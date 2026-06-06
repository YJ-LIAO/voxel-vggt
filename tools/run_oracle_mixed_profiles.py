#!/usr/bin/env python
"""Launch oracle collection with mixed profiles across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


PYTHON = "/mnt/lyj/miniconda3/envs/streamvggt/bin/python"

DEFAULT_PROFILE_WEIGHTS = {
    "real_policy_dedup": 0.4,
    "low_budget_eviction": 0.3,
    "fifo_topk_forced": 0.3,
}


def parse_profile_weights(raw: str | None) -> dict[str, float]:
    """Parse profile weights like 'real_policy_dedup=0.4,low_budget_eviction=0.3'."""
    if raw is None or not str(raw).strip():
        return dict(DEFAULT_PROFILE_WEIGHTS)
    weights: dict[str, float] = {}
    for item in str(raw).split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def allocate_profile_counts(num_shards: int, profile_weights: dict[str, float]) -> dict[str, int]:
    """Allocate integer shard counts using largest remainder, preserving total exactly."""
    num_shards = int(num_shards)
    if num_shards <= 0:
        return {}

    positive_weights = {
        profile: float(weight)
        for profile, weight in profile_weights.items()
        if float(weight) > 0
    }
    total_weight = sum(positive_weights.values())
    if total_weight <= 0:
        raise ValueError("At least one profile weight must be positive")

    raw_counts = {
        profile: num_shards * weight / total_weight
        for profile, weight in positive_weights.items()
    }
    counts = {profile: int(count) for profile, count in raw_counts.items()}
    remaining = num_shards - sum(counts.values())
    remainders = sorted(
        raw_counts.keys(),
        key=lambda profile: (raw_counts[profile] - counts[profile], profile),
        reverse=True,
    )
    for profile in remainders[:remaining]:
        counts[profile] += 1
    return counts


def profile_args(profile: str, events_per_shard: int) -> list[str]:
    """Return collector args for one named profile."""
    common = ["--max-events", str(int(events_per_shard))]
    if profile == "real_policy_dedup":
        return [
            "--oracle-profile", "real_policy",
            "--event-type-quotas", "dedup=8,eviction=0,fifo_topk=0",
            *common,
        ]
    if profile == "low_budget_eviction":
        return [
            "--oracle-profile", "low_budget_eviction",
            "--frontend-per-layer-budget-override", "2500",
            "--event-type-quotas", "eviction=8,dedup=2,fifo_topk=0",
            *common,
        ]
    if profile == "fifo_topk_forced":
        return [
            "--oracle-profile", "fifo_topk",
            "--fifo-keep-topk-override", "32",
            "--oracle-anchor-interval", "4",
            "--oracle-max-anchors", "2",
            "--event-type-quotas", "fifo_topk=8,dedup=2,eviction=2",
            *common,
        ]
    if profile == "late_frame_scan":
        return [
            "--oracle-profile", "real_policy",
            "--oracle-layer-schedule", "random_bucket",
            "--frame-buckets", json.dumps([[4, 8], [9, 23]]),
            "--event-type-quotas", "eviction=4,dedup=2,fifo_topk=2",
            *common,
        ]
    raise ValueError(f"Unknown profile: {profile}")


def generate_profile_jobs(
    base_config: str,
    output_dir: str,
    num_gpus: int,
    num_shards: int,
    events_per_shard: int = 2048,
    start_shard_id: int = 0,
    profile_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Generate one job dict per shard, allocated by profile weights."""
    del base_config, num_gpus
    weights = profile_weights or dict(DEFAULT_PROFILE_WEIGHTS)
    profile_counts = allocate_profile_counts(num_shards, weights)
    jobs: list[dict] = []
    shard_id = int(start_shard_id)
    for profile, count in profile_counts.items():
        for _ in range(count):
            output = os.path.join(output_dir, f"{profile}_shard_{shard_id:03d}.pt")
            jobs.append(
                {
                    "profile": profile,
                    "shard_id": shard_id,
                    "output": output,
                    "args": profile_args(profile, events_per_shard),
                }
            )
            shard_id += 1
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--events-per-shard", type=int, default=2048)
    parser.add_argument("--start-shard-id", type=int, default=0)
    parser.add_argument(
        "--profile-weights",
        default=None,
        help=(
            "Comma-separated profile weights. Default: "
            "real_policy_dedup=0.4,low_budget_eviction=0.3,fifo_topk_forced=0.3"
        ),
    )
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = generate_profile_jobs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_shards=args.num_shards,
        events_per_shard=args.events_per_shard,
        start_shard_id=args.start_shard_id,
        profile_weights=parse_profile_weights(args.profile_weights),
    )

    def build_command(job: dict) -> list[str]:
        return [
            PYTHON, "-u", "tools/collect_counterfactual_oracle.py",
            "--config", args.base_config,
            "--output", job["output"],
            "--max-batches", str(int(args.max_batches)),
            "--event-selection-policy", "quota_stratified",
            "--layers-per-frame", "4",
            *job["args"],
        ]

    if args.dry_run:
        for idx, job in enumerate(jobs):
            gpu_id = idx % int(args.num_gpus)
            cmd = build_command(job)
            print(f"\n# Profile: {job['profile']} (GPU {gpu_id})")
            print(" \\\n  ".join([f"CUDA_VISIBLE_DEVICES={gpu_id}", *cmd]))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    pending = list(jobs)
    running: dict[int, tuple[subprocess.Popen, object, str]] = {}
    failures: list[tuple[str, int]] = []

    while pending or running:
        for gpu_id in range(int(args.num_gpus)):
            if gpu_id in running or not pending:
                continue
            job = pending.pop(0)
            cmd = build_command(job)
            log_path = os.path.splitext(job["output"])[0] + ".log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Launching profile={job['profile']} gpu={gpu_id} log={log_path}")
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running[gpu_id] = (proc, log_f, job["profile"])

        time.sleep(10)

        for gpu_id, (proc, log_f, profile) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_f.close()
            del running[gpu_id]
            if rc != 0:
                failures.append((profile, int(rc)))

    if failures:
        raise SystemExit(f"Failed jobs: {failures}")


if __name__ == "__main__":
    main()
