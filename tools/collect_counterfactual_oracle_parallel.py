#!/usr/bin/env python
"""Launch multiple counterfactual oracle collectors across GPUs."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ParallelCollectorConfig:
    config: str
    dataset_key: str
    output_dir: str
    devices: str
    shards_per_device: int = 1
    start_shard: int = 0
    batch_size: int = 1
    num_workers: int = 0
    max_batches: int = 128
    max_events: int = 4096
    num_samples: int = 8
    oracle_window: int = 4
    num_views: int | None = None
    subset_replay_batch_size: int = 8
    layers_per_frame: int = 4
    max_events_per_sequence: int = 16
    flush_every_events: int = 16
    flush_every_batches: int = 1
    log_every_subsets: int = 1
    max_fetch_errors: int = 256
    seed_base: int = 0
    python: str = sys.executable
    collector_script: str = "tools/collect_counterfactual_oracle.py"
    project_root: str = "/path/to/mount/lyj/voxel-vggt"
    no_high_budget_teacher: bool = False
    overwrite_existing: bool = False
    store_replay_payload: bool = False
    # Phase 1 fields
    max_subsets_per_dedup_event: int = 8
    max_subsets_per_eviction_event: int = 8
    max_subsets_per_fifo_event: int = 8
    max_candidate_events_per_sequence: int = 256
    max_events_per_frame: int = 6
    event_selection_policy: str = "stratified_round_robin"
    stratified_layer_bucket_width: int = 6
    oracle_profile: str = "real_policy"
    frontend_total_budget_override: int | None = None
    fifo_keep_topk_override: int | None = None
    sequence_manifest_path: str | None = None
    sequence_partition_policy: str = "hash_mod"
    num_sequence_shards: int | None = None
    sequence_shard_id: int | None = None


@dataclass
class CollectionJob:
    shard_id: int
    seed: int
    device: str
    output: str
    log: str
    command: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_frontend_finetune.yaml")
    parser.add_argument("--dataset-key", default="train_dataset")
    parser.add_argument("--output-dir", default="checkpoints/token_oracle")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated GPU ids.")
    parser.add_argument("--shards-per-device", type=int, default=1)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--oracle-window", type=int, default=4)
    parser.add_argument("--num-views", type=int, help="Override config num_views for oracle collection only.")
    parser.add_argument("--subset-replay-batch-size", type=int, default=8)
    parser.add_argument("--layers-per-frame", type=int, default=4)
    parser.add_argument("--max-events-per-sequence", type=int, default=64)
    parser.add_argument("--flush-every-events", type=int, default=16)
    parser.add_argument("--flush-every-batches", type=int, default=1)
    parser.add_argument("--log-every-subsets", type=int, default=1)
    parser.add_argument("--max-fetch-errors", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--collector-script", default="tools/collect_counterfactual_oracle.py")
    parser.add_argument("--project-root", default="/path/to/mount/lyj/voxel-vggt")
    parser.add_argument("--no-high-budget-teacher", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--store-replay-payload", action="store_true")
    parser.add_argument("--max-subsets-per-dedup-event", type=int, default=8)
    parser.add_argument("--max-subsets-per-eviction-event", type=int, default=8)
    parser.add_argument("--max-subsets-per-fifo-event", type=int, default=8)
    parser.add_argument("--max-candidate-events-per-sequence", type=int, default=256)
    parser.add_argument("--max-events-per-frame", type=int, default=6)
    parser.add_argument("--event-selection-policy", default="stratified_round_robin")
    parser.add_argument("--stratified-layer-bucket-width", type=int, default=6)
    parser.add_argument("--oracle-profile", default="real_policy")
    parser.add_argument("--frontend-total-budget-override", type=int, default=None)
    parser.add_argument("--fifo-keep-topk-override", type=int, default=None)
    parser.add_argument("--sequence-manifest-path", default=None)
    parser.add_argument("--sequence-partition-policy", default="hash_mod")
    parser.add_argument("--num-sequence-shards", type=int, default=None)
    parser.add_argument("--sequence-shard-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return args


def build_collection_jobs(cfg: ParallelCollectorConfig) -> list[CollectionJob]:
    devices = [item.strip() for item in str(cfg.devices).split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU id")
    shards_per_device = max(int(cfg.shards_per_device), 1)
    total_jobs = len(devices) * shards_per_device
    output_dir = Path(cfg.output_dir)
    jobs: list[CollectionJob] = []
    job_offset = 0
    shard_id = int(cfg.start_shard)
    while len(jobs) < total_jobs:
        device = devices[job_offset % len(devices)]
        output = output_dir / f"oracle_shard_{shard_id:03d}.pt"
        if not cfg.overwrite_existing and output.exists():
            state = _existing_shard_state(output)
            if state == "complete":
                shard_id += 1
                continue
            if state == "partial":
                shard_id += 1
                continue
        seed = int(cfg.seed_base) + (int(shard_id) - int(cfg.start_shard))
        log = output_dir / f"collect_{shard_id:03d}.log"
        command = [
            str(cfg.python),
            "-u",
            str(cfg.collector_script),
            "--config",
            str(cfg.config),
            "--dataset-key",
            str(cfg.dataset_key),
            "--output",
            str(output),
            "--batch-size",
            str(cfg.batch_size),
            "--num-workers",
            str(cfg.num_workers),
            "--max-batches",
            str(cfg.max_batches),
            "--max-events",
            str(cfg.max_events),
            "--num-samples",
            str(cfg.num_samples),
            "--oracle-window",
            str(cfg.oracle_window),
        ]
        if cfg.num_views is not None:
            command.extend(["--num-views", str(cfg.num_views)])
        command.extend([
            "--subset-replay-batch-size",
            str(cfg.subset_replay_batch_size),
            "--layers-per-frame",
            str(cfg.layers_per_frame),
            "--max-events-per-sequence",
            str(cfg.max_events_per_sequence),
            "--flush-every-events",
            str(cfg.flush_every_events),
            "--flush-every-batches",
            str(cfg.flush_every_batches),
            "--log-every-subsets",
            str(cfg.log_every_subsets),
            "--max-fetch-errors",
            str(cfg.max_fetch_errors),
            "--seed",
            str(seed),
            "--device",
            "cuda",
        ])
        if cfg.no_high_budget_teacher:
            command.append("--no-high-budget-teacher")
        if cfg.store_replay_payload:
            command.append("--store-replay-payload")
        # Phase 1 args
        command.extend([
            "--max-subsets-per-dedup-event",
            str(cfg.max_subsets_per_dedup_event),
            "--max-subsets-per-eviction-event",
            str(cfg.max_subsets_per_eviction_event),
            "--max-subsets-per-fifo-event",
            str(cfg.max_subsets_per_fifo_event),
            "--max-candidate-events-per-sequence",
            str(cfg.max_candidate_events_per_sequence),
            "--max-events-per-frame",
            str(cfg.max_events_per_frame),
            "--event-selection-policy",
            str(cfg.event_selection_policy),
            "--stratified-layer-bucket-width",
            str(cfg.stratified_layer_bucket_width),
            "--oracle-profile",
            str(cfg.oracle_profile),
        ])
        if cfg.frontend_total_budget_override is not None:
            command.extend(["--frontend-total-budget-override", str(cfg.frontend_total_budget_override)])
        if cfg.fifo_keep_topk_override is not None:
            command.extend(["--fifo-keep-topk-override", str(cfg.fifo_keep_topk_override)])
        if cfg.sequence_manifest_path is not None:
            command.extend(["--sequence-manifest-path", str(cfg.sequence_manifest_path)])
        command.extend([
            "--sequence-partition-policy",
            str(cfg.sequence_partition_policy),
        ])
        if cfg.num_sequence_shards is not None:
            command.extend(["--num-sequence-shards", str(cfg.num_sequence_shards)])
        if cfg.sequence_shard_id is not None:
            command.extend(["--sequence-shard-id", str(cfg.sequence_shard_id)])
        jobs.append(
            CollectionJob(
                shard_id=shard_id,
                seed=seed,
                device=device,
                output=str(output),
                log=str(log),
                command=command,
            )
        )
        job_offset += 1
        shard_id += 1
    return jobs


def run_collection_jobs(
    jobs: Sequence[CollectionJob],
    project_root: str,
    dry_run: bool = False,
) -> int:
    if dry_run:
        for job in jobs:
            print(_format_job_command(job, project_root))
        return 0

    Path(project_root).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        Path(job.output).parent.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    active: dict[str, tuple[CollectionJob, subprocess.Popen, object]] = {}
    failures: list[CollectionJob] = []

    while pending or active:
        launched = True
        while launched:
            launched = False
            busy_devices = set(active.keys())
            for idx, job in enumerate(list(pending)):
                if job.device in busy_devices:
                    continue
                pending.pop(idx)
                log_file = open(job.log, "w", buffering=1)
                env = _job_env(job, project_root)
                log_file.write(_format_job_header(job, project_root) + "\n")
                print(f"[parallel-oracle] launch shard={job.shard_id:03d} gpu={job.device} log={job.log}", flush=True)
                process = subprocess.Popen(
                    job.command,
                    cwd=project_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                active[job.device] = (job, process, log_file)
                launched = True
                break

        time.sleep(2.0)
        for device, (job, process, log_file) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_file.close()
            del active[device]
            if code != 0:
                print(
                    f"[parallel-oracle] FAILED shard={job.shard_id:03d} gpu={job.device} "
                    f"exit={code} log={job.log}",
                    flush=True,
                )
                print(_format_job_command(job, project_root), flush=True)
                failures.append(job)
            else:
                print(
                    f"[parallel-oracle] done shard={job.shard_id:03d} gpu={job.device} "
                    f"output={job.output}",
                    flush=True,
                )
    return 1 if failures else 0


def _job_env(job: CollectionJob, project_root: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.device)
    src_path = str(Path(project_root) / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    return env


def _format_job_command(job: CollectionJob, project_root: str) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={job.device} "
        f"PYTHONPATH={Path(project_root) / 'src'} "
        + " ".join(job.command)
        + f" > {job.log} 2>&1"
    )


def _existing_shard_state(path: Path) -> str:
    try:
        import torch

        shard = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "unknown"
    if isinstance(shard, dict) and shard.get("partial") is False:
        return "complete"
    if isinstance(shard, dict) and shard.get("partial") is True:
        return "partial"
    return "unknown"


def _format_job_header(job: CollectionJob, project_root: str) -> str:
    config_path = _command_value(job.command, "--config")
    config_hash = _file_sha256(Path(project_root) / config_path) if config_path else "unknown"
    return (
        f"[parallel-oracle] shard={job.shard_id:03d} gpu={job.device} "
        f"seed={job.seed} config={config_path or 'unknown'} config_sha256={config_hash}"
    )


def _command_value(command: Sequence[str], flag: str) -> str | None:
    for idx, item in enumerate(command):
        if item == flag and idx + 1 < len(command):
            return str(command[idx + 1])
    return None


def _file_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    args = parse_args()
    cfg = ParallelCollectorConfig(**{key: value for key, value in vars(args).items() if key != "dry_run"})
    jobs = build_collection_jobs(cfg)
    raise SystemExit(run_collection_jobs(jobs, project_root=cfg.project_root, dry_run=bool(args.dry_run)))


if __name__ == "__main__":
    main()
