#!/usr/bin/env python
"""Collect counterfactual eviction oracle shards from frontend training data."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.training.frontend_oracle_collector import (
    FrontendOracleCollectorConfig,
    collect_oracle_shard_from_config,
    save_oracle_shard,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="/path/to/mount/lyj/voxel-vggt/config/train_frontend_finetune.yaml",
        help="Frontend finetune YAML that defines train_dataset/test_dataset and checkpoints.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt oracle shard. Falls back to 'output' in the config YAML.",
    )
    parser.add_argument("--dataset-key", default="train_dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--oracle-window", type=int, default=4)
    parser.add_argument(
        "--num-views",
        type=int,
        help="Override config num_views for oracle collection only.",
    )
    parser.add_argument(
        "--subset-replay-batch-size",
        type=int,
        default=1,
        help="Replay this many sampled keep subsets for the same event in one batched forward pass.",
    )
    parser.add_argument(
        "--layers-per-frame",
        type=int,
        default=0,
        help="If >0, collect only a rotating stratified subset of layers per frame.",
    )
    parser.add_argument(
        "--max-events-per-sequence",
        type=int,
        default=0,
        help="If >0, cap measured eviction events per input sequence before moving to the next sequence.",
    )
    parser.add_argument(
        "--flush-every-events",
        type=int,
        default=16,
        help="Write a partial shard after this many newly collected events; 0 disables event-based flushing.",
    )
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=1,
        help="Write a partial shard after this many processed batches; 0 disables batch-based flushing.",
    )
    parser.add_argument(
        "--log-every-subsets",
        type=int,
        default=1,
        help="Log replay progress every N candidate subsets; 0 logs every subset.",
    )
    parser.add_argument(
        "--max-fetch-errors",
        type=int,
        default=256,
        help="Skip up to this many dataloader fetch errors before aborting.",
    )
    parser.add_argument(
        "--dataloader-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each dataloader batch fetch; 0 disables. "
             "Prevents indefinite hangs on Lustre/network filesystem I/O stalls.",
    )
    parser.add_argument(
        "--store-replay-payload",
        action="store_true",
        help="Store replay predictions/targets in each subset for debugging; disabled by default for smaller shards.",
    )
    parser.add_argument("--max-subsets-per-dedup-event", type=int, default=8)
    parser.add_argument("--max-subsets-per-eviction-event", type=int, default=8)
    parser.add_argument("--max-subsets-per-fifo-event", type=int, default=8)
    parser.add_argument("--max-candidate-events-per-sequence", type=int, default=256)
    parser.add_argument("--max-events-per-frame", type=int, default=6)
    parser.add_argument("--event-selection-policy", type=str, default="stratified_round_robin", choices=["first_n", "stratified_round_robin"])
    parser.add_argument("--stratified-layer-bucket-width", type=int, default=6)
    parser.add_argument("--oracle-profile", type=str, default="real_policy", choices=["real_policy", "low_budget_eviction", "fifo_topk"])
    parser.add_argument("--frontend-per-layer-budget-override", type=int, default=None)
    parser.add_argument("--fifo-keep-topk-override", type=int, default=None)
    parser.add_argument(
        "--fifo-count-candidates-for-oracle",
        type=str,
        default=None,
        help="Comma-separated list of keep count values to sample for FIFO oracle (e.g., '0,8,16,32,64,128').",
    )
    parser.add_argument("--sequence-manifest-path", type=str, default=None)
    parser.add_argument("--sequence-partition-policy", type=str, default="hash_mod", choices=["hash_mod", "contiguous"])
    parser.add_argument("--num-sequence-shards", type=int, default=None)
    parser.add_argument("--sequence-shard-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument(
        "--no-high-budget-teacher",
        action="store_true",
        help="Use GT targets only; do not run high-budget teacher fallback.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run probe-only diagnostics without replay.",
    )
    parser.add_argument(
        "--probe-output-json",
        type=str,
        default=None,
        help="Write probe diagnostics to this JSON file.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # If the config YAML is a collector-style config (has collection params
    # like max_batches, max_events, etc.), use those as defaults for any
    # argparse parameter that the user did NOT explicitly set on the CLI.
    from omegaconf import OmegaConf as _OC
    _yaml = _OC.load(args.config)
    # Map yaml keys → argparse dest names
    _YAML_TO_ARG = {
        "output": "output", "dataset_key": "dataset_key",
        "batch_size": "batch_size", "num_workers": "num_workers",
        "max_batches": "max_batches", "max_events": "max_events",
        "num_samples": "num_samples", "oracle_window": "oracle_window",
        "num_views": "num_views",
        "subset_replay_batch_size": "subset_replay_batch_size",
        "layers_per_frame": "layers_per_frame",
        "max_events_per_sequence": "max_events_per_sequence",
        "flush_every_events": "flush_every_events",
        "flush_every_batches": "flush_every_batches",
        "log_every_subsets": "log_every_subsets",
        "max_fetch_errors": "max_fetch_errors",
        "max_subsets_per_dedup_event": "max_subsets_per_dedup_event",
        "max_subsets_per_eviction_event": "max_subsets_per_eviction_event",
        "max_subsets_per_fifo_event": "max_subsets_per_fifo_event",
        "max_candidate_events_per_sequence": "max_candidate_events_per_sequence",
        "max_events_per_frame": "max_events_per_frame",
        "event_selection_policy": "event_selection_policy",
        "stratified_layer_bucket_width": "stratified_layer_bucket_width",
        "seed": "seed", "device": "device",
        "oracle_profile": "oracle_profile",
        "store_replay_payload": "store_replay_payload",
        "sequence_manifest_path": "sequence_manifest_path",
        "sequence_partition_policy": "sequence_partition_policy",
        "num_sequence_shards": "num_sequence_shards",
        "sequence_shard_id": "sequence_shard_id",
        "dataloader_timeout": "dataloader_timeout",
        "frontend_per_layer_budget_override": "frontend_per_layer_budget_override",
        "fifo_keep_topk_override": "fifo_keep_topk_override",
        "fifo_count_candidates_for_oracle": "fifo_count_candidates_for_oracle",
        "student_checkpoint": "student_checkpoint",
        "teacher_checkpoint": "teacher_checkpoint",
        "probe_only": "probe_only",
        "probe_output_json": "probe_output_json",
    }
    _explicit = {a for a in sys.argv[1:]}
    for yaml_key, arg_dest in _YAML_TO_ARG.items():
        val = getattr(_yaml, yaml_key, None)
        if val is None:
            continue
        # Check if the user set this flag on the CLI (crude but effective)
        cli_flag = "--" + arg_dest.replace("_", "-")
        if cli_flag not in _explicit:
            setattr(args, arg_dest, val)
    # Boolean flag: only override from yaml if NOT explicitly passed on CLI
    if "--no-high-budget-teacher" not in _explicit:
        hb = getattr(_yaml, "high_budget_teacher", None)
        if hb is not None:
            args.no_high_budget_teacher = not bool(hb)

    if not args.output:
        print("error: --output is required (or set 'output' in the config YAML)", file=sys.stderr)
        sys.exit(1)
    collector_cfg = FrontendOracleCollectorConfig(
        config=args.config,
        output=args.output,
        dataset_key=args.dataset_key,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_batches=int(args.max_batches),
        max_events=int(args.max_events),
        num_samples=int(args.num_samples),
        oracle_window=int(args.oracle_window),
        device=args.device,
        seed=int(args.seed),
        teacher_checkpoint=args.teacher_checkpoint,
        student_checkpoint=args.student_checkpoint,
        high_budget_teacher=not args.no_high_budget_teacher,
        flush_every_events=int(args.flush_every_events),
        flush_every_batches=int(args.flush_every_batches),
        log_every_subsets=int(args.log_every_subsets),
        subset_replay_batch_size=int(args.subset_replay_batch_size),
        layers_per_frame=int(args.layers_per_frame),
        max_events_per_sequence=int(args.max_events_per_sequence),
        max_events_per_frame=int(args.max_events_per_frame),
        max_candidate_events_per_sequence=int(args.max_candidate_events_per_sequence),
        max_subsets_per_dedup_event=int(args.max_subsets_per_dedup_event),
        max_subsets_per_eviction_event=int(args.max_subsets_per_eviction_event),
        max_subsets_per_fifo_event=int(args.max_subsets_per_fifo_event),
        event_selection_policy=args.event_selection_policy,
        stratified_layer_bucket_width=int(args.stratified_layer_bucket_width),
        num_views=int(args.num_views) if args.num_views is not None else None,
        max_fetch_errors=int(args.max_fetch_errors),
        oracle_profile=args.oracle_profile,
        frontend_per_layer_budget_override=int(args.frontend_per_layer_budget_override) if args.frontend_per_layer_budget_override is not None else None,
        fifo_keep_topk_override=int(args.fifo_keep_topk_override) if args.fifo_keep_topk_override is not None else None,
        store_replay_payload=bool(args.store_replay_payload),
        sequence_manifest_path=args.sequence_manifest_path,
        sequence_partition_policy=args.sequence_partition_policy,
        num_sequence_shards=int(args.num_sequence_shards) if args.num_sequence_shards is not None else None,
        sequence_shard_id=int(args.sequence_shard_id) if args.sequence_shard_id is not None else None,
        dataloader_timeout=int(args.dataloader_timeout),
        fifo_count_candidates_for_oracle=args.fifo_count_candidates_for_oracle,
        probe_only=bool(args.probe_only),
        probe_output_json=args.probe_output_json,
    )
    shard = collect_oracle_shard_from_config(collector_cfg)
    shard["collector_config"] = asdict(collector_cfg)
    save_oracle_shard(shard, args.output)
    print(f"Wrote {len(shard['events'])} oracle events to {args.output}")
    if args.probe_output_json and "probe_diagnostics" in shard:
        import json
        with open(args.probe_output_json, "w", encoding="utf-8") as f:
            json.dump(shard["probe_diagnostics"], f, indent=2, default=str)
        print(f"Wrote probe diagnostics to {args.probe_output_json}")


if __name__ == "__main__":
    main()
