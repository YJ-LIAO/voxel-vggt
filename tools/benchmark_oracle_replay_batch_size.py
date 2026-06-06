#!/usr/bin/env python
"""Benchmark and summarize oracle replay batch-size choices."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


ORACLE_LINE_RE = re.compile(r"^\[oracle (?P<ts>[^\]]+)\] (?P<msg>.*)$")
SUBSET_BATCH_RE = re.compile(
    r"replay subsets (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+) "
    r"batch=(?P<batch>\d+) frames=0:(?P<stop>\d+)"
)
SUBSET_SINGLE_RE = re.compile(r"replay subset (?P<idx>\d+)/(?P<total>\d+) frames=0:(?P<stop>\d+)")
MEASURE_RE = re.compile(
    r"measuring event (?P<idx>\d+)/(?P<total_events>\d+) .*"
    r"subsets=(?P<subsets>\d+) future_frames=(?P<future_frames>\d+)"
)
START_RE = re.compile(r"collector start: .*subset_replay_batch_size=(?P<value>\d+)")
TIMING_REPLAY_RE = re.compile(r"timing phase=replay .*elapsed_sec=(?P<elapsed>[0-9.]+)")


@dataclass
class LogSummary:
    path: str
    subset_replay_batch_size: int | None
    measured_events: int
    replay_batches: int
    total_subsets: int
    max_subsets_per_event: int
    total_replay_sec: float
    avg_replay_batch_sec: float
    avg_subsets_per_event: float
    events_per_hour: float | None


@dataclass
class BenchmarkConfig:
    project_root: str = "/path/to/mount/lyj/voxel-vggt"
    python: str = sys.executable
    config: str = "config/train_frontend_finetune.yaml"
    dataset_key: str = "train_dataset"
    output_dir: str = "checkpoints/token_oracle/bench_replay_batch"
    device: str = "0"
    replay_batch_sizes: Sequence[int] = (8, 16, 24, 32)
    max_batches: int = 1
    max_events: int = 16
    num_samples: int = 8
    oracle_window: int = 4
    num_views: int | None = None
    layers_per_frame: int = 4
    max_events_per_sequence: int = 16
    seed: int = 0


def summarize_log(path: str | Path) -> LogSummary:
    path = Path(path)
    rows: list[tuple[datetime, str]] = []
    subset_replay_batch_size: int | None = None
    measured_events = 0
    total_subsets = 0
    max_subsets_per_event = 0

    for line in path.read_text(errors="replace").splitlines():
        match = ORACLE_LINE_RE.match(line)
        if not match:
            continue
        ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
        msg = match.group("msg")
        rows.append((ts, msg))
        start_match = START_RE.search(msg)
        if start_match:
            subset_replay_batch_size = int(start_match.group("value"))
        measure_match = MEASURE_RE.search(msg)
        if measure_match:
            measured_events += 1
            subsets = int(measure_match.group("subsets"))
            total_subsets += subsets
            max_subsets_per_event = max(max_subsets_per_event, subsets)

    replay_batches = 0
    total_replay_sec = 0.0
    for idx, (ts, msg) in enumerate(rows):
        timing_match = TIMING_REPLAY_RE.search(msg)
        if timing_match:
            replay_batches += 1
            total_replay_sec += float(timing_match.group("elapsed"))
            continue
        if SUBSET_BATCH_RE.search(msg) or SUBSET_SINGLE_RE.search(msg):
            replay_batches += 1
            next_ts = _next_oracle_timestamp(rows, idx + 1)
            if next_ts is not None:
                delta = (next_ts - ts).total_seconds()
                if delta >= 0.0:
                    total_replay_sec += delta

    avg_replay_batch_sec = total_replay_sec / replay_batches if replay_batches else 0.0
    avg_subsets_per_event = total_subsets / measured_events if measured_events else 0.0
    events_per_hour = None
    if rows and measured_events > 0:
        elapsed = (rows[-1][0] - rows[0][0]).total_seconds()
        if elapsed > 0:
            events_per_hour = measured_events * 3600.0 / elapsed
    return LogSummary(
        path=str(path),
        subset_replay_batch_size=subset_replay_batch_size,
        measured_events=measured_events,
        replay_batches=replay_batches,
        total_subsets=total_subsets,
        max_subsets_per_event=max_subsets_per_event,
        total_replay_sec=total_replay_sec,
        avg_replay_batch_sec=avg_replay_batch_sec,
        avg_subsets_per_event=avg_subsets_per_event,
        events_per_hour=events_per_hour,
    )


def build_benchmark_commands(cfg: BenchmarkConfig) -> list[list[str]]:
    commands: list[list[str]] = []
    output_dir = Path(cfg.output_dir)
    for replay_batch_size in cfg.replay_batch_sizes:
        output = output_dir / f"oracle_bench_rbs{int(replay_batch_size)}.pt"
        command = [
            str(cfg.python),
            "-u",
            "tools/collect_counterfactual_oracle.py",
            "--config",
            str(cfg.config),
            "--dataset-key",
            str(cfg.dataset_key),
            "--output",
            str(output),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--max-batches",
            str(int(cfg.max_batches)),
            "--max-events",
            str(int(cfg.max_events)),
            "--num-samples",
            str(int(cfg.num_samples)),
            "--oracle-window",
            str(int(cfg.oracle_window)),
            "--subset-replay-batch-size",
            str(int(replay_batch_size)),
            "--layers-per-frame",
            str(int(cfg.layers_per_frame)),
            "--max-events-per-sequence",
            str(int(cfg.max_events_per_sequence)),
            "--seed",
            str(int(cfg.seed)),
            "--device",
            "cuda",
        ]
        if cfg.num_views is not None:
            command.extend(["--num-views", str(int(cfg.num_views))])
        commands.append(command)
    return commands


def format_summary(summary: LogSummary) -> str:
    events_per_hour = "NA" if summary.events_per_hour is None else f"{summary.events_per_hour:.2f}"
    replay_size = "unknown" if summary.subset_replay_batch_size is None else str(summary.subset_replay_batch_size)
    return (
        f"{Path(summary.path).name}: replay_batch_size={replay_size} "
        f"events={summary.measured_events} replay_batches={summary.replay_batches} "
        f"subsets={summary.total_subsets} avg_subsets_event={summary.avg_subsets_per_event:.1f} "
        f"max_subsets_event={summary.max_subsets_per_event} "
        f"total_replay_sec={summary.total_replay_sec:.1f} "
        f"avg_replay_batch_sec={summary.avg_replay_batch_sec:.1f} "
        f"events_per_hour={events_per_hour}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize existing oracle collection logs.")
    summarize_parser.add_argument("logs", nargs="+", help="Log files to summarize.")

    bench_parser = subparsers.add_parser("bench", help="Generate or run replay batch-size benchmark commands.")
    bench_parser.add_argument("--project-root", default="/path/to/mount/lyj/voxel-vggt")
    bench_parser.add_argument("--python", default=sys.executable)
    bench_parser.add_argument("--config", default="config/train_frontend_finetune.yaml")
    bench_parser.add_argument("--dataset-key", default="train_dataset")
    bench_parser.add_argument("--output-dir", default="checkpoints/token_oracle/bench_replay_batch")
    bench_parser.add_argument("--device", default="0")
    bench_parser.add_argument("--replay-batch-sizes", default="8,16,24,32")
    bench_parser.add_argument("--max-batches", type=int, default=1)
    bench_parser.add_argument("--max-events", type=int, default=16)
    bench_parser.add_argument("--num-samples", type=int, default=8)
    bench_parser.add_argument("--oracle-window", type=int, default=4)
    bench_parser.add_argument("--num-views", type=int)
    bench_parser.add_argument("--layers-per-frame", type=int, default=4)
    bench_parser.add_argument("--max-events-per-sequence", type=int, default=16)
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summarize":
        for log_path in args.logs:
            print(format_summary(summarize_log(log_path)))
        return 0

    replay_batch_sizes = [
        int(item.strip())
        for item in str(args.replay_batch_sizes).split(",")
        if item.strip()
    ]
    cfg = BenchmarkConfig(
        project_root=args.project_root,
        python=args.python,
        config=args.config,
        dataset_key=args.dataset_key,
        output_dir=args.output_dir,
        device=args.device,
        replay_batch_sizes=replay_batch_sizes,
        max_batches=args.max_batches,
        max_events=args.max_events,
        num_samples=args.num_samples,
        oracle_window=args.oracle_window,
        num_views=args.num_views,
        layers_per_frame=args.layers_per_frame,
        max_events_per_sequence=args.max_events_per_sequence,
        seed=args.seed,
    )
    for command in build_benchmark_commands(cfg):
        formatted = _format_cuda_command(command, cfg)
        print(formatted, flush=True)
        if not args.dry_run:
            subprocess.run(
                command,
                cwd=cfg.project_root,
                env=_cuda_env(cfg),
                check=True,
            )
    return 0


def _next_oracle_timestamp(rows: Sequence[tuple[datetime, str]], start_idx: int) -> datetime | None:
    for idx in range(start_idx, len(rows)):
        return rows[idx][0]
    return None


def _cuda_env(cfg: BenchmarkConfig) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    src_path = str(Path(cfg.project_root) / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    return env


def _format_cuda_command(command: Sequence[str], cfg: BenchmarkConfig) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={cfg.device} "
        f"PYTHONPATH={Path(cfg.project_root) / 'src'} "
        + " ".join(str(item) for item in command)
    )


if __name__ == "__main__":
    raise SystemExit(main())
