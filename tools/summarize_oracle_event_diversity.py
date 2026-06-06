#!/usr/bin/env python
"""Summarize oracle event diversity across logs and shards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


PROBE_RE = re.compile(
    r"probe captured (?P<eviction>\d+) eviction, "
    r"(?P<dedup>\d+) dedup, (?P<fifo>\d+) fifo candidates, "
    r"(?P<selected>\d+) total"
)


def _frame_bucket_counts(frame_hist: dict) -> dict[str, int]:
    buckets = {"early": 0, "mid": 0, "late": 0}
    for frame_id_raw, count_raw in frame_hist.items():
        frame_id = int(frame_id_raw)
        count = int(count_raw)
        if frame_id <= 3:
            buckets["early"] += count
        elif frame_id <= 8:
            buckets["mid"] += count
        else:
            buckets["late"] += count
    return buckets


def _layer_bucket_counts(layer_hist: dict, layer_bucket_width: int = 6) -> dict[int, int]:
    buckets: dict[int, int] = {}
    for layer_id_raw, count_raw in layer_hist.items():
        bucket = int(layer_id_raw) // int(layer_bucket_width)
        buckets[bucket] = buckets.get(bucket, 0) + int(count_raw)
    return buckets


def summarize_shard(shard: dict) -> dict:
    """Compute diversity summary from a loaded oracle shard."""
    events = list(shard.get("events", []))
    event_type_counts: dict[str, int] = {}
    frame_hist: dict[int, int] = {}
    layer_hist: dict[int, int] = {}
    dataset_counts: dict[str, int] = {}

    for event in events:
        event_type = str(event.get("event_type", "eviction"))
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        frame_id = int(event.get("frame_id", 0))
        layer_id = int(event.get("layer_id", 0))
        frame_hist[frame_id] = frame_hist.get(frame_id, 0) + 1
        layer_hist[layer_id] = layer_hist.get(layer_id, 0) + 1

        provenance = event.get("sequence_provenance") or {}
        dataset = str(provenance.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    return {
        "total_events": len(events),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "frame_histogram": dict(sorted(frame_hist.items())),
        "frame_bucket_counts": _frame_bucket_counts(frame_hist),
        "layer_histogram": dict(sorted(layer_hist.items())),
        "layer_bucket_counts": dict(sorted(_layer_bucket_counts(layer_hist).items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "raw_event_type_counts": {},
    }


def summarize_log(path: str | Path) -> dict:
    """Parse collector logs, including GPU3-style probe and final summary lines."""
    path = Path(path)
    raw_counts = {"eviction": 0, "dedup": 0, "fifo_topk": 0}
    probe_batches = 0
    selected_candidate_events = 0
    final_summary: dict = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROBE_RE.search(line)
        if match is not None:
            probe_batches += 1
            raw_counts["eviction"] += int(match.group("eviction"))
            raw_counts["dedup"] += int(match.group("dedup"))
            raw_counts["fifo_topk"] += int(match.group("fifo"))
            selected_candidate_events += int(match.group("selected"))

        if "flushed final shard" in line and "summary=" in line:
            try:
                final_summary = json.loads(line.split("summary=", 1)[1].strip())
            except json.JSONDecodeError:
                pass

    event_type_counts = dict(final_summary.get("event_type_counts", {}))
    frame_hist = dict(final_summary.get("frame_histogram", {}))
    layer_hist = dict(final_summary.get("layer_histogram", {}))
    total_events = int(final_summary.get("num_events", sum(int(v) for v in event_type_counts.values())))

    return {
        "total_events": total_events,
        "event_type_counts": event_type_counts,
        "frame_histogram": frame_hist,
        "frame_bucket_counts": _frame_bucket_counts(frame_hist),
        "layer_histogram": layer_hist,
        "layer_bucket_counts": dict(sorted(_layer_bucket_counts(layer_hist).items())),
        "dataset_counts": dict(final_summary.get("dataset_counts", {})),
        "raw_event_type_counts": raw_counts,
        "probe_batches": probe_batches,
        "selected_candidate_events": selected_candidate_events,
        "elapsed_sec": float(final_summary.get("elapsed_sec", 0.0) or 0.0),
    }


def check_diversity_thresholds(summary: dict, profile: str = "") -> list[str]:
    """Return diversity threshold issues. Empty list means no threshold issue."""
    issues: list[str] = []
    total = int(summary.get("total_events", 0))
    if total <= 0:
        return ["NO_EVENTS: shard has zero events"]

    profile = str(profile or "")
    event_counts = summary.get("event_type_counts", {})
    dedup_pct = float(event_counts.get("dedup", 0)) / float(total) * 100.0

    if dedup_pct > 70.0 and "dedup" not in profile:
        issues.append(f"DEDUP_DOMINANCE: dedup is {dedup_pct:.1f}% of events")
    if "eviction" in profile and int(event_counts.get("eviction", 0)) == 0:
        issues.append("NO_EVICTION: eviction events are zero in eviction profile")
    if "fifo" in profile and int(event_counts.get("fifo_topk", 0)) == 0:
        issues.append("NO_FIFO: fifo_topk events are zero in FIFO profile")

    frame_buckets = summary.get("frame_bucket_counts", {})
    if int(frame_buckets.get("early", 0)) == total and "dedup" not in profile:
        issues.append("ALL_EARLY_FRAMES: all events are in frames 0-3")

    layer_buckets = summary.get("layer_bucket_counts", {})
    if len(layer_buckets) <= 2:
        issues.append(f"NARROW_LAYERS: only {len(layer_buckets)} layer buckets covered")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Paths to .pt shard files or .log files")
    parser.add_argument("--profile", default="", help="Profile name for threshold checks")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    issues_by_path: dict[str, list[str]] = {}
    for input_path in args.inputs:
        path = Path(input_path)
        if path.suffix == ".pt":
            summary = summarize_shard(torch.load(path, map_location="cpu"))
        elif path.suffix == ".log":
            summary = summarize_log(path)
        else:
            raise ValueError(f"Unsupported input type: {path}")
        summaries[str(path)] = summary
        issues_by_path[str(path)] = check_diversity_thresholds(summary, profile=args.profile)

    if args.json:
        print(json.dumps({"summaries": summaries, "issues": issues_by_path}, indent=2, default=str))
    else:
        for path, summary in summaries.items():
            print(f"\n=== {path} ===")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            issues = issues_by_path[path]
            if issues:
                print("  ISSUES:")
                for issue in issues:
                    print(f"    FAIL {issue}")
            else:
                print("  PASS Diversity thresholds passed")

    if any(issues_by_path.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
