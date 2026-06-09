#!/usr/bin/env python
"""Read-only diagnostic tool for oracle training signal analysis.

Analyzes oracle shards to help choose thresholds and assess training signal
quality.  Does NOT modify any training code or data.

Output sections:
  EVENT_COUNTS, THRESHOLD_SWEEP, SEQUENCE_DISTRIBUTION,
  COUNT_CONFIDENCE, COUNT_LABEL_DISTRIBUTION, SCORE_PROJECTION_STATE
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper: add src/ to sys.path so dataset classes can be imported
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    """Add the project ``src/`` directory to ``sys.path`` if needed."""
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()

from ovggt.training.token_oracle_dataset import (  # noqa: E402
    CounterfactualOracleDataset,
    FifoCountDataset,
    load_oracle_events,
    summarize_oracle_pair_samples,
    summarize_fifo_count_samples,
)


# ---------------------------------------------------------------------------
# Public helpers (testable)
# ---------------------------------------------------------------------------

def summarize_threshold_sweep(
    events: list[dict],
    thresholds: Sequence[float],
    fifo_token_pair_mode: str,
    max_pairs_per_event: int,
    pair_sampling_seed: int,
) -> dict[str, dict]:
    """Build CounterfactualOracleDataset at each threshold and report stats.

    Returns:
        Dict keyed by threshold string (e.g. ``"0.01"``), each value containing:
        total_pairs, by_event_type, margin_p50, usable_events, fifo_cross_keep_frac.
    """
    results: dict[str, dict] = {}
    for threshold in thresholds:
        threshold_str = f"{float(threshold):.2f}" if float(threshold) != round(float(threshold), 4) else str(float(threshold))
        # Normalize to a clean string
        threshold_str = f"{float(threshold)}"
        # Try cleaner representation
        t = float(threshold)
        if t == int(t * 100) / 100:
            threshold_str = f"{t:.2f}"

        ds = CounterfactualOracleDataset.from_events(
            events,
            min_loss_gap=float(threshold),
            fifo_token_pair_mode=fifo_token_pair_mode,
            max_pairs_per_event=max_pairs_per_event,
            pair_sampling_seed=pair_sampling_seed,
        )
        summary = summarize_oracle_pair_samples(ds.samples)
        results[threshold_str] = {
            "total_pairs": summary["count"],
            "by_event_type": dict(summary["by_event_type"]),
            "margin_p50": summary["target_margin_p50"],
            "usable_events": summary["by_event_type_unique_event_count"],
            "fifo_cross_keep_frac": summary["fifo_cross_keep_frac"],
        }
    return results


def summarize_sequence_distribution(events: list[dict]) -> dict:
    """Report per-sequence event counts and distribution statistics.

    Returns:
        Dict with total_sequences, events_per_sequence percentiles (p10/p50/p90),
        and top 10 sequences by count as list of (sequence_id, count) pairs.
    """
    if not events:
        return {
            "total_sequences": 0,
            "events_per_sequence_p10": 0.0,
            "events_per_sequence_p50": 0.0,
            "events_per_sequence_p90": 0.0,
            "top_sequences": [],
        }

    seq_counts: dict[str, int] = Counter()
    for event in events:
        prov = event.get("sequence_provenance") or {}
        seq_id = str(prov.get("sequence_id", event.get("event_id", "unknown")))
        seq_counts[seq_id] += 1

    counts_arr = np.array(list(seq_counts.values()), dtype=np.float64)

    # Top 10 by count descending
    top_sequences = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_sequences": len(seq_counts),
        "events_per_sequence_p10": float(np.percentile(counts_arr, 10)),
        "events_per_sequence_p50": float(np.percentile(counts_arr, 50)),
        "events_per_sequence_p90": float(np.percentile(counts_arr, 90)),
        "top_sequences": [(sid, int(cnt)) for sid, cnt in top_sequences],
    }


def summarize_count_confidence(
    events: list[dict],
    count_candidates: Sequence[int],
    label_reduction: str,
) -> dict:
    """Compute best-vs-second-best gap per event for FifoCountDataset.

    Returns:
        Dict with p10/p50/p90 of count_loss_gap, total_samples,
        above_thresholds dict mapping threshold -> count.
    """
    ds = FifoCountDataset.from_events(
        events,
        count_candidates=list(count_candidates),
        label_reduction=label_reduction,
    )

    if len(ds) == 0:
        return {
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "total_samples": 0,
            "above_thresholds": {},
        }

    gaps = [float(sample.get("count_loss_gap", 0.0)) for sample in ds.samples]
    gaps_arr = np.array(gaps, dtype=np.float64)

    # Count samples above common thresholds
    gap_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    above_thresholds = {}
    for gt in gap_thresholds:
        above_thresholds[str(gt)] = int(np.sum(gaps_arr >= gt))

    return {
        "p10": float(np.percentile(gaps_arr, 10)),
        "p50": float(np.percentile(gaps_arr, 50)),
        "p90": float(np.percentile(gaps_arr, 90)),
        "total_samples": len(ds),
        "above_thresholds": above_thresholds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_event_counts(events: list[dict]) -> None:
    """Print EVENT_COUNTS section."""
    type_counts: dict[str, int] = Counter()
    for event in events:
        et = str(event.get("event_type", "eviction"))
        type_counts[et] += 1
    total = len(events)
    print(f"\nEVENT_COUNTS total={total} {dict(sorted(type_counts.items()))}")


def _print_threshold_sweep(
    events: list[dict],
    thresholds: Sequence[float],
    fifo_token_pair_mode: str,
    max_pairs_per_event: int,
    pair_sampling_seed: int,
) -> None:
    """Print THRESHOLD_SWEEP section."""
    results = summarize_threshold_sweep(
        events, thresholds, fifo_token_pair_mode,
        max_pairs_per_event, pair_sampling_seed,
    )
    print("\nTHRESHOLD_SWEEP")
    for threshold_str, summary in sorted(results.items(), key=lambda x: float(x[0])):
        print(
            f"  threshold={threshold_str} "
            f"total_pairs={summary['total_pairs']} "
            f"by_event_type={summary['by_event_type']} "
            f"margin_p50={summary['margin_p50']:.6f} "
            f"usable_events={summary['usable_events']} "
            f"fifo_cross_keep_frac={summary['fifo_cross_keep_frac']:.3f}"
        )


def _print_sequence_distribution(events: list[dict]) -> None:
    """Print SEQUENCE_DISTRIBUTION section."""
    result = summarize_sequence_distribution(events)
    print(
        f"\nSEQUENCE_DISTRIBUTION "
        f"total_sequences={result['total_sequences']} "
        f"events_per_sequence_p10={result['events_per_sequence_p10']:.2f} "
        f"events_per_sequence_p50={result['events_per_sequence_p50']:.2f} "
        f"events_per_sequence_p90={result['events_per_sequence_p90']:.2f}"
    )
    print("  top_sequences:")
    for seq_id, count in result["top_sequences"]:
        print(f"    {seq_id}: {count}")


def _print_count_confidence(
    events: list[dict],
    count_candidates: Sequence[int],
    label_reduction: str,
) -> None:
    """Print COUNT_CONFIDENCE section."""
    result = summarize_count_confidence(events, count_candidates, label_reduction)
    print(
        f"\nCOUNT_CONFIDENCE "
        f"p10={result['p10']:.6f} "
        f"p50={result['p50']:.6f} "
        f"p90={result['p90']:.6f} "
        f"above_thresholds={result['above_thresholds']}"
    )


def _print_count_label_distribution(
    events: list[dict],
    count_candidates: Sequence[int],
    label_reduction: str,
) -> None:
    """Print COUNT_LABEL_DISTRIBUTION section."""
    ds = FifoCountDataset.from_events(
        events,
        count_candidates=list(count_candidates),
        label_reduction=label_reduction,
    )
    summary = summarize_fifo_count_samples(ds.samples)

    keep_counts = summary.get("target_keep_count", {})
    majority_accuracy = 0.0
    if summary["count"] > 0 and keep_counts:
        majority_count = max(keep_counts.values())
        majority_accuracy = majority_count / summary["count"]

    print(
        f"\nCOUNT_LABEL_DISTRIBUTION "
        f"target_keep_count={keep_counts} "
        f"majority_accuracy={majority_accuracy:.4f}"
    )


def _print_score_projection_state(shard_paths: Sequence[str | Path]) -> None:
    """Print SCORE_PROJECTION_STATE section for each shard."""
    print("\nSCORE_PROJECTION_STATE")
    for shard_path in shard_paths:
        shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        has_projection = False
        key_count = 0
        if isinstance(shard, dict):
            proj_state = shard.get("score_state_projection_state", {})
            if proj_state and isinstance(proj_state, dict):
                filtered = {k: v for k, v in proj_state.items()
                            if k.startswith("aggregator.score_state_projs.")}
                key_count = len(filtered)
                has_projection = key_count > 0
        print(
            f"  shard={shard_path} "
            f"has_projection={has_projection} "
            f"key_count={key_count}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-shards", nargs="+", required=True,
        help="Path(s) to .pt oracle shard files",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.01, 0.02, 0.03, 0.05],
        help="Loss-gap thresholds to sweep",
    )
    parser.add_argument(
        "--fifo-token-pair-mode", default="same_keep_count",
        choices=["any", "same_keep_count"],
        help="Pair mode for fifo_topk events",
    )
    parser.add_argument(
        "--pair-sampling-seed", type=int, default=0,
        help="Deterministic seed for pair sampling",
    )
    parser.add_argument(
        "--max-pairs-per-event", type=int, default=64,
        help="Max pairs to keep per event",
    )
    parser.add_argument(
        "--count-candidates", nargs="+", type=int,
        default=[0, 8, 16, 32, 64, 128],
        help="Candidate keep_count values for FifoCountDataset",
    )
    parser.add_argument(
        "--label-reduction", default="min",
        choices=["min", "mean"],
        help="Label reduction strategy for count dataset",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    print(f"Loading {len(args.oracle_shards)} shard(s) ...", file=sys.stderr)
    events = load_oracle_events(args.oracle_shards)
    print(f"Loaded {len(events)} events.", file=sys.stderr)

    _print_event_counts(events)
    _print_threshold_sweep(
        events, args.thresholds, args.fifo_token_pair_mode,
        args.max_pairs_per_event, args.pair_sampling_seed,
    )
    _print_sequence_distribution(events)
    _print_count_confidence(events, args.count_candidates, args.label_reduction)
    _print_count_label_distribution(events, args.count_candidates, args.label_reduction)
    _print_score_projection_state(args.oracle_shards)


if __name__ == "__main__":
    main()
