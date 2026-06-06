#!/usr/bin/env python
"""Compare learned eviction against heuristic subsets on oracle shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TokenScorer


def evaluate_oracle_shards(
    shard_paths: Sequence[str | Path],
    token_scorer=None,
    device: str | torch.device = "cpu",
) -> dict:
    device = torch.device(device)
    totals = {
        "learned": _new_metric_accumulator(),
        "heuristic": _new_metric_accumulator(),
        "oracle_best": _new_metric_accumulator(),
    }
    distribution = {
        "learned_by_layer": Counter(),
        "learned_by_token_frame": Counter(),
        "heuristic_by_layer": Counter(),
        "heuristic_by_token_frame": Counter(),
    }
    events_evaluated = 0

    if token_scorer is not None:
        token_scorer = token_scorer.to(device).eval()

    for shard_path in shard_paths:
        shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        events = shard.get("events", shard if isinstance(shard, list) else [])
        for event in events:
            subsets = list(event.get("subsets", []))
            if len(subsets) < 1:
                continue
            subset_by_mask = {
                _indices_key(subset["keep_indices"]): subset
                for subset in subsets
            }
            budget = int(event.get("budget", _infer_first_subset_budget(subsets)))
            protected = event.get("protected_indices", [])
            num_tokens = int(torch.as_tensor(event["score_state"]).shape[-2])
            layer_id = int(event.get("layer_id", 0))

            learned_scores = _learned_scores_for_event(event, token_scorer, device)
            heuristic_scores = torch.as_tensor(
                event.get("base_scores", event.get("heuristic_scores", event.get("importance", torch.zeros(num_tokens)))),
                dtype=torch.float32,
            )
            learned_keep = _topk_keep_indices(learned_scores, budget, protected)
            heuristic_keep = _topk_keep_indices(heuristic_scores, budget, protected)
            learned_subset = _match_or_best_subset(subset_by_mask, subsets, learned_keep)
            heuristic_subset = _match_or_best_subset(subset_by_mask, subsets, heuristic_keep)
            oracle_subset = min(subsets, key=lambda subset: float(subset["loss"]))

            _accumulate_subset(totals["learned"], learned_subset)
            _accumulate_subset(totals["heuristic"], heuristic_subset)
            _accumulate_subset(totals["oracle_best"], oracle_subset)
            _accumulate_distribution(distribution, "learned", layer_id, learned_subset, event)
            _accumulate_distribution(distribution, "heuristic", layer_id, heuristic_subset, event)
            events_evaluated += 1

    if events_evaluated == 0:
        raise ValueError("No oracle events were evaluated from the provided shard(s)")

    report = {
        "events": events_evaluated,
        "learned": _finalize_metric_accumulator(totals["learned"]),
        "heuristic": _finalize_metric_accumulator(totals["heuristic"]),
        "oracle_best": _finalize_metric_accumulator(totals["oracle_best"]),
        "cache_distribution": {
            key: {str(k): int(v) for k, v in sorted(counter.items())}
            for key, counter in distribution.items()
        },
    }
    report["delta_learned_minus_heuristic"] = (
        report["learned"]["weighted_total"] - report["heuristic"]["weighted_total"]
    )
    return report


def _new_metric_accumulator() -> dict:
    return {"weighted_total": 0.0, "camera": 0.0, "depth": 0.0, "point_map": 0.0, "count": 0}


def _accumulate_subset(accumulator: dict, subset: dict) -> None:
    accumulator["weighted_total"] += float(subset["loss"])
    components = subset.get("loss_components", {})
    accumulator["camera"] += float(components.get("camera", 0.0))
    accumulator["depth"] += float(components.get("depth", 0.0))
    accumulator["point_map"] += float(components.get("point_map", 0.0))
    accumulator["count"] += 1


def _finalize_metric_accumulator(accumulator: dict) -> dict:
    count = max(int(accumulator["count"]), 1)
    return {
        "weighted_total": accumulator["weighted_total"] / count,
        "camera": accumulator["camera"] / count,
        "depth": accumulator["depth"] / count,
        "point_map": accumulator["point_map"] / count,
    }


def _learned_scores_for_event(event: dict, token_scorer, device: torch.device) -> torch.Tensor:
    explicit = event.get("learned_token_scores")
    if explicit is not None:
        return torch.as_tensor(explicit, dtype=torch.float32)
    if token_scorer is None:
        raise ValueError(
            "Oracle shard event is missing learned_token_scores; pass a TokenScorer checkpoint "
            "or include precomputed learned scores."
        )
    score_state = torch.as_tensor(event["score_state"], dtype=torch.float32, device=device)
    metadata_features = torch.as_tensor(event["metadata_features"], dtype=torch.float32, device=device)
    if score_state.dim() == 2:
        score_state = score_state.unsqueeze(0)
    if metadata_features.dim() == 2:
        metadata_features = metadata_features.unsqueeze(0)
    layer_id = torch.tensor([int(event.get("layer_id", 0))], device=device)
    with torch.inference_mode():
        logits = token_scorer(score_state, metadata_features, layer_id=layer_id)
    return logits[0].detach().cpu().float()


def _topk_keep_indices(scores: torch.Tensor, budget: int, protected_indices: Iterable[int] = ()) -> torch.Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    num_tokens = int(scores.numel())
    protected = torch.as_tensor(list(protected_indices), dtype=torch.long)
    protected = protected[(protected >= 0) & (protected < num_tokens)].unique(sorted=True)
    keep_budget = max(int(budget) - int(protected.numel()), 0)
    protected_set = set(protected.tolist())
    candidates = torch.tensor([idx for idx in range(num_tokens) if idx not in protected_set], dtype=torch.long)
    if keep_budget <= 0 or candidates.numel() == 0:
        return protected[: max(int(budget), 0)].sort().values
    top = candidates[torch.topk(scores[candidates], k=min(keep_budget, candidates.numel())).indices]
    return torch.cat([protected, top]).unique(sorted=True)


def _match_or_best_subset(subset_by_mask: dict, subsets: Sequence[dict], keep_indices: torch.Tensor) -> dict:
    key = _indices_key(keep_indices)
    if key in subset_by_mask:
        return subset_by_mask[key]
    keep_set = set(key)
    return min(
        subsets,
        key=lambda subset: (
            len(keep_set.symmetric_difference(set(_indices_key(subset["keep_indices"])))),
            float(subset["loss"]),
        ),
    )


def _indices_key(indices) -> tuple[int, ...]:
    tensor = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
    return tuple(int(x) for x in tensor.unique(sorted=True).tolist())


def _infer_first_subset_budget(subsets: Sequence[dict]) -> int:
    if not subsets:
        return 0
    return int(torch.as_tensor(subsets[0]["keep_indices"]).numel())


def _accumulate_distribution(distribution: dict, prefix: str, layer_id: int, subset: dict, event: dict) -> None:
    keep_indices = torch.as_tensor(subset["keep_indices"], dtype=torch.long).reshape(-1)
    distribution[f"{prefix}_by_layer"][int(layer_id)] += int(keep_indices.numel())
    token_frame_ids = event.get("token_frame_ids")
    if token_frame_ids is None:
        frame_id = int(event.get("frame_id", -1))
        distribution[f"{prefix}_by_token_frame"][frame_id] += int(keep_indices.numel())
        return
    token_frame_ids = torch.as_tensor(token_frame_ids, dtype=torch.long).reshape(-1)
    for frame_id in token_frame_ids[keep_indices].tolist():
        distribution[f"{prefix}_by_token_frame"][int(frame_id)] += 1


def load_token_scorer_checkpoint(path: str | None, score_state_dim: int, num_layers: int, device: torch.device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("token_scorer") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        model_state = checkpoint.get("model", checkpoint)
        prefix = "aggregator.token_scorers.0."
        state_dict = {
            key[len(prefix):]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
    if not state_dict:
        raise ValueError(f"No TokenScorer state found in {path}")
    scorer = TokenScorer(
        score_state_dim=score_state_dim,
        metadata_dim=TOKEN_METADATA_FEATURE_DIM,
        num_layers=num_layers,
    )
    scorer.load_state_dict(state_dict, strict=False)
    return scorer.to(device)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-shards", nargs="+", required=True)
    parser.add_argument("--token-scorer-checkpoint")
    parser.add_argument("--score-state-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    scorer = load_token_scorer_checkpoint(
        args.token_scorer_checkpoint,
        score_state_dim=args.score_state_dim,
        num_layers=args.num_layers,
        device=device,
    )
    report = evaluate_oracle_shards(args.oracle_shards, token_scorer=scorer, device=device)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
