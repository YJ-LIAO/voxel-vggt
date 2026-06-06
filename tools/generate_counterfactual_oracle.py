#!/usr/bin/env python
"""Generate counterfactual token-retention oracle shards.

This tool writes the shard format consumed by
``ovggt.training.token_oracle_dataset.CounterfactualOracleDataset``.

This script has four implemented modes:
- ``measured-dump`` converts event dumps whose subset losses were already
  measured by an external frozen-model replay job.
- ``synthetic`` creates tiny deterministic shards for smoke tests.
- ``replay`` converts frozen-model replay dumps where each sampled subset stores
  future-window predictions and camera/depth/point-map targets. The expensive
  cache snapshot/restore and frame replay can run in a separate job; this tool
  makes the loss computation and shard format deterministic.
- ``collect-replay`` drives a replay runner factory to snapshot/restore cache
  state, apply sampled keep subsets, replay the future window, and then convert
  the measured records into oracle events.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import List

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.training.counterfactual_replay import (
    CounterfactualReplayCandidateEvent,
    TASK_WEIGHTS,
    collect_counterfactual_replay_event,
    compute_three_task_loss_components,
    sample_group_retention_subsets,
    weighted_three_task_loss,
)


def build_oracle_events_from_dump(event_dump: list, num_samples: int, seed: int) -> list:
    events = []
    for event_idx, event in enumerate(event_dump):
        score_state = torch.as_tensor(event["score_state"], dtype=torch.float32)
        if score_state.dim() == 3 and score_state.shape[0] == 1:
            score_state = score_state[0]
        metadata_features = torch.as_tensor(event["metadata_features"], dtype=torch.float32)
        if metadata_features.dim() == 3 and metadata_features.shape[0] == 1:
            metadata_features = metadata_features[0]
        num_tokens = int(score_state.shape[0])
        budget = int(event["budget"])
        subsets = event.get("subsets")
        if subsets is None:
            sampled = sample_group_retention_subsets(
                num_tokens=num_tokens,
                budget=budget,
                num_samples=num_samples,
                protected_indices=event.get("protected_indices", []),
                group_ids=event.get("group_ids"),
                base_scores=event.get("base_scores"),
                seed=seed + event_idx,
            )
            measured = event.get("measured_losses")
            if measured is None or len(measured) != len(sampled):
                raise ValueError(
                    "Event dump without explicit subsets must include measured_losses "
                    "for each sampled subset after frozen-model replay."
                )
            subsets = []
            for keep_indices, loss_record in zip(sampled, measured):
                loss_components = loss_record.get("loss_components", loss_record)
                subsets.append(
                    {
                        "keep_indices": keep_indices,
                        "loss": weighted_three_task_loss(loss_components),
                        "loss_components": loss_components,
                    }
                )
        else:
            normalized_subsets = []
            for subset in subsets:
                loss_components = subset.get("loss_components", {})
                loss = float(subset.get("loss", weighted_three_task_loss(loss_components)))
                normalized_subsets.append(
                    {
                        "keep_indices": torch.as_tensor(subset["keep_indices"], dtype=torch.long),
                        "loss": loss,
                        "loss_components": loss_components,
                    }
                )
            subsets = normalized_subsets

        events.append(
            {
                "event_id": event.get("event_id", f"event:{event_idx}"),
                "layer_id": int(event.get("layer_id", 0)),
                "frame_id": int(event.get("frame_id", -1)),
                "budget": budget,
                "sequence_provenance": event.get("sequence_provenance"),
                "score_state": score_state,
                "metadata_features": metadata_features,
                "subsets": subsets,
            }
        )
        for optional_key in _ORACLE_EVENT_OPTIONAL_TENSOR_KEYS:
            if optional_key in event:
                events[-1][optional_key] = torch.as_tensor(event[optional_key])
    return events


def build_oracle_events_from_replay_dump(event_dump: list) -> list:
    events = []
    for event_idx, event in enumerate(event_dump):
        score_state = torch.as_tensor(event["score_state"], dtype=torch.float32)
        if score_state.dim() == 3 and score_state.shape[0] == 1:
            score_state = score_state[0]
        metadata_features = torch.as_tensor(event["metadata_features"], dtype=torch.float32)
        if metadata_features.dim() == 3 and metadata_features.shape[0] == 1:
            metadata_features = metadata_features[0]

        subsets = []
        for subset_idx, subset in enumerate(event.get("subsets", [])):
            replay_record = subset.get("replay", subset)
            predictions = replay_record.get("predictions")
            targets = replay_record.get("targets")
            if predictions is None or targets is None:
                raise ValueError(
                    f"Replay event {event_idx} subset {subset_idx} must include "
                    "replay.predictions and replay.targets"
                )
            loss_components = compute_three_task_loss_components(predictions, targets)
            subsets.append(
                {
                    "keep_indices": torch.as_tensor(subset["keep_indices"], dtype=torch.long),
                    "loss": weighted_three_task_loss(loss_components),
                    "loss_components": loss_components,
                }
            )
        if len(subsets) < 2:
            raise ValueError(f"Replay event {event_idx} must contain at least two sampled subsets")

        normalized_event = {
            "event_id": event.get("event_id", f"replay:{event_idx}"),
            "layer_id": int(event.get("layer_id", 0)),
            "frame_id": int(event.get("frame_id", -1)),
            "budget": int(event.get("budget", score_state.shape[0])),
            "sequence_provenance": event.get("sequence_provenance"),
            "score_state": score_state,
            "metadata_features": metadata_features,
            "subsets": subsets,
        }
        for optional_key in _ORACLE_EVENT_OPTIONAL_TENSOR_KEYS:
            if optional_key in event:
                normalized_event[optional_key] = torch.as_tensor(event[optional_key])
        events.append(normalized_event)
    return events


def collect_replay_events_from_plan(
    replay_plan: dict,
    runner_factory_path: str,
) -> list:
    runner_factory = _load_runner_factory(runner_factory_path)
    runner = runner_factory(replay_plan.get("frames", []))
    measured_events = []
    for event in replay_plan.get("events", []):
        future_frames = event.get("future_frames")
        if future_frames is None:
            future_frames = _future_frames_from_plan(replay_plan, event)
        candidate_event = CounterfactualReplayCandidateEvent(
            event_id=event.get("event_id", f"collect:{len(measured_events)}"),
            layer_id=int(event.get("layer_id", 0)),
            frame_id=int(event.get("frame_id", -1)),
            budget=int(event["budget"]),
            sequence_provenance=event.get("sequence_provenance"),
            score_state=torch.as_tensor(event["score_state"], dtype=torch.float32),
            metadata_features=torch.as_tensor(event["metadata_features"], dtype=torch.float32),
            candidate_subsets=[
                _candidate_subset_to_indices(subset)
                for subset in event.get("candidate_subsets", event.get("subsets", []))
            ],
            protected_indices=_optional_tensor(event, "protected_indices", dtype=torch.long),
            base_scores=_optional_tensor(event, "base_scores", dtype=torch.float32),
            learned_token_scores=_optional_tensor(event, "learned_token_scores", dtype=torch.float32),
            group_ids=_optional_tensor(event, "group_ids", dtype=torch.long),
            token_frame_ids=_optional_tensor(event, "token_frame_ids", dtype=torch.long),
        )
        if not candidate_event.candidate_subsets:
            sampled = sample_group_retention_subsets(
                num_tokens=int(candidate_event.score_state.shape[-2]),
                budget=candidate_event.budget,
                num_samples=int(replay_plan.get("num_samples", 8)),
                protected_indices=[] if candidate_event.protected_indices is None else candidate_event.protected_indices.tolist(),
                group_ids=None if candidate_event.group_ids is None else candidate_event.group_ids.tolist(),
                base_scores=None if candidate_event.base_scores is None else candidate_event.base_scores.tolist(),
                seed=int(replay_plan.get("seed", 0)) + len(measured_events),
            )
            candidate_event.candidate_subsets = sampled
        measured_events.append(
            collect_counterfactual_replay_event(
                event=candidate_event,
                runner=runner,
                future_frames=future_frames,
            )
        )
    return build_oracle_events_from_replay_dump(measured_events)


def _load_runner_factory(path: str):
    if ":" not in path:
        raise ValueError("--runner-factory must use module:function format")
    module_name, function_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Runner factory {path} is not callable")
    return factory


def _future_frames_from_plan(replay_plan: dict, event: dict) -> list:
    frames = list(replay_plan.get("frames", []))
    frame_id = int(event.get("frame_id", -1))
    window = int(event.get("oracle_window", replay_plan.get("oracle_window", 4)))
    return [frame for frame in frames if int(frame.get("frame_id", -1)) > frame_id][:window]


def _optional_tensor(event: dict, key: str, dtype: torch.dtype):
    if key not in event or event[key] is None:
        return None
    return torch.as_tensor(event[key], dtype=dtype)


def _candidate_subset_to_indices(subset) -> torch.Tensor:
    if isinstance(subset, dict):
        subset = subset["keep_indices"]
    return torch.as_tensor(subset, dtype=torch.long)


_ORACLE_EVENT_OPTIONAL_TENSOR_KEYS = (
    "base_scores",
    "heuristic_scores",
    "learned_token_scores",
    "group_ids",
    "protected_indices",
    "token_frame_ids",
)


def synthetic_events(num_events: int, num_tokens: int, score_state_dim: int, num_samples: int, seed: int) -> list:
    generator = torch.Generator().manual_seed(seed)
    events = []
    for event_idx in range(num_events):
        score_state = torch.randn(num_tokens, score_state_dim, generator=generator)
        metadata_features = torch.randn(num_tokens, TOKEN_METADATA_FEATURE_DIM, generator=generator)
        oracle_quality = score_state[:, 0] + 0.25 * metadata_features[:, 0]
        subsets = []
        for keep_indices in sample_group_retention_subsets(
            num_tokens=num_tokens,
            budget=max(2, num_tokens // 2),
            num_samples=num_samples,
            protected_indices=[0],
            base_scores=oracle_quality.tolist(),
            seed=seed + event_idx,
        ):
            retained_quality = oracle_quality[keep_indices].mean()
            camera = float((1.0 - retained_quality).abs().item() * 0.01)
            depth = float((1.5 - retained_quality).abs().item() * 0.01)
            point_map = float((2.0 - retained_quality).abs().item() * 0.01)
            loss_components = {"camera": camera, "depth": depth, "point_map": point_map}
            subsets.append(
                {
                    "keep_indices": keep_indices,
                    "loss": weighted_three_task_loss(loss_components),
                    "loss_components": loss_components,
                }
            )
        events.append(
            {
                "event_id": f"synthetic:{event_idx}",
                "layer_id": event_idx % 4,
                "budget": max(2, num_tokens // 2),
                "sequence_provenance": {
                    "dataset_key": "synthetic",
                    "sequence_id": f"synthetic/{event_idx}",
                    "frame_count": 1,
                    "frames": [{"frame_index": 0}],
                },
                "score_state": score_state,
                "metadata_features": metadata_features,
                "subsets": subsets,
            }
        )
    return events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "measured-dump", "synthetic", "replay", "collect-replay"),
        default="auto",
        help=(
            "Generation mode. collect-replay runs a replay runner factory; "
            "replay converts already-collected replay dumps."
        ),
    )
    parser.add_argument("--event-dump", help="JSON or .pt event dump with measured counterfactual losses")
    parser.add_argument("--replay-plan", help="JSON or .pt plan for --mode collect-replay")
    parser.add_argument("--runner-factory", help="module:function returning a CounterfactualReplayRunner")
    parser.add_argument("--output", required=True, help="Output .pt shard path")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-events", type=int, default=0)
    parser.add_argument("--synthetic-tokens", type=int, default=16)
    parser.add_argument("--score-state-dim", type=int, default=128)
    return parser.parse_args(argv)


def resolve_generation_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.synthetic_events > 0:
        return "synthetic"
    if args.event_dump:
        return "measured-dump"
    if getattr(args, "replay_plan", None):
        return "collect-replay"
    raise ValueError("Provide --event-dump, --synthetic-events > 0, or an explicit --mode")


def build_events_from_args(args: argparse.Namespace) -> list:
    mode = resolve_generation_mode(args)
    if mode == "synthetic":
        if args.synthetic_events <= 0:
            raise ValueError("--mode synthetic requires --synthetic-events > 0")
        return synthetic_events(
            num_events=args.synthetic_events,
            num_tokens=args.synthetic_tokens,
            score_state_dim=args.score_state_dim,
            num_samples=args.num_samples,
            seed=args.seed,
        )

    if mode == "measured-dump":
        if not args.event_dump:
            raise ValueError("--mode measured-dump requires --event-dump")
        path = Path(args.event_dump)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as handle:
                event_dump = json.load(handle)
        else:
            event_dump = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(event_dump, dict) and "events" in event_dump:
            event_dump = event_dump["events"]
        return build_oracle_events_from_dump(event_dump, num_samples=args.num_samples, seed=args.seed)

    if mode == "replay":
        if not args.event_dump:
            raise ValueError("--mode replay requires --event-dump")
        path = Path(args.event_dump)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as handle:
                event_dump = json.load(handle)
        else:
            event_dump = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(event_dump, dict) and "events" in event_dump:
            event_dump = event_dump["events"]
        return build_oracle_events_from_replay_dump(event_dump)

    if mode == "collect-replay":
        if not args.replay_plan:
            raise ValueError("--mode collect-replay requires --replay-plan")
        if not args.runner_factory:
            raise ValueError("--mode collect-replay requires --runner-factory")
        replay_plan = _load_structured_file(Path(args.replay_plan))
        if not isinstance(replay_plan, dict):
            raise ValueError("--replay-plan must contain a mapping with events")
        return collect_replay_events_from_plan(replay_plan, args.runner_factory)

    raise ValueError(f"Unsupported oracle generation mode: {mode}")


def _load_structured_file(path: Path):
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    events = build_events_from_args(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ovggt_counterfactual_oracle_v1",
            "task_weights": TASK_WEIGHTS,
            "events": events,
        },
        output,
    )
    print(f"Wrote {len(events)} events to {output}")


if __name__ == "__main__":
    main()
