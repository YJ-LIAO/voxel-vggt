"""Counterfactual replay helpers for TokenScorer oracle collection.

The collector is intentionally runner-based: production OVGGT replay can supply
snapshot/restore/apply/replay methods, while unit tests use a tiny fake runner.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import torch
from torch import Tensor


TASK_WEIGHTS = {"camera": 20.0, "depth": 20.0, "point_map": 10.0}


def sample_group_retention_subsets(
    num_tokens: int,
    budget: int,
    num_samples: int,
    protected_indices: Iterable[int] = (),
    group_ids: Iterable[int] | None = None,
    base_scores: Iterable[float] | None = None,
    seed: int = 0,
) -> list[Tensor]:
    """Sample retention candidates at group granularity.

    The returned keep sets always honor ``budget``. Protected tokens are kept
    first; if they overflow the budget, the most recent protected indices are
    retained, matching the cache overflow policy used by learned eviction.
    """
    generator = torch.Generator().manual_seed(int(seed))
    num_tokens = max(int(num_tokens), 0)
    budget = max(int(budget), 0)
    num_samples = max(int(num_samples), 0)
    protected = torch.as_tensor(list(protected_indices), dtype=torch.long)
    protected = protected[(protected >= 0) & (protected < num_tokens)].unique(sorted=True)
    if budget <= 0 or num_tokens <= 0:
        return [torch.empty(0, dtype=torch.long) for _ in range(num_samples)]
    if protected.numel() >= budget:
        keep = protected[-budget:].clone()
        return [keep for _ in range(num_samples)]

    candidate_budget = budget - int(protected.numel())
    protected_set = set(protected.tolist())
    candidates = torch.tensor(
        [idx for idx in range(num_tokens) if idx not in protected_set],
        dtype=torch.long,
    )
    if candidate_budget <= 0 or candidates.numel() == 0:
        return [protected.clone() for _ in range(num_samples)]

    if base_scores is None:
        scores = torch.rand(num_tokens, generator=generator)
    else:
        scores = torch.as_tensor(list(base_scores), dtype=torch.float32)
        if scores.numel() != num_tokens:
            scores = torch.zeros(num_tokens, dtype=torch.float32)
    if group_ids is None:
        groups = candidates.clone()
    else:
        raw_groups = torch.as_tensor(list(group_ids), dtype=torch.long)
        if raw_groups.numel() != num_tokens:
            raw_groups = torch.arange(num_tokens, dtype=torch.long)
        groups = raw_groups[candidates]

    subsets = []
    unique_groups = torch.unique(groups)
    for sample_idx in range(num_samples):
        jitter = 0.01 * torch.rand(candidates.numel(), generator=generator)
        if sample_idx % 2 == 0:
            order = torch.argsort(scores[candidates] + jitter, descending=True)
        else:
            group_order = unique_groups[torch.randperm(unique_groups.numel(), generator=generator)]
            order_parts = []
            for group in group_order:
                group_local = torch.nonzero(groups == group, as_tuple=False).squeeze(-1)
                group_scores = scores[candidates[group_local]] + jitter[group_local]
                order_parts.append(group_local[torch.argsort(group_scores, descending=True)])
            order = torch.cat(order_parts, dim=0) if order_parts else torch.empty(0, dtype=torch.long)
        selected = candidates[order[:candidate_budget]].sort().values
        subsets.append(torch.cat([protected, selected]).unique(sorted=True))
    return subsets


class CounterfactualReplayRunner(Protocol):
    def snapshot(self): ...

    def restore(self, snapshot) -> None: ...

    def apply_keep_indices(self, layer_id: int, keep_indices: Tensor) -> None: ...

    def replay_future_window(self, start_frame_idx: int, future_frames: Sequence[dict]) -> Sequence[dict]: ...

    def targets_for_future_window(self, future_frames: Sequence[dict]) -> Sequence[dict]: ...


class OVGGTCacheReplayRunner:
    """Base runner for OVGGT frontend-cache counterfactual replay.

    Subclasses are responsible for advancing the frozen model to the event
    boundary and for replaying future frames. This base class handles the common
    cache snapshot/restore and per-layer keep-index application.
    """

    def __init__(self, cache_states: Sequence) -> None:
        self.cache_states = list(cache_states)

    def prepare_event(self, event: "CounterfactualReplayCandidateEvent") -> None:
        return None

    def snapshot(self):
        return copy.deepcopy(self.cache_states)

    def restore(self, snapshot) -> None:
        for idx, snapshot_state in enumerate(snapshot):
            if idx >= len(self.cache_states):
                self.cache_states.append(copy.deepcopy(snapshot_state))
            else:
                _restore_layer_cache_state_(self.cache_states[idx], snapshot_state)
        if len(self.cache_states) > len(snapshot):
            del self.cache_states[len(snapshot):]

    def apply_keep_indices(self, layer_id: int, keep_indices: Tensor) -> None:
        if layer_id < 0 or layer_id >= len(self.cache_states):
            raise IndexError(f"layer_id {layer_id} out of range for {len(self.cache_states)} cache states")
        indices = torch.as_tensor(keep_indices, dtype=torch.long)
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        self.cache_states[layer_id].gather_(indices)

    def replay_future_window(self, start_frame_idx: int, future_frames: Sequence[dict]) -> Sequence[dict]:
        raise NotImplementedError

    def targets_for_future_window(self, future_frames: Sequence[dict]) -> Sequence[dict]:
        raise NotImplementedError


@dataclass
class CounterfactualReplayCandidateEvent:
    event_id: str
    layer_id: int
    frame_id: int
    budget: int
    sequence_provenance: dict | None
    score_state: Tensor
    metadata_features: Tensor
    candidate_subsets: Sequence[Tensor]
    protected_indices: Tensor | None = None
    base_scores: Tensor | None = None
    learned_token_scores: Tensor | None = None
    group_ids: Tensor | None = None
    token_frame_ids: Tensor | None = None


def collect_counterfactual_replay_event(
    event: CounterfactualReplayCandidateEvent,
    runner: CounterfactualReplayRunner,
    future_frames: Sequence[dict],
) -> dict:
    """Evaluate sampled keep subsets by restoring the same cache snapshot.

    The returned event is a measured replay dump record: each subset contains
    keep indices, future-window predictions/targets, unweighted task losses, and
    the weighted total loss used by the ranking dataset.
    """
    prepare_event = getattr(runner, "prepare_event", None)
    if callable(prepare_event):
        prepare_event(event)
    snapshot = runner.snapshot()
    targets = list(runner.targets_for_future_window(future_frames))
    start_frame_idx = int(future_frames[0]["frame_id"]) if future_frames else int(event.frame_id) + 1

    subsets = []
    try:
        for keep_indices in event.candidate_subsets:
            keep_indices = torch.as_tensor(keep_indices, dtype=torch.long).reshape(-1)
            runner.restore(snapshot)
            runner.apply_keep_indices(int(event.layer_id), keep_indices)
            predictions = list(runner.replay_future_window(start_frame_idx, future_frames))
            loss_components = compute_three_task_loss_components(predictions, targets)
            subsets.append(
                {
                    "keep_indices": keep_indices.detach().cpu(),
                    "replay": {
                        "predictions": _detach_tensor_tree(predictions),
                        "targets": _detach_tensor_tree(targets),
                    },
                    "loss_components": loss_components,
                    "loss": weighted_three_task_loss(loss_components),
                }
            )
    finally:
        runner.restore(snapshot)

    output = {
        "event_id": event.event_id,
        "layer_id": int(event.layer_id),
        "frame_id": int(event.frame_id),
        "budget": int(event.budget),
        "sequence_provenance": event.sequence_provenance,
        "score_state": _squeeze_single_batch_matrix(event.score_state).detach().cpu(),
        "metadata_features": _squeeze_single_batch_matrix(event.metadata_features).detach().cpu(),
        "subsets": subsets,
    }
    for key in (
        "protected_indices",
        "base_scores",
        "learned_token_scores",
        "group_ids",
        "token_frame_ids",
    ):
        value = getattr(event, key)
        if value is not None:
            output[key] = torch.as_tensor(value).detach().cpu()
    return output


def compute_three_task_loss_components(predictions: Iterable[dict], targets: Iterable[dict]) -> dict:
    predictions = list(predictions)
    targets = list(targets)
    if len(predictions) != len(targets):
        raise ValueError(
            "future-window length mismatch: "
            f"{len(predictions)} predictions vs {len(targets)} targets"
        )
    camera_terms = []
    depth_terms = []
    point_terms = []
    for pred, target in zip(predictions, targets):
        if "camera_pose" in pred and "camera_pose" in target:
            pred_pose = torch.as_tensor(pred["camera_pose"], dtype=torch.float32)
            target_pose = torch.as_tensor(target["camera_pose"], dtype=torch.float32)
            camera_terms.append((pred_pose - target_pose).abs().mean())

        if "depth" in pred and "depth" in target:
            pred_depth = torch.as_tensor(pred["depth"], dtype=torch.float32)
            target_depth = torch.as_tensor(target["depth"], dtype=torch.float32)
            valid_mask = _valid_mask_for(target.get("valid_mask"), target_depth)
            depth_terms.append(_masked_mean((pred_depth - target_depth).abs(), valid_mask))

        pred_pmap = pred.get("pts3d_in_other_view", pred.get("point_map"))
        target_pmap = target.get("pts3d_in_other_view", target.get("point_map"))
        if pred_pmap is not None and target_pmap is not None:
            pred_points = torch.as_tensor(pred_pmap, dtype=torch.float32)
            target_points = torch.as_tensor(target_pmap, dtype=torch.float32)
            point_error = (pred_points - target_points).abs().mean(dim=-1)
            valid_mask = _valid_mask_for(target.get("valid_mask"), point_error)
            point_terms.append(_masked_mean(point_error, valid_mask))

    return {
        "camera": _mean_scalar_terms(camera_terms),
        "depth": _mean_scalar_terms(depth_terms),
        "point_map": _mean_scalar_terms(point_terms),
    }


def weighted_three_task_loss(loss_components: dict) -> float:
    return sum(float(loss_components.get(name, 0.0)) * weight for name, weight in TASK_WEIGHTS.items())


def _mean_scalar_terms(terms: list[Tensor]) -> float:
    if not terms:
        return 0.0
    return float(torch.stack([term.float() for term in terms]).mean().item())


def _masked_mean(values: Tensor, valid_mask: Tensor) -> Tensor:
    valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
    while valid_mask.dim() < values.dim():
        valid_mask = valid_mask.unsqueeze(-1)
    valid_mask = valid_mask.expand_as(values)
    if not valid_mask.any():
        return values.new_zeros(())
    return values[valid_mask].mean()


def _valid_mask_for(valid_mask, reference: Tensor) -> Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    while mask.dim() > reference.dim():
        mask = mask.squeeze(-1)
    return mask


def _squeeze_single_batch_matrix(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor


def _detach_tensor_tree(payload):
    if isinstance(payload, Tensor):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {key: _detach_tensor_tree(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_detach_tensor_tree(value) for value in payload]
    return payload


def _restore_layer_cache_state_(target, snapshot) -> None:
    for key, value in snapshot.__dict__.items():
        setattr(target, key, copy.deepcopy(value))


@dataclass
class DedupCounterfactualEvent:
    """Event for dedup counterfactual evaluation.

    Records a voxel group with multiple tokens and candidate keep-one strategies.
    """
    event_id: str
    layer_id: int
    frame_id: int
    voxel_group_id: int
    sequence_provenance: dict | None
    score_state: Tensor
    metadata_features: Tensor
    candidate_subsets: Sequence[dict]
    base_scores: Tensor | None = None


def collect_dedup_counterfactual_event(
    event: DedupCounterfactualEvent,
    runner: CounterfactualReplayRunner,
    future_frames: Sequence[dict],
) -> dict:
    """Evaluate dedup candidate subsets by restoring cache and replaying future.

    For dedup events, each candidate subset specifies which token to keep in a
    voxel group. The runner applies the keep set and replays future frames.
    """
    prepare_event = getattr(runner, "prepare_event", None)
    if callable(prepare_event):
        prepare_event(event)
    snapshot = runner.snapshot()
    targets = list(runner.targets_for_future_window(future_frames))
    start_frame_idx = int(future_frames[0]["frame_id"]) if future_frames else int(event.frame_id) + 1

    subsets = []
    try:
        for candidate in event.candidate_subsets:
            keep_indices = torch.as_tensor(candidate["keep_indices"], dtype=torch.long).reshape(-1)
            runner.restore(snapshot)
            runner.apply_keep_indices(int(event.layer_id), keep_indices)
            predictions = list(runner.replay_future_window(start_frame_idx, future_frames))
            loss_components = compute_three_task_loss_components(predictions, targets)
            subsets.append(
                {
                    "keep_indices": keep_indices.detach().cpu(),
                    "keep_index": candidate.get("keep_index"),
                    "evict_indices": candidate.get("evict_indices"),
                    "loss_components": loss_components,
                    "loss": weighted_three_task_loss(loss_components),
                }
            )
    finally:
        runner.restore(snapshot)

    output = {
        "event_id": event.event_id,
        "event_type": "dedup",
        "layer_id": int(event.layer_id),
        "frame_id": int(event.frame_id),
        "voxel_group_id": int(event.voxel_group_id),
        "sequence_provenance": event.sequence_provenance,
        "score_state": _squeeze_single_batch_matrix(event.score_state).detach().cpu(),
        "metadata_features": _squeeze_single_batch_matrix(event.metadata_features).detach().cpu(),
        "subsets": subsets,
    }
    if event.base_scores is not None:
        output["base_scores"] = torch.as_tensor(event.base_scores).detach().cpu()
    return output


def _copy_subset_metadata(subset: dict) -> dict:
    """Copy subset-level metadata keys, detaching tensors and deep-copying other values."""
    copied = {}
    for key in ("strategy", "source", "keep_count", "demoted_keep_indices", "keep_index", "evict_indices"):
        if key in subset:
            value = subset[key]
            copied[key] = value.detach().cpu() if isinstance(value, Tensor) else copy.deepcopy(value)
    return copied


@dataclass
class FifoTopKCounterfactualEvent:
    """Event for FIFO top-K counterfactual evaluation.

    Records the demoted slot tokens and candidate top-K selection strategies.
    """
    event_id: str
    layer_id: int
    frame_id: int
    demoted_slot: int
    sequence_provenance: dict | None
    score_state: Tensor
    metadata_features: Tensor
    candidate_subsets: Sequence[dict]
    keep_count: int | None = None
    base_scores: Tensor | None = None


def collect_fifo_topk_counterfactual_event(
    event: FifoTopKCounterfactualEvent,
    runner: CounterfactualReplayRunner,
    future_frames: Sequence[dict],
) -> dict:
    """Evaluate FIFO top-K candidate subsets by restoring cache and replaying future.

    For FIFO events, each candidate subset specifies which K tokens from the
    demoted slot to promote to slot 0. The runner applies the keep set and
    replays future frames.
    """
    prepare_event = getattr(runner, "prepare_event", None)
    if callable(prepare_event):
        prepare_event(event)
    snapshot = runner.snapshot()
    targets = list(runner.targets_for_future_window(future_frames))
    start_frame_idx = int(future_frames[0]["frame_id"]) if future_frames else int(event.frame_id) + 1

    subsets = []
    try:
        for candidate in event.candidate_subsets:
            keep_indices = torch.as_tensor(candidate["keep_indices"], dtype=torch.long).reshape(-1)
            runner.restore(snapshot)
            runner.apply_keep_indices(int(event.layer_id), keep_indices)
            predictions = list(runner.replay_future_window(start_frame_idx, future_frames))
            loss_components = compute_three_task_loss_components(predictions, targets)
            subsets.append(
                {
                    **_copy_subset_metadata(candidate),
                    "keep_count": int(candidate.get("keep_count", event.keep_count or 0)),
                    "keep_indices": keep_indices.detach().cpu(),
                    "loss_components": loss_components,
                    "loss": weighted_three_task_loss(loss_components),
                }
            )
    finally:
        runner.restore(snapshot)

    output = {
        "event_id": event.event_id,
        "event_type": "fifo_topk",
        "layer_id": int(event.layer_id),
        "frame_id": int(event.frame_id),
        "demoted_slot": int(event.demoted_slot),
        "keep_count": int(event.keep_count or 0),
        "sequence_provenance": event.sequence_provenance,
        "score_state": _squeeze_single_batch_matrix(event.score_state).detach().cpu(),
        "metadata_features": _squeeze_single_batch_matrix(event.metadata_features).detach().cpu(),
        "subsets": subsets,
    }
    if event.base_scores is not None:
        output["base_scores"] = torch.as_tensor(event.base_scores).detach().cpu()
    return output
