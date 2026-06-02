"""Production helpers for collecting TokenScorer counterfactual oracle shards.

The collector uses the same frontend finetuning config that defines the mixed
training dataset. It runs a frozen OVGGT model, records eviction candidates at
cache commit time, then replays short future windows with sampled keep sets.
"""

from __future__ import annotations

import copy
import importlib
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

import torch
from omegaconf import OmegaConf

from ovggt.models.ovggt import OVGGT
from ovggt.training.counterfactual_replay import (
    TASK_WEIGHTS,
    compute_three_task_loss_components,
    sample_group_retention_subsets,
    weighted_three_task_loss,
)
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.geometry import closed_form_inverse_se3
from ovggt.utils.pose_enc import ABS_POSE_ENCODING, world_to_camera_to_pose_encoding


@dataclass
class FrontendOracleCollectorConfig:
    config: str
    output: str
    dataset_key: str = "train_dataset"
    batch_size: int | None = 1
    num_workers: int | None = 0
    max_batches: int = 1
    max_events: int = 64
    num_samples: int = 8
    oracle_window: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    teacher_checkpoint: str | None = None
    student_checkpoint: str | None = None
    high_budget_teacher: bool = True
    flush_every_events: int = 16
    flush_every_batches: int = 1
    log_every_subsets: int = 1
    subset_replay_batch_size: int = 1
    layers_per_frame: int = 0
    max_events_per_sequence: int = 0
    num_views: int | None = None
    max_fetch_errors: int = 256
    store_replay_payload: bool = False


def default_oracle_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[oracle {timestamp}] {message}", flush=True)


class OracleShardFlusher:
    """Incrementally persists partial oracle shards during long collection."""

    def __init__(
        self,
        base_shard: dict,
        output_path: str | Path,
        flush_every_events: int = 16,
        flush_every_batches: int = 1,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.base_shard = dict(base_shard)
        self.output_path = Path(output_path)
        self.flush_every_events = max(int(flush_every_events), 0)
        self.flush_every_batches = max(int(flush_every_batches), 0)
        self.log_fn = log_fn
        self.last_event_count = 0
        self.last_batch_idx = -1

    def maybe_flush(
        self,
        events: Sequence[dict],
        batch_idx: int | None,
        force: bool = False,
        reason: str = "",
    ) -> bool:
        started = time.monotonic()
        event_count = len(events)
        batch_idx_int = -1 if batch_idx is None else int(batch_idx)
        should_flush = bool(force)
        if not should_flush and self.flush_every_events > 0:
            should_flush = event_count - self.last_event_count >= self.flush_every_events
        if not should_flush and self.flush_every_batches > 0 and batch_idx is not None:
            should_flush = batch_idx_int - self.last_batch_idx >= self.flush_every_batches
        if not should_flush:
            return False

        shard = dict(self.base_shard)
        shard["events"] = list(events)
        shard["partial"] = not bool(force)
        shard["num_events"] = event_count
        shard["flush_reason"] = str(reason or ("final" if force else "periodic"))
        shard["flushed_at"] = datetime.now().isoformat(timespec="seconds")
        save_oracle_shard(shard, self.output_path)
        self.last_event_count = event_count
        self.last_batch_idx = batch_idx_int
        if self.log_fn is not None:
            state = "final shard" if force else "partial shard"
            self.log_fn(
                f"flushed {state}: events={event_count} path={self.output_path} "
                f"reason={shard['flush_reason']}"
            )
            self.log_fn(
                "timing phase=flush "
                f"events={event_count} batch_idx={batch_idx_int} "
                f"force={int(bool(force))} elapsed_sec={time.monotonic() - started:.3f}"
            )
        return True


class CounterfactualEvictionProbe:
    """Records cache state just before budget eviction.

    The probe is deliberately lightweight: it stores CPU copies of token state
    and sampled keep subsets. Full model replay is handled by the collector.
    """

    def __init__(
        self,
        num_samples: int = 8,
        oracle_window: int = 4,
        seed: int = 0,
        event_prefix: str = "oracle",
        max_events: int | None = None,
        sequence_provenance: dict[int, dict] | None = None,
        layers_per_frame: int = 0,
        num_layers: int | None = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.oracle_window = int(oracle_window)
        self.seed = int(seed)
        self.event_prefix = str(event_prefix)
        self.max_events = None if max_events is None else int(max_events)
        self.sequence_provenance = dict(sequence_provenance or {})
        self.layers_per_frame = int(layers_per_frame)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.events: list[dict] = []

    def on_eviction_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        budget: int,
        batch_index: int = 0,
    ) -> None:
        if self.max_events is not None and len(self.events) >= self.max_events:
            return
        if not should_record_oracle_layer(
            layer_id=layer_id,
            frame_id=frame_id,
            layers_per_frame=self.layers_per_frame,
            num_layers=self.num_layers,
        ):
            return
        if cache_state.score_state is None or cache_state.metadata is None:
            return
        num_tokens = int(cache_state.num_tokens())
        budget = int(budget)
        if num_tokens <= budget:
            return

        protected_count = min(max(int(cache_state.protected_count), 0), num_tokens)
        protected_indices = torch.arange(protected_count, dtype=torch.long)
        score_state = _select_cache_batch(cache_state.score_state).detach().cpu().float()
        metadata_features = _select_batch(
            cache_state.build_scorer_metadata_features(current_frame_id=int(frame_id)),
            0,
        ).detach().cpu().float()
        base_scores = _select_cache_batch(cache_state.metadata.importance).detach().cpu().float()
        token_frame_ids = _select_cache_batch(cache_state.metadata.frame_id).detach().cpu().long()
        group_ids = self._group_ids_for_cache(cache_state, batch_index=batch_index, frame_id=int(frame_id))
        candidate_subsets = sample_group_retention_subsets(
            num_tokens=num_tokens,
            budget=budget,
            num_samples=self.num_samples,
            protected_indices=protected_indices.tolist(),
            group_ids=group_ids.tolist() if group_ids is not None else None,
            base_scores=base_scores.tolist(),
            seed=self.seed + len(self.events),
        )
        candidate_subsets = deduplicate_keep_subsets(candidate_subsets)
        if len(candidate_subsets) < 2:
            return

        event_id = f"{self.event_prefix}:b{batch_index}:f{int(frame_id)}:l{int(layer_id)}:e{len(self.events)}"
        sequence_provenance = self.sequence_provenance.get(int(batch_index))
        self.events.append(
            {
                "event_id": event_id,
                "layer_id": int(layer_id),
                "frame_id": int(frame_id),
                "batch_index": int(batch_index),
                "budget": budget,
                "score_state": score_state,
                "metadata_features": metadata_features,
                "candidate_subsets": [
                    {"keep_indices": subset.detach().cpu().long()}
                    for subset in candidate_subsets
                ],
                "protected_indices": protected_indices,
                "base_scores": base_scores,
                "group_ids": group_ids.detach().cpu().long() if group_ids is not None else None,
                "token_frame_ids": token_frame_ids,
                "sequence_provenance": copy.deepcopy(sequence_provenance) if sequence_provenance is not None else None,
            }
        )

    @staticmethod
    def _group_ids_for_cache(cache_state, batch_index: int, frame_id: int) -> torch.Tensor | None:
        metadata = cache_state.metadata
        if metadata is None:
            return None
        active_xyz = cache_state._project_slot_local_xyz_to_active(
            metadata.slot_local_xyz,
            metadata.slot_id,
        )
        xyz = _select_cache_batch(active_xyz)
        frame_ids = _select_cache_batch(metadata.frame_id).long()
        xyz_valid = torch.isfinite(xyz).all(dim=-1)
        voxel = torch.floor(torch.nan_to_num(xyz, nan=0.0) / 0.25).long()
        hashed = (
            voxel[:, 0]
            + 4096 * voxel[:, 1]
            + 4096 * 4096 * voxel[:, 2]
            + 17 * frame_ids
        )
        fallback = torch.arange(xyz.shape[0], dtype=torch.long, device=xyz.device)
        return torch.where(xyz_valid, hashed, fallback + (int(frame_id) + 1) * 100000000)

    def clear(self) -> None:
        self.events.clear()


class CounterfactualDedupProbe:
    """Records cache state when voxel dedup is about to discard tokens.

    For each voxel group that has multiple tokens, it samples different
    keep-one strategies and records them for counterfactual replay.
    """

    def __init__(
        self,
        num_samples: int = 8,
        oracle_window: int = 4,
        seed: int = 0,
        event_prefix: str = "dedup",
        max_events: int | None = None,
        sequence_provenance: dict[int, dict] | None = None,
        layers_per_frame: int = 0,
        num_layers: int | None = None,
        voxel_size: float = 0.25,
    ) -> None:
        self.num_samples = int(num_samples)
        self.oracle_window = int(oracle_window)
        self.seed = int(seed)
        self.event_prefix = str(event_prefix)
        self.max_events = None if max_events is None else int(max_events)
        self.sequence_provenance = dict(sequence_provenance or {})
        self.layers_per_frame = int(layers_per_frame)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.voxel_size = float(voxel_size)
        self.events: list[dict] = []

    def on_dedup_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        batch_index: int = 0,
        scores: torch.Tensor | None = None,
        policy_keep_indices: torch.Tensor | None = None,
    ) -> None:
        """Called during apply_voxel_dedup_ when duplicate voxel groups are found."""
        self._last_scores = scores
        self._last_policy_keep_indices = policy_keep_indices
        if self.max_events is not None and len(self.events) >= self.max_events:
            return
        if not should_record_oracle_layer(
            layer_id=layer_id,
            frame_id=frame_id,
            layers_per_frame=self.layers_per_frame,
            num_layers=self.num_layers,
        ):
            return
        if cache_state.score_state is None or cache_state.metadata is None:
            return
        num_tokens = int(cache_state.num_tokens())
        if num_tokens == 0:
            return

        metadata = cache_state.metadata
        projected_xyz = cache_state._project_slot_local_xyz_to_active(
            metadata.slot_local_xyz,
            metadata.slot_id,
        )
        # Select batch 0 for analysis
        xyz = _select_cache_batch(projected_xyz)
        xyz_valid = torch.isfinite(xyz).all(dim=-1)
        patch_mask = _select_cache_batch(metadata.token_kind) == 2  # PATCH kind

        if not xyz_valid.any() or not patch_mask.any():
            return

        # Build voxel groups
        voxel = torch.floor(torch.nan_to_num(xyz, nan=0.0) / self.voxel_size).long()
        voxel_hash = (
            voxel[:, 0]
            + 4096 * voxel[:, 1]
            + 4096 * 4096 * voxel[:, 2]
        )
        # Only consider valid patch tokens
        valid_patch = xyz_valid & patch_mask
        valid_hashes = voxel_hash[valid_patch]
        if valid_hashes.numel() == 0:
            return

        # Find voxel groups with >1 token
        unique_hashes, counts = torch.unique(valid_hashes, return_counts=True)
        multi_token_hashes = unique_hashes[counts > 1]
        if multi_token_hashes.numel() == 0:
            return

        score_state = _select_cache_batch(cache_state.score_state).detach().cpu().float()
        metadata_features = _select_batch(
            cache_state.build_scorer_metadata_features(current_frame_id=int(frame_id)),
            0,
        ).detach().cpu().float()
        base_scores = _select_cache_batch(metadata.importance).detach().cpu().float()

        # For each multi-token voxel, create candidate subsets
        for group_hash in multi_token_hashes[:3]:  # Limit to first 3 groups
            group_mask = valid_patch & (voxel_hash == group_hash.item())
            group_indices = torch.nonzero(group_mask, as_tuple=False).squeeze(-1)

            if group_indices.numel() < 2:
                continue

            # Sample candidate subsets: keep one, evict rest
            candidate_subsets = []
            # Always include all-keep and one-evict-each strategies
            for keep_idx in group_indices.tolist():
                evict_indices = [idx for idx in group_indices.tolist() if idx != keep_idx]
                candidate_subsets.append({
                    "keep_index": keep_idx,
                    "evict_indices": evict_indices,
                    "keep_indices": torch.tensor(
                        [idx for idx in range(num_tokens) if idx not in evict_indices],
                        dtype=torch.long,
                    ),
                })

            if len(candidate_subsets) < 2:
                continue

            event_id = (
                f"{self.event_prefix}:b{batch_index}:f{int(frame_id)}:l{int(layer_id)}"
                f":voxel{int(group_hash.item())}:e{len(self.events)}"
            )
            sequence_provenance = self.sequence_provenance.get(int(batch_index))
            self.events.append({
                "event_id": event_id,
                "event_type": "dedup",
                "layer_id": int(layer_id),
                "frame_id": int(frame_id),
                "batch_index": int(batch_index),
                "voxel_group_id": int(group_hash.item()),
                "score_state": score_state,
                "metadata_features": metadata_features,
                "candidate_subsets": [
                    {
                        "keep_index": cs["keep_index"],
                        "evict_indices": cs["evict_indices"],
                        "keep_indices": cs["keep_indices"].detach().cpu().long(),
                    }
                    for cs in candidate_subsets
                ],
                "base_scores": base_scores,
                "sequence_provenance": copy.deepcopy(sequence_provenance) if sequence_provenance is not None else None,
            })

    def clear(self) -> None:
        self.events.clear()


class CounterfactualFifoTopKProbe:
    """Records cache state when FIFO top-K protection is about to select tokens.

    Samples multiple top-K candidate sets for counterfactual replay.
    """

    def __init__(
        self,
        num_samples: int = 8,
        oracle_window: int = 4,
        seed: int = 0,
        event_prefix: str = "fifo_topk",
        max_events: int | None = None,
        sequence_provenance: dict[int, dict] | None = None,
        layers_per_frame: int = 0,
        num_layers: int | None = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.oracle_window = int(oracle_window)
        self.seed = int(seed)
        self.event_prefix = str(event_prefix)
        self.max_events = None if max_events is None else int(max_events)
        self.sequence_provenance = dict(sequence_provenance or {})
        self.layers_per_frame = int(layers_per_frame)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.events: list[dict] = []

    def on_fifo_topk_candidate(
        self,
        cache_state,
        demoted_slot: int,
        keep_count: int,
        layer_id: int,
        frame_id: int,
        batch_index: int = 0,
    ) -> None:
        """Called during protect_topk_on_demotion_ when selecting top-K."""
        if self.max_events is not None and len(self.events) >= self.max_events:
            return
        if not should_record_oracle_layer(
            layer_id=layer_id,
            frame_id=frame_id,
            layers_per_frame=self.layers_per_frame,
            num_layers=self.num_layers,
        ):
            return
        if cache_state.score_state is None or cache_state.metadata is None:
            return

        metadata = cache_state.metadata
        b_idx = batch_index
        slot_mask = metadata.anchor_slot[b_idx] == demoted_slot
        indices = torch.nonzero(slot_mask, as_tuple=False).squeeze(-1)

        # Compute all indices NOT in the demoted slot — these are always retained.
        non_slot_mask = metadata.anchor_slot[b_idx] != demoted_slot
        non_slot_indices = torch.nonzero(non_slot_mask, as_tuple=False).squeeze(-1)

        num_slot_tokens = indices.numel()
        if num_slot_tokens <= keep_count:
            return

        slot_score_state = cache_state.score_state[b_idx, indices].detach().cpu().float()
        full_metadata_features = cache_state.build_scorer_metadata_features(current_frame_id=int(frame_id))
        slot_metadata_features = full_metadata_features[b_idx, indices].detach().cpu().float()
        slot_importance = metadata.importance[b_idx, indices].detach().cpu().float()
        num_tokens = int(cache_state.num_tokens())

        # Sample candidate top-K subsets
        generator = torch.Generator().manual_seed(self.seed + len(self.events))
        candidate_subsets = []

        # Strategy 1: top-K by importance (current heuristic)
        _, top_by_importance = torch.topk(slot_importance, k=keep_count)
        keep_imp_indices = indices[top_by_importance].sort().values
        full_keep_imp = torch.cat([non_slot_indices, keep_imp_indices]).sort().values

        # Strategy 2: random K
        perm = torch.randperm(num_slot_tokens, generator=generator)[:keep_count]
        keep_rand_indices = indices[perm].sort().values
        full_keep_rand = torch.cat([non_slot_indices, keep_rand_indices]).sort().values

        candidate_subsets.append({
            "strategy": "top_importance",
            "keep_indices": full_keep_imp.detach().cpu().long(),
        })
        candidate_subsets.append({
            "strategy": "random",
            "keep_indices": full_keep_rand.detach().cpu().long(),
        })

        # Strategy 3+: random permutations with jitter
        for _ in range(min(self.num_samples - 2, 6)):
            jitter = 0.01 * torch.rand(num_slot_tokens, generator=generator)
            order = torch.argsort(slot_importance + jitter, descending=True)
            keep_jitter_indices = indices[order[:keep_count]].sort().values
            full_keep_jitter = torch.cat([non_slot_indices, keep_jitter_indices]).sort().values
            candidate_subsets.append({
                "strategy": "jittered_importance",
                "keep_indices": full_keep_jitter.detach().cpu().long(),
            })

        if len(candidate_subsets) < 2:
            return

        event_id = (
            f"{self.event_prefix}:b{batch_index}:f{int(frame_id)}:l{int(layer_id)}"
            f":slot{demoted_slot}:e{len(self.events)}"
        )
        sequence_provenance = self.sequence_provenance.get(int(batch_index))
        self.events.append({
            "event_id": event_id,
            "event_type": "fifo_topk",
            "layer_id": int(layer_id),
            "frame_id": int(frame_id),
            "batch_index": int(batch_index),
            "demoted_slot": demoted_slot,
            "keep_count": keep_count,
            "score_state": cache_state.score_state[b_idx].detach().cpu().float(),
            "metadata_features": full_metadata_features[b_idx].detach().cpu().float(),
            "candidate_subsets": candidate_subsets,
            "base_scores": slot_importance,
            "sequence_provenance": copy.deepcopy(sequence_provenance) if sequence_provenance is not None else None,
        })

    def clear(self) -> None:
        self.events.clear()


class ReplayKeepSetProbe:
    """Forces one cached layer to use a sampled keep set during replay."""

    def __init__(self, target_event: dict, keep_indices: torch.Tensor | Iterable[int]) -> None:
        self.target_layer_id = int(target_event["layer_id"])
        self.target_frame_id = int(target_event["frame_id"])
        self.target_batch_index = int(target_event.get("batch_index", 0))
        self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)
        self.applied = False

    def on_eviction_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        budget: int,
        batch_index: int = 0,
    ):
        if self.applied:
            return None
        if int(layer_id) != self.target_layer_id or int(frame_id) != self.target_frame_id:
            return None
        if int(batch_index) != self.target_batch_index:
            return None
        self.applied = True
        return self.keep_indices.to(device=cache_state.k.device if cache_state.k is not None else "cpu")


class MultiReplayKeepSetProbe:
    """Applies one sampled keep set per replicated batch item during replay."""

    def __init__(self, target_event: dict, keep_indices_batch: Sequence[torch.Tensor | Iterable[int]]) -> None:
        self.target_layer_id = int(target_event["layer_id"])
        self.target_frame_id = int(target_event["frame_id"])
        self.keep_indices_batch = [
            torch.as_tensor(keep_indices, dtype=torch.long).reshape(-1)
            for keep_indices in keep_indices_batch
        ]
        self.applied = [False for _ in self.keep_indices_batch]

    @property
    def applied_count(self) -> int:
        return sum(1 for item in self.applied if item)

    def on_eviction_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        budget: int,
        batch_index: int = 0,
    ):
        if int(layer_id) != self.target_layer_id or int(frame_id) != self.target_frame_id:
            return None
        batch_index = int(batch_index)
        if batch_index < 0 or batch_index >= len(self.keep_indices_batch):
            return None
        if self.applied[batch_index]:
            return None
        self.applied[batch_index] = True
        device = cache_state.k.device if getattr(cache_state, "k", None) is not None else "cpu"
        return self.keep_indices_batch[batch_index].to(device=device)


class ReplayDedupKeepSetProbe:
    """Forces one cached layer to use a specific keep set during dedup replay."""

    def __init__(self, target_event: dict, keep_indices: torch.Tensor | Iterable[int]) -> None:
        self.target_layer_id = int(target_event["layer_id"])
        self.target_frame_id = int(target_event["frame_id"])
        self.target_batch_index = int(target_event.get("batch_index", 0))
        self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)
        self.applied = False

    def on_dedup_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        batch_index: int = 0,
        scores: torch.Tensor | None = None,
        policy_keep_indices: torch.Tensor | None = None,
    ):
        """Called by apply_voxel_dedup_. Return keep_indices to override dedup result."""
        if self.applied:
            return None
        if int(layer_id) != self.target_layer_id or int(frame_id) != self.target_frame_id:
            return None
        if int(batch_index) != self.target_batch_index:
            return None
        self.applied = True
        return self.keep_indices.to(device=cache_state.k.device if cache_state.k is not None else "cpu")


class MultiReplayDedupKeepSetProbe:
    """Applies one sampled dedup keep set per replicated batch item during replay."""

    def __init__(self, target_event: dict, keep_indices_batch: Sequence[torch.Tensor | Iterable[int]]) -> None:
        self.target_layer_id = int(target_event["layer_id"])
        self.target_frame_id = int(target_event["frame_id"])
        self.keep_indices_batch = [
            torch.as_tensor(keep_indices, dtype=torch.long).reshape(-1)
            for keep_indices in keep_indices_batch
        ]
        self.applied = [False for _ in self.keep_indices_batch]

    @property
    def applied_count(self) -> int:
        return sum(1 for item in self.applied if item)

    def on_dedup_candidate(
        self,
        cache_state,
        layer_id: int,
        frame_id: int,
        batch_index: int = 0,
        scores: torch.Tensor | None = None,
        policy_keep_indices: torch.Tensor | None = None,
    ):
        """Called by apply_voxel_dedup_. Return keep_indices to override dedup result."""
        if int(layer_id) != self.target_layer_id or int(frame_id) != self.target_frame_id:
            return None
        batch_index = int(batch_index)
        if batch_index < 0 or batch_index >= len(self.keep_indices_batch):
            return None
        if self.applied[batch_index]:
            return None
        self.applied[batch_index] = True
        device = cache_state.k.device if getattr(cache_state, "k", None) is not None else "cpu"
        return self.keep_indices_batch[batch_index].to(device=device)


def deduplicate_keep_subsets(subsets: Sequence[torch.Tensor | Iterable[int]]) -> list[torch.Tensor]:
    unique_subsets: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    for subset in subsets:
        keep = torch.as_tensor(subset, dtype=torch.long).reshape(-1).unique(sorted=True)
        key = tuple(int(idx) for idx in keep.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_subsets.append(keep)
    return unique_subsets


def should_record_oracle_layer(
    layer_id: int,
    frame_id: int,
    layers_per_frame: int = 0,
    num_layers: int | None = None,
) -> bool:
    layers_per_frame = int(layers_per_frame)
    if layers_per_frame <= 0:
        return True
    num_layers = int(num_layers or 0)
    if num_layers <= 0 or layers_per_frame >= num_layers:
        return True
    stride = max(num_layers // layers_per_frame, 1)
    selected = {
        (int(frame_id) + offset * stride) % num_layers
        for offset in range(layers_per_frame)
    }
    return int(layer_id) in selected


def cuda_memory_log_suffix() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    except Exception:
        return ""
    return f" cuda_allocated_mb={allocated:.1f} cuda_reserved_mb={reserved:.1f}"


def log_replay_timing(
    log_fn: Callable[[str], None] | None,
    event: dict,
    num_subsets: int,
    chunk_size: int,
    stop: int,
    elapsed_sec: float,
    chunk_start: int,
    chunk_end: int,
) -> None:
    if log_fn is None:
        return
    log_fn(
        "timing phase=replay "
        f"event_id={event.get('event_id', '<unknown>')} "
        f"event_type={event.get('event_type', 'eviction')} "
        f"layer_id={int(event.get('layer_id', -1))} "
        f"frame_id={int(event.get('frame_id', -1))} "
        f"num_subsets={int(num_subsets)} chunk_size={int(chunk_size)} "
        f"chunk={int(chunk_start) + 1}-{int(chunk_end)} stop={int(stop)} "
        f"elapsed_sec={float(elapsed_sec):.3f}"
        f"{cuda_memory_log_suffix()}"
    )


def load_frontend_oracle_config(config_path: str | Path, num_views: int | None = None):
    cfg = OmegaConf.load(config_path)
    if num_views is not None:
        cfg.num_views = int(num_views)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_dataset_expression(cfg, dataset_key: str = "train_dataset") -> str:
    if dataset_key not in cfg:
        raise KeyError(f"Config has no dataset key: {dataset_key}")
    return str(OmegaConf.to_container(cfg, resolve=True)[dataset_key])


def build_frontend_oracle_dataloader(
    cfg,
    dataset_key: str = "train_dataset",
    batch_size: int | None = None,
    num_workers: int | None = None,
    drop_last: bool = False,
    seed: int = 0,
    log_fn: Callable[[str], None] | None = None,
):
    import dust3r.datasets as dust3r_datasets
    from dust3r.datasets import get_data_loader
    from dust3r.datasets.collate import frontend_collate_fn

    dataset_expr = resolve_dataset_expression(cfg, dataset_key=dataset_key)
    if log_fn is not None:
        log_fn(
            f"building dataloader: dataset_key={dataset_key} batch_size={batch_size} "
            f"num_workers={num_workers}"
        )
        log_fn(f"dataset expression length={len(dataset_expr)}")
    resolved_batch_size = int(batch_size if batch_size is not None else getattr(cfg, "batch_size", 1))
    resolved_num_workers = int(num_workers if num_workers is not None else getattr(cfg, "num_workers", 0))
    accelerator = SimpleNamespace(num_processes=1)
    collate_fn = frontend_collate_fn if resolved_batch_size > 1 else None
    dataset = eval(dataset_expr, dust3r_datasets.__dict__)
    dataset, prune_stats = prune_empty_oracle_dataset(dataset)
    if dataset is None or len(dataset) <= 0:
        raise ValueError(
            "Oracle dataset is empty after pruning branches with no valid long sequences. "
            f"dataset_key={dataset_key} num_views={getattr(cfg, 'num_views', '<unknown>')}"
        )
    if log_fn is not None and prune_stats["removed"] > 0:
        log_fn(
            "oracle dataset pruned empty branches: "
            f"removed={prune_stats['removed']} kept={prune_stats['kept']} "
            f"dataset_len={len(dataset)}"
        )
    # shuffle=True is required: make_sampler generates (idx, ar_idx, nview) tuples
    # that multi-resolution datasets need. With shuffle=False, it falls back to
    # plain integer indices which fail the len(_resolutions)==1 assertion.
    loader = get_data_loader(
        dataset,
        batch_size=resolved_batch_size,
        num_workers=resolved_num_workers,
        pin_mem=True,
        shuffle=True,
        drop_last=drop_last,
        accelerator=accelerator,
        fixed_length=bool(getattr(cfg, "fixed_length", True)),
        collate_fn=collate_fn,
    )
    # set_epoch is needed by ResizedDataset to build its index mapping,
    # and by CustomRandomSampler to seed the shuffle.
    set_oracle_loader_epoch(loader, seed=seed)
    sampler = getattr(loader, "batch_sampler", None)
    if log_fn is not None:
        dataset_len = _safe_len(getattr(loader, "dataset", None))
        sampler_len = _safe_len(sampler)
        log_fn(f"dataloader ready: dataset_len={dataset_len} sampler_len={sampler_len}")
    return loader


def prune_empty_oracle_dataset(dataset):
    from dust3r.datasets.base.easy_dataset import CatDataset, MulDataset, ResizedDataset

    stats = {"removed": 0, "kept": 0}

    def prune(node):
        if isinstance(node, CatDataset):
            children = []
            for child in node.datasets:
                pruned_child = prune(child)
                if pruned_child is not None:
                    children.append(pruned_child)
            if not children:
                stats["removed"] += 1
                return None
            if len(children) == 1:
                return children[0]
            return CatDataset(children)

        if isinstance(node, ResizedDataset):
            pruned_child = prune(node.dataset)
            if pruned_child is None:
                return None
            if len(pruned_child) == 0:
                stats["removed"] += 1
                return None
            return ResizedDataset(node.new_size, pruned_child)

        if isinstance(node, MulDataset):
            pruned_child = prune(node.dataset)
            if pruned_child is None:
                return None
            if len(pruned_child) == 0:
                stats["removed"] += 1
                return None
            return MulDataset(node.multiplicator, pruned_child)

        if len(node) == 0:
            stats["removed"] += 1
            return None
        stats["kept"] += 1
        return node

    return prune(dataset), stats


def set_oracle_loader_epoch(loader, seed: int = 0) -> None:
    epoch = int(seed)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def build_frozen_frontend_model_from_config(
    cfg,
    device: str | torch.device = "cuda",
    checkpoint_path: str | None = None,
    high_budget: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> OVGGT:
    from train_frontend import adapt_state_dict_for_model, build_frontend_cache_config

    device = torch.device(device)
    frontend_cache_config = build_frontend_cache_config(cfg)
    if frontend_cache_config is None:
        frontend_cache_config = FrontendCacheConfig(
            enabled=True,
            score_state_dim=int(getattr(cfg, "score_state_dim", 128)),
            budget_allocation="uniform",
        )
    frontend_cache_config.enabled = True
    frontend_cache_config.learned_eviction_enabled = False
    if high_budget:
        frontend_cache_config.dedup_enabled = False
        frontend_cache_config.intra_frame_dedup_enabled = False
    total_budget = int(getattr(cfg, "frontend_total_budget", 200000))
    if high_budget:
        total_budget = int(max(total_budget, 10_000_000))
    enable_track_head = int(getattr(cfg, "n_corres_train", 0) or 0) > 0
    model = OVGGT(
        mode="frontend_eval",
        frontend_pose_encoding_type=str(getattr(cfg, "frontend_pose_encoding_type", "absT_quaR_FoV")),
        total_budget=total_budget,
        camera_budget=int(getattr(cfg, "frontend_camera_budget", 384)),
        anchor_overflow_policy=str(getattr(cfg, "anchor_overflow_policy", "recent")),
        frontend_cache_config=frontend_cache_config,
        frontend_head_checkpointing=False,
        enable_track_head=enable_track_head,
        use_token_scorer=True,
    )
    ckpt = checkpoint_path or getattr(cfg, "resume", None) or getattr(cfg, "pretrained", None)
    if ckpt:
        from train_frontend import resolve_state_dict

        if log_fn is not None:
            log_fn(f"loading {'teacher' if high_budget else 'student'} checkpoint: {ckpt}")
        state = adapt_state_dict_for_model(model, resolve_state_dict(str(ckpt), map_location="cpu"))
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_frozen_teacher_from_config(
    cfg,
    device: str | torch.device = "cuda",
    checkpoint_path: str | None = None,
    log_fn: Callable[[str], None] | None = None,
):
    ckpt = checkpoint_path or getattr(cfg, "teacher", None) or getattr(cfg, "pretrained", None)
    if not ckpt:
        return None
    return build_frozen_frontend_model_from_config(
        cfg,
        device=device,
        checkpoint_path=str(ckpt),
        high_budget=True,
        log_fn=log_fn,
    )


@torch.inference_mode()
def collect_oracle_shard_from_config(collector_cfg: FrontendOracleCollectorConfig) -> dict:
    log_fn = default_oracle_log
    started = time.monotonic()
    log_fn(
        "collector start: "
        f"config={collector_cfg.config} dataset_key={collector_cfg.dataset_key} "
        f"output={collector_cfg.output} device={collector_cfg.device} "
        f"max_batches={collector_cfg.max_batches} max_events={collector_cfg.max_events} "
        f"num_samples={collector_cfg.num_samples} oracle_window={collector_cfg.oracle_window} "
        f"high_budget_teacher={collector_cfg.high_budget_teacher} "
        f"subset_replay_batch_size={collector_cfg.subset_replay_batch_size} "
        f"layers_per_frame={collector_cfg.layers_per_frame} "
        f"max_events_per_sequence={collector_cfg.max_events_per_sequence} "
        f"store_replay_payload={collector_cfg.store_replay_payload}"
    )
    log_fn("loading config")
    cfg = load_frontend_oracle_config(collector_cfg.config, num_views=collector_cfg.num_views)
    log_fn("config loaded")
    log_fn("building frozen student model")
    model = build_frozen_frontend_model_from_config(
        cfg,
        device=collector_cfg.device,
        checkpoint_path=collector_cfg.student_checkpoint,
        high_budget=False,
        log_fn=log_fn,
    )
    log_fn("student model ready")
    teacher = (
        build_frozen_teacher_from_config(
            cfg,
            device=collector_cfg.device,
            checkpoint_path=collector_cfg.teacher_checkpoint,
            log_fn=log_fn,
        )
        if collector_cfg.high_budget_teacher
        else None
    )
    if collector_cfg.high_budget_teacher:
        log_fn("teacher model ready" if teacher is not None else "teacher disabled: no checkpoint configured")
    num_layers = int(getattr(getattr(model, "aggregator", None), "depth", 0) or 0)
    data_loader = build_frontend_oracle_dataloader(
        cfg,
        dataset_key=collector_cfg.dataset_key,
        batch_size=collector_cfg.batch_size,
        num_workers=collector_cfg.num_workers,
        drop_last=False,
        seed=collector_cfg.seed,
        log_fn=log_fn,
    )
    base_shard = {
        "format": "ovggt_counterfactual_oracle_v1",
        "task_weights": TASK_WEIGHTS,
        "source_config": str(collector_cfg.config),
        "dataset_key": collector_cfg.dataset_key,
        "score_state_projection_state": extract_score_state_projection_state(model),
        "collector_config": asdict(collector_cfg),
    }
    flusher = OracleShardFlusher(
        base_shard=base_shard,
        output_path=collector_cfg.output,
        flush_every_events=collector_cfg.flush_every_events,
        flush_every_batches=collector_cfg.flush_every_batches,
        log_fn=log_fn,
    )
    log_fn("starting oracle event collection")
    events = collect_oracle_events_from_loader(
        model=model,
        data_loader=data_loader,
        device=torch.device(collector_cfg.device),
        max_batches=collector_cfg.max_batches,
        max_events=collector_cfg.max_events,
        num_samples=collector_cfg.num_samples,
        oracle_window=collector_cfg.oracle_window,
        seed=collector_cfg.seed,
        teacher=teacher,
        dataset_key=collector_cfg.dataset_key,
        log_fn=log_fn,
        flush_callback=lambda current_events, reason, batch_idx: flusher.maybe_flush(
            current_events,
            batch_idx=batch_idx,
            force=False,
            reason=reason,
        ),
        log_every_subsets=collector_cfg.log_every_subsets,
        subset_replay_batch_size=collector_cfg.subset_replay_batch_size,
        layers_per_frame=collector_cfg.layers_per_frame,
        max_events_per_sequence=collector_cfg.max_events_per_sequence,
        num_layers=num_layers,
        max_fetch_errors=collector_cfg.max_fetch_errors,
        store_replay_payload=collector_cfg.store_replay_payload,
    )
    shard = dict(base_shard)
    shard["events"] = events
    shard["partial"] = False
    shard["num_events"] = len(events)
    flusher.maybe_flush(events, batch_idx=None, force=True, reason="final")
    elapsed = time.monotonic() - started
    log_fn(f"collector done: events={len(events)} elapsed_sec={elapsed:.1f}")
    return shard


@torch.inference_mode()
def collect_oracle_events_from_loader(
    model,
    data_loader,
    device: torch.device,
    max_batches: int,
    max_events: int,
    num_samples: int,
    oracle_window: int,
    seed: int = 0,
    teacher=None,
    dataset_key: str | None = None,
    log_fn: Callable[[str], None] | None = None,
    flush_callback: Callable[[Sequence[dict], str, int | None], None] | None = None,
    log_every_subsets: int = 1,
    subset_replay_batch_size: int = 1,
    layers_per_frame: int = 0,
    max_events_per_sequence: int = 0,
    num_layers: int | None = None,
    max_fetch_errors: int = 256,
    store_replay_payload: bool = False,
) -> list[dict]:
    events: list[dict] = []
    max_batches_int = int(max_batches)
    max_events_int = int(max_events)
    max_fetch_errors_int = max(int(max_fetch_errors), 0)
    fetch_errors = 0
    batch_idx = 0
    loader_iter = iter(data_loader)
    while batch_idx < max_batches_int:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except Exception as exc:
            fetch_errors += 1
            if log_fn is not None:
                log_fn(
                    "skipping dataloader batch after fetch error "
                    f"{fetch_errors}/{max_fetch_errors_int}: {type(exc).__name__}: {exc}"
                )
            if fetch_errors > max_fetch_errors_int:
                raise RuntimeError(
                    f"Exceeded max_fetch_errors={max_fetch_errors_int} while collecting oracle data"
                ) from exc
            continue

        if batch_idx >= max_batches_int:
            break
        if log_fn is not None:
            log_fn(f"batch {batch_idx + 1}/{max_batches_int} start: events_so_far={len(events)}")
        sequence_provenance = build_sequence_provenance(
            batch,
            batch_index=batch_idx,
            dataset_key=dataset_key,
        )
        if log_fn is not None:
            log_fn(
                f"batch {batch_idx + 1}/{max_batches_int} provenance: "
                f"{format_provenance_log_summary(sequence_provenance)}"
            )
        batch = move_batch_to_device(normalize_batch_images(copy.deepcopy(batch)), device)
        remaining_events = max_events - len(events)
        if int(max_events_per_sequence) > 0:
            remaining_events = min(remaining_events, int(max_events_per_sequence))
        collected_batch_events: list[dict] = []

        def on_event_collected(event: dict) -> None:
            collected_batch_events.append(event)
            events.append(event)
            if flush_callback is not None:
                flush_callback(
                    events,
                    f"batch {batch_idx + 1}/{max_batches_int} event {len(collected_batch_events)} collected",
                    None,
                )

        batch_events = collect_oracle_events_from_sequence(
            model=model,
            frames=batch,
            device=device,
            max_events=remaining_events,
            num_samples=num_samples,
            oracle_window=oracle_window,
            seed=seed + batch_idx * 1009,
            teacher=teacher,
            event_prefix=f"batch{batch_idx}",
            sequence_provenance=sequence_provenance,
            log_fn=log_fn,
            log_every_subsets=log_every_subsets,
            subset_replay_batch_size=subset_replay_batch_size,
            layers_per_frame=layers_per_frame,
            num_layers=num_layers,
            on_event_collected=on_event_collected,
            store_replay_payload=store_replay_payload,
        )
        for event in batch_events:
            if not any(event is collected for collected in collected_batch_events):
                events.append(event)
        if log_fn is not None:
            log_fn(
                f"batch {batch_idx + 1}/{max_batches_int} done: "
                f"batch_events={len(batch_events)} total_events={len(events)}"
            )
        if flush_callback is not None:
            flush_callback(events, f"batch {batch_idx + 1}/{max_batches_int} done", batch_idx)
        if len(events) >= max_events_int:
            if log_fn is not None:
                log_fn(f"stopping: reached max_events={max_events_int}")
            break
        batch_idx += 1
    return events


@torch.inference_mode()
def collect_oracle_events_from_sequence(
    model,
    frames: Sequence[dict],
    device: torch.device,
    max_events: int,
    num_samples: int,
    oracle_window: int,
    seed: int = 0,
    teacher=None,
    event_prefix: str = "oracle",
    sequence_provenance: dict[int, dict] | None = None,
    log_fn: Callable[[str], None] | None = None,
    log_every_subsets: int = 1,
    subset_replay_batch_size: int = 1,
    layers_per_frame: int = 0,
    num_layers: int | None = None,
    on_event_collected: Callable[[dict], None] | None = None,
    store_replay_payload: bool = False,
) -> list[dict]:
    if max_events <= 0:
        return []
    frames = list(frames)
    if log_fn is not None:
        log_fn(f"{event_prefix}: running probe over {len(frames)} frames")
    probe = CounterfactualEvictionProbe(
        num_samples=num_samples,
        oracle_window=oracle_window,
        seed=seed,
        event_prefix=event_prefix,
        max_events=max_events,
        sequence_provenance=sequence_provenance,
        layers_per_frame=layers_per_frame,
        num_layers=num_layers,
    )
    dedup_probe = CounterfactualDedupProbe(
        num_samples=num_samples,
        oracle_window=oracle_window,
        seed=seed,
        event_prefix=f"{event_prefix}_dedup",
        max_events=max_events,
        sequence_provenance=sequence_provenance,
        layers_per_frame=layers_per_frame,
        num_layers=num_layers,
        voxel_size=0.25,
    )
    fifo_probe = CounterfactualFifoTopKProbe(
        num_samples=num_samples,
        oracle_window=oracle_window,
        seed=seed,
        event_prefix=f"{event_prefix}_fifo",
        max_events=max_events,
        sequence_provenance=sequence_provenance,
        layers_per_frame=layers_per_frame,
        num_layers=num_layers,
    )
    probe_started = time.monotonic()
    _run_frontend_with_probe(model, frames, probe, cache_results=False,
                              dedup_probe=dedup_probe, fifo_probe=fifo_probe)
    probe_elapsed = time.monotonic() - probe_started
    candidate_events = [
        event
        for event in probe.events
        if int(event["frame_id"]) + 1 < len(frames)
    ][:max_events]
    # Combine dedup and FIFO events into candidate_events
    dedup_events = [
        event
        for event in dedup_probe.events
        if int(event["frame_id"]) + 1 < len(frames)
    ]
    fifo_events = [
        event
        for event in fifo_probe.events
        if int(event["frame_id"]) + 1 < len(frames)
    ]
    candidate_events.extend(dedup_events)
    candidate_events.extend(fifo_events)
    candidate_events = candidate_events[:max_events]
    if log_fn is not None:
        log_fn(
            f"{event_prefix}: probe captured {len(probe.events)} eviction, "
            f"{len(dedup_probe.events)} dedup, {len(fifo_probe.events)} fifo candidates, "
            f"{len(candidate_events)} total have future frames"
        )
        log_fn(
            "timing phase=probe "
            f"event_prefix={event_prefix} frames={len(frames)} "
            f"eviction={len(probe.events)} dedup={len(dedup_probe.events)} "
            f"fifo={len(fifo_probe.events)} candidate_events={len(candidate_events)} "
            f"elapsed_sec={probe_elapsed:.3f}"
            f"{cuda_memory_log_suffix()}"
        )
    if not candidate_events:
        return []

    teacher_outputs = None
    if teacher is not None:
        if log_fn is not None:
            log_fn(f"{event_prefix}: running high-budget teacher for {len(frames)} frames")
        teacher_started = time.monotonic()
        teacher_outputs = teacher.inference(
            frames,
            move_to_cpu=False,
            cache_results=True,
            return_views=False,
        )
        teacher_elapsed = time.monotonic() - teacher_started
        if log_fn is not None:
            log_fn(f"{event_prefix}: teacher outputs ready")
            log_fn(
                "timing phase=teacher "
                f"event_prefix={event_prefix} frames={len(frames)} "
                f"elapsed_sec={teacher_elapsed:.3f}"
                f"{cuda_memory_log_suffix()}"
            )

    measured_events = []
    for event_idx, event in enumerate(candidate_events):
        future_frames = frames[int(event["frame_id"]) + 1 : int(event["frame_id"]) + 1 + int(oracle_window)]
        if not future_frames:
            continue
        if log_fn is not None:
            log_fn(
                f"{event_prefix}: measuring event {event_idx + 1}/{len(candidate_events)} "
                f"id={event.get('event_id')} layer={event.get('layer_id')} "
                f"frame={event.get('frame_id')} subsets={len(event.get('candidate_subsets', []))} "
                f"future_frames={len(future_frames)}"
            )
        measured_event = measure_counterfactual_event(
            model=model,
            frames=frames,
            event=event,
            future_frames=future_frames,
            teacher_outputs=teacher_outputs,
            log_fn=log_fn,
            log_every_subsets=log_every_subsets,
            subset_replay_batch_size=subset_replay_batch_size,
            store_replay_payload=store_replay_payload,
        )
        if measured_event is not None:
            measured_events.append(measured_event)
            if on_event_collected is not None:
                on_event_collected(measured_event)
    return measured_events


def build_sequence_provenance(batch, batch_index: int, dataset_key: str | None = None) -> dict[int, dict]:
    if not isinstance(batch, list):
        return {}
    if not batch:
        return {}

    num_sequences = 0
    for frame in batch:
        if isinstance(frame, dict):
            for value in frame.values():
                if isinstance(value, torch.Tensor) and value.dim() >= 1:
                    num_sequences = max(num_sequences, int(value.shape[0]))
                    break

    provenance = {}
    for seq_idx in range(num_sequences):
        frames = []
        dataset_name = None
        sequence_key_parts = []
        for frame_idx, frame in enumerate(batch):
            frame_info = {
                "frame_index": int(frame_idx),
                "dataset": _select_provenance_value(frame.get("dataset"), seq_idx),
                "label": _select_provenance_value(frame.get("label"), seq_idx),
                "instance": _select_provenance_value(frame.get("instance"), seq_idx),
            }
            if dataset_name is None and frame_info["dataset"] is not None:
                dataset_name = str(frame_info["dataset"])
            label = frame_info["label"]
            instance = frame_info["instance"]
            if label is not None:
                sequence_key_parts.append(str(label))
            elif instance is not None:
                sequence_key_parts.append(str(instance))
            frames.append(frame_info)
        provenance[seq_idx] = {
            "dataset_key": dataset_key,
            "dataset": dataset_name,
            "batch_index": int(batch_index),
            "sequence_index": int(seq_idx),
            "sequence_id": ":".join(sequence_key_parts) if sequence_key_parts else f"batch{batch_index}:sequence{seq_idx}",
            "frame_count": len(frames),
            "frames": frames,
        }
    return provenance


def format_provenance_log_summary(provenance: dict[int, dict]) -> str:
    seq_count = len(provenance)
    first_sequence = next(iter(provenance.values()), None)
    if first_sequence is None:
        return f"sequences={seq_count} first=<none>"

    dataset_key = first_sequence.get("dataset_key") or "-"
    dataset = first_sequence.get("dataset") or "-"
    sequence_id = str(first_sequence.get("sequence_id") or "<unknown>")
    frame_count = int(first_sequence.get("frame_count") or 0)
    short_id = _short_sequence_id(sequence_id)
    return f"sequences={seq_count} first={dataset_key}/{dataset}/{short_id} frames={frame_count}"


def _short_sequence_id(sequence_id: str, max_chars: int = 96) -> str:
    if len(sequence_id) <= max_chars:
        return sequence_id
    parts = [part for part in sequence_id.split(":") if part]
    if len(parts) >= 2:
        shortened = f"{parts[0]}:...:{parts[-1]}"
        if len(shortened) <= max_chars:
            return shortened
    keep = max(max_chars - 3, 1)
    head = max(keep // 2, 1)
    tail = max(keep - head, 1)
    return f"{sequence_id[:head]}...{sequence_id[-tail:]}"


def _select_provenance_value(value, batch_index: int):
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and value.shape[0] > batch_index:
            selected = value[batch_index]
        else:
            selected = value
        if selected.numel() == 1:
            return selected.item()
        return selected.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) > batch_index:
            return _select_provenance_value(value[batch_index], 0)
        return None
    return value


@torch.inference_mode()
def measure_counterfactual_event(
    model,
    frames: Sequence[dict],
    event: dict,
    future_frames: Sequence[dict],
    teacher_outputs=None,
    log_fn: Callable[[str], None] | None = None,
    log_every_subsets: int = 1,
    subset_replay_batch_size: int = 1,
    store_replay_payload: bool = False,
) -> dict:
    event_started = time.monotonic()
    measured_subsets = []
    replay_total_sec = 0.0
    target_total_sec = 0.0
    loss_total_sec = 0.0
    start = int(event["frame_id"]) + 1
    stop = start + len(future_frames)
    subsets = list(event.get("candidate_subsets", []))
    log_every = max(int(log_every_subsets), 0)
    replay_batch_size = max(int(subset_replay_batch_size), 1)
    event_type = str(event.get("event_type", "eviction"))

    def append_measured_subset(keep_indices: torch.Tensor, predictions: Sequence[dict]) -> None:
        nonlocal target_total_sec, loss_total_sec
        target_started = time.monotonic()
        targets = build_future_targets(
            future_frames=future_frames,
            predictions=predictions,
            teacher_outputs=teacher_outputs,
            start_frame_idx=start,
        )
        target_total_sec += time.monotonic() - target_started
        loss_started = time.monotonic()
        loss_components = compute_three_task_loss_components(predictions, targets)
        loss = weighted_three_task_loss(loss_components)
        loss_total_sec += time.monotonic() - loss_started
        measured_subset = {
            "keep_indices": keep_indices.detach().cpu(),
            "loss": loss,
            "loss_components": loss_components,
        }
        if store_replay_payload:
            measured_subset["replay"] = {
                "predictions": detach_tensor_tree(predictions),
                "targets": detach_tensor_tree(targets),
            }
        measured_subsets.append(measured_subset)

    if replay_batch_size <= 1:
        for subset_idx, subset in enumerate(subsets):
            if log_fn is not None and (log_every == 0 or subset_idx % log_every == 0 or subset_idx + 1 == len(subsets)):
                log_fn(
                    f"{event.get('event_id', '<unknown>')}: replay subset "
                    f"{subset_idx + 1}/{len(subsets)} frames=0:{stop}"
                )
            keep_indices = torch.as_tensor(subset["keep_indices"], dtype=torch.long)
            if event_type == "dedup":
                replay_probe = ReplayDedupKeepSetProbe(event, keep_indices)
                replay_started = time.monotonic()
                outputs = _run_frontend_with_probe(model, frames[:stop], None, cache_results=True, dedup_replay_probe=replay_probe)
            else:
                replay_probe = ReplayKeepSetProbe(event, keep_indices)
                replay_started = time.monotonic()
                outputs = _run_frontend_with_probe(model, frames[:stop], replay_probe, cache_results=True)
            replay_elapsed = time.monotonic() - replay_started
            replay_total_sec += replay_elapsed
            log_replay_timing(
                log_fn=log_fn,
                event=event,
                num_subsets=len(subsets),
                chunk_size=1,
                stop=stop,
                elapsed_sec=replay_elapsed,
                chunk_start=subset_idx,
                chunk_end=subset_idx + 1,
            )
            if not replay_probe.applied:
                if log_fn is not None:
                    log_fn(
                        f"{event.get('event_id', '<unknown>')}: skipping event; "
                        f"{'dedup' if event_type == 'dedup' else 'serial'} replay keep set was not applied"
                    )
                return None
            predictions = list(outputs.ress[start:stop])
            append_measured_subset(keep_indices, predictions)
    else:
        for chunk_start in range(0, len(subsets), replay_batch_size):
            chunk = subsets[chunk_start:chunk_start + replay_batch_size]
            chunk_size = len(chunk)
            chunk_end = chunk_start + chunk_size
            if log_fn is not None:
                log_fn(
                    f"{event.get('event_id', '<unknown>')}: replay subsets "
                    f"{chunk_start + 1}-{chunk_end}/{len(subsets)} "
                    f"batch={chunk_size} frames=0:{stop}"
                )
            keep_indices_batch = [
                torch.as_tensor(subset["keep_indices"], dtype=torch.long)
                for subset in chunk
            ]
            if event_type == "dedup":
                dedup_replay_probe = MultiReplayDedupKeepSetProbe(event, keep_indices_batch)
                replay_frames = repeat_frames_for_batch(frames[:stop], chunk_size)
                replay_started = time.monotonic()
                outputs = _run_frontend_with_probe(
                    model, replay_frames, None, cache_results=True,
                    dedup_replay_probe=dedup_replay_probe,
                )
                replay_elapsed = time.monotonic() - replay_started
                replay_total_sec += replay_elapsed
                log_replay_timing(
                    log_fn=log_fn,
                    event=event,
                    num_subsets=len(subsets),
                    chunk_size=chunk_size,
                    stop=stop,
                    elapsed_sec=replay_elapsed,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                if dedup_replay_probe.applied_count != chunk_size:
                    if log_fn is not None:
                        log_fn(
                            f"{event.get('event_id', '<unknown>')}: batched dedup replay keep sets "
                            f"applied {dedup_replay_probe.applied_count}/{chunk_size}; "
                            "falling back to serial replay"
                        )
                    serial_subsets = measure_counterfactual_subset_serial(
                        model=model,
                        frames=frames,
                        event=event,
                        future_frames=future_frames,
                        subsets=chunk,
                        start=start,
                        stop=stop,
                        teacher_outputs=teacher_outputs,
                        log_fn=log_fn,
                        log_every=log_every,
                        subset_offset=chunk_start,
                        total_subsets=len(subsets),
                        store_replay_payload=store_replay_payload,
                    )
                    if serial_subsets is None:
                        if log_fn is not None:
                            log_fn(
                                f"{event.get('event_id', '<unknown>')}: skipping event; "
                                "dedup fallback serial replay did not apply all keep sets"
                            )
                        return None
                    measured_subsets.extend(serial_subsets)
                else:
                    for local_idx, (subset, keep_indices) in enumerate(zip(chunk, keep_indices_batch)):
                        predictions = [
                            select_prediction_batch(prediction, local_idx)
                            for prediction in outputs.ress[start:stop]
                        ]
                        append_measured_subset(keep_indices, predictions)
                continue
            replay_probe = MultiReplayKeepSetProbe(event, keep_indices_batch)
            replay_frames = repeat_frames_for_batch(frames[:stop], chunk_size)
            replay_started = time.monotonic()
            outputs = _run_frontend_with_probe(
                model,
                replay_frames,
                replay_probe,
                cache_results=True,
            )
            replay_elapsed = time.monotonic() - replay_started
            replay_total_sec += replay_elapsed
            log_replay_timing(
                log_fn=log_fn,
                event=event,
                num_subsets=len(subsets),
                chunk_size=chunk_size,
                stop=stop,
                elapsed_sec=replay_elapsed,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            if replay_probe.applied_count != chunk_size:
                if log_fn is not None:
                    log_fn(
                        f"{event.get('event_id', '<unknown>')}: batched replay keep sets were applied "
                        f"{replay_probe.applied_count}/{chunk_size}; falling back to serial replay"
                    )
                serial_subsets = measure_counterfactual_subset_serial(
                    model=model,
                    frames=frames,
                    event=event,
                    future_frames=future_frames,
                    subsets=chunk,
                    start=start,
                    stop=stop,
                    teacher_outputs=teacher_outputs,
                    log_fn=log_fn,
                    log_every=log_every,
                    subset_offset=chunk_start,
                    total_subsets=len(subsets),
                    store_replay_payload=store_replay_payload,
                )
                if serial_subsets is None:
                    if log_fn is not None:
                        log_fn(
                            f"{event.get('event_id', '<unknown>')}: skipping event; "
                            "fallback serial replay did not apply all keep sets"
                        )
                    return None
                measured_subsets.extend(serial_subsets)
            else:
                for local_idx, (subset, keep_indices) in enumerate(zip(chunk, keep_indices_batch)):
                    predictions = [
                        select_prediction_batch(prediction, local_idx)
                        for prediction in outputs.ress[start:stop]
                    ]
                    append_measured_subset(keep_indices, predictions)

    output = {
        key: value
        for key, value in event.items()
        if key != "candidate_subsets" and value is not None
    }
    output["subsets"] = measured_subsets
    if log_fn is not None:
        log_fn(
            "timing phase=event "
            f"event_id={event.get('event_id', '<unknown>')} "
            f"event_type={event_type} layer_id={int(event.get('layer_id', -1))} "
            f"frame_id={int(event.get('frame_id', -1))} "
            f"num_subsets={len(subsets)} measured_subsets={len(measured_subsets)} "
            f"replay_sec={replay_total_sec:.3f} target_sec={target_total_sec:.3f} "
            f"loss_sec={loss_total_sec:.3f} elapsed_sec={time.monotonic() - event_started:.3f}"
            f"{cuda_memory_log_suffix()}"
        )
    return output


def measure_counterfactual_subset_serial(
    model,
    frames: Sequence[dict],
    event: dict,
    future_frames: Sequence[dict],
    subsets: Sequence[dict],
    start: int,
    stop: int,
    teacher_outputs=None,
    log_fn: Callable[[str], None] | None = None,
    log_every: int = 1,
    subset_offset: int = 0,
    total_subsets: int | None = None,
    store_replay_payload: bool = False,
) -> list[dict] | None:
    measured_subsets = []
    total = len(subsets) if total_subsets is None else int(total_subsets)
    event_type = str(event.get("event_type", "eviction"))
    for local_idx, subset in enumerate(subsets):
        subset_idx = int(subset_offset) + local_idx
        if log_fn is not None and (log_every == 0 or subset_idx % log_every == 0 or subset_idx + 1 == total):
            log_fn(
                f"{event.get('event_id', '<unknown>')}: replay subset "
                f"{subset_idx + 1}/{total} frames=0:{stop}"
            )
        keep_indices = torch.as_tensor(subset["keep_indices"], dtype=torch.long)
        if event_type == "dedup":
            replay_probe = ReplayDedupKeepSetProbe(event, keep_indices)
            outputs = _run_frontend_with_probe(model, frames[:stop], None, cache_results=True, dedup_replay_probe=replay_probe)
        else:
            replay_probe = ReplayKeepSetProbe(event, keep_indices)
            outputs = _run_frontend_with_probe(model, frames[:stop], replay_probe, cache_results=True)
        if not replay_probe.applied:
            if log_fn is not None:
                log_fn(
                    f"{event.get('event_id', '<unknown>')}: replay subset "
                    f"{subset_idx + 1}/{total} did not apply keep set"
                )
            return None
        predictions = list(outputs.ress[start:stop])
        targets = build_future_targets(
            future_frames=future_frames,
            predictions=predictions,
            teacher_outputs=teacher_outputs,
            start_frame_idx=start,
        )
        loss_components = compute_three_task_loss_components(predictions, targets)
        measured_subset = {
            "keep_indices": keep_indices.detach().cpu(),
            "loss": weighted_three_task_loss(loss_components),
            "loss_components": loss_components,
        }
        if store_replay_payload:
            measured_subset["replay"] = {
                "predictions": detach_tensor_tree(predictions),
                "targets": detach_tensor_tree(targets),
            }
        measured_subsets.append(measured_subset)
    return measured_subsets


def repeat_frames_for_batch(frames: Sequence[dict], batch_size: int) -> list[dict]:
    return [
        repeat_batch_tree(frame, batch_size)
        for frame in frames
    ]


def repeat_batch_tree(value, batch_size: int):
    batch_size = int(batch_size)
    if isinstance(value, torch.Tensor):
        if value.dim() == 0 or value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            repeats = [batch_size] + [1] * (value.dim() - 1)
            return value.repeat(*repeats)
        return value
    if isinstance(value, dict):
        return {key: repeat_batch_tree(item, batch_size) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) == 1:
            return [copy.deepcopy(value[0]) for _ in range(batch_size)]
        return [repeat_batch_tree(item, batch_size) for item in value]
    if isinstance(value, tuple):
        if len(value) == 1:
            return tuple(copy.deepcopy(value[0]) for _ in range(batch_size))
        return tuple(repeat_batch_tree(item, batch_size) for item in value)
    return value


def select_prediction_batch(value, batch_index: int):
    batch_index = int(batch_index)
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and value.shape[0] > batch_index:
            return value[batch_index:batch_index + 1]
        return value
    if isinstance(value, dict):
        return {key: select_prediction_batch(item, batch_index) for key, item in value.items()}
    if isinstance(value, list):
        return [select_prediction_batch(item, batch_index) for item in value]
    if isinstance(value, tuple):
        return tuple(select_prediction_batch(item, batch_index) for item in value)
    return value


def detach_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_tensor_tree(item) for item in value)
    return value


def build_future_targets(
    future_frames: Sequence[dict],
    predictions: Sequence[dict],
    teacher_outputs=None,
    start_frame_idx: int = 0,
) -> list[dict]:
    targets = []
    teacher_preds = None if teacher_outputs is None else teacher_outputs.ress
    for offset, (frame, pred) in enumerate(zip(future_frames, predictions)):
        frame_idx = int(start_frame_idx) + offset
        teacher_pred = None
        if teacher_preds is not None and frame_idx < len(teacher_preds):
            teacher_pred = teacher_preds[frame_idx]
        targets.append(build_frame_target(frame, pred, teacher_pred))
    return targets


def build_frame_target(frame: dict, prediction: dict, teacher_pred: dict | None = None) -> dict:
    target = {}
    if "camera_pose" in frame and "camera_intrinsics" in frame and "img" in frame:
        target["camera_pose"] = _gt_camera_pose_encoding(frame, prediction.get("camera_pose"))
    elif "camera_pose" in frame and prediction.get("camera_pose") is not None:
        raw_pose = torch.as_tensor(frame["camera_pose"])
        if raw_pose.shape == torch.as_tensor(prediction["camera_pose"]).shape:
            target["camera_pose"] = raw_pose.to(
                device=prediction["camera_pose"].device,
                dtype=prediction["camera_pose"].dtype,
            )
    if "camera_pose" not in target and teacher_pred is not None and "camera_pose" in teacher_pred:
        target["camera_pose"] = teacher_pred["camera_pose"]
    elif "camera_pose" not in target and "camera_pose" in prediction:
        target["camera_pose"] = torch.zeros_like(prediction["camera_pose"])

    if "depthmap" in frame:
        target["depth"] = _ensure_depth_shape(frame["depthmap"], prediction.get("depth"))
    elif "depth" in frame:
        target["depth"] = _ensure_depth_shape(frame["depth"], prediction.get("depth"))
    elif teacher_pred is not None and "depth" in teacher_pred:
        target["depth"] = teacher_pred["depth"]

    if "pts3d" in frame:
        target["pts3d_in_other_view"] = frame["pts3d"]
    elif "point_map" in frame:
        target["pts3d_in_other_view"] = frame["point_map"]
    elif teacher_pred is not None and "pts3d_in_other_view" in teacher_pred:
        target["pts3d_in_other_view"] = teacher_pred["pts3d_in_other_view"]

    if "valid_mask" in frame:
        target["valid_mask"] = frame["valid_mask"]
    elif teacher_pred is not None and "valid_mask" in teacher_pred:
        target["valid_mask"] = teacher_pred["valid_mask"]
    return target


def normalize_batch_images(batch):
    if isinstance(batch, list):
        out = []
        for frame in batch:
            if isinstance(frame, dict):
                frame = dict(frame)
                if "img" in frame:
                    img = frame["img"]
                    if isinstance(img, torch.Tensor) and float(img.min()) < 0.0:
                        frame["img"] = (img + 1.0) / 2.0
            out.append(frame)
        return out
    if isinstance(batch, dict):
        batch = dict(batch)
        if "img" in batch:
            img = batch["img"]
            if isinstance(img, torch.Tensor) and float(img.min()) < 0.0:
                batch["img"] = (img + 1.0) / 2.0
        return batch
    return batch


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def save_oracle_shard(shard: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(shard, tmp_path)
    os.replace(tmp_path, path)


def _safe_len(value) -> str:
    if value is None:
        return "unknown"
    try:
        return str(len(value))
    except TypeError:
        return "unknown"


def extract_score_state_projection_state(model) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("aggregator.score_state_projs.")
    }


def load_callable(path: str) -> Callable:
    if ":" not in path:
        raise ValueError("Callable path must use module:function format")
    module_name, function_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def _run_frontend_with_probe(model, frames: Sequence[dict], probe, cache_results: bool,
                              dedup_probe=None, fifo_probe=None, dedup_replay_probe=None):
    previous_probe = getattr(model, "_oracle_eviction_probe", None)
    previous_dedup = getattr(model, "_oracle_dedup_probe", None)
    previous_fifo = getattr(model, "_oracle_fifo_probe", None)
    previous_dedup_replay = getattr(model, "_oracle_dedup_replay_probe", None)
    runtime_snapshot = _snapshot_model_runtime_state(model)
    model._oracle_eviction_probe = probe
    if dedup_probe is not None:
        model._oracle_dedup_probe = dedup_probe
    if fifo_probe is not None:
        model._oracle_fifo_probe = fifo_probe
    if dedup_replay_probe is not None:
        model._oracle_dedup_replay_probe = dedup_replay_probe
    try:
        return model.inference(
            frames,
            move_to_cpu=False,
            cache_results=cache_results,
            return_views=False,
        )
    finally:
        _restore_model_runtime_state(model, runtime_snapshot)
        # restore all probes
        for attr, prev in [
            ("_oracle_eviction_probe", previous_probe),
            ("_oracle_dedup_probe", previous_dedup),
            ("_oracle_fifo_probe", previous_fifo),
            ("_oracle_dedup_replay_probe", previous_dedup_replay),
        ]:
            if prev is None:
                if hasattr(model, attr):
                    delattr(model, attr)
            else:
                setattr(model, attr, prev)


def _snapshot_model_runtime_state(model) -> dict:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        return {}
    return {
        "last_scores": aggregator.last_scores.detach().clone(),
        "attn_anchor_counts": [
            int(getattr(block.attn, "num_anchor_tokens", 0))
            for block in getattr(aggregator, "global_blocks", [])
        ],
    }


def _restore_model_runtime_state(model, snapshot: dict) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is None or not snapshot:
        return
    if "last_scores" in snapshot:
        aggregator.last_scores = snapshot["last_scores"].to(
            device=aggregator.last_scores.device,
            dtype=aggregator.last_scores.dtype,
        )
    for block, count in zip(getattr(aggregator, "global_blocks", []), snapshot.get("attn_anchor_counts", [])):
        if hasattr(block.attn, "num_anchor_tokens"):
            block.attn.num_anchor_tokens = int(count)


def _select_batch(tensor: torch.Tensor, batch_index: int) -> torch.Tensor:
    if tensor.dim() >= 1 and tensor.shape[0] > batch_index:
        return tensor[batch_index]
    return tensor


def _select_cache_batch(tensor: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
    if tensor.dim() >= 1 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.dim() >= 1 and tensor.shape[0] > batch_index:
        return tensor[batch_index]
    return tensor


def _gt_camera_pose_encoding(frame: dict, reference):
    if reference is None:
        return torch.as_tensor(frame["camera_pose"])
    ref = torch.as_tensor(reference)
    device = ref.device
    dtype = ref.dtype
    c2w = torch.as_tensor(frame["camera_pose"], device=device, dtype=dtype)
    intrinsics = torch.as_tensor(frame["camera_intrinsics"], device=device, dtype=dtype)
    img = torch.as_tensor(frame["img"])
    image_size_hw = (int(img.shape[-2]), int(img.shape[-1]))
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0).unsqueeze(1)
    elif c2w.dim() == 3:
        c2w = c2w.unsqueeze(1)
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(1)
    elif intrinsics.dim() == 3:
        intrinsics = intrinsics.unsqueeze(1)
    w2c = closed_form_inverse_se3(c2w.reshape(-1, 4, 4)).reshape_as(c2w)
    pose_enc = world_to_camera_to_pose_encoding(
        w2c,
        intrinsics=intrinsics,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )[:, 0]
    return pose_enc.to(device=device, dtype=dtype)


def _ensure_depth_shape(depth, reference):
    tensor = torch.as_tensor(depth)
    if reference is None:
        if tensor.dim() == 3:
            return tensor.unsqueeze(-1)
        return tensor
    ref = torch.as_tensor(reference)
    if tensor.dim() == ref.dim() - 1:
        tensor = tensor.unsqueeze(-1)
    return tensor.to(device=ref.device, dtype=ref.dtype)
