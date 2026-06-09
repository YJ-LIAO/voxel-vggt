"""Dataset and losses for counterfactual token-retention oracle shards."""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Stable hash helper (shared with train_token_scorer_oracle.py)
# ---------------------------------------------------------------------------

def _stable_bucket(value: str, seed: int = 0, buckets: int = 10000) -> int:
    """Deterministic bucket assignment via MD5 hash."""
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


# ---------------------------------------------------------------------------
# Event-level loading and splitting helpers
# ---------------------------------------------------------------------------

def load_oracle_events(shard_paths: Sequence[str | Path]) -> list[dict]:
    """Load all events from a sequence of oracle shard files.

    Each shard may be a dict with an ``"events"`` key or a plain list of events.
    """
    events: list[dict] = []
    for shard_path in shard_paths:
        shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        events.extend(shard.get("events", shard if isinstance(shard, list) else []))
    return events


def split_oracle_events(
    events: list[dict],
    val_fraction: float = 0.1,
    split_key: str = "sequence_id",
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split *events* (not samples) into train and validation sets.

    Splits at the event level so that downstream token-ranking and count
    datasets built from the same split share exactly the same events with
    no data leakage.

    Args:
        events: List of oracle event dicts.
        val_fraction: Approximate fraction of events assigned to validation.
        split_key: ``"sequence_id"`` groups by
            ``event["sequence_provenance"]["sequence_id"]`` (falls back to
            ``event_id``); ``"event_id_hash"`` groups by ``event_id``.
        seed: Seed for deterministic hashing.

    Returns:
        ``(train_events, val_events)`` tuple.
    """
    threshold = int(float(val_fraction) * 10000)
    train: list[dict] = []
    val: list[dict] = []
    for event in events:
        if split_key == "sequence_id":
            prov = event.get("sequence_provenance") or {}
            key = str(prov.get("sequence_id") or event.get("event_id") or "")
        elif split_key == "event_id_hash":
            key = str(event.get("event_id") or "")
        else:
            raise ValueError(f"Unsupported split_key={split_key}")
        target = val if _stable_bucket(key, seed=seed) < threshold else train
        target.append(event)
    return train, val


class CounterfactualOracleDataset(Dataset):
    """Pairwise ranking samples from counterfactual retention events.

    v2: Supports all three event types: eviction, dedup, and fifo_topk.
    All event types produce pairwise ranking samples using the same
    scorer inputs (score_state + metadata_features + layer_id) and
    the same loss (pairwise margin ranking + regression).

    Each shard stores events with token-level features and multiple sampled
    retention subsets. For every event, this dataset emits all ordered subset
    pairs where the lower-loss subset is preferred.
    """

    # Supported event types for unified scorer training
    SUPPORTED_EVENT_TYPES = {"eviction", "dedup", "fifo_topk"}

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        min_loss_gap: float = 0.0,
        event_types: Sequence[str] | None = None,
        fifo_token_pair_mode: str = "any",
        max_pairs_per_event: int | None = 64,
        pair_sampling_seed: int = 0,
        min_loss_gap_by_event_type: dict[str, float] | None = None,
        max_loss_gap: float | None = None,
    ) -> None:
        self.min_loss_gap = float(min_loss_gap)
        self.pair_sampling_seed = int(pair_sampling_seed)
        self.min_loss_gap_by_event_type = (
            {k: float(v) for k, v in min_loss_gap_by_event_type.items()}
            if min_loss_gap_by_event_type is not None
            else None
        )
        self.max_loss_gap = float(max_loss_gap) if max_loss_gap is not None else None
        if event_types is not None:
            self.event_types = set(event_types)
            unsupported = self.event_types - self.SUPPORTED_EVENT_TYPES
            if unsupported:
                raise ValueError(f"Unsupported event types: {unsupported}")
        else:
            self.event_types = self.SUPPORTED_EVENT_TYPES
        if fifo_token_pair_mode not in ("any", "same_keep_count"):
            raise ValueError(
                f"fifo_token_pair_mode must be 'any' or 'same_keep_count', "
                f"got '{fifo_token_pair_mode}'"
            )
        self.fifo_token_pair_mode = fifo_token_pair_mode
        self.max_pairs_per_event = max_pairs_per_event
        self.samples: list[dict] = []
        for shard_path in shard_paths:
            shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
            events = shard.get("events", shard if isinstance(shard, list) else [])
            for event in events:
                self._append_event_pairs(event)

    @classmethod
    def from_events(
        cls,
        events: list[dict],
        min_loss_gap: float = 0.0,
        event_types: Sequence[str] | None = None,
        fifo_token_pair_mode: str = "any",
        max_pairs_per_event: int | None = 64,
        pair_sampling_seed: int = 0,
        min_loss_gap_by_event_type: dict[str, float] | None = None,
        max_loss_gap: float | None = None,
    ) -> "CounterfactualOracleDataset":
        """Build a dataset from a pre-loaded list of events.

        Avoids re-reading from disk.  Useful when events have already been
        loaded and split via :func:`split_oracle_events`.
        """
        dataset = cls.__new__(cls)
        dataset.min_loss_gap = float(min_loss_gap)
        dataset.pair_sampling_seed = int(pair_sampling_seed)
        dataset.min_loss_gap_by_event_type = (
            {k: float(v) for k, v in min_loss_gap_by_event_type.items()}
            if min_loss_gap_by_event_type is not None
            else None
        )
        dataset.max_loss_gap = float(max_loss_gap) if max_loss_gap is not None else None
        if event_types is not None:
            dataset.event_types = set(event_types)
            unsupported = dataset.event_types - cls.SUPPORTED_EVENT_TYPES
            if unsupported:
                raise ValueError(f"Unsupported event types: {unsupported}")
        else:
            dataset.event_types = cls.SUPPORTED_EVENT_TYPES
        if fifo_token_pair_mode not in ("any", "same_keep_count"):
            raise ValueError(
                f"fifo_token_pair_mode must be 'any' or 'same_keep_count', "
                f"got '{fifo_token_pair_mode}'"
            )
        dataset.fifo_token_pair_mode = fifo_token_pair_mode
        dataset.max_pairs_per_event = max_pairs_per_event
        dataset.samples = []
        for event in events:
            dataset._append_event_pairs(event)
        return dataset

    def _event_min_loss_gap(self, event_type: str) -> float:
        """Return the per-event-type threshold, falling back to global min_loss_gap."""
        if self.min_loss_gap_by_event_type and event_type in self.min_loss_gap_by_event_type:
            return float(self.min_loss_gap_by_event_type[event_type])
        return float(self.min_loss_gap)

    def _append_event_pairs(self, event: dict) -> None:
        import random as _random

        # v2: Filter by event_type
        event_type = str(event.get("event_type", "eviction"))
        if event_type not in self.event_types:
            return

        subsets = list(event.get("subsets", []))
        if len(subsets) < 2:
            return

        losses = torch.tensor([float(subset["loss"]) for subset in subsets], dtype=torch.float32)
        loss_min = float(losses.min().item())
        loss_max = float(losses.max().item())

        # C1 fix: skip events where loss range is below the event-specific
        # min_loss_gap.  All subsets are effectively tied — regression targets
        # would be noise.
        event_min_gap = self._event_min_loss_gap(event_type)
        if loss_max - loss_min < event_min_gap:
            return

        denom = max(loss_max - loss_min, 1e-8)

        score_state = _ensure_token_matrix(event["score_state"]).float()
        metadata_features = _ensure_token_matrix(event["metadata_features"]).float()
        num_tokens = int(score_state.shape[0])
        layer_id = int(event.get("layer_id", 0))
        event_id = event.get("event_id", "")
        sequence_provenance = event.get("sequence_provenance")
        event_keep_count = event.get("keep_count")

        # Collect candidate pair descriptors (subset references), then
        # build valid samples, then cap deterministically.
        candidate_pairs: list[tuple[dict, dict]] = []

        # When same_keep_count mode is active for fifo_topk, group subsets
        # by keep_count so that pairs are only formed within the same group.
        if self.fifo_token_pair_mode == "same_keep_count" and event_type == "fifo_topk":
            groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
            for idx, subset in enumerate(subsets):
                kc = subset.get("keep_count", event_keep_count)
                if kc is not None:
                    groups[int(kc)].append((idx, subset))
            # Generate pairs within each keep_count group
            for _kc, group_members in groups.items():
                for i, (_, better_subset) in enumerate(group_members):
                    for j, (_, worse_subset) in enumerate(group_members):
                        if i == j:
                            continue
                        candidate_pairs.append((better_subset, worse_subset))
        else:
            # Default "any" mode: all pairwise combinations (skip self-pairs)
            for i, better_subset in enumerate(subsets):
                for j, worse_subset in enumerate(subsets):
                    if i == j:
                        continue
                    candidate_pairs.append((better_subset, worse_subset))

        # Phase 1: filter to valid pairs (margin > 0 and >= min_loss_gap)
        valid_samples: list[dict] = []
        for better_subset, worse_subset in candidate_pairs:
            sample = self._build_pair_sample(
                better_subset, worse_subset, event_id, event_type,
                layer_id, score_state, metadata_features, num_tokens,
                loss_max, denom, sequence_provenance, event_keep_count,
            )
            if sample is not None:
                valid_samples.append(sample)

        # Phase 2: deterministic capping using local RNG seeded from
        # pair_sampling_seed and event_id.
        if self.max_pairs_per_event is not None and len(valid_samples) > self.max_pairs_per_event:
            seed_key = f"{self.pair_sampling_seed}:{event_id}"
            seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = _random.Random(seed)
            kept_indices = sorted(rng.sample(range(len(valid_samples)), self.max_pairs_per_event))
            valid_samples = [valid_samples[i] for i in kept_indices]

        self.samples.extend(valid_samples)

    def _build_pair_sample(
        self,
        better_subset: dict,
        worse_subset: dict,
        event_id: str,
        event_type: str,
        layer_id: int,
        score_state: Tensor,
        metadata_features: Tensor,
        num_tokens: int,
        loss_max: float,
        denom: float,
        sequence_provenance: dict | None,
        event_keep_count: int | None,
    ) -> dict | None:
        """Build a sample dict for a pair, or return None if invalid."""
        better_loss = float(better_subset["loss"])
        worse_loss = float(worse_subset["loss"])
        target_margin = worse_loss - better_loss
        min_gap = self._event_min_loss_gap(event_type)
        if target_margin <= 0.0 or target_margin < min_gap:
            return None
        if self.max_loss_gap is not None and target_margin > float(self.max_loss_gap):
            return None

        # FIFO keep_count metadata: store for fifo_topk events, None otherwise
        if event_type == "fifo_topk":
            better_keep_count = better_subset.get("keep_count", event_keep_count)
            worse_keep_count = worse_subset.get("keep_count", event_keep_count)
        else:
            better_keep_count = None
            worse_keep_count = None

        return {
            "event_id": event_id,
            "event_type": event_type,
            "layer_id": layer_id,
            "score_state": score_state,
            "metadata_features": metadata_features,
            "better_mask": _indices_to_mask(better_subset["keep_indices"], num_tokens),
            "worse_mask": _indices_to_mask(worse_subset["keep_indices"], num_tokens),
            "better_loss": better_loss,
            "worse_loss": worse_loss,
            "target_margin": target_margin,
            "better_target": (loss_max - better_loss) / denom,
            "worse_target": (loss_max - worse_loss) / denom,
            "sequence_provenance": sequence_provenance,
            "better_keep_count": better_keep_count,
            "worse_keep_count": worse_keep_count,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def summarize_oracle_pair_samples(samples: Sequence[dict]) -> dict:
    """Compute summary statistics from a sequence of oracle pair samples.

    Pure helper usable by both trainers and diagnostic scripts.
    Required fields: count, by_event_type, unique_event_count, fifo_count,
    fifo_cross_keep_count, fifo_cross_keep_frac, target_margin percentiles,
    by_event_type_margin_p50, by_event_type_unique_event_count.
    """
    import numpy as np

    count = len(samples)
    if count == 0:
        return {
            "count": 0,
            "by_event_type": {},
            "unique_event_count": 0,
            "fifo_count": 0,
            "fifo_cross_keep_count": 0,
            "fifo_cross_keep_frac": 0.0,
            "target_margin_p10": 0.0,
            "target_margin_p25": 0.0,
            "target_margin_p50": 0.0,
            "target_margin_p75": 0.0,
            "target_margin_p90": 0.0,
            "by_event_type_margin_p50": {},
            "by_event_type_unique_event_count": {},
        }

    # by_event_type counts
    by_event_type: dict[str, int] = {}
    # margins grouped by event_type
    margins_by_type: dict[str, list[float]] = {}
    # event_ids grouped by event_type
    event_ids_by_type: dict[str, set[str]] = {}
    all_event_ids: set[str] = set()

    fifo_count = 0
    fifo_cross_keep_count = 0
    all_margins: list[float] = []

    for sample in samples:
        et = str(sample.get("event_type", "eviction"))
        by_event_type[et] = by_event_type.get(et, 0) + 1
        margins_by_type.setdefault(et, []).append(float(sample["target_margin"]))
        event_id = str(sample.get("event_id", ""))
        event_ids_by_type.setdefault(et, set()).add(event_id)
        all_event_ids.add(event_id)
        all_margins.append(float(sample["target_margin"]))

        if et == "fifo_topk":
            fifo_count += 1
            bkc = sample.get("better_keep_count")
            wkc = sample.get("worse_keep_count")
            if bkc is not None and wkc is not None and bkc != wkc:
                fifo_cross_keep_count += 1

    margins_arr = np.array(all_margins, dtype=np.float64)

    def _percentile(arr: np.ndarray, q: float) -> float:
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, q))

    by_event_type_margin_p50: dict[str, float] = {}
    by_event_type_unique_event_count: dict[str, int] = {}
    for et in by_event_type:
        by_event_type_margin_p50[et] = _percentile(np.array(margins_by_type[et], dtype=np.float64), 50)
        by_event_type_unique_event_count[et] = len(event_ids_by_type.get(et, set()))

    fifo_cross_keep_frac = fifo_cross_keep_count / fifo_count if fifo_count > 0 else 0.0

    return {
        "count": count,
        "by_event_type": by_event_type,
        "unique_event_count": len(all_event_ids),
        "fifo_count": fifo_count,
        "fifo_cross_keep_count": fifo_cross_keep_count,
        "fifo_cross_keep_frac": fifo_cross_keep_frac,
        "target_margin_p10": _percentile(margins_arr, 10),
        "target_margin_p25": _percentile(margins_arr, 25),
        "target_margin_p50": _percentile(margins_arr, 50),
        "target_margin_p75": _percentile(margins_arr, 75),
        "target_margin_p90": _percentile(margins_arr, 90),
        "by_event_type_margin_p50": by_event_type_margin_p50,
        "by_event_type_unique_event_count": by_event_type_unique_event_count,
    }


def collate_oracle_pairs(samples: List[dict]) -> dict:
    if not samples:
        raise ValueError("collate_oracle_pairs received an empty batch")
    max_tokens = max(sample["score_state"].shape[0] for sample in samples)
    score_dim = samples[0]["score_state"].shape[1]
    metadata_dim = samples[0]["metadata_features"].shape[1]
    score_state = torch.stack([_pad_matrix(sample["score_state"], max_tokens, score_dim) for sample in samples], dim=0)
    metadata_features = torch.stack(
        [_pad_matrix(sample["metadata_features"], max_tokens, metadata_dim) for sample in samples],
        dim=0,
    )
    better_mask = torch.stack([_pad_mask(sample["better_mask"], max_tokens) for sample in samples], dim=0)
    worse_mask = torch.stack([_pad_mask(sample["worse_mask"], max_tokens) for sample in samples], dim=0)
    return {
        "event_id": [sample["event_id"] for sample in samples],
        "event_type": [sample.get("event_type", "eviction") for sample in samples],
        "layer_id": torch.tensor([sample["layer_id"] for sample in samples], dtype=torch.long),
        "score_state": score_state,
        "metadata_features": metadata_features,
        "better_mask": better_mask,
        "worse_mask": worse_mask,
        "token_mask": torch.stack(
            [torch.arange(max_tokens, dtype=torch.long) < sample["score_state"].shape[0] for sample in samples],
            dim=0,
        ),
        "valid_token_count": torch.tensor([sample["score_state"].shape[0] for sample in samples], dtype=torch.long),
        "better_loss": torch.tensor([sample["better_loss"] for sample in samples], dtype=torch.float32),
        "worse_loss": torch.tensor([sample["worse_loss"] for sample in samples], dtype=torch.float32),
        "target_margin": torch.tensor([sample["target_margin"] for sample in samples], dtype=torch.float32),
        "better_target": torch.tensor([sample["better_target"] for sample in samples], dtype=torch.float32),
        "worse_target": torch.tensor([sample["worse_target"] for sample in samples], dtype=torch.float32),
        "sequence_provenance": [sample["sequence_provenance"] for sample in samples],
    }


class FifoCountDataset(Dataset):
    """Classification dataset for predicting best keep_count from fifo_topk events.

    For each event, groups measured subsets by keep_count, finds the best loss
    per group (min or mean), and picks the keep_count with the lowest group loss.
    The target is the index of that keep_count in the configured count_candidates.

    Features are sliced to demoted_indices when available (giving [K, D] tensors),
    or fall back to full cache tokens for old shards.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        count_candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
        label_reduction: str = "min",
        min_count_loss_gap: float = 0.0,
    ) -> None:
        if label_reduction not in ("min", "mean"):
            raise ValueError(f"label_reduction must be 'min' or 'mean', got '{label_reduction}'")
        self.count_candidates = list(count_candidates)
        self.label_reduction = label_reduction
        self.min_count_loss_gap = float(min_count_loss_gap)
        self._old_shard_warnings = 0
        self.samples: list[dict] = []
        for shard_path in shard_paths:
            shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
            if isinstance(shard, list):
                events = shard
            else:
                events = shard.get("events", [])
            for event in events:
                self._append_event(event)
        if self._old_shard_warnings > 0:
            warnings.warn(
                f"FifoCountDataset: {self._old_shard_warnings} events lacked "
                f"demoted_indices; falling back to full cache tokens.",
                stacklevel=2,
            )

    @classmethod
    def from_events(
        cls,
        events: list[dict],
        count_candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
        label_reduction: str = "min",
        min_count_loss_gap: float = 0.0,
    ) -> "FifoCountDataset":
        """Build a dataset from a pre-loaded list of events.

        Avoids re-reading from disk.  Useful when events have already been
        loaded and split via :func:`split_oracle_events`.
        """
        dataset = cls.__new__(cls)
        if label_reduction not in ("min", "mean"):
            raise ValueError(f"label_reduction must be 'min' or 'mean', got '{label_reduction}'")
        dataset.count_candidates = list(count_candidates)
        dataset.label_reduction = label_reduction
        dataset.min_count_loss_gap = float(min_count_loss_gap)
        dataset._old_shard_warnings = 0
        dataset.samples: list[dict] = []
        for event in events:
            dataset._append_event(event)
        if dataset._old_shard_warnings > 0:
            warnings.warn(
                f"FifoCountDataset: {dataset._old_shard_warnings} events lacked "
                f"demoted_indices; falling back to full cache tokens.",
                stacklevel=2,
            )
        return dataset

    def _append_event(self, event: dict) -> None:
        event_type = str(event.get("event_type", ""))
        if event_type != "fifo_topk":
            return

        subsets = list(event.get("subsets", []))
        if not subsets:
            return

        # Group subsets by keep_count, falling back to event-level keep_count
        event_keep_count = event.get("keep_count")
        groups: dict[int, list[float]] = defaultdict(list)
        for subset in subsets:
            kc = subset.get("keep_count", event_keep_count)
            if kc is None:
                continue
            groups[int(kc)].append(float(subset["loss"]))

        if not groups:
            return

        # Compute candidate losses: reduce list[float] to scalar per group,
        # filtered to configured candidate_set only.
        candidate_set = set(self.count_candidates)
        candidate_losses: dict[int, float] = {}
        for kc, losses in groups.items():
            if kc not in candidate_set:
                continue
            if self.label_reduction == "min":
                candidate_losses[kc] = min(losses)
            else:  # mean
                candidate_losses[kc] = sum(losses) / len(losses)

        if len(candidate_losses) < 1:
            return

        # Rank candidates by reduced loss
        ranked = sorted(candidate_losses.items(), key=lambda item: item[1])
        best_keep_count, best_loss = ranked[0]

        # Confidence filtering: single candidate with positive gap threshold -> drop
        if len(ranked) < 2:
            if self.min_count_loss_gap > 0.0:
                return
            second_loss = best_loss
        else:
            second_loss = ranked[1][1]

        count_loss_gap = second_loss - best_loss
        if count_loss_gap < self.min_count_loss_gap:
            return

        target = self.count_candidates.index(best_keep_count)

        # Feature extraction: prefer demoted_indices, fallback to full tokens
        score_state = _ensure_token_matrix(event["score_state"]).float()
        metadata_features = _ensure_token_matrix(event["metadata_features"]).float()

        if "demoted_indices" in event:
            demoted = torch.as_tensor(event["demoted_indices"], dtype=torch.long)
            score_state = score_state[demoted]
            metadata_features = metadata_features[demoted]
        else:
            self._old_shard_warnings += 1

        self.samples.append(
            {
                "event_id": str(event.get("event_id", "")),
                "layer_id": int(event.get("layer_id", 0)),
                "score_state": score_state,
                "metadata_features": metadata_features,
                "target": target,
                "target_keep_count": best_keep_count,
                "count_loss_gap": count_loss_gap,
                "best_count_loss": best_loss,
                "second_best_count_loss": second_loss,
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate_fifo_count_samples(samples: List[dict]) -> dict:
    """Collate FifoCountDataset samples into a padded batch.

    Pads score_state and metadata_features to the max token count across
    the batch and returns a token_mask indicating real vs padded positions.
    """
    if not samples:
        raise ValueError("collate_fifo_count_samples received an empty batch")

    max_tokens = max(sample["score_state"].shape[0] for sample in samples)
    score_dim = samples[0]["score_state"].shape[1]
    metadata_dim = samples[0]["metadata_features"].shape[1]

    score_state = torch.stack(
        [_pad_matrix(sample["score_state"], max_tokens, score_dim) for sample in samples],
        dim=0,
    )
    metadata_features = torch.stack(
        [_pad_matrix(sample["metadata_features"], max_tokens, metadata_dim) for sample in samples],
        dim=0,
    )
    token_mask = torch.stack(
        [torch.arange(max_tokens, dtype=torch.long) < sample["score_state"].shape[0] for sample in samples],
        dim=0,
    )

    return {
        "event_id": [sample["event_id"] for sample in samples],
        "layer_id": torch.tensor([sample["layer_id"] for sample in samples], dtype=torch.long),
        "score_state": score_state,
        "metadata_features": metadata_features,
        "token_mask": token_mask,
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.long),
        "target_keep_count": torch.tensor([sample["target_keep_count"] for sample in samples], dtype=torch.long),
        "count_loss_gap": torch.tensor(
            [sample.get("count_loss_gap", 0.0) for sample in samples], dtype=torch.float32
        ),
    }


def summarize_fifo_count_samples(samples: Sequence[dict]) -> dict:
    """Compute summary statistics from a sequence of fifo count samples.

    Pure helper usable by both trainers and diagnostic scripts.
    Returns count, target_keep_count distribution, and count_loss_gap percentiles.
    """
    import numpy as np

    count = len(samples)
    if count == 0:
        return {
            "count": 0,
            "target_keep_count": {},
            "count_loss_gap_p10": 0.0,
            "count_loss_gap_p50": 0.0,
            "count_loss_gap_p90": 0.0,
        }

    keep_counts: dict[int, int] = {}
    gaps: list[float] = []
    for sample in samples:
        tkc = int(sample.get("target_keep_count", 0))
        keep_counts[tkc] = keep_counts.get(tkc, 0) + 1
        gaps.append(float(sample.get("count_loss_gap", 0.0)))

    gaps_arr = np.array(gaps, dtype=np.float64)

    return {
        "count": count,
        "target_keep_count": keep_counts,
        "count_loss_gap_p10": float(np.percentile(gaps_arr, 10)),
        "count_loss_gap_p50": float(np.percentile(gaps_arr, 50)),
        "count_loss_gap_p90": float(np.percentile(gaps_arr, 90)),
    }


def token_oracle_ranking_loss(
    logits: Tensor,
    batch: dict,
    margin_scale: float = 1.0,
    regression_weight: float = 0.1,
    score_mode: str = "set_mean",
) -> tuple[Tensor, dict]:
    token_mask = batch.get("token_mask")
    if token_mask is not None:
        token_mask = token_mask.to(device=logits.device)
    better_mask = batch["better_mask"].to(device=logits.device)
    worse_mask = batch["worse_mask"].to(device=logits.device)

    if score_mode == "delta_mean":
        # Compare only tokens unique to each mask
        better_only = better_mask & ~worse_mask
        worse_only = worse_mask & ~better_mask
        # Fall back to set_mean per sample if either delta mask is empty
        better_only_count = better_only.sum(dim=1)
        worse_only_count = worse_only.sum(dim=1)
        fallback = (better_only_count == 0) | (worse_only_count == 0)
        if fallback.any():
            # Compute both and blend per-sample
            delta_better_score = _subset_score(logits, better_only, token_mask, reduction="mean")
            delta_worse_score = _subset_score(logits, worse_only, token_mask, reduction="mean")
            set_better_score = _subset_score(logits, better_mask, token_mask, reduction="mean")
            set_worse_score = _subset_score(logits, worse_mask, token_mask, reduction="mean")
            better_score = torch.where(fallback, set_better_score, delta_better_score)
            worse_score = torch.where(fallback, set_worse_score, delta_worse_score)
        else:
            better_score = _subset_score(logits, better_only, token_mask, reduction="mean")
            worse_score = _subset_score(logits, worse_only, token_mask, reduction="mean")
    else:
        # set_mean: current behavior
        better_score = _subset_score(logits, better_mask, token_mask, reduction="mean")
        worse_score = _subset_score(logits, worse_mask, token_mask, reduction="mean")

    target_margin = batch["target_margin"].to(device=logits.device, dtype=logits.dtype) * margin_scale

    pairwise = F.softplus(target_margin - (better_score - worse_score)).mean()

    # Regression always uses set_mean scores for backward compatibility
    better_regression_score = _subset_score(logits, better_mask, token_mask, reduction="mean")
    worse_regression_score = _subset_score(logits, worse_mask, token_mask, reduction="mean")
    better_target = batch["better_target"].to(device=logits.device, dtype=logits.dtype)
    worse_target = batch["worse_target"].to(device=logits.device, dtype=logits.dtype)
    regression = 0.5 * (
        F.mse_loss(better_regression_score, better_target)
        + F.mse_loss(worse_regression_score, worse_target)
    )
    loss = pairwise + regression_weight * regression
    score_diff = better_score - worse_score
    details = {
        "pairwise": float(pairwise.detach().cpu().item()),
        "regression": float(regression.detach().cpu().item()),
        "rank_acc": float((score_diff > 0).to(dtype=torch.float32).mean().detach().cpu().item()),
        "mean_score_diff": float(score_diff.mean().detach().cpu().item()),
    }
    if score_mode == "delta_mean":
        delta_fallback_count = int(((better_only_count == 0) | (worse_only_count == 0)).sum().item())
        details["delta_fallback_count"] = delta_fallback_count
    return loss, details


def _subset_score(
    logits: Tensor,
    mask: Tensor,
    token_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    mask = mask.to(dtype=logits.dtype)
    if token_mask is not None:
        mask = mask * token_mask.to(dtype=logits.dtype)
    score = (logits * mask).sum(dim=1)
    if reduction == "sum":
        return score
    if reduction == "mean":
        denom = mask.sum(dim=1).clamp_min(1.0)
        return score / denom
    raise ValueError(f"Unsupported subset score reduction: {reduction}")


def _ensure_token_matrix(value: Tensor | Iterable) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise ValueError(f"Expected token matrix [N, D], got {tuple(tensor.shape)}")
    return tensor


def _indices_to_mask(indices: Tensor | Iterable[int], num_tokens: int) -> Tensor:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    if index_tensor.numel() > 0:
        if index_tensor.max() >= num_tokens or index_tensor.min() < 0:
            raise ValueError(
                f"keep_indices out of range [0, {num_tokens - 1}]: "
                f"min={int(index_tensor.min().item())} max={int(index_tensor.max().item())}"
            )
        mask[index_tensor] = True
    return mask


def _pad_matrix(matrix: Tensor | Iterable, max_tokens: int, feature_dim: int) -> Tensor:
    tensor = torch.as_tensor(matrix)
    if tensor.dim() != 2:
        raise ValueError(f"Expected matrix [N, D], got {tuple(tensor.shape)}")
    padded = torch.zeros(max_tokens, feature_dim, dtype=tensor.dtype)
    length = min(int(tensor.shape[0]), max_tokens)
    padded[:length] = tensor[:length]
    return padded


def _pad_mask(mask: Tensor | Iterable, max_tokens: int) -> Tensor:
    tensor = torch.as_tensor(mask, dtype=torch.bool)
    padded = torch.zeros(max_tokens, dtype=torch.bool)
    length = min(int(tensor.shape[0]), max_tokens)
    padded[:length] = tensor[:length]
    return padded
