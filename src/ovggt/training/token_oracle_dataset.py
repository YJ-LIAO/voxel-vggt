"""Dataset and losses for counterfactual token-retention oracle shards."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


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
    ) -> None:
        self.min_loss_gap = float(min_loss_gap)
        if event_types is not None:
            self.event_types = set(event_types)
            unsupported = self.event_types - self.SUPPORTED_EVENT_TYPES
            if unsupported:
                raise ValueError(f"Unsupported event types: {unsupported}")
        else:
            self.event_types = self.SUPPORTED_EVENT_TYPES
        self.samples = []
        for shard_path in shard_paths:
            shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
            events = shard.get("events", shard if isinstance(shard, list) else [])
            for event in events:
                self._append_event_pairs(event)

    def _append_event_pairs(self, event: dict) -> None:
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
        denom = max(loss_max - loss_min, 1e-8)

        score_state = _ensure_token_matrix(event["score_state"]).float()
        metadata_features = _ensure_token_matrix(event["metadata_features"]).float()
        num_tokens = int(score_state.shape[0])
        layer_id = int(event.get("layer_id", 0))
        event_id = event.get("event_id", "")
        sequence_provenance = event.get("sequence_provenance")

        for better_idx, better_subset in enumerate(subsets):
            for worse_idx, worse_subset in enumerate(subsets):
                better_loss = float(better_subset["loss"])
                worse_loss = float(worse_subset["loss"])
                target_margin = worse_loss - better_loss
                if target_margin <= 0.0 or target_margin < self.min_loss_gap:
                    continue
                self.samples.append(
                    {
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
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


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
    ) -> None:
        if label_reduction not in ("min", "mean"):
            raise ValueError(f"label_reduction must be 'min' or 'mean', got '{label_reduction}'")
        self.count_candidates = list(count_candidates)
        self.label_reduction = label_reduction
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

        # Compute group loss using the configured reduction
        candidate_set = set(self.count_candidates)
        best_loss = float("inf")
        best_keep_count: int | None = None
        for kc, losses in groups.items():
            if kc not in candidate_set:
                continue
            if self.label_reduction == "min":
                group_loss = min(losses)
            else:  # mean
                group_loss = sum(losses) / len(losses)
            if group_loss < best_loss:
                best_loss = group_loss
                best_keep_count = kc

        if best_keep_count is None:
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
    }


def token_oracle_ranking_loss(
    logits: Tensor,
    batch: dict,
    margin_scale: float = 1.0,
    regression_weight: float = 0.1,
) -> tuple[Tensor, dict]:
    token_mask = batch.get("token_mask")
    if token_mask is not None:
        token_mask = token_mask.to(device=logits.device)
    better_mask = batch["better_mask"].to(device=logits.device)
    worse_mask = batch["worse_mask"].to(device=logits.device)
    better_score = _subset_score(logits, better_mask, token_mask, reduction="mean")
    worse_score = _subset_score(logits, worse_mask, token_mask, reduction="mean")
    target_margin = batch["target_margin"].to(device=logits.device, dtype=logits.dtype) * margin_scale

    pairwise = F.softplus(target_margin - (better_score - worse_score)).mean()
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
    return loss, {
        "pairwise": float(pairwise.detach().cpu().item()),
        "regression": float(regression.detach().cpu().item()),
        "rank_acc": float((score_diff > 0).to(dtype=torch.float32).mean().detach().cpu().item()),
        "mean_score_diff": float(score_diff.mean().detach().cpu().item()),
    }


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
        mask[index_tensor.clamp(0, num_tokens - 1)] = True
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
