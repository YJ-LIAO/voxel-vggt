"""Learned scorer for frontend KV-cache retention decisions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


TOKEN_METADATA_FEATURE_INDEX = {
    "depth_conf": 0,
    "slot_local_xyz_start": 1,
    "active_xyz_start": 4,
    "frame_age": 7,
    "anchor_slot": 8,
    "is_protected": 9,
    "kind_camera": 10,
    "kind_register": 11,
    "kind_patch": 12,
    "slot_id": 13,
    "keyframe_id": 14,
    "xyz_valid": 15,
    "decision_context": 16,   # 0=pruning, 1=dedup, 2=fifo, 3=eviction
}
TOKEN_METADATA_FEATURE_DIM = 17


class TokenScorer(nn.Module):
    """Score cached tokens for retention.

    Inputs are available at cache commit/eviction time:
    - compact ``score_state`` from the transformer layer, shape ``[B, N, Ds]``
    - metadata features built from ``TokenMetadata``, shape ``[B, N, Dm]``
    - layer id, embedded and broadcast over tokens

    Returns raw logits ``[B, N]``. Higher logits mean "prefer retaining".
    """

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        bottleneck_dim: Optional[int] = None,
        score_state_dim: Optional[int] = None,
        metadata_dim: int = TOKEN_METADATA_FEATURE_DIM,
        hidden_dim: Optional[int] = None,
        num_layers: int = 24,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if score_state_dim is None:
            score_state_dim = embed_dim if embed_dim is not None else 128
        if hidden_dim is None:
            hidden_dim = bottleneck_dim if bottleneck_dim is not None else max(score_state_dim, 64)

        self.score_state_dim = int(score_state_dim)
        self.metadata_dim = int(metadata_dim)
        self.layer_embed = nn.Embedding(max(int(num_layers), 1), self.score_state_dim)
        input_dim = self.score_state_dim + self.metadata_dim + self.score_state_dim
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(int(depth) - 1, 0)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(
        self,
        score_state: Tensor,
        metadata_features: Optional[Tensor] = None,
        layer_id: int | Tensor = 0,
    ) -> Tensor:
        if score_state.dim() != 3:
            raise ValueError(f"score_state must have shape [B, N, D], got {tuple(score_state.shape)}")
        B, N, _ = score_state.shape
        if metadata_features is None:
            metadata_features = torch.zeros(
                B,
                N,
                self.metadata_dim,
                dtype=score_state.dtype,
                device=score_state.device,
            )
        if metadata_features.shape[:2] != (B, N):
            raise ValueError(
                "metadata_features must share [B, N] with score_state, got "
                f"{tuple(metadata_features.shape)} vs {tuple(score_state.shape)}"
            )
        if metadata_features.shape[-1] != self.metadata_dim:
            raise ValueError(
                f"metadata_features dim {metadata_features.shape[-1]} does not match {self.metadata_dim}"
            )

        if isinstance(layer_id, Tensor):
            layer_tensor = layer_id.to(device=score_state.device, dtype=torch.long).reshape(-1)
            if layer_tensor.numel() == 1:
                layer_tensor = layer_tensor.expand(B)
            elif layer_tensor.numel() != B:
                raise ValueError(f"layer_id tensor must have 1 or B={B} elements, got {layer_tensor.numel()}")
        else:
            layer_tensor = torch.full((B,), int(layer_id), dtype=torch.long, device=score_state.device)
        layer_tensor = layer_tensor.clamp(0, self.layer_embed.num_embeddings - 1)
        layer_features = self.layer_embed(layer_tensor).unsqueeze(1).expand(B, N, -1)

        scorer_input = torch.cat(
            [score_state, metadata_features.to(dtype=score_state.dtype), layer_features.to(dtype=score_state.dtype)],
            dim=-1,
        )
        return self.scorer(scorer_input).squeeze(-1)
