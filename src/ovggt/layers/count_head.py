"""Shared classification head for FIFO count prediction.

Supports two architectures:

- ``pooled_v1`` (default): the original implementation that concatenates
  pooled score_state, pooled metadata, token count, and a layer embedding,
  then passes through an MLP.

- ``shared_encoder_v2``: uses ``RetentionTokenEncoder`` to produce per-token
  hidden features, then ``FifoCountClassifier`` to pool and classify.

Both architectures output logits ``[B, num_candidates]`` and share the same
``predict_count`` helper.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class FifoCountHead(nn.Module):
    """Predict a discrete FIFO flush count from pooled token representations.

    Parameters
    ----------
    score_state_dim : int
        Dimensionality of the per-token ``score_state`` vector (``Ds``).
    metadata_dim : int
        Dimensionality of per-token metadata features (``Dm``).
    hidden_dim : int | None
        Hidden width of the classification MLP (v1) or encoder output dim (v2).
        Defaults to ``score_state_dim`` for v1, or 128 for v2.
    num_layers : int
        Number of transformer layers (used for layer embedding).
    candidates : sequence of int
        Discrete count values to classify among.
    arch : str
        Architecture version: ``"pooled_v1"`` (default) or
        ``"shared_encoder_v2"``.
    """

    def __init__(
        self,
        score_state_dim: int = 128,
        metadata_dim: int = 17,
        hidden_dim: int | None = None,
        num_layers: int = 24,
        candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
        arch: str = "pooled_v1",
    ) -> None:
        super().__init__()
        self.arch = arch
        self.score_state_dim = int(score_state_dim)
        self.metadata_dim = int(metadata_dim)

        if arch == "pooled_v1":
            self._build_v1(score_state_dim, metadata_dim, hidden_dim, num_layers, candidates)
        elif arch == "shared_encoder_v2":
            self._build_v2(score_state_dim, metadata_dim, hidden_dim, num_layers, candidates)
        else:
            raise ValueError(f"Unknown FifoCountHead arch: {arch}")

        self.register_buffer(
            "candidates",
            torch.tensor(candidates, dtype=torch.long),
            persistent=True,
        )

    def _build_v1(
        self,
        score_state_dim: int,
        metadata_dim: int,
        hidden_dim: int | None,
        num_layers: int,
        candidates: Sequence[int],
    ) -> None:
        if hidden_dim is None:
            hidden_dim = score_state_dim

        self.layer_embed = nn.Embedding(max(int(num_layers), 1), score_state_dim)

        # MLP input: pooled_score_state (Ds) + pooled_metadata (Dm)
        #           + token_count (1) + layer_embed (Ds)
        mlp_in = score_state_dim + metadata_dim + 1 + score_state_dim
        num_candidates = len(candidates)
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_candidates),
        )

    def _build_v2(
        self,
        score_state_dim: int,
        metadata_dim: int,
        hidden_dim: int | None,
        num_layers: int,
        candidates: Sequence[int],
    ) -> None:
        from ovggt.layers.retention_policy import RetentionTokenEncoder, FifoCountClassifier

        effective_hidden = hidden_dim if hidden_dim is not None else 128
        self._encoder = RetentionTokenEncoder(
            score_state_dim, metadata_dim, effective_hidden, num_layers,
        )
        self._classifier = FifoCountClassifier(
            self._encoder.hidden_dim, candidates,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        score_state: Tensor,
        metadata_features: Tensor | None = None,
        layer_id: int | Tensor = 0,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        """Return logits ``[B, num_candidates]`` for each batch element.

        Parameters
        ----------
        score_state : Tensor [B, N, Ds]
            Per-token compact state from the transformer layer.
        metadata_features : Tensor [B, N, Dm] | None
            Per-token metadata features.  Zeros are used when *None*.
        layer_id : int | Tensor
            Transformer layer index (scalar or ``[B]``).
        token_mask : Tensor [B, N] | None
            Boolean mask; ``True`` marks **valid** tokens.  When *None* all
            tokens are considered valid.

        Returns
        -------
        Tensor [B, num_candidates]
        """
        if self.arch == "pooled_v1":
            return self._forward_v1(score_state, metadata_features, layer_id, token_mask)
        else:
            return self._forward_v2(score_state, metadata_features, layer_id, token_mask)

    def _forward_v1(
        self,
        score_state: Tensor,
        metadata_features: Tensor | None,
        layer_id: int | Tensor,
        token_mask: Tensor | None,
    ) -> Tensor:
        if score_state.dim() != 3:
            raise ValueError(
                f"score_state must have shape [B, N, D], got {tuple(score_state.shape)}"
            )
        B, N, Ds = score_state.shape
        if Ds != self.score_state_dim:
            raise ValueError(
                f"score_state last dim {Ds} does not match score_state_dim {self.score_state_dim}"
            )

        # --- defaults for optional inputs ---
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
                f"metadata_features dim {metadata_features.shape[-1]} does not match "
                f"{self.metadata_dim}"
            )

        if token_mask is None:
            token_mask = torch.ones(B, N, dtype=torch.bool, device=score_state.device)

        # Ensure mask is float for mean computation
        mask_float = token_mask.float().unsqueeze(-1)  # [B, N, 1]
        valid_counts = token_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # Masked mean pool over tokens: sum(masked) / count(valid)
        pooled_score = (score_state * mask_float).sum(dim=1) / valid_counts  # [B, Ds]
        pooled_metadata = (
            metadata_features.to(dtype=score_state.dtype) * mask_float
        ).sum(dim=1) / valid_counts  # [B, Dm]

        # Scalar feature: valid token count / 512
        token_count = (valid_counts / 512.0)  # [B, 1]

        # Layer embedding
        if isinstance(layer_id, Tensor):
            layer_tensor = layer_id.to(
                device=score_state.device, dtype=torch.long
            ).reshape(-1)
            if layer_tensor.numel() == 1:
                layer_tensor = layer_tensor.expand(B)
            elif layer_tensor.numel() != B:
                raise ValueError(
                    f"layer_id tensor must have 1 or B={B} elements, got {layer_tensor.numel()}"
                )
        else:
            layer_tensor = torch.full(
                (B,), int(layer_id), dtype=torch.long, device=score_state.device
            )
        layer_tensor = layer_tensor.clamp(0, self.layer_embed.num_embeddings - 1)
        layer_features = self.layer_embed(layer_tensor)  # [B, Ds]

        # Concatenate: [B, Ds + Dm + 1 + Ds]
        mlp_input = torch.cat(
            [
                pooled_score,
                pooled_metadata,
                token_count,
                layer_features.to(dtype=score_state.dtype),
            ],
            dim=-1,
        )

        return self.mlp(mlp_input)  # [B, num_candidates]

    def _forward_v2(
        self,
        score_state: Tensor,
        metadata_features: Tensor | None,
        layer_id: int | Tensor,
        token_mask: Tensor | None,
    ) -> Tensor:
        features = self._encoder(score_state, metadata_features, layer_id)
        return self._classifier(features, token_mask)  # [B, num_candidates]

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    def predict_count(self, logits: Tensor) -> Tensor:
        """Map logits to predicted candidate values via argmax.

        Parameters
        ----------
        logits : Tensor [B, num_candidates]

        Returns
        -------
        Tensor [B]
            Predicted candidate values (as a tensor, not Python int).
        """
        indices = logits.argmax(dim=-1)  # [B]
        return self.candidates[indices]  # [B]
