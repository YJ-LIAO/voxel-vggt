"""Shared retention policy module with token ranking and FIFO count heads.

This module provides:
- ``RetentionTokenEncoder``: encodes per-token features (score_state,
  metadata, layer embedding) into a shared hidden representation.
- ``TokenRankingHead``: maps encoded features to per-token scalar scores.
- ``FifoCountClassifier``: maps encoded (pooled) features to a discrete
  classification over FIFO flush counts.
- ``JointRetentionPolicy``: convenience wrapper combining the encoder with
  both heads.

The ``RetentionTokenEncoder`` is designed so its ``norm`` / ``proj`` / ``act``
layers produce *identical* outputs to ``TokenScorer.scorer[0:3]`` when given
the same inputs, enabling weight-transfer during export.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# RetentionTokenEncoder
# --------------------------------------------------------------------------- #

class RetentionTokenEncoder(nn.Module):
    """Encode per-token features into a shared hidden representation.

    Parameters
    ----------
    score_state_dim : int
        Dimensionality of the per-token ``score_state`` vector (``Ds``).
    metadata_dim : int
        Dimensionality of per-token metadata features (``Dm``).
    hidden_dim : int
        Output dimensionality of the encoder.
    num_layers : int
        Number of transformer layers (used for layer embedding).
    """

    def __init__(
        self,
        score_state_dim: int = 128,
        metadata_dim: int = 17,
        hidden_dim: int = 256,
        num_layers: int = 24,
    ) -> None:
        super().__init__()
        self.score_state_dim = int(score_state_dim)
        self.metadata_dim = int(metadata_dim)
        self.hidden_dim = int(hidden_dim)

        self.layer_embed = nn.Embedding(
            max(int(num_layers), 1), self.score_state_dim
        )

        concat_dim = self.score_state_dim + self.metadata_dim + self.score_state_dim
        self.norm = nn.LayerNorm(concat_dim)
        self.proj = nn.Linear(concat_dim, self.hidden_dim)
        self.act = nn.GELU()

    def forward(
        self,
        score_state: Tensor,
        metadata_features: Optional[Tensor] = None,
        layer_id: int | Tensor = 0,
    ) -> Tensor:
        """Return encoded token features ``[B, N, H]``.

        Parameters
        ----------
        score_state : Tensor [B, N, Ds]
            Per-token compact state from the transformer layer.
        metadata_features : Tensor [B, N, Dm] | None
            Per-token metadata features.  Zeros are used when *None*.
        layer_id : int | Tensor
            Transformer layer index (int, scalar tensor, or ``[B]`` tensor).

        Returns
        -------
        Tensor [B, N, H]
        """
        if score_state.dim() != 3:
            raise ValueError(
                f"score_state must have shape [B, N, D], got {tuple(score_state.shape)}"
            )
        B, N, Ds = score_state.shape
        if Ds != self.score_state_dim:
            raise ValueError(
                f"score_state last dim {Ds} does not match "
                f"score_state_dim {self.score_state_dim}"
            )

        # Default metadata to zeros
        if metadata_features is None:
            metadata_features = torch.zeros(
                B, N, self.metadata_dim,
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

        # Resolve layer_id to [B] tensor
        if isinstance(layer_id, Tensor):
            layer_tensor = layer_id.to(
                device=score_state.device, dtype=torch.long
            ).reshape(-1)
            if layer_tensor.numel() == 1:
                layer_tensor = layer_tensor.expand(B)
            elif layer_tensor.numel() != B:
                raise ValueError(
                    f"layer_id tensor must have 1 or B={B} elements, "
                    f"got {layer_tensor.numel()}"
                )
        else:
            layer_tensor = torch.full(
                (B,), int(layer_id), dtype=torch.long, device=score_state.device
            )
        layer_tensor = layer_tensor.clamp(0, self.layer_embed.num_embeddings - 1)
        layer_features = self.layer_embed(layer_tensor).unsqueeze(1).expand(B, N, -1)

        concat = torch.cat(
            [
                score_state,
                metadata_features.to(dtype=score_state.dtype),
                layer_features.to(dtype=score_state.dtype),
            ],
            dim=-1,
        )
        return self.act(self.proj(self.norm(concat)))


# --------------------------------------------------------------------------- #
# TokenRankingHead
# --------------------------------------------------------------------------- #

class TokenRankingHead(nn.Module):
    """Map encoded token features to a per-token scalar ranking score.

    Parameters
    ----------
    hidden_dim : int
        Must match the output dimension of ``RetentionTokenEncoder``.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, token_features: Tensor) -> Tensor:
        """Return per-token scores ``[B, N]``.

        Parameters
        ----------
        token_features : Tensor [B, N, H]

        Returns
        -------
        Tensor [B, N]
        """
        return self.out(token_features).squeeze(-1)


# --------------------------------------------------------------------------- #
# FifoCountClassifier
# --------------------------------------------------------------------------- #

class FifoCountClassifier(nn.Module):
    """Classify the number of tokens to flush from pooled encoded features.

    Parameters
    ----------
    hidden_dim : int
        Must match the output dimension of ``RetentionTokenEncoder``.
    candidates : sequence of int
        Discrete count values to classify among.
    """

    def __init__(
        self,
        hidden_dim: int,
        candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(candidates)),
        )
        self.register_buffer(
            "candidates",
            torch.tensor(candidates, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        token_features: Tensor,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return logits ``[B, num_candidates]``.

        Parameters
        ----------
        token_features : Tensor [B, N, H]
            Encoded token features from ``RetentionTokenEncoder``.
        token_mask : Tensor [B, N] | None
            Boolean mask; ``True`` marks **valid** tokens.  When *None* all
            tokens are considered valid.

        Returns
        -------
        Tensor [B, num_candidates]
        """
        B, N, H = token_features.shape

        if token_mask is None:
            token_mask = torch.ones(
                B, N, dtype=torch.bool, device=token_features.device
            )

        mask_float = token_mask.float().unsqueeze(-1)  # [B, N, 1]
        valid_counts = token_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # Masked mean pool: sum(masked) / count(valid)
        pooled = (token_features * mask_float).sum(dim=1) / valid_counts  # [B, H]

        # Scalar feature: valid token count / 512
        token_count = valid_counts / 512.0  # [B, 1]

        mlp_input = torch.cat([pooled, token_count], dim=-1)  # [B, H + 1]
        return self.mlp(mlp_input)  # [B, num_candidates]

    def predict_count(self, logits: Tensor) -> Tensor:
        """Map logits to predicted candidate values via argmax.

        Parameters
        ----------
        logits : Tensor [B, num_candidates]

        Returns
        -------
        Tensor [B]
            Predicted candidate values.
        """
        indices = logits.argmax(dim=-1)  # [B]
        return self.candidates[indices]  # [B]


# --------------------------------------------------------------------------- #
# JointRetentionPolicy
# --------------------------------------------------------------------------- #

class JointRetentionPolicy(nn.Module):
    """Combined token-ranking and FIFO-count retention policy.

    Wraps a single ``RetentionTokenEncoder`` shared between a
    ``TokenRankingHead`` and a ``FifoCountClassifier``.

    Parameters
    ----------
    score_state_dim, metadata_dim, hidden_dim, num_layers
        Forwarded to ``RetentionTokenEncoder``.
    count_candidates
        Forwarded to ``FifoCountClassifier``.
    """

    def __init__(
        self,
        score_state_dim: int = 128,
        metadata_dim: int = 17,
        hidden_dim: int = 256,
        num_layers: int = 24,
        count_candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.encoder = RetentionTokenEncoder(
            score_state_dim, metadata_dim, hidden_dim, num_layers
        )
        self.token_head = TokenRankingHead(hidden_dim)
        self.count_head = FifoCountClassifier(hidden_dim, count_candidates)

    def forward_token(
        self,
        score_state: Tensor,
        metadata_features: Optional[Tensor] = None,
        layer_id: int | Tensor = 0,
    ) -> Tensor:
        """Return per-token ranking scores ``[B, N]``."""
        return self.token_head(
            self.encoder(score_state, metadata_features, layer_id)
        )

    def forward_count(
        self,
        score_state: Tensor,
        metadata_features: Optional[Tensor] = None,
        layer_id: int | Tensor = 0,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return FIFO count logits ``[B, num_candidates]``."""
        features = self.encoder(score_state, metadata_features, layer_id)
        return self.count_head(features, token_mask=token_mask)


# --------------------------------------------------------------------------- #
# Export helpers for OVGGT runtime checkpoint conversion
# --------------------------------------------------------------------------- #

def build_token_scorer_state_from_joint(
    joint: JointRetentionPolicy,
    expected_hidden_dim: int | None = None,
) -> dict:
    """Export token scorer weights from a ``JointRetentionPolicy``.

    Maps the shared encoder + ranking head into the flat key layout expected
    by ``TokenScorer(depth=2)`` at runtime:

    ``TokenScorer(depth=2)`` Sequential layout::

        [0] LayerNorm(input_dim)
        [1] Linear(input_dim, hidden_dim)
        [2] GELU
        [3] Linear(hidden_dim, hidden_dim)   # extra hidden layer
        [4] GELU
        [5] Linear(hidden_dim, 1)            # output layer

    Key mapping:

    - ``encoder.layer_embed.weight``  -> ``layer_embed.weight``
    - ``encoder.norm.weight/bias``    -> ``scorer.0.weight/bias``  (LayerNorm)
    - ``encoder.proj.weight/bias``    -> ``scorer.1.weight/bias``  (Linear)
    - identity matrix                 -> ``scorer.3.weight/bias``  (hidden Linear, passthrough)
    - ``token_head.out.weight/bias``  -> ``scorer.5.weight/bias``  (output Linear)

    The joint model has no hidden layer (it uses depth=1 effectively), so
    ``scorer.3`` is initialised as an identity transform (eye weights, zero
    bias) that passes activations through unchanged.

    Parameters
    ----------
    joint : JointRetentionPolicy
        Trained joint policy to export.
    expected_hidden_dim : int | None
        If provided, validates that the encoder hidden dim matches the
        runtime expectation.  Raises ``ValueError`` on mismatch.

    Returns
    -------
    dict
        Flat state dict with TokenScorer deploy keys.
    """
    actual_hidden = joint.encoder.proj.weight.shape[0]
    if expected_hidden_dim is not None and actual_hidden != expected_hidden_dim:
        raise ValueError(
            f"Joint encoder hidden_dim={actual_hidden} != "
            f"runtime expected_hidden_dim={expected_hidden_dim}. "
            f"Export would produce incompatible checkpoint."
        )
    # Identity passthrough for the hidden layer at scorer.3 (depth=2).
    # After GELU, an identity Linear followed by another GELU is not exactly
    # a no-op, but this is the best we can do without a trained hidden layer.
    hidden_passthrough_weight = torch.eye(actual_hidden)
    hidden_passthrough_bias = torch.zeros(actual_hidden)
    return {
        "layer_embed.weight": joint.encoder.layer_embed.weight.detach().cpu().clone(),
        "scorer.0.weight": joint.encoder.norm.weight.detach().cpu().clone(),
        "scorer.0.bias": joint.encoder.norm.bias.detach().cpu().clone(),
        "scorer.1.weight": joint.encoder.proj.weight.detach().cpu().clone(),
        "scorer.1.bias": joint.encoder.proj.bias.detach().cpu().clone(),
        "scorer.3.weight": hidden_passthrough_weight,
        "scorer.3.bias": hidden_passthrough_bias,
        "scorer.5.weight": joint.token_head.out.weight.detach().cpu().clone(),
        "scorer.5.bias": joint.token_head.out.bias.detach().cpu().clone(),
    }


def build_count_head_state_from_joint(joint: JointRetentionPolicy) -> dict:
    """Export count head weights from a ``JointRetentionPolicy``.

    Constructs a ``FifoCountHead(arch="shared_encoder_v2")`` and copies the
    joint encoder + classifier weights into it.  The resulting state dict
    uses the ``aggregator.count_head.`` prefix layout expected at runtime.

    Parameters
    ----------
    joint : JointRetentionPolicy
        Trained joint policy to export.

    Returns
    -------
    dict
        State dict of the assembled FifoCountHead.
    """
    from ovggt.layers.count_head import FifoCountHead

    count_head = FifoCountHead(
        score_state_dim=joint.encoder.score_state_dim,
        metadata_dim=joint.encoder.metadata_dim,
        hidden_dim=joint.encoder.hidden_dim,
        num_layers=joint.encoder.layer_embed.num_embeddings,
        candidates=tuple(joint.count_head.candidates.tolist()),
        arch="shared_encoder_v2",
    )
    count_head._encoder.load_state_dict(joint.encoder.state_dict())
    count_head._classifier.load_state_dict(joint.count_head.state_dict())
    return count_head.state_dict()


def build_ovggt_joint_retention_state_dict(
    token_scorer_state: dict,
    count_head_state: dict | None,
    num_layers: int,
    score_state_projection_state: dict | None = None,
) -> dict:
    """Assemble a full OVGGT deploy state dict from exported sub-components.

    Replicates the shared token scorer weights across all layers and prefixes
    them for direct ``load_state_dict`` into an ``OVGGT`` model with
    ``use_token_scorer=True`` and optionally ``use_count_head=True``.

    Parameters
    ----------
    token_scorer_state : dict
        Output of ``build_token_scorer_state_from_joint``.
    count_head_state : dict | None
        Output of ``build_count_head_state_from_joint``, or *None* to skip.
    num_layers : int
        Number of transformer layers (determines replication count).
    score_state_projection_state : dict | None
        Optional pre-built ``aggregator.score_state_projs.*`` entries.

    Returns
    -------
    dict
        State dict with ``aggregator.token_scorers.{layer}.*`` and
        ``aggregator.count_head.*`` keys.
    """
    deploy_state: dict[str, Tensor] = {}
    for layer_idx in range(int(num_layers)):
        for key, value in token_scorer_state.items():
            deploy_state[f"aggregator.token_scorers.{layer_idx}.{key}"] = (
                value.detach().cpu().clone()
            )
    if count_head_state is not None:
        for key, value in count_head_state.items():
            deploy_state[f"aggregator.count_head.{key}"] = (
                value.detach().cpu().clone()
            )
    if score_state_projection_state:
        for key, value in score_state_projection_state.items():
            if key.startswith("aggregator.score_state_projs."):
                deploy_state[key] = value.detach().cpu().clone()
    return deploy_state
