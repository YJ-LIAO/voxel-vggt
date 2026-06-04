"""Tests for the shared retention policy module.

Covers:
- Step 1.1: shape and input validation tests for RetentionTokenEncoder
- Step 1.5: gradient flow tests confirming shared encoder receives gradients
  from both token and count heads
"""

import os
import sys

import pytest
import torch
from torch import Tensor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.retention_policy import (
    FifoCountClassifier,
    JointRetentionPolicy,
    RetentionTokenEncoder,
    TokenRankingHead,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, N, Ds, Dm, H = 4, 64, 128, 17, 256


def _make_encoder(**kw):
    defaults = dict(score_state_dim=Ds, metadata_dim=Dm, hidden_dim=H, num_layers=24)
    defaults.update(kw)
    return RetentionTokenEncoder(**defaults)


def _make_joint(**kw):
    defaults = dict(
        score_state_dim=Ds,
        metadata_dim=Dm,
        hidden_dim=H,
        num_layers=24,
        count_candidates=(0, 8, 16, 32, 64, 128),
    )
    defaults.update(kw)
    return JointRetentionPolicy(**defaults)


# ===================================================================
# Step 1.1 — Shape and input validation tests
# ===================================================================


class TestRetentionTokenEncoderShape:
    """Verify output shapes and default behaviour."""

    def test_basic_output_shape(self):
        """RetentionTokenEncoder(score_state, metadata, layer_id) -> [B, N, H]."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        out = enc(score, meta, layer_id=3)
        assert out.shape == (B, N, H), f"expected {(B, N, H)}, got {tuple(out.shape)}"

    def test_metadata_none_pads_zeros(self):
        """metadata_features=None should auto-pad with zeros."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        out = enc(score, metadata_features=None, layer_id=0)
        assert out.shape == (B, N, H)

    def test_layer_id_int(self):
        """layer_id as plain int works."""
        enc = _make_encoder()
        score = torch.randn(2, 10, Ds)
        out = enc(score, None, layer_id=5)
        assert out.shape == (2, 10, H)

    def test_layer_id_scalar_tensor(self):
        """layer_id as a single-element tensor works."""
        enc = _make_encoder()
        score = torch.randn(2, 10, Ds)
        out = enc(score, None, layer_id=torch.tensor(7))
        assert out.shape == (2, 10, H)

    def test_layer_id_batch_tensor(self):
        """layer_id as a [B] tensor works."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        lids = torch.arange(B)
        out = enc(score, None, layer_id=lids)
        assert out.shape == (B, N, H)

class TestRetentionTokenEncoderValidation:
    """Verify that invalid inputs raise ValueError."""

    def test_score_state_wrong_dim(self):
        """score_state not [B, N, Ds] should raise ValueError."""
        enc = _make_encoder()
        # 2-D input
        with pytest.raises(ValueError, match="score_state"):
            enc(torch.randn(B, Ds), None, layer_id=0)

    def test_score_state_wrong_last_dim(self):
        """score_state last dim mismatch should raise ValueError."""
        enc = _make_encoder()
        wrong_dim = Ds + 1
        with pytest.raises(ValueError, match="score_state_dim"):
            enc(torch.randn(B, N, wrong_dim), None, layer_id=0)

    def test_metadata_token_mismatch(self):
        """metadata_features with wrong [B, N] should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_meta = torch.randn(B, N + 1, Dm)
        with pytest.raises(ValueError, match="metadata_features"):
            enc(score, bad_meta, layer_id=0)

    def test_metadata_dim_mismatch(self):
        """metadata_features with wrong last dim should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_meta = torch.randn(B, N, Dm + 1)
        with pytest.raises(ValueError, match="metadata_features dim"):
            enc(score, bad_meta, layer_id=0)

    def test_layer_id_wrong_batch_size(self):
        """layer_id tensor with wrong batch size should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_lids = torch.arange(B + 1)
        with pytest.raises(ValueError, match="layer_id"):
            enc(score, None, layer_id=bad_lids)


# ===================================================================
# Step 1.1 continued — TokenRankingHead and FifoCountClassifier shapes
# ===================================================================


class TestTokenRankingHeadShape:
    def test_output_shape(self):
        """TokenRankingHead maps [B, N, H] -> [B, N]."""
        head = TokenRankingHead(H)
        x = torch.randn(B, N, H)
        out = head(x)
        assert out.shape == (B, N)


class TestFifoCountClassifierShape:
    def test_output_shape(self):
        """FifoCountClassifier maps [B, N, H] -> [B, num_candidates]."""
        head = FifoCountClassifier(H)
        x = torch.randn(B, N, H)
        out = head(x)
        assert out.shape == (B, 6)  # default 6 candidates

    def test_output_shape_with_mask(self):
        """FifoCountClassifier respects token_mask."""
        head = FifoCountClassifier(H)
        x = torch.randn(B, N, H)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, N // 2:] = False
        out = head(x, token_mask=mask)
        assert out.shape == (B, 6)

    def test_predict_count(self):
        """predict_count returns candidate values."""
        head = FifoCountClassifier(H, candidates=(0, 8, 16, 32, 64, 128))
        logits = torch.zeros(B, 6)
        # Point index 2 (value 16) as winner
        logits[:, 2] = 10.0
        preds = head.predict_count(logits)
        assert preds.shape == (B,)
        assert (preds == 16).all()


class TestJointRetentionPolicyShape:
    def test_forward_token_shape(self):
        """forward_token returns [B, N]."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        out = joint.forward_token(score, meta, layer_id=0)
        assert out.shape == (B, N)

    def test_forward_count_shape(self):
        """forward_count returns [B, num_candidates]."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)
        out = joint.forward_count(score, meta, layer_id=0, token_mask=mask)
        assert out.shape == (B, 6)


# ===================================================================
# Step 1.5 — Gradient flow tests
# ===================================================================


class TestSharedEncoderGradients:
    """Verify both losses update the shared encoder."""

    def _assert_encoder_grad(self, joint, name=""):
        assert joint.encoder.layer_embed.weight.grad is not None, (
            f"{name}: layer_embed.weight.grad is None"
        )
        assert joint.encoder.proj.weight.grad is not None, (
            f"{name}: proj.weight.grad is None"
        )

    def test_combined_loss_updates_encoder(self):
        """Token loss + count loss -> encoder gets gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)
        layer_id = torch.tensor(2)

        token_loss = joint.forward_token(score, meta, layer_id).sum()
        count_loss = joint.forward_count(score, meta, layer_id, mask).sum()
        (token_loss + count_loss).backward()

        self._assert_encoder_grad(joint, "combined")

    def test_token_loss_alone_updates_encoder(self):
        """Token loss alone -> encoder gets non-zero gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)

        token_loss = joint.forward_token(score, meta, layer_id=0).sum()
        token_loss.backward()

        self._assert_encoder_grad(joint, "token_only")
        # Check gradients are actually non-zero
        assert joint.encoder.layer_embed.weight.grad.abs().sum() > 0, (
            "token_only: layer_embed grad is all zeros"
        )
        assert joint.encoder.proj.weight.grad.abs().sum() > 0, (
            "token_only: proj grad is all zeros"
        )

    def test_count_loss_alone_updates_encoder(self):
        """Count loss alone -> encoder gets non-zero gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)

        count_loss = joint.forward_count(score, meta, layer_id=0, token_mask=mask).sum()
        count_loss.backward()

        self._assert_encoder_grad(joint, "count_only")
        assert joint.encoder.layer_embed.weight.grad.abs().sum() > 0, (
            "count_only: layer_embed grad is all zeros"
        )
        assert joint.encoder.proj.weight.grad.abs().sum() > 0, (
            "count_only: proj grad is all zeros"
        )

    def test_individual_token_and_count_grads_nonzero(self):
        """Both token and count heads produce non-zero encoder gradients individually."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)

        # Token head
        joint.zero_grad()
        token_loss = joint.forward_token(score, meta, layer_id=0).sum()
        token_loss.backward()
        token_layer_grad = joint.encoder.layer_embed.weight.grad.clone()
        token_proj_grad = joint.encoder.proj.weight.grad.clone()

        # Count head
        joint.zero_grad()
        count_loss = joint.forward_count(score, meta, layer_id=0, token_mask=mask).sum()
        count_loss.backward()
        count_layer_grad = joint.encoder.layer_embed.weight.grad.clone()
        count_proj_grad = joint.encoder.proj.weight.grad.clone()

        # Both must be non-zero
        assert token_layer_grad.abs().sum() > 0, "token head: layer_embed grad zero"
        assert token_proj_grad.abs().sum() > 0, "token head: proj grad zero"
        assert count_layer_grad.abs().sum() > 0, "count head: layer_embed grad zero"
        assert count_proj_grad.abs().sum() > 0, "count head: proj grad zero"
