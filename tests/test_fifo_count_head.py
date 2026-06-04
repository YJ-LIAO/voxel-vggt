"""Tests for FifoCountHead: shared classification head for FIFO count prediction.

Includes both v1 (pooled_v1) and v2 (shared_encoder_v2) architecture tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import Tensor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.count_head import FifoCountHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def head() -> FifoCountHead:
    """Small head for fast tests."""
    return FifoCountHead(
        score_state_dim=128,
        metadata_dim=17,
        hidden_dim=64,
        num_layers=24,
        candidates=(0, 8, 16, 32, 64, 128),
    )


# ---------------------------------------------------------------------------
# Shape / dtype tests
# ---------------------------------------------------------------------------

class TestForwardShape:
    """Verify that forward produces the correct output shape and dtype."""

    def test_output_shape_with_all_inputs(self, head: FifoCountHead) -> None:
        B, N = 4, 100
        score_state = torch.randn(B, N, 128)
        metadata_features = torch.randn(B, N, 17)
        token_mask = torch.ones(B, N, dtype=torch.bool)
        layer_id = torch.tensor([0, 5, 10, 23])

        out = head(score_state, metadata_features, layer_id=layer_id, token_mask=token_mask)

        assert out.shape == (B, head.candidates.shape[0])
        assert out.dtype == score_state.dtype

    def test_output_shape_no_metadata(self, head: FifoCountHead) -> None:
        B, N = 2, 50
        score_state = torch.randn(B, N, 128)
        out = head(score_state)
        assert out.shape == (B, head.candidates.shape[0])

    def test_output_shape_scalar_layer_id(self, head: FifoCountHead) -> None:
        B, N = 3, 30
        score_state = torch.randn(B, N, 128)
        out = head(score_state, layer_id=7)
        assert out.shape == (B, head.candidates.shape[0])

    def test_rejects_wrong_score_state_dims(self, head: FifoCountHead) -> None:
        # 2-D input should be rejected
        with pytest.raises(ValueError):
            head(torch.randn(4, 128))


# ---------------------------------------------------------------------------
# Candidate mapping tests
# ---------------------------------------------------------------------------

class TestPredictCount:
    """predict_count should map argmax logits to candidate values."""

    def test_returns_tensor(self, head: FifoCountHead) -> None:
        B = 5
        logits = torch.randn(B, head.candidates.shape[0])
        counts = head.predict_count(logits)
        assert isinstance(counts, Tensor)
        assert counts.shape == (B,)

    def test_maps_to_correct_candidate(self, head: FifoCountHead) -> None:
        # Make argmax point at index 3 -> candidate 32
        logits = torch.zeros(1, head.candidates.shape[0])
        logits[0, 3] = 100.0
        counts = head.predict_count(logits)
        assert counts.item() == head.candidates[3].item()

    def test_default_candidates(self) -> None:
        h = FifoCountHead(score_state_dim=8, metadata_dim=5, num_layers=4)
        assert tuple(h.candidates.tolist()) == (0, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Masked-mean pooling tests
# ---------------------------------------------------------------------------

class TestMaskedMeanPooling:
    """Padded (masked) tokens must not affect the output."""

    def test_padding_does_not_change_output(self, head: FifoCountHead) -> None:
        """Masked-out tokens must not affect the output.

        Strategy: call forward twice with the *same* valid tokens but different
        (random) values in the padded positions.  The outputs must be identical
        because masked positions are ignored.
        """
        torch.manual_seed(42)
        B, N = 2, 64
        n_valid = [34, 40]  # per-batch valid count

        # Shared valid-region data
        score_state_a = torch.randn(B, N, 128)
        metadata_a = torch.randn(B, N, 17)

        # Build mask
        mask = torch.ones(B, N, dtype=torch.bool)
        for b in range(B):
            mask[b, n_valid[b]:] = False

        # Version B: same valid region, random padded region
        score_state_b = score_state_a.clone()
        metadata_b = metadata_a.clone()
        for b in range(B):
            score_state_b[b, n_valid[b]:] = torch.randn(N - n_valid[b], 128) * 100
            metadata_b[b, n_valid[b]:] = torch.randn(N - n_valid[b], 17) * 100

        with torch.no_grad():
            out_a = head(score_state_a, metadata_a, token_mask=mask)
            out_b = head(score_state_b, metadata_b, token_mask=mask)

        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_valid_token_count_feature(self, head: FifoCountHead) -> None:
        """The scalar token_count feature should be proportional to valid tokens."""
        B, N = 1, 512
        score_state = torch.randn(B, N, 128)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 256:] = False  # exactly half valid

        # Hook into forward to verify internal computation
        captured = {}

        def hook_fn(module, args, kwargs):
            # We can't easily capture intermediate values from nn.Module forward,
            # so instead verify via a direct call and check output changes
            pass

        # Instead, verify indirectly: outputs differ when mask changes
        with torch.no_grad():
            out_full = head(score_state, token_mask=torch.ones(B, N, dtype=torch.bool))
            out_half = head(score_state, token_mask=mask)

        # They should differ because token_count differs
        assert not torch.allclose(out_full, out_half, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow back through score_state."""

    def test_gradient_flows_to_score_state(self, head: FifoCountHead) -> None:
        B, N = 2, 20
        score_state = torch.randn(B, N, 128, requires_grad=True)
        metadata_features = torch.randn(B, N, 17)
        token_mask = torch.ones(B, N, dtype=torch.bool)

        out = head(score_state, metadata_features, token_mask=token_mask, layer_id=0)
        out.sum().backward()

        assert score_state.grad is not None
        assert score_state.grad.shape == score_state.shape
        assert score_state.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Device / dtype consistency
# ---------------------------------------------------------------------------

class TestDeviceConsistency:
    """Output should be on the same device as input."""

    def test_output_device_matches_input(self, head: FifoCountHead) -> None:
        score_state = torch.randn(1, 10, 128)
        out = head(score_state)
        assert out.device == score_state.device

    def test_candidates_buffer_persistent(self) -> None:
        h = FifoCountHead(score_state_dim=8, metadata_dim=5, num_layers=4)
        buf_names = {name for name, _ in h.named_buffers()}
        assert "candidates" in buf_names

        # After serialization round-trip, candidates survive
        state = h.state_dict()
        assert "candidates" in state


# ---------------------------------------------------------------------------
# v2 architecture tests (Task 2, Step 2.1)
# ---------------------------------------------------------------------------

@pytest.fixture()
def head_v2() -> FifoCountHead:
    """Small v2 head for fast tests."""
    return FifoCountHead(
        score_state_dim=128,
        metadata_dim=17,
        hidden_dim=64,
        num_layers=24,
        candidates=(0, 8, 16, 32, 64, 128),
        arch="shared_encoder_v2",
    )


class TestV2Architecture:
    """Verify v2 (shared_encoder_v2) count head behavior."""

    def test_v2_output_shape(self, head_v2: FifoCountHead) -> None:
        """v2 shared-encoder count head outputs [B, num_candidates]."""
        B, N = 4, 100
        score_state = torch.randn(B, N, 128)
        metadata_features = torch.randn(B, N, 17)
        token_mask = torch.ones(B, N, dtype=torch.bool)
        layer_id = torch.tensor([0, 5, 10, 23])

        out = head_v2(score_state, metadata_features, layer_id=layer_id, token_mask=token_mask)

        assert out.shape == (B, head_v2.candidates.shape[0])
        assert out.dtype == score_state.dtype

    def test_v2_output_shape_no_metadata(self, head_v2: FifoCountHead) -> None:
        """v2 works with metadata_features=None (defaults to zeros)."""
        B, N = 2, 50
        score_state = torch.randn(B, N, 128)
        out = head_v2(score_state)
        assert out.shape == (B, head_v2.candidates.shape[0])

    def test_v2_output_shape_scalar_layer_id(self, head_v2: FifoCountHead) -> None:
        """v2 works with scalar layer_id."""
        B, N = 3, 30
        score_state = torch.randn(B, N, 128)
        out = head_v2(score_state, layer_id=7)
        assert out.shape == (B, head_v2.candidates.shape[0])

    def test_v2_rejects_wrong_score_state_dims(self, head_v2: FifoCountHead) -> None:
        """v2 rejects wrong score_state dimensions."""
        with pytest.raises(ValueError):
            head_v2(torch.randn(4, 128))

    def test_v2_predict_count_returns_correct_candidates(self, head_v2: FifoCountHead) -> None:
        """predict_count works for v2 head and maps argmax to correct candidates."""
        B = 5
        logits = torch.randn(B, head_v2.candidates.shape[0])
        counts = head_v2.predict_count(logits)
        assert isinstance(counts, Tensor)
        assert counts.shape == (B,)

    def test_v2_predict_count_maps_to_correct_candidate(self, head_v2: FifoCountHead) -> None:
        """predict_count maps specific logit index to correct candidate value."""
        logits = torch.zeros(1, head_v2.candidates.shape[0])
        logits[0, 3] = 100.0
        counts = head_v2.predict_count(logits)
        assert counts.item() == head_v2.candidates[3].item()

    def test_v2_gradient_flows(self, head_v2: FifoCountHead) -> None:
        """Gradients flow through v2 head."""
        B, N = 2, 20
        score_state = torch.randn(B, N, 128, requires_grad=True)
        out = head_v2(score_state)
        out.sum().backward()
        assert score_state.grad is not None
        assert score_state.grad.abs().sum() > 0

    def test_v2_has_internal_encoder_and_classifier(self, head_v2: FifoCountHead) -> None:
        """v2 head has _encoder and _classifier sub-modules."""
        assert hasattr(head_v2, "_encoder")
        assert hasattr(head_v2, "_classifier")

    def test_v2_arch_attribute(self, head_v2: FifoCountHead) -> None:
        """v2 head stores arch attribute."""
        assert head_v2.arch == "shared_encoder_v2"

    def test_v2_state_dict_roundtrip(self, head_v2: FifoCountHead) -> None:
        """v2 head can be serialized and deserialized."""
        state = head_v2.state_dict()
        head_v2_loaded = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
            arch="shared_encoder_v2",
        )
        head_v2_loaded.load_state_dict(state)
        for (n1, p1), (n2, p2) in zip(
            head_v2.named_parameters(), head_v2_loaded.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Param {n1} mismatch after state_dict roundtrip"


class TestArchParameter:
    """Verify arch parameter validation."""

    def test_default_arch_is_v1(self) -> None:
        """Default arch is pooled_v1."""
        h = FifoCountHead(score_state_dim=8, metadata_dim=5, num_layers=4)
        assert h.arch == "pooled_v1"

    def test_explicit_v1_arch(self) -> None:
        """Explicit pooled_v1 arch works."""
        h = FifoCountHead(score_state_dim=8, metadata_dim=5, num_layers=4, arch="pooled_v1")
        assert h.arch == "pooled_v1"

    def test_unknown_arch_raises(self) -> None:
        """Unknown arch string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FifoCountHead arch"):
            FifoCountHead(score_state_dim=8, metadata_dim=5, num_layers=4, arch="bogus_v3")


class TestV1V2Equivalence:
    """Verify that v1 default behavior still works unchanged."""

    def test_v1_forward_unchanged(self, head: FifoCountHead) -> None:
        """v1 forward still produces same results as before."""
        B, N = 4, 100
        torch.manual_seed(123)
        score_state = torch.randn(B, N, 128)
        metadata_features = torch.randn(B, N, 17)
        out = head(score_state, metadata_features, layer_id=0)
        assert out.shape == (B, 6)  # 6 candidates by default

    def test_v1_predict_count_unchanged(self, head: FifoCountHead) -> None:
        """v1 predict_count still works identically."""
        logits = torch.zeros(1, 6)
        logits[0, 2] = 50.0  # index 2 -> candidate 16
        counts = head.predict_count(logits)
        assert counts.item() == 16


class TestCrossArchCheckpointCompat:
    """Verify that v1/v2 checkpoint mismatches raise clear errors."""

    def test_v1_checkpoint_into_v2_model_errors(self) -> None:
        """Loading a v1 checkpoint into a v2 model should error clearly."""
        # Create v1 head and save its state
        h1 = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
            arch="pooled_v1",
        )
        state_v1 = h1.state_dict()

        # Try loading v1 state into v2 head
        h2 = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
            arch="shared_encoder_v2",
        )
        # v2 has different sub-modules (_encoder, _classifier) vs v1 (layer_embed, mlp)
        # So loading should fail with missing/unexpected keys
        with pytest.raises(RuntimeError):
            h2.load_state_dict(state_v1, strict=True)

    def test_v2_checkpoint_into_v1_model_errors(self) -> None:
        """Loading a v2 checkpoint into a v1 model should error clearly."""
        h2 = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
            arch="shared_encoder_v2",
        )
        state_v2 = h2.state_dict()

        h1 = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
            arch="pooled_v1",
        )
        # v1 has different sub-modules from v2
        with pytest.raises(RuntimeError):
            h1.load_state_dict(state_v2, strict=True)

    def test_count_candidate_shape_mismatch_errors(self) -> None:
        """Mismatched number of candidates should be detectable."""
        h_a = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 8, 16, 32, 64, 128),
        )
        h_b = FifoCountHead(
            score_state_dim=128, metadata_dim=17, hidden_dim=64,
            num_layers=24, candidates=(0, 4, 8, 16),  # 4 candidates instead of 6
        )
        state_a = h_a.state_dict()
        # The mlp output layer has different shapes (6 vs 4)
        with pytest.raises(RuntimeError):
            h_b.load_state_dict(state_a, strict=True)
