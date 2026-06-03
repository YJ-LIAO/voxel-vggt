"""Tests for FifoCountHead: shared classification head for FIFO count prediction."""

from __future__ import annotations

import os
import sys

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
