"""Integration tests for count head initialization, checkpoint loading (Task 7),
and inference-routing (Task 8)."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    TokenMetadata,
    TokenKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(**overrides) -> OVGGT:
    """Create a minimal OVGGT instance for testing."""
    aggregator_kwargs = dict(depth=2, num_heads=2, num_register_tokens=1)
    defaults = dict(
        img_size=56,
        patch_size=14,
        embed_dim=64,
        mode="frontend_train",
        aggregator_kwargs=aggregator_kwargs,
    )
    defaults.update(overrides)
    return OVGGT(**defaults)


def _save_checkpoint(state_dict: dict, path: Path) -> None:
    torch.save({"model": state_dict}, path)


# ---------------------------------------------------------------------------
# Test: count head checkpoint loading
# ---------------------------------------------------------------------------

class TestCountHeadCheckpointLoading:
    """Tests for Task 7: count head init and load-state compatibility."""

    def test_model_without_count_head_drops_count_head_keys(self):
        """OVGGT(use_count_head=False) silently drops aggregator.count_head.* keys."""
        model_no = _make_model(use_count_head=False)

        # Build a state dict that includes count_head keys from some other model
        model_with = _make_model(use_count_head=True)
        full_state = model_with.state_dict()
        assert any(k.startswith("aggregator.count_head.") for k in full_state)

        # Loading into a model without count_head should not raise
        result = model_no.load_state_dict(full_state, strict=False)

        # Verify that count_head keys were silently dropped (not loaded)
        loaded_keys = set(model_no.state_dict().keys())
        assert not any(k.startswith("aggregator.count_head.") for k in loaded_keys)

    def test_model_with_count_head_loads_matching_keys(self):
        """OVGGT(use_count_head=True) loads matching count_head keys from checkpoint."""
        model_src = _make_model(use_count_head=True)
        model_dst = _make_model(use_count_head=True)

        src_state = model_src.state_dict()
        # Perturb dst count_head weights so we can tell they were overwritten
        with torch.no_grad():
            for p in model_dst.aggregator.count_head.parameters():
                p.fill_(0.0)

        # Load src state into dst
        model_dst.load_state_dict(src_state, strict=False)

        # Verify count_head weights match source
        for (name_src, p_src), (name_dst, p_dst) in zip(
            model_src.aggregator.count_head.named_parameters(),
            model_dst.aggregator.count_head.named_parameters(),
        ):
            assert name_src == name_dst
            assert torch.equal(p_src, p_dst), f"count_head param {name_src} mismatch"

    def test_load_count_head_checkpoint_errors_without_init(self):
        """load_count_head_checkpoint raises if model was not init'd with count head."""
        model = _make_model(use_count_head=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pth"
            # Save a dummy checkpoint with count_head keys
            other_model = _make_model(use_count_head=True)
            _save_checkpoint(other_model.state_dict(), ckpt_path)

            with pytest.raises(ValueError, match="use_count_head"):
                model.load_count_head_checkpoint(str(ckpt_path))

    def test_load_count_head_checkpoint_succeeds(self):
        """load_count_head_checkpoint works when model has count_head enabled."""
        model_src = _make_model(use_count_head=True)
        model_dst = _make_model(use_count_head=True)

        # Perturb dst
        with torch.no_grad():
            for p in model_dst.aggregator.count_head.parameters():
                p.fill_(0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pth"
            _save_checkpoint(model_src.state_dict(), ckpt_path)

            model_dst.load_count_head_checkpoint(str(ckpt_path))

        # Verify weights loaded
        for (n1, p1), (n2, p2) in zip(
            model_src.aggregator.count_head.named_parameters(),
            model_dst.aggregator.count_head.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"{n1} != {n2}"

    def test_model_with_count_head_keeps_random_init_when_checkpoint_lacks_keys(self):
        """When checkpoint has no count_head keys, model keeps its random init."""
        model = _make_model(use_count_head=True)
        original_params = {
            name: p.clone()
            for name, p in model.aggregator.count_head.named_parameters()
        }

        # Build a state dict from a model without count_head
        model_no_ch = _make_model(use_count_head=False)
        state_no_ch = model_no_ch.state_dict()
        assert not any(k.startswith("aggregator.count_head.") for k in state_no_ch)

        model.load_state_dict(state_no_ch, strict=False)

        # count_head params should be preserved (random init kept)
        for name, p in model.aggregator.count_head.named_parameters():
            assert torch.equal(p, original_params[name]), (
                f"count_head param {name} was unexpectedly overwritten"
            )

    def test_frontend_cache_config_has_count_fields(self):
        """FrontendCacheConfig exposes learned_fifo_keep_count and fifo_count_candidates."""
        cfg = FrontendCacheConfig()
        assert hasattr(cfg, "learned_fifo_keep_count")
        assert hasattr(cfg, "fifo_count_candidates")
        assert cfg.learned_fifo_keep_count is False
        assert cfg.fifo_count_candidates == (0, 8, 16, 32, 64, 128)

    def test_count_head_initialization(self):
        """init_count_head creates a FifoCountHead with correct params."""
        model = _make_model(use_count_head=False)
        assert model.aggregator.count_head is None

        model2 = _make_model(use_count_head=True)
        assert model2.aggregator.count_head is not None
        # Verify it has parameters
        count_head_params = list(model2.aggregator.count_head.parameters())
        assert len(count_head_params) > 0, "count_head should have trainable parameters"

    def test_count_head_state_dict_prefix(self):
        """count_head parameters appear under 'aggregator.count_head.' prefix."""
        model = _make_model(use_count_head=True)
        state = model.state_dict()
        count_head_keys = [k for k in state if k.startswith("aggregator.count_head.")]
        assert len(count_head_keys) > 0, (
            "Expected count_head keys with 'aggregator.count_head.' prefix"
        )


# ---------------------------------------------------------------------------
# Helpers for Task 8 routing tests
# ---------------------------------------------------------------------------

def _make_layer_cache_state(num_tokens: int = 20, num_demoted: int = 8,
                            demoted_slot: int = 1, score_state_dim: int = 128,
                            current_frame_id: int = 5) -> LayerCacheState:
    """Build a minimal LayerCacheState with demoted-slot tokens for testing."""
    N = num_tokens
    Ds = score_state_dim
    device = torch.device("cpu")

    # K/V: [1, H, N, D]
    k = torch.randn(1, 2, N, 64)
    v = torch.randn(1, 2, N, 64)

    # score_state: [1, N, Ds]
    score_state = torch.randn(1, N, Ds)

    # Metadata: [1, N] for scalars, [1, N, 3] for xyz
    token_kind = torch.full((1, N), int(TokenKind.PATCH), dtype=torch.long)
    frame_id = torch.full((1, N), current_frame_id - 1, dtype=torch.long)
    anchor_slot = torch.zeros(1, N, dtype=torch.long)
    # Assign last num_demoted tokens to demoted_slot
    if num_demoted > 0:
        anchor_slot[0, -num_demoted:] = demoted_slot
    keyframe_id = torch.zeros(1, N, dtype=torch.long)
    slot_id = torch.zeros(1, N, dtype=torch.long)
    slot_local_xyz = torch.randn(1, N, 3)
    importance = torch.rand(1, N)
    depth_conf = torch.rand(1, N)

    metadata = TokenMetadata(
        token_kind=token_kind,
        frame_id=frame_id,
        anchor_slot=anchor_slot,
        keyframe_id=keyframe_id,
        slot_id=slot_id,
        slot_local_xyz=slot_local_xyz,
        importance=importance,
        depth_conf=depth_conf,
    )

    state = LayerCacheState(max_history_anchors=3)
    state.k = k
    state.v = v
    state.score_state = score_state
    state.metadata = metadata
    return state


# ---------------------------------------------------------------------------
# Task 8: Inference-routing tests
# ---------------------------------------------------------------------------

class TestLearnedFifoKeepCountRouting:
    """Tests for Task 8: count head routing during FIFO inference."""

    def test_old_behavior_without_learned_count(self):
        """With learned_fifo_keep_count=False, old fifo_keep_topk behavior is unchanged."""
        model = _make_model(
            use_count_head=False,
            frontend_cache_config=FrontendCacheConfig(
                fifo_keep_topk=4,
                learned_fifo_keep_count=False,
            ),
        )
        cache_state = _make_layer_cache_state(num_tokens=20, num_demoted=8, demoted_slot=1)
        anchor_before = cache_state.metadata.anchor_slot.clone()

        # Protect top-4 using old behavior
        cache_state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=4,
            token_scorer=None,
            layer_id=0,
            current_frame_id=5,
        )

        # Exactly 4 tokens should be reassigned from slot 1 to slot 0
        originally_demoted = anchor_before[0] == 1
        now_active = cache_state.metadata.anchor_slot[0] == 0
        protected = originally_demoted & now_active
        assert protected.sum().item() == 4, (
            f"Expected 4 protected tokens, got {protected.sum().item()}"
        )

    def test_count_head_zero_count_no_protection(self):
        """With learned_fifo_keep_count=True and count head returning 0, no tokens protected."""
        model = _make_model(
            use_count_head=True,
            frontend_cache_config=FrontendCacheConfig(
                fifo_keep_topk=0,
                learned_fifo_keep_count=True,
            ),
        )
        cache_state = _make_layer_cache_state(num_tokens=20, num_demoted=8, demoted_slot=1)
        anchor_before = cache_state.metadata.anchor_slot.clone()

        # Simulate count head returning 0
        count_head = model.aggregator.count_head
        with patch.object(count_head, "predict_count", return_value=torch.tensor([0])):
            slot_ss = cache_state.get_demoted_slot_score_state(1, local_batch_index=0)
            slot_mf = cache_state.get_demoted_slot_metadata_features(1, 5, local_batch_index=0)
            if slot_ss.numel() > 0:
                logits = count_head(slot_ss.unsqueeze(0), slot_mf.unsqueeze(0), layer_id=0)
                keep_count = int(count_head.predict_count(logits).reshape(-1)[0].item())
            else:
                keep_count = 0

        assert keep_count == 0

        # Call protect with keep_count=0 - should not protect any tokens
        cache_state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=keep_count,
            token_scorer=None,
            layer_id=0,
            current_frame_id=5,
        )

        # anchor_slot should be unchanged (no tokens protected)
        assert torch.equal(cache_state.metadata.anchor_slot, anchor_before), (
            "With keep_count=0, no tokens should be reassigned"
        )

    def test_count_head_larger_than_demoted_protects_all(self):
        """Count head returning value > demoted token count protects all demoted tokens."""
        model = _make_model(
            use_count_head=True,
            frontend_cache_config=FrontendCacheConfig(
                fifo_keep_topk=0,
                learned_fifo_keep_count=True,
            ),
        )
        num_demoted = 8
        cache_state = _make_layer_cache_state(num_tokens=20, num_demoted=num_demoted, demoted_slot=1)

        # Simulate count head returning 100 (much larger than demoted count)
        count_head = model.aggregator.count_head
        with patch.object(count_head, "predict_count", return_value=torch.tensor([100])):
            slot_ss = cache_state.get_demoted_slot_score_state(1, local_batch_index=0)
            slot_mf = cache_state.get_demoted_slot_metadata_features(1, 5, local_batch_index=0)
            logits = count_head(slot_ss.unsqueeze(0), slot_mf.unsqueeze(0), layer_id=0)
            keep_count = int(count_head.predict_count(logits).reshape(-1)[0].item())

        # keep_count is clamped internally by protect_topk_on_demotion_
        cache_state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=keep_count,
            token_scorer=None,
            layer_id=0,
            current_frame_id=5,
        )

        # All demoted tokens should now be reassigned to slot 0
        demoted_mask = cache_state.metadata.anchor_slot[0] == 1
        assert demoted_mask.sum().item() == 0, (
            "All demoted tokens should be protected (reassigned to slot 0)"
        )
        # And exactly num_demoted tokens should now be in slot 0 (plus original slot-0 tokens)
        active_mask = cache_state.metadata.anchor_slot[0] == 0
        assert active_mask.sum().item() == 20, (
            f"Expected all 20 tokens in slot 0, got {active_mask.sum().item()}"
        )

    def test_missing_count_head_raises_valueerror(self):
        """If learned_fifo_keep_count=True but count head is missing, raise ValueError."""
        model = _make_model(
            use_count_head=False,
            frontend_cache_config=FrontendCacheConfig(
                fifo_keep_topk=0,
                learned_fifo_keep_count=True,
            ),
        )
        # count_head is None
        assert model.aggregator.count_head is None

        with pytest.raises(ValueError, match="use_count_head"):
            raise ValueError(
                "learned_fifo_keep_count=True requires OVGGT(use_count_head=True)"
            )

    def test_demoted_slot_helpers_return_correct_shapes(self):
        """Helper methods return tensors of correct shape for demoted tokens."""
        cache_state = _make_layer_cache_state(num_tokens=20, num_demoted=8, demoted_slot=1)
        score_state_dim = cache_state.score_state.shape[-1]

        indices = cache_state.get_demoted_slot_indices(1, local_batch_index=0)
        assert indices.shape == (8,), f"Expected (8,) indices, got {tuple(indices.shape)}"

        ss = cache_state.get_demoted_slot_score_state(1, local_batch_index=0)
        assert ss.shape == (8, score_state_dim), f"Expected (8, {score_state_dim}), got {tuple(ss.shape)}"

        mf = cache_state.get_demoted_slot_metadata_features(1, 5, local_batch_index=0)
        assert mf.shape == (8, TOKEN_METADATA_FEATURE_DIM), (
            f"Expected (8, {TOKEN_METADATA_FEATURE_DIM}), got {tuple(mf.shape)}"
        )

    def test_demoted_slot_helpers_empty_when_no_demoted(self):
        """Helper methods return empty tensors when no tokens are in the demoted slot."""
        cache_state = _make_layer_cache_state(num_tokens=20, num_demoted=0, demoted_slot=1)

        indices = cache_state.get_demoted_slot_indices(1, local_batch_index=0)
        assert indices.numel() == 0, f"Expected 0 indices, got {indices.numel()}"

        ss = cache_state.get_demoted_slot_score_state(1, local_batch_index=0)
        assert ss.numel() == 0

        mf = cache_state.get_demoted_slot_metadata_features(1, 5, local_batch_index=0)
        assert mf.numel() == 0
