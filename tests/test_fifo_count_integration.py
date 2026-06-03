"""Integration tests for count head initialization and checkpoint loading (Task 7)."""

import tempfile
from pathlib import Path

import pytest
import torch

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig


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
