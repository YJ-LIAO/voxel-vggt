import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.aggregator import Aggregator
from ovggt.utils.frontend_cache import FrontendCacheConfig, LayerCacheState, PendingLayerUpdate


def test_aggregator_frontend_cache_mode_returns_pending_updates_with_conv_patch_embed():
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_register_tokens=1,
        patch_embed="conv",
        aa_block_size=2,
    )
    cache_states = [LayerCacheState() for _ in range(model.depth)]
    images = torch.rand(1, 1, 3, 28, 28)
    outputs, patch_start_idx, returned_states, pending, distill = model(
        images,
        cache_states=cache_states,
        use_cache=True,
        past_frame_idx=0,
        per_layer_budget=16,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )

    assert len(outputs) == model.depth
    assert patch_start_idx == 2
    assert returned_states is cache_states
    assert distill is None
    assert len(pending) == model.depth
    assert all(isinstance(item, PendingLayerUpdate) for item in pending)


def test_aggregator_frontend_cache_mode_validates_cache_states_length():
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_register_tokens=1,
        patch_embed="conv",
        aa_block_size=2,
    )
    images = torch.rand(1, 1, 3, 28, 28)

    with pytest.raises(ValueError, match="cache_states must have length 2, got 1"):
        model(
            images,
            cache_states=[LayerCacheState()],
            use_cache=True,
            past_frame_idx=0,
            per_layer_budget=16,
            frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        )
