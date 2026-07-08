import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.rope import PositionGetter
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


def test_aggregator_forward_supports_rope_disabled_with_special_tokens():
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_register_tokens=1,
        patch_embed="conv",
        aa_block_size=2,
        rope_freq=-1,
    )
    images = torch.rand(1, 1, 3, 28, 28)

    outputs, patch_start_idx = model(images)

    assert patch_start_idx == 2
    assert len(outputs) == model.depth


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


def test_aggregator_frontend_cache_mode_passes_layer_protected_count_to_blocks(monkeypatch):
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
    cache_states = [LayerCacheState(protected_count=3), LayerCacheState(protected_count=5)]
    images = torch.rand(1, 1, 3, 28, 28)
    seen = []

    def make_forward_stub(layer_idx):
        def forward_stub(tokens, **kwargs):
            seen.append((layer_idx, kwargs.get("anchor_token_count")))
            batch_size, num_tokens, channels = tokens.shape
            num_heads = model.global_blocks[layer_idx].attn.num_heads
            head_dim = channels // num_heads
            k_current = torch.zeros(batch_size, num_heads, num_tokens, head_dim)
            v_current = torch.zeros_like(k_current)
            importance = torch.ones(batch_size, num_tokens)
            return tokens, (k_current, v_current), importance

        return forward_stub

    for idx, block in enumerate(model.global_blocks):
        monkeypatch.setattr(block, "forward", make_forward_stub(idx))

    model(
        images,
        cache_states=cache_states,
        use_cache=True,
        past_frame_idx=3,
        per_layer_budget=16,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )

    assert seen == [(0, 3), (1, 5)]


def test_aggregator_frontend_cache_mode_forwards_window_token_count(monkeypatch):
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
    seen = []

    def make_forward_stub(layer_idx):
        def forward_stub(tokens, **kwargs):
            seen.append((layer_idx, kwargs.get("window_token_count")))
            batch_size, num_tokens, channels = tokens.shape
            num_heads = model.global_blocks[layer_idx].attn.num_heads
            head_dim = channels // num_heads
            k_current = torch.zeros(batch_size, num_heads, num_tokens, head_dim)
            v_current = torch.zeros_like(k_current)
            importance = torch.ones(batch_size, num_tokens)
            return tokens, (k_current, v_current), importance

        return forward_stub

    for idx, block in enumerate(model.global_blocks):
        monkeypatch.setattr(block, "forward", make_forward_stub(idx))

    model(
        images,
        cache_states=[LayerCacheState(), LayerCacheState()],
        use_cache=True,
        past_frame_idx=3,
        per_layer_budget=16,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        window_token_count=7,
    )

    assert seen == [(0, 7), (1, 7)]


def test_aggregator_frontend_cache_mode_preserves_attention_keep_indices(monkeypatch):
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
    expected = [
        torch.tensor([[0, 3, 4]], dtype=torch.long),
        torch.tensor([[1, 2, 5]], dtype=torch.long),
    ]

    def make_forward_stub(layer_idx):
        def forward_stub(tokens, **kwargs):
            batch_size, num_tokens, channels = tokens.shape
            num_heads = model.global_blocks[layer_idx].attn.num_heads
            head_dim = channels // num_heads
            k_current = torch.zeros(batch_size, num_heads, num_tokens, head_dim)
            v_current = torch.zeros_like(k_current)
            importance = torch.ones(batch_size, num_tokens)
            return tokens, (k_current, v_current, expected[layer_idx]), importance

        return forward_stub

    for idx, block in enumerate(model.global_blocks):
        monkeypatch.setattr(block, "forward", make_forward_stub(idx))

    images = torch.rand(1, 1, 3, 28, 28)
    _, _, _, pending, _ = model(
        images,
        cache_states=[LayerCacheState(), LayerCacheState()],
        use_cache=True,
        past_frame_idx=3,
        per_layer_budget=16,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )

    assert torch.equal(pending[0].attention_kept_indices, expected[0])
    assert torch.equal(pending[1].attention_kept_indices, expected[1])


def test_position_getter_cache_is_device_scoped():
    getter = PositionGetter()
    getter(batch_size=1, height=2, width=2, device=torch.device("cpu"))

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    getter(batch_size=1, height=2, width=2, device=torch.device("cuda", 0))
    assert len(getter.position_cache) == 2
