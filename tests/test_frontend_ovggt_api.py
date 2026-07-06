import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig
from ovggt.utils.pose_enc import REL_POSE_ENCODING


def test_ovggt_accepts_frontend_constructor_options():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        per_layer_budget=16,
        total_budget=384,
        mode="frontend_eval",
        frontend_pose_encoding_type=REL_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
        aggregator_kwargs={
            "depth": 2,
            "num_heads": 4,
            "num_register_tokens": 1,
            "patch_embed": "conv",
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        enable_track_head=False,
    )
    assert model.mode == "frontend_eval"
    assert model.frontend_cache_config.enabled
    assert model.per_layer_budget == 16
    assert model.total_budget == 384


def test_frame_image_to_sequence_accepts_single_frame_and_batch():
    single = torch.rand(3, 28, 28)
    batch = torch.rand(2, 3, 28, 28)
    assert OVGGT._frame_image_to_sequence(single).shape == (1, 1, 3, 28, 28)
    assert OVGGT._frame_image_to_sequence(batch).shape == (2, 1, 3, 28, 28)


def test_frontend_mode_respects_explicit_disabled_cache_config_and_routes_legacy():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        total_budget=384,
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=False, dedup_enabled=False),
        aggregator_kwargs={
            "depth": 2,
            "num_heads": 4,
            "num_register_tokens": 1,
            "patch_embed": "conv",
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        enable_track_head=False,
    )
    assert model.frontend_cache_config.enabled is False

    calls = []

    def frontend_stub(*args, **kwargs):
        calls.append("frontend")
        return "frontend"

    def legacy_stub(*args, **kwargs):
        calls.append("legacy")
        return "legacy"

    model._inference_frontend = frontend_stub
    model._inference_legacy = legacy_stub

    assert model.inference([]) == "legacy"
    assert calls == ["legacy"]


def test_legacy_mode_defaults_to_coverage_history_anchor_strategy():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        total_budget=384,
        aggregator_kwargs={
            "depth": 2,
            "num_heads": 4,
            "num_register_tokens": 1,
            "patch_embed": "conv",
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        enable_track_head=False,
    )
    assert model._resolve_history_anchor_strategy(None) == "coverage"
