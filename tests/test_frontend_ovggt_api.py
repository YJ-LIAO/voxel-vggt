import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig
from ovggt.utils.pose_enc import (
    ABS_POSE_ENCODING,
    REL_POSE_ENCODING,
    relative_from_absolute_pose_encoding,
)


def _small_ovggt(**kwargs):
    options = {
        "img_size": 28,
        "patch_size": 14,
        "embed_dim": 32,
        "total_budget": 384,
        "aggregator_kwargs": {
            "depth": 2,
            "num_heads": 4,
            "num_register_tokens": 1,
            "patch_embed": "conv",
        },
        "camera_head_kwargs": {"trunk_depth": 1, "num_heads": 4},
        "enable_track_head": False,
    }
    options.update(kwargs)
    return OVGGT(**options)


def _pose_encoding(tx: float, ty: float = 0.0, tz: float = 0.0) -> torch.Tensor:
    return torch.tensor([[tx, ty, tz, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)


def test_ovggt_accepts_frontend_constructor_options():
    model = _small_ovggt(
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
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=False, dedup_enabled=False),
    )
    assert model.frontend_cache_config.enabled is False

    calls = []

    def frontend_stub(*args, **kwargs):
        calls.append(("frontend", kwargs))
        return "frontend"

    def legacy_stub(*args, **kwargs):
        calls.append(("legacy", kwargs))
        return "legacy"

    model._inference_frontend = frontend_stub
    model._inference_legacy = legacy_stub

    assert model.inference([]) == "legacy"
    assert len(calls) == 1
    call_name, call_kwargs = calls[0]
    assert call_name == "legacy"
    assert call_kwargs["history_anchor_strategy"] == "coverage"
    assert call_kwargs["anchor_interval"] == 250

    calls.clear()
    assert model.inference([], history_anchor_strategy="fixed_interval", anchor_interval=12) == "legacy"
    assert len(calls) == 1
    _, explicit_kwargs = calls[0]
    assert explicit_kwargs["history_anchor_strategy"] == "fixed_interval"
    assert explicit_kwargs["anchor_interval"] == 12


def test_legacy_mode_defaults_to_coverage_history_anchor_strategy():
    model = _small_ovggt()
    assert model._resolve_history_anchor_strategy(None) == "coverage"


def test_frontend_train_forward_uses_legacy_forward_when_cache_disabled():
    model = _small_ovggt(
        mode="frontend_train",
        frontend_cache_config=FrontendCacheConfig(enabled=False, dedup_enabled=False),
    )

    def frontend_stub(*args, **kwargs):
        raise AssertionError("frontend path should not be used when frontend cache is disabled")

    model.forward_frontend_train = frontend_stub
    output = model([{"img": torch.rand(1, 3, 28, 28)}])
    assert len(output.ress) == 1


def test_abs_frontend_camera_pose_rel_uses_active_keyframe_pose():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    anchor_pose = _pose_encoding(0.0)
    current_pose = _pose_encoding(1.0)

    class KeyframeManagerStub:
        def get_active_pose_encoding(self):
            return anchor_pose[0]

    camera_pose, camera_pose_rel = model._resolve_frontend_camera_pose(
        frame_idx=1,
        keyframe_managers=[KeyframeManagerStub()],
        abs_pose_enc=current_pose,
        rel_pose_enc=torch.zeros_like(current_pose),
        image_size_hw=(28, 28),
    )
    expected = relative_from_absolute_pose_encoding(
        anchor_pose.unsqueeze(1),
        current_pose.unsqueeze(1),
        image_size_hw=(28, 28),
    )[:, 0, :]
    self_relative = relative_from_absolute_pose_encoding(
        current_pose.unsqueeze(1),
        current_pose.unsqueeze(1),
        image_size_hw=(28, 28),
    )[:, 0, :]
    assert torch.allclose(camera_pose, current_pose)
    assert torch.allclose(camera_pose_rel, expected)
    assert not torch.allclose(camera_pose_rel, self_relative)


def test_frontend_frame_writer_receives_detached_output_when_move_to_cpu():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    )
    captured = []

    def writer(frame_idx, frame, result):
        captured.append(result)

    model.inference(
        [{"img": torch.rand(3, 28, 28)}],
        frame_writer=writer,
        cache_results=False,
        move_to_cpu=True,
    )
    assert captured
    tensor_values = [value for value in captured[0].values() if isinstance(value, torch.Tensor)]
    assert tensor_values
    assert all(value.device.type == "cpu" for value in tensor_values)
    assert all(not value.requires_grad for value in tensor_values)


def test_learned_fifo_keep_count_raises_without_count_head():
    try:
        _small_ovggt(
            mode="frontend_eval",
            frontend_cache_config=FrontendCacheConfig(
                enabled=True,
                dedup_enabled=False,
                learned_fifo_keep_count=True,
            ),
        )
    except ValueError as exc:
        assert "learned_fifo_keep_count=True requires a count head" in str(exc)
    else:
        raise AssertionError("Expected learned_fifo_keep_count without count head to raise")
