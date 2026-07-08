import os
import sys

import pytest
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


def test_frontend_inference_empty_frames_returns_empty_output():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )

    output = model.inference([], cache_results=False)

    assert output.ress == []
    assert output.views == []


def test_frontend_inference_passes_anchor_keep_ratio_to_frontend_path():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    calls = []

    def frontend_stub(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    model._inference_frontend = frontend_stub

    result = model.inference([], anchor_keep_ratio=0.125)

    assert calls
    assert result["anchor_keep_ratio"] == 0.125


def test_frontend_inference_passes_window_protect_frames_to_frontend_path():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    calls = []

    def frontend_stub(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    model._inference_frontend = frontend_stub

    result = model.inference([], window_protect_frames=3)

    assert calls
    assert result["window_protect_frames"] == 3


def test_frontend_history_anchor_slots_keep_only_confident_subset():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    patch_start_idx = model.aggregator.patch_start_idx
    total_tokens = patch_start_idx + 4
    conf_map = torch.tensor(
        [[
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.9],
            [0.8, 0.8, 0.2, 0.2],
            [0.8, 0.8, 0.2, 0.2],
        ]],
        dtype=torch.float32,
    )

    anchor_slots = model._build_frontend_anchor_slot_tensor(
        anchor_slot=2,
        total_tokens=total_tokens,
        anchor_keep_ratio=(patch_start_idx + 2) / total_tokens,
        conf_map=conf_map,
        image_size_hw=(28, 28),
        patch_start_idx=patch_start_idx,
        device=conf_map.device,
    )

    expected = torch.full((1, total_tokens), -1, dtype=torch.long)
    expected[:, :patch_start_idx] = 2
    expected[:, patch_start_idx + 1] = 2
    expected[:, patch_start_idx + 2] = 2
    assert torch.equal(anchor_slots, expected)


def test_frontend_history_anchor_subsample_always_keeps_all_special_tokens():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    patch_start_idx = model.aggregator.patch_start_idx
    total_tokens = patch_start_idx + 8
    conf_map = torch.ones(1, 4, 4, dtype=torch.float32)

    anchor_slots = model._build_frontend_anchor_slot_tensor(
        anchor_slot=2,
        total_tokens=total_tokens,
        anchor_keep_ratio=1.0 / total_tokens,
        conf_map=conf_map,
        image_size_hw=(28, 28),
        patch_start_idx=patch_start_idx,
        device=conf_map.device,
    )

    assert torch.equal(anchor_slots[0, :patch_start_idx], torch.full((patch_start_idx,), 2, dtype=torch.long))
    assert torch.equal(anchor_slots[0, patch_start_idx:], torch.full((total_tokens - patch_start_idx,), -1, dtype=torch.long))


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


def test_frontend_eval_forward_routes_to_frontend_inference(monkeypatch):
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    calls = []

    def inference_stub(*args, **kwargs):
        calls.append(kwargs)
        return "frontend"

    monkeypatch.setattr(model, "inference", inference_stub)

    result = model([{"img": torch.rand(1, 3, 28, 28)}], return_views=True)
    assert result == "frontend"
    assert len(calls) == 1
    assert calls[0]["move_to_cpu"] is False
    assert calls[0]["return_views"] is True


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


def test_frontend_inference_mutates_resume_state_and_advances_frame_offset():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    )
    state = {}

    model.inference(
        [{"img": torch.rand(3, 28, 28)}],
        past_key_values=state,
        cache_results=False,
        move_to_cpu=True,
    )
    assert state["frame_offset"] == 1
    assert len(state["keyframe_managers"]) == 1
    assert len(state["cache_states"]) == 1
    assert len(state["past_key_values_camera"]) == 1

    model.inference(
        [{"img": torch.rand(3, 28, 28)}],
        past_key_values=state,
        cache_results=False,
        move_to_cpu=True,
    )
    assert state["frame_offset"] == 2


def test_frontend_inference_returns_resume_state_when_not_supplied():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    )

    first = model.inference(
        [{"img": torch.rand(3, 28, 28)}],
        cache_results=False,
        move_to_cpu=True,
    )

    assert first.frontend_state["frame_offset"] == 1
    assert len(first.frontend_state["keyframe_managers"]) == 1

    second = model.inference(
        [{"img": torch.rand(3, 28, 28)}],
        past_key_values=first.frontend_state,
        cache_results=False,
        move_to_cpu=True,
    )
    assert second.frontend_state["frame_offset"] == 2


def test_frontend_keyframe_schedule_keeps_each_batch_event():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    )

    output = model.inference(
        [{"img": torch.rand(2, 3, 28, 28)}],
        cache_results=False,
        move_to_cpu=True,
    )

    assert len(output.keyframe_schedule) == 1
    assert isinstance(output.keyframe_schedule[0], list)
    assert len(output.keyframe_schedule[0]) == 2
    assert all(event.frame_idx == 0 for event in output.keyframe_schedule[0])


def test_frontend_coverage_strategy_is_not_monitor_only_by_default():
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )
    config = model._build_frontend_keyframe_config(
        history_anchor_strategy="coverage",
        anchor_interval=8,
        max_anchors=3,
        coverage_threshold=0.2,
    )
    assert config.strategy == "coverage"
    assert config.coverage_monitor_only is False


def test_frontend_inference_keeps_per_batch_camera_last_scores(monkeypatch):
    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    )
    seen = []
    call_count = {"value": 0}
    original_forward = model.camera_head.forward

    def fake_forward(
        aggregated_tokens_list,
        num_iterations=4,
        past_key_values_camera=None,
        use_cache=False,
        anchor_token_count=None,
        pose_encoding_type=None,
        return_pose_predictions=False,
        return_last_pose_only=False,
    ):
        seen.append(model.camera_head.last_scores.clone())
        call_count["value"] += 1
        marker = float(call_count["value"])
        model.camera_head.last_scores = torch.full_like(model.camera_head.last_scores, marker)
        if use_cache:
            predictions = {
                "abs_pose_enc": torch.zeros(1, 1, 9),
                "rel_pose_enc": torch.zeros(1, 1, 9),
            }
            if return_pose_predictions:
                return predictions, past_key_values_camera
            return [predictions["abs_pose_enc"]], past_key_values_camera
        return original_forward(
            aggregated_tokens_list,
            num_iterations=num_iterations,
            past_key_values_camera=past_key_values_camera,
            use_cache=use_cache,
            anchor_token_count=anchor_token_count,
            pose_encoding_type=pose_encoding_type,
            return_pose_predictions=return_pose_predictions,
            return_last_pose_only=return_last_pose_only,
        )

    monkeypatch.setattr(model.camera_head, "forward", fake_forward)

    frames = [
        {"img": torch.rand(2, 3, 28, 28)},
        {"img": torch.rand(2, 3, 28, 28)},
    ]
    model.inference(frames, cache_results=False, move_to_cpu=True)

    assert len(seen) == 4
    zeros = torch.zeros_like(seen[0])
    ones = torch.ones_like(seen[0])
    twos = torch.full_like(seen[0], 2.0)
    assert torch.equal(seen[0], zeros)
    assert torch.equal(seen[1], zeros)
    assert torch.equal(seen[2], ones)
    assert torch.equal(seen[3], twos)


def test_frontend_depth_and_point_heads_disable_outer_cuda_autocast(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA autocast test requires CUDA")

    model = _small_ovggt(
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    ).cuda()
    seen = []

    def fake_camera_forward(
        aggregated_tokens_list,
        num_iterations=4,
        past_key_values_camera=None,
        use_cache=False,
        anchor_token_count=None,
        pose_encoding_type=None,
        return_pose_predictions=False,
        return_last_pose_only=False,
    ):
        batch_size = aggregated_tokens_list[-1].shape[0]
        pose = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            device=aggregated_tokens_list[-1].device,
            dtype=torch.float32,
        ).reshape(1, 1, 9).expand(batch_size, 1, -1).clone()
        predictions = {"abs_pose_enc": pose, "rel_pose_enc": pose}
        return predictions, past_key_values_camera

    def fake_depth_forward(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8):
        seen.append(("depth", torch.is_autocast_enabled("cuda")))
        batch_size, num_frames, _, height, width = images.shape
        depth = torch.ones(
            batch_size, num_frames, height, width, 1,
            device=images.device,
            dtype=torch.float32,
        )
        depth_conf = torch.ones(
            batch_size, num_frames, height, width,
            device=images.device,
            dtype=torch.float32,
        )
        return depth, depth_conf

    def fake_point_forward(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8):
        seen.append(("point", torch.is_autocast_enabled("cuda")))
        batch_size, num_frames, _, height, width = images.shape
        pts3d = torch.ones(
            batch_size, num_frames, height, width, 3,
            device=images.device,
            dtype=torch.float32,
        )
        pts3d_conf = torch.ones(
            batch_size, num_frames, height, width,
            device=images.device,
            dtype=torch.float32,
        )
        return pts3d, pts3d_conf

    monkeypatch.setattr(model.camera_head, "forward", fake_camera_forward)
    monkeypatch.setattr(model.depth_head, "forward", fake_depth_forward)
    monkeypatch.setattr(model.point_head, "forward", fake_point_forward)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        model.inference(
            [{"img": torch.rand(3, 28, 28, device="cuda")}],
            cache_results=False,
            move_to_cpu=False,
        )

    assert seen == [("depth", False), ("point", False)]
