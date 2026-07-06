import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.heads.camera_head import CameraHead
from ovggt.utils.frontend_keyframe import KeyframeEvent, KeyframeEventType
from ovggt.utils.pose_enc import ABS_POSE_ENCODING, REL_POSE_ENCODING


def test_camera_head_returns_abs_and_rel_pose_predictions():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    tokens = [torch.rand(1, 1, 2, 16)]
    predictions = head(
        tokens,
        num_iterations=1,
        use_cache=False,
        pose_encoding_type=REL_POSE_ENCODING,
        return_pose_predictions=True,
        return_last_pose_only=True,
    )
    assert set(predictions) == {"abs_pose_enc", "rel_pose_enc"}
    assert predictions["abs_pose_enc"].shape == (1, 1, 9)
    assert predictions["rel_pose_enc"].shape == (1, 1, 9)


def test_camera_head_apply_keyframe_event_noop_is_safe():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    event = KeyframeEvent(
        event_type=KeyframeEventType.NOOP,
        frame_idx=1,
        keyframe_id=0,
        anchor_slot=-1,
    )
    past = [None]
    assert head.apply_keyframe_event(past, event, num_cam_iters=1) is past


def test_camera_head_loads_legacy_state_dict_rel_branch_from_pose_branch():
    source = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    old_style_state = {}
    for key, value in source.state_dict().items():
        if key.startswith("rel_pose_branch."):
            continue
        if key.startswith("pose_branch."):
            old_style_state[key] = torch.full_like(value, 0.25)
        else:
            old_style_state[key] = value.clone()

    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    head.load_state_dict(old_style_state)

    for name, rel_value in head.rel_pose_branch.state_dict().items():
        pose_value = head.pose_branch.state_dict()[name]
        assert torch.equal(rel_value, pose_value)


def test_camera_head_cache_returns_pose_prediction_dict_and_populated_kv():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    tokens = [torch.rand(1, 1, 2, 16)]
    past_key_values_camera = [None] * head.trunk_depth

    predictions, past_key_values_camera = head(
        tokens,
        num_iterations=1,
        past_key_values_camera=past_key_values_camera,
        use_cache=True,
        return_pose_predictions=True,
        return_last_pose_only=True,
    )

    assert set(predictions) == {"abs_pose_enc", "rel_pose_enc"}
    assert predictions["abs_pose_enc"].shape == (1, 1, 9)
    assert predictions["rel_pose_enc"].shape == (1, 1, 9)
    assert len(past_key_values_camera) == head.trunk_depth
    assert all(kv is not None for kv in past_key_values_camera)


def test_camera_head_apply_keyframe_event_forwards_fifo_anchor_count(monkeypatch):
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    event = KeyframeEvent(
        event_type=KeyframeEventType.FIFO_SWAP,
        frame_idx=2,
        keyframe_id=1,
        anchor_slot=1,
        num_anchor_frames=3,
    )
    past = [None]
    calls = []

    def sync_anchor_change(past_key_values_camera, anchor_token_count, num_cam_iters=4, is_fifo=False):
        calls.append(
            {
                "past": past_key_values_camera,
                "anchor_token_count": anchor_token_count,
                "num_cam_iters": num_cam_iters,
                "is_fifo": is_fifo,
            }
        )
        return past_key_values_camera

    monkeypatch.setattr(head, "sync_anchor_change", sync_anchor_change)

    assert head.apply_keyframe_event(past, event, num_cam_iters=2) is past
    assert calls == [
        {
            "past": past,
            "anchor_token_count": 6,
            "num_cam_iters": 2,
            "is_fifo": True,
        }
    ]
