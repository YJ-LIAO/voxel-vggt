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
