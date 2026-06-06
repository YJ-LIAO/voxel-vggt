import os
import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.losses.frontend_supervised import FrontendSupervisedLoss


def make_pose(tx: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = tx
    return pose


def make_intrinsics() -> torch.Tensor:
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = 8.0
    intrinsics[1, 1] = 8.0
    intrinsics[0, 2] = 4.0
    intrinsics[1, 2] = 4.0
    return intrinsics


def make_view(tx: float) -> dict:
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, 8),
        torch.linspace(0.0, 1.0, 8),
        indexing="ij",
    )
    depth = torch.ones(1, 8, 8, 1, dtype=torch.float32)
    pts = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).unsqueeze(0)
    conf = torch.ones(1, 8, 8, dtype=torch.float32)
    return {
        "img": torch.zeros(1, 3, 8, 8, dtype=torch.float32),
        "camera_pose": make_pose(tx).unsqueeze(0),
        "camera_intrinsics": make_intrinsics().unsqueeze(0),
        "depthmap": depth.squeeze(0),
        "pts3d": pts.squeeze(0),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        "depth_conf": conf.clone(),
        "conf": conf.clone(),
    }


class FrontendSupervisedLossTests(unittest.TestCase):
    def test_supervised_loss_is_finite_on_matching_targets(self):
        criterion = FrontendSupervisedLoss()
        raw_batch = [make_view(0.0), make_view(0.1), make_view(0.2)]
        schedule = [
            {"event_type": "KeyframeEventType.PROMOTE_KEYFRAME", "frame_idx": 0, "keyframe_id": 0, "anchor_slot": 0},
            {"event_type": "KeyframeEventType.NOOP", "frame_idx": 1, "keyframe_id": 0, "anchor_slot": -1},
            {"event_type": "KeyframeEventType.PROMOTE_KEYFRAME", "frame_idx": 2, "keyframe_id": 1, "anchor_slot": 1},
        ]

        image_hw = criterion._infer_image_hw(raw_batch)
        gt_abs_pose = criterion._build_gt_absolute_pose_encoding(raw_batch, image_hw)
        fixed_keyframe_mask = criterion._build_keyframe_mask(
            num_frames=len(raw_batch),
            device=gt_abs_pose.device,
            keyframe_schedule=schedule,
        )
        gt_rel_pose = criterion._build_gt_relative_pose_targets(gt_abs_pose, fixed_keyframe_mask, image_hw)

        student_preds = []
        for frame_idx in range(len(raw_batch)):
            student_preds.append(
                {
                    "camera_pose": gt_abs_pose[:, frame_idx].clone(),
                    "camera_pose_rel": gt_rel_pose[:, frame_idx].clone(),
                    "depth": raw_batch[frame_idx]["depthmap"].clone(),
                    "depth_conf": raw_batch[frame_idx]["depth_conf"].clone(),
                    "pts3d_in_other_view": raw_batch[frame_idx]["pts3d"].clone(),
                    "conf": raw_batch[frame_idx]["conf"].clone(),
                }
            )

        total, details = criterion(
            raw_batch_gt=raw_batch,
            student_outputs=SimpleNamespace(ress=student_preds, keyframe_schedule=schedule),
            keyframe_schedule=schedule,
        )

        self.assertTrue(torch.isfinite(total))
        self.assertEqual(details["Lcamera_rel"], 0.0)
        self.assertEqual(details["num_keyframes"], 2.0)


if __name__ == "__main__":
    unittest.main()
