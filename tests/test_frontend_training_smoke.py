import os
import sys
import unittest
import importlib.util
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_keyframe import KeyframeEvent, KeyframeEventType

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if SCIPY_AVAILABLE:
    from ovggt.losses.frontend_distill import FrontendDistillLoss
else:
    FrontendDistillLoss = None


def build_tiny_frontend_model():
    return OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        total_budget=64,
        camera_budget=32,
        aggregator_kwargs={
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        camera_head_kwargs={
            "trunk_depth": 1,
            "num_heads": 4,
        },
        depth_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        point_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        mode="frontend_train",
        enable_track_head=False,
    )


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
    return {
        "img": torch.zeros(1, 3, 8, 8, dtype=torch.float32),
        "camera_pose": make_pose(tx).unsqueeze(0),
        "camera_intrinsics": make_intrinsics().unsqueeze(0),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
    }


def make_dense_outputs():
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, 8),
        torch.linspace(0.0, 1.0, 8),
        indexing="ij",
    )
    pts = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).unsqueeze(0)
    depth = torch.ones(1, 8, 8, 1, dtype=torch.float32)
    conf = torch.ones(1, 8, 8, dtype=torch.float32)
    return pts, depth, conf


class FrontendTrainingSmokeTests(unittest.TestCase):
    def test_load_state_dict_upgrades_missing_relative_pose_branch(self):
        model = build_tiny_frontend_model()
        state_dict = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if "camera_head.rel_pose_branch." not in key
        }

        reloaded_model = build_tiny_frontend_model()
        reloaded_model.load_state_dict(state_dict, strict=True)

        self.assertTrue(
            torch.allclose(
                reloaded_model.camera_head.pose_branch.fc1.weight,
                reloaded_model.camera_head.rel_pose_branch.fc1.weight,
            )
        )

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required to import FrontendDistillLoss")
    def test_frontend_distill_loss_is_finite_on_matching_targets(self):
        criterion = FrontendDistillLoss()
        raw_batch = [make_view(0.0), make_view(0.1), make_view(0.2)]
        schedule = [
            KeyframeEvent(KeyframeEventType.PROMOTE_KEYFRAME, 0, 0, 0),
            KeyframeEvent(KeyframeEventType.NOOP, 1, 0, -1),
            KeyframeEvent(KeyframeEventType.PROMOTE_KEYFRAME, 2, 1, 1),
        ]

        image_hw = criterion._infer_image_hw(raw_batch)
        gt_abs_pose = criterion._build_gt_absolute_pose_encoding(raw_batch, image_hw)
        fixed_keyframe_mask = criterion._build_keyframe_mask(
            num_frames=len(raw_batch),
            device=gt_abs_pose.device,
            keyframe_schedule=schedule,
        )
        gt_rel_pose = criterion._build_gt_relative_pose_targets(gt_abs_pose, fixed_keyframe_mask, image_hw)
        pts, depth, conf = make_dense_outputs()

        teacher_preds = []
        student_preds = []
        for frame_idx in range(len(raw_batch)):
            dense_pred = {
                "pts3d_in_other_view": pts.clone(),
                "conf": conf.clone(),
                "depth": depth.clone(),
                "depth_conf": conf.clone(),
            }
            teacher_preds.append(dict(dense_pred))
            student_preds.append(
                {
                    **dense_pred,
                    "camera_pose": gt_abs_pose[:, frame_idx].clone(),
                    "camera_pose_rel": gt_rel_pose[:, frame_idx].clone(),
                }
            )

        total, details = criterion(
            raw_batch_gt=raw_batch,
            teacher_outputs=SimpleNamespace(ress=teacher_preds),
            student_outputs=SimpleNamespace(ress=student_preds),
            keyframe_schedule=schedule,
        )

        self.assertTrue(torch.isfinite(total))
        self.assertEqual(details["Lcamera_rel"], 0.0)
        self.assertIn("Lcamera_abs_consistency", details)
        self.assertGreaterEqual(float(total), 0.0)
        self.assertEqual(details["num_keyframes"], 2.0)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required to import FrontendDistillLoss")
    def test_frontend_distill_combined_depth_and_pmap_terms_match_separate_paths(self):
        criterion = FrontendDistillLoss()
        batch_gt = make_view(0.0)
        pts, depth, conf = make_dense_outputs()
        teacher_pred = {
            "pts3d_in_other_view": pts.clone(),
            "conf": conf.clone(),
            "depth": depth.clone(),
            "depth_conf": conf.clone(),
        }
        student_pred = {
            "pts3d_in_other_view": pts.clone(),
            "conf": conf.clone(),
            "depth": depth.clone(),
            "depth_conf": conf.clone(),
            "camera_pose": torch.zeros(1, 9, dtype=torch.float32),
            "camera_pose_rel": torch.zeros(1, 9, dtype=torch.float32),
        }

        separate_depth = criterion.compute_depth_term(batch_gt, teacher_pred, student_pred)
        separate_pmap = criterion.compute_pmap_term(batch_gt, teacher_pred, student_pred)
        combined_depth, combined_pmap = criterion.compute_depth_and_pmap_terms(
            batch_gt=batch_gt,
            teacher_pred=teacher_pred,
            student_pred=student_pred,
        )

        self.assertTrue(torch.allclose(combined_depth, separate_depth))
        self.assertTrue(torch.allclose(combined_pmap, separate_pmap))


if __name__ == "__main__":
    unittest.main()
