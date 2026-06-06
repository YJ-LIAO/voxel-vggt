import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig


def build_tiny_model(frontend_enabled: bool, mode: str = None):
    shared_kwargs = {
        "img_size": 28,
        "patch_size": 14,
        "embed_dim": 32,
        "per_layer_budget": 3,
        "camera_budget": 32,
        "aggregator_kwargs": {
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        "camera_head_kwargs": {
            "trunk_depth": 1,
            "num_heads": 4,
        },
        "depth_head_kwargs": {
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        "point_head_kwargs": {
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        "enable_track_head": False,
        "mode": mode or ("frontend_eval" if frontend_enabled else "legacy"),
    }
    if frontend_enabled:
        shared_kwargs["frontend_cache_config"] = FrontendCacheConfig(
            enabled=True,
            export_keyframe_packets=True,
            dedup_enabled=True,
            voxel_size=0.5,
        )
        shared_kwargs["keyframe_switch_config"] = KeyframeSwitchConfig(
            strategy="fixed_interval",
            interval=1,
            max_history_anchors=2,
        )
    return OVGGT(**shared_kwargs).eval()


class FrontendInferenceSmokeTests(unittest.TestCase):
    def _make_frames(self, num_frames=3):
        return [{"img": torch.rand(1, 3, 28, 28)} for _ in range(num_frames)]

    def test_legacy_inference_path_runs(self):
        model = build_tiny_model(frontend_enabled=False)
        output = model.inference(self._make_frames())
        self.assertEqual(len(output.ress), 3)
        self.assertIn("camera_pose", output.ress[0])
        self.assertEqual(output.ress[0]["depth"].shape[-1], 1)
        self.assertIsNone(output.keyframe_packets)

    def test_frontend_inference_path_runs(self):
        model = build_tiny_model(frontend_enabled=True)
        output = model.inference(self._make_frames())
        self.assertEqual(len(output.ress), 3)
        self.assertIn("pts3d_in_other_view", output.ress[0])
        self.assertIn("camera_pose_rel", output.ress[0])
        self.assertIsNotNone(output.keyframe_packets)
        self.assertIsNotNone(output.keyframe_schedule)
        self.assertGreaterEqual(len(output.keyframe_packets), 1)

    def test_frontend_train_forward_runs(self):
        model = build_tiny_model(frontend_enabled=True, mode="frontend_train")
        output = model.forward(self._make_frames())
        self.assertEqual(len(output.ress), 3)
        self.assertIn("camera_pose_rel", output.ress[0])
        self.assertIsNotNone(output.keyframe_schedule)
        self.assertIsNone(output.keyframe_packets)

    def test_frontend_train_forward_rejects_inconsistent_batch_size(self):
        model = build_tiny_model(frontend_enabled=True, mode="frontend_train")
        frames = [
            {"img": torch.rand(2, 3, 28, 28)},
            {"img": torch.rand(1, 3, 28, 28)},
        ]
        with self.assertRaises(ValueError):
            model.forward(frames)


if __name__ == "__main__":
    unittest.main()
