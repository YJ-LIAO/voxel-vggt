import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.utils.pose_enc import (
    compose_absolute_from_relative,
    relative_from_absolute_pose_encoding,
)


def make_pose(tx=0.0, ty=0.0, tz=0.0, fov=1.0):
    return torch.tensor([tx, ty, tz, 1.0, 0.0, 0.0, 0.0, fov, fov], dtype=torch.float32)


class PoseEncodingFrontendTests(unittest.TestCase):
    def test_relative_absolute_roundtrip(self):
        image_hw = (240, 320)
        anchor_pose = make_pose(0.0, 0.0, 0.0).view(1, 1, 9)
        current_pose = make_pose(1.0, -0.5, 0.25).view(1, 1, 9)

        rel_pose = relative_from_absolute_pose_encoding(anchor_pose, current_pose, image_hw)
        reconstructed = compose_absolute_from_relative(anchor_pose, rel_pose, image_hw)

        self.assertTrue(torch.allclose(reconstructed, current_pose, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
