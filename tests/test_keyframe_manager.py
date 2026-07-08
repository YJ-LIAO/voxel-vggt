import os
import sys
import unittest

import torch
import ovggt.utils.frontend_keyframe as frontend_keyframe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.utils.frontend_keyframe import (
    FrontendKeyframeManager,
    KeyframeEventType,
    KeyframeSwitchConfig,
)


def make_pose(tx=0.0):
    return torch.tensor([tx, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32)


class FrontendKeyframeManagerTests(unittest.TestCase):
    def test_frame_zero_initializes_global_anchor(self):
        manager = FrontendKeyframeManager(KeyframeSwitchConfig())
        event = manager.update(0, torch.ones(4, 4), make_pose(0.0), (4, 4))
        self.assertEqual(event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event.anchor_slot, 0)
        self.assertEqual(manager.get_num_anchor_frames(), 1)
        self.assertIn(0, event.slot_pose_updates)

    def test_coverage_noop_for_identical_pose(self):
        manager = FrontendKeyframeManager(KeyframeSwitchConfig(strategy="coverage", coverage_threshold=0.5))
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event = manager.update(1, torch.ones(8, 8), make_pose(0.0), (8, 8))
        self.assertEqual(event.event_type, KeyframeEventType.NOOP)
        self.assertEqual(event.anchor_slot, -1)

    def test_translation_threshold_triggers_keyframe(self):
        config = KeyframeSwitchConfig(
            strategy="coverage",
            coverage_threshold=0.0,
            translation_threshold=0.1,
        )
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event = manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        self.assertEqual(event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event.anchor_slot, 1)

    def test_fixed_interval_fifo_rotates_history_slots(self):
        config = KeyframeSwitchConfig(strategy="fixed_interval", interval=1, max_history_anchors=1)
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event1 = manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        event2 = manager.update(2, torch.ones(8, 8), make_pose(2.0), (8, 8))
        self.assertEqual(event1.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event2.event_type, KeyframeEventType.FIFO_SWAP)
        self.assertEqual(event2.demoted_slot, 1)
        self.assertEqual(manager.get_num_anchor_frames(), 2)
        self.assertIn(event2.keyframe_id, event2.slot_pose_updates)

    def test_zero_history_keeps_single_anchor_frame(self):
        config = KeyframeSwitchConfig(strategy="fixed_interval", interval=1, max_history_anchors=0)
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event = manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        self.assertEqual(event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event.anchor_slot, -1)
        self.assertEqual(manager.get_num_anchor_frames(), 1)
        self.assertEqual(len(manager.history_slots), 0)

    def test_coverage_strategy_respects_interval_gate(self):
        original_compute_coverage = frontend_keyframe.compute_coverage
        frontend_keyframe.compute_coverage = lambda *args, **kwargs: 0.0
        try:
            config = KeyframeSwitchConfig(
                strategy="coverage",
                coverage_monitor_only=False,
                coverage_threshold=0.5,
                interval=4,
            )
            manager = FrontendKeyframeManager(config)
            manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))

            early_event = manager.update(1, torch.ones(8, 8), make_pose(0.1), (8, 8))
            self.assertEqual(early_event.event_type, KeyframeEventType.NOOP)

            interval_event = manager.update(4, torch.ones(8, 8), make_pose(0.4), (8, 8))
            self.assertEqual(interval_event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
            self.assertEqual(interval_event.anchor_slot, 1)
        finally:
            frontend_keyframe.compute_coverage = original_compute_coverage

    def test_prune_retired_keyframes_keeps_only_live_or_active_transforms(self):
        config = KeyframeSwitchConfig(strategy="fixed_interval", interval=1, max_history_anchors=1)
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        manager.update(2, torch.ones(8, 8), make_pose(2.0), (8, 8))
        manager.update(3, torch.ones(8, 8), make_pose(3.0), (8, 8))
        self.assertIn(1, manager.retired_keyframes)
        self.assertIn(2, manager.retired_keyframes)

        manager.prune_retired_keyframes({1})

        self.assertIn(1, manager.retired_keyframes)
        self.assertNotIn(2, manager.retired_keyframes)


if __name__ == "__main__":
    unittest.main()
