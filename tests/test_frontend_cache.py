import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.attention import Attention
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    PendingLayerUpdate,
    TokenKind,
    TokenMetadata,
)
from ovggt.utils.frontend_keyframe import KeyframeEvent, KeyframeEventType


def make_transform(tx: float = 0.0) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)
    transform[0, 3] = tx
    return transform


def make_metadata(
    anchor_slots,
    token_kind=None,
    importance=None,
    depth_conf=None,
    local_xyz=None,
    frame_ids=None,
    slot_ids=None,
    keyframe_ids=None,
):
    anchor_slots = torch.tensor([anchor_slots], dtype=torch.long)
    num_tokens = anchor_slots.shape[1]
    token_kind = token_kind or [int(TokenKind.PATCH)] * num_tokens
    importance = importance or [0.0] * num_tokens
    depth_conf = depth_conf or [0.0] * num_tokens
    frame_ids = frame_ids or [0] * num_tokens
    slot_ids = slot_ids or frame_ids
    keyframe_ids = keyframe_ids or slot_ids
    if local_xyz is None:
        local_xyz = [[float(i), 0.0, 0.0] for i in range(num_tokens)]
    return TokenMetadata(
        token_kind=torch.tensor([token_kind], dtype=torch.long),
        frame_id=torch.tensor([frame_ids], dtype=torch.long),
        anchor_slot=anchor_slots,
        keyframe_id=torch.tensor([keyframe_ids], dtype=torch.long),
        slot_id=torch.tensor([slot_ids], dtype=torch.long),
        slot_local_xyz=torch.tensor([local_xyz], dtype=torch.float32),
        importance=torch.tensor([importance], dtype=torch.float32),
        depth_conf=torch.tensor([depth_conf], dtype=torch.float32),
    )


class FrontendCacheTests(unittest.TestCase):
    def test_single_batch_gather_matches_generic_gather(self):
        k = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
        v = k + 100.0
        metadata = make_metadata(
            anchor_slots=[0, 0, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 2, 2],
            slot_ids=[0, 0, 1, 1, 2, 2],
        )
        indices = torch.tensor([[0, 2, 5]], dtype=torch.long)

        generic_state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata.clone())
        generic_state.gather_(indices)

        fast_state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata.clone())
        fast_state._gather_single_batch_(indices[0])

        self.assertTrue(torch.equal(fast_state.k, generic_state.k))
        self.assertTrue(torch.equal(fast_state.v, generic_state.v))
        self.assertTrue(torch.equal(fast_state.metadata.frame_id, generic_state.metadata.frame_id))
        self.assertTrue(torch.equal(fast_state.metadata.slot_id, generic_state.metadata.slot_id))

    def test_gather_keeps_kv_and_metadata_aligned(self):
        k = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
        v = k + 100.0
        metadata = make_metadata(anchor_slots=[0, 0, -1, -1, -1, -1], frame_ids=[0, 0, 1, 1, 2, 2])
        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata)

        gather_indices = torch.tensor([[0, 2, 5]], dtype=torch.long)
        state.gather_(gather_indices)

        self.assertEqual(state.k.shape[2], 3)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(state.metadata.slot_id[0], torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([0.0, 4.0, 10.0])))

    def test_fifo_event_shifts_anchor_slots(self):
        state = LayerCacheState(metadata=make_metadata(anchor_slots=[0, 1, 2, -1], slot_ids=[0, 1, 2, 3]))
        event = KeyframeEvent(
            event_type=KeyframeEventType.FIFO_SWAP,
            frame_idx=2,
            keyframe_id=2,
            anchor_slot=2,
            demoted_slot=1,
            num_anchor_frames=3,
            slot_pose_updates={0: make_transform(0.0), 1: make_transform(1.0), 2: make_transform(2.0)},
        )
        state.apply_keyframe_event_(event)
        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, -1, 1, -1])))

    def test_keyframe_event_keeps_slot_local_xyz_stable(self):
        metadata = make_metadata(
            anchor_slots=[0, 1],
            frame_ids=[0, 1],
            slot_ids=[0, 1],
            local_xyz=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        state = LayerCacheState(metadata=metadata)
        before = state.metadata.slot_local_xyz.clone()

        event = KeyframeEvent(
            event_type=KeyframeEventType.PROMOTE_KEYFRAME,
            frame_idx=1,
            keyframe_id=1,
            anchor_slot=1,
            num_anchor_frames=2,
            slot_pose_updates={0: make_transform(-1.0), 1: make_transform(0.0)},
        )
        state.apply_keyframe_event_(event)

        self.assertTrue(torch.allclose(state.metadata.slot_local_xyz, before))
        self.assertIn(0, state.slot_to_active)
        self.assertTrue(torch.allclose(state.slot_to_active[0], make_transform(-1.0)))

    def test_voxel_dedup_keeps_protected_anchors_and_best_current_patch(self):
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        metadata = make_metadata(
            anchor_slots=[0, -1, -1, -1, -1],
            frame_ids=[0, 1, 1, 1, 1],
            slot_ids=[0, 1, 1, 1, 1],
            importance=[0.0, 0.2, 0.9, 0.8, 0.1],
            depth_conf=[0.0, 0.2, 0.9, 0.1, 0.1],
            token_kind=[
                int(TokenKind.PATCH),
                int(TokenKind.PATCH),
                int(TokenKind.PATCH),
                int(TokenKind.PATCH),
                int(TokenKind.CAMERA),
            ],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )
        state.reorder_by_anchor_slots_()
        state.apply_voxel_dedup_(FrontendCacheConfig(enabled=True, voxel_size=0.5), current_frame_id=1)

        self.assertEqual(state.num_tokens(), 3)
        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, -1, -1])))
        self.assertEqual(int(state.metadata.token_kind[0, -1].item()), int(TokenKind.CAMERA))
        self.assertTrue(torch.allclose(state.metadata.slot_local_xyz[0, 1], torch.tensor([1.0, 0.0, 0.0])))

    def test_commit_pending_update_keeps_metadata_in_sync(self):
        attn = Attention(dim=8, num_heads=2)
        existing_k = torch.randn(1, 2, 3, 4)
        existing_v = torch.randn(1, 2, 3, 4)
        existing_metadata = make_metadata(anchor_slots=[0, 0, -1], frame_ids=[0, 0, 0], slot_ids=[0, 0, 0])
        state = LayerCacheState(
            k=existing_k,
            v=existing_v,
            metadata=existing_metadata,
            protected_count=2,
            max_history_anchors=2,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        pending = PendingLayerUpdate(
            k_current=torch.randn(1, 2, 3, 4),
            v_current=torch.randn(1, 2, 3, 4),
            importance_current=torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float32),
            frame_id=1,
            cache_budget=4,
        )
        current_metadata = make_metadata(
            anchor_slots=[-1, -1, -1],
            frame_ids=[1, 1, 1],
            slot_ids=[1, 1, 1],
            token_kind=[int(TokenKind.CAMERA), int(TokenKind.PATCH), int(TokenKind.PATCH)],
            importance=[0.1, 0.8, 0.2],
            depth_conf=[0.0, 1.0, 0.5],
            local_xyz=[[float("nan"), float("nan"), float("nan")], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )

        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=1.0,
            attn_module=attn,
        )

        self.assertEqual(state.k.shape[2], state.metadata.frame_id.shape[1])
        self.assertLessEqual(state.num_tokens(), 4)


if __name__ == "__main__":
    unittest.main()
