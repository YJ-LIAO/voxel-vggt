import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.attention import Attention, _normalize_scores
from ovggt.layers.block import Block
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    PendingLayerUpdate,
    TokenKind,
    TokenMetadata,
    _normalize_with_mask_batch,
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


def test_masked_normalization_ignores_non_finite_values():
    values = torch.tensor(
        [
            [1.0, float("inf"), 3.0, float("nan")],
            [float("inf"), float("nan"), 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    normalized = _normalize_with_mask_batch(values, mask)

    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.equal(normalized[1], torch.zeros(4))


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

    def test_gather_per_batch_uses_metadata_from_each_batch(self):
        k = torch.arange(2 * 2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4, 2)
        v = k + 100.0
        metadata = TokenMetadata(
            token_kind=torch.full((2, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            anchor_slot=torch.full((2, 4), -1, dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_local_xyz=torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            importance=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32),
            depth_conf=torch.tensor([[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], dtype=torch.float32),
        )
        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata)

        state.gather_per_batch_([
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([1, 3], dtype=torch.long),
        ])

        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([10, 12])))
        self.assertTrue(torch.equal(state.metadata.frame_id[1], torch.tensor([21, 23])))
        self.assertTrue(torch.equal(state.metadata.slot_id[1], torch.tensor([21, 23])))
        self.assertTrue(torch.equal(state.k[1, 0, :, 0], k[1, 0, [1, 3], 0]))

    def test_gather_per_batch_rejects_unequal_keep_lengths_without_fake_padding(self):
        k = torch.arange(2 * 1 * 4 * 1, dtype=torch.float32).reshape(2, 1, 4, 1)
        v = k + 100.0
        metadata = TokenMetadata(
            token_kind=torch.full((2, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            anchor_slot=torch.full((2, 4), -1, dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_local_xyz=torch.zeros(2, 4, 3),
            importance=torch.rand(2, 4),
            depth_conf=torch.rand(2, 4),
        )
        state = LayerCacheState(k=k, v=v, metadata=metadata)

        with self.assertRaisesRegex(ValueError, "same number of tokens"):
            state.gather_per_batch_([
                torch.tensor([0, 2], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
            ])

    def test_protected_count_rejects_mismatched_batch_anchor_counts(self):
        metadata = TokenMetadata(
            token_kind=torch.full((2, 3), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.zeros((2, 3), dtype=torch.long),
            anchor_slot=torch.tensor([[0, -1, -1], [0, 1, -1]], dtype=torch.long),
            keyframe_id=torch.zeros((2, 3), dtype=torch.long),
            slot_id=torch.zeros((2, 3), dtype=torch.long),
            slot_local_xyz=torch.zeros(2, 3, 3),
            importance=torch.zeros(2, 3),
            depth_conf=torch.zeros(2, 3),
        )
        state = LayerCacheState(metadata=metadata)

        with self.assertRaisesRegex(ValueError, "same protected token count"):
            state._compute_protected_count_raw()

    def test_override_indices_ignore_out_of_range_values(self):
        metadata = make_metadata(anchor_slots=[-1, -1, -1])
        state = LayerCacheState(
            k=torch.zeros(1, 1, 3, 1),
            v=torch.zeros(1, 1, 3, 1),
            metadata=metadata,
        )

        rows = state._override_indices_per_batch(torch.tensor([-4, 1, 99], dtype=torch.long))

        self.assertEqual(len(rows), 1)
        self.assertTrue(torch.equal(rows[0], torch.tensor([1], dtype=torch.long)))

    def test_dedup_replay_override_preserves_per_batch_keep_indices(self):
        k = torch.arange(2 * 1 * 4 * 1, dtype=torch.float32).reshape(2, 1, 4, 1)
        v = k + 100.0
        metadata = TokenMetadata(
            token_kind=torch.full((2, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            anchor_slot=torch.full((2, 4), -1, dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_local_xyz=torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            importance=torch.rand(2, 4),
            depth_conf=torch.rand(2, 4),
        )

        class ReplayProbe:
            def on_dedup_candidate(self, **kwargs):
                return torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata)
        state.apply_voxel_dedup_(
            FrontendCacheConfig(enabled=True, dedup_enabled=True),
            current_frame_id=99,
            dedup_replay_probe=ReplayProbe(),
        )

        self.assertEqual(state.k.shape[2], 2)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([10, 12])))
        self.assertTrue(torch.equal(state.metadata.frame_id[1], torch.tensor([21, 23])))
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], k[0, 0, [0, 2], 0]))
        self.assertTrue(torch.equal(state.k[1, 0, :, 0], k[1, 0, [1, 3], 0]))

    def test_eviction_probe_override_preserves_per_batch_keep_indices(self):
        attn = Attention(dim=1, num_heads=1)
        k = torch.arange(2 * 1 * 3 * 1, dtype=torch.float32).reshape(2, 1, 3, 1)
        v = k + 100.0
        metadata = TokenMetadata(
            token_kind=torch.full((2, 3), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.zeros((2, 3), dtype=torch.long),
            anchor_slot=torch.full((2, 3), -1, dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
            slot_local_xyz=torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3),
            importance=torch.rand(2, 3),
            depth_conf=torch.rand(2, 3),
        )
        current_metadata = TokenMetadata(
            token_kind=torch.full((2, 1), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.ones((2, 1), dtype=torch.long),
            anchor_slot=torch.full((2, 1), -1, dtype=torch.long),
            keyframe_id=torch.tensor([[13], [23]], dtype=torch.long),
            slot_id=torch.tensor([[13], [23]], dtype=torch.long),
            slot_local_xyz=torch.ones(2, 1, 3, dtype=torch.float32),
            importance=torch.ones(2, 1, dtype=torch.float32),
            depth_conf=torch.ones(2, 1, dtype=torch.float32),
        )

        class EvictionProbe:
            def on_eviction_candidate(self, **kwargs):
                return torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata)
        state.commit_pending_update_(
            pending_update=PendingLayerUpdate(
                k_current=torch.full((2, 1, 1, 1), 9.0),
                v_current=torch.full((2, 1, 1, 1), 109.0),
                importance_current=torch.ones(2, 1),
                frame_id=1,
                cache_budget=2,
            ),
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=1.0,
            attn_module=attn,
            eviction_probe=EvictionProbe(),
        )

        self.assertEqual(state.k.shape[2], 2)
        self.assertTrue(torch.equal(state.metadata.keyframe_id[0], torch.tensor([10, 12])))
        self.assertTrue(torch.equal(state.metadata.keyframe_id[1], torch.tensor([21, 23])))
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([0.0, 2.0])))
        self.assertTrue(torch.equal(state.k[1, 0, :, 0], torch.tensor([4.0, 9.0])))

    def test_empty_demoted_slot_indices_return_empty_tensor(self):
        metadata = TokenMetadata(
            token_kind=torch.full((1, 2), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.tensor([[1, 2]], dtype=torch.long),
            anchor_slot=torch.tensor([[0, 1]], dtype=torch.long),
            keyframe_id=torch.tensor([[1, 2]], dtype=torch.long),
            slot_id=torch.tensor([[1, 2]], dtype=torch.long),
            slot_local_xyz=torch.zeros(1, 2, 3, dtype=torch.float64),
            importance=torch.zeros(1, 2, dtype=torch.float64),
            depth_conf=torch.ones(1, 2, dtype=torch.float64),
        )
        state = LayerCacheState(
            metadata=metadata,
            slot_to_active={1: make_transform(0.0).to(dtype=torch.float64), 2: make_transform(0.0).to(dtype=torch.float64)},
        )

        empty_indices = state.get_demoted_slot_indices(demoted_slot=99)

        self.assertEqual(empty_indices.shape, (0,))
        self.assertEqual(empty_indices.dtype, torch.long)

    def test_fifo_event_shifts_anchor_slots(self):
        state = LayerCacheState(metadata=make_metadata(anchor_slots=[0, 1, 2, -1], slot_ids=[0, 1, 2, 3]))
        state._cached_protected_count = state._compute_protected_count_raw()
        state.protected_count = state._cached_protected_count
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
        self.assertEqual(state.protected_count, 2)
        self.assertEqual(state._cached_protected_count, 2)

    def test_fifo_probe_override_preserves_per_batch_keep_indices(self):
        metadata = TokenMetadata(
            token_kind=torch.full((2, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.zeros((2, 4), dtype=torch.long),
            anchor_slot=torch.ones((2, 4), dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_local_xyz=torch.zeros(2, 4, 3),
            importance=torch.rand(2, 4),
            depth_conf=torch.rand(2, 4),
        )

        class FifoProbe:
            def on_fifo_topk_candidate(self, **kwargs):
                return torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

        state = LayerCacheState(
            k=torch.zeros(2, 1, 4, 1),
            v=torch.zeros(2, 1, 4, 1),
            metadata=metadata,
        )
        state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=2,
            fifo_probe=FifoProbe(),
        )

        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(state.metadata.anchor_slot[1], torch.tensor([1, 0, 1, 0])))

    def test_fifo_probe_override_respects_keep_count(self):
        metadata = TokenMetadata(
            token_kind=torch.full((1, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.zeros((1, 4), dtype=torch.long),
            anchor_slot=torch.ones((1, 4), dtype=torch.long),
            keyframe_id=torch.arange(4, dtype=torch.long).unsqueeze(0),
            slot_id=torch.arange(4, dtype=torch.long).unsqueeze(0),
            slot_local_xyz=torch.zeros(1, 4, 3),
            importance=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
            depth_conf=torch.ones(1, 4),
        )

        class FifoProbe:
            def on_fifo_topk_candidate(self, **kwargs):
                return torch.tensor([0, 1, 2, 3], dtype=torch.long)

        state = LayerCacheState(
            k=torch.zeros(1, 1, 4, 1),
            v=torch.zeros(1, 1, 4, 1),
            metadata=metadata,
        )
        state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=2,
            fifo_probe=FifoProbe(),
        )

        self.assertEqual(int((state.metadata.anchor_slot[0] == 0).sum().item()), 2)

    def test_fifo_max_protected_cap_is_applied_per_batch(self):
        metadata = TokenMetadata(
            token_kind=torch.full((2, 4), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.zeros((2, 4), dtype=torch.long),
            anchor_slot=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            keyframe_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_id=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
            slot_local_xyz=torch.zeros(2, 4, 3),
            importance=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
            depth_conf=torch.rand(2, 4),
        )
        state = LayerCacheState(
            k=torch.zeros(2, 1, 4, 1),
            v=torch.zeros(2, 1, 4, 1),
            metadata=metadata,
        )

        state.protect_topk_on_demotion_(
            demoted_slot=1,
            keep_count=2,
            max_protected=2,
        )

        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(state.metadata.anchor_slot[1], torch.tensor([0, 0, 1, 1])))

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

        # Protected anchor (token 0) + current token 1 (score 0.2 > protected 0.0, NOT discarded)
        # + best current from voxel (1,0,0) group (token 2, score 0.9) + CAMERA = 4 tokens
        self.assertEqual(state.num_tokens(), 4)
        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, -1, -1, -1])))
        self.assertEqual(int(state.metadata.token_kind[0, -1].item()), int(TokenKind.CAMERA))
        # Token 1 (voxel 0,0,0) is kept because its score (0.2) >= protected score (0.0)
        self.assertTrue(torch.allclose(state.metadata.slot_local_xyz[0, 1], torch.tensor([0.0, 0.0, 0.0])))
        # Best from voxel (1,0,0) group is token 2 (score 0.9)
        self.assertTrue(torch.allclose(state.metadata.slot_local_xyz[0, 2], torch.tensor([1.0, 0.0, 0.0])))

    def test_voxel_dedup_discards_current_patch_when_protected_score_is_higher(self):
        k = torch.randn(1, 2, 2, 4)
        v = torch.randn(1, 2, 2, 4)
        metadata = make_metadata(
            anchor_slots=[0, -1],
            frame_ids=[0, 1],
            slot_ids=[0, 1],
            keyframe_ids=[0, 1],
            importance=[1.0, 0.0],
            depth_conf=[1.0, 0.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                intra_frame_dedup_enabled=False,
            ),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 1)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0])))

    def test_voxel_dedup_keeps_current_frame_anchor_patch(self):
        k = torch.randn(1, 2, 2, 4)
        v = torch.randn(1, 2, 2, 4)
        metadata = make_metadata(
            anchor_slots=[0, 1],
            frame_ids=[0, 1],
            slot_ids=[0, 1],
            keyframe_ids=[0, 1],
            importance=[1.0, 0.0],
            depth_conf=[1.0, 0.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                intra_frame_dedup_enabled=False,
            ),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 2)
        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0, 1])))

    def test_voxel_dedup_keeps_all_tokens_when_current_patch_xyz_is_invalid(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        metadata = make_metadata(
            anchor_slots=[0, -1, -1],
            frame_ids=[0, 1, 1],
            slot_ids=[0, 1, 1],
            keyframe_ids=[0, 1, 1],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [float("nan"), float("nan"), float("nan")],
                [float("nan"), float("nan"), float("nan")],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(enabled=True, voxel_size=0.5),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 3)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 1, 1])))

    def test_soft_reservoir_skips_dedup_below_budget_trigger(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        metadata = make_metadata(
            anchor_slots=[-1, -1, -1],
            frame_ids=[1, 1, 1],
            slot_ids=[1, 1, 1],
            importance=[0.9, 0.5, 0.1],
            depth_conf=[0.9, 0.5, 0.1],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(k=k, v=v, metadata=metadata, slot_to_active={1: make_transform(0.0)})

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                dedup_policy="soft_reservoir",
                dedup_topk_per_voxel=1,
                dedup_budget_trigger_ratio=0.9,
            ),
            current_frame_id=1,
            cache_budget=10,
        )

        self.assertEqual(state.num_tokens(), 3)

    def test_soft_reservoir_keeps_topk_current_tokens_per_voxel(self):
        k = torch.randn(1, 2, 4, 4)
        v = torch.randn(1, 2, 4, 4)
        metadata = make_metadata(
            anchor_slots=[-1, -1, -1, -1],
            frame_ids=[1, 1, 1, 1],
            slot_ids=[1, 1, 1, 1],
            importance=[0.1, 0.9, 0.7, 0.2],
            depth_conf=[0.1, 0.9, 0.7, 0.2],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(k=k, v=v, metadata=metadata, slot_to_active={1: make_transform(0.0)})

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                dedup_policy="soft_reservoir",
                dedup_topk_per_voxel=2,
                dedup_budget_trigger_ratio=0.0,
            ),
            current_frame_id=1,
            cache_budget=4,
        )

        self.assertEqual(state.num_tokens(), 2)
        self.assertTrue(torch.allclose(state.metadata.importance[0], torch.tensor([0.9, 0.7])))

    def test_soft_reservoir_margin_keeps_current_when_protected_not_clearly_better(self):
        k = torch.randn(1, 2, 2, 4)
        v = torch.randn(1, 2, 2, 4)
        metadata = make_metadata(
            anchor_slots=[0, -1],
            frame_ids=[0, 1],
            slot_ids=[0, 1],
            keyframe_ids=[0, 1],
            importance=[0.9, 0.9],
            depth_conf=[0.9, 0.9],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                dedup_policy="soft_reservoir",
                dedup_topk_per_voxel=1,
                dedup_replacement_margin=0.1,
                dedup_budget_trigger_ratio=0.0,
                intra_frame_dedup_enabled=False,
            ),
            current_frame_id=1,
            cache_budget=2,
        )

        self.assertEqual(state.num_tokens(), 2)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 1])))

    def test_soft_reservoir_age_decay_prevents_stale_protected_from_dropping_current(self):
        k = torch.randn(1, 2, 2, 4)
        v = torch.randn(1, 2, 2, 4)
        metadata = make_metadata(
            anchor_slots=[0, -1],
            frame_ids=[0, 100],
            slot_ids=[0, 100],
            keyframe_ids=[0, 100],
            importance=[1.0, 0.0],
            depth_conf=[1.0, 0.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k,
            v=v,
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 100: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                dedup_policy="soft_reservoir",
                dedup_topk_per_voxel=1,
                dedup_age_decay=0.02,
                dedup_budget_trigger_ratio=0.0,
                intra_frame_dedup_enabled=False,
            ),
            current_frame_id=100,
            cache_budget=2,
        )

        self.assertEqual(state.num_tokens(), 2)
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 100])))

    def test_soft_reservoir_age_decay_uses_each_batch_frame_ids(self):
        metadata = TokenMetadata(
            token_kind=torch.full((2, 2), int(TokenKind.PATCH), dtype=torch.long),
            frame_id=torch.tensor([[99, 100], [0, 100]], dtype=torch.long),
            anchor_slot=torch.tensor([[0, -1], [0, -1]], dtype=torch.long),
            keyframe_id=torch.tensor([[99, 100], [0, 100]], dtype=torch.long),
            slot_id=torch.tensor([[99, 100], [0, 100]], dtype=torch.long),
            slot_local_xyz=torch.zeros(2, 2, 3, dtype=torch.float32),
            importance=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            depth_conf=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        )
        state = LayerCacheState(
            k=torch.randn(2, 2, 2, 4),
            v=torch.randn(2, 2, 2, 4),
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 99: make_transform(0.0), 100: make_transform(0.0)},
        )
        config = FrontendCacheConfig(
            enabled=True,
            voxel_size=0.5,
            dedup_policy="soft_reservoir",
            dedup_topk_per_voxel=1,
            dedup_age_decay=0.02,
            dedup_budget_trigger_ratio=0.0,
            intra_frame_dedup_enabled=False,
        )
        projected_xyz = state._project_slot_local_xyz_to_active(
            state.metadata.slot_local_xyz,
            state.metadata.slot_id,
        )

        keep_indices, _ = state._dedup_single_batch(
            b_idx=1,
            protected_patch_mask=torch.tensor([True, False]),
            current_patch_mask=torch.tensor([False, True]),
            scores=torch.tensor([1.0, 0.0]),
            config=config,
            total_tokens=2,
            projected_xyz=projected_xyz[1],
            current_frame_id=100,
        )

        self.assertTrue(torch.equal(keep_indices, torch.tensor([0, 1])))

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

    def test_commit_pending_update_forwards_window_token_count_to_eviction(self):
        attn = Attention(dim=4, num_heads=1)
        seen = {}

        def eviction_spy(k, v, cache_budget, num_anchor_tokens, **kwargs):
            seen["window_token_count"] = kwargs.get("window_token_count")
            return k, v, 0.0, None

        attn.eviction = eviction_spy
        state = LayerCacheState(
            k=torch.randn(1, 1, 3, 4),
            v=torch.randn(1, 1, 3, 4),
            metadata=make_metadata(anchor_slots=[-1, -1, -1], frame_ids=[0, 0, 0], slot_ids=[0, 0, 0]),
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )
        pending = PendingLayerUpdate(
            k_current=torch.randn(1, 1, 2, 4),
            v_current=torch.randn(1, 1, 2, 4),
            importance_current=torch.ones(1, 2),
            frame_id=1,
            cache_budget=3,
        )
        current_metadata = make_metadata(
            anchor_slots=[-1, -1],
            frame_ids=[1, 1],
            slot_ids=[1, 1],
            importance=[1.0, 1.0],
            depth_conf=[1.0, 1.0],
        )

        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=1.0,
            attn_module=attn,
            window_token_count=2,
        )

        self.assertEqual(seen["window_token_count"], 2)

    def test_commit_pending_update_reuses_attention_time_keep_indices(self):
        attn = Attention(dim=1, num_heads=1)
        seen = {"eviction_called": False}

        def eviction_spy(k, v, cache_budget, num_anchor_tokens, **kwargs):
            seen["eviction_called"] = True
            kept_indices = torch.tensor([[2, 3, 4]], dtype=torch.long, device=k.device)
            expanded = kept_indices.view(1, 1, 3, 1).expand(1, 1, 3, 1)
            return (
                torch.gather(k, 2, expanded),
                torch.gather(v, 2, expanded),
                0.0,
                kept_indices,
            )

        attn.eviction = eviction_spy
        state = LayerCacheState(
            k=torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1),
            v=torch.arange(100, 103, dtype=torch.float32).reshape(1, 1, 3, 1),
            metadata=make_metadata(
                anchor_slots=[-1, -1, -1],
                frame_ids=[0, 0, 0],
                slot_ids=[0, 0, 0],
            ),
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )
        pending = PendingLayerUpdate(
            k_current=torch.tensor([[[[10.0], [11.0]]]], dtype=torch.float32),
            v_current=torch.tensor([[[[110.0], [111.0]]]], dtype=torch.float32),
            importance_current=torch.ones(1, 2),
            frame_id=1,
            cache_budget=3,
            attention_kept_indices=torch.tensor([[0, 3, 4]], dtype=torch.long),
        )
        current_metadata = make_metadata(
            anchor_slots=[-1, -1],
            frame_ids=[1, 1],
            slot_ids=[1, 1],
            importance=[1.0, 1.0],
            depth_conf=[1.0, 1.0],
        )

        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=1.0,
            attn_module=attn,
        )

        self.assertFalse(seen["eviction_called"])
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([0.0, 10.0, 11.0])))
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 1, 1])))

    # ---------------------------------------------------------------------------
# Phase 1 Decision 2 instrumentation: apply_voxel_dedup_() must expose
# scores and policy_keep_indices to the probe callback BEFORE the gather.
# ---------------------------------------------------------------------------
import pytest


def _make_dedup_cache_state(num_tokens: int = 10) -> LayerCacheState:
    """Build a LayerCacheState where all PATCH tokens share voxel (0, 0, 0).

    frame_id / keyframe_id / slot_id = 1 so that with current_frame_id=1 the
    tokens fall into current_patch_mask and actually trigger dedup.
    """
    B = 1
    xyz = [(0.0, 0.0, 0.0)] * num_tokens
    return LayerCacheState(
        k=torch.randn(B, 2, num_tokens, 4),
        v=torch.randn(B, 2, num_tokens, 4),
        metadata=TokenMetadata(
            token_kind=torch.tensor([[int(TokenKind.PATCH)] * num_tokens], dtype=torch.long),
            frame_id=torch.tensor([[1] * num_tokens], dtype=torch.long),
            anchor_slot=torch.tensor([[-1] * num_tokens], dtype=torch.long),
            keyframe_id=torch.tensor([[1] * num_tokens], dtype=torch.long),
            slot_id=torch.tensor([[1] * num_tokens], dtype=torch.long),
            slot_local_xyz=torch.tensor([list(xyz)], dtype=torch.float32),
            importance=torch.rand(B, num_tokens),
            depth_conf=torch.rand(B, num_tokens),
        ),
        protected_count=0,
    )


class TestApplyVoxelDedupProbeCallbackOrdering:
    """Verify that apply_voxel_dedup_() computes keep indices BEFORE calling
    the probe callback, and that the callback receives actual scores and
    policy_keep_indices."""

    def test_probe_receives_scores_and_policy_keep_indices(self):
        """The probe callback must receive non-None scores and policy_keep_indices."""
        received = {}

        class InstrumentedProbe:
            def on_dedup_candidate(
                self,
                cache_state,
                layer_id,
                frame_id,
                batch_index=0,
                scores=None,
                policy_keep_indices=None,
            ):
                received["scores"] = scores
                received["policy_keep_indices"] = policy_keep_indices
                received["called"] = True

        cache = _make_dedup_cache_state(num_tokens=10)
        config = FrontendCacheConfig(
            enabled=True, dedup_enabled=True, voxel_size=0.25,
        )
        cache.apply_voxel_dedup_(
            config=config, current_frame_id=1,
            dedup_probe=InstrumentedProbe(),
            batch_index=0,
        )
        assert received.get("called"), "Probe callback was not invoked"
        assert received["scores"] is not None, "scores was not passed to callback"
        assert received["policy_keep_indices"] is not None, "policy_keep_indices was not passed to callback"
        assert received["scores"].dim() == 1, f"scores should be 1D, got {received['scores'].dim()}D"
        assert received["policy_keep_indices"].dim() == 1, (
            f"policy_keep_indices should be 1D, got {received['policy_keep_indices'].dim()}D"
        )

    def test_gather_uses_computed_keep_indices(self):
        """After the callback, the gather must apply the computed policy_keep_indices,
        not a re-derived set."""
        received = {}

        class CapturingProbe:
            def on_dedup_candidate(
                self,
                cache_state,
                layer_id,
                frame_id,
                batch_index=0,
                scores=None,
                policy_keep_indices=None,
            ):
                received["policy_keep_indices"] = (
                    policy_keep_indices.detach().cpu().clone() if policy_keep_indices is not None else None
                )
                received["num_tokens_before"] = cache_state.num_tokens()

        cache = _make_dedup_cache_state(num_tokens=10)
        config = FrontendCacheConfig(
            enabled=True, dedup_enabled=True, voxel_size=0.25,
        )
        cache.apply_voxel_dedup_(
            config=config, current_frame_id=1,
            dedup_probe=CapturingProbe(),
            batch_index=0,
        )
        assert received["policy_keep_indices"] is not None
        expected_kept = received["policy_keep_indices"].shape[0]
        assert received["num_tokens_before"] == 10
        assert expected_kept < received["num_tokens_before"], (
            "Fixture must trigger actual dedup; otherwise this test can pass without exercising gather"
        )
        actual_kept = cache.num_tokens()
        assert actual_kept == expected_kept, (
            f"Cache has {actual_kept} tokens after gather, but policy_keep_indices had {expected_kept}"
        )


def test_attention_global_plus_recent_anchor_overflow_keeps_global_anchor():
    attn = Attention(dim=8, num_heads=2)
    attn.anchor_overflow_policy = "global_plus_recent"
    k = torch.arange(1 * 2 * 6 * 4, dtype=torch.float32).reshape(1, 2, 6, 4)
    v = k + 1000.0

    final_k, final_v, _, kept = attn.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=5,
    )

    assert kept.tolist() == [[0, 3, 4]]
    assert torch.equal(final_k, k[:, :, [0, 3, 4], :])
    assert torch.equal(final_v, v[:, :, [0, 3, 4], :])


def test_attention_global_plus_recent_overflow_when_all_tokens_are_anchors():
    attn = Attention(dim=8, num_heads=2)
    attn.anchor_overflow_policy = "global_plus_recent"
    k = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)
    v = k + 1000.0

    final_k, final_v, _, kept = attn.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=5,
    )

    assert kept.tolist() == [[0, 3, 4]]
    assert torch.equal(final_k, k[:, :, [0, 3, 4], :])
    assert torch.equal(final_v, v[:, :, [0, 3, 4], :])


def test_attention_accepts_legacy_window_token_count_kwarg():
    attn = Attention(dim=8, num_heads=2)
    x = torch.randn(1, 2, 8)
    out, new_kv, _ = attn(
        x,
        use_cache=True,
        cache_budget=4,
        window_token_count=1,
    )
    assert out.shape == x.shape
    assert new_kv[0].shape[2] <= 2


def test_attention_eviction_accepts_legacy_window_token_count_kwarg():
    attn = Attention(dim=8, num_heads=2)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    final_k, final_v, _, kept = attn.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=1,
        window_token_count=1,
    )
    assert final_k.shape[2] <= 3
    assert final_v.shape[2] <= 3


def test_attention_normalize_scores_ignores_non_finite_values():
    scores = torch.tensor([[1.0, float("inf"), float("nan"), 3.0]], dtype=torch.float32)

    normalized = _normalize_scores(scores)

    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_attention_eviction_reserves_newest_window_tokens():
    torch.manual_seed(2)
    attn = Attention(dim=8, num_heads=2)
    k = torch.randn(1, 2, 6, 4)
    v = k + 1000.0

    final_k, final_v, _, kept = attn.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=1,
        window_token_count=2,
    )

    assert kept.tolist() == [[0, 4, 5]]
    assert torch.equal(final_k, k[:, :, [0, 4, 5], :])
    assert torch.equal(final_v, v[:, :, [0, 4, 5], :])


def test_deferred_attention_can_evict_for_attention_without_pruning_returned_current_kv():
    torch.manual_seed(3)
    attn = Attention(dim=4, num_heads=1, qkv_bias=False, proj_bias=False, fused_attn=False)
    x = torch.randn(1, 2, 4)
    past_k = torch.randn(1, 1, 3, 4)
    past_v = torch.randn(1, 1, 3, 4)
    calls = []

    def fake_eviction(k, v, cache_budget, num_anchor_tokens, **kwargs):
        calls.append(
            {
                "input_tokens": k.shape[2],
                "cache_budget": cache_budget,
                "num_anchor_tokens": num_anchor_tokens,
                "num_new_tokens": kwargs.get("num_new_tokens"),
            }
        )
        kept_indices = torch.zeros((k.shape[0], 1), dtype=torch.long, device=k.device)
        return k[:, :, :1, :], torch.zeros_like(v[:, :, :1, :]), 0.0, kept_indices

    attn.eviction = fake_eviction

    output, kv_info, _ = attn(
        x,
        past_key_values=(past_k, past_v),
        use_cache=True,
        cache_budget=1,
        defer_eviction=True,
        evict_for_attention=True,
        anchor_token_count=0,
    )

    k_full, v_full, k_current, v_current, past_kv, attention_kept_indices = kv_info
    assert calls == [
        {
            "input_tokens": 5,
            "cache_budget": 1,
            "num_anchor_tokens": 0,
            "num_new_tokens": 2,
        }
    ]
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    assert k_full.shape[2] == 5
    assert v_full.shape[2] == 5
    assert k_current.shape[2] == 2
    assert v_current.shape[2] == 2
    assert past_kv[0] is past_k
    assert past_kv[1] is past_v
    assert attention_kept_indices.shape == (1, 1)


def test_deferred_attention_returns_attention_time_keep_indices():
    torch.manual_seed(11)
    attn = Attention(dim=4, num_heads=1, qkv_bias=False, proj_bias=False, fused_attn=False)
    x = torch.randn(1, 2, 4)
    past_k = torch.randn(1, 1, 3, 4)
    past_v = torch.randn(1, 1, 3, 4)
    expected_kept = torch.tensor([[0, 3, 4]], dtype=torch.long)

    def fake_eviction(k, v, cache_budget, num_anchor_tokens, **kwargs):
        expanded = expected_kept.to(k.device).view(1, 1, 3, 1).expand(1, 1, 3, 4)
        return torch.gather(k, 2, expanded), torch.gather(v, 2, expanded), 0.0, expected_kept.to(k.device)

    attn.eviction = fake_eviction

    _, kv_info, _ = attn(
        x,
        past_key_values=(past_k, past_v),
        use_cache=True,
        cache_budget=3,
        defer_eviction=True,
        evict_for_attention=True,
        anchor_token_count=0,
    )

    assert torch.equal(kv_info[5].cpu(), expected_kept)


def test_frontend_block_evicts_for_attention_while_deferring_cache_commit():
    block = Block(dim=8, num_heads=2)
    x = torch.randn(1, 2, 8)
    past_k = torch.randn(1, 2, 3, 4)
    past_v = torch.randn(1, 2, 3, 4)
    seen = {}

    def forward_spy(x_in, **kwargs):
        seen["defer_eviction"] = kwargs.get("defer_eviction")
        seen["evict_for_attention"] = kwargs.get("evict_for_attention")
        k_current = torch.randn(1, 2, x_in.shape[1], 4)
        v_current = torch.randn(1, 2, x_in.shape[1], 4)
        k_full = torch.cat([past_k, k_current], dim=2)
        v_full = torch.cat([past_v, v_current], dim=2)
        return torch.zeros_like(x_in), (k_full, v_full, k_current, v_current, (past_k, past_v)), None

    block.attn.forward = forward_spy

    block(
        x,
        past_key_values=(past_k, past_v),
        use_cache=True,
        cache_budget=1,
        frontend_cache_mode=True,
        anchor_token_count=0,
    )

    assert seen["defer_eviction"] is True
    assert seen["evict_for_attention"] is True


def test_block_forwards_window_token_count_to_attention_eviction():
    block = Block(dim=8, num_heads=2)
    x = torch.randn(1, 2, 8)
    past_k = torch.randn(1, 2, 2, 4)
    past_v = torch.randn(1, 2, 2, 4)
    seen = {}

    def eviction_spy(k, v, cache_budget, num_anchor_tokens, **kwargs):
        seen["window_token_count"] = kwargs.get("window_token_count")
        return k, v, None, None

    block.attn.eviction = eviction_spy
    block(
        x,
        past_key_values=(past_k, past_v),
        use_cache=True,
        cache_budget=3,
        anchor_token_count=1,
        window_token_count=7,
    )

    assert seen["window_token_count"] == 7


def test_attention_baseline_fallback_score_remains_similarity_contract():
    attn = Attention(dim=8, num_heads=2)
    uniform_k = torch.ones(1, 2, 4, 4)
    diverse_k = torch.tensor(
        [[
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    uniform_v = uniform_k.clone()
    diverse_v = diverse_k.clone()

    _, _, uniform_score, _ = attn.eviction(
        uniform_k,
        uniform_v,
        cache_budget=2,
        num_anchor_tokens=0,
    )
    _, _, diverse_score, _ = attn.eviction(
        diverse_k,
        diverse_v,
        cache_budget=2,
        num_anchor_tokens=0,
    )

    assert diverse_score < uniform_score


if __name__ == "__main__":
    unittest.main()
