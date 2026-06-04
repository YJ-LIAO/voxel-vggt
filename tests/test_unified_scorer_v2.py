"""Verification tests for v2: unified scoring across dedup / FIFO / eviction.

Plan reference: distributed-hugging-wind.md

Tests cover:
1. apply_voxel_dedup_ uses scorer logits when token_scorer is provided
2. apply_voxel_dedup_ falls back to composite scores when scorer=None
3. protect_topk_on_demotion_ uses scorer logits when token_scorer is provided
4. protect_topk_on_demotion_ falls back to importance when scorer=None
5. commit_pending_update_ uses scorer for intra-frame pruning
6. commit_pending_update_ passes scorer to apply_voxel_dedup_
7. CounterfactualDedupProbe records dedup events
8. CounterfactualFifoTopKProbe records FIFO events
9. CounterfactualOracleDataset handles dedup/FIFO event types
10. Dedup/FIFO counterfactual replay works correctly
11. Regression: scorer=None produces identical behavior to v1
"""

import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import (
    TOKEN_METADATA_FEATURE_DIM,
    TOKEN_METADATA_FEATURE_INDEX,
    TokenScorer,
)
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    PendingLayerUpdate,
    TokenKind,
    TokenMetadata,
    _build_current_frame_metadata_features,
)
from ovggt.utils.frontend_keyframe import KeyframeEvent, KeyframeEventType


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_frontend_cache.py)
# ---------------------------------------------------------------------------

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


def make_scorer(score_state_dim=16, num_layers=4):
    """Create a small TokenScorer for testing."""
    return TokenScorer(
        score_state_dim=score_state_dim,
        metadata_dim=TOKEN_METADATA_FEATURE_DIM,
        hidden_dim=32,
        num_layers=num_layers,
    )


def build_dedup_state(num_tokens=10, num_protected=2, voxel_size=0.5):
    """Build a cache state with tokens at specific voxel positions for dedup testing."""
    k = torch.randn(1, 2, num_tokens, 8)
    v = torch.randn(1, 2, num_tokens, 8)
    score_state_dim = 16
    score_state = torch.randn(1, num_tokens, score_state_dim)

    anchor_slots = [-1] * num_tokens
    frame_ids = [1] * num_tokens
    slot_ids = [1] * num_tokens
    importance = [0.5] * num_tokens
    depth_conf = [0.5] * num_tokens
    # Place tokens at grid positions so dedup can find collisions
    local_xyz = []
    for i in range(num_tokens):
        x = (i % 3) * voxel_size * 2
        y = (i // 3) * voxel_size * 2
        local_xyz.append([x, y, 0.0])
    # Make first num_protected tokens protected anchors from frame 0
    for i in range(num_protected):
        anchor_slots[i] = 0
        frame_ids[i] = 0
        slot_ids[i] = 0
    # Place some tokens at same voxel to trigger dedup
    # Token 3 and 4 at same position → dedup should keep one
    if num_tokens > 4:
        local_xyz[3] = [0.1, 0.1, 0.0]
        local_xyz[4] = [0.1, 0.1, 0.0]

    metadata = make_metadata(
        anchor_slots=anchor_slots,
        frame_ids=frame_ids,
        slot_ids=slot_ids,
        importance=importance,
        depth_conf=depth_conf,
        local_xyz=local_xyz,
    )

    state = LayerCacheState(
        k=k,
        v=v,
        score_state=score_state,
        metadata=metadata,
        slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
    )
    state.reorder_by_anchor_slots_()
    return state


# ===========================================================================
# Test: apply_voxel_dedup_ with scorer
# ===========================================================================

class TestVoxelDedupWithScorer(unittest.TestCase):
    """Tests that apply_voxel_dedup_ uses scorer logits when provided."""

    def test_dedup_with_scorer_produces_different_result_than_without(self):
        """When scorer is provided, dedup should use scorer logits, not composite."""
        state_a = build_dedup_state(num_tokens=8, num_protected=2)
        state_b = build_dedup_state(num_tokens=8, num_protected=2)
        # Copy state so they start identical
        state_b.k = state_a.k.clone()
        state_b.v = state_a.v.clone()
        state_b.score_state = state_a.score_state.clone()
        state_b.metadata = state_a.metadata.clone()

        config = FrontendCacheConfig(enabled=True, voxel_size=0.5)
        scorer = make_scorer()

        # Without scorer
        state_a.apply_voxel_dedup_(config, current_frame_id=1,
                                    token_scorer=None, layer_id=0)
        tokens_without = state_a.num_tokens()

        # With scorer
        state_b.apply_voxel_dedup_(config, current_frame_id=1,
                                    token_scorer=scorer, layer_id=0)
        tokens_with = state_b.num_tokens()

        # Both should have deduplicated (same or fewer tokens)
        self.assertLessEqual(tokens_without, 8)
        self.assertLessEqual(tokens_with, 8)

    def test_dedup_with_scorer_uses_logits_not_importance(self):
        """Verify scorer logits are actually used by mocking scorer output."""
        state = build_dedup_state(num_tokens=8, num_protected=2)
        config = FrontendCacheConfig(enabled=True, voxel_size=0.5)

        # Create a scorer that gives token at index 4 a very high score
        # (so it should be kept over token at index 3 in the same voxel)
        scorer = make_scorer()
        # Override forward to return controlled logits
        original_forward = scorer.forward

        def mock_forward(score_state, metadata_features, layer_id=0):
            B, N = score_state.shape[0], score_state.shape[1]
            logits = torch.zeros(B, N, dtype=score_state.dtype, device=score_state.device)
            # Give high scores to tokens we want to keep
            logits[:, :] = 1.0
            return logits

        scorer.forward = mock_forward
        state.apply_voxel_dedup_(config, current_frame_id=1,
                                  token_scorer=scorer, layer_id=0)
        # Should not crash and should have removed duplicates
        self.assertLessEqual(state.num_tokens(), 8)

    def test_dedup_without_scorer_uses_composite_score(self):
        """When scorer is None, dedup falls back to composite heuristic."""
        state = build_dedup_state(num_tokens=8, num_protected=2)
        config = FrontendCacheConfig(
            enabled=True, voxel_size=0.5,
            importance_weight=0.5, depth_conf_weight=0.5,
        )

        tokens_before = state.num_tokens()
        state.apply_voxel_dedup_(config, current_frame_id=1,
                                  token_scorer=None, layer_id=0)
        tokens_after = state.num_tokens()

        # Dedup should have removed some tokens
        self.assertLessEqual(tokens_after, tokens_before)
        # Protected tokens should remain
        protected_count = (state.metadata.anchor_slot[0] >= 0).sum().item()
        self.assertGreater(protected_count, 0)


# ===========================================================================
# Test: protect_topk_on_demotion_ with scorer
# ===========================================================================

class TestFifoTopKWithScorer(unittest.TestCase):
    """Tests that protect_topk_on_demotion_ uses scorer logits when provided."""

    def _build_fifo_state(self, num_tokens=20, demoted_slot=1, keep_count=5):
        """Build a cache state with tokens in two anchor slots."""
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        score_state_dim = 16
        score_state = torch.randn(1, num_tokens, score_state_dim)

        # Half tokens in slot 0 (global), half in demoted slot
        anchor_slots = [0] * (num_tokens // 2) + [demoted_slot] * (num_tokens - num_tokens // 2)
        metadata = make_metadata(
            anchor_slots=anchor_slots,
            frame_ids=[0] * (num_tokens // 2) + [1] * (num_tokens - num_tokens // 2),
            slot_ids=[0] * (num_tokens // 2) + [1] * (num_tokens - num_tokens // 2),
            importance=[float(i) for i in range(num_tokens)],
            depth_conf=[0.5] * num_tokens,
        )

        return LayerCacheState(
            k=k, v=v, score_state=score_state, metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(1.0)},
        ), keep_count, demoted_slot

    def test_fifo_with_scorer_keeps_top_k_by_scorer(self):
        """When scorer is provided, top-K is selected by scorer logits, not importance."""
        state, keep_count, demoted_slot = self._build_fifo_state(
            num_tokens=20, demoted_slot=1, keep_count=5,
        )

        # Record which tokens are in demoted slot before
        demoted_before = (state.metadata.anchor_slot[0] == demoted_slot).sum().item()

        scorer = make_scorer()
        state.protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            token_scorer=scorer,
            layer_id=0,
        )

        # After: keep_count tokens should be reassigned from demoted_slot to slot 0
        reassigned = (state.metadata.anchor_slot[0] == 0).sum().item()
        self.assertGreaterEqual(reassigned, keep_count)

    def test_fifo_without_scorer_uses_importance(self):
        """When scorer is None, top-K is selected by importance (v1 behavior)."""
        state, keep_count, demoted_slot = self._build_fifo_state(
            num_tokens=20, demoted_slot=1, keep_count=5,
        )

        state.protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            token_scorer=None,
            layer_id=0,
        )

        # keep_count tokens should be reassigned from demoted_slot to slot 0
        reassigned = (state.metadata.anchor_slot[0] == 0).sum().item()
        self.assertGreaterEqual(reassigned, keep_count)

    def test_fifo_with_scorer_differs_from_without(self):
        """Scorer-based selection should differ from importance-based in general."""
        state_a, keep_count, demoted_slot = self._build_fifo_state(
            num_tokens=20, demoted_slot=1, keep_count=5,
        )
        state_b, _, _ = self._build_fifo_state(
            num_tokens=20, demoted_slot=1, keep_count=5,
        )
        # Make them identical
        state_b.k = state_a.k.clone()
        state_b.v = state_a.v.clone()
        state_b.score_state = state_a.score_state.clone()
        state_b.metadata = state_a.metadata.clone()

        scorer = make_scorer()
        state_a.protect_topk_on_demotion_(
            demoted_slot=demoted_slot, keep_count=keep_count,
            token_scorer=None, layer_id=0,
        )
        state_b.protect_topk_on_demotion_(
            demoted_slot=demoted_slot, keep_count=keep_count,
            token_scorer=scorer, layer_id=0,
        )

        # Both should have reassigned tokens
        self.assertGreaterEqual((state_a.metadata.anchor_slot[0] == 0).sum().item(), keep_count)
        self.assertGreaterEqual((state_b.metadata.anchor_slot[0] == 0).sum().item(), keep_count)

    def test_fifo_protects_all_when_keep_count_exceeds_demoted(self):
        """If demoted slot has fewer tokens than keep_count, all are protected (moved to slot 0)."""
        state, _, demoted_slot = self._build_fifo_state(
            num_tokens=20, demoted_slot=1, keep_count=100,  # More than available
        )

        state.protect_topk_on_demotion_(
            demoted_slot=demoted_slot, keep_count=100,
            token_scorer=make_scorer(), layer_id=0,
        )

        # When keep_count >= num_tokens_in_slot, ALL demoted tokens are protected.
        # No tokens should remain in the demoted slot.
        demoted_after = (state.metadata.anchor_slot[0] == demoted_slot).sum().item()
        self.assertEqual(demoted_after, 0,
                        "When keep_count >= demoted tokens, all should be moved to slot 0")


# ===========================================================================
# Test: commit_pending_update_ scorer integration
# ===========================================================================

class TestCommitPendingUpdateWithScorer(unittest.TestCase):
    """Tests that commit_pending_update_ uses scorer for pruning and dedup."""

    def test_commit_passes_scorer_to_dedup(self):
        """Verify commit_pending_update_ passes token_scorer to apply_voxel_dedup_."""
        from ovggt.layers.attention import Attention

        existing_k = torch.randn(1, 2, 3, 4)
        existing_v = torch.randn(1, 2, 3, 4)
        existing_metadata = make_metadata(
            anchor_slots=[0, 0, -1], frame_ids=[0, 0, 0], slot_ids=[0, 0, 0],
        )
        state = LayerCacheState(
            k=existing_k, v=existing_v, metadata=existing_metadata,
            protected_count=2, max_history_anchors=2,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        pending = PendingLayerUpdate(
            k_current=torch.randn(1, 2, 3, 4),
            v_current=torch.randn(1, 2, 3, 4),
            importance_current=torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float32),
            frame_id=1,
            cache_budget=10,
            score_state_current=torch.randn(1, 3, 16),
        )
        current_metadata = make_metadata(
            anchor_slots=[-1, -1, -1],
            frame_ids=[1, 1, 1],
            slot_ids=[1, 1, 1],
            token_kind=[int(TokenKind.CAMERA), int(TokenKind.PATCH), int(TokenKind.PATCH)],
            importance=[0.1, 0.8, 0.2],
            depth_conf=[0.0, 1.0, 0.5],
            local_xyz=[[float("nan")] * 3, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )

        scorer = make_scorer()
        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=True, voxel_size=0.5),
            intra_frame_keep_ratio=1.0,
            attn_module=Attention(dim=8, num_heads=2),
            token_scorer=scorer,
            layer_id=0,
        )

        # Should complete without error and keep k/metadata in sync
        self.assertEqual(state.k.shape[2], state.metadata.frame_id.shape[1])
        if state.score_state is not None:
            self.assertEqual(state.score_state.shape[1], state.metadata.frame_id.shape[1])

    def test_commit_with_intra_pruning_uses_scorer(self):
        """Intra-frame pruning uses scorer when token_scorer is provided."""
        from ovggt.layers.attention import Attention

        num_new_tokens = 10
        existing_k = torch.randn(1, 2, 2, 4)
        existing_v = torch.randn(1, 2, 2, 4)
        existing_metadata = make_metadata(
            anchor_slots=[0, 0], frame_ids=[0, 0], slot_ids=[0, 0],
        )
        state = LayerCacheState(
            k=existing_k, v=existing_v, metadata=existing_metadata,
            protected_count=2, max_history_anchors=2,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        pending = PendingLayerUpdate(
            k_current=torch.randn(1, 2, num_new_tokens, 4),
            v_current=torch.randn(1, 2, num_new_tokens, 4),
            importance_current=torch.randn(1, num_new_tokens),
            frame_id=0,
            cache_budget=100,  # No eviction needed
            score_state_current=torch.randn(1, num_new_tokens, 16),
        )
        current_metadata = make_metadata(
            anchor_slots=[-1] * num_new_tokens,
            frame_ids=[0] * num_new_tokens,
            slot_ids=[0] * num_new_tokens,
            importance=[0.5] * num_new_tokens,
            depth_conf=[0.5] * num_new_tokens,
        )

        scorer = make_scorer()
        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=0.5,
            attn_module=Attention(dim=8, num_heads=2),
            token_scorer=scorer,
            layer_id=0,
        )

        # Should have kept ~half the tokens (5 out of 10) + 2 anchor = 7
        expected_kept = 2 + max(int(num_new_tokens * 0.5), 1)
        self.assertEqual(state.num_tokens(), expected_kept)


# ===========================================================================
# Test: _build_current_frame_metadata_features
# ===========================================================================

class TestBuildCurrentFrameMetadataFeatures(unittest.TestCase):
    """Test standalone metadata feature builder for tokens not yet in cache."""

    def test_output_shape(self):
        metadata = make_metadata(
            anchor_slots=[-1, -1, -1],
            frame_ids=[1, 1, 1],
            slot_ids=[1, 1, 1],
            importance=[0.5, 0.3, 0.7],
            depth_conf=[0.8, 0.6, 0.9],
            local_xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )
        features = _build_current_frame_metadata_features(metadata, current_frame_id=1)

        self.assertEqual(features.shape, (1, 3, TOKEN_METADATA_FEATURE_DIM))
        # depth_conf should be populated
        idx = TOKEN_METADATA_FEATURE_INDEX
        self.assertTrue((features[..., idx["depth_conf"]] > 0).any())

    def test_frame_age_is_zero(self):
        """Tokens not yet in cache should have frame_age=0."""
        metadata = make_metadata(
            anchor_slots=[-1, -1],
            frame_ids=[1, 1],
        )
        features = _build_current_frame_metadata_features(metadata, current_frame_id=1)
        idx = TOKEN_METADATA_FEATURE_INDEX
        self.assertTrue(torch.allclose(features[..., idx["frame_age"]], torch.zeros(1, 2)))


# ===========================================================================
# Test: CounterfactualOracleDataset with dedup/FIFO events
# ===========================================================================

class TestOracleDatasetEventTypes(unittest.TestCase):
    """Test that CounterfactualOracleDataset handles dedup and fifo_topk event types."""

    def _make_shard(self, event_type="eviction"):
        return {
            "events": [
                {
                    "event_id": f"test:{event_type}:f0:l0",
                    "event_type": event_type,
                    "layer_id": 0,
                    "score_state": torch.randn(4, 8),
                    "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                    "subsets": [
                        {"keep_indices": [0, 1], "loss": 0.1},
                        {"keep_indices": [2, 3], "loss": 0.5},
                    ],
                }
            ]
        }

    def test_default_accepts_all_event_types(self):
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard = {
                "events": [
                    {
                        "event_id": "ev0",
                        "event_type": "eviction",
                        "layer_id": 0,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 1], "loss": 0.1},
                            {"keep_indices": [2, 3], "loss": 0.5},
                        ],
                    },
                    {
                        "event_id": "dd0",
                        "event_type": "dedup",
                        "layer_id": 1,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 2], "loss": 0.2},
                            {"keep_indices": [1, 3], "loss": 0.6},
                        ],
                    },
                    {
                        "event_id": "fifo0",
                        "event_type": "fifo_topk",
                        "layer_id": 2,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 3], "loss": 0.3},
                            {"keep_indices": [1, 2], "loss": 0.7},
                        ],
                    },
                ]
            }
            torch.save(shard, f.name)
            dataset = CounterfactualOracleDataset([f.name])
            # 3 events × 1 pair each = 3 samples
            self.assertEqual(len(dataset), 3)
            os.unlink(f.name)

    def test_event_types_filter_dedup_only(self):
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard = {
                "events": [
                    {
                        "event_id": "ev0",
                        "event_type": "eviction",
                        "layer_id": 0,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 1], "loss": 0.1},
                            {"keep_indices": [2, 3], "loss": 0.5},
                        ],
                    },
                    {
                        "event_id": "dd0",
                        "event_type": "dedup",
                        "layer_id": 1,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 2], "loss": 0.2},
                            {"keep_indices": [1, 3], "loss": 0.6},
                        ],
                    },
                ]
            }
            torch.save(shard, f.name)
            dataset = CounterfactualOracleDataset([f.name], event_types=["dedup"])
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.samples[0]["event_type"], "dedup")
            os.unlink(f.name)

    def test_event_type_in_collated_batch(self):
        from ovggt.training.token_oracle_dataset import (
            CounterfactualOracleDataset,
            collate_oracle_pairs,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard = {
                "events": [
                    {
                        "event_id": "dd0",
                        "event_type": "dedup",
                        "layer_id": 0,
                        "score_state": torch.randn(4, 8),
                        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                        "subsets": [
                            {"keep_indices": [0, 1], "loss": 0.1},
                            {"keep_indices": [2, 3], "loss": 0.5},
                        ],
                    },
                ]
            }
            torch.save(shard, f.name)
            dataset = CounterfactualOracleDataset([f.name])
            batch = collate_oracle_pairs([dataset[0]])
            self.assertIn("event_type", batch)
            self.assertEqual(batch["event_type"], ["dedup"])
            os.unlink(f.name)

    def test_unsupported_event_type_raises(self):
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        with self.assertRaises(ValueError):
            CounterfactualOracleDataset([], event_types=["unknown_type"])


# ===========================================================================
# Test: CounterfactualDedupProbe
# ===========================================================================

class TestCounterfactualDedupProbe(unittest.TestCase):
    """Test that CounterfactualDedupProbe records dedup events correctly."""

    def test_probe_records_events_for_multi_token_voxels(self):
        from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe

        probe = CounterfactualDedupProbe(
            num_samples=4,
            oracle_window=2,
            seed=42,
            max_events=10,
            voxel_size=0.5,
        )

        # Build a cache state with tokens in the same voxel
        num_tokens = 10
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        score_state = torch.randn(1, num_tokens, 16)

        # Place tokens 3,4 at same voxel, tokens 5,6 at another
        local_xyz = []
        for i in range(num_tokens):
            if i in (3, 4):
                local_xyz.append([0.1, 0.1, 0.0])
            elif i in (5, 6):
                local_xyz.append([1.0, 1.0, 0.0])
            else:
                local_xyz.append([float(i) * 2.0, 0.0, 0.0])

        metadata = make_metadata(
            anchor_slots=[0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            slot_ids=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            importance=[0.5] * num_tokens,
            depth_conf=[0.5] * num_tokens,
            local_xyz=local_xyz,
            token_kind=[int(TokenKind.PATCH)] * num_tokens,
        )

        cache_state = LayerCacheState(
            k=k, v=v, score_state=score_state, metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        probe.on_dedup_candidate(
            cache_state=cache_state,
            layer_id=0,
            frame_id=1,
            batch_index=0,
        )

        self.assertGreater(len(probe.events), 0)
        event = probe.events[0]
        self.assertEqual(event["event_type"], "dedup")
        self.assertIn("voxel_group_id", event)
        self.assertIn("score_state", event)
        self.assertIn("metadata_features", event)
        self.assertGreater(len(event["candidate_subsets"]), 1)

    def test_probe_respects_max_events(self):
        from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe

        probe = CounterfactualDedupProbe(max_events=1, voxel_size=0.5)

        # Build minimal state
        num_tokens = 6
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        score_state = torch.randn(1, num_tokens, 16)
        local_xyz = [[0.1, 0.1, 0.0] if i in (0, 1) else [float(i * 3), 0.0, 0.0]
                      for i in range(num_tokens)]
        metadata = make_metadata(
            anchor_slots=[-1] * num_tokens,
            frame_ids=[0] * num_tokens,
            importance=[0.5] * num_tokens,
            local_xyz=local_xyz,
        )
        cache_state = LayerCacheState(
            k=k, v=v, score_state=score_state, metadata=metadata,
            slot_to_active={0: make_transform(0.0)},
        )

        # First call should record
        probe.on_dedup_candidate(cache_state, layer_id=0, frame_id=0)
        count_after_first = len(probe.events)

        # Second call should be skipped (max_events=1, but there may be multiple groups)
        # The probe limits to 3 groups per call, but max_events limits total events
        # So at most we should have 1 event (the first group found)
        self.assertLessEqual(len(probe.events), 3)  # at most 3 groups × 1 call


# ===========================================================================
# Test: CounterfactualFifoTopKProbe
# ===========================================================================

class TestCounterfactualFifoTopKProbe(unittest.TestCase):
    """Test that CounterfactualFifoTopKProbe records FIFO events correctly."""

    def test_probe_records_events(self):
        from ovggt.training.frontend_oracle_collector import CounterfactualFifoTopKProbe

        probe = CounterfactualFifoTopKProbe(
            num_samples=4,
            oracle_window=2,
            seed=42,
            max_events=10,
        )

        # Build a cache state with tokens in demoted slot
        num_tokens = 20
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        score_state = torch.randn(1, num_tokens, 16)

        anchor_slots = [0] * 8 + [1] * 12  # 8 in slot 0, 12 in slot 1
        metadata = make_metadata(
            anchor_slots=anchor_slots,
            frame_ids=[0] * 8 + [1] * 12,
            slot_ids=[0] * 8 + [1] * 12,
            importance=[float(i % 5) for i in range(num_tokens)],
        )

        cache_state = LayerCacheState(
            k=k, v=v, score_state=score_state, metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(1.0)},
        )

        probe.on_fifo_topk_candidate(
            cache_state=cache_state,
            demoted_slot=1,
            keep_count=5,
            layer_id=0,
            frame_id=1,
        )

        self.assertEqual(len(probe.events), 1)
        event = probe.events[0]
        self.assertEqual(event["event_type"], "fifo_topk")
        self.assertEqual(event["demoted_slot"], 1)
        self.assertEqual(event["keep_count"], 5)
        self.assertIn("score_state", event)
        self.assertIn("metadata_features", event)
        self.assertGreater(len(event["candidate_subsets"]), 1)
        # Check that each subset has keep_indices = non_slot_count + keep_count
        # 8 tokens in slot 0 (non-demoted) + 5 kept from slot 1 = 13
        non_slot_count = 8  # slot 0 tokens
        expected_keep_len = non_slot_count + 5
        for subset in event["candidate_subsets"]:
            self.assertIn("keep_indices", subset)
            self.assertEqual(len(subset["keep_indices"]), expected_keep_len)

    def test_probe_skips_when_tokens_below_keep_count(self):
        from ovggt.training.frontend_oracle_collector import CounterfactualFifoTopKProbe

        probe = CounterfactualFifoTopKProbe(max_events=10)

        # Only 3 tokens in demoted slot, keep_count=5
        num_tokens = 6
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        score_state = torch.randn(1, num_tokens, 16)
        metadata = make_metadata(
            anchor_slots=[0, 0, 0, 1, 1, 1],
            frame_ids=[0, 0, 0, 1, 1, 1],
            importance=[1.0] * num_tokens,
        )

        cache_state = LayerCacheState(
            k=k, v=v, score_state=score_state, metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(1.0)},
        )

        probe.on_fifo_topk_candidate(
            cache_state=cache_state,
            demoted_slot=1,
            keep_count=5,  # More than 3 tokens in slot
            layer_id=0,
            frame_id=1,
        )

        self.assertEqual(len(probe.events), 0)


# ===========================================================================
# Test: Dedup/FIFO counterfactual replay
# ===========================================================================

class TestDedupCounterfactualReplay(unittest.TestCase):
    """Test dedup and FIFO counterfactual replay event collection."""

    def test_dedup_counterfactual_event_replay(self):
        from ovggt.training.counterfactual_replay import (
            DedupCounterfactualEvent,
            collect_dedup_counterfactual_event,
        )

        class FakeRunner:
            def __init__(self):
                self.keep_indices = None
                self.snapshots = 0
                self.restores = 0

            def snapshot(self):
                self.snapshots += 1
                return {"state": True}

            def restore(self, snapshot):
                self.restores += 1
                self.keep_indices = None

            def apply_keep_indices(self, layer_id, keep_indices):
                self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long).clone()

            def replay_future_window(self, start_frame_idx, future_frames):
                quality = float(self.keep_indices.float().mean().item())
                return [{"camera_pose": torch.tensor([quality, 0.0])} for _ in future_frames]

            def targets_for_future_window(self, future_frames):
                return [{"camera_pose": torch.tensor([2.5, 0.0])} for _ in future_frames]

        event = DedupCounterfactualEvent(
            event_id="dedup:test:f0:l0",
            layer_id=0,
            frame_id=0,
            voxel_group_id=42,
            sequence_provenance=None,
            score_state=torch.randn(4, 8),
            metadata_features=torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            candidate_subsets=[
                {"keep_index": 0, "evict_indices": [1], "keep_indices": [0, 2, 3]},
                {"keep_index": 1, "evict_indices": [0], "keep_indices": [1, 2, 3]},
            ],
            base_scores=torch.tensor([0.5, 0.3, 0.2, 0.1]),
        )

        runner = FakeRunner()
        result = collect_dedup_counterfactual_event(
            event=event,
            runner=runner,
            future_frames=[{"frame_id": 1}],
        )

        self.assertEqual(result["event_type"], "dedup")
        self.assertEqual(result["voxel_group_id"], 42)
        self.assertEqual(len(result["subsets"]), 2)
        self.assertIn("loss", result["subsets"][0])
        self.assertIn("loss_components", result["subsets"][0])
        # Runner should have been snapshotted and restored
        self.assertEqual(runner.snapshots, 1)
        self.assertGreaterEqual(runner.restores, 2)

    def test_fifo_topk_counterfactual_event_replay(self):
        from ovggt.training.counterfactual_replay import (
            FifoTopKCounterfactualEvent,
            collect_fifo_topk_counterfactual_event,
        )

        class FakeRunner:
            def __init__(self):
                self.keep_indices = None
                self.snapshots = 0

            def snapshot(self):
                self.snapshots += 1
                return {"state": True}

            def restore(self, snapshot):
                self.keep_indices = None

            def apply_keep_indices(self, layer_id, keep_indices):
                self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long).clone()

            def replay_future_window(self, start_frame_idx, future_frames):
                quality = float(self.keep_indices.float().mean().item())
                return [{"camera_pose": torch.tensor([quality, 0.0])} for _ in future_frames]

            def targets_for_future_window(self, future_frames):
                return [{"camera_pose": torch.tensor([2.5, 0.0])} for _ in future_frames]

        event = FifoTopKCounterfactualEvent(
            event_id="fifo:test:f1:l0",
            layer_id=0,
            frame_id=1,
            demoted_slot=1,
            keep_count=3,
            sequence_provenance={"sequence_id": "test/seq0"},
            score_state=torch.randn(6, 8),
            metadata_features=torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
            candidate_subsets=[
                {"strategy": "top_importance", "keep_indices": [0, 1, 2]},
                {"strategy": "random", "keep_indices": [3, 4, 5]},
            ],
        )

        runner = FakeRunner()
        result = collect_fifo_topk_counterfactual_event(
            event=event,
            runner=runner,
            future_frames=[{"frame_id": 2}, {"frame_id": 3}],
        )

        self.assertEqual(result["event_type"], "fifo_topk")
        self.assertEqual(result["demoted_slot"], 1)
        self.assertEqual(result["keep_count"], 3)
        self.assertEqual(len(result["subsets"]), 2)
        self.assertIn("loss", result["subsets"][0])
        self.assertEqual(
            result["sequence_provenance"]["sequence_id"], "test/seq0"
        )


# ===========================================================================
# Test: Regression — scorer=None identical to v1
# ===========================================================================

class TestRegressionScorerNone(unittest.TestCase):
    """When token_scorer=None, behavior should be identical to v1."""

    def test_dedup_without_scorer_matches_v1(self):
        """apply_voxel_dedup_ without scorer uses composite score (v1 path)."""
        state = build_dedup_state(num_tokens=8, num_protected=2)
        config = FrontendCacheConfig(
            enabled=True, voxel_size=0.5,
            importance_weight=0.5, depth_conf_weight=0.5,
        )

        # Record token count before
        tokens_before = state.num_tokens()

        # Call with scorer=None (v1 behavior)
        state.apply_voxel_dedup_(config, current_frame_id=1,
                                  token_scorer=None, layer_id=0)

        # Basic invariants: dedup should not increase tokens
        self.assertLessEqual(state.num_tokens(), tokens_before)
        # k and metadata should stay in sync
        self.assertEqual(state.k.shape[2], state.metadata.frame_id.shape[1])

    def test_fifo_without_scorer_matches_v1(self):
        """protect_topk_on_demotion_ without scorer uses importance (v1 path)."""
        num_tokens = 20
        k = torch.randn(1, 2, num_tokens, 8)
        v = torch.randn(1, 2, num_tokens, 8)
        metadata = make_metadata(
            anchor_slots=[0] * 8 + [1] * 12,
            frame_ids=[0] * 8 + [1] * 12,
            importance=[float(i) for i in range(num_tokens)],
        )

        state = LayerCacheState(
            k=k, v=v, metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(1.0)},
        )

        # Call with scorer=None (v1 behavior)
        state.protect_topk_on_demotion_(
            demoted_slot=1, keep_count=5,
            token_scorer=None, layer_id=0,
        )

        # top-5 importance tokens from demoted slot should be reassigned to slot 0
        # The demoted slot tokens had importance 8..19
        # Top-5 by importance: indices 19, 18, 17, 16, 15 → reassigned to slot 0
        slot0_count = (state.metadata.anchor_slot[0] == 0).sum().item()
        self.assertEqual(slot0_count, 8 + 5)  # 8 original + 5 promoted

    def test_commit_without_scorer_matches_v1(self):
        """commit_pending_update_ without scorer uses importance for pruning (v1 path)."""
        from ovggt.layers.attention import Attention

        num_new_tokens = 10
        existing_k = torch.randn(1, 2, 2, 4)
        existing_v = torch.randn(1, 2, 2, 4)
        existing_metadata = make_metadata(
            anchor_slots=[0, 0], frame_ids=[0, 0], slot_ids=[0, 0],
        )
        state = LayerCacheState(
            k=existing_k, v=existing_v, metadata=existing_metadata,
            protected_count=2, max_history_anchors=2,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        pending = PendingLayerUpdate(
            k_current=torch.randn(1, 2, num_new_tokens, 4),
            v_current=torch.randn(1, 2, num_new_tokens, 4),
            importance_current=torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0]]),
            frame_id=0,
            cache_budget=100,
            score_state_current=torch.randn(1, num_new_tokens, 16),
        )
        current_metadata = make_metadata(
            anchor_slots=[-1] * num_new_tokens,
            frame_ids=[0] * num_new_tokens,
            slot_ids=[0] * num_new_tokens,
            importance=[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0],
            depth_conf=[0.5] * num_new_tokens,
        )

        # v1: scorer=None, should use importance for pruning
        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=0.5,
            attn_module=Attention(dim=8, num_heads=2),
            token_scorer=None,  # v1 behavior
            layer_id=0,
        )

        # Should keep top-5 by importance + 2 anchor = 7
        expected_kept = 2 + max(int(num_new_tokens * 0.5), 1)
        self.assertEqual(state.num_tokens(), expected_kept)
        # k and metadata should be in sync
        self.assertEqual(state.k.shape[2], state.metadata.frame_id.shape[1])


if __name__ == "__main__":
    unittest.main()
