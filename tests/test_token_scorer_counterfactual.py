import os
import sys
import copy

import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.block import Block
import ovggt.layers.token_scorer as token_scorer_module
from ovggt.layers.token_scorer import TokenScorer
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    PendingLayerUpdate,
    TokenKind,
    TokenMetadata,
)


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


def make_transform(tx: float = 0.0) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)
    transform[0, 3] = tx
    return transform


def test_token_scorer_accepts_score_state_metadata_and_layer_id():
    scorer = TokenScorer(
        score_state_dim=8,
        metadata_dim=token_scorer_module.TOKEN_METADATA_FEATURE_DIM,
        hidden_dim=16,
        num_layers=4,
    )
    with torch.no_grad():
        for param in scorer.parameters():
            param.zero_()
        scorer.scorer[-1].bias.fill_(2.0)
    score_state = torch.randn(2, 5, 8, requires_grad=True)
    metadata_features = torch.randn(2, 5, token_scorer_module.TOKEN_METADATA_FEATURE_DIM)

    logits = scorer(score_state, metadata_features, layer_id=2)

    assert logits.shape == (2, 5)
    assert torch.allclose(logits, torch.full_like(logits, 2.0))
    logits.sum().backward()
    assert score_state.grad is not None
    assert any(p.grad is not None for p in scorer.parameters())


def test_block_frontend_cache_returns_score_state_without_calling_scorer():
    class RaisingScorer(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("TokenScorer must run at cache commit, not in Block.forward")

    block = Block(dim=8, num_heads=2, eviction_strategy="repr_shift")
    block.token_scorer = RaisingScorer()
    block.score_state_proj = nn.Linear(8, 4)
    x = torch.randn(1, 6, 8)

    tokens, kv_current, importance, score_state = block(
        x,
        use_cache=True,
        frontend_cache_mode=True,
        cache_budget=16,
    )

    assert tokens.shape == x.shape
    assert kv_current[0].shape[2] == x.shape[1]
    assert importance.shape == (1, 6)
    assert score_state.shape == (1, 6, 4)


def test_score_state_follows_kv_through_append_and_gather():
    k = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
    v = k + 100.0
    score_state = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    metadata = make_metadata(anchor_slots=[0, 0, -1, -1, -1, -1])
    state = LayerCacheState(k=k, v=v, score_state=score_state, metadata=metadata)

    indices = torch.tensor([[0, 3, 5]], dtype=torch.long)
    state.gather_(indices)

    assert torch.equal(state.k[0, 0, :, 0], torch.tensor([0.0, 6.0, 10.0]))
    assert torch.equal(state.score_state[0, :, 0], torch.tensor([0.0, 9.0, 15.0]))

    state.append_(
        torch.ones(1, 2, 2, 2),
        torch.ones(1, 2, 2, 2),
        make_metadata(anchor_slots=[-1, -1]),
        score_state_new=torch.full((1, 2, 3), 42.0),
    )

    assert state.k.shape[2] == 5
    assert torch.equal(state.score_state[0, -2:, 0], torch.tensor([42.0, 42.0]))


def test_layer_cache_state_deepcopy_snapshot_does_not_share_tensors():
    state = LayerCacheState(
        k=torch.randn(1, 2, 4, 3),
        v=torch.randn(1, 2, 4, 3),
        score_state=torch.randn(1, 4, 2),
        metadata=make_metadata(anchor_slots=[0, -1, -1, -1]),
        protected_count=1,
        slot_to_active={0: make_transform(1.0)},
    )

    snapshot = copy.deepcopy(state)
    state.k.zero_()
    state.v.zero_()
    state.score_state.zero_()
    state.metadata.depth_conf.fill_(9.0)
    state.slot_to_active[0].zero_()

    assert not torch.equal(snapshot.k, state.k)
    assert not torch.equal(snapshot.v, state.v)
    assert not torch.equal(snapshot.score_state, state.score_state)
    assert not torch.equal(snapshot.metadata.depth_conf, state.metadata.depth_conf)
    assert not torch.equal(snapshot.slot_to_active[0], state.slot_to_active[0])


def test_metadata_feature_builder_uses_commit_time_metadata_and_projection():
    metadata = make_metadata(
        anchor_slots=[0, -1, -1],
        frame_ids=[1, 3, 4],
        keyframe_ids=[1, 2, 4],
        slot_ids=[0, 1, 1],
        token_kind=[int(TokenKind.CAMERA), int(TokenKind.REGISTER), int(TokenKind.PATCH)],
        depth_conf=[0.0, 0.25, 0.75],
        local_xyz=[
            [float("nan"), float("nan"), float("nan")],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    state = LayerCacheState(
        metadata=metadata,
        slot_to_active={0: make_transform(0.0), 1: make_transform(10.0)},
    )

    features = state.build_scorer_metadata_features(current_frame_id=5)

    feature_index = token_scorer_module.TOKEN_METADATA_FEATURE_INDEX
    assert features.shape == (1, 3, token_scorer_module.TOKEN_METADATA_FEATURE_DIM)
    assert features[0, 2, feature_index["depth_conf"]] == 0.75
    assert torch.allclose(
        features[0, 2, feature_index["slot_local_xyz_start"] : feature_index["slot_local_xyz_start"] + 3],
        torch.tensor([0.4, 0.5, 0.6]),
    )
    assert torch.allclose(
        features[0, 2, feature_index["active_xyz_start"] : feature_index["active_xyz_start"] + 3],
        torch.tensor([1.0, 0.5, 0.6]),
    )
    assert torch.allclose(features[0, 2, feature_index["frame_age"]], torch.tensor(1.0 / 128.0))
    assert features[0, 0, feature_index["is_protected"]] == 1.0
    assert features[0, 1, feature_index["kind_register"]] == 1.0
    assert features[0, 2, feature_index["kind_patch"]] == 1.0
    assert features[0, 0, feature_index["xyz_valid"]] == 0.0
    assert features[0, 2, feature_index["xyz_valid"]] == 1.0


def test_metadata_features_are_bounded_and_mark_invalid_xyz():
    metadata = make_metadata(
        anchor_slots=[-1, 0],
        frame_ids=[0, 0],
        keyframe_ids=[999, 999],
        slot_ids=[999, 999],
        depth_conf=[100.0, -1.0],
        local_xyz=[
            [float("nan"), 0.0, 0.0],
            [1000.0, -1000.0, 50.0],
        ],
    )
    state = LayerCacheState(
        metadata=metadata,
        slot_to_active={999: make_transform(1000.0)},
    )

    features = state.build_scorer_metadata_features(current_frame_id=1000)
    feature_index = token_scorer_module.TOKEN_METADATA_FEATURE_INDEX

    assert features.abs().max() <= 1.0
    assert features[0, 0, feature_index["xyz_valid"]] == 0.0
    assert features[0, 1, feature_index["xyz_valid"]] == 1.0
    assert features[0, 0, feature_index["frame_age"]] == 1.0
    assert features[0, 1, feature_index["slot_id"]] == 1.0


def test_learned_eviction_scores_old_and_new_candidates_with_same_scorer():
    class ScoreStateScorer(nn.Module):
        def forward(self, score_state, metadata_features, layer_id):
            return score_state.squeeze(-1)

    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    state = LayerCacheState(
        k=k,
        v=v,
        score_state=torch.tensor([[[0.0], [0.0], [1.0], [10.0], [2.0]]]),
        metadata=make_metadata(
            anchor_slots=[0, 0, -1, -1, -1],
            frame_ids=[0, 0, 0, 0, 0],
            importance=[0.0, 0.0, 0.1, 0.1, 0.1],
        ),
        protected_count=2,
    )
    pending = PendingLayerUpdate(
        k_current=torch.randn(1, 2, 2, 4),
        v_current=torch.randn(1, 2, 2, 4),
        score_state_current=torch.tensor([[[3.0], [4.0]]]),
        importance_current=torch.tensor([[0.0, 1.0]]),
        frame_id=1,
        cache_budget=4,
    )
    current_metadata = make_metadata(
        anchor_slots=[-1, -1],
        frame_ids=[1, 1],
        importance=[0.0, 1.0],
    )

    state.commit_pending_update_(
        pending_update=pending,
        current_metadata=current_metadata,
        config=FrontendCacheConfig(
            enabled=True,
            dedup_enabled=False,
            learned_eviction_enabled=True,
            budget_allocation="uniform",
        ),
        intra_frame_keep_ratio=1.0,
        attn_module=None,
        token_scorer=ScoreStateScorer(),
        layer_id=3,
    )

    assert state.num_tokens() == 4
    assert torch.equal(state.metadata.anchor_slot[0, :2], torch.tensor([0, 0]))
    assert torch.equal(state.score_state[0, :, 0], torch.tensor([0.0, 0.0, 10.0, 4.0]))
    assert torch.equal(state.metadata.frame_id[0], torch.tensor([0, 0, 0, 1]))


def test_learned_eviction_honors_budget_when_protected_tokens_overflow():
    class RaisingScorer(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("Scorer should not run when budget cannot keep candidate tokens")

    state = LayerCacheState(
        k=torch.randn(1, 2, 5, 4),
        v=torch.randn(1, 2, 5, 4),
        score_state=torch.arange(5, dtype=torch.float32).reshape(1, 5, 1),
        metadata=make_metadata(
            anchor_slots=[0, 1, 2, -1, -1],
            frame_ids=[0, 1, 2, 3, 3],
        ),
        protected_count=3,
    )

    state._learned_eviction_(
        cache_budget=2,
        token_scorer=RaisingScorer(),
        layer_id=0,
        current_frame_id=3,
    )

    assert state.num_tokens() == 2
    assert torch.equal(state.metadata.anchor_slot[0], torch.tensor([1, 2]))
    assert torch.equal(state.score_state[0, :, 0], torch.tensor([1.0, 2.0]))


def test_eviction_probe_hook_can_override_commit_eviction_keep_set():
    class HookProbe:
        def __init__(self):
            self.calls = []

        def on_eviction_candidate(self, cache_state, layer_id, frame_id, budget, batch_index=0):
            self.calls.append(
                {
                    "layer_id": layer_id,
                    "frame_id": frame_id,
                    "budget": budget,
                    "score_state": cache_state.score_state.detach().clone(),
                    "metadata_features": cache_state.build_scorer_metadata_features(frame_id).detach().clone(),
                }
            )
            return torch.tensor([0, 2, 4], dtype=torch.long)

    state = LayerCacheState(
        k=torch.randn(1, 2, 3, 4),
        v=torch.randn(1, 2, 3, 4),
        score_state=torch.arange(3, dtype=torch.float32).reshape(1, 3, 1),
        metadata=make_metadata(
            anchor_slots=[0, -1, -1],
            frame_ids=[0, 0, 0],
            importance=[0.0, 0.2, 0.3],
        ),
        protected_count=1,
    )
    pending = PendingLayerUpdate(
        k_current=torch.randn(1, 2, 2, 4),
        v_current=torch.randn(1, 2, 2, 4),
        score_state_current=torch.tensor([[[3.0], [4.0]]]),
        importance_current=torch.tensor([[0.8, 0.9]]),
        frame_id=2,
        cache_budget=3,
    )
    probe = HookProbe()

    state.commit_pending_update_(
        pending_update=pending,
        current_metadata=make_metadata(
            anchor_slots=[-1, -1],
            frame_ids=[2, 2],
            importance=[0.8, 0.9],
        ),
        config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        intra_frame_keep_ratio=1.0,
        attn_module=None,
        layer_id=5,
        eviction_probe=probe,
        batch_index=0,
    )

    assert len(probe.calls) == 1
    assert probe.calls[0]["layer_id"] == 5
    assert probe.calls[0]["frame_id"] == 2
    assert probe.calls[0]["budget"] == 3
    assert probe.calls[0]["score_state"].shape == (1, 5, 1)
    assert probe.calls[0]["metadata_features"].shape[1] == 5
    assert state.num_tokens() == 3
    assert torch.equal(state.score_state[0, :, 0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(state.metadata.frame_id[0], torch.tensor([0, 0, 2]))
