"""TDD for soft-merge intra dedup (frontend short-seq Pareto optimization).

Task 1: config field intra_dedup_mode (drop|merge) + precedence.
Task 2-3 tests added incrementally.
"""
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import torch
import torch.nn.functional as F
from ovggt.utils.frontend_cache import FrontendCacheConfig, TokenMetadata, LayerCacheState


# ---------- Task 1: config field ----------

def test_intra_dedup_mode_default_drop():
    assert FrontendCacheConfig().intra_dedup_mode == "drop"


def test_intra_dedup_mode_accepts_merge():
    assert FrontendCacheConfig(intra_dedup_mode="merge").intra_dedup_mode == "merge"


def test_intra_frame_dedup_enabled_false_disables_regardless_of_mode():
    # precedence: the existing boolean wins; mode is inert when disabled
    # (the `if config.intra_frame_dedup_enabled:` guard at the intra block gates the whole thing)
    cfg = FrontendCacheConfig(intra_frame_dedup_enabled=False, intra_dedup_mode="merge")
    assert cfg.intra_frame_dedup_enabled is False
    assert cfg.intra_dedup_mode == "merge"  # field present, just inert


# ---------- Task 2: _dedup_single_batch returns a merge plan ----------

def test_dedup_single_batch_returns_merge_plan():
    """3 current-frame tokens in the SAME 0.1m voxel, scores [0.2,0.5,0.3].
    Merge mode: rep = highest-score (pos 3), members = all 3, weights = softmax."""
    cs = LayerCacheState(max_history_anchors=3)
    cs.metadata = TokenMetadata(
        token_kind=torch.full((1, 5), 2, dtype=torch.long), frame_id=torch.zeros(1, 5, dtype=torch.long),
        anchor_slot=torch.full((1, 5), -1, dtype=torch.long), keyframe_id=torch.zeros(1, 5, dtype=torch.long),
        slot_id=torch.zeros(1, 5, dtype=torch.long), slot_local_xyz=torch.zeros(1, 5, 3),
        importance=torch.zeros(1, 5), depth_conf=torch.ones(1, 5))
    total = 5
    current_patch_mask = torch.tensor([False, False, True, True, True])
    protected_patch_mask = torch.zeros(total, dtype=torch.bool)
    scores = torch.tensor([0.0, 0.0, 0.2, 0.5, 0.3])  # current at 2,3,4
    xyz = torch.zeros(total, 3); xyz[2:, :] = 0.05      # all 3 current -> voxel (0,0,0)
    cfg = FrontendCacheConfig(intra_frame_dedup_enabled=True, intra_dedup_mode="merge", voxel_size=0.1)
    keep_indices, plan = cs._dedup_single_batch(
        b_idx=0, protected_patch_mask=protected_patch_mask, current_patch_mask=current_patch_mask,
        scores=scores, config=cfg, total_tokens=total, projected_xyz=xyz)
    assert plan is not None, "merge mode must return a plan"
    assert plan["rep_indices"].numel() == 1
    assert plan["rep_indices"][0].item() == 3, "rep = highest-score token (pos 3, score 0.5)"
    assert plan["member_indices"].numel() == 3
    assert sorted(plan["member_indices"].tolist()) == [2, 3, 4], "members = all 3 co-voxel tokens"
    assert plan["member_weights"].sum().item() == pytest_approx(1.0), "weights normalized"
    assert (plan["member_weights"] > 0).all()
    assert plan["member_rep_map"].tolist() == [0, 0, 0], "all 3 map to the single rep"
    # token count unchanged from drop mode: 2 non-current + 1 rep = 3
    assert keep_indices.numel() == 3


def test_dedup_single_batch_drop_mode_returns_no_plan():
    """drop mode returns (keep_indices, None)."""
    cs = LayerCacheState(max_history_anchors=3)
    cs.metadata = TokenMetadata(
        token_kind=torch.full((1, 5), 2, dtype=torch.long), frame_id=torch.zeros(1, 5, dtype=torch.long),
        anchor_slot=torch.full((1, 5), -1, dtype=torch.long), keyframe_id=torch.zeros(1, 5, dtype=torch.long),
        slot_id=torch.zeros(1, 5, dtype=torch.long), slot_local_xyz=torch.zeros(1, 5, 3),
        importance=torch.zeros(1, 5), depth_conf=torch.ones(1, 5))
    total = 5
    current_patch_mask = torch.tensor([False, False, True, True, True])
    protected_patch_mask = torch.zeros(total, dtype=torch.bool)
    scores = torch.tensor([0.0, 0.0, 0.2, 0.5, 0.3])
    xyz = torch.zeros(total, 3); xyz[2:, :] = 0.05
    cfg = FrontendCacheConfig(intra_frame_dedup_enabled=True, intra_dedup_mode="drop", voxel_size=0.1)
    keep_indices, plan = cs._dedup_single_batch(
        b_idx=0, protected_patch_mask=protected_patch_mask, current_patch_mask=current_patch_mask,
        scores=scores, config=cfg, total_tokens=total, projected_xyz=xyz)
    assert plan is None
    assert keep_indices.numel() == 3  # same count as merge (count invariant)


def pytest_approx(v):  # tiny helper to avoid importing pytest approx in module scope
    import pytest
    return pytest.approx(v)


# ---------- Task 3: _apply_intra_merge_ weighted-avg math ----------

def test_apply_intra_merge_weighted_avg():
    """rep slot K = Σ_members w·K; importance = weighted avg; token count via plan."""
    cs = LayerCacheState(max_history_anchors=3)
    K = torch.randn(1, 2, 3, 4); V = torch.randn(1, 2, 3, 4)
    cs.k = K.clone(); cs.v = V.clone()
    cs.metadata = TokenMetadata(
        token_kind=torch.full((1, 3), 2, dtype=torch.long),
        frame_id=torch.tensor([[1, 1, 1]]), anchor_slot=torch.tensor([[-1, -1, -1]]),
        keyframe_id=torch.tensor([[1, 1, 1]]), slot_id=torch.tensor([[1, 1, 1]]),
        slot_local_xyz=torch.zeros(1, 3, 3),
        importance=torch.tensor([[0.2, 0.5, 0.3]]), depth_conf=torch.ones(1, 3),
    )
    # rep=1 (highest), members=[1,2,0] (score-desc), weights=softmax([0.5,0.3,0.2])
    w = F.softmax(torch.tensor([0.5, 0.3, 0.2]), dim=0)
    plan = {"rep_indices": torch.tensor([1]),
            "member_indices": torch.tensor([1, 2, 0]),
            "member_weights": w,
            "member_rep_map": torch.tensor([0, 0, 0])}
    cs._apply_intra_merge_(0, plan)
    expected_k = w[0] * K[0, :, 1, :] + w[1] * K[0, :, 2, :] + w[2] * K[0, :, 0, :]
    assert torch.allclose(cs.k[0, :, 1, :], expected_k, atol=1e-5), "rep K = weighted avg"
    expected_v = w[0] * V[0, :, 1, :] + w[1] * V[0, :, 2, :] + w[2] * V[0, :, 0, :]
    assert torch.allclose(cs.v[0, :, 1, :], expected_v, atol=1e-5), "rep V = weighted avg"
    # importance weighted-avg
    exp_imp = (w[0] * 0.5 + w[1] * 0.3 + w[2] * 0.2).item()
    assert abs(cs.metadata.importance[0, 1].item() - exp_imp) < 1e-5, "rep importance = weighted avg"
