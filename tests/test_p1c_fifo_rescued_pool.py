import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import pytest
import torch
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig, TokenMetadata, LayerCacheState, PendingLayerUpdate,
)


# ---------- Task 1: config + mutual-exclusion assert ----------

def test_config_has_fifo_protected_ring_ratio_default_off():
    cfg = FrontendCacheConfig()
    assert cfg.fifo_protected_ring_ratio == 0.0


def test_config_rejects_ring_and_max_protected_both_set():
    with pytest.raises(ValueError, match="互斥"):
        FrontendCacheConfig(fifo_protected_ring_ratio=0.3, max_protected_ratio=0.5)


def test_config_allows_ring_with_default_max_protected():
    cfg = FrontendCacheConfig(fifo_protected_ring_ratio=0.3)
    assert cfg.fifo_protected_ring_ratio == 0.3


def test_config_allows_max_protected_without_ring():
    cfg = FrontendCacheConfig(max_protected_ratio=0.5)
    assert cfg.max_protected_ratio == 0.5


# ---------- Task 2: _needs_reorder_after_revoke field ----------

def test_layer_cache_state_has_reorder_flag_field():
    cs = LayerCacheState()
    assert cs._needs_reorder_after_revoke is False
    cs._needs_reorder_after_revoke = True
    assert cs._needs_reorder_after_revoke is True


# ---------- helper for Task 3 ring tests ----------

def _make_cache(slot0_kf_ids, demoted_slot=1, demoted_kf=2, n_demoted=10):
    """slot0 has [global(kf=0)] + rescued tokens with given keyframe_ids;
    demoted slot has n_demoted tokens of demoted_kf."""
    frame = [0] + list(slot0_kf_ids) + [demoted_kf] * n_demoted
    anchor = [0] + [0] * len(slot0_kf_ids) + [demoted_slot] * n_demoted
    kf = [0] + list(slot0_kf_ids) + [demoted_kf] * n_demoted
    N = len(frame)
    meta = TokenMetadata(
        token_kind=torch.full((1, N), 2, dtype=torch.long),
        frame_id=torch.tensor([frame]),
        anchor_slot=torch.tensor([anchor]),
        keyframe_id=torch.tensor([kf]),
        slot_id=torch.tensor([kf]),
        slot_local_xyz=torch.zeros(1, N, 3),
        importance=torch.rand(1, N),
        depth_conf=torch.ones(1, N),
    )
    cs = LayerCacheState(max_history_anchors=3)
    cs.k = torch.randn(1, 4, N, 8)
    cs.v = torch.randn(1, 4, N, 8)
    cs.metadata = meta
    cs._cached_protected_count = cs._compute_protected_count_raw()
    cs.protected_count = cs._cached_protected_count
    return cs


# ---------- Task 3: ring revoke logic ----------

def test_ring_revokes_oldest_keyframe_when_over_cap():
    # rescued: kf=1 (5 tokens) + kf=3 (5 tokens). cap=8, keep=4 → 10+4=14>8, overflow=6.
    # oldest rotatable = kf=1 (id smaller). revoke min(6,10)=6 oldest → all 5 of kf=1 + 1 of kf=3.
    cs = _make_cache([1]*5 + [3]*5, n_demoted=10)
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=4, fifo_ring_capacity=8,
                                  global_anchor_keyframe_id=0)
    kf1_slot0 = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert kf1_slot0 == 0, "oldest keyframe (kf=1) fully revoked"


def test_ring_no_revoke_when_under_cap():
    cs = _make_cache([1]*3, n_demoted=4)  # rescued 3, keep 4, cap 8 → 3+4=7<=8
    kf1_before = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=4, fifo_ring_capacity=8,
                                  global_anchor_keyframe_id=0)
    kf1_after = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert kf1_after == kf1_before, "under cap → ring does not revoke existing rescued tokens"


def test_ring_global_anchor_never_revoked():
    cs = _make_cache([1]*10, n_demoted=5)  # force revoke
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=8,
                                  global_anchor_keyframe_id=0)
    global_slot0 = ((cs.metadata.keyframe_id[0] == 0) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert global_slot0 == 1, "global anchor must never be revoked"


def test_ring_disabled_when_capacity_none():
    cs = _make_cache([1]*50, n_demoted=5)
    kf1_before = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=None,
                                  global_anchor_keyframe_id=0)
    kf1_after = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert kf1_after == kf1_before, "ring disabled → no revoke of existing rescued tokens (backward compat)"


def test_ring_clamp_keep_count_when_rotatable_empty():
    # slot0 = only global anchor. cap=5, keep=10 → rotatable empty, clamp keep to cap.
    cs = _make_cache([], n_demoted=10)
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=10, fifo_ring_capacity=5,
                                  global_anchor_keyframe_id=0)
    protected = ((cs.metadata.anchor_slot[0] == 0) & (cs.metadata.keyframe_id[0] == 2)).sum().item()
    assert protected <= 5, f"keep_count clamped to cap, got {protected}"


def test_ring_sets_reorder_flag():
    cs = _make_cache([1]*10, n_demoted=5)
    cs._needs_reorder_after_revoke = False
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=8,
                                  global_anchor_keyframe_id=0)
    assert cs._needs_reorder_after_revoke is True, "revoke must set reorder flag"


def test_ring_clamp_when_overflow_exceeds_rotatable():
    # pass-3 #1: rot_idx=2 (small), keep=20, cap=5 → overflow=17, k=2, remaining=0,
    # keep clamped to 5. Pool after = 0 + 5 = 5 <= cap. Without clamp, pool=20>5.
    cs = _make_cache([1]*2, n_demoted=25)
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=20, fifo_ring_capacity=5,
                                  global_anchor_keyframe_id=0)
    rescued_after = ((cs.metadata.anchor_slot[0] == 0) & (cs.metadata.keyframe_id[0] != 0)).sum().item()
    assert rescued_after <= 5, f"pool must not exceed cap, got {rescued_after}"


# ---------- Task 4: determinism ----------

def test_ring_revoke_deterministic_across_runs():
    def run_once():
        cs = _make_cache([1]*20, n_demoted=5)  # 20 tokens same kf=1 → ties
        cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=10,
                                      global_anchor_keyframe_id=0)
        return cs.metadata.anchor_slot[0].clone()
    a = run_once()
    b = run_once()
    assert torch.equal(a, b), "revoke must be deterministic (argsort stable, token-position tie-break)"


# ---------- Task 5: commit forced reorder ----------

def test_commit_forces_reorder_after_revoke():
    # cache: global(slot0) + revoked token stranded at pos1 (anchor_slot=-1) + candidate
    meta = TokenMetadata(
        token_kind=torch.tensor([[2, 2, 2]]),
        frame_id=torch.tensor([[0, 1, 2]]),
        anchor_slot=torch.tensor([[0, -1, -1]]),  # token1 revoked (now -1) but at pos1
        keyframe_id=torch.tensor([[0, 1, 2]]),
        slot_id=torch.tensor([[0, 1, 2]]),
        slot_local_xyz=torch.zeros(1, 3, 3),
        importance=torch.tensor([[0.5, 0.4, 0.3]]),
        depth_conf=torch.ones(1, 3),
    )
    cs = LayerCacheState(max_history_anchors=3)
    cs.k = torch.randn(1, 4, 3, 8); cs.v = torch.randn(1, 4, 3, 8); cs.metadata = meta
    cs._cached_protected_count = cs._compute_protected_count_raw(); cs.protected_count = cs._cached_protected_count
    cs._needs_reorder_after_revoke = True
    pu = PendingLayerUpdate(k_current=torch.randn(1, 4, 1, 8), v_current=torch.randn(1, 4, 1, 8),
                            importance_current=torch.tensor([[0.2]]), frame_id=5, cache_budget=None)
    cur_meta = TokenMetadata(token_kind=torch.tensor([[2]]), frame_id=torch.tensor([[5]]),
                             anchor_slot=torch.tensor([[-1]]), keyframe_id=torch.tensor([[5]]),
                             slot_id=torch.tensor([[5]]), slot_local_xyz=torch.zeros(1, 1, 3),
                             importance=torch.tensor([[0.2]]), depth_conf=torch.ones(1, 1))
    cs.commit_pending_update_(pu, cur_meta, FrontendCacheConfig(), intra_frame_keep_ratio=1.0, attn_module=None)
    assert cs._needs_reorder_after_revoke is False, "flag reset after reorder"
    # after forced reorder: first token must be protected (global), revoked token moved out
    assert cs.metadata.anchor_slot[0, 0].item() >= 0


# ---------- Task 6: ovggt wiring ----------

def test_ovggt_wiring_uses_keyframe_managers_b():
    import inspect
    import ovggt.models.ovggt as m
    src = inspect.getsource(m)
    assert "keyframe_manager.global_anchor" not in src, "must use keyframe_managers[b], not singular"
    assert "fifo_ring_capacity" in src, "ring capacity must be wired"
    assert "global_anchor_keyframe_id" in src, "global anchor id must be wired"
