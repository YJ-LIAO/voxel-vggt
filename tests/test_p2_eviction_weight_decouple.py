"""P2 Phase-1: decouple eviction_importance_weight from (dedup) importance_weight.

The single config field `importance_weight` was overloaded:
  - eviction (attention.py eviction old/new temporal blend, called at
    frontend_cache.py commit_pending_update_ ~line 1285)
  - dedup composite (_composite_candidate_scores_batch importance-vs-depth_conf
    blend, frontend_cache.py ~line 799)
Different semantics. P2 needs a clean eviction-weight sweep, so we add a
dedicated `eviction_importance_weight` field and route eviction through it,
leaving dedup on `importance_weight`.
"""
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import pytest
import torch
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig, TokenMetadata, LayerCacheState, PendingLayerUpdate,
)


# ---------- Task 1.1: config field exists, default 0.5, independent ----------

def test_config_has_eviction_importance_weight_default():
    cfg = FrontendCacheConfig()
    assert cfg.eviction_importance_weight == 0.5


def test_eviction_importance_weight_independent_of_importance_weight():
    cfg = FrontendCacheConfig(importance_weight=0.9)
    assert cfg.eviction_importance_weight == 0.5  # unchanged
    cfg2 = FrontendCacheConfig(eviction_importance_weight=0.3)
    assert cfg2.importance_weight == 0.5  # unchanged


# ---------- Task 1.2: commit_pending_update_ threads eviction_importance_weight ----------

class _MockAttn:
    """Records the importance_weight passed to eviction(). Returns a trimmed cache."""
    def __init__(self):
        self.recorded_weight = None

    def eviction(self, k, v, cache_budget, num_anchor_tokens,
                 importance_scores=None, num_new_tokens=0, importance_weight=0.5):
        self.recorded_weight = float(importance_weight)
        keep = max(int(cache_budget), 0)
        B, H, N, D = k.shape
        out_k = k[:, :, :keep, :]
        out_v = v[:, :, :keep, :]
        idx = torch.arange(keep, device=k.device).unsqueeze(0).expand(B, -1)
        return out_k, out_v, 0.0, idx


def _cache_with_candidates(n_old=6, n_new=1, D=8, H=4):
    """LayerCacheState with n_old old-frame candidate tokens + (pending) n_new
    current-frame tokens. protected_count=0 (no anchors) so all are candidates."""
    N = n_old
    frame = [0] * n_old
    anchor = [-1] * n_old
    kf = [0] * n_old
    meta = TokenMetadata(
        token_kind=torch.full((1, N), 2, dtype=torch.long),
        frame_id=torch.tensor([frame]),
        anchor_slot=torch.tensor([anchor]),
        keyframe_id=torch.tensor([kf]),
        slot_id=torch.tensor([kf]),
        slot_local_xyz=torch.zeros(1, N, 3),
        importance=torch.rand(1, N),   # old-token repr_shift (varied)
        depth_conf=torch.ones(1, N),
    )
    cs = LayerCacheState(max_history_anchors=3)
    cs.k = torch.randn(1, H, N, D)
    cs.v = torch.randn(1, H, N, D)
    cs.metadata = meta
    cs._cached_protected_count = 0
    cs.protected_count = 0
    # current-frame pending update
    pu = PendingLayerUpdate(
        k_current=torch.randn(1, H, n_new, D),
        v_current=torch.randn(1, H, n_new, D),
        importance_current=torch.rand(1, n_new),
        frame_id=1,
        cache_budget=4,   # < n_old+n_new=7 -> triggers eviction
    )
    cur_meta = TokenMetadata(
        token_kind=torch.full((1, n_new), 2, dtype=torch.long),
        frame_id=torch.tensor([[1]]),
        anchor_slot=torch.tensor([[-1]]),
        keyframe_id=torch.tensor([[1]]),
        slot_id=torch.tensor([[1]]),
        slot_local_xyz=torch.zeros(1, n_new, 3),
        importance=torch.rand(1, n_new),
        depth_conf=torch.ones(1, n_new),
    )
    return cs, pu, cur_meta


def test_commit_threads_eviction_importance_weight_to_eviction():
    # eviction_importance_weight=0.3
    cs1, pu1, cur1 = _cache_with_candidates()
    mock1 = _MockAttn()
    cs1.commit_pending_update_(pu1, cur1, FrontendCacheConfig(eviction_importance_weight=0.3),
                               intra_frame_keep_ratio=1.0, attn_module=mock1)
    # eviction_importance_weight=0.7
    cs2, pu2, cur2 = _cache_with_candidates()
    mock2 = _MockAttn()
    cs2.commit_pending_update_(pu2, cur2, FrontendCacheConfig(eviction_importance_weight=0.7),
                               intra_frame_keep_ratio=1.0, attn_module=mock2)
    assert mock1.recorded_weight == pytest.approx(0.3), \
        f"eviction must receive eviction_importance_weight=0.3, got {mock1.recorded_weight}"
    assert mock2.recorded_weight == pytest.approx(0.7), \
        f"eviction must receive eviction_importance_weight=0.7, got {mock2.recorded_weight}"


def test_commit_default_eviction_weight_is_05():
    cs, pu, cur = _cache_with_candidates()
    mock = _MockAttn()
    cs.commit_pending_update_(pu, cur, FrontendCacheConfig(),  # defaults
                              intra_frame_keep_ratio=1.0, attn_module=mock)
    assert mock.recorded_weight == pytest.approx(0.5)


# ---------- Task 1.3: dedup does NOT read eviction_importance_weight ----------

def test_dedup_unaffected_by_eviction_importance_weight():
    """apply_voxel_dedup_ must keep using config.importance_weight (the
    importance-vs-depth_conf blend), independent of eviction_importance_weight."""
    def run_dedup(imp_w, evict_w):
        N = 12
        meta = TokenMetadata(
            token_kind=torch.full((1, N), 2, dtype=torch.long),
            frame_id=torch.tensor([[0] * 8 + [1] * 4]),  # 8 old, 4 current
            anchor_slot=torch.tensor([[0] * 3 + [-1] * 9]),  # 3 protected (slot0), 9 candidates
            keyframe_id=torch.tensor([[0] * 8 + [1] * 4]),
            slot_id=torch.tensor([[0] * 8 + [1] * 4]),
            slot_local_xyz=torch.zeros(1, N, 3),
            importance=torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5]]),
            depth_conf=torch.ones(1, N),
        )
        cs = LayerCacheState(max_history_anchors=3)
        cs.k = torch.randn(1, 4, N, 8); cs.v = torch.randn(1, 4, N, 8)
        cs.metadata = meta
        cs._cached_protected_count = 3
        cs.protected_count = 3
        cfg = FrontendCacheConfig(importance_weight=imp_w, eviction_importance_weight=evict_w)
        kept = cs.apply_voxel_dedup_(cfg, current_frame_id=1)
        return None if kept is None else int((cs.metadata.anchor_slot[0] >= 0).sum().item())
    # Vary ONLY eviction_importance_weight (hold importance_weight fixed).
    a = run_dedup(imp_w=0.5, evict_w=0.0)
    b = run_dedup(imp_w=0.5, evict_w=1.0)
    assert a == b, "dedup survivor count must not depend on eviction_importance_weight"
