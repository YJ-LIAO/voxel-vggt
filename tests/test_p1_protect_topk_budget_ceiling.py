"""TDD RED test for P1 fix: protect_topk must respect a budget ceiling so
protected_count cannot grow unboundedly and starve eviction.

P1 (confirmed): each FIFO_SWAP adds fifo_keep_topk tokens to slot 0 permanently;
over many swaps protected_count grows monotonically and can reach/exceed
cache_budget, after which eviction has zero candidate budget and the cache
collapses. Fix: cap the number of additionally-protected tokens so that
protected tokens never exceed a configurable fraction of the budget.
"""
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import torch
from ovggt.utils.frontend_cache import FrontendCacheConfig, TokenMetadata, LayerCacheState


def make_meta(n_per_slot, slots, importance_base=0.1):
    frame_ids, anchor_slots, imp = [], [], []
    for s, n in zip(slots, n_per_slot):
        for i in range(n):
            frame_ids.append(s); anchor_slots.append(s); imp.append(importance_base + 0.0001 * i)
    N = len(frame_ids)
    return TokenMetadata(
        token_kind=torch.full((1, N), 2, dtype=torch.long),
        frame_id=torch.tensor([frame_ids], dtype=torch.long),
        anchor_slot=torch.tensor([anchor_slots], dtype=torch.long),
        keyframe_id=torch.zeros((1, N), dtype=torch.long),
        slot_id=torch.tensor([frame_ids], dtype=torch.long),
        slot_local_xyz=torch.tensor([[[0.5 * i, 0.0, 1.0] for i in range(N)]], dtype=torch.float32),
        importance=torch.tensor([imp], dtype=torch.float32),
        depth_conf=torch.ones((1, N), dtype=torch.float32),
    )


def fresh_cache(meta):
    cs = LayerCacheState(max_history_anchors=3)
    N = len(meta.frame_id[0]); H, D = 4, 8
    cs.k = torch.randn(1, H, N, D); cs.v = torch.randn(1, H, N, D)
    cs.metadata = meta
    cs._cached_protected_count = cs._compute_protected_count_raw()
    cs.protected_count = cs._cached_protected_count
    return cs


def test_protect_topk_respects_budget_ceiling():
    """Across many FIFO swaps, the FIFO-protected slot-0 tokens must stay below
    the configured ceiling. P1 root cause: slot 0 accumulates fifo_keep_topk
    tokens per swap and is never reset."""
    BUDGET = 1000
    cfg = FrontendCacheConfig(enabled=True, dedup_enabled=True, voxel_size=0.1,
                               intra_frame_dedup_enabled=False,
                               fifo_keep_topk=80,
                               fifo_protected_ring_ratio=0.0,  # this test exercises the v1 cap, not the ring
                               max_protected_ratio=0.5)  # NEW config field (P1 fix)

    cs = fresh_cache(make_meta([3, 200, 200, 200], [0, 1, 2, 3]))
    max_protected = int(cfg.max_protected_ratio * BUDGET)
    print(f"预算 {BUDGET}, max_protected={max_protected}")

    def slot0_count():
        return int((cs.metadata.anchor_slot[0] == 0).sum().item())

    slot0_history = [slot0_count()]
    for swap in range(1, 13):
        cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=cfg.fifo_keep_topk,
                                      cache_budget=BUDGET, max_protected=max_protected)
        m = cs.metadata.anchor_slot
        m = torch.where(m == 1, torch.full_like(m, -1), m)
        m = torch.where(m > 1, m - 1, m)
        cs.metadata.anchor_slot = m
        new_meta = make_meta([200], [3])
        cs.append_(torch.randn(1, 4, 200, 8), torch.randn(1, 4, 200, 8), new_meta)
        cs._cached_protected_count = cs._compute_protected_count_raw()
        cs.protected_count = cs._cached_protected_count
        slot0_history.append(slot0_count())

    print(f"slot0 count 历史: {slot0_history}")
    peak = max(slot0_history)
    print(f"峰值 slot0 = {peak}, 上限 = {max_protected}")
    assert peak <= max_protected, (
        f"P1 NOT FIXED: slot0 peaked at {peak}, exceeds ceiling {max_protected}"
    )
    print(f">>> P1 fix verified: slot0 (FIFO 累积) 始终 <= {max_protected}")


def test_no_ceiling_means_default_off_backward_compat():
    """When max_protected_ratio is not set (default), behavior is unchanged (no cap)."""
    cfg = FrontendCacheConfig(enabled=True, fifo_keep_topk=80)
    # default max_protected_ratio should be 1.0 (no cap) for backward compat
    assert cfg.max_protected_ratio >= 1.0, "default must not cap (backward compat)"
    print(f">>> 向后兼容: default max_protected_ratio={cfg.max_protected_ratio} (无上限)")


if __name__ == "__main__":
    print("="*60); print("Test 1: protect_topk respects budget ceiling"); print("="*60)
    try:
        test_protect_topk_respects_budget_ceiling()
        print("PASS\n")
    except (AssertionError, TypeError) as e:
        print(f"FAIL (expected RED): {e}\n")
    print("="*60); print("Test 2: backward compat (no default cap)"); print("="*60)
    try:
        test_no_ceiling_means_default_off_backward_compat()
        print("PASS\n")
    except (AssertionError, TypeError) as e:
        print(f"FAIL (expected RED): {e}\n")
