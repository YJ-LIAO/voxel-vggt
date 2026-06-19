# P1 Mechanism C (FIFO Rescued Pool) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement P1 mechanism C (bounded FIFO rescued-pool rotation) in `protect_topk_on_demotion_`, with all pass-1→pass-5 spec fixes, via TDD.

**Architecture:** Add a `fifo_protected_ring_ratio` config that caps the non-global rescued tokens in slot-0 to a fraction of `per_layer_budget`. When a FIFO_SWAP would exceed the cap, revoke the oldest keyframe's rescued tokens (by `keyframe_id`, deterministic argsort) before protecting new ones. Revoke plan computed before probe, mutation applied after; forced reorder via a new dataclass field. v1 `max_protected_ratio` and ring are mutually exclusive (config-level assert).

**Tech Stack:** Python, PyTorch, pytest, conda env `OVGGT`. Run tests with `PYTHONPATH=src python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-14-p1-protection-rotation-p2-scoring-design.md` (section 1.4 pseudocode, section 4 Stage 0 test cases).

---

## File Structure

- **Modify** `src/ovggt/utils/frontend_cache.py`:
  - `FrontendCacheConfig` (~line 43-63): add `fifo_protected_ring_ratio` field + `__post_init__` mutual-exclusion assert.
  - `LayerCacheState` (~line 167): add `_needs_reorder_after_revoke: bool = False` field.
  - `protect_topk_on_demotion_` (~line 437-576): add params `fifo_ring_capacity`, `global_anchor_keyframe_id`; add step 2.5 (revoke plan), step 3.5 (apply revoke), wire `keep_count_by_batch` into step 4 (line ~550).
  - `commit_pending_update_` (~line 1013): force reorder when `_needs_reorder_after_revoke`.
- **Modify** `src/ovggt/models/ovggt.py` (~line 797-846): pass `fifo_ring_capacity` + `global_anchor_keyframe_id` from `keyframe_managers[b]` to `protect_topk_on_demotion_`.
- **Create** `tests/test_p1c_fifo_rescued_pool.py`: 11 RED→GREEN tests (one per Stage 0 case).

---

## Task 1: Config field + mutual-exclusion assert (TDD)

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py` (`FrontendCacheConfig`, ~line 43-63)
- Test: `tests/test_p1c_fifo_rescued_pool.py`

- [ ] **Step 1: Write failing tests for config + assert**

Create `tests/test_p1c_fifo_rescued_pool.py`:
```python
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import pytest
from ovggt.utils.frontend_cache import FrontendCacheConfig


def test_config_has_fifo_protected_ring_ratio_default_off():
    cfg = FrontendCacheConfig()
    assert cfg.fifo_protected_ring_ratio == 0.0  # default disabled, backward compat


def test_config_rejects_ring_and_max_protected_both_set():
    # pass-5 #4: mutual exclusion enforced at config level
    with pytest.raises(ValueError, match="互斥"):
        FrontendCacheConfig(fifo_protected_ring_ratio=0.3, max_protected_ratio=0.5)


def test_config_allows_ring_with_default_max_protected():
    # ring on, max_protected at default 1.0 (disabled) — OK
    cfg = FrontendCacheConfig(fifo_protected_ring_ratio=0.3)
    assert cfg.fifo_protected_ring_ratio == 0.3


def test_config_allows_max_protected_without_ring():
    # v1 path, no ring — OK
    cfg = FrontendCacheConfig(max_protected_ratio=0.5)
    assert cfg.max_protected_ratio == 0.5
```

- [ ] **Step 2: Run tests — verify FAIL (field missing, no assert)**

Run: `cd /path/to/mount/lyj/voxel-vggt && PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py -v`
Expected: FAIL (AttributeError `fifo_protected_ring_ratio`).

- [ ] **Step 3: Add field + `__post_init__`**

In `src/ovggt/utils/frontend_cache.py`, in `FrontendCacheConfig` after `max_protected_ratio`:
```python
    max_protected_ratio: float = 1.0
    # P1 v2: FIFO rescued-token pool capacity as ratio of per_layer_budget.
    # Caps non-global rescued slot-0 tokens. 0.0 = disabled (backward compat).
    fifo_protected_ring_ratio: float = 0.0

    def __post_init__(self):
        if self.fifo_protected_ring_ratio > 0.0 and self.max_protected_ratio < 1.0:
            raise ValueError(
                "fifo_protected_ring_ratio 和 max_protected_ratio 互斥 (两者顺序作用同一 keep_count 会混淆)。"
                "启用 ring 时保持 max_protected_ratio=1.0 (默认, cap 禁用)。"
            )
```

- [ ] **Step 4: Run tests — verify PASS**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_p1c_fifo_rescued_pool.py
git commit -m "feat(frontend-cache): add fifo_protected_ring_ratio config + mutual-exclusion assert"
```

---

## Task 2: `_needs_reorder_after_revoke` dataclass field

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py` (`LayerCacheState`, ~line 167-176)
- Test: append to `tests/test_p1c_fifo_rescued_pool.py`

- [ ] **Step 1: Write failing test**

Append:
```python
from ovggt.utils.frontend_cache import LayerCacheState

def test_layer_cache_state_has_reorder_flag_field():
    cs = LayerCacheState()
    assert cs._needs_reorder_after_revoke is False  # declared field, default False
    cs._needs_reorder_after_revoke = True
    assert cs._needs_reorder_after_revoke is True
```

- [ ] **Step 2: Run — verify FAIL**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_layer_cache_state_has_reorder_flag_field -v`
Expected: FAIL (AttributeError).

- [ ] **Step 3: Add field**

In `LayerCacheState` dataclass (after `_cached_protected_count`):
```python
    _cached_protected_count: int = 0
    _needs_reorder_after_revoke: bool = False  # pass-2 #6: declared field, not runtime attr
```

- [ ] **Step 4: Run — verify PASS**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_p1c_fifo_rescued_pool.py
git commit -m "feat(frontend-cache): add _needs_reorder_after_revoke dataclass field"
```

---

## Task 3: `protect_topk_on_demotion_` signature + ring revoke logic

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py` (`protect_topk_on_demotion_`, ~line 437-576)
- Test: append to `tests/test_p1c_fifo_rescued_pool.py`

This is the core. Use a shared test-fixture helper to build a `LayerCacheState` with controllable slot-0 rescued tokens.

- [ ] **Step 1: Write failing tests for revoke logic**

Append helper + core tests:
```python
import torch
from ovggt.utils.frontend_cache import TokenMetadata


def _make_cache_with_rescued(slot0_kf_ids, demoted_slot=1, demoted_kf=2, n_demoted=10):
    """slot0 has rescued tokens with given keyframe_ids; demoted slot has n_demoted tokens.
    global anchor = keyframe_id 0 (one token)."""
    frame = [0] + slot0_kf_ids + [demoted_kf] * n_demoted
    anchor = [0] + [0] * len(slot0_kf_ids) + [demoted_slot] * n_demoted
    kf = [0] + slot0_kf_ids + [demoted_kf] * n_demoted
    N = len(frame)
    return LayerCacheState(max_history_anchors=3)._with_meta(
        TokenMetadata(
            token_kind=torch.full((1, N), 2, dtype=torch.long),
            frame_id=torch.tensor([frame]),
            anchor_slot=torch.tensor([anchor]),
            keyframe_id=torch.tensor([kf]),
            slot_id=torch.tensor([kf]),
            slot_local_xyz=torch.zeros(1, N, 3),
            importance=torch.rand(1, N),
            depth_conf=torch.ones(1, N),
        )
    )


def test_ring_revokes_oldest_keyframe_when_over_cap():
    # rescued pool: 2 keyframes (kf=1:5 tokens, kf=3:5 tokens). cap=8, keep=4 → 10+4=14>8, overflow=6.
    # rotatable excludes global(kf=0). oldest = kf=1 (5 tokens). revoke min(6,10)... argsort oldest first.
    cs = _make_cache_with_rescued([1]*5 + [3]*5, n_demoted=10)
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=4, fifo_ring_capacity=8, global_anchor_keyframe_id=0)
    # after: kf=1 rescued tokens should be revoked (anchor_slot -> -1), kf=3 kept
    kf1_still_slot0 = ((cs.metadata.keyframe_id[0] == 1) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    kf3_still_slot0 = ((cs.metadata.keyframe_id[0] == 3) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert kf1_still_slot0 == 0, "oldest keyframe (kf=1) should be fully revoked"
    assert kf3_still_slot0 == 5, "newer keyframe (kf=3) should be retained"


def test_ring_no_revoke_when_under_cap():
    cs = _make_cache_with_rescued([1]*3, n_demoted=4)  # rescued 3, keep 4, cap 8 → 3+4=7<=8
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    before = cs.metadata.anchor_slot[0].clone()
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=4, fifo_ring_capacity=8, global_anchor_keyframe_id=0)
    assert torch.equal(before, cs.metadata.anchor_slot[0]), "under cap → no revoke"


def test_ring_global_anchor_never_revoked():
    cs = _make_cache_with_rescued([1]*10, n_demoted=5)  # force revoke; global is kf=0
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=8, global_anchor_keyframe_id=0)
    global_still_slot0 = ((cs.metadata.keyframe_id[0] == 0) & (cs.metadata.anchor_slot[0] == 0)).sum().item()
    assert global_still_slot0 == 1, "global anchor must never be revoked"


def test_ring_disabled_when_capacity_none():
    cs = _make_cache_with_rescued([1]*50, n_demoted=5)
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    before = cs.metadata.anchor_slot[0].clone()
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=None, global_anchor_keyframe_id=0)
    assert torch.equal(before, cs.metadata.anchor_slot[0]), "ring disabled → no revoke (backward compat)"


def test_ring_clamp_keep_count_when_rotatable_empty():
    # slot0 = only global anchor (1 token). cap=5, keep=10 → rotatable empty, clamp keep to cap.
    cs = _make_cache_with_rescued([], n_demoted=10)  # only global in slot0
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    # Track how many demoted tokens get protected (anchor_slot -> 0). With clamp keep<=5, at most 5.
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=10, fifo_ring_capacity=5, global_anchor_keyframe_id=0)
    protected = ((cs.metadata.anchor_slot[0] == 0) & (cs.metadata.keyframe_id[0] == 2)).sum().item()
    assert protected <= 5, f"keep_count clamped to cap, got {protected}"


def test_ring_sets_reorder_flag():
    cs = _make_cache_with_rescued([1]*10, n_demoted=5)
    cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
    cs._needs_reorder_after_revoke = False
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=8, global_anchor_keyframe_id=0)
    assert cs._needs_reorder_after_revoke is True, "revoke must set reorder flag"
```

Also add the `_with_meta` helper to `LayerCacheState` OR construct directly in the test. Simplest: build the `LayerCacheState` then set `.metadata/.k/.v` manually in the helper (no new method needed). Revise `_make_cache_with_rescued` to set fields directly instead of `_with_meta`.

- [ ] **Step 2: Run — verify FAIL**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py -k ring -v`
Expected: FAIL (TypeError: unexpected keyword `fifo_ring_capacity`).

- [ ] **Step 3: Implement signature + step 2.5/3.5 revoke + step-4 wiring**

In `protect_topk_on_demotion_`:
1. Add params `fifo_ring_capacity: int | None = None, global_anchor_keyframe_id: int | None = None` to signature.
2. After step 1 (demoted indices computed), before the probe (step 3), insert step 2.5 computing `revoke_by_batch` and `keep_count_by_batch` per the spec pseudocode (lines 114-146).
3. After the probe block (step 3) and the `keep_count <= 0` guard, insert step 3.5 applying `revoke_by_batch` (set anchor_slot=-1, set `_needs_reorder_after_revoke=True`, recompute protected_count) — per spec lines 149-155.
4. In step 4 (the `for b_idx, indices in demoted_indices_by_batch.items():` loop at ~line 550), change `effective_keep_count = min(max(int(keep_count), 0), int(indices.numel()))` to use `keep_count_by_batch.get(b_idx, keep_count)`.

Exact insertion: follow spec section 1.4 pseudocode verbatim (it is the reviewed, fixed version).

- [ ] **Step 4: Run — verify PASS**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py -v`
Expected: all ring tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_p1c_fifo_rescued_pool.py
git commit -m "feat(frontend-cache): implement FIFO rescued-pool ring rotation in protect_topk_on_demotion_"
```

---

## Task 4: Determinism + tie-break test

**Files:**
- Modify: `tests/test_p1c_fifo_rescued_pool.py` (test only; implementation already uses argsort stable)

- [ ] **Step 1: Write determinism test**

Append:
```python
def test_ring_revoke_deterministic_across_runs():
    # pass-2 #7: same keyframe_id ties broken deterministically (argsort stable)
    # Two runs with identical input must revoke identical token sets.
    def run_once():
        torch.manual_seed(42)
        cs = _make_cache_with_rescued([1]*20, n_demoted=5)  # 20 tokens same kf=1
        cs.k = torch.randn(1, 4, cs.num_tokens(), 8); cs.v = torch.randn(1, 4, cs.num_tokens(), 8)
        cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=5, fifo_ring_capacity=10, global_anchor_keyframe_id=0)
        return cs.metadata.anchor_slot[0].clone()
    a = run_once()
    b = run_once()
    assert torch.equal(a, b), "revoke must be deterministic (argsort stable, token-position tie-break)"
```

- [ ] **Step 2: Run — verify PASS** (implementation already uses argsort stable from Task 3)

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_ring_revoke_deterministic_across_runs -v`
Expected: PASS. If FAIL (non-deterministic), ensure step 2.5 uses `torch.argsort(rot_kf, stable=True)` (CPU path); if CUDA non-determinism, run this test on CPU (`metadata` on cpu in helper).

- [ ] **Step 3: Commit**

```bash
git add tests/test_p1c_fifo_rescued_pool.py
git commit -m "test(frontend-cache): ring revoke determinism (argsort stable tie-break)"
```

---

## Task 5: `commit_pending_update_` forced reorder

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py` (`commit_pending_update_`, ~line 1065-1071)
- Test: append to `tests/test_p1c_fifo_rescued_pool.py`

- [ ] **Step 1: Write failing test (reorder triggered after revoke even when current frame has no anchor tokens)**

This requires a `PendingLayerUpdate` mock — keep it minimal: set `_needs_reorder_after_revoke=True`, build a cache where revoked tokens are stranded in the protected-region positions, call `commit_pending_update_` with a non-anchor current frame, assert they got moved to candidate region.

Append:
```python
def test_commit_forces_reorder_after_revoke():
    from ovggt.utils.frontend_cache import PendingLayerUpdate, FrontendCacheConfig
    # cache: 1 global (slot0) + 1 revoked token stranded at slot0 position with anchor_slot=-1
    # but positioned before candidates (simulating post-revoke-not-yet-reordered)
    meta = TokenMetadata(
        token_kind=torch.tensor([[2, 2, 2]]),
        frame_id=torch.tensor([[0, 1, 2]]),
        anchor_slot=torch.tensor([[0, -1, -1]]),  # token1 was revoked (now -1) but sits at pos 1
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
    # minimal pending update (non-anchor current frame → has_anchor_tokens False on current)
    pu = PendingLayerUpdate(k_current=torch.randn(1,4,1,8), v_current=torch.randn(1,4,1,8),
                            importance_current=torch.tensor([[0.2]]), frame_id=5, cache_budget=None)
    cur_meta = TokenMetadata(token_kind=torch.tensor([[2]]), frame_id=torch.tensor([[5]]),
                             anchor_slot=torch.tensor([[-1]]), keyframe_id=torch.tensor([[5]]),
                             slot_id=torch.tensor([[5]]), slot_local_xyz=torch.zeros(1,1,3),
                             importance=torch.tensor([[0.2]]), depth_conf=torch.ones(1,1))
    cs.commit_pending_update_(pu, cur_meta, FrontendCacheConfig(), intra_frame_keep_ratio=1.0, attn_module=None)
    # after forced reorder: protected (anchor_slot>=0) must be at front, candidates after
    assert cs._needs_reorder_after_revoke is False, "flag reset after reorder"
    # token at index 1 (revoked, anchor_slot=-1) should no longer be in protected region
    assert cs.metadata.anchor_slot[0, 0].item() >= 0  # first token now protected (global)
```

- [ ] **Step 2: Run — verify FAIL**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_commit_forces_reorder_after_revoke -v`
Expected: FAIL (reorder not forced when `_needs_reorder_after_revoke` set).

- [ ] **Step 3: Implement forced reorder**

In `commit_pending_update_`, find the `if metadata_current.has_anchor_tokens(): self.reorder_by_anchor_slots_()` block (after `self.append_(...)`, ~line 1066-1067). Change to:
```python
        if metadata_current.has_anchor_tokens() or self._needs_reorder_after_revoke:
            self.reorder_by_anchor_slots_()
            self._needs_reorder_after_revoke = False
```

- [ ] **Step 4: Run — verify PASS**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_commit_forces_reorder_after_revoke -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_p1c_fifo_rescued_pool.py
git commit -m "feat(frontend-cache): force reorder after ring revoke in commit_pending_update_"
```

---

## Task 6: ovggt.py call-site wiring

**Files:**
- Modify: `src/ovggt/models/ovggt.py` (~line 829-846, the `protect_topk_on_demotion_` call)
- Test: integration — covered by existing `test_frontend_inference_smoke.py` (must still pass) + a wiring test

- [ ] **Step 1: Write wiring test (no NameError, ring flows through)**

Append to `tests/test_p1c_fifo_rescued_pool.py`:
```python
def test_ovggt_wiring_uses_keyframe_managers_b():
    # pass-2 #1: call site must use keyframe_managers[b] (plural), not singular keyframe_manager.
    # Smoke-import + source check that singular form is absent in the call.
    import inspect, ovggt.models.ovggt as m
    src = inspect.getsource(m)
    # the singular variable must not appear as a bare reference in the protect_topk call region
    assert "keyframe_manager.global_anchor" not in src, "must use keyframe_managers[b], not singular"
```

- [ ] **Step 2: Run — verify current state**

Run: `PYTHONPATH=src python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_ovggt_wiring_uses_keyframe_managers_b -v`
Expected: may PASS or FAIL depending on current source. Proceed to wire regardless.

- [ ] **Step 3: Wire call site**

In `src/ovggt/models/ovggt.py`, the existing `protect_topk_on_demotion_` call (~line 829) already passes `cache_budget` and `max_protected`. Add the two new params. The call already lives inside `if is_fifo_swap and (fifo_keep_topk > 0 or learned_fifo_keep_count):` (correct gate — ring is NOT a trigger per pass-2 #2). Add after the existing kwargs:
```python
                            cache_state.protect_topk_on_demotion_(
                                demoted_slot=demoted_slot,
                                keep_count=keep_count,
                                token_scorer=(...),
                                layer_id=layer_idx,
                                current_frame_id=i,
                                fifo_probe=getattr(self, "_oracle_fifo_probe", None),
                                batch_index=b,
                                cache_budget=self.per_layer_budget,
                                max_protected=(
                                    int(self.frontend_cache_config.max_protected_ratio * self.per_layer_budget)
                                    if self.frontend_cache_config.max_protected_ratio < 1.0 else None
                                ),
                                fifo_ring_capacity=(
                                    int(self.frontend_cache_config.fifo_protected_ring_ratio * self.per_layer_budget)
                                    if self.frontend_cache_config.fifo_protected_ring_ratio > 0.0 else None
                                ),
                                global_anchor_keyframe_id=(
                                    keyframe_managers[b].global_anchor["keyframe_id"]
                                    if keyframe_managers[b].global_anchor is not None else 0
                                ),
                            )
```
Note: `max_protected` is already wired from the prior P1 v1 commit; ensure it stays. Only ADD `fifo_ring_capacity` and `global_anchor_keyframe_id`.

- [ ] **Step 4: Run wiring test + smoke test**

Run: `cd /path/to/mount/lyj/voxel-vggt && PYTHONPATH=src CUDA_VISIBLE_DEVICES=4 python -m pytest tests/test_p1c_fifo_rescued_pool.py::test_ovggt_wiring_uses_keyframe_managers_b tests/test_frontend_inference_smoke.py -v`
Expected: PASS (wiring correct, real pipeline doesn't crash).

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/models/ovggt.py tests/test_p1c_fifo_rescued_pool.py
git commit -m "feat(ovggt): wire fifo_ring_capacity + global_anchor_keyframe_id into protect_topk call"
```

---

## Task 7: Full regression

**Files:** none (verification only)

- [ ] **Step 1: Run all P1 tests + existing frontend suite on free GPU**

Run:
```bash
cd /path/to/mount/lyj/voxel-vggt
source /mnt/lyj/miniconda3/etc/profile.d/conda.sh && conda activate OVGGT
PYTHONPATH=src CUDA_VISIBLE_DEVICES=4 python -m pytest \
  tests/test_p1c_fifo_rescued_pool.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p4_voxel_hash_collision_free.py \
  tests/test_p5_fifo_swap_transform_retention.py \
  tests/test_keyframe_manager.py \
  tests/test_frontend_cache.py \
  tests/test_frontend_inference_smoke.py \
  tests/test_pose_enc_frontend.py \
  tests/test_frontend_batch_training.py -q
```
Expected: all pass (batch training needs free GPU → `CUDA_VISIBLE_DEVICES=4`).

- [ ] **Step 2: If any fail, debug (do NOT skip)**

For each failure: read the assertion, trace whether it's a real regression from this plan or an environmental issue (GPU OOM → use free GPU; determinism → ensure CPU for argsort test). Fix the code, not the test.

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A
git commit -m "test(frontend-cache): full regression green for P1 mechanism C"
```

---

## Notes for the implementer

- **Backward compat**: default `fifo_protected_ring_ratio=0.0` + the v1/v2 mutual-exclusion assert means existing configs (no ring) behave exactly as before. Verify `test_frontend_inference_smoke.py` still passes unchanged.
- **`_make_cache_with_rescued` helper**: build `LayerCacheState()` then set `.metadata`, `.k`, `.v`, `._cached_protected_count`, `.protected_count` directly — do NOT add a `_with_meta` method to production code (test-only concern).
- **Determinism test (Task 4)**: run on CPU (helper tensors on cpu). If argsort stable on CPU is non-deterministic (it isn't), that's a PyTorch bug — report it.
- **The spec pseudocode (section 1.4) is the reviewed source of truth** — implement it verbatim; do not "improve" it. All 5 review passes converged on that logic.
- **Do not touch** `retention_policy.py`, `train_joint_retention_policy.py`, `evaluate_learned_eviction.py` — those are unrelated pre-existing modifications.
