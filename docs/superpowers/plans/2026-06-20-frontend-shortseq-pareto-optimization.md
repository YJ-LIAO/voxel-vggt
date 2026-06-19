# Frontend Short-Seq Pareto Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OVGGT frontend beat legacy at 200/1000f (mean ATE) and 500f (conditional on redkitchen fix) via soft-merge intra dedup + coverage anchoring — single fixed config, no runtime adaptation.

**Architecture:** Two code-change levers + one diagnosis. (1) Soft-merge: `_dedup_single_batch` returns a per-voxel-group merge plan (representative + members + softmax weights); `apply_voxel_dedup_` applies the weighted K/V/score_state scatter into the representative slot BEFORE gather (token COUNT unchanged = num unique voxels, only VALUES differ). (2) Coverage anchoring: plumb `coverage_monitor_only` through `_build_frontend_keyframe_config` (currently hardcoded True = dead code). (3) Diagnose + fix 500f redkitchen crash. Each lever gated empirically before combining.

**Tech Stack:** PyTorch, OVGGT frontend cache, 7-Scenes eval, pytest TDD, GPU 4-7.

**Spec:** `docs/superpowers/specs/2026-06-19-frontend-shortseq-pareto-optimization-design.md` (review-passed, fcac767)

---

## File Structure

- `src/ovggt/utils/frontend_cache.py` — config field `intra_dedup_mode`; `_dedup_single_batch` returns merge plan; new `_apply_intra_merge_` helper; `apply_voxel_dedup_` wiring (B==1 + B>1).
- `src/ovggt/models/ovggt.py` — `_build_frontend_keyframe_config` accepts `coverage_monitor_only`.
- `src/ovggt/utils/frontend_keyframe.py` — verify coverage branch (line 248) works when enabled (likely no change).
- `tools/run_legacy_vs_frontend.py` + `tools/test_multi_scene.py` — production call-site: `intra_dedup_mode="merge"`, coverage anchoring.
- `tests/test_soft_merge_intra_dedup.py` (new) — TDD for soft-merge.
- `tests/test_coverage_anchor_plumbing.py` (new) — TDD for coverage plumbing.

---

## Task 1: Config field `intra_dedup_mode` + precedence (TDD)

**Files:** Modify `src/ovggt/utils/frontend_cache.py` (FrontendCacheConfig ~line 52, `__post_init__` ~85). Test: `tests/test_soft_merge_intra_dedup.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_soft_merge_intra_dedup.py
import sys; sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
from ovggt.utils.frontend_cache import FrontendCacheConfig

def test_intra_dedup_mode_default_drop():
    assert FrontendCacheConfig().intra_dedup_mode == "drop"

def test_intra_dedup_mode_accepts_merge():
    assert FrontendCacheConfig(intra_dedup_mode="merge").intra_dedup_mode == "merge"

def test_intra_frame_dedup_enabled_false_disables_regardless_of_mode():
    # precedence: the existing boolean wins; mode is ignored when disabled
    cfg = FrontendCacheConfig(intra_frame_dedup_enabled=False, intra_dedup_mode="merge")
    assert cfg.intra_frame_dedup_enabled is False  # disabled wins
    # no error raised (mode just inert when disabled)
```

- [ ] **Step 2: Run → FAIL** (`intra_dedup_mode` not a field). `PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 python -m pytest tests/test_soft_merge_intra_dedup.py -v`

- [ ] **Step 3: Implement.** In FrontendCacheConfig add after `intra_frame_dedup_enabled`:
```python
intra_dedup_mode: Literal["drop", "merge"] = "drop"  # "merge" = soft-merge co-voxel tokens (weighted K/V avg) instead of hard drop
```
Add `Literal` import if missing. No `__post_init__` change needed for precedence (the existing `if config.intra_frame_dedup_enabled:` guard at line 1114 already gates the whole intra block — when False, mode is inert by construction; document this in the field comment).

- [ ] **Step 4: Run → PASS.** Commit: `feat(cache): add intra_dedup_mode config field (drop|merge)`

---

## Task 2: `_dedup_single_batch` returns a merge plan (TDD)

**Files:** Modify `src/ovggt/utils/frontend_cache.py` `_dedup_single_batch` (1034-1137). Change return to `(keep_indices, merge_plan)`.

**Merge plan contract** (returned only when `config.intra_dedup_mode=="merge"`; else `None`): a dict with vectorized tensors (global token indices into the N-dim cache axis, batch-local):
- `rep_indices`: [G] representative survivor index per multi-member group
- `member_indices`: [M] flat all-member indices across groups
- `member_weights`: [M] softmax(member scores) within each group
- `member_rep_map`: [M] index into rep_indices (which rep each member belongs to)

- [ ] **Step 1: Write failing test** — craft a cache where 3 current-frame tokens share a voxel (same projected_xyz rounded to 0.1m), scores [0.2, 0.5, 0.3]; assert merge_plan: `rep_indices` has 1 entry (the highest-score token), `member_indices` has 3, `member_weights` ≈ softmax([0.2,0.5,0.3]), `member_rep_map` all zeros (all 3 → the 1 rep); and `keep_indices` count = unique-voxel count (merging doesn't change count vs drop). Also assert `len(member_indices) == sum(member_counts for groups with count>1)`.

- [ ] **Step 2: Run → FAIL** (return shape changed / merge_plan absent).

- [ ] **Step 3: Implement.** In `_dedup_single_batch`, after computing `final_order` / `keep_first` / `duplicate_indices` (existing lines 1121-1135): when `config.intra_dedup_mode=="merge"`, build the merge plan from the group structure. **Scope:** merge plan covers ONLY the `intra_frame_dedup_enabled` block (lines 1114-1135); protected-conflict discards (1104-1105) remain hard drops (those are current-vs-protected, not intra-group). Vectorized construction:
  - `unique_g, member_counts = torch.unique(ordered_group_ids, return_counts=True)`; `multi_mask = member_counts > 1`.
  - rep per group = the survivor at the `keep_first` position (= highest score in group).
  - members per group = all `final_order` positions with that group id; `member_rep_map` = each member's group-local rep index (broadcast via `torch.searchsorted` on `unique_g` for each member's group id → rep index).
  - weights = softmax(member scores) within each group.
  - Return `(keep_indices, {rep_indices, member_indices, member_weights, member_rep_map})`. For drop mode, return `(keep_indices, None)`.

- [ ] **Step 4: Run → PASS.** Commit: `feat(cache): _dedup_single_batch returns intra merge plan`

**Note:** the return-contract change requires updating the 2 call sites + the probe in `apply_voxel_dedup_` (Task 3) — do that next or the code breaks. Keep Task 2 + 3 in one commit if cleaner.

---

## Task 3: Apply merge in `apply_voxel_dedup_` before gather (TDD)

**Files:** Modify `src/ovggt/utils/frontend_cache.py`: new `_apply_intra_merge_(self, b_idx, merge_plan)`; `apply_voxel_dedup_` (B==1 ~808-857, B>1 ~859-875) unpacks `(keep_indices, merge_plan)` and calls the helper before gather; probe site (~831) uses `keep_indices`.

- [ ] **Step 1: Write failing test** — drive `apply_voxel_dedup_` on a cache with a 3-token same-voxel group, `intra_dedup_mode="merge"`; assert: (a) after dedup, token at rep slot has K = Σ wᵢ·Kᵢ (weighted avg, within 1e-5), (b) token count = unique-voxel count (same as drop mode), (c) non-rep members removed. Use `score_state=None` (production path).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `_apply_intra_merge_`** — scatter weighted contributions into rep slots via `index_add_` over the token dim:
```python
def _apply_intra_merge_(self, b_idx, plan):
    # plan: {rep_indices[G], member_indices[M], member_weights[M], member_rep_map[M]}
    rep = plan["rep_indices"]            # [G]
    memb = plan["member_indices"]         # [M]
    w = plan["member_weights"]            # [M]
    rep_of = plan["member_rep_map"]       # [M] in [0,G) — which rep each member maps to
    H, _, D = self.k.shape
    for kv in (self.k, self.v):
        contrib = kv[b_idx, :, memb, :] * w.view(1, -1, 1)   # [H,M,D] weighted
        acc = torch.zeros(H, rep.shape[0], D, device=kv.device, dtype=kv.dtype)  # [H,G,D]
        acc.index_add_(dim=1, index=rep_of, source=contrib)  # Σ_members w·kv → rep slot
        kv[b_idx, :, rep, :] = acc                           # write merged into rep slots
    # importance / depth_conf: weighted-avg at rep (1D)
    for field in ("importance", "depth_conf"):
        val = getattr(self.metadata, field)[b_idx, memb] * w          # [M]
        acc1 = torch.zeros(rep.shape[0], device=val.device, dtype=val.dtype)
        acc1.index_add_(0, rep_of, val)
        getattr(self.metadata, field)[b_idx, rep] = acc1
    # slot_local_xyz [B,N,3]: UNWEIGHTED MEAN per group (3D, not weighted-avg)
    xyz = self.metadata.slot_local_xyz[b_idx, memb]                   # [M,3]
    acc_xyz = torch.zeros(rep.shape[0], 3, device=xyz.device, dtype=xyz.dtype)
    cnt = torch.zeros(rep.shape[0], device=xyz.device, dtype=xyz.dtype)
    acc_xyz.index_add_(0, rep_of, xyz); cnt.index_add_(0, rep_of, torch.ones_like(w))
    self.metadata.slot_local_xyz[b_idx, rep] = acc_xyz / cnt.unsqueeze(-1).clamp(min=1.0)
    # score_state: skip when None (production); else weighted-avg analogous to importance (2D: [M,Ds])
```
Test pins the math: `self.k[0,:,rep[0],:] ≈ Σ_m w[m]·K0[m]` for the 3-token group.

- [ ] **Step 4: Wire into apply_voxel_dedup_.** B==1 (~810): `policy_keep_indices, merge_plan = self._dedup_single_batch(...)`. Before `_gather_single_batch_(policy_keep_indices)` (~856): `if merge_plan is not None: self._apply_intra_merge_(0, merge_plan)`. Probe (~831): `policy_keep_indices=policy_keep_indices` (unchanged, it's the keep_indices). B>1 (~863-875): unpack `(keep_mask_b, merge_plan_b)` per batch, apply merge per batch before `gather_per_batch_`.

- [ ] **Step 5: Run → PASS.** Also run existing dedup tests + frontend regression. Commit: `feat(cache): apply soft-merge intra dedup before gather (token count unchanged)`

---

## Task 4: Gate 1 + 1b — soft-merge empirical (fire/office/chess/redkitchen @200f, fire/chess @1000f)

**Files:** temp script (reuse `run_legacy_vs_frontend.py` build with `intra_dedup_mode="merge"` — add a `--intra-mode` arg, or a temp variant). Run on GPU 4-7.

- [ ] **Step 1:** Add `--intra-mode {drop,merge}` arg to `tools/run_legacy_vs_frontend.py` (frontend build sets `intra_dedup_mode`). Smoke 1 run to confirm it loads.

- [ ] **Step 2 (G1 @200f):** Run frontend `--intra-mode merge` on fire/office/chess/redkitchen @200f seed0 (4 GPU parallel). Compare to drop-mode baseline (fire 0.0458, office 0.0377, chess 0.0263, redkitchen 0.0138) and legacy (0.0307/0.0298/0.0257/0.0176).
  - **Pass:** fire < 0.035 (improves toward legacy) AND office/chess/redkitchen within ±0.003 of drop-mode.
  - **Fail/stop:** fire doesn't improve, or any of office/chess/redkitchen regresses >0.003 → soft-merge rejected, revisit.

- [ ] **Step 3 (G1b @1000f):** Run `--intra-mode merge` on fire/chess @1000f seed0. Compare to drop-mode 1000f (fire 0.0477, chess 0.0580).
  - **Pass:** regression < 0.01m at 1000f (soft-merge doesn't erode the long-seq win).
  - **Fail/stop:** 1000f regresses ≥0.01 → stop, soft-merge hurts long-seq.

- [ ] **Step 4:** Record results. If both pass → proceed to Task 5. If fail → stop, report, reconsider (e.g., merge only helps short-seq, abandon or tune). Commit the `--intra-mode` arg.

---

## Task 5: Coverage anchoring plumbing (TDD)

**Files:** Modify `src/ovggt/models/ovggt.py` `_build_frontend_keyframe_config` (1196-1223). Test: `tests/test_coverage_anchor_plumbing.py`.

- [ ] **Step 1: Write failing tests** — two levels:
  - (a) Unit: `model._build_frontend_keyframe_config("coverage", 250, 3, 0.2, coverage_monitor_only=False)` → assert `KeyframeSwitchConfig.coverage_monitor_only is False` and `strategy=="coverage"`. (Currently hardcoded True → fails.)
  - (b) Full-chain propagation: assert `inspect.signature(OVGGT.inference)` AND `inspect.signature(OVGGT._inference_frontend)` both include a `coverage_monitor_only` param; then monkeypatch/spy `_build_frontend_keyframe_config` to record the `coverage_monitor_only` it receives when `_inference_frontend(..., coverage_monitor_only=False)` is invoked (mock the aggregator/frames to avoid GPU), asserting `False` is threaded through the 3-hop chain `inference → _inference_frontend → _build_frontend_keyframe_config`. This prevents the failure mode where the inner function is fixed but `inference()` stays hardcoded.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** Add `coverage_monitor_only: bool = True` param to `_build_frontend_keyframe_config`; pass it through to the non-train `KeyframeSwitchConfig(...)` (line 1217-1223) replacing the hardcoded `True`. (Train branch stays True.) Add `coverage_monitor_only: bool = True` kwarg to BOTH `inference()` (541) and `_inference_frontend()` (586), threading it from `inference → _inference_frontend → _build_frontend_keyframe_config`. Test (b) pins all 3 hops.

- [ ] **Step 4: Run → PASS.** Commit: `feat(keyframe): plumb coverage_monitor_only through _build_frontend_keyframe_config (enable coverage in frontend)`

---

## Task 6: Gate 2 — coverage anchoring empirical (office @200f, verify branch fires)

**Files:** `tools/run_legacy_vs_frontend.py` add `--anchor {fixed_interval,coverage}` + `--coverage-monitor-only` flags. GPU 4-7.

- [ ] **Step 1:** Add anchor flags. First run with a temporary `print`/log in `frontend_keyframe.py:248` coverage branch to CONFIRM it fires when `coverage_monitor_only=False` + strategy=coverage.

- [ ] **Step 2 (G2):** Run frontend coverage-anchor on office/chess/fire/redkitchen @200f seed0 (intra=drop to isolate the anchor effect). Compare to fixed_interval baseline (office 0.0377).
  - **Pass:** office improves toward 0.030 AND coverage branch confirmed firing (log) AND no ring/FIFO errors/overflow.
  - **Fail/stop:** coverage branch doesn't fire, or office doesn't improve, or ring/FIFO breaks → coverage anchoring rejected for office; revisit (the office deficit cause is elsewhere).
  - **Note:** if coverage threshold 0.2 over/under-registers anchors, quick sweep {0.2, 0.3, 0.4} on office.

- [ ] **Step 3:** Record. Commit anchor flags + remove temp log.

---

## Task 7: Gate 3 — diagnose + fix 500f redkitchen crash

**Files:** investigation (diagnostic probe `tools/cache_diag_probe.py` + `run_cache_diag.py` on redkitchen 500f). Then targeted fix TBD by diagnosis.

- [ ] **Step 1 (diagnose):** Run `run_cache_diag.py --scene redkitchen/seq-03 --num_frames 500` (frontend, current config). Examine protected_count growth, anchor_overflow_rate, rescued_pool_count. Hypotheses: (a) anchor overflow on budget-poor layers, (b) intra/dedup interaction, (c) coverage-anchor instability, (d) scene-specific geometry.
  - Also compare to chess 500f (which is stable: 0.0528) to isolate what redkitchen does differently.

- [ ] **Step 2 (fix):** Based on diagnosis. Possibilities: raise ring cap for large scenes, fix overflow policy, adjust anchor interval for redkitchen, etc. **Not predetermined** — let diagnosis dictate.

- [ ] **Step 3 (verify):** Re-run redkitchen 500f → target < 0.05 (from 0.1034). If unfixable (deep scene cause), document honestly; 500f mean goal becomes conditional/unmet.

---

## Task 8: Gate 4 — combined config @200/500f

- [ ] **Step 1:** Build combined frontend config: `intra_dedup_mode="merge"` + coverage anchor + redkitchen fix (from G3). Run 4 scenes × {200,500}f seed0.
  - **Pass:** 200f mean < legacy 0.0260; 500f mean < legacy 0.0738 (conditional on G3); chess/redkitchen no regression; no errors.
  - **Fail:** isolate which component regressed; iterate.

---

## Task 9: Gate 5 — full confirmation sweep

- [ ] **Step 1:** 4 scenes × 3 seeds × {200,500,1000}f, combined config, GPU 4-7 (seed-0 deterministic; seeds 1,2 confirm). Reuse `batch_p1_ablation.sh`-style runner.
- [ ] **Step 2:** Paired diff vs legacy (from `tools/legacy_vs_frontend_lyj/`). Confirm 200/500/1000f mean < legacy, no scene regression, 1000f −26% preserved.

---

## Task 10: Apply final config to production call-sites + commit

**Files:** `tools/run_legacy_vs_frontend.py`, `tools/test_multi_scene.py` (and note `eval_*` scripts).

- [ ] **Step 1:** Set production defaults: `intra_dedup_mode="merge"`, coverage anchor (`history_anchor_strategy="coverage"`, `coverage_monitor_only=False`), keep ring0.2 + budget8334 + intra_frame_dedup_enabled=True. Update the explanatory comment in `test_multi_scene.py`.
- [ ] **Step 2:** Re-run `test_multi_scene.py` 4-scene benchmark → confirm frontend (FE8_opt) beats Legacy at 200f (the original deficit scene).
- [ ] **Step 3:** Commit: `feat(frontend): production soft-merge dedup + coverage anchor (surpass legacy at 200/500/1000f)`. Update `docs/legacy_vs_frontend_comparison.md` with new numbers.

---

## Verification

- **Unit (TDD):** `tests/test_soft_merge_intra_dedup.py`, `tests/test_coverage_anchor_plumbing.py`, + existing `tests/test_p1c_fifo_rescued_pool.py` + `tests/test_p2_eviction_weight_decouple.py` all green.
- **Empirical gates:** G1/G1b/G2/G3/G4/G5 each a hard pass/fail checkpoint.
- **End-to-end:** `tools/test_multi_scene.py` 4-scene benchmark shows FE8_opt ≤ Legacy at 200f (the original gap closed).

## Risks (from spec §4)
- soft-merge may blur (G1 decides); coverage may break ring/FIFO or not help office (G2 decides); 500f redkitchen may be unfixable (G3 decides, 500f goal conditional). Each gate is a stop/proceed decision — no component combined until its gate passes.
