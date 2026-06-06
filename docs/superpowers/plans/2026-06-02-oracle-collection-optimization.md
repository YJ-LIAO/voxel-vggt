# Oracle Collection Optimization Implementation Plan

> **Naming migration (2026-06-04):** `frontend_total_budget` has been renamed to `frontend_per_layer_budget` and now represents the per-layer token count (previously it was the 24-layer sum). Numeric values in this doc have been rescaled by //24 where they appear in code snippets. Conceptual discussion of `200000` / `10410` totals is preserved verbatim for historical context.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-sequence oracle collection time from ~4.5h to ~5-10min by capping dedup subset enumeration, lowering per-sequence event count, lowering layers-per-frame, adding stratified event selection, and adding event-type stress profiles — all while preserving shard format and loss semantics.

**Architecture:** Phase 1 (this plan) is sampling-policy changes only — no changes to model code (except `frontend_cache.py` instrumentation), no changes to loss definitions. The instrumentation restructure in `apply_voxel_dedup_()` (Decision 2) is a prerequisite for the sampling cap because the probe needs access to actual dedup decision scores and policy keep sets. Phase 2 (prefix snapshot reuse) and Phase 3 (model vectorization) are out of scope. Phase 4 (multi-GPU production scheduling) is included as deployment tasks.

**Spec:** `docs/superpowers/specs/2026-06-02-oracle-collection-optimization-design.md`

**Tech Stack:** Python 3.11, PyTorch, OmegaConf, OVGGT frontend cache, pytest.

---

## File Map

| File | Phase 1 Role |
|------|--------------|
| `src/ovggt/utils/frontend_cache.py` | **Modify:** restructure `apply_voxel_dedup_()` to expose scores + policy_keep_indices to probe callback (Decision 2 prerequisite) |
| `src/ovggt/training/frontend_oracle_collector.py` | **Modify:** add `_sample_dedup_keep_indices` with policy baseline, update `on_dedup_candidate` signature across all probes, add `select_oracle_events`, add `load_frontend_oracle_config` override, plumb all new config fields, update probe call sites |
| `tools/collect_counterfactual_oracle.py` | **Modify:** add new CLI args for all Phase 1 parameters, pass through |
| `tools/collect_counterfactual_oracle_parallel.py` | **Modify (Phase 4):** forward Phase 1 args, add manifest partition fields |
| `config/collect_counterfactual_oracle.yaml` | **Modify:** update defaults |
| `src/train_token_scorer_oracle.py` | **Modify:** add held-out event/sequence splits, event-type metrics, optional stress-profile weighted sampling |
| `tests/test_frontend_oracle_collector.py` | **Modify:** add unit tests for sampling function, stratified selection, stress profiles, cap behavior |
| `tests/test_oracle_phase1_equivalence.py` | **Create:** end-to-end equivalence test against current collector |
| `tests/test_frontend_oracle_parallel.py` | **Modify:** regression test for Phase 1 arg forwarding |
| `tests/test_token_oracle_dataset.py` | **Modify:** verify shard format compatibility |
| `tools/summarize_oracle_collection.py` | **Create (Phase 4):** log summarizer with shard summary metrics |

---

## Task 1: Restructure `apply_voxel_dedup_()` To Expose Scores And Policy Keep Set

**Spec Reference:** Decision 2 — Required Instrumentation Change To `apply_voxel_dedup_()`

**Rationale:** The current `apply_voxel_dedup_()` in `frontend_cache.py` calls `on_dedup_candidate()` before computing keep indices, so the probe cannot access actual dedup decision scores or the policy keep set. Phase 1's quality gate requires verifying the current-policy baseline is retained for dedup events. Without this restructure, that gate cannot be tested.

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py` (restructure `apply_voxel_dedup_()`, lines ~543-670)
- Test: `tests/test_frontend_cache.py` (or existing test file for `frontend_cache`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_cache.py` (or create if not exists):

```python
import torch
import pytest

from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    TokenKind,
    TokenMetadata,
)


def _make_dedup_cache_state(num_tokens: int = 10) -> LayerCacheState:
    """Build a LayerCacheState where all PATCH tokens share voxel (0, 0, 0).

    This triggers the dedup code path in apply_voxel_dedup_()
    because multiple tokens map to the same voxel hash.
    """
    B = 1
    xyz = [(0.0, 0.0, 0.0)] * num_tokens
    return LayerCacheState(
        k=torch.randn(B, 2, num_tokens, 4),
        v=torch.randn(B, 2, num_tokens, 4),
        score_state=torch.randn(B, num_tokens, 128),
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
    """Verify that apply_voxel_dedup_() computes keep indices BEFORE calling the probe callback,
    and that the callback receives actual scores and policy_keep_indices."""

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
        not a re-derived set. The probe must not change what gets gathered."""
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
        # After apply_voxel_dedup_, the number of kept tokens must match
        # what the policy_keep_indices said.
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/mount/lyj/voxel-vggt
PYTHONPATH=src pytest tests/test_frontend_cache.py::TestApplyVoxelDedupProbeCallbackOrdering -v
```

Expected: AssertionError — old code invokes the probe but does not pass
`scores` or `policy_keep_indices` (for example: "scores was not passed to callback").

- [ ] **Step 3: Restructure `apply_voxel_dedup_()` in `frontend_cache.py`**

The current order in `apply_voxel_dedup_()` (lines ~543-670) is:

1. Compute `scores` (lines 599-612)
2. Call `dedup_probe.on_dedup_candidate(cache_state, layer_id, frame_id, batch_index)` (lines 615-621)
3. Compute keep indices via `_dedup_single_batch(...)` (lines 644+)
4. Apply gather (`self.gather_(kept_indices)`)

Restructure to:

1. Compute `scores` (unchanged)
2. Compute `policy_keep_indices` via `_dedup_single_batch(...)` — **hoisted** from lines 644+
3. Call `dedup_probe.on_dedup_candidate(cache_state, layer_id, frame_id, batch_index, scores=scores[batch_index], policy_keep_indices=policy_keep_indices)` — **updated signature**
4. Handle `dedup_replay_probe` path (lines 624-642) — **unchanged in behavior**, re-derive keep from replay set
5. Apply gather (`self.gather_(policy_keep_indices)`)

Key implementation details:
- `_dedup_single_batch(...)` call is the same, just moved earlier.
- `scores` passed to the callback is `scores[batch_index]` — a 1D tensor of per-token decision scores.
- `policy_keep_indices` is the full cache keep set the policy is about to apply.
- The `dedup_replay_probe` path (lines 624-642) still runs **before** the gather and is unchanged in behavior — the replay keep decision is re-derived from the replay keep set, not from the policy keep set.

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_cache.py::TestApplyVoxelDedupProbeCallbackOrdering -v
```

Expected: 2 passed.

- [ ] **Step 5: Run existing frontend_cache tests to verify no regression**

```bash
PYTHONPATH=src pytest tests/test_frontend_cache.py -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_frontend_cache.py
git commit -m "refactor(frontend-cache): restructure apply_voxel_dedup_ to expose scores and policy_keep_indices to probe callback

Decision 2 instrumentation change. The probe now receives actual dedup
decision scores and the policy keep set BEFORE the gather is applied.
This is a prerequisite for Phase 1's current-policy baseline quality gate."
```

---

## Task 2: Update `on_dedup_candidate` Signature Across All Probes

**Spec Reference:** Decision 2 — files changed by instrumentation

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (`CounterfactualDedupProbe`, `ReplayDedupKeepSetProbe`, `MultiReplayDedupKeepSetProbe`)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
class TestProbeCallbackSignatureCompatibility:
    """All probes must accept the new optional args without error."""

    def test_dedup_probe_accepts_scores_and_policy_keep(self):
        """CounterfactualDedupProbe.on_dedup_candidate must accept scores and policy_keep_indices."""
        from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe
        probe = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42)
        # Build minimal cache state (reuse _dedup_cache_state from Task 3)
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        # Must not raise TypeError
        probe.on_dedup_candidate(
            cache_state=state,
            layer_id=0,
            frame_id=5,
            batch_index=0,
            scores=torch.randn(5),
            policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )

    def test_replay_probe_accepts_scores_and_policy_keep(self):
        """ReplayDedupKeepSetProbe.on_dedup_candidate must accept and ignore new args."""
        from ovggt.training.frontend_oracle_collector import ReplayDedupKeepSetProbe
        target_event = {"layer_id": 0, "frame_id": 5, "batch_index": 0}
        probe = ReplayDedupKeepSetProbe(target_event=target_event, keep_indices=torch.tensor([0, 1, 2]))
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        result = probe.on_dedup_candidate(
            cache_state=state,
            layer_id=0,
            frame_id=5,
            batch_index=0,
            scores=torch.randn(5),
            policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )
        assert torch.equal(result.cpu(), torch.tensor([0, 1, 2]))

    def test_multi_replay_probe_accepts_scores_and_policy_keep(self):
        """MultiReplayDedupKeepSetProbe.on_dedup_candidate must accept and ignore new args."""
        from ovggt.training.frontend_oracle_collector import MultiReplayDedupKeepSetProbe
        target_event = {"layer_id": 0, "frame_id": 5, "batch_index": 0}
        probe = MultiReplayDedupKeepSetProbe(
            target_event=target_event,
            keep_indices_batch=[torch.tensor([0, 1, 2])],
        )
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        result = probe.on_dedup_candidate(
            cache_state=state,
            layer_id=0,
            frame_id=5,
            batch_index=0,
            scores=torch.randn(5),
            policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )
        assert torch.equal(result.cpu(), torch.tensor([0, 1, 2]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestProbeCallbackSignatureCompatibility -v
```

Expected: TypeError — `on_dedup_candidate()` got unexpected keyword arguments `scores` / `policy_keep_indices`.

- [ ] **Step 3: Update all three probe classes**

In `src/ovggt/training/frontend_oracle_collector.py`:

**CounterfactualDedupProbe.on_dedup_candidate:**
```python
def on_dedup_candidate(
    self,
    cache_state,
    layer_id: int,
    frame_id: int,
    batch_index: int = 0,
    scores: torch.Tensor | None = None,          # NEW: [num_tokens]
    policy_keep_indices: torch.Tensor | None = None,  # NEW: [num_kept]
) -> None:
```
Store `self._last_scores = scores` and `self._last_policy_keep_indices = policy_keep_indices` for use in Task 4.

**ReplayDedupKeepSetProbe.on_dedup_candidate:**
```python
def on_dedup_candidate(
    self,
    cache_state,
    layer_id: int,
    frame_id: int,
    batch_index: int = 0,
    scores: torch.Tensor | None = None,          # Accept and ignore
    policy_keep_indices: torch.Tensor | None = None,  # Accept and ignore
) -> None:
```

**MultiReplayDedupKeepSetProbe.on_dedup_candidate:**
```python
def on_dedup_candidate(
    self,
    cache_state,
    layer_id: int,
    frame_id: int,
    batch_index: int = 0,
    scores: torch.Tensor | None = None,          # Accept and ignore
    policy_keep_indices: torch.Tensor | None = None,  # Accept and ignore
) -> None:
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestProbeCallbackSignatureCompatibility -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): update on_dedup_candidate signature to accept scores and policy_keep_indices

CounterfactualDedupProbe stores them for later sampling use.
ReplayDedupKeepSetProbe and MultiReplayDedupKeepSetProbe accept and ignore.
Matches Decision 2 instrumentation change from the spec."
```

---

## Task 3: Add `_sample_dedup_keep_indices` Helper With Policy Baseline

**Spec Reference:** Decision 2 — Subset Sampling Strategy

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (new function near existing helpers)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
from ovggt.training.frontend_oracle_collector import _sample_dedup_keep_indices


class TestSampleDedupKeepIndices:
    def test_returns_all_indices_when_n_le_cap(self):
        group_indices = torch.tensor([3, 7, 11, 15])
        dedup_scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
        gen = torch.Generator().manual_seed(0)
        result, baseline = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=8, generator=gen,
        )
        assert sorted(result) == [3, 7, 11, 15]
        assert baseline is None  # no policy baseline when N <= cap

    def test_caps_at_requested_size_when_n_gt_cap(self):
        group_indices = torch.arange(50)
        dedup_scores = torch.rand(50)
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=8, generator=gen,
        )
        assert len(result) <= 8
        assert len(result) >= 4  # at least top + bottom
        for idx in result:
            assert 0 <= idx < 50

    def test_includes_extreme_scores(self):
        group_indices = torch.arange(20)
        dedup_scores = torch.arange(20, dtype=torch.float)
        gen = torch.Generator().manual_seed(0)
        result, _ = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=4, generator=gen,
        )
        assert 19 in result  # highest score
        assert 0 in result   # lowest score

    def test_deterministic_with_same_seed(self):
        group_indices = torch.arange(100)
        dedup_scores = torch.rand(100)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        r1, _ = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen1)
        r2, _ = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen2)
        assert r1 == r2

    def test_policy_baseline_single_token_in_group(self):
        """When policy keeps exactly one token in the group, that token must be included."""
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        # Policy keeps token 5 (which is in the group)
        policy_keep = torch.tensor([5, 50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=8, generator=gen,
            policy_keep_indices=policy_keep,
        )
        assert 5 in result  # the single policy-kept token must be included
        assert baseline is None  # single token → no full-cache baseline

    def test_policy_baseline_multiple_tokens_in_group(self):
        """When policy keeps multiple tokens in the group, emit a full-cache policy baseline."""
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        # Policy keeps tokens 3, 7, 11 inside the group
        policy_keep = torch.tensor([3, 7, 11, 50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=8, generator=gen,
            policy_keep_indices=policy_keep,
        )
        # The policy baseline is the full policy_keep tensor
        assert baseline is not None
        assert torch.equal(baseline, policy_keep.long())
        # Total keep-one subsets + baseline must be <= cap
        assert len(result) + (1 if baseline is not None else 0) <= 8

    def test_policy_baseline_zero_tokens_in_group_is_skipped(self):
        """When policy keeps zero tokens in the group, the function returns empty."""
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        # Policy keeps only tokens outside the group
        policy_keep = torch.tensor([50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(
            group_indices, dedup_scores, cap=8, generator=gen,
            policy_keep_indices=policy_keep,
        )
        assert result == []
        assert baseline is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestSampleDedupKeepIndices -v
```

Expected: ImportError or AttributeError on `_sample_dedup_keep_indices`.

- [ ] **Step 3: Write the implementation**

Add to `src/ovggt/training/frontend_oracle_collector.py` after the `deduplicate_keep_subsets` function (around line 622):

```python
def _sample_dedup_keep_indices(
    group_indices: torch.Tensor,
    dedup_scores: torch.Tensor,
    cap: int,
    generator: torch.Generator,
    policy_keep_indices: torch.Tensor | None = None,
) -> tuple[list[int], torch.Tensor | None]:
    """Sample up to `cap` keep-one subset anchors from a voxel group.

    Strategy when N > cap:
      1. Derive policy-kept tokens inside the group.
         - Zero tokens in group → skip (return empty, no baseline).
         - One token → include it as the policy keep-one choice.
         - Multiple tokens → emit one full-cache policy_baseline subset.
      2. Top cap//4 by dedup_scores (highest).
      3. Bottom cap//4 by dedup_scores (lowest).
      4. Rest: uniform random from remaining indices.

    Returns:
      (chosen_token_indices, policy_baseline_or_None)
      - chosen_token_indices: list of global token indices to keep as keep-one choices.
      - policy_baseline: full-cache policy_keep_indices tensor if multiple policy-kept
        tokens exist in the group, else None.

    dedup_scores must be group-local: shape [N], same order as group_indices.
    Deterministic for a fixed `generator` seed.
    """
    cap = int(cap)
    n = int(group_indices.numel())
    if cap <= 0 or n == 0:
        return [], None
    if n <= cap:
        return [int(x) for x in group_indices.tolist()], None

    # --- Policy baseline derivation ---
    chosen: set[int] = set()
    policy_baseline: torch.Tensor | None = None

    if policy_keep_indices is not None:
        policy_set = {int(idx) for idx in policy_keep_indices.reshape(-1).tolist()}
        policy_local = [
            local_idx
            for local_idx, token_idx in enumerate(group_indices.tolist())
            if int(token_idx) in policy_set
        ]
        if len(policy_local) == 0:
            return [], None
        if len(policy_local) == 1:
            chosen.add(int(policy_local[0]))
        else:
            # Multiple policy-kept tokens: emit full-cache baseline
            policy_baseline = policy_keep_indices.reshape(-1).detach().cpu().long()

    # --- Score-based sampling ---
    head = max(cap // 4, 1)
    tail = max(cap // 4, 1)

    _, top_idx = torch.topk(dedup_scores, k=min(head, n))
    _, bot_idx = torch.topk(dedup_scores, k=min(tail, n), largest=False)
    chosen.update(int(i) for i in top_idx.tolist())
    chosen.update(int(i) for i in bot_idx.tolist())

    reserved = 1 if policy_baseline is not None else 0
    remaining_slots = max(cap - reserved - len(chosen), 0)
    remaining = [i for i in range(n) if i not in chosen]
    if remaining and remaining_slots > 0:
        perm = torch.randperm(len(remaining), generator=generator)[:remaining_slots]
        chosen.update(remaining[i] for i in perm.tolist())

    return [int(group_indices[i].item()) for i in sorted(chosen)], policy_baseline
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestSampleDedupKeepIndices -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add _sample_dedup_keep_indices with policy baseline support

Implements Decision 2 subset sampling strategy with:
- Top/bottom score extremes + uniform random
- Policy baseline for zero/one/multiple policy-kept tokens
- Deterministic sampling via torch.Generator"
```

---

## Task 4: Wire Sampling Into `CounterfactualDedupProbe` Using Actual Dedup Scores

**Spec Reference:** Decision 2 — wiring into on_dedup_candidate

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (`CounterfactualDedupProbe.__init__` and `on_dedup_candidate`)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe
from ovggt.utils.frontend_cache import LayerCacheState, TokenKind, TokenMetadata


def _dedup_metadata(num_tokens, frame_id=5, importance=None, xyz_for_tokens=None):
    """Build TokenMetadata where all tokens share the same voxel position."""
    if importance is None:
        importance = [0.5] * num_tokens
    if xyz_for_tokens is None:
        xyz_for_tokens = [(0.0, 0.0, 0.0)] * num_tokens
    return TokenMetadata(
        token_kind=torch.tensor([[int(TokenKind.PATCH)] * num_tokens], dtype=torch.long),
        frame_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        anchor_slot=torch.tensor([[0] * num_tokens], dtype=torch.long),
        keyframe_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        slot_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        slot_local_xyz=torch.tensor([list(xyz_for_tokens)], dtype=torch.float32),
        importance=torch.tensor([importance], dtype=torch.float32),
        depth_conf=torch.tensor([[0.5] * num_tokens], dtype=torch.float32),
    )


def _dedup_cache_state(num_tokens, importance, xyz_for_tokens=None):
    """Build a real LayerCacheState with all tokens in the same voxel group."""
    return LayerCacheState(
        k=torch.randn(1, 2, num_tokens, 4),
        v=torch.randn(1, 2, num_tokens, 4),
        score_state=torch.arange(num_tokens * 4, dtype=torch.float32).reshape(1, num_tokens, 4),
        metadata=_dedup_metadata(num_tokens, importance=importance, xyz_for_tokens=xyz_for_tokens),
        protected_count=0,
    )


class TestDedupProbeSubsetCap:
    def test_probe_caps_subsets_when_voxel_group_exceeds_cap(self):
        importance = [float(i) / 50.0 for i in range(50)]
        state = _dedup_cache_state(num_tokens=50, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        # Pass actual scores (simulating what apply_voxel_dedup_ now provides)
        scores = torch.tensor(importance)
        policy_keep = torch.tensor([0, 1, 2, 3, 4])  # policy keeps first 5 tokens
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=scores, policy_keep_indices=policy_keep,
        )
        assert len(probe.events) == 1
        event = probe.events[0]
        assert len(event["candidate_subsets"]) <= 8
        assert any(
            subset.get("source") == "policy_baseline"
            for subset in event["candidate_subsets"]
        )

    def test_probe_returns_full_enumeration_below_cap(self):
        importance = [0.1, 0.2, 0.3, 0.4, 0.5]
        state = _dedup_cache_state(num_tokens=5, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=torch.tensor(importance), policy_keep_indices=torch.arange(5),
        )
        assert len(probe.events) == 1
        assert len(probe.events[0]["candidate_subsets"]) == 5

    def test_probe_uses_actual_scores_not_importance(self):
        """The sampling must use the scores tensor, not metadata.importance."""
        importance = [0.5] * 50  # uniform importance
        actual_scores = torch.arange(50, dtype=torch.float)  # but real scores are ordered
        state = _dedup_cache_state(num_tokens=50, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=actual_scores, policy_keep_indices=torch.tensor([0]),
        )
        # With actual_scores = arange(50), top-k must include token 49
        subsets = [
            subset for subset in probe.events[0]["candidate_subsets"]
            if subset.get("source") != "policy_baseline"
        ]
        keep_indices_sets = [set(s["keep_indices"].tolist()) for s in subsets]
        # Token 49 (highest score) must appear as a keep-one choice
        assert any(49 in kis for kis in keep_indices_sets)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestDedupProbeSubsetCap -v
```

Expected: TypeError — `CounterfactualDedupProbe.__init__()` got unexpected keyword argument `max_subsets_per_dedup_event`, or the sampling doesn't use `scores`.

- [ ] **Step 3: Update `CounterfactualDedupProbe.__init__` to accept the cap**

```python
class CounterfactualDedupProbe:
    def __init__(
        self,
        num_samples: int = 8,
        oracle_window: int = 4,
        seed: int = 0,
        event_prefix: str = "dedup",
        max_events: int | None = None,
        sequence_provenance: dict[int, dict] | None = None,
        layers_per_frame: int = 0,
        num_layers: int | None = None,
        voxel_size: float = 0.25,
        max_subsets_per_dedup_event: int = 8,   # NEW
    ) -> None:
        # ... existing fields ...
        self.max_subsets_per_dedup_event = int(max_subsets_per_dedup_event)
```

- [ ] **Step 4: Use `_sample_dedup_keep_indices` inside `on_dedup_candidate` with actual scores**

Find the inner loop (around line 346-356) that enumerates all keep-one subsets. Replace with:

```python
# Use actual dedup scores passed from apply_voxel_dedup_ (not metadata.importance)
group_scores = scores[group_indices.tolist()] if scores is not None else base_scores[group_indices.tolist()]
generator = torch.Generator().manual_seed(self.seed + len(self.events))

keep_local_indices, policy_baseline = _sample_dedup_keep_indices(
    group_indices,
    dedup_scores=group_scores,
    cap=self.max_subsets_per_dedup_event,
    generator=generator,
    policy_keep_indices=policy_keep_indices,
)

if not keep_local_indices and policy_baseline is None:
    # Zero policy-kept tokens in group → skip
    continue

candidate_subsets = []
if policy_baseline is not None:
    # IMPORTANT: policy baseline must flow through candidate_subsets so
    # measure_counterfactual_event() replays it and stores it in final subsets.
    candidate_subsets.append({
        "source": "policy_baseline",
        "keep_indices": policy_baseline,
    })

for keep_idx in keep_local_indices:
    evict_indices = [idx for idx in group_indices.tolist() if idx != keep_idx]
    candidate_subsets.append({
        "source": "keep_one",
        "keep_index": keep_idx,
        "evict_indices": evict_indices,
        "keep_indices": torch.tensor(
            [idx for idx in range(num_tokens) if idx not in evict_indices],
            dtype=torch.long,
        ),
    })
```

- [ ] **Step 5: Plumb the cap from `collect_oracle_events_from_sequence`**

Find the `CounterfactualDedupProbe(...)` construction (around line 1092-1102) and pass `max_subsets_per_dedup_event=collector_cfg.max_subsets_per_dedup_event`.

- [ ] **Step 6: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestDedupProbeSubsetCap -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): cap dedup subsets via sampling using actual dedup scores

Wire _sample_dedup_keep_indices into CounterfactualDedupProbe using the
scores and policy_keep_indices now provided by apply_voxel_dedup_.
Policy baseline is stored inside candidate_subsets so replay measures it
and final shards expose it through subsets[]."
```

---

## Task 5: Add Stratified Event Selection (`select_oracle_events`)

**Spec Reference:** Decision 3 — Phase 1 Uses Stratified Event Selection

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (new function + plumbing)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
from ovggt.training.frontend_oracle_collector import select_oracle_events


class TestSelectOracleEvents:
    def _make_events(self, specs: list[tuple[str, int, int, int]]) -> list[dict]:
        """Build candidate events from (event_type, frame_id, layer_id, voxel_group_id) tuples."""
        return [
            {"event_type": et, "frame_id": fid, "layer_id": lid, "voxel_group_id": vgid}
            for et, fid, lid, vgid in specs
        ]

    def test_first_n_policy_takes_first_n(self):
        events = self._make_events([("dedup", f, 0, 0) for f in range(20)])
        selected = select_oracle_events(events, max_events=5, max_events_per_frame=100, policy="first_n")
        assert len(selected) == 5
        assert [e["frame_id"] for e in selected] == [0, 1, 2, 3, 4]

    def test_stratified_round_robin_spreads_across_frames(self):
        """When first 16 candidates are all frame 0, stratified selection must pick later frames
        if the candidate pool contains them."""
        # 20 events on frame 0, then 20 on frame 1, 20 on frame 2
        events = []
        for frame in range(3):
            for _ in range(20):
                events.append({"event_type": "dedup", "frame_id": frame, "layer_id": 0, "voxel_group_id": 0})
        selected = select_oracle_events(
            events, max_events=6, max_events_per_frame=6, policy="stratified_round_robin",
        )
        frames_selected = set(e["frame_id"] for e in selected)
        # Must include at least 2 different frames
        assert len(frames_selected) >= 2

    def test_max_events_per_frame_is_respected(self):
        events = [{"event_type": "dedup", "frame_id": 0, "layer_id": i, "voxel_group_id": 0} for i in range(20)]
        selected = select_oracle_events(
            events, max_events=10, max_events_per_frame=3, policy="stratified_round_robin",
        )
        frame_counts = {}
        for e in selected:
            frame_counts[e["frame_id"]] = frame_counts.get(e["frame_id"], 0) + 1
        assert frame_counts.get(0, 0) <= 3

    def test_deterministic_on_same_input(self):
        events = self._make_events([
            ("dedup", i % 5, i, i) for i in range(50)
        ])
        s1 = select_oracle_events(events, max_events=10, max_events_per_frame=6, policy="stratified_round_robin")
        s2 = select_oracle_events(events, max_events=10, max_events_per_frame=6, policy="stratified_round_robin")
        assert s1 == s2

    def test_groups_by_event_type_frame_layer_and_voxel(self):
        """Events from different event types, frames, layers, and voxel groups must each get a turn."""
        events = []
        for et in ["dedup", "eviction"]:
            for frame in [0, 1]:
                for layer in [0, 12]:
                    events.append({"event_type": et, "frame_id": frame, "layer_id": layer, "voxel_group_id": 0})
        # 8 events total (2 types × 2 frames × 2 layers)
        selected = select_oracle_events(
            events, max_events=8, max_events_per_frame=8, policy="stratified_round_robin",
        )
        assert len(selected) == 8

    def test_layer_bucket_groups_correctly(self):
        """layer_id // 6 should produce 4 buckets for 24 layers. Events from different
        buckets must be spread across selections."""
        events = [
            {"event_type": "dedup", "frame_id": 0, "layer_id": 0, "voxel_group_id": 0},   # bucket 0
            {"event_type": "dedup", "frame_id": 0, "layer_id": 6, "voxel_group_id": 0},   # bucket 1
            {"event_type": "dedup", "frame_id": 0, "layer_id": 12, "voxel_group_id": 0},  # bucket 2
            {"event_type": "dedup", "frame_id": 0, "layer_id": 18, "voxel_group_id": 0},  # bucket 3
        ]
        selected = select_oracle_events(
            events, max_events=4, max_events_per_frame=4, policy="stratified_round_robin",
        )
        assert len(selected) == 4
        layers = [e["layer_id"] for e in selected]
        assert set(layers) == {0, 6, 12, 18}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestSelectOracleEvents -v
```

Expected: ImportError or AttributeError on `select_oracle_events`.

- [ ] **Step 3: Implement `select_oracle_events` and helper**

Add to `src/ovggt/training/frontend_oracle_collector.py`:

```python
def _group_candidates_for_stratified_selection(
    candidate_events: list[dict],
    layer_bucket_width: int = 6,
) -> list[list[dict]]:
    """Group candidate events by (event_type, frame_id, layer_bucket, voxel_group_id) for round-robin."""
    groups: dict[tuple, list[dict]] = {}
    for event in candidate_events:
        et = event.get("event_type", "unknown")
        fid = int(event.get("frame_id", 0))
        lid = int(event.get("layer_id", 0))
        bucket = lid // layer_bucket_width
        vgid = event.get("voxel_group_id", event.get("demoted_slot", id(event)))
        key = (et, fid, bucket, vgid)
        groups.setdefault(key, []).append(event)
    # Return groups sorted by key for determinism
    return [groups[k] for k in sorted(groups.keys())]


def select_oracle_events(
    candidate_events: list[dict],
    max_events: int,
    max_events_per_frame: int,
    policy: str = "stratified_round_robin",
    layer_bucket_width: int = 6,
) -> list[dict]:
    """Select up to max_events from candidate_events according to the given policy.

    Args:
        candidate_events: post-probe candidate pool (capped by max_candidate_events_per_sequence).
        max_events: max events to measure per sequence (default 16).
        max_events_per_frame: prevents a single frame from consuming the entire sequence quota.
        policy: "first_n" (old behavior) or "stratified_round_robin" (Phase 1 default).
        layer_bucket_width: width of layer_id // width for stratified grouping.
    """
    if policy == "first_n":
        return candidate_events[:max_events]

    groups = _group_candidates_for_stratified_selection(candidate_events, layer_bucket_width)
    selected = []
    per_frame_counts: dict[int, int] = {}

    # Round-robin: iterate groups in order, pick one from each, repeat
    group_queues = [list(g) for g in groups]  # copy to consume
    while group_queues and len(selected) < max_events:
        next_round = []
        for queue in group_queues:
            if not queue:
                continue
            if len(selected) >= max_events:
                break
            event = queue.pop(0)
            frame_id = int(event.get("frame_id", 0))
            if per_frame_counts.get(frame_id, 0) >= max_events_per_frame:
                # Skip this event, but keep the queue for next round
                next_round.append(queue)
                continue
            selected.append(event)
            per_frame_counts[frame_id] = per_frame_counts.get(frame_id, 0) + 1
            if queue:
                next_round.append(queue)
        group_queues = next_round

    return selected
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestSelectOracleEvents -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add stratified round-robin event selection (Decision 3)

select_oracle_events groups candidates by event_type, frame_id, layer
bucket, and voxel group, then round-robines across groups. This prevents
early frames or one large voxel group from consuming the entire quota."
```

---

## Task 6: Wire Stratified Selection Into Collection Pipeline

**Spec Reference:** Decision 3 — probe call sites must use `max_candidate_events_per_sequence`

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (update probe construction and event selection)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
class TestCollectorUsesCandidateCap:
    """Verify that probes are constructed with max_candidate_events_per_sequence,
    not max_events_per_sequence, and that select_oracle_events applies the smaller cap."""

    def test_collect_sequence_uses_candidate_cap_before_selection(self, monkeypatch):
        """When max_candidate_events_per_sequence=256 and max_events_per_sequence=16,
        probes must capture 256 candidates, then stratified selection measures <=16."""
        import torch
        import ovggt.training.frontend_oracle_collector as collector

        constructed_max_events = {}
        measured_events = []

        def make_fake_probe(name):
            class FakeProbe:
                def __init__(self, *args, max_events=None, **kwargs):
                    # Accept and ignore all constructor kwargs (num_samples, oracle_window, etc.)
                    self.events = []
                    constructed_max_events[name] = max_events
            return FakeProbe

        def fake_event(event_type, frame_id, idx):
            return {
                "event_id": f"{event_type}_{frame_id}_{idx}",
                "event_type": event_type,
                "frame_id": frame_id,
                "layer_id": idx % 24,
                "voxel_group_id": idx % 8,
                "candidate_subsets": [
                    {"keep_indices": torch.tensor([0])},
                    {"keep_indices": torch.tensor([1])},
                ],
            }

        def fake_run_frontend(model, frames, probe, cache_results=False, dedup_probe=None, fifo_probe=None):
            # Old first-N slicing would measure only frame 0 eviction events here.
            probe.events.extend(fake_event("eviction", 0, i) for i in range(40))
            dedup_probe.events.extend(fake_event("dedup", 1, i) for i in range(40))
            fifo_probe.events.extend(fake_event("fifo_topk", 2, i) for i in range(40))

        def fake_measure_counterfactual_event(*args, event, **kwargs):
            measured_events.append(event)
            measured = dict(event)
            measured["subsets"] = [
                {"keep_indices": torch.tensor([0]), "loss": 0.0},
                {"keep_indices": torch.tensor([1]), "loss": 1.0},
            ]
            return measured

        monkeypatch.setattr(collector, "CounterfactualEvictionProbe", make_fake_probe("eviction"))
        monkeypatch.setattr(collector, "CounterfactualDedupProbe", make_fake_probe("dedup"))
        monkeypatch.setattr(collector, "CounterfactualFifoTopKProbe", make_fake_probe("fifo_topk"))
        monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)
        monkeypatch.setattr(collector, "measure_counterfactual_event", fake_measure_counterfactual_event)

        result = collector.collect_oracle_events_from_sequence(
            model=object(),
            frames=[{}, {}, {}, {}],
            device=torch.device("cpu"),
            max_events=16,
            num_samples=8,
            oracle_window=1,
            seed=0,
            max_candidate_events_per_sequence=256,
            max_events_per_frame=6,
            event_selection_policy="stratified_round_robin",
            stratified_layer_bucket_width=6,
        )

        assert constructed_max_events == {
            "eviction": 256,
            "dedup": 256,
            "fifo_topk": 256,
        }
        assert len(result) <= 16
        assert len(measured_events) == len(result)
        selected_frames = {int(e["frame_id"]) for e in result}
        assert max(selected_frames) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestCollectorUsesCandidateCap -v
```

Expected: AttributeError on `max_candidate_events_per_sequence`.

- [ ] **Step 3: Update function args, probe call sites, and loader/config pass-through**

Add these args to `collect_oracle_events_from_sequence`:

```python
max_candidate_events_per_sequence: int | None = None,
max_events_per_frame: int = 6,
event_selection_policy: str = "stratified_round_robin",
stratified_layer_bucket_width: int = 6,
```

Also pass them through from `collect_oracle_events_from_loader(...)` and
`collect_oracle_shard_from_config(...)` using fields on `FrontendOracleCollectorConfig`.

In `collect_oracle_events_from_sequence` (or equivalent function), find the three probe constructions at lines ~1171, 1181, 1192 that pass `max_events=self.max_events`. Change each to:

```python
candidate_cap = int(max_candidate_events_per_sequence or max_events)
# ...
max_events=candidate_cap
```

Then, after all probes have collected candidate events, remove any direct
`candidate_events[:max_events]` slicing and apply `select_oracle_events`:

```python
all_candidates = dedup_probe.events + eviction_probe.events + fifo_probe.events
selected_events = select_oracle_events(
    all_candidates,
    max_events=max_events,
    max_events_per_frame=max_events_per_frame,
    policy=event_selection_policy,
    layer_bucket_width=stratified_layer_bucket_width,
)
```

Measure `selected_events`, not `all_candidates`. This is the behavior the test
must prove: probes see the larger candidate cap; replay only sees the smaller
measured event cap.

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestCollectorUsesCandidateCap -v
```

Expected: 1 passed. If this test passes without executing
`collect_oracle_events_from_sequence`, it is too weak and must be rewritten.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): separate candidate cap from measured event cap, wire stratified selection

Probes now capture up to max_candidate_events_per_sequence (256) candidates.
select_oracle_events applies the smaller max_events_per_sequence (16) cap
with stratified_round_robin policy for frame/layer/event-type diversity."
```

---

## Task 7: Add Event-Type Stress Profiles

**Spec Reference:** Decision 5 — Add Event-Type Stress Profiles

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (`load_frontend_oracle_config` + config fields)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
from omegaconf import OmegaConf
import tempfile, yaml


class TestStressProfileOverrides:
    def _write_yaml(self, content: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(content, f)
        f.close()
        return f.name

    def test_low_budget_eviction_overrides_total_budget(self):
        """oracle_profile=low_budget_eviction must lower frontend_per_layer_budget."""
        from ovggt.training.frontend_oracle_collector import load_frontend_oracle_config
        path = self._write_yaml({
            "frontend_per_layer_budget": 8000,
            "num_views": 4,
            "frontend_cache": {"fifo_keep_topk": 0},
        })
        collector_cfg = OmegaConf.create({
            "oracle_profile": "low_budget_eviction",
            "frontend_per_layer_budget_override": 833,
            "fifo_keep_topk_override": None,
        })
        cfg = load_frontend_oracle_config(path, collector_cfg=collector_cfg)
        assert cfg.frontend_per_layer_budget == 833

    def test_fifo_topk_overrides_fifo_keep_topk(self):
        """oracle_profile=fifo_topk must enable fifo_keep_topk."""
        from ovggt.training.frontend_oracle_collector import load_frontend_oracle_config
        path = self._write_yaml({
            "frontend_per_layer_budget": 8000,
            "num_views": 4,
            "frontend_cache": {"fifo_keep_topk": 0},
        })
        collector_cfg = OmegaConf.create({
            "oracle_profile": "fifo_topk",
            "frontend_per_layer_budget_override": None,
            "fifo_keep_topk_override": 8,
        })
        cfg = load_frontend_oracle_config(path, collector_cfg=collector_cfg)
        assert cfg.frontend_cache.fifo_keep_topk == 8

    def test_real_policy_applies_no_overrides(self):
        """oracle_profile=real_policy must not change any config values."""
        from ovggt.training.frontend_oracle_collector import load_frontend_oracle_config
        path = self._write_yaml({
            "frontend_per_layer_budget": 8000,
            "num_views": 4,
            "frontend_cache": {"fifo_keep_topk": 0},
        })
        collector_cfg = OmegaConf.create({
            "oracle_profile": "real_policy",
            "frontend_per_layer_budget_override": None,
            "fifo_keep_topk_override": None,
        })
        cfg = load_frontend_oracle_config(path, collector_cfg=collector_cfg)
        assert cfg.frontend_per_layer_budget == 8000
        assert cfg.frontend_cache.fifo_keep_topk == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestStressProfileOverrides -v
```

Expected: ImportError on `load_frontend_oracle_config` or AttributeError.

- [ ] **Step 3: Implement `load_frontend_oracle_config` with override logic**

Add or update in `src/ovggt/training/frontend_oracle_collector.py`:

```python
def load_frontend_oracle_config(config_path, num_views=None, collector_cfg=None):
    cfg = OmegaConf.load(config_path)
    if collector_cfg is not None:
        if collector_cfg.get("oracle_profile", "real_policy") == "low_budget_eviction":
            override = collector_cfg.get("frontend_per_layer_budget_override")
            if override is not None:
                cfg.frontend_per_layer_budget = int(override)
        elif collector_cfg.get("oracle_profile", "real_policy") == "fifo_topk":
            override = collector_cfg.get("fifo_keep_topk_override")
            if override is not None:
                if not hasattr(cfg, "frontend_cache"):
                    cfg.frontend_cache = {}
                cfg.frontend_cache.fifo_keep_topk = int(override)
    if num_views is not None:
        cfg.num_views = int(num_views)
    OmegaConf.resolve(cfg)
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestStressProfileOverrides -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add stress profile overrides in load_frontend_oracle_config

low_budget_eviction overrides frontend_per_layer_budget. fifo_topk overrides
frontend_cache.fifo_keep_topk. real_policy is identity. Overrides applied
before OmegaConf.resolve() so they participate in interpolation."
```

---

## Task 8: Add All Phase 1 Config Fields, CLI Args, And Update YAML

**Spec Reference:** New Parameters section + Recommended Production Defaults

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (`FrontendOracleCollectorConfig`)
- Modify: `tools/collect_counterfactual_oracle.py` (CLI args)
- Modify: `config/collect_counterfactual_oracle.yaml` (defaults)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
from ovggt.training.frontend_oracle_collector import FrontendOracleCollectorConfig


class TestCollectorConfigDefaults:
    def test_all_phase1_defaults(self):
        cfg = FrontendOracleCollectorConfig(config="dummy", output="/tmp/dummy.pt")
        assert cfg.max_subsets_per_dedup_event == 8
        assert cfg.max_subsets_per_eviction_event == 8
        assert cfg.max_subsets_per_fifo_event == 8
        assert cfg.max_candidate_events_per_sequence == 256
        assert cfg.max_events_per_sequence == 16
        assert cfg.max_events_per_frame == 6
        assert cfg.layers_per_frame == 2
        assert cfg.event_selection_policy == "stratified_round_robin"
        assert cfg.stratified_layer_bucket_width == 6
        assert cfg.oracle_profile == "real_policy"
        assert cfg.frontend_per_layer_budget_override is None
        assert cfg.fifo_keep_topk_override is None
        assert cfg.sequence_manifest_path is None
        assert cfg.sequence_partition_policy == "hash_mod"
        assert cfg.num_sequence_shards is None
        assert cfg.sequence_shard_id is None
        assert cfg.store_replay_payload is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestCollectorConfigDefaults -v
```

Expected: AttributeError on new fields.

- [ ] **Step 3: Update `FrontendOracleCollectorConfig` dataclass**

In `src/ovggt/training/frontend_oracle_collector.py`:

```python
@dataclass
class FrontendOracleCollectorConfig:
    config: str
    output: str
    dataset_key: str = "train_dataset"
    batch_size: int | None = 1
    num_workers: int | None = 0
    max_batches: int = 1
    max_events: int = 64
    num_samples: int = 8
    oracle_window: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    teacher_checkpoint: str | None = None
    student_checkpoint: str | None = None
    high_budget_teacher: bool = True
    flush_every_events: int = 16
    flush_every_batches: int = 1
    log_every_subsets: int = 1
    subset_replay_batch_size: int = 16
    # --- Phase 1 sampling-policy parameters ---
    layers_per_frame: int = 2                              # was 4; Decision 4
    max_events_per_sequence: int = 16                      # was 64; Decision 3
    max_events_per_frame: int = 6                          # Decision 3
    max_candidate_events_per_sequence: int = 256           # Decision 3
    max_subsets_per_dedup_event: int = 8                   # Decision 2
    max_subsets_per_eviction_event: int = 8                # Decision 2
    max_subsets_per_fifo_event: int = 8                    # Decision 2
    event_selection_policy: str = "stratified_round_robin" # Decision 3
    stratified_layer_bucket_width: int = 6                 # Decision 3
    # --- Stress profiles (Decision 5) ---
    oracle_profile: str = "real_policy"
    frontend_per_layer_budget_override: int | None = None
    fifo_keep_topk_override: int | None = None
    # --- Phase 4 manifest (Decision 9) ---
    sequence_manifest_path: str | None = None
    sequence_partition_policy: str = "hash_mod"
    num_sequence_shards: int | None = None
    sequence_shard_id: int | None = None
    # --- Existing fields ---
    store_replay_payload: bool = False
    num_views: int | None = None
    max_fetch_errors: int = 256
```

- [ ] **Step 4: Add CLI args in `tools/collect_counterfactual_oracle.py`**

Add after existing args:

```python
# Phase 1 sampling-policy args
parser.add_argument("--max-subsets-per-dedup-event", type=int, default=8)
parser.add_argument("--max-subsets-per-eviction-event", type=int, default=8)
parser.add_argument("--max-subsets-per-fifo-event", type=int, default=8)
parser.add_argument("--max-candidate-events-per-sequence", type=int, default=256)
parser.add_argument("--max-events-per-sequence", type=int, default=16)
parser.add_argument("--max-events-per-frame", type=int, default=6)
parser.add_argument("--layers-per-frame", type=int, default=2)
parser.add_argument("--event-selection-policy", type=str, default="stratified_round_robin",
                    choices=["first_n", "stratified_round_robin"])
parser.add_argument("--stratified-layer-bucket-width", type=int, default=6)
parser.add_argument("--oracle-profile", type=str, default="real_policy",
                    choices=["real_policy", "low_budget_eviction", "fifo_topk"])
parser.add_argument("--frontend-per-layer-budget-override", type=int, default=None)
parser.add_argument("--fifo-keep-topk-override", type=int, default=None)
# Phase 4 manifest args
parser.add_argument("--sequence-manifest-path", type=str, default=None)
parser.add_argument("--sequence-partition-policy", type=str, default="hash_mod",
                    choices=["hash_mod", "contiguous"])
parser.add_argument("--num-sequence-shards", type=int, default=None)
parser.add_argument("--sequence-shard-id", type=int, default=None)
```

Wire all of these into `FrontendOracleCollectorConfig(...)` construction in `main()`.

- [ ] **Step 5: Update `config/collect_counterfactual_oracle.yaml`**

```yaml
# Phase 1 sampling-policy defaults (Recommended Production Defaults from spec)
max_subsets_per_dedup_event: 8
max_subsets_per_eviction_event: 8
max_subsets_per_fifo_event: 8
max_candidate_events_per_sequence: 256
max_events_per_sequence: 16
max_events_per_frame: 6
layers_per_frame: 2
event_selection_policy: stratified_round_robin
stratified_layer_bucket_width: 6
subset_replay_batch_size: 16
store_replay_payload: false
oracle_profile: real_policy
```

- [ ] **Step 6: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestCollectorConfigDefaults -v
```

Expected: 1 passed.

- [ ] **Step 7: Verify YAML loads correctly**

```bash
PYTHONPATH=src python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/collect_counterfactual_oracle.yaml')
print('max_subsets_per_dedup_event:', cfg.get('max_subsets_per_dedup_event'))
print('max_events_per_sequence:', cfg.get('max_events_per_sequence'))
print('layers_per_frame:', cfg.get('layers_per_frame'))
print('event_selection_policy:', cfg.get('event_selection_policy'))
print('max_candidate_events_per_sequence:', cfg.get('max_candidate_events_per_sequence'))
print('max_events_per_frame:', cfg.get('max_events_per_frame'))
print('oracle_profile:', cfg.get('oracle_profile'))
"
```

Expected: all values match the YAML above.

- [ ] **Step 8: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tools/collect_counterfactual_oracle.py config/collect_counterfactual_oracle.yaml tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add all Phase 1 config fields, CLI args, and YAML defaults

Covers Decisions 2-5 and 9: subset caps, event selection policy, stress
profiles, sequence manifest fields. All 16 new parameters from the spec."
```

---

## Task 9: End-to-End Equivalence Test (Cap = ∞ Matches Old Behavior)

**Spec Reference:** Verification Plan — Exact Equivalence Gate

**Files:**
- Create: `tests/test_oracle_phase1_equivalence.py`

- [ ] **Step 1: Write the equivalence test**

```python
"""Phase 1 equivalence: cap disabled must match original full-enumeration behavior.

Spec Verification Plan:
  For matching events: identical score_state and metadata_features.
  For matching subsets: identical keep_indices and loss matches within atol=1e-5, rtol=1e-4.
"""
import pytest
import torch

from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe
from ovggt.utils.frontend_cache import LayerCacheState, TokenKind, TokenMetadata
from test_frontend_oracle_collector import _dedup_cache_state, _dedup_metadata


def _reference_dedup_subsets(num_tokens: int, group_indices: torch.Tensor) -> list[torch.Tensor]:
    """Pre-Phase-1 keep-one-in-N enumeration. Returns a list of keep_indices tensors."""
    keep_sets = []
    group_set = set(int(i) for i in group_indices.tolist())
    for keep in group_indices.tolist():
        keep_tensor = torch.tensor(
            [idx for idx in range(num_tokens) if idx not in group_set or idx == keep],
            dtype=torch.long,
        )
        keep_tensor = keep_tensor.unique(sorted=True)
        keep_sets.append(keep_tensor)
    return keep_sets


@pytest.fixture
def large_voxel_cache():
    importance = [float(i) for i in range(50)]
    return _dedup_cache_state(num_tokens=50, importance=importance)


def test_cap_disabled_emits_same_keep_indices_as_full_enumeration(large_voxel_cache):
    """With cap=1000, the probe's subsets must exactly match the reference enumeration."""
    state = large_voxel_cache
    scores = torch.arange(50, dtype=torch.float)
    probe = CounterfactualDedupProbe(
        num_samples=8, oracle_window=4, seed=42,
        max_subsets_per_dedup_event=1000,
    )
    probe.on_dedup_candidate(
        cache_state=state, layer_id=0, frame_id=5, batch_index=0,
        scores=scores, policy_keep_indices=torch.arange(50),
    )
    assert len(probe.events) == 1
    emitted = [s["keep_indices"] for s in probe.events[0]["candidate_subsets"]]

    group_indices = torch.arange(50)
    reference = _reference_dedup_subsets(num_tokens=50, group_indices=group_indices)
    assert len(emitted) == len(reference) == 50

    emitted_sets = [tuple(int(i) for i in k.tolist()) for k in emitted]
    reference_sets = [tuple(int(i) for i in k.tolist()) for k in reference]
    assert emitted_sets == reference_sets


def test_cap_disabled_emits_identical_score_state_and_metadata(large_voxel_cache):
    """Beyond keep_indices, spec equivalence requires identical score_state and metadata_features."""
    state = large_voxel_cache
    scores = torch.arange(50, dtype=torch.float)
    probe_a = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42,
                                       max_subsets_per_dedup_event=1000)
    probe_b = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42,
                                       max_subsets_per_dedup_event=8)
    probe_a.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0,
                               scores=scores, policy_keep_indices=torch.arange(50))
    probe_b.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0,
                               scores=scores, policy_keep_indices=torch.arange(50))

    e_a, e_b = probe_a.events[0], probe_b.events[0]
    assert torch.equal(e_a["score_state"], e_b["score_state"])
    assert torch.equal(e_a["metadata_features"], e_b["metadata_features"])
    assert e_a["layer_id"] == e_b["layer_id"]
    assert e_a["frame_id"] == e_b["frame_id"]


def test_cap_8_emits_at_most_8_subsets(large_voxel_cache):
    scores = torch.arange(50, dtype=torch.float)
    probe = CounterfactualDedupProbe(
        num_samples=8, oracle_window=4, seed=42,
        max_subsets_per_dedup_event=8,
    )
    probe.on_dedup_candidate(
        cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0,
        scores=scores, policy_keep_indices=torch.arange(50),
    )
    event = probe.events[0]
    assert len(event["candidate_subsets"]) <= 8
    assert len(event["candidate_subsets"]) >= 4


def test_cap_deterministic_with_same_seed(large_voxel_cache):
    """Two probes with seed=42 must emit identical subset keep_indices sequences."""
    scores = torch.arange(50, dtype=torch.float)
    p1 = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42,
                                  max_subsets_per_dedup_event=8)
    p2 = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42,
                                  max_subsets_per_dedup_event=8)
    p1.on_dedup_candidate(cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0,
                          scores=scores, policy_keep_indices=torch.arange(50))
    p2.on_dedup_candidate(cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0,
                          scores=scores, policy_keep_indices=torch.arange(50))
    s1 = p1.events[0]["candidate_subsets"]
    s2 = p2.events[0]["candidate_subsets"]
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert torch.equal(a["keep_indices"], b["keep_indices"])


def test_policy_baseline_always_retained_for_dedup():
    """Quality gate: current-policy baseline must be present for dedup events
    when policy keeps at least one token in the group."""
    importance = [float(i) for i in range(20)]
    state = _dedup_cache_state(num_tokens=20, importance=importance)
    scores = torch.arange(20, dtype=torch.float)
    # Policy keeps tokens 3, 7, 11 (multiple in the group)
    policy_keep = torch.tensor([3, 7, 11, 50, 60])
    probe = CounterfactualDedupProbe(
        num_samples=8, oracle_window=4, seed=42,
        max_subsets_per_dedup_event=8,
    )
    probe.on_dedup_candidate(
        cache_state=state, layer_id=0, frame_id=5, batch_index=0,
        scores=scores, policy_keep_indices=policy_keep,
    )
    event = probe.events[0]
    baselines = [
        subset
        for subset in event["candidate_subsets"]
        if subset.get("source") == "policy_baseline"
    ]
    assert len(baselines) == 1, "Policy baseline must be present once when policy keeps multiple tokens in group"
    assert torch.equal(baselines[0]["keep_indices"], policy_keep.long())
```

- [ ] **Step 2: Run the equivalence tests**

```bash
PYTHONPATH=src pytest tests/test_oracle_phase1_equivalence.py -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_oracle_phase1_equivalence.py
git commit -m "test(oracle): Phase 1 equivalence, determinism, and policy baseline tests

Covers spec Verification Plan: exact equivalence gate (cap disabled =
full enumeration), policy baseline retention, deterministic sampling."
```

---

## Task 10: Shard Summary Metrics

**Spec Reference:** Shard Summary Metrics section

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py` (add summary logging to shard writer)
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_collector.py`:

```python
class TestShardSummaryMetrics:
    def test_shard_summary_includes_required_fields(self):
        """Every shard summary must include: num_sequences, num_events, event type counts,
        frame histogram, layer histogram, subsets/event stats, pair count, events/hour.
        Summary must count final measured subsets; candidate_subsets is only a fallback
        for pre-measurement candidate events."""
        from ovggt.training.frontend_oracle_collector import compute_shard_summary
        events = [
            {"event_type": "dedup", "frame_id": 0, "layer_id": 3,
             "subsets": [{"keep_indices": torch.tensor([0, 1]), "loss": float(i)} for i in range(5)],
             "sequence_provenance": {"sequence_id": "seq_001", "dataset": "WildRGBD"}},
            {"event_type": "dedup", "frame_id": 0, "layer_id": 9,
             "subsets": [{"keep_indices": torch.tensor([0, 1]), "loss": float(i)} for i in range(3)],
             "sequence_provenance": {"sequence_id": "seq_001", "dataset": "WildRGBD"}},
            {"event_type": "eviction", "frame_id": 3, "layer_id": 15,
             "subsets": [{"keep_indices": torch.tensor([0]), "loss": float(i)} for i in range(2)],
             "sequence_provenance": {"sequence_id": "seq_002", "dataset": "DTU"}},
        ]
        summary = compute_shard_summary(events, partial=True, elapsed_sec=3600.0)
        assert summary["num_events"] == 3
        assert summary["num_sequences"] == 2
        assert summary["partial"] is True
        assert "dedup" in summary["event_type_counts"]
        assert summary["event_type_counts"]["dedup"] == 2
        assert summary["event_type_counts"]["eviction"] == 1
        assert 0 in summary["frame_histogram"]
        assert 3 in summary["frame_histogram"]
        assert summary["subsets_per_event"]["min"] == 2
        assert summary["subsets_per_event"]["max"] == 5
        assert "events_per_hour" in summary
        assert summary["dataset_counts"]["WildRGBD"] == 1  # unique sequences
        assert summary["dataset_counts"]["DTU"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestShardSummaryMetrics -v
```

Expected: ImportError on `compute_shard_summary`.

- [ ] **Step 3: Implement `compute_shard_summary`**

Add to `src/ovggt/training/frontend_oracle_collector.py`:

```python
def compute_shard_summary(
    events: list[dict],
    partial: bool = False,
    elapsed_sec: float = 0.0,
) -> dict:
    """Compute shard summary metrics per spec: Shard Summary Metrics section."""
    if not events:
        return {
            "num_sequences": 0, "num_events": 0, "partial": partial,
            "event_type_counts": {}, "dataset_counts": {},
            "frame_histogram": {}, "layer_histogram": {},
            "subsets_per_event": {"min": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0},
            "events_per_hour": 0.0, "elapsed_sec": elapsed_sec,
        }

    # Unique sequences
    seq_ids = set()
    for e in events:
        prov = e.get("sequence_provenance", {})
        seq_id = prov.get("sequence_id", id(e))
        seq_ids.add(seq_id)

    # Event type counts
    et_counts: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "unknown")
        et_counts[et] = et_counts.get(et, 0) + 1

    # Dataset counts (unique sequences per dataset)
    ds_counts: dict[str, int] = {}
    seen_seq_ds: set[tuple] = set()
    for e in events:
        prov = e.get("sequence_provenance", {})
        ds = prov.get("dataset", "unknown")
        sid = prov.get("sequence_id", id(e))
        key = (ds, sid)
        if key not in seen_seq_ds:
            seen_seq_ds.add(key)
            ds_counts[ds] = ds_counts.get(ds, 0) + 1

    # Frame and layer histograms
    frame_hist: dict[int, int] = {}
    layer_hist: dict[int, int] = {}
    for e in events:
        fid = int(e.get("frame_id", 0))
        lid = int(e.get("layer_id", 0))
        frame_hist[fid] = frame_hist.get(fid, 0) + 1
        layer_hist[lid] = layer_hist.get(lid, 0) + 1

    def _subset_count(event: dict) -> int:
        # Final shards store measured results under "subsets". Candidate events
        # before replay use "candidate_subsets"; keep this as a fallback only.
        if "subsets" in event:
            return len(event.get("subsets", []))
        return len(event.get("candidate_subsets", []))

    # Subsets per event stats
    subset_counts = [_subset_count(e) for e in events]
    import numpy as np
    arr = np.array(subset_counts, dtype=float)
    subsets_stats = {
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }

    events_per_hour = len(events) / max(elapsed_sec / 3600, 1e-6)

    return {
        "num_sequences": len(seq_ids),
        "num_events": len(events),
        "partial": partial,
        "event_type_counts": et_counts,
        "dataset_counts": ds_counts,
        "frame_histogram": dict(sorted(frame_hist.items())),
        "layer_histogram": dict(sorted(layer_hist.items())),
        "subsets_per_event": subsets_stats,
        "events_per_hour": events_per_hour,
        "elapsed_sec": elapsed_sec,
        # TODO (Phase 4): add pair_count after min_loss_gap (requires post-hoc pairwise expansion)
        # TODO (Phase 4): add manifest partition coverage (requested, processed, skipped, duplicates)
    }
```

- [ ] **Step 4: Wire summary into shard writer**

Find the shard write point in the collector and add summary logging:

```python
summary = compute_shard_summary(collected_events, partial=is_partial, elapsed_sec=elapsed)
logger.info(f"[oracle] Shard summary: {json.dumps(summary, indent=2, default=str)}")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py::TestShardSummaryMetrics -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add shard summary metrics per spec

compute_shard_summary reports: num_sequences, num_events, event type counts,
dataset counts, frame/layer histograms, subsets/event stats, events/hour."
```

---

## Task 11: Phase 4 — Sequence Manifest Stub + Test

**Spec Reference:** Decision 9 — Phase 1 lands a minimal manifest-builder stub + unit test

**Files:**
- Create: `src/ovggt/training/oracle_manifest.py`
- Test: `tests/test_oracle_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_oracle_manifest.py`:

```python
"""Sequence manifest stub for Phase 4 deterministic shard partitioning."""
import json
import pytest
from pathlib import Path

from ovggt.training.oracle_manifest import (
    build_manifest_from_dataset,
    partition_manifest,
    load_manifest,
)


class TestSequenceManifest:
    def test_build_manifest_creates_jsonl(self, tmp_path):
        """build_manifest_from_dataset should create a JSONL manifest with stable sequence_ids."""
        # Use a mock or tiny dataset config — this is a stub test
        manifest_path = tmp_path / "manifest.jsonl"
        build_manifest_from_dataset(
            config_path="config/train_frontend_finetune.yaml",
            output_path=str(manifest_path),
            max_sequences=10,
        )
        assert manifest_path.exists()
        lines = manifest_path.read_text().strip().splitlines()
        assert len(lines) == 10
        for line in lines:
            entry = json.loads(line)
            assert "sequence_id" in entry
            assert "dataset" in entry
            assert "frame_count" in entry
            assert entry["source_config"] == "config/train_frontend_finetune.yaml"

    def test_partition_manifest_hash_mod(self, tmp_path):
        """hash_mod partition must produce non-overlapping, exhaustive subsets."""
        entries = [{"sequence_id": f"seq_{i:04d}", "dataset": "test"} for i in range(100)]
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        manifest = load_manifest(str(manifest_path))
        part0 = partition_manifest(manifest, num_shards=4, shard_id=0, policy="hash_mod")
        part1 = partition_manifest(manifest, num_shards=4, shard_id=1, policy="hash_mod")
        part2 = partition_manifest(manifest, num_shards=4, shard_id=2, policy="hash_mod")
        part3 = partition_manifest(manifest, num_shards=4, shard_id=3, policy="hash_mod")

        ids0 = {e["sequence_id"] for e in part0}
        ids1 = {e["sequence_id"] for e in part1}
        ids2 = {e["sequence_id"] for e in part2}
        ids3 = {e["sequence_id"] for e in part3}
        parts = [ids0, ids1, ids2, ids3]
        # Non-overlapping and exhaustive
        assert sum(len(part) for part in parts) == len(set().union(*parts)) == 100

    def test_partition_manifest_contiguous(self, tmp_path):
        """contiguous partition must split by index range."""
        entries = [{"sequence_id": f"seq_{i:04d}", "dataset": "test"} for i in range(100)]
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        manifest = load_manifest(str(manifest_path))
        part0 = partition_manifest(manifest, num_shards=4, shard_id=0, policy="contiguous")
        part3 = partition_manifest(manifest, num_shards=4, shard_id=3, policy="contiguous")
        # Part 0 should have the first 25 entries
        assert len(part0) == 25
        assert part0[0]["sequence_id"] == "seq_0000"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_oracle_manifest.py -v
```

Expected: ImportError on `oracle_manifest`.

- [ ] **Step 3: Implement the manifest stub**

Create `src/ovggt/training/oracle_manifest.py`:

```python
"""Sequence manifest utilities for deterministic Phase 4 shard partitioning.

Phase 1 lands a minimal stub + unit test. The manifest is required only at
Phase 4 launch and is not on the Phase 1 critical path for real-policy collection.
"""
import hashlib
import json
from pathlib import Path


def build_manifest_from_dataset(
    config_path: str,
    output_path: str,
    max_sequences: int | None = None,
) -> None:
    """Build a JSONL manifest from the training dataset config.

    Each line: {"sequence_id": str, "dataset": str, "frame_count": int, ...}
    This is a stub — Phase 4 will flesh out the full dataset iteration.
    """
    # TODO: Phase 4 will iterate the actual dataset from config_path.
    # Phase 1 stub emits deterministic synthetic rows when max_sequences is
    # provided so partition tests and launcher plumbing are executable.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = int(max_sequences or 0)
    with open(output_path, "w") as f:
        for idx in range(count):
            f.write(json.dumps({
                "sequence_id": f"stub_seq_{idx:06d}",
                "dataset": "stub",
                "frame_count": 0,
                "source_config": str(config_path),
            }) + "\n")


def load_manifest(manifest_path: str) -> list[dict]:
    """Load a JSONL manifest file."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def partition_manifest(
    manifest: list[dict],
    num_shards: int,
    shard_id: int,
    policy: str = "hash_mod",
) -> list[dict]:
    """Partition manifest entries into non-overlapping subsets.

    Args:
        manifest: list of manifest entries (each must have "sequence_id").
        num_shards: total number of shards.
        shard_id: this shard's index (0-based).
        policy: "hash_mod" (deterministic hash-based) or "contiguous" (index range).
    """
    if policy == "contiguous":
        n = len(manifest)
        chunk = (n + num_shards - 1) // num_shards
        start = shard_id * chunk
        end = min(start + chunk, n)
        return manifest[start:end]

    # hash_mod: deterministic partition by sequence_id hash
    result = []
    for entry in manifest:
        seq_id = entry.get("sequence_id", "")
        h = int(hashlib.md5(seq_id.encode()).hexdigest(), 16)
        if h % num_shards == shard_id:
            result.append(entry)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_oracle_manifest.py -v
```

Expected: 3 passed. The first test must not require a real dataset; it uses
deterministic stub rows when `max_sequences` is provided.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/oracle_manifest.py tests/test_oracle_manifest.py
git commit -m "feat(oracle): add sequence manifest stub for Phase 4 deterministic partitioning

build_manifest_from_dataset stub, load_manifest, partition_manifest with
hash_mod and contiguous policies. Non-overlapping, deterministic partitions."
```

---

## Task 12: Phase 4 — Forward Phase 1 Args In Parallel Launcher

**Spec Reference:** Decision 9 — Phase 4 production multi-GPU scheduling

**Files:**
- Modify: `tools/collect_counterfactual_oracle_parallel.py`
- Test: `tests/test_frontend_oracle_parallel.py`

**Context:** Resume mode is already implemented. Task scope is forwarding new args + manifest fields.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frontend_oracle_parallel.py`:

```python
def test_parallel_launcher_forwards_all_phase1_args(tmp_path):
    """All Phase 1 CLI args must reach each shard command."""
    from tools.collect_counterfactual_oracle_parallel import (
        ParallelCollectorConfig, build_collection_jobs,
    )
    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0",
        shards_per_device=1,
        # Phase 1 args
        max_subsets_per_dedup_event=8,
        max_subsets_per_eviction_event=8,
        max_subsets_per_fifo_event=8,
        max_candidate_events_per_sequence=256,
        max_events_per_sequence=16,
        max_events_per_frame=6,
        layers_per_frame=2,
        event_selection_policy="stratified_round_robin",
        stratified_layer_bucket_width=6,
        oracle_profile="real_policy",
        # Phase 4 manifest args
        sequence_manifest_path=None,
        sequence_partition_policy="hash_mod",
        num_sequence_shards=None,
        sequence_shard_id=None,
    )
    jobs = build_collection_jobs(cfg)
    assert len(jobs) == 1
    cmd = jobs[0].command
    # Spot-check key args
    assert "--max-subsets-per-dedup-event" in cmd
    idx = cmd.index("--max-subsets-per-dedup-event")
    assert cmd[idx + 1] == "8"
    assert "--event-selection-policy" in cmd
    idx = cmd.index("--event-selection-policy")
    assert cmd[idx + 1] == "stratified_round_robin"
    assert "--max-candidate-events-per-sequence" in cmd
    idx = cmd.index("--max-candidate-events-per-sequence")
    assert cmd[idx + 1] == "256"
    assert "--max-events-per-frame" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_parallel.py::test_parallel_launcher_forwards_all_phase1_args -v
```

Expected: TypeError — new fields not in `ParallelCollectorConfig`.

- [ ] **Step 3: Add all Phase 1 + Phase 4 fields to `ParallelCollectorConfig` and `parse_args`**

In `tools/collect_counterfactual_oracle_parallel.py`:

```python
@dataclass
class ParallelCollectorConfig:
    # ... existing fields ...
    # Phase 1
    max_subsets_per_dedup_event: int = 8
    max_subsets_per_eviction_event: int = 8
    max_subsets_per_fifo_event: int = 8
    max_candidate_events_per_sequence: int = 256
    max_events_per_sequence: int = 16
    max_events_per_frame: int = 6
    event_selection_policy: str = "stratified_round_robin"
    stratified_layer_bucket_width: int = 6
    oracle_profile: str = "real_policy"
    frontend_per_layer_budget_override: int | None = None
    fifo_keep_topk_override: int | None = None
    # Phase 4
    sequence_manifest_path: str | None = None
    sequence_partition_policy: str = "hash_mod"
    num_sequence_shards: int | None = None
    sequence_shard_id: int | None = None
```

Add corresponding `parser.add_argument(...)` in `parse_args` and extend `command` in `build_collection_jobs`.

- [ ] **Step 4: Verify existing resume test still passes**

```bash
PYTHONPATH=src pytest tests/test_frontend_oracle_parallel.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/collect_counterfactual_oracle_parallel.py tests/test_frontend_oracle_parallel.py
git commit -m "feat(oracle-parallel): forward all Phase 1 + Phase 4 args to shard commands"
```

---

## Task 13: Phase 4 — Log Summarizer

**Spec Reference:** Shard Summary Metrics section + Decision 9

**Files:**
- Create: `tools/summarize_oracle_collection.py`

- [ ] **Step 1: Implement the summarizer**

```python
"""Summarize oracle collection logs: events/hour, subsets/event, errors.

Reads shard logs and/or .pt shard files to produce the summary metrics
specified in the Shard Summary Metrics section of the spec.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

LOG_RE = re.compile(r"\[oracle (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+)")
MEASURING_RE = re.compile(r"measuring event (\d+)/(\d+).*subsets=(\d+)")
BATCH_DONE_RE = re.compile(r"batch (\d+)/(\d+) done.*total_events=(\d+)")
SUMMARY_RE = re.compile(r"Shard summary: (.+)")


def summarize_log(log_path: Path) -> dict:
    """Parse a single log file for throughput metrics."""
    events_started = 0
    total_subsets = 0
    batches_done = 0
    total_events = 0
    first_ts = None
    last_ts = None
    summary_data = None
    for line in log_path.read_text().splitlines():
        m = LOG_RE.match(line)
        if not m:
            continue
        ts_str, msg = m.groups()
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        m2 = MEASURING_RE.search(msg)
        if m2:
            events_started += 1
            total_subsets += int(m2.group(3))
        m3 = BATCH_DONE_RE.search(msg)
        if m3:
            batches_done += 1
            total_events = max(total_events, int(m3.group(3)))
        m4 = SUMMARY_RE.search(line)
        if m4:
            try:
                summary_data = json.loads(m4.group(1))
            except json.JSONDecodeError:
                pass
    elapsed_sec = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0
    result = {
        "log_path": str(log_path),
        "elapsed_sec": elapsed_sec,
        "events_seen": events_started,
        "avg_subsets_per_event": total_subsets / max(events_started, 1),
        "batches_done": batches_done,
        "total_events_at_end": total_events,
        "events_per_hour": events_started / max(elapsed_sec / 3600, 1e-6),
    }
    if summary_data:
        result["shard_summary"] = summary_data
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize oracle collection logs")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    total_events = 0
    total_elapsed = 0.0
    for log in args.logs:
        s = summarize_log(log)
        total_events += s["events_seen"]
        total_elapsed += s["elapsed_sec"]
        print(f"{log.name}: {s['events_seen']} events, "
              f"avg {s['avg_subsets_per_event']:.1f} subsets/event, "
              f"{s['events_per_hour']:.1f} events/hour, "
              f"{s['elapsed_sec']/60:.1f} min")
        if "shard_summary" in s:
            ss = s["shard_summary"]
            print(f"  event types: {ss.get('event_type_counts', {})}")
            print(f"  sequences: {ss.get('num_sequences', '?')}, datasets: {ss.get('dataset_counts', {})}")
            print(f"  subsets/event: {ss.get('subsets_per_event', {})}")

    if len(args.logs) > 1:
        print(f"\n--- Total: {total_events} events, {total_elapsed/3600:.1f}h ---")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it on the v3 log (if available)**

```bash
python tools/summarize_oracle_collection.py checkpoints/token_oracle/collect_shard000_gpu5_v3.log
```

Expected: output containing events_seen, subsets/event, events/hour.

- [ ] **Step 3: Commit**

```bash
git add tools/summarize_oracle_collection.py
git commit -m "feat(oracle): log summarizer with shard summary metrics support"
```

---

## Task 14: Phase 1 Shard Dataset Compatibility

**Spec Reference:** Training Quality Gate — resulting shards must load cleanly with `CounterfactualOracleDataset`

**Files:**
- Modify: `tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write the failing/coverage test**

Add to `tests/test_token_oracle_dataset.py`:

```python
def test_phase1_shard_with_capped_and_stratified_metadata_loads(tmp_path):
    """Phase 1 shards store measured subsets, including policy_baseline sources,
    and must produce valid pairwise ranking samples."""
    import torch
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    shard_path = tmp_path / "phase1_oracle.pt"
    event = {
        "event_id": "seq001:dedup:frame1:layer0",
        "event_type": "dedup",
        "frame_id": 1,
        "layer_id": 0,
        "score_state": torch.randn(4, 128),
        "metadata_features": torch.randn(4, 13),
        "sequence_provenance": {
            "sequence_id": "seq_001",
            "dataset": "WildRGBD",
        },
        "collector_config": {
            "max_subsets_per_dedup_event": 8,
            "event_selection_policy": "stratified_round_robin",
            "max_candidate_events_per_sequence": 256,
            "max_events_per_sequence": 16,
        },
        "subsets": [
            {
                "source": "policy_baseline",
                "keep_indices": torch.tensor([0, 1, 2]),
                "loss": 0.30,
            },
            {
                "source": "keep_one",
                "keep_indices": torch.tensor([0, 2]),
                "loss": 0.10,
            },
            {
                "source": "keep_one",
                "keep_indices": torch.tensor([1, 2]),
                "loss": 0.45,
            },
        ],
    }
    torch.save({
        "format": "ovggt_counterfactual_oracle_v1",
        "num_events": 1,
        "events": [event],
    }, shard_path)

    dataset = CounterfactualOracleDataset([shard_path], min_loss_gap=0.0)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["event_type"] == "dedup"
    assert sample["sequence_provenance"]["sequence_id"] == "seq_001"
    assert sample["better_mask"].shape[0] == 4
    assert sample["worse_mask"].shape[0] == 4
    assert sample["target_margin"] > 0
```

- [ ] **Step 2: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/test_token_oracle_dataset.py::test_phase1_shard_with_capped_and_stratified_metadata_loads -v
```

Expected: 1 passed. This test is a contract check: Phase 1 must not change shard
format or require the dataset to read `candidate_subsets`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_token_oracle_dataset.py
git commit -m "test(oracle): verify Phase 1 shards load in CounterfactualOracleDataset"
```

---

## Task 15: TokenScorer Held-Out Splits And Event-Type Metrics

**Spec Reference:** Training Quality Gate — early and full held-out rank accuracy thresholds

**Files:**
- Modify: `src/train_token_scorer_oracle.py`
- Test: `tests/test_train_token_scorer_oracle.py`

- [ ] **Step 1: Write split and metrics tests**

Create or extend `tests/test_train_token_scorer_oracle.py`:

```python
import torch

from train_token_scorer_oracle import (
    split_oracle_pair_samples,
    summarize_metrics_by_event_type,
)


def _sample(event_id, event_type="dedup", sequence_id="seq_a"):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "sequence_provenance": {"sequence_id": sequence_id},
    }


def test_event_hash_split_has_no_event_id_leakage():
    samples = []
    for idx in range(50):
        event_id = f"event_{idx:03d}"
        samples.append(_sample(event_id, sequence_id=f"seq_{idx % 5}"))
        samples.append(_sample(event_id, sequence_id=f"seq_{idx % 5}"))

    train, val = split_oracle_pair_samples(
        samples,
        val_fraction=0.2,
        split_key="event_id_hash",
        seed=0,
    )
    train_ids = {s["event_id"] for s in train}
    val_ids = {s["event_id"] for s in val}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)


def test_sequence_split_has_no_sequence_leakage():
    samples = [
        _sample(f"event_{idx:03d}", sequence_id=f"seq_{idx % 10}")
        for idx in range(100)
    ]
    train, val = split_oracle_pair_samples(
        samples,
        val_fraction=0.2,
        split_key="sequence_id",
        seed=0,
    )
    train_seq = {s["sequence_provenance"]["sequence_id"] for s in train}
    val_seq = {s["sequence_provenance"]["sequence_id"] for s in val}
    assert train_seq
    assert val_seq
    assert train_seq.isdisjoint(val_seq)


def test_summarize_metrics_by_event_type():
    rows = [
        {"event_type": "dedup", "rank_correct": torch.tensor(True)},
        {"event_type": "dedup", "rank_correct": torch.tensor(False)},
        {"event_type": "eviction", "rank_correct": torch.tensor(True)},
    ]
    summary = summarize_metrics_by_event_type(rows)
    assert summary["dedup"]["count"] == 2
    assert summary["dedup"]["rank_acc"] == 0.5
    assert summary["eviction"]["count"] == 1
    assert summary["eviction"]["rank_acc"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_train_token_scorer_oracle.py -v
```

Expected: ImportError on `split_oracle_pair_samples` and
`summarize_metrics_by_event_type`.

- [ ] **Step 3: Implement deterministic split helpers**

In `src/train_token_scorer_oracle.py`, add:

```python
import hashlib


def _stable_bucket(value: str, seed: int = 0, buckets: int = 10000) -> int:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def split_oracle_pair_samples(
    samples: list[dict],
    val_fraction: float = 0.1,
    split_key: str = "event_id_hash",
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split pair samples before DataLoader construction.

    Early calibration uses split_key="event_id_hash" so pair samples from the
    same oracle event never appear in both splits. Full production uses
    split_key="sequence_id" so scenes/sequences do not leak.
    """
    threshold = int(float(val_fraction) * 10000)
    train, val = [], []
    for sample in samples:
        if split_key == "sequence_id":
            prov = sample.get("sequence_provenance") or {}
            key = str(prov.get("sequence_id") or sample.get("event_id") or "")
        elif split_key == "event_id_hash":
            key = str(sample.get("event_id") or "")
        else:
            raise ValueError(f"Unsupported split_key={split_key}")
        target = val if _stable_bucket(key, seed=seed) < threshold else train
        target.append(sample)
    return train, val
```

- [ ] **Step 4: Implement per-event-type validation metrics**

Add a validation loop that reports:
- `val_rank_acc`
- `val_rank_acc_dedup`
- `val_rank_acc_eviction`
- `val_rank_acc_fifo_topk`
- `val_count_*`

Keep the training loss unchanged. The split must happen before DataLoader
construction, using `dataset.samples` from `CounterfactualOracleDataset`.

Add CLI/YAML args:

```python
parser.add_argument("--val-fraction", type=float)
parser.add_argument("--split-key", choices=["event_id_hash", "sequence_id"])
parser.add_argument("--split-seed", type=int)
parser.add_argument("--stress-profile-weights", type=str, default=None)
```

Defaults:

```python
"val_fraction": 0.1,
"split_key": "event_id_hash",
"split_seed": 0,
"stress_profile_weights": None,
```

If `stress_profile_weights` is provided, parse it as JSON mapping event type to
sampling weight and use a `WeightedRandomSampler` for the training DataLoader
only. Do not change validation sampling.

- [ ] **Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_train_token_scorer_oracle.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run a tiny trainer parse check**

```bash
PYTHONPATH=src python -m train_token_scorer_oracle --help
```

Expected: help output includes `--val-fraction`, `--split-key`, and `--split-seed`.

- [ ] **Step 7: Commit**

```bash
git add src/train_token_scorer_oracle.py tests/test_train_token_scorer_oracle.py
git commit -m "feat(token-oracle): add held-out splits and event-type validation metrics"
```

---

## Task 16: Smoke Run + Production Verification

**Spec Reference:** Verification Plan — Production Smoke Test (7 steps)

**Files:** none (verification only)

- [ ] **Step 1: Pick a free GPU and run a 10-sequence smoke test**

```bash
cd /path/to/mount/lyj/voxel-vggt
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --output checkpoints/token_oracle/smoke_phase1.pt \
  --max-batches 10 \
  --max-events 4096 \
  --max-events-per-sequence 16 \
  --max-candidate-events-per-sequence 256 \
  --max-events-per-frame 6 \
  --max-subsets-per-dedup-event 8 \
  --layers-per-frame 2 \
  --event-selection-policy stratified_round_robin \
  --subset-replay-batch-size 8 \
  --oracle-profile real_policy \
  --seed 0 \
  --device cuda \
  2>&1 | tee checkpoints/token_oracle/smoke_phase1.log
```

- [ ] **Step 2: Verify wall time and event count**

**Target:** ≤ 9 min per sequence on average (≥ 30× speedup vs v3 baseline).

```bash
python tools/summarize_oracle_collection.py checkpoints/token_oracle/smoke_phase1.log
```

If average per-sequence time exceeds 9 min, stop and investigate.

- [ ] **Step 3: Verify the shard loads correctly with `CounterfactualOracleDataset`**

```bash
PYTHONPATH=src python -c "
import torch
from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset
shard = torch.load('checkpoints/token_oracle/smoke_phase1.pt', map_location='cpu', weights_only=False)
print('num_events:', shard.get('num_events'))
ds = CounterfactualOracleDataset(['checkpoints/token_oracle/smoke_phase1.pt'])
print('num_pairs:', len(ds))
sample = ds[0]
print('sample keys:', list(sample.keys()))
"
```

- [ ] **Step 4: Verify shard summary includes required metrics**

Check the log for the `[oracle] Shard summary:` line. It must include:
- ✅ Nonzero sequence count
- ✅ Event type counts (dedup, eviction, fifo_topk)
- ✅ Frame histogram
- ✅ Layer histogram
- ✅ subsets/event min/mean/p50/p90/max
- ✅ Dataset/scene counts
- ✅ events/hour

- [ ] **Step 5: Verify stratified selection improves frame coverage**

```bash
# Compare: run with first_n and stratified_round_robin on the same 10 sequences
PYTHONPATH=src python -c "
import re
# Parse frame histogram from both logs
# Stratified must show more distinct frames than first_n
"
```

Expected: `stratified_round_robin` selects later frames when candidate pool contains them.

- [ ] **Step 6: Verify candidate cap > measured cap and selected events include later frames**

Check log output for:
- Candidate pool includes frames > 0
- Selected events include frames > 0

- [ ] **Step 7: Document measured throughput**

```bash
git add checkpoints/token_oracle/smoke_phase1.log
git commit -m "chore(oracle): Phase 1 smoke test, X events/Y min"
```

---

## Task 17: Phase 4 — Production 8-GPU Launch

**Files:** none (operational)

- [ ] **Step 1: Confirm GPUs 0-7 are idle**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

- [ ] **Step 2: Launch the parallel collector**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=src nohup python -u tools/collect_counterfactual_oracle_parallel.py \
  --config config/train_frontend_finetune.yaml \
  --output-dir checkpoints/token_oracle_phase1 \
  --devices 0,1,2,3,4,5,6,7 \
  --shards-per-device 5 \
  --start-shard 0 \
  --max-batches 128 \
  --max-events 4096 \
  --max-events-per-sequence 16 \
  --max-candidate-events-per-sequence 256 \
  --max-events-per-frame 6 \
  --max-subsets-per-dedup-event 8 \
  --layers-per-frame 2 \
  --event-selection-policy stratified_round_robin \
  --subset-replay-batch-size 8 \
  --oracle-profile real_policy \
  > checkpoints/token_oracle_phase1/parallel.log 2>&1 &
```

Resume behavior is the default for the current parallel launcher: existing
complete/partial shard outputs are skipped unless `--overwrite-existing` is
supplied. Do not add `--skip-completed` unless Task 12 explicitly implements
that CLI flag.

- [ ] **Step 3: Monitor with the summarizer**

```bash
watch -n 60 'python tools/summarize_oracle_collection.py checkpoints/token_oracle_phase1/collect_*.log'
```

- [ ] **Step 4: Run supplemental stress shards after real-policy smoke test passes**

```bash
# low_budget_eviction stress shard
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --output checkpoints/token_oracle_phase1/stress_eviction.pt \
  --max-batches 30 \
  --oracle-profile low_budget_eviction \
  --frontend-per-layer-budget-override 833 \
  --max-subsets-per-dedup-event 8 \
  --max-events-per-sequence 16 \
  --layers-per-frame 2 \
  --seed 42

# fifo_topk stress shard
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --output checkpoints/token_oracle_phase1/stress_fifo.pt \
  --max-batches 30 \
  --oracle-profile fifo_topk \
  --fifo-keep-topk-override 8 \
  --max-subsets-per-dedup-event 8 \
  --max-events-per-sequence 16 \
  --layers-per-frame 2 \
  --seed 42
```

- [ ] **Step 5: Document production runtime**

Write: `docs/superpowers/notes/2026-06-XX-phase1-production-runtime.md` with:
- Measured events/hour per GPU
- Total wall time
- Sequence coverage count
- Event type distribution
- Any issues encountered

---

## Verification Gate

Before declaring Phase 1 done:

- [ ] **All new tests pass:**
  ```bash
  PYTHONPATH=src pytest tests/test_frontend_oracle_collector.py tests/test_oracle_phase1_equivalence.py tests/test_frontend_oracle_parallel.py tests/test_oracle_manifest.py tests/test_token_oracle_dataset.py tests/test_train_token_scorer_oracle.py -v
  ```

- [ ] **Shard dataset compatibility test (Task 14) passes** with `CounterfactualOracleDataset`

- [ ] **TokenScorer held-out split + event-type metric tests (Task 15) pass**

- [ ] **Smoke test (Task 16) shows ≥ 30× speedup** (≤ 9 min/sequence vs v3 ~4.5h)

- [ ] **Shard summary metrics present in every shard log** (event types, frame/layer histograms, subsets/event stats)

- [ ] **Stratified selection improves frame coverage** vs `first_n` on same seed

- [ ] **Production 8-GPU launch (Task 17) completes at least 5,000 sequences in <24h**

- [ ] **Resulting shards load cleanly** with `CounterfactualOracleDataset`

- [ ] **Phase 1 Sampling Policy Gate** (spec Verification Plan — ≤34 GPU-hours calibration):
  - Run cap=8 / cap=12 / cap=16 / no-cap (3 sequences) on fixed seed=42
  - Select fastest cap whose retained-subset loss-gap agreement ≥ 80% on overlap sequences
  - Verify current-policy baseline retention is 100% on dedup events

- [ ] **Training Quality Gate** (spec Verification Plan):
  - Train `TokenScorer` on calibration shards with 90/10 held-out split
  - Early gate: held-out rank accuracy ≥ 52% on cap=8 shard
  - Full gate (after Phase 1+4): held-out rank accuracy ≥ 60% with sequence-level split

- [ ] **Phase 4 manifest smoke:** two shards from tiny manifest partition → non-overlapping sequence ids, zero duplicates

## Out Of Scope

- Phase 2 (prefix snapshot reuse) — separate plan if Phase 1 < 30× speedup
- Phase 3 (model-level vectorization) — out of scope unless Phase 1+2 insufficient
- Changing `TASK_WEIGHTS`, `compute_three_task_loss_components`, or `token_oracle_ranking_loss`
- Changing the shard format (`ovggt_counterfactual_oracle_v1`)
- Full downstream frontend fine-tuning benchmark (short TokenScorer calibration runs are in scope)
