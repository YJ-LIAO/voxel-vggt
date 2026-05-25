# Frontend Long-Sequence Precision Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce FE int=8 200-frame ATE from 0.0502m to <=0.0300m while maintaining short-sequence accuracy.

**Architecture:** Three independent optimizations to the Frontend cache pipeline: (A) defer reorder until eviction needs it, (B) add cooldown period for newly promoted anchors in voxel dedup, (C) protect high-score tokens during FIFO demotion. Each targets a specific identified error source from ablation data.

**Tech Stack:** Python 3.10, PyTorch, OVGGT/VGGT transformer architecture

---

## Background & Ablation Data

Current ablation results on 7-Scenes chess/seq-03:

| Config | 50 frames | 100 frames | 200 frames |
|--------|-----------|------------|------------|
| Legacy (coverage) | 0.0160 | 0.0160 | 0.0260 |
| FE int=100 noDedup | 0.0159 | 0.0161 | 0.0275 |
| FE int=100 improved_dedup | 0.0214 | 0.0210 | 0.0280 |
| FE int=8 noDedup | 0.0196 | 0.0210 | 0.0358 |
| FE int=8 improved_dedup | 0.0220 | 0.0267 | 0.0502 |
| FE int=8 old_dedup | 0.0265 | 0.0303 | 0.1534 |

Target: FE int=8 + improved_dedup at 200 frames should match FE int=100 + improved_dedup (0.0280m).

Three error sources identified:
1. **Reorder disruption** (commit_pending_update_ line 619-620): Every keyframe promotion triggers reorder_by_anchor_slots_ which reshuffles the entire KV cache, disrupting spatial locality and making subsequent dedup less effective.
2. **Dedup over-pruning** (_dedup_single_batch line 520-567): Newly promoted anchor tokens immediately participate in voxel conflict detection, aggressively discarding current-frame tokens that share their voxels.
3. **FIFO token loss** (frontend_keyframe.py line 157-173): When max_history_anchors (3) is full, demoting the oldest anchor loses all its protected tokens, regardless of their importance scores.

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/ovggt/utils/frontend_cache.py` | LayerCacheState: dedup, reorder, eviction, commit | Modify |
| `src/ovggt/utils/frontend_keyframe.py` | FrontendKeyframeManager: keyframe promotion, FIFO | Modify |
| `tools/test_dedup_ablation.py` | Existing ablation test script | Use for verification |

---

## Task A: Delay Reorder Until Eviction

**Why:** reorder_by_anchor_slots_ is called on every commit_pending_update_ when anchor tokens exist (line 619). With int=8, this happens every 8 frames, reshuffling the full cache each time. The reorder is only needed for eviction to know which tokens are protected. Deferring it until eviction actually runs avoids unnecessary disruption.

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py:594-641` (commit_pending_update_)
- Modify: `src/ovggt/utils/frontend_cache.py:382-403` (reorder_by_anchor_slots_)

- [ ] **Step 1: Add needs_reorder_ flag to LayerCacheState**

In `src/ovggt/utils/frontend_cache.py`, find the `__init__` method of `LayerCacheState` (around line 50-80) and add:

```python
self.needs_reorder_: bool = False
```

- [ ] **Step 2: Set flag instead of calling reorder in commit_pending_update_**

In `commit_pending_update_` (line 618-621), replace the unconditional reorder call:

Before:
```python
self.append_(k_current, v_current, metadata_current)
if metadata_current.has_anchor_tokens():
    self.reorder_by_anchor_slots_()
self.apply_voxel_dedup_(config, current_frame_id=pending_update.frame_id)
```

After:
```python
self.append_(k_current, v_current, metadata_current)
if metadata_current.has_anchor_tokens():
    self.needs_reorder_ = True
self.apply_voxel_dedup_(config, current_frame_id=pending_update.frame_id)
```

- [ ] **Step 3: Execute reorder lazily before eviction**

In `commit_pending_update_` (line 622-641), add reorder check before the eviction section:

Before:
```python
if pending_update.cache_budget is None or self.num_tokens() <= pending_update.cache_budget:
    return None
```

After:
```python
if self.needs_reorder_:
    self.reorder_by_anchor_slots_()
    self.needs_reorder_ = False

if pending_update.cache_budget is None or self.num_tokens() <= pending_update.cache_budget:
    return None
```

- [ ] **Step 4: Smoke test**

Run: `cd /path/to/mount/lyj/voxel-vggt && CUDA_VISIBLE_DEVICES=4 conda run -n OVGGT python tools/test_smoke.py`

Expected: "OK, 10 frames, no crash." with no errors.

- [ ] **Step 5: Run ablation**

Run: `cd /path/to/mount/lyj/voxel-vggt && CUDA_VISIBLE_DEVICES=4 conda run -n OVGGT python tools/test_dedup_ablation.py`

Expected: FE8_dedup ATE at 200 frames should be lower than 0.0502. Even a small improvement validates the reorder delay is correct.

- [ ] **Step 6: Commit**

```bash
cd /path/to/mount/lyj/voxel-vggt
git add src/ovggt/utils/frontend_cache.py
git commit -m "feat: defer cache reorder until eviction actually needs it"
```

---

## Task B: Dedup Cooldown for New Anchors

**Why:** When a frame is promoted to anchor (every 8 frames with int=8), its tokens immediately become "protected" in voxel dedup. This means the next frame's tokens that happen to land in the same voxels get discarded. A cooldown period lets newly promoted anchors coexist with incoming frames for N frames before participating in conflict detection.

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py:436-441` (protected_patch_mask in apply_voxel_dedup_)
- Modify: `src/ovggt/utils/frontend_cache.py:22-29` (FrontendCacheConfig)
- Modify: `src/ovggt/utils/frontend_cache.py` (metadata field usage)

- [ ] **Step 1: Add cooldown config field**

In `FrontendCacheConfig` (line 22-29), add:

```python
dedup_cooldown_frames: int = 0  # New anchors skip dedup for N frames after promotion
```

Default is 0 (no cooldown, preserving backward compatibility).

- [ ] **Step 2: Add promotion_frame field to TokenMetadata**

Find the `TokenMetadata` dataclass (search for `class TokenMetadata`). It should have fields like `frame_id`, `anchor_slot`, `keyframe_id`, etc. Add:

```python
promotion_frame: Optional[Tensor] = None  # frame_id when token was promoted to anchor
```

Also update `TokenMetadata.index_select` and any other methods that manipulate fields to handle `promotion_frame`.

- [ ] **Step 3: Set promotion_frame when anchor is promoted**

In `build_frame_token_metadata_base` (search for this function in the codebase, likely in `ovggt/utils/frontend_cache.py` or a metadata builder), when `anchor_slot >= 0`, set:

```python
promotion_frame = torch.full((1, total_tokens), float(frame_id), dtype=torch.float32)
```

For non-anchor tokens (anchor_slot < 0), set `promotion_frame` to NaN or -1.

- [ ] **Step 4: Modify protected_patch_mask to exclude cooldown anchors**

In `apply_voxel_dedup_` (line 436-441), change the protected_patch_mask to exclude anchors still in cooldown:

Before:
```python
protected_patch_mask = (
    (metadata.anchor_slot >= 0)
    & (metadata.frame_id != current_frame_id)
    & patch_mask
    & valid_xyz_mask
)
```

After:
```python
cooldown_mask = torch.ones_like(metadata.anchor_slot, dtype=torch.bool)
if config.dedup_cooldown_frames > 0 and metadata.promotion_frame is not None:
    frames_since_promotion = current_frame_id - metadata.promotion_frame
    cooldown_mask = frames_since_promotion > config.dedup_cooldown_frames

protected_patch_mask = (
    (metadata.anchor_slot >= 0)
    & (metadata.frame_id != current_frame_id)
    & patch_mask
    & valid_xyz_mask
    & cooldown_mask
)
```

- [ ] **Step 5: Pass cooldown config through OVGGT construction**

In `src/ovggt/models/ovggt.py`, where `FrontendCacheConfig` is constructed (search for `FrontendCacheConfig(`), pass the cooldown parameter. This may already be forwarded from constructor args. Ensure `dedup_cooldown_frames` can be set.

- [ ] **Step 6: Smoke test**

Run: `cd /path/to/mount/lyj/voxel-vggt && CUDA_VISIBLE_DEVICES=4 conda run -n OVGGT python tools/test_smoke.py`

Expected: "OK, 10 frames, no crash."

- [ ] **Step 7: Tune cooldown parameter**

Create a small test script or modify the ablation script to try cooldown values [0, 4, 8, 16]:

```python
for cooldown in [0, 4, 8, 16]:
    config = FrontendCacheConfig(enabled=True, dedup_enabled=True, dedup_cooldown_frames=cooldown)
    # ... run 200-frame test ...
```

Expected: Optimal cooldown should reduce 200-frame ATE. Based on int=8, cooldown=4-8 is likely best (half to full interval).

- [ ] **Step 8: Commit**

```bash
cd /path/to/mount/lyj/voxel-vggt
git add src/ovggt/utils/frontend_cache.py src/ovggt/models/ovggt.py
git commit -m "feat: add dedup cooldown period for newly promoted anchors"
```

---

## Task C: FIFO Protection for High-Score Tokens

**Why:** When max_history_anchors (3) is full and a new keyframe triggers FIFO_SWAP (frontend_keyframe.py line 157-173), the oldest anchor (slot 1) is demoted. All its tokens lose protected status and become eligible for eviction. If some of those tokens have very high importance scores, losing them degrades accuracy. We should retain the top-K tokens by score even after demotion.

**Files:**
- Modify: `src/ovggt/utils/frontend_keyframe.py:157-173` (FIFO_SWAP branch)
- Modify: `src/ovggt/utils/frontend_cache.py` (new method: protect_topk_on_demotion_)

- [ ] **Step 1: Add protect_topk_on_demotion_ method to LayerCacheState**

In `src/ovggt/utils/frontend_cache.py`, add a new method to LayerCacheState:

```python
def protect_topk_on_demotion_(self, demoted_slot: int, keep_count: int) -> None:
    """When an anchor slot is demoted, retain top-K tokens by importance score."""
    if self.metadata is None or self.num_tokens() == 0:
        return
    demoted_mask = self.metadata.anchor_slot[:, :] == demoted_slot
    if not demoted_mask.any():
        return
    for b_idx in range(self.metadata.anchor_slot.shape[0]):
        indices = torch.nonzero(demoted_mask[b_idx], as_tuple=False).squeeze(-1)
        if indices.numel() <= keep_count:
            continue
        scores = self.metadata.importance[b_idx, indices]
        _, top_local = torch.topk(scores, k=keep_count)
        top_indices = indices[top_local]
        # Change demoted non-topK to unassigned (slot -1)
        all_demoted = torch.zeros_like(demoted_mask[b_idx])
        all_demoted[indices] = True
        keep_mask = torch.zeros_like(demoted_mask[b_idx])
        keep_mask[top_indices] = True
        demote_mask = all_demoted & ~keep_mask
        self.metadata.anchor_slot[b_idx, demote_mask] = -1
```

- [ ] **Step 2: Call protect_topk during FIFO_SWAP in ovggt.py**

In `src/ovggt/models/ovggt.py`, find where FIFO_SWAP events are handled. After `apply_keyframe_event_` and before `commit_pending_update_`, add:

```python
if event.event_type == KeyframeEventType.FIFO_SWAP and event.demoted_slot is not None:
    for layer_idx in range(len(cache_states)):
        cache_states[layer_idx].protect_topk_on_demotion_(
            demoted_slot=event.demoted_slot,
            keep_count=50,  # retain top 50 tokens from demoted anchor
        )
```

The `keep_count=50` is a starting point; tune based on results.

- [ ] **Step 3: Smoke test**

Run: `cd /path/to/mount/lyj/voxel-vggt && CUDA_VISIBLE_DEVICES=4 conda run -n OVGGT python tools/test_smoke.py`

Expected: "OK, 10 frames, no crash."

- [ ] **Step 4: Tune keep_count**

Try values [0, 25, 50, 100, 200] on 200-frame test. The optimal value balances keeping useful geometry info vs leaving room for new tokens.

- [ ] **Step 5: Commit**

```bash
cd /path/to/mount/lyj/voxel-vggt
git add src/ovggt/utils/frontend_cache.py src/ovggt/models/ovggt.py
git commit -m "feat: protect top-K tokens by score when FIFO-demoting oldest anchor"
```

---

## Task D: Combined Verification & Final Tuning

**Why:** Each optimization (A, B, C) targets an independent error source. Need to verify they compose well together and hit the target ATE.

**Files:**
- Modify: `tools/test_dedup_ablation.py` (add combined config)

- [ ] **Step 1: Create combined ablation test**

Add to `tools/test_dedup_ablation.py` or create `tools/test_combined_ablation.py`:

```python
# Combined: all three optimizations
a_combined = run('Combined',
    lambda: OVGGT(mode='frontend_eval', total_budget=200000,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            dedup_enabled=True,
            dedup_cooldown_frames=8,  # Task B
        )),
    lambda m: m.inference(inputs,
        history_anchor_strategy='fixed_interval',
        anchor_interval=8, max_anchors=3))
```

- [ ] **Step 2: Run full ablation matrix**

Run the combined test for all frame counts [50, 100, 200] and compare:

| Config | Expected |
|--------|----------|
| Legacy | 0.0160 / 0.0160 / 0.0260 (baseline) |
| Combined (A+B+C) | <=0.0200 / <=0.0250 / <=0.0300 (target) |

- [ ] **Step 3: Tune hyperparameters**

If target not met, try grid:
- cooldown_frames: [4, 8, 12, 16]
- fifo_keep_count: [25, 50, 100, 200]

Pick the combination that minimizes 200-frame ATE without hurting 50-frame accuracy.

- [ ] **Step 4: Run on additional scenes**

Test on at least 2 more 7-Scenes sequences to verify generalization:
- fire/seq-03
- office/seq-05

Expected: Consistent improvement across scenes.

- [ ] **Step 5: Commit final config**

```bash
cd /path/to/mount/lyj/voxel-vggt
git add tools/test_combined_ablation.py
git commit -m "test: add combined optimization ablation test"
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Reorder delay breaks protected_count tracking | protected_count is recomputed in reorder; eviction still gets correct count |
| Cooldown prevents dedup from running at all for early frames | Cooldown only applies to newly promoted anchors, not existing ones |
| FIFO protection leaves too many tokens, exceeds budget | budget eviction still runs after protection; it just prefers non-protected tokens |
| Changes interact negatively | Each task has independent smoke test; combined test in Task D validates composition |

## Success Criteria

- [ ] FE int=8 + improved_dedup 200-frame ATE <= 0.0300m (currently 0.0502m, target 40% reduction)
- [ ] 50-frame ATE does not regress above 0.0250m (currently 0.0220m)
- [ ] No crashes or NaN outputs on any test sequence
