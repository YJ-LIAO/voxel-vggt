# Training Throughput Optimization: Remove batch_size=1 Constraint

> Multi-sequence batched training with independent per-sequence streaming state
> Date: 2026-05-26

---

## 1. Motivation

The OVGGT frontend training pipeline enforces `batch_size=1` through three explicit `ValueError` guards and six implicit architectural assumptions. Each training step processes a single video sequence through the streaming KV-cache architecture. This constrains GPU utilization: a single forward pass saturates only a fraction of available compute, leaving headroom that could be filled by processing multiple independent sequences in parallel.

**Goal**: Enable `batch_size > 1` where each batch element is an independent video sequence with its own keyframe state, KV cache, and camera cache — producing training results mathematically identical to running B separate `batch_size=1` steps and averaging their losses.

**Non-goal**: Optimizing the per-frame sequential loop itself (inherent to the streaming architecture). This spec focuses on the batch dimension only.

---

## 2. Root Cause Analysis: Why batch_size=1 Is Required

Three explicit guards and six singleton state assumptions prevent B>1.

### 2.1 Explicit Guards

| File | Line | Mechanism |
|------|------|-----------|
| `src/ovggt/models/ovggt.py` | 815-834 | `_validate_frontend_batch_size()` raises `ValueError` if `ref_batch_size != 1` |
| `src/train_frontend.py` | 654-658 | Startup check: `if batch_size != 1: raise ValueError` |
| `src/finetune_frontend.py` | 251-255 | Same startup check in finetune script |

### 2.2 Singleton State Assumptions

| # | Component | Singleton | Problem with B>1 |
|---|-----------|-----------|-------------------|
| 1 | `FrontendKeyframeManager` | One `active_keyframe_id`, one `history_slots` list, one `active_pose_encoding` | Each sequence needs independent keyframe slot management |
| 2 | `LayerCacheState` (per layer) | One `TokenMetadata` with `[1, N]` shape, shared anchor semantics | Dedup/eviction uses scalar `frame_id`; `_compute_protected_count` only checks `anchor_slot[0]` |
| 3 | `PendingLayerUpdate` | `frame_id: int` (scalar) | Different batch elements may be at different frame indices (not applicable in training with aligned frames, but the interface blocks it) |
| 4 | `past_key_values_camera` | One list of `(k,v)` per trunk layer | Each sequence needs independent camera KV caches with potentially different anchor counts |
| 5 | `_compute_protected_count` | `anchor_slot[0] >= 0` — indexes only batch 0 | Ignores all other batch elements |
| 6 | `gather_` B=1 fast-path | `_gather_single_batch_` via `index_select` | Multi-batch path (`gather_per_batch_`) pads to equal length, which would corrupt attention with garbage tokens |

### 2.3 Partially Vectorized Lower-Level Code

The low-level utilities (`_dedup_single_batch`, `gather_per_batch_`, `_project_slot_local_xyz_to_active`) already iterate over batch dimension. This suggests multi-batch support was attempted at the tensor operation level, but the **orchestration layer** (`_inference_frontend`, `FrontendKeyframeManager`, `PendingLayerUpdate`) was never updated. The `ValueError` guards correctly prevent silent state corruption.

---

## 3. Design: Per-Batch Independent State

### 3.1 Core Principle

**Each batch element gets a complete, independent copy of all streaming state.** The transformer backbone (aggregator, heads) processes B sequences simultaneously via native tensor batch dimension. The state management layer (`keyframe_manager`, `cache_states`, `camera_cache`) is replicated B times.

### 3.2 State Architecture

```
Current (B=1):
  FrontendKeyframeManager  × 1
  LayerCacheState          × depth  (list of 24)
  past_key_values_camera   × trunk_depth  (list of 4)

After (B≥1):
  FrontendKeyframeManager  × B   (list, independent per-sequence)
  LayerCacheState          × B × depth  (list of lists: B sequences × 24 layers)
  past_key_values_camera   × B × trunk_depth  (list of lists: B × 4)
```

### 3.3 Per-Batch Keyframe Events

`FrontendKeyframeManager.update()` is called B times independently — once per batch element. Each call returns a `KeyframeEvent` specific to that sequence. The events are stored as `List[KeyframeEvent]` (length B).

`apply_keyframe_event_()` receives a single event (per-batch, not a merged event), preserving the existing single-batch semantics. No internal changes to `apply_keyframe_event_` or `KeyframeEvent`.

### 3.4 Aggregator Cache State Interface

The aggregator currently expects `cache_states: List[LayerCacheState]` (length=depth). For B>1, we provide a **merged view**:

For each layer, `k` and `v` tensors from all B `LayerCacheState` instances are concatenated along the batch dimension into `[B, H, N_max, D]` before passing to the aggregator. Since each sequence may have different numbers of cached tokens, the shorter caches are **right-padded with zeros** and the corresponding attention mask entries are set to `-inf` to prevent attending to padding.

This approach avoids modifying `aggregator.py`'s internal attention logic at the cost of some wasted compute on padding tokens.

### 3.5 Training Loss Equivalence

Loss is computed as the mean across batch elements: `loss = criterion_loss.mean() + distill_loss.mean()`. This is mathematically equivalent to accumulating B separate `batch_size=1` steps (with `accum_iter=B`) but in a single forward+backward pass.

Gradient contributions are identical because:
- Each sequence has independent state → independent forward paths in the stateful components
- Shared backbone parameters receive gradients from all B sequences simultaneously
- The batch-mean reduction is linear (identical to serial accumulation + division)

---

## 4. Detailed Changes

### 4.1 `src/ovggt/models/ovggt.py` — `_inference_frontend` Refactor

**State initialization** (replaces lines 385-392):

```python
# Per-batch independent state
keyframe_managers = [
    FrontendKeyframeManager(frontend_keyframe_config)
    for _ in range(B)
]
cache_states = [
    [LayerCacheState(max_history_anchors=frontend_keyframe_config.max_history_anchors)
     for _ in range(self.aggregator.depth)]
    for _ in range(B)
]
past_key_values_camera = [
    [None] * self.camera_head.trunk_depth
    for _ in range(B)
]
total_distill_loss = None
```

**Aggregator call** (replaces line 397):

```python
# Merge per-batch cache states for the aggregator
batch_k, batch_v = merge_cache_states_for_batch(
    [cs[layer] for cs in cache_states], B, layer)
merged_cache = LayerCacheState()
merged_cache.k, merged_cache.v = batch_k, batch_v

aggregated_tokens, ..., frame_distill_loss = self.aggregator(
    images,
    cache_states=[merged_cache for ...],  # one merged state per layer
    ...
)
```

**Per-batch event loop** (replaces lines 498-553):

```python
# Independent keyframe decisions per batch
events = []
for b in range(B):
    event = keyframe_managers[b].update(
        frame_idx=i,
        depth=depth[b],
        pose_abs_enc=camera_pose[b],
        image_size_hw=(img_h, img_w),
    )
    events.append(event)

# Per-batch per-layer cache commit
for b in range(B):
    for layer_idx in range(self.aggregator.depth):
        cache_states[b][layer_idx].apply_keyframe_event_(events[b])
        cache_states[b][layer_idx].commit_pending_update_(...)
```

**Guard removal**: Change `_validate_frontend_batch_size` from raising `ValueError` to a no-op (or remove the call at line 378).

### 4.2 `src/ovggt/utils/frontend_keyframe.py`

Minimal changes. `update()` already returns `KeyframeEvent` per-call. Only verification needed: ensure internal state is fully reset between calls (no cross-contamination from previous batch element).

### 4.3 `src/ovggt/utils/frontend_cache.py` — Sync Point Elimination

| Location | Current | Replacement |
|----------|---------|-------------|
| `has_anchor_tokens()` (L149) | `bool((self.anchor_slot >= 0).any().item())` | Use cached `self._cached_has_anchor` updated after reorder/evict |
| `_compute_protected_count()` (L683) | `int((self.metadata.anchor_slot[0] >= 0).sum().item())` | Cache result after each mutation, return cached int |
| `_current_frame_importance()` (L692) | `int(mask.sum().item())` | Use `mask.sum()` tensor, handle in calling code without sync |
| `_dedup_single_batch()` (L587) | `int(inverse.max().item())` | Defer or use tensor-based conditional logic |

Each replacement stores the computed integer result at the point of mutation (where a GPU sync is already unavoidable due to the mutation op) and reads the cached value on subsequent queries.

### 4.4 `src/ovggt/models/aggregator.py`

Add a helper `merge_cache_states_for_batch()` that:
1. Takes `List[List[LayerCacheState]]` (B × depth)
2. For each layer, concatenates K and V tensors from B caches along dim=0
3. Pads shorter caches to `max_tokens` with zeros
4. Returns padded `[B, H, max_tokens, D]` tensors and a `[B, max_tokens]` attention mask

### 4.5 `src/ovggt/heads/camera_head.py`

Support per-batch `past_key_values_camera`. The camera head's internal KV cache management already operates on `[B, H, N, D]` tensors. The key change is propagating per-batch caches through the sync_anchor_change logic.

### 4.6 `src/train_frontend.py`

- Remove `ValueError` guard at line 654
- In `train()`, pass `use_token_scorer` and `distill_loss_weight` from args to model constructor
- Ensure `freeze_stage_a_scorer_only()` is called between `load_student_pretrained_weights` (line 774) and `get_parameter_groups` (line 797) when `use_token_scorer=True`

### 4.7 Training Config

`config/train_frontend_blendedmvs.yaml`:
```yaml
batch_size: 4          # changed from 1
accum_iter: 1          # batch_size handles parallelism
```

`config/train_token_scorer.yaml` (if scorer is used):
```yaml
batch_size: 4          # changed from 1
accum_iter: 1
```

---

## 5. Side Optimization: GPU Sync Point Elimination

While refactoring the cache layer for B>1 support, we opportunistically eliminate the ~240+ `.item()`-induced host-device sync points per training step.

### 5.1 Mechanism

Each `.item()` call on a GPU tensor forces CUDA kernel completion and transfers a scalar to the host. In `frontend_cache.py`, four methods call `.item()` in the per-frame per-layer hot path:

- `has_anchor_tokens()` → called from `commit_pending_update_`
- `_compute_protected_count()` → called from `gather_`, `append_`, `commit_pending_update_`, `reorder_`
- `_current_frame_importance()` → called from `commit_pending_update_`
- `_dedup_single_batch()` → called from `apply_voxel_dedup_` → `commit_pending_update_`

### 5.2 Strategy: Cache-on-Mutation

For each sync point, we identify the mutation operations that change the underlying state and recompute the cached scalar at those points:

| Cached Value | Updated After |
|-------------|---------------|
| `_protected_count` (int) | `gather_`, `gather_per_batch_`, `append_`, `apply_keyframe_event_` |
| `_has_anchor_tokens` (bool) | `gather_`, `gather_per_batch_`, `append_`, `apply_keyframe_event_` |
| `_num_current_frame_tokens` (per-batch counts tracked as tensor) | `commit_pending_update_` |

On read, return the cached value without GPU sync. The cached values may be slightly stale between mutation and read, but all reads happen after the most recent mutation in the same frame processing step.

### 5.3 Additional Sync Reduction

- **GPU memory logging** (`train_frontend.py:1369-1375`): Increase `print_freq` from 10 to 50, or move `torch.cuda.memory_*` calls behind a `profile_memory=True` flag (default False).
- **`float(loss)` at line 1413**: This sync is unavoidable (needed for `isfinite` check and logging) but is only 1 sync per step (not per-frame or per-layer).

---

## 6. Migration Plan

### Phase 1: B=2 Correctness Validation
1. Implement all changes on `feature/batch-size-optimization` branch
2. Run 100 steps with B=1 (baseline) and B=2, compare:
   - Loss curves (should be within floating-point noise: <1e-4 relative difference)
   - ATE on chess/seq-03 after 1 epoch (should be identical within noise)
3. Fix any discrepancies before proceeding

### Phase 2: B=4 Throughput Validation
1. Increase batch_size to 4
2. Monitor GPU memory: expected increase is B × (cache state memory per sequence)
   - Cache state memory per sequence: ~24 layers × (K cache + V cache + metadata) ≈ 24 × (~50MB) ≈ 1.2GB
   - B=4 total cache overhead: ~4.8GB (should fit within typical 40GB/80GB GPU)
3. If OOM, reduce `frontend_total_budget` proportionally or enable gradient checkpointing

### Phase 3: Scale to B=8
1. Test with B=8 on a single GPU
2. If stable, test with B=8 × world_size=4 (effective batch 32)
3. Adjust learning rate linearly: `lr_new = lr_base × (B_new / B_old)`

---

## 7. Verification Strategy

### 7.1 Numerical Correctness

| Test | Method | Success Criteria |
|------|--------|-----------------|
| Loss equivalence | Run B=1 × 4 steps (accum_iter=4) vs B=4 × 1 step, compare loss | Relative difference < 1e-4 |
| Cache state equivalence | After 10 frames, compare `cache_states[0][0].k` between B=1 and B=4 (batch 0) | Element-wise identical |
| Head output equivalence | Compare depth/point/pose predictions for batch 0 between B=1 and B=4 runs | Max absolute error < 1e-7 |
| Gradient equivalence | Compare parameter gradients between B=1×4 and B=4×1 | Max relative error < 1e-4 |

### 7.2 Throughput Measurement

| Metric | Baseline (B=1) | Target (B=4) |
|--------|---------------|--------------|
| Step time (ms) | current | ≤ 1.3× baseline |
| Sequences/sec | current/step | ≥ 3× baseline |
| GPU memory (GB) | current | ≤ current + 6GB |

### 7.3 Regression Tests

- `tests/test_frontend_training_smoke.py` — pass with batch_size=2
- `tests/test_frontend_cache.py` — pass with multi-batch cache states
- `tools/test_phase2_smoke.py` — pass with use_token_scorer=True (remove stale `use_learned_scorer` field)

---

## 8. Risk Register

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Cache state padding causes attention artifacts | Pad with zeros + `-inf` mask; verify attention weights are zero on padded positions |
| 2 | Per-batch event divergence causes desynchronization | Training uses `fixed_interval` strategy with identical frame indices for all batches → events should be identical. Add assertion in Phase 1 to verify |
| 3 | GPU memory exceeds budget at B=4 | Enable `gradient_checkpointing=True` (40-60% memory reduction, 30% compute increase) or reduce `frontend_total_budget` |
| 4 | Sync point cache invalidation bug | Add debug assertions in Phase 1: after each frame, verify cached `_protected_count` matches ground-truth `.item()` value |
| 5 | DDP communication overhead increases with B | `ddp_static_graph=True` already set; gradient all-reduce communicates parameter gradients (fixed size regardless of B) |
| 6 | Data loader becomes bottleneck | `num_workers=8` with `persistent_workers=True` should handle B=4; increase to 12 if needed |

---

## 9. File Change Summary

| File | Operation | Lines Changed | Description |
|------|-----------|---------------|-------------|
| `src/ovggt/models/ovggt.py` | Modify | ~80 | `_inference_frontend`: per-batch state init, loop refactor, guard removal |
| `src/ovggt/utils/frontend_cache.py` | Modify | ~30 | Sync point elimination: cache-on-mutation for `.item()` calls |
| `src/ovggt/models/aggregator.py` | Modify | ~20 | `merge_cache_states_for_batch()` helper |
| `src/ovggt/heads/camera_head.py` | Modify | ~15 | Per-batch `past_key_values_camera` support |
| `src/ovggt/utils/frontend_keyframe.py` | Modify | ~5 | Verify per-batch isolation |
| `src/train_frontend.py` | Modify | ~10 | Guard removal, batch_size adaptation |
| `src/finetune_frontend.py` | Modify | ~3 | Guard removal |
| `config/train_frontend_blendedmvs.yaml` | Modify | 2 | `batch_size: 4` |
| `config/train_token_scorer.yaml` | Modify | 2 | `batch_size: 4` |
| **Total** | | **~167** | |

---

*Design completed 2026-05-26. See implementation plan for task breakdown.*
