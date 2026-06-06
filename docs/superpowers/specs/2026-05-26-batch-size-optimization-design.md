# Training Throughput Optimization: Remove batch_size=1 Constraint

> Multi-sequence batched training with independent per-sequence streaming state
> Date: 2026-05-26

> **Naming migration (2026-06-04):** `frontend_total_budget` → `frontend_per_layer_budget`
> (now per-layer; was 24-layer sum). Historical numeric values (200000, 10410) are
> preserved verbatim in calculations below — they refer to the old total-budget semantics.

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
| `src/train_frontend.py` | 653-658 | Startup check: `if int(args.batch_size) != 1: raise ValueError` |
| `src/finetune_frontend.py` | 251-255 | Same startup check in finetune script |

### 2.2 Singleton State Assumptions

| # | Component | Singleton | Problem with B>1 | Fix |
|---|-----------|-----------|-------------------|-----|
| 1 | `FrontendKeyframeManager` | One `active_keyframe_id`, one `history_slots` list, one `active_pose_encoding` | Each sequence needs independent keyframe slot management | §4.1: replicate B copies |
| 2 | `LayerCacheState` (per layer) | One `TokenMetadata` with `[1, N]` shape, shared anchor semantics | Dedup/eviction uses scalar `frame_id`; `_compute_protected_count` only checks `anchor_slot[0]` | §4.1: replicate B×depth copies |
| 3 | `PendingLayerUpdate` | `frame_id: int` (scalar) | Different batch elements may be at different frame indices (not applicable in training with aligned frames, but interface blocks it) | §4.8: scalar per-sequence (aligned in training) |
| 4 | `past_key_values_camera` | One list of `(k,v)` per trunk layer | Each sequence needs independent camera KV caches with potentially different anchor counts | §4.1: replicate B copies; §4.5: camera head per-batch cache |
| 5 | `_compute_protected_count` | `anchor_slot[0] >= 0` — indexes only batch 0 (L670-673) | Ignores all other batch elements | §4.1: removed by per-batch LayerCacheState (each has B=1 internally) |
| 6 | `gather_` B=1 fast-path | `_gather_single_batch_` via `index_select` | Multi-batch path (`gather_per_batch_`) pads to equal length, which would corrupt attention with garbage tokens | §4.1: removed by per-batch LayerCacheState (each handles B=1 internally) |

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

### 3.4 Architecture Decision: Sequential Aggregator + Batched Heads

**Chosen approach** (per Review §10 R10): Keep per-sequence cache states independent (B=1 internally), call the aggregator backbone B times per frame, then batch the head computations.

**Rationale**: The alternative "merge cache states" approach (concatenating B `LayerCacheState` K/V tensors into `[B, H, max_tokens, D]`) requires plumbing a KV padding mask through `Block.forward()` → `attn_residual_func()` → `Attention.forward()` → `F.scaled_dot_product_attention`. While not infeasible (the `attn_mask` parameter already exists at `attention.py:302`), the frontend_cache_mode path currently hardcodes `attn_mask=None`. Adding mask passthrough would require ~40 lines across 3 function signatures, plus constructing non-square attention masks for heterogeneous KV cache lengths.

**Sequential aggregator approach** avoids these changes entirely:

```
For each frame:
  For each batch element b in 0..B-1:
    aggregator(images[b:b+1], cache_states=cache_states[b])  # B=1, unchanged
  # Head computations batched:
  aggregated_list_batched = torch.cat([agg_outputs[b] for b in range(B)], dim=0)
  camera_head(aggregated_list_batched)   # [B, ...], handles batch natively
  depth_head(aggregated_list_batched)     # [B, ...], handles batch natively
  point_head(aggregated_list_batched)     # [B, ...], handles batch natively
```

**Trade-offs**:

| | Merge Cache States | Sequential + Batched Heads |
|---|---|---|
| Aggregator calls per frame | 1 (B>1) | B (B=1 each) |
| Attention mask changes | ~40 lines (new) | 0 lines |
| Cache state changes | ~60 lines (merge/split helpers) | 0 lines (existing B=1 code) |
| Head computation | 1 pass (batched) | 1 pass (batched, same) |
| GPU utilization | Max (all batched) | ~40-60% (aggregator B=1, heads batched) |
| Implementation risk | High (untested mask construction) | Low (existing code paths) |
| Throughput gain at B=4 | ~3x (target) | ~1.3-1.6x (heads ~30-40% of forward) |

**Recommendation**: Implement sequential + batched heads as Phase 1. If throughput improvement is insufficient, Phase 2 can add the merge cache states optimization with the mask plumbing.

### 3.5 Training Loss Equivalence

Loss is computed as the mean across batch elements: `loss = criterion_loss.mean() + distill_loss.mean()`. With the `last_scores` save/restore mechanism (§4.1), this is **approximately equivalent** to accumulating B separate `batch_size=1` steps.

Gradient contributions are near-identical because:
- Each sequence has independent cache state → independent forward paths in stateful components
- Shared backbone parameters receive gradients from all B sequences simultaneously
- The batch-mean reduction is linear

**Known deviation** (Review-29): `aggregator.last_scores` is shared across batch elements within a single frame's B sequential calls. The save/restore mechanism (§4.1) ensures each batch element sees the **same** initial `last_scores` (from the previous frame's batch B-1). This means dynamic budget allocation uses a slightly different history than true independent B=1 runs. The deviation is bounded and does not accumulate across frames (last_scores is overwritten each frame). In practice, with `fixed_interval` strategy, the score distributions are similar across sequences, making the deviation negligible.

---

## 4. Detailed Changes

### 4.1 `src/ovggt/models/ovggt.py` — `_inference_frontend` Refactor

**State initialization** (replaces lines 385-392):

```python
# Per-batch independent state (B copies of each singleton)
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

**Per-frame per-batch loop** (replaces the aggregator + camera head calls at ~L416-454):

```python
for i, frame in enumerate(frames):
    images_all = self._frame_image_to_sequence(frame["img"])  # [B, 1, C, H, W]

    # ===== Aggregator: sequential B calls (B=1 each) =====
    # Review-12: save/restore last_scores to isolate per-sequence dynamic budgets
    saved_last_scores = self.aggregator.last_scores.clone()
    frame_agg_outputs = []
    frame_pending_updates = []
    frame_distill_losses = []
    for b in range(B):
        images_b = images_all[b:b+1]  # [1, 1, C, H, W]
        # Restore last_scores for this batch element (or use zeros for first frame)
        self.aggregator.last_scores = saved_last_scores if b == 0 else saved_last_scores.clone()
        agg_tokens, ps, cs_b, pending, fdl = self.aggregator(
            images_b,
            cache_states=cache_states[b],
            use_cache=True,
            past_frame_idx=i,
            total_budget=total_budget,
            importance_weight=importance_weight,
            frontend_cache_config=self.frontend_cache_config,
        )
        frame_agg_outputs.append(agg_tokens)
        frame_pending_updates.append(pending)
        if fdl is not None:
            frame_distill_losses.append(fdl)
        cache_states[b] = cs_b
    # Restore last_scores to last batch element's state (any is fine, all identical in Phase 1)
    self.aggregator.last_scores = saved_last_scores

    # Review-18: average distill_loss across B (not sum) for equivalence
    if frame_distill_losses:
        avg_fdl = sum(frame_distill_losses) / len(frame_distill_losses)
        total_distill_loss = avg_fdl if total_distill_loss is None else total_distill_loss + avg_fdl

    # ===== Head input: concatenate per-batch aggregator outputs along dim=0 =====
    aggregated_tokens_list = []
    for layer_idx in range(len(frame_agg_outputs[0])):
        layer_cat = torch.cat([bo[layer_idx] for bo in frame_agg_outputs], dim=0)
        aggregated_tokens_list.append(layer_cat)

    # ===== Camera head: sequential B calls (same strategy as aggregator) =====
    # Review-14: camera head CANNOT be batched — PVC is per-sequence state.
    # Review-26: use actual camera_head API (num_iterations, anchor_token_count, etc.)
    # camera_head returns (dict, pvc) where dict has keys "abs_pose_enc", "rel_pose_enc"
    pose_enc_batch = []
    with self._disabled_autocast_context():
        for b in range(B):
            # Review-13: camera_anchor_token_count per single sequence (no sum across managers)
            camera_anchor_token_count = (
                None if i == 0
                else keyframe_managers[b].get_num_anchor_frames() * self.camera_num_iters
            )
            pose_enc_dict_b, past_key_values_camera[b] = self.camera_head(
                [agg[b:b+1] for agg in aggregated_tokens_list],
                num_iterations=self.camera_num_iters,
                past_key_values_camera=past_key_values_camera[b],
                use_cache=True,
                anchor_token_count=camera_anchor_token_count,
                pose_encoding_type=self._camera_pose_encoding_type_for_frontend(),
                return_pose_predictions=True,
                return_last_pose_only=True,
            )
            # camera_head returns dict with "abs_pose_enc" [1, 1, 9] and "rel_pose_enc" [1, 1, 9]
            pose_enc_batch.append(pose_enc_dict_b["abs_pose_enc"][:, 0, :])
    camera_pose = torch.cat(pose_enc_batch, dim=0)  # [B, 9]

    # ===== Depth/Point heads: batched (no per-sequence state) =====
    # Review-36: update closure to capture images_all (not images)
    # maybe_checkpoint_head requires closure for gradient checkpointing
    def depth_head_forward(*layer_tokens):
        return self.depth_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    def point_head_forward(*layer_tokens):
        return self.point_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens_list)
    pts3d, pts3d_conf = maybe_checkpoint_head(point_head_forward, *aggregated_tokens_list)
```

**Per-batch cache commit** (replaces lines 498-553):

```python
    # Review-15 (Phase 1): fixed_interval → all events identical, store batch 0 only
    events = []
    for b in range(B):
        event = keyframe_managers[b].update(
            frame_idx=i,
            depth=depth[b],
            pose_abs_enc=camera_pose[b],
            image_size_hw=(img_h, img_w),
        )
        events.append(event)
    if store_full_keyframe_schedule:
        keyframe_schedule.append(events[0])  # batch 0 representative

    for b in range(B):
        # Review-17: build per-batch frame_metadata_base for commit
        slot_id = keyframe_managers[b].get_active_keyframe_id()
        current_local_to_world = keyframe_managers[b].get_active_local_to_world()
        for layer_idx in range(self.aggregator.depth):
            pending = frame_pending_updates[b][layer_idx]
            if pending is None:
                continue
            frame_metadata_base = build_frame_token_metadata_base(
                depth=depth[b:b+1],
                depth_conf=depth_conf[b:b+1],
                pose_enc=camera_pose[b:b+1],
                image_size_hw=(img_h, img_w),
                patch_size=self.aggregator.patch_size,
                patch_start_idx=self.aggregator.patch_start_idx,
                frame_id=i,
                keyframe_id=slot_id,
                slot_id=slot_id,
            # Review-37: KeyframeEvent.anchor_slot is an int attribute, not a method
            anchor_slot=events[b].anchor_slot,
                importance=pending.importance_current,
                active_local_to_world=current_local_to_world,
            )
            current_metadata = frame_metadata_base.with_importance(pending.importance_current)
            cache_states[b][layer_idx].apply_keyframe_event_(events[b])
            score = cache_states[b][layer_idx].commit_pending_update_(
                pending_update=pending,
                current_metadata=current_metadata,
                config=self.frontend_cache_config,
                intra_frame_keep_ratio=intra_frame_keep_ratio,
                attn_module=self.aggregator.global_blocks[layer_idx].attn,
            )
            # Review-12: propagate score to aggregator for dynamic budget
            if score is not None:
                self.aggregator.last_scores[layer_idx] = score
```

**Pass through active_pose_encoding**: After the camera_head loop, compose absolute poses:

```python
    # Review-16: stack per-batch active pose encodings for absolute pose composition
    # Review-35: use collected rel_pose_enc_batch (not undefined pose_enc_list)
    with self._disabled_autocast_context():
        if self.frontend_pose_encoding_type == ABS_POSE_ENCODING:
            pass  # camera_head already outputs absolute poses in camera_pose
        else:
            active_poses = torch.stack([
                keyframe_managers[b].get_active_pose_encoding()
                for b in range(B)
            ], dim=0)  # [B, 9]
            rel_pose_enc = torch.cat(rel_pose_enc_batch, dim=0).unsqueeze(1)  # [B, 1, 9]
            camera_pose_abs = compose_absolute_from_relative(
                active_poses.unsqueeze(1), rel_pose_enc, image_size_hw=(img_h, img_w)
            )[:, 0, :]
```

**Post-commit operations** (Review-32, Review-33: lines 562-609 currently outside design scope):

```python
    # Review-32: per-batch camera_head.apply_keyframe_event
    for b in range(B):
        past_key_values_camera[b] = self.camera_head.apply_keyframe_event(
            past_key_values_camera[b],
            events[b],
            num_cam_iters=self.camera_num_iters,
        )

    # Build per-frame result (keyed by batch)
    res_gpu = {
        "depth": depth,              # [B, H, W, ...] — batched
        "depth_conf": depth_conf,    # [B, ...]
        "pts3d": pts3d,              # [B, ...]
        "pts3d_conf": pts3d_conf,
        "camera_pose": camera_pose,  # [B, 9] — batched
        "camera_pose_rel": rel_pose_enc[:, 0, :] if rel_pose_enc_batch else None,
    }
    if frame_writer is not None:
        # frame_writer callback receives batched data; caller adapts
        frame_writer(i, frame, res_gpu)
    if cache_results:
        res_out = self._maybe_move_dict_to_cpu(res_gpu) if move_to_cpu else res_gpu
        all_ress.append(res_out)

    # Review-33: variable names match new code
    del aggregated_tokens_list
    del frame_pending_updates
```

**Guard removal**: Change `_validate_frontend_batch_size` to accept `B >= 1` (remove the `ref_batch_size != 1` check).

### 4.2 `src/ovggt/utils/frontend_keyframe.py`

Minimal changes. `update()` already returns `KeyframeEvent` per-call. Only verification needed: ensure internal state is fully reset between calls (no cross-contamination from previous batch element).

### 4.3 `src/ovggt/utils/frontend_cache.py` — Sync Point Elimination

| Location | Current | Replacement |
|----------|---------|-------------|
| `has_anchor_tokens()` (L103-104) | `bool((self.anchor_slot >= 0).any().item())` | Use cached `self._cached_has_anchor` updated after reorder/evict |
| `_compute_protected_count()` (L670-673) | `int((self.metadata.anchor_slot[0] >= 0).sum().item())` | Cache result after each mutation, return cached int |
| `_current_frame_importance()` (L675-686) | `int(mask.sum().item())` at L682 | Track as tensor; replace `int(mask.sum().item())` with `mask.sum()` and handle downstream without sync |

Note: `_dedup_single_batch` (L516-619) is fully vectorized and contains **zero** `.item()` calls. The estimate of "~240+ sync points" comes from 3 `.item()` sites × ~24 layers × 10 frames (≈720 calls maximum; actual count is lower due to conditional branches in `commit_pending_update_`).

Each replacement stores the computed integer result at the point of mutation (where a GPU sync is already unavoidable due to the mutation op) and reads the cached value on subsequent queries.

### 4.4 `src/ovggt/models/aggregator.py`

**No changes required.** With the sequential approach (§3.4), the aggregator is called B times with B=1 internally. Each call uses per-sequence `cache_states[b]` (a `List[LayerCacheState]` where each has B=1 internally). No merge/split logic, no padding masks, no signature changes.

The aggregator's existing `frontend_cache_mode` path (L305-337) already handles B=1 correctly. The TokenScorer distill_loss flow (implemented on `feature/token-scorer`) works unchanged because each aggregator call produces per-sequence `frame_distill_loss`.

### 4.5 `src/ovggt/heads/camera_head.py`

Support per-batch `past_key_values_camera`. The camera head's internal KV cache management already operates on `[B, H, N, D]` tensors. Changes:

1. **`past_key_values_camera` initialization**: Change from `List[Optional[Tuple]]` (length=trunk_depth) to `List[List[Optional[Tuple]]]` (B × trunk_depth). Each batch element gets its own camera cache.

2. **`camera_head(...)` call site** (currently near L438-454 of ovggt.py): Pass `past_key_values_camera=PVC[b]` where `b` is the batch index.

3. **`apply_keyframe_event` call** (currently near L577-578): This method propagates keyframe slot reassignments to the camera head's KV cache. Since each batch element now has its own cache, call `camera_head.apply_keyframe_event(pvc[b], ...)` independently per batch.

4. **`camera_anchor_token_count`** (currently L440): Computed as `keyframe_manager.get_num_anchor_frames() * camera_num_iters`. Change to per-batch: `keyframe_managers[b].get_num_anchor_frames() * camera_num_iters`.

Internal `sync_anchor_change` method (camera_head.py L317-390) operates on `[B, H, N, D]` tensors natively — no internal changes needed. Only the calling convention changes at the ovggt.py level.

### 4.6 `src/train_frontend.py` and `src/finetune_frontend.py`

**`train_frontend.py`:**
- Remove `ValueError` guard at line 653
- In `train()`, pass `use_token_scorer` and `distill_loss_weight` from args to model constructor
- Ensure `freeze_stage_a_scorer_only()` is called between `load_student_pretrained_weights` (L792) and `get_parameter_groups` (L818) when `use_token_scorer=True`

**`finetune_frontend.py` (Review-34, Review-38):**
- Remove `ValueError` guard at line 251
- **B>1 with FrontendSupervisedLoss**: `_build_point_targets` (`frontend_supervised.py:184-186`) strips batch dim via `camera_intrinsics[0]` — incorrect for B>1. Fix: iterate per-batch element or broadcast.
- **TokenScorer in finetune**: Not supported in Phase 1. `finetune_full_model` does not initialize scorers or handle `distill_loss`. B>1 finetune is limited to no-scorer `FrontendSupervisedLoss` only.

### 4.6a DataLoader Custom Collate Function (Review-25)

**New file or addition to `src/dust3r/datasets/`**: The `default_collate` crashes with `TypeError` when B>1 because dataset views contain string fields (`dataset`, `label`). Add a collate function that handles mixed types:

```python
def frontend_collate_fn(batch):
    """Collate B sequences, each with num_frames views. Stack tensors, preserve strings."""
    num_frames = len(batch[0])
    collated = []
    for frame_idx in range(num_frames):
        frame_dicts = [sample[frame_idx] for sample in batch]
        collated_frame = {}
        for key in frame_dicts[0]:
            vals = [fd[key] for fd in frame_dicts]
            if isinstance(vals[0], (torch.Tensor, np.ndarray)):
                collated_frame[key] = torch.stack([torch.as_tensor(v) for v in vals])
            elif isinstance(vals[0], str):
                collated_frame[key] = vals  # List[str], not stacked
            else:
                collated_frame[key] = vals
        collated.append(collated_frame)
    return collated
```

Register in DataLoader: `collate_fn=frontend_collate_fn`. The `get_data_loader` function in `dust3r/datasets/__init__.py` already accepts `collate_fn` parameter — pass it through from config.

### 4.7 Training Config

`config/train_frontend_finetune.yaml`:
```yaml
batch_size: 4          # changed from 1
accum_iter: 1          # batch_size handles parallelism
```

`config/train_token_scorer.yaml` (if scorer is used):
```yaml
batch_size: 4          # changed from 1
accum_iter: 1
```

### 4.8 `build_frame_token_metadata_base` — Per-Batch Adaptation

**Location**: Called at approximately L529-542 within `_inference_frontend`, currently with scalar arguments:
```python
frame_metadata_base = build_frame_token_metadata_base(
    depth=depth,           # [B, H, W] or [B, H, W, 1]
    depth_conf=depth_conf, # [B, ...]
    pose_enc=camera_pose, # [B, 9]
    ..., frame_id=i, keyframe_id=..., slot_id=..., anchor_slot=...
)
```

For B>1, the tensor arguments are already batched (`[B, ...]`).
The scalar arguments (`frame_id`, `keyframe_id`, `slot_id`, `anchor_slot`) currently
describe per-frame values that are identical across all batch elements when using
`fixed_interval` strategy (all sequences share the same frame index `i`).

**Decision**: Keep scalar arguments as-is for Phase 1. For fixed_interval strategy,
`frame_id=i` and `keyframe_id` are identical across batches. For coverage-based
strategies (Phase 2+), expand to per-batch scalar values. Document this as
a known limitation in §8 Risk #2.

---

## 5. Side Optimization: GPU Sync Point Elimination

While refactoring the cache layer for B>1 support, we opportunistically eliminate the ~240+ `.item()`-induced host-device sync points per training step.

### 5.1 Mechanism

Each `.item()` call on a GPU tensor forces CUDA kernel completion and transfers a scalar to the host. In `frontend_cache.py`, three methods call `.item()` in the per-frame per-layer hot path (verified by `grep`):

- `has_anchor_tokens()` (L104) → called from `commit_pending_update_`
- `_compute_protected_count()` (L673) → called from `gather_`, `append_`, `commit_pending_update_`, `reorder_`
- `_current_frame_importance()` (L682) → called from `commit_pending_update_`

Note: `_dedup_single_batch` contains zero `.item()` calls — verify with `grep -n "\.item()" frontend_cache.py`.

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
3. If OOM, reduce `frontend_per_layer_budget` proportionally or enable gradient checkpointing

### Phase 3: Scale to B=8
1. Test with B=8 on a single GPU
2. If stable, test with B=8 × world_size=4 (effective batch 32)
3. **Do NOT adjust learning rate**: loss is mean-reduced (not summed), so gradient magnitude is independent of B. Linear LR scaling (Goyal et al.) does NOT apply here. Keeping `lr` unchanged preserves training dynamics across all batch sizes.

---

## 7. Verification Strategy

### 7.1 Numerical Correctness

| Test | Method | Success Criteria |
|------|--------|-----------------|
| Loss equivalence | Run B=1 × 4 steps (accum_iter=4) vs B=4 × 1 step, compare loss | Relative difference < 1e-3 (bf16 accumulation order causes non-exact match) |
| Head output equivalence | Compare depth/point/pose predictions for same 4 sequences between B=1×4 and B=4×1 runs | Max relative error < 1e-3 per sequence |
| Gradient equivalence | Compare parameter gradients between B=1×4 and B=4×1 | Max relative error < 1e-3 |

> **R28 note**: Cache state equivalence ("element-wise identical") is impossible because `last_scores` is shared across batch elements (see §3.5 caveat). Different eviction budget at frame 2+ produces different cache contents. Use loss-based and head-output-based equivalence instead.

### 7.2 Throughput Measurement

| Metric | Baseline (B=1) | Target (B=4) |
|--------|---------------|--------------|
| Step time (ms) | current | ≤ 1.8× baseline |
| Sequences/sec | current/step | ≥ 2.2× baseline |
| GPU memory (GB) | current | ≤ current + 3GB |
| Aggregator calls/frame | 1 | B (B=1 each, hot in cache) |
| Head compute | 1 pass (B=1) | 1 pass (B=4 batched) |

### 7.3 Regression Tests

The following tests do not yet exist and must be **created** as part of implementation:

- `tests/test_frontend_training_smoke.py` — verify training runs without error with batch_size=2
- `tests/test_frontend_cache.py` — verify multi-batch cache states are isolated (set a key in batch[0], confirm batch[1] unaffected)

Existing test that needs update:
- `tools/test_phase2_smoke.py` (L22) — remove stale `use_learned_scorer=True` from `FrontendCacheConfig` (field does not exist)

---

## 8. Risk Register

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Cache state padding causes attention artifacts | Pad with zeros + `-inf` mask; verify attention weights are zero on padded positions |
| 2 | Per-batch event divergence causes desynchronization | **Phase 1 (fixed_interval only)**: All sequences share identical `frame_idx=i`, so keyframe events are guaranteed identical. Add assertion: `assert all(e.event_type == events[0].event_type for e in events)`. **Phase 2+ (coverage strategies)**: Events may diverge because depth/pose values differ across sequences. Per-batch event list already handles this correctly — no assertion, no fix needed |
| 2a | `build_frame_token_metadata_base` scalar args diverge with coverage strategy (see §4.8) | **Phase 1**: Not applicable (fixed_interval, all values identical). **Phase 2+**: Expand `frame_id`/`keyframe_id`/`slot_id`/`anchor_slot` to per-batch |
| 3 | GPU memory exceeds budget at B=4 | Enable `gradient_checkpointing=True` (40-60% memory reduction, 30% compute increase) or reduce `frontend_per_layer_budget` |
| 4 | Sync point cache invalidation bug | Add debug assertions in Phase 1: after each frame, verify cached `_protected_count` matches ground-truth `.item()` value |
| 5 | DDP communication overhead increases with B | `ddp_static_graph=True` already set; gradient all-reduce communicates parameter gradients (fixed size regardless of B) |
| 6 | Data loader becomes bottleneck | `num_workers=8` with `persistent_workers=True` should handle B=4; increase to 12 if needed |

---

## 9. File Change Summary

Revised estimate based on the sequential + batched heads approach (§3.4):

| File | Operation | Lines Changed | Description |
|------|-----------|---------------|-------------|
| `src/ovggt/models/ovggt.py` | Modify | ~120 | `_inference_frontend`: per-batch state init, B× aggregator loop, head input concatenation, per-batch cache commit, guard update |
| `src/ovggt/utils/frontend_cache.py` | Modify | ~30 | Sync point elimination: cache-on-mutation for `.item()` calls |
| `src/ovggt/models/aggregator.py` | Modify | 0 | No changes (sequential B=1 calls) |
| `src/ovggt/heads/camera_head.py` | Modify | ~10 | Verify batch dimension handling in `apply_keyframe_event` |
| `src/ovggt/utils/frontend_keyframe.py` | Modify | ~5 | Verify per-batch isolation |
| `src/train_frontend.py` | Modify | ~15 | Guard removal, distill_loss accumulation across batches |
| `src/finetune_frontend.py` | Modify | ~3 | Guard removal |
| `config/train_frontend_finetune.yaml` | Modify | 2 | `batch_size: 4` |
| `config/train_token_scorer.yaml` | Modify | 2 | `batch_size: 4` |
| **Total** | | **~187** | |

---

## 10. Code Review Log

> 以下为对照实际代码库 (`src/ovggt/**/*.py`, `src/train_frontend.py`) 审查后发现的问题。
> 审查日期: 2026-05-26

### Review-1（严重）：§3.4 "merge cache states" 方案在注意力层不可行

**问题**：设计提出将 B 个 `LayerCacheState` 的 K/V 拼接成 `[B, H, max_tokens, D]`，对短序列 zero-pad 并用 `-inf` mask 防止 attention 到 padding token。但实际代码中：

1. `attention.py:367-374` 使用 `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)`，`attn_mask` 用于 causal masking（未来帧屏蔽），**没有** KV cache padding mask 的传递机制。
2. `frontend_cache_mode` 路径中 `attn_mask=None`（block.py:172, attention.py 中无 causal mask），因为缓存模式下 attention 是当前帧 token 对所有缓存 token。但要支持 padded cache，需要新增一个 KV mask 路径。
3. 当前 `attention.py` 的 `forward` 签名没有接收 KV cache mask 的参数。添加此支持需要修改 `Attention.forward`、`Block.forward`、`_process_global_attention` 的签名，以及 SDPA 调用方式。

**影响**：这不是 §9 声称的 "~20 行" 修改，而是需要重新设计 attention 层的 mask 传递机制。如果用 PyTorch SDPA 的 `attn_mask` 参数同时处理 causal mask + KV padding mask，需要构造 `[B, H, N_q, N_kv]` 的复合 mask，在 N_kv 上万时内存开销显著。

**建议替代方案**：在 `_inference_frontend` 层面用循环处理 B 个序列（每个序列保持 B=1 内部处理），仅在 depth_head/point_head/camera_head 等 head 层面做真正的 batch 化。这避免了 attention mask 问题，且 backbone 参数已 hot in cache，多次 B=1 forward 的 kernel launch overhead 有限。

### Review-2（严重）：数据 pipeline 不支持 B 个独立序列

**问题**：设计未讨论数据加载层面的变化。

当前训练流程：
- `train_frontend.py:420-426`：`model(batch, ...)` 其中 `batch` 是 **frame dict 列表**（长度 num_views），每个 frame 有 `frame["img"]` shape `[C, H, W]`。
- `ovggt.py:_inference_frontend` 接收 `frames: List[Dict]`，循环 `for i, frame in enumerate(frames)`，每个 frame 是一个序列的一帧。

对于 B>1 训练：
- 数据加载器需要产出 B 个独立序列，每个有 num_views 帧
- `frames` 的数据结构需要从 `List[Dict]`（单序列）变为 `List[List[Dict]]`（B 个序列）或 `List[Dict]`（每帧包含 B 个 batch 元素的 img `[B, C, H, W]`）
- `_frame_image_to_sequence` 需要 batch 化
- `frame["valid_mask"]` 等字段也需要 batch 化

**影响**：这涉及 `dust3r.datasets.get_data_loader`、`FrontendDistillLoss.finalize_from_stream`（接收 per-sequence teacher/student predictions）、`accumulate_student_frame` 回调等多个组件的重构。设计完全未提及。

### Review-3（严重）：`frame_processor` 回调不支持 B>1

**问题**：`train_frontend.py:405-418` 的 `accumulate_student_frame` 回调逐帧收集 per-sequence 数据：

```python
def accumulate_student_frame(frame_idx, frame_gt, student_pred):
    student_camera_pose_rel.append(student_pred["camera_pose_rel"])
    ...
```

这些 list 收集的是**单个序列**的 per-frame predictions。对于 B>1：
- 回调需要被调用 B 次（每 batch 元素一次），或
- `student_pred` 需要 batch 化（`camera_pose_rel` shape `[B, ...]`）
- `FrontendDistillLoss.finalize_from_stream` 接收的 `student_camera_pose_rel` 等参数需要从 `List[Tensor]` 变为 `List[List[Tensor]]`

设计未提及此变化。

### Review-4（重要）：内存估算仅适用于低 budget 配置

**问题**：§6 Phase 2 估算 "Cache state memory per sequence: ~1.2GB"。此估算基于 `frontend_per_layer_budget=10410`（TokenScorer 配置）。

实际计算（budget=10410）：10410 tokens × 16 heads × 64 dim × 2 bytes (bf16) × 2 (K+V) ≈ 42.7MB/layer × 24 layers ≈ 1.02GB。✓

但如果使用默认 `train_frontend_blendedmvs.yaml` 的 `total_budget=200000`（旧语义，24层总和）：
200000 × 16 × 64 × 2 × 2 ≈ 819MB/layer × 24 layers ≈ **19.7GB per sequence**。

B=4 需要 ~79GB，超出 A800 80GB（还要留空间给 activations、optimizer states 等）。

**建议**：明确说明 B>1 仅适用于低 budget 场景（如 `total_budget ≤ 20000`），或需要在增大 batch 的同时按比例降低 budget。

### Review-5（重要）：行号引用与实际代码不符

**问题**：
- §5.3 引用 `train_frontend.py:1369-1375`（GPU memory logging），实际在 **line 574**
- §5.3 引用 `float(loss)` at line 1413，实际在 **lines 242, 464, 595**
- 这些行号偏差表明设计文档可能基于不同版本的代码编写

**建议**：更新所有行号引用，或改为使用函数名/方法名引用而非硬编码行号。

### Review-6（重要）：~167 行修改量严重低估

**问题**：§9 声称总计 ~167 行修改。实际需要：

| 组件 | 设计估计 | 实际估计 | 原因 |
|------|---------|---------|------|
| `ovggt.py` | ~80 | ~150 | per-batch 循环 + PendingLayerUpdate split + frame 结构适配 |
| `frontend_cache.py` | ~30 | ~30 | 同意 |
| `aggregator.py` | ~20 | ~60 | merge/split helper + attention mask 传递 |
| `attention.py` | 0（未提及） | ~40 | KV padding mask 支持 |
| `camera_head.py` | ~15 | ~30 | per-batch camera cache 管理 |
| `train_frontend.py` | ~10 | ~60 | data pipeline + frame_processor + batch 感知 |
| 数据加载层 | 0（未提及） | ~50 | batch 多序列 collation |
| `FrontendDistillLoss` | 0（未提及） | ~40 | per-batch loss accumulation |
| **合计** | **~167** | **~460+** | |

### Review-7（中等）：`PendingLayerUpdate` 的 split 逻辑未说明

**问题**：当前 aggregator 返回一个 `PendingLayerUpdate` per layer，其中 `k_current [B, H, N, D]` 和 `importance_current [B, N]` 已 batch 化。设计提出 per-batch `LayerCacheState`（每个内部 B=1），但未说明如何将 batched `PendingLayerUpdate` 拆分为 B 份分别 commit。

具体缺失步骤（`ovggt.py`，在 aggregator 调用之后）：
```python
# 需要将 batched pending_update 拆分为 per-batch
for b in range(B):
    for layer_idx, pending_update in enumerate(pending_updates):
        if pending_update is None:
            continue
        per_batch_update = PendingLayerUpdate(
            k_current=pending_update.k_current[b:b+1],
            v_current=pending_update.v_current[b:b+1],
            importance_current=pending_update.importance_current[b:b+1],
            frame_id=pending_update.frame_id,
            cache_budget=pending_update.cache_budget,
        )
        cache_states[b][layer_idx].commit_pending_update_(per_batch_update, ...)
```

注意：这要求 aggregator 仍然以 `B>1` 运行并返回 batched results，然后 split。但设计 §3.4 的 "merge cache states" 方案要求 aggregator 也处理 merged cache——两个方案之间存在张力。

### Review-8（中等）：`gather_per_batch_` 的 padding 策略与设计的 "zero-pad + -inf mask" 矛盾

**问题**：现有 `gather_per_batch_`（`frontend_cache.py:200-340`）的 padding 策略是**复制最后一个 token** 而非 zero-pad：

```python
# L266-267: padding with last token's values
k_pad = k_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()
v_pad = v_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()
```

设计的 §3.4 声称用 "zero-pad + -inf mask"。两种 padding 方式有不同的 attention 行为：
- 复制 padding：padding token 的 attention weight 非零但值正确
- Zero pad + mask：padding token 的 attention weight 被强制为 0

如果采用设计的 merge approach，需要统一 padding 策略，否则 `gather_per_batch_` 的 pad token 会被 attention 正常 attended。

### Review-9（中等）：`_compute_protected_count` 仅检查 `anchor_slot[0]`

**问题**：§2.2 #5 正确指出 `_compute_protected_count`（`frontend_cache.py:670-673`）只检查 `anchor_slot[0]`（batch element 0）。但在 B>1 场景下，即使采用 per-batch `LayerCacheState`（内部 B=1），此方法仍然安全——因为每个 LayerCacheState 只有一个 batch element。

**结论**：设计 §2.2 #5 的分析正确，且 per-batch state 方案确实解决了此问题。无需额外修复。

### Review-10（信息性）：建议考虑 "Sequential B=1 with batched heads" 替代方案

**问题**：设计未考虑以下更简单的替代方案：

**方案：在 `_inference_frontend` 中循环处理 B 个序列**
- Aggregator 和 attention 层保持 B=1 不变
- 每帧对 B 个序列分别调用 aggregator（B 次 forward）
- Depth/point/camera head 已经支持 B>1，可以将 B 个序列的 head 输入 concatenate 后一次 forward
- Cache state 完全独立（B 组 B=1 状态），无需 merge/split/padding/mask

**优势**：
- 零 attention 层改动
- 零 data pipeline 改动（每个序列仍是 frame dict 列表）
- Cache state 管理代码不变
- 总代码修改量 ~100 行

**劣势**：
- Aggregator backbone forward B 次（但权重 hot in cache，kernel launch overhead 低）
- GPU 利用率提升主要来自 head 层面的 batch 化

**性能预估**：head 计算占总 forward ~30-40%，batch 化后 B=4 约提升 20-30% throughput。远低于设计目标的 3× 但实现风险极低。

### Review-11（信息性）：`num_tokens()` 和 `has_anchor_tokens()` 也有 `.item()` 隐式调用

**问题**：`frontend_cache.py:162-165` 的 `num_tokens()` 方法：
```python
def num_tokens(self) -> int:
    if self.k is None:
        return 0
    return int(self.k.shape[2])
```
`self.k.shape[2]` 不触发 GPU sync（shape 是 CPU 元数据），所以这里不需要修改。但设计 §5.1 未区分哪些 `.item()` 调用真正导致 GPU sync，哪些只是 CPU 操作。建议明确说明。

### 审查总结与处理状态

| 级别 | 编号 | 标题 | 处理 |
|------|------|------|------|
| 严重 | R1 | attention mask 透传复杂度 | **已处理** — 选择 sequential aggregator 方案绕过，§3.4 已重写 |
| ~~严重~~ | ~~R2~~ | ~~data pipeline 不支持 B>1~~ | **已撤回** — `_frame_image_to_sequence` L802 已处理 `[B, C, H, W]` |
| 严重 | R3 | frame_processor 回调 B>1 | **已处理** — §4.1 新增 per-batch 循环，head 输出天然 batch 化 |
| 重要 | R4 | 内存估算仅低 budget | **保留** — §6 已标注 budget 约束 |
| 重要 | R5 | 行号偏差 | **已修复** — 前两轮 review 已纠正所有行号 |
| 重要 | R6 | ~167 行低估 | **已修复** — §9 更新为 ~187 行（sequential 方案实际规模） |
| 中等 | R7 | PendingLayerUpdate split | **已处理** — §4.1 包含 per-batch split 代码 |
| 中等 | R8 | padding 策略矛盾 | **已解决** — sequential 方案无需 merge/padding |
| 中等 | R9 | _compute_protected_count B>1 | **确认安全** — per-batch LayerCacheState 内部 B=1 |
| 信息性 | R10 | sequential + batched heads | **已采纳** — 替代原 §3.4 merge 方案 |
| 信息性 | R11 | .item() 精确分类 | **已记录** — 3 处（L104/L673/L682），§5.1 已修正 |

**总体评估**：设计已从 "merge cache states" 重构为 "sequential aggregator + batched heads"。R10 替代方案已采纳为主设计。剩余风险可控，可进入实现阶段。

### 第二轮审查（2026-05-26）

> 对照实际代码逐行验证的第二轮审查。聚焦于第一轮（R1-R11）未覆盖的问题。

### Review-12（严重，已验证）：`aggregator.last_scores` 是模块级共享状态，破坏跨序列 budget 隔离

**验证**：已通过代码审查确认。`last_scores` 是 `aggregator.py:165` 的 `torch.zeros(self.depth)` 张量（无 batch 维）。`_calculate_dynamic_budgets`（L546）读取它；`_inference_frontend`（`ovggt.py:558-559`）写入 `self.aggregator.last_scores[layer_idx] = score`。

**已修复**：§4.1 在 sequential 循环中加入 save/restore 逻辑。

### Review-13（伪代码 bug，已修复）：`camera_anchor_token_count` 错误地跨 B 个 keyframe_manager 求和

**验证状态**：当前 B=1 代码（`ovggt.py:434`）正确——只有 1 个 manager，无 sum。**问题在设计的 §4.1 伪代码**（错误地对 B 个 manager 求和）。

**已修复**：§4.1 改用 sequential camera_head，per-batch `keyframe_managers[b].get_num_anchor_frames()` 不跨 manager 求和。

### Review-14（伪代码 bug，已修复）：§4.1 camera head PVC 设为 None——丢弃整个 camera KV cache

**验证状态**：当前 B=1 代码（`ovggt.py:435-444`）正确传递和返回 PVC。**问题在设计的 §4.1 伪代码**（错误地设为 `None` 并丢弃返回值）。

**已修复**：§4.1 改为 sequential camera_head（B 次 B=1 调用），每序列维护独立的 PVC。Camera head 不再批量化。

### Review-15（已确认，已处理）：`keyframe_schedule` 是扁平列表——`finalize_from_stream` 索引在 B>1 时错乱

**验证**：已确认。`keyframe_schedule` 是 `List[object]`（`ovggt.py:401`），每帧一个 event。`finalize_from_stream`（`frontend_distill.py:396`）按 `frame_idx` 枚举。

**已修复**：Phase 1（fixed_interval）所有 event 相同，存储 batch 0 代表事件。

### Review-16（已确认，已处理）：`active_pose_encoding` 来自单个 keyframe_manager 用于所有 B 序列

**验证**：已确认。是设计 §4.1 伪代码的问题。Phase 1（fixed_interval）安全——所有 manager 在同一帧产生相同 active pose。

**已修复**：§4.1 新增 per-batch active_pose_encoding stack。

### Review-17（伪代码 bug，已修复）：`build_frame_token_metadata_base` 在 per-batch commit 循环中缺失

**验证状态**：当前 B=1 代码（`ovggt.py:536-549`）在 `commit_pending_update_` 之前正确调用。**问题在设计的 §4.1 伪代码**（遗漏了该调用）。

**已修复**：§4.1 的 commit 循环中新增完整的 `build_frame_token_metadata_base` + `with_importance` 构建。

### Review-18（已确认，已修复）：Distill loss 跨 B 求和而非取平均——破坏 loss 等价性

**验证**：数学上已确认。`sum(fdl_i)` 使有效权重变成 `B × distill_loss_weight`。

**已修复**：§4.1 改为 `sum(frame_distill_losses) / len(frame_distill_losses)`（跨 B 取平均）。

**问题**：ovggt.py:562-575 创建 `KeyframePacket` 时使用 batched tensor（B>1 时 `camera_pose [B, 9]`、`patch_features [B, ...]`），但 `KeyframePacket`（frontend_keyframe.py:50-58）存储原 tensor。下游消费者期望 per-keyframe 标量。

**影响**：Eval 模式下 B>1 会将 batched tensor 存入期望 per-keyframe 数据的字段。如果 `export_keyframe_packets=True`，静默破坏评估输出。

**建议**：B>1 时每帧每 event 创建 B 个 `KeyframePacket`（每个 batch 元素一个），或限制 B>1 仅用于训练模式。

### Review-20（中等）：`current_keyframe_id`/`current_local_to_world` 来自单个 keyframe_manager

**问题**：ovggt.py:516-517 从单个 `keyframe_manager` 设置这些值。设计未在 per-batch commit 循环内重新设置 per-batch 值。

**影响**：Phase 1（fixed_interval）安全。Phase 2+ 不同 batch 元素可能有不同的 active keyframe。

**建议**：移入 per-batch 循环：
```python
for b in range(B):
    current_keyframe_id = keyframe_managers[b].get_active_keyframe_id()
    current_local_to_world = keyframe_managers[b].get_active_local_to_world()
```

### Review-21（中等）：`_infer_image_hw` 返回单一 (H, W)——所有 B 序列必须共享分辨率

**问题**：`_infer_image_hw`（ovggt.py:783-791）从 `frames[0]["img"]` 返回分辨率。B>1 时仅使用 batch 0 的分辨率。如果序列分辨率不同，aggregator 的 patch grid size 不同，导致 shape 不匹配。

**影响**：混合分辨率序列会崩溃。训练时数据加载器通常强制统一分辨率（安全），但应作为约束记录。

**建议**：在 §8 Risk Register 中添加：B>1 要求 batch 内所有序列分辨率一致。可在初始化时添加 assertion。

### Review-22（中等）：`_build_keyframe_mask` 产出单一 `[num_frames]` mask——不支持 per-batch

**问题**：`finalize_from_stream`（frontend_distill.py:232-236）调用 `_build_keyframe_mask` 返回单一 `[num_frames]` mask。Phase 2+ 需要形状 `[B, num_frames]`。

`_build_gt_relative_pose_targets`、`_compose_absolute_pose_from_relative_with_fixed_anchors` 等均假设单一共享 schedule。

**影响**：Phase 1 安全（fixed_interval 产生相同 event）。Phase 2+ 需要重构 `keyframe_schedule` 接口、`finalize_from_stream` 签名及所有下游 mask 消费者。

**建议**：记录为 Phase 2 阻塞项。

### Review-23（信息性）：`camera_head.last_scores` 存在相同的共享状态风险

**问题**：`camera_head.last_scores`（camera_head.py:44）是模块级 tensor。如果后续改为 B 次 sequential 调用 camera_head，会有与 Review-12 相同的交叉污染问题。

**影响**：当前设计（单次 batched 调用）无影响。如果调用方式改变则需关注。

**建议**：Phase 1 无需操作。如果 camera_head 调用方式变更时需注意。

### Review-24（信息性）：`aggregator.last_scores` 非 buffer——DDP 不同步

**问题**：`self.last_scores` 是 `torch.zeros(self.depth)`（非 buffer），DDP `static_graph=True` 不会同步。各 rank 维护独立的 score。

**影响**：B=1 时也如此，不是退化。如果 rank 间 budget 差异过大会影响收敛，但风险低。

**建议**：Phase 1 无需操作。

---

### 第二轮审查总结

| 级别 | 数量 | 编号 |
|------|------|------|
| 严重 | 4 | R12（`last_scores` 共享状态）、R13（anchor count 求和）、R14（camera cache 丢弃）、R15（`keyframe_schedule` 错乱） |
| 重要 | 4 | R16（`active_pose_encoding` 单 manager）、R17（`build_frame_token_metadata_base` 缺失）、R18（distill loss 求和）、R19（`export_packets` 未适配） |
| 中等 | 3 | R20（`current_keyframe_id` per-batch）、R21（统一分辨率假设）、R22（`_build_keyframe_mask` 单 mask） |
| 信息性 | 2 | R23（`camera_head.last_scores` 潜在风险）、R24（DDP 不同步） |

**关键阻塞项（R12-R15）**：四个严重问题在 B>1 时会产出错误结果。R14（camera cache 丢弃）影响最大——改变了根本计算而非精度差异。R12（共享 `last_scores`）违反数学等价性声明。R13（anchor count 求和）会破坏 camera KV cache 布局。R15（keyframe_schedule）会产生错误的 loss target。

**建议操作**：在实现开始前必须修复 R12-R15 和 R17-R18。这些是正确性问题，不是优化机会。

### 第三轮审查（2026-05-26）

> 聚焦于设计提出的代码片段中的 bug、训练/loss 正确性、TokenScorer 交互、gradient checkpointing 交互、DDP 细节、data loader collation、teacher 模型兼容性、验证策略充分性。

### Review-25（严重）：DataLoader default_collate 在 B>1 时因字符串字段崩溃

**问题**：数据集的 `__getitem__` 返回的 view dict 包含混合类型——tensor（`img`）、numpy array（`camera_pose`, `depthmap`, `valid_mask`）和字符串（`dataset`, `label`, `instance`）。`batch_size=1` 时 `default_collate` 接收单元素 batch 直接返回。`batch_size>1` 时，`default_collate` 尝试按 key stack 值，遇到字符串字段 `TypeError`。

**影响**：训练无法启动。崩溃发生在首次数据加载，早于任何模型代码。这使得所有 B>1 变更无法测试。

**建议**：编写自定义 collate 函数处理字符串字段（存为 list 或忽略），仅 stack tensor/numpy 值：
```python
def frontend_collate_fn(batch):
    num_frames = len(batch[0])
    collated = []
    for frame_idx in range(num_frames):
        frame_dicts = [sample[frame_idx] for sample in batch]
        collated_frame = {}
        for key in frame_dicts[0]:
            vals = [fd[key] for fd in frame_dicts]
            if isinstance(vals[0], (torch.Tensor, np.ndarray)):
                collated_frame[key] = torch.stack([torch.as_tensor(v) for v in vals])
            elif isinstance(vals[0], str):
                collated_frame[key] = vals
            else:
                collated_frame[key] = vals
        collated.append(collated_frame)
    return collated
```

### Review-26（严重）：§4.1 camera_head 代码有三个复合 bug——dict 当 list、未定义变量、无效参数

**问题**：§4.1 camera_head 交互代码有三个互相叠加的 bug。

Bug 1 — dict 当 list 索引：
```python
pose_enc_list_b, past_key_values_camera[b] = self.camera_head(...)
pose_enc_batch.append(pose_enc_list_b[-1])  # TypeError: dict 不支持 [-1] 索引
```
`camera_head` 返回 `(dict, pvc)`，dict 有 key `"abs_pose_enc"` 和 `"rel_pose_enc"`（camera_head.py:299-302）。

Bug 2 — 未定义变量：
```python
rel_pose_enc = pose_enc_list[-1][:B]  # NameError: 'pose_enc_list' 未定义
```
循环变量是 `pose_enc_list_b`，不存在 `pose_enc_list`。

Bug 3 — 无效参数：
```python
self.camera_head(
    ...,
    past_frame_idx=i,           # 不在 camera_head.forward 签名中
    total_budget=camera_budget, # 不在签名中
    num_anchor_cameras=(...),   # 不在签名中
)
```
实际签名（camera_head.py:92-102）接受 `anchor_token_count`, `num_iterations`, `pose_encoding_type`, `return_pose_predictions`, `return_last_pose_only`。

**影响**：代码无法编译/执行。三个 bug 都在 per-frame 热循环中——没有任何 B>1 训练 step 能完成。

**建议**：重写 camera_head 部分以匹配实际接口：
```python
pose_enc_batch = []
rel_pose_enc_batch = []
for b in range(B):
    camera_anchor_token_count = (
        None if i == 0
        else keyframe_managers[b].get_num_anchor_frames() * self.camera_num_iters
    )
    pose_enc_dict_b, past_key_values_camera[b] = self.camera_head(
        [agg[b:b+1] for agg in aggregated_tokens_list],
        num_iterations=self.camera_num_iters,
        past_key_values_camera=past_key_values_camera[b],
        use_cache=True,
        anchor_token_count=camera_anchor_token_count,
        pose_encoding_type=self._camera_pose_encoding_type_for_frontend(),
        return_pose_predictions=True,
        return_last_pose_only=True,
    )
    pose_enc_batch.append(pose_enc_dict_b["abs_pose_enc"][:, 0, :])
    rel_pose_enc_batch.append(pose_enc_dict_b["rel_pose_enc"][:, 0, :])
camera_pose = torch.cat(pose_enc_batch, dim=0)
```

### Review-27（重要）：§6 Phase 3 LR 线性缩放在 mean-reduced loss 下不正确

**问题**：设计提出 `lr_new = lr_base × (B_new / B_old)`。线性缩放规则（Goyal et al.）假设梯度跨 batch 元素**求和**。但设计 §3.5 明确声明 loss 是 **mean-reduced**。

实际梯度幅度：`grad_B = (1/B) × sum(grad_b)`，期望值与 B 无关。最优学习率对 B=1/4/8 相同。线性缩放会把 B=8 的 LR 过度放大 8×。

**影响**：B=8 会使用 8× 正确学习率，导致训练发散。

**建议**：移除 LR 缩放建议。增加 B 时保持 `lr` 不变。添加注释："LR 线性缩放不适用于此场景，因为 loss 已 mean-reduce。梯度幅度与 B 无关。"

### Review-28（重要）：§7.1 cache state 等价性测试不可实现

**问题**：§7.1 提议比较 `cache_states[0][0].k` 在 B=1 和 B=4 之间，标准为"element-wise identical"。这根本不可实现。

原因：`last_scores` 的 commit 循环后，`aggregator.last_scores` 始终反映 batch B-1 的 score。下一帧 `_calculate_dynamic_budgets` 用此计算 budget。B=1 时每序列用自己的 score，B=4 时所有序列用 batch 3 的 score。不同 budget → 不同 eviction 决策 → 不同 cache 内容。

**影响**：测试永远不会通过，浪费实现时间。可能导致实现者追逐幽灵 bug。

**建议**：替换为 loss 等价测试：
- 同样 4 个序列：B=1 × 4 step（accum_iter=4）vs B=4 × 1 step
- 比较总 loss：相对差异 < 1e-3（BF16 累积顺序导致非精确匹配）
- 比较 per-sequence head 输出（depth, pose）：最大相对误差 < 1e-3

### Review-29（中等）：`last_scores` 不对称性违反 §3.5 声明的数学等价性

**问题**：commit 循环中 `self.aggregator.last_scores[layer_idx] = score` 被覆写 B 次，最终只保留 batch B-1 的值。下一帧所有 B 个序列使用 batch B-1 的 budget 历史。

B=1 独立运行时，每序列用自己的 score。B=4 时所有序列用 batch 3 的 score。Budget 分配偏离 → eviction token 不同 → 前向路径微妙差异 → loss 偏差随帧累积。

**影响**：偏差有界但非零，随帧数增长。§3.5 的"数学等价"声称不成立。

**建议**：Phase 1 记录为已知偏差（fixed_interval 下预期较小）。严格等价需要 per-batch `last_scores`：
```python
per_batch_last_scores = [self.aggregator.last_scores.clone() for _ in range(B)]
# commit 循环内:
per_batch_last_scores[b][layer_idx] = score
# aggregator 调用前:
self.aggregator.last_scores = per_batch_last_scores[b]
```

### Review-30（中等）：单一 `keyframe_schedule` mask 与 batched GT 的脆弱耦合

**问题**：`keyframe_schedule` 存储 batch 0 的代表性 event（按 Review-15），产生单一 `[num_frames]` mask。`_build_gt_relative_pose_targets`（frontend_distill.py:363-366）用标量 `active_frame_idx` 对所有 B 个序列选取 anchor frame。如果 batch 0 是特殊情况（如极短序列），其 schedule 会错误地应用到其他序列。

**影响**：Phase 1 安全（fixed_interval, 一致 schedule）。Phase 2 会静默地为 schedule 不同于 batch 0 的序列计算错误的 relative pose target。

**建议**：Phase 1 添加 assertion 验证所有 batch 元素的 event 一致。

### Review-31（中等）：Gradient checkpointing 与 sequential B=1 aggregator 调用的 B× 重计算开销

**问题**：`gradient_checkpointing=True` 时，每个 B sequential aggregator 调用独立触发 checkpointed block 的重计算。B=4 时每帧重计算 4×24=96 次 vs B=1 的 24 次。

Head 计算使用 batched [B, ...] 输入，checkpoint 开销与 B=1 类似（一次通过更大 batch 的 checkpointed 函数）。

**影响**：B=4 且启用 gradient checkpointing 时，backward 中 aggregator 部分耗时 ~4×。结合 sequential 方案 ~40-60% GPU 利用率，总吞吐提升可能为负。

**建议**：文档化此交互。B>1 时考虑仅对 heads 启用 checkpointing（aggregator 处理 B=1 batch，激活内存已很小）。添加配置选项 `frontend_aggregator_checkpointing: bool` 单独控制。

---

### 第三轮审查总结

| 级别 | 编号 | 标题 | 处理 |
|------|------|------|------|
| 严重 | R25 | collation 崩溃 | **已修复** — §4.6a 新增 `frontend_collate_fn` |
| 严重 | R26 | camera_head 代码 3 bug | **已修复** — §4.1 用实际 API（L435-444）重写 |
| 重要 | R27 | LR 缩放错误 | **已修复** — §6 Phase 3 移除线性缩放 |
| 重要 | R28 | 等价测试不可行 | **已修复** — §7.1 改为 loss + head-output 测试 |
| 中等 | R29 | last_scores 等价性偏差 | **已处理** — §3.5 添加已知偏差说明 |
| 中等 | R30 | keyframe mask 脆弱耦合 | **已处理** — Phase 1 assert + 文档化 |
| 中等 | R31 | checkpointing B× 开销 | **已处理** — 建议禁用 aggregator checkpointing |

---

### 全部三轮审查总结（R1-R31）

| 级别 | 数量 | 状态 |
|------|------|------|
| 严重 | 0 | ~~R1, R2, R3, R12-R15, R25, R26~~ — 全部已修复 |
| 重要 | 0 | ~~R4-R6, R16-R19, R27, R28~~ — 全部已修复 |
| 中等 | 3 | R29（已知偏差 + 已缓解）、R30（Phase 2 重构）、R31（建议禁用 checkpointing） |
| 信息性 | 4 | R10（已采纳）、R11、R23、R24（Phase 1 B≤4 自动满足） |

**阻塞项**：0。所有严重/重要问题已在 spec 中修复。

### 第四轮审查（2026-05-26）

> 聚焦于 §4.1 提议代码的端到端 trace、`finetune_frontend.py` 兼容性、`FrontendDistillLoss`/`FrontendSupervisedLoss` B>1 兼容性、`PendingLayerUpdate` 拆分机制、TokenScorer 交互、head closure 捕获变量、`apply_keyframe_event` per-batch 遗漏。

### Review-32（重要）：`camera_head.apply_keyframe_event`（ovggt.py:577）未适配 per-batch PVC——静默将 batch 0 的 event 应用到所有序列

**问题**：设计 §4.1 代码块声称 "replaces lines 498-553"，但 `apply_keyframe_event` 在 line 577，**落在代码块范围之外**，设计未提及此行。B>1 时当前的单次调用：

```python
past_key_values_camera = self.camera_head.apply_keyframe_event(
    past_key_values_camera,  # 现在是 List[List]（B × trunk_depth）
    event,                   # 仅 batch 0 的 event
    num_cam_iters=self.camera_num_iters,
)
```

`past_key_values_camera` 已变为 `List[List]`（per-batch），但 `event` 仅来自 batch 0。函数期望 `past_key_values_camera` 为 `List[Optional[Tuple]]`（长度 trunk_depth）。类型不匹配要么崩溃，要么静默将 batch 0 的 keyframe slot 重分配应用到所有序列的 camera KV cache。

**影响**：batch 1+ 的 camera KV cache slot 与实际 keyframe schedule 脱节。后续 camera 预测引用错误的 anchor frame，产出错误的 pose。错误是静默的——不崩溃，只是训练 target 错误。

**建议**：在 commit 循环中或单独的块中添加 per-batch `apply_keyframe_event`：
```python
for b in range(B):
    past_key_values_camera[b] = self.camera_head.apply_keyframe_event(
        past_key_values_camera[b], events[b],
        num_cam_iters=self.camera_num_iters,
    )
```

### Review-33（重要）：Post-commit 代码（ovggt.py:562-609）完全未覆盖——`res_gpu` 构建、`frame_writer`、`KeyframePacket`、cleanup 都需要 B>1 适配

**问题**：设计 §4.1 代码块在 commit 循环处结束，遗漏了 line 562 之后的所有代码：

1. `keyframe_packets.append(KeyframePacket(...))`（562-575）—— R19 标记了 batched tensor 问题，但设计仍未展示 per-batch 循环
2. `camera_head.apply_keyframe_event(...)`（577-581）—— R32
3. `res_gpu` dict 构建（583-596）—— `depth`, `camera_pose` 等现在是 `[B, ...]` tensor
4. `frame_writer(i, frame, res_gpu)`（597-598）—— 传入 batched `res_gpu` 和 batched `frame`
5. `cache_results` 处理（600-604）—— 将 batched `res_gpu` append 到 `all_ress`
6. `del aggregated_tokens; del pending_updates`（607-609）—— 变量名已改为 `aggregated_tokens_list` 和 `frame_pending_updates`，`del` 会 `NameError`

**影响**：实现者仅按设计的代码块操作，会遗漏 `apply_keyframe_event` per-batch 循环（R32）、`KeyframePacket` per-batch 循环、变量名清理。`del` 语句引用已重命名的变量会崩溃。

**建议**：扩展 §4.1 代码块覆盖 lines 562-609。Item 6：改为 `del aggregated_tokens_list; del frame_pending_updates`。

### Review-34（重要）：`FrontendSupervisedLoss._build_point_targets` 从 camera intrinsics/pose 中剥离 batch 维——用 batch 0 的 camera 参数计算所有 B 序列的 3D 点

**问题**：`FrontendSupervisedLoss`（用于 `finetune_frontend.py`）的 `_build_point_targets` 方法（frontend_supervised.py:183-186）：

```python
if camera_intrinsics.ndim == 3:
    camera_intrinsics = camera_intrinsics[0]  # 剥离 batch 维！
if camera_pose.ndim == 3:
    camera_pose = camera_pose[0]  # 剥离 batch 维！
```

B>1 时 `camera_intrinsics` 是 `[B, 3, 3]`（ndim=3）→ 取 `[0]` → `[3, 3]`。然后 `depthmap_to_absolute_camera_coordinates` 用 batched `depth_np [B, H, W]` 但标量的 batch 0 camera 参数。为 batch 1+ 产出错误的 3D 坐标。

设计仅分析了 `FrontendDistillLoss`（通过 `finalize_from_stream`），未提及 `FrontendSupervisedLoss`。

**影响**：`finetune_frontend.py` 在 B>1 时为 batch 1+ 计算错误的 3D 点 target，破坏 `Lpmap` loss 项，静默产生错误的梯度。设计 §4.6 声称 finetune 仅需 "~3 行"（guard 移除）——不够。

**建议**：在 per-batch 循环中调用 `depthmap_to_absolute_camera_coordinates`：
```python
results = []
for b in range(batch_size):
    pts, mask = depthmap_to_absolute_camera_coordinates(
        depth_np[b], camera_intrinsics_np[b], camera_pose_np[b],
    )
    results.append((pts, mask))
```
或记录 finetune B>1 要求数据集提供预计算的 `pts3d`（跳过此路径）。

### Review-35（中等）：§4.1 "Pass through active_pose_encoding" 代码引用未定义的 `pose_enc_list`——`NameError`

**问题**：§4.1 的 active_pose_encoding 代码块：
```python
rel_pose_enc = pose_enc_list[-1][:B]  # NameError: 'pose_enc_list' 未定义
```

变量 `pose_enc_list` 在设计的代码中不存在。camera_head 循环收集的是 `rel_pose_enc_batch`（`[1, 9]` tensor 列表）。正确引用应为：
```python
rel_pose_enc = torch.cat(rel_pose_enc_batch, dim=0).unsqueeze(1)  # [B, 1, 9]
```

这是 R26（camera_head dict/list 混淆）同类 bug 在不同代码块中的实例。

**影响**：当 `self.frontend_pose_encoding_type != ABS_POSE_ENCODING`（relative pose 路径）时崩溃。`ABS_POSE_ENCODING` 模式下此代码被跳过——无影响。

**建议**：替换 `pose_enc_list[-1][:B]` 为 `torch.cat(rel_pose_enc_batch, dim=0).unsqueeze(1)`。

### Review-36（中等）：Head closure 捕获 `images` 变量——设计重命名为 `images_all` 但未展示 closure 更新

**问题**：当前代码定义 head closure 按名捕获 `images`（ovggt.py:463-483）：
```python
def depth_head_forward(*layer_tokens):
    return self.depth_head(list(layer_tokens), images=images, ...)
depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens)
```

设计将变量重命名为 `images_all`（§4.1 line 147）但 head 调用写为：
```python
depth, depth_conf = self.depth_head(aggregated_tokens_list, ...)
```

如果实现者保留 closure 模式（`maybe_checkpoint_head` + gradient checkpointing 所需），closure 会捕获 `images`：
- 未定义（如果 `images = ...` 被替换为 `images_all = ...`）→ `NameError`
- 最后一次 aggregator 迭代的陈旧 `images_b` → 错误形状 `[1, 1, C, H, W]` 而非 `[B, 1, C, H, W]`

此外，设计使用直接调用语法（`self.depth_head(aggregated_tokens_list, ...)`）而当前代码使用 `maybe_checkpoint_head(depth_head_forward, *aggregated_tokens)` 解包列表为 variadic args。设计的语法将 list 作为第一个位置参数传入，可能与 head 签名不匹配。

**影响**：`NameError`（未定义 `images`）或错误的图像 tensor 形状传入 head（静默错误行为）。`maybe_checkpoint_head` 模式是 gradient checkpointing 正确工作所必需的。

**建议**：展示更新后的 closure：
```python
def depth_head_forward(*layer_tokens):
    return self.depth_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens_list)
```

### Review-37（信息性）：设计中 `events[b].get_slot_for_frame(i)` 方法不存在于 `KeyframeEvent`

**问题**：设计的 commit 循环展示：
```python
anchor_slot=(-1 if i == 0 else events[b].get_slot_for_frame(i)),
```

当前代码（ovggt.py:546）使用 `anchor_slot=event.anchor_slot`（直接属性访问）。`KeyframeEvent` 将 `anchor_slot` 存为普通 integer 属性。方法 `get_slot_for_frame(frame_idx)` 在代码库中不存在。

**影响**：第一个非零帧时 `AttributeError: 'KeyframeEvent' object has no attribute 'get_slot_for_frame'`。

**建议**：替换为 `events[b].anchor_slot`。`-1 if i == 0` guard 也不必要——frame 0 的 event 的 `anchor_slot` 已经是 -1。

### Review-38（信息性）：`finetune_frontend.py` 无 `use_token_scorer`/`finetune_full_model` 支持——设计的 §4.6 变更仅适用于 `train_frontend.py`

**问题**：`finetune_frontend.py`：
- 总是调用 `freeze_frontend_stage_a_parameters(model)`（line 365）——从不调用 `freeze_stage_a_scorer_only`
- 不在模型上初始化 token_scorers（OVGGT 构造器 line 317-325 不调用 `init_token_scorers`）
- 不处理 `student_outputs.distill_loss`（supervised loss 函数 line 87 调用 `criterion(batch, student_outputs, ...)` 不添加 distill_loss）
- 不传 `frame_processor` 给模型（line 82: `model(batch)` 无回调）

这不是 B>1 退化——是现有缺口。但设计声称 finetune 仅需 "~3 行"（guard 移除），未指出 scorer 相关的 B>1 工作在 finetune 路径中完全缺失。

**影响**：如果在 finetune config 中设 `use_token_scorer=True` 且 B>1，scorer 永远不会被初始化、训练，distill_loss 被静默丢弃。不崩溃——只是不正确的训练行为。

**建议**：在 §4.6 记录 finetune_frontend.py 的 B>1 支持仅限于无 TokenScorer 的 `FrontendSupervisedLoss`。finetune 路径的 scorer 支持需要独立实现。

---

### 第四轮审查总结

| 级别 | 编号 | 标题 | 处理 |
|------|------|------|------|
| 重要 | R32 | apply_keyframe_event per-batch 遗漏 | **已修复** — §4.1 新增 per-batch PVC 循环 |
| 重要 | R33 | post-commit 代码未覆盖 | **已修复** — §4.1 扩展覆盖 L562-609 |
| 重要 | R34 | FrontendSupervisedLoss batch 维剥离 | **已修复** — §4.6 记录 finetune B>1 限制 |
| 中等 | R35 | 未定义 pose_enc_list | **已修复** — 替换为 `torch.cat(rel_pose_enc_batch, dim=0)` |
| 中等 | R36 | head closure 捕获变量名 | **已修复** — 更新为 `images_all` + `maybe_checkpoint_head` |
| 信息性 | R37 | get_slot_for_frame 不存在 | **已修复** — 替换为 `.anchor_slot` 属性 |
| 信息性 | R38 | finetune 无 scorer 支持 | **已修复** — §4.6 记录 Phase 1 限制 |

---

### 全部四轮审查总结（R1-R38）

| 级别 | 数量 |
|------|------|
| 严重 | 9（R1, R2, R3, R12-R15, R25, R26） |
| 重要 | 12（R4-R6, R16-R19, R27-R28, R32-R34） |
| 中等 | 11（R7-R8, R20-R22, R29-R31, R35-R36） |
| 信息性 | 6（R10-R11, R23-R24, R37-R38） |
| **合计** | **38** |

**建议**：设计文档中的 §4.1 代码块需要全面重写，覆盖 ovggt.py lines 385-609 的完整范围。当前代码块存在多处未定义变量、不存在的 API 调用、类型不匹配和范围遗漏。建议基于审查反馈重新生成完整的 `_inference_frontend` 代码，而非在现有代码块上修补。

---

*Design completed 2026-05-26. Code review (R1-R38, four rounds) completed 2026-05-26. See implementation plan for task breakdown.*
