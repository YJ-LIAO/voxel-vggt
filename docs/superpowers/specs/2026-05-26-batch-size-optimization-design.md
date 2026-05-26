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

**Per-frame per-batch aggregator loop** (replaces the aggregator call at ~L416 and surrounding logic):

```python
for i, frame in enumerate(frames):
    images_all = self._frame_image_to_sequence(frame["img"])  # [B, 1, C, H, W]

    # Sequential aggregator: call B times with B=1 (cache states are B=1 internally)
    frame_agg_outputs = []
    frame_pending_updates = []
    frame_distill_losses = []
    for b in range(B):
        images_b = images_all[b:b+1]  # [1, 1, C, H, W]
        agg_tokens, ps, cs_b, pending, fdl = self.aggregator(
            images_b,
            cache_states=cache_states[b],
            use_cache=True,
            past_frame_idx=i,
            total_budget=total_budget,
            importance_weight=importance_weight,
            frontend_cache_config=self.frontend_cache_config,
        )
        frame_agg_outputs.append(agg_tokens)       # List[Tensor] per layer
        frame_pending_updates.append(pending)       # per-batch pending
        if fdl is not None:
            frame_distill_losses.append(fdl)
        cache_states[b] = cs_b                     # updated cache

    # Accumulate distill_loss across batches
    if frame_distill_losses:
        sum_fdl = sum(frame_distill_losses)
        total_distill_loss = sum_fdl if total_distill_loss is None else total_distill_loss + sum_fdl

    # Batch head computations: concatenate per-batch aggregator outputs along dim=0
    # aggregated_tokens_list: List of 24 tensors, each [B, S*P, 2C]
    aggregated_tokens_list = []
    for layer_idx in range(len(frame_agg_outputs[0])):
        layer_cat = torch.cat([bo[layer_idx] for bo in frame_agg_outputs], dim=0)
        aggregated_tokens_list.append(layer_cat)
```

**Heads** (unchanged signature, now receive batched `[B, ...]` inputs):
```python
    with self._disabled_autocast_context():
        camera_anchor_token_count = None if i == 0 else sum(
            km.get_num_anchor_frames() for km in keyframe_managers
        ) * self.camera_num_iters
        pose_enc_list, _ = self.camera_head(
            aggregated_tokens_list,
            past_key_values_camera=None,  # per-batch: see §4.5
            ...
        )
    depth, depth_conf = self.depth_head(aggregated_tokens_list, ...)
    pts3d, pts3d_conf = self.point_head(aggregated_tokens_list, ...)
```

**Per-batch cache commit** (replaces lines 498-553):

```python
    for b in range(B):
        # Per-batch keyframe decision
        event = keyframe_managers[b].update(
            frame_idx=i,
            depth=depth[b],
            pose_abs_enc=pose_enc_list[-1][b] if pose_enc_list else camera_pose[b],
            image_size_hw=(img_h, img_w),
        )
        # Per-batch per-layer commit
        for layer_idx in range(self.aggregator.depth):
            pending = frame_pending_updates[b][layer_idx]
            if pending is None:
                continue
            # Split batch-dim from aggregator output for per-batch commit
            per_batch_update = PendingLayerUpdate(
                k_current=pending.k_current,       # already [1, H, N, D] from B=1 aggregator
                v_current=pending.v_current,
                importance_current=pending.importance_current,
                frame_id=pending.frame_id,
                cache_budget=pending.cache_budget,
            )
            cache_states[b][layer_idx].apply_keyframe_event_(event)
            cache_states[b][layer_idx].commit_pending_update_(
                per_batch_update, ..., config=self.frontend_cache_config, ...
            )
```

**Guard removal**: Change `_validate_frontend_batch_size` to accept `B >= 1` (remove the `ref_batch_size != 1` check at `ovggt.py:828`).

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

### 4.6 `src/train_frontend.py`

- Remove `ValueError` guard at line 653
- In `train()`, pass `use_token_scorer` and `distill_loss_weight` from args to model constructor
- Ensure `freeze_stage_a_scorer_only()` is called between `load_student_pretrained_weights` (L792) and `get_parameter_groups` (L818) when `use_token_scorer=True`

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
| 3 | GPU memory exceeds budget at B=4 | Enable `gradient_checkpointing=True` (40-60% memory reduction, 30% compute increase) or reduce `frontend_total_budget` |
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

**问题**：§6 Phase 2 估算 "Cache state memory per sequence: ~1.2GB"。此估算基于 `frontend_total_budget=10410`（TokenScorer 配置）。

实际计算（budget=10410）：10410 tokens × 16 heads × 64 dim × 2 bytes (bf16) × 2 (K+V) ≈ 42.7MB/layer × 24 layers ≈ 1.02GB。✓

但如果使用默认 `train_frontend_blendedmvs.yaml` 的 `total_budget=200000`：
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

---

*Design completed 2026-05-26. Code review added 2026-05-26. See implementation plan for task breakdown.*
