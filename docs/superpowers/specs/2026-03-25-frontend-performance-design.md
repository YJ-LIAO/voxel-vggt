# Frontend Performance Optimization Design

## Goal

Improve frontend training-time memory usage, training throughput, and inference latency without changing training results or inference behavior.

## Non-Goals

- No changes to loss formulas.
- No changes to query-point sampling.
- No changes to keyframe scheduling semantics.
- No changes to cache eviction or voxel-dedup decision rules.
- No third-party kernel integration such as `flash-attn`.

## Constraints

- Preserve numerical behavior as closely as possible.
- Keep existing `batch_size=1` frontend training contract.
- Maintain compatibility with current config flags such as `teacher_output_to_cpu`, `teacher_weight_offload`, and `frontend_head_checkpointing`.

## Evidence

### Inference

Measured on real data and weights with the existing hotspot profiler:

- `Aggregator.forward` remains the dominant compute block.
- `LayerCacheState.commit_pending_update_` and `LayerCacheState.apply_voxel_dedup_` together account for roughly 28%-33% of end-to-end frontend inference time.
- `CameraHead.forward` is not a primary bottleneck.
- Attention already dispatches to PyTorch Flash SDPA on the current A800 runtime, so the main opportunity is outside the attention kernel.

### Training

The current training path performs:

1. teacher forward on GPU
2. optional teacher output offload to CPU
3. per-frame tensor copies back to GPU inside loss computation

This preserves memory headroom, but adds host-device transfer overhead and repeated `.to(device)` calls.

## Design

### 1. Add Batch-Size-1 Fast Paths For Cache State Updates

`frontend_train` is already restricted to `batch_size=1`. The cache-state implementation currently pays for a general per-batch gather-and-pad path that is unnecessary in the steady-state training/inference configuration.

Planned changes:

- Add a dedicated `B=1` gather path in `LayerCacheState`.
- Avoid the generic `gather_per_batch_` padding logic when only one batch item exists.
- Reuse compact index-select operations for K/V and metadata fields.

Expected outcome:

- Lower Python overhead.
- Fewer temporary tensors.
- Lower GPU memory churn during dedup and cache commit.

### 2. Keep Voxel-Dedup Semantics But Reduce Implementation Overhead

The current dedup implementation is already vectorized in parts, but still spends time in:

- repeated `nonzero` extraction
- generic batch loops even when `B=1`
- metadata reconstruction after gather

Planned changes:

- Specialize `apply_voxel_dedup_` for `B=1`.
- Keep protected-token conflict detection and intra-frame survivor selection identical.
- Reduce intermediate tensor construction and metadata rebuilding.

Expected outcome:

- Lower latency in `commit_pending_update_`.
- No change to which tokens survive dedup.

### 3. Reduce Teacher Output Transfer Overhead In Training

The current distillation path moves teacher results to CPU and then rehydrates them field-by-field on GPU.

Planned changes:

- Keep teacher offload semantics unchanged.
- Move CPU teacher tensors into pinned memory before later reuse.
- Use non-blocking device transfers consistently.
- Reduce repeated field-wise copies where the same frame data is consumed more than once.

Expected outcome:

- Faster training steps under `teacher_output_to_cpu=True`.
- No change to distillation targets.

### 4. Remove Execution-Level Inefficiencies That Do Not Affect Semantics

Planned changes:

- Replace deprecated `torch.cuda.amp.autocast(...)` usage with `torch.amp.autocast(...)`.
- Tighten tensor lifetime handling where large containers can be dropped earlier.
- Keep config behavior intact while reducing redundant allocations.

## Validation Strategy

### Unit/Smoke Validation

- Extend frontend cache tests to cover the new `B=1` fast path against the existing behavior.
- Run the existing frontend smoke tests.

### Behavior Validation

- Compare key outputs for the cache-update path before and after optimization on deterministic tiny inputs.
- Confirm training loss remains finite and follows the same path on the smoke setup.

### Performance Validation

- Re-run the frontend hotspot profiler on real data.
- Compare wall-clock time and peak memory before and after.
- If feasible, run a minimal training-step profile to confirm reduced transfer overhead.

## Risks

- Fast-path logic can accidentally diverge from the generic path if index ordering is not preserved exactly.
- Pinned-memory staging improves speed but can increase host memory pressure if tensor lifetime is not controlled carefully.

## Rollout

1. Add tests that lock in current cache-state behavior.
2. Implement `B=1` fast paths and teacher-transfer cleanup.
3. Re-run tests and real-data profiling.
4. Keep the generic multi-batch path intact as a correctness fallback.

## Repository Note

This workspace does not currently contain a `.git` directory, so the design is saved on disk but cannot be committed from this location.
