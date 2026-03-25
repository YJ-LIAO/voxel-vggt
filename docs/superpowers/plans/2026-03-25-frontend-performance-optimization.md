# Frontend Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce frontend training memory/transfer overhead and frontend inference latency without changing numerical behavior or training semantics.

**Architecture:** Add batch-size-1 fast paths in cache-state update code, preserve voxel-dedup and eviction rules exactly, and reduce teacher CPU/GPU transfer overhead in the training path. Keep the current generic code as a correctness fallback and validate equivalence with targeted tests plus existing smoke coverage.

**Tech Stack:** PyTorch, Accelerate, pytest, OVGGT frontend cache/keyframe pipeline

---

### Task 1: Lock In Cache-State Behavior With Tests

**Files:**
- Modify: `tests/test_frontend_cache.py`

- [ ] **Step 1: Write failing tests for batch-size-1 cache gather/dedup equivalence**

Add tests that:
- construct a small `LayerCacheState`
- exercise gather on a single batch item
- exercise voxel dedup on a single batch item
- assert outputs match the existing generic behavior

- [ ] **Step 2: Run targeted tests to verify they fail for the missing fast path**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_cache.py -q`

Expected: new assertions fail because the fast-path helpers do not exist yet.

- [ ] **Step 3: Implement the minimal test scaffolding only**

Keep test inputs deterministic and small. Avoid touching production code in this step.

- [ ] **Step 4: Re-run the targeted tests**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_cache.py -q`

Expected: still failing, but now on the intended missing behavior.

### Task 2: Implement Batch-Size-1 Fast Paths In Cache State

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py`
- Test: `tests/test_frontend_cache.py`

- [ ] **Step 1: Add a single-batch gather helper**

Implement a focused helper for `B=1` that:
- gathers `k` and `v` without padding
- index-selects metadata fields directly
- preserves index ordering exactly

- [ ] **Step 2: Route `gather_per_batch_` and dedup updates through the fast path when `B=1`**

Preserve the generic multi-batch path as fallback.

- [ ] **Step 3: Add a single-batch dedup helper**

Specialize the current `apply_voxel_dedup_` path for the common `B=1` case while keeping:
- protected-token conflict handling
- current-frame survivor selection
- voxel grouping semantics

- [ ] **Step 4: Run the cache tests**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_cache.py -q`

Expected: PASS.

### Task 3: Optimize Teacher Transfer Path Without Changing Semantics

**Files:**
- Modify: `src/train_frontend.py`
- Modify: `src/ovggt/losses/frontend_distill.py`
- Test: `tests/test_frontend_training_smoke.py`

- [ ] **Step 1: Write a failing test for teacher-output transfer behavior**

Add a focused smoke/regression test that exercises the distillation path with `teacher_output_to_cpu=True` and validates the step still completes with finite loss.

- [ ] **Step 2: Run the training smoke test to verify the baseline**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_training_smoke.py -q`

Expected: existing tests pass; new transfer-focused assertions fail until the optimized path is implemented.

- [ ] **Step 3: Implement pinned/non-blocking teacher tensor staging**

Make the transfer path more efficient by:
- pinning CPU teacher tensors where appropriate
- preserving non-blocking device copies
- reducing repeated per-field `.to(device)` churn

- [ ] **Step 4: Re-run the training smoke tests**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_training_smoke.py -q`

Expected: PASS.

### Task 4: Clean Up Execution-Level Inefficiencies

**Files:**
- Modify: `src/ovggt/models/ovggt.py`
- Modify: `src/train_frontend.py`
- Modify: `src/finetune_frontend.py`

- [ ] **Step 1: Replace deprecated autocast entry points**

Switch deprecated `torch.cuda.amp.autocast(...)` usage to `torch.amp.autocast(...)` where behavior is unchanged.

- [ ] **Step 2: Tighten tensor lifetime handling**

Drop large containers as early as possible once no longer needed, without changing outputs.

- [ ] **Step 3: Run frontend smoke coverage**

Run: `cd /mnt/lyj/workspace/OVGGT-main && /mnt/lyj/miniconda3/bin/conda run -n OVGGT pytest tests/test_frontend_inference_smoke.py tests/test_frontend_training_smoke.py tests/test_frontend_supervised_loss.py -q`

Expected: PASS.

### Task 5: Verify Real-Data Performance

**Files:**
- No code changes required

- [ ] **Step 1: Re-run real-data frontend hotspot profiling**

Run: `CUDA_VISIBLE_DEVICES=0 /mnt/lyj/miniconda3/bin/conda run -n OVGGT python /mnt/lyj/workspace/OVGGT-main/tools/profile_frontend_hotspots.py --dataset-root /mnt/lyj/workspace/OVGGT-main/blendedmvs_processed --weights /mnt/lyj/workspace/OVGGT-main/ckpt/checkpoints.pth --scene 000000000000000000000000 --sequence 0,1,2 --device cuda:0 --voxel-size 0.1`

Expected: lower time share in `LayerCacheState.commit_pending_update_` and `LayerCacheState.apply_voxel_dedup_`, with no runtime errors.

- [ ] **Step 2: Re-run SDPA backend sanity check if needed**

Run the existing local profiler helper only if attention behavior appears regressed.

- [ ] **Step 3: Summarize before/after results**

Capture:
- cache-update time reduction
- end-to-end inference time change
- peak memory change where observable

## Review Notes

- The repository is not a git checkout here, so plan/spec documents cannot be committed.
- Subagent review was skipped because this session is not authorized for delegation; manual review is required instead.
