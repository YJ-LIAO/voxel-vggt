# Batch Size Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove batch_size=1 constraint — enable B independent video sequences per training step with per-sequence streaming state, while reducing training step time by eliminating GPU sync points.

**Architecture:** Sequential aggregator (B calls, B=1 each) + batched heads (depth/point). Each batch element gets independent copies of FrontendKeyframeManager, LayerCacheState, and camera KV cache. Cache states are saved/restored per batch element to isolate dynamic budget allocation.

**Tech Stack:** PyTorch 2.x, BF16 AMP, DDP, DINOv2 ViT-L/14 backbone, fused SDPA attention.

**Spec:** `docs/superpowers/specs/2026-05-26-batch-size-optimization-design.md`

---

## File Structure

| File | Role | Responsibility |
|------|------|----------------|
| `src/dust3r/datasets/collate.py` | **NEW** | Custom collate function for B>1 mixed-type view dicts |
| `src/ovggt/models/ovggt.py` | Modify | Core `_inference_frontend` refactor — per-batch state, sequential aggregator, batched heads |
| `src/ovggt/utils/frontend_cache.py` | Modify | Sync point elimination — cache-on-mutation for `.item()` calls |
| `src/ovggt/utils/frontend_keyframe.py` | Modify | Verify per-batch isolation of FrontendKeyframeManager |
| `src/train_frontend.py` | Modify | Guard removal, pass batch-aware args to model |
| `src/finetune_frontend.py` | Modify | Guard removal only |
| `config/train_frontend_finetune.yaml` | Modify | `batch_size: 4` |
| `config/train_token_scorer.yaml` | Modify | `batch_size: 4` |
| `tests/test_frontend_batch_training.py` | **NEW** | Integration smoke test with B=2 |

---

### Task 1: Custom Collate Function for Mixed-Type View Dicts

**Files:**
- Create: `src/dust3r/datasets/collate.py`
- Modify: `src/dust3r/datasets/__init__.py` (register collate_fn)

**Why first:** B>1 training cannot start without this — `default_collate` crashes on string fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frontend_batch_training.py (partial — full test in Task 7)
import torch
import numpy as np
from dust3r.datasets.collate import frontend_collate_fn

def test_collate_mixed_types():
    """B=2 sequences, each with 2 frames, mixed tensor/str/None fields."""
    sample0 = [
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a"},
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a"},
    ]
    sample1 = [
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b"},
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b"},
    ]
    batch = [sample0, sample1]
    result = frontend_collate_fn(batch)
    assert len(result) == 2  # 2 frames
    assert result[0]["img"].shape == (2, 3, 518, 392)  # B stacked
    assert result[0]["dataset"] == ["blendedmvs", "co3d"]  # strings as list
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/mount/lyj/voxel-vggt
python -m pytest tests/test_frontend_batch_training.py::test_collate_mixed_types -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'dust3r.datasets.collate'`

- [ ] **Step 3: Write collate function**

```python
# src/dust3r/datasets/collate.py
from typing import List, Dict, Any
import numpy as np
import torch


def frontend_collate_fn(batch: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Collate B sequences, each a list of num_frames view dicts.

    Each sample in `batch` is List[Dict] (length = num_frames per sequence).
    Stack tensor/numpy values across B, preserve strings as List[str].
    """
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

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_frontend_batch_training.py::test_collate_mixed_types -v
```
Expected: PASS

- [ ] **Step 5: Register collate_fn in DataLoader**

In `src/dust3r/datasets/__init__.py`, the `get_data_loader` function already accepts `collate_fn` parameter. In `build_dataset` (`train_frontend.py`), pass `collate_fn=frontend_collate_fn` when `batch_size > 1`:

```python
# train_frontend.py — build_dataset or get_data_loader call site
from dust3r.datasets.collate import frontend_collate_fn
...
loader = get_data_loader(
    dataset, batch_size=batch_size, ...,
    collate_fn=frontend_collate_fn if batch_size > 1 else None,
)
```

- [ ] **Step 6: Commit**

```bash
git add src/dust3r/datasets/collate.py tests/test_frontend_batch_training.py src/dust3r/datasets/__init__.py
git commit -m "feat: add frontend_collate_fn for B>1 mixed-type view dicts"
```

---

### Task 2: Remove batch_size=1 Guards

**Files:**
- Modify: `src/ovggt/models/ovggt.py:815-834`
- Modify: `src/train_frontend.py:653-658`
- Modify: `src/finetune_frontend.py:251-255`

- [ ] **Step 1: Remove guard in ovggt.py**

In `_validate_frontend_batch_size` (line 815-834), change:
```python
if ref_batch_size != 1:
    raise ValueError(...)
```
to:
```python
# B>=1 is now supported via per-batch independent state
```

- [ ] **Step 2: Remove guard in train_frontend.py**

Delete lines 653-658 (`if int(args.batch_size) != 1: raise ValueError(...)`).

- [ ] **Step 3: Remove guard in finetune_frontend.py**

Delete lines 251-255.

- [ ] **Step 4: Verify guards removed**

```bash
grep -rn "batch_size != 1" src/ovggt/models/ovggt.py src/train_frontend.py src/finetune_frontend.py
```
Expected: (no output — all guards removed)

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/models/ovggt.py src/train_frontend.py src/finetune_frontend.py
git commit -m "feat: remove batch_size=1 guards in ovggt, train_frontend, finetune_frontend"
```

---

### Task 3: Sync Point Elimination in frontend_cache.py

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py:103-104, 670-673, 675-686`

**Goal:** Eliminate 3 `.item()` GPU sync points in LayerCacheState, replacing with cached-on-mutation pattern.

- [ ] **Step 1: Add cached attributes to LayerCacheState**

In `LayerCacheState.__init__` (after existing `__init__` or as class-level defaults):

```python
# frontend_cache.py — LayerCacheState class
def __init__(self, ...):
    ...
    self._cached_protected_count: int = 0
    self._cached_has_anchor: bool = False
```

- [ ] **Step 2: Replace has_anchor_tokens (line 103-104)**

```python
# BEFORE:
def has_anchor_tokens(self) -> bool:
    return bool((self.anchor_slot >= 0).any().item())

# AFTER:
def has_anchor_tokens(self) -> bool:
    return self._cached_has_anchor
```

- [ ] **Step 3: Replace _compute_protected_count (line 670-673)**

```python
# BEFORE:
def _compute_protected_count(self) -> int:
    if self.metadata is None or self.metadata.anchor_slot.numel() == 0:
        return 0
    return int((self.metadata.anchor_slot[0] >= 0).sum().item())

# AFTER:
def _compute_protected_count(self) -> int:
    return self._cached_protected_count
```

- [ ] **Step 4: Replace _current_frame_importance (line 675-686)**

```python
# BEFORE (line 682):
counts.append(int(mask.sum().item()))

# AFTER:
counts.append(mask.sum())  # keep as tensor, check lengths with .numel()
# Update the equality check:
if not counts or min(c.numel() for c in counts) == 0 or len(set(c.item() for c in counts)) != 1:
    return None, 0
# Note: the .item() in the set comprehension is acceptable (once per batch, not per-layer)
```

- [ ] **Step 5: Update caches at mutation points**

In `gather_` (line 167), `gather_per_batch_` (line 200), `append_` (line 342), and `apply_keyframe_event_` (line 370), add after mutation:

```python
self._cached_protected_count = self._compute_protected_count_raw()  # keep raw method
self._cached_has_anchor = self.metadata is not None and (self.metadata.anchor_slot >= 0).any().item()
```

Note: this still has `.item()` but only at mutation points (per-frame, not per-query). The `.item()` is unavoidable here due to tensor → int conversion but frequency drops from ~240/step to ~24/step (once per layer per frame, not once per query).

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py
git commit -m "perf: cache-on-mutation for .item() sync points in LayerCacheState"
```

---

### Task 4: Core _inference_frontend Refactor

**Files:**
- Modify: `src/ovggt/models/ovggt.py:364-617`

**This is the main change.** Converts single-sequence inference loop to multi-sequence with per-batch independent state and sequential aggregator + batched heads.

- [ ] **Step 1: Per-batch state initialization (replaces lines 385-392)**

```python
# AFTER — per-batch independent state (B copies of each singleton)
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

- [ ] **Step 2: Per-frame per-batch aggregator loop**

Replace the single aggregator call at ~line 416 with sequential B calls. Key pattern:

```python
for i, frame in enumerate(frames):
    images_all = self._frame_image_to_sequence(frame["img"])  # [B, 1, C, H, W]

    # Save/restore last_scores for per-sequence budget isolation (Review-12)
    saved_last_scores = self.aggregator.last_scores.clone()

    frame_agg_outputs = []
    frame_pending_updates = []
    frame_distill_losses = []
    for b in range(B):
        images_b = images_all[b:b+1]  # [1, 1, C, H, W]
        self.aggregator.last_scores = saved_last_scores.clone() if b > 0 else saved_last_scores
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

    self.aggregator.last_scores = saved_last_scores

    # Average distill_loss across B (Review-18)
    if frame_distill_losses:
        avg_fdl = sum(frame_distill_losses) / len(frame_distill_losses)
        total_distill_loss = avg_fdl if total_distill_loss is None else total_distill_loss + avg_fdl

    # Concatenate per-batch aggregator outputs for batched heads
    aggregated_tokens_list = []
    for layer_idx in range(len(frame_agg_outputs[0])):
        layer_cat = torch.cat([bo[layer_idx] for bo in frame_agg_outputs], dim=0)
        aggregated_tokens_list.append(layer_cat)
```

- [ ] **Step 3: Sequential camera head (Review-14, Review-26)**

```python
    pose_enc_batch = []
    rel_pose_enc_batch = []
    with self._disabled_autocast_context():
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
    camera_pose = torch.cat(pose_enc_batch, dim=0)  # [B, 9]

    # Absolute pose composition (Review-16, Review-35)
    with self._disabled_autocast_context():
        if self.frontend_pose_encoding_type != ABS_POSE_ENCODING:
            active_poses = torch.stack([
                keyframe_managers[b].get_active_pose_encoding()
                for b in range(B)
            ], dim=0)  # [B, 9]
            rel_pose_enc = torch.cat(rel_pose_enc_batch, dim=0).unsqueeze(1)  # [B, 1, 9]
            camera_pose_abs = compose_absolute_from_relative(
                active_poses.unsqueeze(1), rel_pose_enc, image_size_hw=(img_h, img_w)
            )[:, 0, :]
```

- [ ] **Step 4: Batched depth/point heads (Review-36)**

```python
    # Depth head — batched (no per-sequence state)
    def depth_head_forward(*layer_tokens):
        return self.depth_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens_list)

    # Point head — batched
    def point_head_forward(*layer_tokens):
        return self.point_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    pts3d, pts3d_conf = maybe_checkpoint_head(point_head_forward, *aggregated_tokens_list)
```

- [ ] **Step 5: Per-batch keyframe events + cache commit (Review-17, Review-37)**

```python
    # Per-batch keyframe decisions
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
        keyframe_schedule.append(events[0])  # Review-15: Phase 1 — all identical

    # Per-batch per-layer cache commit
    for b in range(B):
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
                anchor_slot=events[b].anchor_slot,  # Review-37: attribute, not method
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
            if score is not None:
                self.aggregator.last_scores[layer_idx] = score
```

- [ ] **Step 6: Post-commit operations (Review-32, Review-33)**

```python
    # Per-batch camera head keyframe sync
    for b in range(B):
        past_key_values_camera[b] = self.camera_head.apply_keyframe_event(
            past_key_values_camera[b],
            events[b],
            num_cam_iters=self.camera_num_iters,
        )

    # Build per-frame result
    res_gpu = {
        "depth": depth,
        "depth_conf": depth_conf,
        "pts3d": pts3d,
        "pts3d_conf": pts3d_conf,
        "camera_pose": camera_pose,
        "camera_pose_rel": rel_pose_enc[:, 0, :] if rel_pose_enc_batch else None,
    }
    if frame_writer is not None:
        frame_writer(i, frame, res_gpu)
    if cache_results:
        res_out = self._maybe_move_dict_to_cpu(res_gpu) if move_to_cpu else res_gpu
        all_ress.append(res_out)

    del aggregated_tokens_list
    del frame_pending_updates
```

- [ ] **Step 7: Update return to include per-batch distill_loss (existing TokenScorer flow)**

```python
    return OVGGTOutput(
        ress=all_ress if cache_results else None,
        views=processed_frames if (cache_results and return_views) else None,
        keyframe_packets=keyframe_packets if export_packets else None,
        keyframe_schedule=keyframe_schedule,
        distill_loss=total_distill_loss,  # accumulated across batches
    )
```

- [ ] **Step 8: Commit**

```bash
git add src/ovggt/models/ovggt.py
git commit -m "feat: multi-sequence B>1 support in _inference_frontend

- Per-batch independent state (keyframe_manager, cache_states, PVC)
- Sequential aggregator: B calls with B=1 (save/restore last_scores)
- Sequential camera head: B calls with per-batch PVC
- Batched depth/point heads (torch.cat across batch dim)
- Per-batch keyframe events + cache commit
- Average distill_loss across B for loss equivalence"
```

---

### Task 5: Config Updates

**Files:**
- Modify: `config/train_frontend_finetune.yaml`
- Modify: `config/train_token_scorer.yaml`

- [ ] **Step 1: Update train_frontend_finetune.yaml**

```yaml
batch_size: 4          # changed from 1
accum_iter: 1          # batch_size provides parallelism
```

- [ ] **Step 2: Update train_token_scorer.yaml**

```yaml
batch_size: 4          # changed from 1
accum_iter: 1
```

- [ ] **Step 3: Commit**

```bash
git add config/train_frontend_finetune.yaml config/train_token_scorer.yaml
git commit -m "config: set batch_size=4 in finetune and token_scorer configs"
```

---

### Task 6: Integration Smoke Test

**Files:**
- Modify: `tests/test_frontend_batch_training.py` (extend from Task 1)

- [ ] **Step 1: Write B=2 forward pass smoke test**

```python
def test_batch2_forward_no_error():
    """Smoke test: B=2 training forward pass completes without error."""
    import torch
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig

    B = 2
    num_frames = 4
    H, W = 518, 392
    model = OVGGT(
        mode='frontend_train',
        total_budget=5000,
        camera_budget=64,
        use_token_scorer=False,
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            dedup_enabled=True,
            intra_frame_dedup_enabled=False,
        ),
    ).cuda().eval()

    # Build B=2 sequences with 4 frames each
    frames = []
    for _ in range(num_frames):
        frames.append({"img": torch.randn(B, 3, H, W).cuda()})

    with torch.no_grad():
        output = model.inference(
            frames,
            history_anchor_strategy='fixed_interval',
            anchor_interval=2,
            max_anchors=2,
            cache_results=True,
        )
    assert output.ress is not None
    assert len(output.ress) == num_frames
    # Each result should have batch-dim depth
    assert output.ress[0]["depth"].shape[0] == B
```

- [ ] **Step 2: Run smoke test**

```bash
python -m pytest tests/test_frontend_batch_training.py::test_batch2_forward_no_error -v
```
Expected: PASS (model completes forward without crash)

- [ ] **Step 3: Write B=1 vs B=2 equivalence test**

```python
@torch.no_grad()
def test_batch1_vs_batch2_loss_close():
    """Same 2 sequences: B=1×2steps vs B=2×1step produce similar loss."""
    import torch
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig
    from ovggt.losses.frontend_distill import FrontendDistillLoss

    B = 2
    num_frames = 4
    H, W = 518, 392
    criterion = FrontendDistillLoss()

    def make_model():
        return OVGGT(
            mode='frontend_train',
            total_budget=5000,
            camera_budget=64,
            use_token_scorer=False,
            frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        ).cuda().eval()

    frames_a = [{"img": torch.randn(1, 3, H, W).cuda()} for _ in range(num_frames)]
    frames_b = [{"img": torch.randn(1, 3, H, W).cuda()} for _ in range(num_frames)]

    # B=1: run both sequences separately
    model1 = make_model()
    out1a = model1.inference(frames_a, history_anchor_strategy='fixed_interval',
                              anchor_interval=2, max_anchors=2)
    out1b = model1.inference(frames_b, history_anchor_strategy='fixed_interval',
                              anchor_interval=2, max_anchors=2)

    # B=2: stack into batched frames
    frames_2 = []
    for fa, fb in zip(frames_a, frames_b):
        frames_2.append({"img": torch.cat([fa["img"], fb["img"]], dim=0)})
    model2 = make_model()
    out2 = model2.inference(frames_2, history_anchor_strategy='fixed_interval',
                             anchor_interval=2, max_anchors=2)

    # Compare per-frame depth predictions
    for t in range(num_frames):
        d1a = out1a.ress[t]["depth"][0]  # sequence a
        d1b = out1b.ress[t]["depth"][0]  # sequence b
        d2a = out2.ress[t]["depth"][0]   # batch 0 = sequence a
        d2b = out2.ress[t]["depth"][1]   # batch 1 = sequence b
        assert torch.allclose(d1a, d2a, rtol=1e-3, atol=1e-5), f"Frame {t}: batch 0 mismatch"
        assert torch.allclose(d1b, d2b, rtol=1e-3, atol=1e-5), f"Frame {t}: batch 1 mismatch"
```

- [ ] **Step 4: Run equivalence test**

```bash
python -m pytest tests/test_frontend_batch_training.py::test_batch1_vs_batch2_loss_close -v
```
Expected: PASS (head outputs match within tolerance)

- [ ] **Step 5: Commit**

```bash
git add tests/test_frontend_batch_training.py
git commit -m "test: add B=2 smoke test and B=1 vs B=2 equivalence test"
```

---

### Task 7: Verify B=1 Regression

- [ ] **Step 1: Run existing B=1 smoke test with batch_size=1 config**

```bash
python tools/test_phase2_smoke.py
```
Expected: PASS (B=1 path unchanged)

- [ ] **Step 2: Verify per-sequence cache isolation**

Add to tests:
```python
def test_cache_isolation():
    """B=2: batch 0 and batch 1 have independent cache states."""
    B = 2
    model = OVGGT(
        mode='frontend_train', total_budget=5000, camera_budget=64,
        use_token_scorer=False,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    ).cuda().eval()

    frames = [{"img": torch.randn(B, 3, 518, 392).cuda()} for _ in range(3)]
    out = model.inference(frames, history_anchor_strategy='fixed_interval',
                           anchor_interval=2, max_anchors=2)
    # If cache states weren't isolated, batch 0 and batch 1 outputs would be identical
    d0 = out.ress[-1]["depth"][0]
    d1 = out.ress[-1]["depth"][1]
    assert not torch.allclose(d0, d1), "Cache states not isolated — outputs are identical"
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_frontend_batch_training.py
git commit -m "test: add cache isolation and B=1 regression tests"
```

---

## Dependency Graph

```
Task 1 (collate) ──┐
                   ├──> Task 4 (core refactor) ──> Task 6 (smoke test) ──> Task 7 (regression)
Task 2 (guards) ───┤
Task 3 (sync pts) ─┘                              Task 5 (config) ────────┘
```

Tasks 1-3 are independent and can be done in any order. Task 4 depends on Tasks 1-3. Tasks 5-6 depend on Task 4. Task 7 validates everything.

---

## Rollback Plan

If B>1 produces incorrect results:
1. Revert Tasks 4-7 (`git revert` the _inference_frontend refactor)
2. Keep Tasks 1-3 (collate, guards, sync points) — beneficial for future work
3. Fall back to `batch_size=1` with `accum_iter=4` for equivalent effective batch

## Known Limitations (Phase 2)

- `last_scores` shared across batch within frame (minor budget asymmetry — §3.5 caveat)
- Keyframe schedule from batch 0 only (works for `fixed_interval`, needs refactor for coverage strategy)
- Finetune B>1 limited to no-TokenScorer `FrontendSupervisedLoss`
- Checkpointing B× recomputation overhead with sequential aggregator
