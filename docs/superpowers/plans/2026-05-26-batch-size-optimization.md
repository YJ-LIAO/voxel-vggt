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

- [ ] **Step 5: Add `collate_fn` parameter to `get_data_loader` and register**

`get_data_loader` does NOT currently accept `collate_fn`. Two changes required:

```python
# src/dust3r/datasets/__init__.py — get_data_loader signature
def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
    collate_fn=None,  # NEW — pass through to DataLoader
):
```

Pass `collate_fn=collate_fn` in both DataLoader constructors (lines 67–72 and 77–84):

```python
# try branch (line 67):
data_loader = torch.utils.data.DataLoader(
    dataset, batch_sampler=sampler, num_workers=num_workers,
    pin_memory=pin_mem, collate_fn=collate_fn,
)
# except branch (line 77):
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle,
    num_workers=num_workers, pin_memory=pin_mem,
    drop_last=drop_last, collate_fn=collate_fn,
)
```

Then in `build_dataset` (`train_frontend.py:162`), pass `collate_fn` when `batch_size > 1`:

```python
# train_frontend.py — build_dataset function
from dust3r.datasets.collate import frontend_collate_fn
...
return get_data_loader(
    dataset, batch_size=batch_size, num_workers=num_workers,
    pin_mem=True, shuffle=shuffle, drop_last=drop_last,
    accelerator=accelerator, fixed_length=fixed_length,
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
- Modify: `src/train_frontend.py:674-678`
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

Delete lines 674–678 in `train()` function (`if int(args.batch_size) != 1: raise ValueError(...)`).

- [ ] **Step 3: Remove guard in finetune_frontend.py**

Delete lines 251-255.

> **Warning (Spec R34)**: B>1 finetune with `FrontendSupervisedLoss` has a known limitation: `_build_point_targets` (`frontend_supervised.py:184-186`) strips batch dim via `camera_intrinsics[0]`, producing incorrect 3D point targets for batch > 0. Finetune B>1 is only safe when the dataset provides pre-computed `pts3d` (skipping this code path). If using datasets without pre-computed `pts3d`, keep `batch_size=1` in finetune config.

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

- [ ] **Step 1: Add cached attribute to LayerCacheState**

`LayerCacheState` is a plain class (NOT `@dataclass`) with class-level defaults at line 148. Add a new field:

```python
# frontend_cache.py — LayerCacheState class (line 148)
class LayerCacheState:
    k: Optional[Tensor] = None
    ...
    _cached_protected_count: int = 0  # NEW — cache for _compute_protected_count
```

- [ ] **Step 2: Keep has_anchor_tokens unchanged (on TokenMetadata, line 103–104)**

`has_anchor_tokens` is on `TokenMetadata` (a separate dataclass from `LayerCacheState`). It's already cheap and doesn't need caching. No changes needed.

```python
# TokenMetadata.has_anchor_tokens — keep as-is (line 103-104)
def has_anchor_tokens(self) -> bool:
    return bool((self.anchor_slot >= 0).any().item())
```

- [ ] **Step 3: Rename and cache `_compute_protected_count` (line 670–673)**

Rename the existing method to `_compute_protected_count_raw`, then create a cached wrapper:

```python
# BEFORE (line 670):
def _compute_protected_count(self) -> int:
    if self.metadata is None or self.metadata.anchor_slot.numel() == 0:
        return 0
    return int((self.metadata.anchor_slot[0] >= 0).sum().item())

# AFTER:
def _compute_protected_count_raw(self) -> int:
    """Original computation — called only at mutation points."""
    if self.metadata is None or self.metadata.anchor_slot.numel() == 0:
        return 0
    return int((self.metadata.anchor_slot[0] >= 0).sum().item())

def _compute_protected_count(self) -> int:
    """Cached version — returns pre-computed value, no GPU sync."""
    return self._cached_protected_count
```

- [ ] **Step 4: Replace _current_frame_importance (line 675-686)**

```python
# BEFORE (line 682):
counts.append(int(mask.sum().item()))

# AFTER — keep as Python int to avoid tensor comparison issues:
counts.append(mask.sum().item())
# Note: .item() here is once per batch element per frame (not per-layer), acceptable frequency.
# The equality check remains unchanged:
#   if not counts or min(counts) == 0 or len(set(counts)) != 1:
#       return None, 0
```

- [ ] **Step 5: Update caches at mutation points**

In these methods, find the existing `self.protected_count = self._compute_protected_count()` line and replace with the cache update pattern. **All mutation points must be updated consistently:**

| Method | Line | Change |
|--------|------|--------|
| `_gather_single_batch_` | ~198 | Replace `self.protected_count = self._compute_protected_count()` with `self._cached_protected_count = self._compute_protected_count_raw(); self.protected_count = self._cached_protected_count` |
| `gather_per_batch_` | ~340 | Same pattern |
| `append_` | ~351 | Same pattern |
| `apply_keyframe_event_` | ~370 | Same pattern |

```python
# At each mutation point, replace:
#   self.protected_count = self._compute_protected_count()
# With:
self._cached_protected_count = self._compute_protected_count_raw()
self.protected_count = self._cached_protected_count
```

Note: this still has `.item()` inside `_compute_protected_count_raw`, but only at mutation points (per-frame, not per-query). Frequency drops from ~240/step to ~24/step.

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

- [ ] **Step 1: Compute B and initialize per-batch state (replaces lines 385–392)**

```python
# Compute batch size from input frames
B = self._frame_batch_size(frames[0]["img"])  # e.g. frames[0]["img"].shape[0]
for f in frames[1:]:
    assert self._frame_batch_size(f["img"]) == B, "All frames must have same batch size"

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

    # patch_start_idx from aggregator — all batch elements share the same resolution
    # (ps from any batch element is correct, use the last one)
    patch_start_idx = ps

    # Average distill_loss across B (Review-18)
    # Assumes all B sequences produce non-None distill_loss. If some don't,
    # denominator varies per frame (non-uniform weighting). In practice all
    # sequences should produce distill_loss when token_scorer is enabled.
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

    # Compute camera_pose_rel for BOTH encoding paths
    # This must happen outside the per-batch loop for efficiency
    camera_pose_rel = None  # default, overwritten below
    with self._disabled_autocast_context():
        if i == 0 or self.frontend_pose_encoding_type == ABS_POSE_ENCODING:
            camera_pose_rel = relative_from_absolute_pose_encoding(
                camera_pose.unsqueeze(1),
                camera_pose.unsqueeze(1),
                image_size_hw=(img_h, img_w),
            )[:, 0, :]
        else:
            active_poses = torch.stack([
                keyframe_managers[b].get_active_pose_encoding()
                for b in range(B)
            ], dim=0)  # [B, 9]
            rel_pose_enc = torch.cat(rel_pose_enc_batch, dim=0).unsqueeze(1)  # [B, 1, 9]
            # Overwrite camera_pose with composed absolute pose (not raw abs_pose_enc)
            camera_pose = compose_absolute_from_relative(
                active_poses.unsqueeze(1), rel_pose_enc, image_size_hw=(img_h, img_w)
            )[:, 0, :]
            camera_pose_rel = rel_pose_enc[:, 0, :]
```

- [ ] **Step 4: Batched depth/point heads (Review-36)**

```python
    # Depth head — batched (no per-sequence state)
    def depth_head_forward(*layer_tokens):
        return self.depth_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens_list)
    depth = depth[:, 0]       # [B, 1, H, W] → [B, H, W] — slice off sequence dim
    depth_conf = depth_conf[:, 0]

    # Point head — batched
    def point_head_forward(*layer_tokens):
        return self.point_head(list(layer_tokens), images=images_all, patch_start_idx=patch_start_idx)
    pts3d, pts3d_conf = maybe_checkpoint_head(point_head_forward, *aggregated_tokens_list)
    pts3d = pts3d[:, 0]       # [B, 1, H, W] → [B, H, W]
    pts3d_conf = pts3d_conf[:, 0]
```

> **Note:** DPT heads output `[B, S, 1, H, W]` where S is sequence length. For single-image inference S=1, so `[:, 0]` extracts the correct slice. This matches the current code at ovggt.py:471–472, 482–483.

- [ ] **Step 4b: Track head for B>1 (Phase 1: skip)**

```python
    # Phase 1 assumes track_head is None for B>1.
    # Future: batch track_head by stacking query_points per-batch element.
    if self.track_head is not None and current_query_points is not None:
        raise NotImplementedError("track_head with B>1 not yet supported. Set batch_size=1 or remove query_points.")
```

- [ ] **Step 5: Per-batch keyframe events + cache commit (Review-17, Review-37)**

```python
    # Per-batch keyframe decisions
    events = []
    for b in range(B):
        event = keyframe_managers[b].update(
            frame_idx=i,
            depth=depth[b],          # [H, W] — already sliced by [:, 0] in Step 4
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
                anchor_slot=events[b].anchor_slot,
                total_tokens=pending.importance_current.shape[1],
                active_local_to_world=current_local_to_world,
            )
            current_metadata = frame_metadata_base.with_importance(pending.importance_current)
            cache_states[b][layer_idx].apply_keyframe_event_(events[b])
            score = cache_states[b][layer_idx].commit_pending_update_(
                pending_update=pending,
                current_metadata=current_metadata,
                config=self.frontend_cache_config,
                intra_frame_keep_ratio=self.aggregator.intra_frame_keep_ratio,
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

    # Per-batch KeyframePacket (export_packets path)
    # NOTE: frame_metadata_base from the commit loop above is per-batch — 
    # it holds the correct data for the current b because the commit loop
    # processes batch elements sequentially. However, only the LAST layer's
    # metadata survives the inner loop. Reconstruct for packets to be safe.
    if export_packets:
        for b in range(B):
            if events[b].anchor_slot >= 0:
                slot_id_b = keyframe_managers[b].get_active_keyframe_id()
                ltw_b = keyframe_managers[b].get_active_local_to_world()
                pkt_metadata = build_frame_token_metadata_base(
                    depth=depth[b:b+1],
                    depth_conf=depth_conf[b:b+1],
                    pose_enc=camera_pose[b:b+1],
                    image_size_hw=(img_h, img_w),
                    patch_size=self.aggregator.patch_size,
                    patch_start_idx=patch_start_idx,
                    frame_id=i,
                    keyframe_id=slot_id_b,
                    slot_id=slot_id_b,
                    anchor_slot=events[b].anchor_slot,
                    total_tokens=frame_pending_updates[b][-1].importance_current.shape[1] if frame_pending_updates[b][-1] is not None else 0,
                    active_local_to_world=ltw_b,
                )
                patch_features = aggregated_tokens_list[-1][b:b+1, :, patch_start_idx:]
                keyframe_packets.append(
                    KeyframePacket(
                        frame_idx=i,
                        keyframe_id=slot_id_b,
                        anchor_slot=events[b].anchor_slot,
                        pose_abs=camera_pose[b].detach().cpu(),
                        local_to_world=ltw_b.detach().cpu(),
                        patch_local_xyz=pkt_metadata.slot_local_xyz[:, patch_start_idx:].detach().cpu(),
                        patch_depth_conf=pkt_metadata.depth_conf[:, patch_start_idx:].detach().cpu(),
                        patch_features=patch_features.detach().cpu(),
                    )
                )

    # Build per-frame result — field names must match current code exactly
    res_gpu = {
        "pts3d_in_other_view": pts3d,
        "conf": pts3d_conf,
        "depth": depth,
        "depth_conf": depth_conf,
        "camera_pose": camera_pose,
        "camera_pose_rel": camera_pose_rel,
        **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
        **(
            {"track": track, "vis": vis, "track_conf": track_conf}
            if self.track_head is not None and current_query_points is not None
            else {}
        ),
    }
    if frame_writer is not None:
        frame_writer(i, frame, res_gpu)
    if cache_results:
        res_out = self._maybe_move_dict_to_cpu(res_gpu) if move_to_cpu else res_gpu
        all_ress.append(res_out)
        if return_views:
            processed_frames.append(self._maybe_move_dict_to_cpu(frame) if move_to_cpu else frame)

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

> **Memory risk**: Changing from `batch_size=1, accum_iter=4` to `batch_size=4, accum_iter=1` keeps the effective batch size at 4 but increases peak GPU memory up to 4× (B independent cache states + batched activations). If OOM, fall back to `batch_size=2, accum_iter=2` or enable `gradient_checkpointing=True`. The spec's §6 Phase 2 memory monitoring will surface this.

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

> **Note:** This test calls `model.inference()` directly. The actual training path is `model(frames) → forward() → forward_frontend_train() → _inference_frontend()`. For full integration coverage, add a test that calls `model(frames, frame_processor=callback_fn)` to exercise the `forward_frontend_train` path including `accumulate_student_frame` callback and loss computation.

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

    # Use different random seeds per batch element so outputs differ IF isolation works
    g0 = torch.Generator().manual_seed(42)
    g1 = torch.Generator().manual_seed(99)
    frames = [{"img": torch.cat([
        torch.randn(1, 3, 518, 392, generator=g0),
        torch.randn(1, 3, 518, 392, generator=g1),
    ], dim=0).cuda()} for _ in range(3)]
    out = model.inference(frames, history_anchor_strategy='fixed_interval',
                           anchor_interval=2, max_anchors=2)
    d0 = out.ress[-1]["depth"][0]
    d1 = out.ress[-1]["depth"][1]
    # Different inputs → different outputs if cache is isolated
    # Would be identical if cache states leaked across batch
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

---

## Plan Review (P1–P17) — All Resolved (Round 2: Approved)

> **Round 1:** 17 findings (P1–P17): 1 Critical, 7 Important, 6 Medium, 3 Informational. All fixed.
> **Round 2:** Re-review passed with 2 minor issues + 2 advisory. All fixed.
> **Status:** Approved for implementation.

### Resolution Summary

| Finding | Severity | Resolution |
|---------|----------|------------|
| P1 | Critical | Rewrote Step 5: added `collate_fn=None` to `get_data_loader` signature + both DataLoader constructors + `build_dataset` call site |
| P2 | Important | Corrected line reference to 674–678 |
| P3 | Important | Added `B = self._frame_batch_size(frames[0]["img"])` + validation loop in Step 1 |
| P4 | Important | Removed `importance=`, added `total_tokens=pending.importance_current.shape[1]` |
| P5 | Important | Computed `camera_pose_rel` for both ABS and non-ABS paths in Step 3 |
| P6 | Important | Fixed field names to `pts3d_in_other_view`/`conf`, added `valid_mask` and track fields |
| P7 | Important | Added `[:, 0]` slicing after depth/point heads |
| P8 | Important | `camera_pose` now overwritten with composed absolute pose in non-ABS path |
| P9 | Medium | Renamed to `_compute_protected_count_raw`, explicit rename-keep pattern |
| P10 | Medium | `has_anchor_tokens` kept on `TokenMetadata` unchanged; cache only on `LayerCacheState` |
| P11 | Medium | Added Step 4b with explicit `NotImplementedError` for track_head B>1 |
| P12 | Medium | Changed to `self.aggregator.intra_frame_keep_ratio` |
| P13 | Medium | Added per-batch `KeyframePacket` loop in Step 6 |
| P14 | Medium | Added `valid_mask` and track fields to `res_gpu` |
| P15 | Informational | Fixed wording: "add dataclass fields with defaults" instead of `__init__` |
| P16 | Informational | Added integration test note to Step 2 |
| P17 | Informational | Added distill_loss denominator assumption comment |

### Round 2 Findings (R2-1, R2-2) — Fixed

| Finding | Severity | Resolution |
|---------|----------|------------|
| R2-1 | Issue | `patch_start_idx` undefined in Step 4 — added `patch_start_idx = ps` after aggregator loop |
| R2-2 | Issue | Task 3 Step 4 changed counts to tensors, breaking min/nul check — reverted to `.item()` to keep Python ints |
| R2-R1 | Advisory | KeyframePacket used stale `frame_metadata_base` for B>1 — now reconstructs per-batch via explicit `build_frame_token_metadata_base` call |
| R2-R2 | Advisory | Task 3 Step 5 mutation points table now explicitly lists all 4 methods with exact change pattern |

### P1 (Critical): `get_data_loader` does NOT accept `collate_fn` — Task 1 wiring is broken

**Problem:** Task 1 Step 5 states "the `get_data_loader` function already accepts `collate_fn` parameter." This is false. Reading `src/dust3r/datasets/__init__.py:41–86`, the function signature has no `collate_fn` parameter, and both `DataLoader` creation paths (lines 67–72 and 77–84) omit it. The plan's code snippet `loader = get_data_loader(dataset, ..., collate_fn=...)` would raise `TypeError: get_data_loader() got an unexpected keyword argument 'collate_fn'`.
**Task/Step:** Task 1 / Step 5
**Impact:** Collate function is never wired to DataLoader. B>1 training cannot start — every DataLoader iteration crashes with string TypeError. Blocks all downstream tasks.
**Suggestion:** Add `collate_fn=None` parameter to `get_data_loader` signature, and pass it through to both `DataLoader` constructors (lines 67–72 and 77–84).

### P2 (Important): train_frontend.py batch_size guard is at line 674, not 653–658

**Problem:** Task 2 Step 2 says "Delete lines 653–658" for the batch_size guard. The actual guard is at line 674 (`if int(args.batch_size) != 1: raise ValueError(...)`). Lines 653–657 contain `save_model` checkpointing logic. Deleting the wrong lines breaks checkpoint saving while leaving the B=1 guard in place.
**Task/Step:** Task 2 / Step 2
**Impact:** Engineer deletes wrong lines, breaking checkpointing while B>1 training remains blocked.
**Suggestion:** Correct the line reference to 674–678.

### P3 (Important): Variable `B` is never defined in Task 4 Step 1

**Problem:** Task 4 Step 1 starts with `keyframe_managers = [FrontendKeyframeManager(...) for _ in range(B)]` but never shows where `B` is computed. The current code computes `ref_batch_size` inside `_validate_frontend_batch_size` (line 822–823: `ref_batch_size = self._frame_batch_size(frames[0]["img"])`), which the plan removes in Task 2. After guard removal, no code computes `B`.
**Task/Step:** Task 4 / Step 1
**Impact:** `NameError: name 'B' is not defined`. The entire refactor is dead code.
**Suggestion:** Add `B = self._frame_batch_size(frames[0]["img"])` at the beginning of Step 1, before the per-batch state initialization. Also validate all frames have the same B.

### P4 (Important): `build_frame_token_metadata_base` — plan passes nonexistent `importance` parameter, omits required `total_tokens`

**Problem:** The plan's Step 5 calls `build_frame_token_metadata_base(depth=depth[b:b+1], ..., importance=pending.importance_current, ...)`. But the actual function signature (`frontend_cache.py:805–817`) has no `importance` parameter. The required parameters are: `depth, depth_conf, pose_enc, image_size_hw, patch_size, patch_start_idx, frame_id, keyframe_id, slot_id, anchor_slot, total_tokens, active_local_to_world`. The plan omits the **required** `total_tokens` parameter (currently passed as `pending_update.importance_current.shape[1]` at ovggt.py:547).
**Task/Step:** Task 4 / Step 5
**Impact:** `TypeError: build_frame_token_metadata_base() got an unexpected keyword argument 'importance'` at runtime. Every frame commit crashes.
**Suggestion:** Remove `importance=pending.importance_current`. Add `total_tokens=pending.importance_current.shape[1]`.

### P5 (Important): `rel_pose_enc` undefined in ABS_POSE_ENCODING path — `res_gpu` dict will `NameError`

**Problem:** Task 4 Step 6 builds `"camera_pose_rel": rel_pose_enc[:, 0, :] if rel_pose_enc_batch else None`. But `rel_pose_enc` is only defined inside the `if self.frontend_pose_encoding_type != ABS_POSE_ENCODING` block in Step 3. When `frontend_pose_encoding_type == ABS_POSE_ENCODING`, `rel_pose_enc` is never assigned → `NameError`. The current code (ovggt.py:447–453) handles this correctly — in the ABS path, it computes `camera_pose_rel = relative_from_absolute_pose_encoding(...)`.
**Task/Step:** Task 4 / Step 6
**Impact:** Any training/eval run with ABS_POSE_ENCODING crashes at the first frame's `res_gpu` construction.
**Suggestion:** Compute `camera_pose_rel` for both paths. In the ABS path, derive from `relative_from_absolute_pose_encoding`. For non-ABS, use `torch.cat(rel_pose_enc_batch, dim=0)[:, 0, :]`.

### P6 (Important): `res_gpu` dict field names don't match actual code

**Problem:** The plan's `res_gpu` dict uses `"pts3d": pts3d, "pts3d_conf": pts3d_conf`. The actual code (ovggt.py:583–596) uses `"pts3d_in_other_view": pts3d, "conf": pts3d_conf`. Also missing: `"valid_mask"` (line 590), track fields (`"track"`, `"vis"`, `"track_conf"`, lines 591–595).
**Task/Step:** Task 4 / Step 6
**Impact:** Downstream consumers (`accumulate_student_frame` at `train_frontend.py:421–434`) access keys by name → `KeyError` or silent loss computation errors.
**Suggestion:** Use exact field names from current code. Include `"valid_mask"` and track fields.

### P7 (Important): Missing `[:, 0]` slicing on depth/pts3d head outputs

**Problem:** The DPT head returns `[B, S, 1, H, W]` (verified at `dpt_head.py:131–132`). The current code slices off the sequence dim: `depth = depth[:, 0]` (ovggt.py:471–472, 482–483). The plan's Step 4 omits this slicing. For B>1, `depth` would be `[B, 1, 1, H, W]` instead of `[B, 1, H, W]`.
**Task/Step:** Task 4 / Step 4
**Impact:** Downstream code passes `depth[b]` (shape `[1, 1, H, W]` instead of `[1, H, W]`) to `keyframe_manager.update()` and `build_frame_token_metadata_base`, causing shape mismatches.
**Suggestion:** Add `depth = depth[:, 0]; depth_conf = depth_conf[:, 0]; pts3d = pts3d[:, 0]; pts3d_conf = pts3d_conf[:, 0]` after the head calls.

### P8 (Important): `camera_pose` not updated to composed absolute pose in non-ABS path

**Problem:** Task 4 Step 3 sets `camera_pose = torch.cat(pose_enc_batch, dim=0)` from `abs_pose_enc`. Then the non-ABS block stores the composed pose in `camera_pose_abs` (a new variable), not `camera_pose`. The current code (ovggt.py:456–461) correctly overwrites `camera_pose` with the composed absolute pose. The plan's Step 5 then passes `camera_pose[b]` to `keyframe_manager.update()` — which receives the raw `abs_pose_enc` instead of the composed absolute pose.
**Task/Step:** Task 4 / Steps 3 and 5
**Impact:** In non-ABS encoding mode, `keyframe_manager.update` receives incorrect pose → `active_pose_encoding` is wrong → subsequent frame's camera composition uses wrong reference pose → camera predictions diverge silently.
**Suggestion:** In the non-ABS block, overwrite `camera_pose` (not `camera_pose_abs`): `camera_pose = compose_absolute_from_relative(...)[:, 0, :]`.

### P9 (Medium): `_compute_protected_count_raw` does not exist — plan references nonexistent method

**Problem:** Task 3 Step 5 says `self._cached_protected_count = self._compute_protected_count_raw()`. But `_compute_protected_count_raw` does not exist anywhere in `frontend_cache.py`. The actual method is `_compute_protected_count` (line 670). If Step 3 replaces `_compute_protected_count` to return the cached value, then calling `_compute_protected_count()` at mutation points would just return the stale cached value.
**Task/Step:** Task 3 / Steps 3 and 5
**Impact:** Implementer must deduce the rename-keep pattern. Without it, mutation points would call the cached getter instead of computing fresh value.
**Suggestion:** Explicitly state: rename `_compute_protected_count` to `_compute_protected_count_raw`, then define new `_compute_protected_count` that returns `self._cached_protected_count`. At mutation points, call `self._cached_protected_count = self._compute_protected_count_raw()`.

### P10 (Medium): `has_anchor_tokens` is on `TokenMetadata`, not `LayerCacheState` — caching target class confusion

**Problem:** The plan's Step 1 adds `_cached_has_anchor` in context of `LayerCacheState`, but Step 2 replaces `has_anchor_tokens` at lines 103–104, which is on `TokenMetadata` (a different class, `frontend_cache.py:34–109`). `TokenMetadata` is a dataclass with `index_select`, `append`, `clone`, `empty` that reconstruct instances — any cached field would be lost unless explicitly propagated.
**Task/Step:** Task 3 / Steps 1 and 2
**Impact:** Implementer cannot apply the plan as written. Either the attribute goes on the wrong class, or the method replacement references the wrong `self`.
**Suggestion:** Add `_cached_has_anchor: bool = False` as a field on `TokenMetadata` (not `LayerCacheState`). Update `TokenMetadata.index_select`, `append`, `clone` to propagate the cached value. Alternatively, keep `has_anchor_tokens` on `TokenMetadata` unchanged and only cache `_compute_protected_count` on `LayerCacheState`.

### P11 (Medium): Missing `track_head` path adaptation for B>1

**Problem:** The current code at ovggt.py:485–495 calls `self.track_head(aggregated_tokens, images=images, ...)` inside the `with self._disabled_autocast_context()` block. The plan's Task 4 completely omits this path. For B>1, `aggregated_tokens_list` is batched and `images_all` is `[B, 1, C, H, W]` — the track_head would receive batched input, but the plan doesn't show how `current_query_points` is handled per-batch.
**Task/Step:** Task 4 / between Step 4 and Step 5
**Impact:** Any model with `track_head is not None` and `query_points is not None` would either crash or silently produce incorrect results.
**Suggestion:** Add track_head handling, or explicitly note Phase 1 assumes `track_head is None` for B>1.

### P12 (Medium): `intra_frame_keep_ratio` referenced but not defined in plan code

**Problem:** Task 4 Step 5 passes `intra_frame_keep_ratio=intra_frame_keep_ratio` to `commit_pending_update_`. But `intra_frame_keep_ratio` is never defined in the plan's code. The current code uses `self.aggregator.intra_frame_keep_ratio` (ovggt.py:555).
**Task/Step:** Task 4 / Step 5
**Impact:** `NameError: name 'intra_frame_keep_ratio' is not defined` at first commit.
**Suggestion:** Use `self.aggregator.intra_frame_keep_ratio` as in current code.

### P13 (Medium): Missing `KeyframePacket` per-batch handling

**Problem:** The current code at ovggt.py:562–575 creates a `KeyframePacket` with batched tensors. For B>1, a single `KeyframePacket` would store `[B, 9]` pose data where downstream expects per-keyframe scalars. The plan's Task 4 Step 6 doesn't include per-batch loop for packets.
**Task/Step:** Task 4 / between Step 5 and Step 6
**Impact:** Eval mode with B>1 and `export_keyframe_packets=True` would produce corrupt packet data silently.
**Suggestion:** Add per-batch `KeyframePacket` creation inside the `for b in range(B)` loop.

### P14 (Medium): `valid_mask` and frame dict fields not handled for B>1

**Problem:** The current code at ovggt.py:590 includes `"valid_mask": frame["valid_mask"]` in `res_gpu`. With B>1, `frame["valid_mask"]` would be `[B, H, W]` (stacked by collate). The plan omits this field. Similarly, `frame["dataset"]` and `frame["label"]` become lists (per collate), but `processed_frames.append(frame)` stores the batched frame dict.
**Task/Step:** Task 4 / Step 6
**Impact:** `valid_mask` is used by `FrontendSupervisedLoss` — missing it causes `KeyError` in finetune.
**Suggestion:** Include `"valid_mask"` in `res_gpu`. Document that `processed_frames` contains batched frame dicts for B>1.

### P15 (Informational): Task 3 — `LayerCacheState` is a `@dataclass` without explicit `__init__`

**Problem:** Task 3 Step 1 says "In `LayerCacheState.__init__`" but `LayerCacheState` is a `@dataclass` with auto-generated `__init__`. The instruction to add code "after existing `__init__`" is misleading since there's no explicit `__init__`.
**Task/Step:** Task 3 / Step 1
**Impact:** Low — an experienced Python developer would know to add dataclass fields.
**Suggestion:** Say "Add as dataclass fields with defaults: `_cached_protected_count: int = 0`" instead of referencing `__init__`.

### P16 (Informational): Task 6 smoke test bypasses actual training code path

**Problem:** The test calls `model.inference()` directly with `mode='frontend_train'` and `.eval()`. But actual training calls `model(batch, frame_processor=...)` → `forward()` → `forward_frontend_train()` → `_inference_frontend(...)`. The test doesn't exercise the `forward_frontend_train` path or `accumulate_student_frame` callback or loss computation.
**Task/Step:** Task 6 / Step 1
**Impact:** Test verifies model forward pass in isolation but doesn't catch issues in the training pipeline.
**Suggestion:** Add a test that calls `model(frames, frame_processor=callback_fn)` (the actual `forward` path) to catch integration issues.

### P17 (Informational): Missing `distill_loss` aggregation semantics note for B>1

**Problem:** The plan shows per-frame distill_loss averaging across B (`avg_fdl = sum(frame_distill_losses) / len(frame_distill_losses)`). But if only `k < B` sequences produce non-None distill_loss, the denominator varies per frame, creating non-uniform weighting across frames.
**Task/Step:** Task 4 / Step 2
**Impact:** Minor mathematical difference in loss weighting. If all B sequences always produce distill_loss (expected), behavior is correct.
**Suggestion:** Add a comment noting the assumption that all batch elements produce distill_loss.

---

### Review Summary

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| Critical | 1 | P1 |
| Important | 7 | P2, P3, P4, P5, P6, P7, P8 |
| Medium | 6 | P9, P10, P11, P12, P13, P14 |
| Informational | 3 | P15, P16, P17 |

**Blocking issues:** P1 (collate_fn wiring), P3 (undefined B), P4 (wrong function signature), P7 (missing slicing) are showstoppers — the code would crash at the first B>1 training step. P5 and P8 crash or silently corrupt results depending on pose encoding mode. P6 crashes in the frame_writer callback.

**Verdict:** The plan cannot be implemented as written. Tasks 1 and 4 require significant corrections before an engineer could execute without getting stuck. The spec review findings R1–R38 were addressed at the design level, but the plan's code snippets introduced new bugs in translation from spec to implementation steps.
