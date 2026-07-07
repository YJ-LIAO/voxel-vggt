# Frontend Cache Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully integrate the production frontend cache path from `/path/to/mount/lyj/voxel-vggt` into `/Train/LYJ/workspace/OVGGT`.

**Architecture:** Preserve the existing legacy `OVGGT.inference()` behavior, and add a production frontend path driven by `FrontendKeyframeManager`, `LayerCacheState`, voxel metadata/dedup, and deferred per-layer cache commits. The frontend path should be selected by `mode in {"frontend_train", "frontend_eval"}` or an enabled `FrontendCacheConfig`.

**Tech Stack:** Python 3.11, PyTorch, pytest/unittest-compatible tests, existing `ovggt` model package.

---

## File Structure

- Create `src/ovggt/utils/frontend_keyframe.py`: production keyframe event manager and event dataclasses.
- Create `src/ovggt/utils/frontend_cache.py`: production cache state, token metadata, voxel hashing/dedup, and pending update commit logic.
- Modify `src/ovggt/layers/attention.py`: production overflow policy, stable score normalization, and frontend-compatible cache return behavior.
- Modify `src/ovggt/layers/block.py`: deferred cache update mode and optional checkpoint support used by frontend cache.
- Modify `src/ovggt/models/aggregator.py`: add `frontend_cache_mode`, per-layer `PendingLayerUpdate`, and per-layer budget semantics.
- Modify `src/ovggt/heads/camera_head.py`: add relative pose support and `apply_keyframe_event()` if not already present.
- Modify `src/ovggt/models/ovggt.py`: add frontend config, frontend inference path, and legacy compatibility aliases.
- Create tests under `tests/`: keyframe manager, frontend cache, voxel hash, FIFO transform retention, protected-ring config, and compile/import smoke tests.

## Task 1: Add Keyframe Manager Tests

**Files:**
- Create: `tests/test_keyframe_manager.py`
- Test: `tests/test_keyframe_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_keyframe_manager.py` by copying the production test and removing hard-coded source-repo path insertion:

```python
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.utils.frontend_keyframe import (
    FrontendKeyframeManager,
    KeyframeEventType,
    KeyframeSwitchConfig,
)


def make_pose(tx=0.0):
    return torch.tensor([tx, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32)


class FrontendKeyframeManagerTests(unittest.TestCase):
    def test_frame_zero_initializes_global_anchor(self):
        manager = FrontendKeyframeManager(KeyframeSwitchConfig())
        event = manager.update(0, torch.ones(4, 4), make_pose(0.0), (4, 4))
        self.assertEqual(event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event.anchor_slot, 0)
        self.assertEqual(manager.get_num_anchor_frames(), 1)
        self.assertIn(0, event.slot_pose_updates)

    def test_coverage_noop_for_identical_pose(self):
        manager = FrontendKeyframeManager(KeyframeSwitchConfig(strategy="coverage", coverage_threshold=0.5))
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event = manager.update(1, torch.ones(8, 8), make_pose(0.0), (8, 8))
        self.assertEqual(event.event_type, KeyframeEventType.NOOP)
        self.assertEqual(event.anchor_slot, -1)

    def test_translation_threshold_triggers_keyframe(self):
        config = KeyframeSwitchConfig(
            strategy="coverage",
            coverage_threshold=0.0,
            translation_threshold=0.1,
        )
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event = manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        self.assertEqual(event.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event.anchor_slot, 1)

    def test_fixed_interval_fifo_rotates_history_slots(self):
        config = KeyframeSwitchConfig(strategy="fixed_interval", interval=1, max_history_anchors=1)
        manager = FrontendKeyframeManager(config)
        manager.update(0, torch.ones(8, 8), make_pose(0.0), (8, 8))
        event1 = manager.update(1, torch.ones(8, 8), make_pose(1.0), (8, 8))
        event2 = manager.update(2, torch.ones(8, 8), make_pose(2.0), (8, 8))
        self.assertEqual(event1.event_type, KeyframeEventType.PROMOTE_KEYFRAME)
        self.assertEqual(event2.event_type, KeyframeEventType.FIFO_SWAP)
        self.assertEqual(event2.demoted_slot, 1)
        self.assertEqual(manager.get_num_anchor_frames(), 2)
        self.assertIn(event2.keyframe_id, event2.slot_pose_updates)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m unittest tests.test_keyframe_manager -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ovggt.utils.frontend_keyframe'`.

- [ ] **Step 3: Add production keyframe manager module**

Copy the production module exactly:

```bash
cp /path/to/mount/lyj/voxel-vggt/src/ovggt/utils/frontend_keyframe.py \
   /Train/LYJ/workspace/OVGGT/src/ovggt/utils/frontend_keyframe.py
```

- [ ] **Step 4: Run keyframe tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m unittest tests.test_keyframe_manager -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_keyframe_manager.py src/ovggt/utils/frontend_keyframe.py
git commit -m "feat: add frontend keyframe manager"
```

## Task 2: Add Frontend Cache Module and Core Tests

**Files:**
- Create: `tests/test_p4_voxel_hash_collision_free.py`
- Create: `tests/test_frontend_cache.py`
- Create: `tests/test_p1_protect_topk_budget_ceiling.py`
- Create: `tests/test_p5_fifo_swap_transform_retention.py`
- Create: `src/ovggt/utils/frontend_cache.py`

- [ ] **Step 1: Write failing voxel hash test**

Create `tests/test_p4_voxel_hash_collision_free.py` with target-local imports:

```python
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def old_hash(voxels):
    mult = torch.tensor([1, 1000, 1000000], dtype=torch.long)
    return (voxels.to(torch.long) * mult).sum(-1)


def test_no_collision_large_coords():
    from ovggt.utils.frontend_cache import voxel_hash_collision_free

    torch.manual_seed(0)
    voxels = torch.randint(-2500, 2501, (50000, 3))
    h = voxel_hash_collision_free(voxels)
    vox_unique = len(set(map(tuple, voxels.tolist())))
    hash_unique = len(set(h.tolist()))
    assert hash_unique == vox_unique


def test_no_collision_negative():
    from ovggt.utils.frontend_cache import voxel_hash_collision_free

    v = torch.tensor([[1000, 0, 0], [0, 1, 0], [-5, -5, -5]], dtype=torch.long)
    h = voxel_hash_collision_free(v)
    assert len(set(h.tolist())) == 3
    assert old_hash(v)[0].item() == old_hash(v)[1].item()
```

- [ ] **Step 2: Run voxel hash test to verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_p4_voxel_hash_collision_free.py
```

Expected: FAIL with missing `ovggt.utils.frontend_cache`.

- [ ] **Step 3: Add production frontend cache module**

Copy the production module exactly:

```bash
cp /path/to/mount/lyj/voxel-vggt/src/ovggt/utils/frontend_cache.py \
   /Train/LYJ/workspace/OVGGT/src/ovggt/utils/frontend_cache.py
```

- [ ] **Step 4: Run voxel hash test**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_p4_voxel_hash_collision_free.py
```

Expected: PASS.

- [ ] **Step 5: Add broader cache tests**

Copy these production tests, then remove hard-coded `/path/to/mount/lyj/voxel-vggt/src` insertions and keep only target-local `ROOT = .../src` inserts:

```bash
cp /path/to/mount/lyj/voxel-vggt/tests/test_frontend_cache.py \
   /Train/LYJ/workspace/OVGGT/tests/test_frontend_cache.py
cp /path/to/mount/lyj/voxel-vggt/tests/test_p1_protect_topk_budget_ceiling.py \
   /Train/LYJ/workspace/OVGGT/tests/test_p1_protect_topk_budget_ceiling.py
cp /path/to/mount/lyj/voxel-vggt/tests/test_p5_fifo_swap_transform_retention.py \
   /Train/LYJ/workspace/OVGGT/tests/test_p5_fifo_swap_transform_retention.py
```

Edit each copied test so the only path setup is:

```python
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

- [ ] **Step 6: Run frontend cache tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q \
  tests/test_frontend_cache.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p5_fifo_swap_transform_retention.py
```

Expected: PASS. Do not add `TokenScorer`; the learned retention route is intentionally excluded from this migration.

- [ ] **Step 7: Commit**

```bash
git add src/ovggt/utils/frontend_cache.py \
  tests/test_p4_voxel_hash_collision_free.py \
  tests/test_frontend_cache.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p5_fifo_swap_transform_retention.py
git commit -m "feat: add frontend cache state and tests"
```

## Task 3: Update Attention and Block for Deferred Frontend Cache

**Files:**
- Modify: `src/ovggt/layers/attention.py`
- Modify: `src/ovggt/layers/block.py`
- Test: `tests/test_frontend_cache.py`

- [ ] **Step 1: Add regression test for anchor overflow policy**

Append to `tests/test_frontend_cache.py`:

```python
def test_attention_global_plus_recent_anchor_overflow_keeps_global_anchor():
    attn = Attention(dim=8, num_heads=2)
    attn.anchor_overflow_policy = "global_plus_recent"
    k = torch.arange(1 * 2 * 6 * 4, dtype=torch.float32).reshape(1, 2, 6, 4)
    v = k + 1000.0

    final_k, final_v, _, kept = attn.eviction(
        k,
        v,
        cache_budget=3,
        num_anchor_tokens=5,
    )

    assert kept.tolist() == [[0, 3, 4]]
    assert torch.equal(final_k, k[:, :, [0, 3, 4], :])
    assert torch.equal(final_v, v[:, :, [0, 3, 4], :])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_cache.py::test_attention_global_plus_recent_anchor_overflow_keeps_global_anchor
```

Expected: FAIL because the target `Attention` does not yet implement `anchor_overflow_policy` behavior.

- [ ] **Step 3: Port production attention changes**

Apply the production changes from:

```text
/path/to/mount/lyj/voxel-vggt/src/ovggt/layers/attention.py
```

Required implementation details:

```python
def _normalize_scores(scores: Tensor, neutral_value: float = 0.5) -> Tensor:
    score_min = scores.min(dim=-1, keepdim=True)[0]
    score_max = scores.max(dim=-1, keepdim=True)[0]
    denom = score_max - score_min
    normalized = (scores - score_min) / (denom + 1e-8)
    if scores.shape[-1] == 0:
        return normalized
    equal_mask = denom <= 1e-8
    if equal_mask.any():
        normalized = torch.where(
            equal_mask.expand_as(normalized),
            torch.full_like(normalized, neutral_value),
            normalized,
        )
    return normalized
```

Add `self.anchor_overflow_policy = "recent"` in `Attention.__init__`, add `_select_anchor_indices_on_overflow()`, remove the old `window_token_count` retention branch, and use the production `eviction()` implementation that returns global-plus-recent anchor indices when candidate budget is zero.

- [ ] **Step 4: Port production block changes**

Apply production `Block.forward()` changes from:

```text
/path/to/mount/lyj/voxel-vggt/src/ovggt/layers/block.py
```

Required behavior:

```python
def forward(..., frontend_cache_mode: bool = False, patch_grid_size: Optional[Tuple[int, int]] = None):
    if frontend_cache_mode:
        # Run attention with defer_eviction=True.
        # Return x_after_mlp, (k_current, v_current), new_importance.
```

Also add:

```python
from torch.utils.checkpoint import checkpoint
self.use_checkpoint = False
```

Use production `current_frame_keys()` for baseline importance so legacy eviction does not infer current keys from an already-evicted cache.

- [ ] **Step 5: Run cache tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q \
  tests/test_frontend_cache.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p5_fifo_swap_transform_retention.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/layers/attention.py src/ovggt/layers/block.py tests/test_frontend_cache.py
git commit -m "feat: support deferred frontend cache updates"
```

## Task 4: Update Aggregator for Frontend Pending Updates

**Files:**
- Modify: `src/ovggt/models/aggregator.py`
- Test: `tests/test_frontend_cache.py`

- [ ] **Step 1: Add aggregator frontend-mode smoke test**

Create `tests/test_frontend_aggregator_smoke.py`:

```python
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.aggregator import Aggregator
from ovggt.utils.frontend_cache import FrontendCacheConfig, LayerCacheState, PendingLayerUpdate


def test_aggregator_frontend_cache_mode_returns_pending_updates_with_conv_patch_embed():
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_register_tokens=1,
        patch_embed="conv",
    )
    cache_states = [LayerCacheState() for _ in range(model.depth)]
    images = torch.rand(1, 1, 3, 28, 28)
    outputs, patch_start_idx, returned_states, pending, distill = model(
        images,
        cache_states=cache_states,
        use_cache=True,
        past_frame_idx=0,
        per_layer_budget=16,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    )

    assert len(outputs) == model.depth
    assert patch_start_idx == 2
    assert returned_states is cache_states
    assert distill is None
    assert len(pending) == model.depth
    assert all(isinstance(item, PendingLayerUpdate) for item in pending)
```

- [ ] **Step 2: Run aggregator smoke test to verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_aggregator_smoke.py
```

Expected: FAIL because target `Aggregator.forward()` does not accept `cache_states`, `per_layer_budget`, or `frontend_cache_config`.

- [ ] **Step 3: Port production aggregator changes**

Apply production changes from:

```text
/path/to/mount/lyj/voxel-vggt/src/ovggt/models/aggregator.py
```

Required interface:

```python
def forward(
    self,
    images: torch.Tensor,
    past_key_values=None,
    cache_states=None,
    use_cache=False,
    past_frame_idx=0,
    per_layer_budget=0,
    anchor_token_count: int = None,
    importance_weight: float = 0.5,
    frontend_cache_config=None,
):
```

Keep target compatibility by accepting the old `total_budget` keyword as a deprecated alias at call sites in `OVGGT`, not inside the aggregator.

Required frontend return:

```python
if frontend_cache_mode:
    return output_list, self.patch_start_idx, cache_states, pending_updates, None
```

Required budget behavior:

```python
def _calculate_budgets(self, per_layer_budget, frontend_cache_config=None):
    if frontend_cache_config is not None and getattr(frontend_cache_config, "budget_allocation", "dynamic") == "uniform":
        return torch.full((self.depth,), int(per_layer_budget), dtype=torch.int64)
    ...
```

- [ ] **Step 4: Run aggregator smoke test**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_aggregator_smoke.py
```

Expected: PASS.

- [ ] **Step 5: Run existing frontend cache tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_cache.py tests/test_frontend_aggregator_smoke.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/models/aggregator.py tests/test_frontend_aggregator_smoke.py
git commit -m "feat: add aggregator frontend cache mode"
```

## Task 5: Update Camera Head for Frontend Pose API

**Files:**
- Modify: `src/ovggt/heads/camera_head.py`
- Test: `tests/test_frontend_camera_head_smoke.py`

- [ ] **Step 1: Add camera head API smoke test**

Create `tests/test_frontend_camera_head_smoke.py`:

```python
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.heads.camera_head import CameraHead
from ovggt.utils.frontend_keyframe import KeyframeEvent, KeyframeEventType
from ovggt.utils.pose_enc import ABS_POSE_ENCODING, REL_POSE_ENCODING


def test_camera_head_returns_abs_and_rel_pose_predictions():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    tokens = [torch.rand(1, 1, 2, 16)]
    predictions = head(
        tokens,
        num_iterations=1,
        use_cache=False,
        pose_encoding_type=REL_POSE_ENCODING,
        return_pose_predictions=True,
        return_last_pose_only=True,
    )
    assert set(predictions) == {"abs_pose_enc", "rel_pose_enc"}
    assert predictions["abs_pose_enc"].shape == (1, 1, 9)
    assert predictions["rel_pose_enc"].shape == (1, 1, 9)


def test_camera_head_apply_keyframe_event_noop_is_safe():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, total_budget=8)
    event = KeyframeEvent(
        event_type=KeyframeEventType.NOOP,
        frame_idx=1,
        keyframe_id=0,
        anchor_slot=-1,
    )
    past = [None]
    assert head.apply_keyframe_event(past, event, num_cam_iters=1) is past
```

- [ ] **Step 2: Run camera test to verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_camera_head_smoke.py
```

Expected: FAIL because target camera head lacks `pose_encoding_type`, relative branch, and/or `apply_keyframe_event()`.

- [ ] **Step 3: Port production camera head API**

Apply production changes from:

```text
/path/to/mount/lyj/voxel-vggt/src/ovggt/heads/camera_head.py
```

Required additions:

```python
from ovggt.utils.pose_enc import ABS_POSE_ENCODING, REL_POSE_ENCODING
self.pose_encoding_type = pose_encoding_type
self.rel_pose_branch = Mlp(...)
self.rel_pose_branch.load_state_dict(self.pose_branch.state_dict())
```

Required forward signature:

```python
def forward(
    self,
    aggregated_tokens_list: list,
    num_iterations: int = 4,
    past_key_values_camera=None,
    use_cache: bool = False,
    anchor_token_count: int = None,
    pose_encoding_type: str = None,
    return_pose_predictions: bool = False,
    return_last_pose_only: bool = False,
):
```

Required cache event method:

```python
def apply_keyframe_event(self, past_key_values_camera, event, num_cam_iters=4):
    if event is None or past_key_values_camera is None:
        return past_key_values_camera
    event_name = str(getattr(event, "event_type", ""))
    if event_name.endswith("NOOP"):
        return past_key_values_camera
    if getattr(event, "frame_idx", None) == 0 and getattr(event, "anchor_slot", -1) == 0:
        return past_key_values_camera
    num_anchor_frames = getattr(event, "num_anchor_frames", 0)
    if num_anchor_frames <= 0:
        return past_key_values_camera
    return self.sync_anchor_change(
        past_key_values_camera,
        anchor_token_count=num_anchor_frames * num_cam_iters,
        num_cam_iters=num_cam_iters,
        is_fifo=event_name.endswith("FIFO_SWAP"),
    )
```

- [ ] **Step 4: Run camera test**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_camera_head_smoke.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/heads/camera_head.py tests/test_frontend_camera_head_smoke.py
git commit -m "feat: add frontend camera pose API"
```

## Task 6: Add OVGGT Frontend Inference Path

**Files:**
- Modify: `src/ovggt/models/ovggt.py`
- Test: `tests/test_frontend_ovggt_api.py`

- [ ] **Step 1: Add OVGGT frontend constructor/API tests**

Create `tests/test_frontend_ovggt_api.py`:

```python
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig
from ovggt.utils.pose_enc import REL_POSE_ENCODING


def test_ovggt_accepts_frontend_constructor_options():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        per_layer_budget=16,
        total_budget=384,
        mode="frontend_eval",
        frontend_pose_encoding_type=REL_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
        aggregator_kwargs={
            "depth": 2,
            "num_heads": 4,
            "num_register_tokens": 1,
            "patch_embed": "conv",
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        enable_track_head=False,
    )
    assert model.mode == "frontend_eval"
    assert model.frontend_cache_config.enabled
    assert model.per_layer_budget == 16
    assert model.total_budget == 384


def test_frame_image_to_sequence_accepts_single_frame_and_batch():
    single = torch.rand(3, 28, 28)
    batch = torch.rand(2, 3, 28, 28)
    assert OVGGT._frame_image_to_sequence(single).shape == (1, 1, 3, 28, 28)
    assert OVGGT._frame_image_to_sequence(batch).shape == (2, 1, 3, 28, 28)
```

- [ ] **Step 2: Run OVGGT API tests to verify they fail**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_ovggt_api.py
```

Expected: FAIL because target `OVGGT.__init__` lacks frontend constructor options and helper methods.

- [ ] **Step 3: Port production OVGGT frontend path**

Apply production changes from:

```text
/path/to/mount/lyj/voxel-vggt/src/ovggt/models/ovggt.py
```

Required constructor compatibility:

```python
def __init__(
    self,
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    total_budget=200000,
    per_layer_budget=None,
    camera_budget=384,
    eviction_strategy="repr_shift_spatial",
    intra_frame_keep_ratio=1.0,
    spatial_alpha=0.5,
    importance_weight: float = 0.5,
    mode: str = "legacy",
    frontend_pose_encoding_type: str = ABS_POSE_ENCODING,
    frontend_cache_config: Optional[FrontendCacheConfig] = None,
    keyframe_switch_config: Optional[KeyframeSwitchConfig] = None,
    aggregator_kwargs: Optional[dict] = None,
    camera_head_kwargs: Optional[dict] = None,
    depth_head_kwargs: Optional[dict] = None,
    point_head_kwargs: Optional[dict] = None,
    enable_track_head: bool = True,
    camera_num_iters: int = 4,
    anchor_overflow_policy: str = "recent",
    frontend_head_checkpointing: bool = False,
):
```

Set:

```python
self.per_layer_budget = int(per_layer_budget if per_layer_budget is not None else max(int(total_budget) // self.aggregator.depth, 0))
self.total_budget = int(total_budget)
```

Required helper methods:

```python
@staticmethod
def _frame_image_to_sequence(frame_img: torch.Tensor) -> torch.Tensor:
    if frame_img.dim() == 3:
        return frame_img.unsqueeze(0).unsqueeze(1)
    if frame_img.dim() == 4:
        return frame_img.unsqueeze(1)
    raise ValueError(...)
```

Port `_inference_frontend()`, `_inference_legacy()`, `_infer_image_hw()`, `_frame_batch_size()`, `_validate_frontend_batch_size()`, `_build_frontend_keyframe_config()`, `_resolve_history_anchor_strategy()`, `_resolve_anchor_interval()`, `_maybe_move_dict_to_cpu()`, `_camera_pose_encoding_type_for_frontend()`, and `_set_anchor_overflow_policy()` from production.

Keep public `inference()` behavior:

```python
if self.mode in {"frontend_train", "frontend_eval"} and self.frontend_cache_config.enabled:
    return self._inference_frontend(...)
return self._inference_legacy(...)
```

- [ ] **Step 4: Run OVGGT API tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q tests/test_frontend_ovggt_api.py
```

Expected: PASS.

- [ ] **Step 5: Run combined unit tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q \
  tests/test_keyframe_manager.py \
  tests/test_frontend_cache.py \
  tests/test_p4_voxel_hash_collision_free.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p5_fifo_swap_transform_retention.py \
  tests/test_frontend_aggregator_smoke.py \
  tests/test_frontend_camera_head_smoke.py \
  tests/test_frontend_ovggt_api.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/models/ovggt.py tests/test_frontend_ovggt_api.py
git commit -m "feat: add ovggt frontend cache inference path"
```

## Task 7: Import and Compile Verification

**Files:**
- Test: core Python modules

- [ ] **Step 1: Compile migrated modules**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m py_compile \
  src/ovggt/models/ovggt.py \
  src/ovggt/models/aggregator.py \
  src/ovggt/utils/frontend_cache.py \
  src/ovggt/utils/frontend_keyframe.py \
  src/ovggt/layers/block.py \
  src/ovggt/layers/attention.py \
  src/ovggt/heads/camera_head.py
```

Expected: command exits 0.

- [ ] **Step 2: Run import smoke**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python - <<'PY'
from ovggt.models.ovggt import OVGGT
from ovggt.models.aggregator import Aggregator
from ovggt.utils.frontend_cache import FrontendCacheConfig, LayerCacheState
from ovggt.utils.frontend_keyframe import FrontendKeyframeManager, KeyframeSwitchConfig

model = OVGGT(
    img_size=28,
    patch_size=14,
    embed_dim=32,
    total_budget=384,
    per_layer_budget=16,
    mode="frontend_eval",
    frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
    aggregator_kwargs={"depth": 2, "num_heads": 4, "num_register_tokens": 1, "patch_embed": "conv"},
    camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
    enable_track_head=False,
)
assert model.mode == "frontend_eval"
assert model.frontend_cache_config.enabled
assert Aggregator is not None
assert LayerCacheState is not None
assert FrontendKeyframeManager is not None
print("frontend cache import smoke OK")
PY
```

Expected: prints `frontend cache import smoke OK`.

- [ ] **Step 3: Run full migrated test subset**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
PYTHONPATH=src python -m pytest -q \
  tests/test_keyframe_manager.py \
  tests/test_frontend_cache.py \
  tests/test_p4_voxel_hash_collision_free.py \
  tests/test_p1_protect_topk_budget_ceiling.py \
  tests/test_p5_fifo_swap_transform_retention.py \
  tests/test_frontend_aggregator_smoke.py \
  tests/test_frontend_camera_head_smoke.py \
  tests/test_frontend_ovggt_api.py
```

Expected: PASS.

- [ ] **Step 4: Commit verification adjustments if any**

If verification required small compatibility fixes, commit them:

```bash
git add src/ovggt tests
git commit -m "fix: stabilize frontend cache migration"
```

If no files changed, skip this commit.

## Self-Review

- Spec coverage: the plan covers keyframe manager, voxel/cache metadata, deferred aggregator commits, block/attention support, camera cache event sync, OVGGT frontend inference, and verification.
- Marker scan: no unfinished markers or unspecified implementation tasks remain.
- Type consistency: the plan consistently uses `FrontendCacheConfig`, `LayerCacheState`, `PendingLayerUpdate`, `FrontendKeyframeManager`, `KeyframeSwitchConfig`, and `per_layer_budget`.
