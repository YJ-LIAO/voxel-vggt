# Persistent Commit Eviction Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a train-free frontend cache improvement that keeps persistent cache eviction at commit time and optionally fuses current-frame patch evidence into cross-frame voxel representatives.

**Architecture:** The implementation keeps attention-time eviction as temporary compute control and removes it from persistent cache maintenance. Persistent `LayerCacheState` updates follow `append -> reorder -> voxel dedup/fusion -> commit-time eviction`. Cross-frame fusion is a config-gated extension inside `src/ovggt/utils/frontend_cache.py`, with CLI plumbing in the 7-Scenes eval launcher.

**Tech Stack:** Python, PyTorch, pytest, accelerate, OVGGT frontend cache, 7-Scenes evaluation.

---

## File Structure

- Modify `src/ovggt/utils/frontend_cache.py`
  - Stop applying `PendingLayerUpdate.attention_kept_indices` to persistent cache.
  - Add `FrontendCacheConfig.cross_frame_dedup_mode`.
  - Add conservative fusion controls.
  - Add helper methods for current-to-representative cross-frame voxel fusion.
  - Add active-to-slot coordinate conversion that fails closed when transforms are missing.
- Modify `src/eval/mv_recon/launch.py`
  - Pass cross-frame fusion config into `FrontendCacheConfig`.
  - Add CLI args for fusion mode and alpha.
- Modify `tests/test_frontend_cache.py`
  - Add regression tests for persistent commit-time eviction.
  - Add fusion unit tests.
- Modify `tests/test_mv_recon_launch.py`
  - Add parser and config plumbing tests for the new CLI args.
- Use `src/eval/mv_recon/run_frontend.sh`
  - No required script edit because it forwards `"$@"` to `launch.py`.

## Base Metrics to Preserve

Record these comparison points before running new ablations:

- Legacy 7-Scenes 200f:
  `/Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration/eval_results/mv_recon/legacy_200f_8gpu/7scenes/logs_all_recovered.txt`
- Current base frontend:
  `/Train/LYJ/workspace/OVGGT/.worktrees/frontend-attention-parity-quality/eval_results/mv_recon/frontend_attention_parity_dedup_soft_200f_8gpu/7scenes/logs_all.txt`
- Current best verified frontend:
  `/path/to/mount/lyj/ovggt_eval_results/frontend_windowfix_interval9_trigger100_200f_8gpu/7scenes/logs_all.txt`

Use `/path/to/mount/lyj/ovggt_eval_results` for all new evaluation outputs.

---

### Task 1: Materialize the Implementation Worktree

**Files:**
- Source worktree: `/Train/LYJ/workspace/OVGGT/.worktrees/frontend-window-protect-fix`
- Create worktree: `/Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion`

- [ ] **Step 1: Verify the source worktree and spec commit**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-window-protect-fix
git branch --show-current
git log --oneline -3
git status --short
```

Expected:

```text
frontend-window-protect-fix
d52b8d6 docs: add persistent commit eviction fusion design
```

The status output may contain existing tracked modifications. These are the current evaluated frontend base changes and must be materialized into the isolated implementation worktree, not edited in place.

- [ ] **Step 2: Create a binary patch for the current evaluated base**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-window-protect-fix
git diff HEAD --binary > /tmp/frontend-window-protect-fix-current-base.patch
test -s /tmp/frontend-window-protect-fix-current-base.patch
```

Expected: exit code `0`.

- [ ] **Step 3: Create the isolated implementation worktree**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT
git worktree add -b frontend-persistent-commit-evict-fusion \
  /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion \
  frontend-window-protect-fix
```

Expected: worktree is created on branch `frontend-persistent-commit-evict-fusion`.

- [ ] **Step 4: Apply and commit the current evaluated base patch**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
git apply --index /tmp/frontend-window-protect-fix-current-base.patch
git status --short
git commit -m "chore: materialize frontend window protect base"
```

Expected: commit succeeds. This commit makes the current evaluated base reproducible in the new worktree without changing the source worktree.

- [ ] **Step 5: Run the current regression baseline**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py \
  tests/test_frontend_aggregator_smoke.py \
  tests/test_frontend_ovggt_api.py \
  tests/test_mv_recon_launch.py \
  tests/test_keyframe_manager.py -q
```

Expected: `89 passed` or the same pass count as the source worktree baseline. If it fails, stop and inspect the materialized base before implementing new behavior.

---

### Task 2: Move Persistent Cache Back to Commit-Time Eviction

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py`
- Test: `tests/test_frontend_cache.py`

- [ ] **Step 1: Add the failing regression test**

Append this test method inside `FrontendCacheTests` in `tests/test_frontend_cache.py`:

```python
    def test_commit_ignores_attention_keep_indices_for_persistent_cache(self):
        class RecordingAttention:
            def __init__(self):
                self.seen_tokens = None
                self.seen_num_new_tokens = None
                self.seen_importance_scores = None

            def eviction(
                self,
                k,
                v,
                cache_budget,
                num_anchor_tokens,
                importance_scores=None,
                num_new_tokens=0,
                importance_weight=0.5,
                window_token_count=0,
            ):
                self.seen_tokens = int(k.shape[2])
                self.seen_num_new_tokens = int(num_new_tokens)
                self.seen_importance_scores = importance_scores.clone()
                kept = torch.tensor([[0, 4, 5]], dtype=torch.long, device=k.device)
                expanded = kept.unsqueeze(1).unsqueeze(-1).expand(k.shape[0], k.shape[1], kept.shape[1], k.shape[-1])
                return torch.gather(k, 2, expanded), torch.gather(v, 2, expanded), 0.5, kept

        k = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        v = k + 100.0
        metadata = make_metadata(
            anchor_slots=[-1, -1, -1, -1],
            frame_ids=[0, 0, 0, 0],
            slot_ids=[0, 0, 0, 0],
            keyframe_ids=[0, 0, 0, 0],
            importance=[0.1, 0.2, 0.3, 0.4],
            depth_conf=[0.1, 0.2, 0.3, 0.4],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata, slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)})
        current_metadata = make_metadata(
            anchor_slots=[-1, -1],
            frame_ids=[1, 1],
            slot_ids=[1, 1],
            keyframe_ids=[1, 1],
            importance=[0.9, 1.0],
            depth_conf=[0.9, 1.0],
            local_xyz=[
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
        )
        pending = PendingLayerUpdate(
            k_current=torch.tensor([[[[4.0], [5.0]]]], dtype=torch.float32),
            v_current=torch.tensor([[[[104.0], [105.0]]]], dtype=torch.float32),
            importance_current=torch.tensor([[0.9, 1.0]], dtype=torch.float32),
            frame_id=1,
            cache_budget=3,
            attention_kept_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        )
        attn = RecordingAttention()

        state.commit_pending_update_(
            pending_update=pending,
            current_metadata=current_metadata,
            config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
            intra_frame_keep_ratio=1.0,
            attn_module=attn,
        )

        self.assertEqual(attn.seen_tokens, 6)
        self.assertEqual(attn.seen_num_new_tokens, 2)
        self.assertTrue(torch.equal(attn.seen_importance_scores, torch.tensor([[0.9, 1.0]])))
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([0.0, 4.0, 5.0])))
        self.assertTrue(torch.equal(state.metadata.frame_id[0], torch.tensor([0, 1, 1])))
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py::FrontendCacheTests::test_commit_ignores_attention_keep_indices_for_persistent_cache -q
```

Expected: FAIL because `attn.seen_tokens` is `None` or not `6`; the current code applies `attention_kept_indices` and can skip commit-time eviction.

- [ ] **Step 3: Remove persistent use of `attention_kept_indices`**

In `src/ovggt/utils/frontend_cache.py`, replace the block after `reordered_for_anchor` with this code:

```python
        self.apply_voxel_dedup_(
            config,
            current_frame_id=pending_update.frame_id,
            layer_id=layer_id,
            dedup_probe=dedup_probe,
            batch_index=batch_index,
            dedup_replay_probe=dedup_replay_probe,
            cache_budget=pending_update.cache_budget,
        )
```

The removed block is:

```python
        if (
            pending_update.attention_kept_indices is not None
            and not reordered_for_anchor
            and intra_frame_keep_ratio >= 1.0
        ):
            self.gather_per_batch_(
                self._override_indices_per_batch(pending_update.attention_kept_indices)
            )
```

Do not remove the `PendingLayerUpdate.attention_kept_indices` field yet; `Block` and `Aggregator` may still pass it for diagnostics.

- [ ] **Step 4: Run the regression test and verify it passes**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py::FrontendCacheTests::test_commit_ignores_attention_keep_indices_for_persistent_cache -q
```

Expected: PASS.

- [ ] **Step 5: Run frontend cache tests**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest tests/test_frontend_cache.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit persistent eviction change**

Run:

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_frontend_cache.py
git commit -m "fix: keep frontend cache eviction at commit time"
```

Expected: commit succeeds.

---

### Task 3: Add Cross-Frame Fusion Config and CLI Plumbing

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py`
- Modify: `src/eval/mv_recon/launch.py`
- Test: `tests/test_mv_recon_launch.py`

- [ ] **Step 1: Add config fields**

In `FrontendCacheConfig` in `src/ovggt/utils/frontend_cache.py`, add these fields after `intra_dedup_mode`:

```python
    cross_frame_dedup_mode: Literal["drop", "merge"] = "drop"
    cross_frame_fusion_max_alpha: float = 0.10
    cross_frame_fusion_min_score: float = 0.25
```

In `__post_init__`, add:

```python
        if self.cross_frame_fusion_max_alpha < 0.0 or self.cross_frame_fusion_max_alpha > 1.0:
            raise ValueError("cross_frame_fusion_max_alpha must be in [0, 1]")
        if self.cross_frame_fusion_min_score < 0.0:
            raise ValueError("cross_frame_fusion_min_score must be >= 0")
```

- [ ] **Step 2: Add eval config test**

Add this test to `tests/test_mv_recon_launch.py`:

```python
def test_build_ovggt_kwargs_frontend_allows_cross_frame_fusion_overrides():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=8,
        frontend_keyframe_strategy="fixed_interval",
        frontend_dedup_policy="soft_reservoir",
        frontend_voxel_size=0.05,
        frontend_dedup_budget_trigger_ratio=1.0,
        frontend_dedup_topk_per_voxel=3,
        frontend_dedup_replacement_margin=0.10,
        frontend_dedup_age_decay=0.02,
        frontend_cross_frame_dedup_mode="merge",
        frontend_cross_frame_fusion_max_alpha=0.20,
        frontend_cross_frame_fusion_min_score=0.30,
    )

    kwargs = build_ovggt_kwargs_for_eval(args)

    config = kwargs["frontend_cache_config"]
    assert config.cross_frame_dedup_mode == "merge"
    assert config.cross_frame_fusion_max_alpha == 0.20
    assert config.cross_frame_fusion_min_score == 0.30
```

- [ ] **Step 3: Add parser test**

Add this test to `tests/test_mv_recon_launch.py`:

```python
def test_parser_exposes_cross_frame_fusion_cli_args():
    parser = mv_launch.get_args_parser()

    args = parser.parse_args([
        "--frontend_cross_frame_dedup_mode", "merge",
        "--frontend_cross_frame_fusion_max_alpha", "0.20",
        "--frontend_cross_frame_fusion_min_score", "0.30",
    ])

    assert args.frontend_cross_frame_dedup_mode == "merge"
    assert args.frontend_cross_frame_fusion_max_alpha == 0.20
    assert args.frontend_cross_frame_fusion_min_score == 0.30
```

- [ ] **Step 4: Run the new CLI tests and verify they fail**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_mv_recon_launch.py::test_build_ovggt_kwargs_frontend_allows_cross_frame_fusion_overrides \
  tests/test_mv_recon_launch.py::test_parser_exposes_cross_frame_fusion_cli_args -q
```

Expected: FAIL because the new args and config fields are not wired yet.

- [ ] **Step 5: Wire config in eval launcher**

In `src/eval/mv_recon/launch.py`, add these arguments to `FrontendCacheConfig(...)`:

```python
            cross_frame_dedup_mode=getattr(args, "frontend_cross_frame_dedup_mode", "drop"),
            cross_frame_fusion_max_alpha=getattr(args, "frontend_cross_frame_fusion_max_alpha", 0.10),
            cross_frame_fusion_min_score=getattr(args, "frontend_cross_frame_fusion_min_score", 0.25),
```

Add these parser args after `--frontend_dedup_age_decay`:

```python
    parser.add_argument(
        "--frontend_cross_frame_dedup_mode",
        type=str,
        default="drop",
        choices=("drop", "merge"),
    )
    parser.add_argument("--frontend_cross_frame_fusion_max_alpha", type=float, default=0.10)
    parser.add_argument("--frontend_cross_frame_fusion_min_score", type=float, default=0.25)
```

- [ ] **Step 6: Run CLI tests and verify they pass**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_mv_recon_launch.py::test_build_ovggt_kwargs_frontend_allows_cross_frame_fusion_overrides \
  tests/test_mv_recon_launch.py::test_parser_exposes_cross_frame_fusion_cli_args -q
```

Expected: PASS.

- [ ] **Step 7: Commit config and CLI plumbing**

Run:

```bash
git add src/ovggt/utils/frontend_cache.py src/eval/mv_recon/launch.py tests/test_mv_recon_launch.py
git commit -m "feat: add frontend cross-frame fusion config"
```

Expected: commit succeeds.

---

### Task 4: Implement Cross-Frame Voxel Representative Fusion

**Files:**
- Modify: `src/ovggt/utils/frontend_cache.py`
- Test: `tests/test_frontend_cache.py`

- [ ] **Step 1: Add fusion tests**

Add these test methods inside `FrontendCacheTests` in `tests/test_frontend_cache.py`:

```python
    def test_cross_frame_merge_fuses_current_patch_into_protected_representative(self):
        k = torch.tensor([[[[1.0], [11.0]]]], dtype=torch.float32)
        v = torch.tensor([[[[101.0], [111.0]]]], dtype=torch.float32)
        metadata = make_metadata(
            anchor_slots=[0, -1],
            frame_ids=[0, 1],
            slot_ids=[0, 1],
            keyframe_ids=[0, 1],
            importance=[0.6, 1.0],
            depth_conf=[0.6, 1.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(
            k=k.clone(),
            v=v.clone(),
            metadata=metadata,
            slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)},
        )

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                intra_frame_dedup_enabled=False,
                cross_frame_dedup_mode="merge",
                cross_frame_fusion_max_alpha=0.25,
                cross_frame_fusion_min_score=0.0,
            ),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 1)
        self.assertTrue(torch.equal(state.metadata.anchor_slot[0], torch.tensor([0])))
        self.assertTrue(torch.allclose(state.k[0, 0, :, 0], torch.tensor([3.5])))
        self.assertTrue(torch.allclose(state.v[0, 0, :, 0], torch.tensor([103.5])))
        self.assertTrue(state.metadata.importance[0, 0].item() > 0.6)

    def test_cross_frame_merge_does_not_change_camera_or_register_tokens(self):
        k = torch.tensor([[[[1.0], [2.0], [12.0]]]], dtype=torch.float32)
        v = k + 100.0
        metadata = make_metadata(
            anchor_slots=[0, -1, -1],
            frame_ids=[0, 1, 1],
            slot_ids=[0, 1, 1],
            keyframe_ids=[0, 1, 1],
            token_kind=[int(TokenKind.PATCH), int(TokenKind.CAMERA), int(TokenKind.REGISTER)],
            importance=[0.8, 1.0, 1.0],
            depth_conf=[0.8, 1.0, 1.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata, slot_to_active={0: make_transform(0.0), 1: make_transform(0.0)})

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                cross_frame_dedup_mode="merge",
                cross_frame_fusion_min_score=0.0,
            ),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 3)
        self.assertTrue(torch.equal(state.metadata.token_kind[0], torch.tensor([int(TokenKind.PATCH), int(TokenKind.CAMERA), int(TokenKind.REGISTER)])))
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([1.0, 2.0, 12.0])))

    def test_cross_frame_merge_skips_when_representative_transform_is_missing(self):
        k = torch.tensor([[[[1.0], [11.0]]]], dtype=torch.float32)
        v = k + 100.0
        metadata = make_metadata(
            anchor_slots=[0, -1],
            frame_ids=[0, 1],
            slot_ids=[99, 1],
            keyframe_ids=[99, 1],
            importance=[0.6, 1.0],
            depth_conf=[0.6, 1.0],
            local_xyz=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        state = LayerCacheState(k=k.clone(), v=v.clone(), metadata=metadata, slot_to_active={1: make_transform(0.0)})

        state.apply_voxel_dedup_(
            FrontendCacheConfig(
                enabled=True,
                voxel_size=0.5,
                intra_frame_dedup_enabled=False,
                cross_frame_dedup_mode="merge",
                cross_frame_fusion_max_alpha=0.25,
                cross_frame_fusion_min_score=0.0,
            ),
            current_frame_id=1,
        )

        self.assertEqual(state.num_tokens(), 2)
        self.assertTrue(torch.equal(state.k[0, 0, :, 0], torch.tensor([1.0, 11.0])))
```

- [ ] **Step 2: Run fusion tests and verify they fail**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_fuses_current_patch_into_protected_representative \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_does_not_change_camera_or_register_tokens \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_skips_when_representative_transform_is_missing -q
```

Expected: FAIL because cross-frame merge is not implemented yet.

- [ ] **Step 3: Add transform helper methods**

Add these methods to `LayerCacheState` after `_get_slot_transform` in `src/ovggt/utils/frontend_cache.py`:

```python
    def _has_slot_transform(self, slot_id: int) -> bool:
        return self.slot_to_active is not None and int(slot_id) in self.slot_to_active

    def _active_points_to_slot_local(self, points_active: Tensor, slot_id: int) -> Optional[Tensor]:
        if points_active.numel() == 0:
            return points_active
        if not self._has_slot_transform(slot_id):
            return None
        slot_to_active = self.slot_to_active[int(slot_id)].to(
            device=points_active.device,
            dtype=points_active.dtype,
        )
        active_to_slot = closed_form_inverse_se3(slot_to_active.unsqueeze(0))[0]
        return transform_points(points_active.unsqueeze(0), active_to_slot.unsqueeze(0))[0]
```

- [ ] **Step 4: Add cross-frame merge plan builder**

Add this method to `LayerCacheState` before `_dedup_single_batch`:

```python
    def _build_cross_frame_merge_plan_single_batch(
        self,
        b_idx: int,
        protected_patch_mask: Tensor,
        current_patch_mask: Tensor,
        scores: Tensor,
        config: FrontendCacheConfig,
        projected_xyz: Tensor,
    ) -> Optional[dict]:
        if config.cross_frame_dedup_mode != "merge":
            return None
        metadata = self.metadata
        device = metadata.anchor_slot.device
        current_indices = torch.nonzero(current_patch_mask, as_tuple=False).squeeze(-1)
        if current_indices.numel() == 0:
            return None

        existing_patch_mask = (
            (metadata.token_kind[b_idx] == int(TokenKind.PATCH))
            & torch.isfinite(projected_xyz).all(dim=-1)
            & (metadata.frame_id[b_idx] != metadata.frame_id[b_idx, current_indices[0]])
        )
        existing_indices = torch.nonzero(existing_patch_mask, as_tuple=False).squeeze(-1)
        if existing_indices.numel() == 0:
            return None

        existing_voxels = torch.floor(projected_xyz[existing_indices] / config.voxel_size).to(torch.long)
        current_voxels = torch.floor(projected_xyz[current_indices] / config.voxel_size).to(torch.long)
        existing_hash = voxel_hash_collision_free(existing_voxels)
        current_hash = voxel_hash_collision_free(current_voxels)

        unique_hash, inv = torch.unique(existing_hash, return_inverse=True)
        best_rep = torch.full((unique_hash.numel(),), -1, dtype=torch.long, device=device)
        best_score = torch.full((unique_hash.numel(),), float("-inf"), dtype=scores.dtype, device=device)
        existing_scores = scores[existing_indices]
        existing_is_protected = protected_patch_mask[existing_indices]
        rep_rank = existing_scores + existing_is_protected.to(existing_scores.dtype) * 2.0
        best_score.scatter_reduce_(0, inv, rep_rank, reduce="amax", include_self=True)
        for local_idx in range(existing_indices.numel()):
            group = int(inv[local_idx].item())
            if rep_rank[local_idx] == best_score[group] and int(best_rep[group].item()) < 0:
                best_rep[group] = existing_indices[local_idx]

        sort_idx = torch.searchsorted(unique_hash, current_hash).clamp(0, max(unique_hash.numel() - 1, 0))
        matched = unique_hash[sort_idx] == current_hash
        if not matched.any():
            return None

        matched_current = current_indices[matched]
        matched_rep = best_rep[sort_idx[matched]]
        valid_rep = matched_rep >= 0
        if not valid_rep.any():
            return None

        matched_current = matched_current[valid_rep]
        matched_rep = matched_rep[valid_rep]
        current_scores = scores[matched_current]
        score_mask = current_scores >= float(config.cross_frame_fusion_min_score)
        if not score_mask.any():
            return None

        matched_current = matched_current[score_mask]
        matched_rep = matched_rep[score_mask]
        current_scores = current_scores[score_mask]

        rep_slot_ids = metadata.slot_id[b_idx, matched_rep]
        keep_current = []
        keep_rep = []
        keep_alpha = []
        keep_xyz_local = []
        for idx in range(matched_current.numel()):
            rep_idx = int(matched_rep[idx].item())
            cur_idx = int(matched_current[idx].item())
            rep_slot_id = int(rep_slot_ids[idx].item())
            if not self._has_slot_transform(rep_slot_id):
                continue
            alpha = min(float(config.cross_frame_fusion_max_alpha), max(0.0, float(current_scores[idx].item()) * float(config.cross_frame_fusion_max_alpha)))
            if alpha <= 0.0:
                continue
            blended_active = (1.0 - alpha) * projected_xyz[rep_idx] + alpha * projected_xyz[cur_idx]
            blended_local = self._active_points_to_slot_local(blended_active.unsqueeze(0), rep_slot_id)
            if blended_local is None:
                continue
            keep_current.append(cur_idx)
            keep_rep.append(rep_idx)
            keep_alpha.append(alpha)
            keep_xyz_local.append(blended_local[0])

        if not keep_current:
            return None

        return {
            "rep_indices": torch.tensor(keep_rep, dtype=torch.long, device=device),
            "current_indices": torch.tensor(keep_current, dtype=torch.long, device=device),
            "alpha": torch.tensor(keep_alpha, dtype=self.k.dtype, device=device),
            "rep_local_xyz": torch.stack(keep_xyz_local, dim=0),
        }
```

- [ ] **Step 5: Add cross-frame merge applier**

Add this method near `_apply_intra_merge_`:

```python
    def _apply_cross_frame_merge_(self, b_idx: int, plan: dict) -> None:
        rep = plan["rep_indices"]
        cur = plan["current_indices"]
        alpha = plan["alpha"].to(device=self.k.device, dtype=self.k.dtype)
        rep_local_xyz = plan["rep_local_xyz"].to(device=self.metadata.slot_local_xyz.device, dtype=self.metadata.slot_local_xyz.dtype)

        for kv in (self.k, self.v):
            old = kv[b_idx, :, rep, :]
            new = kv[b_idx, :, cur, :]
            kv[b_idx, :, rep, :] = old * (1.0 - alpha.view(1, -1, 1)) + new * alpha.view(1, -1, 1)

        for field in ("importance", "depth_conf"):
            values = getattr(self.metadata, field)
            old = values[b_idx, rep]
            new = values[b_idx, cur]
            values[b_idx, rep] = old * (1.0 - alpha.to(values.dtype)) + new * alpha.to(values.dtype)

        self.metadata.slot_local_xyz[b_idx, rep] = rep_local_xyz
```

- [ ] **Step 6: Integrate fusion into `_dedup_single_batch`**

Keep `soft_reservoir` behavior unchanged in this first implementation. Add the
cross-frame merge block immediately after the existing `soft_reservoir` early
return block:

```python
        if config.dedup_policy == "soft_reservoir":
            return self._soft_reservoir_dedup_single_batch(
                b_idx=b_idx,
                current_frame_id=current_frame_id,
                protected_patch_mask=protected_patch_mask,
                current_patch_mask=current_patch_mask,
                scores=scores,
                config=config,
                total_tokens=total_tokens,
                projected_xyz=projected_xyz,
            )
```

The new block is:

```python
        cross_frame_merge_plan = self._build_cross_frame_merge_plan_single_batch(
            b_idx=b_idx,
            protected_patch_mask=protected_patch_mask,
            current_patch_mask=current_patch_mask,
            scores=scores,
            config=config,
            projected_xyz=projected_xyz,
        )
        if cross_frame_merge_plan is not None:
            keep_mask[cross_frame_merge_plan["current_indices"]] = False
```

At the end of `_dedup_single_batch`, return both merge plans:

```python
        if merge_plan is None:
            combined_plan = {"cross_frame": cross_frame_merge_plan, "intra": None}
        else:
            combined_plan = {"cross_frame": cross_frame_merge_plan, "intra": merge_plan}
        return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1), combined_plan
```

This makes cross-frame merge active for the `hard` policy only. The full
7-Scenes fusion ablation uses `--frontend_dedup_policy hard` in Task 6.

- [ ] **Step 7: Apply combined merge plans in `apply_voxel_dedup_`**

Replace the single-batch merge application with:

```python
            if merge_plan_b0 is not None:
                cross_plan = merge_plan_b0.get("cross_frame") if isinstance(merge_plan_b0, dict) else None
                intra_plan = merge_plan_b0.get("intra") if isinstance(merge_plan_b0, dict) else merge_plan_b0
                if cross_plan is not None:
                    self._apply_cross_frame_merge_(0, cross_plan)
                if intra_plan is not None:
                    self._apply_intra_merge_(0, intra_plan)
```

Replace the multi-batch merge application with:

```python
            if merge_plan_b is not None:
                cross_plan = merge_plan_b.get("cross_frame") if isinstance(merge_plan_b, dict) else None
                intra_plan = merge_plan_b.get("intra") if isinstance(merge_plan_b, dict) else merge_plan_b
                if cross_plan is not None:
                    self._apply_cross_frame_merge_(b_idx, cross_plan)
                if intra_plan is not None:
                    self._apply_intra_merge_(b_idx, intra_plan)
```

- [ ] **Step 8: Run fusion tests and fix exact failures**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_fuses_current_patch_into_protected_representative \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_does_not_change_camera_or_register_tokens \
  tests/test_frontend_cache.py::FrontendCacheTests::test_cross_frame_merge_skips_when_representative_transform_is_missing -q
```

Expected: PASS. If a failure shows a shape or dtype mismatch, fix only the helper code touched in this task and rerun this exact command.

- [ ] **Step 9: Run frontend cache tests**

Run:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest tests/test_frontend_cache.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit fusion implementation**

Run:

```bash
git add src/ovggt/utils/frontend_cache.py tests/test_frontend_cache.py
git commit -m "feat: add cross-frame voxel representative fusion"
```

Expected: commit succeeds.

---

### Task 5: Run Full Unit Regression

**Files:**
- No code files changed in this task.

- [ ] **Step 1: Run frontend/cache regression set**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py \
  tests/test_frontend_aggregator_smoke.py \
  tests/test_frontend_ovggt_api.py \
  tests/test_mv_recon_launch.py \
  tests/test_keyframe_manager.py -q
```

Expected: PASS with no failures.

- [ ] **Step 2: Commit only if test snapshots or docs changed**

Run:

```bash
git status --short
```

Expected: no modified tracked files. If pytest generated tracked changes, inspect them and commit only if they are intentional.

---

### Task 6: Run 7-Scenes Smoke and Full Ablations

**Files:**
- No code files changed in this task.

- [ ] **Step 1: Record current comparison metrics**

Run:

```bash
python - <<'PY'
import pathlib, re, statistics
pat = re.compile(r'Idx: ([^,]+), Acc: ([0-9.eE+-]+), Comp: ([0-9.eE+-]+), NC1: ([0-9.eE+-]+), NC2: ([0-9.eE+-]+)')
paths = {
    "legacy": "/Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration/eval_results/mv_recon/legacy_200f_8gpu/7scenes/logs_all_recovered.txt",
    "base": "/Train/LYJ/workspace/OVGGT/.worktrees/frontend-attention-parity-quality/eval_results/mv_recon/frontend_attention_parity_dedup_soft_200f_8gpu/7scenes/logs_all.txt",
    "best": "/path/to/mount/lyj/ovggt_eval_results/frontend_windowfix_interval9_trigger100_200f_8gpu/7scenes/logs_all.txt",
}
for name, path in paths.items():
    rows = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        match = pat.search(line)
        if match:
            acc, comp, nc1, nc2 = map(float, match.groups()[1:])
            rows.append((acc, comp, (nc1 + nc2) / 2))
    print(name, len(rows), statistics.mean(r[0] for r in rows), statistics.mean(r[1] for r in rows), statistics.mean(r[2] for r in rows))
PY
```

Expected: prints 18 scenes for each existing comparison path.

- [ ] **Step 2: Run short persistent-commit-only smoke**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
MAX_FRAMES=40 NUM_PROCESSES=8 FRONTEND_ANCHOR_INTERVAL=9 \
OUTPUT_DIR=/path/to/mount/lyj/ovggt_eval_results/frontend_commit_only_smoke_40f_8gpu \
MAIN_PROCESS_PORT=29631 \
./src/eval/mv_recon/run_frontend.sh \
  --frontend_dedup_policy soft_reservoir \
  --frontend_voxel_size 0.05 \
  --frontend_dedup_budget_trigger_ratio 1.00 \
  --frontend_dedup_topk_per_voxel 3 \
  --frontend_dedup_replacement_margin 0.10 \
  --frontend_dedup_age_decay 0.02
```

Expected: command exits `0` and writes `logs_all.txt` under the output directory.

- [ ] **Step 3: Run short fusion smoke**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
MAX_FRAMES=40 NUM_PROCESSES=8 FRONTEND_ANCHOR_INTERVAL=9 \
OUTPUT_DIR=/path/to/mount/lyj/ovggt_eval_results/frontend_commit_fusion_hard_smoke_40f_8gpu \
MAIN_PROCESS_PORT=29632 \
./src/eval/mv_recon/run_frontend.sh \
  --frontend_dedup_policy hard \
  --frontend_voxel_size 0.05 \
  --frontend_cross_frame_dedup_mode merge \
  --frontend_cross_frame_fusion_max_alpha 0.10 \
  --frontend_cross_frame_fusion_min_score 0.25
```

Expected: command exits `0` and writes `logs_all.txt`.

- [ ] **Step 4: Run full persistent-commit-only 200f evaluation**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
MAX_FRAMES=200 NUM_PROCESSES=8 FRONTEND_ANCHOR_INTERVAL=9 \
OUTPUT_DIR=/path/to/mount/lyj/ovggt_eval_results/frontend_commit_only_i9_t100_200f_8gpu \
MAIN_PROCESS_PORT=29633 \
./src/eval/mv_recon/run_frontend.sh \
  --frontend_dedup_policy soft_reservoir \
  --frontend_voxel_size 0.05 \
  --frontend_dedup_budget_trigger_ratio 1.00 \
  --frontend_dedup_topk_per_voxel 3 \
  --frontend_dedup_replacement_margin 0.10 \
  --frontend_dedup_age_decay 0.02
```

Expected: command exits `0` and writes `logs_all.txt`.

- [ ] **Step 5: Run full fusion 200f evaluation**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
MAX_FRAMES=200 NUM_PROCESSES=8 FRONTEND_ANCHOR_INTERVAL=9 \
OUTPUT_DIR=/path/to/mount/lyj/ovggt_eval_results/frontend_commit_fusion_hard_i9_200f_8gpu \
MAIN_PROCESS_PORT=29634 \
./src/eval/mv_recon/run_frontend.sh \
  --frontend_dedup_policy hard \
  --frontend_voxel_size 0.05 \
  --frontend_cross_frame_dedup_mode merge \
  --frontend_cross_frame_fusion_max_alpha 0.10 \
  --frontend_cross_frame_fusion_min_score 0.25
```

Expected: command exits `0` and writes `logs_all.txt`.

- [ ] **Step 6: Summarize ablation metrics**

Run:

```bash
python - <<'PY'
import pathlib, re, statistics
pat = re.compile(r'Idx: ([^,]+), Acc: ([0-9.eE+-]+), Comp: ([0-9.eE+-]+), NC1: ([0-9.eE+-]+), NC2: ([0-9.eE+-]+)')
paths = {
    "legacy": "/Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration/eval_results/mv_recon/legacy_200f_8gpu/7scenes/logs_all_recovered.txt",
    "best": "/path/to/mount/lyj/ovggt_eval_results/frontend_windowfix_interval9_trigger100_200f_8gpu/7scenes/logs_all.txt",
    "commit_only": "/path/to/mount/lyj/ovggt_eval_results/frontend_commit_only_i9_t100_200f_8gpu/7scenes/logs_all.txt",
    "fusion": "/path/to/mount/lyj/ovggt_eval_results/frontend_commit_fusion_hard_i9_200f_8gpu/7scenes/logs_all.txt",
}
for name, path in paths.items():
    rows = []
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        print(name, "missing")
        continue
    for line in path_obj.read_text(errors="replace").splitlines():
        match = pat.search(line)
        if match:
            acc, comp, nc1, nc2 = map(float, match.groups()[1:])
            rows.append((acc, comp, (nc1 + nc2) / 2))
    if rows:
        print(f"{name:12s} n={len(rows):2d} acc={statistics.mean(r[0] for r in rows):.12f} comp={statistics.mean(r[1] for r in rows):.12f} nc={statistics.mean(r[2] for r in rows):.12f}")
    else:
        print(name, "no metrics")
PY
```

Expected: `commit_only` and `fusion` each report `n=18`. If either has fewer scenes, inspect per-process logs before drawing conclusions.

---

### Task 7: Document Outcome and Recommendation

**Files:**
- Modify: `docs/superpowers/plans/2026-07-07-persistent-commit-evict-fusion.md`

- [ ] **Step 1: Add results section to this plan**

Run this script after Task 6 completes. It reads the evaluation logs, computes
mean metrics, and appends a concrete results section with no manual table
editing.

```bash
python - <<'PY'
import pathlib
import re
import statistics

plan_path = pathlib.Path("docs/superpowers/plans/2026-07-07-persistent-commit-evict-fusion.md")
pat = re.compile(r'Idx: ([^,]+), Acc: ([0-9.eE+-]+), Comp: ([0-9.eE+-]+), NC1: ([0-9.eE+-]+), NC2: ([0-9.eE+-]+)')
paths = {
    "legacy": "/Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration/eval_results/mv_recon/legacy_200f_8gpu/7scenes/logs_all_recovered.txt",
    "current best": "/path/to/mount/lyj/ovggt_eval_results/frontend_windowfix_interval9_trigger100_200f_8gpu/7scenes/logs_all.txt",
    "commit only": "/path/to/mount/lyj/ovggt_eval_results/frontend_commit_only_i9_t100_200f_8gpu/7scenes/logs_all.txt",
    "fusion": "/path/to/mount/lyj/ovggt_eval_results/frontend_commit_fusion_hard_i9_200f_8gpu/7scenes/logs_all.txt",
}

def load_metrics(path):
    rows = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        match = pat.search(line)
        if match:
            acc, comp, nc1, nc2 = map(float, match.groups()[1:])
            rows.append((acc, comp, (nc1 + nc2) / 2))
    if not rows:
        raise RuntimeError(f"No metric rows found in {path}")
    return {
        "n": len(rows),
        "acc": statistics.mean(row[0] for row in rows),
        "comp": statistics.mean(row[1] for row in rows),
        "nc": statistics.mean(row[2] for row in rows),
    }

metrics = {name: load_metrics(path) for name, path in paths.items()}
best = metrics["current best"]

def decision(name, row):
    if name in {"legacy", "current best"}:
        return "comparison"
    acc_ok = row["acc"] <= best["acc"]
    comp_ok = row["comp"] <= best["comp"]
    nc_ok = row["nc"] >= best["nc"]
    if acc_ok and comp_ok and nc_ok:
        return "candidate default"
    if (comp_ok and nc_ok) or (acc_ok and nc_ok) or (acc_ok and comp_ok):
        return "candidate ablation"
    return "reject as default"

lines = [
    "",
    "## Results",
    "",
    "| Variant | Log Path | Scenes | Acc | Comp | Mean NC | Decision |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
]
for name, path in paths.items():
    row = metrics[name]
    lines.append(
        f"| {name} | `{path}` | {row['n']} | {row['acc']:.12f} | {row['comp']:.12f} | {row['nc']:.12f} | {decision(name, row)} |"
    )

commit_decision = decision("commit only", metrics["commit only"])
fusion_decision = decision("fusion", metrics["fusion"])
lines.extend([
    "",
    "Recommendation:",
    "",
    f"- Persistent commit-time eviction: {commit_decision}.",
    f"- Cross-frame voxel fusion: {fusion_decision}.",
    "",
])

text = plan_path.read_text()
if "\n## Results\n" in text:
    text = text.split("\n## Results\n", 1)[0].rstrip() + "\n"
plan_path.write_text(text.rstrip() + "\n" + "\n".join(lines))
PY
```

Expected: the plan file ends with a populated `## Results` section containing
four rows and concrete numeric metrics.

- [ ] **Step 2: Commit results documentation**

Run:

```bash
git add docs/superpowers/plans/2026-07-07-persistent-commit-evict-fusion.md
git commit -m "docs: record persistent commit fusion evaluation"
```

Expected: commit succeeds.
