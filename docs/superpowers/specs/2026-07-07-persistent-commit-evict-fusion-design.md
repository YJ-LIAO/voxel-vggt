# Persistent Commit-Time Eviction and Cross-Frame Voxel Fusion Design

Date: 2026-07-07
Status: Approved for implementation planning

## Context

The current OVGGT frontend streaming path has passed the known cache migration
regression tests and can outperform the legacy 7-Scenes 200f baseline on the
best evaluated settings. Further interval, trigger, and top-k tuning has only
small gains. The next step should target the cache decision mechanism itself.

Two code review findings guide this design:

1. The frontend attention path creates `attention_kept_indices` while computing
   the current frame output. Those indices are scored before the current layer's
   post-MLP importance exists, and they may be based on the previous layer's
   `prev_importance`.
2. `commit_pending_update_()` currently may apply those attention-time indices
   to the persistent cache before geometry deduplication. That can permanently
   drop candidates using a temporary keep set, then skip the later commit-time
   eviction because the cache is already within budget.

The selected structural improvement is to separate temporary attention compute
from persistent cache maintenance, then add train-free cross-frame voxel
representative fusion at the geometry-aware commit stage.

## Goals

- Keep attention-time eviction as a temporary compute control for the current
  attention output.
- Stop using attention-time keep indices as the final persistent cache keep set.
- Make persistent cache maintenance follow one canonical order:

  ```text
  append current K/V -> anchor reorder -> voxel dedup/fusion -> commit-time eviction
  ```

- Add a train-free cross-frame voxel fusion mode that merges current-frame patch
  evidence into an existing representative token instead of only dropping the
  current token.
- Preserve the current better-than-legacy base as a reproducible baseline by
  implementing in an isolated worktree.
- Provide targeted unit tests, smoke tests, and full 7-Scenes 200f evaluation
  evidence before recommending the new mode as default.

## Non-Goals

- No keyframe second pass in this iteration.
- No learned token scorer or training-dependent module.
- No geometry memory bank in this iteration.
- No changes to camera/register token semantics.
- No broad refactor of the frontend state layout.
- No change to current-frame output timing. Fusion affects future cache quality,
  not the already-produced output for the same frame.

## Worktree

Implementation will use a new isolated worktree:

```text
/Train/LYJ/workspace/OVGGT/.worktrees/frontend-persistent-commit-evict-fusion
```

The worktree should be based on the current best frontend branch state, not on
the dirty main workspace. Evaluation outputs must be written under:

```text
/path/to/mount/lyj/ovggt_eval_results
```

because `/Train` has very little free space.

## Design Part 1: Persistent Commit-Time Eviction

### Current Problem

In frontend cache mode, `Attention.forward()` can evict a temporary `k_attn/v_attn`
for the current attention output and return `attention_kept_indices`. The current
commit path can apply those indices to `LayerCacheState` before geometry dedup.

That makes an attention-time compute decision become a persistent memory
decision. The scoring context is weaker than commit-time scoring because:

- current layer post-MLP importance is not available yet;
- geometry metadata is not available yet;
- voxel deduplication has not run yet;
- dedup may reduce token count after early gather, but dropped candidates cannot
  be recovered.

### New Behavior

`attention_kept_indices` remains valid only for temporary attention compute. It
must not be applied to persistent `LayerCacheState`.

`LayerCacheState.commit_pending_update_()` should append current K/V and
metadata, reorder anchors when required, run geometry dedup/fusion, then call
`attn_module.eviction()` if the persistent cache still exceeds the layer budget.

The `PendingLayerUpdate.attention_kept_indices` field can be left in place for
compatibility, but the commit path must ignore it for persistent cache updates.

### Expected Effect

This change may increase commit-time eviction work, but it lets the persistent
cache use the strongest available evidence: current layer importance plus
geometry metadata and dedup output.

## Design Part 2: Cross-Frame Voxel Representative Fusion

### Current Dedup Behavior

The existing voxel dedup path can discard current-frame patch tokens that fall
into a protected voxel with a stronger existing token. It can also deduplicate
within the current frame. An intra-frame merge helper already exists for
same-frame duplicate patches.

### New Fusion Mode

Add a config-controlled mode for cross-frame fusion. The recommended first mode
name is:

```python
cross_frame_dedup_mode: Literal["drop", "merge"] = "drop"
```

`drop` preserves existing behavior. `merge` enables current-to-representative
fusion.

Fusion applies only to patch tokens where:

- both current token and representative token have finite projected XYZ;
- both tokens fall into the same voxel in active coordinates;
- the current token is not protected;
- the representative token is preferably protected, otherwise a higher-scoring
  existing candidate in the same voxel;
- the current token passes a conservative score/confidence gate.

The first implementation should only merge current tokens into an existing
representative. It should not merge arbitrary old tokens with each other.

### Representative Selection

For each active-coordinate voxel:

1. Prefer a protected patch representative.
2. If no protected representative exists, prefer an existing non-current patch
   with the highest composite score.
3. Do not choose camera/register tokens as representatives.
4. Do not choose a current token as the representative for cross-frame fusion in
   the first implementation. Current-current duplicates remain handled by the
   existing intra-frame dedup path.

### Fusion Update

For each current token selected for fusion:

- update representative K and V using a bounded weighted blend;
- update representative `importance` and `depth_conf` using the same weight
  family;
- update representative geometry in active coordinates, then convert back into
  the representative token's local slot coordinate before storing
  `slot_local_xyz`;
- drop the fused current token from the keep set.

The default fusion should be conservative:

```text
rep_new = (1 - alpha) * rep_old + alpha * current
```

`alpha` should be bounded by config and derived from normalized current score,
representative score, and depth confidence. A fixed small maximum alpha is
acceptable for the first implementation.

### Coordinate Rule

Do not average `slot_local_xyz` directly when tokens have different `slot_id`
values. Cross-frame fusion must:

1. use already-computed projected active coordinates for representative and
   current tokens;
2. blend in active coordinates;
3. transform the blended active point back to the representative token's local
   coordinate frame;
4. write that local coordinate to the representative token metadata.

If the inverse transform for the representative slot is not available, skip
geometry fusion for that token rather than falling back to identity silently.

## Invariants

Implementation must preserve these invariants:

- K, V, and metadata token order remain aligned after every gather or merge.
- Protected tokens stay before candidate tokens after anchor reorder.
- `protected_count` is recomputed after every mutation that can change anchor
  slots or token count.
- Current-frame candidate tokens remain a contiguous tail before commit-time
  hybrid eviction, or the existing P6 invariant must fail loudly.
- Fusion must not drop protected representatives.
- Fusion must not change camera/register tokens.
- Per-sample cache state remains independent; do not reintroduce variable-length
  batch gather into a shared `LayerCacheState`.
- Dedup replay and oracle probe hooks should keep existing semantics unless the
  test explicitly opts into fusion.

## Testing

### Unit Tests

Add tests for:

- `attention_kept_indices` is ignored by persistent commit, so commit-time
  eviction can choose a different keep set using current metadata.
- Commit-time eviction still receives `window_token_count`.
- Cross-frame merge fuses a current patch into a protected representative and
  drops the current token.
- Fusion does not alter camera/register tokens.
- Fusion preserves protected count and K/V metadata alignment.
- Fusion skips local XYZ updates if the representative transform is missing.
- Current-tail P6 invariant still holds after fusion and dedup.
- Existing `drop` mode remains unchanged.

### Regression Tests

Run the current frontend/cache regression set:

```bash
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest \
  tests/test_frontend_cache.py \
  tests/test_frontend_aggregator_smoke.py \
  tests/test_frontend_ovggt_api.py \
  tests/test_mv_recon_launch.py \
  tests/test_keyframe_manager.py -q
```

### Evaluation

Run in this order:

1. Short 7-Scenes smoke with `MAX_FRAMES` reduced.
2. Full 7-Scenes 200f 8GPU persistent commit-time eviction only.
3. Full 7-Scenes 200f 8GPU persistent commit-time eviction plus cross-frame
   fusion.

Compare against:

- legacy 200f;
- current best frontend setting;
- the new persistent-commit-only ablation.

Primary metrics are `Acc`, `Comp`, and mean `NC`. The final recommendation
should not rely on one metric if the others regress materially.

The current best frontend comparison point is the best verified 7-Scenes 200f
result available at design time, not an assumed future default. The implementation
plan should record the exact log path and metrics before running new ablations.

## Acceptance Criteria

- Existing frontend/cache unit tests pass.
- New regression tests demonstrate that persistent commit no longer applies
  attention-time keep indices before geometry deduplication.
- The persistent-commit-only ablation is evaluated separately from fusion.
- Fusion is optional and disabled by default unless full 7-Scenes 200f evidence
  shows it improves the selected balance of `Acc`, `Comp`, and mean `NC`.
- If persistent commit-time eviction improves quality but fusion regresses it,
  keep the eviction fix and leave fusion as an experimental option.
- If persistent commit-time eviction regresses quality, investigate whether the
  added commit-time eviction work changes cache budget pressure before enabling
  fusion on top of it.

## Risks

- Ignoring `attention_kept_indices` can increase persistent commit-time work.
  This is acceptable if runtime remains practical on 8 GPUs.
- K/V linear blending can damage transformer feature geometry if the gate is too
  aggressive. Start with conservative alpha and make the mode optional.
- Coordinate conversion bugs can silently corrupt dedup geometry. Add targeted
  tests around non-identity slot transforms.
- Fusion may improve completion while hurting accuracy. Keep ablations separate
  so the persistent eviction fix can be accepted even if fusion is not.

## Rollout

The implementation should keep existing defaults unless a full evaluation shows
the new behavior is better. The final branch should make it easy to run:

- baseline frontend;
- persistent commit-time eviction only;
- persistent commit-time eviction plus cross-frame fusion.

Only promote a new default after 7-Scenes 200f evidence shows a real gain over
the current best frontend result.
