# Frontend Cache Migration Design

## Goal

Migrate the production frontend cache path from `/path/to/mount/lyj/voxel-vggt` into `/Train/LYJ/workspace/OVGGT` so `ovggt.models.OVGGT` can run the production keyframe manager and voxel-aware cache maintenance path during inference.

## Scope

The migration includes:

- Keyframe event scheduling and FIFO history-anchor maintenance.
- Per-layer cache state with K/V tensors, token metadata, protected-region ordering, FIFO rescued-token ring, and voxel deduplication.
- Frontend inference path that commits current-frame K/V only after pose/depth heads produce geometry metadata.
- Aggregator support for deferred cache updates via `PendingLayerUpdate`.
- Block/attention support needed by the deferred update path and anchor overflow policy.
- Focused tests for keyframe events, voxel hash uniqueness, cache metadata/dedup, FIFO protected-ring behavior, and frontend path import/compile compatibility.

The migration excludes:

- Oracle data collection tools.
- Learned TokenScorer/FifoCountHead retention routes.
- Full training loss/config migration unless a compile-time dependency is required.
- Replacing unrelated `vggt` or `dust3r` code.

## Current State

`/Train/LYJ/workspace/OVGGT` already has an `src/ovggt` package with streaming cache support, but its inference path uses the older `HistoryAnchorManager` and direct `Aggregator.sync_anchor_change()` cache rearrangement. It does not have the production `frontend_cache.py` and `frontend_keyframe.py` modules.

The source production branch adds a richer frontend path where cache updates are delayed until after camera/depth heads produce pose and local 3D metadata. That enables voxel deduplication and keyframe-aware metadata maintenance instead of simple token-count anchor zones.

## Architecture

The target repository will support two inference paths:

- Legacy path: preserves the existing behavior based on `HistoryAnchorManager`, useful for backward compatibility.
- Frontend path: enabled through `mode in {"frontend_train", "frontend_eval"}` or `FrontendCacheConfig.enabled`, using production keyframe and voxel cache maintenance.

The frontend path processes frames one at a time. For each frame, `Aggregator.forward()` computes current-frame global-attention K/V and returns one `PendingLayerUpdate` per global layer. `OVGGT._inference_frontend()` then predicts camera pose and depth, asks `FrontendKeyframeManager` for the keyframe event, builds `TokenMetadata` from depth/pose, applies the event to every `LayerCacheState`, and commits the pending K/V with intra-frame pruning, voxel dedup, FIFO protected-ring policy, and final eviction.

## Data Flow

1. Convert one input frame to `[B, 1, C, H, W]`.
2. Run aggregator with `frontend_cache_mode=True`.
3. Run camera head using per-sample camera cache and keyframe-aware anchor counts.
4. Run depth and point heads on the aggregated tokens.
5. Update `FrontendKeyframeManager` using predicted depth and absolute pose.
6. For each layer:
   - optionally protect top-K tokens from a demoted FIFO slot;
   - apply keyframe event to existing metadata;
   - build current-frame token metadata from depth, confidence, pose, and keyframe id;
   - commit `PendingLayerUpdate` into `LayerCacheState`;
   - apply voxel deduplication and eviction.
7. Sync camera-head cache with the keyframe event.
8. Return per-frame predictions and optional keyframe schedule/packets.

## File Changes

Create:

- `src/ovggt/utils/frontend_keyframe.py`
- `src/ovggt/utils/frontend_cache.py`
- Focused tests under `tests/` for migrated cache/keyframe behavior.

Modify:

- `src/ovggt/models/ovggt.py`
- `src/ovggt/models/aggregator.py`
- `src/ovggt/layers/block.py`
- `src/ovggt/layers/attention.py`
- `src/ovggt/heads/camera_head.py` only if the target version lacks production keyframe-event camera cache sync behavior.

Do not modify:

- `src/vggt/**`
- `src/dust3r/**`
- training/oracle scripts unless a required import is missing.

## Compatibility

The public `OVGGT.forward()` behavior remains compatible for non-frontend usage. Existing `OVGGT.inference()` arguments remain accepted where possible. New frontend-specific construction options mirror the production branch:

- `mode`
- `frontend_pose_encoding_type`
- `frontend_cache_config`
- `keyframe_switch_config`
- `per_layer_budget`
- `anchor_overflow_policy`
- `camera_num_iters`

The existing `total_budget` parameter should be mapped conservatively to `per_layer_budget` compatibility or preserved as a legacy alias so current evaluation scripts do not break.

## Error Handling

The migration should fail fast for incompatible cache settings:

- `fifo_protected_ring_ratio > 0` requires uniform budget allocation.
- `fifo_protected_ring_ratio` and `max_protected_ratio < 1.0` are mutually exclusive.
- Frontend cache should reject inconsistent frame batch sizes.
- Cache metadata and K/V gather operations must preserve matching token counts.

## Testing

The first tests should be small CPU tests that do not require model checkpoints:

- `FrontendKeyframeManager` emits initial promotion, fixed-interval promotion, and FIFO swap events with expected slot ids and pose-update retention.
- `voxel_hash_collision_free()` does not collide for large positive/negative voxel coordinates.
- `LayerCacheState` keeps K/V and metadata aligned after gather/reorder.
- FIFO rescued-ring capacity revokes old rescued tokens before protecting new demoted tokens.
- `FrontendCacheConfig` rejects invalid ring/dynamic-budget combinations.

Then add import/compile smoke checks for `OVGGT`, `Aggregator`, `Block`, `Attention`, and frontend utils.

Full model inference with checkpoints is out of scope for the first migration verification unless the local environment already has the required checkpoint and dependencies.

## Risks

The main risk is partial migration: copying utility modules without the deferred commit path would leave keyframe/voxel logic unused. The implementation must wire `OVGGT._inference_frontend()`, aggregator pending updates, and cache commit together in one coherent path.

The second risk is reintroducing unrelated learned-retention training dependencies. The migrated frontend inference path intentionally excludes the learned `TokenScorer` and `FifoCountHead` routes after local validation showed that route is not usable for this migration.

The third risk is budget semantic drift. The target repo currently uses `total_budget`; the production branch uses `per_layer_budget`. The implementation should preserve old call sites while making frontend mode use the production per-layer semantics.
