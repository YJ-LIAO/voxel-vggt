# Oracle FIFO/Dedup Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make oracle collection cover FIFO Top-K events and broaden Dedup frame/layer diversity without changing the training target semantics.

**Architecture:** Keep replay/loss semantics unchanged. Improve only oracle collection controls: force earlier FIFO swaps for oracle runs, separate probe-layer scheduling from event selection, remove selector bias toward early frame/layer groups, and add diagnostics so coverage failures are visible before long runs.

**Tech Stack:** Python, PyTorch, OmegaConf YAML config, pytest, existing `ovggt.training.frontend_oracle_collector` collector.

---

## Baseline Evidence

Current interrupted run:

```bash
python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --dataset-key train_dataset \
  --output checkpoints/token_oracle_phase2/phase2_shard_gpu7.pt \
  --max-batches 128 \
  --max-events 4096 \
  --event-selection-policy quota_stratified \
  --event-type-quotas eviction=8,dedup=4,fifo_topk=4 \
  --oracle-profile fifo_topk \
  --frontend-per-layer-budget-override 2500 \
  --fifo-keep-topk-override 80 \
  --layers-per-frame 2
```

Observed shard after stopping:

```text
events=344
event_type_counts={"dedup":216, "eviction":128}
fifo_topk=0
dedup_frames={0:108, 1:108}
dedup_layers={0:54, 1:54, 12:54, 13:54}
```

Log examples:

```text
probe captured 0 eviction, 144 dedup, 0 fifo candidates, 4 total have future frames
probe captured 8 eviction, 144 dedup, 0 fifo candidates, 12 total have future frames
```

## Root Cause Summary

### No FIFO Top-K

`CounterfactualFifoTopKProbe` is only called from `LayerCacheState.protect_topk_on_demotion_()`, which is only reached when the frontend keyframe manager emits a `FIFO_SWAP` event. The current run did not pass `oracle_anchor_interval`; with the default keyframe schedule, 24-frame sequences often do not produce FIFO swaps. Therefore `fifo candidates=0` is expected.

The requested change `oracle_anchor_interval=4` is the right first fix because it forces more frequent anchor insertion and should cause FIFO slot demotion within a 24-frame sequence. For stronger FIFO pressure, also set `oracle_max_anchors=2` in a dedicated FIFO profile if `oracle_anchor_interval=4` alone still produces no FIFO candidates.

### Dedup Only Four Layers

This is caused by the current default layer schedule and the fact that Dedup events happen early:

```python
selected = {
    (frame_id + offset * stride) % num_layers
    for offset in range(layers_per_frame)
}
```

With `layers_per_frame=2`, `num_layers=24`, `stride=12`:

```text
frame 0 -> layers 0, 12
frame 1 -> layers 1, 13
```

Because raw Dedup candidates are concentrated at frame 0 and frame 1, the scheduler can only record layers `{0, 12, 1, 13}`.

### Dedup Only Frame 0/1

There are two contributing factors:

1. Dedup itself appears to trigger mainly in early frames for many sequences because the first one or two frames produce dense overlapping voxel groups before cache/keyframe state diversifies.
2. `quota_stratified` currently sorts groups by ascending keys and then round-robins. With `dedup=4`, it consistently selects the first `(frame_bucket, layer_bucket)` groups, which amplifies early-frame bias.

The fix is not just increasing `max_events_per_sequence`. That would make collection much slower and still over-sample frame 0/1. We need targeted diversity controls.

## Immediate Operational Change

Use `config/collect_counterfactual_oracle.yaml` instead of passing `config/train_frontend_finetune.yaml` directly, or explicitly pass the oracle arguments on the command line.

The collector config now sets:

```yaml
oracle_anchor_interval: 4
```

If launching with the training YAML directly, the equivalent flag is required:

```bash
--oracle-anchor-interval 4
```

Recommended next smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.pt \
  --max-batches 8 \
  --max-events 64 \
  --max-events-per-sequence 16 \
  --event-selection-policy quota_stratified \
  --event-type-quotas eviction=4,dedup=8,fifo_topk=8 \
  --quota-fill-remaining \
  --layers-per-frame 0 \
  --oracle-layer-schedule random_bucket \
  --oracle-layer-buckets '[[0,5],[6,11],[12,17],[18,23]]' \
  --oracle-anchor-interval 4 \
  --oracle-max-anchors 2 \
  --oracle-profile fifo_topk \
  --frontend-per-layer-budget-override 2500 \
  --fifo-keep-topk-override 80 \
  --fifo-count-candidates-for-oracle 0,8,16,32,64,128 \
  --probe-only \
  --probe-output-json checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.json
```

Expected probe-only success criteria:

```text
fifo candidates > 0 on at least some batches
raw dedup layers hit >= 8 unique layers across 8 batches
selected dedup frame histogram is not only {0,1}
```

If FIFO is still 0, run the same probe with:

```bash
--oracle-anchor-interval 2 --oracle-max-anchors 2
```

## Task 1: Add FIFO-Swap Diagnostics

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write the failing test**

Add a unit test around `_build_probe_diagnostics()` that expects keyframe/FIFO diagnostics fields:

```python
def test_probe_diagnostics_reports_fifo_absence_reason():
    diag = collector._build_probe_diagnostics(
        eviction_events=[],
        dedup_events=[{"event_type": "dedup", "frame_id": 0, "layer_id": 0}],
        fifo_events=[],
        candidate_events=[],
        filtered_total=0,
        keyframe_event_counts={"NOOP": 20, "NEW_ANCHOR": 4},
    )
    assert diag["raw_event_type_counts"]["dedup"] == 1
    assert diag["keyframe_event_counts"]["NOOP"] == 20
    assert diag["fifo_swap_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_probe_diagnostics_reports_fifo_absence_reason -q
```

Expected: FAIL because `_build_probe_diagnostics()` does not accept/report keyframe event counts yet.

- [ ] **Step 3: Implement minimal diagnostics**

Add optional `keyframe_event_counts: dict[str, int] | None = None` to `_build_probe_diagnostics()` and include:

```python
"keyframe_event_counts": dict(sorted((keyframe_event_counts or {}).items())),
"fifo_swap_count": sum(v for k, v in (keyframe_event_counts or {}).items() if "FIFO_SWAP" in str(k)),
```

- [ ] **Step 4: Thread keyframe diagnostics from model outputs**

If `model.inference(..., return_views=False)` still returns `outputs.keyframe_schedule`, histogram it in `collect_oracle_events_from_sequence()`. If not present, add an explicit debug log:

```text
keyframe_schedule unavailable; cannot diagnose FIFO_SWAP count
```

Do not change model inference semantics.

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_probe_diagnostics_reports_fifo_absence_reason -q
```

Expected: PASS.

## Task 2: Split Probe Layer Schedule From Dedup Selection Bias

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write failing selector test for frame-bucket fairness**

Create synthetic Dedup candidates where early frames are abundant but later frames exist. Assert quota selection includes later frame buckets:

```python
def test_quota_stratified_rotates_frame_buckets_by_seed():
    candidates = []
    for frame in [0, 1, 2, 3, 8, 12, 18]:
        for layer in [0, 6, 12, 18]:
            candidates.append({"event_type": "dedup", "frame_id": frame, "layer_id": layer, "voxel_group_id": frame * 100 + layer})

    selected = collector.select_oracle_events(
        candidates,
        max_events=8,
        max_events_per_frame=2,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas={"dedup": 8},
        frame_buckets=[(0, 3), (4, 8), (9, 15), (16, 23)],
        selection_seed=17,
    )

    selected_frames = {e["frame_id"] for e in selected}
    assert any(f >= 8 for f in selected_frames)
    assert len({e["layer_id"] // 6 for e in selected}) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_quota_stratified_rotates_frame_buckets_by_seed -q
```

Expected: FAIL because `selection_seed` does not exist and sorted group order favors low frame buckets.

- [ ] **Step 3: Add deterministic group rotation**

Add `selection_seed: int = 0` to:

- `select_oracle_events()`
- `_select_oracle_events_quota_stratified()`
- `collect_oracle_events_from_sequence()`
- `collect_oracle_events_from_loader()`
- `collect_oracle_shard_from_config()`
- CLI/YAML mapping

Before round-robin, rotate each type's sorted group list:

```python
keys = sorted(type_groups_dict.keys())
if keys:
    offset = int(selection_seed) % len(keys)
    keys = keys[offset:] + keys[:offset]
type_groups = [list(type_groups_dict[k]) for k in keys]
```

Use `seed + batch_idx * 1009` as the per-sequence selection seed so each sequence starts from a different frame/layer bucket.

- [ ] **Step 4: Verify selector tests**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_quota_stratified_rotates_frame_buckets_by_seed tests/test_frontend_oracle_collector.py::test_frame_bucket_selection_prefers_later_frames -q
```

Expected: PASS.

## Task 3: Add Dedup-Specific Probe Caps

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

Problem: one global `layers_per_frame` is currently shared by eviction, dedup, and FIFO. Low `layers_per_frame` is useful for eviction speed, but harmful for Dedup layer diversity.

- [ ] **Step 1: Write failing config/plumbing test**

Add fields:

```python
dedup_layers_per_frame: int | None = None
fifo_layers_per_frame: int | None = None
eviction_layers_per_frame: int | None = None
```

Test that when `dedup_layers_per_frame=0` and `eviction_layers_per_frame=2`, the Dedup probe sees all layers while eviction still uses the cap.

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_collector_uses_event_type_specific_layer_caps -q
```

Expected: FAIL because event-type-specific layer caps do not exist.

- [ ] **Step 3: Implement event-type-specific caps**

In `collect_oracle_events_from_sequence()`:

```python
eviction_lpf = layers_per_frame if eviction_layers_per_frame is None else eviction_layers_per_frame
dedup_lpf = layers_per_frame if dedup_layers_per_frame is None else dedup_layers_per_frame
fifo_lpf = layers_per_frame if fifo_layers_per_frame is None else fifo_layers_per_frame
```

Pass these separately to `CounterfactualEvictionProbe`, `CounterfactualDedupProbe`, and `CounterfactualFifoTopKProbe`.

- [ ] **Step 4: Add CLI/YAML**

Add:

```bash
--dedup-layers-per-frame
--eviction-layers-per-frame
--fifo-layers-per-frame
```

Add to `_YAML_TO_ARG` and `FrontendOracleCollectorConfig`.

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/test_frontend_oracle_collector.py -q
```

Expected: PASS.

## Task 4: Configure Production Diversity Profile

**Files:**
- Modify: `config/collect_counterfactual_oracle.yaml`

Recommended defaults:

```yaml
oracle_anchor_interval: 4
oracle_max_anchors: 2
oracle_layer_schedule: random_bucket
oracle_layer_buckets: [[0, 5], [6, 11], [12, 17], [18, 23]]
frame_buckets: [[0, 3], [4, 8], [9, 15], [16, 23]]
layers_per_frame: 2
dedup_layers_per_frame: 0
eviction_layers_per_frame: 2
fifo_layers_per_frame: 4
event_type_quotas:
  eviction: 4
  dedup: 8
  fifo_topk: 8
quota_fill_remaining: true
```

Rationale:

- Dedup replay is batched and comparatively fast, so allow wider Dedup probe coverage.
- Eviction replay is serial and slow, so keep a narrow cap.
- FIFO should be sampled aggressively only after we verify `FIFO_SWAP` exists.

## Task 5: Probe-Only Validation Before Full Collection

**Files:**
- No code changes.

- [ ] **Step 1: Run probe-only smoke**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.pt \
  --max-batches 16 \
  --max-events 128 \
  --probe-only \
  --probe-output-json checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.json
```

- [ ] **Step 2: Inspect diagnostics**

```bash
PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python tools/summarize_oracle_event_diversity.py \
  --log checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.log \
  --probe-json checkpoints/token_oracle_debug/fifo_dedup_diversity_probe_gpu0.json
```

Expected thresholds:

```text
raw fifo_topk > 0
selected fifo_topk > 0 if raw fifo_topk > 0
selected dedup unique frames >= 4
selected dedup unique layer buckets >= 4
```

If FIFO is still 0:

1. Keep `oracle_anchor_interval=4`.
2. Lower `oracle_max_anchors` to 1 for a FIFO stress-only shard.
3. Verify `keyframe_event_counts` shows `FIFO_SWAP`.

## Task 6: Small Replay Validation

**Files:**
- No code changes.

- [ ] **Step 1: Run small replay collection**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_debug/fifo_dedup_diversity_replay_gpu0.pt \
  --max-batches 8 \
  --max-events 64 \
  --max-events-per-sequence 16 \
  --flush-every-events 8 \
  --flush-every-batches 1
```

- [ ] **Step 2: Verify shard diversity**

```bash
PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python - <<'PY'
import torch, collections
p='checkpoints/token_oracle_debug/fifo_dedup_diversity_replay_gpu0.pt'
d=torch.load(p, map_location='cpu', weights_only=False)
events=d.get('events', [])
print('events', len(events))
print('types', collections.Counter(e.get('event_type') for e in events))
for et in ['dedup','fifo_topk','eviction']:
    xs=[e for e in events if e.get('event_type') == et]
    print(et, 'frames', collections.Counter(e.get('frame_id') for e in xs))
    print(et, 'layers', collections.Counter(e.get('layer_id') for e in xs))
PY
```

Expected:

```text
fifo_topk appears, or diagnostics clearly show no FIFO_SWAP
dedup no longer only frame 0/1
dedup no longer only layers 0/1/12/13
```

## Task 7: Full Collection Command

Status before launch:

- Task 5 probe-only validation passed on GPU0:
  - `raw_types={'dedup': 2061, 'fifo_topk': 96, 'eviction': 40}`
  - `selected_types={'dedup': 52, 'fifo_topk': 64, 'eviction': 12}`
  - `fifo_swap_count=24`
  - raw coverage: 24 unique frames, 24 unique layers
  - selected coverage: 9 unique frames, 15 unique layers
- Task 6 small replay validation initially exposed a FIFO replay bug:
  - `fifo_topk` probe candidates replayed through the eviction hook and were skipped with `serial replay keep set was not applied`.
  - Fixed by adding FIFO-specific replay override through `on_fifo_topk_candidate()`.
  - Regression tests added for FIFO replay override and FIFO replay probe plumbing.
- Task 6 re-run passed on GPU0:
  - shard: `checkpoints/token_oracle_debug/fifo_dedup_diversity_replay_gpu0_fixed.pt`
  - `events=8`, `types={'fifo_topk': 4, 'dedup': 4}`
  - `frames={12: 4, 0: 4}`, `layers={0: 2, 8: 2, 16: 2, 18: 2}`
  - `subsets=[12, 12, 12, 12, 8, 8, 8, 8]`, `partial=False`
- Verification:
  - `python -m pytest tests/test_frontend_oracle_collector.py -q` -> 71 passed
  - `python -m pytest tests/test_token_oracle_dataset.py -q` -> 26 passed
  - `py_compile` passed for collector/cache/CLI files
  - `git diff --check` passed for edited files

Use this only after Task 5 and Task 6 pass:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src nohup \
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu7.pt \
  --max-batches 128 \
  --max-events 4096 \
  --seed 42 \
  > checkpoints/token_oracle_phase2/collect_diverse_gpu7.log 2>&1 &
```

Parallel launch used on 2026-06-06:

- GPU7 was not used because it already had an unrelated `wekws/bin/python` process occupying about 36GB.
- The initial unsharded GPU0 launch was stopped before producing a shard.
- Four mutually exclusive sequence shards were launched on GPU0-3:
  - `oracle_diverse_gpu0_s0` -> `phase2_diverse_shard_gpu0_s0.pt`, log `collect_diverse_gpu0_s0.log`, `sequence_shard_id=0`, `seed=42`
  - `oracle_diverse_gpu1_s1` -> `phase2_diverse_shard_gpu1_s1.pt`, log `collect_diverse_gpu1_s1.log`, `sequence_shard_id=1`, `seed=43`
  - `oracle_diverse_gpu2_s2` -> `phase2_diverse_shard_gpu2_s2.pt`, log `collect_diverse_gpu2_s2.log`, `sequence_shard_id=2`, `seed=44`
  - `oracle_diverse_gpu3_s3` -> `phase2_diverse_shard_gpu3_s3.pt`, log `collect_diverse_gpu3_s3.log`, `sequence_shard_id=3`, `seed=45`
- Startup verification:
  - all four sessions reached `dataloader ready`
  - all four sessions reached `batch 1/128 start`
  - all four first probes captured FIFO candidates:
    - GPU0/s0: `4 eviction, 258 dedup, 12 fifo candidates`
    - GPU1/s1: `3 eviction, 258 dedup, 12 fifo candidates`
    - GPU2/s2: `0 eviction, 258 dedup, 12 fifo candidates`
    - GPU3/s3: `3 eviction, 256 dedup, 12 fifo candidates`

Do not use `config/train_frontend_finetune.yaml` directly unless all oracle collection options are explicitly passed on the CLI.

## Out Of Scope

- Changing model/frontend training semantics.
- Changing Dedup/FIFO replay loss definitions.
- Making eviction replay batched again.
- Increasing `max_events_per_sequence` blindly without diversity diagnostics.
