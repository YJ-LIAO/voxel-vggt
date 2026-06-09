# Joint Retention Policy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the poor Phase 2 TokenScorer result by separating token ranking from FIFO count prediction, using cleaner oracle pairs, and training TokenScorer + FifoCountHead jointly with measurable validation metrics.

**Architecture:** Counterfactual oracle events remain the source of supervision. `CounterfactualOracleDataset` will emit deterministic, task-correct token-ranking pairs; FIFO token pairs will compare subsets with the same `keep_count`, while `FifoCountDataset` learns the best count from `fifo_topk` events. `train_joint_retention_policy.py` will become the main training entry point and will log token-ranking validation by event type plus FIFO count validation metrics.

**Tech Stack:** PyTorch, OmegaConf YAML configs, existing `ovggt.training.token_oracle_dataset`, existing `ovggt.layers.retention_policy.JointRetentionPolicy`, pytest.

---

## Context And Diagnosis

Current log: `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/train_phase2_gpu0.log`.

Observed final metrics:

- Overall validation rank accuracy: `0.5413`.
- `dedup`: `0.5869`.
- `eviction`: `0.5121`.
- `fifo_topk`: `0.5168`.

Local oracle statistics from the four Phase 2 shards:

- Total events: `8192`, across `501` sequences.
- `fifo_topk` usable events with `min_loss_gap=0.01`: `1390 / 4096 = 33.9%`.
- Current FIFO token-ranking samples include about `85.5%` cross-`keep_count` pairs.
- Current pair cap is non-deterministic because it uses Python `random.sample` without a controlled seed.

Main conclusion: the current single-head TokenScorer training asks one scalar score to solve both "which tokens matter" and "how many tokens should remain". That is mis-specified for FIFO. Eviction is also weak because the current loss averages over entire large keep sets, so the few changed tokens are diluted.

## File Structure

Modify:

- `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
  - Deterministic pair generation.
  - FIFO same-`keep_count` pairing support as an explicit training parameter.
  - New token-pair score mode for delta-only ranking.
  - Dataset summary helpers for logging/debugging.

- `/path/to/mount/lyj/voxel-vggt/src/train_token_scorer_oracle.py`
  - Keep as an ablation script.
  - Add CLI/config parameters that mirror the joint trainer.
  - Log dataset statistics so old single-head runs are diagnosable.

- `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
  - Make this the primary optimized trainer.
  - Pass token dataset options explicitly.
  - Build train and validation datasets from the same sequence split.
  - Add validation for token ranking and count head.
  - Save validation metrics and dataset stats into checkpoint.

- `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2.yaml`
  - New production Phase 2 joint-training config using the current Phase 2 shards.

- `/path/to/mount/lyj/voxel-vggt/tools/analyze_oracle_training_signal.py`
  - New read-only diagnostic tool for event counts, loss-gap histograms, pair distribution, and count-label baseline.

Tests:

- `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_token_scorer_counterfactual.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_train_token_scorer_oracle.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_joint_retention_policy.py`

---

### Task 1: Make Oracle Pair Generation Deterministic And FIFO-Correct

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write failing tests for FIFO same-keep pairing**

Add tests that build one synthetic `fifo_topk` event with subsets at `keep_count` values `[8, 16, 32]`.

Expected behavior:

- `fifo_token_pair_mode="any"` can emit cross-count pairs.
- `fifo_token_pair_mode="same_keep_count"` emits zero cross-count pairs.
- same-count pairs still keep valid positive margins.

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: FAIL before implementation because the test should assert new explicit behavior and stats.

- [ ] **Step 2: Write failing tests for deterministic capped pairs**

Add a synthetic event with many valid positive pairs and `max_pairs_per_event=4`.

Expected behavior:

- Two dataset builds with the same `pair_sampling_seed` produce identical `(better_mask, worse_mask, target_margin)` samples.
- Different seeds may produce different samples.
- Capping happens after invalid/tied pairs are filtered.

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: FAIL until `pair_sampling_seed` and post-filter cap are implemented.

- [ ] **Step 3: Add dataset constructor parameters**

Add parameters to both `CounterfactualOracleDataset.__init__` and `CounterfactualOracleDataset.from_events`:

```python
fifo_token_pair_mode: str = "any"
max_pairs_per_event: int | None = 64
pair_sampling_seed: int = 0
```

Keep the default `fifo_token_pair_mode="any"` for backward compatibility, but every optimized config must set `same_keep_count` explicitly.

- [ ] **Step 4: Refactor pair generation to filter before cap**

Change `_append_event_pairs` so it builds a list of already-valid samples or valid pair descriptors before sampling.

Required behavior:

- Apply `target_margin > 0` and `target_margin >= min_loss_gap` before cap.
- For `fifo_topk` + `same_keep_count`, only compare subsets with equal `keep_count`.
- Deterministic sampling uses a local RNG seeded by `(pair_sampling_seed, event_id)`, not global `random`.

Implementation shape:

```python
seed_key = f"{self.pair_sampling_seed}:{event_id}"
seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
rng = random.Random(seed)
kept_indices = sorted(rng.sample(range(len(valid_pairs)), self.max_pairs_per_event))
```

- [ ] **Step 5: Add dataset stats helper**

Add a helper such as:

```python
def summarize_oracle_pair_samples(samples: Sequence[dict]) -> dict:
    ...
```

It should report:

- sample count by `event_type`
- FIFO cross-keep fraction
- target margin p10/p50/p90
- unique event count

- [ ] **Step 6: Verify tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py tests/test_token_oracle_dataset.py
git commit -m "fix: make oracle token pairs deterministic and fifo-aware"
```

---

### Task 2: Add Delta-Only Token Ranking Loss

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_scorer_counterfactual.py`

- [ ] **Step 1: Write failing loss test for changed-token scoring**

Create masks where `better_mask` and `worse_mask` share many tokens, but differ by one token each.

Expected behavior for new `score_mode="delta_mean"`:

- Common tokens do not change the score difference.
- Better-only token score larger than worse-only token score gives `rank_acc=1`.
- Better-only token score smaller gives `rank_acc=0`.

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_token_scorer_counterfactual.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Implement score mode**

Extend `token_oracle_ranking_loss` signature:

```python
def token_oracle_ranking_loss(
    logits: Tensor,
    batch: dict,
    margin_scale: float = 1.0,
    regression_weight: float = 0.1,
    score_mode: str = "set_mean",
) -> tuple[Tensor, dict]:
```

Supported modes:

- `set_mean`: current behavior.
- `delta_mean`: compare `better_mask & ~worse_mask` against `worse_mask & ~better_mask`.

Implementation detail:

```python
better_only = better_mask & ~worse_mask
worse_only = worse_mask & ~better_mask
better_score = _subset_score(logits, better_only, token_mask, reduction="mean")
worse_score = _subset_score(logits, worse_only, token_mask, reduction="mean")
```

If either delta mask is empty, fall back to `set_mean` for that sample to avoid undefined ties.

- [ ] **Step 3: Handle regression explicitly**

For optimized training, set `regression_weight=0.0`. Keep regression computation for `set_mean` compatibility.

If `score_mode="delta_mean"` and `regression_weight > 0`, either:

- compute regression with original `set_mean` scores and document this in code, or
- raise `ValueError`.

Recommended: compute regression with original `set_mean` scores for backward compatibility, but all optimized configs use `regression_weight: 0.0`.

- [ ] **Step 4: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_token_scorer_counterfactual.py tests/test_token_oracle_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py tests/test_token_scorer_counterfactual.py
git commit -m "feat: add delta token ranking loss"
```

---

### Task 3: Propagate Optimized Dataset Options Through Trainers

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/train_token_scorer_oracle.py`
- Modify: `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_token_scorer_oracle.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`

- [ ] **Step 1: Write failing CLI/config tests**

Add tests that pass config keys:

```yaml
fifo_token_pair_mode: same_keep_count
token_score_mode: delta_mean
pair_sampling_seed: 0
max_pairs_per_event: 64
```

Expected behavior:

- Both trainers parse these values from YAML.
- Values are forwarded into `CounterfactualOracleDataset.from_events`.
- `token_oracle_ranking_loss(..., score_mode=token_score_mode)` is used.

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_train_token_scorer_oracle.py tests/test_train_joint_retention_policy.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Add trainer arguments**

In both trainer parse defaults, add:

```python
"fifo_token_pair_mode": "same_keep_count",
"token_score_mode": "delta_mean",
"pair_sampling_seed": 0,
"max_pairs_per_event": 64,
```

For `train_token_scorer_oracle.py`, this makes the ablation script safer. For `train_joint_retention_policy.py`, this is the main path.

- [ ] **Step 3: Pass parameters into token dataset**

Use:

```python
CounterfactualOracleDataset.from_events(
    train_events,
    min_loss_gap=min_loss_gap,
    fifo_token_pair_mode=fifo_token_pair_mode,
    max_pairs_per_event=max_pairs_per_event,
    pair_sampling_seed=pair_sampling_seed,
)
```

Use the same settings for validation.

- [ ] **Step 4: Pass score mode into loss**

Use:

```python
token_oracle_ranking_loss(
    token_logits,
    token_batch,
    regression_weight=regression_weight,
    score_mode=token_score_mode,
)
```

- [ ] **Step 5: Log dataset stats at startup**

Use `summarize_oracle_pair_samples` and print one line each for train/val:

```text
token_dataset train samples=... by_type=... fifo_cross_keep_frac=...
```

Required optimized run invariant:

```text
fifo_cross_keep_frac=0.000
```

- [ ] **Step 6: Verify tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_train_token_scorer_oracle.py tests/test_train_joint_retention_policy.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/train_token_scorer_oracle.py src/train_joint_retention_policy.py tests/test_train_token_scorer_oracle.py tests/test_train_joint_retention_policy.py
git commit -m "feat: expose optimized token oracle training options"
```

---

### Task 4: Add Joint Validation Metrics

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`

- [ ] **Step 1: Write failing validation test**

Construct tiny train/val events and run `train_joint_retention` for one epoch on CPU.

Expected checkpoint fields:

```python
checkpoint["validation_metrics"]["token"]["rank_acc"]
checkpoint["validation_metrics"]["token"]["per_event_type"]
checkpoint["validation_metrics"]["count"]["accuracy"]
checkpoint["validation_metrics"]["count"]["mean_abs_count_error"]
checkpoint["dataset_stats"]["train_token"]
checkpoint["dataset_stats"]["val_token"]
```

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Build validation datasets**

Currently `train_joint_retention_policy.py` discards `val_events` after split. Change it to keep both:

```python
train_events, val_events = split_oracle_events(...)
```

Build:

- `val_token_dataset`
- `val_count_dataset`
- `val_token_loader`
- `val_count_loader`

If `val_fraction == 0` or no samples exist, skip validation but save empty metrics with `count=0`.

- [ ] **Step 3: Implement token validation**

Add `_evaluate_token_ranking(...)` similar to `train_token_scorer_oracle._evaluate_token_scorer`, but local to the joint trainer or imported from a shared helper.

Required output:

- loss
- pairwise
- regression
- rank_acc
- mean_score_diff
- per-event-type rank_acc/count

- [ ] **Step 4: Implement count validation**

Add `_evaluate_count_head(...)` for `JointRetentionPolicy.forward_count`.

Required output:

- cross-entropy loss
- accuracy
- mean absolute count error
- target distribution
- prediction distribution
- majority-label baseline accuracy

The model should be judged against the majority baseline, not raw accuracy alone.

- [ ] **Step 5: Print validation summaries every epoch**

Use compact lines:

```text
validation_token epoch=... samples=... rank_acc=... val/fifo_topk/rank_acc=...
validation_count epoch=... samples=... acc=... majority_acc=... mae=...
```

- [ ] **Step 6: Save validation metrics**

Save latest validation metrics in the checkpoint:

```python
"validation_metrics": {
    "token": token_validation_metrics,
    "count": count_validation_metrics,
},
"dataset_stats": dataset_stats,
```

- [ ] **Step 7: Verify tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest tests/test_train_joint_retention_policy.py tests/test_joint_retention_policy.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/train_joint_retention_policy.py tests/test_train_joint_retention_policy.py
git commit -m "feat: add joint retention validation metrics"
```

---

### Task 5: Add Phase 2 Optimized Joint Training Config

**Files:**

- Create: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2.yaml`
- Optionally modify: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy.yaml`

- [ ] **Step 1: Create Phase 2 config**

Create:

```yaml
# Joint TokenScorer + FifoCountHead training on Phase 2 oracle shards.

oracle_shards:
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu0_s0.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu1_s1.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu2_s2.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu3_s3.pt

output: /path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2.pt
score_state_proj_checkpoint: null

score_state_dim: 128
metadata_dim: 17
hidden_dim: 256
num_layers: 24
count_candidates: [0, 8, 16, 32, 64, 128]
count_head_arch: shared_encoder_v2

batch_size: 64
epochs: 10
lr: 1e-4
weight_decay: 0.01

regression_weight: 0.0
min_loss_gap: 0.01
count_loss_weight: 0.5
count_label_reduction: min
count_repeat_factor: 1

fifo_token_pair_mode: same_keep_count
token_score_mode: delta_mean
pair_sampling_seed: 0
max_pairs_per_event: 64

val_fraction: 0.1
split_key: sequence_id
split_seed: 0
device: cuda
```

- [ ] **Step 2: Keep old config as non-production**

Either leave `config/train_joint_retention_policy.yaml` as a test/example config, or update its comments to point to the new Phase 2 config. Do not silently keep the old `token_oracle_count/oracle_shard_000.pt` as the obvious production path.

- [ ] **Step 3: Verify config parses**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python src/train_joint_retention_policy.py --config config/train_joint_retention_policy_phase2.yaml --epochs 0 --device cpu --output /tmp/joint_retention_policy_phase2_parse.pt
```

Expected: Either a clean parse/no-op save if `epochs=0` is supported, or adjust the script to support a `--dry-run` option. If not adding dry-run, run Task 6 smoke instead.

- [ ] **Step 4: Commit**

```bash
git add config/train_joint_retention_policy_phase2.yaml config/train_joint_retention_policy.yaml
git commit -m "config: add phase2 joint retention training config"
```

---

### Task 6: Add Oracle Training Signal Diagnostic Tool

**Files:**

- Create: `/path/to/mount/lyj/voxel-vggt/tools/analyze_oracle_training_signal.py`
- Test: optional direct invocation; keep implementation read-only.

- [ ] **Step 1: Create diagnostic script**

The tool should accept:

```bash
--oracle-shards ...
--min-loss-gap 0.01
--fifo-token-pair-mode same_keep_count
--pair-sampling-seed 0
--max-pairs-per-event 64
```

It should print:

- event counts by type
- usable event rate by type
- loss-range p10/p50/p90 by type
- actual dataset sample counts by type
- FIFO cross-keep fraction
- count-label distribution and majority baseline
- score projection key availability by shard

- [ ] **Step 2: Run on Phase 2 shards**

Run:

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python tools/analyze_oracle_training_signal.py \
  --oracle-shards \
  checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu0_s0.pt \
  checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu1_s1.pt \
  checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu2_s2.pt \
  checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu3_s3.pt \
  --min-loss-gap 0.01 \
  --fifo-token-pair-mode same_keep_count
```

Expected:

- `fifo_cross_keep_frac=0.000`.
- Count-label majority baseline is printed.
- No shard reports missing `score_state_projection_state`.

- [ ] **Step 3: Commit**

```bash
git add tools/analyze_oracle_training_signal.py
git commit -m "tools: add oracle training signal diagnostics"
```

---

### Task 7: Run Unit And Smoke Verification

**Files:**

- No source edits unless tests expose issues.

- [ ] **Step 1: Run focused tests**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest \
  tests/test_token_oracle_dataset.py \
  tests/test_token_scorer_counterfactual.py \
  tests/test_train_token_scorer_oracle.py \
  tests/test_train_joint_retention_policy.py \
  tests/test_joint_retention_policy.py \
  tests/test_fifo_count_dataset.py \
  tests/test_fifo_count_head.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run one-epoch joint smoke**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python src/train_joint_retention_policy.py \
  --config config/train_joint_retention_policy_phase2.yaml \
  --epochs 1 \
  --output /path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_smoke.pt
```

Expected log checks:

- Token dataset startup line shows `fifo_cross_keep_frac=0.000`.
- Validation token metrics are printed.
- Validation count metrics are printed.
- Checkpoint contains `model`, `token_scorer`, `count_head`, and score projection keys.

- [ ] **Step 3: Inspect smoke checkpoint**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -c "import torch; p='checkpoints/token_scorer_oracle/joint_retention_policy_phase2_smoke.pt'; s=torch.load(p,map_location='cpu'); print(s.keys()); print(s['validation_metrics']); print(sum(k.startswith('aggregator.score_state_projs.') for k in s['model']))"
```

Expected:

- `count_head_trained=True`.
- `aggregator.score_state_projs.*` count is greater than zero.
- Validation metrics are non-empty.

- [ ] **Step 4: Commit any smoke fixes**

```bash
git status --short
git add <changed-files>
git commit -m "fix: stabilize joint retention smoke training"
```

Only commit if source/test/config changes were needed. Do not commit generated checkpoints unless explicitly requested.

---

### Task 8: Run Full Training And Compare Against Baseline

**Files:**

- No source edits expected.
- Output checkpoint: `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2.pt`
- Output log: `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/train_joint_phase2_gpu0.log`

- [ ] **Step 1: Start full training**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python src/train_joint_retention_policy.py \
  --config config/train_joint_retention_policy_phase2.yaml \
  > checkpoints/token_scorer_oracle/train_joint_phase2_gpu0.log 2>&1
```

- [ ] **Step 2: Monitor first epoch**

```bash
tail -n 80 checkpoints/token_scorer_oracle/train_joint_phase2_gpu0.log
```

Abort and debug if:

- `fifo_cross_keep_frac` is not `0.000`.
- `count_head_grad` is always `0.0000`.
- validation count sample count is `0`.
- token validation rank accuracy is exactly random and `mean_score_diff` is near zero after multiple epochs.

- [ ] **Step 3: Compare against current baseline**

Baseline from old log:

- overall val rank_acc: `0.5413`
- `eviction`: `0.5121`
- `fifo_topk`: `0.5168`
- `dedup`: `0.5869`

Minimum acceptance for optimized trainer:

- FIFO token validation uses same-count pairs only.
- Overall token val rank_acc improves over `0.5413`.
- `eviction` val rank_acc improves over `0.5121`.
- `fifo_topk` val rank_acc improves over `0.5168`.
- Count head validation accuracy beats its majority-label baseline.
- Count head mean absolute count error is logged and stable.

Stronger target:

- overall token val rank_acc `>= 0.57`
- `eviction >= 0.55`
- `fifo_topk >= 0.56`
- count accuracy at least `5%` absolute above majority baseline

- [ ] **Step 4: If full training still fails, run controlled ablations**

Create temporary config overrides, one change at a time:

1. `min_loss_gap: 0.02`
2. `min_loss_gap: 0.03`
3. `count_loss_weight: 0.25`
4. `count_loss_weight: 1.0`
5. `token_score_mode: set_mean` with same-keep FIFO, to isolate delta loss impact
6. `regression_weight: 0.05` only if ranking loss underfits

Do not change multiple knobs at once.

---

### Task 9: Runtime Load And Downstream Sanity Check

**Files:**

- Modify only if load/inference incompatibility is found:
  - `/path/to/mount/lyj/voxel-vggt/src/ovggt/models/ovggt.py`
  - `/path/to/mount/lyj/voxel-vggt/src/ovggt/models/aggregator.py`
  - `/path/to/mount/lyj/voxel-vggt/src/train_frontend.py`
- Tests:
  - `/path/to/mount/lyj/voxel-vggt/tests/test_learned_eviction_integration.py`
  - `/path/to/mount/lyj/voxel-vggt/tests/test_fifo_count_integration.py`

- [ ] **Step 1: Run checkpoint loading tests**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -m pytest \
  tests/test_learned_eviction_integration.py \
  tests/test_fifo_count_integration.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Manually inspect exported model keys**

```bash
/mnt/lyj/miniconda3/envs/StreamVGGT/bin/python -c "import torch; p='checkpoints/token_scorer_oracle/joint_retention_policy_phase2.pt'; s=torch.load(p,map_location='cpu'); m=s['model']; print('token',sum(k.startswith('aggregator.token_scorers.') for k in m)); print('count',sum(k.startswith('aggregator.count_head.') for k in m)); print('proj',sum(k.startswith('aggregator.score_state_projs.') for k in m)); print(s.get('count_head_arch'))"
```

Expected:

- token scorer keys exist for all 24 layers.
- count head keys exist.
- score projection keys exist.
- `count_head_arch == "shared_encoder_v2"`.

- [ ] **Step 3: Run downstream smoke evaluation**

Use the existing learned eviction evaluation config, but point it to the new joint checkpoint and enable the count head. If the current eval config lacks count-head fields, create a temporary config copy.

Expected:

- Model loads without missing token scorer/count head key failures.
- No random `score_state_projs` warning.
- Runtime uses `learned_fifo_keep_count=True` only when `use_count_head=True`.

- [ ] **Step 4: Commit runtime fixes if needed**

```bash
git add src/ovggt/models/ovggt.py src/ovggt/models/aggregator.py src/train_frontend.py tests/test_learned_eviction_integration.py tests/test_fifo_count_integration.py
git commit -m "fix: load joint retention policy checkpoint at runtime"
```

Only commit if code changes were needed.

---

## Rollback Strategy

If optimized joint training underperforms:

1. Keep `train_token_scorer_oracle.py` as the single-head ablation path.
2. Use `fifo_token_pair_mode=same_keep_count` and `token_score_mode=set_mean` to isolate whether delta loss caused the regression.
3. Use `regression_weight=0.0` as the default baseline; only re-enable regression after ranking improves.
4. Use `split_key=sequence_id` for all reported validation. Do not compare against old `event_id_hash` validation as a final metric.

## Definition Of Done

- Focused pytest suite passes.
- Diagnostic tool reports FIFO token samples have `fifo_cross_keep_frac=0.000` in optimized config.
- Joint trainer logs validation token metrics by event type and count-head metrics by epoch.
- Full training checkpoint contains deployable TokenScorer, FifoCountHead, and score projection weights.
- Full training improves over old Phase 2 baseline on overall token validation and both weak event types: `eviction` and `fifo_topk`.
- Count head validation beats majority-label baseline.

