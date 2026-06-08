# Retention Oracle Signal Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current `/path/to/mount/lyj/voxel-vggt` source tree train a reliable joint TokenScorer + FifoCountHead policy by first fixing missing trainer wiring, then filtering weak oracle labels, selecting best checkpoints, and running diagnostics that separate data-quality limits from model-capacity limits.

**Architecture:** Treat `JointRetentionPolicy` as the baseline architecture and improve the data/training system around it. The first phase makes the current repo self-contained: deterministic oracle pair generation, `delta_mean` token ranking, real train/val validation, dataset statistics, and Phase 2 configs. The second phase adds high-confidence token/count filtering, checkpoint selection, CountHead deploy gating, diagnostics, and controlled ablations.

**Tech Stack:** PyTorch, OmegaConf YAML configs, `ovggt.training.token_oracle_dataset`, `src/train_joint_retention_policy.py`, pytest, existing Phase 2 oracle shards.

---

## Current State

Run all relative-path commands from:

```bash
cd /path/to/mount/lyj/voxel-vggt
```

Important source/artifact mismatch:

- Current `src/train_joint_retention_policy.py` discards validation events with `train_events, _ = split_oracle_events(...)`.
- Current `src/train_joint_retention_policy.py` does not parse or forward `fifo_token_pair_mode`, `token_score_mode`, `pair_sampling_seed`, `max_pairs_per_event`, `min_count_loss_gap`, `save_best`, or deploy-gating options.
- Current `tools/analyze_oracle_training_signal.py` and `config/train_joint_retention_policy_phase2.yaml` are missing.
- Current `src/ovggt/training/token_oracle_dataset.py` has `fifo_token_pair_mode` and `max_pairs_per_event`, but capping is not deterministic and the trainer does not pass these options.
- Existing checkpoint `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2.pt` contains useful metrics, but it was produced by code that is not present in the current source tree. Use it as evidence only; do not assume the current trainer can reproduce it.

Observed evidence:

- Single-head TokenScorer Phase 2 final validation from `train_phase2_gpu0.log`: overall `0.5413`, `dedup=0.5869`, `eviction=0.5121`, `fifo_topk=0.5168`.
- Existing joint checkpoint metrics: token `rank_acc=0.5189`, `dedup=0.5245`, `eviction=0.5549`, `fifo_topk=0.4225`; CountHead `accuracy=0.3235`, majority baseline `0.3407`.
- Existing joint checkpoint dataset stats: train margin p50 `0.0250`, val margin p50 `0.0228`; val FIFO token samples only `284`.
- CountHead prediction distribution in the existing checkpoint is majority-like: `8` is predicted `370 / 408` times.

Conclusion: do not start by increasing model size. First make the training path reproducible in the current repo, remove mis-specified/low-confidence supervision, and select/deploy checkpoints using validation evidence.

## File Structure

Modify:

- `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
  - Deterministic pair sampling.
  - Token pair summaries.
  - `score_mode` support.
  - Per-event-type token filtering.
  - FIFO count-label confidence filtering.

- `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
  - Parse and forward all dataset/loss options.
  - Build train and validation datasets from the same event split.
  - Log and save token/count validation metrics.
  - Save best checkpoints and support early stopping.
  - Gate CountHead deploy export based on validation quality.

- `/path/to/mount/lyj/voxel-vggt/tools/analyze_oracle_training_signal.py`
  - New diagnostic tool for threshold sweeps, sequence distribution, count-label confidence, and shard projection-state availability.

- `/path/to/mount/lyj/voxel-vggt/tools/overfit_joint_retention_sanity.py`
  - New small-data overfit sanity tool.

- `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2.yaml`
  - New baseline Phase 2 joint-training config.

- `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2_highconf.yaml`
  - New high-confidence config, created after diagnostics choose thresholds.

- `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2_capacity_ablation.yaml`
  - Optional larger hidden-dim ablation.

Tests:

- `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_fifo_count_dataset.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`
- `/path/to/mount/lyj/voxel-vggt/tests/test_token_scorer_counterfactual.py`

---

### Task 1: Make Token Pair Dataset Deterministic And Summarizable

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write failing deterministic cap test**

Add a synthetic event with many valid positive pairs and `max_pairs_per_event=4`.

Expected behavior:

- Two builds with the same `pair_sampling_seed` produce identical `(event_id, better_mask, worse_mask, target_margin)` samples.
- Two builds with different seeds may produce different capped samples.
- Capping happens after invalid/tied pairs are filtered.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: FAIL before implementation because `pair_sampling_seed` does not exist and capping uses global randomness.

- [ ] **Step 2: Add `pair_sampling_seed` to dataset constructors**

Add to both `CounterfactualOracleDataset.__init__` and `CounterfactualOracleDataset.from_events`:

```python
pair_sampling_seed: int = 0
```

Store it:

```python
self.pair_sampling_seed = int(pair_sampling_seed)
```

- [ ] **Step 3: Move pair capping after valid-pair filtering**

Refactor `_append_event_pairs(...)` so it first creates valid pair samples or valid pair descriptors, then applies `max_pairs_per_event`.

Required behavior:

- Apply `target_margin > 0` and `target_margin >= min_loss_gap` before capping.
- In `fifo_token_pair_mode="same_keep_count"`, only compare FIFO subsets with equal `keep_count`.
- Store FIFO pair metadata on emitted samples:

```python
"better_keep_count": better_subset.get("keep_count", event_keep_count),
"worse_keep_count": worse_subset.get("keep_count", event_keep_count),
```

For non-FIFO events or old shards without keep-count metadata, store `None`.
- Use local deterministic RNG:

```python
seed_key = f"{self.pair_sampling_seed}:{event_id}"
seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
rng = random.Random(seed)
kept_indices = sorted(rng.sample(range(len(valid_pairs)), self.max_pairs_per_event))
```

Do not use global `random.sample`.

- [ ] **Step 4: Add token dataset summary helper**

Add:

```python
def summarize_oracle_pair_samples(samples: Sequence[dict]) -> dict:
    ...
```

Required fields:

- `count`
- `by_event_type`
- `unique_event_count`
- `fifo_count`
- `fifo_cross_keep_count`
- `fifo_cross_keep_frac`
- `target_margin_p10`
- `target_margin_p25`
- `target_margin_p50`
- `target_margin_p75`
- `target_margin_p90`
- `by_event_type_margin_p50`
- `by_event_type_unique_event_count`

Compute `fifo_cross_keep_count` from `better_keep_count` and `worse_keep_count`, not from masks or losses. If either side is missing, do not count that sample as cross-keep.
Keep the helper pure and usable by both trainers and diagnostic scripts.

- [ ] **Step 5: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py tests/test_token_oracle_dataset.py
git commit -m "fix: make oracle token pair sampling deterministic"
```

---

### Task 2: Add Delta-Only Token Ranking And High-Confidence Token Filtering

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_scorer_counterfactual.py`

- [ ] **Step 1: Write failing loss test for `score_mode="delta_mean"`**

Create masks where `better_mask` and `worse_mask` share many common tokens and differ by one token each.

Expected behavior:

- Common tokens do not affect the rank score difference.
- If the better-only token score is larger than the worse-only token score, `rank_acc=1`.
- If the better-only token score is smaller, `rank_acc=0`.
- If either delta side is empty, that sample falls back to `set_mean`.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_scorer_counterfactual.py -q
```

Expected: FAIL before implementation because `score_mode` does not exist.

- [ ] **Step 2: Implement `score_mode` in `token_oracle_ranking_loss`**

Extend signature:

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
- `delta_mean`: compare `better_mask & ~worse_mask` to `worse_mask & ~better_mask`.

If `score_mode="delta_mean"` and `regression_weight > 0`, compute regression using the original `set_mean` scores for backward compatibility. Optimized configs set `regression_weight: 0.0`.

- [ ] **Step 3: Write failing tests for per-event-type filtering**

Create synthetic `dedup`, `eviction`, and `fifo_topk` events with margins `[0.01, 0.02, 0.04]`.

Expected behavior:

- `min_loss_gap` still works globally.
- `min_loss_gap_by_event_type={"fifo_topk": 0.03, "dedup": 0.02}` filters by event type.
- Event types not in the mapping fall back to global `min_loss_gap`.
- Unknown mapping keys are ignored.
- If `max_loss_gap < min_loss_gap`, dataset length is `0`.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 4: Add filtering parameters**

Add to `CounterfactualOracleDataset.__init__` and `.from_events`:

```python
min_loss_gap_by_event_type: dict[str, float] | None = None
max_loss_gap: float | None = None
```

Add helper:

```python
def _event_min_loss_gap(self, event_type: str) -> float:
    if self.min_loss_gap_by_event_type and event_type in self.min_loss_gap_by_event_type:
        return float(self.min_loss_gap_by_event_type[event_type])
    return float(self.min_loss_gap)
```

Use the event-specific threshold in both event-level loss-range filtering and pair-level `target_margin` filtering.

Pair acceptance rule:

```python
min_gap = self._event_min_loss_gap(event_type)
if target_margin <= 0.0 or target_margin < min_gap:
    return None
if self.max_loss_gap is not None and target_margin > float(self.max_loss_gap):
    return None
```

- [ ] **Step 5: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest \
  tests/test_token_oracle_dataset.py \
  tests/test_token_scorer_counterfactual.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py tests/test_token_oracle_dataset.py tests/test_token_scorer_counterfactual.py
git commit -m "feat: add delta ranking and high-confidence token filtering"
```

---

### Task 3: Add FIFO Count Label Confidence

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/ovggt/training/token_oracle_dataset.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_fifo_count_dataset.py`

- [ ] **Step 1: Write failing tests for ambiguous labels**

Create one `fifo_topk` event with reduced candidate losses:

```python
{
    0: [0.50],
    8: [0.49],
    16: [0.10],
    32: [0.11],
}
```

Expected behavior:

- With `min_count_loss_gap=0.0`, target keep count is `16`.
- With `min_count_loss_gap=0.05`, the event is dropped because best-vs-second-best gap is `0.01`.
- With one candidate only and `min_count_loss_gap > 0`, the event is dropped because no competing label proves confidence.
- Events with all gaps below threshold produce an empty dataset.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_fifo_count_dataset.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Add `min_count_loss_gap`**

Add to `FifoCountDataset.__init__` and `.from_events`:

```python
min_count_loss_gap: float = 0.0
```

Store it:

```python
self.min_count_loss_gap = float(min_count_loss_gap)
```

- [ ] **Step 3: Compute count-label confidence**

Inside `_append_event`, reduce group losses to scalar candidate losses, sort them, then filter:

```python
candidate_losses: dict[int, float] = {}
for kc, losses in groups.items():
    if kc not in candidate_set:
        continue
    candidate_losses[kc] = min(losses) if self.label_reduction == "min" else sum(losses) / len(losses)

if len(candidate_losses) < 1:
    return

ranked = sorted(candidate_losses.items(), key=lambda item: item[1])
best_keep_count, best_loss = ranked[0]
if len(ranked) < 2:
    if self.min_count_loss_gap > 0.0:
        return
    second_loss = best_loss
else:
    second_loss = ranked[1][1]

count_loss_gap = second_loss - best_loss
if count_loss_gap < self.min_count_loss_gap:
    return

target = self.count_candidates.index(best_keep_count)
```

Store `count_loss_gap`, `best_count_loss`, and `second_best_count_loss` in each sample.

- [ ] **Step 4: Collate confidence fields**

Extend `collate_fifo_count_samples`:

```python
"count_loss_gap": torch.tensor([sample.get("count_loss_gap", 0.0) for sample in samples], dtype=torch.float32)
```

- [ ] **Step 5: Add count sample summary helper**

Add:

```python
def summarize_fifo_count_samples(samples: Sequence[dict]) -> dict:
    ...
```

Required fields:

- `count`
- `target_keep_count`
- `count_loss_gap_p10`
- `count_loss_gap_p50`
- `count_loss_gap_p90`

- [ ] **Step 6: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_fifo_count_dataset.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py tests/test_fifo_count_dataset.py
git commit -m "feat: filter ambiguous fifo count labels"
```

---

### Task 4: Wire Dataset Options And Validation Into Joint Trainer

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`

- [ ] **Step 1: Write failing config-forwarding test**

Create a temporary YAML with:

```yaml
fifo_token_pair_mode: same_keep_count
token_score_mode: delta_mean
pair_sampling_seed: 7
max_pairs_per_event: 4
min_loss_gap_by_event_type:
  eviction: 0.02
max_loss_gap: 0.5
min_count_loss_gap: 0.01
```

Expected behavior:

- `parse_args(["--config", config])` returns these exact values.
- `train_joint_retention(...)` accepts these parameters directly.
- Checkpoint `training_options` stores the exact values.
- A tiny CLI smoke invocation also stores the exact values, proving `main()` forwards parsed YAML/CLI values into `train_joint_retention(...)` instead of silently dropping them.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Add parser defaults and CLI options**

Add defaults:

```python
"fifo_token_pair_mode": "same_keep_count",
"token_score_mode": "delta_mean",
"pair_sampling_seed": 0,
"max_pairs_per_event": 64,
"min_loss_gap_by_event_type": None,
"max_loss_gap": None,
"min_count_loss_gap": 0.0,
```

Add CLI args:

```python
parser.add_argument("--fifo-token-pair-mode", choices=["any", "same_keep_count"])
parser.add_argument("--token-score-mode", choices=["set_mean", "delta_mean"])
parser.add_argument("--pair-sampling-seed", type=int)
parser.add_argument("--max-pairs-per-event", type=int)
parser.add_argument("--min-loss-gap-by-event-type")
parser.add_argument("--max-loss-gap", type=float)
parser.add_argument("--min-count-loss-gap", type=float)
```

Parse CLI JSON for `--min-loss-gap-by-event-type`. Add `import json` at the top of the file (the current imports do not include it). Then in the CLI override loop:

```python
if key == "min_loss_gap_by_event_type" and isinstance(value, str):
    values[key] = json.loads(value)
    continue
```

Without `import json`, the first CLI invocation of `--min-loss-gap-by-event-type` raises `NameError`. YAML mappings are already dictionaries and do not need this conversion.

- [ ] **Step 3: Add function parameters and forward them**

Add the same parameters to `train_joint_retention(...)` and pass them into:

- `CounterfactualOracleDataset.from_events(...)` for train and validation token datasets.
- `token_oracle_ranking_loss(..., score_mode=token_score_mode)`.
- `FifoCountDataset.from_events(..., min_count_loss_gap=min_count_loss_gap)` for train and validation count datasets.

Also update `main()` so parsed args are not dropped at the CLI-to-function boundary.

Add a CLI smoke test that runs the script through `subprocess.run(...)` on a tiny fake shard with `--epochs 1 --device cpu`, then loads the checkpoint and asserts `training_options` contains the YAML values from Step 1. Keep the fake shard tiny so this test stays fast.

- [ ] **Step 4: Write failing validation checkpoint test**

Run tiny CPU training with `val_fraction=0.5` and deterministic split.

Expected checkpoint fields:

```python
checkpoint["validation_metrics"]["token"]["rank_acc"]
checkpoint["validation_metrics"]["token"]["per_event_type"]
checkpoint["validation_metrics"]["count"]["accuracy"]
checkpoint["validation_metrics"]["count"]["majority_accuracy"]
checkpoint["dataset_stats"]["train_token"]
checkpoint["dataset_stats"]["val_token"]
checkpoint["dataset_stats"]["train_count"]
checkpoint["dataset_stats"]["val_count"]
checkpoint["training_options"]
```

Expected: FAIL before validation implementation.

- [ ] **Step 5: Build real validation datasets**

Change:

```python
train_events, _ = split_oracle_events(...)
```

to:

```python
train_events, val_events = split_oracle_events(...)
```

If `val_fraction == 0` or no validation samples exist, save empty metrics with `count=0`.

- [ ] **Step 6: Implement `_evaluate_token_ranking`**

Required output:

- `count`
- `loss`
- `pairwise`
- `regression`
- `rank_acc`
- `mean_score_diff`
- `per_event_type`

When computing per-sample correctness, use the same `token_score_mode` as training.

- [ ] **Step 7: Implement `_evaluate_count_head`**

Required output:

- `count`
- `loss`
- `accuracy`
- `mean_abs_count_error`
- `majority_accuracy`
- `target_distribution`
- `prediction_distribution`

Use checkpoint key `majority_accuracy`; log output may use `majority_acc`.

- [ ] **Step 8: Print and save summaries**

Print compact startup summaries:

```text
token_dataset train {...}
token_dataset val {...}
count_dataset train {...}
count_dataset val {...}
```

Save:

```python
"validation_metrics": {"token": token_validation_metrics, "count": count_validation_metrics},
"dataset_stats": dataset_stats,
"training_options": training_options,
```

- [ ] **Step 9: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/train_joint_retention_policy.py tests/test_train_joint_retention_policy.py
git commit -m "feat: add joint retention validation and config wiring"
```

---

### Task 5: Add Best Checkpoint, Early Stopping, And CountHead Deploy Gating

**Files:**

- Modify: `/path/to/mount/lyj/voxel-vggt/src/train_joint_retention_policy.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`

- [ ] **Step 1: Write failing tests for best checkpoint metadata**

Call `train_joint_retention(...)` with:

```python
save_best=True
best_metric="token.rank_acc"
early_stop_patience=None
```

Expected:

- Main output exists.
- Sibling `.best.pt` exists.
- Both checkpoints include `best_metric`, `best_metric_value`, `best_epoch`, `final_epoch`, `has_validation`, and `best_selection_reason`.
- `.best.pt` includes the validation metrics from the selected best epoch.
- Final checkpoint includes `best_validation_metrics` with the selected best epoch snapshot.

Also test `val_fraction=0.0`:

- `.best.pt` exists.
- `best_metric_value is None`.
- `has_validation is False`.
- `best_selection_reason == "no_validation"`.
- Early stopping is disabled.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 2: Add parser and function options**

Add defaults:

```python
"save_best": False,
"best_metric": "token.rank_acc",
"best_output": None,
"early_stop_patience": None,
"early_stop_min_delta": 0.0,
"deploy_count_head": "auto",
"deploy_count_head_min_delta": 0.0,
```

Add CLI:

```python
parser.add_argument("--save-best", action=argparse.BooleanOptionalAction)
parser.add_argument("--best-metric", choices=[
    "token.rank_acc",
    "token.eviction_rank_acc",
    "token.fifo_topk_rank_acc",
    "token.eviction_fifo_mean_rank_acc",
    "count.accuracy",
])
parser.add_argument("--best-output")
parser.add_argument("--early-stop-patience", type=int)
parser.add_argument("--early-stop-min-delta", type=float)
parser.add_argument("--deploy-count-head", choices=["auto", "always", "never"])
parser.add_argument("--deploy-count-head-min-delta", type=float)
```

Add matching `train_joint_retention(...)` parameters and update `main()`.

- [ ] **Step 3: Implement metric selection helper**

Add:

```python
def _select_best_metric(token_metrics: dict, count_metrics: dict, metric_name: str) -> float:
    ...
```

Rules:

- `token.rank_acc`: `token_metrics["rank_acc"]`
- `token.eviction_rank_acc`: `token_metrics["per_event_type"]["eviction"]["rank_acc"]`
- `token.fifo_topk_rank_acc`: `token_metrics["per_event_type"]["fifo_topk"]["rank_acc"]`
- `token.eviction_fifo_mean_rank_acc`: average available eviction and fifo values
- `count.accuracy`: `count_metrics["accuracy"]`

Missing metrics return `float("-inf")` and print a warning.

- [ ] **Step 4: Implement shared checkpoint save helper**

Extract checkpoint export into:

```python
def _save_joint_checkpoint(..., checkpoint_role: str, validation_metrics: dict, best_info: dict) -> None:
    ...
```

Save existing schema plus:

```python
"checkpoint_role": checkpoint_role,
"validation_metrics": validation_metrics,
"best_validation_metrics": best_validation_metrics,
"best_metric": best_metric,
"best_metric_value": best_metric_value,
"best_epoch": best_epoch,
"final_epoch": final_epoch,
"has_validation": has_validation,
"best_selection_reason": best_selection_reason,
```

- [ ] **Step 5: Track best validation metrics**

Maintain:

```python
best_validation_metrics = {"token": {}, "count": {}}
best_value = float("-inf")
best_epoch = None
epochs_without_improvement = 0
```

On improvement, deep-copy the current token/count validation metrics into `best_validation_metrics` and save `.best.pt` if `save_best=True`.

- [ ] **Step 6: Implement early stopping**

At epoch end:

```python
if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
    print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_metric_value={best_value:.6f}", flush=True)
    break
```

- [ ] **Step 7: Write failing tests for CountHead deploy gating**

Unit-test:

```python
_should_deploy_count_head(
    count_head_trained=True,
    count_metrics={"accuracy": 0.30, "majority_accuracy": 0.40},
    deploy_count_head="auto",
    min_delta=0.0,
) is False
```

Checkpoint behavior:

- `count_head_trained` remains `True`.
- Top-level `count_head` remains present for debugging/resume.
- `count_head_deploy_enabled` is `False`.
- deploy `model` contains no `aggregator.count_head.*` keys.
- With `deploy_count_head="always"`, deploy keys are present.
- With no validation and `deploy_count_head="auto"`, deploy is disabled.

Expected: FAIL before implementation.

- [ ] **Step 8: Implement deploy decision helper**

Add:

```python
def _should_deploy_count_head(
    count_head_trained: bool,
    count_metrics: dict,
    deploy_count_head: str,
    min_delta: float,
) -> bool:
    if not count_head_trained:
        return False
    if deploy_count_head == "always":
        return True
    if deploy_count_head == "never":
        return False
    acc = count_metrics.get("accuracy")
    majority = count_metrics.get("majority_accuracy")
    if acc is None or majority is None:
        return False
    return float(acc) >= float(majority) + float(min_delta)
```

Call this inside `_save_joint_checkpoint(...)`, not only in the final checkpoint path. For `.best.pt`, use that checkpoint's validation metrics.

- [ ] **Step 9: Verify focused tests**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/train_joint_retention_policy.py tests/test_train_joint_retention_policy.py
git commit -m "feat: save best joint retention checkpoint"
```

---

### Task 6: Add Oracle Signal Diagnostic Tool

**Files:**

- Create: `/path/to/mount/lyj/voxel-vggt/tools/analyze_oracle_training_signal.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write failing tests for diagnostic helpers**

Add tests for the diagnostic helpers. Because `tools/` is not a Python package, add the tools directory to `sys.path` at the top of the test before importing:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from analyze_oracle_training_signal import (
    summarize_count_confidence,
    summarize_sequence_distribution,
    summarize_threshold_sweep,
)
```

Test:

```python
summarize_threshold_sweep(events, thresholds, fifo_token_pair_mode, max_pairs_per_event, pair_sampling_seed)
summarize_sequence_distribution(events)
summarize_count_confidence(events, count_candidates, label_reduction)
```

Expected threshold keys include `0.01`, `0.02`, `0.03`, `0.05`.

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -q
```

Expected: FAIL before tool/helper implementation.

- [ ] **Step 2: Implement read-only CLI**

CLI:

```bash
env PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python tools/analyze_oracle_training_signal.py \
  --oracle-shards <paths...> \
  --thresholds 0.01 0.02 0.03 0.05 \
  --fifo-token-pair-mode same_keep_count \
  --pair-sampling-seed 0 \
  --max-pairs-per-event 64 \
  --count-candidates 0 8 16 32 64 128
```

Output must include exact section labels:

- `EVENT_COUNTS`
- `THRESHOLD_SWEEP threshold=<value> total_pairs=<n> by_event_type=<dict> margin_p50=<value> usable_events=<dict>`
- `SEQUENCE_DISTRIBUTION total_sequences=<n> events_per_sequence_p10=<value> events_per_sequence_p50=<value> events_per_sequence_p90=<value>`
- `COUNT_CONFIDENCE p10=<value> p50=<value> p90=<value> above_thresholds=<dict>`
- `COUNT_LABEL_DISTRIBUTION target_keep_count=<dict> majority_accuracy=<value>`
- `SCORE_PROJECTION_STATE shard=<path> has_projection=<bool> key_count=<n>`

- [ ] **Step 3: Run real diagnostic on Phase 2 shards**

Run:

```bash
env PYTHONPATH=src \
  /mnt/lyj/miniconda3/envs/streamvggt/bin/python tools/analyze_oracle_training_signal.py \
  --oracle-shards \
    checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu0_s0.pt \
    checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu1_s1.pt \
    checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu2_s2.pt \
    checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu3_s3.pt \
  --thresholds 0.01 0.02 0.03 0.05 \
  --fifo-token-pair-mode same_keep_count \
  --pair-sampling-seed 0 \
  --max-pairs-per-event 64 \
  --count-candidates 0 8 16 32 64 128
```

Expected:

- Every threshold line is printed.
- `fifo_cross_keep_frac=0.000` appears in threshold summaries.
- Count confidence distribution is printed.
- Projection-state availability is printed for every shard.

- [ ] **Step 4: Record threshold decision**

Create a short note in the plan execution log or final response with:

```text
chosen_min_loss_gap_by_event_type:
  dedup: ?
  eviction: ?
  fifo_topk: ?
chosen_min_count_loss_gap: ?
reason: sample counts and p50/p90 confidence from diagnostics
```

Minimum sample criteria before training high-confidence config:

- Train token samples per event type: at least `500`.
- Validation token samples per event type: at least `100`.
- Train count samples: at least `500`.
- Validation count samples: at least `50`.

If diagnostics do not meet these minima, lower the threshold or collect more sequence-balanced oracle data before training. Do not train a high-confidence config that silently leaves an event type under-sampled.

- [ ] **Step 5: Commit**

```bash
git add tools/analyze_oracle_training_signal.py tests/test_token_oracle_dataset.py
git commit -m "tools: add oracle signal diagnostics"
```

**Threshold Decision (from Phase 2 diagnostics):**

```text
chosen_min_loss_gap_by_event_type:
  dedup: 0.02
  eviction: 0.01
  fifo_topk: 0.01
chosen_min_count_loss_gap: 0.001
reason: |
  Phase 2 diagnostics: 8192 events (3340 dedup, 756 eviction, 4096 fifo_topk)
  across 501 sequences, 48 score_state_projs keys per shard.

  Token pair samples at threshold 0.01:
    total_pairs=23725 (dedup=12978, fifo_topk=4062, eviction=6685)
    unique_events: dedup=1474, fifo_topk=460, eviction=540
  At 90/10 split, train events:
    dedup=1327 (>=500 PASS), eviction=486 (~500), fifo_topk=414 (<500)
  eviction barely meets 500 at threshold 0.01; fifo_topk is under 500.
  Using 0.01 for eviction and fifo_topk, 0.02 for dedup (which has ample samples).

  Count confidence:
    p10=0.000029, p50=0.000672, p90=0.007482
    1714 above 0.001 threshold -> ~1543 train, ~171 val (both pass minima)
    majority_accuracy=0.3350 (label distribution is spread, not collapsed)
```

---

### Task 7: Add Small-Data Overfit Sanity Tool

**Files:**

- Create: `/path/to/mount/lyj/voxel-vggt/tools/overfit_joint_retention_sanity.py`
- Test: `/path/to/mount/lyj/voxel-vggt/tests/test_train_joint_retention_policy.py`

- [ ] **Step 1: Write failing helper and CLI tests**

Add tests that import helpers by adding `tools/` to `sys.path`.

Required helper behavior:

- `select_high_margin_token_samples(samples, n)` returns highest margins in descending order.
- A `main(argv)` smoke test using fake data prints all required markers and returns exit code `0` when thresholds pass.
- A failing fake-data run returns exit code `2`.

Required output markers:

- `TOKEN_DATA`
- `TOKEN_OVERFIT`
- `TOKEN_DONE`
- `COUNT_DATA`
- `COUNT_OVERFIT`
- `COUNT_DONE`
- `SANITY_RESULT token_pass=<bool> count_pass=<bool>`

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_train_joint_retention_policy.py -q
```

Expected: FAIL before tool exists.

- [ ] **Step 2: Implement tool**

CLI:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /mnt/lyj/miniconda3/envs/streamvggt/bin/python tools/overfit_joint_retention_sanity.py \
  --oracle-shards <paths...> \
  --token-samples 128 \
  --count-samples-per-class 8 \
  --min-loss-gap 0.03 \
  --device cuda \
  --max-token-steps 150 \
  --max-count-steps 400
```

Default pass thresholds:

- token rank accuracy `>= 0.98`
- count accuracy `>= 0.90`

Exit code:

- `0` if both pass.
- `2` if either fails.

- [ ] **Step 3: Run real-shard sanity check**

Run:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /mnt/lyj/miniconda3/envs/streamvggt/bin/python tools/overfit_joint_retention_sanity.py \
  --oracle-shards checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu0_s0.pt \
  --token-samples 128 \
  --count-samples-per-class 8 \
  --min-loss-gap 0.03 \
  --device cuda
```

Expected:

- `SANITY_RESULT token_pass=True count_pass=True`

If CUDA is unavailable, run with `--device cpu` and report slower runtime.

- [ ] **Step 4: Commit**

```bash
git add tools/overfit_joint_retention_sanity.py tests/test_train_joint_retention_policy.py
git commit -m "tools: add joint retention overfit sanity check"
```

---

### Task 8: Add Phase 2 Baseline And High-Confidence Configs

**Files:**

- Create: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2.yaml`
- Create: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2_highconf.yaml`
- Create: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy_phase2_capacity_ablation.yaml`
- Optionally modify: `/path/to/mount/lyj/voxel-vggt/config/train_joint_retention_policy.yaml`

- [ ] **Step 1: Create baseline Phase 2 config**

Create `config/train_joint_retention_policy_phase2.yaml`:

```yaml
# Baseline joint TokenScorer + FifoCountHead training on Phase 2 oracle shards.

oracle_shards:
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu0_s0.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu1_s1.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu2_s2.pt
  - /path/to/mount/lyj/voxel-vggt/checkpoints/token_oracle_phase2/phase2_diverse_shard_gpu3_s3.pt

output: /path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_baseline.pt
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
min_loss_gap_by_event_type: null
max_loss_gap: null
min_count_loss_gap: 0.0
count_loss_weight: 0.5
count_label_reduction: min
count_repeat_factor: 1

fifo_token_pair_mode: same_keep_count
token_score_mode: delta_mean
pair_sampling_seed: 0
max_pairs_per_event: 64

save_best: true
best_metric: token.eviction_fifo_mean_rank_acc
early_stop_patience: 3
early_stop_min_delta: 0.001
deploy_count_head: auto
deploy_count_head_min_delta: 0.03

val_fraction: 0.1
split_key: sequence_id
split_seed: 0
device: cuda
```

- [ ] **Step 2: Create full high-confidence config from diagnostic decision**

Use the thresholds chosen in Task 6 Step 4. Create `config/train_joint_retention_policy_phase2_highconf.yaml` as a complete, directly runnable YAML file. Do not write it as a partial override, because the current config loader does not support inheritance or includes.

Start by copying every field from `config/train_joint_retention_policy_phase2.yaml`, then change these fields if the diagnostic minima pass with the candidate values:

```yaml
output: /path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_highconf.pt
min_loss_gap: 0.03
min_loss_gap_by_event_type:
  dedup: 0.03
  eviction: 0.02
  fifo_topk: 0.03
min_count_loss_gap: 0.02
early_stop_patience: 2
```

All other fields should match the baseline config unless diagnostics justify a different threshold. If diagnostics show under-sampling, lower only the threshold(s) that fail the sample criteria and document the chosen values.

- [ ] **Step 3: Create capacity ablation config**

Create `config/train_joint_retention_policy_phase2_capacity_ablation.yaml` as another complete, directly runnable YAML file by copying the full high-confidence config and changing:

```yaml
output: /path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_highconf_h512.pt
hidden_dim: 512
```

Run this only if the high-confidence `hidden_dim=256` baseline has low train accuracy as well as weak validation. If train accuracy is high but validation is weak, collect better data instead of increasing capacity.

- [ ] **Step 4: Add parser preflight that compares values, not just attributes**

Run:

```bash
env PYTHONPATH=src \
  /mnt/lyj/miniconda3/envs/streamvggt/bin/python -c "
import inspect
from omegaconf import OmegaConf
from train_joint_retention_policy import parse_args, train_joint_retention
cfg_path = 'config/train_joint_retention_policy_phase2_highconf.yaml'
cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
args = parse_args(['--config', cfg_path])
sig = inspect.signature(train_joint_retention)
missing_from_signature = sorted(k for k in cfg if k not in sig.parameters)
missing_from_args = sorted(k for k in cfg if not hasattr(args, k))
mismatched = {
    k: (cfg[k], getattr(args, k, '<missing>'))
    for k in cfg
    if getattr(args, k, '<missing>') != cfg[k]
}
print('missing_from_signature:', missing_from_signature)
print('missing_from_args:', missing_from_args)
print('mismatched:', mismatched)
assert not missing_from_signature
assert not missing_from_args
assert not mismatched
assert args.deploy_count_head == 'auto'
assert args.deploy_count_head_min_delta == 0.03
"
```

Expected:

```text
missing_from_signature: []
missing_from_args: []
mismatched: {}
```

If any mismatch appears, fix parser defaults or `main()` forwarding before training. Do not filter unknown keys with `inspect.signature` for production training; that masks broken wiring.
The threshold values themselves may differ from the example defaults if Task 6 diagnostics justify different choices; this preflight only requires the YAML values and parsed values to match exactly.

- [ ] **Step 5: Commit**

```bash
git add \
  config/train_joint_retention_policy_phase2.yaml \
  config/train_joint_retention_policy_phase2_highconf.yaml \
  config/train_joint_retention_policy_phase2_capacity_ablation.yaml
git commit -m "config: add phase2 joint retention configs"
```

---

### Task 9: Run Controlled Training Matrix

**Files:**

- Outputs under `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/`
- Optional results note under `/path/to/mount/lyj/voxel-vggt/docs/superpowers/results/`

- [ ] **Step 1: Run final preflight**

Run the parser preflight from Task 8 Step 4 for both:

- `config/train_joint_retention_policy_phase2.yaml`
- `config/train_joint_retention_policy_phase2_highconf.yaml`

Expected: no missing or mismatched keys.

- [ ] **Step 2: Run high-confidence baseline**

Run:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /mnt/lyj/miniconda3/envs/streamvggt/bin/python src/train_joint_retention_policy.py \
  --config config/train_joint_retention_policy_phase2_highconf.yaml \
  --device cuda
```

Expected:

- Best checkpoint saved at `/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_highconf.best.pt`.
- Training may stop before `epochs` due to early stopping.

- [ ] **Step 3: Inspect best checkpoint metrics**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -c "import torch, pprint; p='/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_highconf.best.pt'; s=torch.load(p,map_location='cpu',weights_only=False); print(s['best_metric'], s['best_metric_value'], s['best_epoch']); pprint.pp(s['validation_metrics']); print('count_head_deploy_enabled', s.get('count_head_deploy_enabled'))"
```

Minimum target:

- Best token validation beats old single-head `eviction=0.5121`.
- Best token validation does not regress `fifo_topk` below `0.5168`.
- CountHead deploy is enabled only if `accuracy >= majority_accuracy + 0.03`.

Strong target:

- Overall token rank accuracy `>= 0.56`.
- `eviction >= 0.57`.
- `fifo_topk >= 0.55`.
- CountHead beats majority by at least `+0.03`.

- [ ] **Step 4: Run capacity ablation only if justified**

Run `hidden_dim=512` only if:

- high-confidence `hidden_dim=256` train metrics remain low, and
- overfit sanity passed, and
- validation is not already overfit-limited.

Skip capacity ablation if train improves while validation stays weak; that points to data quality or data volume.

- [ ] **Step 5: Compare results**

Record:

```text
run                         best_epoch  best_metric  overall  dedup  eviction  fifo_topk  count_acc  majority_acc  deploy_count
single_head_phase2_final     9           -            0.5413   0.5869 0.5121    0.5168     -          -             -
joint_existing_artifact      -           -            0.5189   0.5245 0.5549    0.4225     0.3235     0.3407        unknown
joint_phase2_baseline        ?           ?            ?        ?      ?         ?          ?          ?             ?
joint_highconf               ?           ?            ?        ?      ?         ?          ?          ?             ?
joint_highconf_h512          ?           ?            ?        ?      ?         ?          ?          ?             ?
```

- [ ] **Step 6: Decide whether to collect more oracle data**

Collect more data only if:

- high-confidence filtering leaves too few validation samples, or
- high-confidence train metrics are strong but validation remains near random, or
- CountHead remains below majority after confidence filtering.

If collecting, prioritize:

- more `fifo_topk` events with high best-vs-second-best count loss gap,
- more eviction events from under-represented sequences,
- sequence-balanced sampling instead of simply increasing pair count.

---

### Task 10: Final Verification

**Files:**

- All modified files above.

- [ ] **Step 1: Run focused tests**

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest \
  tests/test_token_oracle_dataset.py \
  tests/test_fifo_count_dataset.py \
  tests/test_token_scorer_counterfactual.py \
  tests/test_train_joint_retention_policy.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run broader tests if time permits**

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests -q
```

Expected: PASS or report unrelated pre-existing failures.

- [ ] **Step 3: Verify deploy checkpoint keys**

Run:

```bash
/mnt/lyj/miniconda3/envs/streamvggt/bin/python -c "import torch; p='/path/to/mount/lyj/voxel-vggt/checkpoints/token_scorer_oracle/joint_retention_policy_phase2_highconf.best.pt'; s=torch.load(p,map_location='cpu',weights_only=False); print('score_state_projs', sum(k.startswith('aggregator.score_state_projs.') for k in s['model'])); print('token_scorers', sum(k.startswith('aggregator.token_scorers.') for k in s['model'])); print('count_head', sum(k.startswith('aggregator.count_head.') for k in s['model'])); print('count_head_deploy_enabled', s.get('count_head_deploy_enabled'))"
```

Expected:

- `token_scorers`: nonzero.
- `score_state_projs`: nonzero if oracle shards or checkpoint provide projection state.
- `count_head`: nonzero only when `count_head_deploy_enabled=True`; otherwise `0`.

- [ ] **Step 4: Save results notes separately**

If adding written results, create a separate note under:

```text
docs/superpowers/results/
```

Do not treat this implementation plan as the mutable training results log.

---

## Success Criteria

Minimum acceptable outcome:

- Current source tree can reproduce joint validation metrics without relying on external worktree code.
- Config preflight proves YAML values are parsed, forwarded, and not silently dropped.
- Overfit sanity tool passes on at least one real shard.
- High-confidence training saves `.best.pt`.
- Best checkpoint beats old single-head `eviction=0.5121` and does not regress below old single-head `fifo_topk=0.5168`.
- CountHead either beats majority baseline by configured margin or is explicitly disabled from deploy while retaining trained weights for debugging.

Strong outcome:

- Overall token rank accuracy `>= 0.56`.
- `eviction >= 0.57`.
- `fifo_topk >= 0.55`.
- CountHead validation accuracy beats majority baseline by at least `+0.03`.

Decision rule:

- If high-confidence filtering improves validation, optimize collection around high-margin and sequence-balanced events.
- If high-confidence filtering does not improve validation but overfit sanity passes, collect more sequence-balanced high-margin oracle data.
- If high-confidence overfit sanity fails, investigate implementation and architecture before collecting more data.
