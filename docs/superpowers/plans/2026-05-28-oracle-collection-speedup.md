# Oracle Collection Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce counterfactual oracle collection time without changing the oracle loss definition, and add a multi-GPU shard collection launcher.

**Architecture:** Keep the existing collector as the single-event source of truth, but batch candidate subset replays for the same event, deduplicate identical keep sets, and optionally stratify layer sampling. Add a separate launcher script that starts one collector process per GPU and writes independent shards/logs.

**Tech Stack:** Python, PyTorch, argparse, existing OVGGT frontend oracle collector and pytest tests.

---

### Task 1: Collector Replay Optimizations

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] Add tests for deduplicating repeated candidate keep sets.
- [ ] Add tests for batched replay applying a different keep set per replicated batch item.
- [ ] Implement `deduplicate_keep_subsets`, `MultiReplayKeepSetProbe`, frame replication, and batched event measurement.
- [ ] Run `pytest tests/test_frontend_oracle_collector.py -v`.

### Task 2: Stratified Event Sampling

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Modify: `config/collect_counterfactual_oracle.yaml`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] Add tests for rotating layer selection, e.g. 4 layers per 24-layer frame.
- [ ] Add config/CLI fields for `layers_per_frame`, `max_events_per_sequence`, and `subset_replay_batch_size`.
- [ ] Wire fields into `CounterfactualEvictionProbe` and sequence collection.
- [ ] Run `pytest tests/test_frontend_oracle_collector.py -v`.

### Task 3: Multi-GPU Collection Launcher

**Files:**
- Create: `tools/collect_counterfactual_oracle_parallel.py`
- Test: `tests/test_frontend_oracle_parallel.py`

- [ ] Add tests that dry-run command generation maps shard ids to GPU ids, seeds, outputs, and logs.
- [ ] Implement launcher using `subprocess.Popen` with per-process `CUDA_VISIBLE_DEVICES`.
- [ ] Ensure failures return a non-zero exit code and print the failed shard command.
- [ ] Run `pytest tests/test_frontend_oracle_parallel.py -v`.

### Task 4: Verification

**Files:**
- Test: `tests/test_frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_parallel.py`

- [ ] Run collector tests.
- [ ] Run parallel launcher tests.
- [ ] Run a tiny CPU synthetic/dummy smoke path where feasible.
