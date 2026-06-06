# TokenScorer Oracle Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the exact input sequence identity for every TokenScorer oracle event so training samples can be traced back to the collected image sequence.

**Architecture:** Attach a compact `sequence_provenance` payload at oracle collection time, keyed by batch index, and carry it through replay dumps, oracle shards, dataset samples, and training batches. Keep the payload textual and lightweight: dataset name plus per-frame labels/instances and a derived sequence key. Do not change scorer math or eviction semantics.

**Tech Stack:** Python, PyTorch, existing `dust3r` dataset metadata, `pytest`.

---

### Task 1: Add provenance coverage tests

**Files:**
- Modify: `tests/test_frontend_oracle_collector.py`
- Modify: `tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sequence_provenance_is_recorded_per_batch():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_frontend_oracle_collector.py tests/test_token_oracle_dataset.py -v`

- [ ] **Step 3: Write minimal implementation**

Add provenance plumbing in collector and dataset code until the tests pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_frontend_oracle_collector.py tests/test_token_oracle_dataset.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_frontend_oracle_collector.py tests/test_token_oracle_dataset.py
git commit -m "test: cover oracle provenance tracking"
```

### Task 2: Collect and persist sequence provenance

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `src/ovggt/training/counterfactual_replay.py`
- Modify: `tools/generate_counterfactual_oracle.py`

- [ ] **Step 1: Write the failing test**

The tests from Task 1 should fail until provenance is attached and preserved.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_frontend_oracle_collector.py -v`

- [ ] **Step 3: Write minimal implementation**

Build `sequence_provenance` from the collated frontend sequence, attach it to each oracle event, and preserve it through replay conversion and synthetic generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_frontend_oracle_collector.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py src/ovggt/training/counterfactual_replay.py tools/generate_counterfactual_oracle.py
git commit -m "feat: preserve oracle sequence provenance"
```

### Task 3: Carry provenance into training batches

**Files:**
- Modify: `src/ovggt/training/token_oracle_dataset.py`
- Modify: `src/train_token_scorer_oracle.py`

- [ ] **Step 1: Write the failing test**

Extend the dataset test so collated batches keep `sequence_provenance`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -v`

- [ ] **Step 3: Write minimal implementation**

Keep `sequence_provenance` in each sample, pass it through collate, and surface it in training logs.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/path/to/mount/lyj/voxel-vggt/src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -m pytest tests/test_token_oracle_dataset.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/ovggt/training/token_oracle_dataset.py src/train_token_scorer_oracle.py
git commit -m "feat: preserve oracle provenance in training"
```

