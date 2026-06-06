# Oracle Collection Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up counterfactual oracle data collection without changing the oracle loss definition, candidate labels, pairwise ranking samples, or training objective.

**Architecture:** Keep `measure_counterfactual_event()` as the semantic source of truth, then remove redundant replay work around it. The main bottleneck is repeated frontend replay of the same short prefixes, especially for dedup events; optimizations must be gated by exact/near-exact shard equivalence tests against the current collector on fixed seeds.

**Tech Stack:** Python, PyTorch, OVGGT frontend cache, existing `tools/collect_counterfactual_oracle.py`, `src/ovggt/training/frontend_oracle_collector.py`, and pytest.

---

## Evidence From Current Run

Source log: `checkpoints/token_oracle/collect_shard000_gpu5_v3.log`.

- Startup/model/dataloader overhead is not the dominant cost. Collection starts at 16:30:09, after config/model/dataloader setup.
- Probe and teacher are cheap relative to replay. For the first 24-frame sequence, probe took about 12s and high-budget teacher took about 4s.
- Replay dominates. By 17:11:42, the collector was still on batch 1, event 28/64.
- Current run: `subset_replay_batch_size=32`, `layers_per_frame=4`, `max_events_per_sequence=64`.
- The first 28 events average 68.1 candidate subsets each, with max 143.
- Replay batches average about 38.5s per logged replay batch in v3.
- Comparing v2 (`subset_replay_batch_size=8`) and v3 (`32`), batching reduces Python/model-call overhead, but the frontend path still scales close to linearly because `_inference_frontend()` runs aggregator and camera head in `for b in range(B)` loops.
- Current partial shard contains 16 events, all `dedup`, from frames 0 and 1 only, already 20MB. The training dataset only consumes `score_state`, `metadata_features`, `keep_indices`, `loss`, and `loss_components`; replay predictions/targets are not used by `CounterfactualOracleDataset`.

## Non-Negotiable Constraints

- Do not change `TASK_WEIGHTS`, `compute_three_task_loss_components()`, subset loss computation, or pairwise ranking construction.
- Do not drop candidate subsets, event types, frames, layers, or datasets as a default optimization. Those are sampling policy changes and can affect training results.
- Every code optimization must have a deterministic equivalence test against the current collector on a small fixed-seed sequence.
- Acceptable numeric tolerance: exact subset IDs/keep masks and event ordering; loss differences only within normal floating-point tolerance, e.g. `atol=1e-5`, `rtol=1e-4`, unless a stricter project precedent exists.

---

### Task 1: Add Replay Timing Instrumentation

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [x] Add per-event timing around probe, teacher, replay, target building, loss building, and shard flush.
- [x] Add per-replay timing fields to logs: `event_id`, `event_type`, `frame_id`, `layer_id`, `num_subsets`, `chunk_size`, `stop`, `elapsed_sec`, and GPU memory allocated/reserved when CUDA is available.
- [x] Keep this instrumentation side-effect-free and disabled only by log level/config if needed.
- [x] Run `pytest tests/test_frontend_oracle_collector.py -v`.

### Task 2: Do Not Store Replay Predictions/Targets In Production Shards

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Modify: `config/collect_counterfactual_oracle.yaml`
- Test: `tests/test_frontend_oracle_collector.py`
- Test: `tests/test_token_oracle_dataset.py`

- [x] Add `store_replay_payload: bool = false` to collector config/CLI.
- [x] In measured subsets, always store `keep_indices`, `loss`, and `loss_components`.
- [x] Store `replay.predictions` and `replay.targets` only when `store_replay_payload=true`.
- [x] Add a dataset test proving `CounterfactualOracleDataset` produces identical samples from shards with and without replay payload.
- [x] Run `pytest tests/test_frontend_oracle_collector.py tests/test_token_oracle_dataset.py -v`.

Expected impact: lower CPU RAM, less `torch.save()` time, smaller partial shards. This does not change training because the training loader ignores replay tensors.

### Task 3: Tune Existing Runtime Parameters Safely

**Files:**
- Modify: `config/collect_counterfactual_oracle.yaml`
- Optional: add `tools/benchmark_oracle_replay_batch_size.py`
- Test: none required for config-only benchmark, but preserve collector tests.

- [x] Benchmark `subset_replay_batch_size` on one fixed sequence with values `8, 16, 24, 32`.
- [x] Use v3 evidence as baseline: batch 32 averages about 38.5s per replay chunk and uses about 50GB on GPU 5.
- [x] Pick the value with best events/hour, not largest batch size. On the current code, `16` or `24` may be better if `32` hits a throughput knee.
- [x] Increase `log_every_subsets` only if log I/O becomes visible in instrumentation; it is not the current bottleneck.
- [x] Keep `num_samples=8`, `oracle_window=4`, `layers_per_frame=4`, and `max_events_per_sequence=64` unchanged unless intentionally changing data coverage.

Expected impact: modest. This is the fastest no-code or low-code optimization, but it will not solve the main repeated-prefix cost.

### Task 4: Cache And Reuse Event Prefix State

**Files:**
- Modify: `src/ovggt/models/ovggt.py`
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] Add an internal frontend replay API that can run frames up to an event boundary, snapshot cache/keyframe/camera state, then replay only future frames after applying a keep set.
- [ ] Snapshot all state needed to reproduce current full-prefix behavior: `cache_states`, keyframe managers, `past_key_values_camera`, `aggregator.last_scores`, attention anchor counts, current query points, and keyframe schedule state.
- [ ] For a fixed event, compare current full-prefix `frames[:stop]` replay against prefix-snapshot replay for every subset.
- [ ] Assert identical event metadata and keep masks; assert losses match within tolerance.
- [ ] Run the equivalence test on eviction, dedup, and fifo events. If only dedup is present in a fixture, create synthetic fixtures for the other two.

Expected impact: high. Current replay recomputes frames before the event for every event and subset chunk. Prefix-state reuse should reduce repeated frame 0/1/2/... work while preserving output semantics.

### Task 5: Group Events By Same Prefix Boundary

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] After Task 4, group candidate events by `(frame_id, stop, event_type-compatible replay path)` within a sequence.
- [ ] Build the prefix snapshot once per `frame_id` group.
- [ ] Replay each event from the shared prefix snapshot, applying only its event-specific keep set.
- [ ] Preserve output event order exactly as today.
- [ ] Add a test comparing grouped-prefix collection to current event-by-event collection.

Expected impact: high. In the v3 log, the first 12 events all use `frame=0, stop=5`; the next group repeats `frame=1, stop=6`. Grouping avoids rebuilding the same prefix state many times.

### Task 6: Vectorize Batched Replay For Real

**Files:**
- Modify: `src/ovggt/models/ovggt.py`
- Modify: `src/ovggt/utils/frontend_cache.py`
- Test: `tests/test_frontend_oracle_collector.py`
- Test: `tests/test_frontend_batch_training.py`
- Test: `tests/test_frontend_inference_smoke.py`

- [ ] Replace the replay-only `for b in range(B)` aggregator loop with a batched path where cache states are stacked across subset candidates.
- [ ] Do the same for camera head state only if prefix reuse still leaves camera as a bottleneck.
- [ ] Keep the existing sequential path as fallback for normal training/inference until batched cache equivalence is proven.
- [ ] Add equivalence tests: batched replay with N subsets must match N serial replays in losses and keep masks.
- [ ] Run focused frontend tests plus oracle collector tests.

Expected impact: high but higher risk. Current `subset_replay_batch_size=32` is not true aggregator batching because `_inference_frontend()` still loops over each batch element for the expensive path.

### Task 7: Production Multi-GPU Scheduling

**Files:**
- Modify: `tools/collect_counterfactual_oracle_parallel.py`
- Optional create: `tools/summarize_oracle_collection_logs.py`
- Test: `tests/test_frontend_oracle_parallel.py`

- [x] Add a resume/skip mode: if a shard exists and `partial=False`, skip it; if `partial=True`, write a new shard id unless explicit overwrite is requested.
- [x] Include `CUDA_VISIBLE_DEVICES`, shard id, seed, and config hash in each log.
- [x] Add a log summarizer that reports events/hour, replay seconds/event, subsets/event, and last event progress per shard.
- [x] Keep one collector process per GPU by default. Multiple processes per GPU should be opt-in only after memory/timing evidence.
- [x] Run `pytest tests/test_frontend_oracle_parallel.py -v`.

Expected impact: better cluster utilization and safer restarts. This does not change per-shard training samples.

---

## Recommended Execution Order

1. Run Task 1 first so every later change has timing evidence.
2. Do Task 2 immediately after; it is low risk and should reduce shard I/O.
3. Benchmark Task 3 to choose `subset_replay_batch_size` for current code.
4. Implement Tasks 4 and 5 together as the main high-impact, semantics-preserving optimization.
5. Attempt Task 6 only if prefix reuse still leaves GPU utilization low or events/hour unacceptable.
6. Use Task 7 to run production collection across GPUs once one-shard equivalence is verified.

## Verification Gate Before Production

- [ ] Fixed-seed tiny shard from old collector and optimized collector has identical event count, event order, event ids, layer/frame ids, and keep masks.
- [ ] Per-subset losses match within tolerance.
- [ ] `CounterfactualOracleDataset` sample count and first N samples match exactly for the same shard.
- [ ] One production dry run logs improved events/hour on the same GPU and config.
- [ ] No changes to `train_token_scorer_oracle.py` loss behavior.
