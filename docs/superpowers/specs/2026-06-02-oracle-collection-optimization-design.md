# Oracle Data Collection Speedup Design Spec

## Goal

Speed up counterfactual oracle data collection so that a broad pass over all
36,750 training sequences in `train_frontend_finetune.yaml` finishes within
~1 week on the existing 8-GPU A800 node.

The priority is now sequence/event/scene coverage under a fixed compute budget,
not exhaustive per-event subset enumeration. The shard format and loss
definitions remain fixed, but Phase 1 is explicitly allowed to change the
sampling policy when that produces much broader data coverage. Those sampling
changes must be deterministic, config-controlled, and measured with quality
gates.

The collector today spends ~4.5 hours on a single 24-frame sequence, so a
full single-GPU pass would take ~18,000 GPU-hours. The bottleneck is
counterfactual replay (99.5% of wall time).

## Background And Evidence

Source log: `checkpoints/token_oracle/collect_shard000_gpu5_v3.log`
Collector script: `tools/collect_counterfactual_oracle.py`
Collector module: `src/ovggt/training/frontend_oracle_collector.py`

For the first 24-frame sequence (WildRGBD/stuffed_toy/scene_458):

| Phase | Wall time | Notes |
|---|---|---|
| Frozen student + teacher load | ~30s | One-time, amortized over all sequences |
| Dataloader build | ~130s | One-time |
| Probe forward (student) | 12s | Captures 66 dedup candidates, 0 eviction, 0 fifo |
| High-budget teacher forward | 4s | Runs with `total_budget=10_000_000` |
| Replay loop | ~4.5h | 64 events, average ~70 subsets/event |
| Partial shard write (16 events) | <1s | Negligible |

Replay dominates because:

1. The dedup probe generates N subsets for a voxel group with N tokens
   (`on_dedup_candidate`, lines 343-388). The v3 log shows 12-142 subsets
   per event.
2. Each subset triggers a fresh `_run_frontend_with_probe(model, frames[:stop], ...)`
   forward pass. There is no reuse of the prefix before `event["frame_id"]`.
3. Even with `subset_replay_batch_size=32`, the OVGGT frontend inference
   path contains a `for b in range(B)` loop in `src/ovggt/models/ovggt.py`
   (lines 472, 508, 580, 602, 669, 678), so Python-level batching does not
   vectorize the expensive aggregator path.

The collected partial shard `oracle_shard_000.pt` was used to train
`TokenScorer` via `train_token_scorer_oracle.py`. After 5 epochs /
21,285 steps, rank accuracy on the training pairs was only 55.2%
(random is 50%).

Current shard evidence from the same run:

- 1,088 measured events from 17 sequences.
- All events are `dedup`; `eviction` and `fifo_topk` are absent.
- The shard contains about 274k pairwise ranking samples, so the issue is
  not absolute pair count. The issue is low sequence, temporal, scene, and
  event-type diversity.
- Events are concentrated in frames 0-5, with 64 events per sequence.
- The collector averaged about 121 events/hour on GPU 5, with high variance
  across sequences.

This weak training result supports prioritizing throughput and coverage: it is
better to collect fewer subsets per event across many more sequences than to
fully enumerate highly correlated subsets for a small number of sequences.

## Non-Negotiable Constraints

- The output shard format (`ovggt_counterfactual_oracle_v1`) and the keys
  consumed by `CounterfactualOracleDataset` (`score_state`,
  `metadata_features`, `subsets[].keep_indices`, `subsets[].loss`,
  `subsets[].loss_components`, and `sequence_provenance`) cannot change.
- `TASK_WEIGHTS`, `compute_three_task_loss_components`,
  `weighted_three_task_loss`, and `token_oracle_ranking_loss` are
  immutable. Optimizations must not change the loss values produced for a
  kept subset.
- Pairwise ranking samples are constructed by `CounterfactualOracleDataset`
  from ordered `(better_subset, worse_subset)` pairs where
  `worse_loss - better_loss > min_loss_gap`. The ordering and identity of
  subsets fed into this construction must stay deterministic on a fixed
  seed.
- Event types may not be silently dropped. Any change to which events are
  emitted must be a documented sampling-policy change controlled by
  config, not a side effect of a "performance fix."
- Phase 1 may intentionally reduce subsets and events per sequence to improve
  coverage, but the policy must be visible in `collector_config`, logged at
  collector startup, and summarized per shard: sequences, event types, frames,
  layers, subsets/event, and pair count.

## Design Decisions

### Decision 1: Optimize In Four Phases, With Phase 1 As The Default Path

Phase 1 changes sampling policy parameters to trade exhaustive replay for
coverage. Phase 2 adds cross-event state reuse. Phase 3 is an optional
model-level vectorization. Phase 4 productionizes multi-GPU scheduling.

Rationale: the current training run already has hundreds of thousands of
pairs but only 17 sequences and one event type. The fastest path to a better
TokenScorer is to collect many more sequences first. Phase 1+4 are the
default critical path. Phase 2 is useful if Phase 1+4 still miss the 1-week
target. Phase 3 remains high-risk and out of scope by default.

### Decision 2: Phase 1 Caps Subsets Per Dedup Event

Today, a voxel group with N tokens emits N `keep-one` subsets. With
voxel groups of 100+ tokens in the v3 log, this is the largest single
source of replay work. Phase 1 introduces `max_subsets_per_dedup_event`
(default 8). When N exceeds the cap, the collector samples selected
keep-one choices plus an explicit current-policy baseline:

- The current policy's actual keep set for this voxel group, derived from
  `policy_keep_indices ∩ group_indices` (see instrumentation below)
- Top-K by the actual dedup decision score (highest)
- Top-K by the actual dedup decision score (lowest)
- A uniform sample of the remaining indices

Rationale: keep-one-in-N subsets from the same voxel group are highly
correlated — they all evict N-1 tokens from the same physical region.
Pairwise ranking needs contrastive signal, not exhaustive enumeration.
The sampling strategy preserves the current policy baseline, both score
extremes, and interior coverage. The current-policy baseline counts against
`max_subsets_per_dedup_event`, so the cap remains a hard upper bound. Using
the actual dedup decision score matters: `metadata.importance` is not always
the exact score used by `apply_voxel_dedup_()` (see Instrumentation Change
below).

Note on semantics: for a dedup event at a voxel group with N tokens, most
sampled `keep_indices` sets are `{all tokens outside the voxel group} ∪
{one chosen token inside the group}` — i.e. "keep everything except N-1
tokens in this voxel group," not "keep only one token." The explicit
current-policy baseline may keep more than one token inside the group if the
real policy does so, for example when protected tokens are already in that
voxel. That baseline is still a valid subset because the shard format accepts
arbitrary `keep_indices`.

Tradeoff: the v1 shard format does not record "how many subsets were
sampled vs. enumerated," so downstream consumers cannot tell whether a
subset was a full enumeration or a sample. This is acceptable because
the dataset uses subsets only through their `keep_indices` and `loss`
values.

#### Required Instrumentation Change To `apply_voxel_dedup_()`

The current `apply_voxel_dedup_()` in
`src/ovggt/utils/frontend_cache.py` (lines 543-670) has the order:

1. Compute `scores` (lines 599-612) — either `token_scorer(...)` logits
   or `_composite_candidate_scores_batch(importance, depth_conf, ...)`.
2. **Call `dedup_probe.on_dedup_candidate(...)` (lines 615-621) — but
   the callback receives only `(self, layer_id, frame_id, batch_index)`,
   not the `scores` tensor and not the eventual policy keep set.**
3. Compute keep indices via `_dedup_single_batch(...)` (lines 644+,
   using `scores`).
4. Apply the gather (`self.gather_(kept_indices)`).

This ordering means the probe cannot currently access either:
(a) the actual dedup decision scores used by the policy, or
(b) which tokens the policy will actually keep for the cache.

Phase 1 therefore **requires** restructuring `apply_voxel_dedup_()` to:

1. Compute `scores` (unchanged from current lines 599-612).
2. Compute `policy_keep_indices` via `_dedup_single_batch(...)` **first**
   (hoisted from current lines 644+).
3. Then call `dedup_probe.on_dedup_candidate(...)` with the new
   signature:
   ```python
   def on_dedup_candidate(
       self,
       cache_state,
       layer_id: int,
       frame_id: int,
       batch_index: int = 0,
       scores: Tensor | None = None,            # NEW: [num_tokens]
       policy_keep_indices: Tensor | None = None,  # NEW: [num_kept]
   ) -> None:
       ...
   ```
   `scores` is a 1D tensor of per-token decision scores used by the
   policy (`scores[batch_index]` from step 1). `policy_keep_indices` is
   the full cache keep set the policy is about to apply.
4. Then apply the gather (`self.gather_(policy_keep_indices)`).

The dedup probe must derive the group-local baseline as:

```python
policy_kept_in_group = sorted(set(policy_keep_indices.tolist()) & set(group_indices.tolist()))
```

If `policy_kept_in_group` is empty, skip that voxel group because the probe
cannot retain a current-policy baseline for it. If it contains one token, that
token is reserved as the policy keep-one choice. If it contains multiple
tokens, emit one `policy_baseline` subset whose `keep_indices` equals the full
`policy_keep_indices`; do not coerce the real policy into a single-token choice.
Any top/bottom/random keep-one choices are added only into the remaining cap
slots. The test must cover all three cases: zero, one, and multiple
policy-kept tokens inside a voxel group.

This is a minimal, mechanical restructure: the same `_dedup_single_batch`
call happens, just earlier. The `dedup_replay_probe` path (lines 624-642)
must still run **before** the gather and remains unchanged in
behavior — only the ordering of the probe callback call shifts.

For replay overrides (`dedup_replay_probe`), the keep decision must be
re-derived from the replay keep set, since the policy keep set is irrelevant
when overriding. The replay probe callback signature does not change.

Files changed by this instrumentation:
- `src/ovggt/utils/frontend_cache.py` — restructure `apply_voxel_dedup_()`
- `src/ovggt/training/frontend_oracle_collector.py` — update
  `CounterfactualDedupProbe.on_dedup_candidate` signature and use
  `scores` (not `metadata.importance`) when sampling
- `src/ovggt/training/frontend_oracle_collector.py` — also update
  `ReplayDedupKeepSetProbe.on_dedup_candidate` and
  `MultiReplayDedupKeepSetProbe.on_dedup_candidate` to accept and
  ignore the new optional args (they don't need the scores or policy
  keep — they override anyway).

This instrumentation is part of Phase 1 because Phase 1's quality gate
requires verifying that the current-policy baseline is always retained for
dedup events (see Phase 1 Sampling Policy Gate). Without
this restructure, that gate cannot be tested.

### Decision 3: Phase 1 Uses Stratified Event Selection

Lower `max_events_per_sequence` from 64 to 16, but do not take the first 16
candidate events. The current collector orders candidates by probe order and
then slices `candidate_events[:max_events]`; that biases toward early frames
and whichever event type appears first.

Phase 1 adds `event_selection_policy=stratified_round_robin`. The selector
groups candidates by `event_type`, `frame_id`, coarse layer bucket, and
voxel group / slot / event id, then round-robins across groups until the
sequence quota is filled. This keeps speed while improving frame, layer, and
scene diversity.

Important: the probe capture cap must be separate from the measured event cap.
The current implementation passes `max_events` directly into each probe, which
can stop candidate capture before later frames are observed. Phase 1 therefore
adds `max_candidate_events_per_sequence` (default 256). The probes may stop at
this candidate cap for memory control, but the final measured set is selected
by `max_events_per_sequence` (default 16). The production smoke test must
compare the frame histogram of the candidate pool and the selected events.

The candidate cap is **per-probe**, not global. The current code threads
`max_events` into all three probes via `max_events=self.max_events` at
`frontend_oracle_collector.py:1171, 1181, 1192`. Phase 1 changes each
call site to use `max_events=self.max_candidate_events_per_sequence`
instead, while the new `select_oracle_events()` applies the smaller
`max_events_per_sequence` cap. This ensures eviction/dedup/fifo probes
do not collectively starve each other under a single global limit —
each probe type can capture up to `max_candidate_events_per_sequence`
events independently.

#### Layer Bucket Definition

The "coarse layer bucket" is defined as `layer_id // 6`, producing 4
buckets for the 24-layer aggregator (buckets 0–3 covering layers
{0-5}, {6-11}, {12-17}, {18-23}). This choice is coordinated with
`layers_per_frame=2` from Decision 4: with `should_record_oracle_layer`
using stride `num_layers // layers_per_frame = 12`, a single frame's
sampled layers all live in different buckets, so the stratified
selector sees maximum bucket diversity per frame.

The bucket width is exposed as a config field
`stratified_layer_bucket_width: int = 6` so it can be retuned without
code changes if `num_layers` ever changes.

### Decision 4: Phase 1 Lowers `layers_per_frame` From 4 To 2

Rationale: `should_record_oracle_layer` already subsamples layers
deterministically. Halving to 2 cuts event count by ~2× with no change
to the per-event computation. The stratified selector must run after probe
collection so that lowering `layers_per_frame` does not further concentrate
events in frame 0.

### Decision 5: Add Event-Type Stress Profiles

The current real-policy shard has only `dedup` events. This is likely expected
under the current `frontend_per_layer_budget=8000` and `fifo_keep_topk=0`: budget
eviction may rarely trigger in 24 frames, and FIFO top-K is disabled.

Phase 1 therefore defines collection profiles:

- `oracle_profile=real_policy`: use the real frontend config. This is the main
  production distribution and may be mostly dedup.
- `oracle_profile=low_budget_eviction`: lower `frontend_per_layer_budget` only for
  a supplemental shard to force eviction events.
- `oracle_profile=fifo_topk`: enable `fifo_keep_topk` and, if needed, reduce
  keyframe capacity only for a supplemental shard to force fifo_topk events.

Stress-profile shards are data augmentation, not replacements for real-policy
shards. Training should either sample them with explicit weights or report
metrics by event type so they do not silently dominate the learned scorer.
Default policy: train and evaluate the first scorer on `real_policy` shards
only. Use stress-profile shards first as diagnostics to verify that the scorer
can rank eviction and fifo_topk events. If stress profiles are mixed into a
training run, the default sampling weights are:

- `real_policy`: 0.8
- `low_budget_eviction`: 0.1
- `fifo_topk`: 0.1

Any mixed-profile run must report metrics both overall and per event type.

#### Override Application Point

`frontend_per_layer_budget_override` and `fifo_keep_topk_override` MUST be
applied in `load_frontend_oracle_config()` (in
`src/ovggt/training/frontend_oracle_collector.py`), immediately after
`OmegaConf.load(config_path)` and BEFORE `OmegaConf.resolve(cfg)`. This
ensures the overrides participate in `${...}` interpolation if needed
and propagate through every downstream path that reads from `cfg`.

```python
def load_frontend_oracle_config(config_path, num_views=None, collector_cfg=None):
    cfg = OmegaConf.load(config_path)
    if collector_cfg is not None:
        if collector_cfg.oracle_profile == "low_budget_eviction":
            if collector_cfg.frontend_per_layer_budget_override is not None:
                cfg.frontend_per_layer_budget = int(collector_cfg.frontend_per_layer_budget_override)
        elif collector_cfg.oracle_profile == "fifo_topk":
            if collector_cfg.fifo_keep_topk_override is not None:
                # NOTE: fifo_keep_topk is a FrontendCacheConfig field, but the
                # YAML key under train_frontend_finetune.yaml is `frontend_cache`
                # (not `frontend_cache_config`). It is read by
                # build_frontend_cache_config() from cfg.frontend_cache.fifo_keep_topk.
                if not hasattr(cfg, "frontend_cache"):
                    cfg.frontend_cache = {}
                cfg.frontend_cache.fifo_keep_topk = int(collector_cfg.fifo_keep_topk_override)
    if num_views is not None:
        cfg.num_views = int(num_views)
    OmegaConf.resolve(cfg)
    return cfg
```

The `real_policy` profile applies no overrides — it is the identity path.
The override code path MUST be unit-tested with a small config fixture in
`tests/test_frontend_oracle_collector.py` to confirm the YAML field actually
changes after the override.

The YAML field name is `frontend_cache` (not `frontend_cache_config`),
matching the convention in `config/train_frontend_finetune.yaml` and
`config/eval_learned_eviction.yaml`. `build_frontend_cache_config()` in
`src/train_frontend.py` reads from `cfg.frontend_cache`.

### Decision 6: Phase 2 Adds Prefix-State Snapshots Inside A Single
Event's Replay

Add a model API to run the frontend once up to `event["frame_id"]`,
snapshot `cache_states`, `aggregator.last_scores`, camera head state,
attention anchor counts, and keyframe manager state, then for each
subset apply the keep set to the cached state and replay only the
future window `frames[frame_id+1:stop]`.

Rationale: today the prefix before the event is recomputed for every
subset. For an event at frame=10 with `oracle_window=4`, the prefix
costs ~10/14 of each replay. Snapshot reuse reduces this to one prefix
per event.

Equivalence contract: for a fixed seed and identical keep set, the loss
computed via snapshot replay must equal the loss from full replay within
`atol=1e-5, rtol=1e-4`. The test must cover eviction, dedup, and
fifo_topk events.

### Decision 7: Phase 2 Groups Events Sharing The Same `frame_id` To
Share One Prefix Snapshot

Rationale: in the v3 log, the first 12 events all use `frame=0`. With
per-event snapshots that would still build the same prefix 12 times.
Building one prefix per `frame_id` and replaying all events at that
frame from it reduces prefix work to ~min(num_frames, num_events).

Equivalence contract: per-event losses must be identical to per-event
snapshots, because the prefix state at `event["frame_id"]` is identical.

### Decision 8: Phase 3 Vectorizes The Frontend Inference `for b in range(B)`
Loop — **Out Of Scope By Default**

This is a model-level change to `ovggt.py` and `aggregator.py` to
support `B > 1` cache states in a single forward.

**Status:** Out of scope unless explicitly re-requested. Phase 3 is
*gated* behind a concrete throughput shortfall: it is only considered
if Phases 1+4 fail to deliver the 1-week full-corpus target on 8
GPUs.

Rationale: Phase 1+4 should be tested first. Phase 3 is high-risk
because it touches training and inference paths shared with the rest of
OVGGT, and any bug here corrupts both oracle collection and downstream
fine-tuning.

### Decision 9: Phase 4 Productionizes Multi-GPU Scheduling

Use the existing `tools/collect_counterfactual_oracle_parallel.py` with
8 devices and `shards_per_device` tuned so each shard finishes in ~6
hours. Add a resume mode that skips completed shards and a log
summarizer.

Phase 4 must also make sequence coverage deterministic. Different random
seeds are not enough: independent shuffled shards can repeat sequences and
miss others. Before a full production run, build a sequence manifest for the
36,750 training sequences and assign each shard a deterministic partition:

- `sequence_manifest_path`: path to a JSONL/CSV manifest with stable
  `sequence_id`, dataset, and frame provenance.
- `sequence_partition_policy`: `contiguous` or `hash_mod`.
- `num_sequence_shards` and `sequence_shard_id`: define the shard's
  non-overlapping manifest partition.

Every shard summary must report manifest coverage: requested sequences,
processed sequences, skipped/error sequences, and duplicate sequence ids. The
full run is not considered complete until the union of complete shards covers
the manifest target or explicitly records why a sequence was skipped.

Rationale: linear 8× speedup is useful only if the collected shards actually
increase sequence diversity. Deterministic partitions make resume, coverage
accounting, and failed-shard repair straightforward.

## Alternatives Considered

### Alternative A: Cap Only `max_events_per_sequence`, Keep Subset Count

Rejected: with 117 subsets/event, even 16 events/sequence still leaves
~1900 replays per sequence. Phase 1's per-event cap is the larger
lever.

### Alternative B: Drop Dedup Events Entirely, Keep Only Eviction + Fifo

Rejected: in the v3 log, 100% of captured events are dedup. Dropping
them would produce an empty shard for this sequence. The training
tokenizer needs all three event types.

### Alternative C: Run Teacher Once Per Dataset Instead Of Once Per Sequence

Rejected: teacher cost is 4s/sequence (~0.2% of replay). Eliminating it
gains negligible speedup and changes target semantics when GT is
missing.

### Alternative D: Skip Probe Forward, Reconstruct Events From Replay

Rejected: the probe captures cache snapshots at the exact frame the
eviction/dedup decision is made. Reconstructing this from a separate
replay would require a separate instrumentation pass with the same cost.

## Phase 1 Detailed Changes

### Files

- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Modify: `tools/collect_counterfactual_oracle_parallel.py`
- Modify: `config/collect_counterfactual_oracle.yaml`
- Modify: `src/train_token_scorer_oracle.py` for held-out event/sequence
  splits, event-type metrics, and optional stress-profile weighted sampling.
- Create or modify: a sequence-manifest utility used by Phase 4 deterministic
  shard partitioning. **Phase 1 lands a minimal manifest-builder stub +
  unit test** (so the smoke test step 7 can run on a tiny synthetic
  manifest); the manifest is **required only at Phase 4 launch** and is
  not on the Phase 1 critical path for the real-policy collection.
- Test: `tests/test_frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_parallel.py`
- Test: `tests/test_token_oracle_dataset.py`

### New Parameters

- `max_subsets_per_dedup_event: int = 8` — when a voxel group has more
  than this many tokens, sample subsets instead of enumerating.
- `max_subsets_per_eviction_event: int = 8` — same cap for the eviction
  probe (currently controlled by `num_samples`, but explicitly named).
- `max_subsets_per_fifo_event: int = 8` — cap fifo_topk candidates when
  stress-profile collection enables FIFO events.
- `max_candidate_events_per_sequence: int = 256` — cap probe-captured
  candidates before event selection; must be larger than
  `max_events_per_sequence` so the stratified selector can see later frames.
- `max_events_per_sequence: int = 16` — was 64.
- `max_events_per_frame: int = 6` — prevents frame 0 from consuming the
  entire sequence quota.
- `layers_per_frame: int = 2` — was 4.
- `event_selection_policy: str = "stratified_round_robin"` — selects
  events across event type, frame, layer bucket, and voxel/slot group instead
  of taking the first N probe results.
- `stratified_layer_bucket_width: int = 6` — maps `layer_id` to
  `layer_id // width` for event selection.
- `oracle_profile: str = "real_policy"` — one of `real_policy`,
  `low_budget_eviction`, or `fifo_topk`.
- `frontend_per_layer_budget_override: int | null = null` — stress-profile-only
  budget override to force eviction events.
- `fifo_keep_topk_override: int | null = null` — stress-profile-only override
  to force fifo_topk candidates.
- `sequence_manifest_path: str | null = null` — Phase 4 manifest for
  deterministic full-corpus coverage.
- `sequence_partition_policy: str = "hash_mod"` — Phase 4 sequence partition
  strategy, one of `hash_mod` or `contiguous`.
- `num_sequence_shards: int | null = null` and
  `sequence_shard_id: int | null = null` — Phase 4 non-overlapping sequence
  partition assignment.

### Subset Sampling Strategy

When a dedup voxel group has N tokens and `N > max_subsets_per_dedup_event`,
sample voxel-local keep choices, then convert each selected choice into a full
cache keep set:

```python
def _sample_dedup_keep_indices(
    group_indices: Tensor,
    dedup_scores: Tensor,
    cap: int,
    generator: torch.Generator,
    policy_keep_indices: Tensor | None = None,
) -> tuple[list[int], Tensor | None]:
    """Return token indices inside group_indices to keep for replay.

    dedup_scores must be group-local: shape [N], same order as group_indices.
    The optional return Tensor is a full-cache current-policy baseline subset.
    """
    N = group_indices.numel()
    if N <= cap:
        return group_indices.tolist(), None
    head = max(cap // 4, 1)
    tail = max(cap // 4, 1)

    chosen: set[int] = set()
    policy_baseline: Tensor | None = None
    if policy_keep_indices is not None:
        policy_set = {int(idx) for idx in policy_keep_indices.reshape(-1).tolist()}
        policy_local = [
            local_idx
            for local_idx, token_idx in enumerate(group_indices.tolist())
            if int(token_idx) in policy_set
        ]
        if len(policy_local) == 0:
            return [], None
        if len(policy_local) == 1:
            chosen.add(int(policy_local[0]))
        else:
            policy_baseline = policy_keep_indices.reshape(-1).detach().cpu().long()

    _, top_idx = torch.topk(dedup_scores, k=min(head, N))
    _, bot_idx = torch.topk(dedup_scores, k=min(tail, N), largest=False)
    chosen.update(int(i) for i in top_idx.tolist())
    chosen.update(int(i) for i in bot_idx.tolist())

    reserved = 1 if policy_baseline is not None else 0
    remaining_slots = max(cap - reserved - len(chosen), 0)
    remaining = [i for i in range(N) if i not in chosen]
    if remaining and remaining_slots > 0:
        perm = torch.randperm(len(remaining), generator=generator)[:remaining_slots]
        chosen.update(remaining[i] for i in perm.tolist())
    return [int(group_indices[i].item()) for i in sorted(chosen)], policy_baseline
```

### Equivalence Testing

A new test verifies that with `cap=8`, the emitted subset count is `<= cap`,
and each emitted dedup subset is either the explicit `policy_baseline` or a
keep-one sampled subset. For keep-one subsets, the test verifies:

- Keeps all tokens outside the voxel group.
- Keeps exactly one token inside the voxel group.
- Uses only selected voxel-local keep choices from
  `_sample_dedup_keep_indices()`.

The test must also verify:

- The cap does not change behavior when `N <= cap`.
- A group with zero policy-kept tokens is skipped.
- A group with one policy-kept token includes that token.
- A group with multiple policy-kept tokens emits one full-cache
  `policy_baseline` subset and still keeps total subsets `<= cap`.

### Event Selection Strategy

After the probe captures all candidate events for a sequence, Phase 1 selects
the events to measure with:

```python
def select_oracle_events(
    candidate_events: list[dict],
    max_events: int,
    max_events_per_frame: int,
    policy: str = "stratified_round_robin",
) -> list[dict]:
    if policy == "first_n":
        return candidate_events[:max_events]
    groups = group_by_event_type_frame_layer_and_voxel(candidate_events)
    selected = []
    per_frame_counts: dict[int, int] = {}
    for event in round_robin(groups):
        frame_id = int(event["frame_id"])
        if per_frame_counts.get(frame_id, 0) >= max_events_per_frame:
            continue
        selected.append(event)
        per_frame_counts[frame_id] = per_frame_counts.get(frame_id, 0) + 1
        if len(selected) >= max_events:
            break
    return selected
```

The exact grouping function must be deterministic and stable across fixed
seeds. It should include `event_type`, `frame_id`, coarse layer bucket, and
the event-specific group id (`voxel_group_id`, `demoted_slot`, or event id).
This prevents the first few frames or one large voxel group from consuming a
whole sequence's quota.

Implementation detail: `candidate_events` must be the post-probe candidate
pool capped by `max_candidate_events_per_sequence`, not by
`max_events_per_sequence`. The measured event cap is applied only inside
`select_oracle_events()`. A regression test must construct a sequence whose
first 16 candidates are all frame 0 and verify that
`stratified_round_robin` selects later frames when the candidate cap is large
enough.

### Recommended Production Defaults

For the first production speed run:

```yaml
max_subsets_per_dedup_event: 8
max_subsets_per_eviction_event: 8
max_subsets_per_fifo_event: 8
max_candidate_events_per_sequence: 256
max_events_per_sequence: 16
max_events_per_frame: 6
layers_per_frame: 2
event_selection_policy: stratified_round_robin
stratified_layer_bucket_width: 6
subset_replay_batch_size: 16
store_replay_payload: false
oracle_profile: real_policy
```

Run supplemental stress shards only after the real-policy shard smoke test
passes:

```yaml
oracle_profile: low_budget_eviction
frontend_per_layer_budget_override: 833  # tune in smoke tests
```

```yaml
oracle_profile: fifo_topk
fifo_keep_topk_override: 8
```

### Shard Summary Metrics

Every partial and final shard log should summarize:

- `num_sequences`, `num_events`, and `partial`
- manifest partition id, requested sequence count, processed sequence count,
  skipped sequence count, and duplicate sequence ids
- event type counts
- dataset/scene counts from `sequence_provenance`
- frame and layer histograms
- subsets/event min, mean, p50, p90, max
- pair count after `min_loss_gap`
- average replay seconds/event and events/hour

### Expected Impact

- ~5-10× from the dedup subset cap (depends on voxel group sizes)
- 4× from `max_events_per_sequence: 64 → 16`
- 2× from `layers_per_frame: 4 → 2`
- Net: ~30-50× on per-sequence wall time

Single-sequence runtime target: ~5-10 minutes (down from ~4.5h).
Single-GPU full-corpus target: ~600-1200h (down from ~18,000h).
8-GPU full-corpus target: ~3-6 days.

## Verification Plan

### Exact Equivalence Gate For Code Optimizations

For changes that are intended to be semantics-preserving, run the existing
collector and the modified collector with sampling caps disabled:

1. Run the existing collector on a fixed-seed 2-sequence fixture with
   the current parameters; save the resulting shard.
2. Run the modified collector on the same fixture.
3. Assert:
   - Same `num_events`
   - Same event ordering by `event_id`
   - For matching events: identical `score_state` and `metadata_features`
   - For matching subsets: identical `keep_indices` and `loss` matches within
     `atol=1e-5, rtol=1e-4`

This gate applies to Phase 2 snapshot reuse and any refactor around replay.
It does not apply to Phase 1 caps when those caps are enabled.

### Phase 1 Sampling Policy Gate

Phase 1 intentionally produces fewer subsets and events. Before production,
run a calibration shard with the following structure. The cap is bounded
because the no-cap reference is the original ~4.5 h/sequence path — only
small reference datasets are feasible.

**Calibration matrix (≤34 GPU-hours total budget):**

| Run | Cap | Sequences | Estimated time |
|-----|-----|-----------|----------------|
| 1 | 8 | 30 | ~2-5h (cap=8 path) |
| 2 | 12 | 30 | ~3-7h |
| 3 | 16 | 30 | ~4-9h |
| 4 | no-cap (reference) | **3** | ~13h (original ~4.5h/seq) |

**Total: ≤34 GPU-hours on a single GPU** (~1.5 days). The no-cap reference
is explicitly capped at **3 sequences** (not 20-50) because the original
collector is ~30× slower than the cap=8 path; 20-50 sequences would
take 90-225 GPU-hours and violate the 1-week total budget. Three
sequences is enough to spot-check which subsets the cap=8 path drops
and to verify loss-gap agreement for retained subsets.

All four runs MUST use the same fixed seed and the same 30 (or 3)
sequences. Sequences are sampled deterministically from the dataset
using a fixed `seed=42` sampler.

**Per-run report:**

- sequences/hour and events/hour
- unique sequences and datasets covered
- event type distribution
- frame and layer distribution
- candidate-pool frame/layer distribution before selection
- selected-event frame/layer distribution after `stratified_round_robin`
- subsets/event and pair count
- loss-gap distribution after `min_loss_gap`
- For cap runs only: percentage of retained subsets whose losses match
  the no-cap reference on the 3 overlap sequences
- For cap runs only: whether the current-policy baseline is always retained
  for dedup events (this requires the Decision 2 instrumentation change)

**Selection rule:** The fastest cap whose retained-subset loss-gap
agreement with the no-cap reference is ≥80% on the 3 overlap
sequences, AND whose current-policy baseline retention is 100% on dedup
events, is selected for production. Default recommendation is cap `8`
for production speed, with cap `12` or `16` as fallback if calibration
shows too much signal loss.

### Training Quality Gate

Before the full run, train `TokenScorer` for a short run on the
calibration shards. The split is applied before pairwise sample expansion.
For the early calibration gate, a 90/10 event-level split by `event_id` hash is
acceptable because it checks whether the cap destroys ranking signal. For the
full gate, use a 90/10 sequence-level split by
`sequence_provenance.sequence_id` or a stable scene hash, because the goal is
sequence and scene generalization. Report:

- Pairwise rank accuracy on the held-out 10%
- Regression loss on the held-out 10%
- Per-event-type rank accuracy (dedup only for real_policy; dedup +
  eviction + fifo for stress profiles)

**Early gate:** Held-out rank accuracy must be ≥ 52% on the cap=8
calibration shard. This is a concrete bar above random (50%), tighter
than the current ~55.2% in-training number (which has no held-out
split and 17 sequences only). If cap=8 fails this gate, retry with
cap=12; if cap=12 also fails, revise the sampling strategy before
launching the full 8-GPU collection.

**Why a held-out bar:** The 55.2% baseline is in-training accuracy on
1,088 events from 17 sequences with no held-out split. Almost any
broader dataset would trivially improve it. A held-out split with a
modest concrete threshold (52%) is the minimum meaningful bar for an
early calibration gate.

**Full gate (after Phase 1+4 produce a broad shard set):** Train on a
sequence-level 90/10 split across the full corpus. Held-out rank accuracy
should be ≥ 60%, with metrics reported both overall and by event type.

### Production Smoke Test

Before launching the full 8-GPU run:

1. Run a single shard on a single GPU for 10 sequences.
2. Verify events/hour is within 2× of the Phase 1 projection.
3. Verify the shard loads correctly with `CounterfactualOracleDataset`.
4. Verify the shard summary includes nonzero sequence count, event type
   counts, frame histogram, layer histogram, subsets/event, pair count, and
   loss-gap histogram.
5. Verify `event_selection_policy=stratified_round_robin` improves frame
   coverage compared with `first_n` on the same fixed-seed sequence.
6. Verify `max_candidate_events_per_sequence > max_events_per_sequence` and
   that the selected events include later frames when the candidate pool
   contains them.
7. For Phase 4 smoke, run two shards from a tiny manifest partition and verify
   their sequence ids are non-overlapping and the shard summaries report zero
   duplicates.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 cap drops too much signal | Medium | Training quality gate; can raise cap from 8 to 12 |
| Phase 1 first-N event selection keeps early-frame bias | High | Use `stratified_round_robin`; include frame/layer histograms in every shard summary |
| Stratified selector sees only early candidates | High | Separate `max_candidate_events_per_sequence` from `max_events_per_sequence`; verify candidate and selected frame histograms |
| Dedup cap misses the current policy baseline | Medium | Pass actual dedup decision scores and policy keep set into the probe; assert current-policy baseline is retained |
| Current-policy dedup baseline is ambiguous for multi-token policy keeps | Medium | Derive `policy_keep_indices ∩ group_indices`; emit full policy baseline for multi-token intersections and test zero/one/multiple cases |
| Stress-profile shards shift the training distribution | Medium | Keep stress shards supplemental; train/evaluate with event-type weights and per-type metrics |
| Full-corpus collection repeats sequences across random shards | High | Use a deterministic sequence manifest and non-overlapping shard partitions; report duplicate ids in every shard summary |
| Phase 2 snapshot equivalence bug | Medium | Strict per-event equivalence test before merge |
| Phase 2 determinism broken by RNG state in snapshot | Medium | Snapshot/restore must preserve RNG stream state for any stochastic op in the prefix (dropout, stochastic depth). Add explicit RNG-state assertion to the equivalence test |
| Phase 2 memory pressure from saved snapshots | Low | Snapshot to CPU; restore on demand. Hard ceiling: snapshot footprint must stay <8 GB per event; validate on the longest sequence in the corpus (24 frames × full budget). Multiple per-`frame_id` snapshots must fit in <24 GB combined |
| Phase 3 breaks training/inference correctness | High | Out of scope unless Phase 1+4 leave throughput below target. Phase 3 is **gated** and not on the default critical path |
| 8-GPU scheduling collides with other workloads | Low | Pin to GPUs 0-7 via SLURM/CUDA_VISIBLE_DEVICES |
| Resume mode encounters heterogeneous shard shapes | Medium | Phase 1 intentionally reduces events-per-sequence and subsets-per-event. Resume logic must tolerate shard-size variance: skip iff shard exists with `partial=False`; do not assume a fixed events/shard count |

## Out Of Scope

- Changing the shard format (`ovggt_counterfactual_oracle_v1`).
- Changing `TASK_WEIGHTS` or loss definitions.
- Full downstream frontend fine-tuning benchmark. Short TokenScorer calibration
  runs are in scope for quality gates.
- Collecting on more than 8 GPUs — the hardware is fixed.
- Multi-node collection — not needed for the 1-week target.
