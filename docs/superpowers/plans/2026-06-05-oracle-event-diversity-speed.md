# Oracle Event Diversity + Speed Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect a much more diverse oracle dataset (dedup, eviction, fifo_topk/FIFO_SWAP, later frames, deeper layers, more scenes) without blindly increasing `max_events_per_sequence`.

**Architecture:** Split collection into two stages. Stage A runs cheap probe-only diagnostics to find which profile actually produces each event type and which frame/layer buckets are available. Stage B replays only quota-selected candidates from profile-specific collectors, so expensive counterfactual replay is spent on scarce/high-value event types instead of the first 16 dedup events.

**Tech Stack:** Python, PyTorch, OVGGT frontend cache, `tools/collect_counterfactual_oracle.py`, `src/ovggt/training/frontend_oracle_collector.py`, pytest.

---

## Evidence From Latest GPU3 Log

Log: `/tmp/oracle_collection_gpu3.log`

Observed facts:

- The run used `config/collect_counterfactual_oracle.yaml`, which applies `oracle_profile=fifo_topk`, `fifo_keep_topk_override=80`, `frontend_per_layer_budget_override=2500`, `subset_replay_batch_size=16`, `layers_per_frame=2`, and `max_events_per_sequence=16`.
- Across 128 probe batches, raw candidates were:
  - `eviction`: 564 total, present in 77/128 batches
  - `dedup`: 18,287 total
  - `fifo_topk`: 0 total
- Despite raw eviction candidates existing, the final shard is still `event_type_counts={"dedup": 2048}`.
- Final frame histogram is almost entirely frames `0, 1, 2`: `{"0": 749, "1": 745, "2": 549, "3": 4, "4": 1}`.
- Final layer histogram is dominated by `0, 1, 2, 12, 13, 14`: `{"0": 373, "1": 368, "2": 382, "12": 376, "13": 377, "14": 167, ...}`.
- Runtime was about `38668 s` for 2048 events, roughly 10.7 hours on GPU3. Later replay chunks slowed to 20-32s/event-subset group, likely due shared GPU load or system contention.

Root-cause conclusions:

- For eviction, this is now clearly a selection/replay allocation bug: raw eviction candidates exist, but all replay budget is consumed by dedup.
- Eviction probe events currently do not reliably participate as a first-class event type in quota selection. In `CounterfactualEvictionProbe`, emitted events should carry `"event_type": "eviction"` before any selector groups by type.
- The current `stratified_round_robin` policy sorts groups by `(event_type, frame, layer_bucket, voxel)` and has no event-type quotas. With hundreds of early dedup candidates per sequence and only `max_events_per_sequence=16`, dedup can consume every measured slot before eviction is reached.
- `layers_per_frame=2` selects layer pairs by frame: for 24 layers, frame 0 -> layers 0/12, frame 1 -> 1/13, frame 2 -> 2/14. Because the first 16 selected candidates are all early dedup, the measured shard is stuck in early frames and early/deep-pair layers.
- FIFO events require `FIFO_SWAP` plus `frontend_cache.fifo_keep_topk > 0` or learned count-head mode. With 24 views, `anchor_interval=8`, and `max_anchors=3`, a FIFO swap usually needs enough keyframe promotions to exceed three history anchors. The current run does not reach such events in the measured candidate pool.
- The GPU3 run already lowered `frontend_per_layer_budget` to 2500 and did produce eviction candidates, so the immediate eviction fix is quota selection, not an even lower budget.

Non-goal:

- Do not solve this by simply raising `max_events_per_sequence`; that mostly increases expensive replay of already abundant dedup events.

---

## Proposed Collection Strategy

Use a profile mixture, not a single global command:

| Profile | Purpose | Key knobs | Replay budget |
| --- | --- | --- | --- |
| `real_policy_dedup` | Keep natural-policy dedup distribution | current budget, dedup enabled | small quota |
| `low_budget_eviction` | Replay existing eviction candidates instead of dropping them | `frontend_per_layer_budget=2500` already works; quota selection is the key fix | medium quota |
| `fifo_topk_forced` | Force FIFO_SWAP/top-k events | `fifo_keep_topk_override`, shorter keyframe interval, lower `max_anchors` | medium/high quota |
| `late_frame_scan` | Improve frame/layer coverage | cheap probe-only pass, later-frame quotas, random layer buckets | replay only selected late buckets |

The important change is quota-first selection:

- Select by `(event_type, frame_bucket, layer_bucket, dataset)` before replay.
- Use per-type caps, e.g. `dedup <= 4`, `eviction <= 8`, `fifo_topk <= 8` per sequence.
- Prefer scarce event types first; by default leave unused slots empty rather than letting dedup exceed its quota. Use `--quota-fill-remaining` only for experiments where full shard density matters more than strict type caps.
- Keep `max_events_per_sequence` modest, but make it type-aware.

---

## Backward Compatibility

- The existing `stratified_round_robin` and `first_n` policies MUST remain unchanged. `quota_stratified` is a new policy added to the choices list.
- Existing callers of `select_oracle_events(...)` without `event_type_quotas` continue to work — the parameter defaults to `None`, and when `None` the function falls through to the existing policy behavior.
- Existing `.pt` shards have eviction events WITHOUT `"event_type"` field. `compute_shard_summary` already handles this via `e.get("event_type", "unknown")` (line 151). No migration needed for old shards, but training code that groups by event type should use `.get("event_type", "eviction")` as the fallback for legacy eviction events.
- New fields on `FrontendOracleCollectorConfig` (`probe_only`, `event_type_quotas`, `frame_buckets`, `oracle_layer_schedule`, etc.) all default to `None`/`False`, so existing config YAMLs work unchanged.

---

## Files

- Modify: `src/ovggt/training/frontend_oracle_collector.py`
  - Add probe-only mode.
  - Add event-type quotas and richer selection.
  - Add keyframe/FIFO trigger overrides.
  - Add layer schedule modes and frame buckets.
  - Improve diagnostics.
- Modify: `tools/collect_counterfactual_oracle.py`
  - Add CLI args for profile/quotas/probe-only/keyframe overrides/layer schedule/frame buckets.
  - Update `choices` for `--event-selection-policy`.
- Modify: `config/collect_counterfactual_oracle.yaml`
  - Add recommended mixed-profile defaults.
- Create: `tools/summarize_oracle_event_diversity.py`
  - Summarize event/frame/layer/dataset coverage across logs and shards.
- Test: `tests/test_frontend_oracle_collector.py`
- Test: `tests/test_frontend_oracle_parallel.py`
- Test: `tests/test_token_oracle_dataset.py`

---

## Task Dependencies

```
Task 1 (Probe-Only) ──────────────────────────────────┐
    │                                                   │
    ▼                                                   │
Task 2 (Quota Selection) ──────────────────────────┐   │
    │                                               │   │
    ├──────────────────┐                            │   │
    ▼                  ▼                            ▼   ▼
Task 3 (FIFO)    Task 4 (Eviction Profile)    Task 5 (Frame/Layer)
    │                  │                            │
    └──────┬───────────┘                            │
           ▼                                        │
    Task 6 (Mixed-Profile Launcher) ◄───────────────┘
           │
           ▼
    Task 7 (Diversity Summarizer)
           │
           ▼
    Task 8 (Production Rollout)
```

- Tasks 1 → 2 are sequential: probe-only diagnostics must exist before quota selection can be tested.
- Tasks 3, 4, 5 are independent of each other but all depend on Task 2 (they need `quota_stratified` to work).
- Task 6 depends on Tasks 3, 4, 5 (it generates commands for all profiles).
- Task 7 is independent of 3-5 but useful before Task 8.
- Task 8 is operational, not code.

---

## Task 1: Add Probe-Only Candidate Diagnostics

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

**Plumbing chain for `probe_only`:**

```
CLI: --probe-only
  → FrontendOracleCollectorConfig.probe_only: bool = False    [NEW FIELD]
  → FrontendOracleCollectorConfig.probe_output_json: str | None = None    [NEW FIELD]
    → collect_oracle_shard_from_config(...)
      → probe_diagnostics_list = []
      → collect_oracle_events_from_loader(..., probe_only=..., probe_diagnostics_list=...)
        → collect_oracle_events_from_sequence(..., probe_only=..., probe_diagnostics_sink=...)
          → if probe_only: append diagnostics to sink, skip teacher + replay, return []
      → shard["probe_diagnostics"] = probe_diagnostics_list
```

- [ ] **Step 1: Write failing test for probe-only mode**

```python
def test_probe_only_does_not_replay_and_reports_candidate_histograms(monkeypatch):
    """probe_only=True should skip measure_counterfactual_event and return diagnostics."""
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from ovggt.training.frontend_oracle_collector import collect_oracle_events_from_sequence

    frames = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(3)]
    diagnostics = []

    def fake_run_frontend(model, frames, probe, cache_results, dedup_probe=None, fifo_probe=None, **kwargs):
        probe.events = [
            {"event_type": "eviction", "frame_id": 0, "layer_id": 0, "candidate_subsets": []}
        ]
        dedup_probe.events = [
            {"event_type": "dedup", "frame_id": 1, "layer_id": 12, "candidate_subsets": []}
        ]
        fifo_probe.events = []
        return SimpleNamespace(ress=[])

    with patch(
        "ovggt.training.frontend_oracle_collector._run_frontend_with_probe",
        side_effect=fake_run_frontend,
    ), patch(
        "ovggt.training.frontend_oracle_collector.measure_counterfactual_event",
        side_effect=AssertionError("measure_counterfactual_event must NOT be called in probe_only mode"),
    ):
        result = collect_oracle_events_from_sequence(
            model=object(),
            frames=frames,
            device=torch.device("cpu"),
            max_events=16,
            num_samples=4,
            oracle_window=2,
            probe_only=True,  # NEW PARAMETER
            probe_diagnostics_sink=diagnostics,  # NEW PARAMETER
        )

    # Should return empty measured events
    assert len(result) == 0  # No measured events in probe-only mode
    assert len(diagnostics) == 1
    assert "raw_event_type_counts" in diagnostics[0]
    assert "selected_event_type_counts" in diagnostics[0]
```

```python
def test_probe_only_shard_returns_diagnostics(tmp_path):
    """collect_oracle_shard_from_config with probe_only returns diagnostics in shard metadata."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import ovggt.training.frontend_oracle_collector as collector
    from ovggt.training.frontend_oracle_collector import (
        FrontendOracleCollectorConfig,
        collect_oracle_shard_from_config,
    )

    def fake_collect_oracle_events_from_loader(**kwargs):
        kwargs["probe_diagnostics_list"].append(
            {
                "raw_event_type_counts": {"dedup": 1},
                "raw_frame_histogram": {"0": 1},
                "raw_layer_histogram": {"0": 1},
            }
        )
        return []

    cfg = FrontendOracleCollectorConfig(
        config="config/collect_counterfactual_oracle.yaml",
        output=str(tmp_path / "probe_test.pt"),
        max_batches=1,
        max_events=16,
        probe_only=True,  # NEW FIELD
    )

    with patch.object(collector, "load_frontend_oracle_config", return_value=SimpleNamespace(n_corres_train=0)), \
         patch.object(collector, "build_frozen_frontend_model_from_config", return_value=SimpleNamespace(aggregator=SimpleNamespace(depth=24), state_dict=lambda: {})), \
         patch.object(collector, "build_frozen_teacher_from_config", return_value=None), \
         patch.object(collector, "build_frontend_oracle_dataloader", return_value=[]), \
         patch.object(collector, "collect_oracle_events_from_loader", side_effect=fake_collect_oracle_events_from_loader):
        shard = collect_oracle_shard_from_config(cfg)

    assert "probe_diagnostics" in shard
    assert isinstance(shard["probe_diagnostics"], list)
    assert len(shard["probe_diagnostics"]) == 1
    diag = shard["probe_diagnostics"][0]
    assert "raw_event_type_counts" in diag
    assert "raw_frame_histogram" in diag
    assert "raw_layer_histogram" in diag
    assert isinstance(diag["raw_event_type_counts"], dict)
    for key in diag["raw_event_type_counts"]:
        assert key in ("dedup", "eviction", "fifo_topk", "unknown")
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_probe_only_does_not_replay_and_reports_candidate_histograms -q
```

Expected: fail because `probe_only` parameter does not exist on `collect_oracle_events_from_sequence`.

- [ ] **Step 3: Add `probe_only` field to `FrontendOracleCollectorConfig`**

In `src/ovggt/training/frontend_oracle_collector.py`, add to the dataclass (after `fifo_count_candidates_for_oracle`):

```python
# --- Probe-only diagnostics mode ---
probe_only: bool = False
probe_output_json: str | None = None
```

- [ ] **Step 4: Add `probe_only` parameter to the plumbing chain**

Each function in the chain gets a new `probe_only: bool = False` parameter with default `False`. The sequence function also gets `probe_diagnostics_sink: list[dict] | None = None`, because the existing return type is `list[dict]` of measured events and should stay backward-compatible.

a) `collect_oracle_events_from_sequence` (line 1630): add:

```python
probe_only: bool = False,
probe_diagnostics_sink: list[dict] | None = None,
```

When `probe_only=True`:
- Run `_run_frontend_with_probe(...)` normally (to populate probe events).
- Build candidate histograms from all three probes (eviction, dedup, fifo).
- **Important:** run the `probe_only` diagnostics branch before any `if not candidate_events: return []` early return, so empty candidate probes still produce diagnostics.
- **Skip teacher entirely** (do NOT call `teacher.inference`). Insert early return before line 1742 (`teacher_outputs = None`).
- **Skip the replay loop entirely** (do NOT call `measure_counterfactual_event`). Insert early return before line 1764 (`for event_idx, event in enumerate(candidate_events)`).
- Build diagnostics dict, append it to `probe_diagnostics_sink` if provided, and return empty list of measured events.

```python
# Insert after line 1730 (after select_oracle_events and logging), before line 1739:
if probe_only:
    diagnostics = _build_probe_diagnostics(
        eviction_events=probe.events,
        dedup_events=dedup_probe.events,
        fifo_events=fifo_probe.events,
        candidate_events=candidate_events,
    )
    if log_fn is not None:
        log_fn(
            f"{event_prefix}: probe_only mode - skipping replay. "
            f"diagnostics: {json.dumps(diagnostics, default=str)}"
        )
    if probe_diagnostics_sink is not None:
        probe_diagnostics_sink.append(diagnostics)
    return []
```

b) `collect_oracle_events_from_loader` (line 1489): add `probe_only: bool = False` and `probe_diagnostics_list: list[dict] | None = None`. Pass `probe_diagnostics_sink=probe_diagnostics_list` to `collect_oracle_events_from_sequence`.

c) `collect_oracle_shard_from_config` (line 1368): read `collector_cfg.probe_only`, create `probe_diagnostics_list: list[dict] = []`, pass it to `collect_oracle_events_from_loader`, and when `probe_only=True`, store it in `shard["probe_diagnostics"]`.

- [ ] **Step 5: Add `_build_probe_diagnostics` helper**

```python
def _build_probe_diagnostics(
    eviction_events: list[dict],
    dedup_events: list[dict],
    fifo_events: list[dict],
    candidate_events: list[dict],
) -> dict:
    """Build histogram diagnostics from probe events for probe-only mode."""
    def _histogram(items: list[dict], key: str, transform=None) -> dict[str, int]:
        h: dict[str, int] = {}
        for item in items:
            val = transform(item[key]) if transform else item.get(key, "unknown")
            val = str(val)
            h[val] = h.get(val, 0) + 1
        return dict(sorted(h.items()))

    all_raw = list(eviction_events) + list(dedup_events) + list(fifo_events)
    raw_et: dict[str, int] = {}
    for e in all_raw:
        et = e.get("event_type", "eviction")  # eviction probe lacks event_type currently
        raw_et[et] = raw_et.get(et, 0) + 1

    return {
        "raw_event_type_counts": dict(sorted(raw_et.items())),
        "raw_frame_histogram": _histogram(all_raw, "frame_id", int),
        "raw_layer_histogram": _histogram(all_raw, "layer_id", int),
        "selected_event_type_counts": _histogram(candidate_events, "event_type"),
        "selected_frame_histogram": _histogram(candidate_events, "frame_id", int),
        "selected_layer_histogram": _histogram(candidate_events, "layer_id", int),
        "raw_total": len(all_raw),
        "selected_total": len(candidate_events),
    }
```

- [ ] **Step 6: Add CLI args**

In `tools/collect_counterfactual_oracle.py`, add to `parse_args`:

```python
parser.add_argument(
    "--probe-only",
    action="store_true",
    help="Run probe-only diagnostics: collect candidate histograms without replay. "
         "Output shard contains 'probe_diagnostics' instead of 'events'.",
)
parser.add_argument(
    "--probe-output-json",
    type=str,
    default=None,
    help="Write probe diagnostics to this JSON file (in addition to the .pt shard).",
)
```

Wire into `FrontendOracleCollectorConfig`:

```python
probe_only=bool(args.probe_only),
probe_output_json=args.probe_output_json,
```

Add `"probe_only"` and `"probe_output_json"` to `_YAML_TO_ARG` mapping.

After `collect_oracle_shard_from_config(...)` returns in `main()`, write JSON diagnostics when requested:

```python
if args.probe_output_json:
    with open(args.probe_output_json, "w", encoding="utf-8") as f:
        json.dump(shard.get("probe_diagnostics", []), f, indent=2, default=str)
```

- [ ] **Step 7: Verify tests pass**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_probe_only_does_not_replay_and_reports_candidate_histograms tests/test_frontend_oracle_collector.py::test_probe_only_shard_returns_diagnostics -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tools/collect_counterfactual_oracle.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add probe-only candidate diagnostics mode

- Add probe_only flag to FrontendOracleCollectorConfig
- Skip teacher + replay when probe_only=True
- Return raw/selected event type, frame, layer histograms
- Add --probe-only and --probe-output-json CLI args"
```

---

## Task 2: Add Event-Type Quota Selection

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

**Plumbing chain for `event_type_quotas`:**

```
CLI: --event-type-quotas eviction=8,dedup=4,fifo_topk=4
  → FrontendOracleCollectorConfig.event_type_quotas: dict[str,int] | None = None  [NEW FIELD]
    → collect_oracle_shard_from_config(...)
      → collect_oracle_events_from_loader(..., event_type_quotas=...)
        → collect_oracle_events_from_sequence(..., event_type_quotas=...)
          → select_oracle_events(..., event_type_quotas=...)
```

Each function in the chain gets a new `event_type_quotas: dict[str, int] | None = None` parameter. When `None`, existing behavior is unchanged.

- [ ] **Step 1: Write failing test that eviction events carry an event type**

```python
def test_eviction_events_have_event_type_field():
    """CounterfactualEvictionProbe events must include 'event_type': 'eviction'."""
    import torch
    from ovggt.training.frontend_oracle_collector import CounterfactualEvictionProbe
    from ovggt.utils.frontend_cache import LayerCacheState

    # Reuse the existing _metadata(...) helper from tests/test_frontend_oracle_collector.py.
    # It builds real TokenMetadata with PATCH tokens and valid slot metadata.
    probe = CounterfactualEvictionProbe(num_samples=4, oracle_window=2, seed=42)
    cache_state = LayerCacheState(
        k=torch.randn(1, 2, 6, 4),
        v=torch.randn(1, 2, 6, 4),
        score_state=torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
        metadata=_metadata(
            anchor_slots=[0, 1, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 2, 2],
            importance=[0.0, 0.0, 0.5, 0.2, 0.9, 0.1],
        ),
        protected_count=2,
    )
    probe.on_eviction_candidate(
        cache_state, layer_id=0, frame_id=1, budget=4, batch_index=0,
    )
    assert len(probe.events) >= 1, "Expected at least one eviction event"
    for event in probe.events:
        assert event.get("event_type") == "eviction", (
            f"Eviction event missing event_type='eviction', got: {event.get('event_type')}"
        )
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_eviction_events_have_event_type_field -q
```

Expected: FAIL — `CounterfactualEvictionProbe.on_eviction_candidate` does not set `"event_type"`.

- [ ] **Step 3: Implement explicit eviction event type**

In `CounterfactualEvictionProbe.on_eviction_candidate(...)`, add to the event dict (line 346 area):

```python
self.events.append(
    {
        "event_id": event_id,
        "event_type": "eviction",  # <-- ADD THIS LINE
        "layer_id": int(layer_id),
        "frame_id": int(frame_id),
        "batch_index": int(batch_index),
        "budget": budget,
        # ... rest unchanged ...
    }
)
```

- [ ] **Step 4: Verify eviction event type test passes**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_eviction_events_have_event_type_field -q
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for quota selection**

```python
def test_quota_stratified_prefers_scarce_event_types():
    """When dedup is abundant and eviction is scarce, quota_stratified selects eviction first."""
    candidates = []
    for i in range(20):
        candidates.append({"event_type": "dedup", "frame_id": i % 5, "layer_id": i % 6, "voxel_group_id": i})
    for i in range(2):
        candidates.append({"event_type": "eviction", "frame_id": i, "layer_id": 0, "voxel_group_id": 100 + i})

    selected = select_oracle_events(
        candidates,
        max_events=10,
        max_events_per_frame=6,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas={"eviction": 8, "dedup": 4, "fifo_topk": 4},
    )

    eviction_count = sum(1 for e in selected if e.get("event_type") == "eviction")
    dedup_count = sum(1 for e in selected if e.get("event_type") == "dedup")
    assert eviction_count == 2, f"Expected all 2 eviction events, got {eviction_count}"
    assert dedup_count <= 4, f"Dedup should be capped by quota, got {dedup_count}"


def test_quota_stratified_dedup_cannot_consume_all_slots():
    """With event_type_quotas set, dedup cannot consume all slots even if max_events is low."""
    candidates = [{"event_type": "dedup", "frame_id": i, "layer_id": 0, "voxel_group_id": i} for i in range(100)]
    candidates += [{"event_type": "eviction", "frame_id": i, "layer_id": 0, "voxel_group_id": 100 + i} for i in range(2)]

    selected = select_oracle_events(
        candidates,
        max_events=16,
        max_events_per_frame=16,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas={"eviction": 8, "dedup": 4},
    )

    eviction_count = sum(1 for e in selected if e.get("event_type") == "eviction")
    dedup_count = sum(1 for e in selected if e.get("event_type") == "dedup")
    assert eviction_count == 2, f"Expected 2 eviction events, got {eviction_count}"
    assert dedup_count <= 4, f"Dedup should be capped by quota, got {dedup_count}"


def test_quota_stratified_respects_frame_layer_diversity():
    """Within each event type, quota_stratified still stratifies by frame/layer."""
    candidates = [
        {"event_type": "eviction", "frame_id": f, "layer_id": l, "voxel_group_id": f * 100 + l}
        for f in range(10)
        for l in range(24)
    ]

    selected = select_oracle_events(
        candidates,
        max_events=16,
        max_events_per_frame=6,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas={"eviction": 16},
    )

    frames = set(e["frame_id"] for e in selected)
    layers = set(e["layer_id"] // 6 for e in selected)
    assert len(frames) > 1, "Expected diversity across frames"
    assert len(layers) > 1, "Expected diversity across layer buckets"


def test_quota_stratified_without_quotas_falls_back_to_stratified():
    """When event_type_quotas is None, quota_stratified behaves like stratified_round_robin."""
    candidates = [
        {"event_type": "dedup", "frame_id": 0, "layer_id": i, "voxel_group_id": i}
        for i in range(20)
    ]
    selected = select_oracle_events(
        candidates,
        max_events=10,
        max_events_per_frame=10,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas=None,
    )
    assert len(selected) == 10
```

- [ ] **Step 6: Run tests and verify they fail**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_quota_stratified_prefers_scarce_event_types -q
```

Expected: FAIL — `quota_stratified` is not a recognized policy value in `select_oracle_events`.

- [ ] **Step 7: Update CLI `choices` for `--event-selection-policy`**

In `tools/collect_counterfactual_oracle.py`, line 107, update:

```python
parser.add_argument(
    "--event-selection-policy",
    type=str,
    default="stratified_round_robin",
    choices=["first_n", "stratified_round_robin", "quota_stratified"],
)
```

- [ ] **Step 8: Implement `quota_stratified` in `select_oracle_events`**

Add `event_type_quotas` parameter to `select_oracle_events` (line 1333):

```python
def select_oracle_events(
    candidate_events: list[dict],
    max_events: int,
    max_events_per_frame: int,
    policy: str = "stratified_round_robin",
    layer_bucket_width: int = 6,
    event_type_quotas: dict[str, int] | None = None,  # NEW PARAMETER
    quota_fill_remaining: bool = False,  # NEW PARAMETER
) -> list[dict]:
```

Add the `quota_stratified` policy branch at the top of the function, before the existing `first_n` check:

```python
    if policy == "quota_stratified":
        if event_type_quotas is None:
            # Fall back to stratified_round_robin behavior when no quotas specified
            policy = "stratified_round_robin"
        else:
            return _select_oracle_events_quota_stratified(
                candidate_events=candidate_events,
                max_events=max_events,
                max_events_per_frame=max_events_per_frame,
                layer_bucket_width=layer_bucket_width,
                event_type_quotas=event_type_quotas,
                quota_fill_remaining=quota_fill_remaining,
            )
    # Existing policies unchanged below this line
```

Add the new helper function (before `select_oracle_events`):

```python
def _select_oracle_events_quota_stratified(
    candidate_events: list[dict],
    max_events: int,
    max_events_per_frame: int,
    layer_bucket_width: int,
    event_type_quotas: dict[str, int],
    quota_fill_remaining: bool = False,
) -> list[dict]:
    """Quota-first stratified selection: fill scarce event types before abundant ones.

    Algorithm:
    1. Group candidates by event_type.
    2. Sort event types by ascending raw count (scarcest first).
    3. For each type, round-robin stratify within (frame_bucket, layer_bucket) groups.
    4. Fill up to the type's quota.
    5. Optionally fill remaining slots with leftovers only when quota_fill_remaining=True.

    Args:
        candidate_events: All candidate events from probe phase.
        max_events: Total events to select.
        max_events_per_frame: Per-frame cap (same as existing policy).
        layer_bucket_width: Width of layer bucket for stratification.
        event_type_quotas: Per-type cap, e.g. {"eviction": 8, "dedup": 4, "fifo_topk": 4}.
        quota_fill_remaining: If False, strict per-type caps are respected. If True,
            unused capacity can be filled from leftovers after all quotas are attempted.
    """
    if not candidate_events or max_events <= 0:
        return []

    # Group by event type
    by_type: dict[str, list[dict]] = {}
    for event in candidate_events:
        et = event.get("event_type", "eviction")  # default for legacy events
        by_type.setdefault(et, []).append(event)

    # Sort types by ascending raw count (scarcest first)
    sorted_types = sorted(by_type.keys(), key=lambda t: len(by_type[t]))

    selected: list[dict] = []
    per_frame_counts: dict[int, int] = {}
    used_ids: set[int] = set()  # track by id() to avoid duplicates

    for event_type in sorted_types:
        type_candidates = by_type[event_type]
        type_quota = event_type_quotas.get(event_type, 0)
        if type_quota <= 0:
            continue  # this type has zero quota

        remaining_total = max_events - len(selected)
        type_budget = min(type_quota, remaining_total)
        if type_budget <= 0:
            continue

        # Stratify within this type by (frame_bucket, layer_bucket)
        groups: dict[tuple, list[dict]] = {}
        for event in type_candidates:
            if id(event) in used_ids:
                continue
            fid = int(event.get("frame_id", 0))
            lid = int(event.get("layer_id", 0))
            bucket = lid // layer_bucket_width
            key = (fid, bucket)
            groups.setdefault(key, []).append(event)

        # Round-robin across groups
        group_queues = [list(groups[k]) for k in sorted(groups.keys())]
        type_selected = 0

        while group_queues and type_selected < type_budget:
            next_round: list[list[dict]] = []
            for queue in group_queues:
                if not queue:
                    continue
                if type_selected >= type_budget:
                    break
                if len(selected) >= max_events:
                    break
                event = queue.pop(0)
                frame_id = int(event.get("frame_id", 0))
                if per_frame_counts.get(frame_id, 0) >= max_events_per_frame:
                    next_round.append(queue)
                    continue
                selected.append(event)
                used_ids.add(id(event))
                per_frame_counts[frame_id] = per_frame_counts.get(frame_id, 0) + 1
                type_selected += 1
                if queue:
                    next_round.append(queue)
            group_queues = next_round

    # Optional fill: strict mode leaves unused slots empty rather than exceeding per-type caps.
    remaining = max_events - len(selected)
    if quota_fill_remaining and remaining > 0:
        leftovers = [e for e in candidate_events if id(e) not in used_ids]
        if leftovers:
            fill_groups: dict[tuple, list[dict]] = {}
            for event in leftovers:
                et = event.get("event_type", "eviction")
                fid = int(event.get("frame_id", 0))
                lid = int(event.get("layer_id", 0))
                bucket = lid // layer_bucket_width
                key = (et, fid, bucket)
                fill_groups.setdefault(key, []).append(event)

            group_queues = [list(fill_groups[k]) for k in sorted(fill_groups.keys())]
            while group_queues and len(selected) < max_events:
                next_round: list[list[dict]] = []
                for queue in group_queues:
                    if not queue:
                        continue
                    if len(selected) >= max_events:
                        break
                    event = queue.pop(0)
                    frame_id = int(event.get("frame_id", 0))
                    if per_frame_counts.get(frame_id, 0) >= max_events_per_frame:
                        next_round.append(queue)
                        continue
                    selected.append(event)
                    per_frame_counts[frame_id] = per_frame_counts.get(frame_id, 0) + 1
                    if queue:
                        next_round.append(queue)
                group_queues = next_round

    return selected
```

- [ ] **Step 9: Add `event_type_quotas` to the plumbing chain**

a) `FrontendOracleCollectorConfig` — add field:

```python
# --- Quota-based event selection ---
event_type_quotas: dict[str, int] | None = None
quota_fill_remaining: bool = False
```

b) `tools/collect_counterfactual_oracle.py` — add CLI arg:

```python
parser.add_argument(
    "--event-type-quotas",
    type=str,
    default=None,
    help="Per-event-type quota for quota_stratified policy, e.g. 'eviction=8,dedup=4,fifo_topk=4'.",
)
parser.add_argument(
    "--quota-fill-remaining",
    action="store_true",
    help="Allow quota_stratified to fill unused slots after strict event-type quotas are attempted.",
)
```

Parse in `main()`:

```python
event_type_quotas = None
if args.event_type_quotas:
    event_type_quotas = {
        k: int(v)
        for k, v in (pair.split("=") for pair in args.event_type_quotas.split(","))
    }
```

Pass to `FrontendOracleCollectorConfig(..., event_type_quotas=event_type_quotas, quota_fill_remaining=bool(args.quota_fill_remaining))`.

c) `collect_oracle_events_from_sequence` — add `event_type_quotas: dict[str, int] | None = None` and `quota_fill_remaining: bool = False`, pass to `select_oracle_events(..., event_type_quotas=event_type_quotas, quota_fill_remaining=quota_fill_remaining)`.

d) `collect_oracle_events_from_loader` — add `event_type_quotas: dict[str, int] | None = None` and `quota_fill_remaining: bool = False`, pass through.

e) `collect_oracle_shard_from_config` — read `collector_cfg.event_type_quotas` and `collector_cfg.quota_fill_remaining`, pass through.

Add `"event_type_quotas"` and `"quota_fill_remaining"` to `_YAML_TO_ARG` mapping in the CLI.

- [ ] **Step 10: Verify tests pass**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_quota_stratified_prefers_scarce_event_types tests/test_frontend_oracle_collector.py::test_quota_stratified_dedup_cannot_consume_all_slots tests/test_frontend_oracle_collector.py::test_quota_stratified_respects_frame_layer_diversity tests/test_frontend_oracle_collector.py::test_quota_stratified_without_quotas_falls_back_to_stratified -q
```

Expected: PASS.

- [ ] **Step 11: Verify against GPU3 evidence**

Run a small replay collection with the same profile that produced `/tmp/oracle_collection_gpu3.log`, but with quota selection:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_debug/quota_eviction_smoke.pt \
  --max-batches 8 \
  --max-events-per-sequence 16 \
  --event-selection-policy quota_stratified \
  --event-type-quotas eviction=8,dedup=4,fifo_topk=4
```

Expected: final shard contains nonzero `eviction` events. If probe logs show eviction candidates but final shard still has none, stop and inspect selector inputs before changing budget/profile knobs.

- [ ] **Step 12: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tools/collect_counterfactual_oracle.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add quota_stratified event selection policy

- Add event_type='eviction' to CounterfactualEvictionProbe events
- Implement quota_stratified: scarce event types selected first
- Add event_type_quotas config field and --event-type-quotas CLI arg
- Update --event-selection-policy choices to include quota_stratified
- Backward compatible: existing policies unchanged, quotas default to None"
```

---

## Task 3: Add Keyframe/FIFO Trigger Overrides

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

**Important note on anchor overrides:** `_run_frontend_with_probe` currently calls `model.inference(frames, move_to_cpu=False, cache_results=cache_results, return_views=False)`. In the current codebase, `OVGGT.inference(...)` already accepts `history_anchor_strategy`, `anchor_interval`, and `max_anchors` directly ([`src/ovggt/models/ovggt.py`](../../src/ovggt/models/ovggt.py)). Do not mutate `aggregator` or `frontend_cache` attributes for this feature; pass the overrides directly into `model.inference(...)`. The existing `_snapshot_model_runtime_state` / `_restore_model_runtime_state` still handles runtime score/probe state as before.

- [ ] **Step 1: Write failing test**

```python
def test_collector_respects_oracle_anchor_interval():
    """Collector passes oracle anchor overrides directly to model.inference."""
    import torch
    from types import SimpleNamespace

    model = SimpleNamespace(inference_calls=[])

    def fake_inference(frames, **kwargs):
        model.inference_calls.append(kwargs)
        return {}

    model.inference = fake_inference

    frames = [{"img": torch.zeros(1, 3, 2, 2)} for _ in range(3)]
    from ovggt.training.frontend_oracle_collector import collect_oracle_events_from_sequence

    collect_oracle_events_from_sequence(
        model=model,
        frames=frames,
        device=torch.device("cpu"),
        max_events=16,
        num_samples=4,
        oracle_window=2,
        oracle_anchor_interval=4,
        oracle_max_anchors=2,
        probe_only=True,
        probe_diagnostics_sink=[],
    )

    assert model.inference_calls, "Expected _run_frontend_with_probe to call model.inference"
    call_kwargs = model.inference_calls[0]
    assert call_kwargs["history_anchor_strategy"] == "fixed_interval"
    assert call_kwargs["anchor_interval"] == 4
    assert call_kwargs["max_anchors"] == 2
```

- [ ] **Step 2: Add collector config fields**

In `FrontendOracleCollectorConfig`, add after `probe_only`:

```python
# --- Keyframe/anchor overrides for FIFO trigger ---
oracle_anchor_interval: int | None = None
oracle_max_anchors: int | None = None
```

- [ ] **Step 3: Add CLI args**

```python
parser.add_argument(
    "--oracle-anchor-interval",
    type=int,
    default=None,
    help="Override anchor_interval for oracle collection only.",
)
parser.add_argument(
    "--oracle-max-anchors",
    type=int,
    default=None,
    help="Override max_anchors for oracle collection only.",
)
```

Wire both args into `FrontendOracleCollectorConfig`. Add both fields to `_YAML_TO_ARG`.

- [ ] **Step 4: Pass anchor overrides directly through `_run_frontend_with_probe`**

Extend `_run_frontend_with_probe` to accept optional overrides:

```python
def _run_frontend_with_probe(
    model, frames, probe, cache_results: bool,
    dedup_probe=None, fifo_probe=None, dedup_replay_probe=None,
    oracle_anchor_interval: int | None = None,
    oracle_max_anchors: int | None = None,
):
```

Before calling `model.inference(...)`, build kwargs only for provided overrides:

```python
    inference_kwargs = {
        "move_to_cpu": False,
        "cache_results": cache_results,
        "return_views": False,
    }
    if oracle_anchor_interval is not None:
        inference_kwargs["history_anchor_strategy"] = "fixed_interval"
        inference_kwargs["anchor_interval"] = int(oracle_anchor_interval)
    if oracle_max_anchors is not None:
        inference_kwargs["max_anchors"] = int(oracle_max_anchors)
    try:
        return model.inference(frames, **inference_kwargs)
    finally:
        # existing runtime/probe restore logic remains unchanged
        ...
```

Thread `oracle_anchor_interval` and `oracle_max_anchors` through the plumbing chain:
- `collect_oracle_events_from_sequence` → all calls to `_run_frontend_with_probe`
- `collect_oracle_events_from_loader` → `collect_oracle_events_from_sequence`
- `collect_oracle_shard_from_config` → `collect_oracle_events_from_loader`

- [ ] **Step 5: Verify FIFO probe production**

Run a small probe-only smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --output checkpoints/token_oracle_probe/fifo_probe.pt \
  --probe-only \
  --max-batches 2 \
  --max-events-per-sequence 16 \
  --oracle-profile fifo_topk \
  --fifo-keep-topk-override 32 \
  --fifo-count-candidates-for-oracle 0,8,16,32,64,128 \
  --oracle-anchor-interval 4 \
  --oracle-max-anchors 2 \
  --layers-per-frame 4 \
  --event-selection-policy quota_stratified
```

Expected: probe diagnostics show nonzero `fifo_topk` candidates. If zero, inspect keyframe schedule in diagnostics before changing replay settings.

**Fallback if FIFO candidates remain zero:** Try progressively lower anchor intervals (3, 2) and higher `fifo_keep_topk_override` values (64, 128). If still zero after `anchor_interval=2`, the model may not produce FIFO_SWAP events at all with this checkpoint — in that case, document the finding and skip FIFO profile in production rollout.

- [ ] **Step 6: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tools/collect_counterfactual_oracle.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add keyframe/anchor overrides for FIFO trigger

- Add oracle_anchor_interval and oracle_max_anchors
- Pass anchor params directly to model.inference during collection
- Add --oracle-anchor-interval, --oracle-max-anchors CLI args"
```

---

## Task 4: Add Low-Budget Eviction Profile

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `config/collect_counterfactual_oracle.yaml`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write failing test**

```python
def test_low_budget_eviction_profile_applies_budget():
    """oracle_profile=low_budget_eviction uses the lowered per-layer budget."""
    from ovggt.training.frontend_oracle_collector import load_frontend_oracle_config

    cfg = load_frontend_oracle_config(
        "config/collect_counterfactual_oracle.yaml",
        collector_cfg=FrontendOracleCollectorConfig(
            config="config/collect_counterfactual_oracle.yaml",
            output="/tmp/test.pt",
            oracle_profile="low_budget_eviction",
            frontend_per_layer_budget_override=2500,
        ),
    )
    assert int(cfg.frontend_per_layer_budget) == 2500, (
        f"Expected budget 2500, got {cfg.frontend_per_layer_budget}"
    )
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_low_budget_eviction_profile_applies_budget -q
```

Expected: PASS — existing override path at `load_frontend_oracle_config` lines 1084-1088 already handles this correctly.

- [ ] **Step 3: Prefer the GPU3-proven budget before sweeping lower**

The GPU3 log proves `frontend_per_layer_budget_override=2500` already creates eviction candidates: 564 raw eviction candidates across 128 batches. Start with 2500 plus quota selection before testing lower budgets.

- [ ] **Step 4: Add recommended low-budget profile command**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/train_frontend_finetune.yaml \
  --output checkpoints/token_oracle_phase2/eviction_shard_000.pt \
  --max-batches 64 \
  --max-events 2048 \
  --max-events-per-sequence 12 \
  --max-candidate-events-per-sequence 512 \
  --event-selection-policy quota_stratified \
  --event-type-quotas eviction=8,dedup=2,fifo_topk=2 \
  --oracle-profile low_budget_eviction \
  --frontend-per-layer-budget-override 2500 \
  --layers-per-frame 4 \
  --subset-replay-batch-size 8
```

Expected: nonzero measured eviction events. If raw eviction candidates remain nonzero but measured eviction is zero, the selector is still wrong. Only run lower-budget sweeps if raw eviction candidates are too rare after selector fixes.

- [ ] **Step 5: Commit**

```bash
git add config/collect_counterfactual_oracle.yaml tests/test_frontend_oracle_collector.py
git commit -m "test(oracle): verify low_budget_eviction profile applies budget override

- Confirm existing override path works at budget=2500
- GPU3 log proves this budget creates eviction candidates"
```

---

## Task 5: Make Layer and Frame Coverage Explicit

**Files:**
- Modify: `src/ovggt/training/frontend_oracle_collector.py`
- Modify: `tools/collect_counterfactual_oracle.py`
- Test: `tests/test_frontend_oracle_collector.py`

- [ ] **Step 1: Write failing test for layer schedule modes**

```python
def test_random_bucket_layer_schedule_covers_all_buckets():
    """random_bucket layer schedule should produce candidates across all layer buckets."""
    from ovggt.training.frontend_oracle_collector import should_record_oracle_layer_with_schedule

    num_layers = 24
    bucket_width = 6
    frames_tested = list(range(8))

    selected_layers: set[int] = set()
    for frame_id in frames_tested:
        for layer_id in range(num_layers):
            if should_record_oracle_layer_with_schedule(
                layer_id=layer_id,
                frame_id=frame_id,
                schedule="random_bucket",
                num_layers=num_layers,
                layer_buckets=[[0, 5], [6, 11], [12, 17], [18, 23]],
                seed=42,
            ):
                selected_layers.add(layer_id)

    buckets_hit = set(l // bucket_width for l in selected_layers)
    assert len(buckets_hit) >= 3, (
        f"Expected coverage across >=3 layer buckets, got {len(buckets_hit)}: {buckets_hit}"
    )


def test_rotating_stride_layer_schedule():
    """rotating_stride should cycle through different layer pairs across frames."""
    from ovggt.training.frontend_oracle_collector import should_record_oracle_layer_with_schedule

    num_layers = 24
    layers_per_frame = 2

    frame0_layers: set[int] = set()
    frame1_layers: set[int] = set()
    for layer_id in range(num_layers):
        if should_record_oracle_layer_with_schedule(
            layer_id=layer_id, frame_id=0,
            schedule="rotating_stride", num_layers=num_layers,
            layers_per_frame=layers_per_frame, seed=0,
        ):
            frame0_layers.add(layer_id)
        if should_record_oracle_layer_with_schedule(
            layer_id=layer_id, frame_id=1,
            schedule="rotating_stride", num_layers=num_layers,
            layers_per_frame=layers_per_frame, seed=0,
        ):
            frame1_layers.add(layer_id)

    assert frame0_layers != frame1_layers, (
        f"Frame 0 and frame 1 selected same layers: {frame0_layers}"
    )
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_random_bucket_layer_schedule_covers_all_buckets tests/test_frontend_oracle_collector.py::test_rotating_stride_layer_schedule -q
```

Expected: FAIL — `should_record_oracle_layer_with_schedule` does not exist.

- [ ] **Step 3: Implement layer schedule modes**

Add `should_record_oracle_layer_with_schedule` function:

```python
def should_record_oracle_layer_with_schedule(
    layer_id: int,
    frame_id: int,
    schedule: str = "default",
    num_layers: int = 24,
    layers_per_frame: int = 2,
    layer_buckets: list[list[int]] | None = None,
    seed: int = 0,
) -> bool:
    """Determine if a layer should be recorded, using the specified schedule mode.

    Schedule modes:
    - "default": use the existing should_record_oracle_layer logic (stride-based).
    - "rotating_stride": like default but offset rotates with frame_id.
    - "random_bucket": deterministically select one layer per bucket per frame.
    - "fixed_buckets": select layers from pre-defined bucket ranges.
    """
    if schedule == "default":
        return should_record_oracle_layer(
            layer_id=layer_id, frame_id=frame_id,
            layers_per_frame=layers_per_frame, num_layers=num_layers,
        )

    if schedule == "rotating_stride":
        if num_layers <= 0 or layers_per_frame <= 0:
            return True
        offset = frame_id % max(num_layers // layers_per_frame, 1)
        stride = max(num_layers // layers_per_frame, 1)
        selected = {(offset + s * stride) % num_layers for s in range(layers_per_frame)}
        return layer_id in selected

    if schedule == "random_bucket":
        if layer_buckets is None:
            layer_buckets = [[i, min(i + 6, num_layers - 1)] for i in range(0, num_layers, 6)]
        import random as _rng
        state = _rng.Random(seed + frame_id * 1000)
        selected: set[int] = set()
        for bucket_range in layer_buckets:
            lo, hi = int(bucket_range[0]), int(bucket_range[1])
            chosen = state.randint(lo, hi)
            selected.add(chosen)
        return layer_id in selected

    if schedule == "fixed_buckets":
        if layer_buckets is None:
            return True
        for bucket_range in layer_buckets:
            lo, hi = int(bucket_range[0]), int(bucket_range[1])
            if lo <= layer_id <= hi:
                return True
        return False

    return True  # unknown schedule → record everything
```

- [ ] **Step 4: Verify layer schedule tests pass**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_random_bucket_layer_schedule_covers_all_buckets tests/test_frontend_oracle_collector.py::test_rotating_stride_layer_schedule -q
```

Expected: PASS.

- [ ] **Step 5: Write failing test for frame bucket selection**

```python
def test_frame_bucket_selection_prefers_later_frames():
    """When frame_buckets are set, selection should include mid/late frame candidates."""
    candidates = [
        # 9 early dedup events (3 frames x 3 groups)
        *({"event_type": "dedup", "frame_id": f, "layer_id": 0, "voxel_group_id": f} for f in range(0, 3)),
        *({"event_type": "dedup", "frame_id": f, "layer_id": 0, "voxel_group_id": 100 + f} for f in range(0, 3)),
        *({"event_type": "dedup", "frame_id": f, "layer_id": 0, "voxel_group_id": 200 + f} for f in range(0, 3)),
        # 2 late eviction events
        {"event_type": "eviction", "frame_id": 10, "layer_id": 0, "voxel_group_id": 300},
        {"event_type": "eviction", "frame_id": 15, "layer_id": 0, "voxel_group_id": 301},
    ]

    selected = select_oracle_events(
        candidates,
        max_events=6,
        max_events_per_frame=6,
        policy="quota_stratified",
        layer_bucket_width=6,
        event_type_quotas={"eviction": 4, "dedup": 2},
        frame_buckets=[(0, 3), (4, 8), (9, 23)],
    )

    frames = [e["frame_id"] for e in selected]
    late_frames = [f for f in frames if f >= 9]
    assert len(late_frames) > 0, (
        f"Expected some late-frame events (frame >= 9), got frames: {frames}"
    )
```

- [ ] **Step 6: Run test and verify it fails**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_frame_bucket_selection_prefers_later_frames -q
```

Expected: FAIL — `frame_buckets` parameter does not exist on `select_oracle_events`.

- [ ] **Step 7: Add frame bucket config and plumbing**

Add to `FrontendOracleCollectorConfig`:

```python
# --- Layer and frame coverage ---
oracle_layer_schedule: str = "default"  # default | rotating_stride | random_bucket | fixed_buckets
oracle_layer_buckets: str | None = None  # JSON string, e.g. '[[0,5],[6,11],[12,17],[18,23]]'
frame_buckets: str | None = None  # JSON string, e.g. '[[0,3],[4,8],[9,23]]'
```

Add CLI args:

```python
parser.add_argument(
    "--oracle-layer-schedule",
    type=str, default="default",
    choices=["default", "rotating_stride", "random_bucket", "fixed_buckets"],
    help="Layer selection schedule for oracle collection.",
)
parser.add_argument(
    "--oracle-layer-buckets",
    type=str, default=None,
    help="JSON list of layer bucket ranges, e.g. '[[0,5],[6,11],[12,17],[18,23]]'.",
)
parser.add_argument(
    "--frame-buckets",
    type=str, default=None,
    help="JSON list of frame bucket ranges for quota selection, e.g. '[[0,3],[4,8],[9,23]]'.",
)
```

Thread through the plumbing chain explicitly:

- Parse `oracle_layer_buckets` and `frame_buckets` with `json.loads(...)` in `tools/collect_counterfactual_oracle.py`.
- Add `oracle_layer_schedule`, parsed `oracle_layer_buckets`, and a deterministic `oracle_layer_schedule_seed` to `CounterfactualEvictionProbe`, `CounterfactualDedupProbe`, and `CounterfactualFifoTopKProbe`.
- In all three probe `on_*_candidate(...)` methods, replace the current `should_record_oracle_layer(...)` call with `should_record_oracle_layer_with_schedule(...)`.
- Add `frame_buckets: list[tuple[int, int]] | None = None` to `select_oracle_events(...)` and `_select_oracle_events_quota_stratified(...)`.
- Add helper:

```python
def _frame_bucket_id(frame_id: int, frame_buckets: list[tuple[int, int]] | None) -> int:
    if not frame_buckets:
        return int(frame_id)
    for idx, (lo, hi) in enumerate(frame_buckets):
        if int(lo) <= int(frame_id) <= int(hi):
            return idx
    return len(frame_buckets)
```

- Inside `_select_oracle_events_quota_stratified(...)`, use `(frame_bucket, layer_bucket)` rather than `(frame_id, layer_bucket)` as the within-type group key when `frame_buckets` is provided.

**Note on `late_frame_scan` profile:** The `late_frame_scan` profile from the strategy table is implemented via these settings:

```bash
--oracle-layer-schedule random_bucket \
--frame-buckets '[[4,8],[9,23]]' \
--event-type-quotas eviction=4,dedup=2,fifo_topk=2
```

No separate profile entry is needed — it's a configuration combination.

- [ ] **Step 8: Verify frame bucket test passes**

```bash
python -m pytest tests/test_frontend_oracle_collector.py::test_frame_bucket_selection_prefers_later_frames -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/ovggt/training/frontend_oracle_collector.py tools/collect_counterfactual_oracle.py tests/test_frontend_oracle_collector.py
git commit -m "feat(oracle): add layer schedule modes and frame bucket selection

- Add rotating_stride, random_bucket, fixed_buckets layer schedules
- Add frame_buckets config for preferring later frames
- late_frame_scan profile implemented as a config combination"
```

---

## Task 6: Add Mixed-Profile Launcher

**Files:**
- Create: `tools/run_oracle_mixed_profiles.py`
- Test: `tests/test_frontend_oracle_parallel.py`

- [ ] **Step 1: Write test for generated jobs**

```python
def test_mixed_profile_generates_weighted_unique_shards():
    """Launcher should generate weighted shard jobs with unique outputs."""
    from tools.run_oracle_mixed_profiles import generate_profile_jobs

    jobs = generate_profile_jobs(
        base_config="config/train_frontend_finetune.yaml",
        output_dir="checkpoints/token_oracle_mixed/",
        num_gpus=4,
        num_shards=10,
        events_per_shard=2048,
        start_shard_id=20,
        profile_weights={
            "real_policy_dedup": 0.4,
            "low_budget_eviction": 0.3,
            "fifo_topk_forced": 0.3,
        },
    )
    assert len(jobs) == 10

    profile_counts = {}
    for job in jobs:
        profile_counts[job["profile"]] = profile_counts.get(job["profile"], 0) + 1
    assert profile_counts == {
        "real_policy_dedup": 4,
        "low_budget_eviction": 3,
        "fifo_topk_forced": 3,
    }

    outputs = {job["output"] for job in jobs}
    assert len(outputs) == len(jobs), "Each job must have a unique output path"
    assert any("shard_020" in out for out in outputs)
    assert any("shard_029" in out for out in outputs)

    for job in jobs:
        assert "--max-events" in job["args"]
        assert str(2048) in job["args"]


def test_mixed_profile_can_include_late_frame_scan():
    """late_frame_scan is optional and included when profile weights request it."""
    from tools.run_oracle_mixed_profiles import generate_profile_jobs

    jobs = generate_profile_jobs(
        base_config="config/train_frontend_finetune.yaml",
        output_dir="checkpoints/token_oracle_mixed/",
        num_gpus=2,
        num_shards=4,
        events_per_shard=128,
        profile_weights={
            "real_policy_dedup": 0.25,
            "low_budget_eviction": 0.25,
            "fifo_topk_forced": 0.25,
            "late_frame_scan": 0.25,
        },
    )
    profiles = {job["profile"] for job in jobs}
    assert profiles == {
        "real_policy_dedup",
        "low_budget_eviction",
        "fifo_topk_forced",
        "late_frame_scan",
    }
```

- [ ] **Step 2: Implement launcher**

Create `tools/run_oracle_mixed_profiles.py`:

```python
#!/usr/bin/env python
"""Launch oracle collection with mixed profiles across GPUs."""
import argparse
import json
import os
import subprocess
import time

PYTHON = "/mnt/lyj/miniconda3/envs/streamvggt/bin/python"

DEFAULT_PROFILE_WEIGHTS = {
    "real_policy_dedup": 0.4,
    "low_budget_eviction": 0.3,
    "fifo_topk_forced": 0.3,
}


def parse_profile_weights(raw: str | None) -> dict[str, float]:
    """Parse profile weights like 'real_policy_dedup=0.4,low_budget_eviction=0.3'."""
    if not raw:
        return dict(DEFAULT_PROFILE_WEIGHTS)
    weights: dict[str, float] = {}
    for item in raw.split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def allocate_profile_counts(num_shards: int, profile_weights: dict[str, float]) -> dict[str, int]:
    """Allocate integer shard counts using largest remainder, preserving total exactly."""
    if num_shards <= 0:
        return {}
    total_weight = sum(float(w) for w in profile_weights.values() if float(w) > 0)
    if total_weight <= 0:
        raise ValueError("At least one profile weight must be positive")

    raw_counts = {
        profile: num_shards * float(weight) / total_weight
        for profile, weight in profile_weights.items()
        if float(weight) > 0
    }
    counts = {profile: int(count) for profile, count in raw_counts.items()}
    remaining = num_shards - sum(counts.values())
    remainders = sorted(
        raw_counts.keys(),
        key=lambda profile: (raw_counts[profile] - counts[profile], profile),
        reverse=True,
    )
    for profile in remainders[:remaining]:
        counts[profile] += 1
    return counts


def profile_args(profile: str, events_per_shard: int) -> list[str]:
    """Return collector args for one named profile."""
    common = ["--max-events", str(events_per_shard)]
    if profile == "real_policy_dedup":
        return [
            "--oracle-profile", "real_policy",
            "--event-type-quotas", "dedup=8,eviction=0,fifo_topk=0",
            *common,
        ]
    if profile == "low_budget_eviction":
        return [
            "--oracle-profile", "low_budget_eviction",
            "--frontend-per-layer-budget-override", "2500",
            "--event-type-quotas", "eviction=8,dedup=2,fifo_topk=0",
            *common,
        ]
    if profile == "fifo_topk_forced":
        return [
            "--oracle-profile", "fifo_topk",
            "--fifo-keep-topk-override", "32",
            "--oracle-anchor-interval", "4",
            "--oracle-max-anchors", "2",
            "--event-type-quotas", "fifo_topk=8,dedup=2,eviction=2",
            *common,
        ]
    if profile == "late_frame_scan":
        return [
            "--oracle-profile", "real_policy",
            "--oracle-layer-schedule", "random_bucket",
            "--frame-buckets", json.dumps([[4, 8], [9, 23]]),
            "--event-type-quotas", "eviction=4,dedup=2,fifo_topk=2",
            *common,
        ]
    raise ValueError(f"Unknown profile: {profile}")


def generate_profile_jobs(
    base_config: str,
    output_dir: str,
    num_gpus: int,
    num_shards: int,
    events_per_shard: int = 2048,
    start_shard_id: int = 0,
    profile_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Generate one job dict per shard, allocated by profile weights."""
    os.makedirs(output_dir, exist_ok=True)

    weights = profile_weights or dict(DEFAULT_PROFILE_WEIGHTS)
    profile_counts = allocate_profile_counts(num_shards, weights)
    jobs: list[dict] = []
    shard_id = int(start_shard_id)
    for profile, count in profile_counts.items():
        for _ in range(count):
            output = os.path.join(output_dir, f"{profile}_shard_{shard_id:03d}.pt")
            jobs.append(
                {
                    "profile": profile,
                    "shard_id": shard_id,
                    "output": output,
                    "args": profile_args(profile, events_per_shard),
                }
            )
            shard_id += 1
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--events-per-shard", type=int, default=2048)
    parser.add_argument("--start-shard-id", type=int, default=0)
    parser.add_argument(
        "--profile-weights",
        default=None,
        help="Comma-separated profile weights. Default: real_policy_dedup=0.4,low_budget_eviction=0.3,fifo_topk_forced=0.3",
    )
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = generate_profile_jobs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_shards=args.num_shards,
        events_per_shard=args.events_per_shard,
        start_shard_id=args.start_shard_id,
        profile_weights=parse_profile_weights(args.profile_weights),
    )

    def build_command(job: dict) -> list[str]:
        return [
            PYTHON, "-u", "tools/collect_counterfactual_oracle.py",
            "--config", args.base_config,
            "--output", job["output"],
            "--max-batches", str(args.max_batches),
            "--event-selection-policy", "quota_stratified",
            "--layers-per-frame", "4",
            *job["args"],
        ]

    if args.dry_run:
        for i, job in enumerate(jobs):
            gpu_id = i % args.num_gpus
            cmd = build_command(job)
            print(f"\n# Profile: {job['profile']} (GPU {gpu_id})")
            print(" \\\n  ".join(["CUDA_VISIBLE_DEVICES=" + str(gpu_id)] + cmd))
        return

    # Slot scheduler: at most one active collector per physical GPU.
    pending = list(jobs)
    running: dict[int, tuple[subprocess.Popen, object, str]] = {}
    failures: list[tuple[str, int]] = []

    while pending or running:
        for gpu_id in range(args.num_gpus):
            if gpu_id in running or not pending:
                continue
            job = pending.pop(0)
            cmd = build_command(job)
            log_path = os.path.splitext(job["output"])[0] + ".log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Launching profile={job['profile']} gpu={gpu_id} log={log_path}")
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running[gpu_id] = (proc, log_f, job["profile"])

        time.sleep(10)

        for gpu_id, (proc, log_f, profile) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_f.close()
            del running[gpu_id]
            if rc != 0:
                failures.append((profile, rc))

    if failures:
        raise SystemExit(f"Failed jobs: {failures}")
```

- [ ] **Step 3: Keep per-GPU concurrency at 1**

Previous benchmark showed multiple collectors on one GPU were slower because SM compute was saturated. The launcher uses a slot scheduler with one active `subprocess.Popen(...)` per GPU. If there are more jobs than GPUs, it waits for a GPU slot to finish before launching the next job on that GPU.

- [ ] **Step 4: Commit**

```bash
git add tools/run_oracle_mixed_profiles.py tests/test_frontend_oracle_parallel.py
git commit -m "feat(oracle): add mixed-profile launcher for multi-GPU collection

- Generate weighted shard jobs with unique output paths
- Default allocation is 40% dedup, 30% eviction, 30% FIFO
- One collector per GPU (concurrency=1 per GPU)
- Add --num-shards, --events-per-shard, --profile-weights, and --dry-run"
```

---

## Task 7: Add Diversity Summarizer

**Files:**
- Create: `tools/summarize_oracle_event_diversity.py`
- Test: `tests/test_token_oracle_dataset.py`

- [ ] **Step 1: Write failing test**

```python
def test_summarizer_detects_all_dedup_shard():
    """Summarizer should report 100% dedup and flag diversity failure."""
    from tools.summarize_oracle_event_diversity import summarize_shard, check_diversity_thresholds

    shard = {
        "events": [
            {"event_type": "dedup", "frame_id": 0, "layer_id": 0, "sequence_provenance": {"dataset": "test"}},
            {"event_type": "dedup", "frame_id": 1, "layer_id": 1, "sequence_provenance": {"dataset": "test"}},
        ],
    }
    summary = summarize_shard(shard)
    assert summary["event_type_counts"]["dedup"] == 2
    assert summary["event_type_counts"].get("eviction", 0) == 0

    issues = check_diversity_thresholds(summary, profile="low_budget_eviction")
    assert any("eviction" in issue for issue in issues), f"Expected eviction diversity issue, got: {issues}"


def test_summarizer_parses_gpu3_style_log(tmp_path):
    """Log summarizer should parse raw probe counts and final shard summary."""
    from tools.summarize_oracle_event_diversity import summarize_log

    log_path = tmp_path / "oracle_collection_gpu3.log"
    log_path.write_text(
        "\n".join(
            [
                "[oracle] batch0: probe captured 2 eviction, 20 dedup, 0 fifo candidates, 16 total have future frames",
                '[oracle] flushed final shard summary={"num_events": 16, '
                '"event_type_counts": {"dedup": 16}, '
                '"frame_histogram": {"0": 8, "1": 8}, '
                '"layer_histogram": {"0": 8, "12": 8}, '
                '"elapsed_sec": 100.0}',
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_log(log_path)
    assert summary["raw_event_type_counts"] == {"eviction": 2, "dedup": 20, "fifo_topk": 0}
    assert summary["event_type_counts"] == {"dedup": 16}
    assert summary["total_events"] == 16
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python -m pytest tests/test_token_oracle_dataset.py::test_summarizer_detects_all_dedup_shard tests/test_token_oracle_dataset.py::test_summarizer_parses_gpu3_style_log -q
```

Expected: FAIL — `summarize_oracle_event_diversity` module does not exist.

- [ ] **Step 3: Implement summary**

Create `tools/summarize_oracle_event_diversity.py`:

```python
#!/usr/bin/env python
"""Summarize oracle event diversity across logs and shards."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch


PROBE_RE = re.compile(
    r"probe captured (?P<eviction>\d+) eviction, "
    r"(?P<dedup>\d+) dedup, (?P<fifo>\d+) fifo candidates, "
    r"(?P<selected>\d+) total"
)


def _frame_bucket_counts(frame_hist: dict) -> dict[str, int]:
    frame_buckets = {"early": 0, "mid": 0, "late": 0}
    for fid_raw, count in frame_hist.items():
        fid = int(fid_raw)
        if fid <= 3:
            frame_buckets["early"] += int(count)
        elif fid <= 8:
            frame_buckets["mid"] += int(count)
        else:
            frame_buckets["late"] += int(count)
    return frame_buckets


def _layer_bucket_counts(layer_hist: dict, layer_bucket_width: int = 6) -> dict[int, int]:
    layer_bucket_counts: dict[int, int] = {}
    for lid_raw, count in layer_hist.items():
        bucket = int(lid_raw) // int(layer_bucket_width)
        layer_bucket_counts[bucket] = layer_bucket_counts.get(bucket, 0) + int(count)
    return layer_bucket_counts


def summarize_shard(shard: dict) -> dict:
    """Compute diversity summary from a loaded shard dict."""
    events = shard.get("events", [])

    et_counts: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "eviction")  # legacy fallback
        et_counts[et] = et_counts.get(et, 0) + 1

    frame_hist: dict[int, int] = {}
    layer_hist: dict[int, int] = {}
    ds_counts: dict[str, int] = {}
    for e in events:
        fid = int(e.get("frame_id", 0))
        lid = int(e.get("layer_id", 0))
        frame_hist[fid] = frame_hist.get(fid, 0) + 1
        layer_hist[lid] = layer_hist.get(lid, 0) + 1
        prov = e.get("sequence_provenance") or {}
        ds = prov.get("dataset", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

    total = len(events)

    return {
        "total_events": total,
        "event_type_counts": dict(sorted(et_counts.items())),
        "frame_histogram": dict(sorted(frame_hist.items())),
        "frame_bucket_counts": dict(sorted(_frame_bucket_counts(frame_hist).items())),
        "layer_histogram": dict(sorted(layer_hist.items())),
        "layer_bucket_counts": dict(sorted(_layer_bucket_counts(layer_hist).items())),
        "dataset_counts": dict(sorted(ds_counts.items())),
        "raw_event_type_counts": {},
    }


def summarize_log(path: str | Path) -> dict:
    """Parse collector logs, including /tmp/oracle_collection_gpu3.log style output."""
    path = Path(path)
    raw_counts = {"eviction": 0, "dedup": 0, "fifo_topk": 0}
    probe_batches = 0
    selected_candidate_events = 0
    final_summary = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = PROBE_RE.search(line)
        if m:
            probe_batches += 1
            raw_counts["eviction"] += int(m.group("eviction"))
            raw_counts["dedup"] += int(m.group("dedup"))
            raw_counts["fifo_topk"] += int(m.group("fifo"))
            selected_candidate_events += int(m.group("selected"))
        if "flushed final shard" in line and "summary=" in line:
            try:
                final_summary = json.loads(line.split("summary=", 1)[1].strip())
            except json.JSONDecodeError:
                pass

    final_summary = final_summary or {}
    event_type_counts = final_summary.get("event_type_counts", {})
    frame_hist = final_summary.get("frame_histogram", {})
    layer_hist = final_summary.get("layer_histogram", {})
    total = int(final_summary.get("num_events", sum(int(v) for v in event_type_counts.values())))

    return {
        "total_events": total,
        "event_type_counts": dict(event_type_counts),
        "frame_histogram": dict(frame_hist),
        "frame_bucket_counts": dict(sorted(_frame_bucket_counts(frame_hist).items())),
        "layer_histogram": dict(layer_hist),
        "layer_bucket_counts": dict(sorted(_layer_bucket_counts(layer_hist).items())),
        "dataset_counts": final_summary.get("dataset_counts", {}),
        "raw_event_type_counts": raw_counts,
        "probe_batches": probe_batches,
        "selected_candidate_events": selected_candidate_events,
        "elapsed_sec": final_summary.get("elapsed_sec", 0.0),
    }


def check_diversity_thresholds(summary: dict, profile: str = "") -> list[str]:
    """Check diversity thresholds. Returns list of issue strings (empty = pass)."""
    issues: list[str] = []
    total = summary["total_events"]
    if total == 0:
        issues.append("NO_EVENTS: shard has zero events")
        return issues

    et = summary["event_type_counts"]
    dedup_pct = et.get("dedup", 0) / total * 100

    if dedup_pct > 70 and "dedup" not in profile:
        issues.append(f"DEDUP_DOMINANCE: dedup is {dedup_pct:.1f}% of events (threshold: 70%)")

    if "fifo" in profile and et.get("fifo_topk", 0) == 0:
        issues.append("NO_FIFO: fifo_topk events are zero in FIFO profile")

    if "eviction" in profile and et.get("eviction", 0) == 0:
        issues.append("NO_EVICTION: eviction events are zero in eviction profile")

    fb = summary.get("frame_bucket_counts", {})
    if fb.get("early", 0) == total and "dedup" not in profile:
        issues.append("ALL_EARLY_FRAMES: all events are in early frames (0-3)")

    lb = summary.get("layer_bucket_counts", {})
    if len(lb) <= 2:
        issues.append(f"NARROW_LAYERS: only {len(lb)} layer buckets covered (threshold: >2)")

    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Paths to .pt shard files or .log files")
    parser.add_argument("--profile", default="", help="Profile name for threshold checks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    all_summaries: dict[str, dict] = {}
    all_issues: dict[str, list[str]] = {}
    for path_str in args.inputs:
        path = Path(path_str)
        if path.suffix == ".pt":
            shard = torch.load(path, map_location="cpu")
            summary = summarize_shard(shard)
        elif path.suffix == ".log":
            summary = summarize_log(path)
        else:
            raise ValueError(f"Unsupported input type: {path}")
        issues = check_diversity_thresholds(summary, profile=args.profile)
        all_summaries[str(path)] = summary
        all_issues[str(path)] = issues

    if args.json:
        print(json.dumps({"summaries": all_summaries, "issues": all_issues}, indent=2, default=str))
    else:
        for path, summary in all_summaries.items():
            print(f"\n=== {path} ===")
            for key, val in summary.items():
                print(f"  {key}: {val}")
            if all_issues.get(path):
                print("  ISSUES:")
                for issue in all_issues[path]:
                    print(f"    FAIL {issue}")
            else:
                print("  PASS Diversity thresholds passed")

    has_issues = any(issues for issues in all_issues.values())
    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify tests pass**

```bash
python -m pytest tests/test_token_oracle_dataset.py::test_summarizer_detects_all_dedup_shard tests/test_token_oracle_dataset.py::test_summarizer_parses_gpu3_style_log -q
```

Expected: PASS.

- [ ] **Step 5: Verify on latest GPU3 log**

```bash
python tools/summarize_oracle_event_diversity.py /tmp/oracle_collection_gpu3.log --profile low_budget_eviction
```

Expected: report fails diversity thresholds and prints the current GPU3 diagnosis: raw eviction candidates exist, raw FIFO candidates are zero, measured events are all dedup, and measured frames/layers are concentrated in early buckets.

- [ ] **Step 6: Commit**

```bash
git add tools/summarize_oracle_event_diversity.py tests/test_token_oracle_dataset.py
git commit -m "feat(oracle): add diversity summarizer with acceptance thresholds

- Summarize event type, frame, layer, dataset coverage
- Check thresholds: dedup <= 70%, nonzero eviction/fifo in named profiles
- Exit code 1 if diversity issues detected"
```

---

## Task 8: Production Rollout

**Files:** none (operational procedures)

**Note:** This task is operational, not code. All code changes are in Tasks 1-7.

- [ ] **Step 1: Probe-only budget/keyframe sweep**

Run cheap probe-only sweeps first:

```bash
# Eviction quota smoke with the GPU3-proven budget
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
  --config config/collect_counterfactual_oracle.yaml \
  --output checkpoints/token_oracle_probe/eviction_quota_2500.pt \
  --max-batches 8 \
  --oracle-profile low_budget_eviction \
  --frontend-per-layer-budget-override 2500 \
  --layers-per-frame 4 \
  --event-selection-policy quota_stratified \
  --event-type-quotas eviction=8,dedup=4,fifo_topk=0

# FIFO trigger sweep
for interval in 8 6 4 3; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /mnt/lyj/miniconda3/envs/streamvggt/bin/python -u tools/collect_counterfactual_oracle.py \
    --config config/train_frontend_finetune.yaml \
    --output checkpoints/token_oracle_probe/fifo_interval_${interval}.pt \
    --probe-only \
    --max-batches 8 \
    --oracle-profile fifo_topk \
    --fifo-keep-topk-override 32 \
    --oracle-anchor-interval ${interval} \
    --oracle-max-anchors 2 \
    --layers-per-frame 4 \
    --event-selection-policy quota_stratified
done
```

- [ ] **Step 2: Choose fastest successful profiles**

Pick:

- first confirm quota selection replays the eviction candidates already seen at budget 2500
- largest anchor interval that still yields FIFO candidates
- smallest `layers_per_frame` that covers all layer buckets over multiple frames

- [ ] **Step 3: Launch mixed collection**

Suggested initial allocation:

- 40% real-policy dedup shards
- 30% low-budget eviction shards
- 30% FIFO shards

Adjust after the first 2-4 shards based on summarizer output.

```bash
python tools/run_oracle_mixed_profiles.py \
  --base-config config/train_frontend_finetune.yaml \
  --output-dir checkpoints/token_oracle_mixed/ \
  --num-gpus 4 \
  --num-shards 40 \
  --events-per-shard 2048 \
  --max-batches 128 \
  --profile-weights real_policy_dedup=0.4,low_budget_eviction=0.3,fifo_topk_forced=0.3
```

- [ ] **Step 4: Train with type-aware validation**

When training token scorer/count head, report validation metrics per event type. Do not trust aggregate rank accuracy if FIFO/eviction are underrepresented.

---

## Expected Outcome

Compared with `/tmp/oracle_collection_gpu3.log` (`2048/2048` measured dedup, 564 raw eviction candidates dropped by selection, zero raw FIFO candidates, frames mostly `0-2`, layers mostly `0-2/12-14`), this plan should produce:

- nonzero measured eviction events before launching long production shards
- nonzero FIFO candidates before expensive FIFO replay is launched
- measured shards with explicit type balance, not accidental dedup dominance
- broader frame and layer coverage without increasing replay count blindly
- faster iteration because probe-only sweeps are used to tune trigger knobs before production replay
