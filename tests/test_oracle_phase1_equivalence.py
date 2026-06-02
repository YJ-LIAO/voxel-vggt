"""Phase 1 equivalence: cap disabled must match original full-enumeration behavior."""
import pytest
import torch

from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe
from ovggt.utils.frontend_cache import LayerCacheState, TokenKind, TokenMetadata
from test_frontend_oracle_collector import _dedup_cache_state


def _reference_dedup_subsets(num_tokens, group_indices):
    """Pre-Phase-1 keep-one-in-N enumeration."""
    keep_sets = []
    group_set = set(int(i) for i in group_indices.tolist())
    for keep in group_indices.tolist():
        keep_tensor = torch.tensor(
            [idx for idx in range(num_tokens) if idx not in group_set or idx == keep],
            dtype=torch.long,
        )
        keep_tensor = keep_tensor.unique(sorted=True)
        keep_sets.append(keep_tensor)
    return keep_sets


@pytest.fixture
def large_voxel_cache():
    importance = [float(i) for i in range(50)]
    return _dedup_cache_state(num_tokens=50, importance=importance)


def test_cap_disabled_emits_same_keep_indices_as_full_enumeration(large_voxel_cache):
    state = large_voxel_cache
    scores = torch.arange(50, dtype=torch.float)
    probe = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=1000)
    probe.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    assert len(probe.events) == 1
    emitted = [s["keep_indices"] for s in probe.events[0]["candidate_subsets"]]
    group_indices = torch.arange(50)
    reference = _reference_dedup_subsets(num_tokens=50, group_indices=group_indices)
    assert len(emitted) == len(reference) == 50
    emitted_sets = [tuple(int(i) for i in k.tolist()) for k in emitted]
    reference_sets = [tuple(int(i) for i in k.tolist()) for k in reference]
    assert emitted_sets == reference_sets


def test_cap_disabled_emits_identical_score_state_and_metadata(large_voxel_cache):
    state = large_voxel_cache
    scores = torch.arange(50, dtype=torch.float)
    probe_a = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=1000)
    probe_b = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=8)
    probe_a.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    probe_b.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    e_a, e_b = probe_a.events[0], probe_b.events[0]
    assert torch.equal(e_a["score_state"], e_b["score_state"])
    assert torch.equal(e_a["metadata_features"], e_b["metadata_features"])
    assert e_a["layer_id"] == e_b["layer_id"]
    assert e_a["frame_id"] == e_b["frame_id"]


def test_cap_8_emits_at_most_8_subsets(large_voxel_cache):
    scores = torch.arange(50, dtype=torch.float)
    probe = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=8)
    probe.on_dedup_candidate(cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    event = probe.events[0]
    assert len(event["candidate_subsets"]) <= 8
    assert len(event["candidate_subsets"]) >= 4


def test_cap_deterministic_with_same_seed(large_voxel_cache):
    scores = torch.arange(50, dtype=torch.float)
    p1 = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=8)
    p2 = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=8)
    p1.on_dedup_candidate(cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    p2.on_dedup_candidate(cache_state=large_voxel_cache, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=torch.arange(50))
    s1 = p1.events[0]["candidate_subsets"]
    s2 = p2.events[0]["candidate_subsets"]
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert torch.equal(a["keep_indices"], b["keep_indices"])


def test_policy_baseline_always_retained_for_dedup():
    importance = [float(i) for i in range(20)]
    state = _dedup_cache_state(num_tokens=20, importance=importance)
    scores = torch.arange(20, dtype=torch.float)
    policy_keep = torch.tensor([3, 7, 11, 50, 60])
    probe = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42, max_subsets_per_dedup_event=8)
    probe.on_dedup_candidate(cache_state=state, layer_id=0, frame_id=5, batch_index=0, scores=scores, policy_keep_indices=policy_keep)
    event = probe.events[0]
    baselines = [s for s in event["candidate_subsets"] if s.get("source") == "policy_baseline"]
    assert len(baselines) == 1
    assert torch.equal(baselines[0]["keep_indices"], policy_keep.long())
