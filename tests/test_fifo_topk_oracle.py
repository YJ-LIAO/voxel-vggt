"""Tests for FIFO top-K protection probe pre-mutation semantics.

Ensures that the probe sees the original demoted-slot token set before any
metadata mutation, even when keep_count=0 or keep_count >= demoted_token_count.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import copy

import torch
import pytest

from ovggt.utils.frontend_cache import LayerCacheState, TokenMetadata
from ovggt.training.frontend_oracle_collector import CounterfactualFifoTopKProbe
from ovggt.training.counterfactual_replay import (
    FifoTopKCounterfactualEvent,
    collect_fifo_topk_counterfactual_event,
)


def _make_layer_cache(
    num_tokens: int = 10,
    demoted_slot: int = 1,
    num_in_demoted_slot: int = 5,
    batch_size: int = 1,
    importance_values: list[float] | None = None,
    score_state_dim: int = 128,
) -> LayerCacheState:
    """Build a minimal LayerCacheState with tokens in a demoted slot.

    Tokens 0..num_in_demoted_slot-1 are placed in `demoted_slot`.
    Remaining tokens are placed in slot 0 (active/protected).
    """
    device = torch.device("cpu")
    dtype = torch.float32

    # anchor_slot: first num_in_demoted_slot tokens in demoted_slot, rest in slot 0
    anchor_slot = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device)
    anchor_slot[:, :num_in_demoted_slot] = demoted_slot

    # importance: deterministic values for reproducibility
    if importance_values is not None:
        imp = torch.tensor(importance_values, dtype=dtype, device=device)
        imp = imp.unsqueeze(0).expand(batch_size, -1)
    else:
        # Use arange so tokens have distinct importances
        imp = torch.arange(num_tokens, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1).clone()

    metadata = TokenMetadata(
        token_kind=torch.full((batch_size, num_tokens), 2, dtype=torch.long, device=device),
        frame_id=torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device),
        anchor_slot=anchor_slot,
        keyframe_id=torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device),
        slot_id=torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device),
        slot_local_xyz=torch.full((batch_size, num_tokens, 3), float("nan"), dtype=dtype, device=device),
        importance=imp,
        depth_conf=torch.ones(batch_size, num_tokens, dtype=dtype, device=device),
    )

    # Build dummy k, v tensors
    num_heads = 1
    head_dim = 4
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, num_tokens, head_dim, dtype=dtype, device=device)

    # Build score_state
    score_state = torch.randn(batch_size, num_tokens, score_state_dim, dtype=dtype, device=device)

    state = LayerCacheState(
        k=k,
        v=v,
        score_state=score_state,
        metadata=metadata,
        protected_count=num_tokens - num_in_demoted_slot,
    )
    return state


class ProbeRecorder:
    """Simple probe that records arguments passed to on_fifo_topk_candidate."""

    def __init__(self):
        self.calls = []

    def on_fifo_topk_candidate(
        self,
        cache_state,
        demoted_slot: int,
        keep_count: int,
        layer_id: int,
        frame_id: int,
        batch_index: int = 0,
        demoted_indices_by_batch=None,
    ):
        # Snapshot the demoted-slot indices from cache_state at probe time
        b_idx = batch_index
        slot_mask = cache_state.metadata.anchor_slot[b_idx] == demoted_slot
        live_indices = torch.nonzero(slot_mask, as_tuple=False).squeeze(-1).detach().clone()

        self.calls.append({
            "demoted_slot": demoted_slot,
            "keep_count": keep_count,
            "layer_id": layer_id,
            "frame_id": frame_id,
            "batch_index": batch_index,
            "demoted_indices_by_batch": demoted_indices_by_batch,
            "live_indices_at_probe_time": live_indices,
        })


class TestFifoProbeObservesPreMutationDemotedSlot:
    """Test that the FIFO probe sees the original demoted-slot token set."""

    def test_fifo_probe_observes_pre_mutation_demoted_slot(self):
        """Probe receives 5 demoted-slot indices before any mutation.

        Build a LayerCacheState with 5 tokens in anchor_slot=1.
        Call protect_topk_on_demotion_(demoted_slot=1, keep_count=2, fifo_probe=probe).
        Probe receives 5 demoted-slot indices.
        After protection, exactly 2 tokens moved to slot 0.
        """
        num_tokens = 10
        num_in_demoted_slot = 5
        demoted_slot = 1
        keep_count = 2

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        probe = ProbeRecorder()

        cache.protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            fifo_probe=probe,
            layer_id=0,
            current_frame_id=0,
            batch_index=0,
        )

        # Probe must have been called exactly once
        assert len(probe.calls) == 1, f"Expected 1 probe call, got {len(probe.calls)}"

        call = probe.calls[0]

        # The probe must see all 5 demoted-slot indices BEFORE mutation
        live_indices = call["live_indices_at_probe_time"]
        assert live_indices.numel() == num_in_demoted_slot, (
            f"Probe should see {num_in_demoted_slot} demoted-slot tokens, "
            f"but saw {live_indices.numel()}"
        )

        # demoted_indices_by_batch should be provided and match
        if call["demoted_indices_by_batch"] is not None:
            for b_idx, indices in call["demoted_indices_by_batch"].items():
                assert indices.numel() == num_in_demoted_slot, (
                    f"demoted_indices_by_batch[{b_idx}] should have "
                    f"{num_in_demoted_slot} indices, got {indices.numel()}"
                )

        # After protection, exactly 2 tokens should have been moved from
        # demoted_slot to slot 0
        anchor = cache.metadata.anchor_slot[0]
        remaining_in_slot = (anchor == demoted_slot).sum().item()
        moved_to_slot0 = (anchor == 0).sum().item()

        assert remaining_in_slot == num_in_demoted_slot - keep_count, (
            f"Expected {num_in_demoted_slot - keep_count} tokens remaining in "
            f"demoted_slot={demoted_slot}, but found {remaining_in_slot}"
        )

        # Original slot-0 tokens (5) + newly protected (2) = 7
        original_slot0 = num_tokens - num_in_demoted_slot
        assert moved_to_slot0 == original_slot0 + keep_count, (
            f"Expected {original_slot0 + keep_count} tokens in slot 0, "
            f"but found {moved_to_slot0}"
        )


class TestFifoProbeKeepCountZero:
    """Test that keep_count=0 is recorded by the probe without mutation."""

    def test_fifo_probe_sees_keep_count_zero(self):
        """keep_count=0 should still fire the probe but not mutate anchor_slot."""
        num_tokens = 10
        num_in_demoted_slot = 5
        demoted_slot = 1

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        # Snapshot anchor_slot before
        anchor_before = cache.metadata.anchor_slot.clone()

        probe = ProbeRecorder()

        cache.protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=0,
            fifo_probe=probe,
            layer_id=0,
            current_frame_id=0,
            batch_index=0,
        )

        # Probe must have been called even with keep_count=0
        assert len(probe.calls) == 1, (
            f"Probe should be called even with keep_count=0, got {len(probe.calls)} calls"
        )

        call = probe.calls[0]
        assert call["keep_count"] == 0

        # Probe should see all 5 demoted-slot tokens
        live_indices = call["live_indices_at_probe_time"]
        assert live_indices.numel() == num_in_demoted_slot, (
            f"Probe should see {num_in_demoted_slot} demoted-slot tokens with keep_count=0, "
            f"but saw {live_indices.numel()}"
        )

        # anchor_slot must NOT be mutated
        assert torch.equal(cache.metadata.anchor_slot, anchor_before), (
            "keep_count=0 must not mutate anchor_slot"
        )


class TestFifoProbeProtectAllTokens:
    """Test that keep_count >= demoted_token_count protects all tokens."""

    def test_fifo_probe_keep_count_exceeds_demoted(self):
        """keep_count >= demoted_token_count should protect all demoted tokens."""
        num_tokens = 10
        num_in_demoted_slot = 5
        demoted_slot = 1
        keep_count = 10  # More than 5 demoted tokens

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        probe = ProbeRecorder()

        cache.protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            fifo_probe=probe,
            layer_id=0,
            current_frame_id=0,
            batch_index=0,
        )

        # Probe must have been called
        assert len(probe.calls) == 1

        call = probe.calls[0]
        # Probe should see all 5 demoted-slot tokens
        live_indices = call["live_indices_at_probe_time"]
        assert live_indices.numel() == num_in_demoted_slot

        # All demoted tokens should now be in slot 0
        anchor = cache.metadata.anchor_slot[0]
        remaining_in_slot = (anchor == demoted_slot).sum().item()
        assert remaining_in_slot == 0, (
            f"Expected 0 tokens remaining in demoted_slot={demoted_slot} "
            f"when keep_count={keep_count}, but found {remaining_in_slot}"
        )

        all_in_slot0 = (anchor == 0).sum().item()
        assert all_in_slot0 == num_tokens, (
            f"Expected all {num_tokens} tokens in slot 0, but found {all_in_slot0}"
        )


class TestFifoProbeSamplesMultipleKeepCounts:
    """Test that count_candidates causes the probe to emit subsets for multiple
    keep_count values, clamped to [0, num_slot_tokens].
    """

    def test_fifo_probe_samples_multiple_keep_counts_with_zero_and_all(self):
        """Given demoted slot token count K=10 and count_candidates [0, 4, 20],
        the probe should emit candidates for counts [0, 4, 10] after clamping.

        Expected behavior:
        - keep_count=0: keep_indices contains only non-demoted-slot tokens.
        - keep_count=4: keep_indices contains non-demoted-slot tokens plus 4 demoted tokens.
        - keep_count=10: keep_indices contains non-demoted-slot tokens plus all 10 demoted tokens.
        - Every candidate subset has subset["keep_count"].
        """
        num_tokens = 20
        num_in_demoted_slot = 10  # K=10
        demoted_slot = 1
        keep_count = 5  # runtime keep_count (not used when count_candidates is set)

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        # Non-demoted indices: tokens 10..19
        expected_non_slot = set(range(num_in_demoted_slot, num_tokens))
        # Demoted indices: tokens 0..9
        expected_demoted = set(range(num_in_demoted_slot))

        probe = CounterfactualFifoTopKProbe(
            count_candidates=[0, 4, 20],  # 20 will be clamped to K=10
            max_subsets_per_fifo_event=16,
        )

        probe.on_fifo_topk_candidate(
            cache_state=cache,
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            layer_id=0,
            frame_id=0,
            batch_index=0,
            demoted_indices_by_batch=None,
        )

        assert len(probe.events) == 1, f"Expected 1 event, got {len(probe.events)}"
        event = probe.events[0]

        # Event must include demoted_indices
        assert "demoted_indices" in event, "Event must contain 'demoted_indices'"
        demoted_indices = event["demoted_indices"]
        assert demoted_indices.numel() == num_in_demoted_slot, (
            f"demoted_indices should have {num_in_demoted_slot} elements, "
            f"got {demoted_indices.numel()}"
        )

        subsets = event["candidate_subsets"]
        # Collect the unique keep_counts present across subsets
        keep_counts_found = sorted({int(s["keep_count"]) for s in subsets})
        # Should be [0, 4, 10] (20 clamped to K=10)
        assert keep_counts_found == [0, 4, 10], (
            f"Expected keep_counts [0, 4, 10], got {keep_counts_found}"
        )

        # Check keep_count=0 subset
        zero_subsets = [s for s in subsets if int(s["keep_count"]) == 0]
        assert len(zero_subsets) >= 1, "Expected at least 1 subset with keep_count=0"
        for s in zero_subsets:
            keep_set = set(s["keep_indices"].tolist())
            assert keep_set == expected_non_slot, (
                f"keep_count=0 subset should contain only non-demoted tokens, "
                f"got {keep_set}"
            )

        # Check keep_count=10 (all) subset
        all_subsets = [s for s in subsets if int(s["keep_count"]) == num_in_demoted_slot]
        assert len(all_subsets) >= 1, (
            f"Expected at least 1 subset with keep_count={num_in_demoted_slot}"
        )
        for s in all_subsets:
            keep_set = set(s["keep_indices"].tolist())
            assert keep_set == set(range(num_tokens)), (
                f"keep_count=10 subset should contain all tokens, got {keep_set}"
            )

        # Check keep_count=4 subsets
        four_subsets = [s for s in subsets if int(s["keep_count"]) == 4]
        assert len(four_subsets) >= 1, "Expected at least 1 subset with keep_count=4"
        for s in four_subsets:
            keep_set = set(s["keep_indices"].tolist())
            # Must include all non-slot tokens
            assert expected_non_slot.issubset(keep_set), (
                f"keep_count=4 subset must include all non-demoted tokens; "
                f"missing {expected_non_slot - keep_set}"
            )
            # Must include exactly 4 demoted tokens
            demoted_kept = keep_set & expected_demoted
            assert len(demoted_kept) == 4, (
                f"keep_count=4 subset should have exactly 4 demoted tokens, "
                f"got {len(demoted_kept)}"
            )
            # Must have demoted_keep_indices
            assert "demoted_keep_indices" in s, (
                "subset must include 'demoted_keep_indices'"
            )
            assert s["demoted_keep_indices"].numel() == 4, (
                f"demoted_keep_indices should have 4 elements, "
                f"got {s['demoted_keep_indices'].numel()}"
            )

    def test_fifo_probe_default_count_candidates_uses_runtime_keep_count(self):
        """When count_candidates is None (default), probe should use the
        runtime keep_count value only.
        """
        num_tokens = 15
        num_in_demoted_slot = 5
        demoted_slot = 1
        keep_count = 2

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        probe = CounterfactualFifoTopKProbe()  # default: no count_candidates

        probe.on_fifo_topk_candidate(
            cache_state=cache,
            demoted_slot=demoted_slot,
            keep_count=keep_count,
            layer_id=0,
            frame_id=0,
            batch_index=0,
            demoted_indices_by_batch=None,
        )

        assert len(probe.events) == 1
        subsets = probe.events[0]["candidate_subsets"]
        # All subsets should have keep_count=2 (the runtime value)
        keep_counts = {int(s.get("keep_count", keep_count)) for s in subsets}
        # Old behavior: subsets may or may not have keep_count field,
        # but if they do, they should all be 2.
        # With new behavior, every subset must have keep_count.
        for s in subsets:
            assert "keep_count" in s, "Every subset must have 'keep_count' field"
            assert int(s["keep_count"]) == keep_count, (
                f"Expected keep_count={keep_count}, got {s['keep_count']}"
            )

    def test_fifo_probe_cap_semantics(self):
        """max_subsets_per_fifo_event caps total subsets across all count
        candidates. Deterministic subsets (keep_count=0 and all) are always
        included.
        """
        num_tokens = 20
        num_in_demoted_slot = 10
        demoted_slot = 1

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        # Very tight cap: only 2 subsets allowed
        probe = CounterfactualFifoTopKProbe(
            count_candidates=[0, 4, 10],
            max_subsets_per_fifo_event=2,
        )

        probe.on_fifo_topk_candidate(
            cache_state=cache,
            demoted_slot=demoted_slot,
            keep_count=5,
            layer_id=0,
            frame_id=0,
            batch_index=0,
            demoted_indices_by_batch=None,
        )

        assert len(probe.events) == 1
        event = probe.events[0]
        subsets = event["candidate_subsets"]
        keep_counts = sorted({int(s["keep_count"]) for s in subsets})

        # With cap=2 and 3 candidate counts (0, 4, 10), the cap should be
        # auto-raised to at least 3 (one per count) so we get all three.
        # Check that all counts are represented.
        assert 0 in keep_counts, "keep_count=0 must always be included"
        assert num_in_demoted_slot in keep_counts, "all-token subset must be included"
        assert 4 in keep_counts, (
            "At least one positive non-all count must be represented; "
            "cap should auto-raise if needed"
        )

    def test_fifo_probe_demoted_indices_in_event(self):
        """Event-level output must include demoted_indices for FifoCountDataset."""
        num_tokens = 15
        num_in_demoted_slot = 5
        demoted_slot = 1

        cache = _make_layer_cache(
            num_tokens=num_tokens,
            demoted_slot=demoted_slot,
            num_in_demoted_slot=num_in_demoted_slot,
        )

        probe = CounterfactualFifoTopKProbe(
            count_candidates=[0, 5],
        )

        probe.on_fifo_topk_candidate(
            cache_state=cache,
            demoted_slot=demoted_slot,
            keep_count=3,
            layer_id=0,
            frame_id=0,
            batch_index=0,
            demoted_indices_by_batch=None,
        )

        assert len(probe.events) == 1
        event = probe.events[0]
        assert "demoted_indices" in event
        demoted_indices = event["demoted_indices"]
        assert demoted_indices.numel() == num_in_demoted_slot
        # Must be on CPU and long dtype
        assert demoted_indices.device == torch.device("cpu")
        assert demoted_indices.dtype == torch.long


# ---------------------------------------------------------------------------
# Helper: fake CounterfactualReplayRunner for testing replay output schemas
# ---------------------------------------------------------------------------

class _FakeRunner:
    """Minimal fake runner that satisfies CounterfactualReplayRunner protocol."""

    def __init__(self):
        self._snapshot = None

    def snapshot(self):
        self._snapshot = copy.deepcopy({"cache": "dummy"})
        return self._snapshot

    def restore(self, snapshot):
        pass

    def apply_keep_indices(self, layer_id, keep_indices):
        pass

    def replay_future_window(self, start_frame_idx, future_frames):
        # Return trivial predictions so loss computation works
        return [
            {"camera_pose": torch.zeros(4, 4)}
            for _ in future_frames
        ]

    def targets_for_future_window(self, future_frames):
        return [
            {"camera_pose": torch.zeros(4, 4)}
            for _ in future_frames
        ]


def _make_fifo_event_with_subsets(
    keep_counts=(0, 4),
    num_tokens=10,
    demoted_slot=1,
    num_in_demoted=5,
):
    """Build a FifoTopKCounterfactualEvent with candidate subsets."""
    score_state_dim = 8
    metadata_dim = 4
    # Non-demoted tokens are those NOT in the demoted slot
    non_demoted = list(range(num_in_demoted, num_tokens))
    demoted = list(range(num_in_demoted))

    candidate_subsets = []
    for kc in keep_counts:
        # keep_indices includes all non-demoted + first kc demoted
        keep = sorted(non_demoted + demoted[:kc])
        demoted_keep = demoted[:kc]
        candidate_subsets.append({
            "keep_indices": torch.tensor(keep, dtype=torch.long),
            "strategy": "fifo_topk",
            "source": "fifo_probe",
            "keep_count": kc,
            "demoted_keep_indices": torch.tensor(demoted_keep, dtype=torch.long),
        })

    return FifoTopKCounterfactualEvent(
        event_id="test-fifo-001",
        layer_id=0,
        frame_id=5,
        demoted_slot=demoted_slot,
        sequence_provenance={"dataset": "test"},
        score_state=torch.randn(1, num_tokens, score_state_dim),
        metadata_features=torch.randn(1, num_tokens, metadata_dim),
        candidate_subsets=candidate_subsets,
        keep_count=2,
        base_scores=torch.randn(num_tokens),
    )


class TestFifoReplayPreservesSubsetKeepCount:
    """Tests that collect_fifo_topk_counterfactual_event preserves subset-level
    keep_count and demoted_keep_indices in the replay output.
    """

    def test_fifo_replay_preserves_subset_keep_count(self):
        """Replay output must preserve keep_count, strategy, source,
        and demoted_keep_indices from each candidate subset.
        """
        event = _make_fifo_event_with_subsets(keep_counts=(0, 4))
        runner = _FakeRunner()
        future_frames = [
            {"frame_id": torch.tensor(6)},
            {"frame_id": torch.tensor(7)},
        ]

        result = collect_fifo_topk_counterfactual_event(event, runner, future_frames)

        # Check event-level keep_count
        assert "keep_count" in result, "Result must contain keep_count"

        subsets = result["subsets"]
        assert len(subsets) == 2, f"Expected 2 subsets, got {len(subsets)}"

        expected_counts = {0, 4}
        found_counts = set()
        for subset in subsets:
            assert "keep_count" in subset, (
                "Each measured subset must contain 'keep_count'"
            )
            assert "strategy" in subset, (
                "Each measured subset must contain 'strategy'"
            )
            assert "source" in subset, (
                "Each measured subset must contain 'source'"
            )
            assert "demoted_keep_indices" in subset, (
                "Each measured subset must contain 'demoted_keep_indices'"
            )

            kc = int(subset["keep_count"])
            found_counts.add(kc)

            assert subset["strategy"] == "fifo_topk"
            assert subset["source"] == "fifo_probe"

            # demoted_keep_indices should be a tensor on CPU
            dki = subset["demoted_keep_indices"]
            assert isinstance(dki, torch.Tensor), (
                f"demoted_keep_indices should be a Tensor, got {type(dki)}"
            )
            assert dki.device == torch.device("cpu")

            if kc == 0:
                assert dki.numel() == 0
            elif kc == 4:
                assert dki.numel() == 4

        assert found_counts == expected_counts, (
            f"Expected keep_counts {expected_counts}, got {found_counts}"
        )


class TestFrontendOracleCollectorPreservesFifoSubsetMetadata:
    """Tests that measure_counterfactual_event and measure_counterfactual_subset_serial
    preserve FIFO subset-level metadata (keep_count, strategy, source, demoted_keep_indices).
    """

    def _make_event_dict_with_subsets(
        self,
        keep_counts=(0, 4),
        num_tokens=10,
        num_in_demoted=5,
    ):
        """Build an event dict (as passed to measure_counterfactual_event) with subsets."""
        non_demoted = list(range(num_in_demoted, num_tokens))
        demoted = list(range(num_in_demoted))

        candidate_subsets = []
        for kc in keep_counts:
            keep = sorted(non_demoted + demoted[:kc])
            demoted_keep = demoted[:kc]
            candidate_subsets.append({
                "keep_indices": torch.tensor(keep, dtype=torch.long),
                "strategy": "fifo_topk",
                "source": "fifo_probe",
                "keep_count": kc,
                "demoted_keep_indices": torch.tensor(demoted_keep, dtype=torch.long),
            })

        event = {
            "event_id": "test-event-001",
            "event_type": "eviction",
            "layer_id": 0,
            "frame_id": 5,
            "candidate_subsets": candidate_subsets,
            "voxel_group_id": 0,
        }
        return event

    def test_frontend_oracle_collector_preserves_fifo_subset_metadata(self):
        """measure_counterfactual_event (serial path) must preserve subset metadata."""
        from unittest.mock import MagicMock, patch

        from ovggt.training.frontend_oracle_collector import measure_counterfactual_event

        event = self._make_event_dict_with_subsets(keep_counts=(0, 4))
        num_future = 2
        frames = [{"frame_id": torch.tensor(i)} for i in range(8)]
        future_frames = frames[6:6 + num_future]

        mock_model = MagicMock()

        class FakeOutputs:
            def __init__(self, res_list):
                self.ress = res_list

        def make_fake_outputs():
            return FakeOutputs([
                {"camera_pose": torch.zeros(4, 4)} for _ in range(8)
            ])

        # Patch both the runner and the replay probe so that probe.applied is True
        with patch("ovggt.training.frontend_oracle_collector._run_frontend_with_probe") as mock_run, \
             patch("ovggt.training.frontend_oracle_collector.ReplayKeepSetProbe") as MockProbe:

            # Make the mock probe report applied=True immediately
            fake_probe = MagicMock()
            fake_probe.applied = True
            MockProbe.return_value = fake_probe

            mock_run.return_value = make_fake_outputs()

            result = measure_counterfactual_event(
                model=mock_model,
                frames=frames,
                event=event,
                future_frames=future_frames,
                subset_replay_batch_size=1,
            )

        assert result is not None, "measure_counterfactual_event should not return None"
        subsets = result["subsets"]
        assert len(subsets) == 2, f"Expected 2 measured subsets, got {len(subsets)}"

        expected_counts = {0, 4}
        found_counts = set()
        for subset in subsets:
            assert "keep_count" in subset, (
                "Measured subset must contain 'keep_count'"
            )
            assert "strategy" in subset, (
                "Measured subset must contain 'strategy'"
            )
            assert "source" in subset, (
                "Measured subset must contain 'source'"
            )
            assert "demoted_keep_indices" in subset, (
                "Measured subset must contain 'demoted_keep_indices'"
            )

            kc = int(subset["keep_count"])
            found_counts.add(kc)

            assert subset["strategy"] == "fifo_topk"
            assert subset["source"] == "fifo_probe"

            dki = subset["demoted_keep_indices"]
            assert isinstance(dki, torch.Tensor), (
                f"demoted_keep_indices should be Tensor, got {type(dki)}"
            )
            assert dki.device == torch.device("cpu")

        assert found_counts == expected_counts, (
            f"Expected keep_counts {expected_counts}, got {found_counts}"
        )

    def test_frontend_oracle_collector_batched_preserves_fifo_subset_metadata(self):
        """measure_counterfactual_event (batched path) must preserve subset metadata."""
        from unittest.mock import MagicMock, patch

        from ovggt.training.frontend_oracle_collector import measure_counterfactual_event

        event = self._make_event_dict_with_subsets(keep_counts=(0, 4))
        num_future = 2
        frames = [{"frame_id": torch.tensor(i)} for i in range(8)]
        future_frames = frames[6:6 + num_future]

        mock_model = MagicMock()

        class FakeOutputs:
            def __init__(self, res_list):
                self.ress = res_list

        def make_fake_batched_outputs(batch_size):
            return FakeOutputs([
                {"camera_pose": torch.zeros(batch_size, 4, 4)}
                for _ in range(8)
            ])

        with patch("ovggt.training.frontend_oracle_collector._run_frontend_with_probe") as mock_run, \
             patch("ovggt.training.frontend_oracle_collector.MultiReplayKeepSetProbe") as MockMultiProbe:

            # Make the mock multi probe report applied_count == chunk_size
            fake_multi = MagicMock()
            fake_multi.applied_count = 2  # matches chunk_size
            MockMultiProbe.return_value = fake_multi

            mock_run.return_value = make_fake_batched_outputs(2)

            result = measure_counterfactual_event(
                model=mock_model,
                frames=frames,
                event=event,
                future_frames=future_frames,
                subset_replay_batch_size=2,
            )

        assert result is not None
        subsets = result["subsets"]
        assert len(subsets) == 2, f"Expected 2 measured subsets, got {len(subsets)}"

        expected_counts = {0, 4}
        found_counts = set()
        for subset in subsets:
            assert "keep_count" in subset, "Measured subset must contain 'keep_count'"
            assert "strategy" in subset, "Measured subset must contain 'strategy'"
            assert "source" in subset, "Measured subset must contain 'source'"
            assert "demoted_keep_indices" in subset, "Measured subset must contain 'demoted_keep_indices'"

            kc = int(subset["keep_count"])
            found_counts.add(kc)
            assert subset["strategy"] == "fifo_topk"
            assert subset["source"] == "fifo_probe"

        assert found_counts == expected_counts, (
            f"Expected keep_counts {expected_counts}, got {found_counts}"
        )

    def test_measure_counterfactual_subset_serial_preserves_fifo_subset_metadata(self):
        """measure_counterfactual_subset_serial must preserve subset metadata."""
        from unittest.mock import MagicMock, patch

        from ovggt.training.frontend_oracle_collector import measure_counterfactual_subset_serial

        num_tokens = 10
        num_in_demoted = 5
        non_demoted = list(range(num_in_demoted, num_tokens))
        demoted = list(range(num_in_demoted))

        subsets_input = []
        for kc in (0, 4):
            keep = sorted(non_demoted + demoted[:kc])
            demoted_keep = demoted[:kc]
            subsets_input.append({
                "keep_indices": torch.tensor(keep, dtype=torch.long),
                "strategy": "fifo_topk",
                "source": "fifo_probe",
                "keep_count": kc,
                "demoted_keep_indices": torch.tensor(demoted_keep, dtype=torch.long),
            })

        event = {
            "event_id": "test-serial-001",
            "event_type": "eviction",
            "layer_id": 0,
            "frame_id": 5,
        }
        frames = [{"frame_id": torch.tensor(i)} for i in range(8)]
        future_frames = frames[6:8]
        mock_model = MagicMock()

        class FakeOutputs:
            def __init__(self, res_list):
                self.ress = res_list

        with patch("ovggt.training.frontend_oracle_collector._run_frontend_with_probe") as mock_run, \
             patch("ovggt.training.frontend_oracle_collector.ReplayKeepSetProbe") as MockProbe:

            fake_probe = MagicMock()
            fake_probe.applied = True
            MockProbe.return_value = fake_probe

            mock_run.return_value = FakeOutputs([
                {"camera_pose": torch.zeros(4, 4)} for _ in range(8)
            ])

            result = measure_counterfactual_subset_serial(
                model=mock_model,
                frames=frames,
                event=event,
                future_frames=future_frames,
                subsets=subsets_input,
                start=6,
                stop=8,
            )

        assert result is not None, "measure_counterfactual_subset_serial should not return None"
        assert len(result) == 2, f"Expected 2 measured subsets, got {len(result)}"

        expected_counts = {0, 4}
        found_counts = set()
        for subset in result:
            assert "keep_count" in subset, "Measured subset must contain 'keep_count'"
            assert "strategy" in subset, "Measured subset must contain 'strategy'"
            assert "source" in subset, "Measured subset must contain 'source'"
            assert "demoted_keep_indices" in subset, "Measured subset must contain 'demoted_keep_indices'"

            kc = int(subset["keep_count"])
            found_counts.add(kc)
            assert subset["strategy"] == "fifo_topk"
            assert subset["source"] == "fifo_probe"

            dki = subset["demoted_keep_indices"]
            assert isinstance(dki, torch.Tensor), (
                f"demoted_keep_indices should be Tensor, got {type(dki)}"
            )
            assert dki.device == torch.device("cpu")

        assert found_counts == expected_counts, (
            f"Expected keep_counts {expected_counts}, got {found_counts}"
        )
