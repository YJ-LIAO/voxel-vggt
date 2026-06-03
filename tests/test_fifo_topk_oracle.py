"""Tests for FIFO top-K protection probe pre-mutation semantics.

Ensures that the probe sees the original demoted-slot token set before any
metadata mutation, even when keep_count=0 or keep_count >= demoted_token_count.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import pytest

from ovggt.utils.frontend_cache import LayerCacheState, TokenMetadata


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
