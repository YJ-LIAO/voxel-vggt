"""Tests for the shared retention policy module.

Covers:
- Step 1.1: shape and input validation tests for RetentionTokenEncoder
- Step 1.5: gradient flow tests confirming shared encoder receives gradients
  from both token and count heads
- Step 3.1: event-level split leak tests
- Step 3.5: fifo_token_pair_mode same_keep_count tests
"""

import os
import sys

import pytest
import torch
from torch import Tensor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.retention_policy import (
    FifoCountClassifier,
    JointRetentionPolicy,
    RetentionTokenEncoder,
    TokenRankingHead,
)
from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, N, Ds, Dm, H = 4, 64, 128, 17, 256


def _make_encoder(**kw):
    defaults = dict(score_state_dim=Ds, metadata_dim=Dm, hidden_dim=H, num_layers=24)
    defaults.update(kw)
    return RetentionTokenEncoder(**defaults)


def _make_joint(**kw):
    defaults = dict(
        score_state_dim=Ds,
        metadata_dim=Dm,
        hidden_dim=H,
        num_layers=24,
        count_candidates=(0, 8, 16, 32, 64, 128),
    )
    defaults.update(kw)
    return JointRetentionPolicy(**defaults)


# ===================================================================
# Step 1.1 — Shape and input validation tests
# ===================================================================


class TestRetentionTokenEncoderShape:
    """Verify output shapes and default behaviour."""

    def test_basic_output_shape(self):
        """RetentionTokenEncoder(score_state, metadata, layer_id) -> [B, N, H]."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        out = enc(score, meta, layer_id=3)
        assert out.shape == (B, N, H), f"expected {(B, N, H)}, got {tuple(out.shape)}"

    def test_metadata_none_pads_zeros(self):
        """metadata_features=None should auto-pad with zeros."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        out = enc(score, metadata_features=None, layer_id=0)
        assert out.shape == (B, N, H)

    def test_layer_id_int(self):
        """layer_id as plain int works."""
        enc = _make_encoder()
        score = torch.randn(2, 10, Ds)
        out = enc(score, None, layer_id=5)
        assert out.shape == (2, 10, H)

    def test_layer_id_scalar_tensor(self):
        """layer_id as a single-element tensor works."""
        enc = _make_encoder()
        score = torch.randn(2, 10, Ds)
        out = enc(score, None, layer_id=torch.tensor(7))
        assert out.shape == (2, 10, H)

    def test_layer_id_batch_tensor(self):
        """layer_id as a [B] tensor works."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        lids = torch.arange(B)
        out = enc(score, None, layer_id=lids)
        assert out.shape == (B, N, H)

class TestRetentionTokenEncoderValidation:
    """Verify that invalid inputs raise ValueError."""

    def test_score_state_wrong_dim(self):
        """score_state not [B, N, Ds] should raise ValueError."""
        enc = _make_encoder()
        # 2-D input
        with pytest.raises(ValueError, match="score_state"):
            enc(torch.randn(B, Ds), None, layer_id=0)

    def test_score_state_wrong_last_dim(self):
        """score_state last dim mismatch should raise ValueError."""
        enc = _make_encoder()
        wrong_dim = Ds + 1
        with pytest.raises(ValueError, match="score_state_dim"):
            enc(torch.randn(B, N, wrong_dim), None, layer_id=0)

    def test_metadata_token_mismatch(self):
        """metadata_features with wrong [B, N] should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_meta = torch.randn(B, N + 1, Dm)
        with pytest.raises(ValueError, match="metadata_features"):
            enc(score, bad_meta, layer_id=0)

    def test_metadata_dim_mismatch(self):
        """metadata_features with wrong last dim should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_meta = torch.randn(B, N, Dm + 1)
        with pytest.raises(ValueError, match="metadata_features dim"):
            enc(score, bad_meta, layer_id=0)

    def test_layer_id_wrong_batch_size(self):
        """layer_id tensor with wrong batch size should raise ValueError."""
        enc = _make_encoder()
        score = torch.randn(B, N, Ds)
        bad_lids = torch.arange(B + 1)
        with pytest.raises(ValueError, match="layer_id"):
            enc(score, None, layer_id=bad_lids)


# ===================================================================
# Step 1.1 continued — TokenRankingHead and FifoCountClassifier shapes
# ===================================================================


class TestTokenRankingHeadShape:
    def test_output_shape(self):
        """TokenRankingHead maps [B, N, H] -> [B, N]."""
        head = TokenRankingHead(H)
        x = torch.randn(B, N, H)
        out = head(x)
        assert out.shape == (B, N)


class TestFifoCountClassifierShape:
    def test_output_shape(self):
        """FifoCountClassifier maps [B, N, H] -> [B, num_candidates]."""
        head = FifoCountClassifier(H)
        x = torch.randn(B, N, H)
        out = head(x)
        assert out.shape == (B, 6)  # default 6 candidates

    def test_output_shape_with_mask(self):
        """FifoCountClassifier respects token_mask."""
        head = FifoCountClassifier(H)
        x = torch.randn(B, N, H)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, N // 2:] = False
        out = head(x, token_mask=mask)
        assert out.shape == (B, 6)

    def test_predict_count(self):
        """predict_count returns candidate values."""
        head = FifoCountClassifier(H, candidates=(0, 8, 16, 32, 64, 128))
        logits = torch.zeros(B, 6)
        # Point index 2 (value 16) as winner
        logits[:, 2] = 10.0
        preds = head.predict_count(logits)
        assert preds.shape == (B,)
        assert (preds == 16).all()


class TestJointRetentionPolicyShape:
    def test_forward_token_shape(self):
        """forward_token returns [B, N]."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        out = joint.forward_token(score, meta, layer_id=0)
        assert out.shape == (B, N)

    def test_forward_count_shape(self):
        """forward_count returns [B, num_candidates]."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)
        out = joint.forward_count(score, meta, layer_id=0, token_mask=mask)
        assert out.shape == (B, 6)


# ===================================================================
# Step 1.5 — Gradient flow tests
# ===================================================================


class TestSharedEncoderGradients:
    """Verify both losses update the shared encoder."""

    def _assert_encoder_grad(self, joint, name=""):
        assert joint.encoder.layer_embed.weight.grad is not None, (
            f"{name}: layer_embed.weight.grad is None"
        )
        assert joint.encoder.proj.weight.grad is not None, (
            f"{name}: proj.weight.grad is None"
        )

    def test_combined_loss_updates_encoder(self):
        """Token loss + count loss -> encoder gets gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)
        layer_id = torch.tensor(2)

        token_loss = joint.forward_token(score, meta, layer_id).sum()
        count_loss = joint.forward_count(score, meta, layer_id, mask).sum()
        (token_loss + count_loss).backward()

        self._assert_encoder_grad(joint, "combined")

    def test_token_loss_alone_updates_encoder(self):
        """Token loss alone -> encoder gets non-zero gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)

        token_loss = joint.forward_token(score, meta, layer_id=0).sum()
        token_loss.backward()

        self._assert_encoder_grad(joint, "token_only")
        # Check gradients are actually non-zero
        assert joint.encoder.layer_embed.weight.grad.abs().sum() > 0, (
            "token_only: layer_embed grad is all zeros"
        )
        assert joint.encoder.proj.weight.grad.abs().sum() > 0, (
            "token_only: proj grad is all zeros"
        )

    def test_count_loss_alone_updates_encoder(self):
        """Count loss alone -> encoder gets non-zero gradients."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)

        count_loss = joint.forward_count(score, meta, layer_id=0, token_mask=mask).sum()
        count_loss.backward()

        self._assert_encoder_grad(joint, "count_only")
        assert joint.encoder.layer_embed.weight.grad.abs().sum() > 0, (
            "count_only: layer_embed grad is all zeros"
        )
        assert joint.encoder.proj.weight.grad.abs().sum() > 0, (
            "count_only: proj grad is all zeros"
        )

    def test_individual_token_and_count_grads_nonzero(self):
        """Both token and count heads produce non-zero encoder gradients individually."""
        joint = _make_joint()
        score = torch.randn(B, N, Ds)
        meta = torch.randn(B, N, Dm)
        mask = torch.ones(B, N, dtype=torch.bool)

        # Token head
        joint.zero_grad()
        token_loss = joint.forward_token(score, meta, layer_id=0).sum()
        token_loss.backward()
        token_layer_grad = joint.encoder.layer_embed.weight.grad.clone()
        token_proj_grad = joint.encoder.proj.weight.grad.clone()

        # Count head
        joint.zero_grad()
        count_loss = joint.forward_count(score, meta, layer_id=0, token_mask=mask).sum()
        count_loss.backward()
        count_layer_grad = joint.encoder.layer_embed.weight.grad.clone()
        count_proj_grad = joint.encoder.proj.weight.grad.clone()

        # Both must be non-zero
        assert token_layer_grad.abs().sum() > 0, "token head: layer_embed grad zero"
        assert token_proj_grad.abs().sum() > 0, "token head: proj grad zero"
        assert count_layer_grad.abs().sum() > 0, "count head: layer_embed grad zero"
        assert count_proj_grad.abs().sum() > 0, "count head: proj grad zero"


# ===================================================================
# Step 3.1 — Event-level split leak tests
# ===================================================================


def _make_fake_event(
    event_id: str,
    event_type: str = "eviction",
    sequence_id: str = "seq_a",
    num_tokens: int = 6,
    keep_count: int | None = None,
    demoted_indices: list[int] | None = None,
) -> dict:
    """Create a minimal fake event for split / dataset tests."""
    subsets = [
        {"keep_indices": torch.tensor([0, 1, 2]), "loss": 0.20 + 0.01 * hash(event_id) % 5},
        {"keep_indices": torch.tensor([3, 4, 5]), "loss": 0.55 + 0.01 * hash(event_id) % 5},
    ]
    if keep_count is not None:
        subsets[0]["keep_count"] = keep_count
        subsets[1]["keep_count"] = keep_count
    event: dict = {
        "event_id": event_id,
        "event_type": event_type,
        "layer_id": 0,
        "sequence_provenance": {"sequence_id": sequence_id},
        "score_state": torch.randn(num_tokens, 8),
        "metadata_features": torch.randn(num_tokens, TOKEN_METADATA_FEATURE_DIM),
        "subsets": subsets,
    }
    if demoted_indices is not None:
        event["demoted_indices"] = demoted_indices
    if keep_count is not None:
        event["keep_count"] = keep_count
    return event


class TestEventLevelSplitNoLeakage:
    """Step 3.1: event-level split avoids token/count data leakage."""

    @pytest.fixture()
    def fake_events(self) -> list[dict]:
        """100 events spanning 3 event types and 10 sequences."""
        events = []
        for idx in range(100):
            event_type = ["eviction", "dedup", "fifo_topk"][idx % 3]
            seq_id = f"seq_{idx % 10}"
            demoted = list(range(3)) if event_type == "fifo_topk" else None
            kc = 3 if event_type == "fifo_topk" else None
            events.append(
                _make_fake_event(
                    event_id=f"event_{idx:04d}",
                    event_type=event_type,
                    sequence_id=seq_id,
                    demoted_indices=demoted,
                    keep_count=kc,
                )
            )
        return events

    def test_event_id_hash_split_no_event_leakage(self, fake_events):
        """split_oracle_events with event_id_hash: no event_id in both train and val."""
        from ovggt.training.token_oracle_dataset import split_oracle_events

        train_events, val_events = split_oracle_events(
            fake_events, val_fraction=0.2, split_key="event_id_hash", seed=42,
        )
        train_ids = {e["event_id"] for e in train_events}
        val_ids = {e["event_id"] for e in val_events}
        assert train_ids, "train split should not be empty"
        assert val_ids, "val split should not be empty"
        assert train_ids.isdisjoint(val_ids), (
            f"event_id leakage: {train_ids & val_ids}"
        )

    def test_sequence_id_split_no_sequence_leakage(self, fake_events):
        """split_oracle_events with sequence_id: no sequence in both train and val."""
        from ovggt.training.token_oracle_dataset import split_oracle_events

        train_events, val_events = split_oracle_events(
            fake_events, val_fraction=0.2, split_key="sequence_id", seed=42,
        )
        train_seqs = {e["sequence_provenance"]["sequence_id"] for e in train_events}
        val_seqs = {e["sequence_provenance"]["sequence_id"] for e in val_events}
        assert train_seqs, "train split should not be empty"
        assert val_seqs, "val split should not be empty"
        assert train_seqs.isdisjoint(val_seqs), (
            f"sequence_id leakage: {train_seqs & val_seqs}"
        )

    def test_token_and_count_datasets_share_same_events(self, fake_events):
        """Token dataset and count dataset built from same split use the same event set."""
        from ovggt.training.token_oracle_dataset import (
            CounterfactualOracleDataset,
            FifoCountDataset,
            split_oracle_events,
        )

        train_events, val_events = split_oracle_events(
            fake_events, val_fraction=0.2, split_key="event_id_hash", seed=0,
        )

        # Build both datasets from the same train events
        token_ds = CounterfactualOracleDataset.from_events(train_events)
        count_ds = FifoCountDataset.from_events(train_events)

        token_event_ids = {s["event_id"] for s in token_ds.samples}
        count_event_ids = {s["event_id"] for s in count_ds.samples}

        # Count dataset only uses fifo_topk events, but every event_id it uses
        # must be a subset of the token dataset's event_ids
        assert count_event_ids.issubset(token_event_ids), (
            f"count dataset has event_ids not in token dataset: "
            f"{count_event_ids - token_event_ids}"
        )

    def test_val_events_not_in_train_events(self, fake_events):
        """Events in val must not appear in train."""
        from ovggt.training.token_oracle_dataset import split_oracle_events

        train_events, val_events = split_oracle_events(
            fake_events, val_fraction=0.15, split_key="sequence_id", seed=0,
        )
        train_ids = {e["event_id"] for e in train_events}
        val_ids = {e["event_id"] for e in val_events}
        assert train_ids.isdisjoint(val_ids)

        # Also check sequence_id consistency
        train_seqs = {e["sequence_provenance"]["sequence_id"] for e in train_events}
        val_seqs = {e["sequence_provenance"]["sequence_id"] for e in val_events}
        assert train_seqs.isdisjoint(val_seqs)

    def test_sequence_id_split_falls_back_to_event_id(self):
        """When sequence_provenance is missing, falls back to event_id."""
        from ovggt.training.token_oracle_dataset import split_oracle_events

        events = []
        for idx in range(50):
            e = _make_fake_event(event_id=f"ev_{idx:04d}", sequence_id=f"seq_{idx % 5}")
            # Remove sequence_provenance from some events to test fallback
            if idx % 3 == 0:
                del e["sequence_provenance"]
            events.append(e)

        train_events, val_events = split_oracle_events(
            events, val_fraction=0.2, split_key="sequence_id", seed=0,
        )
        train_ids = {e["event_id"] for e in train_events}
        val_ids = {e["event_id"] for e in val_events}
        assert train_ids.isdisjoint(val_ids)


# ===================================================================
# Step 3.5 — fifo_token_pair_mode="same_keep_count" tests
# ===================================================================


class TestFifoTokenPairModeSameKeepCount:
    """Step 3.5: same_keep_count mode restricts pairs to same keep_count."""

    def test_same_keep_count_restricts_pairs(self):
        """With same_keep_count, no pairs span different keep_count values."""
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        event = {
            "event_id": "fifo_ev_1",
            "event_type": "fifo_topk",
            "layer_id": 0,
            "sequence_provenance": {"sequence_id": "seq_a"},
            "score_state": torch.randn(8, 8),
            "metadata_features": torch.randn(8, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1, 2], "loss": 0.10, "keep_count": 3},
                {"keep_indices": [3, 4, 5], "loss": 0.30, "keep_count": 3},
                {"keep_indices": [0, 1, 2, 3], "loss": 0.20, "keep_count": 4},
                {"keep_indices": [4, 5, 6, 7], "loss": 0.40, "keep_count": 4},
            ],
        }

        ds = CounterfactualOracleDataset.from_events(
            [event], fifo_token_pair_mode="same_keep_count",
        )

        for sample in ds.samples:
            better_mask = sample["better_mask"]
            worse_mask = sample["worse_mask"]
            better_count = int(better_mask.sum().item())
            worse_count = int(worse_mask.sum().item())
            assert better_count == worse_count, (
                f"same_keep_count violation: better_keep={better_count} "
                f"worse_keep={worse_count}"
            )

    def test_any_mode_allows_cross_keep_count_pairs(self):
        """With any mode, cross-keep_count pairs are allowed."""
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        event = {
            "event_id": "fifo_ev_2",
            "event_type": "fifo_topk",
            "layer_id": 0,
            "sequence_provenance": {"sequence_id": "seq_a"},
            "score_state": torch.randn(8, 8),
            "metadata_features": torch.randn(8, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1, 2], "loss": 0.10, "keep_count": 3},
                {"keep_indices": [3, 4, 5], "loss": 0.30, "keep_count": 3},
                {"keep_indices": [0, 1, 2, 3], "loss": 0.20, "keep_count": 4},
                {"keep_indices": [4, 5, 6, 7], "loss": 0.40, "keep_count": 4},
            ],
        }

        ds_any = CounterfactualOracleDataset.from_events(
            [event], fifo_token_pair_mode="any",
        )
        ds_same = CounterfactualOracleDataset.from_events(
            [event], fifo_token_pair_mode="same_keep_count",
        )

        # "any" mode should produce more or equal pairs than "same_keep_count"
        assert len(ds_any) >= len(ds_same)
        # And with different keep_counts, "any" should strictly have more
        assert len(ds_any) > len(ds_same), (
            f"'any' mode ({len(ds_any)}) should produce more pairs than "
            f"'same_keep_count' ({len(ds_same)}) with different keep_counts"
        )

    def test_same_keep_count_does_not_affect_non_fifo_events(self):
        """same_keep_count mode only applies to fifo_topk events."""
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        eviction_event = {
            "event_id": "eviction_ev",
            "event_type": "eviction",
            "layer_id": 0,
            "sequence_provenance": {"sequence_id": "seq_a"},
            "score_state": torch.randn(6, 8),
            "metadata_features": torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1, 2], "loss": 0.10},
                {"keep_indices": [3, 4, 5], "loss": 0.30},
            ],
        }

        ds_any = CounterfactualOracleDataset.from_events(
            [eviction_event], fifo_token_pair_mode="any",
        )
        ds_same = CounterfactualOracleDataset.from_events(
            [eviction_event], fifo_token_pair_mode="same_keep_count",
        )

        assert len(ds_any) == len(ds_same), (
            "same_keep_count should not affect non-fifo_topk events"
        )

    def test_invalid_fifo_token_pair_mode_raises(self):
        """Invalid fifo_token_pair_mode should raise ValueError."""
        from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

        with pytest.raises(ValueError, match="fifo_token_pair_mode"):
            CounterfactualOracleDataset.from_events(
                [], fifo_token_pair_mode="invalid",
            )
