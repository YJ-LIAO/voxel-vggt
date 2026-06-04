"""Tests for Task 5: Export JointRetentionPolicy to OVGGT runtime checkpoint.

Covers:
- Step 5.1: Export mapping correctness (token scorer keys, count head keys,
  weight replication across layers).
- Step 5.5: End-to-end load into an OVGGT model with use_token_scorer=True
  and use_count_head=True (shared_encoder_v2).
"""

import os
import sys
from typing import Optional

import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.retention_policy import (
    JointRetentionPolicy,
    RetentionTokenEncoder,
    TokenRankingHead,
    FifoCountClassifier,
    build_token_scorer_state_from_joint,
    build_count_head_state_from_joint,
    build_ovggt_joint_retention_state_dict,
)
from ovggt.layers.token_scorer import TokenScorer
from ovggt.layers.count_head import FifoCountHead
from ovggt.models.ovggt import OVGGT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use small dimensions for fast tests
_SCORE_STATE_DIM = 32
_METADATA_DIM = 17
_HIDDEN_DIM = 16
_NUM_LAYERS = 4
_CANDIDATES = (0, 4, 8, 16)


def _make_joint() -> JointRetentionPolicy:
    """Create a small JointRetentionPolicy for testing."""
    return JointRetentionPolicy(
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=_HIDDEN_DIM,
        num_layers=_NUM_LAYERS,
        count_candidates=_CANDIDATES,
    )


def _make_ovggt(
    use_token_scorer: bool = True,
    use_count_head: bool = True,
    score_state_dim: int = _SCORE_STATE_DIM,
    count_head_hidden_dim: int = _HIDDEN_DIM,
) -> OVGGT:
    """Create a minimal OVGGT for export round-trip testing.

    Dimensions are configured to match the test JointRetentionPolicy so that
    exported weights can be loaded back.
    """
    aggregator_kwargs = dict(depth=_NUM_LAYERS, num_heads=2, num_register_tokens=1)
    from ovggt.utils.frontend_cache import FrontendCacheConfig
    fcc = FrontendCacheConfig(
        score_state_dim=score_state_dim,
        fifo_count_candidates=_CANDIDATES,
    )
    return OVGGT(
        img_size=56,
        patch_size=14,
        embed_dim=64,
        mode="frontend_train",
        frontend_cache_config=fcc,
        aggregator_kwargs=aggregator_kwargs,
        use_token_scorer=use_token_scorer,
        scorer_bottleneck_dim=_HIDDEN_DIM,
        use_count_head=use_count_head,
        count_head_hidden_dim=count_head_hidden_dim,
        count_head_arch="shared_encoder_v2",
    )


# ===========================================================================
# Step 5.1: Export mapping tests
# ===========================================================================

class TestTokenScorerExportMapping:
    """Verify the token scorer export produces correct deploy keys."""

    def test_export_keys_match_token_scorer_structure(self):
        """Exported keys match TokenScorer's named_parameters layout."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)

        expected_keys = {
            "layer_embed.weight",
            "scorer.0.weight",
            "scorer.0.bias",
            "scorer.1.weight",
            "scorer.1.bias",
            "scorer.3.weight",
            "scorer.3.bias",
        }
        assert set(state.keys()) == expected_keys, (
            f"Key mismatch: got {sorted(state.keys())}, "
            f"expected {sorted(expected_keys)}"
        )

    def test_layer_embed_weight_shape(self):
        """layer_embed.weight has shape [num_layers, score_state_dim]."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)
        le = state["layer_embed.weight"]
        assert le.shape == (_NUM_LAYERS, _SCORE_STATE_DIM), (
            f"Expected ({_NUM_LAYERS}, {_SCORE_STATE_DIM}), got {tuple(le.shape)}"
        )

    def test_scorer_0_weight_shape(self):
        """scorer.0 (LayerNorm) weight has correct input dim."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)
        concat_dim = _SCORE_STATE_DIM + _METADATA_DIM + _SCORE_STATE_DIM
        assert state["scorer.0.weight"].shape == (concat_dim,)
        assert state["scorer.0.bias"].shape == (concat_dim,)

    def test_scorer_1_weight_shape(self):
        """scorer.1 (Linear) weight has shape [hidden_dim, concat_dim]."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)
        concat_dim = _SCORE_STATE_DIM + _METADATA_DIM + _SCORE_STATE_DIM
        assert state["scorer.1.weight"].shape == (_HIDDEN_DIM, concat_dim)
        assert state["scorer.1.bias"].shape == (_HIDDEN_DIM,)

    def test_scorer_3_weight_shape(self):
        """scorer.3 (final Linear) weight has shape [1, hidden_dim]."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)
        assert state["scorer.3.weight"].shape == (1, _HIDDEN_DIM)
        assert state["scorer.3.bias"].shape == (1,)

    def test_exported_values_are_detached_cpu_clones(self):
        """All exported tensors are on CPU and detached from the graph."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(joint)
        for key, tensor in state.items():
            assert tensor.device == torch.device("cpu"), f"{key} not on CPU"
            assert not tensor.requires_grad, f"{key} still requires grad"

    def test_hidden_dim_mismatch_raises(self):
        """expected_hidden_dim mismatch raises ValueError."""
        joint = _make_joint()
        with pytest.raises(ValueError, match="hidden_dim"):
            build_token_scorer_state_from_joint(joint, expected_hidden_dim=999)

    def test_hidden_dim_match_succeeds(self):
        """Matching expected_hidden_dim does not raise."""
        joint = _make_joint()
        state = build_token_scorer_state_from_joint(
            joint, expected_hidden_dim=_HIDDEN_DIM
        )
        assert len(state) == 7


class TestCountHeadExportMapping:
    """Verify the count head export produces v2 architecture state dict."""

    def test_count_head_state_has_v2_keys(self):
        """Exported count head state has _encoder and _classifier sub-keys."""
        joint = _make_joint()
        state = build_count_head_state_from_joint(joint)
        assert len(state) > 0

        # Should have v2 architecture keys
        has_encoder = any(k.startswith("_encoder.") for k in state)
        has_classifier = any(k.startswith("_classifier.") for k in state)
        assert has_encoder, "Missing _encoder.* keys in count head state"
        assert has_classifier, "Missing _classifier.* keys in count head state"

    def test_count_head_encoder_weights_match_joint(self):
        """Count head encoder weights are copied from the joint encoder."""
        joint = _make_joint()
        state = build_count_head_state_from_joint(joint)

        joint_enc_state = joint.encoder.state_dict()
        for key, value in joint_enc_state.items():
            exported_key = f"_encoder.{key}"
            assert exported_key in state, f"Missing {exported_key}"
            assert torch.equal(state[exported_key], value), (
                f"Mismatch for {exported_key}"
            )

    def test_count_head_classifier_weights_match_joint(self):
        """Count head classifier weights are copied from joint count_head."""
        joint = _make_joint()
        state = build_count_head_state_from_joint(joint)

        joint_cls_state = joint.count_head.state_dict()
        for key, value in joint_cls_state.items():
            exported_key = f"_classifier.{key}"
            assert exported_key in state, f"Missing {exported_key}"
            assert torch.equal(state[exported_key], value), (
                f"Mismatch for {exported_key}"
            )

    def test_count_head_candidates_preserved(self):
        """Exported count head preserves the candidate values."""
        joint = _make_joint()
        state = build_count_head_state_from_joint(joint)
        assert "candidates" in state
        assert torch.equal(state["candidates"], torch.tensor(_CANDIDATES))


class TestFullDeployStateDict:
    """Verify build_ovggt_joint_retention_state_dict layout."""

    def test_token_scorer_keys_per_layer(self):
        """Each layer gets identical token scorer deploy keys."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=None,
            num_layers=_NUM_LAYERS,
        )
        for layer_idx in range(_NUM_LAYERS):
            for key in ts_state:
                deploy_key = f"aggregator.token_scorers.{layer_idx}.{key}"
                assert deploy_key in deploy, f"Missing key: {deploy_key}"

    def test_all_layers_have_identical_token_scorer_weights(self):
        """All layers share identical (replicated) weights from shared encoder."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=None,
            num_layers=_NUM_LAYERS,
        )

        # Pick a reference key from layer 0
        ref_key = "aggregator.token_scorers.0.scorer.1.weight"
        ref_val = deploy[ref_key]

        for layer_idx in range(1, _NUM_LAYERS):
            layer_key = f"aggregator.token_scorers.{layer_idx}.scorer.1.weight"
            assert torch.equal(deploy[layer_key], ref_val), (
                f"Layer {layer_idx} scorer weights differ from layer 0"
            )

    def test_count_head_keys_prefixed(self):
        """Count head keys appear under aggregator.count_head.* prefix."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        ch_state = build_count_head_state_from_joint(joint)
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=ch_state,
            num_layers=_NUM_LAYERS,
        )

        for key in ch_state:
            deploy_key = f"aggregator.count_head.{key}"
            assert deploy_key in deploy, f"Missing: {deploy_key}"

    def test_score_state_projection_passthrough(self):
        """score_state_projection_state keys pass through unchanged."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        proj_state = {
            "aggregator.score_state_projs.0.weight": torch.randn(32, 64),
            "aggregator.score_state_projs.0.bias": torch.randn(32),
        }
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=None,
            num_layers=_NUM_LAYERS,
            score_state_projection_state=proj_state,
        )
        for key in proj_state:
            assert key in deploy, f"Missing projection key: {key}"
            assert torch.equal(deploy[key], proj_state[key])

    def test_non_projection_keys_in_projection_state_ignored(self):
        """Keys not starting with aggregator.score_state_projs. are ignored."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        bogus = {"some.other.key": torch.randn(3)}
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=None,
            num_layers=_NUM_LAYERS,
            score_state_projection_state=bogus,
        )
        assert "some.other.key" not in deploy


# ===========================================================================
# Step 5.5: End-to-end OVGGT loading
# ===========================================================================

class TestOVGGTDeployLoadRoundTrip:
    """Verify exported deploy state loads into OVGGT runtime model."""

    def test_exported_token_scorers_load_into_ovggt(self):
        """Exported token scorer state loads without error into OVGGT."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        ch_state = build_count_head_state_from_joint(joint)
        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=ch_state,
            num_layers=_NUM_LAYERS,
        )

        model = _make_ovggt(use_token_scorer=True, use_count_head=True)

        # Load the exported deploy state
        result = model.load_state_dict(deploy, strict=False)

        # No missing scorer/count_head keys (they were all provided)
        scorer_or_count = [
            k for k in result.missing_keys
            if k.startswith("aggregator.token_scorers.")
            or k.startswith("aggregator.count_head.")
            or k.startswith("aggregator.score_state_projs.")
        ]
        assert len(scorer_or_count) == 0, (
            f"Missing scorer/count_head keys: {scorer_or_count}"
        )

    def test_loaded_token_scorer_weights_match_joint(self):
        """After loading, token scorer weights match the original joint encoder."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        ch_state = build_count_head_state_from_joint(joint)

        # Build projection state to match the model's expected shapes
        model = _make_ovggt(use_token_scorer=True, use_count_head=True)
        proj_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.startswith("aggregator.score_state_projs.")
        }

        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=ch_state,
            num_layers=_NUM_LAYERS,
            score_state_projection_state=proj_state,
        )

        model.load_state_dict(deploy, strict=False)

        # Verify each layer's token scorer has the exported weights
        for layer_idx in range(_NUM_LAYERS):
            scorer = model.aggregator.token_scorers[layer_idx]

            # layer_embed
            assert torch.equal(
                scorer.layer_embed.weight,
                joint.encoder.layer_embed.weight,
            ), f"Layer {layer_idx}: layer_embed mismatch"

            # scorer.0 (LayerNorm) -> encoder.norm
            assert torch.equal(
                scorer.scorer[0].weight,
                joint.encoder.norm.weight,
            ), f"Layer {layer_idx}: scorer.0.weight mismatch"
            assert torch.equal(
                scorer.scorer[0].bias,
                joint.encoder.norm.bias,
            ), f"Layer {layer_idx}: scorer.0.bias mismatch"

            # scorer.1 (Linear) -> encoder.proj
            assert torch.equal(
                scorer.scorer[1].weight,
                joint.encoder.proj.weight,
            ), f"Layer {layer_idx}: scorer.1.weight mismatch"
            assert torch.equal(
                scorer.scorer[1].bias,
                joint.encoder.proj.bias,
            ), f"Layer {layer_idx}: scorer.1.bias mismatch"

            # scorer.3 (final Linear) -> token_head.out
            assert torch.equal(
                scorer.scorer[3].weight,
                joint.token_head.out.weight,
            ), f"Layer {layer_idx}: scorer.3.weight mismatch"
            assert torch.equal(
                scorer.scorer[3].bias,
                joint.token_head.out.bias,
            ), f"Layer {layer_idx}: scorer.3.bias mismatch"

    def test_loaded_count_head_weights_match_joint(self):
        """After loading, count head encoder/classifier match joint weights."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        ch_state = build_count_head_state_from_joint(joint)

        model = _make_ovggt(use_token_scorer=True, use_count_head=True)
        proj_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.startswith("aggregator.score_state_projs.")
        }

        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=ch_state,
            num_layers=_NUM_LAYERS,
            score_state_projection_state=proj_state,
        )
        model.load_state_dict(deploy, strict=False)

        # Verify count head encoder matches joint encoder
        count_head = model.aggregator.count_head
        assert count_head is not None
        assert count_head.arch == "shared_encoder_v2"

        for key, value in joint.encoder.state_dict().items():
            model_key = f"_encoder.{key}"
            model_val = dict(count_head._encoder.named_parameters())
            if key in dict(count_head._encoder.state_dict()):
                assert torch.equal(
                    count_head._encoder.state_dict()[key], value
                ), f"count_head._encoder.{key} mismatch"

        for key, value in joint.count_head.state_dict().items():
            if key == "candidates":
                continue  # buffer, already checked elsewhere
            if key in dict(count_head._classifier.state_dict()):
                assert torch.equal(
                    count_head._classifier.state_dict()[key], value
                ), f"count_head._classifier.{key} mismatch"

    def test_exported_state_no_unexpected_scorer_keys(self):
        """No unexpected keys from scorer/count_head after loading."""
        joint = _make_joint()
        ts_state = build_token_scorer_state_from_joint(joint)
        ch_state = build_count_head_state_from_joint(joint)

        model = _make_ovggt(use_token_scorer=True, use_count_head=True)
        proj_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.startswith("aggregator.score_state_projs.")
        }

        deploy = build_ovggt_joint_retention_state_dict(
            token_scorer_state=ts_state,
            count_head_state=ch_state,
            num_layers=_NUM_LAYERS,
            score_state_projection_state=proj_state,
        )

        result = model.load_state_dict(deploy, strict=False)

        # All deploy keys should be consumed (none unexpected)
        unexpected_scorer = [
            k for k in result.unexpected_keys
            if k.startswith("aggregator.token_scorers.")
            or k.startswith("aggregator.count_head.")
            or k.startswith("aggregator.score_state_projs.")
        ]
        assert len(unexpected_scorer) == 0, (
            f"Unexpected scorer/count_head keys: {unexpected_scorer}"
        )


class TestNumericEquivalence:
    """Verify that exported weights produce numerically identical outputs."""

    def test_token_scorer_forward_matches_joint_forward(self):
        """TokenScorer loaded from export produces same output as JointRetentionPolicy."""
        joint = _make_joint()
        joint.eval()

        ts_state = build_token_scorer_state_from_joint(joint)

        # Build a standalone TokenScorer with matching dims
        scorer = TokenScorer(
            score_state_dim=_SCORE_STATE_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_layers=_NUM_LAYERS,
        )
        scorer.load_state_dict(ts_state)
        scorer.eval()

        B, N = 2, 10
        score_state = torch.randn(B, N, _SCORE_STATE_DIM)
        metadata = torch.randn(B, N, _METADATA_DIM)
        layer_id = 1

        with torch.no_grad():
            joint_out = joint.forward_token(score_state, metadata, layer_id)
            scorer_out = scorer(score_state, metadata, layer_id)

        assert torch.allclose(joint_out, scorer_out, atol=1e-6), (
            f"TokenScorer output differs from joint. "
            f"Max diff: {(joint_out - scorer_out).abs().max().item()}"
        )
