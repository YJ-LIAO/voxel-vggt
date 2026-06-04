"""Tests for joint training of token-ranking and FIFO-count retention policy.

Step 4.1: CPU smoke training test with a fake oracle shard containing:
- One eviction/dedup event that produces pairwise token samples
- One fifo_topk event with multiple keep_count values and demoted_indices

Asserts the checkpoint contains all required keys and deploy state layout.
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM


# --------------------------------------------------------------------------- #
# Fake shard helpers
# --------------------------------------------------------------------------- #

_SCORE_STATE_DIM = 32
_METADATA_DIM = TOKEN_METADATA_FEATURE_DIM  # 17
_NUM_TOKENS = 10


def _make_eviction_event(event_id: str, layer_id: int = 0) -> dict:
    """Eviction event with 2 subsets producing pairwise ranking samples."""
    return {
        "event_id": event_id,
        "event_type": "eviction",
        "layer_id": layer_id,
        "sequence_provenance": {"sequence_id": f"seq_{event_id}"},
        "score_state": torch.randn(_NUM_TOKENS, _SCORE_STATE_DIM),
        "metadata_features": torch.randn(_NUM_TOKENS, _METADATA_DIM),
        "subsets": [
            {"keep_indices": [0, 1, 2, 3], "loss": 0.10},
            {"keep_indices": [4, 5, 6, 7], "loss": 0.50},
        ],
    }


def _make_fifo_topk_event(event_id: str, layer_id: int = 5) -> dict:
    """Fifo_topk event with multiple keep_count values and demoted_indices."""
    n_tokens = 20
    n_demoted = 12
    return {
        "event_id": event_id,
        "event_type": "fifo_topk",
        "layer_id": layer_id,
        "sequence_provenance": {"sequence_id": f"seq_{event_id}"},
        "score_state": torch.randn(n_tokens, _SCORE_STATE_DIM),
        "metadata_features": torch.randn(n_tokens, _METADATA_DIM),
        "demoted_indices": torch.randperm(n_tokens)[:n_demoted].tolist(),
        "subsets": [
            {"keep_count": 0, "keep_indices": list(range(0)), "loss": 3.0},
            {"keep_count": 4, "keep_indices": list(range(4)), "loss": 0.8},
            {"keep_count": 8, "keep_indices": list(range(8)), "loss": 1.2},
            {"keep_count": 16, "keep_indices": list(range(16)), "loss": 2.0},
        ],
    }


def _make_fake_shard(path, num_eviction=2, num_fifo=3):
    """Create a fake shard with both eviction and fifo_topk events."""
    events = []
    for i in range(num_eviction):
        events.append(_make_eviction_event(f"eviction_{i}", layer_id=i % 4))
    for i in range(num_fifo):
        events.append(_make_fifo_topk_event(f"fifo_{i}", layer_id=(i + 2) % 4))
    shard = {"events": events}
    torch.save(shard, path)
    return path


# --------------------------------------------------------------------------- #
# Step 4.1: CPU smoke training test
# --------------------------------------------------------------------------- #

def test_train_one_epoch_produces_valid_checkpoint(tmp_path):
    """Run 1 CPU epoch and verify checkpoint contains all required keys."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=3,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    count_candidates = [0, 4, 8, 16]

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=count_candidates,
        count_head_arch="shared_encoder_v2",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        weight_decay=0.01,
        regression_weight=0.1,
        min_loss_gap=0.01,
        count_loss_weight=1.0,
        count_label_reduction="min",
        count_repeat_factor=1.0,
        val_fraction=0.0,
        split_key="event_id_hash",
        split_seed=0,
        device="cpu",
    )

    assert output_path.exists(), f"Checkpoint not written to {output_path}"
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)

    # --- Top-level metadata keys ---
    assert checkpoint["joint_arch"] == "shared_token_encoder_v1", (
        f"Expected joint_arch='shared_token_encoder_v1', got {checkpoint.get('joint_arch')}"
    )
    assert checkpoint["count_head_arch"] == "shared_encoder_v2", (
        f"Expected count_head_arch='shared_encoder_v2', got {checkpoint.get('count_head_arch')}"
    )
    assert checkpoint["count_head_trained"] is True, (
        f"Expected count_head_trained=True, got {checkpoint.get('count_head_trained')}"
    )
    assert checkpoint["count_candidates"] == count_candidates, (
        f"Expected count_candidates={count_candidates}, got {checkpoint.get('count_candidates')}"
    )

    # --- Native joint state ---
    assert "joint_policy" in checkpoint, "Missing 'joint_policy' in checkpoint"
    assert isinstance(checkpoint["joint_policy"], dict)

    # --- Exported TokenScorer state ---
    assert "token_scorer" in checkpoint, "Missing 'token_scorer' in checkpoint"
    token_scorer_state = checkpoint["token_scorer"]
    assert isinstance(token_scorer_state, dict)
    expected_scorer_keys = {
        "layer_embed.weight",
        "scorer.0.weight", "scorer.0.bias",
        "scorer.1.weight", "scorer.1.bias",
        "scorer.3.weight", "scorer.3.bias",
    }
    assert set(token_scorer_state.keys()) == expected_scorer_keys, (
        f"token_scorer keys mismatch: {sorted(token_scorer_state.keys())}"
    )

    # --- Exported CountHead state ---
    assert "count_head" in checkpoint, "Missing 'count_head' in checkpoint"
    assert checkpoint["count_head"] is not None, "count_head should not be None when trained"
    count_head_state = checkpoint["count_head"]
    assert isinstance(count_head_state, dict)
    assert any(k.startswith("_encoder.") for k in count_head_state), (
        "count_head state missing _encoder.* keys"
    )
    assert any(k.startswith("_classifier.") for k in count_head_state), (
        "count_head state missing _classifier.* keys"
    )

    # --- Full deploy state ---
    assert "model" in checkpoint, "Missing 'model' in checkpoint"
    deploy_state = checkpoint["model"]
    assert isinstance(deploy_state, dict)

    # Deploy keys for token scorers
    assert "aggregator.token_scorers.0.layer_embed.weight" in deploy_state, (
        "Missing aggregator.token_scorers.0.layer_embed.weight"
    )
    assert "aggregator.token_scorers.0.scorer.1.weight" in deploy_state, (
        "Missing aggregator.token_scorers.0.scorer.1.weight"
    )

    # Deploy keys for count head
    assert "aggregator.count_head._encoder.layer_embed.weight" in deploy_state, (
        "Missing aggregator.count_head._encoder.layer_embed.weight"
    )

    # --- Dimension keys ---
    assert checkpoint["score_state_dim"] == _SCORE_STATE_DIM
    assert checkpoint["metadata_dim"] == _METADATA_DIM
    assert checkpoint["hidden_dim"] == 16
    assert checkpoint["num_layers"] == 4
    assert checkpoint["score_state_projection_checkpoint"] is None


def test_train_token_only_when_no_fifo_events(tmp_path):
    """When no fifo_topk events exist, training runs token-only and count_head_trained=False."""
    from train_joint_retention_policy import train_joint_retention

    # Shard with only eviction events
    events = []
    for i in range(5):
        events.append(_make_eviction_event(f"ev_only_{i}", layer_id=i % 4))
    shard_path = tmp_path / "eviction_only_shard.pt"
    torch.save({"events": events}, shard_path)

    output_path = tmp_path / "output" / "joint_token_only.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        batch_size=2,
        epochs=1,
        lr=1e-3,
        min_loss_gap=0.01,
        regression_weight=0.1,
        count_loss_weight=1.0,
        count_label_reduction="min",
        count_repeat_factor=1.0,
        device="cpu",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["count_head_trained"] is False, (
        f"Expected count_head_trained=False for token-only training"
    )
    assert checkpoint["count_head"] is None, (
        "count_head should be None when not trained"
    )
    # Token scorer should still be present
    assert "token_scorer" in checkpoint
    assert "model" in checkpoint


def test_train_with_count_repeat_factor(tmp_path):
    """count_repeat_factor > 1 causes the count loader to repeat within an epoch."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=2,
        num_fifo=2,
    )
    output_path = tmp_path / "output" / "joint_repeat.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        batch_size=2,
        epochs=1,
        lr=1e-3,
        min_loss_gap=0.01,
        regression_weight=0.1,
        count_loss_weight=1.0,
        count_label_reduction="min",
        count_repeat_factor=3.0,
        device="cpu",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["count_head_trained"] is True
    assert "count_head" in checkpoint
    assert checkpoint["count_head"] is not None


def test_parse_args_with_yaml_override(tmp_path):
    """Config from YAML is overridden by CLI args."""
    from train_joint_retention_policy import parse_args

    # Write a YAML config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "oracle_shards:\n  - /fake/shard.pt\n"
        "output: /fake/output.pt\n"
        "hidden_dim: 512\n"
        "lr: 0.001\n"
    )

    shard = tmp_path / "shard.pt"
    output = tmp_path / "out.pt"

    args = parse_args([
        "--config", str(config_path),
        "--oracle-shards", str(shard),
        "--output", str(output),
        "--hidden-dim", "128",
    ])

    # CLI should override YAML for hidden_dim
    assert args.hidden_dim == 128, f"Expected 128, got {args.hidden_dim}"
    # YAML values that were NOT overridden should come through
    assert args.lr == 0.001, f"Expected 0.001 from YAML, got {args.lr}"
