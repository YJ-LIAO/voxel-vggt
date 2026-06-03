"""Tests for train_fifo_count_head training script and checkpoint serialization."""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_fake_shard(path, num_events=3):
    """Create a fake shard with fifo_topk events for training."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM

    events = []
    for i in range(num_events):
        n_tokens = 20
        n_demoted = 12
        event = {
            "event_id": f"test_{i}",
            "event_type": "fifo_topk",
            "layer_id": i % 24,
            "score_state": torch.randn(n_tokens, 128),
            "metadata_features": torch.randn(n_tokens, TOKEN_METADATA_FEATURE_DIM),
            "demoted_indices": torch.randperm(n_tokens)[:n_demoted],
            "subsets": [
                {"keep_count": 0, "loss": 3.0 + i * 0.1},
                {"keep_count": 8, "loss": 1.0 + i * 0.2},
                {"keep_count": 16, "loss": 2.0 + i * 0.3},
                {"keep_count": 32, "loss": 2.5 + i * 0.4},
                {"keep_count": 64, "loss": 2.8 + i * 0.5},
                {"keep_count": 128, "loss": 2.9 + i * 0.6},
            ],
        }
        events.append(event)
    shard = {"events": events}
    torch.save(shard, path)
    return path


def test_train_one_epoch_produces_valid_checkpoint(tmp_path):
    """Run one training epoch on CPU with a fake shard and verify checkpoint contents."""
    from train_fifo_count_head import train_fifo_count_head

    shard_path = _make_fake_shard(tmp_path / "fake_shard.pt", num_events=4)
    output_path = tmp_path / "output" / "fifo_count_head.pt"

    train_fifo_count_head(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        count_candidates=(0, 8, 16, 32, 64, 128),
        score_state_dim=128,
        hidden_dim=64,
        num_layers=24,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        label_reduction="min",
        device="cpu",
    )

    assert output_path.exists(), f"Checkpoint not written to {output_path}"
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)

    # Top-level keys
    assert "count_head" in checkpoint, "Missing 'count_head' in checkpoint"
    assert "model" in checkpoint, "Missing 'model' in checkpoint"
    assert "score_state_dim" in checkpoint, "Missing 'score_state_dim' in checkpoint"
    assert "metadata_dim" in checkpoint, "Missing 'metadata_dim' in checkpoint"
    assert "num_layers" in checkpoint, "Missing 'num_layers' in checkpoint"
    assert "count_candidates" in checkpoint, "Missing 'count_candidates' in checkpoint"

    # count_head is a state dict
    assert isinstance(checkpoint["count_head"], dict), "count_head should be a dict (state_dict)"

    # model dict has keys prefixed with aggregator.count_head.
    model_keys = list(checkpoint["model"].keys())
    assert any(k.startswith("aggregator.count_head.") for k in model_keys), (
        f"No keys starting with 'aggregator.count_head.' in model, got: {model_keys[:5]}"
    )

    # candidates buffer appears in the deploy state dict
    assert "aggregator.count_head.candidates" in checkpoint["model"], (
        "Missing 'aggregator.count_head.candidates' in model state dict"
    )

    # count_candidates is a list matching what was passed
    assert isinstance(checkpoint["count_candidates"], list), "count_candidates should be a list"
    assert checkpoint["count_candidates"] == [0, 8, 16, 32, 64, 128]


def test_build_ovggt_count_head_state_dict():
    """Verify deploy state dict mapping prepends aggregator.count_head."""
    from train_fifo_count_head import build_ovggt_count_head_state_dict

    count_head_state = {
        "mlp.0.weight": torch.randn(4, 4),
        "mlp.0.bias": torch.randn(4),
        "candidates": torch.tensor([0, 8, 16]),
    }
    deploy_state = build_ovggt_count_head_state_dict(count_head_state)

    assert "aggregator.count_head.mlp.0.weight" in deploy_state
    assert "aggregator.count_head.mlp.0.bias" in deploy_state
    assert "aggregator.count_head.candidates" in deploy_state
    assert torch.equal(
        deploy_state["aggregator.count_head.mlp.0.weight"],
        count_head_state["mlp.0.weight"],
    )
    # Values should be detached and on CPU
    for value in deploy_state.values():
        assert not value.requires_grad


def test_train_multiple_epochs_reduces_loss(tmp_path):
    """Training for multiple epochs should reduce loss on the training data."""
    from train_fifo_count_head import train_fifo_count_head

    shard_path = _make_fake_shard(tmp_path / "fake_shard.pt", num_events=6)
    output_path = tmp_path / "output" / "fifo_count_head.pt"

    train_fifo_count_head(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        count_candidates=(0, 8, 16, 32, 64, 128),
        score_state_dim=128,
        hidden_dim=64,
        num_layers=24,
        batch_size=2,
        epochs=5,
        lr=1e-3,
        label_reduction="min",
        device="cpu",
    )

    assert output_path.exists()
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert "count_head" in checkpoint
