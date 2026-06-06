import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _sample(event_id, event_type="dedup", sequence_id="seq_a"):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "sequence_provenance": {"sequence_id": sequence_id},
    }


def _make_pairwise_event(event_id, sequence_id):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM

    return {
        "event_id": event_id,
        "event_type": "dedup",
        "layer_id": 0,
        "sequence_provenance": {"sequence_id": sequence_id},
        "score_state": torch.randn(8, 16),
        "metadata_features": torch.randn(8, TOKEN_METADATA_FEATURE_DIM),
        "subsets": [
            {"keep_indices": [0, 1, 2, 3], "loss": 0.1},
            {"keep_indices": [4, 5, 6, 7], "loss": 0.7},
        ],
    }


def _make_fake_oracle_shard(path, num_events=30):
    events = [
        _make_pairwise_event(f"event_{idx:03d}", f"seq_{idx:03d}")
        for idx in range(num_events)
    ]
    torch.save({"events": events}, path)
    return path


def test_event_hash_split_has_no_event_id_leakage():
    from train_token_scorer_oracle import split_oracle_pair_samples

    samples = []
    for idx in range(50):
        event_id = f"event_{idx:03d}"
        samples.append(_sample(event_id, sequence_id=f"seq_{idx % 5}"))
        samples.append(_sample(event_id, sequence_id=f"seq_{idx % 5}"))

    train, val = split_oracle_pair_samples(
        samples,
        val_fraction=0.2,
        split_key="event_id_hash",
        seed=0,
    )
    train_ids = {s["event_id"] for s in train}
    val_ids = {s["event_id"] for s in val}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)


def test_sequence_split_has_no_sequence_leakage():
    from train_token_scorer_oracle import split_oracle_pair_samples

    samples = [
        _sample(f"event_{idx:03d}", sequence_id=f"seq_{idx % 10}")
        for idx in range(100)
    ]
    train, val = split_oracle_pair_samples(
        samples,
        val_fraction=0.2,
        split_key="sequence_id",
        seed=0,
    )
    train_seq = {s["sequence_provenance"]["sequence_id"] for s in train}
    val_seq = {s["sequence_provenance"]["sequence_id"] for s in val}
    assert train_seq
    assert val_seq
    assert train_seq.isdisjoint(val_seq)


def test_summarize_metrics_by_event_type():
    from train_token_scorer_oracle import summarize_metrics_by_event_type

    rows = [
        {"event_type": "dedup", "rank_correct": torch.tensor(True)},
        {"event_type": "dedup", "rank_correct": torch.tensor(False)},
        {"event_type": "eviction", "rank_correct": torch.tensor(True)},
    ]
    summary = summarize_metrics_by_event_type(rows)
    assert summary["dedup"]["count"] == 2
    assert summary["dedup"]["rank_acc"] == 0.5
    assert summary["eviction"]["count"] == 1
    assert summary["eviction"]["rank_acc"] == 1.0


def test_train_token_scorer_records_validation_split(tmp_path):
    from train_token_scorer_oracle import train_token_scorer_oracle

    shard_path = _make_fake_oracle_shard(tmp_path / "oracle.pt", num_events=30)
    output_path = tmp_path / "token_scorer.pt"

    train_token_scorer_oracle(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=16,
        hidden_dim=32,
        num_layers=4,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        regression_weight=0.1,
        min_loss_gap=0.01,
        val_fraction=0.5,
        split_key="event_id_hash",
        split_seed=3,
        device="cpu",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["train_sample_count"] > 0
    assert checkpoint["val_sample_count"] > 0
    assert checkpoint["train_sample_count"] + checkpoint["val_sample_count"] == 30
    assert "validation_metrics" in checkpoint
