"""Tests for FifoCountDataset and collate_fifo_count_samples."""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_fifo_shard(path, event_overrides=None, shard_overrides=None):
    """Helper to create a fake shard with one fifo_topk event and measured subsets."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM

    base_event = {
        "event_id": "sceneA:frame4:layer2",
        "event_type": "fifo_topk",
        "layer_id": 2,
        "score_state": torch.randn(20, 128),
        "metadata_features": torch.randn(20, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([3, 7, 10, 14]),
        "subsets": [
            {"keep_count": 0, "loss": 3.0},
            {"keep_count": 8, "loss": 1.0},
            {"keep_count": 8, "loss": 1.2},
            {"keep_count": 16, "loss": 2.0},
        ],
    }
    if event_overrides:
        base_event.update(event_overrides)

    shard = {
        "format": "ovggt_counterfactual_oracle_v1",
        "num_events": 1,
        "events": [base_event],
    }
    if shard_overrides:
        shard.update(shard_overrides)

    torch.save(shard, path)
    return path


def test_fifo_count_dataset_target_is_candidate_index_for_count_8(tmp_path):
    """Dataset target is the candidate index for keep_count=8 (best loss)."""
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    shard_path = _make_fifo_shard(tmp_path / "shard.pt")

    dataset = FifoCountDataset(
        [shard_path],
        count_candidates=(0, 8, 16, 32, 64, 128),
        label_reduction="min",
    )
    assert len(dataset) == 1

    sample = dataset[0]
    # Best grouped loss: keep_count=8 has min loss 1.0 (vs 0->3.0, 16->2.0)
    # Index of 8 in count_candidates=(0, 8, 16, 32, 64, 128) is 1
    assert sample["target"] == 1
    assert sample["target_keep_count"] == 8


def test_fifo_count_dataset_slices_to_demoted_indices(tmp_path):
    """score_state and metadata_features are sliced to demoted_indices, not full cache tokens."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    demoted = torch.tensor([3, 7, 10, 14])
    score_state = torch.arange(20 * 128, dtype=torch.float32).reshape(20, 128)
    metadata = torch.arange(20 * TOKEN_METADATA_FEATURE_DIM, dtype=torch.float32).reshape(
        20, TOKEN_METADATA_FEATURE_DIM
    )

    shard_path = _make_fifo_shard(
        tmp_path / "shard.pt",
        event_overrides={
            "score_state": score_state,
            "metadata_features": metadata,
            "demoted_indices": demoted,
        },
    )

    dataset = FifoCountDataset([shard_path])
    sample = dataset[0]

    # Should be sliced to demoted_indices: [K=4, Ds] and [K=4, Dm]
    assert sample["score_state"].shape == (4, 128)
    assert sample["metadata_features"].shape == (4, TOKEN_METADATA_FEATURE_DIM)

    # Verify the actual values are correct slices
    assert torch.equal(sample["score_state"], score_state[demoted])
    assert torch.equal(sample["metadata_features"], metadata[demoted])


def test_fifo_count_dataset_fallback_when_no_demoted_indices(tmp_path):
    """When demoted_indices is missing, fall back to full cache tokens."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    shard_path = _make_fifo_shard(
        tmp_path / "shard.pt",
        event_overrides={
            # Remove demoted_indices to simulate old shard
        },
    )
    # Load, remove demoted_indices, re-save
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    del shard["events"][0]["demoted_indices"]
    torch.save(shard, shard_path)

    dataset = FifoCountDataset([shard_path])
    assert len(dataset) == 1
    sample = dataset[0]

    # Should use full 20 tokens
    assert sample["score_state"].shape == (20, 128)
    assert sample["metadata_features"].shape == (20, TOKEN_METADATA_FEATURE_DIM)


def test_fifo_count_dataset_missing_subset_keep_count_falls_back(tmp_path):
    """Missing subset keep_count falls back to event-level keep_count for old shards."""
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM

    event = {
        "event_id": "old_shard_event",
        "event_type": "fifo_topk",
        "layer_id": 3,
        "keep_count": 8,  # event-level fallback
        "score_state": torch.randn(10, 128),
        "metadata_features": torch.randn(10, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([1, 4, 6]),
        "subsets": [
            # No keep_count in subsets -> should fall back to event-level 8
            {"loss": 0.5},
            {"loss": 0.8},
        ],
    }
    shard_path = tmp_path / "old_shard.pt"
    torch.save({"events": [event]}, shard_path)

    dataset = FifoCountDataset(
        [shard_path],
        count_candidates=(0, 8, 16),
        label_reduction="min",
    )
    assert len(dataset) == 1
    sample = dataset[0]
    # All subsets have keep_count=8 (fallback), so best is 8 -> index 1
    assert sample["target"] == 1
    assert sample["target_keep_count"] == 8


def test_fifo_count_dataset_skips_events_with_no_candidate_match(tmp_path):
    """Skip events where no measured count is in configured candidates."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    event = {
        "event_id": "no_match_event",
        "event_type": "fifo_topk",
        "layer_id": 0,
        "score_state": torch.randn(10, 128),
        "metadata_features": torch.randn(10, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([1, 2]),
        "subsets": [
            {"keep_count": 5, "loss": 0.1},
            {"keep_count": 7, "loss": 0.2},
        ],
    }
    shard_path = tmp_path / "no_match.pt"
    torch.save({"events": [event]}, shard_path)

    # candidates don't include 5 or 7
    dataset = FifoCountDataset(
        [shard_path],
        count_candidates=(0, 8, 16, 32),
    )
    assert len(dataset) == 0


def test_fifo_count_dataset_skips_non_fifo_topk_events(tmp_path):
    """Only fifo_topk events are processed."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    event = {
        "event_id": "eviction_event",
        "event_type": "eviction",
        "layer_id": 0,
        "score_state": torch.randn(10, 128),
        "metadata_features": torch.randn(10, TOKEN_METADATA_FEATURE_DIM),
        "subsets": [
            {"keep_count": 4, "loss": 0.1},
            {"keep_count": 8, "loss": 0.2},
        ],
    }
    shard_path = tmp_path / "eviction.pt"
    torch.save({"events": [event]}, shard_path)

    dataset = FifoCountDataset([shard_path])
    assert len(dataset) == 0


def test_fifo_count_dataset_label_reduction_min_selects_best_per_group(tmp_path):
    """With label_reduction='min', the best loss per keep_count group is used."""
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    # keep_count=8 has two subsets with losses [1.0, 1.2] -> min is 1.0
    # keep_count=16 has loss 0.5 -> would be best
    # But let's check a scenario where grouping matters
    event_overrides = {
        "subsets": [
            {"keep_count": 8, "loss": 1.0},
            {"keep_count": 8, "loss": 0.3},  # min for group 8 is 0.3
            {"keep_count": 16, "loss": 0.5},
            {"keep_count": 16, "loss": 0.6},
        ],
    }
    shard_path = _make_fifo_shard(tmp_path / "shard.pt", event_overrides=event_overrides)

    dataset = FifoCountDataset(
        [shard_path],
        count_candidates=(0, 8, 16, 32),
        label_reduction="min",
    )
    sample = dataset[0]
    # Group 8: min loss = 0.3, group 16: min loss = 0.5
    # Best is group 8 -> index 1
    assert sample["target"] == 1
    assert sample["target_keep_count"] == 8


def test_fifo_count_dataset_returns_correct_keys(tmp_path):
    """Each sample has the expected keys."""
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    shard_path = _make_fifo_shard(tmp_path / "shard.pt")

    dataset = FifoCountDataset([shard_path])
    sample = dataset[0]

    expected_keys = {"event_id", "layer_id", "score_state", "metadata_features", "target", "target_keep_count"}
    assert set(sample.keys()) == expected_keys
    assert isinstance(sample["event_id"], str)
    assert isinstance(sample["layer_id"], int)
    assert isinstance(sample["target"], int)
    assert isinstance(sample["target_keep_count"], int)
    assert isinstance(sample["score_state"], torch.Tensor)
    assert isinstance(sample["metadata_features"], torch.Tensor)


def test_fifo_count_dataset_multiple_events(tmp_path):
    """Multiple events in a shard are all processed."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    events = [
        {
            "event_id": f"event_{i}",
            "event_type": "fifo_topk",
            "layer_id": i,
            "score_state": torch.randn(10, 128),
            "metadata_features": torch.randn(10, TOKEN_METADATA_FEATURE_DIM),
            "demoted_indices": torch.tensor([1, 3, 5]),
            "subsets": [
                {"keep_count": 8, "loss": float(i) + 0.1},
                {"keep_count": 16, "loss": float(i) + 0.5},
            ],
        }
        for i in range(3)
    ]
    shard_path = tmp_path / "multi.pt"
    torch.save({"events": events}, shard_path)

    dataset = FifoCountDataset([shard_path], count_candidates=(0, 8, 16, 32))
    assert len(dataset) == 3
    for i in range(3):
        sample = dataset[i]
        # keep_count=8 has lower loss; index of 8 in (0,8,16,32) is 1
        assert sample["target"] == 1
        assert sample["target_keep_count"] == 8


def test_collate_fifo_count_samples_pads_and_returns_mask(tmp_path):
    """collate_fifo_count_samples pads score_state/metadata_features and returns token_mask."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset, collate_fifo_count_samples

    # Create two events with different demoted_indices sizes
    event1 = {
        "event_id": "e1",
        "event_type": "fifo_topk",
        "layer_id": 1,
        "score_state": torch.randn(20, 128),
        "metadata_features": torch.randn(20, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([1, 3, 5]),  # K=3
        "subsets": [
            {"keep_count": 8, "loss": 0.5},
            {"keep_count": 16, "loss": 1.0},
        ],
    }
    event2 = {
        "event_id": "e2",
        "event_type": "fifo_topk",
        "layer_id": 2,
        "score_state": torch.randn(20, 128),
        "metadata_features": torch.randn(20, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([0, 2, 4, 6, 8]),  # K=5
        "subsets": [
            {"keep_count": 8, "loss": 1.5},
            {"keep_count": 16, "loss": 0.3},
        ],
    }
    shard_path = tmp_path / "two_events.pt"
    torch.save({"events": [event1, event2]}, shard_path)

    dataset = FifoCountDataset([shard_path], count_candidates=(0, 8, 16, 32))
    assert len(dataset) == 2

    batch = collate_fifo_count_samples([dataset[0], dataset[1]])

    # Padded to max K=5
    assert batch["score_state"].shape == (2, 5, 128)
    assert batch["metadata_features"].shape == (2, 5, TOKEN_METADATA_FEATURE_DIM)
    assert batch["token_mask"].shape == (2, 5)
    assert batch["token_mask"][0].tolist() == [True, True, True, False, False]
    assert batch["token_mask"][1].tolist() == [True, True, True, True, True]

    assert batch["layer_id"].tolist() == [1, 2]
    # e1: best=8 (loss 0.5 < 1.0), index of 8 in (0,8,16,32) is 1
    # e2: best=16 (loss 0.3 < 1.5), index of 16 in (0,8,16,32) is 2
    assert batch["target"].tolist() == [1, 2]
    assert batch["target_keep_count"].tolist() == [8, 16]
    assert batch["event_id"] == ["e1", "e2"]


def test_collate_fifo_count_samples_single_sample(tmp_path):
    """collate_fifo_count_samples works with a single sample."""
    from ovggt.training.token_oracle_dataset import FifoCountDataset, collate_fifo_count_samples

    shard_path = _make_fifo_shard(tmp_path / "shard.pt")
    dataset = FifoCountDataset([shard_path])
    sample = dataset[0]

    batch = collate_fifo_count_samples([sample])

    assert batch["score_state"].shape[0] == 1
    assert batch["token_mask"].all().item()  # single sample, no padding needed


def test_fifo_count_dataset_handles_list_shard_format(tmp_path):
    """Handles shards that are plain lists (old format) instead of dicts with 'events' key."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    event = {
        "event_id": "list_event",
        "event_type": "fifo_topk",
        "layer_id": 0,
        "score_state": torch.randn(10, 128),
        "metadata_features": torch.randn(10, TOKEN_METADATA_FEATURE_DIM),
        "demoted_indices": torch.tensor([1, 2]),
        "subsets": [
            {"keep_count": 8, "loss": 0.5},
            {"keep_count": 16, "loss": 1.0},
        ],
    }
    shard_path = tmp_path / "list_shard.pt"
    torch.save([event], shard_path)  # plain list

    dataset = FifoCountDataset([shard_path], count_candidates=(0, 8, 16))
    assert len(dataset) == 1
    # keep_count=8 has loss 0.5 < 1.0; index of 8 in (0,8,16) is 1
    assert dataset[0]["target"] == 1
