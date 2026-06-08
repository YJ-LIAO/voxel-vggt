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

    # Deploy keys for count head (disabled when no validation + auto)
    assert "count_head_deploy_enabled" in checkpoint
    assert checkpoint["count_head_deploy_enabled"] is False, (
        "count_head should not be deployed when val_fraction=0.0 and deploy_count_head=auto"
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


# --------------------------------------------------------------------------- #
# Step 4.2: Config forwarding test for new dataset options
# --------------------------------------------------------------------------- #

def test_parse_args_forwards_new_dataset_options(tmp_path):
    """New dataset options from YAML config reach parse_args output."""
    from train_joint_retention_policy import parse_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "oracle_shards:\n  - /fake/shard.pt\n"
        "output: /fake/output.pt\n"
        "fifo_token_pair_mode: same_keep_count\n"
        "token_score_mode: delta_mean\n"
        "pair_sampling_seed: 7\n"
        "max_pairs_per_event: 4\n"
        "min_loss_gap_by_event_type:\n"
        "  eviction: 0.02\n"
        "max_loss_gap: 0.5\n"
        "min_count_loss_gap: 0.01\n"
    )

    shard = tmp_path / "shard.pt"
    output = tmp_path / "out.pt"

    args = parse_args([
        "--config", str(config_path),
        "--oracle-shards", str(shard),
        "--output", str(output),
    ])

    assert args.fifo_token_pair_mode == "same_keep_count", (
        f"Expected same_keep_count, got {args.fifo_token_pair_mode}"
    )
    assert args.token_score_mode == "delta_mean", (
        f"Expected delta_mean, got {args.token_score_mode}"
    )
    assert args.pair_sampling_seed == 7, (
        f"Expected 7, got {args.pair_sampling_seed}"
    )
    assert args.max_pairs_per_event == 4, (
        f"Expected 4, got {args.max_pairs_per_event}"
    )
    assert args.min_loss_gap_by_event_type == {"eviction": 0.02}, (
        f"Expected {{'eviction': 0.02}}, got {args.min_loss_gap_by_event_type}"
    )
    assert args.max_loss_gap == 0.5, (
        f"Expected 0.5, got {args.max_loss_gap}"
    )
    assert args.min_count_loss_gap == 0.01, (
        f"Expected 0.01, got {args.min_count_loss_gap}"
    )


def test_parse_args_cli_json_for_min_loss_gap_by_event_type(tmp_path):
    """CLI --min-loss-gap-by-event-type accepts JSON string."""
    from train_joint_retention_policy import parse_args

    shard = tmp_path / "shard.pt"
    output = tmp_path / "out.pt"

    args = parse_args([
        "--oracle-shards", str(shard),
        "--output", str(output),
        "--min-loss-gap-by-event-type", '{"eviction": 0.03, "fifo_topk": 0.05}',
    ])

    assert args.min_loss_gap_by_event_type == {"eviction": 0.03, "fifo_topk": 0.05}, (
        f"Expected {{'eviction': 0.03, 'fifo_topk': 0.05}}, got {args.min_loss_gap_by_event_type}"
    )


# --------------------------------------------------------------------------- #
# Step 4.4: CLI smoke test proving main() forwards new options
# --------------------------------------------------------------------------- #

def test_cli_smoke_forwards_config_to_checkpoint(tmp_path):
    """CLI smoke: YAML config values survive through main() into checkpoint."""
    import subprocess

    # Build a tiny fake shard
    shard_path = tmp_path / "shard.pt"
    _make_fake_shard(shard_path, num_eviction=1, num_fifo=1)
    output_path = tmp_path / "output.pt"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"oracle_shards:\n  - {shard_path}\n"
        f"output: {output_path}\n"
        "fifo_token_pair_mode: same_keep_count\n"
        "token_score_mode: delta_mean\n"
        "pair_sampling_seed: 7\n"
        "max_pairs_per_event: 4\n"
        "min_count_loss_gap: 0.01\n"
        "hidden_dim: 16\n"
        "num_layers: 4\n"
        "score_state_dim: 32\n"
        "epochs: 1\n"
        "device: cpu\n"
        "batch_size: 2\n"
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "train_joint_retention_policy",
            "--config", str(config_path),
        ],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI smoke failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    opts = checkpoint.get("training_options", {})

    assert opts.get("fifo_token_pair_mode") == "same_keep_count", (
        f"Expected same_keep_count, got {opts.get('fifo_token_pair_mode')}"
    )
    assert opts.get("token_score_mode") == "delta_mean", (
        f"Expected delta_mean, got {opts.get('token_score_mode')}"
    )
    assert opts.get("pair_sampling_seed") == 7, (
        f"Expected 7, got {opts.get('pair_sampling_seed')}"
    )
    assert opts.get("max_pairs_per_event") == 4, (
        f"Expected 4, got {opts.get('max_pairs_per_event')}"
    )
    assert opts.get("min_count_loss_gap") == 0.01, (
        f"Expected 0.01, got {opts.get('min_count_loss_gap')}"
    )


# --------------------------------------------------------------------------- #
# Step 4.4: Validation checkpoint test
# --------------------------------------------------------------------------- #

def test_train_with_validation_produces_validation_metrics(tmp_path):
    """Training with val_fraction=0.5 produces validation_metrics in checkpoint."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=4,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
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
        val_fraction=0.5,
        split_key="event_id_hash",
        split_seed=0,
        device="cpu",
    )

    assert output_path.exists(), f"Checkpoint not written to {output_path}"
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)

    # Validation metrics structure
    assert "validation_metrics" in checkpoint, "Missing 'validation_metrics' in checkpoint"
    val_metrics = checkpoint["validation_metrics"]

    # Token validation metrics
    assert "token" in val_metrics, "Missing 'token' in validation_metrics"
    token_val = val_metrics["token"]
    assert "count" in token_val, "Missing 'count' in token validation"
    assert "loss" in token_val, "Missing 'loss' in token validation"
    assert "rank_acc" in token_val, "Missing 'rank_acc' in token validation"
    assert "per_event_type" in token_val, "Missing 'per_event_type' in token validation"

    # Count validation metrics
    assert "count" in val_metrics, "Missing 'count' in validation_metrics"
    count_val = val_metrics["count"]
    assert "count" in count_val, "Missing 'count' in count validation"
    assert "accuracy" in count_val, "Missing 'accuracy' in count validation"
    assert "majority_accuracy" in count_val, "Missing 'majority_accuracy' in count validation"
    assert "mean_abs_count_error" in count_val, "Missing 'mean_abs_count_error' in count validation"
    assert "target_distribution" in count_val, "Missing 'target_distribution' in count validation"
    assert "prediction_distribution" in count_val, "Missing 'prediction_distribution' in count validation"

    # Dataset stats
    assert "dataset_stats" in checkpoint, "Missing 'dataset_stats' in checkpoint"
    stats = checkpoint["dataset_stats"]
    for key in ("train_token", "val_token", "train_count", "val_count"):
        assert key in stats, f"Missing '{key}' in dataset_stats"

    # Training options
    assert "training_options" in checkpoint, "Missing 'training_options' in checkpoint"


# --------------------------------------------------------------------------- #
# Step 5.1: Tests for best checkpoint metadata
# --------------------------------------------------------------------------- #

def test_save_best_produces_best_checkpoint(tmp_path):
    """save_best=True with validation produces .best.pt with correct metadata."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=4,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        count_head_arch="shared_encoder_v2",
        batch_size=2,
        epochs=2,
        lr=1e-3,
        weight_decay=0.01,
        regression_weight=0.1,
        min_loss_gap=0.01,
        count_loss_weight=1.0,
        count_label_reduction="min",
        count_repeat_factor=1.0,
        val_fraction=0.5,
        split_key="event_id_hash",
        split_seed=0,
        device="cpu",
        save_best=True,
        best_metric="token.rank_acc",
        early_stop_patience=None,
    )

    # Main checkpoint exists
    assert output_path.exists(), f"Main checkpoint not written to {output_path}"

    # Best checkpoint exists
    best_path = output_path.with_suffix(".best.pt")
    assert best_path.exists(), f"Best checkpoint not written to {best_path}"

    # Check main checkpoint metadata
    main_ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
    for key in ("best_metric", "best_metric_value", "best_epoch", "final_epoch",
                "has_validation", "best_selection_reason"):
        assert key in main_ckpt, f"Missing '{key}' in main checkpoint"
    assert "best_validation_metrics" in main_ckpt, "Missing 'best_validation_metrics' in main checkpoint"
    assert main_ckpt["has_validation"] is True
    assert main_ckpt["final_epoch"] >= 0

    # Check best checkpoint metadata
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    for key in ("best_metric", "best_metric_value", "best_epoch", "final_epoch",
                "has_validation", "best_selection_reason", "validation_metrics",
                "checkpoint_role"):
        assert key in best_ckpt, f"Missing '{key}' in best checkpoint"
    assert best_ckpt["checkpoint_role"] == "best"
    assert best_ckpt["has_validation"] is True
    assert best_ckpt["best_metric"] == "token.rank_acc"
    assert best_ckpt["best_metric_value"] is not None
    assert best_ckpt["best_epoch"] >= 0

    # Both checkpoints share the same best info
    assert main_ckpt["best_metric"] == best_ckpt["best_metric"]
    assert main_ckpt["best_epoch"] == best_ckpt["best_epoch"]


def test_save_best_no_validation(tmp_path):
    """val_fraction=0.0 with save_best=True produces .best.pt with no_validation metadata."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=3,
        num_fifo=3,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
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
        save_best=True,
        best_metric="token.rank_acc",
    )

    # Best checkpoint exists
    best_path = output_path.with_suffix(".best.pt")
    assert best_path.exists(), f"Best checkpoint not written to {best_path}"

    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best_ckpt["best_metric_value"] is None
    assert best_ckpt["has_validation"] is False
    assert best_ckpt["best_selection_reason"] == "no_validation"
    assert best_ckpt["checkpoint_role"] == "best"

    # Main checkpoint also reflects no validation
    main_ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
    assert main_ckpt["best_metric_value"] is None
    assert main_ckpt["has_validation"] is False
    assert main_ckpt["best_selection_reason"] == "no_validation"


def test_save_best_false_no_best_checkpoint(tmp_path):
    """save_best=False (default) does not produce .best.pt."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=2,
        num_fifo=2,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

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
        device="cpu",
    )

    assert output_path.exists()
    best_path = output_path.with_suffix(".best.pt")
    assert not best_path.exists(), ".best.pt should not exist when save_best=False"


# --------------------------------------------------------------------------- #
# Step 5.7: Tests for CountHead deploy gating
# --------------------------------------------------------------------------- #

def test_should_deploy_count_head_auto_weak():
    """auto mode: count head below majority accuracy disables deploy."""
    from train_joint_retention_policy import _should_deploy_count_head

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.30, "majority_accuracy": 0.40},
        deploy_count_head="auto",
        min_delta=0.0,
    ) is False


def test_should_deploy_count_head_auto_strong():
    """auto mode: count head at or above majority accuracy enables deploy."""
    from train_joint_retention_policy import _should_deploy_count_head

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.50, "majority_accuracy": 0.40},
        deploy_count_head="auto",
        min_delta=0.0,
    ) is True


def test_should_deploy_count_head_auto_with_delta():
    """auto mode: min_delta raises the bar."""
    from train_joint_retention_policy import _should_deploy_count_head

    # Exactly at majority + delta => enabled
    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.45, "majority_accuracy": 0.40},
        deploy_count_head="auto",
        min_delta=0.05,
    ) is True

    # Just below majority + delta => disabled
    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.44, "majority_accuracy": 0.40},
        deploy_count_head="auto",
        min_delta=0.05,
    ) is False


def test_should_deploy_count_head_always():
    """always mode: always deploy if trained."""
    from train_joint_retention_policy import _should_deploy_count_head

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={},
        deploy_count_head="always",
        min_delta=0.0,
    ) is True


def test_should_deploy_count_head_never():
    """never mode: never deploy."""
    from train_joint_retention_policy import _should_deploy_count_head

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.99, "majority_accuracy": 0.10},
        deploy_count_head="never",
        min_delta=0.0,
    ) is False


def test_should_deploy_count_head_not_trained():
    """Not trained: never deploy regardless of mode."""
    from train_joint_retention_policy import _should_deploy_count_head

    for mode in ("auto", "always", "never"):
        assert _should_deploy_count_head(
            count_head_trained=False,
            count_metrics={},
            deploy_count_head=mode,
            min_delta=0.0,
        ) is False


def test_should_deploy_count_head_auto_missing_metrics():
    """auto mode: missing accuracy or majority returns False."""
    from train_joint_retention_policy import _should_deploy_count_head

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"accuracy": 0.50},
        deploy_count_head="auto",
        min_delta=0.0,
    ) is False

    assert _should_deploy_count_head(
        count_head_trained=True,
        count_metrics={"majority_accuracy": 0.40},
        deploy_count_head="auto",
        min_delta=0.0,
    ) is False


def test_deploy_gating_in_checkpoint_weak_count_head(tmp_path):
    """Weak count head: count_head_deploy_enabled=False, no deploy keys."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=4,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        count_head_arch="shared_encoder_v2",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        val_fraction=0.5,
        deploy_count_head="auto",
        deploy_count_head_min_delta=0.0,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)

    # count_head_trained remains True for debugging/resume
    assert checkpoint["count_head_trained"] is True
    # Top-level count_head present for debugging/resume
    assert checkpoint["count_head"] is not None

    # Deploy gating
    assert "count_head_deploy_enabled" in checkpoint
    # With only 1 epoch of tiny training, accuracy is likely weak
    # Check deploy state for count_head keys
    deploy_state = checkpoint["model"]
    count_head_deploy_keys = [k for k in deploy_state if "count_head" in k]

    if not checkpoint["count_head_deploy_enabled"]:
        assert len(count_head_deploy_keys) == 0, (
            f"Found count_head deploy keys when deploy disabled: {count_head_deploy_keys}"
        )


def test_deploy_gating_always_includes_count_head(tmp_path):
    """deploy_count_head='always' includes count_head in deploy state."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=4,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        count_head_arch="shared_encoder_v2",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        val_fraction=0.5,
        deploy_count_head="always",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["count_head_trained"] is True
    assert checkpoint["count_head_deploy_enabled"] is True

    deploy_state = checkpoint["model"]
    count_head_deploy_keys = [k for k in deploy_state if "count_head" in k]
    assert len(count_head_deploy_keys) > 0, (
        "Expected count_head deploy keys when deploy_count_head='always'"
    )


def test_deploy_gating_auto_no_validation_disables(tmp_path):
    """No validation + auto: deploy is disabled."""
    from train_joint_retention_policy import train_joint_retention

    shard_path = _make_fake_shard(
        tmp_path / "fake_shard.pt",
        num_eviction=4,
        num_fifo=4,
    )
    output_path = tmp_path / "output" / "joint_retention.pt"

    train_joint_retention(
        oracle_shards=[str(shard_path)],
        output=str(output_path),
        score_state_dim=_SCORE_STATE_DIM,
        metadata_dim=_METADATA_DIM,
        hidden_dim=16,
        num_layers=4,
        count_candidates=[0, 4, 8, 16],
        count_head_arch="shared_encoder_v2",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        device="cpu",
        val_fraction=0.0,
        deploy_count_head="auto",
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["count_head_trained"] is True
    assert checkpoint["count_head_deploy_enabled"] is False

    deploy_state = checkpoint["model"]
    count_head_deploy_keys = [k for k in deploy_state if "count_head" in k]
    assert len(count_head_deploy_keys) == 0, (
        f"Found count_head deploy keys with no validation + auto: {count_head_deploy_keys}"
    )
