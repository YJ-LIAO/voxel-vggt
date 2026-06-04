import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_counterfactual_oracle_dataset_trains_scorer_one_step(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TokenScorer
    from ovggt.training.token_oracle_dataset import (
        CounterfactualOracleDataset,
        collate_oracle_pairs,
        token_oracle_ranking_loss,
    )

    shard_path = tmp_path / "oracle_shard.pt"
    torch.save(
        {
            "events": [
                {
                    "event_id": "sceneA:frame4:layer2",
                    "layer_id": 2,
                    "sequence_provenance": {
                        "dataset_key": "dataset_arkitscenes",
                        "sequence_id": "arkitscenes/sceneA",
                        "frame_count": 3,
                        "frames": [
                            {"frame_index": 0, "label": "sceneA_0000", "instance": "/data/sceneA/0000.jpg"},
                            {"frame_index": 1, "label": "sceneA_0001", "instance": "/data/sceneA/0001.jpg"},
                            {"frame_index": 2, "label": "sceneA_0002", "instance": "/data/sceneA/0002.jpg"},
                        ],
                    },
                    "score_state": torch.randn(6, 8),
                    "metadata_features": torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
                    "subsets": [
                        {
                            "keep_indices": torch.tensor([0, 2, 4]),
                            "loss": 0.20,
                            "loss_components": {"camera": 0.003, "depth": 0.004, "point_map": 0.006},
                        },
                        {
                            "keep_indices": torch.tensor([1, 3, 5]),
                            "loss": 0.55,
                            "loss_components": {"camera": 0.011, "depth": 0.010, "point_map": 0.013},
                        },
                        {
                            "keep_indices": torch.tensor([0, 1, 5]),
                            "loss": 0.35,
                            "loss_components": {"camera": 0.007, "depth": 0.006, "point_map": 0.009},
                        },
                    ],
                }
            ]
        },
        shard_path,
    )

    dataset = CounterfactualOracleDataset([shard_path])
    assert len(dataset) == 3

    batch = collate_oracle_pairs([dataset[0], dataset[1], dataset[2]])
    assert batch["score_state"].shape == (3, 6, 8)
    assert batch["better_mask"].shape == (3, 6)
    assert batch["worse_mask"].shape == (3, 6)
    assert torch.all(batch["target_margin"] > 0)
    assert batch["sequence_provenance"][0]["sequence_id"] == "arkitscenes/sceneA"
    assert batch["sequence_provenance"][0]["frames"][2]["instance"] == "/data/sceneA/0002.jpg"

    scorer = TokenScorer(score_state_dim=8, metadata_dim=TOKEN_METADATA_FEATURE_DIM, hidden_dim=16, num_layers=4)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    logits = scorer(batch["score_state"], batch["metadata_features"], batch["layer_id"])
    loss, loss_details = token_oracle_ranking_loss(logits, batch)

    assert torch.isfinite(loss)
    assert loss_details["pairwise"] >= 0
    assert loss_details["regression"] >= 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(param.grad is not None for param in scorer.parameters())


def test_counterfactual_oracle_dataset_ignores_replay_payload(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    base_event = {
        "event_id": "sceneA:frame4:layer2",
        "layer_id": 2,
        "sequence_provenance": {"sequence_id": "sceneA"},
        "score_state": torch.arange(24, dtype=torch.float32).reshape(3, 8),
        "metadata_features": torch.arange(3 * TOKEN_METADATA_FEATURE_DIM, dtype=torch.float32).reshape(
            3,
            TOKEN_METADATA_FEATURE_DIM,
        ),
        "subsets": [
            {
                "keep_indices": torch.tensor([0, 1]),
                "loss": 0.1,
                "loss_components": {"camera": 0.1, "depth": 0.0, "point_map": 0.0},
            },
            {
                "keep_indices": torch.tensor([1, 2]),
                "loss": 0.4,
                "loss_components": {"camera": 0.4, "depth": 0.0, "point_map": 0.0},
            },
        ],
    }
    payload_event = {
        **base_event,
        "subsets": [
            {
                **subset,
                "replay": {
                    "predictions": [{"depth": torch.ones(1, 2, 2, 1)}],
                    "targets": [{"depth": torch.zeros(1, 2, 2, 1)}],
                },
            }
            for subset in base_event["subsets"]
        ],
    }
    compact_path = tmp_path / "compact.pt"
    payload_path = tmp_path / "payload.pt"
    torch.save({"events": [base_event]}, compact_path)
    torch.save({"events": [payload_event]}, payload_path)

    compact = CounterfactualOracleDataset([compact_path])
    payload = CounterfactualOracleDataset([payload_path])

    assert len(compact) == len(payload) == 1
    for key, compact_value in compact[0].items():
        payload_value = payload[0][key]
        if isinstance(compact_value, torch.Tensor):
            assert torch.equal(compact_value, payload_value)
        else:
            assert compact_value == payload_value


def test_collate_oracle_pairs_preserves_sequence_provenance_list():
    from ovggt.training.token_oracle_dataset import collate_oracle_pairs

    samples = [
        {
            "event_id": "e0",
            "layer_id": 1,
            "score_state": torch.zeros(2, 3),
            "metadata_features": torch.zeros(2, 16),
            "better_mask": torch.tensor([True, False]),
            "worse_mask": torch.tensor([False, True]),
            "better_loss": 0.1,
            "worse_loss": 0.2,
            "target_margin": 0.1,
            "better_target": 1.0,
            "worse_target": 0.0,
            "sequence_provenance": {"sequence_id": "seq-a"},
        },
        {
            "event_id": "e1",
            "layer_id": 1,
            "score_state": torch.zeros(2, 3),
            "metadata_features": torch.zeros(2, 16),
            "better_mask": torch.tensor([True, False]),
            "worse_mask": torch.tensor([False, True]),
            "better_loss": 0.3,
            "worse_loss": 0.4,
            "target_margin": 0.1,
            "better_target": 1.0,
            "worse_target": 0.0,
            "sequence_provenance": {"sequence_id": "seq-b"},
        },
    ]

    batch = collate_oracle_pairs(samples)
    assert batch["sequence_provenance"][0]["sequence_id"] == "seq-a"
    assert batch["sequence_provenance"][1]["sequence_id"] == "seq-b"


def test_collate_oracle_pairs_pads_variable_token_lengths():
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TokenScorer
    from ovggt.training.token_oracle_dataset import collate_oracle_pairs, token_oracle_ranking_loss

    samples = [
        {
            "event_id": "e0",
            "layer_id": 1,
            "score_state": torch.randn(2, 8),
            "metadata_features": torch.randn(2, TOKEN_METADATA_FEATURE_DIM),
            "better_mask": torch.tensor([True, False]),
            "worse_mask": torch.tensor([False, True]),
            "better_loss": 0.1,
            "worse_loss": 0.2,
            "target_margin": 0.1,
            "better_target": 1.0,
            "worse_target": 0.0,
            "sequence_provenance": {"sequence_id": "seq-a"},
        },
        {
            "event_id": "e1",
            "layer_id": 1,
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "better_mask": torch.tensor([True, False, False, True]),
            "worse_mask": torch.tensor([False, True, True, False]),
            "better_loss": 0.3,
            "worse_loss": 0.7,
            "target_margin": 0.4,
            "better_target": 1.0,
            "worse_target": 0.0,
            "sequence_provenance": {"sequence_id": "seq-b"},
        },
    ]

    batch = collate_oracle_pairs(samples)

    assert batch["score_state"].shape == (2, 4, 8)
    assert batch["metadata_features"].shape == (2, 4, TOKEN_METADATA_FEATURE_DIM)
    assert batch["better_mask"].shape == (2, 4)
    assert batch["worse_mask"].shape == (2, 4)
    assert batch["token_mask"].tolist() == [[True, True, False, False], [True, True, True, True]]
    assert batch["better_mask"][0].tolist() == [True, False, False, False]
    assert batch["worse_mask"][0].tolist() == [False, True, False, False]

    scorer = TokenScorer(score_state_dim=8, metadata_dim=TOKEN_METADATA_FEATURE_DIM, hidden_dim=16, num_layers=2)
    logits = scorer(batch["score_state"], batch["metadata_features"], batch["layer_id"])
    loss, details = token_oracle_ranking_loss(logits, batch)

    assert torch.isfinite(loss)
    assert details["pairwise"] >= 0


def test_oracle_training_config_and_deploy_state_dict(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TokenScorer
    from train_token_scorer_oracle import (
        build_ovggt_token_scorer_state_dict,
        load_score_state_projection_state_from_oracle_shards,
        parse_args,
    )

    shard_path = tmp_path / "oracle.pt"
    output_path = tmp_path / "token_scorer.pt"
    config_path = tmp_path / "train_token_scorer_oracle.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"oracle_shards: [{shard_path}]",
                f"output: {output_path}",
                "score_state_dim: 8",
                "hidden_dim: 16",
                "num_layers: 2",
                "batch_size: 4",
                "epochs: 5",
                "lr: 0.001",
                "regression_weight: 0.2",
                "min_loss_gap: 0.02",
                "device: cpu",
            ]
        )
    )

    args = parse_args(["--config", str(config_path), "--epochs", "3", "--min-loss-gap", "0.03"])

    assert args.oracle_shards == [str(shard_path)]
    assert args.output == str(output_path)
    assert args.score_state_dim == 8
    assert args.hidden_dim == 16
    assert args.num_layers == 2
    assert args.epochs == 3
    assert args.min_loss_gap == 0.03
    assert args.device == "cpu"

    scorer = TokenScorer(
        score_state_dim=8,
        metadata_dim=TOKEN_METADATA_FEATURE_DIM,
        hidden_dim=16,
        num_layers=2,
    )
    projection_state = {
        "aggregator.score_state_projs.0.weight": torch.randn(8, 32),
        "aggregator.score_state_projs.0.bias": torch.randn(8),
        "aggregator.score_state_projs.1.weight": torch.randn(8, 32),
        "aggregator.score_state_projs.1.bias": torch.randn(8),
    }

    deploy_state = build_ovggt_token_scorer_state_dict(
        scorer_state=scorer.state_dict(),
        num_layers=2,
        score_state_projection_state=projection_state,
    )

    assert "aggregator.token_scorers.0.scorer.1.weight" in deploy_state
    assert "aggregator.token_scorers.1.scorer.1.weight" in deploy_state
    assert torch.equal(
        deploy_state["aggregator.token_scorers.0.scorer.1.weight"],
        deploy_state["aggregator.token_scorers.1.scorer.1.weight"],
    )
    assert torch.equal(
        deploy_state["aggregator.score_state_projs.0.weight"],
        projection_state["aggregator.score_state_projs.0.weight"],
    )

    shard_with_projection = tmp_path / "oracle_with_projection.pt"
    torch.save({"score_state_projection_state": projection_state, "events": []}, shard_with_projection)
    loaded_projection = load_score_state_projection_state_from_oracle_shards([shard_with_projection])
    assert torch.equal(
        loaded_projection["aggregator.score_state_projs.1.bias"],
        projection_state["aggregator.score_state_projs.1.bias"],
    )


def test_oracle_training_log_line_excludes_provenance_fields():
    from train_token_scorer_oracle import format_epoch_summary_line, format_training_log_line

    line = format_training_log_line(
        epoch=0,
        step=50,
        loss=0.123456,
        details={
            "pairwise": 0.111111,
            "regression": 0.222222,
            "rank_acc": 0.75,
            "mean_score_diff": 1.25,
        },
    )

    assert "provenance" not in line
    assert "sequence" not in line
    assert line == (
        "epoch=0 step=50 loss=0.123456 pairwise=0.111111 "
        "regression=0.222222 rank_acc=0.7500 mean_score_diff=1.250000"
    )

    summary = format_epoch_summary_line(
        epoch=0,
        metrics={
            "count": 2,
            "loss": 0.4,
            "pairwise": 0.3,
            "regression": 0.1,
            "rank_acc": 1.5,
            "mean_score_diff": 3.0,
        },
    )
    assert "provenance" not in summary
    assert "sequence" not in summary
    assert summary == (
        "epoch_summary=0 batches=2 loss=0.200000 pairwise=0.150000 "
        "regression=0.050000 rank_acc=0.7500 mean_score_diff=1.500000"
    )


def test_counterfactual_oracle_dataset_filters_tiny_loss_gap_pairs(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    shard_path = tmp_path / "oracle.pt"
    torch.save(
        {
            "events": [
                {
                    "event_id": "event0",
                    "layer_id": 0,
                    "score_state": torch.randn(4, 8),
                    "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                    "subsets": [
                        {"keep_indices": [0, 1], "loss": 1.000},
                        {"keep_indices": [1, 2], "loss": 1.005},
                        {"keep_indices": [2, 3], "loss": 1.030},
                    ],
                }
            ]
        },
        shard_path,
    )

    dataset = CounterfactualOracleDataset([shard_path], min_loss_gap=0.01)

    assert len(dataset) == 2
    assert [round(sample["target_margin"], 3) for sample in dataset] == [0.03, 0.025]


def test_token_oracle_ranking_loss_uses_sum_scores_for_pairwise_ranking():
    import torch.nn.functional as F
    from ovggt.training.token_oracle_dataset import token_oracle_ranking_loss

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    batch = {
        "better_mask": torch.tensor([[True, False, True, False]]),
        "worse_mask": torch.tensor([[False, True, False, True]]),
        "target_margin": torch.tensor([0.0]),
        "better_target": torch.tensor([0.0]),
        "worse_target": torch.tensor([0.0]),
    }

    loss, details = token_oracle_ranking_loss(logits, batch, regression_weight=0.0)

    assert torch.allclose(loss, F.softplus(torch.tensor(1.0)))
    assert details["rank_acc"] == 0.0
    assert details["mean_score_diff"] == -1.0


def test_oracle_generator_replay_mode_computes_three_task_losses(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from tools.generate_counterfactual_oracle import (
        build_events_from_args,
        parse_args,
    )

    synthetic_args = parse_args(
        [
            "--mode",
            "synthetic",
            "--synthetic-events",
            "1",
            "--synthetic-tokens",
            "4",
            "--score-state-dim",
            "8",
            "--output",
            str(tmp_path / "synthetic.pt"),
        ]
    )
    synthetic_events = build_events_from_args(synthetic_args)
    assert len(synthetic_events) == 1
    assert synthetic_events[0]["metadata_features"].shape[-1] == TOKEN_METADATA_FEATURE_DIM

    replay_dump = tmp_path / "replay_dump.pt"
    torch.save(
        {
            "events": [
                {
                    "event_id": "scene:frame4:layer1",
                    "layer_id": 1,
                    "frame_id": 4,
                    "budget": 2,
                    "score_state": torch.randn(4, 8),
                    "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                    "subsets": [
                        {
                            "keep_indices": [0, 1],
                            "replay": {
                                "predictions": [
                                    {
                                        "camera_pose": torch.tensor([1.0, 0.0]),
                                        "depth": torch.tensor([[2.0, 4.0]]),
                                        "pts3d_in_other_view": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
                                    }
                                ],
                                "targets": [
                                    {
                                        "camera_pose": torch.tensor([0.0, 0.0]),
                                        "depth": torch.tensor([[1.0, 1.0]]),
                                        "pts3d_in_other_view": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
                                        "valid_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                                    }
                                ],
                            },
                        },
                        {
                            "keep_indices": [2, 3],
                            "replay": {
                                "predictions": [
                                    {
                                        "camera_pose": torch.tensor([0.0, 0.0]),
                                        "depth": torch.tensor([[1.0, 1.0]]),
                                        "pts3d_in_other_view": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
                                    }
                                ],
                                "targets": [
                                    {
                                        "camera_pose": torch.tensor([0.0, 0.0]),
                                        "depth": torch.tensor([[1.0, 1.0]]),
                                        "pts3d_in_other_view": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
                                        "valid_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                                    }
                                ],
                            },
                        },
                    ],
                }
            ]
        },
        replay_dump,
    )

    replay_args = parse_args(
        [
            "--mode",
            "replay",
            "--event-dump",
            str(replay_dump),
            "--output",
            str(tmp_path / "replay.pt"),
        ]
    )
    replay_events = build_events_from_args(replay_args)

    assert len(replay_events) == 1
    assert replay_events[0]["subsets"][0]["loss_components"] == {
        "camera": 0.5,
        "depth": 2.0,
        "point_map": 1.0,
    }
    assert replay_events[0]["subsets"][0]["loss"] == 20.0 * 0.5 + 20.0 * 2.0 + 10.0 * 1.0
    assert replay_events[0]["subsets"][1]["loss"] == 0.0


def test_oracle_generator_preserves_sequence_provenance_from_replay_dump(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from tools.generate_counterfactual_oracle import build_oracle_events_from_replay_dump

    provenance = {
        "dataset_key": "train_dataset",
        "dataset": "arkitscenes",
        "sequence_id": "arkitscenes/sceneA",
        "frames": [{"frame_index": 0, "instance": "/data/sceneA/0000.jpg"}],
    }
    events = build_oracle_events_from_replay_dump(
        [
            {
                "event_id": "scene:frame0:layer1",
                "layer_id": 1,
                "frame_id": 0,
                "budget": 2,
                "sequence_provenance": provenance,
                "score_state": torch.randn(4, 8),
                "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
                "subsets": [
                    {
                        "keep_indices": [0, 1],
                        "replay": {
                            "predictions": [{"camera_pose": torch.zeros(2)}],
                            "targets": [{"camera_pose": torch.zeros(2)}],
                        },
                    },
                    {
                        "keep_indices": [2, 3],
                        "replay": {
                            "predictions": [{"camera_pose": torch.ones(2)}],
                            "targets": [{"camera_pose": torch.zeros(2)}],
                        },
                    },
                ],
            }
        ]
    )

    assert events[0]["sequence_provenance"] == provenance


def test_learned_eviction_oracle_eval_compares_against_heuristic(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from tools.evaluate_learned_eviction import evaluate_oracle_shards

    shard_path = tmp_path / "oracle_eval.pt"
    torch.save(
        {
            "events": [
                {
                    "event_id": "event0",
                    "layer_id": 2,
                    "frame_id": 5,
                    "budget": 2,
                    "score_state": torch.zeros(4, 8),
                    "metadata_features": torch.zeros(4, TOKEN_METADATA_FEATURE_DIM),
                    "base_scores": torch.tensor([10.0, 9.0, 0.0, 0.0]),
                    "learned_token_scores": torch.tensor([0.0, 0.0, 10.0, 9.0]),
                    "token_frame_ids": torch.tensor([0, 0, 5, 5]),
                    "subsets": [
                        {
                            "keep_indices": torch.tensor([0, 1]),
                            "loss": 2.0,
                            "loss_components": {"camera": 0.05, "depth": 0.025, "point_map": 0.05},
                        },
                        {
                            "keep_indices": torch.tensor([2, 3]),
                            "loss": 0.5,
                            "loss_components": {"camera": 0.01, "depth": 0.01, "point_map": 0.01},
                        },
                    ],
                }
            ]
        },
        shard_path,
    )

    report = evaluate_oracle_shards([shard_path], token_scorer=None, device="cpu")

    assert report["learned"]["weighted_total"] == 0.5
    assert report["heuristic"]["weighted_total"] == 2.0
    assert report["delta_learned_minus_heuristic"] == -1.5
    assert report["cache_distribution"]["learned_by_layer"] == {"2": 2}
    assert report["cache_distribution"]["learned_by_token_frame"] == {"5": 2}


def test_learned_eviction_eval_rejects_empty_oracle_shards(tmp_path):
    from tools.evaluate_learned_eviction import evaluate_oracle_shards

    shard_path = tmp_path / "empty_oracle.pt"
    torch.save({"events": []}, shard_path)

    try:
        evaluate_oracle_shards([shard_path], token_scorer=None, device="cpu")
    except ValueError as exc:
        assert "No oracle events were evaluated" in str(exc)
    else:
        raise AssertionError("Empty oracle evaluation must fail explicitly")


def test_counterfactual_replay_collector_snapshots_restores_and_replays_future_window():
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.counterfactual_replay import (
        CounterfactualReplayCandidateEvent,
        collect_counterfactual_replay_event,
    )

    class FakeReplayRunner:
        def __init__(self):
            self.keep_indices = None
            self.state = {"cache_tokens": torch.arange(4)}
            self.snapshots = 0
            self.restores = 0
            self.applied = []
            self.replayed = []
            self.prepared = []

        def prepare_event(self, event):
            self.prepared.append(event.event_id)

        def snapshot(self):
            self.snapshots += 1
            return {"state": self.state["cache_tokens"].clone()}

        def restore(self, snapshot):
            self.restores += 1
            self.state["cache_tokens"] = snapshot["state"].clone()
            self.keep_indices = None

        def apply_keep_indices(self, layer_id, keep_indices):
            self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long).clone()
            self.state["cache_tokens"] = self.state["cache_tokens"][self.keep_indices]
            self.applied.append((layer_id, tuple(self.keep_indices.tolist())))

        def replay_future_window(self, start_frame_idx, future_frames):
            self.replayed.append((start_frame_idx, tuple(frame["frame_id"] for frame in future_frames)))
            quality = float(self.keep_indices.float().mean().item())
            return [
                {
                    "camera_pose": torch.tensor([quality, 0.0]),
                    "depth": torch.full((1, 2), quality),
                    "pts3d_in_other_view": torch.full((1, 2, 3), quality),
                }
                for _ in future_frames
            ]

        def targets_for_future_window(self, future_frames):
            return [
                {
                    "camera_pose": torch.tensor([2.5, 0.0]),
                    "depth": torch.full((1, 2), 2.5),
                    "pts3d_in_other_view": torch.full((1, 2, 3), 2.5),
                    "valid_mask": torch.ones(1, 2, dtype=torch.bool),
                }
                for _ in future_frames
            ]

    runner = FakeReplayRunner()
    event = CounterfactualReplayCandidateEvent(
        event_id="scene:frame4:layer0",
        layer_id=0,
        frame_id=4,
        budget=2,
        sequence_provenance={"sequence_id": "synthetic/seq0"},
        score_state=torch.zeros(4, 8),
        metadata_features=torch.zeros(4, TOKEN_METADATA_FEATURE_DIM),
        protected_indices=torch.tensor([], dtype=torch.long),
        base_scores=torch.tensor([3.0, 2.0, 1.0, 0.0]),
        candidate_subsets=[torch.tensor([0, 1]), torch.tensor([2, 3])],
    )
    future_frames = [{"frame_id": 5}, {"frame_id": 6}]

    replay_event = collect_counterfactual_replay_event(
        event=event,
        runner=runner,
        future_frames=future_frames,
    )

    assert runner.prepared == ["scene:frame4:layer0"]
    assert runner.snapshots == 1
    assert runner.restores == 3
    assert runner.applied == [(0, (0, 1)), (0, (2, 3))]
    assert runner.replayed == [(5, (5, 6)), (5, (5, 6))]
    assert torch.equal(runner.state["cache_tokens"], torch.arange(4))
    assert replay_event["subsets"][0]["loss"] > replay_event["subsets"][1]["loss"]
    assert replay_event["subsets"][1]["loss_components"] == {
        "camera": 0.0,
        "depth": 0.0,
        "point_map": 0.0,
    }
    assert replay_event["sequence_provenance"]["sequence_id"] == "synthetic/seq0"


def test_counterfactual_replay_loss_rejects_mismatched_window_lengths():
    from ovggt.training.counterfactual_replay import compute_three_task_loss_components

    try:
        compute_three_task_loss_components(
            predictions=[{"camera_pose": torch.zeros(2)}],
            targets=[{"camera_pose": torch.zeros(2)}, {"camera_pose": torch.zeros(2)}],
        )
    except ValueError as exc:
        assert "future-window length mismatch" in str(exc)
    else:
        raise AssertionError("Mismatched replay predictions/targets must fail explicitly")


def test_ovggt_cache_replay_runner_applies_keep_indices_to_layer_cache():
    from ovggt.training.counterfactual_replay import OVGGTCacheReplayRunner
    from ovggt.utils.frontend_cache import LayerCacheState
    from tests.test_token_scorer_counterfactual import make_metadata

    class TinyRunner(OVGGTCacheReplayRunner):
        def replay_future_window(self, start_frame_idx, future_frames):
            return []

        def targets_for_future_window(self, future_frames):
            return []

    cache_state = LayerCacheState(
        k=torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        v=torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 100.0,
        score_state=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        metadata=make_metadata(anchor_slots=[0, -1, -1, -1]),
    )
    runner = TinyRunner(cache_states=[cache_state])
    snapshot = runner.snapshot()

    runner.apply_keep_indices(layer_id=0, keep_indices=torch.tensor([0, 3]))

    assert torch.equal(cache_state.k[0, 0, :, 0], torch.tensor([0.0, 12.0]))
    assert torch.equal(cache_state.score_state[0, :, 0], torch.tensor([0.0, 6.0]))

    runner.restore(snapshot)

    assert torch.equal(cache_state.k[0, 0, :, 0], torch.tensor([0.0, 4.0, 8.0, 12.0]))
    assert torch.equal(cache_state.score_state[0, :, 0], torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_oracle_generator_collect_replay_mode_uses_runner_factory(tmp_path):
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from tools.generate_counterfactual_oracle import build_events_from_args, parse_args

    runner_module = tmp_path / "fake_runner_factory.py"
    runner_module.write_text(
        """
import torch

class Runner:
    def __init__(self, frames):
        self.frames = frames
        self.keep_indices = None
        self.cache = torch.arange(4)

    def snapshot(self):
        return self.cache.clone()

    def restore(self, snapshot):
        self.cache = snapshot.clone()
        self.keep_indices = None

    def apply_keep_indices(self, layer_id, keep_indices):
        self.keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)
        self.cache = self.cache[self.keep_indices]

    def replay_future_window(self, start_frame_idx, future_frames):
        quality = float(self.keep_indices.float().mean().item())
        return [
            {
                "camera_pose": torch.tensor([quality, 0.0]),
                "depth": torch.full((1, 2), quality),
                "pts3d_in_other_view": torch.full((1, 2, 3), quality),
            }
            for _ in future_frames
        ]

    def targets_for_future_window(self, future_frames):
        return [
            {
                "camera_pose": torch.tensor([2.5, 0.0]),
                "depth": torch.full((1, 2), 2.5),
                "pts3d_in_other_view": torch.full((1, 2, 3), 2.5),
                "valid_mask": torch.ones(1, 2, dtype=torch.bool),
            }
            for _ in future_frames
        ]

def build_runner(frames):
    return Runner(frames)
""",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    plan_path = tmp_path / "replay_plan.pt"
    torch.save(
        {
            "events": [
                {
                    "event_id": "event:auto",
                    "layer_id": 0,
                    "frame_id": 4,
                    "budget": 2,
                    "score_state": torch.zeros(4, 8),
                    "metadata_features": torch.zeros(4, TOKEN_METADATA_FEATURE_DIM),
                    "base_scores": torch.tensor([3.0, 2.0, 1.0, 0.0]),
                    "candidate_subsets": [torch.tensor([0, 1]), torch.tensor([2, 3])],
                    "future_frames": [{"frame_id": 5}, {"frame_id": 6}],
                }
            ],
            "frames": [{"frame_id": idx} for idx in range(7)],
        },
        plan_path,
    )

    args = parse_args(
        [
            "--mode",
            "collect-replay",
            "--replay-plan",
            str(plan_path),
            "--runner-factory",
            "fake_runner_factory:build_runner",
            "--output",
            str(tmp_path / "oracle.pt"),
        ]
    )
    events = build_events_from_args(args)

    assert len(events) == 1
    assert events[0]["event_id"] == "event:auto"
    assert events[0]["subsets"][0]["loss"] > events[0]["subsets"][1]["loss"]
    assert torch.equal(events[0]["base_scores"], torch.tensor([3.0, 2.0, 1.0, 0.0]))


def test_legacy_train_token_scorer_config_is_deprecated():
    config_text = open("config/train_token_scorer.yaml", "r", encoding="utf-8").read()

    assert "DEPRECATED" in config_text
    assert "train_token_scorer_oracle.py" in config_text
    assert "train_token_scorer_oracle.yaml" in config_text
    assert "scorer_only: True" not in config_text
    assert "FrontendDistillLoss()" not in config_text


def test_phase1_shard_with_capped_and_stratified_metadata_loads(tmp_path):
    """Phase 1 shards store measured subsets, including policy_baseline sources,
    and must produce valid pairwise ranking samples."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    shard_path = tmp_path / "phase1_oracle.pt"
    event = {
        "event_id": "seq001:dedup:frame1:layer0",
        "event_type": "dedup",
        "frame_id": 1,
        "layer_id": 0,
        "score_state": torch.randn(4, 128),
        "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
        "sequence_provenance": {
            "sequence_id": "seq_001",
            "dataset": "WildRGBD",
        },
        "collector_config": {
            "max_subsets_per_dedup_event": 8,
            "event_selection_policy": "stratified_round_robin",
            "max_candidate_events_per_sequence": 256,
            "max_events_per_sequence": 16,
        },
        "subsets": [
            {
                "source": "policy_baseline",
                "keep_indices": torch.tensor([0, 1, 2]),
                "loss": 0.30,
            },
            {
                "source": "keep_one",
                "keep_indices": torch.tensor([0, 2]),
                "loss": 0.10,
            },
            {
                "source": "keep_one",
                "keep_indices": torch.tensor([1, 2]),
                "loss": 0.45,
            },
        ],
    }
    torch.save({
        "format": "ovggt_counterfactual_oracle_v1",
        "num_events": 1,
        "events": [event],
    }, shard_path)

    dataset = CounterfactualOracleDataset([shard_path], min_loss_gap=0.0)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["event_type"] == "dedup"
    assert sample["sequence_provenance"]["sequence_id"] == "seq_001"
    assert sample["better_mask"].shape[0] == 4
    assert sample["worse_mask"].shape[0] == 4
    assert sample["target_margin"] > 0


# ===================================================================
# Step 3.2–3.4: load_oracle_events, split_oracle_events, from_events
# ===================================================================


def test_load_oracle_events_from_shard_files(tmp_path):
    """load_oracle_events reads events from shard files."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import load_oracle_events

    events_a = [
        {
            "event_id": f"ev_{i}",
            "event_type": "eviction",
            "layer_id": 0,
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.1},
                {"keep_indices": [2, 3], "loss": 0.3},
            ],
        }
        for i in range(3)
    ]
    events_b = [
        {
            "event_id": f"ev_{i+3}",
            "event_type": "dedup",
            "layer_id": 1,
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.2},
                {"keep_indices": [2, 3], "loss": 0.5},
            ],
        }
        for i in range(2)
    ]

    shard_a = tmp_path / "shard_a.pt"
    shard_b = tmp_path / "shard_b.pt"
    torch.save({"events": events_a}, shard_a)
    torch.save({"events": events_b}, shard_b)

    loaded = load_oracle_events([shard_a, shard_b])
    assert len(loaded) == 5
    assert loaded[0]["event_id"] == "ev_0"
    assert loaded[4]["event_id"] == "ev_4"


def test_split_oracle_events_produces_disjoint_splits():
    """split_oracle_events splits at event level, not sample level."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import split_oracle_events

    events = [
        {
            "event_id": f"ev_{i:04d}",
            "event_type": "eviction",
            "layer_id": 0,
            "sequence_provenance": {"sequence_id": f"seq_{i % 5}"},
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.1},
                {"keep_indices": [2, 3], "loss": 0.3},
            ],
        }
        for i in range(50)
    ]

    train, val = split_oracle_events(events, val_fraction=0.2, split_key="event_id_hash", seed=0)
    train_ids = {e["event_id"] for e in train}
    val_ids = {e["event_id"] for e in val}
    assert train_ids.isdisjoint(val_ids)
    assert len(train) + len(val) == 50


def test_counterfactual_oracle_dataset_from_events():
    """from_events classmethod builds dataset without reading disk."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    events = [
        {
            "event_id": "ev_0",
            "event_type": "dedup",
            "layer_id": 0,
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.10},
                {"keep_indices": [2, 3], "loss": 0.30},
            ],
        },
        {
            "event_id": "ev_1",
            "event_type": "eviction",
            "layer_id": 1,
            "score_state": torch.randn(6, 8),
            "metadata_features": torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1, 2], "loss": 0.05},
                {"keep_indices": [3, 4, 5], "loss": 0.25},
            ],
        },
    ]

    ds = CounterfactualOracleDataset.from_events(events)
    assert len(ds) == 2
    assert ds.samples[0]["event_id"] == "ev_0"
    assert ds.samples[1]["event_id"] == "ev_1"


def test_counterfactual_oracle_dataset_from_events_with_event_type_filter():
    """from_events classmethod respects event_types filter."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import CounterfactualOracleDataset

    events = [
        {
            "event_id": "ev_0",
            "event_type": "dedup",
            "layer_id": 0,
            "score_state": torch.randn(4, 8),
            "metadata_features": torch.randn(4, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.10},
                {"keep_indices": [2, 3], "loss": 0.30},
            ],
        },
        {
            "event_id": "ev_1",
            "event_type": "eviction",
            "layer_id": 1,
            "score_state": torch.randn(6, 8),
            "metadata_features": torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
            "subsets": [
                {"keep_indices": [0, 1, 2], "loss": 0.05},
                {"keep_indices": [3, 4, 5], "loss": 0.25},
            ],
        },
    ]

    ds = CounterfactualOracleDataset.from_events(events, event_types=["dedup"])
    assert len(ds) == 1
    assert ds.samples[0]["event_type"] == "dedup"


def test_fifo_count_dataset_from_events():
    """FifoCountDataset.from_events classmethod builds dataset without reading disk."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import FifoCountDataset

    events = [
        {
            "event_id": "fifo_0",
            "event_type": "fifo_topk",
            "layer_id": 0,
            "score_state": torch.randn(8, 8),
            "metadata_features": torch.randn(8, TOKEN_METADATA_FEATURE_DIM),
            "demoted_indices": [0, 1, 2, 3],
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.10, "keep_count": 0},
                {"keep_indices": [0, 1], "loss": 0.05, "keep_count": 8},
                {"keep_indices": [0, 1], "loss": 0.02, "keep_count": 16},
            ],
        },
    ]

    ds = FifoCountDataset.from_events(
        events,
        count_candidates=(0, 8, 16, 32, 64, 128),
        label_reduction="min",
    )
    assert len(ds) == 1
    assert ds.samples[0]["event_id"] == "fifo_0"
    # Best keep_count is 16 (loss=0.02), which is index 2 in candidates
    assert ds.samples[0]["target"] == 2


def test_from_events_matches_shard_constructor(tmp_path):
    """from_events produces same results as loading from shards."""
    from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
    from ovggt.training.token_oracle_dataset import (
        CounterfactualOracleDataset,
        FifoCountDataset,
        load_oracle_events,
    )

    events = [
        {
            "event_id": f"ev_{i}",
            "event_type": "fifo_topk",
            "layer_id": 0,
            "sequence_provenance": {"sequence_id": f"seq_{i % 3}"},
            "score_state": torch.randn(6, 8),
            "metadata_features": torch.randn(6, TOKEN_METADATA_FEATURE_DIM),
            "demoted_indices": [0, 1, 2],
            "subsets": [
                {"keep_indices": [0, 1], "loss": 0.10 + 0.05 * i, "keep_count": 8},
                {"keep_indices": [2, 3], "loss": 0.20 + 0.05 * i, "keep_count": 16},
            ],
        }
        for i in range(4)
    ]

    shard_path = tmp_path / "test_shard.pt"
    torch.save({"events": events}, shard_path)

    # From shard
    token_from_shard = CounterfactualOracleDataset([shard_path])
    count_from_shard = FifoCountDataset([shard_path])

    # From events
    loaded_events = load_oracle_events([shard_path])
    token_from_events = CounterfactualOracleDataset.from_events(loaded_events)
    count_from_events = FifoCountDataset.from_events(loaded_events)

    assert len(token_from_shard) == len(token_from_events)
    assert len(count_from_shard) == len(count_from_events)

    for s, e in zip(token_from_shard.samples, token_from_events.samples):
        assert s["event_id"] == e["event_id"]
        assert torch.equal(s["better_mask"], e["better_mask"])
        assert torch.equal(s["worse_mask"], e["worse_mask"])
        assert abs(s["target_margin"] - e["target_margin"]) < 1e-6

    for s, e in zip(count_from_shard.samples, count_from_events.samples):
        assert s["event_id"] == e["event_id"]
        assert s["target"] == e["target"]
