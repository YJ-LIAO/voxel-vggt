import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.utils.frontend_cache import LayerCacheState, TokenKind, TokenMetadata


def _metadata(anchor_slots, frame_ids=None, importance=None, depth_conf=None):
    num_tokens = len(anchor_slots)
    frame_ids = frame_ids or [0] * num_tokens
    importance = importance or [0.0] * num_tokens
    depth_conf = depth_conf or [0.0] * num_tokens
    return TokenMetadata(
        token_kind=torch.tensor([[int(TokenKind.PATCH)] * num_tokens], dtype=torch.long),
        frame_id=torch.tensor([frame_ids], dtype=torch.long),
        anchor_slot=torch.tensor([anchor_slots], dtype=torch.long),
        keyframe_id=torch.tensor([frame_ids], dtype=torch.long),
        slot_id=torch.tensor([frame_ids], dtype=torch.long),
        slot_local_xyz=torch.tensor(
            [[[float(i), 0.0, 0.0] for i in range(num_tokens)]],
            dtype=torch.float32,
        ),
        importance=torch.tensor([importance], dtype=torch.float32),
        depth_conf=torch.tensor([depth_conf], dtype=torch.float32),
    )


def test_eviction_probe_records_commit_time_candidate_features():
    from ovggt.training.frontend_oracle_collector import CounterfactualEvictionProbe

    state = LayerCacheState(
        k=torch.randn(1, 2, 6, 4),
        v=torch.randn(1, 2, 6, 4),
        score_state=torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
        metadata=_metadata(
            anchor_slots=[0, 1, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 2, 2],
            importance=[0.0, 0.0, 0.5, 0.2, 0.9, 0.1],
            depth_conf=[0.0, 0.0, 0.1, 0.2, 0.3, 0.4],
        ),
        protected_count=2,
    )
    probe = CounterfactualEvictionProbe(
        num_samples=4,
        oracle_window=2,
        seed=7,
        event_prefix="unit",
    )

    probe.on_eviction_candidate(
        cache_state=state,
        layer_id=3,
        frame_id=2,
        budget=4,
        batch_index=0,
    )

    assert len(probe.events) == 1
    event = probe.events[0]
    assert event["event_id"] == "unit:b0:f2:l3:e0"
    assert event["batch_index"] == 0
    assert event["score_state"].shape == (6, 3)
    assert event["metadata_features"].shape == (6, TOKEN_METADATA_FEATURE_DIM)
    assert event["protected_indices"].tolist() == [0, 1]
    assert event["token_frame_ids"].tolist() == [0, 0, 1, 1, 2, 2]
    assert 2 <= len(event["candidate_subsets"]) <= 4
    for subset in event["candidate_subsets"]:
        keep = torch.as_tensor(subset["keep_indices"], dtype=torch.long)
        assert keep.numel() == 4
        assert {0, 1}.issubset(set(keep.tolist()))


def test_eviction_probe_records_sequence_provenance_for_input_sequence():
    from ovggt.training.frontend_oracle_collector import CounterfactualEvictionProbe

    state = LayerCacheState(
        k=torch.randn(1, 2, 6, 4),
        v=torch.randn(1, 2, 6, 4),
        score_state=torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
        metadata=_metadata(
            anchor_slots=[0, 1, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 2, 2],
            importance=[0.0, 0.0, 0.5, 0.2, 0.9, 0.1],
        ),
        protected_count=2,
    )
    provenance = {
        "dataset_key": "dataset_arkitscenes",
        "batch_index": 0,
        "sequence_id": "arkitscenes/scene0001",
        "frame_count": 3,
        "frames": [
            {"frame_index": 0, "dataset": "arkitscenes", "label": "scene0001_0000", "instance": "/data/scene0001/0000.jpg"},
            {"frame_index": 1, "dataset": "arkitscenes", "label": "scene0001_0001", "instance": "/data/scene0001/0001.jpg"},
            {"frame_index": 2, "dataset": "arkitscenes", "label": "scene0001_0002", "instance": "/data/scene0001/0002.jpg"},
        ],
    }
    probe = CounterfactualEvictionProbe(
        num_samples=2,
        oracle_window=1,
        seed=11,
        event_prefix="unit",
        sequence_provenance={0: provenance},
    )

    probe.on_eviction_candidate(
        cache_state=state,
        layer_id=1,
        frame_id=2,
        budget=4,
        batch_index=0,
    )

    event = probe.events[0]
    assert event["sequence_provenance"] == provenance
    assert event["event_id"].startswith("unit:b0:f2:l1")


def test_frontend_finetune_config_dataset_expression_is_resolved():
    from pathlib import Path

    from ovggt.training.frontend_oracle_collector import (
        load_frontend_oracle_config,
        resolve_dataset_expression,
    )

    cfg = load_frontend_oracle_config(
        Path("/path/to/mount/lyj/voxel-vggt/config/train_frontend_finetune.yaml")
    )
    expression = resolve_dataset_expression(cfg, dataset_key="train_dataset")

    assert "${" not in expression
    assert "Co3d_Multi" in expression
    assert "PointOdyssey_Multi" in expression
    assert "/path/to/mount/lyj/" in expression


def test_oracle_config_can_override_num_views_before_dataset_resolution():
    from pathlib import Path

    from ovggt.training.frontend_oracle_collector import (
        load_frontend_oracle_config,
        resolve_dataset_expression,
    )

    cfg = load_frontend_oracle_config(
        Path("/path/to/mount/lyj/voxel-vggt/config/train_frontend_finetune.yaml"),
        num_views=96,
    )
    expression = resolve_dataset_expression(cfg, dataset_key="train_dataset")

    assert cfg.num_views == 96
    assert "num_views=96" in expression
    assert "num_views=24" not in expression


def test_collector_model_config_defaults_enable_score_state_when_yaml_has_no_frontend_cache(monkeypatch):
    from omegaconf import OmegaConf

    import ovggt.training.frontend_oracle_collector as collector

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.parameters_called = False

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr(collector, "OVGGT", FakeModel)
    cfg = OmegaConf.create(
        {
            "frontend_mode": "frontend_train",
            "frontend_total_budget": 10410,
            "frontend_camera_budget": 128,
            "frontend_pose_encoding_type": "relT_quaR_FoV",
            "anchor_overflow_policy": "global_plus_recent",
            "n_corres_train": 0,
        }
    )

    model = collector.build_frozen_frontend_model_from_config(
        cfg,
        device="cpu",
        checkpoint_path="",
    )

    assert isinstance(model, FakeModel)
    assert captured["use_token_scorer"] is True
    assert captured["frontend_cache_config"].enabled is True
    assert captured["frontend_cache_config"].learned_eviction_enabled is False
    assert captured["frontend_cache_config"].budget_allocation == "uniform"
    assert captured["frontend_cache_config"].score_state_dim == 128


def test_high_budget_teacher_config_disables_dedup(monkeypatch):
    from omegaconf import OmegaConf

    import ovggt.training.frontend_oracle_collector as collector

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr(collector, "OVGGT", FakeModel)
    cfg = OmegaConf.create({"n_corres_train": 0, "frontend_total_budget": 16})

    collector.build_frozen_frontend_model_from_config(
        cfg,
        device="cpu",
        checkpoint_path="",
        high_budget=True,
    )

    assert captured["total_budget"] >= 10_000_000
    assert captured["frontend_cache_config"].dedup_enabled is False
    assert captured["frontend_cache_config"].intra_frame_dedup_enabled is False


def test_replay_keep_set_probe_requires_matching_batch_index():
    from ovggt.training.frontend_oracle_collector import ReplayKeepSetProbe

    class CacheState:
        k = torch.empty(1, 1, 1, 1)

    probe = ReplayKeepSetProbe(
        {"layer_id": 2, "frame_id": 3, "batch_index": 1},
        keep_indices=torch.tensor([0, 2]),
    )

    assert probe.on_eviction_candidate(CacheState(), 2, 3, 4, batch_index=0) is None
    keep = probe.on_eviction_candidate(CacheState(), 2, 3, 4, batch_index=1)
    assert torch.equal(keep.cpu(), torch.tensor([0, 2]))


def test_oracle_shard_flusher_writes_partial_shard_atomically(tmp_path):
    from ovggt.training.frontend_oracle_collector import OracleShardFlusher

    logs = []
    output = tmp_path / "oracle_shard.pt"
    flusher = OracleShardFlusher(
        base_shard={"format": "unit", "task_weights": {"camera": 20.0}},
        output_path=output,
        flush_every_events=2,
        flush_every_batches=0,
        log_fn=logs.append,
    )

    events = [{"event_id": "e0"}, {"event_id": "e1"}]

    assert flusher.maybe_flush(events, batch_idx=0, force=False, reason="batch") is True
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["partial"] is True
    assert saved["num_events"] == 2
    assert saved["events"] == events
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    assert any("flushed partial shard" in message for message in logs)


def test_collect_loader_logs_and_flushes_after_batches(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    logs = []
    flushes = []
    loader = [
        [{"img": torch.zeros(1, 3, 2, 2), "dataset": ["unit"], "label": ["seq0:f0"]}],
        [{"img": torch.zeros(1, 3, 2, 2), "dataset": ["unit"], "label": ["seq1:f0"]}],
    ]

    def fake_collect_sequence(**kwargs):
        prefix = kwargs["event_prefix"]
        return [{"event_id": f"{prefix}:event0", "subsets": []}]

    monkeypatch.setattr(collector, "collect_oracle_events_from_sequence", fake_collect_sequence)

    events = collector.collect_oracle_events_from_loader(
        model=object(),
        data_loader=loader,
        device=torch.device("cpu"),
        max_batches=2,
        max_events=10,
        num_samples=2,
        oracle_window=1,
        seed=5,
        teacher=None,
        dataset_key="dataset_unit",
        log_fn=logs.append,
        flush_callback=lambda current_events, reason, batch_idx: flushes.append(
            (len(current_events), reason, batch_idx)
        ),
    )

    assert [event["event_id"] for event in events] == ["batch0:event0", "batch1:event0"]
    assert any("batch 1/2 start" in message for message in logs)
    assert any("dataset_unit/unit/seq0:f0" in message for message in logs)
    assert any("batch 2/2 done" in message for message in logs)
    assert flushes == [
        (1, "batch 1/2 done", 0),
        (2, "batch 2/2 done", 1),
    ]


def test_collect_loader_flushes_after_events_within_long_batch(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    flushes = []
    loader = [
        [{"img": torch.zeros(1, 3, 2, 2), "dataset": ["unit"], "label": ["seq0:f0"]}],
    ]

    def fake_collect_sequence(**kwargs):
        on_event_collected = kwargs["on_event_collected"]
        events = []
        for idx in range(3):
            event = {"event_id": f"batch0:event{idx}", "subsets": []}
            events.append(event)
            on_event_collected(event)
        return events

    monkeypatch.setattr(collector, "collect_oracle_events_from_sequence", fake_collect_sequence)

    events = collector.collect_oracle_events_from_loader(
        model=object(),
        data_loader=loader,
        device=torch.device("cpu"),
        max_batches=1,
        max_events=10,
        num_samples=2,
        oracle_window=1,
        seed=5,
        teacher=None,
        dataset_key="dataset_unit",
        flush_callback=lambda current_events, reason, batch_idx: flushes.append(
            (len(current_events), reason, batch_idx)
        ),
    )

    assert [event["event_id"] for event in events] == [
        "batch0:event0",
        "batch0:event1",
        "batch0:event2",
    ]
    assert flushes[:3] == [
        (1, "batch 1/1 event 1 collected", None),
        (2, "batch 1/1 event 2 collected", None),
        (3, "batch 1/1 event 3 collected", None),
    ]


def test_collect_loader_skips_fetch_errors_and_continues(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    logs = []

    class FlakyLoader:
        def __iter__(self):
            return self

        def __init__(self):
            self.calls = 0

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return [{"img": torch.zeros(1, 3, 2, 2), "dataset": ["unit"], "label": ["seq0:f0"]}]
            if self.calls == 2:
                raise AssertionError("Non-zero skew cx=-0.7500760555267334")
            if self.calls == 3:
                return [{"img": torch.zeros(1, 3, 2, 2), "dataset": ["unit"], "label": ["seq2:f0"]}]
            raise StopIteration

    def fake_collect_sequence(**kwargs):
        prefix = kwargs["event_prefix"]
        return [{"event_id": f"{prefix}:event0", "subsets": []}]

    monkeypatch.setattr(collector, "collect_oracle_events_from_sequence", fake_collect_sequence)

    events = collector.collect_oracle_events_from_loader(
        model=object(),
        data_loader=FlakyLoader(),
        device=torch.device("cpu"),
        max_batches=3,
        max_events=10,
        num_samples=2,
        oracle_window=1,
        seed=5,
        teacher=None,
        dataset_key="dataset_unit",
        log_fn=logs.append,
        max_fetch_errors=1,
    )

    assert [event["event_id"] for event in events] == ["batch0:event0", "batch1:event0"]
    assert any("skipping dataloader batch after fetch error" in message for message in logs)


def test_provenance_log_summary_keeps_long_sequence_ids_short():
    from ovggt.training.frontend_oracle_collector import format_provenance_log_summary

    provenance = {
        0: {
            "dataset_key": "train_dataset",
            "dataset": "co3d",
            "sequence_id": ":".join(f"frame_{idx:03d}" for idx in range(96)),
            "frame_count": 96,
        }
    }

    summary = format_provenance_log_summary(provenance)

    assert summary.startswith("sequences=1 first=train_dataset/co3d/frame_000")
    assert "frame_095" in summary
    assert "frames=96" in summary
    assert len(summary) < 180


def test_oracle_loader_epoch_uses_collector_seed():
    from ovggt.training.frontend_oracle_collector import set_oracle_loader_epoch

    class Dataset:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class Sampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class Loader:
        def __init__(self):
            self.dataset = Dataset()
            self.batch_sampler = Sampler()

    loader = Loader()

    set_oracle_loader_epoch(loader, seed=7)

    assert loader.dataset.epochs == [7]
    assert loader.batch_sampler.epochs == [7]


def test_oracle_dataset_prunes_resized_empty_branches():
    from ovggt.training.frontend_oracle_collector import prune_empty_oracle_dataset
    from dust3r.datasets.base.easy_dataset import EasyDataset

    class EmptyDataset(EasyDataset):
        _resolutions = [(2, 2)]
        num_views = 192

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError(idx)

    class NonEmptyDataset(EasyDataset):
        _resolutions = [(2, 2)]
        num_views = 192

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return idx

    dataset = (10 @ EmptyDataset()) + (5 @ NonEmptyDataset())

    pruned, stats = prune_empty_oracle_dataset(dataset)

    assert stats["removed"] == 1
    assert len(pruned) == 5
    pruned.set_epoch(3)


def test_collect_oracle_builds_model_before_dataloader(monkeypatch):
    from omegaconf import OmegaConf

    import ovggt.training.frontend_oracle_collector as collector

    calls = []

    class FakeModel:
        def __init__(self):
            self.aggregator = SimpleNamespace(depth=0)

        def state_dict(self):
            return {}

    def fake_load_cfg(config_path, num_views=None):
        return OmegaConf.create(
            {
                "frontend_total_budget": 16,
                "frontend_camera_budget": 4,
                "frontend_pose_encoding_type": "absT_quaR_FoV",
                "anchor_overflow_policy": "recent",
                "n_corres_train": 0,
                "train_dataset": "dummy_dataset",
            }
        )

    def fake_build_model(*args, **kwargs):
        calls.append("model")
        return FakeModel()

    def fake_build_loader(*args, **kwargs):
        calls.append("loader")
        return object()

    def fake_collect(*args, **kwargs):
        return []

    monkeypatch.setattr(collector, "load_frontend_oracle_config", fake_load_cfg)
    monkeypatch.setattr(collector, "build_frozen_frontend_model_from_config", fake_build_model)
    monkeypatch.setattr(collector, "build_frozen_teacher_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, "build_frontend_oracle_dataloader", fake_build_loader)
    monkeypatch.setattr(collector, "collect_oracle_events_from_loader", fake_collect)

    shard = collector.collect_oracle_shard_from_config(
        collector.FrontendOracleCollectorConfig(
            config="config/train_frontend_finetune.yaml",
            output="/tmp/oracle.pt",
            device="cuda",
            high_budget_teacher=False,
            max_batches=1,
            max_events=1,
            num_samples=1,
            oracle_window=1,
        )
    )

    assert shard["events"] == []
    assert calls[:2] == ["model", "loader"]


def test_eviction_probe_deduplicates_sampled_keep_subsets(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    state = LayerCacheState(
        k=torch.randn(1, 2, 6, 4),
        v=torch.randn(1, 2, 6, 4),
        score_state=torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
        metadata=_metadata(
            anchor_slots=[0, 1, -1, -1, -1, -1],
            frame_ids=[0, 0, 1, 1, 2, 2],
            importance=[0.0, 0.0, 0.5, 0.2, 0.9, 0.1],
        ),
        protected_count=2,
    )

    def fake_sample(**kwargs):
        return [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 4, 5]),
        ]

    monkeypatch.setattr(collector, "sample_group_retention_subsets", fake_sample)
    probe = collector.CounterfactualEvictionProbe(num_samples=3, seed=13)
    probe.on_eviction_candidate(state, layer_id=0, frame_id=2, budget=4, batch_index=0)

    event = probe.events[0]
    keeps = [tuple(item["keep_indices"].tolist()) for item in event["candidate_subsets"]]
    assert keeps == [(0, 1, 2, 3), (0, 1, 4, 5)]


def test_rotating_layer_sampling_selects_evenly_spaced_layers():
    from ovggt.training.frontend_oracle_collector import should_record_oracle_layer

    selected_frame0 = [
        layer for layer in range(24)
        if should_record_oracle_layer(layer, frame_id=0, layers_per_frame=4, num_layers=24)
    ]
    selected_frame1 = [
        layer for layer in range(24)
        if should_record_oracle_layer(layer, frame_id=1, layers_per_frame=4, num_layers=24)
    ]

    assert selected_frame0 == [0, 6, 12, 18]
    assert selected_frame1 == [1, 7, 13, 19]
    assert should_record_oracle_layer(5, frame_id=0, layers_per_frame=0, num_layers=24)


def test_measure_counterfactual_event_batches_subset_replays(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "unit-event",
        "layer_id": 0,
        "frame_id": 0,
        "batch_index": 0,
        "budget": 2,
        "candidate_subsets": [
            {"keep_indices": torch.tensor([0, 1])},
            {"keep_indices": torch.tensor([0, 2])},
        ],
    }
    run_calls = []

    class CacheState:
        k = torch.empty(1, 1, 4, 1)

    def fake_run_frontend(model, replay_frames, probe, cache_results, **kwargs):
        run_calls.append((replay_frames, probe, cache_results))
        assert replay_frames[0]["img"].shape[0] == 2
        keep0 = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=0)
        keep1 = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=1)
        assert torch.equal(keep0.cpu(), torch.tensor([0, 1]))
        assert torch.equal(keep1.cpu(), torch.tensor([0, 2]))
        return SimpleNamespace(
            ress=[
                {},
                {
                    "depth": torch.stack(
                        [
                            torch.zeros(2, 2, 1),
                            torch.ones(2, 2, 1),
                        ],
                        dim=0,
                    ),
                    "pts3d_in_other_view": torch.stack(
                        [
                            torch.zeros(2, 2, 3),
                            torch.ones(2, 2, 3),
                        ],
                        dim=0,
                    ),
                },
            ]
        )

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    measured = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=2,
    )

    assert len(run_calls) == 1
    assert [subset["keep_indices"].tolist() for subset in measured["subsets"]] == [[0, 1], [0, 2]]
    assert measured["subsets"][0]["loss"] == 0.0
    assert measured["subsets"][1]["loss"] > 0.0


def test_measure_counterfactual_event_logs_replay_timing(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "timed-event",
        "event_type": "dedup",
        "layer_id": 3,
        "frame_id": 0,
        "batch_index": 0,
        "candidate_subsets": [
            {"keep_indices": torch.tensor([0, 1])},
            {"keep_indices": torch.tensor([0, 2])},
        ],
    }
    logs = []

    class CacheState:
        k = torch.empty(1, 1, 4, 1)

    def fake_run_frontend(model, replay_frames, probe, cache_results, dedup_replay_probe=None, **kwargs):
        active_probe = dedup_replay_probe or probe
        keep0 = active_probe.on_dedup_candidate(CacheState(), 3, 0, batch_index=0)
        keep1 = active_probe.on_dedup_candidate(CacheState(), 3, 0, batch_index=1)
        assert torch.equal(keep0.cpu(), torch.tensor([0, 1]))
        assert torch.equal(keep1.cpu(), torch.tensor([0, 2]))
        return SimpleNamespace(
            ress=[
                {},
                {
                    "depth": torch.zeros(2, 2, 2, 1),
                    "pts3d_in_other_view": torch.zeros(2, 2, 2, 3),
                },
            ]
        )

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    measured = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=2,
        log_fn=logs.append,
    )

    assert measured is not None
    assert any(
        "timing phase=replay" in message
        and "event_id=timed-event" in message
        and "event_type=dedup" in message
        and "layer_id=3" in message
        and "num_subsets=2" in message
        and "chunk_size=2" in message
        and "elapsed_sec=" in message
        for message in logs
    )
    assert any("timing phase=event" in message and "target_sec=" in message and "loss_sec=" in message for message in logs)


def test_measure_counterfactual_event_stores_replay_payload_only_when_requested(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "payload-event",
        "layer_id": 0,
        "frame_id": 0,
        "batch_index": 0,
        "candidate_subsets": [{"keep_indices": torch.tensor([0, 1])}],
    }

    class CacheState:
        k = torch.empty(1, 1, 4, 1)

    def fake_run_frontend(model, replay_frames, probe, cache_results, **kwargs):
        keep = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=0)
        assert torch.equal(keep.cpu(), torch.tensor([0, 1]))
        return SimpleNamespace(
            ress=[
                {},
                {
                    "depth": torch.zeros(1, 2, 2, 1),
                    "pts3d_in_other_view": torch.zeros(1, 2, 2, 3),
                },
            ]
        )

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    default_event = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=1,
    )
    payload_event = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=1,
        store_replay_payload=True,
    )

    assert "replay" not in default_event["subsets"][0]
    assert set(payload_event["subsets"][0]["replay"]) == {"predictions", "targets"}
    assert torch.equal(payload_event["subsets"][0]["replay"]["predictions"][0]["depth"], torch.zeros(1, 2, 2, 1))


def test_measure_counterfactual_event_falls_back_to_serial_when_batched_replay_misses(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "unit-event",
        "layer_id": 0,
        "frame_id": 0,
        "batch_index": 0,
        "budget": 2,
        "candidate_subsets": [
            {"keep_indices": torch.tensor([0, 1])},
            {"keep_indices": torch.tensor([0, 2])},
        ],
    }
    logs = []
    calls = []

    class CacheState:
        k = torch.empty(1, 1, 4, 1)

    def fake_run_frontend(model, replay_frames, probe, cache_results, **kwargs):
        calls.append(type(probe).__name__)
        if type(probe).__name__ == "MultiReplayKeepSetProbe":
            assert replay_frames[0]["img"].shape[0] == 2
            return SimpleNamespace(ress=[{}, {"depth": torch.zeros(2, 2, 2, 1)}])
        keep = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=0)
        assert keep is not None
        offset = 0.0 if torch.equal(keep.cpu(), torch.tensor([0, 1])) else 1.0
        return SimpleNamespace(
            ress=[
                {},
                {
                    "depth": torch.full((1, 2, 2, 1), offset),
                    "pts3d_in_other_view": torch.full((1, 2, 2, 3), offset),
                },
            ]
        )

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    measured = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=2,
        log_fn=logs.append,
    )

    assert calls == ["MultiReplayKeepSetProbe", "ReplayKeepSetProbe", "ReplayKeepSetProbe"]
    assert any("falling back to serial replay" in message for message in logs)
    assert [subset["keep_indices"].tolist() for subset in measured["subsets"]] == [[0, 1], [0, 2]]
    assert measured["subsets"][0]["loss"] == 0.0
    assert measured["subsets"][1]["loss"] > 0.0


def test_measure_counterfactual_event_skips_when_fallback_serial_subset_misses(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "partially-unstable-event",
        "layer_id": 0,
        "frame_id": 0,
        "batch_index": 0,
        "budget": 2,
        "candidate_subsets": [
            {"keep_indices": torch.tensor([0, 1])},
            {"keep_indices": torch.tensor([0, 2])},
        ],
    }
    logs = []
    calls = []

    class CacheState:
        k = torch.empty(1, 1, 4, 1)

    def fake_run_frontend(model, replay_frames, probe, cache_results, **kwargs):
        calls.append(type(probe).__name__)
        if type(probe).__name__ == "MultiReplayKeepSetProbe":
            keep0 = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=0)
            assert torch.equal(keep0.cpu(), torch.tensor([0, 1]))
            return SimpleNamespace(ress=[{}, {"depth": torch.zeros(2, 2, 2, 1)}])
        if len(calls) == 2:
            keep = probe.on_eviction_candidate(CacheState(), 0, 0, 2, batch_index=0)
            assert torch.equal(keep.cpu(), torch.tensor([0, 1]))
        return SimpleNamespace(
            ress=[
                {},
                {
                    "depth": torch.zeros(1, 2, 2, 1),
                    "pts3d_in_other_view": torch.zeros(1, 2, 2, 3),
                },
            ]
        )

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    measured = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=2,
        log_fn=logs.append,
    )

    assert measured is None
    assert calls == ["MultiReplayKeepSetProbe", "ReplayKeepSetProbe", "ReplayKeepSetProbe"]
    assert any("fallback serial replay did not apply all keep sets" in message for message in logs)


def test_measure_counterfactual_event_skips_event_when_replay_never_matches(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {
            "img": torch.zeros(1, 3, 2, 2),
            "depthmap": torch.zeros(1, 2, 2),
            "pts3d": torch.zeros(1, 2, 2, 3),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
    ]
    event = {
        "event_id": "unmatched-event",
        "layer_id": 0,
        "frame_id": 0,
        "batch_index": 0,
        "budget": 2,
        "candidate_subsets": [
            {"keep_indices": torch.tensor([0, 1])},
            {"keep_indices": torch.tensor([0, 2])},
        ],
    }
    logs = []

    def fake_run_frontend(model, replay_frames, probe, cache_results, **kwargs):
        return SimpleNamespace(ress=[{}, {"depth": torch.zeros(1, 2, 2, 1)}])

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    measured = collector.measure_counterfactual_event(
        model=object(),
        frames=frames,
        event=event,
        future_frames=frames[1:],
        subset_replay_batch_size=2,
        log_fn=logs.append,
    )

    assert measured is None
    assert any("skipping event" in message for message in logs)


def test_collect_sequence_logs_probe_timing(monkeypatch):
    import ovggt.training.frontend_oracle_collector as collector

    frames = [
        {"img": torch.zeros(1, 3, 2, 2)},
        {"img": torch.zeros(1, 3, 2, 2)},
    ]
    logs = []

    def fake_run_frontend(model, frames_arg, probe, cache_results, dedup_probe=None, fifo_probe=None):
        assert cache_results is False
        assert len(frames_arg) == 2
        assert probe is not None
        assert dedup_probe is not None
        assert fifo_probe is not None
        return SimpleNamespace(ress=[])

    monkeypatch.setattr(collector, "_run_frontend_with_probe", fake_run_frontend)

    events = collector.collect_oracle_events_from_sequence(
        model=object(),
        frames=frames,
        device=torch.device("cpu"),
        max_events=4,
        num_samples=2,
        oracle_window=1,
        log_fn=logs.append,
    )

    assert events == []
    assert any("timing phase=probe" in message and "event_prefix=oracle" in message for message in logs)


def _dedup_metadata(num_tokens, frame_id=5, importance=None, xyz_for_tokens=None):
    if importance is None:
        importance = [0.5] * num_tokens
    if xyz_for_tokens is None:
        xyz_for_tokens = [(0.0, 0.0, 0.0)] * num_tokens
    return TokenMetadata(
        token_kind=torch.tensor([[int(TokenKind.PATCH)] * num_tokens], dtype=torch.long),
        frame_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        anchor_slot=torch.tensor([[0] * num_tokens], dtype=torch.long),
        keyframe_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        slot_id=torch.tensor([[frame_id] * num_tokens], dtype=torch.long),
        slot_local_xyz=torch.tensor([list(xyz_for_tokens)], dtype=torch.float32),
        importance=torch.tensor([importance], dtype=torch.float32),
        depth_conf=torch.tensor([[0.5] * num_tokens], dtype=torch.float32),
    )


def _dedup_cache_state(num_tokens, importance, xyz_for_tokens=None):
    return LayerCacheState(
        k=torch.randn(1, 2, num_tokens, 4),
        v=torch.randn(1, 2, num_tokens, 4),
        score_state=torch.arange(num_tokens * 4, dtype=torch.float32).reshape(1, num_tokens, 4),
        metadata=_dedup_metadata(num_tokens, importance=importance, xyz_for_tokens=xyz_for_tokens),
        protected_count=0,
    )


class TestProbeCallbackSignatureCompatibility:
    def test_dedup_probe_accepts_scores_and_policy_keep(self):
        from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe
        probe = CounterfactualDedupProbe(num_samples=8, oracle_window=4, seed=42)
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=torch.randn(5), policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )

    def test_replay_probe_accepts_scores_and_policy_keep(self):
        from ovggt.training.frontend_oracle_collector import ReplayDedupKeepSetProbe
        target_event = {"layer_id": 0, "frame_id": 5, "batch_index": 0}
        probe = ReplayDedupKeepSetProbe(target_event=target_event, keep_indices=torch.tensor([0, 1, 2]))
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        result = probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=torch.randn(5), policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )
        assert torch.equal(result.cpu(), torch.tensor([0, 1, 2]))

    def test_multi_replay_probe_accepts_scores_and_policy_keep(self):
        from ovggt.training.frontend_oracle_collector import MultiReplayDedupKeepSetProbe
        target_event = {"layer_id": 0, "frame_id": 5, "batch_index": 0}
        probe = MultiReplayDedupKeepSetProbe(
            target_event=target_event, keep_indices_batch=[torch.tensor([0, 1, 2])],
        )
        state = _dedup_cache_state(num_tokens=5, importance=[0.1, 0.2, 0.3, 0.4, 0.5])
        result = probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=torch.randn(5), policy_keep_indices=torch.tensor([0, 1, 2, 3, 4]),
        )
        assert torch.equal(result.cpu(), torch.tensor([0, 1, 2]))


from ovggt.training.frontend_oracle_collector import _sample_dedup_keep_indices

class TestSampleDedupKeepIndices:
    def test_returns_all_indices_when_n_le_cap(self):
        group_indices = torch.tensor([3, 7, 11, 15])
        dedup_scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
        gen = torch.Generator().manual_seed(0)
        result, baseline = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen)
        assert sorted(result) == [3, 7, 11, 15]
        assert baseline is None

    def test_caps_at_requested_size_when_n_gt_cap(self):
        group_indices = torch.arange(50)
        dedup_scores = torch.rand(50)
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen)
        assert len(result) <= 8
        assert len(result) >= 4
        for idx in result:
            assert 0 <= idx < 50

    def test_includes_extreme_scores(self):
        group_indices = torch.arange(20)
        dedup_scores = torch.arange(20, dtype=torch.float)
        gen = torch.Generator().manual_seed(0)
        result, _ = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=4, generator=gen)
        assert 19 in result
        assert 0 in result

    def test_deterministic_with_same_seed(self):
        group_indices = torch.arange(100)
        dedup_scores = torch.rand(100)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        r1, _ = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen1)
        r2, _ = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen2)
        assert r1 == r2

    def test_policy_baseline_single_token_in_group(self):
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        policy_keep = torch.tensor([5, 50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen, policy_keep_indices=policy_keep)
        assert 5 in result
        assert baseline is None

    def test_policy_baseline_multiple_tokens_in_group(self):
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        policy_keep = torch.tensor([3, 7, 11, 50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen, policy_keep_indices=policy_keep)
        assert baseline is not None
        assert torch.equal(baseline, policy_keep.long())
        assert len(result) + (1 if baseline is not None else 0) <= 8

    def test_policy_baseline_zero_tokens_in_group_is_skipped(self):
        group_indices = torch.arange(20)
        dedup_scores = torch.rand(20)
        policy_keep = torch.tensor([50, 60, 70, 80])
        gen = torch.Generator().manual_seed(42)
        result, baseline = _sample_dedup_keep_indices(group_indices, dedup_scores, cap=8, generator=gen, policy_keep_indices=policy_keep)
        assert result == []
        assert baseline is None


from ovggt.training.frontend_oracle_collector import CounterfactualDedupProbe

class TestDedupProbeSubsetCap:
    def test_probe_caps_subsets_when_voxel_group_exceeds_cap(self):
        importance = [float(i) / 50.0 for i in range(50)]
        state = _dedup_cache_state(num_tokens=50, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        scores = torch.tensor(importance)
        policy_keep = torch.tensor([0, 1, 2, 3, 4])
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=scores, policy_keep_indices=policy_keep,
        )
        assert len(probe.events) == 1
        event = probe.events[0]
        assert len(event["candidate_subsets"]) <= 8
        assert any(subset.get("source") == "policy_baseline" for subset in event["candidate_subsets"])

    def test_probe_returns_full_enumeration_below_cap(self):
        importance = [0.1, 0.2, 0.3, 0.4, 0.5]
        state = _dedup_cache_state(num_tokens=5, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=torch.tensor(importance), policy_keep_indices=torch.arange(5),
        )
        assert len(probe.events) == 1
        assert len(probe.events[0]["candidate_subsets"]) == 5

    def test_probe_uses_actual_scores_not_importance(self):
        importance = [0.5] * 50
        actual_scores = torch.arange(50, dtype=torch.float)
        state = _dedup_cache_state(num_tokens=50, importance=importance)
        probe = CounterfactualDedupProbe(
            num_samples=8, oracle_window=4, seed=42,
            max_subsets_per_dedup_event=8,
        )
        probe.on_dedup_candidate(
            cache_state=state, layer_id=0, frame_id=5, batch_index=0,
            scores=actual_scores, policy_keep_indices=torch.tensor([0]),
        )
        subsets = [s for s in probe.events[0]["candidate_subsets"] if s.get("source") != "policy_baseline"]
        keep_indices_sets = [set(s["keep_indices"].tolist()) for s in subsets]
        assert any(49 in kis for kis in keep_indices_sets)


from ovggt.training.frontend_oracle_collector import select_oracle_events


class TestSelectOracleEvents:
    def _make_events(self, specs):
        return [{"event_type": et, "frame_id": fid, "layer_id": lid, "voxel_group_id": vgid} for et, fid, lid, vgid in specs]

    def test_first_n_policy_takes_first_n(self):
        events = self._make_events([("dedup", f, 0, 0) for f in range(20)])
        selected = select_oracle_events(events, max_events=5, max_events_per_frame=100, policy="first_n")
        assert len(selected) == 5
        assert [e["frame_id"] for e in selected] == [0, 1, 2, 3, 4]

    def test_stratified_round_robin_spreads_across_frames(self):
        events = []
        for frame in range(3):
            for _ in range(20):
                events.append({"event_type": "dedup", "frame_id": frame, "layer_id": 0, "voxel_group_id": 0})
        selected = select_oracle_events(events, max_events=6, max_events_per_frame=6, policy="stratified_round_robin")
        frames_selected = set(e["frame_id"] for e in selected)
        assert len(frames_selected) >= 2

    def test_max_events_per_frame_is_respected(self):
        events = [{"event_type": "dedup", "frame_id": 0, "layer_id": i, "voxel_group_id": 0} for i in range(20)]
        selected = select_oracle_events(events, max_events=10, max_events_per_frame=3, policy="stratified_round_robin")
        frame_counts = {}
        for e in selected:
            frame_counts[e["frame_id"]] = frame_counts.get(e["frame_id"], 0) + 1
        assert frame_counts.get(0, 0) <= 3

    def test_deterministic_on_same_input(self):
        events = self._make_events([("dedup", i % 5, i, i) for i in range(50)])
        s1 = select_oracle_events(events, max_events=10, max_events_per_frame=6, policy="stratified_round_robin")
        s2 = select_oracle_events(events, max_events=10, max_events_per_frame=6, policy="stratified_round_robin")
        assert s1 == s2

    def test_groups_by_event_type_frame_layer_and_voxel(self):
        events = []
        for et in ["dedup", "eviction"]:
            for frame in [0, 1]:
                for layer in [0, 12]:
                    events.append({"event_type": et, "frame_id": frame, "layer_id": layer, "voxel_group_id": 0})
        selected = select_oracle_events(events, max_events=8, max_events_per_frame=8, policy="stratified_round_robin")
        assert len(selected) == 8

    def test_layer_bucket_groups_correctly(self):
        events = [
            {"event_type": "dedup", "frame_id": 0, "layer_id": 0, "voxel_group_id": 0},
            {"event_type": "dedup", "frame_id": 0, "layer_id": 6, "voxel_group_id": 0},
            {"event_type": "dedup", "frame_id": 0, "layer_id": 12, "voxel_group_id": 0},
            {"event_type": "dedup", "frame_id": 0, "layer_id": 18, "voxel_group_id": 0},
        ]
        selected = select_oracle_events(events, max_events=4, max_events_per_frame=4, policy="stratified_round_robin")
        assert len(selected) == 4
        layers = [e["layer_id"] for e in selected]
        assert set(layers) == {0, 6, 12, 18}


import unittest.mock as _mock
from ovggt.training.frontend_oracle_collector import (
    collect_oracle_events_from_sequence,
    select_oracle_events as _real_select,
)


class TestTask6Wiring:
    """Tests that select_oracle_events is wired into the collection pipeline."""

    @staticmethod
    def _make_event(idx, frame_id=0, layer_id=0, event_type="eviction"):
        return {
            "event_id": idx,
            "frame_id": frame_id,
            "layer_id": layer_id,
            "event_type": event_type,
            "candidate_subsets": [{"keep_indices": [0], "source": "test"}],
        }

    def test_select_oracle_events_called_with_stratified_args(self):
        """collect_oracle_events_from_sequence should call select_oracle_events
        with the stratified selection parameters."""
        import ovggt.training.frontend_oracle_collector as mod

        events = [self._make_event(i, frame_id=i % 3, layer_id=i, event_type="eviction") for i in range(10)]
        for e in events:
            e["candidate_subsets"] = [{"keep_indices": [0], "source": "test"}]

        select_calls = []

        def fake_select(candidates, max_events, max_events_per_frame, policy, layer_bucket_width):
            select_calls.append({
                "max_events": max_events,
                "max_events_per_frame": max_events_per_frame,
                "policy": policy,
                "layer_bucket_width": layer_bucket_width,
                "num_candidates": len(candidates),
            })
            return candidates[:max_events]

        with _mock.patch.object(mod, "_run_frontend_with_probe") as fake_run, \
             _mock.patch.object(mod, "measure_counterfactual_event", return_value=None), \
             _mock.patch.object(mod, "select_oracle_events", side_effect=fake_select):

            def populate_probes(model, frames, probe, **kwargs):
                probe.events = [e for e in events if e["event_type"] == "eviction"]
                dedup = kwargs.get("dedup_probe")
                if dedup:
                    dedup.events = [e for e in events if e["event_type"] == "dedup"]
                fifo = kwargs.get("fifo_probe")
                if fifo:
                    fifo.events = []
            fake_run.side_effect = populate_probes

            result = collect_oracle_events_from_sequence(
                model=None,
                frames=[{"f": i} for i in range(12)],
                device=torch.device("cpu"),
                max_events=5,
                num_samples=2,
                oracle_window=4,
                max_candidate_events_per_sequence=100,
                max_events_per_frame=3,
                event_selection_policy="stratified_round_robin",
                stratified_layer_bucket_width=4,
            )

        assert len(select_calls) == 1
        call = select_calls[0]
        assert call["max_events"] == 5
        assert call["max_events_per_frame"] == 3
        assert call["policy"] == "stratified_round_robin"
        assert call["layer_bucket_width"] == 4

    def test_candidate_cap_used_for_probe_max_events(self):
        """When max_candidate_events_per_sequence is set, probes should use it as cap."""
        import ovggt.training.frontend_oracle_collector as mod

        probe_max_events = []

        class FakeProbe:
            def __init__(self, **kwargs):
                self.events = []
                probe_max_events.append(kwargs.get("max_events"))

        with _mock.patch.object(mod, "CounterfactualEvictionProbe", FakeProbe), \
             _mock.patch.object(mod, "CounterfactualDedupProbe", FakeProbe), \
             _mock.patch.object(mod, "CounterfactualFifoTopKProbe", FakeProbe), \
             _mock.patch.object(mod, "_run_frontend_with_probe"), \
             _mock.patch.object(mod, "select_oracle_events", return_value=[]):

            collect_oracle_events_from_sequence(
                model=None,
                frames=[{"f": i} for i in range(12)],
                device=torch.device("cpu"),
                max_events=5,
                num_samples=2,
                oracle_window=4,
                max_candidate_events_per_sequence=200,
            )

        # All three probes should get candidate_cap=200
        assert probe_max_events == [200, 200, 200]

    def test_candidate_cap_defaults_to_max_events(self):
        """When max_candidate_events_per_sequence is None, probes use max_events."""
        import ovggt.training.frontend_oracle_collector as mod

        probe_max_events = []

        class FakeProbe:
            def __init__(self, **kwargs):
                self.events = []
                probe_max_events.append(kwargs.get("max_events"))

        with _mock.patch.object(mod, "CounterfactualEvictionProbe", FakeProbe), \
             _mock.patch.object(mod, "CounterfactualDedupProbe", FakeProbe), \
             _mock.patch.object(mod, "CounterfactualFifoTopKProbe", FakeProbe), \
             _mock.patch.object(mod, "_run_frontend_with_probe"), \
             _mock.patch.object(mod, "select_oracle_events", return_value=[]):

            collect_oracle_events_from_sequence(
                model=None,
                frames=[{"f": i} for i in range(12)],
                device=torch.device("cpu"),
                max_events=5,
                num_samples=2,
                oracle_window=4,
            )

        assert probe_max_events == [5, 5, 5]

    def test_default_policy_is_stratified_round_robin(self):
        """When event_selection_policy is not passed, default should be stratified_round_robin."""
        import ovggt.training.frontend_oracle_collector as mod

        select_calls = []

        def fake_select(candidates, max_events, max_events_per_frame, policy, layer_bucket_width):
            select_calls.append({"policy": policy})
            return []

        with _mock.patch.object(mod, "_run_frontend_with_probe"), \
             _mock.patch.object(mod, "select_oracle_events", side_effect=fake_select):

            collect_oracle_events_from_sequence(
                model=None,
                frames=[{"f": i} for i in range(12)],
                device=torch.device("cpu"),
                max_events=5,
                num_samples=2,
                oracle_window=4,
            )

        assert len(select_calls) == 1
        assert select_calls[0]["policy"] == "stratified_round_robin"
