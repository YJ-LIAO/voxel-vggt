import os
import sys
from pathlib import Path

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def test_parallel_launcher_builds_one_command_per_shard(tmp_path):
    from collect_counterfactual_oracle_parallel import ParallelCollectorConfig, build_collection_jobs

    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0,1",
        shards_per_device=2,
        start_shard=3,
        max_batches=7,
        max_events=11,
        num_samples=4,
        oracle_window=2,
        num_views=96,
        subset_replay_batch_size=4,
        layers_per_frame=6,
        max_events_per_sequence=48,
        seed_base=100,
        python="python",
        collector_script="tools/collect_counterfactual_oracle.py",
    )

    jobs = build_collection_jobs(cfg)

    assert [job.device for job in jobs] == ["0", "1", "0", "1"]
    assert [job.shard_id for job in jobs] == [3, 4, 5, 6]
    assert [job.seed for job in jobs] == [100, 101, 102, 103]
    assert [Path(job.output).name for job in jobs] == [
        "oracle_shard_003.pt",
        "oracle_shard_004.pt",
        "oracle_shard_005.pt",
        "oracle_shard_006.pt",
    ]
    assert [Path(job.log).name for job in jobs] == [
        "collect_003.log",
        "collect_004.log",
        "collect_005.log",
        "collect_006.log",
    ]
    first_cmd = " ".join(jobs[0].command)
    assert "--num-views 96" in first_cmd
    assert "--subset-replay-batch-size 4" in first_cmd
    assert "--layers-per-frame 6" in first_cmd
    assert "--max-events-per-sequence 48" in first_cmd


def test_parallel_launcher_skips_complete_shards_and_moves_partial_outputs(tmp_path):
    from collect_counterfactual_oracle_parallel import ParallelCollectorConfig, build_collection_jobs

    torch.save({"partial": False, "events": []}, tmp_path / "oracle_shard_000.pt")
    torch.save({"partial": True, "events": [{"event_id": "partial"}]}, tmp_path / "oracle_shard_001.pt")

    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0,1,2",
        start_shard=0,
        max_batches=1,
        max_events=1,
        seed_base=10,
        python="python",
        collector_script="tools/collect_counterfactual_oracle.py",
    )

    jobs = build_collection_jobs(cfg)

    assert [job.shard_id for job in jobs] == [2, 3, 4]
    assert [job.seed for job in jobs] == [12, 13, 14]
    assert [Path(job.output).name for job in jobs] == [
        "oracle_shard_002.pt",
        "oracle_shard_003.pt",
        "oracle_shard_004.pt",
    ]
    assert [Path(job.log).name for job in jobs] == [
        "collect_002.log",
        "collect_003.log",
        "collect_004.log",
    ]


def test_parallel_launcher_overwrite_keeps_existing_shard_ids(tmp_path):
    from collect_counterfactual_oracle_parallel import ParallelCollectorConfig, build_collection_jobs

    torch.save({"partial": False, "events": []}, tmp_path / "oracle_shard_000.pt")
    torch.save({"partial": True, "events": [{"event_id": "partial"}]}, tmp_path / "oracle_shard_001.pt")

    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0,1",
        start_shard=0,
        max_batches=1,
        max_events=1,
        seed_base=10,
        overwrite_existing=True,
        python="python",
        collector_script="tools/collect_counterfactual_oracle.py",
    )

    jobs = build_collection_jobs(cfg)

    assert [job.shard_id for job in jobs] == [0, 1]
    assert [job.seed for job in jobs] == [10, 11]


def test_parallel_launcher_can_enable_replay_payload(tmp_path):
    from collect_counterfactual_oracle_parallel import ParallelCollectorConfig, build_collection_jobs

    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0",
        max_batches=1,
        max_events=1,
        store_replay_payload=True,
        python="python",
        collector_script="tools/collect_counterfactual_oracle.py",
    )

    command = " ".join(build_collection_jobs(cfg)[0].command)

    assert "--store-replay-payload" in command


def test_collect_cli_parses_event_type_quotas_from_yaml_mapping():
    """Collector CLI should accept event_type_quotas loaded from YAML as a mapping."""
    from omegaconf import OmegaConf
    from collect_counterfactual_oracle import parse_event_type_quotas

    quotas = OmegaConf.create({"eviction": 8, "dedup": 4, "fifo_topk": 4})

    assert parse_event_type_quotas(quotas) == {
        "eviction": 8,
        "dedup": 4,
        "fifo_topk": 4,
    }


def test_collect_cli_detects_equals_style_explicit_flags():
    """YAML defaults must not override CLI args passed as --flag=value."""
    from collect_counterfactual_oracle import cli_flag_was_explicit

    argv = ["--event-type-quotas=eviction=1,dedup=0", "--max-batches=2"]

    assert cli_flag_was_explicit(argv, "event_type_quotas")
    assert cli_flag_was_explicit(argv, "max_batches")
    assert not cli_flag_was_explicit(argv, "num_samples")


def test_parallel_launcher_forwards_all_phase1_args(tmp_path):
    """All Phase 1 CLI args must reach each shard command."""
    from collect_counterfactual_oracle_parallel import ParallelCollectorConfig, build_collection_jobs

    cfg = ParallelCollectorConfig(
        config="config/train_frontend_finetune.yaml",
        dataset_key="train_dataset",
        output_dir=str(tmp_path),
        devices="0,1",
        shards_per_device=1,
        start_shard=0,
        max_batches=1,
        max_events=1,
        python="python",
        collector_script="tools/collect_counterfactual_oracle.py",
        # Phase 1 fields
        max_subsets_per_dedup_event=5,
        max_subsets_per_eviction_event=6,
        max_subsets_per_fifo_event=7,
        max_candidate_events_per_sequence=128,
        max_events_per_frame=3,
        event_selection_policy="greedy_topk",
        stratified_layer_bucket_width=4,
        oracle_profile="fast_profile",
        frontend_per_layer_budget_override=1024,
        fifo_keep_topk_override=32,
        sequence_manifest_path="/tmp/manifest.csv",
        sequence_partition_policy="sequential",
        num_sequence_shards=4,
        sequence_shard_id=1,
    )

    jobs = build_collection_jobs(cfg)
    assert len(jobs) >= 1, "expected at least one job"

    for job in jobs:
        cmd = " ".join(job.command)
        # Spot-check key Phase 1 args
        assert "--max-subsets-per-dedup-event 5" in cmd, f"missing --max-subsets-per-dedup-event in: {cmd}"
        assert "--event-selection-policy greedy_topk" in cmd, f"missing --event-selection-policy in: {cmd}"
        assert "--max-candidate-events-per-sequence 128" in cmd, f"missing --max-candidate-events-per-sequence in: {cmd}"
        assert "--max-events-per-frame 3" in cmd, f"missing --max-events-per-frame in: {cmd}"
        assert "--max-subsets-per-eviction-event 6" in cmd, f"missing --max-subsets-per-eviction-event in: {cmd}"
        assert "--max-subsets-per-fifo-event 7" in cmd, f"missing --max-subsets-per-fifo-event in: {cmd}"
        assert "--stratified-layer-bucket-width 4" in cmd, f"missing --stratified-layer-bucket-width in: {cmd}"
        assert "--oracle-profile fast_profile" in cmd, f"missing --oracle-profile in: {cmd}"
        assert "--frontend-per-layer-budget-override 1024" in cmd, f"missing --frontend-per-layer-budget-override in: {cmd}"
        assert "--fifo-keep-topk-override 32" in cmd, f"missing --fifo-keep-topk-override in: {cmd}"
        assert "--sequence-manifest-path /tmp/manifest.csv" in cmd, f"missing --sequence-manifest-path in: {cmd}"
        assert "--sequence-partition-policy sequential" in cmd, f"missing --sequence-partition-policy in: {cmd}"
        assert "--num-sequence-shards 4" in cmd, f"missing --num-sequence-shards in: {cmd}"
        assert "--sequence-shard-id 1" in cmd, f"missing --sequence-shard-id in: {cmd}"


def test_mixed_profile_generates_weighted_unique_shards():
    """Mixed launcher should generate weighted shard jobs with unique outputs."""
    from run_oracle_mixed_profiles import generate_profile_jobs

    jobs = generate_profile_jobs(
        base_config="config/train_frontend_finetune.yaml",
        output_dir="checkpoints/token_oracle_mixed/",
        num_gpus=4,
        num_shards=10,
        events_per_shard=2048,
        start_shard_id=20,
        profile_weights={
            "real_policy_dedup": 0.4,
            "low_budget_eviction": 0.3,
            "fifo_topk_forced": 0.3,
        },
    )
    assert len(jobs) == 10

    profile_counts = {}
    for job in jobs:
        profile_counts[job["profile"]] = profile_counts.get(job["profile"], 0) + 1
    assert profile_counts == {
        "real_policy_dedup": 4,
        "low_budget_eviction": 3,
        "fifo_topk_forced": 3,
    }

    outputs = {job["output"] for job in jobs}
    assert len(outputs) == len(jobs), "Each job must have a unique output path"
    assert any("shard_020" in out for out in outputs)
    assert any("shard_029" in out for out in outputs)

    for job in jobs:
        assert "--max-events" in job["args"]
        assert "2048" in job["args"]


def test_mixed_profile_can_include_late_frame_scan():
    """late_frame_scan is optional and included when profile weights request it."""
    from run_oracle_mixed_profiles import generate_profile_jobs

    jobs = generate_profile_jobs(
        base_config="config/train_frontend_finetune.yaml",
        output_dir="checkpoints/token_oracle_mixed/",
        num_gpus=2,
        num_shards=4,
        events_per_shard=128,
        profile_weights={
            "real_policy_dedup": 0.25,
            "low_budget_eviction": 0.25,
            "fifo_topk_forced": 0.25,
            "late_frame_scan": 0.25,
        },
    )
    profiles = {job["profile"] for job in jobs}
    assert profiles == {
        "real_policy_dedup",
        "low_budget_eviction",
        "fifo_topk_forced",
        "late_frame_scan",
    }
