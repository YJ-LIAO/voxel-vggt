import os
import sys
from pathlib import Path

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def test_summarize_legacy_oracle_log_estimates_replay_seconds(tmp_path):
    from benchmark_oracle_replay_batch_size import summarize_log

    log_path = tmp_path / "collect.log"
    log_path.write_text(
        "\n".join(
            [
                "[oracle 2026-06-01 16:00:00] collector start: subset_replay_batch_size=8 layers_per_frame=4",
                "[oracle 2026-06-01 16:00:01] batch0: measuring event 1/2 id=e0 layer=0 frame=0 subsets=9 future_frames=4",
                "[oracle 2026-06-01 16:00:01] e0: replay subsets 1-8/9 batch=8 frames=0:5",
                "[oracle 2026-06-01 16:00:11] e0: replay subsets 9-9/9 batch=1 frames=0:5",
                "[oracle 2026-06-01 16:00:13] batch0: measuring event 2/2 id=e1 layer=0 frame=1 subsets=4 future_frames=4",
                "[oracle 2026-06-01 16:00:13] e1: replay subsets 1-4/4 batch=4 frames=0:6",
                "[oracle 2026-06-01 16:00:18] batch 1/1 done: batch_events=2 total_events=2",
            ]
        )
    )

    summary = summarize_log(log_path)

    assert summary.subset_replay_batch_size == 8
    assert summary.measured_events == 2
    assert summary.replay_batches == 3
    assert summary.total_subsets == 13
    assert summary.max_subsets_per_event == 9
    assert summary.total_replay_sec == 17.0
    assert summary.avg_replay_batch_sec == 17.0 / 3.0


def test_build_benchmark_commands_vary_replay_batch_size(tmp_path):
    from benchmark_oracle_replay_batch_size import BenchmarkConfig, build_benchmark_commands

    cfg = BenchmarkConfig(
        project_root="/repo",
        python="python",
        config="config/train_frontend_finetune.yaml",
        output_dir=str(tmp_path),
        device="5",
        replay_batch_sizes=[8, 16, 24, 32],
        max_batches=1,
        max_events=4,
        num_views=24,
    )

    commands = build_benchmark_commands(cfg)

    assert len(commands) == 4
    joined = [" ".join(command) for command in commands]
    assert "--subset-replay-batch-size 8" in joined[0]
    assert "--subset-replay-batch-size 16" in joined[1]
    assert "--subset-replay-batch-size 24" in joined[2]
    assert "--subset-replay-batch-size 32" in joined[3]
    assert all("--store-replay-payload" not in item for item in joined)
    assert all(str(tmp_path) in item for item in joined)
