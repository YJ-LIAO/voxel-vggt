import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import eval.mv_recon.launch as mv_launch
from eval.mv_recon.launch import (
    build_7scenes_kwargs,
    build_ovggt_inference_kwargs_for_eval,
    build_ovggt_kwargs_for_eval,
    resolve_7scenes_root,
    should_run_reconstruction_eval,
    validate_model_mode,
)


def test_build_ovggt_kwargs_legacy_mode():
    args = SimpleNamespace(
        ovggt_mode="legacy",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=8,
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    assert kwargs["mode"] == "legacy"
    assert "frontend_cache_config" not in kwargs
    assert "keyframe_switch_config" not in kwargs


def test_build_ovggt_kwargs_frontend_mode():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=False,
        frontend_anchor_interval=12,
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    assert kwargs["mode"] == "frontend_eval"
    assert kwargs["frontend_cache_config"].enabled is True
    assert kwargs["frontend_cache_config"].dedup_enabled is False
    assert kwargs["frontend_cache_config"].fifo_keep_topk == 80
    assert kwargs["keyframe_switch_config"].strategy == "fixed_interval"
    assert kwargs["keyframe_switch_config"].interval == 12


def test_build_ovggt_kwargs_frontend_passes_pose_encoding_type():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=8,
        frontend_pose_encoding_type="relT_quaR_FoV",
    )

    kwargs = build_ovggt_kwargs_for_eval(args)

    assert kwargs["frontend_pose_encoding_type"] == "relT_quaR_FoV"


def test_build_ovggt_kwargs_frontend_allows_coverage_strategy():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=12,
        frontend_keyframe_strategy="coverage",
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    assert kwargs["keyframe_switch_config"].strategy == "coverage"
    assert kwargs["keyframe_switch_config"].interval == 12
    assert kwargs["keyframe_switch_config"].coverage_monitor_only is False


def test_build_ovggt_kwargs_frontend_allows_cache_policy_overrides():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=False,
        frontend_anchor_interval=8,
        frontend_keyframe_strategy="fixed_interval",
        frontend_max_anchors=5,
        frontend_coverage_threshold=0.35,
        frontend_fifo_keep_topk=0,
        frontend_budget_allocation="dynamic",
        frontend_fifo_protected_ring_ratio=0.0,
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    config = kwargs["frontend_cache_config"]
    assert config.dedup_enabled is False
    assert config.fifo_keep_topk == 0
    assert config.budget_allocation == "dynamic"
    assert config.fifo_protected_ring_ratio == 0.0
    assert kwargs["keyframe_switch_config"].max_history_anchors == 5
    assert kwargs["keyframe_switch_config"].coverage_threshold == 0.35


def test_build_ovggt_kwargs_frontend_allows_soft_reservoir_overrides():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=8,
        frontend_keyframe_strategy="fixed_interval",
        frontend_dedup_policy="soft_reservoir",
        frontend_voxel_size=0.05,
        frontend_dedup_budget_trigger_ratio=0.85,
        frontend_dedup_topk_per_voxel=4,
        frontend_dedup_replacement_margin=0.12,
        frontend_dedup_age_decay=0.03,
    )

    kwargs = build_ovggt_kwargs_for_eval(args)

    config = kwargs["frontend_cache_config"]
    assert config.dedup_policy == "soft_reservoir"
    assert config.voxel_size == 0.05
    assert config.dedup_budget_trigger_ratio == 0.85
    assert config.dedup_topk_per_voxel == 4
    assert config.dedup_replacement_margin == 0.12
    assert config.dedup_age_decay == 0.03


def test_parser_exposes_soft_reservoir_cli_args():
    parser = mv_launch.get_args_parser()

    args = parser.parse_args([
        "--frontend_dedup_policy", "soft_reservoir",
        "--frontend_voxel_size", "0.05",
        "--frontend_dedup_budget_trigger_ratio", "0.85",
        "--frontend_dedup_topk_per_voxel", "4",
        "--frontend_dedup_replacement_margin", "0.12",
        "--frontend_dedup_age_decay", "0.03",
        "--frontend_window_protect_frames", "2",
        "--frontend_anchor_keep_ratio", "0.08",
        "--frontend_max_anchors", "5",
        "--frontend_coverage_threshold", "0.35",
    ])

    assert args.frontend_dedup_policy == "soft_reservoir"
    assert args.frontend_voxel_size == 0.05
    assert args.frontend_dedup_budget_trigger_ratio == 0.85
    assert args.frontend_dedup_topk_per_voxel == 4
    assert args.frontend_dedup_replacement_margin == 0.12
    assert args.frontend_dedup_age_decay == 0.03
    assert args.frontend_window_protect_frames == 2
    assert args.frontend_anchor_keep_ratio == 0.08
    assert args.frontend_max_anchors == 5
    assert args.frontend_coverage_threshold == 0.35


def test_parser_exposes_frontend_pose_encoding_type():
    parser = mv_launch.get_args_parser()

    args = parser.parse_args(["--frontend_pose_encoding_type", "relT_quaR_FoV"])

    assert args.frontend_pose_encoding_type == "relT_quaR_FoV"


def test_build_ovggt_inference_kwargs_passes_frontend_window_and_anchor_ratio():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_window_protect_frames=2,
        frontend_anchor_keep_ratio=0.08,
        frontend_max_anchors=5,
        frontend_coverage_threshold=0.35,
    )

    kwargs = build_ovggt_inference_kwargs_for_eval(args)

    assert kwargs["window_protect_frames"] == 2
    assert kwargs["anchor_keep_ratio"] == 0.08
    assert kwargs["max_anchors"] == 5
    assert kwargs["coverage_threshold"] == 0.35


def test_resolve_7scenes_root_prefers_explicit_path():
    assert resolve_7scenes_root("/tmp/seven") == "/tmp/seven"


def test_resolve_7scenes_root_falls_back_to_repo_relative_default():
    assert resolve_7scenes_root("") == "./data/7scenes"


def test_validate_model_mode_rejects_frontend_mode_for_vggt():
    with pytest.raises(ValueError, match="frontend mode"):
        validate_model_mode("VGGT", "frontend_eval")


def test_reconstruction_eval_model_gate_is_not_constant_true():
    assert should_run_reconstruction_eval("OVGGT") is True
    assert should_run_reconstruction_eval("VGGT") is True
    assert should_run_reconstruction_eval("unsupported") is False


def test_build_7scenes_kwargs_passes_max_frames():
    kwargs = build_7scenes_kwargs(
        data_root="/tmp/seven",
        resolution=(518, 392),
        max_frames=17,
    )
    assert kwargs["ROOT"] == "/tmp/seven"
    assert kwargs["resolution"] == (518, 392)
    assert kwargs["max_frames"] == 17


def test_filter_finite_point_pairs_uses_rowwise_pred_and_gt_masks():
    pred = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    gt = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [np.inf, 5.0, 6.0],
            [7.0, 7.0, 7.0],
        ],
        dtype=np.float32,
    )
    colors = np.arange(12, dtype=np.float32).reshape(4, 3)

    pred_f, gt_f, colors_f = mv_launch.filter_finite_point_pairs(pred, gt, colors)

    assert np.array_equal(pred_f, pred[[0, 3]])
    assert np.array_equal(gt_f, gt[[0, 3]])
    assert np.array_equal(colors_f, colors[[0, 3]])


def test_accelerator_kwargs_use_long_timeout_for_imbalanced_scene_shards():
    handlers = mv_launch.build_accelerator_kwargs_handlers()

    assert handlers
    assert handlers[0].timeout >= timedelta(hours=2)


def test_default_run_script_uses_script_relative_paths():
    script = Path(ROOT, "eval", "mv_recon", "run.sh").read_text()

    assert "SCRIPT_DIR=" in script
    assert "REPO_ROOT=" in script
    assert 'workdir=\'..\'' not in script
    assert '"${SCRIPT_DIR}/launch.py"' in script
    assert '--data_root "${DATA_ROOT}"' in script
    assert '"$@"' in script


def test_frontend_run_script_forwards_extra_cli_args():
    script = Path(ROOT, "eval", "mv_recon", "run_frontend.sh").read_text()

    assert '"${SCRIPT_DIR}/launch.py"' in script
    assert '"$@"' in script


def test_write_merged_eval_log_rejects_missing_scene_metric(tmp_path):
    save_path = tmp_path / "7scenes"
    save_path.mkdir()
    (save_path / "logs_0.txt").write_text(
        "Idx: chess/seq-03, Acc: 0.1, Comp: 0.2, NC1: 0.3, NC2: 0.4 - "
        "Acc_med: 0.5, Compc_med: 0.6, NC1c_med: 0.7, NC2c_med: 0.8\n"
    )

    with pytest.raises(RuntimeError, match="Missing metrics.*heads/seq-01"):
        mv_launch.write_merged_eval_log(
            str(save_path),
            expected_scene_ids=["chess/seq-03", "heads/seq-01"],
            num_processes=2,
        )
