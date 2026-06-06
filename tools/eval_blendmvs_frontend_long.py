#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from eval_blendmvs_frontend import (
    DEFAULT_SCENE,
    build_frames,
    decode_predicted_poses,
    depth_metrics_ssi,
    prepare_sequence_data,
    rotation_error_deg,
    umeyama_similarity,
)
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long contiguous legacy/frontend OVGGT sequences.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/lyj/workspace/StreamVGGT/data/train/processed_blendedmvs",
        help="BlendMVS root. Supports both raw BlendMVS1 layout and processed_blendedmvs layout.",
    )
    parser.add_argument(
        "--weights",
        default="/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Scene id under dataset root.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index for each contiguous sequence.",
    )
    parser.add_argument(
        "--length",
        action="append",
        type=int,
        default=[],
        help="Contiguous sequence length. Can be repeated.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Frontend voxel size.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def load_model(weights_path: str, device: torch.device, frontend_enabled: bool, voxel_size: float) -> OVGGT:
    model_kwargs = {"mode": "frontend_eval" if frontend_enabled else "legacy"}
    if frontend_enabled:
        model_kwargs["frontend_cache_config"] = FrontendCacheConfig(
            enabled=True,
            dedup_enabled=True,
            export_keyframe_packets=False,
            voxel_size=voxel_size,
        )
    model = OVGGT(**model_kwargs).to(device)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_inference_light(
    model: OVGGT,
    frames: List[dict],
    device: torch.device,
    frontend_enabled: bool,
) -> Dict[str, object]:
    depth_list: List = []
    pose_list: List[torch.Tensor] = []
    inference_kwargs = (
        {"history_anchor_strategy": "fixed_interval", "anchor_interval": 48}
        if frontend_enabled
        else {"history_anchor_strategy": "none"}
    )

    def frame_writer(_frame_idx: int, _frame: dict, res_cpu: dict) -> None:
        depth_list.append(res_cpu["depth"].squeeze(0).squeeze(-1).numpy())
        pose_list.append(res_cpu["camera_pose"])

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    output = None
    with torch.no_grad():
        if device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model.inference(
                    frames,
                    frame_writer=frame_writer,
                    cache_results=False,
                    **inference_kwargs,
                )
        else:
            output = model.inference(
                frames,
                frame_writer=frame_writer,
                cache_results=False,
                **inference_kwargs,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_sec = time.perf_counter() - start

    depth = np.stack(depth_list, axis=0)
    c2w = decode_predicted_poses(pose_list, image_hw=depth.shape[-2:])

    metrics = {
        "runtime_sec": runtime_sec,
        "runtime_ms_per_frame": runtime_sec * 1000.0 / max(len(depth_list), 1),
    }
    if device.type == "cuda":
        metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if frontend_enabled:
        keyframe_frames = [
            int(event.frame_idx)
            for event in (output.keyframe_schedule or [])
            if getattr(event, "anchor_slot", -1) >= 0
        ]
        metrics["num_keyframes"] = float(len(keyframe_frames))
        if len(keyframe_frames) >= 2:
            intervals = np.diff(np.asarray(keyframe_frames, dtype=np.int64))
            metrics["keyframe_interval_min"] = float(intervals.min())
            metrics["keyframe_interval_mean"] = float(intervals.mean())
            metrics["keyframe_interval_max"] = float(intervals.max())
    return {
        "depth": depth,
        "c2w": c2w,
        "metrics": metrics,
    }


def evaluate_depth_pose(prediction: Dict[str, object], sequence_data: Dict[str, object]) -> Dict[str, float]:
    pred_depth = prediction["depth"]
    pred_c2w = prediction["c2w"]

    gt_depth = sequence_data["gt_depths"]
    gt_c2w = sequence_data["gt_c2w"]

    depth_metric_list = []
    for idx in range(pred_depth.shape[0]):
        mask = torch.isfinite(torch.from_numpy(gt_depth[idx])) & (torch.from_numpy(gt_depth[idx]) > 0) & torch.isfinite(
            torch.from_numpy(pred_depth[idx])
        )
        depth_metric_list.append(depth_metrics_ssi(pred_depth[idx], gt_depth[idx], mask.numpy()))

    sequence_metrics = {
        key: float(sum(item[key] for item in depth_metric_list) / len(depth_metric_list))
        for key in depth_metric_list[0]
    }

    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    pose_scale, pose_R, pose_t = umeyama_similarity(pred_centers, gt_centers)
    aligned_centers = pose_scale * (pred_centers @ pose_R.T) + pose_t
    trans_err = ((aligned_centers - gt_centers) ** 2).sum(axis=1) ** 0.5
    aligned_rot = pose_R[None] @ pred_c2w[:, :3, :3]
    rot_err = [
        rotation_error_deg(aligned_rot[idx], gt_c2w[idx, :3, :3])
        for idx in range(pred_c2w.shape[0])
    ]
    sequence_metrics["pose_ate_rmse"] = float((trans_err ** 2).mean() ** 0.5)
    sequence_metrics["pose_rot_deg_mean"] = float(sum(rot_err) / len(rot_err))
    sequence_metrics["pose_rot_deg_max"] = float(max(rot_err))
    return sequence_metrics


def main() -> None:
    args = parse_args()
    if not args.length:
        args.length = [200, 500]

    device = torch.device(args.device)
    scene_dir = os.path.join(args.dataset_root, args.scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(scene_dir)

    legacy_model = load_model(args.weights, device, frontend_enabled=False, voxel_size=args.voxel_size)
    frontend_model = load_model(args.weights, device, frontend_enabled=True, voxel_size=args.voxel_size)

    per_length = []
    for length in args.length:
        frame_ids = tuple(range(args.start, args.start + length))
        sequence_data = prepare_sequence_data(scene_dir, frame_ids)
        frames = build_frames(sequence_data["image_paths"], device)

        legacy_prediction = run_inference_light(legacy_model, frames, device, frontend_enabled=False)
        legacy_eval = {**legacy_prediction["metrics"], **evaluate_depth_pose(legacy_prediction, sequence_data)}
        del legacy_prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        frontend_prediction = run_inference_light(frontend_model, frames, device, frontend_enabled=True)
        frontend_eval = {**frontend_prediction["metrics"], **evaluate_depth_pose(frontend_prediction, sequence_data)}
        del frontend_prediction
        del frames
        del sequence_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        item = {
            "scene": args.scene,
            "start": args.start,
            "length": length,
            "legacy": legacy_eval,
            "frontend": frontend_eval,
            "runtime_ratio_frontend_vs_legacy": frontend_eval["runtime_sec"] / legacy_eval["runtime_sec"],
            "runtime_delta_sec": frontend_eval["runtime_sec"] - legacy_eval["runtime_sec"],
            "runtime_delta_pct": (frontend_eval["runtime_sec"] / legacy_eval["runtime_sec"] - 1.0) * 100.0,
        }
        per_length.append(item)
        print(json.dumps(item, indent=2))

    summary = {
        "dataset_root": args.dataset_root,
        "scene": args.scene,
        "device": args.device,
        "start": args.start,
        "lengths": args.length,
        "results": per_length,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
