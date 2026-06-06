#!/usr/bin/env python3
"""
对比 Frontend 和 Legacy 模式的推理效果

Usage:
    conda activate streamvggt
    python compare_frontend_legacy.py --num_frames 10
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ovggt.models.ovggt import OVGGT
from dust3r.utils.image import load_images_for_eval


def load_model(checkpoint_path: str, mode: str, device: torch.device, per_layer_budget: int = 8000):
    """加载模型"""
    print(f"Loading model in {mode} mode from {checkpoint_path}")

    if mode == "frontend_eval":
        model = OVGGT(
            mode="frontend_eval",
            per_layer_budget=per_layer_budget,
            frontend_pose_encoding_type="relT_quaR_FoV",
        )
    else:  # legacy
        model = OVGGT(
            mode="legacy",
            per_layer_budget=per_layer_budget,
        )

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


def load_test_images(dataset_path: str, scene_id: str, num_frames: int, img_size: int = 518):
    """加载测试图片"""
    scene_path = Path(dataset_path) / scene_id
    image_dir = scene_path / "images"

    if not image_dir.exists():
        # Try alternative structure
        image_dir = scene_path

    # Get image files
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Select evenly spaced frames
    if len(image_files) > num_frames:
        indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
        image_files = [image_files[i] for i in indices]

    print(f"Loading {len(image_files)} images from {scene_id}")
    for f in image_files:
        print(f"  - {f.name}")

    # Load images
    images = load_images_for_eval([str(f) for f in image_files], size=img_size)

    return images, image_files


def run_inference(model, images, device: torch.device):
    """运行推理"""
    # Move images to device
    for img_dict in images:
        img_dict["img"] = img_dict["img"].to(device)

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        output = model.inference(images)
        elapsed = time.time() - start_time

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

    return output, elapsed, peak_memory


def compute_metrics(output, device: torch.device):
    """计算输出指标"""
    metrics = {}

    # Collect depth statistics
    depths = []
    for res in output["ress"]:
        depth = res["depth"].to(device)
        depths.append(depth)
    depths = torch.stack(depths)

    metrics["depth_mean"] = depths.mean().item()
    metrics["depth_std"] = depths.std().item()
    metrics["depth_min"] = depths.min().item()
    metrics["depth_max"] = depths.max().item()

    # Collect pose statistics
    poses = []
    for res in output["ress"]:
        pose = res["camera_pose"].to(device)
        poses.append(pose)
    poses = torch.stack(poses)

    metrics["pose_mean_norm"] = poses.norm(dim=-1).mean().item()

    # Collect point cloud statistics
    pts3d = []
    for res in output["ress"]:
        pts = res["pts3d_in_other_view"].to(device)
        pts3d.append(pts)
    pts3d = torch.stack(pts3d)

    metrics["pts3d_mean"] = pts3d.mean().item()
    metrics["pts3d_std"] = pts3d.std().item()

    return metrics


def compare_outputs(frontend_output, legacy_output, device: torch.device):
    """对比两个输出的差异"""
    differences = {}

    # Compare depth
    f_depths = torch.stack([res["depth"] for res in frontend_output["ress"]]).to(device)
    l_depths = torch.stack([res["depth"] for res in legacy_output["ress"]]).to(device)

    depth_diff = (f_depths - l_depths).abs()
    differences["depth_mae"] = depth_diff.mean().item()
    differences["depth_rmse"] = torch.sqrt((depth_diff ** 2).mean()).item()

    # Compare poses
    f_poses = torch.stack([res["camera_pose"] for res in frontend_output["ress"]]).to(device)
    l_poses = torch.stack([res["camera_pose"] for res in legacy_output["ress"]]).to(device)

    pose_diff = (f_poses - l_poses).abs()
    differences["pose_mae"] = pose_diff.mean().item()
    differences["pose_rmse"] = torch.sqrt((pose_diff ** 2).mean()).item()

    # Compare point clouds
    f_pts = torch.stack([res["pts3d_in_other_view"] for res in frontend_output["ress"]]).to(device)
    l_pts = torch.stack([res["pts3d_in_other_view"] for res in legacy_output["ress"]]).to(device)

    pts_diff = (f_pts - l_pts).abs()
    differences["pts3d_mae"] = pts_diff.mean().item()
    differences["pts3d_rmse"] = torch.sqrt((pts_diff ** 2).mean()).item()

    return differences


def main():
    parser = argparse.ArgumentParser(description="Compare Frontend vs Legacy inference")
    parser.add_argument("--frontend_ckpt", type=str,
                        default="/mnt/lyj/workspace/OVGGT-main/checkpoints/OVGGT_frontend_custom_b1_nv10/checkpoint-last.pth",
                        help="Frontend checkpoint path")
    parser.add_argument("--legacy_ckpt", type=str,
                        default="/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth",
                        help="Legacy checkpoint path")
    parser.add_argument("--dataset_path", type=str,
                        default="/mnt/lyj/workspace/StreamVGGT/dataset/blendedmvs_processed/",
                        help="Dataset path")
    parser.add_argument("--scene", type=str, default="000000000000000000000002",
                        help="Scene ID")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames to test")
    parser.add_argument("--img_size", type=int, default=518,
                        help="Image size")
    parser.add_argument("--per_layer_budget", type=int, default=434,
                        help="Per-layer KV cache budget for Frontend mode")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--mode", type=str, default="both", choices=["frontend", "legacy", "both"],
                        help="Which mode to run")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    print("=" * 60)
    print("Frontend vs Legacy Inference Comparison")
    print("=" * 60)
    print(f"Frontend checkpoint: {args.frontend_ckpt}")
    print(f"Legacy checkpoint: {args.legacy_ckpt}")
    print(f"Scene: {args.scene}")
    print(f"Num frames: {args.num_frames}")
    print(f"Per-layer budget (Frontend): {args.per_layer_budget}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    # Load images
    images, image_files = load_test_images(
        args.dataset_path, args.scene, args.num_frames, args.img_size
    )

    frontend_output = None
    frontend_time = 0
    frontend_memory = 0
    frontend_metrics = {}

    legacy_output = None
    legacy_time = 0
    legacy_memory = 0
    legacy_metrics = {}

    # Run Frontend inference
    if args.mode in ["frontend", "both"]:
        print("\n[1] Loading Frontend model...")
        frontend_model = load_model(args.frontend_ckpt, "frontend_eval", device, args.per_layer_budget)

        print("\n[2] Running Frontend inference...")
        frontend_output, frontend_time, frontend_memory = run_inference(frontend_model, images, device)
        frontend_metrics = compute_metrics(frontend_output, device)

        print(f"    Time: {frontend_time:.2f}s")
        print(f"    Peak memory: {frontend_memory:.2f} GB")

        # Clear cache
        del frontend_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Run Legacy inference
    if args.mode in ["legacy", "both"]:
        print("\n[3] Loading Legacy model...")
        legacy_model = load_model(args.legacy_ckpt, "legacy", device, 8000)  # Large budget for legacy

        print("\n[4] Running Legacy inference...")
        legacy_output, legacy_time, legacy_memory = run_inference(legacy_model, images, device)
        legacy_metrics = compute_metrics(legacy_output, device)

        print(f"    Time: {legacy_time:.2f}s")
        print(f"    Peak memory: {legacy_memory:.2f} GB")

        # Clean up
        del legacy_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Compare outputs
    if args.mode == "both" and frontend_output and legacy_output:
        print("\n[5] Comparing outputs...")
        differences = compare_outputs(frontend_output, legacy_output, device)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n[Performance]")
    print(f"{'Metric':<25} {'Frontend':<15} {'Legacy':<15}")
    print("-" * 55)
    print(f"{'Inference Time (s)':<25} {frontend_time:<15.2f} {legacy_time:<15.2f}")
    print(f"{'Peak Memory (GB)':<25} {frontend_memory:<15.2f} {legacy_memory:<15.2f}")
    if legacy_memory > 0:
        print(f"{'Memory Reduction':<25} {(1 - frontend_memory/legacy_memory)*100:.1f}%")

    if frontend_metrics:
        print("\n[Depth Statistics]")
        print(f"{'Metric':<25} {'Frontend':<15} {'Legacy':<15}")
        print("-" * 55)
        print(f"{'Mean':<25} {frontend_metrics.get('depth_mean', 0):<15.4f} {legacy_metrics.get('depth_mean', 0):<15.4f}")
        print(f"{'Std':<25} {frontend_metrics.get('depth_std', 0):<15.4f} {legacy_metrics.get('depth_std', 0):<15.4f}")

        print("\n[Pose Statistics]")
        print(f"{'Metric':<25} {'Frontend':<15} {'Legacy':<15}")
        print("-" * 55)
        print(f"{'Mean Norm':<25} {frontend_metrics.get('pose_mean_norm', 0):<15.4f} {legacy_metrics.get('pose_mean_norm', 0):<15.4f}")

    if args.mode == "both" and frontend_output and legacy_output:
        print("\n[Differences (Frontend vs Legacy)]")
        print(f"{'Metric':<25} {'MAE':<15} {'RMSE':<15}")
        print("-" * 55)
        print(f"{'Depth':<25} {differences['depth_mae']:<15.6f} {differences['depth_rmse']:<15.6f}")
        print(f"{'Pose':<25} {differences['pose_mae']:<15.6f} {differences['pose_rmse']:<15.6f}")
        print(f"{'Points3D':<25} {differences['pts3d_mae']:<15.6f} {differences['pts3d_rmse']:<15.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
