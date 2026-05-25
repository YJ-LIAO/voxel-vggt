#!/usr/bin/env python3
"""
Evaluate voxel-vggt Frontend mode vs OVGGT Legacy mode on 7-Scenes.

Usage (single scene):
    CUDA_VISIBLE_DEVICES=4 python eval_7scenes_compare.py --scene_id chess/seq-03 --gpu_id 0

Usage (parallel across GPUs, see run_7scenes_compare.sh):
    bash run_7scenes_compare.sh
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

# Only use voxel-vggt source for this process
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOXEL_SRC = os.path.join(ROOT, "src")
OVGGT_SRC = os.path.join(ROOT, "..", "OVGGT", "src")

sys.path.insert(0, VOXEL_SRC)

DATASET_ROOT = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
DEFAULT_CHECKPOINT = "/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth"
OUTPUT_DIR = os.path.join(ROOT, "tools", "eval_results")


def load_7scenes_frames(scene_id, num_frames, dataset_root=DATASET_ROOT):
    """Load 7-Scenes color images and GT poses for a given scene/sequence."""
    from ovggt.utils.load_fn import load_and_preprocess_images

    scene_path = os.path.join(dataset_root, scene_id)
    if not os.path.isdir(scene_path):
        raise FileNotFoundError(f"Scene directory not found: {scene_path}")

    color_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".color.png")])
    if len(color_files) == 0:
        raise FileNotFoundError(f"No color images found in {scene_path}")

    color_files = color_files[:num_frames]
    color_paths = [os.path.join(scene_path, f) for f in color_files]

    images = load_and_preprocess_images(color_paths).cuda()
    inputs = [{"img": img.unsqueeze(0)} for img in images]

    gt_poses = []
    for f in color_files:
        frame_idx = f.replace("frame-", "").replace(".color.png", "")
        pose_path = os.path.join(scene_path, f"frame-{frame_idx}.pose.txt")
        pose = np.loadtxt(pose_path).astype(np.float32)
        gt_poses.append(pose)
    gt_poses = np.array(gt_poses)

    print(f"Loaded {len(inputs)} frames from {scene_id} (image shape: {images.shape})")
    return inputs, gt_poses


def load_state_dict(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            return state_dict["model"]
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
    return state_dict


def predict_poses(output, img_h, img_w):
    from ovggt.utils.pose_enc import pose_encoding_to_extri_intri

    pose_enc_list = []
    for res in output.ress:
        pose_enc = res["camera_pose"]
        if pose_enc.dim() == 1:
            pose_enc = pose_enc.unsqueeze(0)
        pose_enc_list.append(pose_enc)

    pose_enc = torch.cat(pose_enc_list, dim=0)
    ext, _ = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(0), image_size_hw=(img_h, img_w)
    )
    ext = ext.squeeze(0).cpu().numpy()

    N = ext.shape[0]
    w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
    w2c[:, :3, :] = ext
    c2w = np.linalg.inv(w2c)
    return c2w


def align_sim3(gt_poses, pred_poses):
    n = min(len(gt_poses), len(pred_poses))
    gt_p = gt_poses[:n, :3, 3]
    pred_p = pred_poses[:n, :3, 3]

    mu_gt, mu_pred = gt_p.mean(0), pred_p.mean(0)
    gt_c, pred_c = gt_p - mu_gt, pred_p - mu_pred

    H = gt_c.T @ pred_c
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2] *= -1
        R = Vh.T @ U.T

    scale = S.sum() / np.trace(gt_c.T @ gt_c) if S.sum() > 1e-6 else 1.0

    gt_t0_aligned = scale * (R @ gt_poses[0, :3, 3])
    t_anchor = pred_poses[0, :3, 3] - gt_t0_aligned

    aligned = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        aligned[i] = np.eye(4, dtype=np.float32)
        aligned[i, :3, :3] = R @ gt_poses[i, :3, :3]
        aligned[i, :3, 3] = scale * (R @ gt_poses[i, :3, 3]) + t_anchor
    return aligned


def compute_ate_rmse(gt_poses, pred_poses):
    aligned_gt = align_sim3(gt_poses, pred_poses)
    n = len(aligned_gt)
    errors = aligned_gt[:, :3, 3] - pred_poses[:n, :3, 3]
    return float(np.sqrt((errors ** 2).sum(axis=1).mean()))


def compute_rotation_error(gt_poses, pred_poses):
    aligned_gt = align_sim3(gt_poses, pred_poses)
    n = len(aligned_gt)
    errors = []
    for i in range(n):
        R_err = aligned_gt[i, :3, :3].T @ pred_poses[i, :3, :3]
        trace = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        errors.append(np.degrees(np.arccos(trace)))
    return float(np.mean(errors))


def run_frontend_inference(checkpoint, inputs, device, budget):
    """Run voxel-vggt Frontend mode inference."""
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig
    from ovggt.utils.pose_enc import ABS_POSE_ENCODING

    print("Loading voxel-vggt Frontend model...")
    model = OVGGT(
        mode="frontend_eval",
        total_budget=budget,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True),
    )
    state_dict = load_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    del state_dict
    gc.collect()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.time()

    with torch.no_grad():
        output = model.inference(
            inputs,
            history_anchor_strategy="fixed_interval",
            anchor_interval=8,
            max_anchors=3,
        )

    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output, elapsed, peak_mem


def run_legacy_via_subprocess(checkpoint, scene_id, num_frames, gpu_id, budget):
    """Run OVGGT Legacy inference in a subprocess with OVGGT's Python path."""
    script = f'''
import sys, os, time, gc, json
sys.path.insert(0, "{OVGGT_SRC}")
import torch
import numpy as np

from ovggt.models.ovggt import OVGGT
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri

# Load data
scene_path = os.path.join("{DATASET_ROOT}", "{scene_id}")
color_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".color.png")])[:{num_frames}]
color_paths = [os.path.join(scene_path, f) for f in color_files]
images = load_and_preprocess_images(color_paths).cuda()
inputs = [{{"img": img.unsqueeze(0)}} for img in images]

# Load GT poses
gt_poses = []
for f in color_files:
    idx = f.replace("frame-", "").replace(".color.png", "")
    pose = np.loadtxt(os.path.join(scene_path, f"frame-{{idx}}.pose.txt")).astype(np.float32)
    gt_poses.append(pose)
gt_poses = np.array(gt_poses)

# Model
device = torch.device("cuda:{gpu_id}")
model = OVGGT(total_budget={budget})
state_dict = torch.load("{checkpoint}", map_location="cpu", weights_only=False)
if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict, strict=True)
model = model.to(device).eval()
del state_dict
gc.collect()

# Inference
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
t0 = time.time()
with torch.no_grad():
    output = model.inference(inputs, history_anchor_strategy="coverage", anchor_interval=250)
torch.cuda.synchronize(device)
elapsed = time.time() - t0
peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

# Extract poses
pose_enc_list = []
for res in output.ress:
    pe = res["camera_pose"]
    if pe.dim() == 1:
        pe = pe.unsqueeze(0)
    pose_enc_list.append(pe)
pose_enc = torch.cat(pose_enc_list, dim=0)
sample = inputs[0]["img"]
img_h, img_w = (sample.shape[2], sample.shape[3]) if sample.dim() == 4 else (sample.shape[1], sample.shape[2])
ext, _ = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), image_size_hw=(img_h, img_w))
ext = ext.squeeze(0).cpu().numpy()
N = ext.shape[0]
w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
w2c[:, :3, :] = ext
pred_c2w = np.linalg.inv(w2c)

# Save results
results = {{
    "pred_c2w": pred_c2w.tolist(),
    "gt_poses": gt_poses.tolist(),
    "time_s": elapsed,
    "peak_mem_gb": peak_mem,
    "num_frames": {num_frames},
}}
with open(os.path.join("{OUTPUT_DIR}", "{scene_id.replace('/', '_')}_legacy.json"), "w") as f:
    json.dump(results, f)
print(f"[Legacy] Time: {{elapsed:.1f}}s | Mem: {{peak_mem:.1f}}GB")
'''
    return script


def evaluate_scene(scene_id, gpu_id, num_frames, checkpoint, budget):
    device = torch.device(f"cuda:{gpu_id}")
    inputs, gt_poses = load_7scenes_frames(scene_id, num_frames)

    sample_img = inputs[0]["img"]
    if sample_img.dim() == 4:
        img_h, img_w = sample_img.shape[2], sample_img.shape[3]
    else:
        img_h, img_w = sample_img.shape[1], sample_img.shape[2]

    results = {"scene": scene_id, "num_frames": num_frames, "budget": budget}

    # --- Frontend inference (voxel-vggt) ---
    try:
        fe_output, fe_time, fe_mem = run_frontend_inference(
            checkpoint, inputs, device, budget
        )
        fe_c2w = predict_poses(fe_output, img_h, img_w)
        fe_ate = compute_ate_rmse(gt_poses, fe_c2w)
        fe_rot = compute_rotation_error(gt_poses, fe_c2w)
        results["frontend"] = {
            "ate_rmse": fe_ate,
            "rot_error_deg": fe_rot,
            "time_s": fe_time,
            "fps": num_frames / fe_time,
            "peak_mem_gb": fe_mem,
        }
        print(f"[Frontend] ATE RMSE: {fe_ate:.4f}m | Rot: {fe_rot:.2f}deg | "
              f"Time: {fe_time:.1f}s ({num_frames/fe_time:.1f} FPS) | Mem: {fe_mem:.1f}GB")
        del fe_output
    except Exception as e:
        results["frontend"] = {"error": str(e)}
        print(f"[Frontend] FAILED: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()

    # --- Legacy inference (OVGGT, via subprocess with separate PYTHONPATH) ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        legacy_json = os.path.join(OUTPUT_DIR, f"{scene_id.replace('/', '_')}_legacy.json")
        script = run_legacy_via_subprocess(
            checkpoint, scene_id, num_frames, gpu_id, budget
        )
        print("Running OVGGT Legacy inference via subprocess...")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Legacy subprocess failed:\n{proc.stderr[-2000:]}")
        print(proc.stdout.strip())

        with open(legacy_json) as f:
            lg_data = json.load(f)

        lg_c2w = np.array(lg_data["pred_c2w"])
        lg_gt_poses = np.array(lg_data["gt_poses"])
        lg_time = lg_data["time_s"]
        lg_mem = lg_data["peak_mem_gb"]
        lg_ate = compute_ate_rmse(lg_gt_poses, lg_c2w)
        lg_rot = compute_rotation_error(lg_gt_poses, lg_c2w)

        results["legacy"] = {
            "ate_rmse": lg_ate,
            "rot_error_deg": lg_rot,
            "time_s": lg_time,
            "fps": num_frames / lg_time,
            "peak_mem_gb": lg_mem,
        }
        print(f"[Legacy]   ATE RMSE: {lg_ate:.4f}m | Rot: {lg_rot:.2f}deg | "
              f"Time: {lg_time:.1f}s ({num_frames/lg_time:.1f} FPS) | Mem: {lg_mem:.1f}GB")

        os.remove(legacy_json)
    except Exception as e:
        results["legacy"] = {"error": str(e)}
        print(f"[Legacy] FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def summarize_results(all_results):
    print("\n" + "=" * 90)
    print("SUMMARY: voxel-vggt Frontend vs OVGGT Legacy on 7-Scenes (200 frames)")
    print("=" * 90)

    header = f"{'Scene':<20} {'Mode':<10} {'ATE RMSE(m)':<14} {'Rot(deg)':<12} {'Time(s)':<10} {'FPS':<8} {'Mem(GB)':<8}"
    print(header)
    print("-" * 90)

    for r in all_results:
        scene = r["scene"]
        for mode in ["frontend", "legacy"]:
            if mode in r and "error" not in r[mode]:
                d = r[mode]
                label = "Frontend" if mode == "frontend" else "Legacy"
                print(f"{scene:<20} {label:<10} {d['ate_rmse']:<14.4f} "
                      f"{d['rot_error_deg']:<12.2f} {d['time_s']:<10.1f} "
                      f"{d['fps']:<8.1f} {d['peak_mem_gb']:<8.1f}")
            elif mode in r:
                label = "Frontend" if mode == "frontend" else "Legacy"
                print(f"{scene:<20} {label:<10} {'FAILED':<14}")
        print("-" * 90)

    fe_data = [r["frontend"] for r in all_results if "frontend" in r and "error" not in r["frontend"]]
    lg_data = [r["legacy"] for r in all_results if "legacy" in r and "error" not in r["legacy"]]

    if fe_data and lg_data:
        fe_ate = np.mean([d["ate_rmse"] for d in fe_data])
        lg_ate = np.mean([d["ate_rmse"] for d in lg_data])
        fe_rot = np.mean([d["rot_error_deg"] for d in fe_data])
        lg_rot = np.mean([d["rot_error_deg"] for d in lg_data])
        fe_time = np.mean([d["time_s"] for d in fe_data])
        lg_time = np.mean([d["time_s"] for d in lg_data])
        fe_fps = np.mean([d["fps"] for d in fe_data])
        lg_fps = np.mean([d["fps"] for d in lg_data])
        fe_mem = np.mean([d["peak_mem_gb"] for d in fe_data])
        lg_mem = np.mean([d["peak_mem_gb"] for d in lg_data])

        print(f"{'AVERAGE':<20} {'Frontend':<10} {fe_ate:<14.4f} {fe_rot:<12.2f} "
              f"{fe_time:<10.1f} {fe_fps:<8.1f} {fe_mem:<8.1f}")
        print(f"{'AVERAGE':<20} {'Legacy':<10} {lg_ate:<14.4f} {lg_rot:<12.2f} "
              f"{lg_time:<10.1f} {lg_fps:<8.1f} {lg_mem:<8.1f}")
        print(f"\nATE ratio (Frontend/Legacy): {fe_ate/lg_ate:.3f}")
        print(f"Memory change: {(fe_mem/lg_mem - 1)*100:+.1f}%")
        print(f"Speed change: {(fe_fps/lg_fps - 1)*100:+.1f}%")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="7-Scenes Frontend vs Legacy comparison")
    parser.add_argument("--scene_id", type=str, default="chess/seq-03")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--budget", type=int, default=200000)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.summarize:
        all_results = []
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if fname.endswith(".json") and not fname.endswith("_legacy.json"):
                with open(os.path.join(OUTPUT_DIR, fname)) as f:
                    all_results.append(json.load(f))
        summarize_results(all_results)
        return

    results = evaluate_scene(
        scene_id=args.scene_id,
        gpu_id=args.gpu_id,
        num_frames=args.num_frames,
        checkpoint=args.checkpoint,
        budget=args.budget,
    )

    safe_name = args.scene_id.replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
