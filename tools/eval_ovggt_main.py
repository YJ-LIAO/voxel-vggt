#!/usr/bin/env python3
"""
Evaluate OVGGT-main Frontend mode on 7-Scenes.
Runs from voxel-vggt directory but uses OVGGT-main source via PYTHONPATH.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

# Use OVGGT-main source (not voxel-vggt source)
OVGGT_MAIN_SRC = "/path/to/mount/lyj/OVGGT-main/src"
sys.path.insert(0, OVGGT_MAIN_SRC)

DATASET_ROOT = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
DEFAULT_CHECKPOINT = "/path/to/mount/lyj/OVGGT-main/ckpt/model.pt"
OUTPUT_DIR = "/path/to/mount/lyj/voxel-vggt/tools/eval_results_ovggt_main"


def load_7scenes(scene_id, num_frames, data_root=DATASET_ROOT, target_size=518):
    seq_dir = os.path.join(data_root, scene_id)
    color_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(".color.png")])[:num_frames]
    to_tensor = TF.ToTensor()
    images, gt_poses = [], []

    for cf in color_files:
        img = Image.open(os.path.join(seq_dir, cf)).convert("RGB")
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))

        prefix = cf.replace(".color.png", "")
        pp = os.path.join(seq_dir, f"{prefix}.pose.txt")
        gt_poses.append(np.loadtxt(pp).reshape(4, 4))

    print(f"Loaded {len(images)} frames from {scene_id}")
    return torch.stack(images), gt_poses


def build_frontend_model(ckpt_path, device, budget):
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig
    from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig
    from ovggt.utils.pose_enc import ABS_POSE_ENCODING

    kw = {
        "mode": "frontend_eval",
        "per_layer_budget": budget,
        "frontend_pose_encoding_type": ABS_POSE_ENCODING,
        "frontend_cache_config": FrontendCacheConfig(
            enabled=True, export_keyframe_packets=False, dedup_enabled=True, voxel_size=0.5
        ),
        "keyframe_switch_config": KeyframeSwitchConfig(
            strategy="fixed_interval", interval=8, max_history_anchors=3
        ),
    }
    model = OVGGT(**kw)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def build_legacy_model(ckpt_path, device, budget):
    from ovggt.models.ovggt import OVGGT

    model = OVGGT(mode="legacy", per_layer_budget=budget)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def run_frontend_streaming(model, images_tensor, device, image_size_hw):
    from ovggt.utils.pose_enc import pose_encoding_to_camera_to_world

    torch.cuda.empty_cache()
    gc.collect()
    n = images_tensor.shape[0]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    pred_R, pred_t = [], []
    for i in range(n):
        frame = [{"img": images_tensor[i].unsqueeze(0)}]
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = model.inference(frame)
        res = output.ress[0]
        if "camera_pose" in res:
            pe = res["camera_pose"].detach().cpu().reshape(1, 1, -1)
            c2w = pose_encoding_to_camera_to_world(pe, image_size_hw=image_size_hw).reshape(-1, 4, 4)[0]
            pred_R.append(c2w[:3, :3].numpy())
            pred_t.append(c2w[:3, 3].numpy())

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    return {"time_s": elapsed, "fps": n / elapsed, "peak_mem_gb": peak_mem,
            "pred_R": pred_R, "pred_t": pred_t}


def run_legacy_joint(model, images_tensor, device, image_size_hw):
    from ovggt.utils.pose_enc import pose_encoding_to_camera_to_world

    torch.cuda.empty_cache()
    gc.collect()
    n = images_tensor.shape[0]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    frames = [{"img": images_tensor[i].unsqueeze(0)} for i in range(n)]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            output = model.inference(frames)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    pred_R, pred_t = [], []
    for res in output.ress:
        if "camera_pose" in res:
            pe = res["camera_pose"].detach().cpu().reshape(1, 1, -1)
            c2w = pose_encoding_to_camera_to_world(pe, image_size_hw=image_size_hw).reshape(-1, 4, 4)[0]
            pred_R.append(c2w[:3, :3].numpy())
            pred_t.append(c2w[:3, 3].numpy())

    return {"time_s": elapsed, "fps": n / elapsed, "peak_mem_gb": peak_mem,
            "pred_R": pred_R, "pred_t": pred_t}


def ate_umeyama(pred_t_list, gt_poses):
    idx = [i for i in range(len(gt_poses)) if i < len(pred_t_list)]
    if len(idx) < 3:
        return float("nan")
    gt_t = np.array([gt_poses[i][:3, 3] for i in idx])
    pr_t = np.array([pred_t_list[i] for i in idx])

    pm, gm = pr_t.mean(0), gt_t.mean(0)
    pc, gc = pr_t - pm, gt_t - gm
    H = pc.T @ gc
    U, S, Vt = np.linalg.svd(H)
    Ra = Vt.T @ U.T
    if np.linalg.det(Ra) < 0:
        Vt[-1] *= -1
        Ra = Vt.T @ U.T
    s = np.sum(S) / (np.sum(pc**2) + 1e-8)
    al = (s * (Ra @ pr_t.T)).T + (gm - s * Ra @ pm)
    return float(np.sqrt(np.mean(np.sum((al - gt_t)**2, axis=1))))


def rpe_rotation(pred_R_list, gt_poses):
    idx = [i for i in range(len(gt_poses)) if i < len(pred_R_list)]
    if len(idx) < 3:
        return float("nan")
    gt_R = [gt_poses[i][:3, :3] for i in idx]
    pr_R = [pred_R_list[i] for i in idx]
    rpe = []
    for j in range(1, len(idx)):
        Rd = (pr_R[j] @ pr_R[j-1].T) @ (gt_R[j] @ gt_R[j-1].T).T
        rpe.append(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)) * 180 / np.pi)
    return float(np.mean(rpe))


def evaluate_scene(scene_id, gpu_id, num_frames, checkpoint, budget):
    device = torch.device(f"cuda:{gpu_id}")
    image_size_hw = (518, 518)
    images, gt_poses = load_7scenes(scene_id, num_frames)
    images = images.to(device)

    results = {"scene": scene_id, "num_frames": num_frames, "budget": budget}

    # Frontend
    try:
        print("Loading OVGGT-main Frontend model...")
        model_f = build_frontend_model(checkpoint, device, budget)
        res_f = run_frontend_streaming(model_f, images, device, image_size_hw)
        del model_f
        gc.collect()
        torch.cuda.empty_cache()

        ate_f = ate_umeyama(res_f["pred_t"], gt_poses)
        rpe_f = rpe_rotation(res_f["pred_R"], gt_poses)
        results["frontend"] = {
            "ate_rmse": ate_f, "rot_error_deg": rpe_f,
            "time_s": res_f["time_s"], "fps": res_f["fps"],
            "peak_mem_gb": res_f["peak_mem_gb"],
        }
        print(f"[Frontend] ATE: {ate_f:.4f}m | RPE: {rpe_f:.2f}deg | "
              f"{res_f['time_s']:.1f}s ({res_f['fps']:.1f} FPS) | {res_f['peak_mem_gb']:.1f}GB")
    except Exception as e:
        results["frontend"] = {"error": str(e)}
        print(f"[Frontend] FAILED: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()

    # Legacy
    try:
        print("Loading OVGGT-main Legacy model...")
        model_l = build_legacy_model(checkpoint, device, budget)
        res_l = run_legacy_joint(model_l, images, device, image_size_hw)
        del model_l
        gc.collect()
        torch.cuda.empty_cache()

        ate_l = ate_umeyama(res_l["pred_t"], gt_poses)
        rpe_l = rpe_rotation(res_l["pred_R"], gt_poses)
        results["legacy"] = {
            "ate_rmse": ate_l, "rot_error_deg": rpe_l,
            "time_s": res_l["time_s"], "fps": res_l["fps"],
            "peak_mem_gb": res_l["peak_mem_gb"],
        }
        print(f"[Legacy]   ATE: {ate_l:.4f}m | RPE: {rpe_l:.2f}deg | "
              f"{res_l['time_s']:.1f}s ({res_l['fps']:.1f} FPS) | {res_l['peak_mem_gb']:.1f}GB")
    except Exception as e:
        results["legacy"] = {"error": str(e)}
        print(f"[Legacy] FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default="chess/seq-03")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--budget", type=int, default=200000)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.summarize:
        all_results = []
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if fname.endswith(".json"):
                with open(os.path.join(OUTPUT_DIR, fname)) as f:
                    all_results.append(json.load(f))

        print("\n" + "=" * 90)
        print("SUMMARY: OVGGT-main Frontend vs Legacy on 7-Scenes (200 frames)")
        print("=" * 90)
        header = f"{'Scene':<20} {'Mode':<10} {'ATE RMSE(m)':<14} {'RPE(deg)':<12} {'Time(s)':<10} {'FPS':<8} {'Mem(GB)':<8}"
        print(header)
        print("-" * 90)
        for r in all_results:
            for mode in ["frontend", "legacy"]:
                if mode in r and "error" not in r[mode]:
                    d = r[mode]
                    label = "Frontend" if mode == "frontend" else "Legacy"
                    print(f"{r['scene']:<20} {label:<10} {d['ate_rmse']:<14.4f} "
                          f"{d['rot_error_deg']:<12.2f} {d['time_s']:<10.1f} "
                          f"{d['fps']:<8.1f} {d['peak_mem_gb']:<8.1f}")
            print("-" * 90)
        return

    results = evaluate_scene(args.scene_id, args.gpu_id, args.num_frames, args.checkpoint, args.budget)

    safe_name = args.scene_id.replace("/", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
