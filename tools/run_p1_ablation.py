#!/usr/bin/env python3
"""
P1 cache-config ablation on 7-Scenes (spec §3.1, Stage 1).

Runs voxel-vggt FrontendEval with an EXPLICIT FrontendCacheConfig (the eval_7scenes_compare.py
defaults of fifo_keep_topk=0 / intra_frame_dedup=True do NOT match the production noIntra+fifo80
baseline, so we build the config explicitly here).

Per the spec §3.1 fixed config table, production baseline is:
    per_layer_budget=8000, dedup_enabled=True, intra_frame_dedup_enabled=False,
    fifo_keep_topk=80, budget_allocation='dynamic', learned_fifo_keep_count=False.

The ring_capacity (ovggt.py:849) = fifo_protected_ring_ratio * per_layer_budget, so with the
production budget 8000 the swept ratios are r0.1->800, r0.2->1600, r0.3->2400, r0.5->4000 tokens.

Determinism (spec §3.2 determinism pre-check): --seed sets all RNG; --deterministic turns on
torch.use_deterministic_algorithms(True) + cudnn.deterministic + CUBLAS_WORKSPACE_CONFIG. The
pre-check runs the same (config, scene, num_frames, seed) twice and compares ATE for bit-identical.

Metric: ATE = Sim3-aligned absolute trajectory RMSE (meters), same as eval_7scenes_compare.py.

Usage (single config, one scene):
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python tools/run_p1_ablation.py \\
        --config P1-current --scene chess/seq-03 --num_frames 200 --seed 42 --deterministic --tag r1
"""
import os
# Must be set before CUDA context init for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import time
import gc
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri, ABS_POSE_ENCODING

DATASET_ROOT = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
CKPT = "/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth"
PER_LAYER_BUDGET = 8000  # production noIntra+fifo80 budget (test_multi_scene.py:66-69)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_ablation_results_lyj")

# Spec §3.1 fixed config table (P1, P2 held at baseline).
# All share: dedup_enabled=True, intra_frame_dedup_enabled=False, budget_allocation='dynamic',
#            learned_fifo_keep_count=False, per_layer_budget=8000.
# v1/v2 mutex (FrontendCacheConfig.__post_init__): ring_ratio>0 requires max_protected_ratio==1.0.
CONFIGS = {
    "P1-none":       dict(fifo_keep_topk=0,  fifo_protected_ring_ratio=0.0, max_protected_ratio=1.0),
    "P1-current":    dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.0, max_protected_ratio=1.0),
    "P1-v1cap":      dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.0, max_protected_ratio=0.5),
    "P1-mechC-r0.1": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.1, max_protected_ratio=1.0),
    "P1-mechC-r0.2": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.2, max_protected_ratio=1.0),
    "P1-mechC-r0.3": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.3, max_protected_ratio=1.0),
    "P1-mechC-r0.5": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.5, max_protected_ratio=1.0),
    # --- P2 Phase 1: eviction old/new-token weight sweep on the P1-selected
    # production config (ring0.2 + budget8334 + intra ON + uniform). Only
    # eviction_importance_weight varies; dedup's importance_weight stays 0.5
    # (decoupled in FrontendCacheConfig). The shared baseline overrides below
    # set the P1 production params; these dicts only set the swept weight.
    "P2-w0.3": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.2, max_protected_ratio=1.0,
                    eviction_importance_weight=0.3),
    "P2-w0.5": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.2, max_protected_ratio=1.0,
                    eviction_importance_weight=0.5),
    "P2-w0.7": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.2, max_protected_ratio=1.0,
                    eviction_importance_weight=0.7),
}


def set_seed_and_determinism(seed, deterministic):
    """Set all RNG. Optionally force deterministic CUDA algos (may raise on unsupported ops)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def load_scene(scene_id, num_frames):
    scene_path = os.path.join(DATASET_ROOT, scene_id)
    if not os.path.isdir(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    color_files = sorted(f for f in os.listdir(scene_path) if f.endswith(".color.png"))[:num_frames]
    if len(color_files) < num_frames:
        raise FileNotFoundError(f"Scene {scene_id} has {len(color_files)} frames, need {num_frames}")
    imgs = load_and_preprocess_images([os.path.join(scene_path, f) for f in color_files]).cuda()
    inputs = [{"img": i.unsqueeze(0)} for i in imgs]
    gt = np.array([
        np.loadtxt(os.path.join(scene_path, f.replace(".color.png", ".pose.txt"))).astype(np.float32)
        for f in color_files
    ])
    h, w = imgs.shape[2], imgs.shape[3]
    return inputs, gt, h, w


def predict_c2w(output, h, w):
    pe = torch.cat([r["camera_pose"] for r in output.ress], 0)
    if pe.dim() == 1:
        pe = pe.unsqueeze(0)
    ext, _ = pose_encoding_to_extri_intri(pe.unsqueeze(0), image_size_hw=(h, w))
    ext = ext.squeeze(0).cpu().numpy()
    n = ext.shape[0]
    w2c = np.eye(4, dtype=np.float32)[None].repeat(n, 0)
    w2c[:, :3, :] = ext
    return np.linalg.inv(w2c)


def compute_ate_rmse(gt_poses, pred_poses):
    """Sim3-aligned ATE RMSE, identical math to eval_7scenes_compare.compute_ate_rmse."""
    n = min(len(gt_poses), len(pred_poses))
    gp, pp = gt_poses[:n, :3, 3], pred_poses[:n, :3, 3]
    gm, pm = gp.mean(0), pp.mean(0)
    gc, pc = gp - gm, pp - pm
    H = pc.T @ gc
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2] *= -1
        R = Vh.T @ U.T
    s = S.sum() / (np.trace(pc.T @ pc) + 1e-8)
    al = (s * (R @ pp.T)).T + (gm - s * R @ pm)
    return float(np.sqrt(np.mean(np.sum((al - gp) ** 2, axis=1))))


def build_model(config_name):
    p = CONFIGS[config_name]
    # P2-* configs run on the P1-selected PRODUCTION baseline (ring0.2 + budget8334
    # + intra ON + uniform); legacy P1-* keep their original noIntra+dynamic+8000
    # baseline for historical comparability.
    if config_name.startswith("P2-"):
        base = dict(enabled=True, dedup_enabled=True, intra_frame_dedup_enabled=True,
                    budget_allocation="uniform", learned_fifo_keep_count=False)
        budget = 8334
    else:
        base = dict(enabled=True, dedup_enabled=True, intra_frame_dedup_enabled=False,
                    budget_allocation="dynamic", learned_fifo_keep_count=False)
        budget = PER_LAYER_BUDGET
    cfg = FrontendCacheConfig(**base, **p)
    model = OVGGT(
        mode="frontend_eval",
        per_layer_budget=budget,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=cfg,
    )
    return model


def run_once(config_name, scene_id, num_frames, seed, deterministic, state_dict, device, save_poses=False):
    set_seed_and_determinism(seed, deterministic)
    inputs, gt, h, w = load_scene(scene_id, num_frames)
    model = build_model(config_name)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
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
    pred_c2w = predict_c2w(output, h, w)
    ate = compute_ate_rmse(gt, pred_c2w)
    result = {
        "config": config_name,
        "scene": scene_id,
        "num_frames": num_frames,
        "seed": seed,
        "deterministic": deterministic,
        "ate_rmse": ate,
        "time_s": elapsed,
        "fps": num_frames / elapsed,
        "peak_mem_gb": peak_mem,
        "per_layer_budget": PER_LAYER_BUDGET,
    }
    if save_poses:
        result["pred_c2w"] = pred_c2w.tolist()
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="P1 cache-config ablation on 7-Scenes")
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--scene", default="chess/seq-03")
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--save_poses", action="store_true")
    parser.add_argument("--determinism_check", action="store_true",
                        help="Run the same config twice and report whether ATE is bit-identical.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    print(f"Loading checkpoint {CKPT} ...")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    if args.determinism_check:
        r1 = run_once(args.config, args.scene, args.num_frames, args.seed,
                      args.deterministic, sd, device, save_poses=True)
        r2 = run_once(args.config, args.scene, args.num_frames, args.seed,
                      args.deterministic, sd, device, save_poses=True)
        p1 = np.array(r1["pred_c2w"])
        p2 = np.array(r2["pred_c2w"])
        poses_identical = np.array_equal(p1, p2)
        ate_diff = abs(r1["ate_rmse"] - r2["ate_rmse"])
        print("\n=== DETERMINISM PRE-CHECK (spec §3.2) ===")
        print(f"  config={args.config} scene={args.scene} frames={args.num_frames} "
              f"seed={args.seed} deterministic={args.deterministic}")
        print(f"  run1 ATE={r1['ate_rmse']:.6f}  run2 ATE={r2['ate_rmse']:.6f}")
        print(f"  |ATE diff| = {ate_diff:.2e}")
        print(f"  poses bit-identical: {poses_identical}")
        print(f"  ATE bit-identical:   {r1['ate_rmse'] == r2['ate_rmse']}")
        out = {"run1": r1, "run2": r2, "poses_identical": bool(poses_identical),
               "ate_diff": ate_diff}
        fname = f"detcheck_{args.config}_{args.scene.replace('/', '_')}_f{args.num_frames}_s{args.seed}{'_det' if args.deterministic else ''}.json"
        with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved {os.path.join(OUTPUT_DIR, fname)}")
        return

    r = run_once(args.config, args.scene, args.num_frames, args.seed,
                 args.deterministic, sd, device, save_poses=args.save_poses)
    tag = f"_{args.tag}" if args.tag else ""
    fname = (f"{args.config}_{args.scene.replace('/', '_')}_f{args.num_frames}"
             f"_s{args.seed}{'_det' if args.deterministic else ''}{tag}.json")
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        json.dump(r, f, indent=2)
    print(f"[{args.config}] {args.scene} f{args.num_frames} s{args.seed} | "
          f"ATE={r['ate_rmse']:.4f}m | {r['fps']:.1f} FPS | {r['peak_mem_gb']:.1f}GB")
    print(f"  saved {os.path.join(OUTPUT_DIR, fname)}")


if __name__ == "__main__":
    main()
