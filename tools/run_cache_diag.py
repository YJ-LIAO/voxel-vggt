#!/usr/bin/env python3
"""
Diagnostic runner: confirms the P1 "protection crowds the budget" mechanism by
directly measuring per-frame protected_count, slot0_count, and anchor_overflow
during a live OVGGT frontend inference run, alongside the same ATE as the
ablation harness (so the diagnostic run is comparable to the ATE table).

Reuses run_p1_ablation.py's exact config construction + ATE math (same scene
loader, same checkpoint, same FrontendCacheConfig build) so results align with
the ablation. The only addition: attaching a read-only CacheDiagProbe.

Usage:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python tools/run_cache_diag.py \
      --config P1-current --scene chess/seq-03 --num_frames 500
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import time
import gc
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import torch

# reuse the ablation harness verbatim (config table, loader, ATE, build_model)
from run_p1_ablation import (CONFIGS, PER_LAYER_BUDGET, load_scene, predict_c2w,
                             compute_ate_rmse, build_model, set_seed_and_determinism)
from cache_diag_probe import CacheDiagProbe

CKPT = "/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth"
OUTPUT_DIR = os.path.join(HERE, "p1_ablation_results_lyj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="P1-current", choices=list(CONFIGS.keys()))
    ap.add_argument("--scene", default="chess/seq-03")
    ap.add_argument("--num_frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="diag")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")
    print(f"Loading checkpoint {CKPT} ...")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    set_seed_and_determinism(args.seed, deterministic=False)
    inputs, gt, h, w = load_scene(args.scene, args.num_frames)

    model = build_model(args.config)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    probe = CacheDiagProbe()
    model._oracle_eviction_probe = probe  # read-only callback, returns None

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

    pred_c2w = predict_c2w(output, h, w)
    ate = compute_ate_rmse(gt, pred_c2w)

    head = probe.headline()
    head["config"] = args.config
    head["scene"] = args.scene
    head["num_frames"] = args.num_frames
    head["seed"] = args.seed
    head["ate_rmse"] = ate
    head["time_s"] = elapsed
    head["per_layer_budget"] = PER_LAYER_BUDGET

    # save headline + per-frame time series (sampled to keep size bounded: every
    # frame that had an eviction event, plus all overflow frames)
    pf = probe.per_frame_summary()
    ts = [{"frame": f, **pf[f]} for f in sorted(pf)]

    out_path = os.path.join(
        OUTPUT_DIR, f"diag_{args.config}_{args.scene.replace('/', '_')}"
        f"_f{args.num_frames}_s{args.seed}_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"headline": head, "per_frame": ts}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DIAG: {args.config} {args.scene} f{args.num_frames} s{args.seed}")
    print("=" * 70)
    print(json.dumps(head, indent=2))
    print(f"\nATE={ate:.4f}m  ({args.num_frames/elapsed:.1f} FPS)")
    print(f"saved {out_path}")

    del model, output
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
