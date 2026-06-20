#!/usr/bin/env python3
"""Legacy-vs-frontend distribution comparison on 7-Scenes (single run).

- baseline (legacy): OVGGT(mode='legacy'), coverage anchor / interval=250
- current (frontend): mode='frontend_eval' + ring0.2 + budget8334 + intra ON
Same checkpoint, same scene; only inference mode differs -> fair comparison.
ATE math identical to tools/test_multi_scene.py (Sim3-aligned RMSE).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/run_legacy_vs_frontend.py \
      --mode legacy --scene chess/seq-03 --num_frames 500 --seed 0
"""
import os, sys, json, time, gc, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, torch
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri, ABS_POSE_ENCODING

DATASET = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
CKPT = "/path/to/mount/lyj/voxel-vggt/ckpt/checkpoints.pth"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_vs_frontend_lyj")


def getp(o, h, w):
    pe = torch.cat([r["camera_pose"] for r in o.ress], 0)
    ext, _ = pose_encoding_to_extri_intri(pe.unsqueeze(0), image_size_hw=(h, w))
    ext = ext.squeeze(0).cpu().numpy(); N = ext.shape[0]
    w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0); w2c[:, :3, :] = ext
    return np.linalg.inv(w2c)


def ate(gt, p):
    n = min(len(gt), len(p)); gp, pp = gt[:n, :3, 3], p[:n, :3, 3]
    gm, pm = gp.mean(0), pp.mean(0); gc, pc = gp - gm, pp - pm
    H = pc.T @ gc; U, S, Vh = np.linalg.svd(H); R = Vh.T @ U.T
    if np.linalg.det(R) < 0: Vh[2] *= -1; R = Vh.T @ U.T
    s = S.sum() / (np.trace(pc.T @ pc) + 1e-8); al = (s * (R @ pp[:n].T)).T + (gm - s * R @ pm)
    return float(np.sqrt(np.mean(np.sum((al - gp) ** 2, axis=1))))


def build(mode, intra_mode="drop"):
    if mode == "legacy":
        return OVGGT(mode="legacy", per_layer_budget=8334)
    return OVGGT(mode="frontend_eval", per_layer_budget=8334,
                 frontend_pose_encoding_type=ABS_POSE_ENCODING,
                 frontend_cache_config=FrontendCacheConfig(
                     enabled=True, dedup_enabled=True, intra_frame_dedup_enabled=True,
                     fifo_keep_topk=80, fifo_protected_ring_ratio=0.2,
                     intra_dedup_mode=intra_mode))


def infer(m, mode, inputs):
    if mode == "legacy":
        return m.inference(inputs, history_anchor_strategy="coverage", anchor_interval=250)
    return m.inference(inputs, history_anchor_strategy="fixed_interval", anchor_interval=8, max_anchors=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["legacy", "frontend"])
    ap.add_argument("--scene", default="chess/seq-03")
    ap.add_argument("--num_frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intra-mode", default="drop", choices=["drop", "merge"])
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: builtin CKPT)")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = args.ckpt if args.ckpt else CKPT

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    scene_path = os.path.join(DATASET, args.scene)
    cfs = sorted(f for f in os.listdir(scene_path) if f.endswith(".color.png"))[:args.num_frames]
    imgs = load_and_preprocess_images([os.path.join(scene_path, f) for f in cfs]).cuda()
    inputs = [{"img": i.unsqueeze(0)} for i in imgs]
    gt = np.array([np.loadtxt(os.path.join(scene_path, f.replace(".color.png", ".pose.txt"))).astype(np.float32) for f in cfs])
    h, w = imgs.shape[2], imgs.shape[3]

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    m = build(args.mode, intra_mode=args.intra_mode); m.load_state_dict(sd, strict=False); m = m.cuda().eval()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): o = infer(m, args.mode, inputs)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    a = ate(gt, getp(o, h, w))
    del m, o; gc.collect(); torch.cuda.empty_cache()

    res = dict(mode=args.mode, scene=args.scene, num_frames=args.num_frames, seed=args.seed,
               ate_rmse=a, time_s=elapsed, fps=args.num_frames/elapsed, peak_mem_gb=peak_mem)
    out = os.path.join(OUTPUT_DIR, f"{args.mode}{args.intra_mode}_{args.scene.replace('/', '_')}_f{args.num_frames}_s{args.seed}.json")
    with open(out, "w") as f: json.dump(res, f, indent=2)
    print(f"[{args.mode}] {args.scene} f{args.num_frames} s{args.seed} | ATE={a:.4f}m | {res['fps']:.1f}FPS | {peak_mem:.1f}GB")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
