"""Compare reconstruction (depth) accuracy: frontend vs legacy on 7-Scenes.
Extracts predicted depth from model output, compares to GT depth maps.
Metrics: AbsRel, RMSE, δ1 (threshold accuracy), valid pixel coverage.
"""
import os, sys, json, argparse, gc
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import numpy as np
import torch
from PIL import Image
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import ABS_POSE_ENCODING

DATASET = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
CKPT = "/path/to/mount/lyj/voxel-vggt/ckpt/checkpoints.pth"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["legacy", "frontend"])
ap.add_argument("--scene", default="chess/seq-03")
ap.add_argument("--num_frames", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("cuda:0")
sp = os.path.join(DATASET, args.scene)
cfs = sorted(f for f in os.listdir(sp) if f.endswith(".color.png"))[:args.num_frames]
imgs = load_and_preprocess_images([os.path.join(sp, f) for f in cfs]).to(device)
inputs = [{"img": i.unsqueeze(0)} for i in imgs]
H_gt, W_gt = 480, 640  # 7-Scenes native resolution

# Load GT depths
gt_depths = []
for f in cfs:
    d = np.array(Image.open(os.path.join(sp, f.replace(".color.png", ".depth.png")))).astype(np.float32) / 1000.0
    gt_depths.append(d)
gt_depths = np.stack(gt_depths)  # [N, 480, 640] in meters

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd: sd = sd["model"]

# Build model
if args.mode == "legacy":
    model = OVGGT(mode="legacy", per_layer_budget=8334)
else:
    model = OVGGT(mode="frontend_eval", per_layer_budget=8334,
                  frontend_pose_encoding_type=ABS_POSE_ENCODING,
                  frontend_cache_config=FrontendCacheConfig(
                      enabled=True, dedup_enabled=True, intra_frame_dedup_enabled=True,
                      fifo_keep_topk=80, fifo_protected_ring_ratio=0.2,
                      budget_allocation="uniform", eviction_importance_weight=0.5))
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()

with torch.no_grad():
    if args.mode == "legacy":
        output = model.inference(inputs, history_anchor_strategy="coverage", anchor_interval=250)
    else:
        output = model.inference(inputs, history_anchor_strategy="fixed_interval", anchor_interval=8, max_anchors=3)

# Extract predicted depth per frame
pred_depths = []
for i, res in enumerate(output.ress):
    d = res.get("depth")  # [1, 1, H, W] or similar
    if d is None:
        pred_depths.append(None)
        continue
    d = d.squeeze().cpu().numpy()  # [H, W]
    pred_depths.append(d)

# Compute metrics
abs_rel_all, rmse_all, delta1_all, valid_ratio_all = [], [], [], []
for i in range(min(len(pred_depths), len(gt_depths))):
    pd = pred_depths[i]
    if pd is None or pd.size == 0:
        continue
    gd = gt_depths[i]

    # Resize pred to GT resolution if needed
    if pd.shape != gd.shape:
        pd_resized = np.array(Image.fromarray(pd).resize((W_gt, H_gt), Image.BILINEAR))
    else:
        pd_resized = pd

    # Valid mask: GT depth > 0 and < 10m (remove invalid/sky)
    valid = (gd > 0.1) & (gd < 10.0)
    if valid.sum() < 100:
        continue

    gp = gd[valid]
    pp = pd_resized[valid]

    # Remove pred outliers (>50m or <=0)
    pred_valid = pp > 0
    gp, pp = gp[pred_valid], pp[pred_valid]
    if len(gp) < 50:
        continue

    # Standard depth metrics
    abs_rel = np.mean(np.abs(pp - gp) / gp)
    rmse = np.sqrt(np.mean((pp - gp) ** 2))
    thresh = np.maximum(pp / gp, gp / pp)
    delta1 = (thresh < 1.25).mean()
    valid_ratio = valid.sum() / valid.size

    abs_rel_all.append(abs_rel)
    rmse_all.append(rmse)
    delta1_all.append(delta1)
    valid_ratio_all.append(valid_ratio)

n = len(abs_rel_all)
if n == 0:
    print(f"[{args.mode}] {args.scene} f{args.num_frames}: NO VALID FRAMES for depth comparison")
else:
    result = {
        "mode": args.mode, "scene": args.scene, "num_frames": args.num_frames,
        "n_valid_frames": n,
        "abs_rel": float(np.mean(abs_rel_all)),
        "rmse_m": float(np.mean(rmse_all)),
        "delta1": float(np.mean(delta1_all)),
        "valid_pixel_ratio": float(np.mean(valid_ratio_all)),
    }
    out_path = f"/path/to/mount/lyj/voxel-vggt/tools/legacy_vs_frontend_lyj/depth_{args.mode}_{args.scene.replace('/', '_')}_f{args.num_frames}_s0.json"
    with open(out_path, "w") as f: json.dump(result, f, indent=2)
    print(f"[{args.mode}] {args.scene} f{args.num_frames} | n_frames={n}")
    print(f"  AbsRel={result['abs_rel']:.4f}  RMSE={result['rmse_m']:.4f}m  δ1={result['delta1']:.4f}  valid_px={result['valid_pixel_ratio']:.2f}")
    print(f"  saved {out_path}")
