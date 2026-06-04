#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_SCENE = "588c989a90414422fbe86d96"
DEFAULT_SEQUENCES = [(0, 3, 4), (12, 13, 14), (31, 32, 34)]
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OVGGT legacy/frontend inference on BlendMVS1.")
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
        "--sequence",
        action="append",
        default=[],
        help="Comma-separated frame indices, e.g. 0,3,4. Can be repeated.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Frontend voxel size.",
    )
    parser.add_argument(
        "--use-token-scorer",
        action="store_true",
        help="Enable token scorer in the model.",
    )
    parser.add_argument(
        "--token-scorer-checkpoint",
        default=None,
        help="Path to token scorer checkpoint (requires --use-token-scorer).",
    )
    parser.add_argument(
        "--use-count-head",
        action="store_true",
        help="Enable count head in the model.",
    )
    parser.add_argument(
        "--count-head-checkpoint",
        default=None,
        help="Path to count head checkpoint (requires --use-count-head).",
    )
    parser.add_argument(
        "--learned-fifo-keep-count",
        action="store_true",
        help="Use learned count head for fifo keep count.",
    )
    parser.add_argument(
        "--fifo-keep-topk",
        type=int,
        default=80,
        help="Top-K tokens to retain by score when demoting oldest anchor.",
    )
    parser.add_argument(
        "--fifo-count-candidates",
        default="0,8,16,32,64,128",
        help="Comma-separated candidate counts for learned fifo keep count.",
    )
    return parser.parse_args()


def parse_sequence_args(sequence_args: Sequence[str]) -> List[Tuple[int, ...]]:
    if not sequence_args:
        return list(DEFAULT_SEQUENCES)
    sequences = []
    for item in sequence_args:
        seq = tuple(int(part.strip()) for part in item.split(",") if part.strip())
        if not seq:
            raise ValueError(f"Invalid empty sequence: {item!r}")
        sequences.append(seq)
    return sequences


def load_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        header = handle.readline().decode("utf-8").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}: {header}")
        match = re.match(r"^(\d+)\s(\d+)\s$", handle.readline().decode("utf-8"))
        if match is None:
            raise ValueError(f"Invalid PFM size line in {path}")
        width, height = map(int, match.groups())
        scale = float(handle.readline().decode("utf-8").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(handle.read(), dtype=dtype)
        if header == "PF":
            data = data.reshape(height, width, 3)
        else:
            data = data.reshape(height, width)
        data = np.flipud(data)
    return data.astype(np.float32)


def load_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        world_to_cam = np.loadtxt(handle, skiprows=1, max_rows=4, dtype=np.float32)
        intrinsics = np.loadtxt(handle, skiprows=2, max_rows=3, dtype=np.float32)
    return intrinsics, world_to_cam


def compute_preprocess_geometry(height: int, width: int, target_size: int = 518) -> Dict[str, float]:
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    crop_top = 0
    if new_height > target_size:
        crop_top = (new_height - target_size) // 2
        final_height = target_size
    else:
        final_height = new_height
    return {
        "new_width": new_width,
        "new_height": new_height,
        "final_height": final_height,
        "scale_x": new_width / width,
        "scale_y": new_height / height,
        "crop_top": crop_top,
    }


def preprocess_depth_and_intrinsics(depth: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape[:2]
    geom = compute_preprocess_geometry(height, width)
    depth_resized = cv2.resize(
        depth,
        (int(geom["new_width"]), int(geom["new_height"])),
        interpolation=cv2.INTER_NEAREST,
    )
    valid_resized = cv2.resize(
        (depth > 0).astype(np.uint8),
        (int(geom["new_width"]), int(geom["new_height"])),
        interpolation=cv2.INTER_NEAREST,
    )
    if geom["crop_top"] > 0:
        start = int(geom["crop_top"])
        end = start + int(geom["final_height"])
        depth_resized = depth_resized[start:end]
        valid_resized = valid_resized[start:end]
    depth_resized = depth_resized.astype(np.float32)
    depth_resized[valid_resized == 0] = 0.0

    intrinsics_out = intrinsics.copy().astype(np.float32)
    intrinsics_out[0, 0] *= geom["scale_x"]
    intrinsics_out[1, 1] *= geom["scale_y"]
    intrinsics_out[0, 2] *= geom["scale_x"]
    intrinsics_out[1, 2] = intrinsics_out[1, 2] * geom["scale_y"] - geom["crop_top"]
    return depth_resized, intrinsics_out


def make_4x4(extrinsics_3x4: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = extrinsics_3x4
    return out


def depth_metrics_ssi(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    x = pred[mask].reshape(-1).astype(np.float64)
    y = gt[mask].reshape(-1).astype(np.float64)
    if x.size == 0:
        return {
            "depth_mae_ssi": float("nan"),
            "depth_rmse_ssi": float("nan"),
            "depth_absrel_ssi": float("nan"),
        }
    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    aligned = scale * pred + shift
    err = aligned[mask] - gt[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    absrel = float(np.mean(np.abs(err) / np.clip(gt[mask], EPS, None)))
    return {
        "depth_mae_ssi": mae,
        "depth_rmse_ssi": rmse,
        "depth_absrel_ssi": absrel,
    }


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 arrays, got {src.shape} and {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, singular, Vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign[-1] = -1.0
    R = U @ np.diag(sign) @ Vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = float(np.sum(singular * sign) / max(src_var, EPS))
    t = dst_mean - scale * (R @ src_mean)
    return scale, R.astype(np.float32), t.astype(np.float32)


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_pred @ R_gt.T
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def backproject_depth_to_world(depth: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    z = depth.astype(np.float32)
    x = (xs - intrinsics[0, 2]) * z / max(intrinsics[0, 0], EPS)
    y = (ys - intrinsics[1, 2]) * z / max(intrinsics[1, 1], EPS)
    cam_points = np.stack([x, y, z], axis=-1)
    world_points = cam_points @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
    return world_points.astype(np.float32)


def prepare_sequence_data(scene_dir: str, frame_ids: Sequence[int]) -> Dict[str, object]:
    image_paths = []
    gt_depths = []
    gt_intrinsics = []
    gt_c2w = []

    raw_layout = os.path.isdir(os.path.join(scene_dir, "blended_images"))
    processed_stems = None
    if not raw_layout:
        processed_stems = sorted(os.path.splitext(name)[0] for name in os.listdir(scene_dir) if name.endswith(".jpg"))
        if not processed_stems:
            raise FileNotFoundError(f"No BlendMVS frames found under {scene_dir}")

    for frame_id in frame_ids:
        if raw_layout:
            stem = f"{frame_id:08d}"
            image_path = os.path.join(scene_dir, "blended_images", f"{stem}.jpg")
            depth_path = os.path.join(scene_dir, "rendered_depth_maps", f"{stem}.pfm")
            cam_path = os.path.join(scene_dir, "cams", f"{stem}_cam.txt")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(depth_path)
            if not os.path.isfile(cam_path):
                raise FileNotFoundError(cam_path)

            intrinsics, world_to_cam = load_cam_file(cam_path)
            depth = load_pfm(depth_path)
            cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)
        else:
            if frame_id < 0 or frame_id >= len(processed_stems):
                raise IndexError(f"Frame index {frame_id} is out of range for {scene_dir} with {len(processed_stems)} frames")
            stem = processed_stems[frame_id]
            image_path = os.path.join(scene_dir, f"{stem}.jpg")
            depth_path = os.path.join(scene_dir, f"{stem}.exr")
            cam_path = os.path.join(scene_dir, f"{stem}.npz")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(depth_path)
            if not os.path.isfile(cam_path):
                raise FileNotFoundError(cam_path)

            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise IOError(f"Could not load EXR depth: {depth_path}")
            with np.load(cam_path) as camera_params:
                intrinsics = np.float32(camera_params["intrinsics"])
                cam_to_world = np.eye(4, dtype=np.float32)
                cam_to_world[:3, :3] = np.float32(camera_params["R_cam2world"])
                cam_to_world[:3, 3] = np.float32(camera_params["t_cam2world"])

        depth_processed, intrinsics_processed = preprocess_depth_and_intrinsics(depth, intrinsics)

        image_paths.append(image_path)
        gt_depths.append(depth_processed)
        gt_intrinsics.append(intrinsics_processed)
        gt_c2w.append(cam_to_world)

    return {
        "image_paths": image_paths,
        "gt_depths": np.stack(gt_depths, axis=0),
        "gt_intrinsics": np.stack(gt_intrinsics, axis=0),
        "gt_c2w": np.stack(gt_c2w, axis=0),
    }


def build_frames(image_paths: Sequence[str], device: torch.device) -> List[dict]:
    images = load_and_preprocess_images(list(image_paths)).to(device)
    return [{"img": images[idx].unsqueeze(0)} for idx in range(images.shape[0])]


def load_model(
    weights_path: str,
    device: torch.device,
    frontend_enabled: bool,
    voxel_size: float,
    use_token_scorer: bool = False,
    use_count_head: bool = False,
    learned_fifo_keep_count: bool = False,
    fifo_keep_topk: int = 0,
    fifo_count_candidates: tuple = (0, 8, 16, 32, 64, 128),
    token_scorer_checkpoint: str = None,
    count_head_checkpoint: str = None,
    count_head_arch: str = None,
    count_head_hidden_dim: int = None,
) -> OVGGT:
    # --- Read checkpoint metadata to auto-detect count head config ---
    if count_head_checkpoint and use_count_head:
        ckpt_meta = torch.load(count_head_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt_meta, dict):
            # Auto-detect arch from checkpoint metadata if not explicitly provided
            if count_head_arch is None and "count_head_arch" in ckpt_meta:
                count_head_arch = ckpt_meta["count_head_arch"]
            # Auto-detect hidden_dim from checkpoint metadata
            if count_head_hidden_dim is None and "count_head_hidden_dim" in ckpt_meta:
                count_head_hidden_dim = ckpt_meta["count_head_hidden_dim"]
            # Auto-detect candidates from checkpoint metadata
            if "count_candidates" in ckpt_meta:
                count_head_candidates_meta = tuple(ckpt_meta["count_candidates"])
                # Use checkpoint candidates unless explicitly overridden via CLI
                if fifo_count_candidates == (0, 8, 16, 32, 64, 128):
                    fifo_count_candidates = count_head_candidates_meta

    # Defaults if not detected from checkpoint
    if count_head_arch is None:
        count_head_arch = "pooled_v1"

    model_kwargs = {
        "mode": "frontend_eval" if frontend_enabled else "legacy",
        "use_token_scorer": use_token_scorer,
        "use_count_head": use_count_head,
        "count_head_arch": count_head_arch,
    }
    if count_head_hidden_dim is not None and use_count_head:
        model_kwargs["count_head_hidden_dim"] = count_head_hidden_dim
    if frontend_enabled:
        model_kwargs["frontend_cache_config"] = FrontendCacheConfig(
            enabled=True,
            dedup_enabled=True,
            export_keyframe_packets=True,
            voxel_size=voxel_size,
            fifo_keep_topk=fifo_keep_topk,
            learned_fifo_keep_count=learned_fifo_keep_count,
            fifo_count_candidates=fifo_count_candidates,
        )
    model = OVGGT(**model_kwargs).to(device)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    if token_scorer_checkpoint:
        model.load_token_scorer_checkpoint(token_scorer_checkpoint)
    if count_head_checkpoint:
        model.load_count_head_checkpoint(count_head_checkpoint)
    model.eval()
    return model


def decode_predicted_poses(camera_pose_list: List[torch.Tensor], image_hw: Tuple[int, int]) -> np.ndarray:
    pose_enc = torch.stack([pose.squeeze(0) for pose in camera_pose_list], dim=0)
    extrinsic, _ = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), image_hw)
    w2c = extrinsic.squeeze(0).detach().cpu().numpy()
    c2w = np.stack([np.linalg.inv(make_4x4(item)) for item in w2c], axis=0).astype(np.float32)
    return c2w


def run_inference(
    model: OVGGT,
    frames: List[dict],
    device: torch.device,
    frontend_enabled: bool,
) -> Dict[str, object]:
    inference_kwargs = (
        {"history_anchor_strategy": "fixed_interval"}
        if frontend_enabled
        else {"history_anchor_strategy": "none"}
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        if device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model.inference(frames, **inference_kwargs)
        else:
            output = model.inference(frames, **inference_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_sec = time.perf_counter() - start

    depth = np.stack(
        [res["depth"].squeeze(0).squeeze(-1).detach().cpu().numpy() for res in output.ress],
        axis=0,
    )
    points = np.stack(
        [res["pts3d_in_other_view"].squeeze(0).detach().cpu().numpy() for res in output.ress],
        axis=0,
    )
    c2w = decode_predicted_poses(
        [res["camera_pose"] for res in output.ress],
        image_hw=depth.shape[-2:],
    )
    metrics = {
        "runtime_sec": runtime_sec,
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
    return {
        "depth": depth,
        "points": points,
        "c2w": c2w,
        "metrics": metrics,
    }


def evaluate_sequence(prediction: Dict[str, object], sequence_data: Dict[str, object]) -> Dict[str, float]:
    pred_depth = prediction["depth"]
    pred_points = prediction["points"]
    pred_c2w = prediction["c2w"]

    gt_depth = sequence_data["gt_depths"]
    gt_intrinsics = sequence_data["gt_intrinsics"]
    gt_c2w = sequence_data["gt_c2w"]

    depth_metric_list = []
    pred_clouds = []
    gt_clouds = []

    for idx in range(pred_depth.shape[0]):
        mask = np.isfinite(gt_depth[idx]) & (gt_depth[idx] > 0) & np.isfinite(pred_depth[idx])
        depth_metric_list.append(depth_metrics_ssi(pred_depth[idx], gt_depth[idx], mask))

        gt_world = backproject_depth_to_world(gt_depth[idx], gt_intrinsics[idx], gt_c2w[idx])
        point_mask = mask & np.isfinite(pred_points[idx]).all(axis=-1) & np.isfinite(gt_world).all(axis=-1)
        if np.any(point_mask):
            pred_clouds.append(pred_points[idx][point_mask])
            gt_clouds.append(gt_world[point_mask])

    sequence_metrics = {
        key: float(np.mean([item[key] for item in depth_metric_list]))
        for key in depth_metric_list[0]
    }

    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    pose_scale, pose_R, pose_t = umeyama_similarity(pred_centers, gt_centers)
    aligned_centers = pose_scale * (pred_centers @ pose_R.T) + pose_t
    trans_err = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    aligned_rot = pose_R[None] @ pred_c2w[:, :3, :3]
    rot_err = np.array(
        [rotation_error_deg(aligned_rot[idx], gt_c2w[idx, :3, :3]) for idx in range(pred_c2w.shape[0])],
        dtype=np.float32,
    )
    sequence_metrics["pose_ate_rmse"] = float(np.sqrt(np.mean(trans_err ** 2)))
    sequence_metrics["pose_rot_deg_mean"] = float(np.mean(rot_err))
    sequence_metrics["pose_rot_deg_max"] = float(np.max(rot_err))

    pred_points_all = np.concatenate(pred_clouds, axis=0)
    gt_points_all = np.concatenate(gt_clouds, axis=0)
    point_scale, point_R, point_t = umeyama_similarity(pred_points_all, gt_points_all)
    aligned_points = point_scale * (pred_points_all @ point_R.T) + point_t
    point_err = np.linalg.norm(aligned_points - gt_points_all, axis=1)
    sequence_metrics["point_mae_sim3"] = float(np.mean(point_err))
    sequence_metrics["point_rmse_sim3"] = float(np.sqrt(np.mean(point_err ** 2)))

    return sequence_metrics


def average_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    scene_dir = os.path.join(args.dataset_root, args.scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(scene_dir)

    # Validate learned-fifo-keep-count requires count head
    if args.learned_fifo_keep_count and not args.use_count_head:
        raise ValueError("--learned-fifo-keep-count requires --use-count-head")
    if args.learned_fifo_keep_count and not args.count_head_checkpoint:
        raise ValueError("--learned-fifo-keep-count requires --count-head-checkpoint")

    # Parse fifo_count_candidates from comma-separated string to tuple
    fifo_count_candidates = tuple(
        int(c.strip()) for c in args.fifo_count_candidates.split(",") if c.strip()
    )

    sequences = parse_sequence_args(args.sequence)
    legacy_model = load_model(args.weights, device, frontend_enabled=False, voxel_size=args.voxel_size)
    frontend_model = load_model(
        args.weights,
        device,
        frontend_enabled=True,
        voxel_size=args.voxel_size,
        use_token_scorer=args.use_token_scorer,
        use_count_head=args.use_count_head,
        learned_fifo_keep_count=args.learned_fifo_keep_count,
        fifo_keep_topk=args.fifo_keep_topk,
        fifo_count_candidates=fifo_count_candidates,
        token_scorer_checkpoint=args.token_scorer_checkpoint,
        count_head_checkpoint=args.count_head_checkpoint,
    )

    per_sequence = []
    legacy_metrics = []
    frontend_metrics = []
    diff_metrics = []

    for frame_ids in sequences:
        seq_key = ",".join(str(idx) for idx in frame_ids)
        sequence_data = prepare_sequence_data(scene_dir, frame_ids)
        frames = build_frames(sequence_data["image_paths"], device)

        legacy_prediction = run_inference(legacy_model, frames, device, frontend_enabled=False)
        frontend_prediction = run_inference(frontend_model, frames, device, frontend_enabled=True)

        legacy_eval = {**legacy_prediction["metrics"], **evaluate_sequence(legacy_prediction, sequence_data)}
        frontend_eval = {**frontend_prediction["metrics"], **evaluate_sequence(frontend_prediction, sequence_data)}
        diff_eval = {
            "legacy_frontend_depth_l1": float(
                np.mean(np.abs(legacy_prediction["depth"] - frontend_prediction["depth"]))
            ),
            "legacy_frontend_pts3d_l1": float(
                np.mean(np.abs(legacy_prediction["points"] - frontend_prediction["points"]))
            ),
        }

        legacy_metrics.append(legacy_eval)
        frontend_metrics.append(frontend_eval)
        diff_metrics.append(diff_eval)
        per_sequence.append(
            {
                "sequence": seq_key,
                "legacy": legacy_eval,
                "frontend": frontend_eval,
                "diff": diff_eval,
            }
        )
        print(json.dumps(per_sequence[-1], indent=2))

    summary = {
        "dataset_root": args.dataset_root,
        "scene": args.scene,
        "device": args.device,
        "sequences": [list(item) for item in sequences],
        "legacy_mean": average_metrics(legacy_metrics),
        "frontend_mean": average_metrics(frontend_metrics),
        "diff_mean": average_metrics(diff_metrics),
        "per_sequence": per_sequence,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
