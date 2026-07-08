import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm
import uuid
import json
from collections import defaultdict
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


def resolve_7scenes_root(data_root: str) -> str:
    return data_root or "./data/7scenes"


def validate_model_mode(model_name: str, ovggt_mode: str) -> None:
    if model_name != "OVGGT" and ovggt_mode != "legacy":
        raise ValueError(
            f"frontend mode is only supported for OVGGT, got model_name={model_name!r}, "
            f"ovggt_mode={ovggt_mode!r}"
        )


def should_run_reconstruction_eval(model_name: str) -> bool:
    return model_name in {"OVGGT", "VGGT", "stream3r"}


def filter_finite_point_pairs(pred_points, gt_points, colors=None):
    valid = np.isfinite(pred_points).all(axis=-1) & np.isfinite(gt_points).all(axis=-1)
    if colors is None:
        return pred_points[valid], gt_points[valid]
    return pred_points[valid], gt_points[valid], colors[valid]


def build_accelerator_kwargs_handlers(timeout_seconds: int = 7200):
    return [InitProcessGroupKwargs(timeout=timedelta(seconds=int(timeout_seconds)))]


def build_ovggt_kwargs_for_eval(args):
    if args.ovggt_mode == "legacy":
        return {"mode": "legacy"}

    from ovggt.utils.frontend_cache import FrontendCacheConfig
    from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig

    frontend_keyframe_strategy = getattr(args, "frontend_keyframe_strategy", "fixed_interval")
    return {
        "mode": "frontend_eval",
        "frontend_pose_encoding_type": getattr(
            args,
            "frontend_pose_encoding_type",
            "absT_quaR_FoV",
        ),
        "frontend_cache_config": FrontendCacheConfig(
            enabled=True,
            dedup_enabled=args.frontend_dedup_enabled,
            voxel_size=getattr(args, "frontend_voxel_size", 0.1),
            dedup_policy=getattr(args, "frontend_dedup_policy", "hard"),
            dedup_budget_trigger_ratio=getattr(args, "frontend_dedup_budget_trigger_ratio", 0.9),
            dedup_topk_per_voxel=getattr(args, "frontend_dedup_topk_per_voxel", 3),
            dedup_replacement_margin=getattr(args, "frontend_dedup_replacement_margin", 0.05),
            dedup_age_decay=getattr(args, "frontend_dedup_age_decay", 0.02),
            fifo_keep_topk=getattr(args, "frontend_fifo_keep_topk", 80),
            budget_allocation=getattr(args, "frontend_budget_allocation", "uniform"),
            fifo_protected_ring_ratio=getattr(args, "frontend_fifo_protected_ring_ratio", 0.2),
        ),
        "keyframe_switch_config": KeyframeSwitchConfig(
            strategy=frontend_keyframe_strategy,
            interval=args.frontend_anchor_interval,
            coverage_threshold=getattr(args, "frontend_coverage_threshold", 0.2),
            max_history_anchors=getattr(args, "frontend_max_anchors", 3),
            coverage_monitor_only=(frontend_keyframe_strategy != "coverage"),
        ),
    }


def build_ovggt_inference_kwargs_for_eval(args):
    if getattr(args, "ovggt_mode", "legacy") != "frontend_eval":
        return {}
    return {
        "window_protect_frames": getattr(args, "frontend_window_protect_frames", 0),
        "anchor_keep_ratio": getattr(args, "frontend_anchor_keep_ratio", 0.05),
        "max_anchors": getattr(args, "frontend_max_anchors", 3),
        "coverage_threshold": getattr(args, "frontend_coverage_threshold", 0.2),
    }


def build_7scenes_kwargs(data_root: str, resolution, max_frames: int):
    return {
        "split": "test",
        "ROOT": resolve_7scenes_root(data_root),
        "resolution": resolution,
        "num_seq": 1,
        "full_video": True,
        "kf_every": 2,
        "max_frames": max_frames,
    }


def append_eval_log_line(log_file: str, line: str) -> None:
    with open(log_file, "a") as f:
        print(line, file=f)


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--ovggt_mode",
        type=str,
        default="legacy",
        choices=("legacy", "frontend_eval"),
    )
    parser.add_argument(
        "--frontend_dedup_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--frontend_keyframe_strategy",
        type=str,
        default="fixed_interval",
        choices=("fixed_interval", "coverage"),
    )
    parser.add_argument("--frontend_anchor_interval", type=int, default=8)
    parser.add_argument(
        "--frontend_pose_encoding_type",
        type=str,
        default="absT_quaR_FoV",
        choices=("absT_quaR_FoV", "relT_quaR_FoV"),
    )
    parser.add_argument(
        "--frontend_dedup_policy",
        type=str,
        default="hard",
        choices=("hard", "soft_reservoir", "pressure_only"),
    )
    parser.add_argument("--frontend_voxel_size", type=float, default=0.1)
    parser.add_argument("--frontend_dedup_budget_trigger_ratio", type=float, default=0.9)
    parser.add_argument("--frontend_dedup_topk_per_voxel", type=int, default=3)
    parser.add_argument("--frontend_dedup_replacement_margin", type=float, default=0.05)
    parser.add_argument("--frontend_dedup_age_decay", type=float, default=0.02)
    parser.add_argument("--frontend_window_protect_frames", type=int, default=0)
    parser.add_argument("--frontend_anchor_keep_ratio", type=float, default=0.05)
    parser.add_argument("--frontend_max_anchors", type=int, default=3)
    parser.add_argument("--frontend_coverage_threshold", type=float, default=0.2)
    parser.add_argument("--frontend_fifo_keep_topk", type=int, default=80)
    parser.add_argument(
        "--frontend_budget_allocation",
        type=str,
        default="uniform",
        choices=("uniform", "dynamic"),
    )
    parser.add_argument("--frontend_fifo_protected_ring_ratio", type=float, default=0.2)
    return parser


def main(args):
    validate_model_mode(args.model_name, args.ovggt_mode)
    seven_scenes_root = resolve_7scenes_root(args.data_root)
    if args.weights and not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if seven_scenes_root and not os.path.exists(seven_scenes_root):
        raise FileNotFoundError(f"7-Scenes root not found: {seven_scenes_root}")

    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
        # resolution = (518, 336)
    else:
        raise NotImplementedError
    datasets_all = {
        "7scenes": SevenScenes(**build_7scenes_kwargs(seven_scenes_root, resolution, args.max_frames)),
        # "NRGBD": NRGBD(
        #     split="test",
        #     ROOT="./data/neural_rgbd_data",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=500,
        # ),
    }

    accelerator = Accelerator(kwargs_handlers=build_accelerator_kwargs_handlers())
    device = accelerator.device
    model_name = args.model_name
    if model_name == "OVGGT":
        from ovggt.models.ovggt import OVGGT
        from ovggt.utils.pose_enc import pose_encoding_to_extri_intri
        from ovggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = OVGGT(**build_ovggt_kwargs_for_eval(args))
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to(device)
    elif model_name == "VGGT":
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy
        model = VGGT()
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to(device)

    else:
        raise NotImplementedError
    del ckpt
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []
                    in_camera1 = None  

                    if should_run_reconstruction_eval(model_name):
                        revisit = args.revisit
                        update = not args.freeze
                        if revisit > 1:
                            # repeat input for 'revisit' times
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views
                        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                        with torch.cuda.amp.autocast(dtype=dtype):
                            if isinstance(batch, dict) and "img" in batch:
                                batch["img"] = (batch["img"] + 1.0) / 2.0
                            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                                for view in batch:
                                    view["img"] = (view["img"] + 1.0) / 2.0

                        with torch.cuda.amp.autocast(dtype=dtype):
                            with torch.no_grad():
                                results = model.inference(batch, **build_ovggt_inference_kwargs_for_eval(args))

                            preds, batch = results.ress, results.views 

                            if args.use_proj:
                                pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
                                depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
                                depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
                                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                                                    batch[0]["img"].shape[-2:])

                                if "DTU" in name_data:
                                    depth_map = depth_map * 1000.0
                                    extrinsic[..., :3, 3] *= 1000.0

                                point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                                                extrinsic.squeeze(0),
                                                                                                intrinsic.squeeze(0))
                            valid_length = len(preds) // args.revisit
                            if args.revisit > 1:
                                preds = preds[-valid_length:]
                                batch = batch[-valid_length:]
                                

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            if args.use_proj:
                                pts = point_map_by_unprojection[j]
                                conf = depth_conf[0, j].cpu().data.numpy()
                            else:
                                pts = pred_pts[j].cpu().numpy()[0]
                                conf = preds[j]["conf"].cpu().data.numpy()[0]

                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]

                            # Align predicted 3D points to the ground truth
                            # pts = geotrf(in_camera1, pts)
                            # pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append(image[None, ...])
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        save_params,
                    )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    pts_all_masked, pts_gt_all_masked, images_all_masked = filter_finite_point_pairs(
                        pts_all_masked,
                        pts_gt_all_masked,
                        images_all_masked,
                    )

                    if args.use_proj:
                        def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
                            assert src.shape == dst.shape
                            N, dim = src.shape

                            mu_src = src.mean(axis=0)
                            mu_dst = dst.mean(axis=0)
                            src_c = src - mu_src
                            dst_c = dst - mu_dst

                            Sigma = dst_c.T @ src_c / N  # (3,3)

                            U, D, Vt = np.linalg.svd(Sigma) 

                            S = np.eye(dim)
                            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                                S[-1, -1] = -1

                            R = U @ S @ Vt

                            if with_scale:
                                var_src = (src_c ** 2).sum() / N
                                s = (D * S.diagonal()).sum() / var_src
                            else:
                                s = 1.0

                            t = mu_dst - s * R @ mu_src

                            return s, R, t

                        pts_all_masked = pts_all_masked.reshape(-1, 3)
                        pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                        s, R, t = umeyama_alignment(pts_all_masked, pts_gt_all_masked, with_scale=True)
                        pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                        pts_all_masked = pts_all_aligned

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                        ),
                        pcd,
                    )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                        pcd_gt,
                    )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)

                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask_align.ply"
                        ),
                        pcd,
                    )

                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    log_line = (
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                        f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(log_line)
                    append_eval_log_line(log_file, log_line)

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                write_merged_eval_log(
                    save_path,
                    expected_scene_ids=expected_scene_ids_for_dataset(dataset),
                    num_processes=accelerator.num_processes,
                )



from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


def expected_scene_ids_for_dataset(dataset) -> list[str] | None:
    scene_list = getattr(dataset, "scene_list", None)
    num_seq = int(getattr(dataset, "num_seq", 1))
    if scene_list is None or num_seq <= 0:
        return None
    return [scene_list[idx // num_seq] for idx in range(len(dataset))]


def write_merged_eval_log(
    save_path: str,
    expected_scene_ids: list[str] | None = None,
    num_processes: int = 8,
) -> dict:
    to_write = ""
    for i in range(int(num_processes)):
        log_path = osp.join(save_path, f"logs_{i}.txt")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r") as f_sub:
            to_write += f_sub.read()

    metrics = defaultdict(list)
    seen_scene_ids = []
    for line in to_write.strip().split("\n"):
        match = regex.match(line)
        if not match:
            continue
        data = match.groupdict()
        seen_scene_ids.append(data["scene_id"])
        for key, value in data.items():
            if key != "scene_id":
                metrics[key].append(float(value))
        metrics["nc"].append((float(data["nc1"]) + float(data["nc2"])) / 2)
        metrics["nc_med"].append((float(data["nc1_med"]) + float(data["nc2_med"])) / 2)

    if expected_scene_ids is not None:
        expected = set(expected_scene_ids)
        seen = set(seen_scene_ids)
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"Missing metrics for {missing}")
            if extra:
                parts.append(f"Unexpected metrics for {extra}")
            raise RuntimeError("; ".join(parts))

    if not seen_scene_ids:
        raise RuntimeError(f"No evaluation metrics found in {save_path}")

    mean_metrics = {
        metric: sum(values) / len(values)
        for metric, values in metrics.items()
    }

    c_name = "mean"
    print_str = f"{c_name.ljust(20)}: "
    for m_name in mean_metrics:
        print_num = np.mean(mean_metrics[m_name])
        print_str = print_str + f"{m_name}: {print_num:.3f} | "
    print_str = print_str + "\n"
    with open(osp.join(save_path, "logs_all.txt"), "w") as f:
        f.write(to_write + print_str)
    return mean_metrics


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
