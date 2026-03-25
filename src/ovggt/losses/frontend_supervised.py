from typing import Sequence

import torch

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from ovggt.losses.frontend_distill import FrontendDistillLoss


class FrontendSupervisedLoss(FrontendDistillLoss):
    def __init__(
        self,
        keyframe_interval: int = 8,
        forced_keyframe_frames=(0,),
        abs_pose_loss_weight: float = 20.0,
        abs_pose_consistency_weight: float = 5.0,
        depth_loss_weight: float = 20.0,
        pmap_loss_weight: float = 10.0,
        track_loss_weight: float = 0.5,
    ):
        super().__init__(
            keyframe_interval=keyframe_interval,
            forced_keyframe_frames=forced_keyframe_frames,
            abs_pose_loss_weight=abs_pose_loss_weight,
            abs_pose_consistency_weight=abs_pose_consistency_weight,
        )
        self.depth_loss_weight = float(depth_loss_weight)
        self.pmap_loss_weight = float(pmap_loss_weight)
        self.track_loss_weight = float(track_loss_weight)

    def forward(
        self,
        raw_batch_gt: Sequence[dict],
        student_outputs,
        keyframe_schedule: Sequence[object],
    ):
        student_preds = student_outputs.ress
        assert student_preds is not None
        assert len(student_preds) == len(raw_batch_gt)

        image_size_hw = self._infer_image_hw(raw_batch_gt)
        gt_abs_pose_enc = self._build_gt_absolute_pose_encoding(raw_batch_gt, image_size_hw)
        num_frames = gt_abs_pose_enc.shape[1]
        fixed_keyframe_mask = self._build_keyframe_mask(
            num_frames=num_frames,
            device=gt_abs_pose_enc.device,
            keyframe_schedule=keyframe_schedule,
        )
        gt_rel_pose_enc = self._build_gt_relative_pose_targets(
            gt_abs_pose_enc,
            fixed_keyframe_mask,
            image_size_hw,
        )

        cam_pr = torch.stack([pred["camera_pose_rel"] for pred in student_preds], dim=1)
        Lcamera_rel = self.cam_loss(cam_pr, gt_rel_pose_enc)
        cam_abs_pr = torch.stack([pred["camera_pose"] for pred in student_preds], dim=1)
        Lcamera_abs = self._masked_camera_loss(cam_abs_pr, gt_abs_pose_enc, fixed_keyframe_mask)
        non_keyframe_mask = ~fixed_keyframe_mask
        abs_pose_consistency_target = self._compose_absolute_pose_from_relative_with_fixed_anchors(
            anchor_abs_pose=gt_abs_pose_enc,
            pred_rel_pose=cam_pr,
            keyframe_mask=fixed_keyframe_mask,
            image_size_hw=image_size_hw,
        )
        Lcamera_abs_consistency = self._masked_camera_loss(
            cam_abs_pr,
            abs_pose_consistency_target,
            non_keyframe_mask,
        )

        depth_terms = []
        for batch_gt, student_pred in zip(raw_batch_gt, student_preds):
            gt_depth, gt_valid_mask = self._build_depth_targets(batch_gt, student_pred["depth"])
            depth_terms.append(
                self.depth_loss(
                    student_pred["depth"],
                    gt_depth,
                    student_pred["depth_conf"],
                    torch.ones_like(student_pred["depth_conf"]),
                    gt_valid_mask,
                )
            )
        Ldepth = torch.stack(depth_terms).mean() if depth_terms else Lcamera_rel.new_zeros(())

        pmap_terms = []
        for batch_gt, student_pred in zip(raw_batch_gt, student_preds):
            gt_pts3d, gt_valid_mask = self._build_point_targets(batch_gt, student_pred["pts3d_in_other_view"])
            pmap_terms.append(
                self.pmap_loss(
                    student_pred["pts3d_in_other_view"],
                    gt_pts3d,
                    student_pred["conf"],
                    torch.ones_like(student_pred["conf"]),
                    gt_valid_mask,
                )
            )
        Lpmap = torch.stack(pmap_terms).mean() if pmap_terms else Lcamera_rel.new_zeros(())

        if self._has_track_targets(raw_batch_gt, student_preds):
            y_gt = torch.stack([self._get_track_tensor(view_gt, "track") for view_gt in raw_batch_gt], dim=1)
            vis_gt = torch.stack([self._get_track_tensor(view_gt, "vis") for view_gt in raw_batch_gt], dim=1)
            y_pr = torch.stack([pred["track"] for pred in student_preds], dim=1)
            vis_pr = torch.stack([pred["vis"] for pred in student_preds], dim=1)
            w_p = torch.stack([pred["track_conf"] for pred in student_preds], dim=1)
            w_g = torch.stack([self._get_track_tensor(view_gt, "track_conf") for view_gt in raw_batch_gt], dim=1)
            Ltrack = self.track_loss(y_pr, y_gt, vis_pr, vis_gt, w_p, w_g)
        else:
            Ltrack = Lcamera_rel.new_zeros(())

        total = (
            20.0 * Lcamera_rel
            + self.abs_pose_loss_weight * Lcamera_abs
            + self.abs_pose_consistency_weight * Lcamera_abs_consistency
            + self.depth_loss_weight * Ldepth
            + self.pmap_loss_weight * Lpmap
            + self.track_loss_weight * Ltrack
        )

        details = {
            "Lcamera_rel": float(Lcamera_rel) * 20.0,
            "Lcamera_abs": float(Lcamera_abs) * self.abs_pose_loss_weight,
            "Lcamera_abs_consistency": float(Lcamera_abs_consistency) * self.abs_pose_consistency_weight,
            "Ldepth": float(Ldepth) * self.depth_loss_weight,
            "Lpmap": float(Lpmap) * self.pmap_loss_weight,
            "Ltrack": float(Ltrack) * self.track_loss_weight,
            "total": float(total),
            "num_keyframes": float(fixed_keyframe_mask.sum().item()),
        }
        return total, details

    @staticmethod
    def _to_tensor(value, device: torch.device, dtype: torch.dtype = None) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _build_depth_targets(self, batch_gt: dict, reference_tensor: torch.Tensor):
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        depth = batch_gt.get("depthmap", batch_gt.get("depth"))
        if depth is None:
            raise KeyError("Frontend supervised training expects depthmap or depth in the batch.")
        gt_depth = self._to_tensor(depth, device=device, dtype=dtype)
        if gt_depth.dim() == 2:
            gt_depth = gt_depth.unsqueeze(0).unsqueeze(-1)
        elif gt_depth.dim() == 3:
            if gt_depth.shape[-1] == 1:
                gt_depth = gt_depth.unsqueeze(0)
            else:
                gt_depth = gt_depth.unsqueeze(-1)

        if "valid_mask" in batch_gt:
            valid_mask = self._to_tensor(batch_gt["valid_mask"], device=device).bool()
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(0)
            elif valid_mask.dim() == 3 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
        else:
            valid_mask = torch.isfinite(gt_depth[..., 0]) & (gt_depth[..., 0] > 0)

        return gt_depth, valid_mask

    def _build_point_targets(self, batch_gt: dict, reference_tensor: torch.Tensor):
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        pts3d = batch_gt.get("pts3d")
        if pts3d is not None:
            gt_pts3d = self._to_tensor(pts3d, device=device, dtype=dtype)
            if gt_pts3d.dim() == 3:
                gt_pts3d = gt_pts3d.unsqueeze(0)
        else:
            depth = batch_gt.get("depthmap", batch_gt.get("depth"))
            if depth is None:
                raise KeyError("Frontend supervised training expects pts3d or depthmap in the batch.")
            depth_np = self._to_tensor(depth, device="cpu").detach().cpu().numpy()
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            camera_intrinsics = self._to_tensor(batch_gt["camera_intrinsics"], device="cpu").detach().cpu().numpy()
            camera_pose = self._to_tensor(batch_gt["camera_pose"], device="cpu").detach().cpu().numpy()
            if camera_intrinsics.ndim == 3:
                camera_intrinsics = camera_intrinsics[0]
            if camera_pose.ndim == 3:
                camera_pose = camera_pose[0]
            gt_pts3d_np, valid_mask_np = depthmap_to_absolute_camera_coordinates(
                depth_np,
                camera_intrinsics,
                camera_pose,
            )
            gt_pts3d = self._to_tensor(gt_pts3d_np, device=device, dtype=dtype).unsqueeze(0)
            valid_mask = self._to_tensor(valid_mask_np, device=device).bool().unsqueeze(0)
            return gt_pts3d, valid_mask

        if "valid_mask" in batch_gt:
            valid_mask = self._to_tensor(batch_gt["valid_mask"], device=device).bool()
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(0)
            elif valid_mask.dim() == 3 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
        else:
            valid_mask = torch.isfinite(gt_pts3d).all(dim=-1)

        return gt_pts3d, valid_mask

    @staticmethod
    def _get_track_tensor(view_gt: dict, key: str) -> torch.Tensor:
        if key not in view_gt:
            raise KeyError(key)
        return torch.as_tensor(view_gt[key])

    @staticmethod
    def _has_track_targets(raw_batch_gt: Sequence[dict], student_preds: Sequence[dict]) -> bool:
        track_keys = {"track", "vis", "track_conf"}
        return all(track_keys.issubset(view.keys()) for view in raw_batch_gt) and all(
            track_keys.issubset(pred.keys()) for pred in student_preds
        )
