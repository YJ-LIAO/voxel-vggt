from typing import List, Sequence

import torch
import torch.nn as nn

from dust3r.losses import CameraLoss, DepthOrPmapLoss, TrackLoss
from ovggt.utils.geometry import closed_form_inverse_se3
from ovggt.utils.pose_enc import (
    ABS_POSE_ENCODING,
    compose_absolute_from_relative,
    relative_from_absolute_pose_encoding,
    world_to_camera_to_pose_encoding,
)


class FrontendDistillLoss(nn.Module):
    def __init__(
        self,
        keyframe_interval: int = 8,
        forced_keyframe_frames=(0,),
        abs_pose_loss_weight: float = 20.0,
        abs_pose_consistency_weight: float = 5.0,
    ):
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)
        self.pmap_loss = DepthOrPmapLoss(alpha=0.1)
        self.track_loss = TrackLoss()
        self.abs_pose_loss_weight = float(abs_pose_loss_weight)
        self.abs_pose_consistency_weight = float(abs_pose_consistency_weight)
        self.keyframe_interval = max(int(keyframe_interval), 1)
        self.forced_keyframe_frames = tuple(int(x) for x in forced_keyframe_frames)

    def forward(
        self,
        raw_batch_gt: Sequence[dict],
        teacher_outputs,
        student_outputs,
        keyframe_schedule: Sequence[object],
    ):
        teacher_preds = teacher_outputs.ress
        student_preds = student_outputs.ress
        assert teacher_preds is not None and student_preds is not None
        assert len(teacher_preds) == len(student_preds) == len(raw_batch_gt)

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
        for batch_gt, teacher_pred, student_pred in zip(raw_batch_gt, teacher_preds, student_preds):
            device = student_pred["depth"].device
            teacher_depth = self._tensor_to_device(teacher_pred["depth"], device)
            teacher_depth_conf = self._tensor_to_device(teacher_pred["depth_conf"], device)
            valid_mask = self._get_valid_mask(batch_gt, teacher_depth)
            valid_mask = self._tensor_to_device(valid_mask, device)
            depth_terms.append(
                self.depth_loss(
                    student_pred["depth"],
                    teacher_depth,
                    student_pred["depth_conf"],
                    teacher_depth_conf,
                    valid_mask,
                )
            )
        Ldepth = torch.stack(depth_terms).mean() if depth_terms else Lcamera_rel.new_zeros(())

        pmap_terms = []
        for batch_gt, teacher_pred, student_pred in zip(raw_batch_gt, teacher_preds, student_preds):
            device = student_pred["pts3d_in_other_view"].device
            teacher_pmap = self._tensor_to_device(teacher_pred["pts3d_in_other_view"], device)
            teacher_conf = self._tensor_to_device(teacher_pred["conf"], device)
            valid_mask = self._get_valid_mask(batch_gt, teacher_pmap[..., 0])
            valid_mask = self._tensor_to_device(valid_mask, device)
            pmap_terms.append(
                self.pmap_loss(
                    student_pred["pts3d_in_other_view"],
                    teacher_pmap,
                    student_pred["conf"],
                    teacher_conf,
                    valid_mask,
                )
            )
        Lpmap = torch.stack(pmap_terms).mean() if pmap_terms else Lcamera_rel.new_zeros(())

        if (
            teacher_preds
            and "track" in teacher_preds[0]
            and "track" in student_preds[0]
        ):
            track_device = student_preds[0]["track"].device
            y_gt = torch.stack([pred["track"] for pred in teacher_preds], dim=1)
            vis_gt = torch.stack([pred["vis"] for pred in teacher_preds], dim=1)
            y_gt = self._tensor_to_device(y_gt, track_device)
            vis_gt = self._tensor_to_device(vis_gt, track_device)
            y_pr = torch.stack([pred["track"] for pred in student_preds], dim=1)
            vis_pr = torch.stack([pred["vis"] for pred in student_preds], dim=1)
            w_p = torch.stack([pred["track_conf"] for pred in student_preds], dim=1)
            w_g = torch.stack([pred["track_conf"] for pred in teacher_preds], dim=1)
            w_g = self._tensor_to_device(w_g, track_device)
            Ltrack = self.track_loss(y_pr, y_gt, vis_pr, vis_gt, w_p, w_g)
        else:
            Ltrack = Lcamera_rel.new_zeros(())

        total = (
            20.0 * Lcamera_rel
            + self.abs_pose_loss_weight * Lcamera_abs
            + self.abs_pose_consistency_weight * Lcamera_abs_consistency
            + 20.0 * Ldepth
            + 10.0 * Lpmap
            + 0.5 * Ltrack
        )

        details = {
            "Lcamera_rel": float(Lcamera_rel) * 20.0,
            "Lcamera_abs": float(Lcamera_abs) * self.abs_pose_loss_weight,
            "Lcamera_abs_consistency": float(Lcamera_abs_consistency) * self.abs_pose_consistency_weight,
            "Ldepth": float(Ldepth) * 20.0,
            "Lpmap": float(Lpmap) * 10.0,
            "Ltrack": float(Ltrack) * 0.5,
            "total": float(total),
            "num_keyframes": float(fixed_keyframe_mask.sum().item()),
        }
        return total, details

    def compute_depth_term(self, batch_gt: dict, teacher_pred: dict, student_pred: dict) -> torch.Tensor:
        staged = self._stage_teacher_dense_targets(batch_gt, teacher_pred, student_pred)
        return self.depth_loss(
            student_pred["depth"],
            staged["teacher_depth"],
            student_pred["depth_conf"],
            staged["teacher_depth_conf"],
            staged["valid_mask"],
        )

    def compute_pmap_term(self, batch_gt: dict, teacher_pred: dict, student_pred: dict) -> torch.Tensor:
        staged = self._stage_teacher_dense_targets(batch_gt, teacher_pred, student_pred)
        return self.pmap_loss(
            student_pred["pts3d_in_other_view"],
            staged["teacher_pmap"],
            student_pred["conf"],
            staged["teacher_conf"],
            staged["valid_mask"],
        )

    def compute_depth_and_pmap_terms(self, batch_gt: dict, teacher_pred: dict, student_pred: dict):
        staged = self._stage_teacher_dense_targets(batch_gt, teacher_pred, student_pred)
        depth_term = self.depth_loss(
            student_pred["depth"],
            staged["teacher_depth"],
            student_pred["depth_conf"],
            staged["teacher_depth_conf"],
            staged["valid_mask"],
        )
        pmap_term = self.pmap_loss(
            student_pred["pts3d_in_other_view"],
            staged["teacher_pmap"],
            student_pred["conf"],
            staged["teacher_conf"],
            staged["valid_mask"],
        )
        return depth_term, pmap_term

    def compute_track_term(
        self,
        teacher_track: Sequence[torch.Tensor],
        teacher_vis: Sequence[torch.Tensor],
        teacher_track_conf: Sequence[torch.Tensor],
        student_track: Sequence[torch.Tensor],
        student_vis: Sequence[torch.Tensor],
        student_track_conf: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not student_track:
            ref_device = teacher_track[0].device if teacher_track else torch.device("cpu")
            return torch.zeros((), device=ref_device)
        track_device = student_track[0].device
        y_gt = self._tensor_to_device(torch.stack(list(teacher_track), dim=1), track_device)
        vis_gt = self._tensor_to_device(torch.stack(list(teacher_vis), dim=1), track_device)
        y_pr = torch.stack(list(student_track), dim=1)
        vis_pr = torch.stack(list(student_vis), dim=1)
        w_p = torch.stack(list(student_track_conf), dim=1)
        w_g = self._tensor_to_device(torch.stack(list(teacher_track_conf), dim=1), track_device)
        return self.track_loss(y_pr, y_gt, vis_pr, vis_gt, w_p, w_g)

    def finalize_from_stream(
        self,
        raw_batch_gt: Sequence[dict],
        teacher_outputs,
        student_camera_pose_rel: Sequence[torch.Tensor],
        student_camera_pose_abs: Sequence[torch.Tensor],
        keyframe_schedule: Sequence[object],
        depth_terms: Sequence[torch.Tensor],
        pmap_terms: Sequence[torch.Tensor],
        student_track: Sequence[torch.Tensor] | None = None,
        student_vis: Sequence[torch.Tensor] | None = None,
        student_track_conf: Sequence[torch.Tensor] | None = None,
    ):
        if not student_camera_pose_rel or not student_camera_pose_abs:
            raise RuntimeError("Missing student camera predictions for frontend distillation.")

        image_size_hw = self._infer_image_hw(raw_batch_gt)
        device = student_camera_pose_abs[0].device
        gt_abs_pose_enc = self._build_gt_absolute_pose_encoding(raw_batch_gt, image_size_hw).to(device)
        num_frames = gt_abs_pose_enc.shape[1]
        fixed_keyframe_mask = self._build_keyframe_mask(
            num_frames=num_frames,
            device=device,
            keyframe_schedule=keyframe_schedule,
        )
        gt_rel_pose_enc = self._build_gt_relative_pose_targets(
            gt_abs_pose_enc,
            fixed_keyframe_mask,
            image_size_hw,
        )

        cam_pr = torch.stack(list(student_camera_pose_rel), dim=1)
        Lcamera_rel = self.cam_loss(cam_pr, gt_rel_pose_enc)
        cam_abs_pr = torch.stack(list(student_camera_pose_abs), dim=1)
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

        Ldepth = torch.stack(list(depth_terms)).mean() if depth_terms else Lcamera_rel.new_zeros(())
        Lpmap = torch.stack(list(pmap_terms)).mean() if pmap_terms else Lcamera_rel.new_zeros(())

        teacher_preds = teacher_outputs.ress
        if (
            teacher_preds
            and "track" in teacher_preds[0]
            and student_track
            and student_vis
            and student_track_conf
        ):
            Ltrack = self.compute_track_term(
                teacher_track=[pred["track"] for pred in teacher_preds],
                teacher_vis=[pred["vis"] for pred in teacher_preds],
                teacher_track_conf=[pred["track_conf"] for pred in teacher_preds],
                student_track=student_track,
                student_vis=student_vis,
                student_track_conf=student_track_conf,
            )
        else:
            Ltrack = Lcamera_rel.new_zeros(())

        total = (
            20.0 * Lcamera_rel
            + self.abs_pose_loss_weight * Lcamera_abs
            + self.abs_pose_consistency_weight * Lcamera_abs_consistency
            + 20.0 * Ldepth
            + 10.0 * Lpmap
            + 0.5 * Ltrack
        )
        details = {
            "Lcamera_rel": float(Lcamera_rel) * 20.0,
            "Lcamera_abs": float(Lcamera_abs) * self.abs_pose_loss_weight,
            "Lcamera_abs_consistency": float(Lcamera_abs_consistency) * self.abs_pose_consistency_weight,
            "Ldepth": float(Ldepth) * 20.0,
            "Lpmap": float(Lpmap) * 10.0,
            "Ltrack": float(Ltrack) * 0.5,
            "total": float(total),
            "num_keyframes": float(fixed_keyframe_mask.sum().item()),
        }
        return total, details

    @staticmethod
    def _infer_image_hw(raw_batch_gt: Sequence[dict]):
        sample_img = raw_batch_gt[0]["img"]
        if sample_img.dim() == 4:
            return sample_img.shape[-2], sample_img.shape[-1]
        return sample_img.shape[-3], sample_img.shape[-2]

    @staticmethod
    def _get_valid_mask(batch_gt: dict, reference_tensor: torch.Tensor) -> torch.Tensor:
        if "valid_mask" in batch_gt:
            return batch_gt["valid_mask"].bool()
        return torch.ones_like(reference_tensor[..., 0] if reference_tensor.dim() == 4 else reference_tensor, dtype=torch.bool)

    @staticmethod
    def _tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return tensor.to(device=device, non_blocking=True)

    def _stage_teacher_dense_targets(self, batch_gt: dict, teacher_pred: dict, student_pred: dict) -> dict:
        device = student_pred["depth"].device
        teacher_depth = self._tensor_to_device(teacher_pred["depth"], device)
        teacher_depth_conf = self._tensor_to_device(teacher_pred["depth_conf"], device)
        teacher_pmap = self._tensor_to_device(teacher_pred["pts3d_in_other_view"], device)
        teacher_conf = self._tensor_to_device(teacher_pred["conf"], device)
        valid_mask = self._tensor_to_device(self._get_valid_mask(batch_gt, teacher_depth), device)
        return {
            "teacher_depth": teacher_depth,
            "teacher_depth_conf": teacher_depth_conf,
            "teacher_pmap": teacher_pmap,
            "teacher_conf": teacher_conf,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _build_gt_absolute_pose_encoding(raw_batch_gt: Sequence[dict], image_size_hw) -> torch.Tensor:
        gt_c2w = torch.stack([view["camera_pose"] for view in raw_batch_gt], dim=1).float()
        gt_w2c = closed_form_inverse_se3(gt_c2w.reshape(-1, 4, 4)).reshape_as(gt_c2w)
        gt_intrinsics = torch.stack([view["camera_intrinsics"] for view in raw_batch_gt], dim=1).float()
        return world_to_camera_to_pose_encoding(
            gt_w2c,
            intrinsics=gt_intrinsics,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
        )

    @staticmethod
    def _build_gt_relative_pose_targets(
        gt_abs_pose_enc: torch.Tensor,
        keyframe_mask: torch.Tensor,
        image_size_hw,
    ) -> torch.Tensor:
        batch_size, num_frames = gt_abs_pose_enc.shape[:2]
        if keyframe_mask.ndim != 1 or keyframe_mask.shape[0] != num_frames:
            raise ValueError(
                f"Expected keyframe_mask shape ({num_frames},), got {tuple(keyframe_mask.shape)}"
            )

        active_frame_idx = 0
        targets: List[torch.Tensor] = []
        for frame_idx in range(num_frames):
            if bool(keyframe_mask[frame_idx].item()):
                active_frame_idx = frame_idx
            anchor_pose = gt_abs_pose_enc[:, active_frame_idx : active_frame_idx + 1]
            current_pose = gt_abs_pose_enc[:, frame_idx : frame_idx + 1]
            rel_target = relative_from_absolute_pose_encoding(
                anchor_pose,
                current_pose,
                image_size_hw=image_size_hw,
            )
            targets.append(rel_target[:, 0])
        return torch.stack(targets, dim=1)

    def _build_fixed_keyframe_mask(self, num_frames: int, device: torch.device) -> torch.Tensor:
        if num_frames <= 0:
            return torch.zeros(0, device=device, dtype=torch.bool)
        mask = torch.zeros(num_frames, device=device, dtype=torch.bool)
        mask[0] = True
        mask[:: self.keyframe_interval] = True
        for frame_idx in self.forced_keyframe_frames:
            if 0 <= frame_idx < num_frames:
                mask[frame_idx] = True
        return mask

    def _build_keyframe_mask(
        self,
        num_frames: int,
        device: torch.device,
        keyframe_schedule: Sequence[object] = None,
    ) -> torch.Tensor:
        if not keyframe_schedule:
            return self._build_fixed_keyframe_mask(num_frames=num_frames, device=device)

        mask = torch.zeros(num_frames, device=device, dtype=torch.bool)
        for frame_idx, event in enumerate(keyframe_schedule[:num_frames]):
            event_type = ""
            anchor_slot = -1
            if isinstance(event, dict):
                event_type = str(event.get("event_type", ""))
                anchor_slot = int(event.get("anchor_slot", -1))
            else:
                event_type = str(getattr(event, "event_type", ""))
                anchor_slot = int(getattr(event, "anchor_slot", -1))
            if anchor_slot >= 0 and "NOOP" not in event_type:
                mask[frame_idx] = True

        mask[0] = True
        if not torch.any(mask):
            return self._build_fixed_keyframe_mask(num_frames=num_frames, device=device)
        return mask

    @staticmethod
    def _compose_absolute_pose_from_relative_with_fixed_anchors(
        anchor_abs_pose: torch.Tensor,
        pred_rel_pose: torch.Tensor,
        keyframe_mask: torch.Tensor,
        image_size_hw,
    ) -> torch.Tensor:
        if anchor_abs_pose.shape != pred_rel_pose.shape:
            raise ValueError(
                f"Expected anchor_abs_pose and pred_rel_pose to have same shape, got "
                f"{tuple(anchor_abs_pose.shape)} vs {tuple(pred_rel_pose.shape)}"
            )
        batch_size, num_frames = anchor_abs_pose.shape[:2]
        if keyframe_mask.ndim != 1 or keyframe_mask.shape[0] != num_frames:
            raise ValueError(
                f"Expected keyframe_mask shape ({num_frames},), got {tuple(keyframe_mask.shape)}"
            )

        active_frame_idx = 0
        composed_abs = []
        for frame_idx in range(num_frames):
            if bool(keyframe_mask[frame_idx].item()):
                active_frame_idx = frame_idx
            anchor_pose = anchor_abs_pose[:, active_frame_idx : active_frame_idx + 1]
            rel_pose = pred_rel_pose[:, frame_idx : frame_idx + 1]
            abs_pose = compose_absolute_from_relative(
                anchor_pose,
                rel_pose,
                image_size_hw=image_size_hw,
            )
            composed_abs.append(abs_pose[:, 0])
        return torch.stack(composed_abs, dim=1)

    def _masked_camera_loss(
        self,
        pred_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if frame_mask.ndim != 1:
            raise ValueError(f"Expected 1D frame mask, got shape={tuple(frame_mask.shape)}")
        if not torch.any(frame_mask):
            return pred_pose.new_zeros(())
        pred_selected = pred_pose[:, frame_mask]
        gt_selected = gt_pose[:, frame_mask]
        return self.cam_loss(pred_selected, gt_selected)
