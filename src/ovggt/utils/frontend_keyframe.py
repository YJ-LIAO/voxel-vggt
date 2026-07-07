from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch

from .geometry import closed_form_inverse_se3
from .history_anchor import compute_coverage
from .pose_enc import (
    ABS_POSE_ENCODING,
    pose_encoding_to_camera_to_world,
    pose_encoding_to_world_to_camera,
)


@dataclass
class KeyframeSwitchConfig:
    strategy: str = "fixed_interval"
    coverage_threshold: float = 0.2
    sample_ratio: float = 0.1
    max_history_anchors: int = 3
    interval: int = 48
    translation_threshold: Optional[float] = None
    rotation_threshold_deg: Optional[float] = None
    coverage_monitor_only: bool = True
    forced_keyframe_frames: Sequence[int] = field(default_factory=tuple)


class KeyframeEventType(str, Enum):
    NOOP = "NOOP"
    PROMOTE_KEYFRAME = "PROMOTE_KEYFRAME"
    FIFO_SWAP = "FIFO_SWAP"


@dataclass
class KeyframeEvent:
    event_type: KeyframeEventType
    frame_idx: int
    keyframe_id: int
    anchor_slot: int
    reorder_indices: Optional[torch.Tensor] = None
    demoted_slot: Optional[int] = None
    num_anchor_frames: int = 0
    slot_pose_updates: Optional[Dict[int, torch.Tensor]] = None
    local_to_world: Optional[torch.Tensor] = None
    active_pose_encoding: Optional[torch.Tensor] = None


@dataclass
class KeyframePacket:
    frame_idx: int
    keyframe_id: int
    anchor_slot: int
    pose_abs: torch.Tensor
    local_to_world: torch.Tensor
    patch_local_xyz: torch.Tensor
    patch_depth_conf: torch.Tensor
    patch_features: torch.Tensor


class FrontendKeyframeManager:
    def __init__(self, config: KeyframeSwitchConfig):
        self.config = config
        self.initialized = False
        self.next_keyframe_id = 0
        self.active_keyframe_id = -1
        self.active_local_to_world: Optional[torch.Tensor] = None
        self.active_pose_encoding: Optional[torch.Tensor] = None
        self.global_anchor: Optional[dict] = None
        self.latest_anchor_depth: Optional[torch.Tensor] = None
        self.latest_anchor_pose: Optional[torch.Tensor] = None
        self.latest_anchor_frame_idx: Optional[int] = None
        self.history_slots: List[dict] = []
        self.retired_keyframes: Dict[int, torch.Tensor] = {}

    def get_num_anchor_frames(self) -> int:
        if not self.initialized:
            return 0
        return 1 + len(self.history_slots)

    def get_active_keyframe_id(self) -> int:
        return self.active_keyframe_id

    def get_active_local_to_world(self) -> torch.Tensor:
        if self.active_local_to_world is None:
            raise RuntimeError("Active keyframe is not initialized")
        return self.active_local_to_world

    def get_active_pose_encoding(self) -> torch.Tensor:
        if self.active_pose_encoding is None:
            raise RuntimeError("Active keyframe pose is not initialized")
        return self.active_pose_encoding

    def update(
        self,
        frame_idx: int,
        depth: torch.Tensor,
        pose_abs_enc: torch.Tensor,
        image_size_hw,
    ) -> KeyframeEvent:
        current_local_to_world = pose_encoding_to_c2w(pose_abs_enc, image_size_hw)

        if not self.initialized:
            keyframe_id = self.next_keyframe_id
            self.next_keyframe_id += 1
            self.initialized = True
            self.active_keyframe_id = keyframe_id
            self.active_local_to_world = current_local_to_world
            self.active_pose_encoding = pose_abs_enc.clone()
            self.latest_anchor_depth = depth.clone()
            self.latest_anchor_pose = pose_abs_enc.clone()
            self.latest_anchor_frame_idx = int(frame_idx)
            self.global_anchor = {
                "frame_idx": frame_idx,
                "keyframe_id": keyframe_id,
                "anchor_slot": 0,
                "local_to_world": current_local_to_world,
            }
            slot_pose_updates = {keyframe_id: torch.eye(4, dtype=current_local_to_world.dtype, device=current_local_to_world.device)}
            return KeyframeEvent(
                event_type=KeyframeEventType.PROMOTE_KEYFRAME,
                frame_idx=frame_idx,
                keyframe_id=keyframe_id,
                anchor_slot=0,
                reorder_indices=torch.tensor([-1], dtype=torch.long, device=current_local_to_world.device),
                demoted_slot=None,
                num_anchor_frames=1,
                slot_pose_updates=slot_pose_updates,
                local_to_world=current_local_to_world,
                active_pose_encoding=pose_abs_enc.clone(),
            )

        should_register = self._should_register(frame_idx, depth, pose_abs_enc, image_size_hw)
        if not should_register:
            return KeyframeEvent(
                event_type=KeyframeEventType.NOOP,
                frame_idx=frame_idx,
                keyframe_id=self.active_keyframe_id,
                anchor_slot=-1,
                reorder_indices=None,
                demoted_slot=None,
                num_anchor_frames=self.get_num_anchor_frames(),
                slot_pose_updates=self._build_slot_pose_updates(self.active_local_to_world),
                local_to_world=self.active_local_to_world,
                active_pose_encoding=self.active_pose_encoding.clone() if self.active_pose_encoding is not None else None,
            )

        keyframe_id = self.next_keyframe_id
        self.next_keyframe_id += 1
        max_history_anchors = max(int(self.config.max_history_anchors), 0)

        if max_history_anchors <= 0:
            anchor_slot = -1
            reorder_indices = torch.tensor(
                [0, -1],
                dtype=torch.long,
                device=current_local_to_world.device,
            )
            event_type = KeyframeEventType.PROMOTE_KEYFRAME
            demoted_slot = None
            demoted_record = None
            if (
                self.active_keyframe_id >= 0
                and self.global_anchor is not None
                and self.active_keyframe_id != int(self.global_anchor["keyframe_id"])
                and self.active_local_to_world is not None
            ):
                self.retired_keyframes[self.active_keyframe_id] = self.active_local_to_world.clone()
        elif len(self.history_slots) < max_history_anchors:
            anchor_slot = len(self.history_slots) + 1
            reorder_indices = torch.tensor(
                [0] + [slot["anchor_slot"] for slot in self.history_slots] + [-1],
                dtype=torch.long,
                device=current_local_to_world.device,
            )
            event_type = KeyframeEventType.PROMOTE_KEYFRAME
            demoted_slot = None
            demoted_record = None
        else:
            demoted_slot = 1
            # Capture the demoted (oldest) keyframe's pose BEFORE history_slots is
            # renumbered, so we can retain its transform for tokens still referencing
            # it (e.g. tokens rescued by protect_topk_on_demotion_). Without this,
            # FIFO_SWAP drops the demoted keyframe from slot_to_active, and any token
            # whose slot_id still points to it falls back to an identity transform in
            # _project_slot_local_xyz_to_active (P5: identity-fallback projection bug).
            demoted_record = dict(self.history_slots[0]) if self.history_slots else None
            remaining_slots = self.history_slots[1:]
            reorder_indices = torch.tensor(
                [0] + [slot["anchor_slot"] for slot in remaining_slots] + [-1],
                dtype=torch.long,
                device=current_local_to_world.device,
            )
            self.history_slots = [
                {
                    **slot,
                    "anchor_slot": new_anchor_slot,
                }
                for new_anchor_slot, slot in enumerate(remaining_slots, start=1)
            ]
            anchor_slot = max_history_anchors
            event_type = KeyframeEventType.FIFO_SWAP

        if demoted_record is not None:
            demoted_kf_id = int(demoted_record["keyframe_id"])
            self.retired_keyframes[demoted_kf_id] = demoted_record["local_to_world"].clone()

        if anchor_slot >= 0:
            self.history_slots.append(
                {
                    "frame_idx": frame_idx,
                    "keyframe_id": keyframe_id,
                    "anchor_slot": anchor_slot,
                    "local_to_world": current_local_to_world,
                }
            )
        self.active_keyframe_id = keyframe_id
        self.active_local_to_world = current_local_to_world
        self.active_pose_encoding = pose_abs_enc.clone()
        self.latest_anchor_depth = depth.clone()
        self.latest_anchor_pose = pose_abs_enc.clone()
        self.latest_anchor_frame_idx = int(frame_idx)

        slot_pose_updates = self._build_slot_pose_updates(current_local_to_world)

        return KeyframeEvent(
            event_type=event_type,
            frame_idx=frame_idx,
            keyframe_id=keyframe_id,
            anchor_slot=anchor_slot,
            reorder_indices=reorder_indices,
            demoted_slot=demoted_slot,
            num_anchor_frames=self.get_num_anchor_frames(),
            slot_pose_updates=slot_pose_updates,
            local_to_world=current_local_to_world,
            active_pose_encoding=pose_abs_enc.clone(),
        )

    def _build_slot_pose_updates(self, active_local_to_world: torch.Tensor) -> Dict[int, torch.Tensor]:
        world_to_active = closed_form_inverse_se3(active_local_to_world.unsqueeze(0))[0]
        updates: Dict[int, torch.Tensor] = {}
        if self.global_anchor is not None:
            updates[self.global_anchor["keyframe_id"]] = world_to_active @ self.global_anchor["local_to_world"]
        for slot in self.history_slots:
            updates[slot["keyframe_id"]] = world_to_active @ slot["local_to_world"]
        for keyframe_id, local_to_world in self.retired_keyframes.items():
            updates[int(keyframe_id)] = world_to_active @ local_to_world
        if self.active_keyframe_id not in updates:
            updates[self.active_keyframe_id] = torch.eye(
                4,
                dtype=active_local_to_world.dtype,
                device=active_local_to_world.device,
            )
        return updates

    def _should_register(
        self,
        frame_idx: int,
        depth: torch.Tensor,
        pose_abs_enc: torch.Tensor,
        image_size_hw,
    ) -> bool:
        if frame_idx in set(self.config.forced_keyframe_frames):
            return True

        strategy_trigger = False
        if self.config.strategy == "fixed_interval":
            strategy_trigger = frame_idx > 0 and frame_idx % max(self.config.interval, 1) == 0
        elif self.config.strategy == "coverage" and not self.config.coverage_monitor_only:
            if (
                self.latest_anchor_depth is not None
                and self.latest_anchor_pose is not None
                and (
                    self.latest_anchor_frame_idx is None
                    or frame_idx - int(self.latest_anchor_frame_idx) >= max(int(self.config.interval), 1)
                )
            ):
                coverage = compute_coverage(
                    self.latest_anchor_depth,
                    self.latest_anchor_pose,
                    pose_abs_enc,
                    image_size_hw,
                    self.config.sample_ratio,
                )
                strategy_trigger = coverage < self.config.coverage_threshold

        threshold_trigger = False
        if (
            self.config.translation_threshold is not None
            or self.config.rotation_threshold_deg is not None
        ):
            active_pose = self.latest_anchor_pose
            if active_pose is not None:
                translation, rotation_deg = pose_delta_metrics(active_pose, pose_abs_enc, image_size_hw)
                if self.config.translation_threshold is not None:
                    threshold_trigger = threshold_trigger or (translation > self.config.translation_threshold)
                if self.config.rotation_threshold_deg is not None:
                    threshold_trigger = threshold_trigger or (rotation_deg > self.config.rotation_threshold_deg)
        return strategy_trigger or threshold_trigger


def pose_encoding_to_c2w(pose_enc: torch.Tensor, image_size_hw) -> torch.Tensor:
    pose_batched = pose_enc.unsqueeze(0).unsqueeze(0)
    return pose_encoding_to_camera_to_world(
        pose_batched,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )[0, 0]


def pose_delta_metrics(pose_a: torch.Tensor, pose_b: torch.Tensor, image_size_hw):
    c2w_a = pose_encoding_to_c2w(pose_a, image_size_hw)
    c2w_b = pose_encoding_to_c2w(pose_b, image_size_hw)
    translation = torch.norm(c2w_a[:3, 3] - c2w_b[:3, 3]).item()

    rel_rot = c2w_a[:3, :3].transpose(0, 1) @ c2w_b[:3, :3]
    trace = torch.trace(rel_rot).clamp(min=-1.0, max=3.0)
    cos_theta = ((trace - 1.0) / 2.0).clamp(min=-1.0, max=1.0)
    rotation_deg = torch.rad2deg(torch.acos(cos_theta)).item()
    return translation, rotation_deg


def pose_encoding_to_w2c(pose_enc: torch.Tensor, image_size_hw) -> torch.Tensor:
    pose_batched = pose_enc.unsqueeze(0).unsqueeze(0)
    return pose_encoding_to_world_to_camera(
        pose_batched,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )[0, 0]
