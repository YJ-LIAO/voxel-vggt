from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .geometry import closed_form_inverse_se3
from .pose_enc import pose_encoding_to_extri_intri

_TOKEN_KIND_CACHE: Dict[tuple, Tensor] = {}
_PATCH_GRID_CACHE: Dict[tuple, tuple[Tensor, Tensor]] = {}


class TokenKind(IntEnum):
    CAMERA = 0
    REGISTER = 1
    PATCH = 2


@dataclass
class FrontendCacheConfig:
    enabled: bool = False
    voxel_size: float = 0.1
    dedup_enabled: bool = True
    export_keyframe_packets: bool = False
    depth_conf_weight: float = 0.5
    importance_weight: float = 0.5


@dataclass
class TokenMetadata:
    token_kind: Tensor
    frame_id: Tensor
    anchor_slot: Tensor
    keyframe_id: Tensor
    slot_id: Tensor
    slot_local_xyz: Tensor
    importance: Tensor
    depth_conf: Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TokenMetadata":
        empty_long = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        empty_xyz = torch.empty(batch_size, 0, 3, dtype=dtype, device=device)
        empty_float = torch.empty(batch_size, 0, dtype=dtype, device=device)
        return cls(
            token_kind=empty_long.clone(),
            frame_id=empty_long.clone(),
            anchor_slot=empty_long.clone(),
            keyframe_id=empty_long.clone(),
            slot_id=empty_long.clone(),
            slot_local_xyz=empty_xyz,
            importance=empty_float.clone(),
            depth_conf=empty_float.clone(),
        )

    def clone(self) -> "TokenMetadata":
        return TokenMetadata(
            token_kind=self.token_kind.clone(),
            frame_id=self.frame_id.clone(),
            anchor_slot=self.anchor_slot.clone(),
            keyframe_id=self.keyframe_id.clone(),
            slot_id=self.slot_id.clone(),
            slot_local_xyz=self.slot_local_xyz.clone(),
            importance=self.importance.clone(),
            depth_conf=self.depth_conf.clone(),
        )

    def index_select(self, indices: Tensor) -> "TokenMetadata":
        gather_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
        return TokenMetadata(
            token_kind=torch.gather(self.token_kind, 1, indices),
            frame_id=torch.gather(self.frame_id, 1, indices),
            anchor_slot=torch.gather(self.anchor_slot, 1, indices),
            keyframe_id=torch.gather(self.keyframe_id, 1, indices),
            slot_id=torch.gather(self.slot_id, 1, indices),
            slot_local_xyz=torch.gather(self.slot_local_xyz, 1, gather_xyz),
            importance=torch.gather(self.importance, 1, indices),
            depth_conf=torch.gather(self.depth_conf, 1, indices),
        )

    def append(self, other: "TokenMetadata") -> "TokenMetadata":
        return TokenMetadata(
            token_kind=torch.cat([self.token_kind, other.token_kind], dim=1),
            frame_id=torch.cat([self.frame_id, other.frame_id], dim=1),
            anchor_slot=torch.cat([self.anchor_slot, other.anchor_slot], dim=1),
            keyframe_id=torch.cat([self.keyframe_id, other.keyframe_id], dim=1),
            slot_id=torch.cat([self.slot_id, other.slot_id], dim=1),
            slot_local_xyz=torch.cat([self.slot_local_xyz, other.slot_local_xyz], dim=1),
            importance=torch.cat([self.importance, other.importance], dim=1),
            depth_conf=torch.cat([self.depth_conf, other.depth_conf], dim=1),
        )

    def has_anchor_tokens(self) -> bool:
        return bool((self.anchor_slot >= 0).any().item())

    @property
    def local_xyz(self) -> Tensor:
        return self.slot_local_xyz


@dataclass
class FrameTokenMetadataBase:
    token_kind: Tensor
    frame_id: Tensor
    anchor_slot: Tensor
    keyframe_id: Tensor
    slot_id: Tensor
    slot_local_xyz: Tensor
    depth_conf: Tensor

    def with_importance(self, importance: Tensor) -> TokenMetadata:
        return TokenMetadata(
            token_kind=self.token_kind,
            frame_id=self.frame_id,
            anchor_slot=self.anchor_slot,
            keyframe_id=self.keyframe_id,
            slot_id=self.slot_id,
            slot_local_xyz=self.slot_local_xyz,
            importance=importance.to(self.slot_local_xyz.dtype),
            depth_conf=self.depth_conf,
        )

    @property
    def local_xyz(self) -> Tensor:
        return self.slot_local_xyz


@dataclass
class PendingLayerUpdate:
    k_current: Tensor
    v_current: Tensor
    importance_current: Tensor
    frame_id: int
    cache_budget: Optional[int] = None


@dataclass
class LayerCacheState:
    k: Optional[Tensor] = None
    v: Optional[Tensor] = None
    metadata: Optional[TokenMetadata] = None
    protected_count: int = 0
    max_history_anchors: int = 3
    slot_to_active: Optional[Dict[int, Tensor]] = None

    def as_past_key_values(self):
        if self.k is None or self.v is None:
            return None
        return self.k, self.v

    def num_tokens(self) -> int:
        if self.k is None:
            return 0
        return int(self.k.shape[2])

    def gather_(self, indices: Optional[Tensor]) -> None:
        if indices is None or self.k is None or self.v is None or self.metadata is None:
            return
        if self.k.shape[0] == 1 and indices.shape[0] == 1:
            self._gather_single_batch_(indices[0])
            return
        B, H, _, D = self.k.shape
        expanded = indices.unsqueeze(1).unsqueeze(-1).expand(B, H, indices.shape[1], D)
        self.k = torch.gather(self.k, 2, expanded)
        self.v = torch.gather(self.v, 2, expanded)
        self.metadata = self.metadata.index_select(indices)
        self.protected_count = self._compute_protected_count()

    def _gather_single_batch_(self, indices: Tensor) -> None:
        if self.k is None or self.v is None or self.metadata is None:
            return
        if indices.dim() != 1:
            raise ValueError(f"Expected 1D indices for single-batch gather, got shape {tuple(indices.shape)}")
        indices = indices.to(device=self.k.device, dtype=torch.long)
        self.k = self.k.index_select(2, indices)
        self.v = self.v.index_select(2, indices)
        self.metadata = TokenMetadata(
            token_kind=self.metadata.token_kind.index_select(1, indices),
            frame_id=self.metadata.frame_id.index_select(1, indices),
            anchor_slot=self.metadata.anchor_slot.index_select(1, indices),
            keyframe_id=self.metadata.keyframe_id.index_select(1, indices),
            slot_id=self.metadata.slot_id.index_select(1, indices),
            slot_local_xyz=self.metadata.slot_local_xyz.index_select(1, indices),
            importance=self.metadata.importance.index_select(1, indices),
            depth_conf=self.metadata.depth_conf.index_select(1, indices),
        )
        self.protected_count = self._compute_protected_count()

    def gather_per_batch_(self, indices_list: List[Tensor]) -> None:
        """
        分别处理每个batch的gather操作，支持每个batch保留不同数量的token。

        Args:
            indices_list: List of [N_b] tensors, 每个batch要保留的索引
        """
        if self.k is None or self.v is None or self.metadata is None:
            return
        if len(indices_list) == 1:
            self._gather_single_batch_(indices_list[0])
            return

        B = len(indices_list)
        device = self.k.device
        dtype = self.k.dtype
        H = self.k.shape[1]
        D = self.k.shape[3]

        # 找到每个batch需要保留的最大token数量
        max_kept = max(idx.shape[0] for idx in indices_list) if indices_list else 0

        if max_kept == 0:
            # 所有batch都为空
            self.k = torch.zeros((B, H, 0, D), dtype=dtype, device=device)
            self.v = torch.zeros((B, H, 0, D), dtype=dtype, device=device)
            self.metadata = TokenMetadata.empty(B, device, dtype)
            self.protected_count = 0
            return

        # 对每个batch单独处理gather并padding
        padded_k = []
        padded_v = []
        # 直接构建完整的metadata fields
        token_kinds = []
        frame_ids = []
        anchor_slots = []
        keyframe_ids = []
        slot_ids = []
        slot_local_xyzs = []
        importances = []
        depth_confs = []

        for b_idx, indices in enumerate(indices_list):
            num_kept = indices.shape[0]

            if num_kept == 0:
                # 创建空的padding
                padded_k.append(torch.zeros((1, H, max_kept, D), dtype=dtype, device=device))
                padded_v.append(torch.zeros((1, H, max_kept, D), dtype=dtype, device=device))
                token_kinds.append(torch.zeros((max_kept,), dtype=torch.long, device=device))
                frame_ids.append(torch.zeros((max_kept,), dtype=torch.long, device=device))
                anchor_slots.append(torch.full((max_kept,), -1, dtype=torch.long, device=device))
                keyframe_ids.append(torch.zeros((max_kept,), dtype=torch.long, device=device))
                slot_ids.append(torch.zeros((max_kept,), dtype=torch.long, device=device))
                slot_local_xyzs.append(torch.zeros((max_kept, 3), dtype=dtype, device=device))
                importances.append(torch.zeros((max_kept,), dtype=dtype, device=device))
                depth_confs.append(torch.zeros((max_kept,), dtype=dtype, device=device))
            elif num_kept < max_kept:
                # Gather当前batch
                expanded = indices.unsqueeze(0).unsqueeze(-1).expand(1, H, num_kept, D)
                k_b = torch.gather(self.k[b_idx:b_idx+1], 2, expanded)
                v_b = torch.gather(self.v[b_idx:b_idx+1], 2, expanded)

                # Padding
                pad_size = max_kept - num_kept
                k_pad = k_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()
                v_pad = v_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()

                padded_k.append(torch.cat([k_b, k_pad], dim=2))
                padded_v.append(torch.cat([v_b, v_pad], dim=2))

                # Gather metadata并padding
                metadata_b = self.metadata.index_select(indices.unsqueeze(0))
                token_kinds.append(torch.cat([
                    metadata_b.token_kind[0],
                    metadata_b.token_kind[0, -1:].expand(pad_size).clone()
                ]))
                frame_ids.append(torch.cat([
                    metadata_b.frame_id[0],
                    metadata_b.frame_id[0, -1:].expand(pad_size).clone()
                ]))
                anchor_slots.append(torch.cat([
                    metadata_b.anchor_slot[0],
                    torch.full((pad_size,), -1, dtype=torch.long, device=device)
                ]))
                keyframe_ids.append(torch.cat([
                    metadata_b.keyframe_id[0],
                    metadata_b.keyframe_id[0, -1:].expand(pad_size).clone()
                ]))
                slot_ids.append(torch.cat([
                    metadata_b.slot_id[0],
                    metadata_b.slot_id[0, -1:].expand(pad_size).clone()
                ]))
                slot_local_xyzs.append(torch.cat([
                    metadata_b.slot_local_xyz[0],
                    metadata_b.slot_local_xyz[0, -1:, :].expand(pad_size, 3).clone()
                ], dim=0))
                importances.append(torch.cat([
                    metadata_b.importance[0],
                    metadata_b.importance[0, -1:].expand(pad_size).clone()
                ]))
                depth_confs.append(torch.cat([
                    metadata_b.depth_conf[0],
                    metadata_b.depth_conf[0, -1:].expand(pad_size).clone()
                ]))
            else:
                # num_kept == max_kept，无需padding
                expanded = indices.unsqueeze(0).unsqueeze(-1).expand(1, H, num_kept, D)
                k_b = torch.gather(self.k[b_idx:b_idx+1], 2, expanded)
                v_b = torch.gather(self.v[b_idx:b_idx+1], 2, expanded)
                metadata_b = self.metadata.index_select(indices.unsqueeze(0))

                padded_k.append(k_b)
                padded_v.append(v_b)
                token_kinds.append(metadata_b.token_kind[0])
                frame_ids.append(metadata_b.frame_id[0])
                anchor_slots.append(metadata_b.anchor_slot[0])
                keyframe_ids.append(metadata_b.keyframe_id[0])
                slot_ids.append(metadata_b.slot_id[0])
                slot_local_xyzs.append(metadata_b.slot_local_xyz[0])
                importances.append(metadata_b.importance[0])
                depth_confs.append(metadata_b.depth_conf[0])

        # 合并K/V
        self.k = torch.cat(padded_k, dim=0)
        self.v = torch.cat(padded_v, dim=0)

        # 合并metadata - 使用stack构建batch维度
        self.metadata = TokenMetadata(
            token_kind=torch.stack(token_kinds, dim=0),
            frame_id=torch.stack(frame_ids, dim=0),
            anchor_slot=torch.stack(anchor_slots, dim=0),
            keyframe_id=torch.stack(keyframe_ids, dim=0),
            slot_id=torch.stack(slot_ids, dim=0),
            slot_local_xyz=torch.stack(slot_local_xyzs, dim=0),
            importance=torch.stack(importances, dim=0),
            depth_conf=torch.stack(depth_confs, dim=0),
        )

        self.protected_count = self._compute_protected_count()

    def append_(self, k_new: Tensor, v_new: Tensor, metadata_new: TokenMetadata) -> None:
        if self.k is None or self.v is None or self.metadata is None:
            self.k = k_new
            self.v = v_new
            self.metadata = metadata_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.metadata = self.metadata.append(metadata_new)
        self.protected_count = self._compute_protected_count()

    def apply_keyframe_event_(self, event) -> None:
        slot_pose_updates = getattr(event, "slot_pose_updates", None)
        if slot_pose_updates:
            self.slot_to_active = {
                int(slot_id): transform.clone()
                for slot_id, transform in slot_pose_updates.items()
            }

        if self.metadata is None:
            return

        event_type = getattr(event, "event_type", None)
        if event_type is None or event_type == "NOOP":
            return
        if str(event_type) == "KeyframeEventType.NOOP":
            return
        if str(event_type).endswith("FIFO_SWAP"):
            demoted_slot = getattr(event, "demoted_slot", 1)
            if demoted_slot is None:
                demoted_slot = 1
            demoted_mask = self.metadata.anchor_slot == demoted_slot
            shifted_mask = self.metadata.anchor_slot > demoted_slot
            self.metadata.anchor_slot = torch.where(
                demoted_mask,
                torch.full_like(self.metadata.anchor_slot, -1),
                self.metadata.anchor_slot,
            )
            self.metadata.anchor_slot = torch.where(
                shifted_mask,
                self.metadata.anchor_slot - 1,
                self.metadata.anchor_slot,
            )

    def reorder_by_anchor_slots_(self) -> Optional[Tensor]:
        if self.metadata is None or self.num_tokens() == 0:
            self.protected_count = 0
            return None
        batch_indices = []
        for b_idx in range(self.metadata.anchor_slot.shape[0]):
            ordered = []
            for slot in range(self.max_history_anchors + 1):
                idx = torch.nonzero(self.metadata.anchor_slot[b_idx] == slot, as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    ordered.append(idx)
            candidate_idx = torch.nonzero(self.metadata.anchor_slot[b_idx] < 0, as_tuple=False).squeeze(-1)
            if candidate_idx.numel() > 0:
                ordered.append(candidate_idx)
            if ordered:
                batch_indices.append(torch.cat(ordered, dim=0))
            else:
                batch_indices.append(torch.empty(0, dtype=torch.long, device=self.metadata.anchor_slot.device))
        indices = torch.stack(batch_indices, dim=0)
        self.gather_(indices)
        self.protected_count = self._compute_protected_count()
        return indices

    def apply_voxel_dedup_(self, config: FrontendCacheConfig, current_frame_id: int) -> None:
        """
        向量化体素去重实现。

        优化策略：
        1. 批量构建mask，避免逐batch循环
        2. 批量投影3D坐标和计算分数
        3. 单batch去重逻辑完全向量化
        """
        if (
            self.metadata is None
            or self.k is None
            or self.v is None
            or not config.dedup_enabled
            or config.voxel_size <= 0
        ):
            return

        metadata = self.metadata
        B, N = metadata.anchor_slot.shape
        device = metadata.anchor_slot.device
        projected_xyz = self._project_slot_local_xyz_to_active(
            metadata.slot_local_xyz,
            metadata.slot_id,
        )

        # ========== 阶段1：批量构建mask ==========
        valid_xyz_mask = torch.isfinite(projected_xyz).all(dim=-1)  # [B, N]
        patch_mask = metadata.token_kind == int(TokenKind.PATCH)  # [B, N]

        # 受保护的patch tokens（锚点token，非当前帧）
        protected_patch_mask = (
            (metadata.anchor_slot >= 0)
            & (metadata.frame_id != current_frame_id)
            & patch_mask
            & valid_xyz_mask
        )  # [B, N]

        # 当前帧的patch tokens
        current_patch_mask = (
            (metadata.frame_id == current_frame_id)
            & patch_mask
            & valid_xyz_mask
        )  # [B, N]

        # ========== 阶段2：批量计算分数 ==========
        scores = _composite_candidate_scores_batch(
            metadata.importance,
            metadata.depth_conf,
            current_patch_mask,
            config.importance_weight,
            config.depth_conf_weight,
        )  # [B, N]

        if B == 1:
            kept_indices = self._dedup_single_batch(
                b_idx=0,
                protected_patch_mask=protected_patch_mask[0],
                current_patch_mask=current_patch_mask[0],
                scores=scores[0],
                config=config,
                total_tokens=N,
                projected_xyz=projected_xyz[0],
            )
            self._gather_single_batch_(kept_indices)
            return

        # ========== 阶段3：逐batch处理去重逻辑 ==========
        kept_per_batch = []

        for b_idx in range(B):
            keep_mask_b = self._dedup_single_batch(
                b_idx=b_idx,
                protected_patch_mask=protected_patch_mask[b_idx],
                current_patch_mask=current_patch_mask[b_idx],
                scores=scores[b_idx],
                config=config,
                total_tokens=N,
                projected_xyz=projected_xyz[b_idx],
            )
            kept_per_batch.append(keep_mask_b)

        # ========== 阶段4：使用gather_per_batch_处理不同数量的token ==========
        self.gather_per_batch_(kept_per_batch)

    def _dedup_single_batch(
        self,
        b_idx: int,
        protected_patch_mask: Tensor,
        current_patch_mask: Tensor,
        scores: Tensor,
        config: FrontendCacheConfig,
        total_tokens: int,
        projected_xyz: Tensor,
    ) -> Tensor:
        """
        单batch的去重逻辑，完全向量化。

        返回：保留的token索引 [num_kept]
        """
        metadata = self.metadata
        device = metadata.anchor_slot.device

        keep_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)

        current_patch_indices = torch.nonzero(current_patch_mask, as_tuple=False).squeeze(-1)

        if current_patch_indices.numel() == 0:
            return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

        current_patch_xyz = projected_xyz[current_patch_indices]

        current_patch_scores = scores[current_patch_indices]
        current_patch_voxels = torch.floor(current_patch_xyz / config.voxel_size).to(torch.long)

        # 检测与保护token的冲突
        if protected_patch_mask.any():
            protected_patch_indices = torch.nonzero(protected_patch_mask, as_tuple=False).squeeze(-1)
            protected_patch_xyz = projected_xyz[protected_patch_indices]
            protected_voxels = torch.floor(protected_patch_xyz / config.voxel_size).to(torch.long)

            all_voxels = torch.cat([protected_voxels, current_patch_voxels], dim=0)
            _, inverse = torch.unique(all_voxels, dim=0, return_inverse=True)

            num_protected = protected_voxels.shape[0]
            protected_group_ids = inverse[:num_protected].unique()
            current_group_ids = inverse[num_protected:]

            protected_group_mask = torch.zeros(
                int(inverse.max().item()) + 1,
                dtype=torch.bool,
                device=device,
            )
            protected_group_mask[protected_group_ids] = True
            protected_conflict_mask = protected_group_mask[current_group_ids]
        else:
            _, current_group_ids = torch.unique(current_patch_voxels, dim=0, return_inverse=True)
            protected_conflict_mask = torch.zeros_like(current_group_ids, dtype=torch.bool)

        # 标记冲突token为丢弃
        if protected_conflict_mask.any():
            keep_mask[current_patch_indices[protected_conflict_mask]] = False

        # 帧内去重：保留每个体素中评分最高的token
        survivor_mask = ~protected_conflict_mask
        survivor_indices = current_patch_indices[survivor_mask]
        survivor_group_ids = current_group_ids[survivor_mask]
        survivor_scores = current_patch_scores[survivor_mask]

        if survivor_indices.numel() > 0:
            # 按评分降序排序，再按组ID稳定排序
            order_by_score = torch.argsort(survivor_scores, descending=True, stable=True)
            grouped_order = torch.argsort(survivor_group_ids[order_by_score], stable=True)
            final_order = order_by_score[grouped_order]
            ordered_group_ids = survivor_group_ids[final_order]

            # 检测重复：同一组内第一个保留，其余丢弃
            keep_first = torch.ones_like(ordered_group_ids, dtype=torch.bool)
            if ordered_group_ids.numel() > 1:
                keep_first[1:] = ordered_group_ids[1:] != ordered_group_ids[:-1]

            duplicate_indices = survivor_indices[final_order[~keep_first]]
            if duplicate_indices.numel() > 0:
                keep_mask[duplicate_indices] = False

        return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

    def commit_pending_update_(
        self,
        pending_update: PendingLayerUpdate,
        current_metadata: TokenMetadata,
        config: FrontendCacheConfig,
        intra_frame_keep_ratio: float,
        attn_module,
    ) -> Optional[float]:
        k_current = pending_update.k_current
        v_current = pending_update.v_current
        metadata_current = current_metadata

        if not metadata_current.has_anchor_tokens() and intra_frame_keep_ratio < 1.0:
            keep_count = max(int(k_current.shape[2] * intra_frame_keep_ratio), 1)
            if keep_count < k_current.shape[2]:
                _, top_indices = torch.topk(metadata_current.importance, k=keep_count, dim=-1)
                top_indices = top_indices.sort(dim=-1).values
                expanded = top_indices.unsqueeze(1).unsqueeze(-1).expand(
                    k_current.shape[0], k_current.shape[1], keep_count, k_current.shape[-1]
                )
                k_current = torch.gather(k_current, 2, expanded)
                v_current = torch.gather(v_current, 2, expanded)
                metadata_current = metadata_current.index_select(top_indices)

        self.append_(k_current, v_current, metadata_current)
        if metadata_current.has_anchor_tokens():
            self.reorder_by_anchor_slots_()
        self.apply_voxel_dedup_(config, current_frame_id=pending_update.frame_id)

        if pending_update.cache_budget is None or self.num_tokens() <= pending_update.cache_budget:
            return None

        importance_scores, num_new_tokens = self._current_frame_importance(pending_update.frame_id)
        final_k, final_v, avg_score, kept_indices = attn_module.eviction(
            self.k,
            self.v,
            pending_update.cache_budget,
            self.protected_count,
            importance_scores=importance_scores,
            num_new_tokens=num_new_tokens,
            importance_weight=config.importance_weight,
        )
        self.k = final_k
        self.v = final_v
        if kept_indices is not None:
            self.metadata = self.metadata.index_select(kept_indices)
        self.protected_count = self._compute_protected_count()
        return avg_score

    def _compute_protected_count(self) -> int:
        if self.metadata is None or self.metadata.anchor_slot.numel() == 0:
            return 0
        return int((self.metadata.anchor_slot[0] >= 0).sum().item())

    def _current_frame_importance(self, frame_id: int):
        if self.metadata is None:
            return None, 0
        batch_scores = []
        counts = []
        for b_idx in range(self.metadata.frame_id.shape[0]):
            mask = (self.metadata.frame_id[b_idx] == frame_id) & (self.metadata.anchor_slot[b_idx] < 0)
            counts.append(int(mask.sum().item()))
            batch_scores.append(self.metadata.importance[b_idx, mask])
        if not counts or min(counts) == 0 or len(set(counts)) != 1:
            return None, 0
        return torch.stack(batch_scores, dim=0), counts[0]

    def _project_slot_local_xyz_to_active(self, slot_local_xyz: Tensor, slot_ids: Tensor) -> Tensor:
        if slot_local_xyz.numel() == 0:
            return slot_local_xyz
        if slot_local_xyz.shape[0] == 1:
            return self._project_slot_local_xyz_to_active_single_batch(slot_local_xyz, slot_ids)

        batch_size = slot_local_xyz.shape[0]
        valid_xyz_mask = torch.isfinite(slot_local_xyz).all(dim=-1)
        active_xyz = slot_local_xyz.clone()
        unique_slot_ids = torch.unique(slot_ids[valid_xyz_mask]).tolist() if valid_xyz_mask.any() else []
        for raw_slot_id in unique_slot_ids:
            slot_id = int(raw_slot_id)
            transform = self._get_slot_transform(
                slot_id=slot_id,
                batch_size=batch_size,
                device=slot_local_xyz.device,
                dtype=slot_local_xyz.dtype,
            )
            slot_mask = (slot_ids == slot_id) & valid_xyz_mask
            for b_idx in range(batch_size):
                mask_b = slot_mask[b_idx]
                if not mask_b.any():
                    continue
                transformed = transform_points(
                    slot_local_xyz[b_idx : b_idx + 1, mask_b],
                    transform[b_idx : b_idx + 1],
                )[0]
                active_xyz[b_idx, mask_b] = transformed
        return active_xyz

    def _project_slot_local_xyz_to_active_single_batch(self, slot_local_xyz: Tensor, slot_ids: Tensor) -> Tensor:
        valid_xyz_mask = torch.isfinite(slot_local_xyz[0]).all(dim=-1)
        if not valid_xyz_mask.any():
            return slot_local_xyz

        active_xyz = slot_local_xyz.clone()
        unique_slot_ids = torch.unique(slot_ids[0, valid_xyz_mask]).tolist()
        for raw_slot_id in unique_slot_ids:
            slot_id = int(raw_slot_id)
            slot_mask = (slot_ids[0] == slot_id) & valid_xyz_mask
            if not slot_mask.any():
                continue
            transform = self._get_slot_transform(
                slot_id=slot_id,
                batch_size=1,
                device=slot_local_xyz.device,
                dtype=slot_local_xyz.dtype,
            )
            active_xyz[:, slot_mask] = transform_points(slot_local_xyz[:, slot_mask], transform)
        return active_xyz

    def _get_slot_transform(
        self,
        slot_id: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        transform = None
        if self.slot_to_active is not None:
            transform = self.slot_to_active.get(int(slot_id))
        if transform is None:
            transform = torch.eye(4, dtype=dtype, device=device)
        if transform.dim() == 2:
            transform = transform.unsqueeze(0).expand(batch_size, -1, -1)
        return transform.to(device=device, dtype=dtype)


def build_token_kind_tensor(
    batch_size: int,
    total_tokens: int,
    patch_start_idx: int,
    device: torch.device,
) -> Tensor:
    cache_key = (total_tokens, patch_start_idx, device)
    token_kind = _TOKEN_KIND_CACHE.get(cache_key)
    if token_kind is None:
        token_kind = torch.full((total_tokens,), int(TokenKind.PATCH), dtype=torch.long, device=device)
        if total_tokens > 0:
            token_kind[0] = int(TokenKind.CAMERA)
        if patch_start_idx > 1:
            token_kind[1:patch_start_idx] = int(TokenKind.REGISTER)
        _TOKEN_KIND_CACHE[cache_key] = token_kind
    return token_kind.unsqueeze(0).expand(batch_size, -1)


def build_frame_token_metadata(
    depth: Tensor,
    depth_conf: Tensor,
    pose_enc: Tensor,
    image_size_hw,
    patch_size: int,
    patch_start_idx: int,
    frame_id: int,
    keyframe_id: int,
    slot_id: int,
    anchor_slot: int,
    importance: Tensor,
    active_local_to_world: Tensor,
) -> TokenMetadata:
    metadata_base = build_frame_token_metadata_base(
        depth=depth,
        depth_conf=depth_conf,
        pose_enc=pose_enc,
        image_size_hw=image_size_hw,
        patch_size=patch_size,
        patch_start_idx=patch_start_idx,
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        slot_id=slot_id,
        anchor_slot=anchor_slot,
        total_tokens=importance.shape[1],
        active_local_to_world=active_local_to_world,
    )
    return metadata_base.with_importance(importance)


def build_frame_token_metadata_base(
    depth: Tensor,
    depth_conf: Tensor,
    pose_enc: Tensor,
    image_size_hw,
    patch_size: int,
    patch_start_idx: int,
    frame_id: int,
    keyframe_id: int,
    slot_id: int,
    anchor_slot: int,
    total_tokens: int,
    active_local_to_world: Tensor,
) -> FrameTokenMetadataBase:
    B = depth.shape[0]
    device = depth.device
    dtype = depth.dtype
    token_kind = build_token_kind_tensor(B, total_tokens, patch_start_idx, device)
    slot_local_xyz = torch.full((B, total_tokens, 3), float("nan"), dtype=dtype, device=device)
    token_depth_conf = torch.zeros((B, total_tokens), dtype=dtype, device=device)

    patch_local_xyz, patch_conf = sample_patch_local_xyz(
        depth=depth,
        depth_conf=depth_conf,
        pose_enc=pose_enc,
        image_size_hw=image_size_hw,
        patch_size=patch_size,
        active_local_to_world=active_local_to_world,
    )
    slot_local_xyz[:, patch_start_idx:] = patch_local_xyz
    token_depth_conf[:, patch_start_idx:] = patch_conf

    fill_long = torch.full((B, total_tokens), frame_id, dtype=torch.long, device=device)
    keyframe_long = torch.full((B, total_tokens), keyframe_id, dtype=torch.long, device=device)
    slot_long = torch.full((B, total_tokens), slot_id, dtype=torch.long, device=device)
    anchor_long = torch.full((B, total_tokens), anchor_slot, dtype=torch.long, device=device)

    return FrameTokenMetadataBase(
        token_kind=token_kind,
        frame_id=fill_long,
        anchor_slot=anchor_long,
        keyframe_id=keyframe_long,
        slot_id=slot_long,
        slot_local_xyz=slot_local_xyz,
        depth_conf=token_depth_conf,
    )


def sample_patch_local_xyz(
    depth: Tensor,
    depth_conf: Tensor,
    pose_enc: Tensor,
    image_size_hw,
    patch_size: int,
    active_local_to_world: Tensor,
):
    if depth.dim() == 4:
        depth_2d = depth.squeeze(-1)
    else:
        depth_2d = depth
    B, H, W = depth_2d.shape
    patch_h = H // patch_size
    patch_w = W // patch_size

    grid_key = (H, W, patch_size, depth.device)
    grid = _PATCH_GRID_CACHE.get(grid_key)
    if grid is None:
        y_centers = torch.arange(patch_h, device=depth.device) * patch_size + (patch_size // 2)
        x_centers = torch.arange(patch_w, device=depth.device) * patch_size + (patch_size // 2)
        y_centers = torch.clamp(y_centers, 0, H - 1)
        x_centers = torch.clamp(x_centers, 0, W - 1)
        grid = torch.meshgrid(y_centers, x_centers, indexing="ij")
        _PATCH_GRID_CACHE[grid_key] = grid
    grid_y, grid_x = grid

    patch_depth = depth_2d[:, grid_y, grid_x].reshape(B, -1)
    patch_conf = depth_conf[:, grid_y, grid_x].reshape(B, -1)

    pose_batched = pose_enc.unsqueeze(1)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_batched, image_size_hw)
    extrinsics = extrinsics[:, 0]
    intrinsics = intrinsics[:, 0]

    w2c = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device).unsqueeze(0).repeat(B, 1, 1)
    w2c[:, :3, :4] = extrinsics
    c2w = closed_form_inverse_se3(w2c)

    if active_local_to_world.dim() == 2:
        active_local_to_world = active_local_to_world.unsqueeze(0).expand(B, -1, -1)
    world_to_local = closed_form_inverse_se3(active_local_to_world)

    fx = intrinsics[:, 0, 0].unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)

    flat_x = grid_x.reshape(1, -1).expand(B, -1).to(patch_depth.dtype)
    flat_y = grid_y.reshape(1, -1).expand(B, -1).to(patch_depth.dtype)

    x_cam = (flat_x - cx) * patch_depth / fx
    y_cam = (flat_y - cy) * patch_depth / fy
    z_cam = patch_depth
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    world_points = transform_points(cam_points, c2w)
    local_points = transform_points(world_points, world_to_local)
    return local_points, patch_conf


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    if transform.dim() == 2:
        transform = transform.unsqueeze(0).expand(points.shape[0], -1, -1)
    rot = transform[:, :3, :3]
    trans = transform[:, :3, 3]
    return torch.matmul(points, rot.transpose(1, 2)) + trans.unsqueeze(1)


def _normalize_with_mask(values: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return torch.zeros_like(values)
    selected = values[mask]
    v_min = selected.min()
    v_max = selected.max()
    if torch.isclose(v_min, v_max):
        out = torch.zeros_like(values)
        out[mask] = 0.5
        return out
    out = torch.zeros_like(values)
    out[mask] = (selected - v_min) / (v_max - v_min + 1e-8)
    return out


def _composite_candidate_scores(
    importance: Tensor,
    depth_conf: Tensor,
    mask: Tensor,
    importance_weight: float,
    depth_conf_weight: float,
) -> Tensor:
    norm_importance = _normalize_with_mask(importance, mask)
    norm_depth_conf = _normalize_with_mask(depth_conf, mask)
    return importance_weight * norm_importance + depth_conf_weight * norm_depth_conf


def _composite_candidate_scores_batch(
    importance: Tensor,
    depth_conf: Tensor,
    mask: Tensor,
    importance_weight: float,
    depth_conf_weight: float,
) -> Tensor:
    """
    批量计算复合评分，完全向量化实现。

    Args:
        importance: [B, N] 重要性分数
        depth_conf: [B, N] 深度置信度
        mask: [B, N] 只在该位置计算评分
        importance_weight: 重要性权重
        depth_conf_weight: 深度置信度权重

    Returns:
        [B, N] 复合评分，非mask位置为0
    """
    norm_importance = _normalize_with_mask_batch(importance, mask)
    norm_depth_conf = _normalize_with_mask_batch(depth_conf, mask)
    return importance_weight * norm_importance + depth_conf_weight * norm_depth_conf


def _normalize_with_mask_batch(values: Tensor, mask: Tensor) -> Tensor:
    """
    批量归一化，只在mask为True的位置进行归一化。
    完全向量化实现，消除循环。

    Args:
        values: [B, N] 输入值
        mask: [B, N] 归一化范围掩码

    Returns:
        [B, N] 归一化后的值，非mask位置为0
    """
    B, N = values.shape
    device = values.device
    dtype = values.dtype

    # 使用极值填充非mask位置，以便正确计算min/max
    inf = float('inf')
    values_for_min = torch.where(mask, values, torch.full_like(values, inf))
    values_for_max = torch.where(mask, values, torch.full_like(values, -inf))

    # 计算每个batch的min/max
    v_min = values_for_min.min(dim=1, keepdim=True)[0]  # [B, 1]
    v_max = values_for_max.max(dim=1, keepdim=True)[0]  # [B, 1]

    # 处理全为False的mask情况
    valid_batch = mask.any(dim=1, keepdim=True)  # [B, 1]
    v_min = torch.where(valid_batch, v_min, torch.zeros_like(v_min))
    v_max = torch.where(valid_batch, v_max, torch.ones_like(v_max))

    # 检查是否所有值相等
    range_val = v_max - v_min
    equal_mask = range_val <= 1e-8  # [B, 1]

    # 归一化
    denom = range_val.clamp(min=1e-8)
    normalized = (values - v_min) / denom

    # 处理相等情况：设为中性值0.5
    normalized = torch.where(
        equal_mask & valid_batch,
        torch.full_like(normalized, 0.5),
        normalized,
    )

    # 非mask位置设为0
    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))

    return normalized
