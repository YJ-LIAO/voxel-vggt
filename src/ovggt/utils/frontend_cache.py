from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TOKEN_METADATA_FEATURE_INDEX

from .geometry import closed_form_inverse_se3
from .pose_enc import pose_encoding_to_extri_intri

_TOKEN_KIND_CACHE: Dict[tuple, Tensor] = {}
_PATCH_GRID_CACHE: Dict[tuple, tuple[Tensor, Tensor]] = {}
_SCORER_XYZ_SCALE = 10.0
_SCORER_FRAME_AGE_SCALE = 128.0
_SCORER_ID_SCALE = 64.0


def voxel_hash_collision_free(voxels: Tensor) -> Tensor:
    """Collision-free hash of integer voxel coordinates to 1D ids.

    P4 fix: the previous `x + 1000y + 1000000z` linear hash collides for
    |voxel| >= 1000 (scene scale >= 100m at voxel_size=0.1). This offset-based
    scheme is collision-free over the full int64 range: each coord is shifted to
    non-negative and packed into disjoint bit ranges. With 21 bits per axis it
    covers ±~1M voxels per axis (~100km at voxel_size=0.1), far beyond any scene.
    """
    voxels = voxels.to(torch.long)
    OFFSET = 1 << 20  # shift negatives into non-negative range (±1M range)
    shifted = voxels + OFFSET
    packed = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
    # pack into int64 stays unique because each 21-bit field is disjoint.
    return packed


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
    # P2: eviction old/new-token blend weight (old=cosine diversity, new=repr_shift),
    # decoupled from `importance_weight` (which the dedup composite uses as its
    # importance-vs-depth_conf blend). Default 0.5 = prior behavior (backward compat).
    # Only the eviction call in commit_pending_update_ reads this; dedup is unaffected.
    eviction_importance_weight: float = 0.5
    dedup_cooldown_frames: int = 0
    intra_frame_dedup_enabled: bool = True
    # soft-merge intra dedup: "drop" (hard-drop all but best-score per 0.1m voxel, legacy behavior)
    # or "merge" (importance-weighted K/V avg of co-voxel tokens — preserves multi-view info).
    # Only takes effect when intra_frame_dedup_enabled=True (that boolean gates the whole intra block).
    intra_dedup_mode: Literal["drop", "merge"] = "drop"
    fifo_keep_topk: int = 0  # Retain top-K tokens by score when demoting oldest anchor (0=disable)
    # NOTE: production noIntra+fifo80 baseline sets this to 80 at the call site
    # (e.g. tools/test_multi_scene.py). Unbounded 80/swap protection accumulates
    # and starves eviction at long sequences; it is only safe with a bounded
    # rescued pool (see fifo_protected_ring_ratio).
    learned_eviction_enabled: bool = False
    score_state_dim: int = 128
    oracle_window: int = 4
    budget_allocation: Literal["dynamic", "uniform"] = "uniform"
    learned_fifo_keep_count: bool = False
    fifo_count_candidates: tuple = (0, 8, 16, 32, 64, 128)
    # P1 fix: cap the FIFO-protected slot-0 tokens so they cannot accumulate
    # unboundedly across swaps. max_protected_ratio * cache_budget = ceiling on
    # slot-0 count. Default 1.0 disables the cap (backward compatible).
    max_protected_ratio: float = 1.0
    # P1 v2 (mechanism C): FIFO rescued-token pool capacity as a ratio of
    # per_layer_budget. Caps the non-global rescued slot-0 tokens; when a
    # FIFO_SWAP would exceed it, the oldest keyframe's rescued tokens are
    # revoked (by keyframe_id, argsort-stable) before protecting new ones.
    # 0.0 = disabled (backward compat, falls back to v1 max_protected behavior).
    # Production default 0.2: with per_layer_budget=8334 (depth=24 → total
    # ~200016) the ring caps rescued tokens at 1666/layer. MUST be paired with
    # budget_allocation='uniform', else the dynamic per-layer budget can dip
    # below protected_count on budget-poor layers and trigger anchor overflow
    # (verified: dynamic→4.7–10.2% overflow_rate; uniform→0.0%).
    fifo_protected_ring_ratio: float = 0.2

    def __post_init__(self):
        # pass-5 #4: ring and v1 max_protected are mutually exclusive — both
        # clamp the same keep_count in sequence, which is confusing. Enforce at
        # config construction so a conflicting setup fails fast.
        if self.fifo_protected_ring_ratio > 0.0 and self.max_protected_ratio < 1.0:
            raise ValueError(
                "fifo_protected_ring_ratio 和 max_protected_ratio 互斥 (两者顺序作用同一 keep_count 会混淆)。"
                "启用 ring 时保持 max_protected_ratio=1.0 (默认, cap 禁用)。"
            )
        # ring sizes capacity against the static per_layer_budget, but eviction
        # uses the per-layer budget. Under budget_allocation='dynamic' the
        # softmax can push a layer's budget below protected_count (global anchor
        # + ring rescued) and trigger anchor overflow (verified 4.7–10.2% on
        # 7-Scenes). Require uniform allocation whenever the ring is on.
        if self.fifo_protected_ring_ratio > 0.0 and self.budget_allocation == "dynamic":
            raise ValueError(
                "fifo_protected_ring_ratio>0 需要 budget_allocation='uniform'："
                "ring 按 per_layer_budget 算容量，而 dynamic 分配会把某些层 budget 压到低于 protected_count，"
                "触发 anchor overflow（实测 4.7–10.2%）。请设 budget_allocation='uniform'。"
            )



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
    score_state: Optional[Tensor] = None
    score_state_current: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.score_state is None and self.score_state_current is not None:
            self.score_state = self.score_state_current
        elif self.score_state_current is None:
            self.score_state_current = self.score_state


@dataclass
class LayerCacheState:
    k: Optional[Tensor] = None
    v: Optional[Tensor] = None
    score_state: Optional[Tensor] = None
    metadata: Optional[TokenMetadata] = None
    protected_count: int = 0
    max_history_anchors: int = 3
    slot_to_active: Optional[Dict[int, Tensor]] = None
    needs_reorder_: bool = False
    _cached_protected_count: int = 0  # cache for _compute_protected_count, updated at mutation points
    # pass-2 #6: declared field (not a runtime attribute) — set by ring revoke
    # in protect_topk_on_demotion_, read+reset by commit_pending_update_ to force
    # a reorder that relocates revoked tokens out of the protected region.
    _needs_reorder_after_revoke: bool = False

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
        if self.score_state is not None:
            score_dim = self.score_state.shape[-1]
            score_indices = indices.unsqueeze(-1).expand(B, indices.shape[1], score_dim)
            self.score_state = torch.gather(self.score_state, 1, score_indices)
        self.metadata = self.metadata.index_select(indices)
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count

    def _gather_single_batch_(self, indices: Tensor) -> None:
        if self.k is None or self.v is None or self.metadata is None:
            return
        if indices.dim() != 1:
            raise ValueError(f"Expected 1D indices for single-batch gather, got shape {tuple(indices.shape)}")
        indices = indices.to(device=self.k.device, dtype=torch.long)
        self.k = self.k.index_select(2, indices)
        self.v = self.v.index_select(2, indices)
        if self.score_state is not None:
            self.score_state = self.score_state.index_select(1, indices)
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
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count

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
        score_dim = self.score_state.shape[-1] if self.score_state is not None else 0

        # 找到每个batch需要保留的最大token数量
        max_kept = max(idx.shape[0] for idx in indices_list) if indices_list else 0

        if max_kept == 0:
            # 所有batch都为空
            self.k = torch.zeros((B, H, 0, D), dtype=dtype, device=device)
            self.v = torch.zeros((B, H, 0, D), dtype=dtype, device=device)
            if self.score_state is not None:
                self.score_state = torch.zeros((B, 0, score_dim), dtype=dtype, device=device)
            self.metadata = TokenMetadata.empty(B, device, dtype)
            self.protected_count = 0
            return

        # 对每个batch单独处理gather并padding
        padded_k = []
        padded_v = []
        padded_score_state = []
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
                if self.score_state is not None:
                    padded_score_state.append(torch.zeros((1, max_kept, score_dim), dtype=dtype, device=device))
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
                if self.score_state is not None:
                    score_indices = indices.unsqueeze(0).unsqueeze(-1).expand(1, num_kept, score_dim)
                    score_b = torch.gather(self.score_state[b_idx:b_idx+1], 1, score_indices)

                # Padding
                pad_size = max_kept - num_kept
                k_pad = k_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()
                v_pad = v_b[:, :, -1:, :].expand(1, H, pad_size, D).clone()
                if self.score_state is not None:
                    score_pad = score_b[:, -1:, :].expand(1, pad_size, score_dim).clone()

                padded_k.append(torch.cat([k_b, k_pad], dim=2))
                padded_v.append(torch.cat([v_b, v_pad], dim=2))
                if self.score_state is not None:
                    padded_score_state.append(torch.cat([score_b, score_pad], dim=1))

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
                if self.score_state is not None:
                    score_indices = indices.unsqueeze(0).unsqueeze(-1).expand(1, num_kept, score_dim)
                    score_b = torch.gather(self.score_state[b_idx:b_idx+1], 1, score_indices)
                metadata_b = self.metadata.index_select(indices.unsqueeze(0))

                padded_k.append(k_b)
                padded_v.append(v_b)
                if self.score_state is not None:
                    padded_score_state.append(score_b)
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
        if self.score_state is not None:
            self.score_state = torch.cat(padded_score_state, dim=0)

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

        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count

    def append_(
        self,
        k_new: Tensor,
        v_new: Tensor,
        metadata_new: TokenMetadata,
        score_state_new: Optional[Tensor] = None,
    ) -> None:
        if score_state_new is None and self.score_state is not None:
            score_state_new = torch.zeros(
                k_new.shape[0],
                k_new.shape[2],
                self.score_state.shape[-1],
                dtype=k_new.dtype,
                device=k_new.device,
            )
        if self.k is None or self.v is None or self.metadata is None:
            self.k = k_new
            self.v = v_new
            self.metadata = metadata_new
            self.score_state = score_state_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.metadata = self.metadata.append(metadata_new)
            if self.score_state is not None or score_state_new is not None:
                if self.score_state is None:
                    self.score_state = torch.zeros(
                        self.k.shape[0],
                        self.k.shape[2] - k_new.shape[2],
                        score_state_new.shape[-1],
                        dtype=score_state_new.dtype,
                        device=score_state_new.device,
                    )
                if score_state_new is None:
                    score_state_new = torch.zeros(
                        k_new.shape[0],
                        k_new.shape[2],
                        self.score_state.shape[-1],
                        dtype=self.score_state.dtype,
                        device=self.score_state.device,
                    )
                self.score_state = torch.cat([self.score_state, score_state_new], dim=1)
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count

    def protect_topk_on_demotion_(self, demoted_slot: int, keep_count: int,
                                    token_scorer=None, layer_id: int = 0,
                                    current_frame_id: int | None = None,
                                    fifo_probe=None, batch_index: int = 0,
                                    cache_budget: int | None = None,
                                    max_protected: int | None = None,
                                    fifo_ring_capacity: int | None = None,
                                    global_anchor_keyframe_id: int | None = None) -> None:
        """Before FIFO_SWAP demotion, reassign top-K tokens from the demoted
        anchor to slot 0 (global anchor) so they survive eviction.

        v2: When token_scorer is available, use scorer logits instead of
        importance for ranking tokens in the demoted slot.

        v3: The probe is called BEFORE any metadata mutation so it observes
        the original demoted-slot token set.  keep_count=0 is recorded by the
        probe while never mutating anchor_slot.  keep_count >= demoted_token_count
        protects all demoted-slot tokens.

        v4 (P1 fix): if max_protected is provided, cap keep_count so the slot-0
        (FIFO-protected) token count never exceeds it. Without this cap, slot 0
        accumulates fifo_keep_topk tokens per swap indefinitely and can starve
        eviction (P1: permanent protection accumulation).
        """
        if self.metadata is None or self.num_tokens() == 0:
            return

        # v2: 当 scorer 可用时，获取当前帧ID用于构建 metadata features
        if current_frame_id is None:
            current_frame_id = int(self.metadata.frame_id.max().item()) if self.metadata.frame_id.numel() > 0 else 0

        # 1. Compute demoted-slot indices FIRST (before any mutation)
        demoted_indices_by_batch: dict[int, Tensor] = {}
        for b_idx in range(self.metadata.anchor_slot.shape[0]):
            slot_mask = self.metadata.anchor_slot[b_idx] == demoted_slot
            indices = torch.nonzero(slot_mask, as_tuple=False).squeeze(-1)
            demoted_indices_by_batch[b_idx] = indices

        # 2.5 (P1 mechanism C): if the FIFO rescued pool is enabled and the new
        # protection would exceed its capacity, compute a revoke plan for the
        # oldest non-global rescued tokens. The plan is computed here (before the
        # probe) but applied after (step 3.5), so the probe still observes the
        # original demoted-slot candidate set and the override early-return cannot
        # bypass the revoke.
        revoke_by_batch: dict[int, Tensor] = {}
        keep_count_by_batch: dict[int, int] = {}
        if fifo_ring_capacity is not None and fifo_ring_capacity > 0:
            gaid = int(global_anchor_keyframe_id) if global_anchor_keyframe_id is not None else -1
            for b_idx in range(self.metadata.anchor_slot.shape[0]):
                requested_keep_count = int(keep_count)  # batch-local; don't reuse a clamped value across batches
                slot0_mask = self.metadata.anchor_slot[b_idx] == 0
                slot0_indices = torch.nonzero(slot0_mask, as_tuple=False).squeeze(-1)
                slot0_kf_ids = self.metadata.keyframe_id[b_idx, slot0_indices]
                rotatable = slot0_kf_ids != gaid  # global anchor never rotates
                rot_idx = slot0_indices[rotatable]
                rot_kf = slot0_kf_ids[rotatable]
                if rot_idx.numel() + requested_keep_count <= fifo_ring_capacity:
                    keep_count_by_batch[b_idx] = requested_keep_count
                    continue  # under cap, no revoke needed
                overflow = (rot_idx.numel() + requested_keep_count) - fifo_ring_capacity
                k = 0
                if rot_idx.numel() > 0:
                    k = min(overflow, rot_idx.numel())
                    # deterministic tie-break: oldest keyframe first (argsort ascending),
                    # same keyframe_id broken by original position (stable sort).
                    order = torch.argsort(rot_kf, stable=True)
                    revoke_by_batch[b_idx] = rot_idx[order[:k]]
                # pass-3 #1: clamp keep_count to the room remaining after revoke,
                # regardless of whether revoke fully covered the overflow.
                remaining_after_revoke = rot_idx.numel() - k
                keep_count_by_batch[b_idx] = min(
                    requested_keep_count,
                    max(0, fifo_ring_capacity - remaining_after_revoke),
                )

        # 2. Fire probe BEFORE any metadata mutation (even if keep_count=0)
        keep_indices_override = None
        if fifo_probe is not None:
            keep_indices_override = fifo_probe.on_fifo_topk_candidate(
                cache_state=self,
                demoted_slot=demoted_slot,
                keep_count=keep_count,
                layer_id=layer_id,
                frame_id=current_frame_id if current_frame_id is not None else 0,
                batch_index=batch_index,
                demoted_indices_by_batch=demoted_indices_by_batch,
            )

        # 3.5 (P1 mechanism C): apply the revoke plan regardless of whether the
        # probe overrode the keep set — the override early-return must not bypass
        # the pool-capacity revoke.
        for b_idx, revoke_indices in revoke_by_batch.items():
            if revoke_indices.numel() > 0:
                self.metadata.anchor_slot[b_idx, revoke_indices] = -1
                self._needs_reorder_after_revoke = True
        if revoke_by_batch:
            self._cached_protected_count = self._compute_protected_count_raw()
            self.protected_count = self._cached_protected_count

        if keep_indices_override is not None:
            keep_indices_override = torch.as_tensor(
                keep_indices_override,
                dtype=torch.long,
                device=self.metadata.anchor_slot.device,
            ).reshape(-1)
            if keep_indices_override.numel() == 0:
                return
            max_idx = self.num_tokens() - 1
            if max_idx >= 0:
                keep_indices_override = keep_indices_override.clamp(0, max_idx)
                keep_indices_override = torch.unique(keep_indices_override, sorted=True)
            for b_idx, indices in demoted_indices_by_batch.items():
                if indices.numel() <= 0:
                    continue
                keep_mask = torch.isin(indices, keep_indices_override)
                top_indices = indices[keep_mask]
                if top_indices.numel() > 0:
                    self.metadata.anchor_slot[b_idx, top_indices] = 0
            self._cached_protected_count = self._compute_protected_count_raw()
            self.protected_count = self._cached_protected_count
            return

        # 3. Now guard: if keep_count <= 0, skip token reassignment
        if keep_count <= 0:
            return

        # 3.5 (P1 fix): cap keep_count so the FIFO-protected slot-0 token count
        # does not exceed max_protected. slot 0 is the permanently-protected region
        # that accumulates fifo_keep_topk tokens per swap; without a cap it grows
        # unboundedly and starves eviction.
        if max_protected is not None and self.metadata is not None:
            try:
                slot0_count = int((self.metadata.anchor_slot[0] == 0).sum().item())
            except Exception:
                slot0_count = 0
            available = max(0, int(max_protected) - slot0_count)
            if keep_count > available:
                keep_count = available
            if keep_count <= 0:
                return

        # 4. Per batch: clamp count and protect tokens
        for b_idx, indices in demoted_indices_by_batch.items():
            # pass-5 #1: honor the per-batch keep_count clamped by the ring (step 2.5).
            # B=1 (frontend) collapses to the single value; B>1 keeps batches independent.
            batch_keep = keep_count_by_batch.get(b_idx, keep_count) if keep_count_by_batch else keep_count
            effective_keep_count = min(max(int(batch_keep), 0), int(indices.numel()))
            if effective_keep_count <= 0:
                continue
            if effective_keep_count == int(indices.numel()):
                # All demoted-slot tokens are protected
                top_indices = indices
            elif token_scorer is not None and self.score_state is not None:
                # v2: 提取 demoted slot 的 score_state 和 metadata
                slot_score_state = self.score_state[b_idx, indices]  # [K, Ds]
                # 构建 slot 级别的 metadata features
                full_metadata_features = self.build_scorer_metadata_features(current_frame_id, decision_context=2)
                slot_features = full_metadata_features[b_idx, indices]  # [K, Dm]
                logits = token_scorer(
                    slot_score_state.unsqueeze(0),
                    slot_features.unsqueeze(0),
                    layer_id,
                )
                scores = logits[0]  # [K]
                _, top_local = torch.topk(scores, k=effective_keep_count)
                top_indices = indices[top_local]
            else:
                scores = self.metadata.importance[b_idx, indices]
                _, top_local = torch.topk(scores, k=effective_keep_count)
                top_indices = indices[top_local]
            # Reassign top-K to slot 0 so they survive the FIFO demotion
            self.metadata.anchor_slot[b_idx, top_indices] = 0

        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count

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
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count
        return indices

    def apply_voxel_dedup_(self, config: FrontendCacheConfig, current_frame_id: int,
                           token_scorer=None, layer_id: int = 0,
                           dedup_probe=None, batch_index: int = 0,
                           dedup_replay_probe=None) -> None:
        """
        向量化体素去重实现。

        优化策略：
        1. 批量构建mask，避免逐batch循环
        2. 批量投影3D坐标和计算分数
        3. 单batch去重逻辑完全向量化

        v2: 当 token_scorer 可用时，用 scorer logits 替代启发式 composite score。
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

        # Cooldown: recently promoted anchors skip dedup for N frames
        if config.dedup_cooldown_frames > 0:
            frames_since = current_frame_id - metadata.frame_id
            protected_patch_mask = protected_patch_mask & (frames_since > config.dedup_cooldown_frames)

        # 当前帧的patch tokens
        current_patch_mask = (
            (metadata.frame_id == current_frame_id)
            & patch_mask
            & valid_xyz_mask
        )  # [B, N]

        # ========== 阶段2：批量计算分数 ==========
        if token_scorer is not None and self.score_state is not None:
            # v2: 统一 scorer：score_state + metadata → logits
            metadata_features = self.build_scorer_metadata_features(current_frame_id, decision_context=1)
            logits = token_scorer(self.score_state, metadata_features, layer_id=layer_id)
            scores = logits  # [B, N]，高分 = 优先保留
        else:
            # fallback 到启发式 composite score
            scores = _composite_candidate_scores_batch(
                metadata.importance,
                metadata.depth_conf,
                current_patch_mask,
                config.importance_weight,
                config.depth_conf_weight,
            )  # [B, N]

        # ========== 阶段2.5：计算 policy_keep_indices（在 probe callback 之前） ==========
        if B == 1:
            policy_keep_indices, merge_plan_b0 = self._dedup_single_batch(
                b_idx=0,
                protected_patch_mask=protected_patch_mask[0],
                current_patch_mask=current_patch_mask[0],
                scores=scores[0],
                config=config,
                total_tokens=N,
                projected_xyz=projected_xyz[0],
            )
        else:
            policy_keep_indices = None  # computed per-batch below

        # After computing keep indices, call dedup probe if present
        if dedup_probe is not None:
            local_batch_index = 0 if B == 1 else int(batch_index)
            dedup_probe.on_dedup_candidate(
                cache_state=self,
                layer_id=layer_id,
                frame_id=current_frame_id,
                batch_index=batch_index,
                scores=scores[local_batch_index],
                policy_keep_indices=policy_keep_indices,
            )

        # Check for dedup replay override
        if dedup_replay_probe is not None:
            override_indices = dedup_replay_probe.on_dedup_candidate(
                cache_state=self,
                layer_id=layer_id,
                frame_id=current_frame_id,
                batch_index=batch_index,
            )
            if override_indices is not None:
                # Apply the override keep set directly
                override_indices = torch.as_tensor(override_indices, dtype=torch.long, device=self.k.device)
                if override_indices.dim() == 1:
                    override_indices = override_indices.unsqueeze(0)
                max_idx = self.num_tokens() - 1
                if max_idx >= 0:
                    override_indices = override_indices.clamp(0, max_idx)
                    override_indices = torch.unique(override_indices, sorted=True)
                    override_indices = override_indices.reshape(1, -1)
                self.gather_(override_indices)
                return

        if B == 1:
            if merge_plan_b0 is not None:
                self._apply_intra_merge_(0, merge_plan_b0)
            self._gather_single_batch_(policy_keep_indices)
            return

        # ========== 阶段3：逐batch处理去重逻辑 ==========
        kept_per_batch = []

        for b_idx in range(B):
            keep_mask_b, merge_plan_b = self._dedup_single_batch(
                b_idx=b_idx,
                protected_patch_mask=protected_patch_mask[b_idx],
                current_patch_mask=current_patch_mask[b_idx],
                scores=scores[b_idx],
                config=config,
                total_tokens=N,
                projected_xyz=projected_xyz[b_idx],
            )
            if merge_plan_b is not None:
                self._apply_intra_merge_(b_idx, merge_plan_b)
            kept_per_batch.append(keep_mask_b)

        # ========== 阶段4：使用gather_per_batch_处理不同数量的token ==========
        self.gather_per_batch_(kept_per_batch)

    # ------------------------------------------------------------------
    # Demoted-slot helpers for count head integration
    # ------------------------------------------------------------------

    def get_demoted_slot_indices(
        self, demoted_slot: int, local_batch_index: int = 0
    ) -> Tensor:
        """Return token indices belonging to *demoted_slot* for one batch element.

        Parameters
        ----------
        demoted_slot : int
            The anchor-slot value identifying the demoted anchor.
        local_batch_index : int
            Which batch element to query (default 0, which is the only
            element when cache_state is already per-sample).

        Returns
        -------
        Tensor[K]  (1-D long tensor of token positions)
        """
        if self.metadata is None:
            return torch.empty(0, dtype=torch.long)
        slot_mask = self.metadata.anchor_slot[local_batch_index] == demoted_slot
        return torch.nonzero(slot_mask, as_tuple=False).squeeze(-1)

    def get_demoted_slot_score_state(
        self, demoted_slot: int, local_batch_index: int = 0
    ) -> Tensor:
        """Return the score_state rows for tokens in the demoted slot.

        Parameters
        ----------
        demoted_slot : int
        local_batch_index : int

        Returns
        -------
        Tensor[K, Ds]  (empty tensor with Ds columns if no demoted tokens)
        """
        indices = self.get_demoted_slot_indices(demoted_slot, local_batch_index)
        if indices.numel() == 0 or self.score_state is None:
            Ds = self.score_state.shape[-1] if self.score_state is not None else 0
            return torch.empty(0, Ds, dtype=torch.float)
        # score_state is [B, N, Ds]; index via [local_batch_index, indices]
        return self.score_state[local_batch_index, indices]

    def get_demoted_slot_metadata_features(
        self,
        demoted_slot: int,
        current_frame_id: int,
        local_batch_index: int = 0,
    ) -> Tensor:
        """Return scorer metadata-feature rows for tokens in the demoted slot.

        Builds the full metadata feature tensor first, then slices out only
        the demoted-slot rows for the requested batch element.

        Parameters
        ----------
        demoted_slot : int
        current_frame_id : int
        local_batch_index : int

        Returns
        -------
        Tensor[K, Dm]  (empty tensor with Dm columns if no demoted tokens)
        """
        indices = self.get_demoted_slot_indices(demoted_slot, local_batch_index)
        if indices.numel() == 0:
            return torch.empty(0, TOKEN_METADATA_FEATURE_DIM, dtype=torch.float)
        full_features = self.build_scorer_metadata_features(
            current_frame_id, decision_context=2
        )
        return full_features[local_batch_index, indices]

    def build_scorer_metadata_features(self, current_frame_id: int, decision_context: int = 3) -> Tensor:
        if self.metadata is None:
            raise ValueError("Cannot build scorer metadata features without TokenMetadata")
        metadata = self.metadata
        dtype = metadata.slot_local_xyz.dtype
        device = metadata.slot_local_xyz.device
        B, N = metadata.frame_id.shape
        features = torch.zeros(B, N, TOKEN_METADATA_FEATURE_DIM, dtype=dtype, device=device)

        xyz_valid = torch.isfinite(metadata.slot_local_xyz).all(dim=-1)
        active_xyz = self._project_slot_local_xyz_to_active(
            metadata.slot_local_xyz,
            metadata.slot_id,
        )
        local_xyz = torch.nan_to_num(metadata.slot_local_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        active_xyz = torch.nan_to_num(active_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        local_xyz = torch.where(xyz_valid.unsqueeze(-1), local_xyz, torch.zeros_like(local_xyz))
        active_xyz = torch.where(xyz_valid.unsqueeze(-1), active_xyz, torch.zeros_like(active_xyz))
        local_xyz = (local_xyz / _SCORER_XYZ_SCALE).clamp(-1.0, 1.0)
        active_xyz = (active_xyz / _SCORER_XYZ_SCALE).clamp(-1.0, 1.0)

        idx = TOKEN_METADATA_FEATURE_INDEX
        features[..., idx["depth_conf"]] = metadata.depth_conf.to(dtype).clamp(0.0, 1.0)
        start = idx["slot_local_xyz_start"]
        features[..., start : start + 3] = local_xyz
        start = idx["active_xyz_start"]
        features[..., start : start + 3] = active_xyz
        features[..., idx["frame_age"]] = (
            torch.as_tensor(current_frame_id, dtype=dtype, device=device) - metadata.frame_id.to(dtype)
        ).clamp_min(0).div(_SCORER_FRAME_AGE_SCALE).clamp(0.0, 1.0)
        features[..., idx["anchor_slot"]] = metadata.anchor_slot.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
        features[..., idx["is_protected"]] = (metadata.anchor_slot >= 0).to(dtype)
        features[..., idx["kind_camera"]] = (metadata.token_kind == int(TokenKind.CAMERA)).to(dtype)
        features[..., idx["kind_register"]] = (metadata.token_kind == int(TokenKind.REGISTER)).to(dtype)
        features[..., idx["kind_patch"]] = (metadata.token_kind == int(TokenKind.PATCH)).to(dtype)
        features[..., idx["slot_id"]] = metadata.slot_id.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
        features[..., idx["keyframe_id"]] = metadata.keyframe_id.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
        features[..., idx["xyz_valid"]] = xyz_valid.to(dtype)
        features[..., idx["decision_context"]] = float(decision_context) / 3.0  # normalize to [0,1]
        return features

    def _learned_eviction_(
        self,
        cache_budget: int,
        token_scorer,
        layer_id: int,
        current_frame_id: int,
    ) -> Optional[float]:
        if self.k is None or self.v is None or self.metadata is None:
            return None
        if self.score_state is None:
            raise ValueError("learned_eviction_enabled=True requires LayerCacheState.score_state")

        B, _, N, _ = self.k.shape
        cache_budget = max(int(cache_budget), 0)
        protected_count = min(max(int(self.protected_count), 0), N)
        if N <= cache_budget:
            return None

        keep_from_candidates = min(max(cache_budget - protected_count, 0), N - protected_count)
        device = self.k.device
        if keep_from_candidates <= 0:
            keep_protected = min(cache_budget, protected_count)
            start = max(protected_count - keep_protected, 0)
            kept_indices = torch.arange(start, protected_count, device=device).unsqueeze(0).expand(B, -1)
            self.gather_(kept_indices)
            return None

        metadata_features = self.build_scorer_metadata_features(current_frame_id=current_frame_id)
        logits = token_scorer(self.score_state, metadata_features, layer_id=layer_id)
        if logits.shape != (B, N):
            raise ValueError(f"TokenScorer returned {tuple(logits.shape)}, expected {(B, N)}")

        candidate_logits = logits[:, protected_count:]
        _, top_candidate_indices = torch.topk(candidate_logits, k=keep_from_candidates, dim=-1)
        top_candidate_indices = top_candidate_indices.sort(dim=-1).values + protected_count
        protected_indices = torch.arange(protected_count, device=device).unsqueeze(0).expand(B, -1)
        kept_indices = torch.cat([protected_indices, top_candidate_indices], dim=-1)
        self.gather_(kept_indices)
        return float(candidate_logits.mean().detach().cpu().item())

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
        # Build per-voxel best score map from protected tokens, then compare
        # with current-frame tokens: only discard current tokens that are worse.
        protected_patch_indices = torch.empty(0, dtype=torch.long, device=device)
        if protected_patch_mask.any():
            protected_patch_indices = torch.nonzero(protected_patch_mask, as_tuple=False).squeeze(-1)

        if protected_patch_indices.numel() > 0:
            protected_xyz = projected_xyz[protected_patch_indices]
            protected_voxels = torch.floor(protected_xyz / config.voxel_size).to(torch.long)
            protected_scores = scores[protected_patch_indices]

            # Map voxel key → best protected score using scatter
            voxel_hash = voxel_hash_collision_free(protected_voxels)
            unique_vhash, inv = torch.unique(voxel_hash, return_inverse=True)
            num_v = unique_vhash.shape[0]
            # Initialize with -inf so any real score wins
            best_protected = torch.full((num_v,), float('-inf'), device=device)
            best_protected.scatter_reduce_(0, inv, protected_scores, reduce='amax', include_self=True)

            # For each current token, check if it conflicts with a protected voxel
            current_voxel_hash = voxel_hash_collision_free(current_patch_voxels)
            protected_conflict_mask = torch.zeros(current_patch_indices.shape[0], dtype=torch.bool, device=device)
            discard_current_mask = torch.zeros(current_patch_indices.shape[0], dtype=torch.bool, device=device)

            # Map current voxel hash to the best_protected index
            current_to_best = torch.full((current_voxel_hash.shape[0],), -1, dtype=torch.long, device=device)
            sort_idx = torch.searchsorted(unique_vhash, current_voxel_hash)
            sort_idx = sort_idx.clamp(0, num_v - 1)
            matched = unique_vhash[sort_idx] == current_voxel_hash
            current_to_best[matched] = sort_idx[matched]

            has_match = current_to_best >= 0
            if has_match.any():
                protected_conflict_mask[has_match] = True
                # Only discard current token if its score is worse than the best protected token in that voxel
                worse = current_patch_scores[has_match] < best_protected[current_to_best[has_match]]
                discard_idx = torch.nonzero(has_match, as_tuple=False).squeeze(-1)
                discard_current_mask[discard_idx[worse]] = True

            if discard_current_mask.any():
                keep_mask[current_patch_indices[discard_current_mask]] = False

            _, current_group_ids = torch.unique(current_patch_voxels, dim=0, return_inverse=True)
        else:
            _, current_group_ids = torch.unique(current_patch_voxels, dim=0, return_inverse=True)
            protected_conflict_mask = torch.zeros_like(current_group_ids, dtype=torch.bool)
            discard_current_mask = torch.zeros_like(current_group_ids, dtype=torch.bool)

        # 帧内去重：保留每个体素中评分最高的token
        merge_plan = None
        if config.intra_frame_dedup_enabled:
            # Only exclude tokens that were actually discarded, not all conflicting ones
            survivor_mask = ~discard_current_mask if protected_patch_indices.numel() > 0 else torch.ones(current_patch_indices.shape[0], dtype=torch.bool, device=device)
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

                # soft-merge plan: for each multi-member voxel group, record the
                # representative (highest-score survivor) + all members + softmax
                # weights, so apply_voxel_dedup_ can weighted-avg their K/V before
                # gather. Token COUNT is unchanged (= unique voxels); only VALUES differ.
                if config.intra_dedup_mode == "merge" and ordered_group_ids.numel() > 1:
                    group_starts = torch.nonzero(keep_first, as_tuple=False).squeeze(-1)  # [Gt]
                    Gt = group_starts.numel()
                    ends = torch.cat([group_starts[1:],
                                      torch.tensor([ordered_group_ids.numel()], device=device, dtype=group_starts.dtype)])
                    counts = ends - group_starts  # [Gt] members per group
                    multi = counts > 1
                    if multi.any():
                        ms = group_starts[multi]               # [G] start pos per multi group
                        mc = counts[multi]                     # [G] member count per multi group
                        G = mc.numel()
                        rep_indices = survivor_indices[final_order[ms]]  # [G] global token idx
                        maxc = int(mc.max().item())
                        offs = torch.arange(maxc, device=device).unsqueeze(0)      # [1, maxc]
                        pos = ms.unsqueeze(1) + offs                              # [G, maxc]
                        mmask = offs < mc.unsqueeze(1)                           # [G, maxc]
                        pos_flat = pos[mmask]                                     # [M] final_order positions
                        member_indices = survivor_indices[final_order[pos_flat]]  # [M] global
                        member_scores = survivor_scores[final_order[pos_flat]]    # [M]
                        member_rep_map = torch.repeat_interleave(
                            torch.arange(G, device=device), mc)                   # [M] -> [0,G)
                        # per-group softmax weights (numerically stable)
                        max_pg = torch.full((G,), float('-inf'), device=device)
                        max_pg.scatter_reduce_(0, member_rep_map, member_scores,
                                               reduce='amax', include_self=True)
                        ex = torch.exp(member_scores - max_pg[member_rep_map])
                        sx = torch.zeros(G, device=device).index_add_(0, member_rep_map, ex)
                        member_weights = ex / sx[member_rep_map].clamp(min=1e-12)
                        merge_plan = {"rep_indices": rep_indices, "member_indices": member_indices,
                                      "member_weights": member_weights, "member_rep_map": member_rep_map}

        return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1), merge_plan

    def _apply_intra_merge_(self, b_idx: int, plan: dict) -> None:
        """Apply a soft-merge plan (from _dedup_single_batch, merge mode): for each
        multi-member voxel group, write the importance-weighted average of the members'
        K/V (+ score_state, importance, depth_conf) into the representative slot.
        Token count is unchanged (gather afterwards drops the non-rep members; the rep
        slot already holds the merged value). Call BEFORE _gather_single_batch_."""
        rep = plan["rep_indices"]            # [G]
        memb = plan["member_indices"]         # [M]
        w = plan["member_weights"]            # [M]
        rep_of = plan["member_rep_map"]       # [M] in [0, G)
        _, H, _, D = self.k.shape
        for kv in (self.k, self.v):
            contrib = kv[b_idx, :, memb, :] * w.view(1, -1, 1)            # [H, M, D]
            acc = torch.zeros(H, rep.shape[0], D, device=kv.device, dtype=kv.dtype)  # [H, G, D]
            acc.index_add_(dim=1, index=rep_of, source=contrib)           # Σ_members w·kv -> rep slot
            kv[b_idx, :, rep, :] = acc
        # importance / depth_conf: importance-weighted average (1D)
        for field in ("importance", "depth_conf"):
            val = getattr(self.metadata, field)[b_idx, memb] * w           # [M]
            acc1 = torch.zeros(rep.shape[0], device=val.device, dtype=val.dtype)
            acc1.index_add_(0, rep_of, val)
            getattr(self.metadata, field)[b_idx, rep] = acc1
        # slot_local_xyz [B, N, 3]: unweighted MEAN per group (3D, not weighted)
        xyz = self.metadata.slot_local_xyz[b_idx, memb]                    # [M, 3]
        acc_xyz = torch.zeros(rep.shape[0], 3, device=xyz.device, dtype=xyz.dtype)
        cnt = torch.zeros(rep.shape[0], device=xyz.device, dtype=xyz.dtype)
        acc_xyz.index_add_(0, rep_of, xyz)
        cnt.index_add_(0, rep_of, torch.ones_like(w))
        self.metadata.slot_local_xyz[b_idx, rep] = acc_xyz / cnt.unsqueeze(-1).clamp(min=1.0)
        # score_state [B, N, Ds]: weighted average (only when present, i.e. scorer enabled)
        if self.score_state is not None:
            ss = self.score_state[b_idx, memb] * w.view(-1, 1)             # [M, Ds]
            acc_ss = torch.zeros(rep.shape[0], ss.shape[-1], device=ss.device, dtype=ss.dtype)
            acc_ss.index_add_(0, rep_of, ss)
            self.score_state[b_idx, rep] = acc_ss

    def _scorer_scores_or_importance_(
        self,
        token_scorer,
        current_frame_id: int,
        layer_id: int,
        fallback_scores: Tensor,
        score_state_slice: Tensor,
    ) -> Tensor:
        """Use scorer to produce retention scores; fall back to heuristic scores."""
        if token_scorer is None or self.score_state is None:
            return fallback_scores
        # Build metadata features for the current set of tokens
        metadata_features = self.build_scorer_metadata_features(current_frame_id)
        logits = token_scorer(score_state_slice, metadata_features, layer_id=layer_id)
        return logits

    def commit_pending_update_(
        self,
        pending_update: PendingLayerUpdate,
        current_metadata: TokenMetadata,
        config: FrontendCacheConfig,
        intra_frame_keep_ratio: float,
        attn_module,
        token_scorer=None,
        layer_id: int = 0,
        eviction_probe=None,
        batch_index: int = 0,
        dedup_probe=None,
        dedup_replay_probe=None,
    ) -> Optional[float]:
        k_current = pending_update.k_current
        v_current = pending_update.v_current
        metadata_current = current_metadata
        score_state_current = pending_update.score_state_current

        if not metadata_current.has_anchor_tokens() and intra_frame_keep_ratio < 1.0:
            keep_count = max(int(k_current.shape[2] * intra_frame_keep_ratio), 1)
            if keep_count < k_current.shape[2]:
                # v2: Use scorer for intra-frame pruning when available
                if token_scorer is not None and score_state_current is not None:
                    # Build metadata features for current frame tokens only
                    # Use a temporary cache state view to build features
                    temp_metadata_features = _build_current_frame_metadata_features(
                        metadata_current, pending_update.frame_id,
                        decision_context=0,
                    )
                    pruning_logits = token_scorer(
                        score_state_current, temp_metadata_features, layer_id=layer_id,
                    )
                    _, top_indices = torch.topk(pruning_logits, k=keep_count, dim=-1)
                else:
                    _, top_indices = torch.topk(metadata_current.importance, k=keep_count, dim=-1)
                top_indices = top_indices.sort(dim=-1).values
                expanded = top_indices.unsqueeze(1).unsqueeze(-1).expand(
                    k_current.shape[0], k_current.shape[1], keep_count, k_current.shape[-1]
                )
                k_current = torch.gather(k_current, 2, expanded)
                v_current = torch.gather(v_current, 2, expanded)
                if score_state_current is not None:
                    score_dim = score_state_current.shape[-1]
                    score_indices = top_indices.unsqueeze(-1).expand(
                        top_indices.shape[0],
                        top_indices.shape[1],
                        score_dim,
                    )
                    score_state_current = torch.gather(score_state_current, 1, score_indices)
                metadata_current = metadata_current.index_select(top_indices)

        self.append_(k_current, v_current, metadata_current, score_state_new=score_state_current)
        # pass-1 #3: force reorder when the ring revoked tokens (their anchor_slot
        # changed but they're still positioned in the protected region), so the
        # subsequent eviction binary split is correct.
        if metadata_current.has_anchor_tokens() or self._needs_reorder_after_revoke:
            self.reorder_by_anchor_slots_()
            self._needs_reorder_after_revoke = False
        self.apply_voxel_dedup_(config, current_frame_id=pending_update.frame_id,
                                token_scorer=token_scorer, layer_id=layer_id,
                                dedup_probe=dedup_probe, batch_index=batch_index,
                                dedup_replay_probe=dedup_replay_probe)

        if pending_update.cache_budget is None or self.num_tokens() <= pending_update.cache_budget:
            return None

        if eviction_probe is not None:
            keep_indices_override = eviction_probe.on_eviction_candidate(
                cache_state=self,
                layer_id=layer_id,
                frame_id=pending_update.frame_id,
                budget=pending_update.cache_budget,
                batch_index=batch_index,
            )
            if keep_indices_override is not None:
                keep_indices_override = torch.as_tensor(
                    keep_indices_override,
                    dtype=torch.long,
                    device=self.k.device,
                ).reshape(1, -1)
                # Clamp indices to valid range: replay cache may have fewer
                # tokens than the probe phase due to nondeterministic dedup.
                max_idx = self.num_tokens() - 1
                if max_idx >= 0:
                    keep_indices_override = keep_indices_override.clamp(0, max_idx)
                    # Remove duplicates that may result from clamping
                    keep_indices_override = torch.unique(keep_indices_override, sorted=True)
                    keep_indices_override = keep_indices_override.reshape(1, -1)
                self.gather_(keep_indices_override)
                return None

        if config.learned_eviction_enabled:
            if token_scorer is None:
                raise ValueError("learned_eviction_enabled=True requires token_scorer")
            return self._learned_eviction_(
                cache_budget=pending_update.cache_budget,
                token_scorer=token_scorer,
                layer_id=layer_id,
                current_frame_id=pending_update.frame_id,
            )

        importance_scores, num_new_tokens = self._current_frame_importance(pending_update.frame_id)

        # P6 invariant guard (protects the P3-verified ordering): eviction()'s hybrid
        # path assumes the LAST num_new_tokens candidate positions are the current-frame
        # tokens. append_ -> reorder_by_anchor_slots_ -> apply_voxel_dedup_ preserves
        # this by always returning survivors in ascending positional order. If a future
        # refactor breaks that ordering, this assert catches it before silent eviction
        # misalignment. Only checked when hybrid scoring will be used.
        if (importance_scores is not None and num_new_tokens > 0
                and self.metadata is not None and self.k is not None):
            total_tokens = self.k.shape[2]
            num_candidates = total_tokens - self.protected_count
            num_old_candidates = num_candidates - num_new_tokens
            if num_old_candidates >= 0 and num_new_tokens > 0:
                tail_start = self.protected_count + num_old_candidates
                tail_frame_ids = self.metadata.frame_id[0, tail_start:total_tokens]
                expected_frame = int(pending_update.frame_id)
                if not bool((tail_frame_ids == expected_frame).all().item()):
                    n_mismatch = int((tail_frame_ids != expected_frame).sum().item())
                    raise AssertionError(
                        f"P6 invariant violated: candidate tail (positions {tail_start}:"
                        f"{total_tokens}) should all be frame {expected_frame} but "
                        f"{n_mismatch}/{num_new_tokens} differ. Eviction would misalign "
                        f"importance scores. Check reorder/dedup ordering."
                    )

        final_k, final_v, avg_score, kept_indices = attn_module.eviction(
            self.k,
            self.v,
            pending_update.cache_budget,
            self.protected_count,
            importance_scores=importance_scores,
            num_new_tokens=num_new_tokens,
            importance_weight=config.eviction_importance_weight,
        )
        self.k = final_k
        self.v = final_v
        if kept_indices is not None:
            if self.score_state is not None:
                score_dim = self.score_state.shape[-1]
                score_indices = kept_indices.unsqueeze(-1).expand(
                    kept_indices.shape[0],
                    kept_indices.shape[1],
                    score_dim,
                )
                self.score_state = torch.gather(self.score_state, 1, score_indices)
            self.metadata = self.metadata.index_select(kept_indices)
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count
        return avg_score

    def _compute_protected_count_raw(self) -> int:
        """Original computation — called only at mutation points (has GPU sync)."""
        if self.metadata is None or self.metadata.anchor_slot.numel() == 0:
            return 0
        return int((self.metadata.anchor_slot[0] >= 0).sum().item())

    def _compute_protected_count(self) -> int:
        """Cached version — returns pre-computed value, no GPU sync."""
        return self._cached_protected_count

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


def _build_current_frame_metadata_features(
    metadata: TokenMetadata,
    current_frame_id: int,
    decision_context: int = 0,  # pruning by default
) -> Tensor:
    """Build scorer metadata features for a set of tokens (used in intra-frame pruning).

    This is a standalone version that doesn't require a LayerCacheState, so it
    can be called before tokens are appended to the cache.
    """
    dtype = metadata.slot_local_xyz.dtype
    device = metadata.slot_local_xyz.device
    B, N = metadata.frame_id.shape
    features = torch.zeros(B, N, TOKEN_METADATA_FEATURE_DIM, dtype=dtype, device=device)

    xyz_valid = torch.isfinite(metadata.slot_local_xyz).all(dim=-1)
    local_xyz = torch.nan_to_num(metadata.slot_local_xyz, nan=0.0, posinf=0.0, neginf=0.0)
    local_xyz = torch.where(xyz_valid.unsqueeze(-1), local_xyz, torch.zeros_like(local_xyz))
    local_xyz = (local_xyz / _SCORER_XYZ_SCALE).clamp(-1.0, 1.0)

    # active_xyz = slot_local_xyz for tokens not yet in cache (no transform available)
    active_xyz = local_xyz.clone()

    idx = TOKEN_METADATA_FEATURE_INDEX
    features[..., idx["depth_conf"]] = metadata.depth_conf.to(dtype).clamp(0.0, 1.0)
    start = idx["slot_local_xyz_start"]
    features[..., start : start + 3] = local_xyz
    start = idx["active_xyz_start"]
    features[..., start : start + 3] = active_xyz
    features[..., idx["frame_age"]] = torch.zeros(B, N, dtype=dtype, device=device)
    features[..., idx["anchor_slot"]] = metadata.anchor_slot.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
    features[..., idx["is_protected"]] = (metadata.anchor_slot >= 0).to(dtype)
    features[..., idx["kind_camera"]] = (metadata.token_kind == int(TokenKind.CAMERA)).to(dtype)
    features[..., idx["kind_register"]] = (metadata.token_kind == int(TokenKind.REGISTER)).to(dtype)
    features[..., idx["kind_patch"]] = (metadata.token_kind == int(TokenKind.PATCH)).to(dtype)
    features[..., idx["slot_id"]] = metadata.slot_id.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
    features[..., idx["keyframe_id"]] = metadata.keyframe_id.to(dtype).clamp_min(0).div(_SCORER_ID_SCALE).clamp(0.0, 1.0)
    features[..., idx["xyz_valid"]] = xyz_valid.to(dtype)
    features[..., idx["decision_context"]] = float(decision_context) / 3.0
    return features


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
    patch_conf = depth_conf.reshape(B, patch_h, patch_size, patch_w, patch_size).sum(dim=(2, 4)).reshape(B, -1)

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
