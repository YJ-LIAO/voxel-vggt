import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from torch.utils.checkpoint import checkpoint

from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from transformers.file_utils import ModelOutput

from ovggt.heads.camera_head import CameraHead
from ovggt.heads.dpt_head import DPTHead
from ovggt.models.aggregator import Aggregator
from ovggt.utils.frontend_cache import (
    FrontendCacheConfig,
    LayerCacheState,
    build_frame_token_metadata_base,
)
from ovggt.utils.frontend_keyframe import (
    FrontendKeyframeManager,
    KeyframePacket,
    KeyframeSwitchConfig,
)
from ovggt.utils.history_anchor import HistoryAnchorConfig, HistoryAnchorManager
from ovggt.utils.pose_enc import (
    ABS_POSE_ENCODING,
    REL_POSE_ENCODING,
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
    pose_encoding_to_world_to_camera,
)

try:
    from ovggt.utils.pose_enc import (
        compose_absolute_from_relative,
        relative_from_absolute_pose_encoding,
    )
except ImportError:
    def _inverse_se3(matrix_4x4: torch.Tensor) -> torch.Tensor:
        rot = matrix_4x4[..., :3, :3]
        trans = matrix_4x4[..., :3, 3:]
        rot_t = rot.transpose(-1, -2)
        inv = torch.eye(
            4,
            dtype=matrix_4x4.dtype,
            device=matrix_4x4.device,
        ).expand(matrix_4x4.shape[:-2] + (4, 4)).clone()
        inv[..., :3, :3] = rot_t
        inv[..., :3, 3:] = -torch.matmul(rot_t, trans)
        return inv

    def _world_to_camera_to_pose_encoding(
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size_hw,
        pose_encoding_type: str,
    ) -> torch.Tensor:
        return extri_intri_to_pose_encoding(
            world_to_camera[..., :3, :4],
            intrinsics,
            image_size_hw=image_size_hw,
            pose_encoding_type=pose_encoding_type,
        )

    def compose_absolute_from_relative(
        anchor_abs_pose_encoding: torch.Tensor,
        relative_pose_encoding: torch.Tensor,
        image_size_hw,
    ) -> torch.Tensor:
        anchor_w2c = pose_encoding_to_world_to_camera(
            anchor_abs_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
        )
        relative_w2c = pose_encoding_to_world_to_camera(
            relative_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=REL_POSE_ENCODING,
        )
        _, intrinsics = pose_encoding_to_extri_intri(
            relative_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=REL_POSE_ENCODING,
            build_intrinsics=True,
        )
        current_w2c = torch.matmul(relative_w2c, anchor_w2c)
        return _world_to_camera_to_pose_encoding(
            current_w2c,
            intrinsics=intrinsics,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
        )

    def relative_from_absolute_pose_encoding(
        anchor_abs_pose_encoding: torch.Tensor,
        current_abs_pose_encoding: torch.Tensor,
        image_size_hw,
    ) -> torch.Tensor:
        anchor_w2c = pose_encoding_to_world_to_camera(
            anchor_abs_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
        )
        current_w2c = pose_encoding_to_world_to_camera(
            current_abs_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
        )
        _, intrinsics = pose_encoding_to_extri_intri(
            current_abs_pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=ABS_POSE_ENCODING,
            build_intrinsics=True,
        )
        relative_w2c = torch.matmul(current_w2c, _inverse_se3(anchor_w2c))
        return _world_to_camera_to_pose_encoding(
            relative_w2c,
            intrinsics=intrinsics,
            image_size_hw=image_size_hw,
            pose_encoding_type=REL_POSE_ENCODING,
        )


@dataclass
class OVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[Any] = None
    keyframe_packets: Optional[List[KeyframePacket]] = None
    keyframe_schedule: Optional[List[Any]] = None
    distill_loss: Optional[torch.Tensor] = None


class OVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        total_budget=200000,
        per_layer_budget=None,
        camera_budget=384,
        eviction_strategy="repr_shift_spatial",
        intra_frame_keep_ratio=1.0,
        spatial_alpha=0.5,
        importance_weight: float = 0.5,
        mode: str = "legacy",
        frontend_pose_encoding_type: str = ABS_POSE_ENCODING,
        frontend_cache_config: Optional[FrontendCacheConfig] = None,
        keyframe_switch_config: Optional[KeyframeSwitchConfig] = None,
        aggregator_kwargs: Optional[dict] = None,
        camera_head_kwargs: Optional[dict] = None,
        depth_head_kwargs: Optional[dict] = None,
        point_head_kwargs: Optional[dict] = None,
        enable_track_head: bool = True,
        camera_num_iters: int = 4,
        anchor_overflow_policy: str = "recent",
        frontend_head_checkpointing: bool = False,
    ):
        super().__init__()

        aggregator_kwargs = aggregator_kwargs or {}
        camera_head_kwargs = camera_head_kwargs or {}
        depth_head_kwargs = depth_head_kwargs or {}
        point_head_kwargs = point_head_kwargs or {}

        if mode not in {"legacy", "frontend_train", "frontend_eval"}:
            raise ValueError(f"Unsupported OVGGT mode: {mode}")
        if anchor_overflow_policy not in {"recent", "global_plus_recent"}:
            raise ValueError(
                "Unsupported anchor_overflow_policy: "
                f"{anchor_overflow_policy}. Expected one of: recent, global_plus_recent."
            )

        self.intra_frame_keep_ratio = intra_frame_keep_ratio
        self.spatial_alpha = spatial_alpha
        self.mode = mode
        self.frontend_pose_encoding_type = frontend_pose_encoding_type
        frontend_cache_config_provided = frontend_cache_config is not None
        self.frontend_cache_config = frontend_cache_config or FrontendCacheConfig()
        if self.mode in {"frontend_train", "frontend_eval"} and not frontend_cache_config_provided:
            self.frontend_cache_config.enabled = True
        self.keyframe_switch_config = keyframe_switch_config
        self._keyframe_switch_config_provided = keyframe_switch_config is not None

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            eviction_strategy=eviction_strategy,
            intra_frame_keep_ratio=intra_frame_keep_ratio,
            spatial_alpha=spatial_alpha,
            **aggregator_kwargs,
        )

        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            total_budget=camera_budget,
            **camera_head_kwargs,
        )

        if "intermediate_layer_idx" not in depth_head_kwargs and self.aggregator.depth < 24:
            depth_head_kwargs["intermediate_layer_idx"] = self._dpt_indices_for_depth(self.aggregator.depth)
        if "intermediate_layer_idx" not in point_head_kwargs and self.aggregator.depth < 24:
            point_head_kwargs["intermediate_layer_idx"] = self._dpt_indices_for_depth(self.aggregator.depth)

        depth_kwargs = {
            "output_dim": 2,
            "activation": "exp",
            "conf_activation": "expp1",
        }
        depth_kwargs.update(depth_head_kwargs)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, **depth_kwargs)

        point_kwargs = {
            "output_dim": 4,
            "activation": "inv_log",
            "conf_activation": "expp1",
        }
        point_kwargs.update(point_head_kwargs)
        self.point_head = DPTHead(dim_in=2 * embed_dim, **point_kwargs)

        self.track_head = (
            self._build_track_head(embed_dim=embed_dim, patch_size=patch_size)
            if enable_track_head
            else None
        )

        self.per_layer_budget = int(
            per_layer_budget
            if per_layer_budget is not None
            else max(int(total_budget) // self.aggregator.depth, 0)
        )
        self.total_budget = int(total_budget)
        self.eviction_strategy = eviction_strategy
        self.importance_weight = importance_weight
        self.camera_num_iters = camera_num_iters
        self.anchor_overflow_policy = anchor_overflow_policy
        self.frontend_head_checkpointing = frontend_head_checkpointing
        self._gradient_checkpointing = False
        self._set_anchor_overflow_policy(anchor_overflow_policy)

    @staticmethod
    def _disabled_autocast_context():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.amp.autocast(device_type=device_type, enabled=False)

    def gradient_checkpointing_enable(self, enable: bool = True):
        self._gradient_checkpointing = bool(enable)
        enable = bool(enable)

        patch_embed = getattr(self.aggregator, "patch_embed", None)
        if patch_embed is not None:
            if hasattr(patch_embed, "use_checkpoint"):
                patch_embed.use_checkpoint = enable
            if hasattr(patch_embed, "use_reentrant"):
                patch_embed.use_reentrant = False

        for block in getattr(self.aggregator, "frame_blocks", []):
            if hasattr(block, "use_checkpoint"):
                block.use_checkpoint = enable
        for block in getattr(self.aggregator, "global_blocks", []):
            if hasattr(block, "use_checkpoint"):
                block.use_checkpoint = enable

    @staticmethod
    def _build_track_head(embed_dim: int, patch_size: int):
        from ovggt.heads.track_head import TrackHead

        return TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    @staticmethod
    def _dpt_indices_for_depth(depth: int) -> List[int]:
        if depth <= 0:
            raise ValueError(f"Aggregator depth must be positive, got {depth}")
        if depth == 1:
            return [0, 0, 0, 0]
        return [
            min(int(round(pos * (depth - 1) / 3)), depth - 1)
            for pos in range(4)
        ]

    def _set_anchor_overflow_policy(self, policy: str) -> None:
        for module in self.modules():
            if hasattr(module, "anchor_overflow_policy"):
                module.anchor_overflow_policy = policy

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        frame_processor: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        return_views: bool = False,
    ):
        if self.mode == "frontend_train":
            return self.forward_frontend_train(
                views=views,
                query_points=query_points,
                frame_processor=frame_processor,
                cache_results=cache_results,
                return_views=return_views,
            )

        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            _, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    "pts3d_in_other_view": predictions["world_points"][:, s],
                    "conf": predictions["world_points_conf"][:, s],
                    "depth": predictions["depth"][:, s],
                    "depth_conf": predictions["depth_conf"][:, s],
                    "camera_pose": predictions["pose_enc"][:, s, :],
                    **({"valid_mask": views[s]["valid_mask"]} if "valid_mask" in views[s] else {}),
                    **(
                        {
                            "track": predictions["track"][:, s],
                            "vis": predictions["vis"][:, s],
                            "track_conf": predictions["conf"][:, s],
                        }
                        if "track" in predictions
                        else {}
                    ),
                }
                ress.append(res)
        return OVGGTOutput(ress=ress, views=views)

    def forward_frontend_train(
        self,
        views,
        query_points: torch.Tensor = None,
        frame_processor: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        return_views: bool = False,
    ):
        return self._inference_frontend(
            frames=views,
            query_points=query_points,
            frame_writer=frame_processor,
            cache_results=cache_results,
            history_anchor_strategy="fixed_interval",
            anchor_interval=8,
            max_anchors=3,
            coverage_threshold=0.2,
            move_to_cpu=False,
            export_keyframe_packets=False,
            return_views=return_views,
            store_full_keyframe_schedule=False,
        )

    def inference(
        self,
        frames,
        query_points: torch.Tensor = None,
        past_key_values=None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        history_anchor_strategy: Optional[str] = None,
        anchor_interval: Optional[int] = None,
        min_anchor_interval: Optional[int] = 100,
        window_protect_frames: int = 0,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        move_to_cpu: bool = True,
        return_views: bool = True,
    ):
        resolved_history_anchor_strategy = self._resolve_history_anchor_strategy(history_anchor_strategy)
        resolved_anchor_interval = self._resolve_anchor_interval(anchor_interval)
        if self.mode in {"frontend_train", "frontend_eval"} and self.frontend_cache_config.enabled:
            return self._inference_frontend(
                frames=frames,
                query_points=query_points,
                frame_writer=frame_writer,
                cache_results=cache_results,
                history_anchor_strategy=resolved_history_anchor_strategy,
                anchor_interval=resolved_anchor_interval,
                max_anchors=max_anchors,
                coverage_threshold=coverage_threshold,
                move_to_cpu=move_to_cpu,
                export_keyframe_packets=None,
                return_views=return_views,
            )
        return self._inference_legacy(
            frames=frames,
            query_points=query_points,
            frame_writer=frame_writer,
            cache_results=cache_results,
            history_anchor_strategy=resolved_history_anchor_strategy,
            anchor_interval=resolved_anchor_interval,
            min_anchor_interval=min_anchor_interval,
            window_protect_frames=window_protect_frames,
            max_anchors=max_anchors,
            coverage_threshold=coverage_threshold,
            anchor_keep_ratio=anchor_keep_ratio,
            move_to_cpu=move_to_cpu,
            return_views=return_views,
        )

    def _inference_frontend(
        self,
        frames,
        query_points: torch.Tensor = None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        history_anchor_strategy: str = "fixed_interval",
        anchor_interval: int = 48,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        move_to_cpu: bool = True,
        export_keyframe_packets: Optional[bool] = None,
        return_views: bool = True,
        store_full_keyframe_schedule: Optional[bool] = None,
    ):
        self._validate_frontend_batch_size(frames)
        frontend_keyframe_config = self._build_frontend_keyframe_config(
            history_anchor_strategy,
            anchor_interval,
            max_anchors,
            coverage_threshold,
        )
        B = self._frame_batch_size(frames[0]["img"])
        for frame in frames[1:]:
            if self._frame_batch_size(frame["img"]) != B:
                raise ValueError("All frames must have same batch size")

        keyframe_managers = [
            FrontendKeyframeManager(frontend_keyframe_config)
            for _ in range(B)
        ]
        cache_states = [
            [
                LayerCacheState(max_history_anchors=frontend_keyframe_config.max_history_anchors)
                for _ in range(self.aggregator.depth)
            ]
            for _ in range(B)
        ]
        past_key_values_camera = [
            [None] * self.camera_head.trunk_depth
            for _ in range(B)
        ]
        per_layer_budget = self.per_layer_budget
        importance_weight = self.importance_weight
        intra_frame_keep_ratio = self.aggregator.intra_frame_keep_ratio

        img_h, img_w = self._infer_image_hw(frames)
        patch_grid_size = (img_h // self.aggregator.patch_size, img_w // self.aggregator.patch_size)
        export_packets = self.frontend_cache_config.export_keyframe_packets
        if export_keyframe_packets is not None:
            export_packets = export_keyframe_packets
        if store_full_keyframe_schedule is None:
            store_full_keyframe_schedule = self.mode == "frontend_eval"
        keyframe_packets: List[KeyframePacket] = []
        keyframe_schedule = []
        all_ress = []
        processed_frames = []
        current_query_points = query_points

        def maybe_checkpoint_head(forward_fn, *inputs):
            if self._gradient_checkpointing and self.frontend_head_checkpointing and self.training:
                return checkpoint(forward_fn, *inputs, use_reentrant=False)
            return forward_fn(*inputs)

        total_distill_loss = None

        for i, frame in enumerate(frames):
            images_all = self._frame_image_to_sequence(frame["img"])

            saved_last_scores = self.aggregator.last_scores.clone()
            frame_agg_outputs = []
            frame_pending_updates = []
            frame_distill_losses = []
            ps = None
            for b in range(B):
                images_b = images_all[b:b + 1]
                if b > 0:
                    self.aggregator.last_scores = saved_last_scores.clone()
                agg_tokens, ps, cs_b, pending, fdl = self.aggregator(
                    images_b,
                    cache_states=cache_states[b],
                    use_cache=True,
                    past_frame_idx=i,
                    per_layer_budget=per_layer_budget,
                    importance_weight=importance_weight,
                    frontend_cache_config=self.frontend_cache_config,
                )
                frame_agg_outputs.append(agg_tokens)
                frame_pending_updates.append(pending)
                if fdl is not None:
                    frame_distill_losses.append(fdl)
                cache_states[b] = cs_b
            self.aggregator.last_scores = saved_last_scores
            patch_start_idx = ps

            if frame_distill_losses:
                avg_fdl = sum(frame_distill_losses) / len(frame_distill_losses)
                total_distill_loss = avg_fdl if total_distill_loss is None else total_distill_loss + avg_fdl

            aggregated_tokens = []
            for layer_idx in range(len(frame_agg_outputs[0])):
                aggregated_tokens.append(
                    torch.cat([bo[layer_idx] for bo in frame_agg_outputs], dim=0)
                )

            pose_enc_batch = []
            rel_pose_enc_batch = []
            with self._disabled_autocast_context():
                for b in range(B):
                    camera_anchor_token_count = (
                        None
                        if i == 0
                        else keyframe_managers[b].get_num_anchor_frames() * self.camera_num_iters
                    )
                    pose_enc_dict_b, past_key_values_camera[b] = self.camera_head(
                        [agg[b:b + 1] for agg in aggregated_tokens],
                        num_iterations=self.camera_num_iters,
                        past_key_values_camera=past_key_values_camera[b],
                        use_cache=True,
                        anchor_token_count=camera_anchor_token_count,
                        pose_encoding_type=self._camera_pose_encoding_type_for_frontend(),
                        return_pose_predictions=True,
                        return_last_pose_only=True,
                    )
                    pose_enc_batch.append(pose_enc_dict_b["abs_pose_enc"][:, 0, :])
                    rel_pose_enc_batch.append(pose_enc_dict_b["rel_pose_enc"][:, 0, :])
                abs_pose_enc = torch.cat(pose_enc_batch, dim=0)
                rel_pose_enc = torch.cat(rel_pose_enc_batch, dim=0)
                if i == 0 or self.frontend_pose_encoding_type == ABS_POSE_ENCODING:
                    camera_pose = abs_pose_enc
                    camera_pose_rel = relative_from_absolute_pose_encoding(
                        camera_pose.unsqueeze(1),
                        camera_pose.unsqueeze(1),
                        image_size_hw=(img_h, img_w),
                    )[:, 0, :]
                else:
                    active_poses = torch.stack(
                        [km.get_active_pose_encoding() for km in keyframe_managers],
                        dim=0,
                    )
                    camera_pose = compose_absolute_from_relative(
                        active_poses.unsqueeze(1),
                        rel_pose_enc.unsqueeze(1),
                        image_size_hw=(img_h, img_w),
                    )[:, 0, :]
                    camera_pose_rel = rel_pose_enc

            def depth_head_forward(*layer_tokens):
                return self.depth_head(
                    list(layer_tokens),
                    images=images_all,
                    patch_start_idx=patch_start_idx,
                )

            depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens)
            depth = depth[:, 0]
            depth_conf = depth_conf[:, 0]

            def point_head_forward(*layer_tokens):
                return self.point_head(
                    list(layer_tokens),
                    images=images_all,
                    patch_start_idx=patch_start_idx,
                )

            pts3d, pts3d_conf = maybe_checkpoint_head(point_head_forward, *aggregated_tokens)
            pts3d = pts3d[:, 0]
            pts3d_conf = pts3d_conf[:, 0]

            track = vis = track_conf = None
            if self.track_head is not None and current_query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens,
                    images=images_all,
                    patch_start_idx=patch_start_idx,
                    query_points=current_query_points,
                )
                track = track_list[-1][:, 0]
                current_query_points = track
                vis = vis[:, 0]
                track_conf = conf[:, 0]

            events = []
            for b in range(B):
                event = keyframe_managers[b].update(
                    frame_idx=i,
                    depth=depth[b],
                    pose_abs_enc=camera_pose[b],
                    image_size_hw=(img_h, img_w),
                )
                events.append(event)
            if store_full_keyframe_schedule:
                keyframe_schedule.append(events[0])
            else:
                keyframe_schedule.append(
                    {
                        "event_type": str(events[0].event_type),
                        "frame_idx": int(events[0].frame_idx),
                        "keyframe_id": int(events[0].keyframe_id),
                        "anchor_slot": int(events[0].anchor_slot),
                        "num_anchor_frames": int(events[0].num_anchor_frames),
                    }
                )

            for b in range(B):
                current_keyframe_id = keyframe_managers[b].get_active_keyframe_id()
                current_local_to_world = keyframe_managers[b].get_active_local_to_world()
                frame_metadata_base = None

                for layer_idx in range(self.aggregator.depth):
                    pending = frame_pending_updates[b][layer_idx]
                    if pending is None:
                        continue
                    is_fifo_swap = str(getattr(events[b], "event_type", None)).endswith("FIFO_SWAP")
                    if is_fifo_swap and (
                        self.frontend_cache_config.fifo_keep_topk > 0
                        or self.frontend_cache_config.learned_fifo_keep_count
                    ):
                        demoted_slot = getattr(events[b], "demoted_slot", None)
                        if demoted_slot is not None:
                            cache_state = cache_states[b][layer_idx]
                            if self.frontend_cache_config.learned_fifo_keep_count:
                                raise ValueError(
                                    "learned_fifo_keep_count=True requires a count head, "
                                    "which is not available in this OVGGT build"
                                )
                            keep_count = int(self.frontend_cache_config.fifo_keep_topk)
                            cache_state.protect_topk_on_demotion_(
                                demoted_slot=demoted_slot,
                                keep_count=keep_count,
                                token_scorer=(
                                    self.aggregator.token_scorers[layer_idx]
                                    if self.aggregator.token_scorers is not None
                                    else None
                                ),
                                layer_id=layer_idx,
                                current_frame_id=i,
                                fifo_probe=getattr(self, "_oracle_fifo_probe", None),
                                batch_index=b,
                                cache_budget=self.per_layer_budget,
                                max_protected=(
                                    int(
                                        self.frontend_cache_config.max_protected_ratio
                                        * self.per_layer_budget
                                    )
                                    if self.frontend_cache_config.max_protected_ratio < 1.0
                                    else None
                                ),
                                fifo_ring_capacity=(
                                    int(
                                        self.frontend_cache_config.fifo_protected_ring_ratio
                                        * self.per_layer_budget
                                    )
                                    if self.frontend_cache_config.fifo_protected_ring_ratio > 0.0
                                    else None
                                ),
                                global_anchor_keyframe_id=(
                                    keyframe_managers[b].global_anchor["keyframe_id"]
                                    if keyframe_managers[b].global_anchor is not None
                                    else 0
                                ),
                            )
                    cache_states[b][layer_idx].apply_keyframe_event_(events[b])
                    if (
                        frame_metadata_base is None
                        or frame_metadata_base.frame_id.shape[1] != pending.importance_current.shape[1]
                    ):
                        frame_metadata_base = build_frame_token_metadata_base(
                            depth=depth[b:b + 1],
                            depth_conf=depth_conf[b:b + 1],
                            pose_enc=camera_pose[b:b + 1],
                            image_size_hw=(img_h, img_w),
                            patch_size=self.aggregator.patch_size,
                            patch_start_idx=patch_start_idx,
                            frame_id=i,
                            keyframe_id=current_keyframe_id,
                            slot_id=current_keyframe_id,
                            anchor_slot=events[b].anchor_slot,
                            total_tokens=pending.importance_current.shape[1],
                            active_local_to_world=current_local_to_world,
                        )
                    current_metadata = frame_metadata_base.with_importance(pending.importance_current)
                    score = cache_states[b][layer_idx].commit_pending_update_(
                        pending_update=pending,
                        current_metadata=current_metadata,
                        config=self.frontend_cache_config,
                        intra_frame_keep_ratio=intra_frame_keep_ratio,
                        attn_module=self.aggregator.global_blocks[layer_idx].attn,
                        token_scorer=(
                            self.aggregator.token_scorers[layer_idx]
                            if self.aggregator.token_scorers is not None
                            else None
                        ),
                        layer_id=layer_idx,
                        eviction_probe=getattr(self, "_oracle_eviction_probe", None),
                        batch_index=b,
                        dedup_probe=getattr(self, "_oracle_dedup_probe", None),
                        dedup_replay_probe=getattr(self, "_oracle_dedup_replay_probe", None),
                    )
                    if score is not None:
                        self.aggregator.last_scores[layer_idx] = score

            for b in range(B):
                past_key_values_camera[b] = self.camera_head.apply_keyframe_event(
                    past_key_values_camera[b],
                    events[b],
                    num_cam_iters=self.camera_num_iters,
                )

            if export_packets:
                for b in range(B):
                    if events[b].anchor_slot >= 0:
                        slot_id_b = keyframe_managers[b].get_active_keyframe_id()
                        ltw_b = keyframe_managers[b].get_active_local_to_world()
                        pkt_metadata = build_frame_token_metadata_base(
                            depth=depth[b:b + 1],
                            depth_conf=depth_conf[b:b + 1],
                            pose_enc=camera_pose[b:b + 1],
                            image_size_hw=(img_h, img_w),
                            patch_size=self.aggregator.patch_size,
                            patch_start_idx=patch_start_idx,
                            frame_id=i,
                            keyframe_id=slot_id_b,
                            slot_id=slot_id_b,
                            anchor_slot=events[b].anchor_slot,
                            total_tokens=(
                                frame_pending_updates[b][-1].importance_current.shape[1]
                                if frame_pending_updates[b][-1] is not None
                                else 0
                            ),
                            active_local_to_world=ltw_b,
                        )
                        patch_features = aggregated_tokens[-1][b:b + 1, :, patch_start_idx:]
                        keyframe_packets.append(
                            KeyframePacket(
                                frame_idx=i,
                                keyframe_id=slot_id_b,
                                anchor_slot=events[b].anchor_slot,
                                pose_abs=camera_pose[b].detach().cpu(),
                                local_to_world=ltw_b.detach().cpu(),
                                patch_local_xyz=pkt_metadata.slot_local_xyz[:, patch_start_idx:].detach().cpu(),
                                patch_depth_conf=pkt_metadata.depth_conf[:, patch_start_idx:].detach().cpu(),
                                patch_features=patch_features.detach().cpu(),
                            )
                        )

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                "camera_pose_rel": camera_pose_rel,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if self.track_head is not None and current_query_points is not None
                    else {}
                ),
            }
            if frame_writer is not None:
                frame_writer(i, frame, res_gpu)

            if cache_results:
                res_out = self._maybe_move_dict_to_cpu(res_gpu) if move_to_cpu else res_gpu
                all_ress.append(res_out)
                if return_views:
                    processed_frames.append(self._maybe_move_dict_to_cpu(frame) if move_to_cpu else frame)

            del aggregated_tokens
            del frame_pending_updates
            del res_gpu

        return OVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if (cache_results and return_views) else None,
            keyframe_packets=keyframe_packets if export_packets else None,
            keyframe_schedule=keyframe_schedule,
            distill_loss=total_distill_loss,
        )

    def _inference_legacy(
        self,
        frames,
        query_points: torch.Tensor = None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        history_anchor_strategy: str = "none",
        anchor_interval: int = 250,
        min_anchor_interval: Optional[int] = 100,
        window_protect_frames: int = 0,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        move_to_cpu: bool = True,
        return_views: bool = True,
    ):
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        per_layer_budget = self.per_layer_budget
        importance_weight = self.importance_weight

        img_h, img_w = self._infer_image_hw(frames)
        patch_size = self.aggregator.patch_size
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        tokens_per_frame = 1 + 4 + num_patches
        window_token_count = max(int(window_protect_frames), 0) * tokens_per_frame

        anchor_config = HistoryAnchorConfig(
            strategy=history_anchor_strategy,
            interval=anchor_interval,
            min_anchor_interval=min_anchor_interval,
            max_anchors=max_anchors,
            coverage_threshold=coverage_threshold,
            anchor_keep_ratio=anchor_keep_ratio,
        )
        anchor_manager = HistoryAnchorManager(anchor_config, tokens_per_frame)
        anchor_manager.image_size_hw = (img_h, img_w)

        all_ress = []
        processed_frames = []
        current_query_points = query_points

        def _select_anchor_token_indices(conf_map: torch.Tensor) -> Optional[torch.Tensor]:
            if conf_map is None:
                return None

            special_count = self.aggregator.patch_start_idx
            anchor_chunk = max(int(tokens_per_frame * anchor_keep_ratio), 1)
            if anchor_chunk <= special_count:
                return torch.arange(anchor_chunk, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)

            keep_patches = anchor_chunk - special_count
            patch_h = img_h // patch_size
            patch_w = img_w // patch_size

            if conf_map.dim() == 4:
                conf_map = conf_map.squeeze(1)
            if conf_map.dim() != 3:
                return None

            pooled = F.adaptive_avg_pool2d(conf_map.unsqueeze(1), (patch_h, patch_w)).squeeze(1)
            flat = pooled.reshape(conf_map.shape[0], -1)
            keep_patches = min(keep_patches, flat.shape[1])
            if keep_patches <= 0:
                return torch.arange(anchor_chunk, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)

            topk = torch.topk(flat, k=keep_patches, dim=1).indices
            special_indices = torch.arange(special_count, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)
            return torch.cat([special_indices, topk + special_count], dim=1)

        for i, frame in enumerate(frames):
            fixed_interval_registered = False
            fixed_interval_is_fifo = False
            if history_anchor_strategy == "fixed_interval":
                should_register, is_fifo, reason = anchor_manager.should_become_anchor(frame_idx=i)
                if should_register:
                    anchor_manager.register_anchor(i)
                    fixed_interval_registered = True
                    fixed_interval_is_fifo = is_fifo
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

            anchor_token_count = anchor_manager.get_protected_token_count()
            images = self._frame_image_to_sequence(frame["img"])

            aggregated_tokens, patch_start_idx, past_key_values = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
                per_layer_budget=per_layer_budget,
                anchor_token_count=anchor_token_count,
                importance_weight=importance_weight,
                window_token_count=window_token_count,
            )

            with self._disabled_autocast_context():
                num_cam_iters = self.camera_num_iters
                total_anchors = anchor_manager.get_num_anchors()
                camera_anchor_token_count = total_anchors * num_cam_iters

                pose_enc, past_key_values_camera = self.camera_head(
                    aggregated_tokens,
                    num_iterations=num_cam_iters,
                    past_key_values_camera=past_key_values_camera,
                    use_cache=True,
                    anchor_token_count=camera_anchor_token_count,
                )
                pose_enc = pose_enc[-1]
                camera_pose = pose_enc[:, 0, :]

                if fixed_interval_registered:
                    past_key_values_camera = self.camera_head.sync_anchor_change(
                        past_key_values_camera,
                        anchor_token_count=camera_anchor_token_count,
                        num_cam_iters=num_cam_iters,
                        is_fifo=fixed_interval_is_fifo,
                    )

                depth, depth_conf = self.depth_head(
                    aggregated_tokens,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                depth = depth[:, 0]
                depth_conf = depth_conf[:, 0]

                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                pts3d = pts3d[:, 0]
                pts3d_conf = pts3d_conf[:, 0]

                track = vis = track_conf = None
                if self.track_head is not None and current_query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        query_points=current_query_points,
                    )
                    track = track_list[-1][:, 0]
                    current_query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            if fixed_interval_registered:
                anchor_token_indices = _select_anchor_token_indices(
                    pts3d_conf if self.point_head is not None else None
                )
                past_key_values = self.aggregator.sync_anchor_change(
                    past_key_values,
                    anchor_token_count=anchor_token_count,
                    tokens_per_frame=tokens_per_frame,
                    anchor_keep_ratio=anchor_keep_ratio,
                    anchor_token_indices=anchor_token_indices,
                    is_fifo=fixed_interval_is_fifo,
                )

            if history_anchor_strategy == "coverage":
                should_register, is_fifo, reason, _ = anchor_manager.should_become_anchor_coverage(
                    frame_idx=i,
                    current_depth=depth[0],
                    current_pose=camera_pose[0],
                )
                if should_register:
                    anchor_manager.register_anchor_coverage(i, depth[0], camera_pose[0])
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")
                    anchor_token_count_post = anchor_manager.get_protected_token_count()
                    anchor_token_indices = _select_anchor_token_indices(
                        pts3d_conf if self.point_head is not None else None
                    )
                    past_key_values = self.aggregator.sync_anchor_change(
                        past_key_values,
                        anchor_token_count=anchor_token_count_post,
                        tokens_per_frame=tokens_per_frame,
                        anchor_keep_ratio=anchor_keep_ratio,
                        anchor_token_indices=anchor_token_indices,
                        is_fifo=is_fifo,
                    )
                    cam_anchor_count_post = anchor_manager.get_num_anchors() * self.camera_num_iters
                    past_key_values_camera = self.camera_head.sync_anchor_change(
                        past_key_values_camera,
                        anchor_token_count=cam_anchor_count_post,
                        num_cam_iters=self.camera_num_iters,
                        is_fifo=is_fifo,
                    )

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **(
                    {
                        "valid_mask": (
                            frame["valid_mask"].to(pts3d.device)
                            if isinstance(frame["valid_mask"], torch.Tensor)
                            else frame["valid_mask"]
                        )
                    }
                    if "valid_mask" in frame
                    else {}
                ),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if self.track_head is not None and current_query_points is not None
                    else {}
                ),
            }
            res_out = self._maybe_move_dict_to_cpu(res_gpu) if move_to_cpu else res_gpu
            if frame_writer is not None:
                frame_writer(i, frame, res_out)

            if cache_results:
                all_ress.append(res_out)
                if return_views:
                    processed_frames.append(
                        self._maybe_move_dict_to_cpu(frame) if move_to_cpu else frame
                    )

        return OVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if (cache_results and return_views) else None,
        )

    def _infer_image_hw(self, frames):
        img_h, img_w = 392, 518
        if len(frames) > 0 and "img" in frames[0]:
            sample_img = frames[0]["img"]
            if sample_img.dim() == 3:
                img_h, img_w = sample_img.shape[1], sample_img.shape[2]
            elif sample_img.dim() == 4:
                img_h, img_w = sample_img.shape[2], sample_img.shape[3]
        return img_h, img_w

    @staticmethod
    def _frame_image_to_sequence(frame_img: torch.Tensor) -> torch.Tensor:
        if frame_img.dim() == 3:
            return frame_img.unsqueeze(0).unsqueeze(1)
        if frame_img.dim() == 4:
            return frame_img.unsqueeze(1)
        raise ValueError(f"Expected frame image with 3 or 4 dims, got shape={tuple(frame_img.shape)}")

    @staticmethod
    def _frame_batch_size(frame_img: torch.Tensor) -> int:
        if frame_img.dim() == 3:
            return 1
        if frame_img.dim() == 4:
            return int(frame_img.shape[0])
        raise ValueError(f"Expected frame image with 3 or 4 dims, got shape={tuple(frame_img.shape)}")

    def _validate_frontend_batch_size(self, frames) -> None:
        if not frames:
            return
        ref_batch_size = self._frame_batch_size(frames[0]["img"])
        for frame_idx, frame in enumerate(frames):
            cur_batch_size = self._frame_batch_size(frame["img"])
            if cur_batch_size != ref_batch_size:
                raise ValueError(
                    f"Inconsistent frontend frame batch size at frame {frame_idx}: "
                    f"expected {ref_batch_size}, got {cur_batch_size}"
                )

    def _build_frontend_keyframe_config(
        self,
        history_anchor_strategy: str,
        anchor_interval: int,
        max_anchors: int,
        coverage_threshold: float,
    ) -> KeyframeSwitchConfig:
        if self._keyframe_switch_config_provided and self.keyframe_switch_config is not None:
            return self.keyframe_switch_config
        strategy = (
            history_anchor_strategy
            if history_anchor_strategy in {"fixed_interval", "coverage"}
            else "fixed_interval"
        )
        schedule_interval = max(int(anchor_interval), 1)
        max_history_anchors = max(int(max_anchors), 0)
        if self.mode == "frontend_train":
            return KeyframeSwitchConfig(
                strategy="fixed_interval",
                coverage_threshold=coverage_threshold,
                max_history_anchors=max_history_anchors,
                interval=schedule_interval,
                coverage_monitor_only=True,
                forced_keyframe_frames=(0, schedule_interval, 2 * schedule_interval),
            )
        return KeyframeSwitchConfig(
            strategy=strategy,
            coverage_threshold=coverage_threshold,
            max_history_anchors=max_history_anchors,
            interval=schedule_interval,
            coverage_monitor_only=True,
        )

    def _resolve_history_anchor_strategy(self, history_anchor_strategy: Optional[str]) -> str:
        if history_anchor_strategy is not None:
            return history_anchor_strategy
        if self.mode in {"frontend_train", "frontend_eval"}:
            return "fixed_interval"
        return "coverage"

    def _resolve_anchor_interval(self, anchor_interval: Optional[int]) -> int:
        if anchor_interval is not None:
            return anchor_interval
        if self.mode == "frontend_train":
            return 8
        if self.mode == "frontend_eval":
            return 8
        return 250

    def _camera_pose_encoding_type_for_frontend(self) -> str:
        if self.frontend_pose_encoding_type not in {ABS_POSE_ENCODING, REL_POSE_ENCODING}:
            raise ValueError(
                f"Unsupported frontend pose encoding type: {self.frontend_pose_encoding_type}"
            )
        return self.frontend_pose_encoding_type

    @staticmethod
    def _maybe_move_dict_to_cpu(payload: dict) -> dict:
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in payload.items()
        }
