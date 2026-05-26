import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from torch.utils.checkpoint import checkpoint

from model_hub_compat import ModelOutput, PyTorchModelHubMixin

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
    compose_absolute_from_relative,
    relative_from_absolute_pose_encoding,
)


@dataclass
class OVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[Any] = None
    keyframe_packets: Optional[List[KeyframePacket]] = None
    keyframe_schedule: Optional[List[Any]] = None
    distill_loss: Optional[torch.Tensor] = None  # TokenScorer distillation loss


class OVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        total_budget=200000,
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
        use_token_scorer: bool = False,
        scorer_bottleneck_dim: Optional[int] = None,
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
        self.frontend_cache_config = frontend_cache_config or FrontendCacheConfig()
        if self.mode in {"frontend_train", "frontend_eval"}:
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

        if use_token_scorer:
            self.aggregator.init_token_scorers(
                embed_dim=embed_dim,
                bottleneck_dim=scorer_bottleneck_dim or embed_dim // 4,
            )

        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            total_budget=camera_budget,
            **camera_head_kwargs,
        )

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

        self.total_budget = total_budget
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
        """Enable or disable gradient checkpointing for memory-efficient training.

        When enabled, intermediate activations are recomputed during backward pass,
        trading compute for memory. This can reduce activation memory by ~40-60%.

        Implementation note:
        - Enables checkpointing in patch embedding ViT blocks.
        - Enables checkpointing in aggregator blocks with a safe strategy:
          full residual checkpoint for non-cache paths and MLP-only checkpoint
          for cache-enabled paths.
        """
        self._gradient_checkpointing = enable
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

    def _set_anchor_overflow_policy(self, policy: str) -> None:
        for module in self.modules():
            if hasattr(module, "anchor_overflow_policy"):
                module.anchor_overflow_policy = policy

    def load_state_dict(self, state_dict, strict: bool = True):
        upgraded_state_dict = dict(state_dict)
        self._upgrade_camera_head_state_dict(upgraded_state_dict)
        # TokenScorer compatibility: handle scorer keys in/out of checkpoint
        scorer_keys = [k for k in upgraded_state_dict if 'token_scorers' in k]
        model_has_scorer = hasattr(self.aggregator, 'token_scorers') and self.aggregator.token_scorers is not None
        if scorer_keys and not model_has_scorer:
            for k in scorer_keys:
                del upgraded_state_dict[k]
        if model_has_scorer and not scorer_keys:
            return super().load_state_dict(upgraded_state_dict, strict=False)
        return super().load_state_dict(upgraded_state_dict, strict=strict)

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
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        move_to_cpu: bool = True,  # Whether to move results to CPU
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
        cache_states = [
            LayerCacheState(max_history_anchors=frontend_keyframe_config.max_history_anchors)
            for _ in range(self.aggregator.depth)
        ]
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
        importance_weight = self.importance_weight
        keyframe_manager = FrontendKeyframeManager(frontend_keyframe_config)

        img_h, img_w = self._infer_image_hw(frames)
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
            images = self._frame_image_to_sequence(frame["img"])

            aggregated_tokens, patch_start_idx, cache_states, pending_updates, frame_distill_loss = self.aggregator(
                images,
                cache_states=cache_states,
                use_cache=True,
                past_frame_idx=i,
                total_budget=total_budget,
                importance_weight=importance_weight,
                frontend_cache_config=self.frontend_cache_config,
            )

            if frame_distill_loss is not None:
                total_distill_loss = (
                    frame_distill_loss
                    if total_distill_loss is None
                    else total_distill_loss + frame_distill_loss
                )

            with self._disabled_autocast_context():
                camera_anchor_token_count = None if i == 0 else keyframe_manager.get_num_anchor_frames() * self.camera_num_iters
                pose_enc_list, past_key_values_camera = self.camera_head(
                    aggregated_tokens,
                    num_iterations=self.camera_num_iters,
                    past_key_values_camera=past_key_values_camera,
                    use_cache=True,
                    anchor_token_count=camera_anchor_token_count,
                    pose_encoding_type=self._camera_pose_encoding_type_for_frontend(),
                    return_pose_predictions=True,
                    return_last_pose_only=True,
                )
                abs_pose_enc = pose_enc_list["abs_pose_enc"]
                rel_pose_enc = pose_enc_list["rel_pose_enc"]
                if i == 0 or self.frontend_pose_encoding_type == ABS_POSE_ENCODING:
                    camera_pose = abs_pose_enc[:, 0, :]
                    camera_pose_rel = relative_from_absolute_pose_encoding(
                        camera_pose.unsqueeze(1),
                        camera_pose.unsqueeze(1),
                        image_size_hw=(img_h, img_w),
                    )[:, 0, :]
                else:
                    active_pose_encoding = keyframe_manager.get_active_pose_encoding().unsqueeze(0).unsqueeze(0)
                    camera_pose = compose_absolute_from_relative(
                        active_pose_encoding,
                        rel_pose_enc,
                        image_size_hw=(img_h, img_w),
                    )[:, 0, :]
                    camera_pose_rel = rel_pose_enc[:, 0, :]

                def depth_head_forward(*layer_tokens):
                    return self.depth_head(
                        list(layer_tokens),
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )

                depth, depth_conf = maybe_checkpoint_head(depth_head_forward, *aggregated_tokens)
                depth = depth[:, 0]
                depth_conf = depth_conf[:, 0]

                def point_head_forward(*layer_tokens):
                    return self.point_head(
                        list(layer_tokens),
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )

                pts3d, pts3d_conf = maybe_checkpoint_head(point_head_forward, *aggregated_tokens)
                pts3d = pts3d[:, 0]
                pts3d_conf = pts3d_conf[:, 0]

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

            event = keyframe_manager.update(
                frame_idx=i,
                depth=depth[0],
                pose_abs_enc=camera_pose[0],
                image_size_hw=(img_h, img_w),
            )
            if store_full_keyframe_schedule:
                keyframe_schedule.append(event)
            else:
                keyframe_schedule.append(
                    {
                        "event_type": str(event.event_type),
                        "frame_idx": int(event.frame_idx),
                        "keyframe_id": int(event.keyframe_id),
                        "anchor_slot": int(event.anchor_slot),
                        "num_anchor_frames": int(event.num_anchor_frames),
                    }
                )

            current_keyframe_id = keyframe_manager.get_active_keyframe_id()
            current_local_to_world = keyframe_manager.get_active_local_to_world()
            frame_metadata_base = None

            for layer_idx, pending_update in enumerate(pending_updates):
                if pending_update is None:
                    continue
                is_fifo_swap = str(getattr(event, "event_type", None)).endswith("FIFO_SWAP")
                if is_fifo_swap and self.frontend_cache_config.fifo_keep_topk > 0:
                    demoted_slot = getattr(event, "demoted_slot", None)
                    if demoted_slot is not None:
                        cache_states[layer_idx].protect_topk_on_demotion_(
                            demoted_slot=demoted_slot,
                            keep_count=self.frontend_cache_config.fifo_keep_topk,
                        )
                cache_states[layer_idx].apply_keyframe_event_(event)
                if (
                    frame_metadata_base is None
                    or frame_metadata_base.frame_id.shape[1] != pending_update.importance_current.shape[1]
                ):
                    frame_metadata_base = build_frame_token_metadata_base(
                        depth=depth,
                        depth_conf=depth_conf,
                        pose_enc=camera_pose,
                        image_size_hw=(img_h, img_w),
                        patch_size=self.aggregator.patch_size,
                        patch_start_idx=patch_start_idx,
                        frame_id=i,
                        keyframe_id=current_keyframe_id,
                        slot_id=current_keyframe_id,
                        anchor_slot=event.anchor_slot,
                        total_tokens=pending_update.importance_current.shape[1],
                        active_local_to_world=current_local_to_world,
                    )
                current_metadata = frame_metadata_base.with_importance(pending_update.importance_current)
                score = cache_states[layer_idx].commit_pending_update_(
                    pending_update=pending_update,
                    current_metadata=current_metadata,
                    config=self.frontend_cache_config,
                    intra_frame_keep_ratio=self.aggregator.intra_frame_keep_ratio,
                    attn_module=self.aggregator.global_blocks[layer_idx].attn,
                )
                if score is not None:
                    self.aggregator.last_scores[layer_idx] = score
                pending_updates[layer_idx] = None

            if export_packets and event.anchor_slot >= 0 and frame_metadata_base is not None:
                patch_features = aggregated_tokens[-1][:, 0, patch_start_idx:]
                keyframe_packets.append(
                    KeyframePacket(
                        frame_idx=i,
                        keyframe_id=current_keyframe_id,
                        anchor_slot=event.anchor_slot,
                        pose_abs=camera_pose.detach().cpu(),
                        local_to_world=current_local_to_world.detach().cpu(),
                        patch_local_xyz=frame_metadata_base.slot_local_xyz[:, patch_start_idx:].detach().cpu(),
                        patch_depth_conf=frame_metadata_base.depth_conf[:, patch_start_idx:].detach().cpu(),
                        patch_features=patch_features.detach().cpu(),
                    )
                )

            past_key_values_camera = self.camera_head.apply_keyframe_event(
                past_key_values_camera,
                event,
                num_cam_iters=self.camera_num_iters,
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

            # Drop references that are no longer needed in this frame iteration.
            del aggregated_tokens
            del pending_updates
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
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        move_to_cpu: bool = True,  # Add move_to_cpu parameter
        return_views: bool = True,
    ):
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
        importance_weight = self.importance_weight

        img_h, img_w = self._infer_image_hw(frames)
        patch_size = self.aggregator.patch_size
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        tokens_per_frame = 1 + 4 + num_patches

        anchor_config = HistoryAnchorConfig(
            strategy=history_anchor_strategy,
            interval=anchor_interval,
            max_anchors=max_anchors,
            coverage_threshold=coverage_threshold,
            anchor_keep_ratio=anchor_keep_ratio,
        )
        anchor_manager = HistoryAnchorManager(anchor_config, tokens_per_frame)
        anchor_manager.image_size_hw = (img_h, img_w)

        all_ress = []
        processed_frames = []
        current_query_points = query_points

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
                total_budget=total_budget,
                anchor_token_count=anchor_token_count,
                importance_weight=importance_weight,
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
                **({"valid_mask": frame["valid_mask"].to(pts3d.device) if isinstance(frame["valid_mask"], torch.Tensor) else frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if self.track_head is not None and current_query_points is not None
                    else {}
                ),
            }
            res_out = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) and move_to_cpu else v
                for k, v in res_gpu.items()
            }
            if frame_writer is not None:
                frame_writer(i, frame, res_out)

            if cache_results:
                all_ress.append(res_out)
                if return_views:
                    processed_frames.append(
                        {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) and move_to_cpu else nv for nk, nv in frame.items()}
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
        """
        Convert one frame image tensor into [B, S=1, C, H, W] format expected by
        cached frontend inference.
        """
        if frame_img.dim() == 3:
            # [C, H, W] -> [1, 1, C, H, W]
            return frame_img.unsqueeze(0).unsqueeze(1)
        if frame_img.dim() == 4:
            # [B, C, H, W] -> [B, 1, C, H, W]
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
        """
        Frontend cache/keyframe scheduling currently assumes one sequence per step.
        Explicitly enforce batch_size=1 to avoid silent training corruption.
        """
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
        # B>=1 is now supported; no guard needed.
        pass

    def _build_frontend_keyframe_config(
        self,
        history_anchor_strategy: str,
        anchor_interval: int,
        max_anchors: int,
        coverage_threshold: float,
    ) -> KeyframeSwitchConfig:
        if self._keyframe_switch_config_provided and self.keyframe_switch_config is not None:
            return self.keyframe_switch_config
        strategy = history_anchor_strategy if history_anchor_strategy in {"fixed_interval", "coverage"} else "fixed_interval"
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
        return "none"

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
    def _upgrade_camera_head_state_dict(state_dict: dict) -> None:
        abs_branch_prefix = "camera_head.pose_branch."
        rel_branch_prefix = "camera_head.rel_pose_branch."
        abs_branch_keys = [key for key in state_dict if key.startswith(abs_branch_prefix)]
        if not abs_branch_keys:
            return
        rel_branch_keys = [key for key in state_dict if key.startswith(rel_branch_prefix)]
        if rel_branch_keys:
            return
        for key in abs_branch_keys:
            rel_key = rel_branch_prefix + key[len(abs_branch_prefix):]
            state_dict[rel_key] = state_dict[key].clone()

    @staticmethod
    def _maybe_move_dict_to_cpu(payload: dict) -> dict:
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in payload.items()
        }
