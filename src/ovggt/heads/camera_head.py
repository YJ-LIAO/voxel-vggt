import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ovggt.layers import Mlp
from ovggt.layers.block import Block
from ovggt.heads.head_act import activate_pose
from ovggt.utils.pose_enc import ABS_POSE_ENCODING, REL_POSE_ENCODING


class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        # Field of view activations: ensures FOV values are positive.
        fl_act: str = "relu",
        total_budget: int = 384,
    ):
        super().__init__()

        if pose_encoding_type in {ABS_POSE_ENCODING, REL_POSE_ENCODING}:
            self.target_dim = 9
        else:
            raise ValueError(
                f"Unsupported camera encoding type: {pose_encoding_type}")

        self.pose_encoding_type = pose_encoding_type
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.total_budget = total_budget
        # Initialize last_scores to 0 (all tokens important initially, or rather, equal)
        self.last_scores = torch.zeros(self.trunk_depth)

        self.enable_cache_merge = True
        self.cache_merge_ratio = 0.5

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(
            torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(
            dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )
        self.rel_pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )
        self.rel_pose_branch.load_state_dict(self.pose_branch.state_dict())

    def forward(
        self,
        aggregated_tokens_list: list,
        num_iterations: int = 4,
        past_key_values_camera=None,
        use_cache: bool = False,
        anchor_token_count: int = None,
        pose_encoding_type: str = None,
        return_pose_predictions: bool = False,
        return_last_pose_only: bool = False,
    ) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            anchor_token_count (int, optional): Number of camera KV cache tokens to
                protect from eviction, synced with the Aggregator's anchor manager.
                When None, falls back to first-frame-only protection.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]
        pose_encoding_type = pose_encoding_type or self.pose_encoding_type
        if pose_encoding_type not in {ABS_POSE_ENCODING, REL_POSE_ENCODING}:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        if use_cache:
            pred_pose_enc_list, past_key_values_camera = self.trunk_fn(
                pose_tokens,
                num_iterations,
                past_key_values_camera,
                use_cache,
                anchor_token_count=anchor_token_count,
                pose_encoding_type=pose_encoding_type,
                return_pose_predictions=return_pose_predictions,
                return_last_pose_only=return_last_pose_only,
            )
            return pred_pose_enc_list, past_key_values_camera
        else:
            pred_pose_enc_list = self.trunk_fn(
                pose_tokens,
                num_iterations,
                past_key_values_camera=None,
                use_cache=use_cache,
                anchor_token_count=anchor_token_count,
                pose_encoding_type=pose_encoding_type,
                return_pose_predictions=return_pose_predictions,
                return_last_pose_only=return_last_pose_only,
            )
            return pred_pose_enc_list

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        num_iterations: int,
        past_key_values_camera,
        use_cache: bool,
        anchor_token_count: int = None,
        pose_encoding_type: str = None,
        return_pose_predictions: bool = False,
        return_last_pose_only: bool = False,
    ) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.
            anchor_token_count (int, optional): Number of camera KV cache tokens to
                protect from eviction, synced with the Aggregator's anchor manager.
                When None, falls back to first-frame-only protection.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_rel_pose_enc = None
        keep_full_pose_history = return_pose_predictions and not return_last_pose_only
        abs_pose_enc_list = [] if keep_full_pose_history else None
        rel_pose_enc_list = [] if keep_full_pose_history else None
        last_abs_pose_enc = None
        last_rel_pose_enc = None

        current_budgets = self._calculate_dynamic_budgets(self.total_budget)
        scores_list = []

        if use_cache:
            incoming_past_kv = [kv for kv in past_key_values_camera]
            is_first_frame = all(kv is None for kv in incoming_past_kv)
            if anchor_token_count is None:
                anchor_token_count = num_iterations if not is_first_frame else None
            elif is_first_frame:
                anchor_token_count = None
            temp_kv = list(incoming_past_kv)

        for iter_idx in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(
                    self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(
                module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * \
                modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            if not use_cache:
                L = S * 1
                # [0,0,...,1,1,...,S-1]
                frame_ids = torch.arange(
                    L, device=pose_tokens_modulated.device) // 1
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(
                    pose_tokens_modulated.dtype) * torch.finfo(pose_tokens_modulated.dtype).min
            else:
                attn_mask = None

            if use_cache:
                is_last_iter = (iter_idx == num_iterations - 1)
                scores_iter = []
                prev_importance = None

                for idx in range(self.trunk_depth):
                    layer_budget = None if is_first_frame else current_budgets[idx].item(
                    )
                    pose_tokens_modulated, block_kv, scores, new_importance, _ = self.trunk[idx](
                        pose_tokens_modulated,
                        attn_mask=attn_mask,
                        past_key_values=temp_kv[idx],
                        use_cache=True,
                        cache_budget=layer_budget,
                        prev_importance=prev_importance,
                        anchor_token_count=anchor_token_count,
                    )
                    prev_importance = new_importance
                    temp_kv[idx] = block_kv
                    if is_last_iter:
                        past_key_values_camera[idx] = block_kv

                    if scores is not None:
                        scores_iter.append(scores)
                    else:
                        scores_iter.append(self.last_scores[idx].item())
                scores_list = scores_iter
            else:
                for idx in range(self.trunk_depth):
                    pose_tokens_modulated = self.trunk[idx](
                        pose_tokens_modulated, attn_mask=attn_mask)

            # Compute the delta update for the pose encoding.
            trunk_tokens = self.trunk_norm(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(trunk_tokens)
            pred_rel_pose_enc_delta = self.rel_pose_branch(trunk_tokens)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            if pred_rel_pose_enc is None:
                pred_rel_pose_enc = pred_rel_pose_enc_delta
            else:
                pred_rel_pose_enc = pred_rel_pose_enc + pred_rel_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_abs_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            activated_rel_pose = activate_pose(
                pred_rel_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            last_abs_pose_enc = activated_abs_pose
            last_rel_pose_enc = activated_rel_pose
            if keep_full_pose_history:
                abs_pose_enc_list.append(activated_abs_pose)
                rel_pose_enc_list.append(activated_rel_pose)

        if scores_list:
            self.last_scores = torch.tensor(
                scores_list, device=self.last_scores.device, dtype=self.last_scores.dtype)

        if return_pose_predictions:
            if return_last_pose_only:
                pose_predictions = {
                    "abs_pose_enc": last_abs_pose_enc,
                    "rel_pose_enc": last_rel_pose_enc,
                }
            else:
                pose_predictions = {
                    "abs_pose_enc_list": abs_pose_enc_list,
                    "rel_pose_enc_list": rel_pose_enc_list,
                }
        elif pose_encoding_type == REL_POSE_ENCODING:
            pose_predictions = rel_pose_enc_list if rel_pose_enc_list is not None else [last_rel_pose_enc]
        else:
            pose_predictions = abs_pose_enc_list if abs_pose_enc_list is not None else [last_abs_pose_enc]

        if use_cache:
            return pose_predictions, past_key_values_camera
        return pose_predictions

    def sync_anchor_change(self, past_key_values_camera, anchor_token_count, num_cam_iters=4, is_fifo=False):
        """
        Rearrange camera KV cache after an Aggregator anchor registration event.

        Ensures the anchor zone (first anchor_token_count tokens) contains only
        tokens from currently-active anchor frames, synced with the Aggregator's
        HistoryAnchorManager.

        When a new anchor is registered (non-FIFO):
          - The new frame's tokens (last num_cam_iters in cache) are moved into
            the anchor zone at the correct position.

        When FIFO triggers:
          - The oldest history anchor's tokens are removed from the anchor zone
            and moved to the candidate zone (can be evicted).
          - The new frame's tokens are promoted into the freed anchor slot.

        Args:
            past_key_values_camera: list of (k, v) per trunk layer
            anchor_token_count: anchor zone size AFTER the change (total_anchors * num_cam_iters)
            num_cam_iters: camera iterations per frame (tokens per anchor, default 4)
            is_fifo: whether FIFO demotion occurred (oldest history anchor demoted)

        Returns:
            Modified past_key_values_camera
        """
        global_anchor_end = num_cam_iters

        for idx in range(self.trunk_depth):
            if past_key_values_camera[idx] is None:
                continue
            k, v = past_key_values_camera[idx]
            N = k.shape[2]

            if N <= anchor_token_count:
                continue

            new_frame_start = N - num_cam_iters

            if is_fifo:
                demote_start = global_anchor_end
                demote_end = global_anchor_end + num_cam_iters

                k_new = torch.cat([
                    k[:, :, :demote_start, :],
                    k[:, :, demote_end:anchor_token_count, :],
                    k[:, :, new_frame_start:, :],
                    k[:, :, anchor_token_count:new_frame_start, :],
                    k[:, :, demote_start:demote_end, :],
                ], dim=2)
                v_new = torch.cat([
                    v[:, :, :demote_start, :],
                    v[:, :, demote_end:anchor_token_count, :],
                    v[:, :, new_frame_start:, :],
                    v[:, :, anchor_token_count:new_frame_start, :],
                    v[:, :, demote_start:demote_end, :],
                ], dim=2)
            else:
                old_anchor_end = anchor_token_count - num_cam_iters

                k_new = torch.cat([
                    k[:, :, :old_anchor_end, :],
                    k[:, :, new_frame_start:, :],
                    k[:, :, old_anchor_end:new_frame_start, :],
                ], dim=2)
                v_new = torch.cat([
                    v[:, :, :old_anchor_end, :],
                    v[:, :, new_frame_start:, :],
                    v[:, :, old_anchor_end:new_frame_start, :],
                ], dim=2)

            past_key_values_camera[idx] = (k_new, v_new)

        return past_key_values_camera

    def apply_keyframe_event(self, past_key_values_camera, event, num_cam_iters=4):
        """
        Apply a frontend keyframe event to the camera cache while preserving the
        existing tuple-based KV representation.
        """
        if event is None or past_key_values_camera is None:
            return past_key_values_camera

        event_name = str(getattr(event, "event_type", ""))
        if event_name.endswith("NOOP"):
            return past_key_values_camera

        if getattr(event, "frame_idx", None) == 0 and getattr(event, "anchor_slot", -1) == 0:
            return past_key_values_camera

        num_anchor_frames = getattr(event, "num_anchor_frames", 0)
        if num_anchor_frames <= 0:
            return past_key_values_camera

        anchor_token_count = num_anchor_frames * num_cam_iters
        return self.sync_anchor_change(
            past_key_values_camera,
            anchor_token_count=anchor_token_count,
            num_cam_iters=num_cam_iters,
            is_fifo=event_name.endswith("FIFO_SWAP"),
        )

    def _calculate_dynamic_budgets(self, total_budget):
        with torch.no_grad():
            diversity_scores = 1.0 - self.last_scores
            scaled_scores = diversity_scores / 0.5
            proportions = torch.softmax(scaled_scores, dim=0)
            if total_budget < 0:
                total_budget = 0
            budgets = proportions * total_budget
        return budgets.int()


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    return x * (1 + scale) + shift
