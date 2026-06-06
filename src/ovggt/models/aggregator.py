# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any, Sequence

from ovggt.layers import PatchEmbed
from ovggt.layers.block import Block
from ovggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ovggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from ovggt.utils.frontend_cache import PendingLayerUpdate

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        eviction_strategy='repr_shift',
        intra_frame_keep_ratio=1.0,
        spatial_alpha=0.3,
    ):
        super().__init__()

        self.eviction_strategy = eviction_strategy
        self.intra_frame_keep_ratio = intra_frame_keep_ratio
        self.spatial_alpha = spatial_alpha
        self.token_scorers = None  # initialized by init_token_scorers()
        self.score_state_projs = None
        self.count_head = None  # initialized by init_count_head()

        # Patch tokens start after camera(1) + register tokens
        self._patch_start_idx = 1 + num_register_tokens
        # Store patch_size for dynamic grid size computation
        self._patch_size = patch_size
        # The active patch grid is derived from the current input shape in forward().
        self.patch_grid_size = None

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    eviction_strategy=eviction_strategy,
                    spatial_alpha=spatial_alpha,
                    patch_start_idx=self._patch_start_idx,
                    patch_grid_size=self.patch_grid_size,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
        self.last_scores = torch.zeros(self.depth)

    def init_token_scorers(
        self,
        embed_dim: int,
        bottleneck_dim: int = None,
        score_state_dim: int = 128,
    ):
        from ovggt.layers.token_scorer import TokenScorer
        self.score_state_projs = nn.ModuleList([
            nn.Linear(embed_dim, score_state_dim)
            for _ in range(self.depth)
        ])
        self.token_scorers = nn.ModuleList([
            TokenScorer(
                score_state_dim=score_state_dim,
                bottleneck_dim=bottleneck_dim,
                num_layers=self.depth,
            )
            for _ in range(self.depth)
        ])
        for idx, block in enumerate(self.global_blocks):
            block.token_scorer = self.token_scorers[idx]
            block.score_state_proj = self.score_state_projs[idx]

    def init_count_head(
        self,
        score_state_dim: int = 128,
        metadata_dim: int = 17,
        hidden_dim: int | None = None,
        candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
        arch: str = "pooled_v1",
    ) -> None:
        from ovggt.layers.count_head import FifoCountHead
        self.count_head = FifoCountHead(
            score_state_dim=score_state_dim,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            num_layers=self.depth,
            candidates=candidates,
            arch=arch,
        )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        cache_states=None,
        use_cache=False,
        past_frame_idx=0,
        per_layer_budget=0,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        frontend_cache_config=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        # Compute patch grid size dynamically from actual image dimensions
        grid_h = H // self._patch_size
        grid_w = W // self._patch_size
        current_patch_grid_size = (grid_h, grid_w)

        frontend_cache_mode = use_cache and frontend_cache_config is not None and frontend_cache_config.enabled
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            use_first_frame_tokens = past_frame_idx == 0
            camera_token = select_cached_special_token(
                self.camera_token,
                batch_size=B,
                use_first_frame_tokens=use_first_frame_tokens,
            )
            register_token = select_cached_special_token(
                self.register_token,
                batch_size=B,
                use_first_frame_tokens=use_first_frame_tokens,
            )
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        current_budgets = self._calculate_budgets(per_layer_budget, frontend_cache_config)
        scores = []
        pending_updates: List[Optional[PendingLayerUpdate]] = [None] * self.depth

        # Track importance within current frame processing (layer to layer)
        prev_importance = None
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if frontend_cache_mode:
                        layer_budget = None if current_budgets is None else current_budgets[global_idx].item()
                        tokens, global_idx, global_intermediates, pending_update, new_importance, score_state = self._process_global_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            past_key_values_block=cache_states[global_idx].as_past_key_values()
                            if cache_states is not None and cache_states[global_idx] is not None
                            else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=layer_budget,
                            prev_importance=prev_importance,
                            intra_frame_keep_ratio=self.intra_frame_keep_ratio,
                            anchor_token_count=anchor_token_count,
                            importance_weight=importance_weight,
                            frontend_cache_mode=True,
                            patch_grid_size=current_patch_grid_size,
                        )
                        prev_importance = new_importance
                        layer_idx = global_idx - 1
                        if pending_update is not None:
                            pending_updates[layer_idx] = PendingLayerUpdate(
                                k_current=pending_update[0],
                                v_current=pending_update[1],
                                importance_current=new_importance,
                                frame_id=past_frame_idx,
                                cache_budget=layer_budget,
                                score_state_current=score_state,
                            )
                    elif use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]

                        # Determine cache budget for this layer (None = no eviction)
                        layer_budget = None if current_budgets is None else current_budgets[global_idx].item()

                        # For repr_shift: use previous layer's importance for new tokens
                        # Falls back to baseline (cosine diversity) when shapes don't match
                        tokens, global_idx, global_intermediates, new_kv, current_scores, new_importance, kept_indices = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=layer_budget,
                            prev_importance=prev_importance,
                            intra_frame_keep_ratio=self.intra_frame_keep_ratio,
                            anchor_token_count=anchor_token_count,
                            importance_weight=importance_weight,
                            patch_grid_size=current_patch_grid_size,
                        )

                        # Pass new importance to next layer (within same frame)
                        prev_importance = new_importance

                        layer_idx = global_idx - 1
                        past_key_values[layer_idx] = new_kv
                        if current_scores is not None: 
                            scores.append(current_scores)
                        else:
                            scores.append(self.last_scores[layer_idx].item())
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            patch_grid_size=current_patch_grid_size,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        if scores:  
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if frontend_cache_mode:
            return output_list, self.patch_start_idx, cache_states, pending_updates, None
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):

            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0,
        cache_budget=None,
        prev_importance=None,
        intra_frame_keep_ratio=1.0,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        frontend_cache_mode: bool = False,
        patch_grid_size: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).

        Args:
            prev_importance: Optional [B, N] accumulated importance scores for KV cache.
                Used for eviction decision when eviction_strategy='repr_shift'.

        Returns:
            When use_cache=True: (tokens, global_idx, intermediates, block_kv, scores, new_importance, kept_indices)
            Otherwise: (tokens, global_idx, intermediates)
        """

        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)

        intermediates = []
        new_importance = None
        kept_indices = None
        pending_update = None
        score_state = None

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None

            scores = None
            if frontend_cache_mode:
                tokens, pending_update, new_importance, score_state = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    prev_importance=prev_importance,
                    intra_frame_keep_ratio=intra_frame_keep_ratio,
                    anchor_token_count=anchor_token_count,
                    importance_weight=importance_weight,
                    frontend_cache_mode=True,
                    patch_grid_size=patch_grid_size,
                )
            elif use_cache:
                # Skip intra-frame pruning for anchor frame (first frame)
                effective_keep_ratio = 1.0 if past_frame_idx == 0 else intra_frame_keep_ratio
                tokens, block_kv, scores, new_importance, kept_indices = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    prev_importance=prev_importance,
                    intra_frame_keep_ratio=effective_keep_ratio,
                    anchor_token_count=anchor_token_count,
                    importance_weight=importance_weight,
                    patch_grid_size=patch_grid_size,
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    patch_grid_size=patch_grid_size,
                )

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if frontend_cache_mode:
            return tokens, global_idx, intermediates, pending_update, new_importance, score_state
        if use_cache:
            return tokens, global_idx, intermediates, block_kv, scores, new_importance, kept_indices
        return tokens, global_idx, intermediates
        
    def _calculate_budgets(self, per_layer_budget, frontend_cache_config=None):
        # Handle None budget (eviction paused for History Anchor window)
        # NOTE: ``per_layer_budget`` is the token budget PER LAYER (not the 24-layer
        # sum). The dynamic path multiplies by ``self.depth`` so the per-layer
        # average equals ``per_layer_budget`` even when proportions are non-uniform.
        if per_layer_budget is None:
            return None

        with torch.no_grad():
            if per_layer_budget < 0:
                per_layer_budget = 0
            if frontend_cache_config is not None and getattr(frontend_cache_config, "budget_allocation", "dynamic") == "uniform":
                budgets = torch.full((self.depth,), int(per_layer_budget), dtype=torch.int64)
                return budgets

            diversity_scores = 1.0 - self.last_scores
            scaled_scores = diversity_scores / 0.5
            proportions = torch.softmax(scaled_scores, dim=0)
            # proportions sum to 1.0; multiply by per_layer_budget*depth so the
            # mean across layers equals per_layer_budget (preserves prior behavior
            # where the caller used to pass total = per_layer_budget * depth).
            budgets = proportions * (per_layer_budget * self.depth)

        return budgets.int()


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined


def select_cached_special_token(token_tensor, batch_size: int, use_first_frame_tokens: bool):
    token_idx = 0 if use_first_frame_tokens else 1
    selected = token_tensor[:, token_idx : token_idx + 1, ...]
    selected = selected.expand(batch_size, 1, *token_tensor.shape[2:])
    return selected.reshape(batch_size, *token_tensor.shape[2:])
