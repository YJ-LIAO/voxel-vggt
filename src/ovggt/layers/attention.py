import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

XFORMERS_AVAILABLE = False


def _normalize_scores(scores: Tensor, neutral_value: float = 0.5) -> Tensor:
    """
    Normalize scores to [0, 1] while keeping equal-score groups neutral instead of
    collapsing them to zeros, which would bias eviction decisions.
    """
    score_min = scores.min(dim=-1, keepdim=True)[0]
    score_max = scores.max(dim=-1, keepdim=True)[0]
    denom = score_max - score_min
    normalized = (scores - score_min) / (denom + 1e-8)
    if scores.shape[-1] == 0:
        return normalized
    equal_mask = denom <= 1e-8
    if equal_mask.any():
        normalized = torch.where(
            equal_mask.expand_as(normalized),
            torch.full_like(normalized, neutral_value),
            normalized,
        )
    return normalized



class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0
        # Overflow behavior when cache budget cannot keep all anchor tokens.
        # - "recent": keep the most recent anchors
        # - "global_plus_recent": always keep global anchor (index 0) plus recent anchors
        self.anchor_overflow_policy = "recent"

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0

    def _select_anchor_indices_on_overflow(
        self,
        num_anchor_tokens: int,
        keep_anchor_count: int,
        device: torch.device,
    ) -> Tensor:
        if keep_anchor_count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if self.anchor_overflow_policy == "global_plus_recent" and num_anchor_tokens > 0:
            if keep_anchor_count == 1:
                return torch.tensor([0], dtype=torch.long, device=device)
            recent_count = keep_anchor_count - 1
            start_idx = max(num_anchor_tokens - recent_count, 1)
            recent_indices = torch.arange(start_idx, num_anchor_tokens, dtype=torch.long, device=device)
            return torch.cat(
                [torch.tensor([0], dtype=torch.long, device=device), recent_indices],
                dim=0,
            )

        start_idx = max(num_anchor_tokens - keep_anchor_count, 0)
        return torch.arange(start_idx, num_anchor_tokens, dtype=torch.long, device=device)

    def intra_frame_prune(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance: torch.Tensor,
        keep_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prune current frame tokens based on repr_shift importance.

        This is Stage 1 of the two-stage eviction strategy:
        - Intra-frame pruning: prune current frame tokens BEFORE storing to KV cache
        - Only affects what gets stored, not what's used for current layer attention

        Args:
            k (torch.Tensor): Current frame K [B, H, N, D]
            v (torch.Tensor): Current frame V [B, H, N, D]
            importance (torch.Tensor): Importance scores [B, N] from repr_shift
                Higher = more important = KEEP
            keep_count (int): Number of tokens to keep

        Returns:
            (pruned_k, pruned_v, kept_indices)
            pruned_k: [B, H, keep_count, D]
            pruned_v: [B, H, keep_count, D]
            kept_indices: [B, keep_count] indices of kept tokens, or None if no pruning
        """
        B, H, N, D = k.shape

        if keep_count >= N:
            return k, v, None

        # Keep at least 1 token
        keep_count = max(keep_count, 1)

        # Keep top-k by importance (higher = more important)
        _, top_indices = torch.topk(importance, k=keep_count, dim=-1)
        top_indices = top_indices.sort(dim=-1).values  # Maintain temporal order

        # Gather pruned K, V
        expanded_indices = top_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, keep_count, D)
        pruned_k = torch.gather(k, 2, expanded_indices)
        pruned_v = torch.gather(v, 2, expanded_indices)

        return pruned_k, pruned_v, top_indices

    def eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        importance_scores: torch.Tensor = None,
        num_new_tokens: int = 0,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Evicts tokens from the key-value cache using hybrid scoring strategy.

        Hybrid strategy:
        - Old tokens (from past frames): use baseline (cosine diversity) weighted by (1 - importance_weight)
        - New tokens (from current frame): use repr_shift importance weighted by importance_weight

        Args:
            k (torch.Tensor): The key tensor of shape [B, H, N, D].
            v (torch.Tensor): The value tensor of shape [B, H, N, D].
            cache_budget (int): The maximum number of tokens to retain.
            num_anchor_tokens (int): The number of initial tokens to preserve.
            importance_scores (torch.Tensor): Optional [B, N_new] tensor for new tokens.
                Higher = more important = KEEP.
            num_new_tokens (int): Number of new tokens in the current frame.
            importance_weight (float): Weight for current frame importance scores (default=0.5).
                New token scores = importance_weight * normalized_importance
                Old token scores = (1 - importance_weight) * normalized_diversity

        Returns:
            A tuple of (pruned_k, pruned_v, avg_scores, kept_indices).
            kept_indices: [B, cache_budget] global indices of kept tokens, or None if no eviction.
        """
        B, H, N, D = k.shape
        cache_budget = max(int(cache_budget), 0)
        num_anchor_tokens = min(max(int(num_anchor_tokens), 0), N)

        if N <= cache_budget:
            return k, v, None, None

        anchor_k, candidate_k = k.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)
        anchor_v, candidate_v = v.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)

        num_candidates = N - num_anchor_tokens
        num_to_keep_from_candidates = min(max(cache_budget - num_anchor_tokens, 0), num_candidates)
        num_old_candidates = max(num_candidates - num_new_tokens, 0)

        # Edge case: the budget cannot even hold all protected anchors.
        # Keep the most recent anchors so the returned cache still honors the
        # requested size while favoring short-term geometry consistency.
        if num_to_keep_from_candidates <= 0:
            keep_anchor_count = min(max(cache_budget, 0), num_anchor_tokens)
            keep_anchor_indices = self._select_anchor_indices_on_overflow(
                num_anchor_tokens=num_anchor_tokens,
                keep_anchor_count=keep_anchor_count,
                device=k.device,
            )
            anchor_indices = keep_anchor_indices.unsqueeze(0).expand(B, -1)
            if keep_anchor_count <= 0:
                return (
                    anchor_k[:, :, :0, :],
                    anchor_v[:, :, :0, :],
                    None,
                    anchor_indices,
                )
            expanded_anchor_indices = keep_anchor_indices.view(1, 1, keep_anchor_count, 1).expand(
                B, H, keep_anchor_count, D
            )
            return (
                torch.gather(anchor_k, 2, expanded_anchor_indices),
                torch.gather(anchor_v, 2, expanded_anchor_indices),
                None,
                anchor_indices,
            )

        # Edge case: if we can keep all candidates
        if num_to_keep_from_candidates >= num_candidates:
            return k, v, None, None

        protected_window_count = min(
            max(int(window_token_count), 0),
            num_candidates,
            num_to_keep_from_candidates,
        )

        def _select_candidate_indices(scores: Tensor) -> Tensor:
            num_scored = num_candidates - protected_window_count
            num_from_scores = num_to_keep_from_candidates - protected_window_count
            if num_from_scores > 0:
                scored = scores[:, :num_scored]
                _, top_indices = torch.topk(scored, k=num_from_scores, dim=-1)
                top_indices_sorted = top_indices.sort(dim=-1).values
            else:
                top_indices_sorted = torch.empty(B, 0, dtype=torch.long, device=k.device)

            if protected_window_count <= 0:
                return top_indices_sorted

            window_indices = torch.arange(
                num_scored,
                num_candidates,
                dtype=torch.long,
                device=k.device,
            ).unsqueeze(0).expand(B, -1)
            return torch.cat([top_indices_sorted, window_indices], dim=-1)

        # Compute baseline scores (cosine diversity) for ALL candidates
        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)
        baseline_scores = torch.sum(candidate_k_norm * mean_vector, dim=-1)  # [B, H, N_cand]
        # Convert to diversity: lower similarity = higher diversity = keep
        baseline_diversity = 1.0 - baseline_scores  # [B, H, N_cand]
        # Average across heads for unified scoring
        baseline_diversity_avg = baseline_diversity.mean(dim=1)  # [B, N_cand]

        # Check if we can use hybrid scoring
        use_hybrid = (
            importance_scores is not None
            and importance_scores.shape[1] == num_new_tokens
            and num_new_tokens > 0
            and num_old_candidates > 0
        )

        if use_hybrid:
            # Hybrid strategy: baseline for old, repr_shift for new
            old_scores = baseline_diversity_avg[:, :num_old_candidates]  # [B, N_old]
            new_importance = importance_scores  # [B, N_new]

            # Normalize both to [0, 1] range for fair comparison.
            old_normalized = _normalize_scores(old_scores)
            new_normalized = _normalize_scores(new_importance)

            # Apply importance_weight: current frame weighted by importance_weight, past frames by (1 - importance_weight)
            weighted_old = (1.0 - importance_weight) * old_normalized
            weighted_new = importance_weight * new_normalized

            # Combine: [B, N_cand]
            combined_scores = torch.cat([weighted_old, weighted_new], dim=-1)
            avg_scores = combined_scores.mean().item()

            # Keep tokens with HIGHEST combined score
            top_indices_sorted = _select_candidate_indices(combined_scores)

            # Expand for gather across heads: [B, H, num_to_keep, D]
            expanded_indices = top_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)

            # Build kept_indices
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted + num_anchor_tokens
            kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)
        else:
            # Fallback: pure baseline (cosine diversity)
            avg_scores = baseline_scores.mean().item()

            # Keep tokens with HIGHEST diversity using one unified selection across
            # heads so K/V ordering stays consistent with metadata indices.
            top_indices_sorted = _select_candidate_indices(baseline_diversity_avg)

            expanded_indices = top_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(
                B,
                H,
                num_to_keep_from_candidates,
                D,
            )

            # Build kept_indices.
            anchor_indices = torch.arange(num_anchor_tokens, device=k.device).unsqueeze(0).expand(B, -1)
            kept_candidate_indices = top_indices_sorted + num_anchor_tokens
            kept_indices = torch.cat([anchor_indices, kept_candidate_indices], dim=-1)

        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
        final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)

        return final_k, final_v, avg_scores, kept_indices

    def forward(self,
        x: torch.Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget=2000,
        importance_scores=None,
        intra_frame_keep_ratio=1.0,
        defer_eviction=False,
        anchor_token_count: int = None,
        importance_weight: float = 0.5,
        window_token_count: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass with optional two-stage eviction support.

        Args:
            intra_frame_keep_ratio: Ratio of current frame tokens to keep (1.0 = keep all)
            defer_eviction: If True, skip inter-frame eviction here (handled by Block)

        When defer_eviction=True, returns additional info for Block to handle eviction:
            (output, (k_full, v_full, k_current, v_current, past_kv_or_none), scores)
        """
        B, N, C = x.shape
        num_new_tokens = N  # Number of new tokens from current frame
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = None
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if use_cache and self.num_anchor_tokens == 0:
            self.num_anchor_tokens = k.shape[2]

        # Keep direct references for current-frame K/V.
        # These tensors are not modified in-place before concatenation, so clones
        # only increase peak memory without changing behavior.
        k_current, v_current = k, v
        past_kv_for_block = None

        if use_cache:
            if past_key_values is not None:
                past_k, past_v = past_key_values
                past_kv_for_block = (past_k, past_v)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            kept_indices = None
            if not defer_eviction:
                # Legacy path: apply eviction in attention
                if cache_budget is not None and k.shape[2] > cache_budget:
                    # Use anchor_token_count if provided (History Anchor), otherwise fall back to num_anchor_tokens
                    effective_anchor_count = anchor_token_count if anchor_token_count is not None else self.num_anchor_tokens
                    k, v, scores, kept_indices = self.eviction(
                        k, v, cache_budget, effective_anchor_count,
                        importance_scores=importance_scores,
                        num_new_tokens=num_new_tokens,
                        importance_weight=importance_weight,
                        window_token_count=window_token_count,
                    )
                new_kv = (k, v, kept_indices)
            else:
                # Two-stage path: return info for Block to handle eviction
                # Block will apply: 1) intra-frame pruning, 2) inter-frame eviction
                new_kv = (k, v, k_current, v_current, past_kv_for_block)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
                return x, new_kv, scores
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
