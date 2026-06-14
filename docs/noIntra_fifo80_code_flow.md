# noIntra+fifo80 完整代码流程

> 基线 commit: `8d0a09f` (noIntra+fifo80: establish long-sequence optimization baseline)
> 核心文件: `src/ovggt/models/ovggt.py`, `src/ovggt/models/aggregator.py`, `src/ovggt/layers/block.py`, `src/ovggt/layers/attention.py`, `src/ovggt/utils/frontend_cache.py`
> 推荐配置: `intra_frame_dedup_enabled=False, fifo_keep_topk=80, dedup_enabled=True`

---

## 1. 全局架构：24 层交替注意力

OVGGT 的 Frontend 增量推理模式采用 24 层交替的 Frame/Global Attention。每处理一帧新图像，所有 24 层依次执行，然后统一更新 KV Cache。

```
每帧输入: image → patch_embed → tokens [B, P+5, C]
     (B=1, P=1370 patches, +1 camera +4 register, C=768)

24 层循环 (layer_idx = 0..23):
  ┌─────────────────────────────────────────────────┐
  │ Frame Block[layer_idx]:  tokens 自注意力 (无缓存)│
  │        ↓                                         │
  │ Global Block[layer_idx]: tokens 与 KV Cache 交叉 │
  │        ↓                                         │
  │ 输出: tokens + (k_current, v_current, importance)│
  └─────────────────────────────────────────────────┘

所有 24 层完成后 → Keyframe 事件判断 → FIFO 保护 → commit pending updates → cache 更新
```

每个 Layer 拥有独立的 `LayerCacheState`，持有 `(k, v, metadata)` 三组张量。24 个 cache 之间完全独立，互不影响。

---

## 2. 入口：OVGGT._inference_frontend() (ovggt.py)

### 2.1 初始化

```python
# 每层一个独立的 cache state
cache_states = [LayerCacheState(max_history_anchors=3) for _ in range(24)]
past_key_values_camera = [None] * camera_head.trunk_depth
keyframe_manager = FrontendKeyframeManager(anchor_interval=8, max_anchors=3)
```

### 2.2 逐帧处理循环

```python
for i, frame in enumerate(frames):
    # 1. Aggregator 前向：24层 Frame+Global attention
    aggregated_tokens, patch_start_idx, cache_states, pending_updates = \
        self.aggregator(images, cache_states=cache_states, use_cache=True,
                        past_frame_idx=i, total_budget=total_budget, ...)

    # 2. 任务头：camera / depth / point_map / track
    pose = self.camera_head(aggregated_tokens, ...)
    depth = self.depth_head(aggregated_tokens, ...)
    pts = self.point_head(aggregated_tokens, ...)

    # 3. Keyframe 事件判断
    event = keyframe_manager.update(current_frame_id=i, ...)

    # 4. FIFO topK 保护 (仅 FIFO_SWAP 时)
    is_fifo_swap = str(event.event_type).endswith("FIFO_SWAP")
    if is_fifo_swap and fifo_keep_topk > 0:
        for layer_idx in range(24):
            cache_states[layer_idx].protect_topk_on_demotion_(
                demoted_slot=event.demoted_slot,
                keep_count=fifo_keep_topk,  # 80
            )

    # 5. 应用 keyframe 事件 (anchor slot 变更)
    for layer_idx in range(24):
        cache_states[layer_idx].apply_keyframe_event_(event)

    # 6. 提交 pending updates: 每层独立执行 cache 更新
    for layer_idx in range(24):
        pu = pending_updates[layer_idx]
        frame_metadata = build_frame_token_metadata_base(depth, pose, ...)
        current_metadata = frame_metadata.with_importance(pu.importance_current)

        cache_states[layer_idx].commit_pending_update_(
            pu, current_metadata, config,
            intra_frame_keep_ratio=1.0,  # noIntra: 不做帧内剪枝
            attn_module=self.aggregator.global_blocks[layer_idx].attn,
        )
```

---

## 3. Aggregator.forward() — 24 层注意力调度

### 3.1 Token 组装

```python
# Patch embedding
patch_tokens = self.patch_embed(images)  # [B*S, P, C]

# 特殊 token (camera + register)
camera_token = select_cached_special_token(...)  # [B, 1, C]
register_tokens = ...                             # [B, 4, C]

# 拼接
tokens = [camera_token, register_tokens, patch_tokens]  # [B, N, C], N = P+5
# 添加 RoPE 位置编码
```

### 3.2 Budget 分配

```python
# 基于 softmax(last_scores) 将 total_budget 分配到 24 层
# last_scores 初始为 0，随帧更新
current_budgets = self._calculate_dynamic_budgets(total_budget)
# 返回 [24] 的 per-layer budget，如 [8333, 8333, ...] (200000/24)
```

### 3.3 交替注意力循环

```python
aa_order = ["frame", "global"]
depth = 24
global_idx = 0

for layer_idx in range(depth):
    # --- Frame Attention (无缓存) ---
    tokens = self.frame_blocks[layer_idx](tokens, pos=pos)
    # tokens: [B, N, C] → [B, N, C]
    # 纯自注意力，当前帧内所有 token 互相 attend

    # --- Global Attention (带 KV Cache) ---
    layer_budget = current_budgets[global_idx]
    tokens, global_idx, _, pending_update, new_importance = \
        self._process_global_attention(
            tokens, cache_states[global_idx].as_past_key_values(),
            cache_budget=layer_budget,
            prev_importance=prev_importance,
            frontend_cache_mode=True,
        )
    prev_importance = new_importance

    # 保存当前层的 pending update
    pending_updates[layer_idx] = PendingLayerUpdate(
        k_current=pending_update[0],       # [B, H, N, D]
        v_current=pending_update[1],       # [B, H, N, D]
        importance_current=new_importance, # [B, N]
        frame_id=past_frame_idx,
        cache_budget=layer_budget,
    )
    global_idx += 1
```

---

## 4. Block.forward() — Global Block 的 frontend_cache_mode 路径

当 `use_cache=True` 且 `frontend_cache_mode=True` 时，Block 的执行流程：

### 4.1 Attention（延迟 eviction）

```python
# attention.py — defer_eviction=True 路径

# Step A: QKV 投影
qkv = self.qkv(LayerNorm(x)).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)  # 各 [B, H, N, D], N = P+5

# Step B: RoPE 位置编码
q, k = self.rope(q, pos), self.rope(k, pos)

# Step C: 拼接历史 KV Cache
k_current, v_current = k, v  # 保留当前帧的 KV 引用
if past_key_values is not None:
    past_k, past_v = past_key_values
    k = torch.cat([past_k, k], dim=2)  # [B, H, N_past+N, D]
    v = torch.cat([past_v, v], dim=2)  # [B, H, N_past+N, D]

# Step D: ⚠️ 不做 eviction — 完整交叉注意力
# Q 来自当前帧 [B, H, N, D]
# K, V 包含完整历史 + 当前 [B, H, N_past+N, D]
output = F.scaled_dot_product_attention(q, k, v)
# → 每个当前帧 token attend 到 ALL 历史 + 当前 token

# Step E: 输出投影
output = self.proj(output.transpose(1, 2).reshape(B, N, C))

# 返回: 注意力输出 + 当前帧的 KV（历史 KV 不动）
return output, (k, v, k_current, v_current, past_kv), scores=None
```

**关键设计**: eviction 被延迟到 commit 阶段。attention 计算时使用完整的 cache 历史，保证注意力质量不受 eviction 影响。

### 4.2 MLP + 残差

```python
x_after_attn = x + attn_output
mlp_output = FFN(x_after_attn)
x_after_mlp = x_after_attn + mlp_output
```

### 4.3 计算 Importance 分数

```python
# repr_shift_spatial 策略
importance = ||x_before_mlp||² - ||x_before_mlp - mlp_output||²
# 加上 3×3 空间平滑 (patch grid 上 avg pool)

# 直觉: MLP residual 越大的 token → 对重建越重要
# 这是 token 在经过 attention + FFN 后的 "表征偏移量"
```

### 4.4 返回

```python
return x_after_mlp, (k_current, v_current), new_importance
# 返回 3 元组: tokens, pending KV, importance scores
```

---

## 5. FIFO topK=80 保护 (ovggt.py + frontend_cache.py)

### 5.1 触发条件

FIFO_SWAP 在 anchor 数量超过 `max_anchors=3` 时触发。例如：

```
frame 0:  slot 0 (global anchor)
frame 8:  slot 1 promoted (keyframe)
frame 16: slot 2 promoted (keyframe)
frame 24: slot 3 promoted → FIFO_SWAP!
          → slot 1 demoted (最老的 keyframe)
          → slot 1 的所有 token: anchor_slot 从 1 → -1
          → 这些 token 变成 eviction candidates
```

### 5.2 protect_topk_on_demotion_() 逻辑

```python
def protect_topk_on_demotion_(self, demoted_slot: int, keep_count: int):
    """FIFO demotion 前，将 demoted slot 中分数最高的 K 个 token
    重新分配到 slot 0 (global anchor)，使其在 demotion 后仍受保护。"""

    for b_idx in range(B):
        # 1. 找到所有属于被降级 slot 的 token
        slot_mask = (metadata.anchor_slot[b_idx] == demoted_slot)
        indices = torch.nonzero(slot_mask)  # ~200-300 个

        # 2. 按 importance 降序排列
        scores = metadata.importance[b_idx, indices]
        _, top_local = torch.topk(scores, k=keep_count)  # k=80

        # 3. 将 top-80 token 的 anchor_slot 改为 0 (global anchor)
        top_indices = indices[top_local]
        metadata.anchor_slot[b_idx, top_indices] = 0

    # 重新计算 protected_count
    self.protected_count = self._compute_protected_count()
```

### 5.3 为什么 topK=80

| topK | 50帧 ATE | 200帧 ATE | 说明 |
|------|---------|----------|------|
| 0 | 0.0196 | 0.0358 | 无保护 |
| 60 | 0.0198 | 0.0757 | 不足 |
| 75 | 0.0203 | 0.0272 | 较好 |
| **80** | **0.0203** | **0.0258** | **最优** |
| 90 | 0.0204 | 0.0265 | 略过 |
| 110 | 0.0204 | 0.1234 | 崩溃 |
| 200 | 0.0203 | 0.4863 | 崩溃 |

- 太少: 关键几何锚点丢失，远期 ATE 退化
- 太多: protected token 占据 cache 空间，新帧 token 挤不进来 → 崩溃
- 80: 仅占总 budget 200000 的 0.04%，最优平衡

---

## 6. commit_pending_update_() — Cache 更新核心 (frontend_cache.py)

每层 cache 独立执行以下 5 个步骤：

### 6.1 帧内 Pruning（noIntra: 跳过）

```python
if not metadata.has_anchor_tokens() and intra_frame_keep_ratio < 1.0:
    # 按 importance 保留 top-K% 的当前帧 token
    keep_count = max(int(N * intra_frame_keep_ratio), 1)
    _, top_indices = torch.topk(importance, k=keep_count)
    k_current = k_current.gather(top_indices)
    v_current = v_current.gather(top_indices)
```

**noIntra 配置**: `intra_frame_keep_ratio = 1.0`，此步骤完全跳过。所有当前帧 token 原封不动进入 cache。

### 6.2 Append — 追加到 Cache

```python
self.append_(k_current, v_current, metadata_current)
# K:  [B, H, N_past, D] → [B, H, N_past+N_new, D]
# V:  同上
# metadata: 追加 {anchor_slot, importance, frame_id, 3d_position}
```

当前帧的所有 token 无条件追加到 cache 尾部。

### 6.3 Reorder by Anchor Slots

```python
if metadata.has_anchor_tokens():
    self.reorder_by_anchor_slots_()
```

重排 token 顺序，保证 eviction 时 anchor tokens 不会被误删：

```
重排后顺序:
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ slot 0 (全局) │ slot 1 (KF1) │ slot 2 (KF2) │ slot -1      │
  │ protected    │ protected    │ protected    │ candidates   │
  └──────────────┴──────────────┴──────────────┴──────────────┘
   ← protected_count = 所有 anchor_slot >= 0 的 token 数 →
```

### 6.4 Voxel Deduplication（帧内关闭，仅跨帧）

```python
self.apply_voxel_dedup_(config, current_frame_id)
```

内部调用 `_dedup_single_batch()`：

```python
def _dedup_single_batch(self, b, config, current_frame_id):
    # 1. 将所有 token 的 3D 位置投影到 active coordinate frame
    # 2. 量化到 10cm³ voxel grid
    # 3. 找到所有 voxel 内的 token 冲突

    # ---- 跨帧 dedup (保留) ----
    # protected (anchor_slot >= 0) vs current (frame_id == current)
    for voxel in occupied_voxels:
        if has_protected_token and has_current_token:
            # improved score comparison
            if current_score < protected_score - margin:
                discard_current(voxel.current_token)
    # 只在当前帧 token 明确劣于已缓存 protected token 时才丢弃
    # 不会误删有效 token

    # ---- 帧内 dedup (关闭!) ----
    if config.intra_frame_dedup_enabled:   # ← False, 整块跳过
        # 原本逻辑: 同一帧内投影到同一 voxel 的 token 只保留最高分
        # 问题: 同一 voxel 可能包含来自不同 patch 的有效观察
        #       只保留 1 个会丢失遮挡边界/纹理细节
        #       这是 200帧 ATE 从 0.026 退化到 0.050 的根因
        for voxel in occupied_voxels:
            tokens_in_voxel = [t for t in voxel if t.frame_id == current]
            keep_only_highest_score(tokens_in_voxel)
```

**noIntra 的效果**:
- ✅ 跨帧 dedup: 防止 cache 被重复 token 填满，安全
- ❌ 帧内 dedup: 跳过激进剪枝，保留同一帧内的所有 token

实验验证:

| 配置 | 50帧 ATE | 200帧 ATE |
|------|---------|----------|
| full_dedup (帧内=ON) | 0.0220 | 0.0502 |
| **no_intra_dedup (帧内=OFF)** | **0.0196** | **0.0358** |
| no_dedup_at_all | 0.0196 | 0.0358 |

关闭帧内 dedup 的效果等同于完全关闭 dedup，说明跨帧 dedup（improved score comparison）零副作用。

### 6.5 Budget Eviction — 超限淘汰

```python
if self.num_tokens() <= cache_budget:
    return  # 没超限，不需要 eviction

# 计算 importance scores (当前帧 token 用新计算的, 老 token 用缓存的)
importance_scores, num_new_tokens = self._current_frame_importance(frame_id)

# 调用 hybrid eviction
final_k, final_v, avg_score, kept_indices = attn_module.eviction(
    self.k, self.v,
    cache_budget,            # 该层允许保留的 token 总数
    self.protected_count,    # anchor tokens 数量（永远保留）
    importance_scores=importance_scores,
    num_new_tokens=num_new_tokens,
    importance_weight=importance_weight,
)

# 更新 cache
self.k = final_k
self.v = final_v
self.metadata = self.metadata.index_select(kept_indices)
```

#### Hybrid Eviction 的具体逻辑 (attention.py eviction()):

```
总 token 数: N_total = N_protected + N_old + N_new
需要保留:    cache_budget (如 8333 per layer)
需要淘汰:    N_total - cache_budget

Step 1: 分离 protected 和 candidates
  anchor_k:    [B, H, N_protected, D]  ← 永远保留
  candidate_k: [B, H, N_candidates, D] ← 从中淘汰
  (N_protected = protected_count = 所有 anchor_slot >= 0 的 token)

Step 2: 对 candidate 分类打分

  老 token (来自之前帧):
    score = cosine_diversity = 1 - mean cosine_similarity
    → 越独特 (与均值差异越大) → 越高分 → 越应该保留
    weight = (1 - importance_weight)

  新 token (来自当前帧):
    score = importance (repr_shift_spatial)
    → MLP residual 越大 → 越高分 → 越重要
    weight = importance_weight

Step 3: 归一化 + 加权合并
  combined = cat(weighted_old_scores, weighted_new_scores)

Step 4: Top-K 选择
  _, kept_indices = torch.topk(combined, k=num_to_keep)

Step 5: Gather + 拼回
  final_k = cat(anchor_k, candidate_k[:, :, kept_indices])
  final_v = cat(anchor_v, candidate_v[:, :, kept_indices])
```

---

## 7. 完整一帧的时序图

```
Frame i 进入
    │
    ▼
┌── Patch Embed + Token Assembly ──────────────────────────────────┐
│  image [1,1,3,H,W] → patch_embed → patch_tokens [1, 1370, 768]  │
│  tokens = [camera(1), register(4), patch(1370)] = [1, 1375, 768] │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  24 层 × (Frame+Global) │
                    │                         │
                    │  Global Block[j]:        │
                    │  ┌───────────────────┐  │
                    │  │ Q = current tokens│  │
                    │  │ K,V = past + curr │  │
                    │  │ SDPA → attn_out   │  │
                    │  │ (完整交叉注意力,   │  │
                    │  │  不做 eviction)    │  │
                    │  ├───────────────────┤  │
                    │  │ MLP → residual    │  │
                    │  ├───────────────────┤  │
                    │  │ importance =      │  │
                    │  │  ||x||²-||x-mlp||²│  │
                    │  │  + spatial smooth │  │
                    │  └───────────────────┘  │
                    │  输出: pending_update[j] │
                    │  {k_curr, v_curr, imp,  │
                    │   frame_id, budget}      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ 任务头 (Pose/Depth/3D)   │
                    │ camera_head → pose       │
                    │ depth_head → depth_map   │
                    │ point_head → point_map   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Keyframe Event 判断      │
                    │                         │
                    │ frame 0:  slot 0 (global)│
                    │ frame 8:  slot 1 promote │
                    │ frame 16: slot 2 promote │
                    │ frame 24: slot 3 promote │
                    │   → FIFO_SWAP!           │
                    │   → slot 1 demoted       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ FIFO topK=80 保护       │
                    │ (仅 FIFO_SWAP 时触发)    │
                    │                         │
                    │ slot 1 的 ~250 个 token │
                    │ 按 importance 排序      │
                    │ top-80 → reassign slot 0│
                    │ 剩余 → anchor_slot = -1 │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ apply_keyframe_event_   │
                    │ 更新 anchor_slot 元数据  │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────▼──────────────────────┐
           │  对 24 层各自 commit_pending_update         │
           │                                           │
           │  Layer[j]:                                │
           │                                           │
           │  ① Append                                │
           │     k_current, v_current + metadata       │
           │     全部追加到 cache                       │
           │                                           │
           │  ② Reorder by anchor slots                │
           │     protected tokens 排前面               │
           │     candidates 排后面                      │
           │                                           │
           │  ③ Voxel Dedup                            │
           │     ✅ 跨帧: protected vs current         │
           │        只在 current 明确劣于 protected 时  │
           │        才丢弃 (improved score comparison)  │
           │     ❌ 帧内: 跳过!                         │
           │        不对同一帧内 token 做激进剪枝        │
           │                                           │
           │  ④ Budget Eviction (仅超限时)              │
           │     if tokens > budget:                   │
           │       protected 永远保留                   │
           │       candidates: hybrid scoring           │
           │         old → cosine diversity             │
           │         new → importance score             │
           │       topk → 保留最高分                     │
           └───────────────────────────────────────────┘
                                 │
                                 ▼
                          Frame i 完成
                    cache_states[0..23] 更新完毕
                         等待 Frame i+1
```

---

## 8. 关键设计总结

### 8.1 核心原则

| 原则 | 实现 |
|------|------|
| Attention 质量 | eviction 延迟到 commit 阶段，attention 看到完整 cache |
| 保护关键 token | anchor slot 机制 + protected_count 保证 keyframe token 不被 evict |
| FIFO 降级保护 | topK=80 将最重要的历史几何锚点 reassign 到 global slot |
| 跨帧去重 | improved score comparison，只在确认冗余时才丢弃 |
| 不做帧内剪枝 | 关闭 intra-frame dedup，避免丢失同一帧内的有效 token |
| 重要性信号 | repr_shift_spatial: MLP residual 的 L2 范数变化量 |

### 8.2 noIntra+fifo80 vs 其他配置

| 配置 | 50帧 ATE | 200帧 ATE | 说明 |
|------|---------|----------|------|
| Legacy (coverage) | 0.0160 | 0.0260 | 批量推理基线 |
| FE8 + full dedup | 0.0220 | 0.0502 | 帧内 dedup 导致退化 |
| **FE8 + noIntra + fifo80** | **0.0203** | **0.0258** | **本文档配置** |
| FE8 + noIntra (无 fifo) | 0.0196 | 0.0358 | 缺少 FIFO 保护 |
| FE8 + ToMe | 0.0177 | 0.0297 | 短序列强，长序列差 |

### 8.3 已知限制

1. **topK 敏感性**: 最优值 80 在 7-Scenes/chess 上验证，不同场景可能需要微调
2. **topK > 100 崩溃**: 过多 protected token 占据 cache，新帧 token 无法正常加入
3. **office 场景**: 改善有限 (0.0517→0.0500)，可能需要不同参数
4. **极长序列 (>500帧)**: 缺少帧内 dedup 可能导致 cache 中冗余 token 累积
