# Frontend 长序列精度优化记录

## 问题

voxel-vggt 的 Frontend 增量推理模式在短序列（10-50帧）表现良好，但长序列（200帧）ATE RMSE 相比 Legacy 批量推理大幅退化：

| 配置 | 50帧 | 100帧 | 200帧 |
|------|------|------|-------|
| Legacy (coverage) | 0.0160 | 0.0160 | 0.0260 |
| FE int=8 原始 dedup | 0.0265 | 0.0303 | 0.1534 |
| FE int=8 improved dedup | 0.0220 | 0.0267 | 0.0502 |
| FE int=8 noDedup | 0.0196 | 0.0210 | 0.0358 |

200帧从 Legacy 的 0.0260m 退化到 0.0502m（+93%）。

## 根因分析

通过系统性 ablation 定位到两个独立误差源：

### 误差源 1：帧内 voxel dedup 过度剪枝（占 100% 的退化）

`_dedup_single_batch` 中的帧内去重逻辑将同一帧内投影到相同 10cm voxel 的 token 只保留分数最高的 1 个，丢弃其余。在纹理丰富或遮挡频繁的区域，同一 voxel 可能包含来自不同图像 patch 的有效观察，全部丢弃导致几何信息丢失。

验证方法：对比 `no_intra_dedup`（仅关闭帧内）和 `no_dedup_at_all`（全部关闭），结果完全相同：

| 配置 | 50帧 | 200帧 |
|------|------|-------|
| full_dedup (帧内=ON) | 0.0220 | 0.0502 |
| no_intra_dedup (帧内=OFF) | 0.0196 | 0.0358 |
| no_dedup_at_all | 0.0196 | 0.0358 |

这说明跨帧 dedup（protected vs current，使用 improved score comparison）零副作用，帧内 dedup 是唯一退化源。

### 误差源 2：FIFO swap 时 token 流失（额外 28% 可恢复）

当 `max_history_anchors=3` 满后，新 keyframe promotion 触发 FIFO_SWAP，最老 anchor（slot 1）被降级。所有属于该 anchor 的 token 失去保护（`anchor_slot` 设为 -1），在后续 budget eviction 中被移除。这些 token 包含重要的历史几何信息。

验证：在关闭帧内 dedup 基础上，加入 FIFO topK 保护后 200帧 ATE 从 0.0358 降至 0.0258。

### 排除的假设

- **Task A（延迟 reorder）**：将 `reorder_by_anchor_slots_` 延迟到 eviction 前执行。实验表明 reorder 位置变化对精度无影响（dedup 用 `metadata.anchor_slot` 不依赖 reorder），已撤销。
- **Task B（dedup cooldown）**：新增 `dedup_cooldown_frames` 参数，让新 promote 的 anchor 在 N 帧内不参与跨帧冲突检测。实验表明跨帧 dedup 在 improved score comparison 下本就不丢弃任何 token，cooldown 无效果。

## 最终修改方案

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/ovggt/utils/frontend_cache.py` | 新增 `intra_frame_dedup_enabled`、`fifo_keep_topk`、`dedup_cooldown_frames` 配置项；新增 `protect_topk_on_demotion_` 方法；帧内 dedup 加条件开关 |
| `src/ovggt/models/ovggt.py` | FIFO_SWAP 事件时调用 `protect_topk_on_demotion_` |

### 修改 1：帧内 dedup 开关

**文件**: `src/ovggt/utils/frontend_cache.py`

FrontendCacheConfig 新增字段（第 30 行）：
```python
intra_frame_dedup_enabled: bool = True
```

`_dedup_single_batch` 中帧内去重逻辑加条件判断（第 595-616 行）：
```python
# 帧内去重：保留每个体素中评分最高的token
if config.intra_frame_dedup_enabled:
    survivor_mask = ~discard_current_mask if ...
    # ... 原有帧内去重逻辑 ...
```

设为 `False` 时跳过帧内去重，仅保留跨帧 score-aware 去重。

### 修改 2：FIFO topK 保护

**文件**: `src/ovggt/utils/frontend_cache.py`

FrontendCacheConfig 新增字段（第 31 行）：
```python
fifo_keep_topk: int = 0  # 0=disable
```

LayerCacheState 新增方法（第 353-368 行）：
```python
def protect_topk_on_demotion_(self, demoted_slot: int, keep_count: int) -> None:
    """FIFO demotion 前，将 demoted slot 中分数最高的 K 个 token
    重新分配到 slot 0（global anchor），使其在 demotion 后仍受保护。"""
    if self.metadata is None or self.num_tokens() == 0 or keep_count <= 0:
        return
    for b_idx in range(self.metadata.anchor_slot.shape[0]):
        slot_mask = self.metadata.anchor_slot[b_idx] == demoted_slot
        indices = torch.nonzero(slot_mask, as_tuple=False).squeeze(-1)
        if indices.numel() <= keep_count:
            continue
        scores = self.metadata.importance[b_idx, indices]
        _, top_local = torch.topk(scores, k=keep_count)
        top_indices = indices[top_local]
        self.metadata.anchor_slot[b_idx, top_indices] = 0
    self.protected_count = self._compute_protected_count()
```

### 修改 3：ovggt.py 集成

**文件**: `src/ovggt/models/ovggt.py`（第 497-504 行）

在 `_inference_frontend` 的 per-layer 循环中，FIFO_SWAP 事件触发时在 `apply_keyframe_event_` 前调用保护：
```python
is_fifo_swap = str(getattr(event, "event_type", None)).endswith("FIFO_SWAP")
if is_fifo_swap and self.frontend_cache_config.fifo_keep_topk > 0:
    demoted_slot = getattr(event, "demoted_slot", None)
    if demoted_slot is not None:
        cache_states[layer_idx].protect_topk_on_demotion_(
            demoted_slot=demoted_slot,
            keep_count=self.frontend_cache_config.fifo_keep_topk,
        )
cache_states[layer_idx].apply_keyframe_event_(event)
```

## 最优配置与实验结果

### 推荐配置

```python
OVGGT(
    mode='frontend_eval',
    total_budget=200000,
    frontend_pose_encoding_type=ABS_POSE_ENCODING,
    frontend_cache_config=FrontendCacheConfig(
        enabled=True,
        dedup_enabled=True,           # 保留跨帧 score-aware 去重
        intra_frame_dedup_enabled=False,  # 关闭帧内 voxel 去重
        fifo_keep_topk=80,            # FIFO 时保留 top-80 token
    )
)
# 推理调用
model.inference(inputs,
    history_anchor_strategy='fixed_interval',
    anchor_interval=8,
    max_anchors=3)
```

### fifo_keep_topk 参数敏感性

在 chess/seq-03 200帧上的扫描结果：

| topk | 50帧 ATE | 200帧 ATE | 状态 |
|------|---------|----------|------|
| 0 | 0.0196 | 0.0358 | 无保护 |
| 60 | 0.0198 | 0.0757 | 不足 |
| 70 | 0.0203 | 0.0286 | 接近 |
| 75 | 0.0203 | 0.0272 | 较好 |
| **80** | **0.0203** | **0.0258** | **最优** |
| 85 | 0.0204 | 0.0267 | 略过 |
| 90 | 0.0204 | 0.0265 | 过量 |
| 100 | 0.0203 | 0.0282 | 过量 |
| 110 | 0.0204 | 0.1234 | 崩溃 |
| 200 | 0.0203 | 0.4863 | 崩溃 |

topk 在 75-90 区间效果最好，超过 100 后保护 token 过多导致 cache 溢出，新帧 token 无法正常加入。

### 多场景验证（200帧）

| 场景 | Legacy | FE8 原始 | FE8 优化 | 改善幅度 |
|------|--------|---------|---------|---------|
| chess/seq-03 | 0.0260 | 0.0502 | **0.0258** | -48.6% |
| fire/seq-03 | 0.0271 | 0.0387 | **0.0281** | -27.4% |
| office/seq-03 | 0.0287 | 0.0517 | 0.0500 | -3.3% |
| redkitchen/seq-03 | 0.0175 | 0.1195 | 0.0307 | -74.3% |

- chess、fire：优化后基本追平 Legacy
- redkitchen：从严重退化恢复到可用水平，但仍高于 Legacy（可能需要针对大场景调整 topk）
- office：改善有限，可能需要不同参数或该场景特性导致

## 技术原理

### 为什么帧内 dedup 有害

帧内 dedup 的设计初衷是避免同一帧内重复 token 占用 cache 空间。但在 voxel_size=0.1m（10cm）分辨率下：
- 同一 voxel 可能包含来自不同图像 patch 的多个 token
- 这些 token 虽然投影到同一 3D 位置附近，但携带不同的视觉特征信息
- 只保留分数最高的 1 个会丢失遮挡边界、纹理细节等

跨帧 dedup 则不同：它用 score-aware 比较（improved dedup），只在当前帧 token 明确劣于已缓存 protected token 时才丢弃，不会误删有效信息。

### 为什么 FIFO topK 保护有效

每次 FIFO_SWAP 降级最老 anchor 时，其全部 token（通常 ~200-300 个）立即变为可驱逐状态。budget eviction 倾向于移除这些"旧"token 给新帧让路，导致关键的历史几何锚点丢失。

保留 top-80 个最高分 token 的原理：
- 这些 token 是历史 anchor 中最重要的几何支撑点
- 将它们 reassign 到 slot 0（global anchor）使其持续受保护
- 80 个 token 仅占总 cache 的 ~0.04%（200000 budget），开销极小
- 超过 ~100 个后，过多 protected token 占据 cache 空间，新帧 token 被挤出导致崩溃

## 已知限制

1. **topk 敏感性**：最优值 80 在 chess 上验证，不同场景/序列可能需要微调。topk>100 存在崩溃风险。
2. **office 场景**：改善有限（0.0517→0.0500），可能需要针对该场景的特殊处理。
3. **redkitchen 场景**：虽大幅改善但仍高于 Legacy，可能因为该场景 texture-less 区域多，对 token 保留策略更敏感。
4. **帧内 dedup 关闭的长期影响**：在极长序列（>500帧）中，缺少帧内去重可能导致 cache 中冗余 token 累积，需要后续验证。

---

## Phase 1 SOTA 方法实验记录（2026-05-21）

### 实验目标

用 SOTA 方法系统性替换启发式规则：
- **Phase 1a**: ToMe token merging 替换帧内 voxel dedup（训练免费）
- **Phase 1b**: SAGE-KV 累积 attention 替换 eviction importance score（训练免费）

### Phase 1a: ToMe Token Merging

**方法**：用 bipartite soft matching 按 key cosine similarity 合并相似 token（加权平均），替代硬去重（保留 top-1 per voxel）。

**实现**：在 `frontend_cache.py` 中添加 `_tome_merge_current_frame_` 方法，在 `apply_voxel_dedup_` 中当 `tome_merge_enabled=True` 时调用。

**配置**：`tome_merge_enabled=True, tome_similarity_metric='key'`

**结果**：

| 配置 | 50帧 | 200帧 |
|------|------|-------|
| Legacy | 0.0160 | 0.0260 |
| full_dedup | 0.0220 | 0.0502 |
| noIntra+fifo80 | 0.0203 | 0.0258 |
| **ToMe_key** | **0.0177** | **0.0297** |
| ToMe+fifo80 | 0.0176 | 0.0386 |

**结论**：ToMe 在短序列（50帧）显著优于所有其他配置（0.0177 vs noIntra 0.0196），是最接近 Legacy 的 Frontend 配置。200帧 ATE=0.0297 优于 noIntra (0.0358) 但差于 noIntra+fifo80 (0.0258)。ToMe 的 soft merging 确实保留了更多信息。

### Phase 1b: SAGE-KV 累积 Attention

**方法**：在 attention forward 中提取 column sums（每个 KV token 被当前帧 query 关注的平均注意力），累积跨帧作为 eviction 的重要性信号。

**实现**：
- `attention.py`：添加 `return_attn_col_sums` 参数，分块计算 column sums 避免长序列 OOM
- `block.py`、`aggregator.py`：传递 attn_col_sums
- `frontend_cache.py`：在 `LayerCacheState` 中累积 attention，用于 eviction 评分

**配置**：`sage_kv_enabled=True`

**结果**：

| 配置 | 50帧 | 200帧 |
|------|------|-------|
| noIntra (baseline) | 0.0196 | 0.0358 |
| SAGE-KV only | 0.0250 | 0.0467 |
| ToMe+SAGE-KV | 0.0177 | 0.0326 |
| ToMe+SAGE-KV+fifo80 | 0.0179 | 0.1514 |

**结论**：SAGE-KV 在此场景下**无效甚至有害**。原因：
1. 累积 attention 偏向旧 token（存在越久积累越多），导致 eviction 决策偏差
2. 3D ViT attention 比 LLM 更密集空间化，"受欢迎" token 不等于"对重建重要"的 token
3. SAGE-KV 设计用于 LLM 长上下文 KV cache，不适用于 3D 视觉 transformer

### Phase 1 总结

| 方法 | 适用性 | 50帧最佳 | 200帧最佳 |
|------|--------|---------|----------|
| noIntra+fifo80 | 启发式，已验证 | 否 (0.0203) | **是 (0.0258)** |
| ToMe | 训练免费，短序列强 | **是 (0.0177)** | 否 (0.0297) |
| SAGE-KV | 不适用于 3D ViT | 否 (0.0250) | 否 (0.0467) |

**决策**：Phase 1 的 SOTA 方法中只有 ToMe 有效但仅限于短序列。保留已验证的启发式方案（noIntra+fifo80）作为基线。回退 ToMe 和 SAGE-KV 代码，进入 Phase 2 端到端学习。

---

## Phase 2 VisionSelector 实现记录（2026-05-21）

### 设计

用轻量级可学习 scorer 替换启发式 importance score（repr_shift / MLP residual L2 norm）。每个 global block 独立一个 scorer，在 `frontend_cache_mode` 分支中覆盖 heuristic importance。

### 架构

```
TokenScorer:
  LayerNorm(embed_dim) → Linear(D, D//4) → GELU → Linear(D//4, 1) → squeeze
  Input: [B, N, C] token embeddings → Output: [B, N] importance scores
  Parameters per scorer: ~149K (embed_dim=768) or ~263K (embed_dim=1024)
  Total: 24 scorers = 3.6M / 6.3M parameters (0.4% of total model)
```

训练时使用 sigmoid-based soft top-K（temperature annealing），推理时直接输出 raw scores 供 top-K eviction 使用。

### 文件清单

| 文件 | 修改/新建 | 说明 |
|------|----------|------|
| `src/ovggt/layers/token_scorer.py` | 新建 | TokenScorer 模块 |
| `src/ovggt/layers/block.py` | 修改 | Block.token_scorer 属性 + frontend_cache_mode 中覆盖 importance |
| `src/ovggt/models/aggregator.py` | 修改 | Aggregator.init_token_scorers() 方法 |
| `src/ovggt/models/ovggt.py` | 修改 | use_token_scorer 参数 |
| `src/ovggt/utils/frontend_cache.py` | 修改 | use_learned_scorer 配置字段 |
| `src/train_token_scorer.py` | 新建 | 训练脚本（仅训练 scorer，冻结其余） |
| `config/train_token_scorer.yaml` | 新建 | 训练配置 |

### 训练策略

- **冻结**：除 token_scorer 外全部冻结，仅训练 0.4% 参数
- **Loss**：FrontendSupervisedLoss（depth + point map + camera pose，GT 监督）
- **Curriculum annealing**：temperature 从 1.0 → 0.01，cosine schedule
  - 初期：soft sigmoid selection，梯度流畅
  - 后期：接近 hard top-K，匹配推理行为
- **学习率**：1e-4（比 backbone 高 10x，因为随机初始化）
- **Epochs**：10（轻量模块，快速收敛）

### 使用方式

**训练**：
```bash
cd src
accelerate launch train_token_scorer.py
```

**推理**（训练完成后）：
```python
model = OVGGT(
    mode='frontend_eval',
    use_token_scorer=True,
    frontend_cache_config=FrontendCacheConfig(
        intra_frame_dedup_enabled=False,
        fifo_keep_topk=80,
        use_learned_scorer=True,
    ),
)
# Load pretrained + scorer weights
sd = torch.load('checkpoints.pth', ...)
model.load_state_dict(sd, strict=False)
scorer_sd = torch.load('token_scorers_final.pth')
model.aggregator.token_scorers.load_state_dict(scorer_sd)
```

### 待验证

- [ ] 训练 scorer 后 200帧 ATE 是否低于启发式基线 0.0258
- [ ] 多场景泛化性（chess/fire/office/redkitchen）
- [ ] 极长序列（>500帧）稳定性
