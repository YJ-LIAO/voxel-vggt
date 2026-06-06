# Phase 2: TokenScorer 可学习 Importance 评分方案

> 基于 noIntra+fifo80 基线，用可学习网络替代启发式 importance 评分
> 日期: 2026-05-25

---

## 1. 背景：当前启发式评分的问题

noIntra+fifo80 基线使用 `repr_shift_spatial` 计算 token importance：

```
importance = ||mlp_residual||₂    (MLP 残差 L2 范数)
           → 3×3 高斯平滑
           → 0.3×smoothed + 0.7×raw
```

该方案存在 6 个问题：

| # | 问题 | 说明 |
|---|------|------|
| 1 | 代理指标 ≠ 任务目标 | MLP 修正幅度大不代表对重建重要（噪声 token 也会高） |
| 2 | 新旧 token 评分标准不统一 | 旧 token 用 cosine diversity，新 token 用 repr_shift，两个不同度量在 [0,1] 上直接竞争 |
| 3 | 层间独立无一致性 | 24 层各自独立计算，同一 token 在不同层分数差异大 |
| 4 | 固定空间假设 | 3×3 高斯核 + α=0.3 硬编码，假设重要 token 应成片出现（遮挡边界不满足） |
| 5 | depth_conf 未校准 | depth head 训练目标是深度质量，不是 token importance，0.5/0.5 权重无依据 |
| 6 | 与下游任务脱节 | 评分和最终 depth/point/track loss 之间没有梯度通路 |

---

## 2. 设计目标

1. **可学习**：用轻量神经网络替代启发式公式，参数通过训练优化
2. **统一评分**：所有 token（无论新旧）使用同一个 scorer 输出的分数
3. **端到端可优化**：scorer 的分数直接影响重建损失，梯度可以反传
4. **最小侵入**：scorer 作为可选模块，不影响现有推理路径
5. **参数高效**：新增参数 < 模型总参数的 1%

---

## 3. 架构设计

### 3.1 TokenScorer 模块

```python
class TokenScorer(nn.Module):
    """轻量级可学习 importance 评分器。

    架构: LayerNorm → Linear(D, D//4) → GELU → Linear(D//4, 1) → Sigmoid
    输入: token embeddings [B, N, C]  (x_after_mlp)
    输出: importance scores [B, N]，范围 (0, 1)
    """
    def __init__(self, embed_dim: int, bottleneck_dim: int = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 4

        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.scorer(x).squeeze(-1)  # [B, N, C] → [B, N]
```

**输出范围约束**：末尾使用 `Sigmoid` 将输出约束到 (0, 1)。原因：teacher 的 heuristic importance（`mlp_output.norm(dim=-1)`）数值范围不确定，随机初始化的 scorer 输出可能差几个数量级，导致 MSE loss 极大、梯度不稳定。将 scorer 输出约束到 (0, 1) 并对 teacher target 做 min-max 归一化，保证两者同分布。

**参数量**：
- embed_dim=1024: 每层 Linear(1024→256)=262,400 + Linear(256→1)=257 + LayerNorm(1024)=2,048 = **~264K**
- 24 层合计: 264K × 24 = **~6.3M**（模型总参数的 ~0.7%）

**输入选择**：使用 `x_after_mlp`（MLP 后的完整 token 表示），包含 attention + MLP 的全部信息。

### 3.2 集成位置

每个 global block 的 `frontend_cache_mode` 分支中，在 MLP 计算之后：

```
当前流程:
  Attention → x_after_attn → MLP → x_after_mlp
                                      ↓
                              heuristic importance (repr_shift_spatial)
                                      ↓
                              return (x_after_mlp, (k, v), importance)

修改后:
  Attention → x_after_attn → MLP → x_after_mlp
                                      ↓
                              ├── heuristic importance (teacher, 训练时用)
                              │
                              └── TokenScorer(x_after_mlp) → learned_score + distill_loss
                                      ↓
                              return (x_after_mlp, (k, v), learned_score, distill_loss)
```

注意：`distill_loss` 是通过函数返回值沿调用链向上传递（Block → Aggregator → OVGGT → 训练脚本），**不使用** Python list 或模块属性存储，以保证 autograd 梯度链完整且支持 DDP 多 GPU 训练。

### 3.3 数据流

```
Block (frontend_cache_mode):
  x_after_mlp [B, N, 1024]
      │
      ├── heuristic: repr_shift_spatial → heuristic_score [B, N]  (训练时作 teacher)
      │
      └── TokenScorer(x_after_mlp) → learned_score [B, N]
              │
              ├── 训练+蒸馏: distill_loss = MSE(learned_score, heuristic_score.detach())
              │    该 loss 沿返回值链传至训练脚本，与主 loss 加权求和
              │
              └── 推理: 直接输出 learned_score（使用 torch.no_grad()）

learned_score 进入 PendingLayerUpdate.importance_current
      │
      ├── voxel dedup: 复合分数 = importance_weight×norm(learned_score) + depth_conf_weight×norm(depth_conf)
      │    （Stage A 蒸馏时 depth_conf 仍参与 voxel dedup 评分；Stage B 可考虑降低 depth_conf_weight）
      ├── budget eviction: hybrid scoring 用 learned_score 替代 heuristic 的新 token 打分
      │    （旧 token 仍用 cosine diversity，新旧 token 统一归一化后竞争 — 比旧方案改善但未彻底解决）
      └── FIFO protect_topk: 用 learned_score 选 top-80 token
```

**关于 depth_conf 的说明**：Stage A 蒸馏阶段，scorer 仅学习模仿 heuristic importance，不涉及 depth_conf。因此 voxel dedup 的复合评分仍需 `depth_conf` 参与（与现有逻辑一致）。Stage B 端到端训练后，scorer 可能隐式学到 depth 相关信息，此时可以尝试降低 `depth_conf_weight` 或设为零，让 scorer 独立承担评分职责。这不作为 Stage A 的目标。

---

## 4. 修改文件清单

| 文件 | 操作 | 修改内容 |
|------|------|----------|
| `src/ovggt/layers/token_scorer.py` | **新建** | TokenScorer 类定义 |
| `src/ovggt/layers/block.py` | 修改 | Block 新增 `token_scorer` 属性，frontend_cache_mode 中集成，返回值增加 `distill_loss` |
| `src/ovggt/models/aggregator.py` | 修改 | 新增 `init_token_scorers()`，forward 返回值增加 `distill_losses` |
| `src/ovggt/models/ovggt.py` | 修改 | 新增 `use_token_scorer` 参数，forward 返回值携带 `distill_losses` |
| `src/train_frontend.py` | 修改 | 新增 `freeze_stage_a_scorer_only()` 冻结函数，loss 计算中加入 `distill_loss` |
| `config/train_token_scorer.yaml` | **新建** | 蒸馏训练配置 |

删除的修改项：
- ~~`FrontendCacheConfig` 新增 `use_learned_scorer` 字段~~ — 不需要，直接通过 `model.aggregator.token_scorers is not None` 判断是否启用

---

## 5. 详细修改方案

### 5.1 新建 `src/ovggt/layers/token_scorer.py`

```python
class TokenScorer(nn.Module):
    """轻量级可学习 importance 评分器。

    架构: LayerNorm → Linear(D, D//4) → GELU → Linear(D//4, 1) → Sigmoid
    输入: token embeddings [B, N, C]  (x_after_mlp)
    输出: importance scores [B, N]，范围 (0, 1)

    注意：当前训练配置 batch_size=1，forward 接口支持 B≥1 但在多 batch
    下 PendingLayerUpdate 的 frame_id 管理需要验证（详见第 12 节）。
    """
    def __init__(self, embed_dim: int, bottleneck_dim: int = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 4
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.scorer(x).squeeze(-1)  # [B, N, C] → [B, N]
```

说明：
- 末尾 `Sigmoid` 将输出约束到 (0, 1)，与归一化后的 teacher target 同分布，避免训练初期 MSE 爆炸（详见第 12 节 Review-6）
- 去掉了原始设计中的 `temperature` buffer 和 `soft_select`/`hard_select` 方法 — scoring 是回归任务（MSE to heuristic），不需要 soft/hard selection 操作
- 去掉了 `get_importance_scores` — 推理时 `torch.no_grad()` 由调用方控制

### 5.2 修改 `src/ovggt/layers/block.py`

> **注意：需要新增 `import torch.nn.functional as F`**。当前 `block.py:1-15` 的 import 列表中不含 `torch.nn.functional`，但修改后的代码使用 `F.mse_loss()` 计算蒸馏损失。缺少此 import 在运行时会报 `AttributeError`。

**`__init__` 新增**：
```python
self.token_scorer = None  # 由 Aggregator.init_token_scorers() 设置
```

**`forward` 的 frontend_cache_mode 分支**（当前第 169-198 行），修改为：

```python
# Step 1: 如果有 learned scorer，优先使用
distill_loss = None
if self.token_scorer is not None:
    learned_scores = self.token_scorer(x_after_mlp)  # [B, N]

    # 训练时：计算 heuristic 作为 teacher，并计算蒸馏损失
    if self.training:
        if self.eviction_strategy in ('repr_shift', 'repr_shift_spatial'):
            heuristic_importance = self.importance_scorer.compute(
                x_before_mlp=x_before_mlp,
                mlp_output=mlp_residual,
                patch_start_idx=self.patch_start_idx,
                grid_size=resolved_patch_grid_size,
            )
        else:
            heuristic_importance = self.importance_scorer.compute(k=k_current)

        # 归一化 teacher target 到 [0, 1]，与 scorer 的 Sigmoid 输出对齐
        h_min = heuristic_importance.min(dim=-1, keepdim=True)[0]
        h_max = heuristic_importance.max(dim=-1, keepdim=True)[0]
        heuristic_normalized = (heuristic_importance - h_min) / (h_max - h_min + 1e-8)

        distill_loss = F.mse_loss(learned_scores, heuristic_normalized.detach())

    new_importance = learned_scores
else:
    # 无 scorer，使用原始 heuristic importance
    if self.eviction_strategy in ('repr_shift', 'repr_shift_spatial'):
        new_importance = self.importance_scorer.compute(
            x_before_mlp=x_before_mlp,
            mlp_output=mlp_residual,
            patch_start_idx=self.patch_start_idx,
            grid_size=resolved_patch_grid_size,
        )
    else:
        new_importance = self.importance_scorer.compute(k=k_current)

return x_after_mlp, (k_current, v_current), new_importance, distill_loss
```

**关键原则**：`distill_loss` 通过函数返回值沿调用链传递，绝不使用 Python list 或 `self._distill_losses` 属性存储。
原因：(1) autograd 需要 loss tensor 保持在计算图中直到 backward；(2) DDP 多 GPU 训练时动态添加的属性不会被同步。

**推理路径**：推理时（`not training`），heuristic importance **不会**被计算——`self.training` 为 False 时，即使 `token_scorer` 存在也不会进入 heuristic 计算分支，直接用 scorer 输出。这避免了推理时多余的 `repr_shift` + 空间平滑计算（详见第 12 节 Review-5）。


### 5.3 修改 `src/ovggt/models/aggregator.py`

**`__init__` 新增**：
```python
self.token_scorers = None  # 由 init_token_scorers() 初始化
```

**新增方法**：
```python
def init_token_scorers(self, embed_dim: int, bottleneck_dim: int = None):
    from ovggt.layers.token_scorer import TokenScorer
    self.token_scorers = nn.ModuleList([
        TokenScorer(embed_dim=embed_dim, bottleneck_dim=bottleneck_dim)
        for _ in range(self.depth)
    ])
    for idx, block in enumerate(self.global_blocks):
        block.token_scorer = self.token_scorers[idx]
```

**`_process_global_attention` 修改**：当前 `frontend_cache_mode` 分支（第 473-487 行）调用 block 后接收 3 个返回值，需改为接收 4 个：

> 注意：需在 `for _ in range(self.aa_block_size):` 循环**之前**初始化 `distill_loss = None`（与 `new_importance = None`、`pending_update = None` 同级，见 `aggregator.py:459-461`）。这是因为 `distill_loss` 在 `frontend_cache_mode` 分支内赋值后需要传递到函数末尾的 `return` 语句。

```python
# 在 _process_global_attention 中，循环前新增:
distill_loss = None

# 循环内 frontend_cache_mode 分支:
# 当前:
tokens, pending_update, new_importance = self.global_blocks[global_idx](...)
# 修改为:
tokens, pending_update, new_importance, distill_loss = self.global_blocks[global_idx](...)
```

返回值也需相应增加 `distill_loss`：
```python
# 当前返回值:
return tokens, global_idx, intermediates, pending_update, new_importance
# 修改为:
return tokens, global_idx, intermediates, pending_update, new_importance, distill_loss
```

> **返回类型注解更新**：当前 `_process_global_attention` 的返回类型注解（`aggregator.py:439`）只覆盖了 2 种返回形状，已不完整。修改后 `frontend_cache_mode` 路径返回 6 个值，建议将注解更新为准确的 tuple 类型或使用 `@overload` 装饰器。

> **`aa_block_size > 1` 时的注意**：当前默认 `aa_block_size=1`（`aggregator.py:65`），循环只执行一次，`distill_loss` 取唯一值。如果后续 `aa_block_size > 1`，多次迭代的 `distill_loss` 会互相覆盖——只有最后一层的 `distill_loss` 被传递。当前不需要处理，但记录此限制。

**`forward` 修改**：在 `frontend_cache_mode` 分支中调用 `_process_global_attention` 处（第 307 行），接收新增的 `distill_loss`，在循环中累加：

> 变量初始化位置：在 `aggregator.py` 第 297 行 `prev_importance = None` 之后、第 298 行 `for _ in range(self.aa_block_num):` 之前新增 `total_distill_loss` 和 `distill_layer_count`。

```python
# 在 forward 的 frontend_cache_mode 分支中（第 297 行之后）:
total_distill_loss = None  # 初始化
distill_layer_count = 0    # 计数有效层数，用于后续取均值

# ... 在循环内 frontend_cache_mode 分支中（第 307 行）:
# 当前:
tokens, global_idx, global_intermediates, pending_update, new_importance = self._process_global_attention(...)
# 修改为:
tokens, global_idx, global_intermediates, pending_update, new_importance, layer_distill_loss = self._process_global_attention(...)

if layer_distill_loss is not None:
    total_distill_loss = layer_distill_loss if total_distill_loss is None else total_distill_loss + layer_distill_loss
    distill_layer_count += 1

# 主循环结束后（第 385 行 output_list.append 之后、第 386 行 if scores 之前）:
if total_distill_loss is not None:
    total_distill_loss = total_distill_loss / distill_layer_count
```

> **为什么取均值**：Sigmoid 输出 + 归一化 teacher 使单层 MSE ~0.05，24 层求和 ~1.2。但不是所有 24 层都用 scorer（取决于 `global_idx` 的迭代），所以用实际层数取均值更稳健。均值化的 distill_loss 量级 ~0.05，与 criterion_loss（10-100）通过 `distill_loss_weight` 调和。

**`forward` 返回值修改**：frontend_cache_mode 时返回值增加 `total_distill_loss`：
```python
if frontend_cache_mode:
    return output_list, self.patch_start_idx, cache_states, pending_updates, total_distill_loss
```

### 5.4 修改 `src/ovggt/models/ovggt.py`

**`OVGGTOutput` 新增字段**（`distill_loss` 必须通过 `OVGGTOutput` 传递，不能通过返回值 tuple）：

> **Review-1 修正**：`model(batch, ...)` 走的是 `OVGGT.forward()` → `forward_frontend_train()` → `_inference_frontend()`，最终返回**单个 `OVGGTOutput` 对象**，不是 tuple。因此 `student_outputs, total_distill_loss = model(batch, ...)` 这种写法会报错。`distill_loss` 必须作为 `OVGGTOutput` 的字段传递。

```python
@dataclass
class OVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[Any] = None
    keyframe_packets: Optional[List[KeyframePacket]] = None
    keyframe_schedule: Optional[List[Any]] = None
    distill_loss: Optional[torch.Tensor] = None  # 新增
```

**`__init__` 新增参数**：
```python
use_token_scorer: bool = False,
scorer_bottleneck_dim: Optional[int] = None,
```

在 aggregator 构造后：
```python
if use_token_scorer:
    self.aggregator.init_token_scorers(
        embed_dim=embed_dim,
        bottleneck_dim=scorer_bottleneck_dim or embed_dim // 4,
    )
```

**`_inference_frontend` 修改**：

> **Review-2 修正**：aggregator 在 `_inference_frontend` 中是**逐帧调用**的（在 `for i, frame in enumerate(frames)` 循环内）。方案需在**跨帧维度**也累加 `distill_loss`，而不能只依赖 aggregator 单次调用内的累加。

```python
# 在 _inference_frontend 中新增跨帧累加变量
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

    # 跨帧累加 distill_loss
    # 注意：此处是逐帧求和（未取均值），总 scale 取决于 num_views。
    # aggregator 内已取层间均值（~0.05/帧），24 帧求和后 ~1.2。
    # 若 num_views 变化，需同步调整 distill_loss_weight。
    if frame_distill_loss is not None:
        total_distill_loss = (
            frame_distill_loss
            if total_distill_loss is None
            else total_distill_loss + frame_distill_loss
        )

    # ... 后续 depth/point/camera 计算不变 ...

# 修改最终返回值，将 distill_loss 放入 OVGGTOutput
return OVGGTOutput(
    ress=all_ress if cache_results else None,
    views=processed_frames if (cache_results and return_views) else None,
    keyframe_packets=keyframe_packets if export_packets else None,
    keyframe_schedule=keyframe_schedule,
    distill_loss=total_distill_loss,  # 新增
)
```

**`forward_frontend_train` 无需额外修改**：`_inference_frontend` 已将 `total_distill_loss` 放入 `OVGGTOutput`，训练脚本从 `student_outputs.distill_loss` 获取。

**`load_state_dict` 修改**：

> **Review-15（严重）：`load_state_dict(strict=True)` 会在 scorer 启用时崩溃**
>
> 当前代码中有两处使用 `strict=True` 加载权重，启用 scorer 后新增的 `token_scorers` 参数不存在于旧 checkpoint 中，将直接报错 `RuntimeError: Missing key(s)`：
>
> | 调用点 | 文件:行号 | 问题 |
> |--------|-----------|------|
> | `load_student_pretrained_weights()` | `train_frontend.py:104` | 加载 pretrained backbone（无 scorer 参数）→ `strict=True` 报错 |
> | `misc.load_model()` (resume) | `croco/utils/misc.py:454` | 从 checkpoint 恢复训练（无 scorer 参数）→ `strict=True` 报错 |
>
> 此外，所有 eval/demo 脚本（`demo_gradio.py:37`、`eval_blendmvs_frontend.py:310`、`eval_ovggt_main.py:66` 等）加载模型时也使用 `strict=True`。
>
> **修复方案**：在 `OVGGT.load_state_dict()` 中（`ovggt.py:181-184`）增加 scorer 参数兼容处理：

```python
def load_state_dict(self, state_dict, strict: bool = True):
    upgraded_state_dict = dict(state_dict)
    self._upgrade_camera_head_state_dict(upgraded_state_dict)
    # Review-15: scorer 参数兼容
    # 如果模型启用了 scorer 但 checkpoint 不含 scorer 权重，
    # 从 state_dict 中移除模型中不存在的 key，改用 strict=False 容忍
    scorer_keys = [k for k in upgraded_state_dict if 'token_scorers' in k]
    model_has_scorer = hasattr(self.aggregator, 'token_scorers') and self.aggregator.token_scorers is not None
    if scorer_keys and not model_has_scorer:
        # checkpoint 含 scorer 权重但模型未启用，跳过这些 key
        for k in scorer_keys:
            del upgraded_state_dict[k]
    if model_has_scorer and not scorer_keys:
        # 模型启用 scorer 但 checkpoint 不含 scorer 权重，用 strict=False 容忍
        return super().load_state_dict(upgraded_state_dict, strict=False)
    return super().load_state_dict(upgraded_state_dict, strict=strict)
```

> 对于 `load_student_pretrained_weights`（`train_frontend.py:104`），建议改为：
> ```python
> printer.info(model.load_state_dict(pretrained_state, strict=False))
> ```
> 因为 Stage A 训练时模型包含 scorer 参数但 pretrained 不含，`strict=False` 是必须的。

> 对于 resume（`misc.py:454`），如果 resume 的 checkpoint 是 Stage A 训练中间产物（含 scorer 权重），则 `strict=True` 可以工作。但如果 resume from pretrained（不含 scorer 权重），则需要在调用前过滤。建议在 `train()` 函数中 resume 加载时统一使用 `strict=False`，或在 `OVGGT.load_state_dict` 中如上处理。

### 5.5 修改 `src/train_frontend.py`

**新增冻结函数**：
```python
def freeze_stage_a_scorer_only(model: OVGGT) -> None:
    """Stage A 蒸馏：冻结所有参数，仅放开 TokenScorer"""
    for _, param in model.named_parameters():
        param.requires_grad = False
    # 放开 scorer
    if model.aggregator.token_scorers is not None:
        for param in model.aggregator.token_scorers.parameters():
            param.requires_grad = True
```

> **Review-8 注意**：此函数必须在 `load_student_pretrained_weights` 之后、**`optimizer = torch.optim.AdamW(param_groups, ...)` 之前** 调用。时间线如下（参见 `train_frontend.py`）：
> 1. `load_student_pretrained_weights(model, ...)` — 第 774 行 → 覆盖 scorer 权重
> 2. `freeze_stage_a_scorer_only(model)` — **在此插入** → 冻结 backbone，放开 scorer
> 3. `param_groups = misc.get_parameter_groups(model, ...)` — 第 797 行 → 只收集 `requires_grad=True` 的参数
> 4. `optimizer = torch.optim.AdamW(param_groups, ...)` — 第 798 行 → optimizer 仅含 scorer 参数
> 5. `accelerator.prepare(model, optimizer, ...)` — 第 831-837 行
>
> 如果在 `get_parameter_groups` **之前**未冻结 backbone，optimizer 会包含所有参数（~1B），Stage A 训练将更新整个模型而非仅 scorer。

**`frontend_loss_of_one_batch` 修改**：

> **Review-1 修正**：`model(batch, ...)` 返回**单个 `OVGGTOutput`**，不是 tuple。通过 `student_outputs.distill_loss` 获取 distill_loss。

> **Review-4 修正**：`FrontendDistillLoss.finalize_from_stream` 返回 `(loss, loss_details)` tuple（当前代码 `loss, loss_details = frontend_loss_of_one_batch(...)` 的解包方式证实了这一点）。distill_loss 应在 `finalize_from_stream` 返回的 loss 上叠加，而非单独返回。

```python
# 当前代码（FrontendDistillLoss 分支）:
student_outputs = model(
    batch,
    query_points=query_points,
    frame_processor=accumulate_student_frame,
    cache_results=False,
    return_views=False,
)

# ... criterion.finalize_from_stream(...) 已经计算了 (loss, loss_details) ...

# 修改为: 在 finalize_from_stream 返回后叠加 distill_loss
with torch.amp.autocast(device_type=autocast_device_type, enabled=False):
    if isinstance(criterion, FrontendDistillLoss):
        loss, loss_details = criterion.finalize_from_stream(
            raw_batch_gt=batch,
            teacher_outputs=teacher_outputs,
            ...
        )
        # 叠加 distill_loss
        total_distill_loss = student_outputs.distill_loss
        if total_distill_loss is not None:
            loss = loss + distill_loss_weight * total_distill_loss
    else:
        loss, loss_details = criterion(batch, teacher_outputs, student_outputs, ...)
        total_distill_loss = student_outputs.distill_loss
        if total_distill_loss is not None:
            loss = loss + distill_loss_weight * total_distill_loss

# 注意: student_outputs 在叠加 distill_loss 之后才能 del
del student_outputs
del teacher_outputs
del query_points
return loss, loss_details
```

其中 `distill_loss_weight` 为新增超参数（建议默认值 0.1，配置文件中可调）。

> **注意：`loss_details["total"]` 需同步更新**。`finalize_from_stream` 返回的 `loss_details` 中 `"total"` 是 criterion 自己的总 loss（不含 distill）。叠加 distill_loss 后，`loss_details["total"]` 与实际用于 backward 的 `loss` 不一致。建议在叠加后更新：
> ```python
> if total_distill_loss is not None:
>     loss = loss + distill_loss_weight * total_distill_loss
>     # 同步更新 loss_details 以便日志/监控准确
>     loss_details["distill_loss"] = float(distill_loss_weight * total_distill_loss)
>     loss_details["total"] = float(loss)
> ```
> 否则训练日志中显示的 total loss 会比实际 backward 的 loss 小，NaN 检测时打印的 `loss_details` 也会误导调试。

### 5.6 不需要修改：`src/ovggt/utils/frontend_cache.py`

**不需要修改**。原设计计划在 `FrontendCacheConfig` 中添加 `use_learned_scorer` 字段，但该字段实际无需使用：
- 判断 scorer 是否启用：`model.aggregator.token_scorers is not None`
- `commit_pending_update_` 等方法无需感知 scorer 存在——importance 分数在 block 层已被替换为 learned scores，通过 `PendingLayerUpdate.importance_current` 自动流入缓存系统

---

## 6. 训练流程

### 6.1 Stage A：蒸馏训练

**目标**：让 TokenScorer 学会模仿 heuristic importance（behavior cloning）

```
Teacher (heuristic): repr_shift_spatial(x_before_mlp, mlp_residual) → score_teacher [B, N]
Student (learned):   TokenScorer(x_after_mlp)                       → score_student [B, N]
Distill Loss = MSE(score_student, score_teacher.detach())
Total Loss   = FrontendDistillLoss + distill_loss_weight × Distill Loss
```

**流程**：
1. 加载 pretrained OVGGT（noIntra+fifo80 基线权重）
2. `use_token_scorer=True` 创建模型，scorer 随机初始化
3. **冻结 backbone**：使用 `freeze_stage_a_scorer_only()`（见 5.5 节），仅放开 `token_scorers` 参数
4. 逐 batch forward，`distill_loss` 沿 Block→Aggregator→OVGGT→训练脚本的返回值链传递
5. 在训练脚本中：`total_loss = criterion_loss + distill_loss_weight × distill_loss`
6. 保存 `token_scorers_final.pth`

**超参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| lr | 1e-4 | 比 backbone 高 10x（仅 scorer 参数参与更新） |
| epochs | 10 | 轻量模块快速收敛 |
| dataset | Co3D / BlendedMVS | 任意多视角数据集 |
| total_budget | 10410 | 逼迫 eviction 发生 |
| camera_budget | 128 | 逼迫 eviction 发生 |
| accum_iter | 4 | 梯度累积 |
| distill_loss_weight | 1.0 | 蒸馏 loss 权重。scorer 输出 Sigmoid→(0,1)，teacher 归一化到 [0,1]，层间取均值后单帧 distill_loss ~0.05。`_inference_frontend` 跨帧求和，num_views=24 时总 distill_loss ~1.2。权重 1.0 使 distill 贡献约等于 criterion（10-100）的 1-10%。**注意：若 num_views 改变，需同步调整此权重**。建议首轮实验尝试 0.1/1.0/10.0 |

> **Review-9 注意**：Stage A 的 lr=1e-4 是初始值，实际训练时仍受 `misc.adjust_learning_rate` 的 warmup + cosine schedule 控制（`train_frontend.py:551`）。由于 optimizer 仅包含 scorer 参数，schedule 会正常应用，但最终 lr 会衰减到 `min_lr`（默认 1e-7）。建议在配置中将 `min_lr` 设为 1e-6，避免训练末期 lr 过低导致收敛停滞。

> 注意：原设计中的 temperature annealing 已删除。TokenScorer 输出经 Sigmoid 约束到 (0, 1)，不需要 temperature 参数。

### 6.2 Stage B：端到端微调

> **Review-3 重要修正：Stage B 梯度链在 eviction 处断裂**
>
> 原方案声称 scorer 可以通过重建 loss 端到端优化，但实际 eviction 使用 `torch.topk` + `torch.gather` 做**硬选择**（`attention.py:256`，`frontend_cache.py:636`），这是**离散不可微**操作。梯度路径分析：
>
> ```
> scorer → learned_score → PendingLayerUpdate.importance_current
>     → metadata.importance → topk (离散, 梯度断裂!) → 选中 token 的 K/V
>     → 后续帧 attention → depth/point/pose → loss
> ```
>
> 重建 loss 的梯度**无法通过 topk 选择步骤反传到 scorer 参数**。因此 Stage B 的原始设计（直接端到端优化）不可行。

**目标**：让 TokenScorer 尽可能优化重建精度，超越 teacher 的上限

**可选策略**：

| 策略 | 方法 | 可行性 | 复杂度 |
|------|------|--------|--------|
| **B-1: STE 直通估计** | 前向用 hard topk，反向用恒等函数传递梯度 | 中等（STE 近似有偏，收敛不稳定） | 低 |
| **B-2: Soft eviction 训练** | 训练时用 softmax 加权替代 topk，推理时用 hard topk | 高（梯度可传递，训练推理不一致需验证） | 中 |
| **B-3: 策略梯度 (REINFORCE)** | 将 scorer 输出视为 policy，用 variance-reduced REINFORCE 优化 | 高（理论正确） | 高（需要 baseline 网络、reward shaping） |
| **B-4: 更好的 teacher 信号** | 不做端到端，改用 GT 深度误差作为蒸馏 target | 高（无梯度链问题） | 低 |
| **B-5: 放弃 Stage B** | 仅依赖 Stage A 蒸馏，投入精力改进 heuristic 本身 | 高 | 最低 |

**推荐路线**：**先实施 B-4**，即 Stage B 保持 Stage A 的蒸馏框架，但将 teacher target 从 heuristic importance 替换为基于 GT 深度误差的重要性指标（如：token 的深度预测误差越大 → importance 越高）。如果效果仍不满意，再考虑 B-2（soft eviction）。

**B-4 流程**：
1. 加载 pretrained backbone + Stage A scorer 权重
2. `use_token_scorer=True`，仍然冻结 backbone（仅放开 scorer）
3. 对每个 token，计算其深度预测与 GT 深度的误差作为 importance target
4. 用 MSE(scorer_output, gt_importance_target) 替代原来的 heuristic 蒸馏 target
5. scorer 学习到"哪些 token 的深度质量对重建真正重要"
6. 保存微调后的 scorer 权重

> **B-4 循环依赖注意**：B-4 需要 scorer 先选择 token 才能得到 depth prediction，但 GT teacher 的目标正是训练 scorer。实际实现可采用**在线 bootstrapping** 方式：
> - 当前 scorer 选择 token → model forward → 得到 depth prediction → 计算 per-token GT depth error → 作为**下一 step** 的 teacher target
> - 训练初期 scorer 评分接近随机（从 Stage A 蒸馏初始化），随着训练进行逐步改善
> - 或者：用**当前帧 without scorer** 跑一次额外 forward 获取"无 eviction"的 depth 作为 oracle quality reference（仅用于计算 teacher target，不参与 backward），避免 scorer ↔ token selection 的循环依赖。**注意：此方法需临时禁用 scorer 或维护第二个不含 scorer 的模型实例，实现复杂度较高，优先使用 bootstrapping 方案**

**B-2 流程（备选）**：
1. 训练时，在 `commit_pending_update_` 的 eviction 步骤中，用 softmax 加权求和代替 topk：
   ```python
   # 替代 topk 的 soft selection:
   weights = F.softmax(combined_scores / temperature, dim=-1)
   soft_k = torch.einsum('bhsd,bs->bhd', candidate_k, weights)
   ```
2. 推理时恢复 hard topk（与 Stage A 一致）
3. temperature 从高到低退火，逐渐逼近 hard selection
4. 需要修改 `frontend_cache.py` 的 `commit_pending_update_` 和 `attention.py` 的 `eviction`

**注意**：当前 `train_frontend.py` 默认使用 `FrontendDistillLoss`。如要使用 `FrontendSupervisedLoss`（GT 深度+点云监督），需确认 `src/ovggt/losses/frontend_supervised.py` 已实现且与当前数据 pipeline 兼容。

---

## 7. 推理集成

```python
# 创建模型
model = OVGGT(
    mode='frontend_eval',
    use_token_scorer=True,
    total_budget=200000,
    frontend_cache_config=FrontendCacheConfig(
        intra_frame_dedup_enabled=False,
        fifo_keep_topk=80,
    ),
)

# 加载 backbone 权重（scorer 权重不存在时 strict=False 容忍）
state_dict = torch.load('checkpoints.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)

# 加载 scorer 权重
scorer_sd = torch.load('token_scorers_final.pth', map_location='cpu')
model.aggregator.token_scorers.load_state_dict(scorer_sd)

# 正常推理（与 noIntra+fifo80 完全相同的调用方式）
output = model.inference(frames, history_anchor_strategy='fixed_interval',
                         anchor_interval=8, max_anchors=3)
```

**推理模式说明**：
- 推理时 `self.training=False`，`distill_loss` 不会被计算（第 5.2 节的 `if self.training` 条件保护）
- 推理时 heuristic importance **不会被计算**——scorer 存在时直接使用 scorer 输出，跳过 heuristic 分支（详见 5.2 节代码逻辑）
- Scorer forward 的输出直接作为 `new_importance` 使用，推理时应在 `torch.no_grad()` 上下文中运行
- 整个模型的调用方式与 noIntra+fifo80 基线完全一致

---

## 8. 解决的 6 个问题对照

| 问题 | TokenScorer 如何解决 |
|------|---------------------|
| 代理指标不匹配 | scorer 端到端训练（Stage B），直接用重建 loss 反传优化（注意：Stage B 需使用 STE/soft eviction 等策略绕过 topk 不可微问题，详见 6.2 节） |
| 新旧标准不统一 | 所有 token 统一由 scorer 打分（commit 时 importance_current 已是 learned）。注：budget eviction 中旧 token 仍用 cosine diversity，但新旧统一归一化后竞争 |
| 层间独立 | 每层独立 scorer，但共享训练信号（同一个 loss 反传） |
| 固定空间假设 | 无硬编码假设，网络自己学 spatial pattern |
| depth_conf 未校准 | Stage A 蒸馏后 scorer 替代了 importance 部分，depth_conf 仍参与 voxel dedup 复合评分；Stage B 可尝试降低 `depth_conf_weight`，让 scorer 独立承担职责 |
| 与下游任务脱节 | Stage B 可通过 STE/soft eviction/B-4（GT teacher）等方式间接连接重建 loss，但直接端到端梯度链在 eviction 处断裂（详见 6.2 节 Review-3） |

---

## 9. 验证方案

### 9.1 单元测试

- `tests/test_token_scorer.py`：验证 TokenScorer forward 输出 shape [B, N]
- 验证 scorer 参数梯度正常（backward 后 grad 非 None 且非零）
- 验证 B>1 时输出 shape 正确

### 9.2 集成测试

- `src/test_accel_scorer.py`：验证 accelerate 分布式下 scorer 梯度流通
- 从 BlendedMVS 加载一个 batch，forward + backward，检查 scorer 参数梯度非零
- 验证 `distill_loss` 沿 Block→Aggregator→OVGGT 返回值链正确传递

### 9.3 精度基准

在 7Scenes 数据集上，chess/seq-03 场景 200 帧对比：

| 配置 | 50帧 ATE | 200帧 ATE | 目标 |
|------|---------|----------|------|
| noIntra+fifo80 (baseline) | 0.0203 | 0.0258 | — |
| TokenScorer 蒸馏后 | ≤0.0203 | ≤0.0258 | 模仿 teacher |
| TokenScorer GT 微调后 | <0.0203 | <0.0258 | 超越 teacher |

### 9.4 多场景验证

| 场景 | noIntra+fifo80 | TokenScorer 目标 |
|------|---------------|-----------------|
| chess/seq-03 | 0.0258 | < 0.0258 |
| fire/seq-03 | 0.0281 | < 0.0281 |
| office/seq-03 | 0.0500 | < 0.0500 |
| redkitchen/seq-03 | 0.0307 | < 0.0307 |

---

## 10. 风险与缓解

| # | 风险 | 缓解措施 |
|---|------|---------|
| 1 | scorer 蒸馏后无法超越 heuristic | Stage B 用更好的 teacher 信号（GT 深度误差）或 soft eviction 优化重建精度；如果仍不行，说明当前 cache 管理机制是瓶颈而非 scoring 质量 |
| 2 | 不同场景 scorer 泛化差 | 在多个场景混合训练；或按场景微调 top-K |
| 3 | scorer 训练不稳定（梯度消失/爆炸） | LayerNorm 前置 + Sigmoid 输出约束提供数值稳定；teacher target 归一化到 [0, 1]；`distill_loss_weight` 可调（建议 0.1/1.0/10.0 网格搜索）；仅 scorer 参数更新，backbone 冻结 |
| 4 | budget eviction 中新旧 token 仍不统一 | 旧 token 仍用 cosine diversity，新 token 用 learned score，两者归一化后竞争。后续可考虑用 scorer 为旧 token 也重新打分（周期性 refresh） |
| 5 | 蒸馏 loss 与主 loss 尺度不匹配 | `distill_loss_weight` 可调；首轮实验用 0.1/1.0/10.0 三个值快速搜索 |
| 6 | DDP 多 GPU 训练下返回值链断裂 | `distill_loss` 通过 `OVGGTOutput.distill_loss` 字段传递（tensor 在 dataclass 中），与 DDP 兼容 |
| 7 | **Stage B 梯度链在 eviction 处断裂**（Review-3） | topk/gather 是离散不可微操作，无法直接端到端优化。推荐使用 B-4（GT teacher 信号）或 B-2（soft eviction），详见 6.2 节 |
| 8 | **scorer 输出与 teacher 尺度不匹配**（Review-6） | Sigmoid 约束输出到 (0, 1)，teacher target 归一化到 [0, 1]，两者同分布。详见 5.1 节和 5.2 节 |
| 9 | **`OVGGTOutput` 返回值类型不匹配**（Review-1） | `model()` 返回单个 `OVGGTOutput`，不是 tuple。distill_loss 必须作为 dataclass 字段传递。详见 5.4 节 |
| 10 | **跨帧 distill_loss 累加遗漏**（Review-2） | aggregator 逐帧调用，distill_loss 需在 `_inference_frontend` 中跨帧累加。详见 5.4 节 |
| 11 | **`load_state_dict(strict=True)` 在 scorer 启用时崩溃**（Review-15） | `load_student_pretrained_weights` 和 `misc.load_model` 使用 `strict=True`，旧 checkpoint 不含 scorer 参数。需在 `OVGGT.load_state_dict` 中增加兼容处理，详见 5.4 节 |
| 12 | **Sigmoid 饱和导致训练初期 importance 信号失效**（Review-16） | 随机初始化的 scorer 输出集中在 ~0.5，归一化后所有 token importance 相同（0.5），intra-frame pruning 和 voxel dedup 退化为随机选择。预期随着训练推进（10-100 step）动态范围恢复，但初期 eviction 质量下降。缓解：可考虑用 `nn.init.xavier_uniform_` 初始化最后一层 Linear 使 Sigmoid 输出更分散 |
| 13 | **混合精度 (bf16) 对 scorer 的影响**（Review-17） | scorer 运行在 autocast(bf16) 上下文中，参数 fp32 但输入 bf16，输出 bf16。MSE loss 在 bf16 下计算精度略低于 fp32。对 topk/argmax 排序无影响（bf16 精度足够区分排序）。若需要更高精度，可在 distill_loss 计算前 `.float()` 转换，但通常不必要 |

---

## 11. 实现注意事项

### 11.1 batch_size 限制

当前 `train_frontend_blendedmvs.yaml` 中 `batch_size=1`。本设计中 Block 的 `forward` 使用 `[B, N, C]` 形状，接口上支持 B>1，但在多 batch 下以下组件需要验证：
- `PendingLayerUpdate.frame_id` 在 B>1 时语义（当前为单个 int）
- `cache_states` 的初始化与 batch 大小关系
- `FrontendKeyframeManager` 的 batch 行为

**建议**：Stage A/B 都在 `batch_size=1` + `accum_iter` 下运行，与现有训练流程一致。B>1 支持留作后续优化。

### 11.2 `eviction_strategy` 参数兼容

当启用 TokenScorer 时，Block 的 `eviction_strategy` 参数仍会传递给 `TokenImportanceScorer`（用于计算 heuristic teacher）。TokenScorer 的 `new_importance` 输出是回归值，其语义与 `repr_shift` 的输出一致（越大越重要）。

**建议**：启用力设定 `eviction_strategy='repr_shift_spatial'`（与 noIntra+fifo80 基线一致），确保 teacher 信号和 scorer 输出的 score 同分布。

### 11.3 Tensor 返回值与 JIT/torch.compile 兼容性

`distill_loss` 新增返回值改变了 Block → Aggregator → OVGGT 的函数签名。如果后续使用 `torch.compile`，需要在编译前验证返回值签名兼容。当前训练未使用 `torch.compile`，不影响。

### 11.4 使用 `torch.no_grad()` 包装推理时的 scorer

推理时 scorer 不需要梯度，应在模型级别的 `inference()` 方法中确认已有 `torch.no_grad()` 上下文（当前 `_inference_frontend` 在 `evaluation` 时不调用 `model.eval()` 也会被 `@torch.no_grad()` 包裹）。但为了安全，scorer forward 耗时极小，即使漏了 no_grad 也不影响正确性，只浪费显存。

---

## 12. 代码审查修正记录（Review Log）

> 以下为对照实际代码库审查后发现的问题及修正方案。已在文档各节中标记 `Review-N` 引用。

### Review-1（严重）：`distill_loss` 返回值链与 `OVGGTOutput` 不兼容

**问题**：原方案 5.5 节写 `student_outputs, total_distill_loss = model(batch, ...)`，假设 `model()` 返回 tuple。但实际 `OVGGT.forward()` 在 `frontend_train` 模式下调用 `_inference_frontend()`，返回的是**单个 `OVGGTOutput` dataclass**（`ovggt.py:585-590`），不是 tuple。

**修正**：将 `distill_loss` 作为 `OVGGTOutput` 的字段传递：
- 5.4 节：`OVGGTOutput` 新增 `distill_loss: Optional[torch.Tensor] = None`
- 5.5 节：训练脚本通过 `student_outputs.distill_loss` 获取

**影响文件**：`ovggt.py`（`OVGGTOutput` 定义）、`train_frontend.py`（`frontend_loss_of_one_batch`）

### Review-2（严重）：跨帧 `distill_loss` 累加缺失

**问题**：方案中 aggregator 的 `forward()` 在**单次调用**（处理一帧的所有层）中累加 `total_distill_loss`。但 `_inference_frontend` 中 aggregator 是**逐帧调用**的（`ovggt.py:397` 在 `for i, frame in enumerate(frames)` 循环内）。原方案只描述了单次 aggregator 调用内的层间累加，**没有提到跨帧累加**，导致多帧训练时只有最后一帧的 distill_loss 被返回。

**修正**：在 `_inference_frontend` 中增加跨帧累加逻辑，详见 5.4 节代码。

### Review-3（重要）：Stage B 端到端梯度链在 eviction 处断裂

**问题**：eviction 使用 `torch.topk` + `torch.gather` 做**硬选择**（`attention.py:256`，`frontend_cache.py:636`），这是离散不可微操作。重建 loss 的梯度无法通过 topk 反传到 scorer 参数。Stage B 原始设计（直接端到端优化）不可行。

**修正**：重写 6.2 节，提供 5 种可选策略（B-1 到 B-5），推荐先实施 B-4（GT teacher 信号），备选 B-2（soft eviction）。同时在第 8 节和第 10 节中更新了相关描述。

### Review-4（中等）：`loss_details` 返回值处理遗漏

**问题**：`FrontendDistillLoss.finalize_from_stream` 返回 `(loss, loss_details)` tuple（从 `train_frontend.py:564` 的 `loss, loss_details = frontend_loss_of_one_batch(...)` 解包方式可确认）。原方案的 `total_loss = criterion_loss + ...` 没有考虑 `loss_details` 的存在。

**修正**：在 5.5 节中明确 `loss, loss_details = criterion.finalize_from_stream(...)` 后再叠加 distill_loss，返回值保持 `(loss, loss_details)` 不变。

### Review-5（中等）：推理时多余的 heuristic 计算

**问题**：原方案说"推理时 heuristic 也会计算，计算量极小可忽略"。但 `repr_shift_spatial` 包含 `mlp_output.norm(dim=-1)` + 3×3 conv2d + grid reshape，对 24 层累积并非零开销。

**修正**：修改 5.2 节代码逻辑，scorer 存在且非训练时跳过 heuristic 计算：
```python
if self.token_scorer is not None:
    learned_scores = self.token_scorer(x_after_mlp)
    if self.training:
        heuristic_importance = ...  # 只在训练时计算
        distill_loss = F.mse_loss(...)
    new_importance = learned_scores
else:
    new_importance = heuristic  # 无 scorer 时才用 heuristic
```

### Review-6（中等）：scorer 输出尺度未约束

**问题**：原方案 TokenScorer 最后一层 `Linear(D, 1)` 无激活函数，输出无界。随机初始化的输出与 teacher（`mlp_output.norm`）可能差几个数量级，导致 MSE 爆炸。

**修正**：在 3.1 节和 5.1 节中增加 `nn.Sigmoid()` 末尾激活，将输出约束到 (0, 1)。同时在 5.2 节中增加 teacher target 的 min-max 归一化：
```python
h_min = heuristic_importance.min(dim=-1, keepdim=True)[0]
h_max = heuristic_importance.max(dim=-1, keepdim=True)[0]
heuristic_normalized = (heuristic_importance - h_min) / (h_max - h_min + 1e-8)
distill_loss = F.mse_loss(learned_scores, heuristic_normalized.detach())
```

### Review-7（小）：`aggregator.forward()` 返回值 breaking change

**问题**：`aggregator.forward()` 在 `frontend_cache_mode` 时返回 4 个值（`aggregator.py:393`），改为 5 个值后需确认所有调用点同步更新。

**修正**：已确认 `_inference_frontend` 是唯一调用点（`ovggt.py:397`），在 5.4 节中已更新解包。`_inference_legacy` 走的是 `use_cache=True` 但非 `frontend_cache_mode` 的路径，不受影响。

### Review-8（小）：`freeze_stage_a_scorer_only` 调用时机

**问题**：冻结函数必须在 `load_student_pretrained_weights` 之后、`accelerator.prepare()` 之前调用。

**修正**：在 5.5 节中添加调用时机说明。

### Review-9（小）：lr schedule 对纯 scorer optimizer 的行为

**问题**：Stage A 的 lr=1e-4 受 `misc.adjust_learning_rate` 的 cosine schedule 控制，最终衰减到 `min_lr`（默认 1e-7），可能过低。

**修正**：在 6.1 节中添加注意事项，建议 `min_lr` 设为 1e-6。

### Review-10（小）：`distill_loss` 层间累加应取均值而非求和

**问题**：原方案 aggregator 中 24 层 `distill_loss` 直接求和，loss 幅值随层数线性放大。加上 Sigmoid + teacher 归一化后单层 MSE ~0.05-0.10，求和 ~1.2-2.4，但不同模型深度下 scale 不一致，且与 criterion_loss 调和时 weight 需要针对不同 depth 调参。

**修正**：在 5.3 节中改为按实际层数取均值，使 distill_loss 幅值 ~0.05-0.10，与模型深度无关。训练脚本和 distill_loss_weight 不受影响。

### Review-11（小）：5.1/3.1 节交叉引用写错章节号

**问题**：Sigmoid 输出约束的说明引用了 "第 13 节 Review-6"，但 Review Log 位于第 12 节。

**修正**：将所有 "第 13 节" 改为 "第 12 节"。

### Review-12（小）：B-4（GT teacher）的循环依赖未说明

**问题**：B-4 需要 scorer 选择 token 后才能得到 depth prediction，但 GT teacher 的目标正是训练 scorer。存在 scorer ↔ token selection ↔ depth prediction 的循环依赖。

**修正**：在 6.2 节 B-4 流程后补充两种解决方案：
- 在线 bootstrapping（当前 scorer 选 token，计算出的 GT error 作为下一 step 的 target）
- Oracle forward（额外一次无 scorer eviction 的 forward 获取 reference depth quality）

### Review-13（中等）：`freeze_stage_a_scorer_only` 必须在 `get_parameter_groups` 之前调用

**问题**：Review-8 仅强调该函数需在 `load_student_pretrained_weights` 之后、`accelerator.prepare()` 之前调用。但实际代码中 `get_parameter_groups`（第 797 行）和 `optimizer = AdamW(...)`（第 798 行）在 `accelerator.prepare()` 之前执行。如果 `freeze_stage_a_scorer_only` 在 optimizer 创建之后调用，optimizer 已持有 backbone 参数的引用，冻结无效。

**确认的实际代码顺序**（`train_frontend.py`）：
1. 第 774 行: `load_student_pretrained_weights(model, args.pretrained)`
2. 第 797 行: `param_groups = misc.get_parameter_groups(model, args.weight_decay)`
3. 第 798 行: `optimizer = torch.optim.AdamW(param_groups, ...)`
4. 第 831-837 行: `accelerator.prepare(model, optimizer, ...)`

**修正**：在 5.5 节中明确要求 `freeze_stage_a_scorer_only` 在第 1 步和第 2 步之间调用。

### Review-14（小）：B-4 oracle forward 实现复杂度

**问题**：B-4 的 oracle forward 方案需要"临时禁用 scorer 或维护第二个不含 scorer 的模型实例"，实现复杂度较高但未在文档中标注。

**修正**：在 6.2 节 B-4 描述中增加"优先使用 bootstrapping 方案"的建议。

### Review-15（严重）：`load_state_dict(strict=True)` 在 scorer 启用时崩溃

**问题**：当前代码中有多处使用 `strict=True` 加载权重。当 `use_token_scorer=True` 创建模型后，模型包含 `aggregator.token_scorers` 的参数，但 pretrained checkpoint 不含这些参数，导致 `RuntimeError: Missing key(s) in state_dict`。

受影响的调用点：

| 调用点 | 文件:行号 | 场景 |
|--------|-----------|------|
| `load_student_pretrained_weights()` | `train_frontend.py:104` | Stage A 加载 pretrained backbone → scorer 参数缺失 → **必崩** |
| `misc.load_model()` | `croco/utils/misc.py:454` | 从 checkpoint 恢复训练，若 checkpoint 不含 scorer → **必崩** |
| `teacher.load_state_dict()` | `train_frontend.py:787` | teacher 不启用 scorer → 无影响 |
| 所有 eval/demo 脚本 | `demo_gradio.py:37` 等 | 若用 `use_token_scorer=True` 创建模型 → **必崩** |

**修正**：详见 5.4 节 `load_state_dict` 修改方案，在 `OVGGT.load_state_dict()` 中增加 scorer 参数兼容处理。同时 `load_student_pretrained_weights` 需改为 `strict=False`。

### Review-16（中等）：Sigmoid 饱和导致训练初期 importance 信号失效

**问题**：随机初始化的 scorer 最后一层 `Linear(256, 1)` 权重较小，经 Sigmoid 后输出集中在 ~0.5 附近（如 [0.48, 0.52]）。这会导致以下下游问题：

1. **Intra-frame pruning**（`frontend_cache.py:636`）：`torch.topk(importance, k=keep_count)` 中，所有 token 的 importance 差异仅 ~0.04，排序由浮点噪声决定 → 退化为随机选择
2. **Voxel dedup 复合评分**（`frontend_cache.py:948-970`）：`_normalize_with_mask_batch` 检测到 `range ≤ 1e-8` 时赋值 0.5，importance 分量对所有 token 相同 → 区分完全依赖 `depth_conf`
3. **Eviction hybrid scoring**（`attention.py:244-245`）：`_normalize_scores` 同样会检测到低动态范围

**影响评估**：这是训练初期的预期行为。随着 scorer 学习到有意义的 importance 模式（预计 10-100 step），动态范围会自然扩大。但初期 eviction 质量可能低于 heuristic baseline。

**可选缓解措施**（非必须）：
- 对 scorer 最后一层 Linear 使用 `nn.init.xavier_uniform_` 初始化，使 Sigmoid 输出更分散（初始化后 range 可能扩大到 [0.3, 0.7]）
- 在 `TokenScorer.__init__` 中增加：
  ```python
  nn.init.xavier_uniform_(self.scorer[3].weight)  # Linear(256, 1)
  nn.init.zeros_(self.scorer[3].bias)
  ```

### Review-17（小）：混合精度 (bf16) 对 scorer 的影响

**问题**：当前训练使用 `accelerator(mixed_precision="bf16")`（`train_frontend.py:646`）。scorer 运行在 autocast 上下文中：

- scorer 参数：fp32（autocast 不转换参数）
- scorer 输入 `x_after_mlp`：bf16（来自 block forward）
- scorer 输出 `learned_scores`：bf16（autocast 自动转换中间计算）
- heuristic importance：bf16（来自 `mlp_output.norm(dim=-1)`）
- `distill_loss = F.mse_loss(learned_scores, heuristic_normalized)`：bf16 下计算

**影响评估**：
- topk/argmax 排序：bf16 精度足够，不影响排序结果
- MSE loss 精度：bf16 的 mantissa 有 7 位有效数字，对 0-1 范围的 MSE (~0.05) 精度足够
- scorer 梯度：autocast 会自动处理梯度精度转换

**结论**：无需额外处理。若后续发现 scorer 训练不稳定，可在 distill_loss 计算前对 learned_scores 和 heuristic_normalized 调用 `.float()`，但这会增加显存开销。

### Review-18（信息）：`OVGGTOutput` 新增字段的向后兼容性

**验证结果**：`OVGGTOutput` 新增 `distill_loss: Optional[torch.Tensor] = None` 字段后，以下所有消费点均安全（dataclass 默认值为 None，不影响现有属性访问）：

- `demo_gradio.py:103` — `output.ress`
- `demo_viser.py:96` — `output.ress`
- `tools/eval_*.py` — `output.ress`, `output.keyframe_schedule`
- `scripts/compare_frontend_legacy.py:108` — `output["ress"]`（dict-style access，`ModelOutput` 支持）
- `tests/test_frontend_inference_smoke.py:67-89` — `output.ress`, `output.keyframe_packets`

此外，`_inference_legacy`（`ovggt.py:751-754`）和 `forward()` legacy 路径（`ovggt.py:276`）创建的 `OVGGTOutput` 不含 `distill_loss`，默认为 `None`，不影响任何消费点。

**结论**：无需额外修改。

### Review-19（信息）：`tools/test_phase2_smoke.py` 已存在 scorer 测试

**发现**：`tools/test_phase2_smoke.py` 已使用 `use_token_scorer=True` 创建模型并运行推理。注意该文件使用了 `FrontendCacheConfig(use_learned_scorer=True)` 参数（第 22 行），但当前 `FrontendCacheConfig` 定义中没有 `use_learned_scorer` 字段。

**建议**：该文件可能是前期原型代码。正式实现时应同步更新此测试脚本，移除 `use_learned_scorer=True` 参数（5.6 节已说明不需要此字段）。

### Review-20（信息）：`block.token_scorer` 的 `model.eval()` 传播机制

**验证**：`block.token_scorer` 通过 `init_token_scorers` 设置为 `self.token_scorers[idx]` 的引用。虽然不是通过 `nn.Module.__setattr__` 注册到 Block 的子模块，但因为 scorer 已注册在 `aggregator.token_scorers`（`nn.ModuleList`）中，`model.eval()` 会沿 `OVGGT → aggregator → token_scorers → scorer[idx]` 链正确传播 `training` 标志。

`block.token_scorer.training` 和 `aggregator.token_scorers[idx].training` 是同一个 Python 对象的同一属性，因此 `model.eval()` 后 `block.forward()` 中的 `self.training` 检查（第 198 行）能正确感知 eval 模式。

**结论**：`model.eval()` / `model.train()` 传播正确，无需额外处理。

### Review-21（小）：`loss_details["total"]` 在叠加 distill_loss 后未更新

**问题**：`criterion.finalize_from_stream()` 在内部设置 `loss_details["total"] = float(total)`（`frontend_distill.py:297`）。叠加 `distill_loss` 后，实际用于 backward 的 `loss` 改变了，但 `loss_details["total"]` 仍是叠加前的值。这会导致：

1. 训练日志中的 `loss_details["total"]` 与实际 `metric_logger.update(loss=loss_value)` 的 loss 不一致
2. NaN 检测时 `printer.error("Loss is %s, stopping training. Details: %s", loss_value, loss_details)` 打印的 total 会误导——`loss_value` 可能是 NaN 但 `loss_details["total"]` 显示正常值

**修正**：在 5.5 节中增加 `loss_details["distill_loss"]` 和 `loss_details["total"]` 的更新代码。

**验证**：所有消费 `loss_details` 的代码路径（`metric_logger.update(**loss_details)` at line 596, `log_writer.add_scalar(...)` at line 603-609）通过 `**loss_details` 解包，新增 key 会自动记录到 TensorBoard，属于正向改进。

### Review-22（严重）：`block.py` 缺少 `import torch.nn.functional as F`

**问题**：设计文档 5.2 节的修改代码使用 `F.mse_loss(learned_scores, heuristic_normalized.detach())`，但 `block.py:1-15` 的 import 列表中不含 `torch.nn.functional`。Python 会在运行时抛出 `AttributeError: module 'torch.nn' has no attribute 'functional'`（或 `NameError: name 'F' is not defined`）。

**修正**：在 5.2 节开头添加 import 注意事项，实现时需在 `block.py` 头部加入：
```python
import torch.nn.functional as F
```
或直接在代码中写 `torch.nn.functional.mse_loss(...)`。

### Review-23（建议）：`_process_global_attention` 返回类型注解需更新

**问题**：`aggregator.py:439` 的返回类型注解 `-> Union[Tuple[..., int, List[Tensor]], Tuple[..., int, List[Tensor], List]]` 只覆盖了 2 种返回形状，已过时（实际有 3 种）。添加 `distill_loss` 后进一步偏离。

**修正**：在 5.3 节中添加注解更新建议。非阻塞。
