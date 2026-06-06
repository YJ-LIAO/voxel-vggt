# Voxel-VGGT 修改记录

> 基于 OVGGT 项目的修改对比，由 voxel-vggt 项目维护
> 生成日期: 2026-05-20

---

## 1. 总览

Voxel-VGGT 在 OVGGT 基础上新增了一套完整的 **Frontend 推理与训练管线**，核心目标是将 OVGGT 从「离线批量推理」改造为「在线增量推理」，支持流式逐帧输入、KV Cache 管理、关键帧调度和前端训练。

主要修改方向：

| 方向 | 简述 |
|------|------|
| Frontend 推理模式 | 新增 `_inference_frontend` 路径，支持逐帧增量推理 |
| Frontend Cache 系统 | 全新 KV Cache 管理（`frontend_cache.py`，~970 行） |
| 关键帧调度 | 新增 `FrontendKeyframeManager`（`frontend_keyframe.py`，~290 行） |
| 相对位姿编码 | 扩展 `pose_enc.py`，支持绝对/相对位姿互转与组合 |
| 相机头双分支 | `camera_head.py` 新增 `rel_pose_branch`，同时输出绝对与相对位姿 |
| 训练/蒸馏损失 | 新增 `FrontendDistillLoss` 和 `FrontendSupervisedLoss` |
| 训练脚本 | 新增 `train_frontend.py` 和 `finetune_frontend.py` |
| 梯度检查点 | Block 层支持可选的 gradient checkpointing |
| 数据集 | 新增 `BlendedMVSContinuous_Multi` 连续帧数据集 |
| 鲁棒性 | 数据加载支持坏样本自动重试 |
| 测试 | 新增 6 个测试文件 |

---

## 2. 新增文件（OVGGT 中不存在）

### 2.1 核心新增

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/ovggt/utils/frontend_cache.py` | ~972 | Frontend KV Cache 状态管理：`LayerCacheState`、`PendingLayerUpdate`、`TokenMetadata`、`FrontendCacheConfig` 等。负责逐帧 token 元数据追踪、voxel dedup、importance 评分与 budget 驱动的 token 淘汰。 |
| `src/ovggt/utils/frontend_keyframe.py` | ~286 | 关键帧调度管理器：`FrontendKeyframeManager`、`KeyframeSwitchConfig`、`KeyframeEvent`、`KeyframePacket`。支持 fixed_interval 和 coverage 两种关键帧策略，FIFO 淘汰历史锚点。 |
| `src/ovggt/losses/frontend_distill.py` | ~180 | Teacher-Student 蒸馏损失：使用 Teacher（完整 VGGT）输出监督 Student（Frontend 模式），包含绝对/相对位姿损失、位姿一致性损失。 |
| `src/ovggt/losses/frontend_supervised.py` | ~100 | 监督训练损失（继承自 `FrontendDistillLoss`）：使用 GT 位姿/深度监督，包含 depth loss、pointmap loss、track loss。 |
| `src/train_frontend.py` | ~951 | Frontend 蒸馏训练入口（Stage A）：Teacher-Student 蒸馏训练管线，支持 gradient checkpointing、teacher 权重 offload 到 CPU、混合精度训练。 |
| `src/finetune_frontend.py` | ~520 | Frontend 监督微调入口（Stage B）：在 Stage A 基础上使用 GT 监督微调，支持 activation offload 到 CPU。 |
| `src/model_hub_compat.py` | 12 | HuggingFace Hub 兼容层，优雅降级 `PyTorchModelHubMixin` 和 `ModelOutput` 的 import。 |

### 2.2 配置文件

| 文件 | 说明 |
|------|------|
| `config/train_frontend_blendedmvs.yaml` | Stage A 蒸馏训练配置：teacher 模型路径、BlendedMVS 连续帧数据集、`FrontendDistillLoss`、`relT_quaR_FoV` 编码、budget=200000、4GPU DDP。 |
| `config/finetune_frontend_blendedmvs.yaml` | Stage B 监督微调配置：`FrontendSupervisedLoss`、gradient checkpointing、accum_iter=4、budget=10410。 |
| `config/train_frontend_blendedmvs_custom.yaml` | 自定义训练配置（参数变体）。 |

### 2.3 工具与脚本

| 文件 | 行数 | 说明 |
|------|------|------|
| `scripts/compare_frontend_legacy.py` | ~302 | 对比 Frontend 与 Legacy 模式的推理效果（精度、速度）。 |
| `tools/eval_blendmvs_frontend.py` | ~508 | BlendedMVS 数据集上的 Frontend 模式评估。 |
| `tools/eval_blendmvs_frontend_long.py` | ~267 | 长序列场景的 Frontend 评估。 |
| `tools/profile_frontend_hotspots.py` | ~225 | Frontend 推理性能热点分析。 |

### 2.4 测试文件

| 文件 | 说明 |
|------|------|
| `tests/test_frontend_cache.py` | Frontend Cache 状态管理单元测试 |
| `tests/test_frontend_inference_smoke.py` | Frontend 推理冒烟测试 |
| `tests/test_frontend_supervised_loss.py` | 监督损失单元测试 |
| `tests/test_frontend_training_smoke.py` | 训练冒烟测试 |
| `tests/test_keyframe_manager.py` | 关键帧调度器测试 |
| `tests/test_pose_enc_frontend.py` | 位姿编码前端模式测试 |

---

## 3. 修改的文件

### 3.1 `src/ovggt/models/ovggt.py`（365 行 → 872 行，+507 行）

**核心改动：新增 Frontend 推理模式**

- **三模式架构**：引入 `mode` 参数（`legacy` / `frontend_train` / `frontend_eval`），`forward()` 根据 mode 分发到不同推理路径
- **`forward_frontend_train()`**：训练模式入口，固定使用 `fixed_interval` 策略，anchor_interval=8，max_anchors=3
- **`_inference_frontend()`**：Frontend 增量推理核心实现
  - 使用 `FrontendKeyframeManager` 管理关键帧调度
  - 使用 `LayerCacheState` 管理 aggregator 层级 KV Cache
  - 支持绝对/相对位姿编码切换
  - 支持 gradient checkpointing for depth/point head
  - 支持导出 keyframe packets 和 schedule
- **`_inference_legacy()`**：保留原始 OVGGT 推理路径，新增 `move_to_cpu` / `return_views` 参数
- **`inference()`**：统一入口，根据 mode 自动分发到 frontend 或 legacy 路径
- **参数化构造**：`OVGGT.__init__` 接受 `mode`、`frontend_pose_encoding_type`、`frontend_cache_config`、`keyframe_switch_config`、各 head 的 kwargs 等
- **`gradient_checkpointing_enable()`**：控制 patch_embed、frame_blocks、global_blocks 的梯度检查点
- **`_upgrade_camera_head_state_dict()`**：自动从旧 checkpoint 中复制 `pose_branch` 权重到新增的 `rel_pose_branch`
- **`OVGGTOutput` 扩展**：新增 `keyframe_packets` 和 `keyframe_schedule` 字段
- **`_validate_frontend_batch_size()`**：确保 frontend 模式下 batch_size=1
- **`_frame_image_to_sequence()`**：将单帧图像转为 `[B, S=1, C, H, W]` 格式
- **anchor_overflow_policy**：支持 `recent` 和 `global_plus_recent` 两种 anchor 溢出策略

### 3.2 `src/ovggt/models/aggregator.py`

**核心改动：集成 Frontend Cache 机制**

- **`__init__`**：`patch_start_idx` 默认值从 5 改为 0，`patch_grid_size` 改为动态计算
- **`forward()`**：
  - 新增 `cache_states` 参数替代 `past_key_values`，用于 frontend cache 模式
  - 新增 `frontend_cache_config` 参数
  - 在 frontend cache 模式下使用 `PendingLayerUpdate` 追踪每层 KV 更新
  - 相机/register token 选择逻辑改为 `select_cached_special_token()`
- **`_process_global_attention()`**：
  - 新增 `frontend_cache_mode` 和 `patch_grid_size` 参数
  - frontend cache 模式下先做 attention + MLP，再计算 importance，然后 defer eviction 返回 pending update
  - `patch_grid_size` 参数传递到子模块而非使用 `self.patch_grid_size`
- **非缓存路径**：保持原始逻辑不变

### 3.3 `src/ovggt/layers/attention.py`

**核心改动：KV Cache 淘汰策略优化**

- **`_normalize_scores()`**（新增）：归一化 importance 分数到 [0,1]，对 equal-score 组保持中性值而非强制归零，避免偏差
- **`anchor_overflow_policy`**（新增）：当 cache budget 不足以容纳所有 anchor token 时的溢出策略
  - `recent`：保留最近的 anchor
  - `global_plus_recent`：始终保留全局 anchor (index=0) + 最近的 anchor
- **`_select_anchor_indices_on_overflow()`**（新增）：根据溢出策略选择要保留的 anchor 索引
- **`evict_kv_cache()`**：改用 `_normalize_scores()` 替代简单的 min-max 归一化；支持 `defer_eviction`（延迟淘汰模式用于 frontend cache）
- **Token 淘汰选择**：从 "保留最低相似度 token" 改为 "保留最高 diversity token"，使用多头平均统一选择

### 3.4 `src/ovggt/layers/block.py`

**核心改动：Frontend Cache 模式 + Gradient Checkpointing**

- **`__init__`**：`patch_start_idx` 默认值从 5 改为 0
- **`forward()`**：
  - 新增 `frontend_cache_mode` 和 `patch_grid_size` 参数
  - frontend cache 模式：先计算 attention（defer eviction），然后 MLP，然后计算 importance，返回 `(x_after_mlp, (k_current, v_current), new_importance)`
  - 新增 `current_frame_keys()` 方法：在 eviction 前重新计算当前帧的 keys，避免从已淘汰的 cache 中推断新 token
  - 新增 `ffn_residual_maybe_checkpoint()`：支持可选的 gradient checkpointing for MLP
  - 非缓存路径也支持 gradient checkpointing

### 3.5 `src/ovggt/layers/vision_transformer.py`

- 新增 `self.use_reentrant = False` 配置，避免 gradient checkpointing 的 reentrant 模式警告

### 3.6 `src/ovggt/layers/rope.py`

**核心改动：RoPE 实现优化**

- `get_2d_position_embeddings()`：移除 `.clone()`，避免不必要的内存拷贝
- `_apply_1d_rope()`：新增 `angles = torch.cat((angles, angles), dim=-1)` 拼接
- 新增 `_rotate_features()` 静态方法：封装特征旋转操作
- `_apply_1d_rope()` 改用 `tokens * cos + _rotate_features(tokens) * sin` 统一计算，替代分开的 x1/x2 乘法
- `_apply_2d_rope()`：改用 `torch.chunk()` 分割 + `torch.cat()` 重组，替代硬编码的 feature_dim 索引

### 3.7 `src/ovggt/layers/importance_scorer.py`

- `patch_start_idx` 默认值从 5 改为 0（与 aggregator/block 一致）

### 3.8 `src/ovggt/heads/camera_head.py`

**核心改动：双分支位姿预测**

- 新增 `rel_pose_branch`（MLP）：与 `pose_branch` 结构相同，独立权重，专门预测相对位姿编码
  - 初始化时从 `pose_branch` 复制权重（`load_state_dict`）
- `forward()` 新增参数：
  - `pose_encoding_type`：支持 `absT_quaR_FoV` 和 `relT_quaR_FoV`
  - `return_pose_predictions`：返回完整位姿预测字典
  - `return_last_pose_only`：只返回最后一轮迭代的预测
- `trunk_fn()`：同时计算绝对位姿增量和相对位姿增量，累积更新
- 移除对 `ABS_POSE_ENCODING` / `REL_POSE_ENCODING` 常量的依赖
- 新增 `apply_keyframe_event()`：将关键帧事件应用到 camera KV cache

### 3.9 `src/ovggt/utils/pose_enc.py`

**核心改动：绝对/相对位姿编码互转**

- 新增常量定义：`ABS_POSE_ENCODING = "absT_quaR_FoV"`，`REL_POSE_ENCODING = "relT_quaR_FoV"`
- 新增 `_validate_pose_encoding_type()`：位姿编码类型校验
- 新增 `_inverse_se3()`：SE(3) 矩阵求逆
- 新增 `pose_encoding_to_world_to_camera()`：位姿编码 → world-to-camera 矩阵
- 新增 `pose_encoding_to_camera_to_world()`：位姿编码 → camera-to-world 矩阵
- 新增 `world_to_camera_to_pose_encoding()`：world-to-camera 矩阵 → 位姿编码
- 新增 `compose_absolute_from_relative()`：从锚点绝对位姿 + 相对位姿组合出当前绝对位姿
- 新增 `relative_from_absolute_pose_encoding()`：从两个绝对位姿计算相对位姿

### 3.10 `src/ovggt/utils/history_anchor.py`

- 新增 `min_anchor_interval` 配置字段：两个历史锚点之间的最小帧间隔
- 新增 `last_anchor_frame` 追踪：记录最后一次注册锚点的帧索引
- `should_become_anchor_coverage()`：增加最小间隔检查，避免过于频繁的锚点注册

### 3.11 `src/vggt/models/vggt.py`

- 将 HuggingFace Hub 的 `ModelOutput` / `PyTorchModelHubMixin` 导入替换为 `model_hub_compat.py` 的兼容版本

### 3.12 `src/dust3r/datasets/blendedmvs.py`

**新增 `BlendedMVSContinuous_Multi` 类**（~70 行）

- 继承自 `BlendedMVS_Multi`
- 生成连续帧窗口序列（`contiguous_step` 步长），用于 Frontend 训练
- 支持任意起点 + 等间距采样，保证训练数据是时序连续的

### 3.13 `src/dust3r/datasets/__init__.py`

- 新增 `BlendedMVSContinuous_Multi` 的导出

### 3.14 `src/dust3r/datasets/base/base_multiview_dataset.py`

**核心改动：数据加载鲁棒性增强**

- 新增 `max_bad_view_retries` 参数（默认 16）
- `__getitem__()`：增加 retry 循环，遇到坏样本（AssertionError）自动重试
- 新增 `_is_retryable_bad_view_error()`：判断错误是否可重试（坏主点、NaN pose、NaN depth）
- 新增 `_sample_retry_index()`：随机选择重试索引
- 新增 `_finalize_views()`：提取 view finalization 逻辑，复用于 retry 路径

### 3.15 `src/croco/models/curope/kernels.cu`

- `.type()` → `.scalar_type()`（适配新版 PyTorch ATen API）

### 3.16 `src/croco/models/pos_embed.py`

- import 路径从 `models.curope` 改为 `croco.models.curope`

### 3.17 `src/croco/utils/misc.py`

- `SmoothedValue` 的统计方法增加空 deque 保护，避免除零错误
- 参数组摘要打印优化：展示每个参数组的 `weight_decay`、`lr_scale`、`num_tensors`

---

## 4. 架构对比

### 4.1 OVGGT 原始推理流程（Legacy 模式）

```
多帧图像 [S帧] → Patch Embed → Frame Blocks → 全局 KV Cache + eviction
  → Camera Head → Depth Head → Point Head → Track Head
```

所有帧一次性输入，全局 attention 后再逐 head 预测。

### 4.2 Voxel-VGGT Frontend 推理流程

```
逐帧输入 → Patch Embed → Frame Blocks
  → Aggregator (with LayerCacheState + PendingLayerUpdate)
  → FrontendKeyframeManager.update() → KeyframeEvent
  → Camera Head (abs + rel 双分支, anchor sync)
  → Depth Head (可选 gradient checkpoint)
  → Point Head (可选 gradient checkpoint)
  → Track Head
  → 可能触发 anchor promotion / FIFO swap
```

### 4.3 训练管线

```
Stage A (蒸馏训练):
  Teacher (完整 VGGT, 权重 offload 到 CPU)
    ↓ 提供 GT 伪标签
  Student (Frontend 模式, relT_quaR_FoV)
    → FrontendDistillLoss (abs/rel 位姿损失 + 一致性损失)

Stage B (监督微调):
  Student (Frontend 模式)
    → FrontendSupervisedLoss (cam + depth + pmap + track loss)
    → gradient checkpointing + activation offload
```

---

## 5. 关键参数对比

| 参数 | OVGGT (Legacy) | Voxel-VGGT (Frontend) |
|------|----------------|----------------------|
| 推理模式 | 批量 | 逐帧增量 |
| KV Cache | past_key_values (tuple) | LayerCacheState + PendingLayerUpdate |
| 关键帧调度 | 无 | FrontendKeyframeManager (fixed_interval / coverage) |
| 位姿编码 | absT_quaR_FoV | absT_quaR_FoV + relT_quaR_FoV |
| Camera Head | 单分支 | 双分支 (abs + rel) |
| patch_start_idx | 5 | 0 |
| Anchor 溢出策略 | 无 | recent / global_plus_recent |
| 梯度检查点 | 无 | 可选 (patch_embed, blocks, heads) |
| 数据集 | BlendedMVS_Multi | + BlendedMVSContinuous_Multi |
| 坏样本处理 | 直接报错 | 自动重试 (max 16 次) |
