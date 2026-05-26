# DDP Deadlock & NaN Loss Fix Design

> **Goal:** 修复 4 卡 DDP 训练在 Step 20-30 死锁的问题，根因是 NaN Loss 导致的跨 rank 梯度不同步。

## Root Cause Analysis

### 死锁链路

```
异常 depth 数据 → teacher 产生 inf/nan 预测 → DepthOrPmapLoss 产生 NaN
→ train_frontend.py 执行 loss * 0 + 0（IEEE 754: NaN * 0 = NaN，修复无效）
→ NaN 进入 backward()，产生 NaN 梯度
→ DDP all-reduce 时各 rank 梯度状态不一致（部分 rank 正常，部分 rank 梯度为 NaN）
→ NCCL 通信卡死，GPU 0-2 100% 占用，GPU 3 0%
```

### TokenScorer 验证

经代码确认，TokenScorer **始终在计算图中**（[block.py:192](src/ovggt/layers/block.py#L192) 的 `self.token_scorer is not None` 训练时恒为 True，`self.training` 恒为 True，每帧每层都会计算 `distill_loss`）。非 scorer 条件性参与导致死锁。

### 核心 Bug

[train_frontend.py:599](src/train_frontend.py#L597-L599) 的 `loss * 0 + 0` 在 IEEE 754 下无法将 NaN 替换为 0：`NaN × 0 = NaN`（任意精度均成立，包括 fp32/bf16/fp16）。

## Architecture: Three-Layer Defense

```
数据层 (teacher output sanitize)
    ↓
Loss 层 (数值稳定性保护)
    ↓
DDP 安全网 (torch.nan_to_num 兜底)
```

## Changes

### Layer 1: Teacher Output Sanitize

**File:** `src/train_frontend.py`
**Location:** [lines 365-370](src/train_frontend.py#L365-L370), after `teacher.inference()`

新增 `_sanitize_teacher_outputs()` 函数，在 teacher 输出进入 loss 计算前检测并 clamp inf/nan：

```python
def _sanitize_teacher_outputs(teacher_outputs):
    for i, pred in enumerate(teacher_outputs.ress):
        for key in ("depth", "pts3d_in_other_view"):
            if key in pred:
                t = pred[key]
                if not torch.isfinite(t).all():
                    logger.warning("Teacher %s has inf/nan at frame %d, clamping.", key, i)
                    pred[key] = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
                    pred[key] = pred[key].clamp(-1e4, 1e4)
```

### Layer 2: Loss Numerical Stability

**File:** `src/dust3r/losses.py`

#### 2a. `DepthOrPmapLoss.forward` (lines 1265-1291)

- 入口处 `torch.nan_to_num` + clamp pred/gt 到 `[-1e4, 1e4]`
- 出口处检测 loss 是否为 NaN/inf，如果是则替换为 0 tensor（保留计算图）

#### 2b. `closed_form_scale_and_shift` (lines 1078-1117)

- scale/shift 计算后增加 fallback：`torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))`
- shift 同理：`torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))`

### Layer 3: DDP Safety Net

**File:** `src/train_frontend.py`
**Location:** [lines 597-599](src/train_frontend.py#L597-L599)

```python
# Before (broken):
loss = loss * 0 + 0

# After (correct):
loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
```

增加 rank 和 step 信息到 warning 日志，方便追溯。

## Not Changed

- **DDP 配置** (`ddp_static_graph`, `ddp_find_unused_parameters`) — 经验证 scorer 始终在计算图中，无需调整
- **`finetune_frontend.py`** — 使用 `FloatingPointError` 直接终止，行为明确，可作为后续独立优化
- **Dataset 层** — depth 数据在 dataset 阶段尚未加载（on-demand），且 NaN 来源是 teacher 推理而非 raw 数据
- **`src/train.py`** (legacy) — 旧训练脚本不涉及 TokenScorer

## Risk Assessment

| 风险 | 缓解 |
|------|------|
| clamp 阈值 (-1e4) 可能截断合法的大深度值 | 1e4 远大于任何合理 depth 范围（室内 <100m，室外 <1000m），仅截断传感器异常值 |
| NaN→0 可能掩盖模型发散 | 保留 warning 日志，且 3 层都有日志，连续 NaN 会体现为 loss 曲线异常 |
| `torch.nan_to_num` 在 bf16 下行为 | `nan_to_num` 是 PyTorch op，正确性不依赖精度 |
