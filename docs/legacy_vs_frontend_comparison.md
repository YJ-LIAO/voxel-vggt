# Legacy vs Frontend 分布对比（7-Scenes）

> 日期: 2026-06-19
> 数据: 7-Scenes chess/fire/office/redkitchen 各 seq-03
> ckpt: `/path/to/mount/lyj/voxel-vggt/ckpt/checkpoints.pth`
> 工具: `tools/run_legacy_vs_frontend.py`，结果 `tools/legacy_vs_frontend_lyj/`
> 目的: 对比当前版本（frontend ring0.2）与 legacy baseline 在不同帧数的分布表现

## 设置

- **baseline (legacy)**: `OVGGT(mode='legacy').inference(inputs, history_anchor_strategy='coverage', anchor_interval=250)` —— 逐帧 + coverage history-anchor 流式
- **current (frontend)**: `mode='frontend_eval'` + ring0.2 + per_layer_budget=8334 + intra_frame_dedup=True + `history_anchor_strategy='fixed_interval', anchor_interval=8, max_anchors=3`
- **公平性**: 同 ckpt、同模型，唯一变量是推理模式（coverage-anchor vs bounded-ring）
- **确定性**: 实测 seed-std ≈ 0（eval 模式无随机性），故 seed 0 即完整对比；多 seed 仅作冗余确认

## 结果：ATE RMSE (m)，seed 0

### 逐场景

| scene | mode | 200f | 500f | 1000f |
|-------|------|------|------|-------|
| chess | legacy | 0.0257 | 0.0512 | 0.0977 |
| chess | frontend | 0.0263 | 0.0528 | 0.0580 |
| fire | legacy | 0.0307 | 0.0486 | 0.0448 |
| fire | frontend | 0.0458 | 0.0558 | 0.0477 |
| office | legacy | 0.0298 | 0.1681 | 0.2256 |
| office | frontend | 0.0377 | 0.1038 | 0.1723 |
| redkitchen | legacy | 0.0176 | 0.0274 | 0.2338 |
| redkitchen | frontend | 0.0138 | 0.1034 | 0.1664 |

### 4 场景 mean ± std

| 帧数 | legacy | frontend | 谁优 |
|------|--------|----------|------|
| 200f | 0.0260 ± 0.0052 | 0.0309 ± 0.0121 | legacy 略优（+0.0049）|
| 500f | 0.0738 ± 0.055 | 0.0789 ± **0.025** | 持平，frontend **std 小一半** |
| 1000f | 0.1505 ± 0.081 | **0.1111 ± 0.058** | **frontend −26%，更稳** |

### Paired diff（frontend − legacy，负 = frontend 更好）

| scene | 200f | 500f | 1000f |
|-------|------|------|-------|
| chess | +0.0005 | +0.0016 | **−0.0397** |
| fire | +0.0151 | +0.0072 | +0.0028 |
| office | +0.0079 | **−0.0643** | **−0.0533** |
| redkitchen | −0.0038 | +0.0759 | **−0.0674** |
| frontend 胜场 | 1/4 | 1/4 | **3/4** |

## 解读

1. **短序列（200f）**：legacy 略优（全帧 O(N²) attention 信息优势），差距小（+0.0049）。
2. **中序列（500f）**：持平，但分布最分散——**office legacy 崩**（0.168）、**redkitchen frontend 崩**（0.103）。frontend 的跨场景 std（0.025）比 legacy（0.055）小一半，整体更一致。
3. **长序列（1000f）**：**frontend 全面更优**——mean **−26%**（0.111 vs 0.150），3/4 场景胜，std 更小（0.058 vs 0.081）。**这是 frontend streaming 的核心价值兑现**：legacy 的 coverage anchor 在长序列漂移/累积误差，bounded ring 更稳。
4. **legacy 最致命失败**：office/redkitchen 长序列严重退化（0.168/0.226/0.234）——coverage anchor 在大场景长序列失效；frontend ring0.2 同样退化但轻得多。

## 结论

| 维度 | 结论 |
|------|------|
| 短序列 | legacy 略优（全帧信息）|
| 长序列 | **frontend 明显更优（−26%）且更稳定** ← streaming 价值兑现 |
| 稳定性 | frontend 在 500f/1000f 的跨场景 std 都更小 |
| 公平性 | 同 ckpt 同模型，仅推理模式不同 |

## 限定

- 单序列（每场景 seq-03）、seed 0。确定性已验证（seed-std≈0）。
- 完整分布可扩展到每场景多序列（chess 6 seq, office 10 等），但跨场景 4 点已清晰展示短/中/长序列模式。
