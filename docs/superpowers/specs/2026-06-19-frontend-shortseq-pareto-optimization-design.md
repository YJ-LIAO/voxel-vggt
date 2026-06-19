# Frontend 短序列 Pareto 优化设计（mean 超越 + 多数场景超越 legacy）

> 日期: 2026-06-19
> 状态: 设计（brainstorming 产出，待 spec review）
> 关联: `docs/legacy_vs_frontend_comparison.md`（legacy vs frontend 分布对比）、`docs/p1_ablation_results.md`、RCA workflow（13-agent 短序列 deficit 根因分析）
> 目标: 让 frontend 在 **200/1000f 的 4 场景 mean ATE < legacy，500f mean 在 redkitchen 修复后 < legacy**（即 "mean 超越 + 多数场景超越"），且不回退 chess/redkitchen、不损害 1000f 的 −26% 优势。注：500f mean 目标**条件依赖于 Gate 3（redkitchen 崩溃修复）**，若根因不可修则诚实保留。

---

## 0. 背景与动机

legacy vs frontend 对比（7-Scenes seq-03, seed 0, 同 ckpt, ATE RMSE m）：

| 帧数 | legacy | frontend | 差距 |
|------|--------|----------|------|
| 200f | 0.0260 | 0.0309 | frontend **+0.0049 输** |
| 500f | 0.0738 | 0.0789 | frontend **+0.0051 输**（但 std 更小）|
| 1000f | 0.1505 | 0.1111 | frontend **−0.0394 赢** |

frontend 在长序列（1000f）碾压 legacy（−26%），但短/中序列（200/500f）略输。目标：关闭短/中序列差距。

### RCA 已确认的根因（13-agent workflow + 决定性实测）

200f 的 deficit **场景相关、混合原因**：
- **fire（+0.0151）= intra-frame voxel dedup**：实测 intra-OFF 让 fire 0.0458→0.0279（恢复并超 legacy）。intra 的硬丢弃把 fire 多视角有用信息误当冗余丢弃。
- **office（+0.0079）≠ intra**：intra-OFF 反而更差（0.0377→0.0450）。次要嫌疑 = anchor 策略差异（frontend `fixed_interval/8/max3` vs legacy `coverage/250`）。
- **chess（+0.0006）≈ 持平**；**redkitchen（−0.0038）frontend 反超**。

### RCA 排除的机制（代码+数据双重验证）
- eviction budget 8334（两 mode 共享同一 cap）
- hybrid eviction 评分（两 mode 共享同一 `eviction()` 代码）
- ring0.2 保护池（200f 4 场景 bit-equivalent，影响 <0.001m）
- 跨帧 dedup（无 scorer 生产路径下 `protected_scores=0`，`discard_current_mask` 恒空，不触发）
- pose 编码（frontend/legacy 走同一 camera-head，ABS_POSE）

### 关键约束
intra dedup 对 fire 有害、对 office/chess 有益 —— **单一全局开关无法两全**。因此方案形式 = **改代码 + 单一固定生产配置**（无运行时场景判断），追求 Pareto 改善。

---

## 1. 设计

### 1.1 Component 1: Soft-merge intra dedup（核心，修复 fire 的确认根因）

**现状**：`_dedup_single_batch`（`frontend_cache.py:1113-1135`）对当前帧内同一 0.1m voxel 的多个 patch token，按 importance 评分保留最高分的一个，**硬丢弃**其余（`keep_mask[duplicate_indices]=False`）。

**改动**：每个 voxel group（>1 个当前帧 token）合并成 **1 个 token = 组内 importance 加权平均**：
- `merged_K[b,h,:] = Σ_i w_i · K[b,h,i]`，`merged_V` 同理（per head），权重 `w_i = softmax(importance_i)` over the group
- `merged score_state = Σ_i w_i · score_state_i`（**仅当 scorer 启用时**；生产推理 `score_state=None` → 跳过）
- `merged importance = Σ_i w_i · importance_i`；`merged depth_conf = Σ_i w_i · depth_conf_i`
- `merged slot_local_xyz = 组内平均`（同 voxel，近似一致）
- `frame_id / anchor_slot / keyframe_id / slot_id / token_kind` = 保留代表 token（最高权重者）的值
- **结果 token 数 = unique voxel 数**（与 drop 模式完全相同 → eviction/budget 行为不变），仅 token **值**不同（merged vs best-only）

**实现要点（关键：merge 发生在 `apply_voxel_dedup_`，不是 `_dedup_single_batch`）**：
- `_dedup_single_batch`（`frontend_cache.py:1034-1137`）**只读 `self.metadata` + 返回 keep_mask 索引，无法访问 K/V**。因此它需要额外返回一个 **merge plan**：每个多 token voxel group 的（代表 survivor 索引 + 成员索引 + softmax 权重）。
- `apply_voxel_dedup_`（`frontend_cache.py:712+`，B==1 在 ~856 调 `_gather_single_batch_`、B>1 在 ~875 调 `gather_per_batch_`）在 **gather 之前** 应用 merge plan：对每个 group，用 `scatter_reduce`/`index_add` 把成员的 K/V/score_state 加权平均写入代表 token 的槽位（`self.k[b,:,rep,:]` 等），**然后** gather（丢弃非代表 token）。
- B==1 与 B>1 两条 gather 路径都要接入 merge plan（B==1 走 `_gather_single_batch_`，B>1 走 `gather_per_batch_`）。
- merge 仅作用于当前帧 token（`current_patch_mask`，`frame_id==current`，`anchor_slot<0`，见 `frontend_cache.py:773-790`），所以 metadata 的 `frame_id/anchor_slot/keyframe_id` 都相同，保留代表值即可（无跨帧冲突）。
- **不改变 token 数**（= unique voxel 数，与 drop 模式一致），所以下游 reorder/P6 不变量都不受影响；但 **eviction 排序可能因 importance 被平均（动态范围压缩）而轻微偏移** —— 由 Gate 1 实测裁决。

**Config**：新增 `intra_dedup_mode: Literal["drop","merge"] = "drop"`（默认 drop = 向后兼容；生产设 "merge"）。`__post_init__` 加优先级规则（沿用 `frontend_cache.py:85-104` 的 ring/max_protected 互斥校验模式）：`intra_frame_dedup_enabled=False` 优先（整体关闭帧内 dedup，mode 被忽略）；mode 仅在 `intra_frame_dedup_enabled=True` 时生效。

**为何 Pareto**：fire 的"重复" token 携带多视角信息 → 加权平均**保留**信息（而非丢弃）；office/chess 的真冗余 → 平均 ≈ 任一单 token → 中性。token 数不变 → 不影响 budget/eviction。

**风险**：平均不同视角 token 的 K/V 可能模糊；需实测验证。历史 ToMe（bipartite soft merge）单独 200f=0.0297（可用），但那是 ring/budget 修复前 + 不同合并方式。

### 1.2 Component 2: Coverage anchoring（修复 office 的非 intra deficit）

**现状**：frontend 用 `fixed_interval/8/max3`（每 8 帧固定锚点 + FIFO）；legacy 用 `coverage/250`（按覆盖率自适应布锚，跨轨迹分布）。

**关键障碍（spec-review 发现）**：frontend 的 coverage 路径**当前是死代码**。`_build_frontend_keyframe_config`（`ovggt.py:1196-1223`）对所有非 train 模式**硬编码 `coverage_monitor_only=True`**（`ovggt.py:1222`），所以即使调用方传 `history_anchor_strategy='coverage'`，`frontend_keyframe.py:248` 的 `elif strategy=="coverage" and not coverage_monitor_only:` 分支**永不触发**。只改调用点的 strategy 字符串无效。

**改动（两部分）**：
1. **打通 coverage**：修改 `_build_frontend_keyframe_config`（`ovggt.py:1196-1223`）接受 `coverage_monitor_only` 参数（默认 True 保现状；strategy='coverage' 时允许传 False 启用分支），并接受 `coverage_threshold`。**注意阈值默认不一致**：frontend `KeyframeSwitchConfig.coverage_threshold=0.2`，legacy `HistoryAnchorManager` 有效阈值 ~0.4（`history_anchor.py:36`）—— 实验需 sweep 或对齐 legacy。
2. **生产调用点**：`tools/run_legacy_vs_frontend.py`、`tools/test_multi_scene.py`、`eval_*` 在打通后设 `history_anchor_strategy='coverage'` + `coverage_monitor_only=False`。

**为何帮 office**：office 的 200f deficit 不是 intra → 嫌疑是锚点调度。coverage 锚点跨轨迹分布，提供更好的长程几何参考；fixed_interval/8 锚点聚集在局部。

**风险（较高）**：
- coverage 改变 keyframe promotion 逻辑，与 ring/FIFO_SWAP 动力学交互 —— **Gate 2 必须先验证 coverage 分支确实触发**（log keyframe registration），否则会静默重测 fixed_interval。
- office 的非 intra 根因只是"嫌疑"anchor，未确认；若实为别因，coverage 无效。
- 场景无关 → 只是小均匀增益，不修 fire。

### 1.3 Component 3: 500f redkitchen 崩溃诊断（待查，非预定改动）

500f redkitchen：frontend 0.1034 vs legacy 0.0274（frontend 崩，+0.076）。这是与 200f 不同的不稳定性，**尚未 RCA**。诊断方向：是 intra？anchor？某层 overflow？budget-poor 层？诊断后再定修复，不预设方案。

---

## 2. 验证关卡（empirical gates，单场景快测在前，全量在后）

每个关卡都是"通过才进入下一步"的硬门槛，避免回退 chess/redkitchen 或 1000f 优势。

- **Gate 1（soft-merge @200f 验证）**：fire/office/chess/redkitchen @200f，soft-merge vs drop。
  - 通过：fire 改善趋向 legacy（< 0.035），且 office/chess/redkitchen 不回退（各 ±0.003 内）
- **Gate 1b（soft-merge @1000f 不回退）**：fire/chess @1000f，soft-merge vs drop。
  - 通过：回退 < 0.01m（soft-merge 每帧改当前帧 K/V，1000f 也受影响；必须早查，不能拖到 Gate 5）
- **Gate 2（coverage anchor 验证）**：office（+其他 3 场景）@200f，coverage vs fixed_interval。**先 log 确认 coverage 分支触发**（keyframe registration 打印），否则静默重测 fixed_interval。
  - 通过：office 改善（趋向 0.030），且 ring/FIFO 不破坏（无异常/overflow）、其他场景不显著回退
- **Gate 3（500f redkitchen 诊断）**：诊断 500f redkitchen 崩溃根因 + 针对性修复
- **Gate 4（组合验证）**：soft-merge + coverage + redkitchen 修复，4 场景 × {200,500}f seed0。
  - 通过：200f mean < legacy；500f mean < legacy（**条件依赖 Gate 3**）；chess/redkitchen 不回退；1000f 无回退
- **Gate 5（全量确认）**：4 场景 × 3 seeds × {200,500,1000}f（deterministic，seeds 作确认），paired diff vs legacy。

---

## 3. 判定标准（成功）

- **200f**：frontend 4 场景 mean < legacy 0.0260；且 fire/office 各 < legacy（fire<0.0307, office<0.0298）
- **500f**：frontend mean < legacy 0.0738；redkitchen 不崩（< 0.05）。**条件依赖 Gate 3（redkitchen 修复）**——若 redkitchen 0.1034 无法压到使 mean<0.0738（其他三场景需均 <0.064），则 500f mean 目标不达成，诚实保留。
- **1000f**：保持 frontend −26% 优势（mean < 0.12，不回退）
- 无场景显著回退（±0.005 内）

---

## 4. 风险与诚实保留

1. **soft-merge 模糊风险**：加权平均多视角 K/V 可能不如保留单 token —— Gate 1 实测裁决。
2. **coverage anchor 工程风险**：改变 keyframe promotion，可能与 ring/FIFO 冲突 —— Gate 2 单独验证；若冲突不可调和，回退 fixed_interval，office 用其他方式。
3. **500f redkitchen 未知**：Gate 3 诊断结果决定是否可修；若根因深层（如场景固有），可能无法完全修复，接受 500f mean 仍略输（但 1000f 赢）。
4. **Pareto 不保证**：soft-merge 可能对某些场景中性偏负 —— 关卡设计确保任何回退即停止。
5. **"全面超越" 可能不完全达成**：若某场景/长度的 deficit 不可逆（场景固有），诚实报告，追求 mean 超越 + 多数场景超越。

---

## 5. 范围

**本 spec 范围**：soft-merge intra dedup + coverage anchoring + 500f redkitchen 诊断修复，目标 mean 超越 legacy。

**不在范围**（单独 spec）：
- 学习型 eviction / token scorer（已证无收益，spec §2.2 排除）
- unified-reprshift/cosine 评分重构（P2，weight 已证 0.5 最优，unified 是高风险结构改动）
- 场景自适应运行时逻辑（用户已排除）

---

## 附录: RCA 证据索引
- fire intra 确认：`tools/legacy_vs_frontend_lyj/` + 决定性 intra-OFF 测试（fire 0.0458→0.0279）
- 排除证据：13-agent RCA workflow（eviction/scoring/ring/cross-frame/pose 逐一代码+数据排除）
- 200f/500f/1000f 基线：`docs/legacy_vs_frontend_comparison.md`
