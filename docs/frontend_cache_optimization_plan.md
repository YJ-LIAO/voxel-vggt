# Frontend Cache 管理优化方案

> 基于代码审查 + SOTA 论文调研 (2024-2026) + 实证验证
> 日期: 2026-06-13 (第 3 版，经实证验证修订)
> 目标: 解决 noIntra+fifo80 的已知问题，提升长序列重建质量

---

## ⚠️ 方法论声明：先验证，再结论

本方案经过多轮修订，核心原则如下：

1. **实证验证优先于推理**：任何"看起来是 bug"的判断，必须用构造性测试（实际跑代码）验证，不能只靠读代码推断。本方案的 P3 曾被两轮 code review 判为"CONFIRMED 高危 bug"，但用 systematic-debugging 方法实测后证明是**幽灵 bug**（详见附录 C）。这种教训说明对并发/排序相关的代码，推理不可靠。

2. **实验先行**：任何评分/保护策略的改动都必须先通过 ablation 验证，而非先验断定优劣。SAGE-KV 在"看起来合理"层面完全说得通，但实测有害（ATE 0.0358 → 0.0467）。

3. **区分三类问题**：
   - **活跃 bug**：当前运行就会出错，可直接修复（P1, P4）
   - **设计选择**：当前实现的取舍，是否更优需实验（P2）
   - **潜在脆弱性**：依赖某个隐式不变量、当前碰巧成立但易被未来改动打破，应加断言而非改逻辑（P3, P5, P6）

4. **不采用** 学习型保护决策（见附录 A 的信用分配证伪）

---

## 第一部分：已确认的代码问题

### 问题 P1: FIFO topK 永久保护累积 [严重度: 高]

**位置**: `frontend_cache.py:353-368` (`protect_topk_on_demotion_`)

**现状**: 每次 FIFO swap 将 80 个 token 的 `anchor_slot` 设为 0（永久保护），之后再无代码将其重置。`protected_count` 单调递增。

**后果**:
```
FIFO swap 1: protected_count += 80
FIFO swap 2: protected_count += 80
...
FIFO swap 10: protected_count = 原始anchors + 800

当 protected_count ≥ cache_budget 时:
→ eviction 无法淘汰任何 candidate token
→ cache 被陈旧的 "永久保护" token 占满
→ 新帧 token 被挤出 → 重建质量崩塌
```

**实验佐证**: topK=110 时 200帧 ATE 从 0.026 飙升到 0.123，topK=200 崩塌到 0.486。**推测**这与 protected token 累积导致 budget 被占满有关，但因果链尚未严格验证（崩塌也可能由 budget 算术、保护 token 间相互作用等其他因素导致）。实验 3 应专门验证这一因果，而非预设结论。

---

### 问题 P2: Hybrid Eviction 用双指标 + 独立归一化 [类别: 设计选择，待验证]

**位置**: `attention.py:240-252`

**现状**: 老 token 用 cosine diversity，新 token 用 importance score，各自独立 min-max 归一化到 [0,1] 后拼接。

```python
# 两组独立归一化，不在同一尺度
old_normalized = _normalize_scores(old_scores)    # cosine diversity → [0,1]
new_normalized = _normalize_scores(new_importance) # repr_shift     → [0,1]
combined = cat([weighted_old, weighted_new])       # 直接比较！
```

**为什么是"设计选择"而非"bug"**:
- 独立 min-max 归一化是标准技术，其设计意图正是把两组拉到可比范围
- "两组 [0,1] 不代表同一重要程度"是一个**断言**，是否真的导致次优结果需实验
- `importance_weight=0.5` 是经验值，但当前生产配置就是靠它达到 0.0258 ATE

**待实验回答的问题**: 统一用 repr_shift（老 token 也有缓存的 importance）是否真的优于当前混合？还是当前混合的尺度差异其实是设计上的平衡？

---

### ~~问题 P3: Eviction 位置假设 vs 元数据选择不一致~~ [类别: 幽灵 bug，已证伪 ❌]

> **更新（第 3 版）**：本条曾在前两版被判为"CONFIRMED 高危 bug"，经 systematic-debugging 实证测试后**证伪**。详见附录 C 的测试报告。这里保留说明，供未来避免重复踩坑。

**最初的担忧**（已被证伪）:
- `_current_frame_importance()` 按元数据条件选择 token
- `eviction()` 按物理位置切分 candidates
- 担心 reorder + dedup 后位置不再对齐

**实测结论**：对齐在所有场景下成立，包括对抗性的帧内/跨帧 dedup 删除场景。

**为什么推理会错**：code review 漏看了关键一行——`_dedup_single_batch` 的**返回值**是 `torch.nonzero(keep_mask)`（升序位置序），而**不是**内部按 score-then-group 排序的 `final_order`。内部重排序只用来决定哪些 duplicate 索引被标记 False。整个链路每一步都保持幸存者的相对位置序，所以"新 token 始终在候选区尾部"这个不变量成立。

**正确处理方式**：不当 bug 改，而是把这个隐式不变量显式化——加 invariant 断言 + 注释（归入 P6 脆弱性）。

---

### 问题 P4: Voxel Hash 负坐标碰撞 [类别: 条件性 bug，已验证边界]

**位置**: `frontend_cache.py:935,943`

**实证验证**: hash `x + 1000y + 1000000z` 是混合基数表示，碰撞发生在 `|voxel| >= 1000` 时（即 x 溢出到 y 的位）。精确碰撞点：`voxel(1000,0,0)` 和 `voxel(0,1,0)` 都 hash 到 1000。

**边界条件**（voxel_size=0.1）:
- 室内场景（7-Scenes ~5m，voxel 范围 ±50）: **无碰撞** ❌ 不是问题
- 大场景（>=100m，voxel 范围 ±1000+）: **有碰撞** ✅ 真问题

**结论**: 对当前室内评估不是活跃 bug，但对室外/大场景部署是真实隐患。优先级低于 P1/P5。

---

### 问题 P5: 被保护 token 坐标投影错误（identity fallback）[类别: 活跃 bug，已验证 ✅]

**位置**: `frontend_cache.py:497,533` (protect_topk) + `frontend_keyframe.py:159` (FIFO_SWAP 丢弃 demoted slot) + `frontend_cache.py:1221` (identity fallback)

**实证验证**: 构造性测试确认，FIFO_SWAP 后被保护 token 的坐标投影**错误**（用 identity 而非正确变换）。

**根因链**（全部经代码 + 测试验证）:
1. token 创建时 `slot_id = current_keyframe_id`（ovggt.py:856 确认契约）
2. `protect_topk_on_demotion_` 把被保护 token 的 `anchor_slot` 改成 0，但**不改 `slot_id`**（仍指向被 demote 的 keyframe）
3. FIFO_SWAP 时 `remaining_slots = history_slots[1:]`（frontend_keyframe.py:159）**丢弃 demoted keyframe**
4. `_build_slot_pose_updates` 只为存活 keyframe 建 transform，demoted keyframe 的 transform 不在 dict 里
5. `apply_keyframe_event_` 用这个 dict **替换** `slot_to_active`（line 541），demoted keyframe 的 transform 消失
6. `_get_slot_transform(slot_id)` 找不到 → 返回 **identity**（line 1222）
7. 被保护 token 的 `slot_local_xyz`（在被 demote 的 keyframe 帧里）用 identity 投影 → **完全错误的世界坐标**

**实测**：正确投影应为 `(11,1,1)`，实际得到 `(1,1,1)`（identity fallback）。

**影响**：错误的 xyz 进入 `apply_voxel_dedup_` 的 cross-frame 比较，可能导致误删当前帧 token 或漏删重复。每次 FIFO_SWAP（fifo_keep_topk>0 时）都会触发。

**注意**：原方案误以为 P5 是"潜在脆弱性/疑似幽灵 bug"，实测证明是**活跃 bug**。修复 D（分离 anchor_slot 语义）**不能修复它**——根因是 slot_id 指向已删除的 transform，与语义混淆无关。需要专门的修复（见修复 F）。

---

### 问题 P6: 三机制协调依赖调用顺序 [严重度: 中]

**现状**: `protect_topk_on_demotion_` → `apply_keyframe_event_` → `commit_pending_update_` → `reorder` → `dedup` → `eviction` 的执行顺序硬编码在 caller 中。任何重排都会静默破坏正确性，且没有不变量断言检查一致性。

---

## 第二部分：SOTA 论文调研

### 论文 1: ZipVL (ICCV 2025) — 统一注意力重要性评分

**核心思想**: 用归一化的 attention scores 作为**单一统一指标**，驱动稀疏注意力和 KV cache 保留。不为老/新 token 用不同评分。

**关键机制**:
1. 对每层 attention 输出计算 token importance = attention weights 归一化分数
2. Top-K 选择 important tokens，其余从 KV cache 丢弃
3. **每层、每帧重新评估** importance — 没有 "永久保护"

**对我们的启发**: 统一评分思路值得借鉴，但 attention-based 重要性在 3D ViT 上已被 SAGE-KV 实验证伪（见代码库 FRONTEND_LONGSEQ_OPTIMIZATION.md Phase 1b），不能直接照搬。

### 论文 2: LongSplat (AAAI 2026) — 学习型更新掩码

**核心思想**: learned binary update mask 决定每帧删除哪些 3D Gaussians，解决 "永久保护累积"。

**关键机制**:
1. Gaussian-Image Representation (GIR): identity-aware 追踪
2. History Fusion Transformer: 预测 binary update mask
3. 用 **3D IoU overlap** 作为监督信号（关键：这是 per-Gaussian 的几何信号）
4. 44% Gaussian 数量减少，PSNR 稳定

**对我们的启发**: 动态保护思路值得借鉴。但**不能直接借鉴其监督方法**——见附录 A 的信用分配分析。

### 论文 3: StreamGS (arXiv 2025) — 跨帧特征聚合去冗余

**核心思想**: 通过相邻帧像素对应关系合并冗余 Gaussians，比 voxel-based dedup 精准。

**对我们的启发**: 跨帧 dedup 可考虑特征相似度，但需实验验证。

---

## 第三部分：确定性 Bug 修复（可直接实施）

只有活跃 bug 可直接修复。脆弱性加固见第六部分 Stage 0。

### 修复 C: 修复 Voxel Hash (解决 P4)

```python
def _compute_voxel_hash(self, voxels, config):
    """改进的 voxel hash，处理负坐标。"""
    # 推荐: 直接用 torch.unique 替代手工 hash
    unique_voxels, inverse = torch.unique(voxels, dim=0, return_inverse=True)
    return inverse  # 每个 token 所属的 voxel id，无碰撞
```

**改动**: 约 5 行。**风险**: 极低。**优先级**: 低（P4 是条件性 bug，室内评估无碰撞，仅大场景受影响）。

### 修复 F: 修复 P5 的 identity fallback（活跃 bug，已实施 ✅）

> P5 已实测确认为活跃 bug。修复 D（分离 anchor_slot 语义）**不能修复它**——根因是 slot_id 指向已删除的 transform，与语义混淆无关。

**采用方案 F2（从根上修复）**: 在 `frontend_keyframe.py` 的 `update()` FIFO_SWAP 分支，捕获被 demote 的 keyframe 的 `local_to_world`，按新 active 帧重算其 transform，保留到 `slot_pose_updates`。这样 `apply_keyframe_event_` 替换 `slot_to_active` 时不会丢失该 transform，所有引用它的 token（被保护 token + 普通 candidate）都能正确投影。

**改动**（`src/ovggt/utils/frontend_keyframe.py`，+19 行）:
```python
# FIFO_SWAP 分支: 捕获 demoted keyframe (在 history_slots 重编号前)
demoted_record = dict(self.history_slots[0]) if self.history_slots else None
...
# 重算并保留 demoted keyframe 的 transform
slot_pose_updates = self._build_slot_pose_updates(current_local_to_world)
if demoted_record is not None:
    demoted_kf_id = int(demoted_record["keyframe_id"])
    world_to_active = closed_form_inverse_se3(current_local_to_world.unsqueeze(0))[0]
    slot_pose_updates[demoted_kf_id] = world_to_active @ demoted_record["local_to_world"]
```

**TDD 验证**（`tests/test_p5_fifo_swap_transform_retention.py`）:
- RED: 修复前两个测试均失败（demoted keyframe 1 缺失；投影 (1,0,0) 而非 (-29,0,0)）
- GREEN: 修复后均通过（keys 含 demoted 1；投影正确 (-29,0,0)）
- 回归: 现有 12 个 keyframe/cache 测试全过；PROMOTE 路径无影响

**为什么 F2 而非 F1**: F1（缓存 active_xyz）需新增 metadata 字段，波及 append/gather/scorer 等多处。F2 从根因修复（transform 丢失），改动仅 19 行集中在单文件，且对所有引用该 keyframe 的 token 都正确（不只被保护 token）。

### 修复 D: 分离 anchor_slot 语义 (脆弱性加固，仅 P6)

> **第 3 版注**：P5 已实测为活跃 bug，但**修复 D 不能修 P5**（P5 用修复 F）。修复 D 仅解决 P6 的语义混淆脆弱性，让 anchor_slot 不再同时承载"保护状态"语义。

```python
@dataclass
class TokenMetadata:
    # 现有字段不变
    anchor_slot: Tensor     # 仅用于坐标帧标识 + 排序
    slot_id: Tensor         # 仅用于坐标变换查找
    importance: Tensor
    
    # 新字段
    protection_tier: Tensor  # [B, N] 保护等级 (0=无, 1=anchor, 2=FIFO临时)
    
    @property
    def is_protected(self):
        return self.protection_tier > 0
```

**收益**: 仅 P6 脆弱性加固。**优先级**: 低（纯架构清理，非活跃 bug）。

---

## 第四部分：待验证的算法改进（必须先跑实验）

以下方案属于**假设**，在没有 ablation 数据前不实施。

### 假设 A: 统一 Eviction 评分 (仅探索 P2)

> **第 3 版更新**：P3 已证伪，本假设的立论基础塌了一半。原来"统一评分修复 P2+P3 两个问题"现在只剩 P2（评分尺度）这一个理由。优先级相应降低。

**假设**: 老 token 的 `metadata.importance` 存储的就是 repr_shift_spatial（和新 token 同度量），统一用单一指标可能比当前双指标 + 独立归一化更公平。

**为什么优先级降低**:
- P3（位置不一致）已证伪，统一评分不再解决"对齐问题"
- 只剩 P2 一个理由，而 P2 本身是设计选择，当前混合方案已经达到 0.0258 ATE
- 统一用 repr_shift **完全可能更差**——我们不知道 cosine diversity 对老 token 是否提供了 repr_shift 没有的信号

**改动范围**（如果实验支持）:
- `frontend_cache.py`: `_current_frame_importance` → `_all_candidate_importance`
- `attention.py`: `eviction()` 简化，不分 old/new
- 消除 `importance_weight` 参数

**待验证问题**: 统一 repr_shift vs 统一 cosine diversity vs 当前混合，哪个最好？

### 假设 B: 保护机制改进 (解决 P1)

P1 是真实 bug，但**修复方式有多种，各有权衡**，需要实验决定：

**方案 B1 — 保护预算上限** (最简单)
```python
max_total_protected_ratio: float = 0.5  # protected 不超过 budget 的 50%
def protect_topk_on_demotion_(...):
    max_protected = int(config.max_total_protected_ratio * cache_budget)
    available = max(0, max_protected - self.protected_count)
    actual_keep = min(keep_count, available)
```
- 优点: 一行 config
- 缺点: 预算满后无法保护新 token

**方案 B2 — 保护衰减** (中等复杂)
```python
protection_decay_frames: int = 24
# token 获得临时保护，24 帧后自动失去保护
# 每帧 tick_protection() 衰减计数器
```
- 优点: 保护动态化，自动淘汰陈旧 token
- 缺点: decay_frames 是新魔法数，需调参

**方案 B3 — 完全移除 fifo 保护** (baseline 对照)
```python
fifo_keep_topk = 0
```
- 用作 ablation 的对照组

**待验证问题**: B0(当前永久保护) vs B1(预算上限) vs B2(衰减) vs B3(无保护)，长序列哪个最好？

---

## 第五部分：实验设计（实施前置条件）

### 实验 1: Eviction 评分对比 (验证假设 A)

**目的**: 确定 P2（双指标 + 独立归一化）是否真的次优。注意 P3 已证伪，本实验只回答 P2。

| 配置 | 评分方式 | 说明 |
|------|---------|------|
| Exp1-baseline | cosine(old) + repr_shift(new), 独立归一化 | 当前实现 |
| Exp1-A1 | 统一 repr_shift, 单一归一化 | 假设 A |
| Exp1-A2 | 统一 cosine diversity | 对照 |
| Exp1-A3 | 统一累积 attention (ZipVL 风格) | SOTA 复现 |

**评估**: chess / fire / office / redkitchen 各 200帧 ATE
**成本**: 只改 `eviction()` 评分部分 (~30行)，无需训练，每配置约几十分钟
**统计要求**: 4 场景 × 3 seeds (降低方差)，报告均值 ± 标准差；只有当 Exp1-A1 的 4 场景平均 ATE **在标准差范围外**低于 Exp1-baseline 才采纳。避免 0.0258 vs 0.0262 这种落在噪声内的"改进"。
**混淆注意**: Eviction 评分影响哪些 token 存活 → 影响 protected_count 动态 → 与实验 2 的保护机制交互。因此实验 1 和实验 2 应在**相同保护配置**下跑，或先固定保护机制再做评分实验。

### 实验 2: 保护机制对比 (验证假设 B)

**目的**: 确定 P1 的最佳修复方式，并量化永久保护的累积危害。

| 配置 | 保护机制 | 说明 |
|------|---------|------|
| Exp2-baseline | fifo_keep_topk=80, 永久保护 | 当前实现 |
| Exp2-B1 | 保护预算上限 (ratio=0.5) | |
| Exp2-B2 | 衰减保护 (decay 待调参) | |
| Exp2-B3 | fifo_keep_topk=0, 无保护 | 下界对照 |

**B2 的 decay 值不能拍脑袋**: 需要先做一个小 sweep（decay ∈ {8, 16, 24, 48}），避免重蹈 `fifo_keep_topk=80` 魔法数的覆辙。

**评估**:
- 各场景 200帧 ATE（3 seeds，报告均值 ± 标准差）
- **protected_count 随帧数增长曲线**（关键：量化 P1 危害）
- 500帧/1000帧长序列是否崩塌

**成本**: 只改 `protect_topk_on_demotion_` (~20行)，无需训练
**判定标准**: 短序列 (200帧) 不退化，长序列 (500+) 不崩塌

### 实验 3: 长序列压力测试 (验证 P1 因果)

**目的**: 实验 2 的扩展，**专门验证 P1 的因果链**——protected_count 增长是否真的导致 ATE 崩塌（而非其他原因）。

```
对每个保护配置，跑 chess/seq-03 到 500帧 和 1000帧:
  - 记录每帧 protected_count
  - 记录 ATE 曲线
  - 观察是否在某个帧数后突然崩塌
  - 关键: 观察 ATE 崩塌点是否与 protected_count 接近 cache_budget 的时刻吻合
    （若吻合 → 因果成立；若不吻合 → P1 的因果推断被推翻，需重新归因）
```

**数据可用性前置检查**: 先确认 chess/seq-03 是否真有 500+ 帧；若无，需换其他长序列或合成数据。

### 实验 0: P5 验证 (已完成 ✅)

**目的**: 用构造性测试确认 `_project_slot_local_xyz_to_active` 对被保护 token 是否真的出错。

**结果**: P5 已实测**确认为活跃 bug**——FIFO_SWAP 后 demoted keyframe 的 transform 丢失，被保护 token 的 slot_id 仍指向它，`_get_slot_transform` 返回 identity，投影错误（实测 (1,1,1) 而非正确的 (11,1,1)）。详见附录 D。**已升级为活跃 bug，用修复 F 处理**。

---

## 第五点五部分：所有问题的最终判定汇总（实证）

| 问题 | 原判定 | 实证后判定 | 依据 |
|------|--------|-----------|------|
| **P1** 永久保护累积 | 高危 bug | ✅ **CONFIRMED 活跃 bug** | 流式模拟：slot0 从 3→643 (8 swaps × 80)，单调不减 |
| **P2** 双指标评分 | 高危 bug | ⚙️ **设计选择** | 代码行为确认；是否次优待实验 1 |
| **P3** 位置/元数据不一致 | 高危 bug | ❌ **REFUTED 幽灵** | 3 场景构造测试全部对齐（附录 C） |
| **P4** voxel hash 碰撞 | 中危 bug | ⚠️ **CONDITIONAL** | 室内(±50 voxel)无碰撞；>=100m 场景有 |
| **P5** 坐标投影错误 | 中危/疑似幽灵 | ✅ **CONFIRMED 活跃 bug** | identity fallback 实测（附录 D）；slot_id==keyframe_id 契约确认 |
| **P6** 无阶段间断言 | 中危 | ✅ **CONFIRMED 脆弱性** | frontend_cache.py 0 个 assert |

**修复方案匹配审查**:
- 修复 C → P4：有效，但 P4 仅大场景受影响，低优先级
- 修复 D → ~~P5~~ P6：**不能修 P5**（语义混淆 ≠ slot_id/transform 失配），仅修 P6 脆弱性
- 修复 F → P5：新增，针对 identity fallback 根因（推荐 F1）
- 假设 A → P2：仅剩 P2 一个理由，需实验 1
- 假设 B → P1：实验 2/3 验证修复方式

---

## 第六部分：实施路径（修订版）

### Stage 0: 确定性 Bug 修复 + 脆弱性加固 [立即可做]

| 任务 | 解决 | 风险 | 验证 |
|------|------|------|------|
| 修复 F: P5 identity fallback | P5 (活跃 bug，已确认) | 低 | P5 构造测试作回归用例 |
| 加固 E: P3 不变量断言 | P3 (脆弱性) | 极低 | commit 前断言候选区尾部是当前帧 |
| 修复 C: voxel hash | P4 (条件性 bug) | 极低 | 单元测试验证无碰撞（低优先级，仅大场景） |
| 修复 D: 分离 anchor_slot 语义 | P6 (脆弱性) | 中 | 7-Scenes 评估确认无回归 |

**关键变化（实证后）**：P5 已确认为活跃 bug（非幽灵），优先用修复 F。P3 仍是脆弱性（加断言）。P4 降为条件性（室内无影响）。

### Stage 1: 实验 Ablation [决策依据]

| 任务 | 目的 | 产出 |
|------|------|------|
| 实验 1 (评分对比，3 seeds) | 决定 eviction 评分方案 | 选出最佳评分（或确认当前混合最优） |
| 实验 2 (保护对比 + B2 decay sweep) | 决定保护机制 | 选出最佳保护 |
| 实验 3 (长序列压力测试) | 验证 P1 因果 | 确认崩塌点与 protected_count 吻合 |

**重要**：实验 1 和 2 存在混淆，需固定一个维度做另一个。

### Stage 2: 实施实验胜出的方案 [基于数据]

根据 Stage 1 的 ablation 结果，实施实验中表现最好的配置。
- 如果 Exp1-A1 胜出 → 实施统一 repr_shift 评分
- 如果 Exp2-B2 胜出 → 实施衰减保护
- 没有实验证据的方案**一律不实施**

### Stage 3: 未来探索 [低优先级]

以下方向在 Stage 1-2 完成后再考虑，且同样需要实验验证：
- 特征相似度 dedup (StreamGS 启发)
- 自适应帧内 dedup (替代 noIntra 一刀切)
- ~~学习型保护决策~~ — 见附录 A，已被证伪，不采用

---

## 附录 A: 为什么不采用学习型保护决策

### 信用分配难题（证伪）

原方案 B3 提议用 reconstruction loss 监督一个保护决策网络。但每次 eviction 批量淘汰上百个 token，reconstruction loss 是单个标量，**无法归因到具体 token**：

```
eviction 淘汰 200 个 token → 未来帧 loss 上升 → 哪个 token 的锅?
❌ 无法从单个 loss 标量反推到具体 token
```

要获得 per-token 标签需要逐 token 做 counterfactual（每次只 evict 一个），这比 oracle 收集还贵，且有同样的孤立评估问题。

### 为什么不能照搬 LongSplat

LongSplat 用 3D IoU overlap 作为监督，这是 **per-Gaussian 的几何信号**——每个 Gaussian 独立判断"是否被新 Gaussian 覆盖"。KV cache 里的 token 没有这种干净的几何对应关系，所以这条路不通。

### 为什么不能照搬 ZipVL 的 attention 监督

ZipVL 用 attention 作为 per-token 重要性。但在 3D ViT 上，Phase 1b 的 SAGE-KV 实验已证明纯 attention 监督**有害**（ATE 0.0358 → 0.0467），因为：
1. 累积 attention 偏向旧 token
2. 3D ViT attention 密集空间化，"受欢迎" ≠ "对重建重要"

### 结论

学习型保护决策在当前条件下不可行。Stage 1-2 用 cheap 的确定性机制（预算上限/衰减）+ 实验验证是更现实的路径。

---

## 附录 B: 实验验证状态追踪

| 假设 | 实验 | 状态 | 结论 |
|------|------|------|------|
| 永久保护累积 (P1) | 流式模拟测试 | ✅ 已确认+修复 | slot0 累积，修复 F(P1) 加预算上限 (TDD) |
| 统一评分优于混合评分 (P2) | 实验 1 | ⏳ 未开始 | 设计选择，待验证 |
| Eviction 位置/元数据不一致 (P3) | 构造测试 | ✅ 已证伪 | 幽灵 bug，归入脆弱性 |
| voxel hash 碰撞 (P4) | 边界测试 | ✅ 已确认+修复 | 室内无碰撞，>=100m 有，修复 C 改无碰撞 hash (TDD) |
| protect_topk 破坏坐标投影 (P5) | identity fallback 测试 | ✅ 已确认+修复 | 活跃 bug，修复 F(P5) 保留 transform (TDD) |
| 三机制无断言 (P6) | 静态检查 | ✅ 已确认+加固 | 加 invariant 断言 (P6 guard) |
| 纯 attention 监督有效 | SAGE-KV (历史) | ✅ 已证伪 | 有害，不采用 |

---

## 附录 C: P3 证伪报告（实证验证记录）

### 背景

P3 曾在两轮 code review 中被判为"CONFIRMED 高危 bug"：`_current_frame_importance()` 按元数据选择 token，`eviction()` 按物理位置切分 candidates，担心 reorder + dedup 后两者不对齐。

### 方法

按 systematic-debugging 流程，构造性测试而非推理。写了 `/tmp/test_p3_alignment.py`，直接构造 LayerCacheState，跑完整的 `append_ → reorder_by_anchor_slots_ → apply_voxel_dedup_ → _current_frame_importance` 序列，检查：
1. 候选区尾部 N 个位置是否全是当前帧 token
2. `importance_scores[i]` 是否对应位置 `num_old + i` 的实际 importance

跑了 3 个对抗场景：S1（无删除）、S2（帧内 dedup 删除，触发内部 score/group 重排序）、S3（跨帧 dedup 删除当前 token）。

### 结果

| 场景 | 尾部都是当前帧? | importance 对齐? |
|------|---------------|----------------|
| S1 | ✅ | ✅ True |
| S2 | ✅ | ✅ True |
| S3 | ✅ | ✅ True |

**全部对齐**。P3 是幽灵 bug。

### 推理为什么会错（根因）

code review 漏看了 `frontend_cache.py:995`：
```python
return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)  # 升序位置序，不是 final_order
```

`_dedup_single_batch` 内部确实有 score-then-group 重排序（line 981-983），但那只用来**标记哪些 duplicate 索引被设 False**（line 993）。返回值是 `nonzero(keep_mask)`——升序位置序。整个链路每步都保持幸存者相对位置序，所以"新 token 在候选区尾部"不变量成立。

### 诚实限定

- 测试直接构造 cache state 调用方法，覆盖了 `commit_pending_update_` 的完整序列，但未走完整推理管线。
- 前端强制 B=1，B>1 的 `gather_per_batch_` 路径未测（生产路径不走）。
- `dedup_replay_probe` override 路径（oracle 收集时）用 probe 索引可能非升序，但该路径 return 早退，不影响正常推理的 eviction。

### 教训

对涉及排序/顺序依赖的代码，**读代码推理不可靠**，必须构造性测试。本方案因此新增"先验证再结论"为第一原则。

---

## 附录 D: P1/P4/P5/P6 验证报告（实证记录）

### P1：FIFO topK 永久保护累积 — CONFIRMED

**方法**：构造 cache（slot0=3 + slots1/2/3 各 200 token），模拟 8 次真实 FIFO 循环（每次 protect_topk=80 → demote slot1 → shift → 填充新 keyframe）。

**结果**：slot0 数 `[3, 83, 163, 243, 323, 403, 483, 563, 643]`——每次精确 +80，单调不减。代码审查确认无任何路径将 `anchor_slot=0` 重置为 -1。

### P4：voxel hash 碰撞 — CONDITIONAL

**方法**：解析 hash `x + 1000y + 1000000z`。精确碰撞点 `voxel(1000,0,0) === voxel(0,1,0) = 1000`。

**结果**：碰撞发生在 `|voxel| >= 1000`（x 溢出到 y 位）。voxel_size=0.1 时对应场景尺度 >=100m。室内（±50 voxel）无碰撞，大场景（>=100m）有。

### P5：identity fallback — CONFIRMED（最严重发现）

**方法**：构造 token（slot_id=1，slot_local_xyz=(1,1,1) 在 kf1 帧，正确投影应为 (11,1,1)）。先验证契约 `slot_id == current_keyframe_id`（ovggt.py:856 确认）。然后 protect_topk → 模拟 FIFO_SWAP 删除 kf1 transform → 投影。

**结果**：
```
FIFO_SWAP 前 投影: (11,1,1)  正确
FIFO_SWAP 后 投影: (1,1,1)   错误! identity fallback
```

**根因链**（全部代码确认）：protect_topk 改 anchor_slot=0 但不改 slot_id → FIFO_SWAP 丢弃 demoted keyframe → slot_to_active 替换丢失该 transform → _get_slot_transform 返回 identity → 投影错误。**每次 FIFO_SWAP（fifo_keep_topk>0）触发**。

**关键教训**：原方案把 P5 误判为"疑似幽灵 bug"，实测是活跃 bug。且原修复 D（分离 anchor_slot 语义）**不能修它**——根因是 slot_id/transform 失配，与语义混淆无关。新增修复 F1。

### P6：无阶段间断言 — CONFIRMED（脆弱性）

**方法**：静态检查 `frontend_cache.py` 中 assert 数量。

**结果**：全文 0 个 assert，`commit_pending_update_` 内 0 个。流水线阶段（protect_topk → apply_keyframe → commit → reorder → dedup → eviction）交接处无任何 invariant 断言。确认脆弱性存在（非活跃 bug，是缺乏防护）。

---

## 参考文献

1. **ZipVL** — He et al., "ZipVL: Accelerating Vision-Language Models through Dynamic Token Sparsity", ICCV 2025. [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/He_ZipVL_Accelerating_Vision-Language_Models_through_Dynamic_Token_Sparsity_ICCV_2025_paper.pdf)

2. **LongSplat** — Huang et al., "LongSplat: Online Generalizable 3D Gaussian Splatting from Long Video Sequence", AAAI 2026. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/42504/46465)

3. **StreamGS** — "StreamGS: Streaming 3D Gaussian Splatting with Cross-frame Feature Aggregation", arXiv 2025. [arXiv](https://arxiv.org/abs/2503.06235)
