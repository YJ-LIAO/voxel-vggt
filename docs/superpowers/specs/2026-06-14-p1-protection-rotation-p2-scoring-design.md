# P1 保护策略候选 + P2 评分整合 Spec

> 日期: 2026-06-14
> 状态: 设计待批准 (brainstorming 产出)
> 关联: `docs/frontend_cache_optimization_plan.md` (P1 当前实现有缺陷, 需重设计; P2 待验证)
> 范围: P1 (保护策略候选验证) + P2 (eviction 评分), 一起执行便于统一验证记录

---

## 0. 背景与动机

### P1 当前实现的问题

P1 第一版修复 (`max_protected_ratio` 预算上限) **有缺陷**:
- 它只是把"无限累积"换成"硬截断"——预算满时**直接停止保护任何新 keyframe token**
- 这把一种崩塌换成另一种僵化: 新 keyframe 的 token 无法进入保护区
- 用户反馈: "P1 也算设计选择问题, 不能把关键帧 token 一直保护。流式输入有大量帧, 一直保护会导致后面关键帧的 token 无法加入 cache。目前的改动也有问题"

**本轮不预设需求**: "保护必须轮换" 不再作为已确认需求写死。P1 的确定问题只是: 永久保护会累积, 简单硬截断也可能僵化。轮换、TTL、维持当前 cap 或关闭 FIFO topK 都作为候选策略, 由实验决定是否采用。

### P2 现状

P2 (eviction 双指标 + 独立归一化) 是**设计选择**, 未做任何代码改动。需实验决定是否优化。与 P1 同属"保护/评分机制", 合并到一个执行单元便于统一验证。

### 严格原则 (继承自前置方案)

- **实证优先**: 任何机制优劣判断需实验数据, 不先验断定
- **不动 eviction 核心二分** (除非必要): P3 不变量刚验证过, 改 eviction 核心风险高
- **TDD**: 每个机制改动配 RED→GREEN 回归测试

---

## 1. P1 保护策略候选设计

### 1.1 核心约束 (已确认)

- eviction 在 `protected_count` 处**硬二分**: `[0:protected_count]` 无条件保留, `[protected_count:]` 才淘汰
- `protected_count = count(anchor_slot >= 0)`
- 当前 `protect_topk_on_demotion_` 永久设 `anchor_slot=0`, 无撤销机制
- slot 0 (global anchor) 永不被 FIFO demote, 故移入即永久

### 1.2 三类候选机制 (对比)

| 机制 | eviction 改动 | 新增字段 | 触发 | 轮换/退出语义 | 回归风险 |
|------|-------------|---------|------|------|---------|
| **A: TTL 衰减** | 无 | protection_remaining | 每帧 tick | 突变 (TTL到期整批) | 低 |
| **B: 软评分** | **大 (改核心二分)** | 无 (用keyframe年龄) | 每次 eviction | 渐变 | **高** |
| **C: 有界 FIFO rescued pool** | 无 | 无 (复用 keyframe_id) | 仅 keyframe 事件 | 按容量撤销最老 rescued tokens | 低 |

### 1.3 推荐优先验证机制: C (有界 FIFO rescued pool)

**优先验证理由** (排序):
1. **eviction 不变** → 不触碰 P3 不变量, 回归风险最低
2. **无新字段** → 复用 keyframe_id 作为年龄排序, 不增加 metadata 传播负担
3. **架构自洽** → 与现有 `history_slots` FIFO 语义对齐 (keyframe 本身就是 FIFO 环)
4. **负载自适应** → 保护请求多时老的 rescued tokens 退出更快, 适合作为长序列候选
5. **K 有自然取值** → `max_history_anchors × keep_per_anchor`, 非凭空魔法数

本 spec 不把 C 视为最终答案。Stage 1 至少需要和 P1-current、P1-none 对照; TTL/软评分只在 C 相对 current/none 没有收益或暴露明显副作用时再实现。**B 暂缓** (改 eviction 核心, 代价/风险最高)。

### 1.4 机制 C 详细设计

#### 数据结构 (无新字段)

复用现有 `metadata.keyframe_id` 作为"保护年龄"。keyframe_id 越小 = 越老。

#### 配置

```python
@dataclass
class FrontendCacheConfig:
    # ... 现有字段 ...
    fifo_keep_topk: int = 0
    # P1 v2 (本 spec): FIFO rescued token 保护池的有界容量。
    # 只限制 slot-0 中 keyframe_id != global_anchor_keyframe_id 的 rescued tokens;
    # global anchor 自身不计入 ring capacity, 也不参与 rescued pool 撤销。
    # 注意: ring_ratio 只控制累计保护池大小, 不决定每次保护多少 token。
    # 每次 FIFO_SWAP 仍由 fifo_keep_topk 或 learned_fifo_keep_count 决定 keep_count。
    # 0.0 = 禁用 rescued pool 容量约束 (回退到 v1 的 max_protected_ratio 行为); 0.3 = rescued pool 占 30%。
    fifo_protected_ring_ratio: float = 0.0
```

当 `fifo_protected_ring_ratio > 0` 且本次 FIFO_SWAP 有保护请求 (`fifo_keep_topk>0` 或 learned count > 0) 时启用 rescued pool 容量约束; 否则保留 v1 行为 (向后兼容)。
**绝对上限** `ring_capacity = fifo_protected_ring_ratio × per_layer_budget`, 在调用方计算后传入。ring 不是 "ring-only protection" 开关: 如果 `fifo_keep_topk=0` 且未启用 learned count, 不应因为 ring_ratio>0 而把 demoted slot 全量保护。

**v1/v2 互斥的 config 层强制 (pass-5 发现 4)**: 不只依赖调用方传 None, 在 `FrontendCacheConfig.__post_init__` 加断言, 防止用户同时设两者:
```python
def __post_init__(self):
    if self.fifo_protected_ring_ratio > 0.0 and self.max_protected_ratio < 1.0:
        raise ValueError(
            "fifo_protected_ring_ratio 和 max_protected_ratio 互斥 (两者顺序作用同一 keep_count 会混淆)。"
            "启用 ring 时保持 max_protected_ratio=1.0 (默认, cap 禁用)。"
        )
```

#### protect_topk_on_demotion_ 改动

**关键设计决策 (回应 spec review pass 1 + pass 2 发现)**:
- **撤销计划必须在 probe 之前计算, 但 mutation 在 probe 之后应用** (保留 probe 对原始候选集的观察语义, 同时避免 override 早退绕过撤销)
- **global anchor 用显式 id, 不用 keyframe_id.min() 推断** (pass 1: 避免脆弱代理)
- **ring capacity 只统计非 global 的 rescued slot-0 token** (global anchor 不占用 rescued pool 容量)
- **撤销后强制 reorder** (pass 1: reorder 受 has_anchor_tokens 门控, revoke 后需触发)
- **v1/v2 互斥**: ring 启用时 max_protected=None (pass 1: 避免顺序作用同一 keep_count)
- **per-batch 处理, 不 assert B==1** (pass 2: 代码已支持 B>1, assert 会误报)
- **topk tie 用次级 key 打破** (pass 2: 同 keyframe 的 token id 相同, 需确定性 tie-break)
- **`_needs_reorder_after_revoke` 声明为 dataclass field** (pass 2: 非运行时属性)
- **ring_capacity 按稳定 per_layer_budget 基准算** (保护池是长期结构; 逐帧动态 budget 仍只用于 eviction)

```python
def protect_topk_on_demotion_(self, demoted_slot, keep_count, ...,
                                fifo_ring_capacity=None,
                                global_anchor_keyframe_id=None):
    # 步骤 1: 计算 demoted indices (现有)
    # 步骤 2.5 (P1 v2, probe 之前): 若启用 FIFO 环且会超容, 计算最老 keyframe 的撤销计划
    revoke_by_batch = {}
    keep_count_by_batch = {}
    if fifo_ring_capacity is not None and fifo_ring_capacity > 0:
        # pass 2: per-batch (不 assert B==1; 代码已支持 B>1)
        for b_idx in range(self.metadata.anchor_slot.shape[0]):
            requested_keep_count = int(keep_count)  # batch-local; 不要在 batch 间复用被 clamp 后的值
            slot0_mask = (self.metadata.anchor_slot[b_idx] == 0)
            slot0_indices = torch.nonzero(slot0_mask, as_tuple=False).squeeze(-1)
            slot0_kf_ids = self.metadata.keyframe_id[b_idx, slot0_indices]
            gaid = int(global_anchor_keyframe_id) if global_anchor_keyframe_id is not None else -1
            rotatable = slot0_kf_ids != gaid
            rot_idx = slot0_indices[rotatable]
            rot_kf = slot0_kf_ids[rotatable]
            if rot_idx.numel() + requested_keep_count <= fifo_ring_capacity:
                keep_count_by_batch[b_idx] = requested_keep_count
                continue  # 未超容, 无需撤销
            overflow = (rot_idx.numel() + requested_keep_count) - fifo_ring_capacity
            k = 0
            if rot_idx.numel() > 0:
                k = min(overflow, rot_idx.numel())
                # pass 2: 确定性 tie-break — 主 key=keyframe_id(老优先), 次 key=token位置(稳定)
                # 用 argsort(stable=True) 代替 topk, 避免 tie 不确定
                order = torch.argsort(rot_kf, stable=True)  # 升序 = 老 keyframe 优先, 稳定=同 id 按原位置
                revoke_by_batch[b_idx] = rot_idx[order[:k]]
            # pass-3 发现 1 修复: 无论是否撤销 (含 rot_idx 不足以覆盖 overflow 的情况),
            # 都把 keep_count clamp 到撤销后剩余的可用容量, 否则 pool 会超容。
            # 旧版仅在 rot_idx==0 的 else 分支 clamp, 漏了 rot_idx>0 但 overflow>rot_idx 的情况。
            remaining_after_revoke = rot_idx.numel() - k  # 撤销后仍存活的 rescued
            keep_count_by_batch[b_idx] = min(
                requested_keep_count,
                max(0, fifo_ring_capacity - remaining_after_revoke),
            )
    # 步骤 3 (现有): probe 在任何 metadata mutation 前执行, 仍观察原始 demoted-slot 候选集
    # 步骤 3.5: 无论 probe 是否 override, 先应用 revoke_by_batch, 防止 override 早退绕过撤销
    for b_idx, revoke_indices in revoke_by_batch.items():
        if revoke_indices.numel() > 0:
            self.metadata.anchor_slot[b_idx, revoke_indices] = -1
            self._needs_reorder_after_revoke = True  # dataclass field (见下)
    if revoke_by_batch:
        self._cached_protected_count = self._compute_protected_count_raw()
        self.protected_count = self._cached_protected_count
    # 步骤 4 (现有 per-batch topk 保护, 设 anchor_slot=0):
    #   pass-5 发现 1 修复: 实现时把现有 step-4 循环 (frontend_cache.py:550-551) 的单一
    #   keep_count 改为按 batch 取 keep_count_by_batch, 否则 step 2.5 的 clamp 被计算却不使用:
    #     for b_idx, indices in demoted_indices_by_batch.items():
    #         effective_keep_count = min(keep_count_by_batch.get(b_idx, keep_count), indices.numel())
    #   B=1 (前端) 下 keep_count_by_batch[0] == clamp 后值, 单一/per-batch 结果相同;
    #   B>1 时 per-batch 才有意义 (spec 的 B>1 主张与此接线一致)。
```

**`_needs_reorder_after_revoke` 必须声明为 `LayerCacheState` 的 dataclass field** (pass 2 发现 #6), 否则是脆弱的运行时属性:
```python
@dataclass
class LayerCacheState:
    # ... 现有字段 ...
    _needs_reorder_after_revoke: bool = False  # 新增 field
```

#### 调用方接线 (ovggt.py) — 修正 pass 2 发现 #1/#2/#4

**pass 2 发现 #1**: 调用点在 `for b in range(B)` 内, 作用域里是 `keyframe_managers[b]` (列表), 不是单数 `keyframe_manager`。
**pass 2 发现 #2 修订**: ring_ratio 不是保护触发条件, 只是在已有保护请求发生时控制累计 rescued pool。若只设 ring_ratio>0 且 `fifo_keep_topk=0` / `learned_fifo_keep_count=False`, 不应把 demoted slot 全量保护, 因为这会绕过 top-K 质量筛选。
**pass 2 发现 #4**: budget 是动态 `current_budgets[layer_idx]`, 不是 `self.per_layer_budget` 平均值。但 ring_capacity 是"保护池比例", 应按一个稳定的容量基准算 (用 per_layer_budget 作为比例基准是合理的, 因为 ring 是长期结构而非逐帧 budget); 明确这一点。
**pass 4 发现 #2**: 逐帧动态 layer budget 可能低于稳定 per_layer_budget, 仍可能触发 `eviction()` 的 anchor overflow 分支。实验必须记录 `anchor_overflow_count/rate`; 如出现 overflow, 增加一个保守对照 `ring_capacity = ratio * min(per_layer_budget, current_layer_budget)`。

```python
# ovggt.py 修正后的门控:
is_fifo_swap = str(getattr(events[b], "event_type", None)).endswith("FIFO_SWAP")
ring_ratio = self.frontend_cache_config.fifo_protected_ring_ratio
ring_enabled = ring_ratio > 0.0
if is_fifo_swap and (
    self.frontend_cache_config.fifo_keep_topk > 0
    or self.frontend_cache_config.learned_fifo_keep_count
):
    demoted_slot = getattr(events[b], "demoted_slot", None)
    # ... keep_count 计算 ...
    if self.frontend_cache_config.fifo_keep_topk > 0:
        keep_count = int(self.frontend_cache_config.fifo_keep_topk)
    elif self.frontend_cache_config.learned_fifo_keep_count:
        keep_count = predicted_count

    cache_states[b][layer_idx].protect_topk_on_demotion_(
        demoted_slot=demoted_slot,
        keep_count=keep_count,
        ...,
        # pass 2 #4: 默认 ring_capacity 按比例 × per_layer_budget (作为长期结构基准, 非逐帧动态 budget)
        # pass 4 #2: 若 anchor_overflow_rate>0, 增加保守对照 ratio * min(per_layer_budget, current_layer_budget)
        fifo_ring_capacity=(int(ring_ratio * self.per_layer_budget) if ring_enabled else None),
        # pass 2 #1: global anchor id 从 keyframe_managers[b] 取 (单数 keyframe_manager 不存在)
        global_anchor_keyframe_id=(
            keyframe_managers[b].global_anchor["keyframe_id"]
            if keyframe_managers[b].global_anchor is not None else 0
        ),
    )
```

#### commit_pending_update_ 配套改动 (强制 reorder, pass 1 #3)

reorder 受 `if metadata_current.has_anchor_tokens()` 门控。撤销改了 anchor_slot 后, 若当前帧非 anchor 帧, reorder 不触发, revoked token 仍在保护区位置。修复 (用声明的 dataclass field):

```python
# commit_pending_update_ 中 append_ 之后:
if metadata_current.has_anchor_tokens() or self._needs_reorder_after_revoke:
    self.reorder_by_anchor_slots_()
    self._needs_reorder_after_revoke = False  # reset
```

注: protect_topk (设 flag) → apply_keyframe_event_ (不改 flag) → commit_pending_update_ (读+reset flag)。flag 在同一 cache_states[b][layer_idx] 对象上, 跨这三个调用存活 (pass 2 #3 确认调用顺序)。

#### 关键语义

- **撤销单位**: 按 keyframe_id 升序 (argsort stable) 撤销最老的 rescued tokens, 同 id 按原位置 tie-break (确定性)
- **保护区上限**: 仅限制非 global 的 rescued slot-0 token, 上限 = ratio × per_layer_budget (per_layer_budget 作为长期基准; 逐帧动态 budget 用于 eviction), sweep {0.1,0.2,0.3,0.5}
- **保护数量来源**: 每次 FIFO_SWAP 的新增保护数量仍来自 `fifo_keep_topk` 或 learned count; ring_ratio 不做 ring-only 全量保护
- **global anchor 保护**: 不参与 rescued pool 撤销 (显式 `global_anchor_keyframe_id`, 从 `keyframe_managers[b].global_anchor` 取)
- **撤销后行为**: anchor_slot=-1 → candidate 区 → eviction 决定去留; **强制 reorder** (dataclass field flag)
- **撤销时机**: probe 前计算 revoke plan, probe 后应用 mutation (避免 override 早退绕过, 同时保留 probe 原始观察语义)
- **B 契约**: per-batch 循环处理 (代码已支持 B>1, 不 assert)
- **v1/v2 互斥**: ring 启用时 max_protected=None
- **边界 fallback (pass-3 修复)**: 无论是否撤销, 都把 keep_count clamp 到撤销后剩余可用容量 (修复 rot_idx>0 但 overflow>rot_idx 时 pool 超容的缺口)
- **门控**: ring_enabled 不单独触发 protect_topk 路径; 它只约束已有 top-K/learned 保护请求
- **动态 budget 交互**: ring 默认使用稳定 per_layer_budget; 实验记录 anchor overflow。若任一配置出现 protected anchors 被 overflow 裁剪, 增加 `min(per_layer_budget, current_layer_budget)` 的保守 cap 对照。

#### 实现注意事项 (pass-3 发现 2/3, 不阻塞, 实现时确认)

- **发现 2 (probe override 交互)**: 撤销计划基于请求 keep_count 计算 (step 2.5), 但 `fifo_probe` override (仅 oracle 收集路径) 可能保护不同数量 → 轻微 over/under-revoke。生产推理无 probe, 不受影响。over-revoke 仅让 pool 略小, 不破坏正确性。**接受此行为, 不额外处理**。
- **发现 3 (CUDA 确定性)**: `torch.argsort(stable=True)` 在 CUDA 上跨 run 的确定性需验证。实现时确认 `torch.use_deterministic_algorithms(True)` 下可复现; 若不确定, 对 keyframe_id tie 加显式次级 key (token 全局 index) 排序。这影响 paired-test 可复现性。

### 1.5 机制 A (TTL, 实验对照) 简述

仅作为 fallback 候选; 若 C/current/none 的对照已经足够决策, 则不实现 A:
- 新增 `protection_remaining [B,N]` 字段
- protect_topk 设 protection_remaining = ttl_keyframes
- 每帧 tick: protection_remaining -= 1; 归零 → anchor_slot = -1

---

## 2. P2 Eviction 评分设计

### 2.1 现状 (设计选择, 未改动)

`attention.py:238-252`: 老 token 用 cosine diversity, 新 token 用 repr_shift importance, 各自独立 min-max 归一化到 [0,1] 后拼接, `importance_weight=0.5` 加权。

### 2.2 待验证假设

老 token 的 `metadata.importance` 本就是 repr_shift_spatial (与新 token 同度量)。**假设**: 统一用 repr_shift 可能比双指标 + 独立归一化更公平。但这未经验证, 且当前混合方案达到 0.0258 ATE, 不能假设统一一定更好。

补充约束: `TokenScorer` / learned eviction 路线已被实验证实无收益, 本 spec 不再把它作为 P2 候选或主线对照。

### 2.3 实验设计 (heuristic ablation)

| 配置 | 评分方式 | 说明 |
|------|---------|------|
| P2-baseline | cosine(old) + repr_shift(new), 独立归一化 | 当前实现 |
| P2-baseline-weight-sweep | 同 baseline, sweep `importance_weight={0.3,0.5,0.7}` | 低成本检验 old/new 权重是否比结构重构更关键 |
| P2-unified-reprshift | 统一 repr_shift, 单一归一化 | 假设方向 |
| P2-unified-cosine | 统一 cosine diversity | 对照 |

**实现 (pass 2 修正发现 #3)**: 当前 `eviction()` 无 `scoring_mode` 参数, hybrid 评分硬编码 inline (`attention.py:238-252`)。这不是"参数切换", 是**结构性重构**:
- 需把 inline 的 hybrid 评分抽成独立函数 (按 scoring_mode 分派)
- eviction 签名需加 `scoring_mode` 参数, commit_pending_update_ 透传
- **unified-reprshift/cosine 需要所有 candidate 的指标**: 当前 `_current_frame_importance` 只返回当前帧 token 的 importance (`frame_id==current & anchor_slot<0`); unified 需所有 candidate 的 importance (老的用 `metadata.importance` 缓存值)。这改变了传给 eviction 的 importance_scores 形状 (从 [N_new] → [N_candidates])
- **unified-attention 不进入本轮实验**: SAGE-KV 的 attention 列累积基础设施 (`return_attn_col_sums` 等) 已在 `FRONTEND_LONGSEQ_OPTIMIZATION.md` "回退 ToMe 和 SAGE-KV 代码" 中移除, 且历史实验证明它在 3D ViT 上有害 (SAGE-KV ATE 0.0358→0.0467)。

**P3 不变量交互 (pass 2 发现)**: unified 评分改变 importance_scores 形状后, 需复查 P3 不变量 (candidate 尾部=当前帧) 是否仍适用。unified-reprshift/cosine 不依赖新/老分拆, 所以 P3 的位置假设不再相关 (所有 candidate 统一评分), 但 P6 invariant guard (检查尾部=当前帧) 需相应调整或禁用。实施时必须重验 P3/P6。

---

## 3. 统一实验计划 (P1 + P2)

### 3.0 数据可用性 (已审计 ✅)

7-Scenes 数据集 (`/path/to/mount/lyj/OpenDataLab___7-Scenes/raw/`) 实测: 4 个目标场景**所有序列均为 1000 帧** (chess 6 个 seq, fire 4, office 10, redkitchen 11)。500/1000 帧实验在 4 场景都能干净测量。原审查的 CRITICAL 担忧 (数据不存在) 已推翻。

### 3.1 实验矩阵

**P1 (保护机制)** × **P2 (评分)** 机制耦合, 不能假设正交。分两阶段 + 一阶段确认 factorial:

**阶段 1: 固定 P2=baseline, 扫 P1**
- P1-current (v1 max_protected_ratio) — 已实现, baseline
- P1-mechC (有界 FIFO rescued pool) — 本 spec 新增, **ratio 作为 swept 轴: {0.1, 0.2, 0.3, 0.5}** (避免 0.3 成为未测试魔法数)
- P1-none (fifo_keep_topk=0) — 下界
- ~~P1-mechA (TTL)~~ — **从 Stage 1 剔除** (避免实现负担; 仅当 mechC 全部 ratio 都没有优于 current/none, 才作为 fallback 实现)

**阶段 1 固定配置表 (保证可复现)**:

| 配置 | fifo_keep_topk | fifo_protected_ring_ratio | max_protected_ratio | learned_fifo_keep_count | intra_frame_dedup_enabled | budget_allocation | total_budget |
|------|----------------|---------------------------|---------------------|-------------------------|---------------------------|------------------|--------------|
| P1-none | 0 | 0.0 | 1.0 | False | False | dynamic | 200000 |
| P1-current (生产 fifo80) | 80 | 0.0 | **1.0** (cap 禁用) | False | False | dynamic | 200000 |
| P1-v1cap | 80 | 0.0 | **0.5** | False | False | dynamic | 200000 |
| P1-mechC-r0.1 | 80 | 0.1 | ignored/None | False | False | dynamic | 200000 |
| P1-mechC-r0.2 | 80 | 0.2 | ignored/None | False | False | dynamic | 200000 |
| P1-mechC-r0.3 | 80 | 0.3 | ignored/None | False | False | dynamic | 200000 |
| P1-mechC-r0.5 | 80 | 0.5 | ignored/None | False | False | dynamic | 200000 |

说明:
- **P1-current 是真正的生产 baseline**: `max_protected_ratio=1.0` (cap 禁用 = 原始 noIntra+fifo80 永久保护行为)。这才是与生产对齐、mechC 要超越的对象。
- **P1-v1cap 是单独对照**: `max_protected_ratio=0.5` 对齐 `tests/test_p1_protect_topk_budget_ceiling.py` 的 v1 修复语义 (硬截断)。单列一行, 不覆盖生产 baseline。
- P1-mechC 启用 ring 时 `max_protected_ratio` 不参与同一 keep_count 的二次裁剪; 实现中传 `max_protected=None`。
- 上表固定 `intra_frame_dedup_enabled=False`, 保持 noIntra+fifo80 长序列基线语境; 如需评估 dedup 交互, 单独开实验, 不混入 P1 主矩阵。

**阶段 2: 固定 P1=阶段1选定策略, 扫 P2**
- 先跑 P2-baseline 的 `importance_weight={0.3,0.5,0.7}` sweep
- 若 weight sweep 不能解释主要差异, 再跑 P2-unified-reprshift / P2-unified-cosine

**阶段 1.5: 确认 factorial (防 P1×P2 交互误判)**
- 阶段1 P1 选定策略 × 阶段2 P2 选定策略
- 阶段1 P1 选定策略 × P2-baseline
- P1-current × 阶段2 P2 选定策略
- 若 factorial 显示交互导致某组合灾难性退化, 回退分析

### 3.2 评估协议

- **场景**: chess / fire / office / redkitchen, 每场景用 seq-03 (1000 帧)
- **确定性前置检查 (pass-5 发现 3)**: paired 设计要求相同 seed 跨配置产生配对差异, 前提是管线确定性。**实验前先跑同一配置 2 次 (相同 seed)**, 若 ATE 不完全一致 → 管线非确定 (CUDA argsort 等), paired 退化为 unpaired 并增加 seeds 到 8。先验证 `torch.use_deterministic_algorithms(True)` 能否让两次运行 ATE 完全一致。
- **seeds**: 每配置 **5 seeds** (3 seeds 区分 ATE ~0.001 差异功效不足), 确定性验证通过后采用 **paired 设计** (相同 seed 跨配置, 差分配对, 大幅降方差); 非确定则 unpaired + 8 seeds
- **帧数**: ATE 曲线测 **50/200/500/1000 帧** 四档 (数据支持到 1000)

**关键指标** (精确定义):
1. **ATE 曲线** (50/200/500/1000 帧) — 主指标, 报告均值 ± 标准差
2. **每帧 protected_count + rescued_pool_count 曲线** — `protected_count` 用于确认总保护规模不挤占 budget; `rescued_pool_count = count(anchor_slot==0 and keyframe_id!=global_anchor_kf_id)` 用于量化 C 机制是否按设计受容量约束
3. **slot-0 token 的 keyframe_id 刷新率** — 操作定义: 每帧统计 slot-0 中 **keyframe_id != global_anchor_kf_id** 的 token 数及其 id 集合; 这是 C 机制的行为诊断指标, 不是预设成功条件
4. **eviction 淘汰来源** — 每次 eviction 中, 淘汰的 token 来自 (刚撤销的保护 / 老候选 / 新候选) 的占比
5. **anchor_overflow_count/rate** — 统计 `cache_budget < protected_count` 或 `eviction()` 进入 anchor overflow 分支的次数; 若非 0, 说明稳定 per_layer_budget ring cap 与动态 layer budget 存在冲突, 需跑保守 cap 对照

**判定标准**:
- P1: 以 1000 帧 ATE 为主指标; 若 C 没有显著优于 P1-current/P1-none, 不采纳轮换机制。行为指标只用于解释结果, 不单独作为成功条件。若 `anchor_overflow_rate>0`, 当前配置不得直接采纳, 需先评估保守 cap 对照。
- P2: 4 场景平均 ATE 在 paired-test 下显著优于 baseline (p<0.05) 才采纳

### 3.3 混淆处理

阶段 1 固定 P2, 阶段 2 固定 P1, 阶段 1.5 确认 factorial 捕获交互。

---

## 4. 实施计划 (待 spec 批准后转 writing-plans)

### Stage 0: 实现 P1 候选机制 C (TDD)
1. RED: 写测试矩阵 (枚举见下)
2. GREEN: 实现 `fifo_protected_ring_ratio` + 最老 keyframe 撤销逻辑 (含 probe 原始观察语义/显式 global anchor id)
3. 回归: 现有 23 frontend 测试 + 新增 P1-mechC 测试

**Stage 0 测试用例 (枚举)**:
- multi-swap-ordering: 多次 FIFO swap, rescued pool 按最老 keyframe 撤销
- global-anchor-never-revoked: global anchor token 永不被撤销 (用显式 global_anchor_kf_id 从 keyframe_managers[b] 取)
- exactly-at-cap: rescued_pool_count 恰好等于 ring_capacity 时无操作
- over-cap-partial-revoke: 超容时部分撤销最老
- over-cap-when-rotatable-empty: rescued pool 为空但 keep_count 大于容量时 fallback clamp
- probe-path-includes-rotation: fifo_probe 触发 override 时撤销仍执行, 且 probe 看到 mutation 前的 demoted-slot 候选集; 不断言精确 pool size, 因 override 可能改变实际保护数
- tie-break-determinism: 同 keyframe_id 的 token 撤销顺序确定性 (argsort stable, 跨 seed 可复现)
- b-per-batch: B>1 时 per-batch 处理 (不 assert, 验证不误报)
- reorder-after-revoke: 撤销后 reorder_by_anchor_slots_ 强制触发 (dataclass field flag)
- ring-only-does-not-protect-all: 只设 ring_ratio>0 且 fifo_keep_topk=0/learned disabled 时, 不触发全量保护
- ring-cap-excludes-global-anchor: global anchor 不计入 rescued pool capacity
- wiring-no-nameerror: 接线用 keyframe_managers[b] (非单数), global_anchor id 正确取

### Stage 1: P1 实验 (阶段 1)
- 跑 P1-current / P1-mechC(ratio×4) / P1-none × 4 场景 × 5 seeds (paired)
- 分析 ATE(50/200/500/1000) + protected_count/rescued_pool_count + slot-0 keyframe_id 刷新率 + anchor_overflow_count/rate
- 选出 P1 策略 + ratio; 若 C 无显著收益, 保留 current 或 none

### Stage 1.5: 确认 factorial
- 阶段1 P1 选定策略 × 阶段2 P2 候选 (实验 2 后)

### Stage 2: P2 实验 (阶段 2)
- 固定 P1 选定策略, 先跑 baseline `importance_weight={0.3,0.5,0.7}` × 4 场景 × 5 seeds (paired)
- 如仍有必要, 再跑 P2 **2 个结构重构配置** (unified-reprshift / unified-cosine; unified-attention 剔除, SAGE-KV 已回退且历史证明有害) × 4 场景 × 5 seeds (paired)
- 选出 P2 评分 (或确认 baseline 最优)

### Stage 3: 实施选定方案 + 最终验证
- 实施实验选定的 P1 + P2 配置
- 全量回归 + 长序列验证

---

## 5. 风险与诚实保留

1. **机制 C 的激进度**: 撤销最老 keyframe 保护时, 那批 token 瞬间变 candidate 可能被立即淘汰。若这些 token 对近期重建仍有用, C 比 TTL 给固定缓冲更激进。实验揭示此 tradeoff。
2. **ratio 的取值**: sweep {0.1,0.2,0.3,0.5} 覆盖; 不同场景可能需要不同 ratio。
3. **P2 unified-reprshift 可能更差**: 老 token 的 repr_shift 是插入时算的, 可能已过时 (token 内容随 attention 演化但 importance 未更新)。cosine diversity 可能正是补偿了这一点。实验定论。
4. ~~**长序列数据可用性**~~: ✅ 已审计, 7-Scenes 全为 1000 帧, 数据充足。
5. **B 机制 (软评分) 的回归风险**: 本 spec 明确暂缓 B; 仅当 current/none/C 都不能解释或改善长序列表现时才重新评估 TTL/软评分, 届时需重做 P3 不变量验证。
6. **P1×P2 交互**: 阶段 1.5 factorial 捕获, 防止 OFAT 误判。

---

## 6. 待决问题 (已确认)

1. **global anchor token 是否参与 rescued pool 撤销?** ✅ **不参与** (它是坐标基准; 用显式 `global_anchor_keyframe_id` 标识, 非 keyframe_id.min() 推断)。
2. **撤销的最小粒度**: ✅ **按 token 逐个撤销** (与现有 topk 粒度一致; 实验后看是否需调整为整批)。
3. **保护区上限取值**: ✅ **总容量 (per_layer_budget) 的比例**, 但只约束非 global 的 rescued pool; sweep {0.1,0.2,0.3,0.5}。字段 `fifo_protected_ring_ratio`, 默认 0.0 (禁用, 向后兼容)。
4. **v1 max_protected_ratio 处置**: ✅ **与 ring 互斥** (ring 启用时 max_protected=None), 避免两者顺序作用同一 keep_count 混淆。

---

## 附录: Spec Review 发现处置 (2026-06-14 adversarial review)

4 视角审查 → 24 原始发现 → 代码验证保留 15 → 修订后处置:

| # | 严重度 | 发现 | 处置 |
|---|--------|------|------|
| 1 | ~~CRITICAL~~ | 7-Scenes 无 500+ 帧 | ❌ **推翻** — 实测全为 1000 帧 (见 3.0) |
| 2 | HIGH | probe 早退绕过撤销 | ✅ 修订 — probe 前计算 revoke plan, probe 后应用 mutation |
| 3 | HIGH | revoke 后 reorder 条件 | ✅ 修订 — `_needs_reorder_after_revoke` 强制 reorder |
| 4 | HIGH | 30% 魔法数未测试 | ✅ 修订 — ratio sweep {0.1,0.2,0.3,0.5} |
| 5 | HIGH | 两阶段漏 P1×P2 交互 | ✅ 修订 — 阶段 1.5 确认 factorial |
| 6 | HIGH | mechA 矛盾 | ✅ 修订 — 从 Stage 1 剔除, 作 fallback |
| 7 | HIGH | global anchor 用 min() 太脆 | ✅ 修订 — 显式 `global_anchor_keyframe_id` |
| 8 | HIGH | 统计功效 (3 seeds) | ✅ 修订 — 5 seeds + paired 设计 |
| 9 | MED | v1/v2 优先级 | ✅ 修订 — 互斥 (ring 启用时 max_protected=None) |
| 10 | MED | "无可撤销"边界 | ✅ 修订 — fallback clamp keep_count |
| 11 | MED | 测试计划太笼统 | ✅ 修订 — Stage 0 枚举测试用例 |
| 12 | MED | slot-0 keyframe_id 指标定义 | ✅ 修订 — 3.2 操作定义 |
| 13 | LOW | topk k=0 退化 | ✅ 修订 — `if rot_idx.numel() > 0` 守卫 + fallback |
| 14 | LOW | global anchor 识别 (重复 #7) | ✅ 同 #7 |
| 15 | HIGH | B>1 hardcoded [0] | ✅ 修订 — per-batch 循环 (不 assert) |

所有 14 个有效发现已修订入 spec。1 个 CRITICAL 经实测推翻 (数据充足)。

### Pass 2 (修订后复审, 2026-06-14) — 5 核心发现全部 CONFIRMED

Pass 2 verify 阶段因 StructuredOutput 故障失效 (23 个 verify agent 均未返回), 但 review 阶段原始发现经**手动对照代码验证**, 5 个实现性硬伤全部 CONFIRMED:

| # | 严重度 | 发现 | 代码验证 | 处置 |
|---|--------|------|---------|------|
| P2-1 | CRITICAL | ring-only 触发会绕过 top-K 质量筛选并把 demoted slot 全量保护 | ✅ 当前门控只含 fifo_keep_topk/learned; 不应为 ring 单独加触发条件 | ✅ 重新修订 — ring_ratio 不是保护触发条件, 只约束已有 top-K/learned 保护请求; ring-only 不全量保护 |
| P2-2 | CRITICAL | unified-attention 不可实现 (SAGE-KV 已回退) | ✅ LONGSEQ doc "回退 ToMe 和 SAGE-KV 代码" | ✅ 修订 — 从 P2 实验剔除 unified-attention |
| P2-3 | HIGH | 接线用单数 `keyframe_manager` (NameError) | ✅ grep 单数返回空; 调用点是 `keyframe_managers[b]` | ✅ 修订 — 接线改 `keyframe_managers[b]` |
| P2-4 | HIGH | eviction 无 scoring_mode, P2 是结构性重构非参数切换 | ✅ `eviction()` 签名确认无 scoring_mode; hybrid 硬编码 inline | ✅ 修订 — 2.3 标注重构性质 + importance_scores 形状变化 + P3 重验 |
| P2-5 | HIGH | assert B==1 会误报 (代码已支持 B>1) | ✅ `# B>=1 is now supported; no guard needed.` | ✅ 修订 — 改 per-batch 循环 |
| P2-6 | HIGH | per_layer_budget 是平均, 实际动态 | ✅ `_calculate_budgets` 产生 per-layer 动态 | ✅ 修订 — ring_capacity 明确用 per_layer_budget 作长期基准 |
| P2-7 | HIGH | topk tie 非确定性破坏 paired-test | ✅ 同 keyframe token id 相同 | ✅ 修订 — argsort(stable=True) 替代 topk |
| P2-8 | HIGH | `_needs_reorder_after_revoke` 未声明 dataclass field | ✅ LayerCacheState field 枚举确认无 | ✅ 修订 — 声明为 dataclass field |

Pass 2 价值: 聚焦"代码能否真按 spec 写出来", 挖出 pass 1 (设计/实验向) 漏掉的实现性硬伤。修订后 spec 接线代码经对照验证可编译/可触发。

### Pass 3 (用户精修后复审, 2026-06-14) — 边界 case 鲁棒性

用户在 pass 2 基础上精修了关键语义 (ring 非触发条件、pool 容量排除 global anchor、撤销计划 probe 前算/probe 后用), 消除了 pass-2 修订中"ring 当触发条件会绕过 top-K"的风险。pass 3 对精修后的新逻辑做聚焦审查, 发现 3 个边界 case (比前两轮小得多):

| # | 严重度 | 发现 | 处置 |
|---|--------|------|------|
| P3-1 | MED-LOW | overflow>rot_idx.numel()>0 时 keep_count 未 clamp → pool 超容 (else 分支 clamp 位置错) | ✅ 修订 — clamp 移出 else, 按"撤销后剩余容量"统一 clamp |
| P3-2 | LOW-MED | 撤销计划基于请求 keep_count, probe override 改变实际保护数 → 过/欠撤销 | ✅ 文档化 — 仅 oracle 路径, over-revoke 不破坏正确性, 接受 |
| P3-3 | LOW | CUDA 上 argsort(stable=True) 确定性未验证 | ✅ 文档化 — 实现时确认, 必要时加 token-index 次级 key |

Pass 3 价值: 精修后的逻辑已扎实, 主要剩边界鲁棒性微调。进入实施前仍需补齐 pass 4 的 B>1、dynamic budget、实验配置可复现性约束。

### Pass 4 (继续复审, 2026-06-14) — 实施前约束补齐

| # | 严重度 | 发现 | 处置 |
|---|--------|------|------|
| P4-1 | MED | B>1 时共享 scalar `keep_count` 被 batch 0 clamp 后会污染 batch 1 | ✅ 修订 — 伪代码新增 `keep_count_by_batch`, 每个 batch 独立计算 requested/effective keep_count |
| P4-2 | MED | ring cap 用稳定 per_layer_budget, 但 eviction 用动态 layer budget, 可能仍触发 anchor overflow | ✅ 修订 — 指标新增 `anchor_overflow_count/rate`; 若非 0, 增加 `ratio * min(per_layer_budget, current_layer_budget)` 保守 cap 对照 |
| P4-3 | MED | Stage 1 配置不够可复现, 未固定 fifo_keep_topk/max_protected_ratio/dedup/budget 等关键参数 | ✅ 修订 — 新增阶段 1 固定配置表 |
| P4-4 | LOW-MED | probe override 接受 over/under-revoke, 但测试若断言精确 pool size 会误判 | ✅ 修订 — probe 测试只断言 revoke 未被早退绕过和 probe 看到 mutation 前候选集 |
| P4-5 | LOW | Pass 3 结论“可进入实施”过强 | ✅ 修订 — 降级为“核心方向可实施前需补齐 pass 4 约束” |

Pass 4 后状态: spec 的核心候选机制仍可作为实验对象, 但实施计划必须按新增 per-batch keep_count、anchor overflow 记录和固定配置表落地。

### Pass 5 (用户精修后复审, 2026-06-14, 手动审查) — 接线/对照/一致性

Pass 5 的 workflow verify 因 StructuredOutput 系统性故障失效, 改为手动对照代码审查。4 个 low-medium 发现, 全部修订:

| # | 严重度 | 发现 | 代码证据 | 处置 |
|---|--------|------|---------|------|
| P5-1 | MED | `keep_count_by_batch` 在 step 2.5 计算但 step 4 只注释未显式接线 | `frontend_cache.py:551` step-4 用单一 `keep_count`, 非 per-batch | ✅ 修订 — step 4 注释改为显式接线指令 (改 551 行用 `keep_count_by_batch.get(b_idx, keep_count)`) |
| P5-2 | LOW | P1-current baseline 设 `max_protected_ratio=0.5` 与生产不符 (生产是 cap 禁用) | 生产 fifo80 无 v1 cap 字段; 0.5 是本轮新加的 v1 修复 | ✅ 修订 — P1-current 改 `max_protected_ratio=1.0` (真生产 baseline); 0.5 单列为 P1-v1cap 对照 |
| P5-3 | LOW | paired-test 需确定性, 但 CUDA argsort 确定性未验证 (内部矛盾) | 实现注意事项发现 3 承认未验证 | ✅ 修订 — 3.2 新增确定性前置检查 (同 seed 跑 2 次, 不一致则 paired→unpaired+8seeds) |
| P5-4 | LOW | v1/v2 互斥仅文档化, 无 config 层强制 | 依赖调用方传 None, 用户误配无报错 | ✅ 修订 — `FrontendCacheConfig.__post_init__` 加互斥断言 |

Pass 5 价值: 补齐接线显式化、baseline 对齐生产、确定性前置、互斥强制。spec 经 5 轮审查 (设计→实现→边界→精修→接线/对照), 无遗留设计/实现性硬伤。
