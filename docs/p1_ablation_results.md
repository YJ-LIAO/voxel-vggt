# P1 保护策略消融实验结果报告

> 日期: 2026-06-15
> 关联 spec: `docs/superpowers/specs/2026-06-14-p1-protection-rotation-p2-scoring-design.md` (Stage 1)
> 数据: 420 runs (7 configs × 4 scenes × 5 seeds × {200,500,1000}f), 完整
> 结论先行: **生产 `fifo_keep_topk=80` 保护机制是长序列性能灾难的根因；应采纳 `fifo_keep_topk=0`（关闭保护）。**

---

## 1. 执行摘要

对 OVGGT 前端 KV cache 的 `fifo_keep_topk` 保护机制做了完整消融。**结论明确且经统计验证**：

1. **生产 baseline（`fifo_keep_topk=80`）在 ≥500 帧时灾难性退化**：ATE 从 200f 的 0.035m 飙升到 500f 的 0.524m、1000f 的 0.648m —— 比关闭保护差 **7×（500f）/ 3.7×（1000f）**，统计上极显著（paired t-test p < 1e-13，win%=100%）。
2. **根因是设计的无界累积，非代码 bug**：保护 token 一旦进入 slot 0 即永久驻留、永不淘汰，持续累积（~80/帧交换）直至挤满甚至超过 KV budget。直接测量证实：500f 时 protected_count 达 8084（最差层 2.42× budget），17% 的帧触发 anchor overflow（当前帧 token 被全部淘汰）。P5 transform 修复有效，问题在机制本身。
3. **v1 cap（`max_protected_ratio=0.5`）实测无效**：它只限单次保护请求、不限累积池，overflow 与生产等同（18%），ATE 与生产同灾难。**spec "v1 cap 有缺陷" 的前提被实测证实。**
4. **`P1-none`（关保护）与 `P1-mechC-r0.1`（ring 10%）统计打平且均极显著优于生产**；按 spec §3.2 判定（"C 不显著优于 current/none 则不采纳轮换"）→ **采纳 `P1-none`**（最简），r0.1 等价但增加复杂度、无测得收益。

---

## 2. 实验设置

### 2.1 配置矩阵（固定 P2=baseline，sweep P1）

| 配置 | fifo_keep_topk | fifo_protected_ring_ratio | max_protected_ratio |
|------|---------------|--------------------------|---------------------|
| P1-none | 0 | 0.0 | 1.0 |
| P1-current (生产) | 80 | 0.0 | 1.0 (cap 禁用) |
| P1-v1cap | 80 | 0.0 | 0.5 |
| P1-mechC-r0.1 | 80 | 0.1 | 1.0 |
| P1-mechC-r0.2 | 80 | 0.2 | 1.0 |
| P1-mechC-r0.3 | 80 | 0.3 | 1.0 |
| P1-mechC-r0.5 | 80 | 0.5 | 1.0 |

所有配置共享: `per_layer_budget=8000`, `dedup_enabled=True`, `intra_frame_dedup_enabled=False`, `budget_allocation='dynamic'`, `learned_fifo_keep_count=False`。ring_capacity = ratio × 8000（r0.1→800, r0.5→4000）。

### 2.2 协议

- **数据**: 7-Scenes, chess/fire/office/redkitchen 各 seq-03（均 1000 帧）。
- **帧数**: 200 / 500 / 1000。
- **seeds**: 0–4（5 seeds），**paired 设计**：相同 (scene, seed) 跨配置配对差分。
- **确定性前置检查（spec §3.2）**：同 seed 跑 2 次，**ATE bit-identical** → 管线确定性成立，paired 设计有效。
- **主指标**: Sim3-aligned ATE RMSE（米），与 `eval_7scenes_compare.py` 同算法。

### 2.3 工具

- `tools/run_p1_ablation.py` — 单 run harness，显式构造 FrontendCacheConfig。
- `tools/batch_p1_ablation.sh` — 2 GPU 并行 + 断点续跑。
- `tools/cache_diag_probe.py` + `tools/run_cache_diag.py` — 只读探针（复用既有 `_oracle_eviction_probe` 钩子，零 ATE 扰动），直接测量 protected_count / overflow。
- `tools/analyze_p1_ablation.py`（描述统计）+ `tools/analyze_p1_ablation_stats.py`（paired t-test + 95% CI）。

---

## 3. 结果：ATE 表

ATE RMSE (m), mean ± std over 4 scenes × 5 seeds (n=20)。**越低越好。**

| config | @200f | @500f | @1000f |
|--------|-------|-------|--------|
| P1-none | 0.0356 ± 0.0070 | **0.0678 ± 0.0264** | **0.1767 ± 0.1174** |
| P1-current (生产) | 0.0349 ± 0.0072 | **0.5237 ± 0.1172** | **0.6476 ± 0.0630** |
| P1-v1cap | 0.0349 ± 0.0072 | 0.5422 ± 0.1168 | 0.6341 ± 0.0848 |
| P1-mechC-r0.1 | 0.0354 ± 0.0077 | **0.0700 ± 0.0342** | **0.1695 ± 0.1084** |
| P1-mechC-r0.2 | 0.0348 ± 0.0072 | 0.2937 ± 0.2537 | 0.4536 ± 0.2414 |
| P1-mechC-r0.3 | 0.0349 ± 0.0072 | 0.5142 ± 0.1230 | 0.6330 ± 0.0543 |
| P1-mechC-r0.5 | 0.0349 ± 0.0072 | 0.5237 ± 0.1172 | 0.6392 ± 0.0870 |

**观察**:
- **200f** 全部 ≈0.035，无差异（序列太短，保护池未填满——这正是 smoke test 无法区分的原因）。
- **500f+** 出现明确的阈值效应：保护池越小越好。none/r0.1 ≈ 0.07；r0.2 = 0.29（中间）；r0.3/r0.5/current ≈ 0.52（灾难）。
- **r0.5 ≡ current 完全相同**（ATE 0.5237 = 0.5237）：ring 容量 4000 在这些帧数下从未被触发 → ring 失效 → 行为退化成生产。实测两 run 时间不同（293s vs 297s）确认是两个独立 run，非 bug。

---

## 4. 统计显著性（paired t-test vs P1-current, n=20）

diff = config − current（负 = 优于生产）。

### @500f
| config | mean diff | 95% CI | p | win% |
|--------|-----------|--------|---|------|
| P1-none | **−0.456** | [−0.504, −0.408] | 1.1e-13 *** | 100% |
| P1-mechC-r0.1 | **−0.454** | [−0.501, −0.407] | 9.2e-14 *** | 100% |
| P1-mechC-r0.2 | −0.230 | [−0.335, −0.125] | 3.9e-04 *** | 100% |
| P1-mechC-r0.3 | −0.0095 | [−0.012, −0.007] | 9.4e-07 *** | 100% |
| P1-v1cap | +0.0185 | [+0.008, +0.029] | 2.3e-03 | 25% |
| P1-mechC-r0.5 | −0.0000 | ≈0 | 2.1e-02 | 25% |

### @1000f
| config | mean diff | 95% CI | p | win% |
|--------|-----------|--------|---|------|
| P1-none | **−0.471** | [−0.504, −0.437] | 9.2e-17 *** | 100% |
| P1-mechC-r0.1 | **−0.478** | [−0.506, −0.450] | 2.1e-18 *** | 100% |
| P1-mechC-r0.2 | −0.194 | [−0.277, −0.111] | 2.0e-04 *** | 100% |
| P1-v1cap | −0.0135 | [−0.024, −0.003] | 2.3e-02 | 75% |
| P1-mechC-r0.3 | −0.0146 | [−0.030, +0.001] | 8.5e-02 | 75% |
| P1-mechC-r0.5 | −0.0050 | [−0.017, +0.007] | 4.4e-01 | 28% |

**直接 paired（P1-none vs P1-mechC-r0.1，两个并列最优）**: 200f/500f/1000f 均 **p > 0.07，无显著差异** → 两者打平。

---

## 5. 机制：直接测量（500f, chess/seq-03, seed 0）

只读探针测量每个 eviction 事件的 protected_count、budget、overflow（定义：`protected_count ≥ 该层 budget` → 当前帧 token 被全部淘汰）。

| config | ATE | max protected | 最差层 ratio | overflow 帧 | overflow 率 |
|--------|-----|--------------|-------------|------------|------------|
| P1-current | 0.458 | **8084** | **2.42×** | 84 | **17.0%** |
| P1-v1cap | 0.514 | 7123 | **2.42×** | 89 | **18.1%** |
| P1-mechC-r0.1 | 0.068 | 4964 | 1.64× | 18 | **3.7%** |
| P1-none | 0.065 | 4164 | 1.34× | 4 | 0.8% |

**机制链（全部直接测量，非推断）**:

1. **`protect_topk` 累积无界**。protected_count 单调增长 ~80/帧交换：current 在 500f 达 **8084**（=budget 1.01×），单层最高 **2.42× budget**。因为一旦 `anchor_slot=0` 就永久驻留、永不淘汰。
2. **挤占 budget → 当前帧 token 全部淘汰**。eviction 保留 `[0:budget]`，其中 `[0:protected_count]` 被锁；当 `protected_count ≥ budget` 时，**当前帧候选 token 零存活** → 当前帧几何丢失 → 位姿漂移 → 灾难 ATE。current/v1cap 17–18% 帧发生此情况，与 7× ATE 退化相关。
3. **v1 cap 失效**：`max_protected_ratio=0.5` 只把峰值 protected 从 8084 降到 7123（cap 确实触发了），但**最差层 ratio 与 overflow 率与 current 完全一致（2.42× / 18%）**——cap 限单次请求、不限累积池。
4. **mechC ring 有效**：r0.1 把保护池约束在 4964，ratio 1.64×，overflow **3.7%**（vs current 17%），ATE 0.068 ≈ P1-none。
5. **可复现性**：诊断 run 的 ATE 与消融 harness **bit-identical**（current 500f s0 两处均 0.4579119）→ 只读探针零扰动。

---

## 6. 结论与措施建议

### 判定（按 spec §3.2）
- 1000 帧 ATE 为主指标。
- **C（mechC）不显著优于 P1-none**（直接 paired p > 0.07）→ **不采纳轮换机制**。
- 保护（current/v1cap/r0.3/r0.5）在长序列极显著劣于 none/r0.1。

### 措施
| 优先级 | 措施 | 依据 |
|--------|------|------|
| **采纳** | 生产 `fifo_keep_topk: 80 → 0` | none 在所有帧数统计最优或打平；保护无任何测得收益，500f/1000f 灾难 |
| 不采纳 | P1-mechC-r0.1（ring） | 与 none 统计打平，增加复杂度无收益 |
| 移除/搁置 | v1 cap (`max_protected_ratio=0.5`) | 实测无效（overflow 18% ≡ current） |

### 边界备注（诚实保留）
- 本结论基于 **ATE 单指标 + 4 场景 + 7-Scenes 数据 + budget=8000**。机制（无界累积）对任意 `fifo_keep_topk>0` 都成立，只是随 keep_topk 变小而延迟爆发；`=0` 是唯一根治。
- 1000f 直接 paired（none vs r0.1）有微弱趋势（mean r0.1−none = −0.0072, p=0.079）——**未达显著**，但若需验证"极小有界池在超长序列是否略优"，可后续增大 seeds（n=20→40）复测。

---

## 7. 数据与可复现

- 原始 run: `tools/p1_ablation_results_lyj/*_f{200,500,1000}_s{0..4}_full.json`（420 个）。
- 诊断: `tools/p1_ablation_results_lyj/diag_*_f500_*.json`。
- 复现全矩阵: `bash tools/batch_p1_ablation.sh full`（断点续跑）。
- 复现分析: `python tools/analyze_p1_ablation.py` / `python tools/analyze_p1_ablation_stats.py`。
- 复现机制诊断: `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:tools python tools/run_cache_diag.py --config P1-current --scene chess/seq-03 --num_frames 500`。
