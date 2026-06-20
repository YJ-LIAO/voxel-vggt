# Soft-merge intra dedup — 负面结果记录（Gate 1 失败）

> 日期: 2026-06-20
> 状态: **实验失败，假设被实证推翻**（soft-merge 比 drop 全面更差）
> 关联 spec: `docs/superpowers/specs/2026-06-19-frontend-shortseq-pareto-optimization-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-20-frontend-shortseq-pareto-optimization.md`（Gate 1）
> 实现提交: `b4d8d4a`（config 字段）+ `45b8fa7`（merge plan + `_apply_intra_merge_`）
> 数据: `tools/legacy_vs_frontend_lyj/frontendmerge_*_f200_s0.json`（4 场景）

---

## 1. 假设（被推翻）

frontend 在 200f 输 legacy 的根因之一是 **fire 的 intra-frame voxel dedup 把多视角有用信息当冗余硬丢弃**（RCA 确认：intra-OFF 让 fire 0.0458→0.0279）。

**假设**：把 intra dedup 的"硬丢弃"（keep best-score per voxel, discard rest）改成 **soft-merge**（co-voxel tokens 的 K/V 做 importance 加权平均），就能**既保留多视角信息（帮 fire）又仍去冗余（不害 office/chess）** —— 一个 Pareto 改善。

## 2. 实现（正确，已 TDD 验证）

- `_dedup_single_batch` 返回 merge plan（per 多成员 voxel group: rep + members + softmax weights + member_rep_map）。
- `_apply_intra_merge_`：把成员的 K/V/importance/depth_conf 加权平均写入 rep 槽位（slot_local_xyz 取均值，score_state 加权），在 gather **之前**应用。token **数量不变**（= unique voxel 数），仅**值**不同。
- 6 个 TDD 测试全绿（plan 构造 + 加权数学）；29 个前端测试全绿（drop 默认不回退）。

## 3. Gate 1 结果（frontend @200f, seed 0, ATE RMSE m）

| scene | drop（当前生产）| **merge** | merge vs drop |
|-------|----------------|-----------|---------------|
| fire | 0.0458 | 0.0531 | **+0.0073 更差** |
| office | 0.0377 | 0.0466 | **+0.0089 更差** |
| **chess** | 0.0263 | **0.5002** | **+0.474 灾难（20×）** |
| redkitchen | 0.0138 | 0.0445 | **+0.031 更差（3×）** |

**merge 在全部 4 个场景都比 drop 差**，chess 灾难性崩溃（0.5 = 跟踪发散）。Gate 1 判定标准（fire < 0.035 且 office/chess/redkitchen ±0.003）**全部不满足** → **Gate 1 FAIL，soft-merge 否决**。

## 4. 排除 bug（确认是"方法本身有害"，非实现错误）

- **merge plan 构造正确**：多组实测 reps `[1,4]`、members 映射正确、weights = within-group softmax（sum=1）。
- **加权数学正确**：单测 `test_apply_intra_merge_weighted_avg` 验证 rep K = Σ w·K（atol 1e-5）。
- **chess 0.5 是真值退化**（非 NaN，fps 正常 1.8）—— averaging K/V 把棋盘高频空间信息糊掉，稠密/精密场景崩溃。

结论：实现无误，**averaging co-voxel tokens 的 K/V 本身有害**（模糊表示），soft-merge 假设被推翻。

## 5. 更深层结论：200f deficit 场景冲突，单配置"全面超越"不可达

RCA + Gate 1 共同揭示 200f deficit 的**场景冲突**：
- **fire** 需要**少 dedup**（intra-OFF 0.0458→0.0279 才好；merge/off 都比 ON-drop 差或持平）
- **office / chess / redkitchen** 需要**多 dedup**（intra-ON drop 最优；OFF 更差、merge 更差）

fire 与 office/chess 对 dedup 的需求**相反**。在用户约束（**单一固定配置，无运行时场景判断**）下，**无法同时优化两者** → "全面超越 legacy at 200f（所有场景）"在该约束下**不可达**。

| intra 设置 | fire | office/chess | 结论 |
|-----------|------|--------------|------|
| OFF（不 dedup）| ✅ 0.0279 最好 | ❌ 更差 | 帮 fire 害其他 |
| drop（当前）| 中 0.0458 | ✅ 最优 | 折中，生产采用 |
| merge（本实验）| ❌ 0.0531 | ❌❌ chess 0.5 | 全面更差，否决 |

## 6. 处置与建议

- **soft-merge 代码保留为 opt-in**（`intra_dedup_mode="merge"`，默认 `"drop"`，不影响生产）。保留以便复现/未来参考，不删。
- **当前生产配置不变**：ring0.2 + budget8334 + intra_frame_dedup_enabled=True + intra_dedup_mode=drop。
- **不再追求"全面超越 200f"**：单配置约束下达不成（场景冲突）。1000f frontend −26% 胜 legacy 是 streaming 的真正价值，保持。
- 若未来要"全面超越"，唯一路径是**放宽约束**（运行时场景自适应：fire 关 dedup、office/chess 开），但工程复杂 + 过拟合风险，不建议除非有强需求。

## 7. 教训（避免重复）
1. **averaging K/V 不是"保留信息"** —— 它模糊表示，对稠密/高频场景（chess）灾难性。ToMe 式按特征相似度合并 ≠ 按空间 voxel 合并。
2. **场景冲突的 deficit 无法用单配置修** —— fire vs office/chess 的几何差异决定了相反的最优。诊断（RCA）应早期识别这种冲突，避免在不可达目标上投入。
3. **负面结果要留档**（本文档），避免日后重复尝试 soft-merge。

## 8. log(n) bias 假设的证伪（SOTA 研究后追加实验，2026-06-20）

SOTA 研究（Co-Me, [arxiv 2511.14751](https://arxiv.org/abs/2511.14751)）提出：加权平均 K/V 后 merged token 在 softmax attention 里系统性欠权重 → 加 **log(n) attention bias** 恢复 mass 即可修复。

**实验**：保留 rep 的原始 K（不平均 K，只平均 V）—— 这等价于"无 attention mass loss"（rep K 全 distinctiveness）。若 log(n)/attention-mass 是根因，chess 应恢复到 ~drop。

**结果**：chess = **0.1543**（vs full merge 0.5002，drop 0.0263）。

**结论**：K-preserve（消除 attention mass loss）只把 chess 从 0.5 改善到 0.15，**仍比 drop 差 6×**。剩余 gap 纯来自 **V-averaging**（信息混合），log(n) attention bias **不解决 V 侧**。所以：

- **log(n) bias 不是解药** —— V-averaging 才是 chess 灾难的根因，不是 attention mass loss。
- Co-Me 真正有效的是 **低置信度 gating**（不合并高精度 token），不是 log(n) bias。
- 即使用 importance 作 gating proxy（只合并低 importance token），固定阈值无法同时满足 fire（要少合并）和 chess（要保护高 importance）—— **同样的场景冲突**。

**最终判定**：token averaging（任何形式：K+V / V-only）对稠密精确场景有害；"全面超越 legacy at 200f"经 3 轮实验（full merge / K-preserve / drop vs OFF）**确认在单配置约束下不可达**。merge 方向彻底关闭。
