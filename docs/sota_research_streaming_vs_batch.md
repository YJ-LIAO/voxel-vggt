# SOTA Research: 流式 ViT 如何全面超越 batch baseline（2024-2026 文献综述）

> 日期: 2026-06-20
> 来源: deep-research workflow（102 agents, 79 claims → 25 verified → 17 confirmed, 8 killed）
> 目的: 为"全面超越 legacy"找文献支撑的方向（soft-merge 失败后的下一步）

---

## 核心结论（诚实）

**没有任何已验证的方法能在短序列上证明流式 ViT 匹配 batch full-attention（针对 3D pose）。** 所有"matches full attention/cache"的标题（PyramidKV 12% retention、ZipVL 0.5% drop、SnapKV）都是 **LLM 文本长上下文**或 **batch VGGT** 结果，不是流式 ViT 短序列。短序列冗余少，可压缩空间小 —— 这正是我们的症状。**但研究给出了 3 个高价值、可测试的机制方向。**

---

## 关键发现（按对我们问题的直接相关性排序）

### 1. ⭐ Co-Me 的 log(n) attention-bias 校正 —— 直接修复我们 soft-merge 的失败模式
**Co-Me (CMU/Field AI, 2025, [arxiv 2511.14751](https://arxiv.org/abs/2511.14751))**，几何 transformer token merging：
- **关键洞察**：加权平均 K/V 后，merged token 在 softmax attention 里**系统性欠权重**（averaging 降低了 distinctiveness）→ 信息丢失。Co-Me 加一个 **`log(n)` bias 到 merged token 的 attention logit**，数学上恢复原始 n 个 token 的总 attention mass。
- **这直接解释了我们的 chess 0.5002 灾难**：merged rep token 被 attention 低估 → 关键几何信息丢失 → 跟踪发散。log(n) 校正正是缺失的那一块。
- Co-Me 还**只 merge 低置信度 token**（蒸馏的不确定性预测器），高置信度 token 保留 —— 避免在精确区域（chess 棋盘）合并。
- **诚实限定**：confidence predictor 需蒸馏（~1 GPU-hour，self-supervised，backbone 冻结）—— 不是纯 training-free；且 Co-Me 在细/高频结构上有 stated failure mode（Fig 13）。

### 2. ATM Theorem 1 —— 严格解释为何我们的 soft-merge 有害
**ATM (ECAI 2024, [arxiv 2505.15160](https://arxiv.org/abs/2505.15160))**：
- **定理**：merge error = Θ((n_i·n_j)/(n_i+n_j) · δ(X_i,X_j))，其中 n 是"merging size"（token 已代表的子 token 数），δ 是 cosine 距离。
- **关键**：累积的 voxel token（大 n）merge error 远大于 fresh token（n=1）。fresh n=1,1 → factor 0.5；累积 n=10,10 → factor 5（10×）。**这解释了场景冲突**：累积多的场景（chess 稠密）merge 更有害。
- **启示**：cap merging-size n（不合并已累积的 token），或 reset voxel-token 累积 —— 一个 theorem-motivated 的干预。

### 3. ToSA —— 推翻"feature-similarity 总是优于 spatial"
**ToSA (2025, [arxiv 2506.20066](https://arxiv.org/abs/2506.20066))**：
- **在 EARLY ViT 层，spatial（空间邻近）比 feature-similarity 更可靠**（early layer 特征弱）；later 层 feature-similarity 更好。
- **答案不是单一全局规则，而是 layer-regime-dependent**。我们的 voxel-dedup（spatial）在 early 层可能恰当，问题是**均匀应用到所有层**。
- 限定：ToSA 需 depth 输入（RGB-D），RGB-only 需单目深度模块。

### 4. VGGT 家族的 feature-similarity 加速（最近亲参考实现）
- **LiteVGGT ([arxiv 2512.04939](https://arxiv.org/abs/2512.04939))**：cosine-similarity merge（已验证 repo 代码 `/mnt/lyj/workspace/LiteVGGT-repo/merging/merge.py`，用 `scatter_reduce mean`，非 spatial）。per-token geometric importance 选 anchor。
- **HTTM ([arxiv 2511.21317](https://arxiv.org/abs/2511.21317))**：training-free 3D token merging for VGGT，7× 加速。
- 限定：都加速 **batch** VGGT（非 streaming），不声称短序列 gap closure（该子声明被 refute）。

### 5. 层自适应 / 内容感知 KV cache（原则性教训）
- **PyramidKV**：低层多 cache 高层少，12% retention 匹配 full-cache（LongBench LLM）。
- **ZipVL (ICCV 2025)**：normalized attention score 选 token，per-layer/per-task 自适应比例。
- **AdaMerge**：salience-weighted bipartite matching。
- **统一教训**：token 决策应 **per-layer + per-content**，非 per-voxel 或固定数量。

---

## 对"全面超越 legacy"的可执行方向（按 ROI 排序）

### 方向 A（最直接、可立即测试）：Co-Me log(n) bias 校正
在我们**现有的 soft-merge 代码**（`_apply_intra_merge_`）上加 **log(n) attention bias**：merged token 的 attention logit 加 `log(n)`（n = 合并的成员数），恢复 attention mass。这是**小改动**，直接针对 chess 0.5 灾难的根因（attention 欠权重）。先测这个 —— 如果 log(n) 校正让 chess 从 0.5 回到 ~0.03，soft-merge 就复活了。

### 方向 B：只 merge 低置信度 token（Co-Me 的另一半）
不 merge 所有 co-voxel token，只 merge 低置信度/低 importance 的（高 importance 保留）。chess 棋盘的高 importance token 不被合并 → 避免灾难。可用现有 importance score 作代理（不需蒸馏 predictor）。

### 方向 C：ATM merge-size cap
cap 每个 voxel token 的 merging-size n（累积到阈值就 stop merge / reset），避免大 n 的高 error。

### 方向 D：layer-regime split（ToSA）
early 层保留 spatial voxel dedup，later 层换 feature-similarity merge。需要分层的 merge 策略。

### 方向 E：distillation（最重，但可能唯一能 provably 匹配 batch）
full-attention teacher → streaming student 的 KD/consistency loss。研究 angle 5 返回**零个已验证方法** —— 这是最未开发但也可能最彻底的路径。需训练。

---

## 推荐：先测方向 A（log(n) bias）
理由：(1) 最小改动（在我们已有的 merge 代码上加一行 bias）；(2) 直接针对已诊断的失败模式（attention 欠权重）；(3) 可立即在 chess/fire 上验证；(4) 若成功，soft-merge 复活 + 可能解锁 Pareto。若 A 单独不够，叠加 B（低置信度才 merge）。

## 诚实保留
- log(n) bias 是 Co-Me 在 **geometric transformer batch** 上验证的，未在 streaming ViT KV cache 短序列上验证。
- 即使 log(n) 修复了 chess 灾难，soft-merge 仍可能在 fire/office 上不如 drop（需 G1 重测）。
- "全面超越"可能仍不可达（场景冲突是几何本质），但 log(n) bias 是迄今最有理论支撑的尝试。

## Sources
- Co-Me: https://arxiv.org/abs/2511.14751
- ATM: https://arxiv.org/abs/2505.15160
- ToSA: https://arxiv.org/abs/2506.20066
- LiteVGGT: https://arxiv.org/abs/2512.04939
- HTTM: https://arxiv.org/abs/2511.21317
- PyramidKV: https://arxiv.org/abs/2406.02069
- ZipVL: https://arxiv.org/abs/2410.08584
- AdaMerge: https://arxiv.org/abs/2605.27465
- SnapKV: https://arxiv.org/abs/2404.14469
