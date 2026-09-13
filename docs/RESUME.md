# 简历与面试项目表述

## 项目名

**TencentGR-1M 大规模生成式推荐复现与 HSTU/SID 消融**

推荐关键词：`生成式推荐`、`HSTU`、`InfoNCE`、`Semantic ID`、`Top-K 检索`、`PyTorch`、`CUDA`、`可复现评测`

## 简单语言版：两条即可

- 在约 100 万用户的推荐数据上，复现 HSTU 模型并固定 78,921 人、66 万候选的精确 Top-10 评测；单次训练对照中，同规模 HSTU 比 Transformer 综合分提高 4.10%，扩大到 8 层、512 维后比小配置提高 25.63%。
- 排查加入语义编号预测后下降 5.87% 的实验，将重复编号占用数从 225 万降到 0，并调整辅助权重和首轮预热，使得分恢复到接近原水平；整理断点恢复、结果逐行核对、文件校验与项目交付。当前尚未证明 SID 带来稳定额外收益。

口述时先说“看过去行为、预测下一次点击”，再按面试官的追问解释术语。完整解释见 [项目白话说明](PROJECT_EXPLAINED_CN.md)，现场讲述见 [简单语言讲稿](INTERVIEW_SCRIPTS_CN.md)。

## 两条精简版

- 基于 2025 腾讯广告算法大赛 TencentGR-1M，并参考公开 OnePiece 方法、固定上游提交，复现 HSTU + sample-bias-corrected InfoNCE 训练链路；在 100 万用户、478 万 Item 索引和官方 66 万候选上构建 78,921 人目标泄漏受控的精确 Top-10 评测。同预算 4×128 HSTU 相对 Transformer 综合分提高 **4.10%**，8×512 相对 4×128 再提高 **25.63%**。
- 从训练后 Item Tower 生成两级语义 ID，将 478 万 Item 的重复编号占用数由 **2,250,041 降至 0**；通过线性 warm-up 与 0.02/0.05 权重对照，把第一轮 SID 的 **-5.87%** 退化修复到接近无 SID，尚未证明稳定增益；并实现断点恢复、逐行预测对齐、SHA-256 回读与 GitHub/Hugging Face 分层发布。

## 三条完整版

- 复现 OnePiece 的 HSTU、LogQ 采样偏差校正与 in-batch InfoNCE 核心链路，支持最高 14.23 亿参数单卡训练；冻结 78,921 个用户、Target、历史长度和 66 万候选，使用分块精确内积评测 HR@10、NDCG@10 与比赛风格综合分。
- 设计同预算 Transformer/HSTU 架构对照与 4×128、8×256、8×512 容量实验；固定 seed 下，HSTU 4×128 综合分 0.06634，相对 Transformer 提高 **4.10%**，8×512 达到 0.08335，相对 4×128 提高 **25.63%**，并通过逐用户配对区间和历史切片限制结论范围。
- 构建模型派生两级 SID 并诊断首轮负结果：将 225 万重复编号占用降至 0，以首个 Epoch 线性 warm-up 和 0.02/0.05 辅助权重把 -5.87% 退化修复到接近基线；补齐候选过滤与历史过滤的同 checkpoint 2×2 重评，明确 +2.61% 属于评测协议效应。同步搭建原子断点、哈希回读、远端清理和公共代码/私有大产物治理流程。

## 一句话版

在 TencentGR-1M 上完成最高 14.23 亿参数 HSTU/InfoNCE 资源缩放、严格 Transformer 对照与模型派生 SID 消融，并建立覆盖数据契约、精确 Top-10、逐行统计和产物哈希的可审计推荐系统闭环。

## 推荐算法岗位版本

- 复现 HSTU 与 sample-bias-corrected InfoNCE，执行同预算 Transformer 对照和三档容量实验；4×128 HSTU 固定 seed 综合分提高 4.10%，8×512 相对 4×128 提高 25.63%。
- 从 Item Tower 表征构造两级 SID，使用容量平衡重分配把 478 万 Item 的重复编号占用降至 0；比较辅助权重 0.02/0.05，分别观察全候选排序与 Beam 生成召回，避免把两种指标混为同一结论。
- 对 78,921 个用户执行逐行配对、历史长度切片和候选协议 2×2，区分模型差异、评测协议效应与单 seed 不确定性。

## 机器学习系统岗位版本

- 将设备、路径、数据 Schema 与候选检索从官方 Baseline 中解耦，统一 CPU、CUDA、Ascend 路由，构建无泄漏最后点击评测和分块精确 Top-10 后端。
- 为最高 14.23 亿参数训练加入原子 checkpoint、断点恢复、共享 GPU 隔离和逐级 smoke test；对约 1 GB 权重使用分片续传与端到端 SHA-256，避免失败重跑和静默损坏。
- 将代码、测试和聚合指标发布到 GitHub，将大权重与受限衍生产物按许可分层保存；149 项公开回归测试通过，另有 2 项依赖特定运行环境而跳过。

## English bullets

- Reproduced the OnePiece HSTU pipeline with sampling-bias-corrected in-batch InfoNCE on TencentGR-1M, and built leakage-aware exact Top-10 evaluation for 78,921 users against the official 660k candidate set. Under a fixed seed, matched-budget HSTU improved the weighted score by **4.10%** over a causal Transformer, while the 8×512 configuration improved **25.63%** over 4×128.
- Generated frozen two-level semantic IDs from trained item-tower embeddings and reduced duplicate code-pair assignments from **2.25M to zero** across 4.78M items. Remapping, linear warm-up, and 0.02/0.05 auxiliary-weight comparisons recovered an initial **-5.87%** regression to near the no-SID score; the paired interval includes zero, so additional gains remain unproven.
- Built resumable single-GPU execution, row-aligned evaluation, SHA-256 read-back, remote cleanup, and public-GitHub/private-artifact release controls. A same-checkpoint protocol audit separated a +2.61% evaluation-protocol effect from model improvement.

## 60 秒面试开场

这个项目起点是腾讯广告算法大赛的 TencentGR-1M Baseline。我的工作重点是把它整理成一套能解释、能复算、能经受追问的推荐实验系统。

我先冻结无泄漏评测：对验证用户保留最后一次点击，只使用目标之前的历史，在 78,921 个用户和官方 66 万候选上做分块精确 Top-10。模型侧复现了 OnePiece 的 HSTU 与带 LogQ 校正的 in-batch InfoNCE。为了区分架构和规模效应，我先做同预算 HSTU/Transformer 对照，HSTU 的固定 seed 综合分提高 4.10%；再把 HSTU 从 4×128 扩到 8×512，综合分提高 25.63%。

最值得讲的是 SID 负结果。第一轮辅助目标加入后，综合分下降 5.87%；排查发现约 225 万重复编号占用，以及等权辅助损失和总损失反弹。我调整编号、首轮预热与两档权重后，分数恢复到接近无 SID，但尚未证明稳定提升，也不能拆分各项修改的贡献。整个过程保存逐行对照和文件校验值，每个数字都能查到配置、预测和对应指标。

## 结论边界

- 这些数字来自固定 seed 的本地最后点击留出，不是官方榜单成绩或比赛名次。
- HSTU 与 Transformer 的 +4.10% 来自同配置架构对照；HSTU 相对基础 Baseline 的大幅差异同时包含容量、损失和特征参数化变化。
- 4×128 到 8×512 同时增加深度与宽度，只支持整体容量扩展结论。
- SID 0.02 相对无 SID 的历史协议点估计为 +0.14%，配对区间跨 0；对齐协议下的 SID 差异也跨 0，所以简历写“恢复到接近基线，尚未证明稳定提升”。区间含 0 不等于已经证明严格等价。
- 逐用户区间不包含训练 seed 方差；稳定性仍需多 seed 验证。

## 证据入口

- [项目展示总览](PROJECT_PORTFOLIO_CN.md)
- [3 页面试展示稿与预览](INTERVIEW_DECK_CN.md)
- [逐页讲稿](INTERVIEW_SCRIPTS_CN.md)
- [OnePiece 对齐结果](ONEPIECE_ALIGNMENT_RESULTS.md)
- [OnePiece 技术问答](ONEPIECE_INTERVIEW_CN.md)
- [机器可读容量结果](../metrics/onepiece_scaling_comparison.json)
- [机器可读 SID 结果](../metrics/onepiece_alignment_comparison.json)
- [机器可读协议补充](../metrics/onepiece_followup_comparison.json)
