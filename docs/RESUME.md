# Resume and Interview Kit

## 中文简历项目名

**TencentGR-1M 大规模生成式推荐复现与 HSTU/SID 消融**

## 中文三条版

- 基于 2025 腾讯广告算法大赛 TencentGR-1M 与获许可的 OnePiece 方案，复现
  HSTU + sample-bias-corrected InfoNCE 训练链路；在 100 万用户、478 万 Item 索引
  与官方 66 万候选上构建无泄漏最后点击评测，8×512 HSTU 固定 seed 达到
  **HR@10 0.12082、NDCG@10 0.06651、综合分 0.08335**。
- 设计 Transformer/HSTU 架构对照与 4×128→8×256→8×512 容量实验，冻结
  78,921 行用户、Target、历史长度和候选集合；HSTU 4×128 相对同预算 Transformer
  综合分提高 **4.10%**，扩展到 8×512 后较 4×128 再提高 **25.63%**，并用逐用户
  配对区间和历史切片限制单 seed 结论。
- 从训练后 Item Tower 生成两级 residual spherical-k-means 语义 ID，并完成同规模
  SID 辅助损失消融；识别未加权辅助目标导致 Epoch 3 后 loss 抬升、综合分下降
  **5.87%** 的负结果。搭建原子断点、私有状态心跳、SHA-256 回读、预测行对齐、
  远端自动清理，以及 Hugging Face 私有大产物 / GitHub 公共证据分层发布流程。

## 中文一句话版

在 TencentGR-1M 上完成最高 14.0 亿参数 OnePiece/HSTU 资源缩放复现、严格
Transformer 对照和模型派生 SID 消融，覆盖 78,921 用户 × 66 万候选精确 Top-10；
8×512 达到 HR@10 0.12082、NDCG@10 0.06651，并将失败的 SID 实验定位为辅助损失
平衡与 codebook 碰撞问题。

## English bullets

- Reproduced the OnePiece HSTU plus sampling-bias-corrected InfoNCE pipeline on
  TencentGR-1M and built a leakage-aware exact Top-10 evaluation over 78,921
  users and the official 660k candidate set; the fixed-seed 8×512 HSTU reached
  HR@10 0.12082, NDCG@10 0.06651, and a 0.08335 weighted score.
- Ran a controlled causal-Transformer/HSTU comparison and a 4×128 → 8×256 →
  8×512 capacity study with row-aligned predictions; HSTU improved 4.10% over
  the matched Transformer, while 8×512 improved 25.63% over 4×128 under the
  same offline contract.
- Generated frozen two-level semantic IDs from trained item-tower embeddings
  and diagnosed a matched SID auxiliary-loss ablation that regressed 5.87%; the
  evidence points to unweighted auxiliary gradients and code collisions. Built
  resumable shared-GPU execution, SHA-256/read-back verification, remote cleanup,
  and private-HF/public-GitHub artifact governance.

## 60 秒面试开场

我没有停在官方 Baseline，而是选择获许可的 OnePiece 核心路线，先把 HSTU 和
sample-bias-corrected InfoNCE 缩放到共享 H200 可执行的配置，再冻结 78,921 个用户和
66 万候选的无泄漏协议。第一步用除编码器外完全相同的 Transformer 做单变量对照，
HSTU 4×128 综合分高 4.10%；第二步扩到 8×512，综合分从 0.06634 提到 0.08335；
第三步从已训练 Item Tower 生成两级语义 ID 做同规模消融。SID 没有带来收益，反而
下降 5.87%，而且 loss 从第三个 Epoch 起明显抬升。我没有隐藏这个负结果，而是结合
辅助目标未加权和 225 万碰撞 Item 给出下一轮 loss weight、warm-up 与 codebook 设计。
整个过程还实现了断点恢复、哈希回读、行对齐、远端清理和 HF/GitHub 分层交付。

## 面试结论边界

- 所有数字都是固定 seed 的本地最后点击留出结果，不是官方榜单分数或比赛名次。
- 逐用户 95% 区间只覆盖固定模型在评测人口上的差异，不包含训练 seed 不确定性。
- 4×128→8×512 同时改变宽度与深度，只能归因于整体 capacity scaling。
- SID 负结果只针对当前两级聚类、碰撞率和未加权联合损失，不能外推为语义 ID 无效。

## 证据入口

- 容量扩展：`docs/ONEPIECE_SCALING_RESULTS.md`
- SID 消融：`docs/ONEPIECE_SID_RESULTS.md`
- 面试深挖：`docs/ONEPIECE_INTERVIEW_CN.md`
- 机器可读结果：`metrics/onepiece_scaling_comparison.json`、
  `metrics/onepiece_sid_comparison.json`

## Search keywords

`generative recommendation`, `HSTU`, `InfoNCE`, `semantic ID`, `retrieval`,
`Top-K`, `PyTorch`, `CUDA`, `offline evaluation`, `reproducibility`,
`artifact governance`
