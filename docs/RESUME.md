# Resume and Interview Kit

## 中文简历项目名

**TencentGR-1M 大规模生成式推荐复现与 HSTU/SID 消融**

## 中文三条版

- 基于 2025 腾讯广告算法大赛 TencentGR-1M 与获许可的 OnePiece 方案，复现
  HSTU + sample-bias-corrected InfoNCE 训练链路；在 100 万用户、478 万 Item 索引
  与官方 66 万候选上构建无泄漏最后点击评测，8×512 HSTU 固定 seed 达到
  **HR@10 0.12077、NDCG@10 0.06670、综合分 0.08346**（无冲突 SID 0.02）。
- 设计 Transformer/HSTU 架构对照与 4×128→8×256→8×512 容量实验，冻结
  78,921 行用户、Target、历史长度和候选集合；HSTU 4×128 相对同预算 Transformer
  综合分提高 **4.10%**，扩展到 8×512 后较 4×128 再提高 **25.63%**，并用逐用户
  配对区间和历史切片限制单 seed 结论。
- 从训练后 Item Tower 生成两级语义 ID，为 4,783,154 个 Item 完成容量平衡重分配并
  将 pair 碰撞由 2,250,041 降至 **0**；用线性 warm-up 和 0.02/0.05 权重消融将早期
  SID -5.87% 退化修复到与无 SID 持平（点估计 +0.14%），并通过 Beam 20×384→重排
  定位生成召回与全候选排序的目标冲突。搭建断点、SHA-256 回读、预测行对齐、远端
  自动清理，以及 Hugging Face 私有大产物 / GitHub 公共证据分层发布流程。

## 中文一句话版

在 TencentGR-1M 上完成最高 14.0 亿参数 OnePiece/HSTU 资源缩放复现、严格
Transformer 对照和模型派生 SID 消融，覆盖 78,921 用户 × 66 万候选精确 Top-10；
8×512 + 无冲突 SID(0.02) 达到 HR@10 0.12077、NDCG@10 0.06670，并将早期 SID
退化修复到与无 SID 基本持平。

## English bullets

- Reproduced the OnePiece HSTU plus sampling-bias-corrected InfoNCE pipeline on
  TencentGR-1M and built a leakage-aware exact Top-10 evaluation over 78,921
  users and the official 660k candidate set; the fixed-seed 8×512 HSTU reached
  HR@10 0.12077, NDCG@10 0.06670, and a 0.08346 weighted score with
  collision-free two-level SIDs at a 0.02 auxiliary weight.
- Ran a controlled causal-Transformer/HSTU comparison and a 4×128 → 8×256 →
  8×512 capacity study with row-aligned predictions; HSTU improved 4.10% over
  the matched Transformer, while 8×512 improved 25.63% over 4×128 under the
  same offline contract.
- Generated frozen two-level semantic IDs from trained item-tower embeddings and
  reduced pair collisions from 2.25M to zero across 4.78M items. Linear warm-up
  and a 0.02 auxiliary weight recovered the initial -5.87% SID regression to a
  statistical tie with no SID (+0.14% point estimate); Beam 20×384 ablations
  exposed a generation-recall/full-ranking trade-off. Built resumable execution,
  SHA-256 read-back, remote cleanup, and private-HF/public-GitHub governance.

## 60 秒面试开场

我没有停在官方 Baseline，而是选择获许可的 OnePiece 核心路线，先把 HSTU 和
sample-bias-corrected InfoNCE 缩放到共享 H200 可执行的配置，再冻结 78,921 个用户和
66 万候选的无泄漏协议。第一步用除编码器外完全相同的 Transformer 做单变量对照，
HSTU 4×128 综合分高 4.10%；第二步扩到 8×512，综合分从 0.06634 提到 0.08335；
第三步从已训练 Item Tower 生成两级语义 ID 做同规模消融。第一轮下降 5.87%，我把
原因定位到辅助目标未加权和 225 万碰撞 Item；随后实现零碰撞映射、线性 warm-up 和
0.02/0.05 权重对照，把 0.02 修复到与无 SID 持平，并用 Beam→重排结果定位了生成
召回与全候选排序之间的冲突。
整个过程还实现了断点恢复、哈希回读、行对齐、远端清理和 HF/GitHub 分层交付。

## 面试结论边界

- 所有数字都是固定 seed 的本地最后点击留出结果，不是官方榜单分数或比赛名次。
- 逐用户 95% 区间只覆盖固定模型在评测人口上的差异，不包含训练 seed 不确定性。
- 4×128→8×512 同时改变宽度与深度，只能归因于整体 capacity scaling。
- 0.02 对无 SID 的 +0.14% 点估计区间跨 0，不能写成稳定提升。

## 证据入口

- 容量扩展：`docs/ONEPIECE_SCALING_RESULTS.md`
- SID 消融：`docs/ONEPIECE_SID_RESULTS.md`
- 对齐总结：`docs/ONEPIECE_ALIGNMENT_RESULTS.md`
- 面试深挖：`docs/ONEPIECE_INTERVIEW_CN.md`
- 机器可读结果：`metrics/onepiece_scaling_comparison.json`、
  `metrics/onepiece_sid_comparison.json`、`metrics/onepiece_alignment_comparison.json`

## Search keywords

`generative recommendation`, `HSTU`, `InfoNCE`, `semantic ID`, `retrieval`,
`Top-K`, `PyTorch`, `CUDA`, `offline evaluation`, `reproducibility`,
`artifact governance`
