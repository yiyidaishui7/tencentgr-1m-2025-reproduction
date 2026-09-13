# 项目汇报、简历与面试材料总览

这份文档是项目的中文展示入口。面试官应当先记住这一句话：

> 用过去的行为预测用户下一次点击。在约 100 万用户的数据上，我复现推荐模型、固定评测规则、比较结构和规模，并把一次加入 SID 后的下降排查、修复到接近原水平。

第一次了解项目，可以先读 [项目白话说明](PROJECT_EXPLAINED_CN.md)；准备展示时，使用 [3 页信息增强版 PPT](INTERVIEW_DECK_CN.md) 和 [简单语言讲稿](INTERVIEW_SCRIPTS_CN.md)。

所有数字均来自本地最后点击留出协议，不是官方榜单成绩。每个训练配置只有一个 seed；逐用户区间描述固定模型在当前评测人口上的差异，不包含训练随机性。

## 1. 按场景使用

| 场景 | 建议材料 | 用时 |
|---|---|---:|
| HR 初筛或简历一页版 | [简历项目表述](RESUME.md) 的“两条精简版” | 20–30 秒 |
| 技术面试开场 | [3 页面试展示稿](INTERVIEW_DECK_CN.md) 与 [逐页讲稿](INTERVIEW_SCRIPTS_CN.md) | 3 分钟；可展开至 6–8 分钟 |
| 算法深挖 | [OnePiece 专项问答](ONEPIECE_INTERVIEW_CN.md) | 10–30 分钟 |
| 数据、评测与工程追问 | [基础链路问答](INTERVIEW_QA_CN.md) 与 [故障复盘](POSTMORTEM_CN.md) | 按问题展开 |
| 汇报或项目评审 | 本页、[架构与实验设计](ARCHITECTURE_AND_EXPERIMENTS_CN.md)、[对齐结果](ONEPIECE_ALIGNMENT_RESULTS.md) | 10–15 分钟 |
| 复现实验 | [中文运行手册](REPRODUCTION_RUNBOOK_CN.md) 与 [OnePiece 运行手册](ONEPIECE_RUNBOOK.md) | 按环境执行 |

## 2. 项目内容地图

| 层次 | 已完成内容 | 主要入口 |
|---|---|---|
| 数据契约 | 固定 seed 的用户划分、最后点击留出、目标前历史、候选与 Item ID 对齐 | `dataset.py`、`offline_eval.py`、`scripts/audit_id_alignment.py` |
| 基础模型 | SASRec 风格因果 Transformer、可选多模态融合、BCE 训练 | `model.py`、`main.py` |
| OnePiece 路线 | HSTU、sample-bias-corrected InfoNCE、容量扩展、两级 SID 辅助目标 | `scripts/run_onepiece_formal.py`、`docs/ONEPIECE_RUNBOOK.md` |
| 检索评测 | 78,921 个用户、66 万候选、分块精确内积 Top-10、HR@10/NDCG@10 | `infer.py`、`offline_eval_utils.py` |
| 对照实验 | HSTU/Transformer、4×128 到 8×512、SID 权重与协议对齐；Baseline 2×2 作为待复跑的历史记录保留 | `metrics/*.json`、`docs/EXPERIMENT_LEDGER.md` |
| 工程交付 | CPU/CUDA/Ascend 路由、断点、哈希回读、行对齐、远端清理、GitHub/HF 分层发布 | `runtime_utils.py`、`scripts/verify_onepiece_artifact.py` |
| 质量门槛 | 单元测试、来源契约、评测契约、结果渲染与产物校验 | `tests/`、`docs/ACCEPTANCE_CHECKLIST.md` |

## 3. 汇报主线

### 问题

官方 Baseline 能说明模型入口，却没有直接提供一套跨设备、可审计的离线闭环。序列、特征、多模态数据和候选文件还可能使用不同 Schema 或 ID 表示。模型即使正常运行，目标泄漏或静默错位仍会让指标失真。

### 方法

项目先冻结评测契约，再扩展模型：固定 90/10 用户划分，对验证用户保留最后一次点击，只用目标之前的历史构造 Query；候选检索使用分块精确内积，消除 ANN 误差。模型侧先复现基础 Transformer，再加入 HSTU 与 LogQ 校正的 InfoNCE，最后执行容量扩展和 SID 单变量消融。

### 证据

1. 同预算 4×128 对照中，HSTU 综合分为 0.06634，Transformer 为 0.06373，固定 seed 点估计提高 4.10%。
2. HSTU 从 4×128 扩展到 8×512 后，综合分达到 0.08335，提高 25.63%。该实验同时改变深度和宽度，只支持整体容量扩展结论。
3. 第一轮 SID 综合分下降 5.87%；排查发现约 225 万个重复占用编号的对象，以及等权加入的辅助损失。调整编号、首轮逐渐增加辅助权重，并采用 0.02 权重后，分数恢复到 0.08346。相对无 SID 仅高 0.14%，差值区间包含 0，因此结论是“恢复到接近原水平，尚未证明稳定提升”；不能单独归因于某项修改，也没有证明严格等价。
4. 对齐候选与历史过滤后，同一控制 checkpoint 的综合分从 0.08335 变为 0.08552。这个 +2.61% 来自评测协议变化；掩码与 SID 的对齐模型对照区间均跨 0。

### 工程闭环

公开仓库保存代码、测试、配置与聚合指标；大权重和受限衍生产物按许可放到 Hugging Face 或私有归档。每次同步保留清单和 SHA-256，预测文件还要通过用户、Target 与历史长度逐行对齐，结果验收后再清理远端临时目录。

## 4. 面试中优先展示的数字

| 数字 | 回答的问题 | 必须同时说明的边界 |
|---:|---|---|
| 78,921 × 660,000 | 评测规模与精确检索难度 | 66 万是官方候选池，不是 478 万完整 Item 空间 |
| +4.10% | HSTU 是否优于同预算 Transformer | 单个训练 seed；区间只覆盖固定评测人口 |
| +25.63% | 当前资源范围内扩容是否有效 | 深度和宽度同时变化，不能拆分单独贡献 |
| 2,250,041 → 0 | SID 工程修复是否真实 | 零碰撞映射是全局 residual 近似，不是上游严格层级算法 |
| -5.87% → 接近基线 | 是否形成失败诊断闭环 | 0.02 相对无 SID 仅 +0.14% 点估计，区间跨 0，未证明稳定提升 |
| 149 passed, 2 skipped | 公开代码是否经过回归验证 | 跳过项依赖特定运行环境，不代表完整训练已在本机重跑 |

## 5. 个人贡献表述

面试时把个人工作分成两类：

- 算法与实验：复现 HSTU/InfoNCE；设计同预算架构对照、容量实验、SID 权重消融和候选协议 2×2；用逐用户配对与历史切片解释结果。
- 系统与交付：修复路径和设备耦合；实现无泄漏评测、分块精确 Top-10、ID 审计、断点恢复、哈希回读与分层发布；把失败恢复过程固化成测试和验收清单。

项目使用了腾讯官方 Baseline，并参考固定提交的公开 OnePiece 方法与代码。公开材料保留来源、提交和许可状态；面试时使用“复现、扩展、修复、验证”，不要把上游方法描述成个人原创。OnePiece 上游未提供明确许可证，当前树不再分发其衍生补丁；历史风险单独记录在[已知限制](KNOWN_LIMITATIONS_CN.md)。

## 6. 结论边界

- 不写官方榜单成绩、比赛名次或完整复现 OnePiece 最终方案。
- 不把 HSTU 相对基础 Baseline 的系统级差异当成纯架构收益；纯架构证据来自同配置 Transformer/HSTU 对照。
- 不把单 seed 的逐用户区间解释为训练稳定性。
- 不把协议变化带来的分数变化描述成模型提升。
- 不把 SID 0.02 的 +0.14% 点估计写成显著收益。
- 不把历史 Baseline maxlen/MM 2×2 用作简历主结论；其训练 user-token 构造与评测不一致，修复后仍待重跑。

## 7. 证据入口

- 基础 2×2 历史记录：[`metrics/four_way_comparison.json`](../metrics/four_way_comparison.json)，使用前先看[已知限制](KNOWN_LIMITATIONS_CN.md)
- HSTU/Transformer：[`metrics/onepiece_architecture_comparison.json`](../metrics/onepiece_architecture_comparison.json)
- 容量扩展：[`metrics/onepiece_scaling_comparison.json`](../metrics/onepiece_scaling_comparison.json)
- SID 修复：[`metrics/onepiece_alignment_comparison.json`](../metrics/onepiece_alignment_comparison.json)
- 对齐协议补充：[`metrics/onepiece_followup_comparison.json`](../metrics/onepiece_followup_comparison.json)
- 实验记录：[`docs/EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)
- 资源预算：[`docs/RESOURCE_BUDGET_CN.md`](RESOURCE_BUDGET_CN.md)
