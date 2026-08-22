# Resume and Interview Kit

## 中文简历项目名

**TencentGR-1M 全模态生成式推荐系统复现与工程化**

## 中文三条版

- 基于 2025 腾讯广告算法大赛官方 Baseline 与 TencentGR-1M，完成 SASRec
  风格全模态序列推荐的训练、候选编码、Top-10 召回及最后点击留出评估，覆盖
  78,921 名有效用户，取得 HR@10 0.03135、NDCG@10 0.01597。
- 重构硬编码设备与路径依赖，支持 CPU/CUDA/Ascend，增加确定性数据划分、断点
  恢复、PyTorch 精确内积召回、候选 ID 对齐审计及同口径消融；no-MM 综合分
  0.02124，点估计提升 2.44%，并通过配对区间与单随机种子限制约束实验结论。
- 建立权重与实验产物治理流程：远端 SHA-256 校验、自动同步与清理、SafeTensors
  无损转换及公开/私有分仓发布；48 个张量逐项验证一致，并为核心逻辑补充回归测试。

## 中文一句话版

在 TencentGR-1M 上完成全模态序列推荐端到端复现，将官方 Baseline 工程化为支持
多设备、可审计离线评估、精确 Top-10 召回和安全模型发布的可复现系统。

## English bullets

- Reproduced the 2025 Tencent Ads SASRec-style multimodal recommendation
  baseline end to end on TencentGR-1M, evaluating 78,921 eligible users with
  HR@10 0.03135 and NDCG@10 0.01597 under a leakage-aware last-click protocol.
- Reworked device/path assumptions for CPU, CUDA, and Ascend; added seeded data
  splits, checkpoint resume, exact PyTorch Top-10 retrieval, candidate-ID
  audits, and a row-aligned no-MM ablation whose 0.02124 point estimate was
  2.44% higher while paired uncertainty crossed zero.
- Built a verifiable artifact pipeline with remote/local SHA-256 checks,
  automatic cleanup, lossless SafeTensors conversion, public/private model
  publishing, and regression tests for critical evaluation logic.

## 60-second interview pitch

The official baseline was tied to a specific environment and did not provide a
complete auditable offline evaluation. I first made training and inference
portable, then audited ID alignment across sequences, item features,
multimodal embeddings, and the candidate pool. I implemented an exact Top-10
retrieval backend and a seeded last-click holdout evaluator, including excluded
population counts and history-length slices. Finally, I ran the multimodal
baseline and a row-aligned no-MM ablation. The no-MM point estimate was 2.44%
higher, but its paired interval crossed zero, so I treated it as a fusion
design hypothesis rather than proof against multimodal information. I also
verified all downloaded artifacts with SHA-256 and published a
tensor-equivalent SafeTensors checkpoint. The key lesson was that evaluation
contracts and ID alignment can matter as much as model code in a large
recommendation pipeline.

## Claims to avoid

- Do not describe this as a competition placement or official leaderboard score.
- Do not claim that the ablation proves multimodal features are ineffective or
  harmful; it used the same formal protocol, but only one training seed and a
  paired 95% interval that includes zero.
- Do not describe the 660k candidate pool as the full item universe.

## Search keywords

`recommender systems`, `generative recommendation`, `SASRec`, `multimodal`,
`retrieval`, `Top-K`, `PyTorch`, `SafeTensors`, `CUDA`, `Ascend NPU`,
`offline evaluation`, `reproducibility`, `artifact management`
