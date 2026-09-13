# 项目交付索引 / Project Delivery Index

本页按“先展示、再深挖、后复现”的顺序组织公开材料。

## 面试与汇报优先入口

- [项目总览与汇报主线](docs/PROJECT_PORTFOLIO_CN.md)
- [项目白话说明：流程、术语、实验与追问入口](docs/PROJECT_EXPLAINED_CN.md)
- [3 页面试 PowerPoint、预览与使用说明](docs/INTERVIEW_DECK_CN.md)
- [30 秒、3 分钟与 6–8 分钟简单语言讲稿](docs/INTERVIEW_SCRIPTS_CN.md)
- [简历两条版、三条版与英文版](docs/RESUME.md)
- [已知限制与可安全使用的结论](docs/KNOWN_LIMITATIONS_CN.md)

## Public technical evidence

- [Chinese project overview](README_CN.md)
- [English project overview](README.md)
- [Evaluation results and audit](docs/RESULTS.md)
- [Architecture and experiment design (Chinese)](docs/ARCHITECTURE_AND_EXPERIMENTS_CN.md)
- [Controlled experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Resume material](docs/RESUME.md)
- [3-minute and 6–8-minute interview scripts (plain Chinese)](docs/INTERVIEW_SCRIPTS_CN.md)
- [Technical interview Q&A (Chinese)](docs/INTERVIEW_QA_CN.md)
- [Engineering incident review (Chinese)](docs/POSTMORTEM_CN.md)
- [Known limitations and publication boundaries (Chinese)](docs/KNOWN_LIMITATIONS_CN.md)
- [Reproduction runbook (Chinese)](docs/REPRODUCTION_RUNBOOK_CN.md)
- [Resource, runtime, and artifact budget (Chinese)](docs/RESOURCE_BUDGET_CN.md)
- [End-to-end acceptance checklist](docs/ACCEPTANCE_CHECKLIST.md)
- [OnePiece resource-scaled reproduction](docs/ONEPIECE_REPRODUCTION_CN.md)
- [OnePiece verified six-run results](docs/ONEPIECE_RESULTS.md)
- [OnePiece HSTU capacity-scaling results](docs/ONEPIECE_SCALING_RESULTS.md)
- [OnePiece semantic-ID ablation](docs/ONEPIECE_SID_RESULTS.md)
- [OnePiece collision-free SID and protocol-alignment audit](docs/ONEPIECE_ALIGNMENT_RESULTS.md)
- [OnePiece path-neutral runbook](docs/ONEPIECE_RUNBOOK.md)
- [OnePiece resume draft](docs/ONEPIECE_RESUME_DRAFT_CN.md)
- [OnePiece interview Q&A](docs/ONEPIECE_INTERVIEW_CN.md)

## Reproduction surfaces

- Code: <https://github.com/yiyidaishui7/tencentgr-1m-2025-reproduction>
- Public model and small artifacts: <https://huggingface.co/sixteensun/tencentgr-1m-2025-reproduction>
- Dataset source: <https://huggingface.co/datasets/TAAC2025/TencentGR-1M>
- Drift-resistant 2x2 plan generator: `python scripts/plan_2x2_experiments.py --help`
- OnePiece HSTU/Transformer runner: `python scripts/run_onepiece_formal.py`
- Machine-readable scaling and SID comparisons: `metrics/onepiece_scaling_comparison.json`, `metrics/onepiece_sid_comparison.json`, and `metrics/onepiece_alignment_comparison.json`
- 2026-09-07 historical-release post-cutoff mask comparison on the 660k/no-history-filtering protocol: `metrics/onepiece_post_cutoff_mask_comparison.json`
- 2026-09-07 historical-release same-`s8512` aligned-control comparison on 511,029 warm candidates with history filtering: `metrics/onepiece_aligned_control_comparison.json`
- [2026-09-08 complete mask × protocol 2×2 and aligned SID comparison](metrics/onepiece_followup_comparison.json): eight verified artifact sets, four existing checkpoints, no retraining; the historical six-row table and both 2026-09-07 comparison JSONs remain unchanged.
- [Eight-artifact comparison CLI](scripts/compare_onepiece_followup.py): `python scripts/compare_onepiece_followup.py --help`; the complete command and paired intervals are in the [alignment report](docs/ONEPIECE_ALIGNMENT_RESULTS.md#补充比较的复算入口).

The 2026-09-08 supplement compares exact full-candidate Top-10 only. Its aligned control has the
highest point estimate, but the mask/SID model contrasts and the 2×2 interaction
all have overall score intervals crossing zero. Positive same-checkpoint overall score protocol effects
are not model improvements. The evidence is single-seed, with unadjusted paired
per-user intervals rather than training-seed uncertainty.

The full private evidence archive is intentionally not linked from the public repository. It contains restricted raw artifacts, PyTorch checkpoints, predictions, and execution logs. Public claims must remain reproducible from the documented protocol and published aggregate evidence.
