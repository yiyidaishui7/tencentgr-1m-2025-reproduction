# Project Delivery Index

This page is the final entry point for resume use, reproduction, and technical interviews.

## Public technical evidence

- [Chinese project overview](README_CN.md)
- [English project overview](README.md)
- [Evaluation results and audit](docs/RESULTS.md)
- [Architecture and experiment design (Chinese)](docs/ARCHITECTURE_AND_EXPERIMENTS_CN.md)
- [Controlled experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Resume material](docs/RESUME.md)
- [3-minute and 10-minute interview scripts (Chinese)](docs/INTERVIEW_SCRIPTS_CN.md)
- [Technical interview Q&A (Chinese)](docs/INTERVIEW_QA_CN.md)
- [Engineering incident review (Chinese)](docs/POSTMORTEM_CN.md)
- [Reproduction runbook (Chinese)](docs/REPRODUCTION_RUNBOOK_CN.md)
- [Resource, runtime, and artifact budget (Chinese)](docs/RESOURCE_BUDGET_CN.md)
- [End-to-end acceptance checklist](docs/ACCEPTANCE_CHECKLIST.md)
- [OnePiece resource-scaled reproduction](docs/ONEPIECE_REPRODUCTION_CN.md)
- [OnePiece path-neutral runbook](docs/ONEPIECE_RUNBOOK.md)
- [OnePiece resume draft](docs/ONEPIECE_RESUME_DRAFT_CN.md)
- [OnePiece interview Q&A](docs/ONEPIECE_INTERVIEW_CN.md)

## Reproduction surfaces

- Code: <https://github.com/yiyidaishui7/tencentgr-1m-2025-reproduction>
- Public model and small artifacts: <https://huggingface.co/sixteensun/tencentgr-1m-2025-reproduction>
- Dataset source: <https://huggingface.co/datasets/TAAC2025/TencentGR-1M>
- Drift-resistant 2x2 plan generator: `python scripts/plan_2x2_experiments.py --help`
- OnePiece HSTU/Transformer runner: `python scripts/run_onepiece_formal.py`

The full private evidence archive is intentionally not linked from the public repository. It contains restricted raw artifacts, PyTorch checkpoints, predictions, and execution logs. Public claims must remain reproducible from the documented protocol and published aggregate evidence.
