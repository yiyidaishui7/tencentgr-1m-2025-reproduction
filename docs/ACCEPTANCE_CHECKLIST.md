# End-to-End Acceptance Checklist

The project is complete only when every required item below has current evidence.

## Experiment evidence

- [x] MM101 training and formal offline evaluation complete.
- [x] no-MM101 training and row-aligned formal offline evaluation complete.
- [x] MM50 training/evaluation complete; model, metrics, predictions, and log SHA-verified locally; remote temporary data cleaned.
- [x] no-MM50 training/evaluation complete with the same artifact checks.
- [x] Four-way target/user/history alignment and slice comparison complete.
- [x] Final point-estimate configuration selected with single-seed and non-leaderboard limitations stated.

## Reproducibility and publishing

- [x] Public code includes deterministic split, portable device selection, exact Top-10 evaluation, no-MM ablation, and regression tests.
- [x] Review-only 2x2 planner emits the canonical MM/no-MM x maxlen 101/50 argv, checkpoint, and comparison contract without starting jobs.
- [x] Public GitHub CI passes on the current public HEAD.
- [x] Public Hugging Face contains the audited MM101 and best-point-estimate no-MM50 SafeTensors checkpoints; both LFS SHA-256 values are verified.
- [x] Expanded private Hugging Face archive inventory and all 77 manifest hashes verified.
- [x] Recent-window metrics and promoted small artifacts published and reverified.
- [x] Final run-side temporary directories confirmed absent; private parent permissions confirmed.

## Resume and interview package

- [x] Chinese and English resume bullets drafted.
- [x] Architecture and controlled-experiment explanation drafted.
- [x] Technical deep-dive Q&A drafted.
- [x] Engineering incident and trade-off review drafted.
- [x] Final 3-minute and 10-minute interview scripts updated with the four-way result.
- [x] Measured evaluation runtime, memory scaling, and public/private artifact budget documented with scope limitations.
- [x] Delivery index relative links and current public artifact paths verified.

## Final gate

- [x] 32 unit tests complete after the planner/checkpoint update (30 pass, 2 expected local PyTorch skips); 23 Python files compile and 7 JSON files parse.
- [x] The generated review plan contains four distinct variants and the expected row-aligned comparison inputs.
- [x] All 29 relative links across 15 Markdown files resolve.
- [x] Public repository has no private paths, credentials, or restricted raw artifacts.
- [x] Final GitHub commit/CI and public Hugging Face release subset are reachable and hash-verified; all 77 private-manifest entries are present and hash-matched.
- [x] Final handoff summary contains metrics, limitations, resume text, interview kit, reproduction commands, and artifact links.
