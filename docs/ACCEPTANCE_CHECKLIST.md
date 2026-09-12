# End-to-End Acceptance Checklist

The project is complete only when every required item below has current evidence.

## Experiment evidence

- [x] Historical MM101/no-MM101/MM50/no-MM50 artifacts are complete, row-aligned, and SHA-verified.
- [x] Training/evaluation user-token contract mismatch identified, fixed, and covered by regression tests.
- [ ] Corrected Baseline 2×2 retraining and re-evaluation complete.
- [x] Historical 2×2 interpretations withdrawn until that retraining is complete.
- [x] OnePiece architecture, capacity, SID, and protocol comparisons verified independently.

## Reproducibility and publishing

- [x] Public code includes deterministic split, portable device selection, exact Top-10 evaluation, no-MM ablation, and regression tests.
- [x] Review-only 2x2 planner emits the canonical MM/no-MM x maxlen 101/50 argv, checkpoint, and comparison contract without starting jobs.
- [x] Public GitHub CI passes on the current public HEAD.
- [x] Public Hugging Face contains the historical MM101 and no-MM50 SafeTensors checkpoints; both LFS SHA-256 values are verified and are not promoted as current effect evidence.
- [x] Expanded private Hugging Face archive inventory and all 77 manifest hashes verified.
- [x] Recent-window metrics and promoted small artifacts published and reverified.
- [x] Final run-side temporary directories confirmed absent; private parent permissions confirmed.

## Resume and interview package

- [x] Chinese and English resume bullets drafted.
- [x] Architecture and controlled-experiment explanation drafted.
- [x] Technical deep-dive Q&A drafted.
- [x] Engineering incident and trade-off review drafted.
- [x] Final 3-minute and 10-minute interview scripts use the defensible OnePiece evidence chain.
- [x] Three-page editable PowerPoint and rendered previews are published.
- [x] Measured evaluation runtime, memory scaling, and public/private artifact budget documented with scope limitations.
- [x] Delivery index relative links and current public artifact paths verified.

## Final gate

- [x] Current unit tests pass locally (149 passed, 2 expected skips) and Python sources compile.
- [x] The generated review plan contains four distinct variants and the expected row-aligned comparison inputs.
- [x] Relative links in the current Markdown delivery package resolve.
- [x] Public repository has no private paths, credentials, or restricted raw artifacts.
- [x] Final GitHub commit/CI and public Hugging Face release subset are reachable and hash-verified; all 77 private-manifest entries are present and hash-matched.
- [x] Final handoff summary contains metrics, limitations, resume text, interview kit, reproduction commands, and artifact links.
