# Controlled Experiment Ledger

## Fixed contract

- Training/evaluation seed: 2025
- Deterministic 90/10 user split and the same official candidate pool
- Hidden size 32, one Transformer block, one attention head
- Batch size 2,048, Adam 1e-3 with betas 0.9/0.98, three epochs
- Exact Top-10 evaluation on the same 78,921 row-aligned users and targets
- Primary metric: `0.31 * HR@10 + 0.69 * NDCG@10`

The only experimental factors are sequence `maxlen` (101 or 50, including the directly dependent position-embedding table length) and whether field-81 multimodal embeddings are enabled.

## Search budget and promotion gate

The bounded search adds exactly two variants, MM50 and no-MM50, to the already verified MM101/no-MM101 pair. Every variant is trained once on one GPU. A result can be named the best measured point estimate only after:

1. model, metrics, predictions, and logs pass SHA-256 verification;
2. all four prediction files have identical user, target, and original-history rows;
3. overall and four history-slice metrics are recomputed from predictions;
4. paired score intervals and the 2×2 interaction are reported; and
5. single-seed, offline, and non-leaderboard limitations remain attached.

## Runs

| Variant | maxlen | MM | State | HR@10 | NDCG@10 | Score | Evidence note |
|---|---:|---|---|---:|---:|---:|---|
| MM101 | 101 | enabled | complete | 0.0313478035 | 0.0159694453 | 0.0207367364 | Formal baseline, verified artifacts |
| no-MM101 | 101 | disabled | complete | 0.0317532723 | 0.0165207806 | 0.0212428530 | Row-aligned ablation, verified artifacts |
| MM50 | 50 | enabled | complete | 0.0320827156 | 0.0160502759 | 0.0210203322 | Final BCE 0.2061; verified and remotely cleaned |
| no-MM50 | 50 | disabled | complete | **0.0337045907** | **0.0172092099** | **0.0223227779** | Final BCE 0.2055; row-aligned metrics verified |

The best measured point estimate is no-MM50: +7.65% versus MM101, +5.08% versus no-MM101, and +6.20% versus MM50. The overall interaction interval includes zero, and a stable causal or production-default claim would require additional training seeds. The current search intentionally stops at the fixed-seed 2×2 design.
