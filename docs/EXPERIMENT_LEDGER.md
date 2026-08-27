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

## OnePiece controlled architecture pair

This pair shares the same frozen 78,921-user/660,000-candidate protocol,
hidden size 128, four blocks, four heads, Batch 32, six epochs, AdamW schedule,
sample-bias-corrected InfoNCE, seed, and exact retrieval implementation. The
sequence encoder is the only intended factor.

| Variant | Encoder | State | HR@10 | NDCG@10 | Score | Parameters |
|---|---|---|---:|---:|---:|---:|
| OnePiece HSTU 4×128 | HSTU | complete/verified | **0.0977433** | **0.0522325** | **0.0663408** | 491,793,216 |
| OnePiece Transformer 4×128 | causal Transformer | complete/verified | 0.0939167 | 0.0501666 | 0.0637291 | 492,013,584 |

HSTU improves the fixed-seed score by 0.0026117 (4.10%). The paired fixed-
population interval is `[0.0017145, 0.0035089]`, but does not include training-
seed uncertainty. Slice effects are heterogeneous: +58.65%, +33.53%, +7.07%,
and -0.93% for history lengths 0–20, 21–50, 51–80, and 81+ respectively.

## OnePiece capacity scaling and SID follow-up

The follow-up freezes the same 78,921 rows, 660,000 candidates, six-epoch
InfoNCE protocol and seed. Capacity changes jointly alter depth and width; the
SID row adds only the frozen two-level semantic-ID auxiliary objective at the
selected 8×512 scale.

| Variant | State | HR@10 | NDCG@10 | Score | Evidence note |
|---|---|---:|---:|---:|---|
| HSTU 4×128 | complete/verified | 0.0977433 | 0.0522325 | 0.0663408 | Frozen reference |
| HSTU 8×256 | complete/verified | 0.1112251 | 0.0602908 | 0.0760805 | +14.68% vs 4×128 |
| HSTU 8×512 | complete/verified | **0.1208170** | **0.0665108** | **0.0833458** | +25.63% vs 4×128; selected for SID |
| HSTU 8×512 + old SID (collisions, weight 1.0) | complete/verified | 0.1145576 | 0.0622381 | 0.0784571 | -5.87% vs matched no-SID run |
| HSTU 8×512 + collision-free SID, weight 0.02 | complete/verified | 0.1207663 | **0.0666985** | **0.0834596** | +0.14% vs no SID; interval crosses zero |
| HSTU 8×512 + collision-free SID, weight 0.05 | complete/verified | 0.1199427 | 0.0660450 | 0.0827533 | -0.71% vs no SID; interval crosses zero |

Evidence milestones:

- Both scaling checkpoints, predictions, metrics and logs passed SHA-256 and
  frozen-row verification; all private run outputs were cleaned after download.
- The SID mapping covers 4,783,154 items with 2,533,113 unique two-level pairs,
  and is bound to the selected checkpoint by source and mapping hashes.
- SID loss rises sharply after epoch 2, so the negative effect is recorded as
  an auxiliary-loss/codebook diagnosis, not a general claim against semantic IDs.
- The alignment follow-up removes all pair collisions and linearly warms the SID
  objective for 28,176 steps. Weight 0.02 recovers 6.38% versus the old SID run
  and is statistically tied with no SID on the fixed population.
- Beam 20×384 reveals a trade-off: weight 0.05 reaches 0.0252147 versus 0.0051892
  at weight 0.02, but its full-candidate score is 0.85% lower.
- The private Hugging Face archive passed read-back verification at commit
  `686e5760a89014c9dcb32f58f7f32f559779de4e`: 151 files and 29,874,511,230 bytes;
  19 LFS objects matched remote OIDs and 132 small files matched downloaded SHA-256.
