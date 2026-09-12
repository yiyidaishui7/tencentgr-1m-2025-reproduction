# Controlled Experiment Ledger

> **2026-09-12 contract audit:** the historical Baseline 2×2 training pipeline
> inserted repeated user tokens, while evaluation inserted one. The archived
> metrics and hashes remain valid evidence of those runs, but the matrix is no
> longer eligible for factor-effect or model-promotion claims. The independent
> OnePiece runs below use a separate sequence builder and are unaffected.

## Fixed contract

- Training/evaluation seed: 2025
- Deterministic 90/10 user split and the same official candidate pool
- Hidden size 32, one Transformer block, one attention head
- Batch size 2,048, Adam 1e-3 with betas 0.9/0.98, three epochs
- Exact Top-10 evaluation on the same 78,921 row-aligned users and targets
- Primary metric: `0.31 * HR@10 + 0.69 * NDCG@10`

The intended experimental factors were sequence `maxlen` and whether field-81
multimodal embeddings were enabled. The later token-contract audit found an
additional maxlen-dependent change in repeated user-token context, so the
historical matrix must be rerun after the fix.

## Historical search budget and promotion gate

The bounded search adds exactly two variants, MM50 and no-MM50, to the already verified MM101/no-MM101 pair. Every variant is trained once on one GPU. A result can be named the best measured point estimate only after:

1. model, metrics, predictions, and logs pass SHA-256 verification;
2. all four prediction files have identical user, target, and original-history rows;
3. overall and four history-slice metrics are recomputed from predictions;
4. paired score intervals and the 2×2 interaction are reported; and
5. single-seed, offline, and non-leaderboard limitations remain attached.

## Historical Baseline runs

| Variant | maxlen | MM | State | HR@10 | NDCG@10 | Score | Evidence note |
|---|---:|---|---|---:|---:|---:|---|
| MM101 | 101 | enabled | historical | 0.0313478035 | 0.0159694453 | 0.0207367364 | Verified archived artifacts |
| no-MM101 | 101 | disabled | historical | 0.0317532723 | 0.0165207806 | 0.0212428530 | Verified archived artifacts |
| MM50 | 50 | enabled | historical | 0.0320827156 | 0.0160502759 | 0.0210203322 | Verified and remotely cleaned |
| no-MM50 | 50 | disabled | historical | 0.0337045907 | 0.0172092099 | 0.0223227779 | Row-aligned metrics verified |

no-MM50 was the highest point estimate in the archived implementation. It is
not promoted as a better model after the contract audit. A corrected four-run
matrix and multiple training seeds are required before revisiting the maxlen or
multimodal hypotheses.

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

## 2026-09-07 historical release: OnePiece mask and aligned-control follow-up

The original six-row table above remains the historical 660k-candidate result
set. Two release-only comparisons extend the audit without rewriting it:

| Evidence row | Checkpoint and evaluation protocol | Score | Delta vs historical `s8512` | Normal 95% interval |
|---|---|---:|---:|---:|
| Historical control | `s8512`; 660k candidates, no history filtering | 0.0833458 | reference | — |
| Post-cutoff exposure mask | `xm512`; historical 660k candidates, no history filtering | 0.0828782 | -0.0004676 (-0.56%) | [-0.0013758, 0.0004406] |
| Aligned control | same `s8512`; 511,029 warm candidates, history filtering | 0.0855212 | +0.0021755 (+2.61%) | [0.0019130, 0.0024380] |

The post-cutoff interval crosses zero. The aligned-control delta is a
same-checkpoint evaluation-protocol effect, **not a model improvement**. At that
release, this was not a complete 2x2 because `xm512` had not been re-evaluated on
the aligned protocol; the SID variants had not been re-evaluated either. Those follow-ups do not exercise
Beam retrieval, and the aligned receipt records `beam_eval=false` and
`beam_ann_fallback=false`. Machine-readable evidence is in
[`onepiece_post_cutoff_mask_comparison.json`](../metrics/onepiece_post_cutoff_mask_comparison.json)
and [`onepiece_aligned_control_comparison.json`](../metrics/onepiece_aligned_control_comparison.json).

## 2026-09-08 supplement: complete mask × protocol 2×2 and aligned SID

Three new evaluations reuse the mask and SID 0.02/0.05 checkpoints without
retraining. Together with the existing aligned control and four historical
prediction sets, all eight artifacts passed SHA-256 verification, source-model
binding, frozen-contract checks, row alignment and metric recomputation.

<!-- AUTO-GENERATED values from metrics/onepiece_followup_comparison.json -->
| Existing checkpoint | Historical score | Aligned score | State |
|---|---:|---:|---|
| Control (`s8512`) | 0.0833458 | 0.0855212 | complete/verified |
| Post-cutoff mask (`xm512`) | 0.0828782 | 0.0848123 | complete/verified |
| Collision-free SID (0.02) | 0.0834596 | 0.0851821 | complete/verified |
| Collision-free SID (0.05) | 0.0827533 | 0.0849867 | complete/verified |

| Comparison | Overall score delta | Paired normal 95% interval |
|---|---:|---:|
| Aligned mask − aligned control | -0.0007089 | [-0.0016229, 0.0002051] |
| Aligned SID 0.02 − aligned control | -0.0003391 | [-0.0012484, 0.0005701] |
| Aligned SID 0.05 − aligned control | -0.0005345 | [-0.0014464, 0.0003774] |
| Mask × protocol interaction | -0.0002413 | [-0.0005553, 0.0000726] |
| Control: aligned − historical | +0.0021755 | [0.0019130, 0.0024380] |
| Mask: aligned − historical | +0.0019341 | [0.0016792, 0.0021890] |
| SID 0.02: aligned − historical | +0.0017226 | [0.0014813, 0.0019638] |
| SID 0.05: aligned − historical | +0.0022334 | [0.0019700, 0.0024968] |
<!-- END AUTO-GENERATED values -->

All comparisons use the same 78,921 users. Historical evaluation retains
660,000 candidates and does not filter user history. Every aligned run excludes
148,971 cold candidates, evaluates 511,029 warm candidates, masks 3,531,517
user-history candidate pairs, records zero history overlap and reverifies the
dataset receipt. The supplement compares exact full-candidate Top-10, not Beam results; aligned
receipts record `beam_eval=false` and `beam_ann_fallback=false`.

The aligned control is the highest point estimate, but the three overall score
model-difference intervals and the overall score interaction interval cross zero. The interaction is
`(aligned_mask - aligned_control) - (historical_mask - historical_control)`.
The four positive same-checkpoint overall score protocol intervals measure the joint
candidate/history filtering effect, **not a model improvement**. Each configuration
has one training seed; per-user intervals are unadjusted for multiple comparisons
and exclude training-seed variance. No stable model gain/loss or interaction is
established. The original six-row table and the two 2026-09-07 comparison JSONs
remain unchanged.

Full-precision metrics, history slices and source hashes are in
[`onepiece_followup_comparison.json`](../metrics/onepiece_followup_comparison.json).
Recompute using [`compare_onepiece_followup.py`](../scripts/compare_onepiece_followup.py);
the eight-input command is documented in the
[alignment report](ONEPIECE_ALIGNMENT_RESULTS.md#补充比较的复算入口).
This completion state refers to the verified evaluation artifacts, not a new
remote-archive acceptance claim.
