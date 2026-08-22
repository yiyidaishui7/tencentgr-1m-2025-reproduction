# Evaluation Results and Audit

## Evaluation contract

| Item | Definition |
|---|---|
| Split | Seeded 90/10 user split |
| Seed | 2025 |
| Target | Last `action_type=1` item for each validation user |
| History | Events strictly before the held-out click |
| Candidate pool | Official TencentGR-1M 660k candidates |
| Exclusions | No click target, or target absent from candidate pool |

This is an offline protocol implemented for the reproduction. It is not an
official leaderboard evaluation.

## Population audit

| Population | Users |
|---|---:|
| Validation users | 100,184 |
| Users with a click | 98,439 |
| No click target | 1,745 |
| Target absent from candidate pool | 19,518 |
| Eligible/evaluated users | 78,921 |

## Aggregate metrics

| Metric | Multimodal | No-MM | No-MM relative delta |
|---|---:|---:|---:|
| Hits@10 | 2,474 | **2,506** | +32 hits |
| HR@10 | 0.0313478035 | **0.0317532723** | +1.29% |
| NDCG@10 | 0.0159694453 | **0.0165207806** | +3.45% |
| Competition-style score | 0.0207367364 | **0.0212428530** | +2.44% |
| Runtime | 241.503 s | **133.926 s** | -44.54% |

The score is computed with the competition-style weighting implemented by the
project's evaluation utility. Machine-readable records are available at
`metrics/offline_metrics.json`, `metrics/offline_metrics_no_mm.json`, and
`metrics/ablation_comparison.json`.

## History-length slices

| History length | Users | Multimodal score | No-MM score | No-MM relative delta |
|---|---:|---:|---:|---:|
| 0–20 | 1,125 | 0.0245890 | **0.0251632** | +2.34% |
| 21–50 | 4,366 | **0.0258747** | 0.0251520 | -2.79% |
| 51–80 | 23,339 | 0.0245931 | **0.0250301** | +1.78% |
| 81+ | 50,091 | 0.0184056 | **0.0190495** | +3.50% |

The longest-history group is the largest and has the lowest retrieval quality,
which suggests that a stronger long-context architecture or recency-aware
sampling would be a useful next experiment.

## Paired ablation audit

The two prediction files contain the same 78,921 user IDs, target IDs, and
history lengths in the same order. Hit transitions are:

| Outcome | Users |
|---|---:|
| Hit under both configurations | 1,364 |
| Multimodal only | 1,110 |
| No-MM only | 1,142 |
| Miss under both configurations | 75,305 |

The no-MM score delta is `+0.0005061`. A paired normal 95% interval is
`[-0.0002519, 0.0012641]`, which includes zero. Each configuration was trained
with one seed. The defensible conclusion is therefore that the current
multimodal fusion did not demonstrate a stable gain in this run—not that
multimodal information is generally ineffective or harmful.

## Training and ablation

| Epoch / step | Multimodal validation BCE | No-MM validation BCE |
|---|---:|---:|
| 1 / 441 | 0.2889 | 0.2899 |
| 2 / 882 | 0.2240 | 0.2236 |
| 3 / 1323 | 0.2043 | 0.2040 |

The BCE curves are nearly identical at this scale. Both final checkpoints were
evaluated with the same formal Top-10 protocol; the paired uncertainty and
single-seed limitation above still prevent a general causal claim.

## Reproducibility evidence

- Baseline, inference, ablation, and both formal offline-evaluation stages
  completed.
- Downloaded outputs were verified against remote SHA-256 manifests before
  remote temporary outputs were cleaned.
- The published SafeTensors checkpoint contains 48 tensors and matches the
  evaluated PyTorch state dict for every key, shape, dtype, and value.
- The model SHA-256 is
  `1d53197a6c09fca20ad1c24d702a92a58adfc972f77f7236be3283d065db859b`.
