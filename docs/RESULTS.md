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

| Metric | Value |
|---|---:|
| Hits@10 | 2,474 |
| HR@10 | 0.0313478035 |
| NDCG@10 | 0.0159694453 |
| Competition-style score | 0.0207367364 |
| Runtime | 241.503 s |

The score is computed with the competition-style weighting implemented by the
project's evaluation utility. The exact machine-readable record is available
at `metrics/offline_metrics.json`.

## History-length slices

| History length | Users | HR@10 | NDCG@10 | Score |
|---|---:|---:|---:|---:|
| 0–20 | 1,125 | 0.0382222 | 0.0184639 | 0.0245890 |
| 21–50 | 4,366 | 0.0387082 | 0.0201089 | 0.0258747 |
| 51–80 | 23,339 | 0.0371481 | 0.0189525 | 0.0245931 |
| 81+ | 50,091 | 0.0278493 | 0.0141627 | 0.0184056 |

The longest-history group is the largest and has the lowest retrieval quality,
which suggests that a stronger long-context architecture or recency-aware
sampling would be a useful next experiment.

## Training and ablation

| Epoch / step | Multimodal validation BCE | No-MM validation BCE |
|---|---:|---:|
| 1 / 441 | 0.2889 | 0.2899 |
| 2 / 882 | 0.2240 | 0.2236 |
| 3 / 1323 | 0.2043 | 0.2040 |

The BCE curves are nearly identical at this scale. Only the multimodal model
was run through the formal Top-10 offline protocol, so the table must not be
interpreted as proof that multimodal features do or do not improve retrieval.

## Reproducibility evidence

- Baseline, inference, ablation, and offline-evaluation stages completed.
- Downloaded outputs were verified against remote SHA-256 manifests before
  remote temporary outputs were cleaned.
- The published SafeTensors checkpoint contains 48 tensors and matches the
  evaluated PyTorch state dict for every key, shape, dtype, and value.
- The model SHA-256 is
  `1d53197a6c09fca20ad1c24d702a92a58adfc972f77f7236be3283d065db859b`.

