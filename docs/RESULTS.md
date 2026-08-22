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

## Fixed-seed 2x2 aggregate metrics

The score is `0.31 * HR@10 + 0.69 * NDCG@10`.

| Variant | maxlen | MM | Hits@10 | HR@10 | NDCG@10 | Score | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|
| MM101 | 101 | on | 2,474 | 0.0313478035 | 0.0159694453 | 0.0207367364 | 241.503 s |
| no-MM101 | 101 | off | 2,506 | 0.0317532723 | 0.0165207806 | 0.0212428530 | 133.926 s |
| MM50 | 50 | on | 2,532 | 0.0320827156 | 0.0160502759 | 0.0210203322 | 212.630 s |
| **no-MM50** | 50 | off | **2,660** | **0.0337045907** | **0.0172092099** | **0.0223227779** | 137.461 s |

no-MM50 is the best measured point estimate, with a score `+7.65%` above
MM101. Runtime is reported for traceability but is not used as a controlled
selection metric because the runs shared hardware with unrelated load.

## Row alignment and paired statistics

All four prediction files contain the same 78,921 user IDs, target IDs, and
original history lengths in the same order.

| Paired effect (variant − reference) | Score delta | Relative delta | Normal 95% interval |
|---|---:|---:|---:|
| no-MM101 − MM101 | +0.0005061 | +2.44% | [-0.0002519, 0.0012641] |
| no-MM50 − MM50 | +0.0013024 | +6.20% | [0.0005414, 0.0020635] |
| MM50 − MM101 | +0.0002836 | +1.37% | [-0.0005245, 0.0010917] |
| no-MM50 − no-MM101 | +0.0010799 | +5.08% | [0.0001977, 0.0019621] |
| Difference-in-differences interaction | +0.0007963 | — | [-0.0002521, 0.0018447] |

The no-MM effect at maxlen 50 and the short-window effect without MM are
positive for this fixed evaluation population. The overall interaction
interval still includes zero, so the 2x2 result does not establish a stable
interaction between recency and multimodal removal. Training-seed uncertainty
is not estimated because each configuration was trained once.

## Row-aligned history-length slices

| Original history | Users | MM101 | no-MM101 | MM50 | no-MM50 | Point winner |
|---|---:|---:|---:|---:|---:|---|
| 0–20 | 1,125 | 0.0245890 | **0.0251632** | 0.0208526 | 0.0242034 | no-MM101 |
| 21–50 | 4,366 | **0.0258747** | 0.0251520 | 0.0224903 | 0.0222058 | MM101 |
| 51–80 | 23,339 | 0.0245931 | **0.0250301** | 0.0241666 | 0.0239811 | no-MM101 |
| 81+ | 50,091 | 0.0184056 | 0.0190495 | 0.0194300 | **0.0215181** | no-MM50 |

The 81+ slice contains 63.47% of evaluated users. In that slice, no-MM50 is
`+12.96%` above no-MM101 with a paired interval
`[0.0013953, 0.0035419]`; its slice interaction interval is also positive.
Shorter-history slice winners vary and their relevant intervals include zero.
The defensible interpretation is that the overall point-estimate gain is
concentrated in long histories, while confirmation across training seeds and
correction for exploratory slice comparisons remain future work.

Machine-readable evidence is in `metrics/offline_metrics*.json`,
`metrics/ablation_comparison.json`, and `metrics/four_way_comparison.json`.

## Training and ablation

| Epoch / step | MM101 | no-MM101 | MM50 | no-MM50 |
|---|---:|---:|---:|---:|
| 1 / 441 | 0.2889 | 0.2899 | 0.2840 | 0.2820 |
| 2 / 882 | 0.2240 | 0.2236 | 0.2263 | 0.2247 |
| 3 / 1323 | 0.2043 | **0.2040** | 0.2061 | 0.2055 |

Validation BCE does not rank the retrieval variants in the same order as
Top-10 score. Selection therefore follows the held-out retrieval metric rather
than training loss, while retaining the single-seed limitation.

## Reproducibility evidence

- All four fixed-seed formal offline evaluations completed on the same 78,921
  row-aligned examples.
- Overall, paired, interaction, and four history-slice metrics were recomputed
  directly from the prediction arrays.
- Downloaded outputs were verified against remote SHA-256 manifests before
  remote temporary outputs were cleaned.
- The published SafeTensors checkpoint contains 48 tensors and matches the
  evaluated PyTorch state dict for every key, shape, dtype, and value.
- The model SHA-256 is
  `1d53197a6c09fca20ad1c24d702a92a58adfc972f77f7236be3283d065db859b`.
- The MM50 and no-MM50 PyTorch model SHA-256 values are respectively
  `69f190d60abcacd5496a2aa279db4fe7c61bb8472a597dfff23f23946251439d` and
  `0d3352689daf095dfc1acff9915c53d517a6164549c5f1e5b9c6d44bacc81a6e`;
  both are retained in the private evidence archive.
