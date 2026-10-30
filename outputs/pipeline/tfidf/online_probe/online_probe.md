# A probe that stops a failing hop before it ends (TF-IDF Phase 1 split, block 12)

Train 6325 hops, test 1586 hops (fail 0.41); hop length median 21 tokens. Prefix probe: StandardScaler + LR (C = 0.01) on the running mean of the hop's block-12 states after each token t ≥ 3, plus log t; label = the hop's label. Thresholds from grouped 5-fold out-of-fold predictions inside the train split only. Deployed probe: τ = 0.650 at `</hopK>`.

Sanity: deployed probe on these features, test AUROC 0.9542 (bundle 0.9542).

## 1. Ranking quality by how much of the hop has been generated (test AUROC)

| prefix | prefix probe [95% CI] | deployed probe (full-hop fit) |
|---|---|---|
| 10 % | 0.834 [0.813, 0.854] | 0.710 |
| 20 % | 0.848 [0.829, 0.866] | 0.722 |
| 30 % | 0.858 [0.839, 0.875] | 0.757 |
| 40 % | 0.877 [0.860, 0.893] | 0.789 |
| 50 % | 0.885 [0.869, 0.901] | 0.792 |
| 60 % | 0.891 [0.875, 0.907] | 0.815 |
| 70 % | 0.904 [0.889, 0.919] | 0.860 |
| 80 % | 0.918 [0.905, 0.932] | 0.908 |
| 90 % | 0.927 [0.914, 0.941] | 0.943 |
| 100 % | 0.929 [0.917, 0.943] | 0.954 |

## 2. Online triggers on the test split (hop level)

θ (train OOF): F0.5-optimal 0.962; strict 0.993 (precision ≥ the deployed probe's train precision 0.903).

| trigger | precision | recall | F0.5 | FPR | fires before `</hopK>` (true failures) | median share of the hop at firing | hop tokens saved per caught failure |
|---|---|---|---|---|---|---|---|
| Deployed probe at `</hopK>` (now) | 0.899 | 0.803 | 0.878 | 0.063 | 0% | 100% | 0.0 |
| Prefix probe, online, F0.5 θ | 0.825 | 0.731 | 0.804 | 0.108 | 96% | 20% | 14.5 |
| Prefix probe, online, strict θ | 0.900 | 0.510 | 0.780 | 0.040 | 96% | 31% | 13.2 |
| **Hybrid**: strict online, else deployed at `</hopK>` | 0.884 | 0.820 | 0.871 | 0.075 | 60% | 70% | 8.2 |

Tokens saved are hop tokens not generated before the restart (the pipeline's restart cost is unchanged). The end-to-end effect is not simulated: every cached restart started from a complete hop.

## Note: the early signal is not in the words (checked locally, test split)

95 % of failing hops and 99 % of correct hops open with the identical words "From the context, I find"; hedges ("There is no information…") are 10.8 % of failing hops, and removing them leaves the first-token NLL AUROC almost unchanged (0.727 → 0.710). So the prefix probe's AUROC 0.83 after ~3 tokens is read from hidden states over (almost always) identical text.

(Kaggle run, `scripts/online_probe.py --stage all`; the ~1 GB prefix features were not kept.)
