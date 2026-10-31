# Trained text diagnoser vs the probe (`microsoft/deberta-v3-base`)

Same TF-IDF Phase 1 rows, labels and split as the probe (counterfactuals excluded). Train 5693 / validation 632 (10 % of the train question groups) / test 1586 hop rows. Input: question, step 1 (when checking step 2), the step to check, and the context (truncated first) — max 512 tokens (mean 370 used). 3 epochs, lr 2e-05, batch 8×2, balanced BCE; best epoch by validation AUROC (0.9247, 0.9594, 0.9660); τ = 0.948 (F0.5-optimal on validation). Probe: the deployed shared probe (τ = 0.650), stored test predictions on the identical rows. CIs: bootstrap over question groups.

## Hop-failure detection, test split

| rows | n | fail | detector | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − diagnoser, AUROC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| hops 1–2 | 1586 | 651 | **probe** | 0.954 [0.944, 0.964] | 0.899 | 0.803 | 0.878 | 0.063 | |
| hops 1–2 | 1586 | 651 | trained text diagnoser | 0.959 [0.949, 0.968] | 0.921 | 0.826 | 0.901 | 0.049 | -0.005 [-0.013, +0.004] |
| hop 1 | 798 | 139 | **probe** | 0.941 [0.919, 0.960] | 0.832 | 0.604 | 0.773 | 0.026 | |
| hop 1 | 798 | 139 | trained text diagnoser | 0.939 [0.913, 0.962] | 0.885 | 0.662 | 0.829 | 0.018 | +0.002 [-0.017, +0.022] |
| hop 2 | 788 | 512 | **probe** | 0.931 [0.912, 0.951] | 0.913 | 0.857 | 0.901 | 0.152 | |
| hop 2 | 788 | 512 | trained text diagnoser | 0.936 [0.915, 0.954] | 0.929 | 0.871 | 0.917 | 0.123 | -0.005 [-0.021, +0.012] |

Recall on failing hops by kind (at each detector's τ):

| failure kind | n | probe | diagnoser |
|---|---|---|---|
| hedge | 70 | 0.900 | 0.957 |
| confident (no hedge) | 581 | 0.792 | 0.811 |

Spearman ρ between probe and diagnoser scores: +0.85. Cost of one check: diagnoser 37.8 ms (batch 1, 370 tokens, a separate deberta-v3-base forward pass) vs the probe's one dot product on states the generator already computed (pipeline: 9.5 ms incl. pooling).

## End to end on `pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12` (one restart, same cached generations)

Diagnoser accuracy-optimal (θ1, θ2): {'fold0': (0.98, 0.06), 'fold1': (1.0, 0.06), 'all_2wiki': (0.08, 0.06)}.

### 2wiki (n = 500)

Deployable thresholds (fitted on training data only):

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.538 | 0.64 | 108 | 10 | – |
| Trained text diagnoser (τ = 0.948) | 0.534 | 0.64 | 107 | 11 | +0.004 [-0.016, +0.024] |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 | 0.454 | 0.46 | 63 | 7 | +0.084 [+0.050, +0.118] |
| Always restart after hop 2 | 0.554 | 0.98 | 126 | 20 | -0.016 [-0.038, +0.006] |

Accuracy-optimal thresholds (2-fold on 2Wiki; HotpotQA uses all 2Wiki):

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.554 | 0.78 | 122 | 16 | – |
| Trained text diagnoser, accuracy-optimal | 0.556 | 0.72 | 119 | 12 | -0.002 [-0.016, +0.012] |

Matched restart budget — accuracy, probe vs diagnoser (probe − diagnoser [95% CI]):

| budget | probe | diagnoser | difference |
|---|---|---|---|
| 10% | 0.380 | 0.376 | +0.004 [-0.018, +0.026] |
| 20% | 0.414 | 0.414 | +0.000 [-0.030, +0.030] |
| 30% | 0.446 | 0.462 | -0.016 [-0.044, +0.012] |
| 40% | 0.488 | 0.492 | -0.004 [-0.028, +0.022] |
| 50% | 0.506 | 0.516 | -0.010 [-0.030, +0.012] |
| 60% | 0.524 | 0.530 | -0.006 [-0.026, +0.014] |
| 70% | 0.548 | 0.540 | +0.008 [-0.010, +0.026] |
| 80% | 0.540 | 0.546 | -0.006 [-0.026, +0.012] |
| 90% | 0.522 | 0.520 | +0.002 [-0.024, +0.026] |

### hotpotqa (n = 500)

Deployable thresholds (fitted on training data only):

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.570 | 0.59 | 55 | 22 | – |
| Trained text diagnoser (τ = 0.948) | 0.570 | 0.72 | 67 | 34 | +0.000 [-0.030, +0.030] |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 | 0.526 | 0.23 | 24 | 13 | +0.044 [+0.010, +0.078] |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | -0.016 [-0.050, +0.020] |

Accuracy-optimal thresholds (2-fold on 2Wiki; HotpotQA uses all 2Wiki):

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.580 | 0.78 | 73 | 35 | – |
| Trained text diagnoser, accuracy-optimal | 0.574 | 0.92 | 81 | 46 | +0.006 [-0.026, +0.040] |

Matched restart budget — accuracy, probe vs diagnoser (probe − diagnoser [95% CI]):

| budget | probe | diagnoser | difference |
|---|---|---|---|
| 10% | 0.516 | 0.530 | -0.014 [-0.032, +0.002] |
| 20% | 0.536 | 0.544 | -0.008 [-0.030, +0.012] |
| 30% | 0.546 | 0.556 | -0.010 [-0.036, +0.016] |
| 40% | 0.556 | 0.556 | +0.000 [-0.026, +0.028] |
| 50% | 0.570 | 0.572 | -0.002 [-0.030, +0.026] |
| 60% | 0.570 | 0.576 | -0.006 [-0.034, +0.022] |
| 70% | 0.578 | 0.566 | +0.012 [-0.016, +0.040] |
| 80% | 0.574 | 0.584 | -0.010 [-0.040, +0.018] |
| 90% | 0.582 | 0.584 | -0.002 [-0.030, +0.024] |

### Live 2Wiki hops, AUROC [95% CI]

| hop | n | fail | probe | diagnoser |
|---|---|---|---|---|
| 1 | 499 | 85 | 0.947 [0.923, 0.968] | 0.933 [0.896, 0.965] |
| 2 | 492 | 327 | 0.925 [0.897, 0.948] | 0.931 [0.901, 0.955] |

(Kaggle T4 run of `scripts/text_diagnoser.py --run …`, seed 42.)
