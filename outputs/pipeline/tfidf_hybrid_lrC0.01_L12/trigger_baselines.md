# Is the probe needed? Trivial triggers vs the probe (`pipeline_tfidf_v2_r2_shared_hybrid_probes_lr_C0.01_L12`)

All triggers act on the same cached generations: same model, prompt, repair and restart outputs, one restart. Only the decision *whether and where to restart* differs. Correct = token F1 ≥ 0.5. CIs: paired bootstrap of probe − trigger accuracy.

Trained text trigger: TF-IDF (1–2-grams) + logistic regression on the hop text, trained on the probe's train split (6325 labelled hops 1–2), threshold F0.5-optimal on out-of-fold predictions.

## 2wiki (n = 500)

| trigger | accuracy | restart rate | fixed | broken | gen tokens | probe − trigger [95% CI] |
|---|---|---|---|---|---|---|
| No intervention | 0.342 | 0.00 | 0 | 0 | 143.4 | +0.216 [+0.180, +0.256] |
| **Probe trigger (ours)** | 0.558 | 0.64 | 113 | 5 | 160.2 | – |
| Always restart after hop 2 | 0.582 | 0.98 | 134 | 14 | 180.7 | -0.024 [-0.046, -0.002] |
| Always restart after hop 1 | 0.434 | 1.00 | 68 | 22 | 166.3 | +0.124 [+0.084, +0.166] |
| Hedge-keyword trigger | 0.372 | 0.08 | 15 | 0 | 140.9 | +0.186 [+0.148, +0.224] |
| Trained text trigger (TF-IDF+LR on hop text, τ=0.62) | 0.538 | 0.70 | 106 | 8 | 163.8 | +0.020 [-0.004, +0.046] |
| Random trigger at the probe's per-hop restart rate (200 seeds) | 0.477 ± 0.012 | 0.63 | 77.4 | 9.8 | 165.6 | +0.081 (all 200 seeds below the probe: True) |

## hotpotqa (n = 500)

| trigger | accuracy | restart rate | fixed | broken | gen tokens | probe − trigger [95% CI] |
|---|---|---|---|---|---|---|
| No intervention | 0.504 | 0.00 | 0 | 0 | 134.3 | +0.046 [+0.012, +0.080] |
| **Probe trigger (ours)** | 0.550 | 0.59 | 53 | 30 | 155.5 | – |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | 192.2 | -0.036 [-0.068, -0.004] |
| Always restart after hop 1 | 0.570 | 1.00 | 85 | 52 | 163.5 | -0.020 [-0.056, +0.018] |
| Hedge-keyword trigger | 0.506 | 0.00 | 1 | 0 | 135.2 | +0.044 [+0.010, +0.078] |
| Trained text trigger (TF-IDF+LR on hop text, τ=0.62) | 0.576 | 0.82 | 79 | 43 | 181.8 | -0.026 [-0.058, +0.006] |
| Random trigger at the probe's per-hop restart rate (200 seeds) | 0.551 ± 0.012 | 0.57 | 51.7 | 28.4 | 163.3 | -0.001 (all 200 seeds below the probe: False) |

