# Is the probe needed? Trivial triggers vs the probe (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

All triggers act on the same cached generations: same model, prompt, repair and restart outputs, one restart. Only the decision *whether and where to restart* differs. Correct = token F1 ≥ 0.5. CIs: paired bootstrap of probe − trigger accuracy.

Trained text trigger: TF-IDF (1–2-grams) + logistic regression on the hop text, trained on the probe's train split (6325 labelled hops 1–2), threshold F0.5-optimal on out-of-fold predictions.

## 2wiki (n = 500)

| trigger | accuracy | restart rate | fixed | broken | gen tokens | probe − trigger [95% CI] |
|---|---|---|---|---|---|---|
| No intervention | 0.342 | 0.00 | 0 | 0 | 143.4 | +0.196 [+0.156, +0.236] |
| **Probe trigger (ours)** | 0.538 | 0.64 | 108 | 10 | 160.1 | – |
| Always restart after hop 2 | 0.554 | 0.98 | 126 | 20 | 181.4 | -0.016 [-0.038, +0.006] |
| Always restart after hop 1 | 0.448 | 1.00 | 74 | 21 | 167.9 | +0.090 [+0.052, +0.126] |
| Hedge-keyword trigger | 0.364 | 0.08 | 11 | 0 | 141.2 | +0.174 [+0.136, +0.212] |
| Trained text trigger (TF-IDF+LR on hop text, τ=0.62) | 0.516 | 0.70 | 98 | 11 | 163.6 | +0.022 [-0.004, +0.048] |
| Random trigger at the probe's per-hop restart rate (200 seeds) | 0.465 ± 0.013 | 0.63 | 74.1 | 12.6 | 166.1 | +0.073 (all 200 seeds below the probe: True) |

## hotpotqa (n = 500)

| trigger | accuracy | restart rate | fixed | broken | gen tokens | probe − trigger [95% CI] |
|---|---|---|---|---|---|---|
| No intervention | 0.504 | 0.00 | 0 | 0 | 134.3 | +0.066 [+0.034, +0.098] |
| **Probe trigger (ours)** | 0.570 | 0.59 | 55 | 22 | 153.7 | – |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | 192.3 | -0.016 [-0.050, +0.020] |
| Always restart after hop 1 | 0.568 | 1.00 | 85 | 53 | 164.8 | +0.002 [-0.032, +0.038] |
| Hedge-keyword trigger | 0.506 | 0.00 | 1 | 0 | 135.2 | +0.064 [+0.032, +0.096] |
| Trained text trigger (TF-IDF+LR on hop text, τ=0.62) | 0.576 | 0.82 | 75 | 39 | 181.9 | -0.006 [-0.038, +0.026] |
| Random trigger at the probe's per-hop restart rate (200 seeds) | 0.550 ± 0.012 | 0.57 | 51.6 | 28.6 | 163.7 | +0.020 (all 200 seeds below the probe: False) |


## Detection on live 2Wiki hops: probe vs the trained text trigger

| hop | AUROC probe | AUROC trained text | failure recall (probe / text) | false alarms (probe / text) |
|---|---|---|---|---|
| 1 (n = 475, 80 failures) | **0.949** | 0.789 | 0.61 / 0.36 | 0.03 / 0.01 |
| 2 (n = 468, 307 failures) | **0.928** | 0.677 | 0.86 / 0.77 | 0.16 / 0.56 |

## Accuracy at a matched restart budget (same repair; restart at the earliest hop above a threshold chosen to hit the budget)

Judges really decide after the full answer; here only their *choice* of questions and hops is compared. No CIs.

| dataset | budget | probe | trained text | judge (Llama-3.2-3B) | judge (Qwen2.5-7B) |
|---|---|---|---|---|---|
| 2Wiki | 20% | 0.414 | 0.414 | 0.392 | 0.386 |
| 2Wiki | 30% | 0.446 | 0.448 | 0.408 | 0.422 |
| 2Wiki | 40% | **0.488** | 0.474 | 0.436 | 0.448 |
| 2Wiki | 50% | **0.506** | 0.498 | 0.458 | 0.490 |
| 2Wiki | 60% | **0.524** | 0.502 | 0.456 | 0.504 |
| 2Wiki | 70% | **0.546** | 0.510 | 0.454 | 0.520 |
| HotpotQA | 20% | 0.536 | 0.534 | 0.518 | 0.528 |
| HotpotQA | 30% | 0.546 | 0.542 | 0.530 | **0.552** |
| HotpotQA | 40% | **0.556** | 0.544 | 0.548 | 0.552 |
| HotpotQA | 50% | **0.570** | 0.548 | 0.548 | 0.548 |
| HotpotQA | 60% | **0.572** | 0.566 | 0.558 | 0.550 |
| HotpotQA | 70% | **0.578** | 0.568 | 0.568 | 0.564 |

## Reading

- Most of the end-to-end gain comes from the repair, because hop 2 fails 65% of the time (the first stage usually drops the hop-2 paragraph). Restarting after every hop 2 reaches the probe's accuracy (2Wiki 0.554 vs 0.538, n.s.), with 98% restarts vs 64% and twice the broken answers (20 vs 10).
- The probe's value is **selectivity**. It detects failing hops far better than a text classifier trained on the same labels (AUROC 0.93–0.95 vs 0.68–0.79). It beats random restarts at the same rate by +7.3 points (all 200 seeds). At a matched restart budget it is the best or tied-best trigger from 40% upward (2Wiki: +2 to +4 points over the next best at 60–70%).
- The paper should not claim that the probe's accuracy beats trigger-free strategies. It should claim equal accuracy with fewer restarts and fewer broken answers, earlier detection, and better accuracy at any fixed restart budget.
