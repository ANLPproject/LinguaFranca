# The probe as a verifier: which answer to keep (`pipeline_tfidf_v2_r2_shared_hybrid_probes_lr_C0.01_L12`)

Each candidate trajectory's failure score = max hop P(fail), from the probe (hidden states) or the text judge (Llama-3.2-3B, the pipeline's judge); both scored every candidate during the run. The candidate with the lower score is kept (ties → the original). Same candidates for every row; one restart per candidate. Correct = token F1 ≥ 0.5; CIs: paired bootstrap vs 'always restart after hop 2'.

## 2wiki (n = 500)

| system | accuracy | fixed | broken | vs always-restart-after-hop-2 [95% CI] |
|---|---|---|---|---|
| Original answer only (no intervention) | 0.342 | 0 | 0 | -0.240 [-0.284, -0.198] |
| Always restart after hop 2 (no verification) | 0.582 | 134 | 14 | – |
| **Always restart after hop 2 + probe verification** | 0.558 | 113 | 5 | -0.024 [-0.046, -0.004] |
| Always restart after hop 2 + judge verification (Llama-3.2-3B) | 0.550 | 110 | 6 | -0.032 [-0.056, -0.010] |
| **Probe trigger (τ = 0.650) + probe verification** | 0.540 | 103 | 4 | -0.042 [-0.068, -0.016] |
| **Best of 3 (original, hop-1 restart, hop-2 restart) chosen by the probe** | 0.552 | 111 | 6 | -0.030 [-0.062, +0.000] |
| Best of 3 chosen by the judge (Llama-3.2-3B) | 0.528 | 107 | 14 | -0.054 [-0.084, -0.026] |
| Best of 3 chosen at random | 0.476 | 78 | 11 | -0.106 [-0.142, -0.070] |
| Best of 3, oracle choice (upper bound) | 0.650 | 154 | 0 | +0.068 [+0.046, +0.090] |

## hotpotqa (n = 500)

| system | accuracy | fixed | broken | vs always-restart-after-hop-2 [95% CI] |
|---|---|---|---|---|
| Original answer only (no intervention) | 0.504 | 0 | 0 | -0.082 [-0.124, -0.036] |
| Always restart after hop 2 (no verification) | 0.586 | 89 | 48 | – |
| **Always restart after hop 2 + probe verification** | 0.558 | 63 | 36 | -0.028 [-0.052, -0.004] |
| Always restart after hop 2 + judge verification (Llama-3.2-3B) | 0.588 | 72 | 30 | +0.002 [-0.022, +0.024] |
| **Probe trigger (τ = 0.650) + probe verification** | 0.544 | 43 | 23 | -0.042 [-0.078, -0.006] |
| **Best of 3 (original, hop-1 restart, hop-2 restart) chosen by the probe** | 0.570 | 68 | 35 | -0.016 [-0.046, +0.012] |
| Best of 3 chosen by the judge (Llama-3.2-3B) | 0.590 | 82 | 39 | +0.004 [-0.026, +0.034] |
| Best of 3 chosen at random | 0.526 | 54 | 43 | -0.060 [-0.090, -0.026] |
| Best of 3, oracle choice (upper bound) | 0.724 | 110 | 0 | +0.138 [+0.108, +0.170] |

