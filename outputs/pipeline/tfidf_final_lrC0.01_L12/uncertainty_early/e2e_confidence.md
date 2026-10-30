# End-to-end: probe vs token-confidence triggers (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

Same cached generations and restarts as every other system; one restart; correct = token F1 ≥ 0.5; CIs: paired bootstrap over questions. Confidence signals are computed at the same hop boundary as the probe (as cheap: no extra model call), from one teacher-forced pass over each first-stage trajectory.

Parity: the probe on this pass's features vs the run's stored features: |ΔP(fail)| mean 0.0007, max 0.1197 over 1977 hops. Restartable hops without confidence signals (never trigger): 0.

## 1. Deployable thresholds (each fitted on the probe's train split only)

Probe: its deployed τ. Each confidence trigger: its own F0.5-optimal threshold on the train hops (max_nll 0.77, mean_nll 0.11, max_ent 1.52, mean_ent 0.28). Last column vs the probe.

### 2wiki (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.538 | 0.64 | 108 | 10 | – |
| max token NLL (FLARE: min token prob) | 0.454 | 0.62 | 68 | 12 | +0.084 [+0.046, +0.118] |
| mean token NLL | 0.456 | 0.49 | 66 | 9 | +0.082 [+0.046, +0.120] |
| max token entropy (DRAGIN-style) | 0.456 | 0.51 | 65 | 8 | +0.082 [+0.046, +0.116] |
| mean token entropy | 0.454 | 0.42 | 62 | 6 | +0.084 [+0.050, +0.118] |
| Always restart after hop 2 | 0.554 | 0.98 | 126 | 20 | -0.016 [-0.038, +0.006] |

### hotpotqa (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.570 | 0.59 | 55 | 22 | – |
| max token NLL (FLARE: min token prob) | 0.564 | 0.70 | 67 | 37 | +0.006 [-0.026, +0.040] |
| mean token NLL | 0.552 | 0.55 | 54 | 30 | +0.018 [-0.010, +0.050] |
| max token entropy (DRAGIN-style) | 0.554 | 0.66 | 61 | 36 | +0.016 [-0.018, +0.050] |
| mean token entropy | 0.570 | 0.53 | 59 | 26 | +0.000 [-0.030, +0.034] |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | -0.016 [-0.050, +0.020] |

## 2. Accuracy-optimal thresholds (2-fold on 2Wiki; HotpotQA uses all 2Wiki), last column vs the accuracy-optimal probe

### 2wiki

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.554 | 0.78 | 122 | 16 | – |
| max token NLL (FLARE: min token prob), accuracy-optimal | 0.534 | 0.89 | 114 | 18 | +0.020 [-0.002, +0.042] |
| mean token NLL, accuracy-optimal | 0.542 | 0.96 | 120 | 20 | +0.012 [-0.004, +0.028] |
| max token entropy (DRAGIN-style), accuracy-optimal | 0.472 | 0.99 | 83 | 18 | +0.082 [+0.046, +0.120] |
| mean token entropy, accuracy-optimal | 0.536 | 0.90 | 116 | 19 | +0.018 [+0.002, +0.036] |

### hotpotqa

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.580 | 0.78 | 73 | 35 | – |
| max token NLL (FLARE: min token prob), accuracy-optimal | 0.568 | 0.95 | 84 | 52 | +0.012 [-0.018, +0.046] |
| mean token NLL, accuracy-optimal | 0.580 | 0.97 | 85 | 47 | +0.000 [-0.024, +0.024] |
| max token entropy (DRAGIN-style), accuracy-optimal | 0.566 | 0.99 | 86 | 55 | +0.014 [-0.020, +0.052] |
| mean token entropy, accuracy-optimal | 0.576 | 0.92 | 81 | 45 | +0.004 [-0.020, +0.030] |

## 3. Matched restart budget. Cells: accuracy (probe − trigger [95% CI])

### 2wiki

| budget | Probe | max token NLL (FLARE: min token prob) | mean token NLL | max token entropy (DRAGIN-style) | mean token entropy |
|---|---|---|---|---|---|
| 10% | **0.380** | 0.376 (+0.004 [-0.018, +0.026]) | 0.388 (-0.008 [-0.032, +0.014]) | 0.370 (+0.010 [-0.012, +0.032]) | 0.384 (-0.004 [-0.026, +0.018]) |
| 20% | **0.414** | 0.402 (+0.012 [-0.014, +0.038]) | 0.410 (+0.004 [-0.024, +0.030]) | 0.398 (+0.016 [-0.012, +0.044]) | 0.412 (+0.002 [-0.026, +0.032]) |
| 30% | **0.446** | 0.424 (+0.022 [-0.010, +0.054]) | 0.426 (+0.020 [-0.012, +0.054]) | 0.430 (+0.016 [-0.018, +0.048]) | 0.438 (+0.008 [-0.024, +0.042]) |
| 40% | **0.488** | 0.430 (+0.058 [+0.022, +0.092]) | 0.438 (+0.050 [+0.016, +0.084]) | 0.440 (+0.048 [+0.012, +0.082]) | 0.452 (+0.036 [+0.002, +0.070]) |
| 50% | **0.506** | 0.440 (+0.066 [+0.030, +0.100]) | 0.452 (+0.054 [+0.018, +0.090]) | 0.454 (+0.052 [+0.016, +0.086]) | 0.472 (+0.034 [+0.002, +0.066]) |
| 60% | **0.524** | 0.452 (+0.072 [+0.034, +0.108]) | 0.476 (+0.048 [+0.012, +0.082]) | 0.472 (+0.052 [+0.016, +0.084]) | 0.482 (+0.042 [+0.010, +0.074]) |
| 70% | **0.548** | 0.464 (+0.084 [+0.048, +0.118]) | 0.480 (+0.068 [+0.034, +0.102]) | 0.458 (+0.090 [+0.056, +0.124]) | 0.478 (+0.070 [+0.038, +0.104]) |
| 80% | **0.540** | 0.468 (+0.072 [+0.040, +0.104]) | 0.474 (+0.066 [+0.032, +0.102]) | 0.460 (+0.080 [+0.046, +0.116]) | 0.476 (+0.064 [+0.030, +0.100]) |
| 90% | **0.522** | 0.470 (+0.052 [+0.020, +0.082]) | 0.456 (+0.066 [+0.032, +0.098]) | 0.458 (+0.064 [+0.032, +0.098]) | 0.466 (+0.056 [+0.022, +0.088]) |

### hotpotqa

| budget | Probe | max token NLL (FLARE: min token prob) | mean token NLL | max token entropy (DRAGIN-style) | mean token entropy |
|---|---|---|---|---|---|
| 10% | **0.516** | 0.518 (-0.002 [-0.022, +0.018]) | 0.520 (-0.004 [-0.022, +0.016]) | 0.520 (-0.004 [-0.024, +0.016]) | 0.520 (-0.004 [-0.024, +0.016]) |
| 20% | **0.536** | 0.526 (+0.010 [-0.018, +0.038]) | 0.530 (+0.006 [-0.020, +0.030]) | 0.522 (+0.014 [-0.014, +0.044]) | 0.524 (+0.012 [-0.012, +0.036]) |
| 30% | **0.546** | 0.532 (+0.014 [-0.014, +0.044]) | 0.544 (+0.002 [-0.026, +0.030]) | 0.526 (+0.020 [-0.008, +0.050]) | 0.552 (-0.006 [-0.036, +0.024]) |
| 40% | **0.556** | 0.534 (+0.022 [-0.010, +0.054]) | 0.556 (+0.000 [-0.030, +0.032]) | 0.544 (+0.012 [-0.020, +0.044]) | 0.568 (-0.012 [-0.040, +0.018]) |
| 50% | **0.570** | 0.546 (+0.024 [-0.010, +0.058]) | 0.554 (+0.016 [-0.012, +0.046]) | 0.548 (+0.022 [-0.012, +0.056]) | 0.572 (-0.002 [-0.032, +0.030]) |
| 60% | **0.570** | 0.566 (+0.004 [-0.028, +0.036]) | 0.560 (+0.010 [-0.020, +0.042]) | 0.550 (+0.020 [-0.014, +0.054]) | 0.560 (+0.010 [-0.022, +0.042]) |
| 70% | **0.578** | 0.564 (+0.014 [-0.018, +0.048]) | 0.560 (+0.018 [-0.012, +0.050]) | 0.552 (+0.026 [-0.006, +0.056]) | 0.554 (+0.024 [-0.006, +0.056]) |
| 80% | **0.574** | 0.560 (+0.014 [-0.016, +0.046]) | 0.566 (+0.008 [-0.024, +0.040]) | 0.568 (+0.006 [-0.022, +0.036]) | 0.560 (+0.014 [-0.018, +0.046]) |
| 90% | **0.582** | 0.542 (+0.040 [+0.012, +0.070]) | 0.564 (+0.018 [-0.010, +0.048]) | 0.564 (+0.018 [-0.008, +0.046]) | 0.566 (+0.016 [-0.012, +0.046]) |

## 4. Hop-level detection on live 2Wiki hops, AUROC [95% CI]

| hop | n | fail | probe | max token NLL (FLARE: min token prob) | mean token NLL | max token entropy (DRAGIN-style) | mean token entropy |
|---|---|---|---|---|---|---|---|
| 1 | 499 | 85 | 0.947 [0.923, 0.968] | 0.620 [0.551, 0.686] | 0.697 [0.630, 0.755] | 0.663 [0.592, 0.726] | 0.727 [0.658, 0.782] |
| 2 | 492 | 327 | 0.925 [0.897, 0.948] | 0.651 [0.600, 0.699] | 0.707 [0.661, 0.755] | 0.676 [0.625, 0.725] | 0.734 [0.687, 0.783] |