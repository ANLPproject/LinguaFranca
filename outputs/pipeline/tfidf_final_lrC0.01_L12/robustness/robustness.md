# Robustness checks and extra analyses (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

Items: 1000 (2wiki 500, hotpotqa 500). All systems act on the same cached generations; correct = token F1 ≥ 0.5; CIs are paired bootstraps over questions (2,000 resamples). Probe: `outputs/pipeline/tfidf/probes_lr_C0.01_L12` (τ = 0.650); old probe: `outputs/pipeline/tfidf/probes` (layer 10, τ = 0.639).

## 1. Fair thresholds: every trigger with the same accuracy-optimal (θ1, θ2)

The headline compares the probe at its train-CV τ with judges at their own yes/no decision (θ = 0.5). Here every trigger also gets the procedure the probe's 'accuracy-optimal τ' row uses: (θ1, θ2) maximising accuracy on one half of the 2Wiki questions, applied to the other half; HotpotQA uses the fit on all 2Wiki. The last column compares with the accuracy-optimal probe.

Fitted thresholds (fold 0 / fold 1 / all 2Wiki): probe: (0.78, 0.18) / (1.0, 0.2) / (0.86, 0.18); Llama-3.2-3B-Instruct: (0.94, 0.02) / (1.0, 0.0) / (1.0, 0.0); Llama-3.1-8B-Instruct: (0.98, 0.0) / (0.84, 0.0) / (0.98, 0.0); Qwen2.5-3B-Instruct: (1.0, 0.0) / (1.0, 0.0) / (1.0, 0.0); Qwen2.5-7B-Instruct: (0.22, 0.0) / (1.0, 0.0) / (1.0, 0.0)

### 2wiki (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| Probe, train-CV τ (headline) | 0.538 | 0.64 | 108 | 10 | +0.016 [-0.002, +0.036] |
| **Probe, accuracy-optimal** | 0.554 | 0.78 | 122 | 16 | – |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 (headline) | 0.454 | 0.46 | 63 | 7 | +0.100 [+0.066, +0.136] |
| Judge Llama-3.2-3B-Instruct, accuracy-optimal | 0.546 | 0.93 | 122 | 20 | +0.008 [-0.008, +0.024] |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 (headline) | 0.410 | 0.27 | 37 | 3 | +0.144 [+0.106, +0.184] |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.558 | 0.99 | 127 | 19 | -0.004 [-0.018, +0.010] |
| Judge Qwen2.5-3B-Instruct, θ = 0.5 (headline) | 0.456 | 0.89 | 73 | 16 | +0.098 [+0.060, +0.136] |
| Judge Qwen2.5-3B-Instruct, accuracy-optimal | 0.518 | 0.99 | 109 | 21 | +0.036 [+0.012, +0.060] |
| Judge Qwen2.5-7B-Instruct, θ = 0.5 (headline) | 0.508 | 0.63 | 91 | 8 | +0.046 [+0.016, +0.078] |
| Judge Qwen2.5-7B-Instruct, accuracy-optimal | 0.546 | 0.98 | 123 | 21 | +0.008 [-0.010, +0.026] |
| Always restart after hop 2 | 0.554 | 0.98 | 126 | 20 | +0.000 [-0.012, +0.012] |

### hotpotqa (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| Probe, train-CV τ (headline) | 0.570 | 0.59 | 55 | 22 | +0.010 [-0.012, +0.034] |
| **Probe, accuracy-optimal** | 0.580 | 0.78 | 73 | 35 | – |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 (headline) | 0.526 | 0.23 | 24 | 13 | +0.054 [+0.018, +0.092] |
| Judge Llama-3.2-3B-Instruct, accuracy-optimal | 0.586 | 0.97 | 89 | 48 | -0.006 [-0.030, +0.018] |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 (headline) | 0.520 | 0.16 | 17 | 9 | +0.060 [+0.022, +0.100] |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.586 | 0.97 | 89 | 48 | -0.006 [-0.030, +0.018] |
| Judge Qwen2.5-3B-Instruct, θ = 0.5 (headline) | 0.556 | 0.75 | 68 | 42 | +0.024 [-0.010, +0.058] |
| Judge Qwen2.5-3B-Instruct, accuracy-optimal | 0.574 | 0.98 | 88 | 53 | +0.006 [-0.020, +0.034] |
| Judge Qwen2.5-7B-Instruct, θ = 0.5 (headline) | 0.546 | 0.49 | 49 | 28 | +0.034 [+0.002, +0.068] |
| Judge Qwen2.5-7B-Instruct, accuracy-optimal | 0.582 | 0.97 | 89 | 50 | -0.002 [-0.026, +0.024] |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | -0.006 [-0.030, +0.018] |

## 2. Old (C = 1, block 10) vs tuned (C = 0.01, block 12) probe on the same generations

Old probe accuracy-optimal (θ1, θ2): {'fold0': (0.94, 0.0), 'fold1': (1.0, 0.08), 'all_2wiki': (1.0, 0.12)}. Last column: tuned − old.

### 2wiki

| threshold | tuned | old | tuned − old [95% CI] | restart rate tuned / old |
|---|---|---|---|---|
| train-CV τ | 0.538 | 0.506 | +0.032 [+0.016, +0.050] | 0.64 / 0.60 |
| accuracy-optimal | 0.554 | 0.548 | +0.006 [+0.000, +0.014] | 0.78 / 0.92 |

### hotpotqa

| threshold | tuned | old | tuned − old [95% CI] | restart rate tuned / old |
|---|---|---|---|---|
| train-CV τ | 0.570 | 0.560 | +0.010 [-0.016, +0.036] | 0.59 / 0.59 |
| accuracy-optimal | 0.580 | 0.564 | +0.016 [-0.008, +0.040] | 0.78 / 0.76 |

Hop-level detection on live 2Wiki hops (AUROC [95% CI]):

| hop | n | fail | tuned | old |
|---|---|---|---|---|
| 1 | 499 | 85 | 0.947 [0.923, 0.968] | 0.918 [0.884, 0.950] |
| 2 | 492 | 327 | 0.925 [0.897, 0.948] | 0.902 [0.873, 0.928] |

## 3. Answer forcing: accuracy with and without forced answers

Generations that hit the token cap without `<answer>` get an answer forced from hops 1–2 (stage `force`). 'without' scores those as empty (wrong). 'forced' = how many final answers of the system come from forcing.

### 2wiki

| system | with forcing | without | forced answers | probe − system, without forcing [95% CI] |
|---|---|---|---|---|
| No intervention | 0.342 | 0.334 | 66 | +0.194 [+0.154, +0.234] |
| Probe trigger (ours) | 0.538 | 0.528 | 51 | – |
| Always retrieve | 0.356 | 0.344 | 67 | +0.184 [+0.140, +0.226] |
| Always restart after hop 2 | 0.554 | 0.540 | 51 | -0.012 [-0.034, +0.010] |
| Judge Llama-3.2-3B-Instruct | 0.454 | 0.440 | 57 | +0.088 [+0.054, +0.122] |
| Judge Llama-3.1-8B-Instruct | 0.410 | 0.396 | 57 | +0.132 [+0.094, +0.168] |
| Judge Qwen2.5-3B-Instruct | 0.456 | 0.436 | 61 | +0.092 [+0.056, +0.130] |
| Judge Qwen2.5-7B-Instruct | 0.508 | 0.494 | 52 | +0.034 [+0.004, +0.064] |

### hotpotqa

| system | with forcing | without | forced answers | probe − system, without forcing [95% CI] |
|---|---|---|---|---|
| No intervention | 0.504 | 0.488 | 39 | +0.064 [+0.030, +0.096] |
| Probe trigger (ours) | 0.570 | 0.552 | 28 | – |
| Always retrieve | 0.546 | 0.520 | 46 | +0.032 [-0.004, +0.072] |
| Always restart after hop 2 | 0.586 | 0.562 | 40 | -0.010 [-0.044, +0.026] |
| Judge Llama-3.2-3B-Instruct | 0.526 | 0.510 | 37 | +0.042 [+0.006, +0.078] |
| Judge Llama-3.1-8B-Instruct | 0.520 | 0.506 | 38 | +0.046 [+0.012, +0.082] |
| Judge Qwen2.5-3B-Instruct | 0.556 | 0.530 | 37 | +0.022 [-0.014, +0.058] |
| Judge Qwen2.5-7B-Instruct | 0.546 | 0.530 | 29 | +0.022 [-0.010, +0.056] |

## 4. Accuracy at a matched restart budget (paired-bootstrap CIs)

Each trigger restarts at its earliest hop whose score exceeds the threshold that gives the budget's restart rate (one threshold for both hops). Judges' token cost is ignored here (they need the full answer first): only their choice of questions and hops is compared. Cells: accuracy (probe − trigger [95% CI]).

### 2wiki

| budget | Probe (tuned) | Probe (old) | Judge Llama-3.2-3B-Instruct | Judge Llama-3.1-8B-Instruct | Judge Qwen2.5-3B-Instruct | Judge Qwen2.5-7B-Instruct | Text trigger (TF-IDF+LR) |
|---|---|---|---|---|---|---|---|
| 10% | **0.380** (rate 0.10) | 0.374 (+0.006 [-0.012, +0.024]) | 0.370 (+0.010 [-0.010, +0.030]) | 0.366 (+0.014 [-0.006, +0.034]) | 0.416 (-0.036 [-0.068, -0.006]) | 0.386 (-0.006 [-0.030, +0.020]) | 0.364 (+0.016 [-0.004, +0.034]) |
| 20% | **0.414** (rate 0.20) | 0.402 (+0.012 [-0.008, +0.032]) | 0.392 (+0.022 [-0.006, +0.050]) | 0.394 (+0.020 [-0.010, +0.050]) | 0.416 (-0.002 [-0.036, +0.032]) | 0.392 (+0.022 [-0.008, +0.052]) | 0.414 (+0.000 [-0.028, +0.028]) |
| 30% | **0.446** (rate 0.30) | 0.436 (+0.010 [-0.014, +0.034]) | 0.410 (+0.036 [+0.006, +0.066]) | 0.422 (+0.024 [-0.010, +0.060]) | 0.416 (+0.030 [-0.006, +0.066]) | 0.418 (+0.028 [-0.004, +0.062]) | 0.448 (-0.002 [-0.032, +0.030]) |
| 40% | **0.488** (rate 0.40) | 0.464 (+0.024 [+0.004, +0.046]) | 0.434 (+0.054 [+0.022, +0.088]) | 0.446 (+0.042 [+0.006, +0.078]) | 0.416 (+0.072 [+0.034, +0.110]) | 0.448 (+0.040 [+0.006, +0.076]) | 0.474 (+0.014 [-0.016, +0.042]) |
| 50% | **0.506** (rate 0.50) | 0.490 (+0.016 [+0.000, +0.034]) | 0.458 (+0.048 [+0.014, +0.082]) | 0.448 (+0.058 [+0.022, +0.094]) | 0.446 (+0.060 [+0.024, +0.098]) | 0.490 (+0.016 [-0.018, +0.050]) | 0.496 (+0.010 [-0.020, +0.038]) |
| 60% | **0.524** (rate 0.60) | 0.506 (+0.018 [+0.002, +0.034]) | 0.458 (+0.066 [+0.028, +0.104]) | 0.458 (+0.066 [+0.030, +0.102]) | 0.456 (+0.068 [+0.030, +0.104]) | 0.504 (+0.020 [-0.014, +0.052]) | 0.502 (+0.022 [-0.008, +0.052]) |
| 70% | **0.548** (rate 0.70) | 0.528 (+0.020 [+0.006, +0.036]) | 0.454 (+0.094 [+0.060, +0.132]) | 0.466 (+0.082 [+0.048, +0.116]) | 0.466 (+0.082 [+0.046, +0.118]) | 0.520 (+0.028 [+0.000, +0.054]) | 0.512 (+0.036 [+0.008, +0.062]) |
| 80% | **0.540** (rate 0.80) | 0.528 (+0.012 [-0.004, +0.028]) | 0.442 (+0.098 [+0.062, +0.134]) | 0.462 (+0.078 [+0.046, +0.112]) | 0.462 (+0.078 [+0.042, +0.114]) | 0.528 (+0.012 [-0.014, +0.038]) | 0.532 (+0.008 [-0.016, +0.032]) |
| 90% | **0.522** (rate 0.90) | 0.514 (+0.008 [-0.016, +0.032]) | 0.444 (+0.078 [+0.048, +0.110]) | 0.472 (+0.050 [+0.018, +0.082]) | 0.456 (+0.066 [+0.034, +0.100]) | 0.502 (+0.020 [-0.010, +0.050]) | 0.548 (-0.026 [-0.052, +0.000]) |

### hotpotqa

| budget | Probe (tuned) | Probe (old) | Judge Llama-3.2-3B-Instruct | Judge Llama-3.1-8B-Instruct | Judge Qwen2.5-3B-Instruct | Judge Qwen2.5-7B-Instruct | Text trigger (TF-IDF+LR) |
|---|---|---|---|---|---|---|---|
| 10% | **0.516** (rate 0.10) | 0.532 (-0.016 [-0.034, +0.000]) | 0.518 (-0.002 [-0.024, +0.018]) | 0.508 (+0.008 [-0.012, +0.030]) | 0.520 (-0.004 [-0.030, +0.022]) | 0.520 (-0.004 [-0.026, +0.018]) | 0.516 (+0.000 [-0.020, +0.020]) |
| 20% | **0.536** (rate 0.20) | 0.534 (+0.002 [-0.018, +0.024]) | 0.520 (+0.016 [-0.010, +0.044]) | 0.520 (+0.016 [-0.012, +0.046]) | 0.520 (+0.016 [-0.012, +0.046]) | 0.526 (+0.010 [-0.016, +0.038]) | 0.534 (+0.002 [-0.020, +0.026]) |
| 30% | **0.546** (rate 0.30) | 0.542 (+0.004 [-0.018, +0.028]) | 0.530 (+0.016 [-0.014, +0.046]) | 0.530 (+0.016 [-0.018, +0.050]) | 0.518 (+0.028 [-0.008, +0.064]) | 0.550 (-0.004 [-0.032, +0.026]) | 0.542 (+0.004 [-0.024, +0.030]) |
| 40% | **0.556** (rate 0.40) | 0.534 (+0.022 [-0.002, +0.048]) | 0.548 (+0.008 [-0.026, +0.042]) | 0.528 (+0.028 [-0.004, +0.064]) | 0.516 (+0.040 [+0.008, +0.074]) | 0.552 (+0.004 [-0.028, +0.036]) | 0.546 (+0.010 [-0.020, +0.040]) |
| 50% | **0.570** (rate 0.50) | 0.554 (+0.016 [-0.010, +0.044]) | 0.548 (+0.022 [-0.014, +0.058]) | 0.536 (+0.034 [+0.002, +0.070]) | 0.526 (+0.044 [+0.010, +0.080]) | 0.548 (+0.022 [-0.010, +0.054]) | 0.548 (+0.022 [-0.008, +0.056]) |
| 60% | **0.570** (rate 0.60) | 0.562 (+0.008 [-0.018, +0.034]) | 0.558 (+0.012 [-0.022, +0.048]) | 0.546 (+0.024 [-0.008, +0.058]) | 0.540 (+0.030 [-0.004, +0.066]) | 0.550 (+0.020 [-0.012, +0.052]) | 0.564 (+0.006 [-0.024, +0.036]) |
| 70% | **0.578** (rate 0.70) | 0.550 (+0.028 [+0.002, +0.054]) | 0.568 (+0.010 [-0.026, +0.046]) | 0.552 (+0.026 [-0.006, +0.058]) | 0.550 (+0.028 [-0.008, +0.064]) | 0.564 (+0.014 [-0.018, +0.044]) | 0.568 (+0.010 [-0.020, +0.040]) |
| 80% | **0.574** (rate 0.80) | 0.568 (+0.006 [-0.020, +0.032]) | 0.564 (+0.010 [-0.020, +0.040]) | 0.552 (+0.022 [-0.008, +0.052]) | 0.552 (+0.022 [-0.010, +0.054]) | 0.558 (+0.016 [-0.016, +0.046]) | 0.580 (-0.006 [-0.038, +0.026]) |
| 90% | **0.582** (rate 0.90) | 0.572 (+0.010 [-0.012, +0.034]) | 0.566 (+0.016 [-0.010, +0.042]) | 0.556 (+0.026 [-0.002, +0.054]) | 0.558 (+0.024 [-0.004, +0.052]) | 0.554 (+0.028 [+0.000, +0.058]) | 0.584 (-0.002 [-0.032, +0.028]) |

## 5. Sensitivity of the probe's end-to-end accuracy to τ (one τ for both hops)

The same model gave τ = 0.585 or 0.650 depending only on the CV folds, so the result should not hinge on one τ. Last columns: probe − Llama-3.2-3B-Instruct judge (θ = 0.5).

### 2wiki

| τ | accuracy | restart rate | broken | probe − judge [95% CI] |
|---|---|---|---|---|
| 0.30 | 0.546 | 0.77 | 14 | +0.092 [+0.054, +0.128] |
| 0.35 | 0.546 | 0.74 | 13 | +0.092 [+0.056, +0.128] |
| 0.40 | 0.542 | 0.73 | 12 | +0.088 [+0.052, +0.124] |
| 0.45 | 0.540 | 0.72 | 12 | +0.086 [+0.050, +0.122] |
| 0.50 | 0.546 | 0.70 | 10 | +0.092 [+0.058, +0.128] |
| 0.55 | 0.548 | 0.69 | 10 | +0.094 [+0.060, +0.130] |
| 0.60 | 0.544 | 0.66 | 10 | +0.090 [+0.056, +0.124] |
| 0.65 | 0.538 | 0.64 | 10 | +0.084 [+0.050, +0.118] |
| 0.70 | 0.532 | 0.62 | 10 | +0.078 [+0.042, +0.112] |
| 0.75 | 0.524 | 0.59 | 10 | +0.070 [+0.034, +0.106] |
| 0.80 | 0.518 | 0.56 | 8 | +0.064 [+0.030, +0.100] |
| 0.85 | 0.508 | 0.53 | 7 | +0.054 [+0.018, +0.088] |
| 0.90 | 0.500 | 0.45 | 4 | +0.046 [+0.012, +0.080] |

### hotpotqa

| τ | accuracy | restart rate | broken | probe − judge [95% CI] |
|---|---|---|---|---|
| 0.30 | 0.580 | 0.79 | 35 | +0.054 [+0.016, +0.094] |
| 0.35 | 0.578 | 0.76 | 34 | +0.052 [+0.014, +0.092] |
| 0.40 | 0.578 | 0.74 | 33 | +0.052 [+0.014, +0.092] |
| 0.45 | 0.578 | 0.70 | 30 | +0.052 [+0.016, +0.088] |
| 0.50 | 0.580 | 0.68 | 28 | +0.054 [+0.018, +0.090] |
| 0.55 | 0.580 | 0.65 | 25 | +0.054 [+0.018, +0.090] |
| 0.60 | 0.576 | 0.63 | 24 | +0.050 [+0.016, +0.086] |
| 0.65 | 0.570 | 0.59 | 22 | +0.044 [+0.010, +0.078] |
| 0.70 | 0.570 | 0.53 | 20 | +0.044 [+0.012, +0.078] |
| 0.75 | 0.570 | 0.50 | 19 | +0.044 [+0.012, +0.078] |
| 0.80 | 0.562 | 0.46 | 16 | +0.036 [+0.004, +0.070] |
| 0.85 | 0.556 | 0.41 | 16 | +0.030 [-0.002, +0.064] |
| 0.90 | 0.546 | 0.33 | 12 | +0.020 [-0.010, +0.050] |

## 6. Hop-level detection on live 2Wiki hops, AUROC [95% CI]

'phase1 only' drops the trajectories that did not reproduce the stored Phase 1 text and were relabeled with a presence check (no LLM tier).

| hop | rows | n | fail | probe (tuned) | judge Llama-3.2-3B-Instruct | judge Llama-3.1-8B-Instruct | judge Qwen2.5-3B-Instruct | judge Qwen2.5-7B-Instruct |
|---|---|---|---|---|---|---|---|---|
| 1 | all | 499 | 85 | 0.947 [0.923, 0.968] | 0.569 [0.501, 0.628] | 0.597 [0.532, 0.660] | 0.558 [0.493, 0.619] | 0.779 [0.723, 0.831] |
| 1 | phase1 only | 475 | 80 | 0.949 [0.927, 0.971] | 0.565 [0.504, 0.632] | 0.584 [0.519, 0.654] | 0.556 [0.490, 0.629] | 0.781 [0.726, 0.837] |
| 2 | all | 492 | 327 | 0.925 [0.897, 0.948] | 0.678 [0.626, 0.727] | 0.729 [0.677, 0.777] | 0.562 [0.506, 0.618] | 0.690 [0.640, 0.735] |
| 2 | phase1 only | 468 | 307 | 0.928 [0.900, 0.952] | 0.682 [0.628, 0.732] | 0.724 [0.669, 0.776] | 0.563 [0.508, 0.617] | 0.692 [0.635, 0.741] |

## 7. The probe on restarted trajectories (repair verifier)

Every hop-1 / hop-2 restart was re-scored by the probe during the run (stage `rescore`; the restart prompt contains the retrieved evidence, so this is a distribution shift for the probe).

- **2wiki**: max hop P(fail) predicts a wrong final answer — first trajectories AUROC 0.837 [0.800, 0.872] (n = 499); restarted trajectories 0.774 [0.745, 0.803] (n = 991)
- **hotpotqa**: max hop P(fail) predicts a wrong final answer — first trajectories AUROC 0.595 [0.548, 0.644] (n = 500); restarted trajectories 0.557 [0.517, 0.592] (n = 986)

Probe-triggered restarts (one round), split by whether the probe still flags the restarted trajectory (any hop ≥ τ): if the probe is a useful verifier, flagged restarts should be much less accurate.

| dataset | restarts | still flagged | accuracy, still flagged | accuracy, cleared | answer fixed (cleared / flagged) |
|---|---|---|---|---|---|
| 2wiki | 320 | 175 | 0.263 | 0.717 | 71 / 37 |
| hotpotqa | 293 | 167 | 0.521 | 0.627 | 30 / 25 |

Hop-level detection on the restarted 2Wiki trajectories (labels: presence check of the gold bridge entity, no LLM tier):

| hop | n | fail | probe AUROC [95% CI] | recall | FPR |
|---|---|---|---|---|---|
| 1 | 991 | 150 | 0.901 [0.873, 0.928] | 0.300 | 0.012 |
| 2 | 980 | 576 | 0.871 [0.848, 0.893] | 0.682 | 0.094 |

## 8. Calibration on live 2Wiki hops 1–2

| trigger | n | ECE | Brier |
|---|---|---|---|
| Probe (tuned) | 991 | 0.024 | 0.080 |
| Probe (old) | 991 | 0.033 | 0.095 |
| Judge Llama-3.2-3B-Instruct | 991 | 0.221 | 0.290 |
| Judge Llama-3.1-8B-Instruct | 991 | 0.264 | 0.305 |
| Judge Qwen2.5-3B-Instruct | 991 | 0.482 | 0.487 |
| Judge Qwen2.5-7B-Instruct | 991 | 0.241 | 0.249 |

Figure: `reliability.png`.

## 9. Where the probe restarts and what the repair retrieves

| dataset | restart at | n | accuracy before | after | fixed | broken |
|---|---|---|---|---|---|---|
| 2wiki | hop 1 | 61 | 0.049 | 0.148 | 7 | 1 |
| 2wiki | hop 2 | 259 | 0.189 | 0.544 | 101 | 9 |
| hotpotqa | hop 1 | 88 | 0.466 | 0.568 | 17 | 8 |
| hotpotqa | hop 2 | 205 | 0.449 | 0.566 | 38 | 14 |

2Wiki: the gold paragraph(s) missing from the first-stage context, and whether the restart's retrieval brought one back (all hop-1 / hop-2 restarts, whatever the trigger):

| restart at | n (≥ 1 gold paragraph missing) | retrieval found a missing gold paragraph | accuracy if found | if not found |
|---|---|---|---|---|
| hop 1 | 296 | 0.247 | 0.726 | 0.130 |
| hop 2 | 293 | 0.532 | 0.718 | 0.124 |

## 10. HotpotQA by question type (the probe was trained on 2Wiki bridge chains)

| type | n | none | probe | judge (θ 0.5) | always hop 2 | probe − none [95% CI] | probe − judge [95% CI] | probe restart rate |
|---|---|---|---|---|---|---|---|---|
| bridge | 404 | 0.522 | 0.601 | 0.550 | 0.609 | +0.079 [+0.042, +0.119] | +0.052 [+0.015, +0.089] | 0.63 |
| comparison | 96 | 0.427 | 0.438 | 0.427 | 0.490 | +0.010 [-0.062, +0.083] | +0.010 [-0.062, +0.083] | 0.39 |

## 11. What the probe misses and its false alarms (live 2Wiki hops, τ = train-CV)

| hop | failing hops | missed | missed: hedge | missed: final answer still correct | false alarms | false alarms: final answer wrong anyway |
|---|---|---|---|---|---|---|
| 1 | 85 | 34 (40%) | 1 (3%) | 5 (15%) | 10 (2%) | 8 (80%) |
| 2 | 327 | 50 (15%) | 4 (8%) | 10 (20%) | 27 (16%) | 19 (70%) |

Missed failing hops by first-stage retrieval condition: hop 1 only_hop1_paragraph: 26, hop 1 both_present: 5, hop 1 neither: 2, hop 1 only_hop2_paragraph: 1, hop 2 only_hop1_paragraph: 28, hop 2 both_present: 22

