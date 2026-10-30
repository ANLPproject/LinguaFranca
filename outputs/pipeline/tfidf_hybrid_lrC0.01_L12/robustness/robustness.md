# Robustness checks and extra analyses (`pipeline_tfidf_v2_r2_shared_hybrid_probes_lr_C0.01_L12`)

Items: 1000 (2wiki 500, hotpotqa 500). All systems act on the same cached generations; correct = token F1 ≥ 0.5; CIs are paired bootstraps over questions (2,000 resamples). Probe: `outputs/pipeline/tfidf/probes_lr_C0.01_L12` (τ = 0.650); old probe: `outputs/pipeline/tfidf/probes` (layer 10, τ = 0.639).

## 1. Fair thresholds: every trigger with the same accuracy-optimal (θ1, θ2)

The headline compares the probe at its train-CV τ with judges at their own yes/no decision (θ = 0.5). Here every trigger also gets the procedure the probe's 'accuracy-optimal τ' row uses: (θ1, θ2) maximising accuracy on one half of the 2Wiki questions, applied to the other half; HotpotQA uses the fit on all 2Wiki. The last column compares with the accuracy-optimal probe.

Fitted thresholds (fold 0 / fold 1 / all 2Wiki): probe: (0.82, 0.18) / (0.98, 0.02) / (0.86, 0.14); Llama-3.2-3B-Instruct: (0.9, 0.04) / (0.96, 0.0) / (0.96, 0.0); Llama-3.1-8B-Instruct: (0.98, 0.02) / (0.88, 0.0) / (0.98, 0.0); Qwen2.5-3B-Instruct: (1.0, 0.48) / (1.0, 0.0) / (1.0, 0.0); Qwen2.5-7B-Instruct: (0.02, 0.0) / (1.0, 0.0) / (1.0, 0.0)

### 2wiki (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| Probe, train-CV τ (headline) | 0.558 | 0.64 | 113 | 5 | +0.024 [+0.004, +0.046] |
| **Probe, accuracy-optimal** | 0.582 | 0.87 | 132 | 12 | – |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 (headline) | 0.452 | 0.46 | 59 | 4 | +0.130 [+0.092, +0.170] |
| Judge Llama-3.2-3B-Instruct, accuracy-optimal | 0.564 | 0.91 | 125 | 14 | +0.018 [+0.002, +0.034] |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 (headline) | 0.402 | 0.27 | 35 | 5 | +0.180 [+0.142, +0.220] |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.574 | 0.89 | 129 | 13 | +0.008 [-0.006, +0.022] |
| Judge Qwen2.5-3B-Instruct, θ = 0.5 (headline) | 0.446 | 0.89 | 70 | 18 | +0.136 [+0.096, +0.176] |
| Judge Qwen2.5-3B-Instruct, accuracy-optimal | 0.506 | 0.90 | 99 | 17 | +0.076 [+0.050, +0.104] |
| Judge Qwen2.5-7B-Instruct, θ = 0.5 (headline) | 0.510 | 0.63 | 89 | 5 | +0.072 [+0.040, +0.104] |
| Judge Qwen2.5-7B-Instruct, accuracy-optimal | 0.566 | 0.98 | 127 | 15 | +0.016 [+0.000, +0.034] |
| Always restart after hop 2 | 0.582 | 0.98 | 134 | 14 | +0.000 [-0.010, +0.010] |

### hotpotqa (n = 500)

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| Probe, train-CV τ (headline) | 0.550 | 0.59 | 53 | 30 | +0.016 [-0.006, +0.040] |
| **Probe, accuracy-optimal** | 0.566 | 0.81 | 70 | 39 | – |
| Judge Llama-3.2-3B-Instruct, θ = 0.5 (headline) | 0.536 | 0.23 | 25 | 9 | +0.030 [-0.006, +0.066] |
| Judge Llama-3.2-3B-Instruct, accuracy-optimal | 0.586 | 0.97 | 89 | 48 | -0.020 [-0.042, +0.002] |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 (headline) | 0.528 | 0.16 | 16 | 4 | +0.038 [+0.000, +0.076] |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.588 | 0.97 | 89 | 47 | -0.022 [-0.044, +0.000] |
| Judge Qwen2.5-3B-Instruct, θ = 0.5 (headline) | 0.556 | 0.75 | 68 | 42 | +0.010 [-0.024, +0.042] |
| Judge Qwen2.5-3B-Instruct, accuracy-optimal | 0.582 | 0.98 | 89 | 50 | -0.016 [-0.040, +0.008] |
| Judge Qwen2.5-7B-Instruct, θ = 0.5 (headline) | 0.546 | 0.49 | 49 | 28 | +0.020 [-0.012, +0.052] |
| Judge Qwen2.5-7B-Instruct, accuracy-optimal | 0.580 | 0.97 | 89 | 51 | -0.014 [-0.038, +0.010] |
| Always restart after hop 2 | 0.586 | 0.97 | 89 | 48 | -0.020 [-0.042, +0.002] |

## 2. Old (C = 1, block 10) vs tuned (C = 0.01, block 12) probe on the same generations

Old probe accuracy-optimal (θ1, θ2): {'fold0': (0.94, 0.0), 'fold1': (1.0, 0.04), 'all_2wiki': (1.0, 0.02)}. Last column: tuned − old.

### 2wiki

| threshold | tuned | old | tuned − old [95% CI] | restart rate tuned / old |
|---|---|---|---|---|
| train-CV τ | 0.558 | 0.522 | +0.036 [+0.018, +0.056] | 0.64 / 0.60 |
| accuracy-optimal | 0.582 | 0.574 | +0.008 [-0.004, +0.020] | 0.87 / 0.94 |

### hotpotqa

| threshold | tuned | old | tuned − old [95% CI] | restart rate tuned / old |
|---|---|---|---|---|
| train-CV τ | 0.550 | 0.562 | -0.012 [-0.038, +0.014] | 0.59 / 0.59 |
| accuracy-optimal | 0.566 | 0.578 | -0.012 [-0.032, +0.008] | 0.81 / 0.91 |

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
| No intervention | 0.342 | 0.334 | 66 | +0.212 [+0.176, +0.250] |
| Probe trigger (ours) | 0.558 | 0.546 | 47 | – |
| Always retrieve | 0.384 | 0.372 | 74 | +0.174 [+0.134, +0.216] |
| Always restart after hop 2 | 0.582 | 0.564 | 45 | -0.018 [-0.042, +0.006] |
| Judge Llama-3.2-3B-Instruct | 0.452 | 0.434 | 55 | +0.112 [+0.074, +0.148] |
| Judge Llama-3.1-8B-Instruct | 0.402 | 0.386 | 60 | +0.160 [+0.124, +0.196] |
| Judge Qwen2.5-3B-Instruct | 0.446 | 0.426 | 54 | +0.120 [+0.078, +0.158] |
| Judge Qwen2.5-7B-Instruct | 0.510 | 0.492 | 47 | +0.054 [+0.022, +0.086] |

### hotpotqa

| system | with forcing | without | forced answers | probe − system, without forcing [95% CI] |
|---|---|---|---|---|
| No intervention | 0.504 | 0.488 | 39 | +0.044 [+0.008, +0.078] |
| Probe trigger (ours) | 0.550 | 0.532 | 35 | – |
| Always retrieve | 0.554 | 0.528 | 38 | +0.004 [-0.036, +0.044] |
| Always restart after hop 2 | 0.586 | 0.562 | 44 | -0.030 [-0.062, +0.004] |
| Judge Llama-3.2-3B-Instruct | 0.536 | 0.522 | 35 | +0.010 [-0.024, +0.044] |
| Judge Llama-3.1-8B-Instruct | 0.528 | 0.512 | 40 | +0.020 [-0.014, +0.054] |
| Judge Qwen2.5-3B-Instruct | 0.556 | 0.538 | 38 | -0.006 [-0.042, +0.030] |
| Judge Qwen2.5-7B-Instruct | 0.546 | 0.530 | 32 | +0.002 [-0.030, +0.036] |

## 4. Accuracy at a matched restart budget (paired-bootstrap CIs)

Each trigger restarts at its earliest hop whose score exceeds the threshold that gives the budget's restart rate (one threshold for both hops). Judges' token cost is ignored here (they need the full answer first): only their choice of questions and hops is compared. Cells: accuracy (probe − trigger [95% CI]).

### 2wiki

| budget | Probe (tuned) | Probe (old) | Judge Llama-3.2-3B-Instruct | Judge Llama-3.1-8B-Instruct | Judge Qwen2.5-3B-Instruct | Judge Qwen2.5-7B-Instruct | Text trigger (TF-IDF+LR) |
|---|---|---|---|---|---|---|---|
| 10% | **0.380** (rate 0.10) | 0.372 (+0.008 [-0.010, +0.026]) | 0.372 (+0.008 [-0.014, +0.030]) | 0.368 (+0.012 [-0.010, +0.034]) | 0.422 (-0.042 [-0.072, -0.014]) | 0.384 (-0.004 [-0.028, +0.020]) | 0.372 (+0.008 [-0.014, +0.030]) |
| 20% | **0.422** (rate 0.20) | 0.400 (+0.022 [+0.000, +0.042]) | 0.402 (+0.020 [-0.010, +0.050]) | 0.388 (+0.034 [+0.004, +0.064]) | 0.422 (+0.000 [-0.034, +0.032]) | 0.394 (+0.028 [-0.002, +0.056]) | 0.422 (+0.000 [-0.028, +0.028]) |
| 30% | **0.454** (rate 0.30) | 0.440 (+0.014 [-0.010, +0.038]) | 0.416 (+0.038 [+0.006, +0.070]) | 0.418 (+0.036 [+0.002, +0.068]) | 0.422 (+0.032 [-0.004, +0.068]) | 0.422 (+0.032 [+0.002, +0.064]) | 0.454 (+0.000 [-0.034, +0.032]) |
| 40% | **0.502** (rate 0.40) | 0.474 (+0.028 [+0.006, +0.050]) | 0.434 (+0.068 [+0.034, +0.102]) | 0.434 (+0.068 [+0.032, +0.104]) | 0.422 (+0.080 [+0.044, +0.116]) | 0.452 (+0.050 [+0.014, +0.086]) | 0.486 (+0.016 [-0.014, +0.046]) |
| 50% | **0.528** (rate 0.50) | 0.498 (+0.030 [+0.012, +0.050]) | 0.454 (+0.074 [+0.038, +0.110]) | 0.440 (+0.088 [+0.052, +0.126]) | 0.434 (+0.094 [+0.058, +0.130]) | 0.482 (+0.046 [+0.012, +0.080]) | 0.514 (+0.014 [-0.014, +0.042]) |
| 60% | **0.546** (rate 0.60) | 0.522 (+0.024 [+0.004, +0.044]) | 0.442 (+0.104 [+0.068, +0.142]) | 0.456 (+0.090 [+0.056, +0.126]) | 0.442 (+0.104 [+0.066, +0.140]) | 0.502 (+0.044 [+0.010, +0.076]) | 0.528 (+0.018 [-0.012, +0.046]) |
| 70% | **0.566** (rate 0.70) | 0.548 (+0.018 [+0.002, +0.036]) | 0.446 (+0.120 [+0.080, +0.160]) | 0.462 (+0.104 [+0.068, +0.140]) | 0.456 (+0.110 [+0.072, +0.146]) | 0.526 (+0.040 [+0.010, +0.068]) | 0.538 (+0.028 [+0.002, +0.054]) |
| 80% | **0.562** (rate 0.80) | 0.542 (+0.020 [+0.002, +0.038]) | 0.440 (+0.122 [+0.084, +0.162]) | 0.464 (+0.098 [+0.064, +0.134]) | 0.454 (+0.108 [+0.070, +0.146]) | 0.538 (+0.024 [-0.004, +0.052]) | 0.552 (+0.010 [-0.012, +0.034]) |
| 90% | **0.542** (rate 0.90) | 0.512 (+0.030 [+0.004, +0.056]) | 0.446 (+0.096 [+0.066, +0.128]) | 0.462 (+0.080 [+0.048, +0.114]) | 0.448 (+0.094 [+0.058, +0.130]) | 0.514 (+0.028 [-0.002, +0.060]) | 0.564 (-0.022 [-0.048, +0.004]) |

### hotpotqa

| budget | Probe (tuned) | Probe (old) | Judge Llama-3.2-3B-Instruct | Judge Llama-3.1-8B-Instruct | Judge Qwen2.5-3B-Instruct | Judge Qwen2.5-7B-Instruct | Text trigger (TF-IDF+LR) |
|---|---|---|---|---|---|---|---|
| 10% | **0.520** (rate 0.10) | 0.540 (-0.020 [-0.038, -0.004]) | 0.520 (+0.000 [-0.022, +0.022]) | 0.514 (+0.006 [-0.014, +0.028]) | 0.526 (-0.006 [-0.032, +0.022]) | 0.516 (+0.004 [-0.020, +0.028]) | 0.518 (+0.002 [-0.018, +0.022]) |
| 20% | **0.530** (rate 0.20) | 0.536 (-0.006 [-0.026, +0.014]) | 0.530 (+0.000 [-0.028, +0.028]) | 0.532 (-0.002 [-0.028, +0.026]) | 0.526 (+0.004 [-0.022, +0.032]) | 0.520 (+0.010 [-0.014, +0.038]) | 0.534 (-0.004 [-0.028, +0.020]) |
| 30% | **0.532** (rate 0.30) | 0.534 (-0.002 [-0.026, +0.020]) | 0.542 (-0.010 [-0.040, +0.020]) | 0.536 (-0.004 [-0.034, +0.026]) | 0.536 (-0.004 [-0.036, +0.028]) | 0.544 (-0.012 [-0.040, +0.020]) | 0.534 (-0.002 [-0.032, +0.028]) |
| 40% | **0.534** (rate 0.40) | 0.534 (+0.000 [-0.026, +0.026]) | 0.556 (-0.022 [-0.054, +0.010]) | 0.540 (-0.006 [-0.040, +0.028]) | 0.536 (-0.002 [-0.034, +0.032]) | 0.552 (-0.018 [-0.048, +0.014]) | 0.538 (-0.004 [-0.036, +0.030]) |
| 50% | **0.548** (rate 0.50) | 0.548 (+0.000 [-0.028, +0.028]) | 0.554 (-0.006 [-0.042, +0.028]) | 0.552 (-0.004 [-0.034, +0.030]) | 0.546 (+0.002 [-0.032, +0.036]) | 0.548 (+0.000 [-0.032, +0.032]) | 0.542 (+0.006 [-0.028, +0.042]) |
| 60% | **0.550** (rate 0.60) | 0.564 (-0.014 [-0.040, +0.012]) | 0.564 (-0.014 [-0.050, +0.024]) | 0.562 (-0.012 [-0.044, +0.022]) | 0.548 (+0.002 [-0.034, +0.038]) | 0.532 (+0.018 [-0.014, +0.052]) | 0.556 (-0.006 [-0.038, +0.028]) |
| 70% | **0.560** (rate 0.70) | 0.568 (-0.008 [-0.036, +0.018]) | 0.568 (-0.008 [-0.044, +0.028]) | 0.556 (+0.004 [-0.026, +0.036]) | 0.548 (+0.012 [-0.022, +0.046]) | 0.546 (+0.014 [-0.014, +0.044]) | 0.566 (-0.006 [-0.038, +0.026]) |
| 80% | **0.560** (rate 0.80) | 0.578 (-0.018 [-0.044, +0.006]) | 0.570 (-0.010 [-0.040, +0.018]) | 0.556 (+0.004 [-0.026, +0.034]) | 0.564 (-0.004 [-0.036, +0.026]) | 0.566 (-0.006 [-0.036, +0.022]) | 0.578 (-0.018 [-0.048, +0.012]) |
| 90% | **0.582** (rate 0.90) | 0.570 (+0.012 [-0.008, +0.032]) | 0.582 (+0.000 [-0.028, +0.026]) | 0.566 (+0.016 [-0.010, +0.042]) | 0.568 (+0.014 [-0.012, +0.040]) | 0.568 (+0.014 [-0.012, +0.038]) | 0.582 (+0.000 [-0.032, +0.032]) |

## 5. Sensitivity of the probe's end-to-end accuracy to τ (one τ for both hops)

The same model gave τ = 0.585 or 0.650 depending only on the CV folds, so the result should not hinge on one τ. Last columns: probe − Llama-3.2-3B-Instruct judge (θ = 0.5).

### 2wiki

| τ | accuracy | restart rate | broken | probe − judge [95% CI] |
|---|---|---|---|---|
| 0.30 | 0.568 | 0.77 | 11 | +0.116 [+0.078, +0.156] |
| 0.35 | 0.564 | 0.74 | 10 | +0.112 [+0.074, +0.152] |
| 0.40 | 0.562 | 0.73 | 9 | +0.110 [+0.072, +0.150] |
| 0.45 | 0.560 | 0.72 | 9 | +0.108 [+0.070, +0.146] |
| 0.50 | 0.566 | 0.70 | 7 | +0.114 [+0.076, +0.152] |
| 0.55 | 0.564 | 0.69 | 7 | +0.112 [+0.074, +0.150] |
| 0.60 | 0.562 | 0.66 | 6 | +0.110 [+0.074, +0.146] |
| 0.65 | 0.558 | 0.64 | 5 | +0.106 [+0.068, +0.144] |
| 0.70 | 0.550 | 0.62 | 5 | +0.098 [+0.060, +0.134] |
| 0.75 | 0.546 | 0.59 | 5 | +0.094 [+0.058, +0.130] |
| 0.80 | 0.540 | 0.56 | 4 | +0.088 [+0.052, +0.124] |
| 0.85 | 0.528 | 0.53 | 4 | +0.076 [+0.040, +0.112] |
| 0.90 | 0.516 | 0.45 | 3 | +0.064 [+0.028, +0.100] |

### hotpotqa

| τ | accuracy | restart rate | broken | probe − judge [95% CI] |
|---|---|---|---|---|
| 0.30 | 0.564 | 0.79 | 39 | +0.028 [-0.010, +0.066] |
| 0.35 | 0.562 | 0.76 | 38 | +0.026 [-0.012, +0.062] |
| 0.40 | 0.560 | 0.74 | 37 | +0.024 [-0.014, +0.062] |
| 0.45 | 0.560 | 0.70 | 34 | +0.024 [-0.012, +0.058] |
| 0.50 | 0.562 | 0.68 | 34 | +0.026 [-0.010, +0.060] |
| 0.55 | 0.564 | 0.65 | 31 | +0.028 [-0.008, +0.062] |
| 0.60 | 0.558 | 0.63 | 31 | +0.022 [-0.014, +0.056] |
| 0.65 | 0.550 | 0.59 | 30 | +0.014 [-0.020, +0.048] |
| 0.70 | 0.552 | 0.53 | 26 | +0.016 [-0.016, +0.050] |
| 0.75 | 0.548 | 0.50 | 24 | +0.012 [-0.020, +0.044] |
| 0.80 | 0.542 | 0.46 | 21 | +0.006 [-0.026, +0.036] |
| 0.85 | 0.534 | 0.41 | 20 | -0.002 [-0.032, +0.028] |
| 0.90 | 0.526 | 0.33 | 16 | -0.010 [-0.038, +0.018] |

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

- **2wiki**: max hop P(fail) predicts a wrong final answer — first trajectories AUROC 0.837 [0.800, 0.872] (n = 499); restarted trajectories 0.755 [0.725, 0.783] (n = 991)
- **hotpotqa**: max hop P(fail) predicts a wrong final answer — first trajectories AUROC 0.595 [0.548, 0.644] (n = 500); restarted trajectories 0.580 [0.542, 0.617] (n = 986)

Probe-triggered restarts (one round), split by whether the probe still flags the restarted trajectory (any hop ≥ τ): if the probe is a useful verifier, flagged restarts should be much less accurate.

| dataset | restarts | still flagged | accuracy, still flagged | accuracy, cleared | answer fixed (cleared / flagged) |
|---|---|---|---|---|---|
| 2wiki | 320 | 181 | 0.320 | 0.734 | 75 / 38 |
| hotpotqa | 293 | 172 | 0.506 | 0.570 | 30 / 23 |

Hop-level detection on the restarted 2Wiki trajectories (labels: presence check of the gold bridge entity, no LLM tier):

| hop | n | fail | probe AUROC [95% CI] | recall | FPR |
|---|---|---|---|---|---|
| 1 | 991 | 152 | 0.900 [0.873, 0.925] | 0.322 | 0.018 |
| 2 | 984 | 575 | 0.872 [0.851, 0.892] | 0.701 | 0.086 |

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
| 2wiki | hop 1 | 61 | 0.049 | 0.115 | 5 | 1 |
| 2wiki | hop 2 | 259 | 0.189 | 0.591 | 108 | 4 |
| hotpotqa | hop 1 | 88 | 0.466 | 0.489 | 15 | 13 |
| hotpotqa | hop 2 | 205 | 0.449 | 0.551 | 38 | 17 |

2Wiki: the gold paragraph(s) missing from the first-stage context, and whether the restart's retrieval brought one back (all hop-1 / hop-2 restarts, whatever the trigger):

| restart at | n (≥ 1 gold paragraph missing) | retrieval found a missing gold paragraph | accuracy if found | if not found |
|---|---|---|---|---|
| hop 1 | 296 | 0.264 | 0.679 | 0.115 |
| hop 2 | 293 | 0.659 | 0.663 | 0.110 |

## 10. HotpotQA by question type (the probe was trained on 2Wiki bridge chains)

| type | n | none | probe | judge (θ 0.5) | always hop 2 | probe − none [95% CI] | probe − judge [95% CI] | probe restart rate |
|---|---|---|---|---|---|---|---|---|
| bridge | 404 | 0.522 | 0.577 | 0.562 | 0.621 | +0.054 [+0.015, +0.097] | +0.015 [-0.025, +0.054] | 0.63 |
| comparison | 96 | 0.427 | 0.438 | 0.427 | 0.438 | +0.010 [-0.073, +0.094] | +0.010 [-0.062, +0.083] | 0.39 |

## 11. What the probe misses and its false alarms (live 2Wiki hops, τ = train-CV)

| hop | failing hops | missed | missed: hedge | missed: final answer still correct | false alarms | false alarms: final answer wrong anyway |
|---|---|---|---|---|---|---|
| 1 | 85 | 34 (40%) | 1 (3%) | 5 (15%) | 10 (2%) | 8 (80%) |
| 2 | 327 | 50 (15%) | 4 (8%) | 10 (20%) | 27 (16%) | 19 (70%) |

Missed failing hops by first-stage retrieval condition: hop 1 only_hop1_paragraph: 26, hop 1 both_present: 5, hop 1 neither: 2, hop 1 only_hop2_paragraph: 1, hop 2 only_hop1_paragraph: 28, hop 2 both_present: 22

