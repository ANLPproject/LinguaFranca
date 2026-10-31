# Few-shot judge (`meta-llama/Llama-3.1-8B-Instruct`, 2 demonstrations) — `pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`

Same judge prompt and P(fail) read-out as the pipeline's zero-shot judges; the chat starts with demonstration turns taken from the probe's TRAIN split (shortest contexts, confident non-hedge hop-2 steps): no (86c2414c), yes (1ab2864a). Correct = token F1 ≥ 0.5; CIs: paired bootstrap over questions.

Cost: 1173 ms and 1039 prompt tokens per check (mean), after the full answer is generated; the probe: ~9.5 ms at the hop boundary.

## Hop-level detection on live 2Wiki hops, AUROC [95% CI]

| hop | n | fail | probe | judge Llama-3.1-8B-Instruct | judge Llama-3.1-8B-Instruct (few-shot) |
|---|---|---|---|---|---|
| 1 | 499 | 85 | 0.947 [0.923, 0.968] | 0.597 [0.532, 0.660] | 0.547 [0.473, 0.618] |
| 2 | 492 | 327 | 0.925 [0.897, 0.948] | 0.729 [0.677, 0.777] | 0.678 [0.622, 0.731] |

Calibration on live 2Wiki hops: judge Llama-3.1-8B-Instruct ECE 0.264, Brier 0.305; judge Llama-3.1-8B-Instruct (few-shot) ECE 0.333, Brier 0.350 (probe: ECE 0.024, Brier 0.080).

## End to end (one restart, same cached generations)

Few-shot judge accuracy-optimal (θ1, θ2): {'fold0': (0.94, 0.0), 'fold1': (0.84, 0.0), 'all_2wiki': (0.84, 0.0)}.

### 2wiki (n = 500)

The judges' own decision (θ = 0.5) vs the probe's train-CV τ:

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.538 | 0.64 | 108 | 10 | – |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 | 0.410 | 0.27 | 37 | 3 | +0.128 [+0.090, +0.166] |
| Judge Llama-3.1-8B-Instruct (few-shot), θ = 0.5 | 0.396 | 0.18 | 29 | 2 | +0.142 [+0.104, +0.180] |

Accuracy-optimal thresholds for every trigger:

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.554 | 0.78 | 122 | 16 | – |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.558 | 0.99 | 127 | 19 | -0.004 [-0.018, +0.010] |
| Judge Llama-3.1-8B-Instruct (few-shot), accuracy-optimal | 0.556 | 0.98 | 126 | 19 | -0.002 [-0.016, +0.012] |

Matched restart budget — accuracy (probe − judge [95% CI]):

| budget | probe | judge Llama-3.1-8B-Instruct | judge Llama-3.1-8B-Instruct (few-shot) |
|---|---|---|---|
| 10% | **0.380** | 0.366 (+0.014 [-0.006, +0.034]) | 0.372 (+0.008 [-0.014, +0.030]) |
| 20% | **0.414** | 0.394 (+0.020 [-0.010, +0.050]) | 0.400 (+0.014 [-0.016, +0.044]) |
| 30% | **0.446** | 0.422 (+0.024 [-0.010, +0.060]) | 0.414 (+0.032 [-0.002, +0.066]) |
| 40% | **0.488** | 0.446 (+0.042 [+0.006, +0.078]) | 0.440 (+0.048 [+0.012, +0.086]) |
| 50% | **0.506** | 0.448 (+0.058 [+0.022, +0.094]) | 0.450 (+0.056 [+0.016, +0.094]) |
| 60% | **0.524** | 0.458 (+0.066 [+0.030, +0.102]) | 0.444 (+0.080 [+0.044, +0.118]) |
| 70% | **0.548** | 0.466 (+0.082 [+0.048, +0.116]) | 0.458 (+0.090 [+0.058, +0.126]) |
| 80% | **0.540** | 0.462 (+0.078 [+0.046, +0.112]) | 0.456 (+0.084 [+0.050, +0.120]) |
| 90% | **0.522** | 0.472 (+0.050 [+0.018, +0.082]) | 0.466 (+0.056 [+0.024, +0.088]) |

### hotpotqa (n = 500)

The judges' own decision (θ = 0.5) vs the probe's train-CV τ:

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe trigger (ours)** | 0.570 | 0.59 | 55 | 22 | – |
| Judge Llama-3.1-8B-Instruct, θ = 0.5 | 0.520 | 0.16 | 17 | 9 | +0.050 [+0.018, +0.084] |
| Judge Llama-3.1-8B-Instruct (few-shot), θ = 0.5 | 0.520 | 0.13 | 13 | 5 | +0.050 [+0.016, +0.084] |

Accuracy-optimal thresholds for every trigger:

| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |
|---|---|---|---|---|---|
| **Probe, accuracy-optimal** | 0.580 | 0.78 | 73 | 35 | – |
| Judge Llama-3.1-8B-Instruct, accuracy-optimal | 0.586 | 0.97 | 89 | 48 | -0.006 [-0.030, +0.018] |
| Judge Llama-3.1-8B-Instruct (few-shot), accuracy-optimal | 0.582 | 0.97 | 89 | 50 | -0.002 [-0.026, +0.024] |

Matched restart budget — accuracy (probe − judge [95% CI]):

| budget | probe | judge Llama-3.1-8B-Instruct | judge Llama-3.1-8B-Instruct (few-shot) |
|---|---|---|---|
| 10% | **0.516** | 0.508 (+0.008 [-0.012, +0.030]) | 0.510 (+0.006 [-0.012, +0.026]) |
| 20% | **0.536** | 0.520 (+0.016 [-0.012, +0.046]) | 0.524 (+0.012 [-0.016, +0.040]) |
| 30% | **0.546** | 0.530 (+0.016 [-0.018, +0.050]) | 0.528 (+0.018 [-0.016, +0.052]) |
| 40% | **0.556** | 0.528 (+0.028 [-0.004, +0.064]) | 0.542 (+0.014 [-0.020, +0.050]) |
| 50% | **0.570** | 0.536 (+0.034 [+0.002, +0.070]) | 0.544 (+0.026 [-0.008, +0.060]) |
| 60% | **0.570** | 0.546 (+0.024 [-0.008, +0.058]) | 0.540 (+0.030 [-0.004, +0.064]) |
| 70% | **0.578** | 0.552 (+0.026 [-0.006, +0.058]) | 0.542 (+0.036 [+0.002, +0.072]) |
| 80% | **0.574** | 0.552 (+0.022 [-0.008, +0.052]) | 0.550 (+0.024 [-0.008, +0.056]) |
| 90% | **0.582** | 0.556 (+0.026 [-0.002, +0.054]) | 0.550 (+0.032 [+0.002, +0.062]) |

(Kaggle T4 x2 run of `scripts/fewshot_judge.py --run …`, default `--shots 2`.)
