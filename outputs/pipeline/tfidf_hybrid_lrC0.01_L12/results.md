# Phase 3–5 results — setting `tfidf`

Items evaluated: 1000.  Main probe: one shared probe for both hops; layers {1: 12, 2: 12}, thresholds {1: 0.65, 2: 0.65} (F0.5-optimal on train-split CV).
Answers forced from hops 1-2 after hitting the token cap without <answer>: {'first answers': 105, 'restarts': 437}.

## RQ1 — failure detection (2Wiki held-out test, live trajectories)

Label sources: {'phase1': 476, 'relabeled': 24, 'unavailable': 0}

| rows | trigger | n | fail | AUROC | precision | recall | F0.5 | FPR | flag rate |
|---|---|---|---|---|---|---|---|---|---|
| hop1 | probe | 499 | 85 | 0.947 | 0.836 | 0.600 | 0.775 | 0.024 | 0.122 |
| hop1 | probe_dedicated | 499 | 85 | 0.949 | 0.797 | 0.694 | 0.774 | 0.036 | 0.148 |
| hop1 | judge (Qwen2.5-3B-Instruct) | 499 | 85 | 0.558 | 0.180 | 0.788 | 0.213 | 0.737 | 0.745 |
| hop1 | judge (Qwen2.5-7B-Instruct) | 499 | 85 | 0.779 | 0.426 | 0.541 | 0.445 | 0.150 | 0.216 |
| hop1 | judge (Llama-3.1-8B-Instruct) | 499 | 85 | 0.597 | 0.262 | 0.200 | 0.246 | 0.116 | 0.130 |
| hop1 | judge | 499 | 85 | 0.569 | 0.198 | 0.306 | 0.213 | 0.254 | 0.263 |
| hop2 | probe | 492 | 327 | 0.925 | 0.911 | 0.847 | 0.898 | 0.164 | 0.618 |
| hop2 | probe_dedicated | 492 | 327 | 0.922 | 0.915 | 0.823 | 0.895 | 0.152 | 0.598 |
| hop2 | judge (Qwen2.5-3B-Instruct) | 492 | 327 | 0.562 | 0.680 | 0.807 | 0.703 | 0.752 | 0.789 |
| hop2 | judge (Qwen2.5-7B-Instruct) | 492 | 327 | 0.690 | 0.786 | 0.676 | 0.762 | 0.364 | 0.571 |
| hop2 | judge (Llama-3.1-8B-Instruct) | 492 | 327 | 0.729 | 0.818 | 0.220 | 0.530 | 0.097 | 0.179 |
| hop2 | judge | 492 | 327 | 0.678 | 0.796 | 0.382 | 0.654 | 0.194 | 0.319 |
| trajectory | probe | 500 | 351 | – | 0.909 | 0.829 | 0.892 | 0.195 | 0.640 |
| trajectory | probe_dedicated | 500 | 351 | – | 0.921 | 0.832 | 0.902 | 0.168 | 0.634 |
| trajectory | judge (Qwen2.5-3B-Instruct) | 500 | 351 | – | 0.704 | 0.889 | 0.735 | 0.879 | 0.886 |
| trajectory | judge (Qwen2.5-7B-Instruct) | 500 | 351 | – | 0.777 | 0.695 | 0.759 | 0.470 | 0.628 |
| trajectory | judge (Llama-3.1-8B-Instruct) | 500 | 351 | – | 0.812 | 0.308 | 0.612 | 0.168 | 0.266 |
| trajectory | judge | 500 | 351 | – | 0.795 | 0.519 | 0.718 | 0.315 | 0.458 |

First-failure localization accuracy (on failing trajectories): probe 0.744, judge 0.248. Exact first-fail hop incl. 'no failure' (all trajectories): probe 0.762, judge 0.378.

## RQ3 — end-to-end accuracy, 2wiki

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.342 | – | 0.258 | 0.319 | 0.000 | 0 | 0 | 143.4 | 911 | – |
| Probe trigger (ours) | 0.558 | +0.216 [+0.180, +0.256] | 0.424 | 0.518 | 0.640 | 113 | 5 | 160.2 | 1595 | 53.8 |
| Judge trigger (Doctor-RAG-style) | 0.452 | +0.110 [+0.082, +0.140] | 0.330 | 0.415 | 0.458 | 59 | 4 | 213.7 | 2589 | 169.7 |
| Always retrieve | 0.384 | +0.042 [+0.014, +0.072] | 0.276 | 0.348 | 1.000 | 38 | 17 | 152.6 | 1235 | 0.0 |
| Oracle trigger | 0.566 | +0.224 [+0.186, +0.262] | 0.426 | 0.525 | 0.690 | 118 | 6 | 155.0 | 1639 | 52.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.582 | +0.240 [+0.198, +0.282] | 0.426 | 0.531 | 0.872 | 132 | 12 | 173.1 | 1882 | 57.4 |
| Probe trigger, up to 2 restarts | 0.604 | +0.262 [+0.220, +0.302] | 0.460 | 0.565 | 0.640 | 137 | 6 | 172.8 | 2106 | 53.8 |
| Judge trigger, up to 2 restarts | 0.470 | +0.128 [+0.096, +0.158] | 0.338 | 0.430 | 0.458 | 68 | 4 | 246.7 | 3647 | 169.7 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.554 | +0.212 [+0.174, +0.250] | 0.412 | 0.510 | 0.634 | 111 | 5 | 156.6 | 1587 | 52.8 |
| Judge trigger (Qwen2.5-3B-Instruct) | 0.446 | +0.104 [+0.068, +0.140] | 0.324 | 0.411 | 0.886 | 70 | 18 | 262.2 | 3101 | 148.0 |
| Judge trigger (Qwen2.5-7B-Instruct) | 0.510 | +0.168 [+0.132, +0.204] | 0.384 | 0.469 | 0.628 | 89 | 5 | 227.3 | 2805 | 156.7 |
| Judge trigger (Llama-3.1-8B-Instruct) | 0.402 | +0.060 [+0.036, +0.086] | 0.306 | 0.374 | 0.266 | 35 | 5 | 181.2 | 2369 | 158.2 |

Probe − judge accuracy, paired bootstrap 95% CI: [+0.068, +0.144]
Probe − judge (Qwen2.5-3B-Instruct) accuracy, paired bootstrap 95% CI: [+0.072, +0.148]
Probe − judge (Qwen2.5-7B-Instruct) accuracy, paired bootstrap 95% CI: [+0.016, +0.078]
Probe − judge (Llama-3.1-8B-Instruct) accuracy, paired bootstrap 95% CI: [+0.120, +0.194]
Probe − judge accuracy, both with up to 2 restarts: [+0.094, +0.172]; share of questions restarted twice: {'probe_restart_2r': 0.362, 'judge_restart_2r': 0.202}
Accuracy-optimal τ (τ1, τ2), chosen on the other half of the 2Wiki questions: {'fold0_fit_on_fold1': (0.82, 0.18), 'fold1_fit_on_fold0': (0.98, 0.02), 'all_2wiki_used_for_hotpotqa': (0.86, 0.14)}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Oracle trigger | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | Judge trigger (Qwen2.5-3B-Instruct) | Judge trigger (Qwen2.5-7B-Instruct) | Judge trigger (Llama-3.1-8B-Instruct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both_present | 203 | 0.680 | 0.749 | 0.719 | 0.734 | 0.729 | 0.744 | 0.734 | 0.709 | 0.754 | 0.695 | 0.749 | 0.690 |
| neither | 9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.556 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| only_hop1_paragraph | 282 | 0.113 | 0.436 | 0.280 | 0.131 | 0.461 | 0.486 | 0.511 | 0.319 | 0.426 | 0.273 | 0.355 | 0.213 |
| only_hop2_paragraph | 6 | 0.167 | 0.667 | 0.167 | 1.000 | 0.833 | 0.500 | 0.667 | 0.167 | 0.667 | 0.833 | 0.500 | 0.167 |

## RQ3 — end-to-end accuracy, hotpotqa

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.504 | – | 0.362 | 0.487 | 0.000 | 0 | 0 | 134.3 | 1072 | – |
| Probe trigger (ours) | 0.550 | +0.046 [+0.012, +0.080] | 0.382 | 0.526 | 0.586 | 53 | 30 | 155.5 | 1850 | 52.4 |
| Judge trigger (Doctor-RAG-style) | 0.536 | +0.032 [+0.010, +0.054] | 0.370 | 0.508 | 0.226 | 25 | 9 | 160.6 | 2863 | 124.6 |
| Always retrieve | 0.554 | +0.050 [+0.006, +0.094] | 0.396 | 0.535 | 1.000 | 81 | 56 | 133.3 | 1484 | 0.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.566 | +0.062 [+0.022, +0.100] | 0.384 | 0.536 | 0.806 | 70 | 39 | 174.6 | 2156 | 58.8 |
| Probe trigger, up to 2 restarts | 0.552 | +0.048 [+0.014, +0.082] | 0.386 | 0.533 | 0.586 | 55 | 31 | 176.5 | 2434 | 52.4 |
| Judge trigger, up to 2 restarts | 0.534 | +0.030 [+0.008, +0.054] | 0.374 | 0.508 | 0.226 | 26 | 11 | 170.2 | 3478 | 124.6 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.560 | +0.056 [+0.018, +0.092] | 0.386 | 0.534 | 0.612 | 62 | 34 | 154.3 | 1893 | 50.1 |
| Judge trigger (Qwen2.5-3B-Instruct) | 0.556 | +0.052 [+0.014, +0.094] | 0.372 | 0.526 | 0.750 | 68 | 42 | 235.3 | 3629 | 136.6 |
| Judge trigger (Qwen2.5-7B-Instruct) | 0.546 | +0.042 [+0.008, +0.076] | 0.374 | 0.513 | 0.492 | 49 | 28 | 195.4 | 3266 | 139.0 |
| Judge trigger (Llama-3.1-8B-Instruct) | 0.528 | +0.024 [+0.008, +0.042] | 0.368 | 0.503 | 0.156 | 16 | 4 | 153.1 | 2774 | 121.7 |

Probe − judge accuracy, paired bootstrap 95% CI: [-0.020, +0.048]
Probe − judge (Qwen2.5-3B-Instruct) accuracy, paired bootstrap 95% CI: [-0.042, +0.030]
Probe − judge (Qwen2.5-7B-Instruct) accuracy, paired bootstrap 95% CI: [-0.028, +0.036]
Probe − judge (Llama-3.1-8B-Instruct) accuracy, paired bootstrap 95% CI: [-0.012, +0.056]
Probe − judge accuracy, both with up to 2 restarts: [-0.018, +0.054]; share of questions restarted twice: {'probe_restart_2r': 0.344, 'judge_restart_2r': 0.076}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | Judge trigger (Qwen2.5-3B-Instruct) | Judge trigger (Qwen2.5-7B-Instruct) | Judge trigger (Llama-3.1-8B-Instruct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both_present | 273 | 0.711 | 0.678 | 0.700 | 0.674 | 0.674 | 0.678 | 0.703 | 0.678 | 0.674 | 0.689 | 0.714 |
| neither | 15 | 0.067 | 0.600 | 0.333 | 0.533 | 0.600 | 0.533 | 0.200 | 0.533 | 0.600 | 0.333 | 0.200 |
| one_present | 212 | 0.269 | 0.382 | 0.340 | 0.401 | 0.425 | 0.392 | 0.340 | 0.410 | 0.401 | 0.377 | 0.311 |

## Final-answer failure prediction (AUROC of max hop P(fail) vs. wrong final answer)

| dataset | n | failure rate | probe | judge |
|---|---|---|---|---|
| 2wiki | 500 | 0.658 | 0.837 | 0.682 |
| hotpotqa | 500 | 0.496 | 0.595 | 0.570 |

## Latency (sequential, live subset n=100)

| system | mean ms | median ms |
|---|---|---|
| No intervention | 7124 | 4654 |
| Probe trigger, up to 2 restarts | 10907 | 8615 |
| Judge trigger (Doctor-RAG-style) | 10381 | 8659 |
| Always retrieve | 8797 | 6245 |

Other judges, cost per trajectory: Qwen2.5-3B-Instruct 649 ms, Qwen2.5-7B-Instruct 1556 ms, Llama-3.1-8B-Instruct 1667 ms

Probe cost per hop boundary: 9.45 ms; judge cost per trajectory: 737.5 ms; retrieval: 153.6 ms.
Simulated-vs-live equivalence: {'first_pass_is_prefix': 1.0, 'same_trigger': 1.0, 'same_n_restarts': 1.0, 'same_answer': 0.98, 'same_correctness': 1.0}

## Sanity: live features vs stored Phase 1 features

Reproduced the Phase 1 trajectory exactly: 476/500. On 395 hops: mean cosine at probe layer 0.99999, |ΔP(fail)| mean 0.0004, max 0.0037.
