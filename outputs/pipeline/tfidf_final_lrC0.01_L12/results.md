# Phase 3–5 results — setting `tfidf`

Items evaluated: 1000.  Main probe: one shared probe for both hops; layers {1: 12, 2: 12}, thresholds {1: 0.65, 2: 0.65} (F0.5-optimal on train-split CV).
Answers forced from hops 1-2 after hitting the token cap without <answer>: {'first answers': 105, 'restarts': 441}.

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
| Probe trigger (ours) | 0.538 | +0.196 [+0.156, +0.236] | 0.406 | 0.501 | 0.640 | 108 | 10 | 160.1 | 1574 | 53.8 |
| Judge trigger (Doctor-RAG-style) | 0.454 | +0.112 [+0.080, +0.142] | 0.326 | 0.416 | 0.458 | 63 | 7 | 214.8 | 2576 | 169.7 |
| Always retrieve | 0.356 | +0.014 [-0.016, +0.044] | 0.258 | 0.325 | 1.000 | 30 | 23 | 147.5 | 1209 | 0.0 |
| Oracle trigger | 0.550 | +0.208 [+0.168, +0.246] | 0.410 | 0.510 | 0.690 | 114 | 10 | 157.8 | 1619 | 52.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.554 | +0.212 [+0.168, +0.254] | 0.410 | 0.509 | 0.782 | 122 | 16 | 168.5 | 1743 | 58.0 |
| Probe trigger, up to 2 restarts | 0.594 | +0.252 [+0.210, +0.294] | 0.444 | 0.548 | 0.640 | 133 | 7 | 172.1 | 2049 | 53.8 |
| Judge trigger, up to 2 restarts | 0.470 | +0.128 [+0.096, +0.158] | 0.344 | 0.431 | 0.458 | 69 | 5 | 251.7 | 3620 | 169.7 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.530 | +0.188 [+0.148, +0.226] | 0.396 | 0.490 | 0.634 | 104 | 10 | 158.1 | 1566 | 52.8 |
| Judge trigger (Qwen2.5-3B-Instruct) | 0.456 | +0.114 [+0.078, +0.150] | 0.334 | 0.419 | 0.886 | 73 | 16 | 266.7 | 3068 | 148.0 |
| Judge trigger (Qwen2.5-7B-Instruct) | 0.508 | +0.166 [+0.128, +0.204] | 0.370 | 0.463 | 0.628 | 91 | 8 | 230.8 | 2785 | 156.7 |
| Judge trigger (Llama-3.1-8B-Instruct) | 0.410 | +0.068 [+0.044, +0.092] | 0.306 | 0.382 | 0.266 | 37 | 3 | 180.3 | 2359 | 158.2 |

Probe − judge accuracy, paired bootstrap 95% CI: [+0.050, +0.118]
Probe − judge (Qwen2.5-3B-Instruct) accuracy, paired bootstrap 95% CI: [+0.046, +0.118]
Probe − judge (Qwen2.5-7B-Instruct) accuracy, paired bootstrap 95% CI: [+0.000, +0.060]
Probe − judge (Llama-3.1-8B-Instruct) accuracy, paired bootstrap 95% CI: [+0.090, +0.166]
Probe − judge accuracy, both with up to 2 restarts: [+0.088, +0.162]; share of questions restarted twice: {'probe_restart_2r': 0.35, 'judge_restart_2r': 0.218}
Accuracy-optimal τ (τ1, τ2), chosen on the other half of the 2Wiki questions: {'fold0_fit_on_fold1': (0.78, 0.18), 'fold1_fit_on_fold0': (1.0, 0.2), 'all_2wiki_used_for_hotpotqa': (0.86, 0.18)}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Oracle trigger | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | Judge trigger (Qwen2.5-3B-Instruct) | Judge trigger (Qwen2.5-7B-Instruct) | Judge trigger (Llama-3.1-8B-Instruct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both_present | 203 | 0.680 | 0.734 | 0.700 | 0.650 | 0.729 | 0.734 | 0.754 | 0.704 | 0.744 | 0.709 | 0.754 | 0.709 |
| neither | 9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.556 | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 |
| only_hop1_paragraph | 282 | 0.113 | 0.415 | 0.298 | 0.145 | 0.440 | 0.443 | 0.482 | 0.319 | 0.394 | 0.287 | 0.351 | 0.213 |
| only_hop2_paragraph | 6 | 0.167 | 0.500 | 0.167 | 0.833 | 0.500 | 0.500 | 0.500 | 0.167 | 0.500 | 0.500 | 0.333 | 0.167 |

## RQ3 — end-to-end accuracy, hotpotqa

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.504 | – | 0.362 | 0.487 | 0.000 | 0 | 0 | 134.3 | 1072 | – |
| Probe trigger (ours) | 0.570 | +0.066 [+0.034, +0.098] | 0.398 | 0.543 | 0.586 | 55 | 22 | 153.7 | 1831 | 52.4 |
| Judge trigger (Doctor-RAG-style) | 0.526 | +0.022 [-0.000, +0.046] | 0.374 | 0.503 | 0.226 | 24 | 13 | 164.4 | 2860 | 124.6 |
| Always retrieve | 0.546 | +0.042 [-0.004, +0.082] | 0.378 | 0.521 | 1.000 | 72 | 51 | 136.1 | 1471 | 0.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.580 | +0.076 [+0.036, +0.116] | 0.398 | 0.548 | 0.776 | 73 | 35 | 173.7 | 2088 | 58.9 |
| Probe trigger, up to 2 restarts | 0.564 | +0.060 [+0.024, +0.094] | 0.394 | 0.539 | 0.586 | 60 | 30 | 169.3 | 2368 | 52.4 |
| Judge trigger, up to 2 restarts | 0.534 | +0.030 [+0.008, +0.054] | 0.382 | 0.511 | 0.226 | 26 | 11 | 174.2 | 3456 | 124.6 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.564 | +0.060 [+0.026, +0.094] | 0.388 | 0.535 | 0.612 | 57 | 27 | 154.8 | 1866 | 50.1 |
| Judge trigger (Qwen2.5-3B-Instruct) | 0.556 | +0.052 [+0.012, +0.090] | 0.378 | 0.526 | 0.750 | 68 | 42 | 235.7 | 3606 | 136.6 |
| Judge trigger (Qwen2.5-7B-Instruct) | 0.546 | +0.042 [+0.008, +0.076] | 0.374 | 0.518 | 0.492 | 49 | 28 | 195.2 | 3248 | 139.0 |
| Judge trigger (Llama-3.1-8B-Instruct) | 0.520 | +0.016 [-0.004, +0.036] | 0.366 | 0.498 | 0.156 | 17 | 9 | 153.8 | 2770 | 121.7 |

Probe − judge accuracy, paired bootstrap 95% CI: [+0.010, +0.078]
Probe − judge (Qwen2.5-3B-Instruct) accuracy, paired bootstrap 95% CI: [-0.020, +0.048]
Probe − judge (Qwen2.5-7B-Instruct) accuracy, paired bootstrap 95% CI: [-0.004, +0.056]
Probe − judge (Llama-3.1-8B-Instruct) accuracy, paired bootstrap 95% CI: [+0.018, +0.084]
Probe − judge accuracy, both with up to 2 restarts: [-0.006, +0.066]; share of questions restarted twice: {'probe_restart_2r': 0.334, 'judge_restart_2r': 0.074}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | Judge trigger (Qwen2.5-3B-Instruct) | Judge trigger (Qwen2.5-7B-Instruct) | Judge trigger (Llama-3.1-8B-Instruct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both_present | 273 | 0.711 | 0.700 | 0.685 | 0.667 | 0.678 | 0.670 | 0.692 | 0.689 | 0.674 | 0.689 | 0.700 |
| neither | 15 | 0.067 | 0.533 | 0.267 | 0.600 | 0.533 | 0.533 | 0.200 | 0.467 | 0.533 | 0.333 | 0.200 |
| one_present | 212 | 0.269 | 0.406 | 0.340 | 0.387 | 0.458 | 0.429 | 0.354 | 0.410 | 0.406 | 0.377 | 0.311 |

## Final-answer failure prediction (AUROC of max hop P(fail) vs. wrong final answer)

| dataset | n | failure rate | probe | judge |
|---|---|---|---|---|
| 2wiki | 500 | 0.658 | 0.837 | 0.682 |
| hotpotqa | 500 | 0.496 | 0.595 | 0.570 |

## Latency (sequential, live subset n=100)

| system | mean ms | median ms |
|---|---|---|
| No intervention | 7124 | 4654 |
| Probe trigger, up to 2 restarts | 9822 | 7583 |
| Judge trigger (Doctor-RAG-style) | 10178 | 8558 |
| Always retrieve | 8308 | 5686 |

Other judges, cost per trajectory: Qwen2.5-3B-Instruct 649 ms, Qwen2.5-7B-Instruct 1556 ms, Llama-3.1-8B-Instruct 1667 ms

Probe cost per hop boundary: 9.45 ms; judge cost per trajectory: 737.5 ms; retrieval: 105.0 ms.
Simulated-vs-live equivalence: {'first_pass_is_prefix': 1.0, 'same_trigger': 1.0, 'same_n_restarts': 1.0, 'same_answer': 0.98, 'same_correctness': 0.99}

## Sanity: live features vs stored Phase 1 features

Reproduced the Phase 1 trajectory exactly: 476/500. On 395 hops: mean cosine at probe layer 0.99999, |ΔP(fail)| mean 0.0004, max 0.0037.
