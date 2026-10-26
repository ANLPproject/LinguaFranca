# Phase 3–5 results — setting `tfidf`

Items evaluated: 1000.  Probe layers {1: 10, 2: 10}, thresholds {1: 0.987, 2: 0.917} (F0.5-optimal on train-split CV).

## RQ1 — failure detection (2Wiki held-out test, live trajectories)

Label sources: {'phase1': 476, 'relabeled': 24, 'unavailable': 0}

| rows | trigger | n | fail | AUROC | precision | recall | F0.5 | FPR | flag rate |
|---|---|---|---|---|---|---|---|---|---|
| hop1 | probe | 499 | 85 | 0.927 | 0.847 | 0.588 | 0.779 | 0.022 | 0.118 |
| hop1 | judge | 499 | 85 | 0.569 | 0.198 | 0.306 | 0.213 | 0.254 | 0.263 |
| hop2 | probe | 492 | 327 | 0.910 | 0.918 | 0.817 | 0.895 | 0.145 | 0.591 |
| hop2 | judge | 492 | 327 | 0.678 | 0.796 | 0.382 | 0.654 | 0.194 | 0.319 |
| trajectory | probe | 500 | 351 | – | 0.928 | 0.809 | 0.902 | 0.148 | 0.612 |
| trajectory | judge | 500 | 351 | – | 0.795 | 0.519 | 0.718 | 0.315 | 0.458 |

First-failure localization accuracy (on failing trajectories): probe 0.744, judge 0.248. Exact first-fail hop incl. 'no failure' (all trajectories): probe 0.776, judge 0.378.

## RQ3 — end-to-end accuracy, 2wiki

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.334 | – | 0.258 | 0.313 | 0.000 | 0 | 0 | 142.1 | 909 | – |
| Probe trigger (ours) | 0.472 | +0.138 [+0.104, +0.174] | 0.348 | 0.435 | 0.612 | 80 | 11 | 165.1 | 1535 | 54.2 |
| Judge trigger (Doctor-RAG-style) | 0.452 | +0.118 [+0.086, +0.150] | 0.332 | 0.414 | 0.458 | 66 | 7 | 210.9 | 2567 | 167.9 |
| Always retrieve | 0.346 | +0.012 [-0.020, +0.044] | 0.264 | 0.320 | 1.000 | 34 | 28 | 154.4 | 1166 | 0.0 |
| Oracle trigger | 0.502 | +0.168 [+0.134, +0.204] | 0.378 | 0.466 | 0.690 | 93 | 9 | 164.3 | 1615 | 52.0 |
| Probe trigger, one all-hops probe (ablation) | 0.464 | +0.130 [+0.096, +0.164] | 0.346 | 0.429 | 0.600 | 76 | 11 | 163.9 | 1525 | 52.0 |

Probe − judge accuracy, paired bootstrap 95% CI: [-0.014, +0.052]

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Oracle trigger |
|---|---|---|---|---|---|---|
| both_present | 203 | 0.670 | 0.680 | 0.675 | 0.675 | 0.714 |
| neither | 9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| only_p1 | 123 | 0.146 | 0.382 | 0.341 | 0.171 | 0.398 |
| only_p2 | 165 | 0.079 | 0.309 | 0.285 | 0.091 | 0.345 |

## RQ3 — end-to-end accuracy, hotpotqa

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.488 | – | 0.352 | 0.470 | 0.000 | 0 | 0 | 133.6 | 1071 | – |
| Probe trigger (ours) | 0.576 | +0.088 [+0.052, +0.124] | 0.416 | 0.546 | 0.610 | 69 | 25 | 165.8 | 1887 | 53.0 |
| Judge trigger (Doctor-RAG-style) | 0.516 | +0.028 [+0.006, +0.050] | 0.370 | 0.495 | 0.226 | 24 | 10 | 163.5 | 2855 | 124.0 |
| Always retrieve | 0.508 | +0.020 [-0.022, +0.062] | 0.342 | 0.481 | 1.000 | 62 | 52 | 131.6 | 1418 | 0.0 |
| Probe trigger, one all-hops probe (ablation) | 0.556 | +0.068 [+0.034, +0.104] | 0.384 | 0.521 | 0.592 | 59 | 25 | 163.8 | 1848 | 48.6 |

Probe − judge accuracy, paired bootstrap 95% CI: [+0.024, +0.096]

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve |
|---|---|---|---|---|---|
| both_present | 273 | 0.692 | 0.700 | 0.689 | 0.648 |
| neither | 15 | 0.067 | 0.533 | 0.133 | 0.400 |
| one_present | 212 | 0.255 | 0.420 | 0.321 | 0.335 |

## Final-answer failure prediction (AUROC of max hop P(fail) vs. wrong final answer)

| dataset | n | failure rate | probe | judge |
|---|---|---|---|---|
| 2wiki | 500 | 0.666 | 0.824 | 0.686 |
| hotpotqa | 500 | 0.512 | 0.575 | 0.571 |

## Latency (sequential, live subset n=100)

| system | mean ms | median ms |
|---|---|---|
| No intervention | 7124 | 4654 |
| Probe trigger (ours) | 7844 | 6167 |
| Judge trigger (Doctor-RAG-style) | 10411 | 8724 |
| Always retrieve | 8127 | 5314 |

Probe cost per hop boundary: 9.45 ms; judge cost per trajectory: 737.5 ms; retrieval: 21.5 ms.
Simulated-vs-live equivalence: {'first_pass_is_prefix': 1.0, 'same_trigger': 1.0, 'same_answer': 0.99, 'same_correctness': 1.0}

## RQ4 — where failure becomes linearly decodable (train-split CV AUROC)

* hop1: best layer 13 (AUROC 0.943); earliest layer within 0.01 of best: 10
* hop2: best layer 14 (AUROC 0.925); earliest layer within 0.01 of best: 9
* hop12: best layer 14 (AUROC 0.940); earliest layer within 0.01 of best: 10

## Sanity: live features vs stored Phase 1 features

Reproduced the Phase 1 trajectory exactly: 476/500. On 395 hops: mean cosine at probe layer 0.99999, |ΔP(fail)| mean 0.0004, max 0.0128.
