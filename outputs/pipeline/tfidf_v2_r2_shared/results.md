# Phase 3–5 results — setting `tfidf`

> **Correction (Oct 28):** the stored TF-IDF `retrieval_condition` labels were in *alphabetical* title order (the data predates the hop-order fix), so `only_p1` / `only_p2` below are **not** hop 1 / hop 2. In true hop order, the TF-IDF first stage almost always keeps the hop-1 paragraph: of the 500 2Wiki questions, 203 keep both, 282 miss the hop-2 paragraph, 6 miss the hop-1 paragraph, 9 miss both. The union of the two `only_*` rows (n = 288, "one gold paragraph missing") is still correct. Corrected per-hop tables: `outputs/pipeline/tfidf_v2_r2/results.md`, `outputs/pipeline/full_stats.md`. Accuracy and detection numbers are unaffected.


Items evaluated: 1000.  Main probe: one shared probe for both hops; layers {1: 10, 2: 10}, thresholds {1: 0.639, 2: 0.639} (F0.5-optimal on train-split CV).

## RQ1 — failure detection (2Wiki held-out test, live trajectories)

Label sources: {'phase1': 476, 'relabeled': 24, 'unavailable': 0}

| rows | trigger | n | fail | AUROC | precision | recall | F0.5 | FPR | flag rate |
|---|---|---|---|---|---|---|---|---|---|
| hop1 | probe | 499 | 85 | 0.918 | 0.772 | 0.718 | 0.761 | 0.043 | 0.158 |
| hop1 | probe_dedicated | 499 | 85 | 0.927 | 0.847 | 0.588 | 0.779 | 0.022 | 0.118 |
| hop1 | judge | 499 | 85 | 0.569 | 0.198 | 0.306 | 0.213 | 0.254 | 0.263 |
| hop2 | probe | 492 | 327 | 0.902 | 0.940 | 0.807 | 0.910 | 0.103 | 0.571 |
| hop2 | probe_dedicated | 492 | 327 | 0.910 | 0.918 | 0.817 | 0.895 | 0.145 | 0.591 |
| hop2 | judge | 492 | 327 | 0.678 | 0.796 | 0.382 | 0.654 | 0.194 | 0.319 |
| trajectory | probe | 500 | 351 | – | 0.937 | 0.801 | 0.906 | 0.128 | 0.600 |
| trajectory | probe_dedicated | 500 | 351 | – | 0.928 | 0.809 | 0.902 | 0.148 | 0.612 |
| trajectory | judge | 500 | 351 | – | 0.795 | 0.519 | 0.718 | 0.315 | 0.458 |

First-failure localization accuracy (on failing trajectories): probe 0.724, judge 0.248. Exact first-fail hop incl. 'no failure' (all trajectories): probe 0.768, judge 0.378.

## RQ3 — end-to-end accuracy, 2wiki

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.334 | – | 0.258 | 0.313 | 0.000 | 0 | 0 | 142.1 | 909 | – |
| Probe trigger (ours) | 0.496 | +0.162 [+0.124, +0.200] | 0.382 | 0.462 | 0.600 | 92 | 11 | 157.4 | 1526 | 52.0 |
| Judge trigger (Doctor-RAG-style) | 0.440 | +0.106 [+0.074, +0.138] | 0.322 | 0.404 | 0.458 | 62 | 9 | 212.4 | 2573 | 167.9 |
| Always retrieve | 0.344 | +0.010 [-0.020, +0.038] | 0.258 | 0.315 | 1.000 | 29 | 24 | 145.9 | 1207 | 0.0 |
| Oracle trigger | 0.540 | +0.206 [+0.166, +0.244] | 0.406 | 0.498 | 0.690 | 113 | 10 | 156.5 | 1617 | 52.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.532 | +0.198 [+0.154, +0.238] | 0.400 | 0.488 | 0.916 | 118 | 19 | 176.6 | 1887 | 58.1 |
| Probe trigger, up to 2 restarts | 0.552 | +0.218 [+0.178, +0.258] | 0.418 | 0.509 | 0.600 | 117 | 8 | 170.6 | 2007 | 52.0 |
| Judge trigger, up to 2 restarts | 0.456 | +0.122 [+0.090, +0.154] | 0.338 | 0.419 | 0.458 | 68 | 7 | 248.3 | 3617 | 167.9 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.510 | +0.176 [+0.138, +0.214] | 0.390 | 0.475 | 0.612 | 98 | 10 | 157.5 | 1541 | 54.2 |

Probe − judge accuracy, paired bootstrap 95% CI: [+0.022, +0.092]
Probe − judge accuracy, both with up to 2 restarts: [+0.056, +0.134]; share of questions restarted twice: {'probe_restart_2r': 0.364, 'judge_restart_2r': 0.218}
Accuracy-optimal τ (τ1, τ2), chosen on the other half of the 2Wiki questions: {'fold0_fit_on_fold1': (0.94, 0.0), 'fold1_fit_on_fold0': (1.0, 0.08), 'all_2wiki_used_for_hotpotqa': (0.94, 0.0)}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Oracle trigger | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) |
|---|---|---|---|---|---|---|---|---|---|---|
| both_present | 203 | 0.670 | 0.690 | 0.680 | 0.635 | 0.714 | 0.700 | 0.719 | 0.690 | 0.700 |
| neither | 9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.556 | 0.111 | 0.000 |
| only_p1 | 123 | 0.146 | 0.415 | 0.301 | 0.187 | 0.463 | 0.472 | 0.488 | 0.309 | 0.439 |
| only_p2 | 165 | 0.079 | 0.345 | 0.273 | 0.121 | 0.412 | 0.400 | 0.394 | 0.297 | 0.358 |

## RQ3 — end-to-end accuracy, hotpotqa

| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 0.488 | – | 0.352 | 0.470 | 0.000 | 0 | 0 | 133.6 | 1071 | – |
| Probe trigger (ours) | 0.538 | +0.050 [+0.014, +0.084] | 0.374 | 0.510 | 0.592 | 57 | 32 | 158.1 | 1838 | 48.6 |
| Judge trigger (Doctor-RAG-style) | 0.510 | +0.022 [-0.002, +0.046] | 0.364 | 0.487 | 0.226 | 25 | 14 | 163.6 | 2859 | 124.0 |
| Always retrieve | 0.520 | +0.032 [-0.012, +0.072] | 0.366 | 0.496 | 1.000 | 70 | 54 | 135.3 | 1470 | 0.0 |
| Probe trigger, accuracy-optimal τ (2-fold CV) | 0.562 | +0.074 [+0.028, +0.118] | 0.386 | 0.533 | 0.974 | 88 | 51 | 187.9 | 2349 | 60.5 |
| Probe trigger, up to 2 restarts | 0.530 | +0.042 [+0.006, +0.078] | 0.364 | 0.504 | 0.592 | 54 | 33 | 177.0 | 2459 | 48.6 |
| Judge trigger, up to 2 restarts | 0.518 | +0.030 [+0.006, +0.054] | 0.372 | 0.494 | 0.226 | 27 | 12 | 173.3 | 3455 | 124.0 |
| Probe trigger, dedicated hop-1 / hop-2 probes (ablation) | 0.558 | +0.070 [+0.032, +0.108] | 0.400 | 0.530 | 0.610 | 64 | 29 | 156.3 | 1868 | 53.0 |

Probe − judge accuracy, paired bootstrap 95% CI: [-0.008, +0.064]
Probe − judge accuracy, both with up to 2 restarts: [-0.024, +0.048]; share of questions restarted twice: {'probe_restart_2r': 0.392, 'judge_restart_2r': 0.074}

Accuracy by first-stage retrieval condition:

| condition | n | No intervention | Probe trigger (ours) | Judge trigger (Doctor-RAG-style) | Always retrieve | Probe trigger, accuracy-optimal τ (2-fold CV) | Probe trigger, up to 2 restarts | Judge trigger, up to 2 restarts | Probe trigger, dedicated hop-1 / hop-2 probes (ablation) |
|---|---|---|---|---|---|---|---|---|---|
| both_present | 273 | 0.692 | 0.663 | 0.667 | 0.648 | 0.637 | 0.678 | 0.674 | 0.674 |
| neither | 15 | 0.067 | 0.400 | 0.267 | 0.533 | 0.400 | 0.400 | 0.200 | 0.533 |
| one_present | 212 | 0.255 | 0.387 | 0.325 | 0.354 | 0.476 | 0.349 | 0.340 | 0.410 |

## Final-answer failure prediction (AUROC of max hop P(fail) vs. wrong final answer)

| dataset | n | failure rate | probe | judge |
|---|---|---|---|---|
| 2wiki | 500 | 0.666 | 0.798 | 0.686 |
| hotpotqa | 500 | 0.512 | 0.595 | 0.571 |

## Latency (sequential, live subset n=100)

| system | mean ms | median ms |
|---|---|---|
| No intervention | 7124 | 4654 |
| Probe trigger, up to 2 restarts | 9862 | 7850 |
| Judge trigger (Doctor-RAG-style) | 10094 | 8483 |
| Always retrieve | 7942 | 5488 |

Probe cost per hop boundary: 9.45 ms; judge cost per trajectory: 737.5 ms; retrieval: 128.4 ms.
Simulated-vs-live equivalence: {'first_pass_is_prefix': 1.0, 'same_trigger': 1.0, 'same_n_restarts': 1.0, 'same_answer': 0.95, 'same_correctness': 1.0}

## RQ4 — where failure becomes linearly decodable (train-split CV AUROC)

* hop1: best layer 13 (AUROC 0.943); earliest layer within 0.01 of best: 10
* hop2: best layer 14 (AUROC 0.925); earliest layer within 0.01 of best: 9
* hop12: best layer 14 (AUROC 0.940); earliest layer within 0.01 of best: 10

## Sanity: live features vs stored Phase 1 features

Reproduced the Phase 1 trajectory exactly: 476/500. On 395 hops: mean cosine at probe layer 0.99999, |ΔP(fail)| mean 0.0006, max 0.0063.
