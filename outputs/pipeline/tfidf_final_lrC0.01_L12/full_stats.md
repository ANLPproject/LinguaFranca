# Complete statistics — Phases 3–5 (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`, main probe: shared)

Every number below is computed from the run's output files (`scripts/full_stats.py`). Correct = token F1 ≥ 0.5 with the gold answer.

## 1. Data and splits

| | count |
|---|---|
| 2WikiMultihopQA questions (TF-IDF top-5 setting, all with hidden states) | 4000 |
| → train split (probe training + threshold CV) | 3200 |
| → test split (held out) | 800 |
| 2Wiki questions evaluated end to end (random sample of the test split) | 500 |
| HotpotQA questions evaluated (validation, distractor setting, out of domain) | 500 |
| → HotpotQA bridge / comparison questions | 404 / 96 |
| Retrieval corpus (paragraphs) | 94,031 |

- **all 4,000**: hop-1 failures 782 (19.6%); hop-2 failures 2620 (65.5%); first-stage condition {'both_present': 1594, 'only_hop1_paragraph': 2269, 'neither': 92, 'only_hop2_paragraph': 39, 'unresolved': 6}
- **train**: hop-1 failures 641 (20.0%); hop-2 failures 2096 (65.5%); first-stage condition {'both_present': 1271, 'only_hop1_paragraph': 1817, 'neither': 75, 'only_hop2_paragraph': 32, 'unresolved': 5}
- **test**: hop-1 failures 141 (17.6%); hop-2 failures 524 (65.5%); first-stage condition {'both_present': 323, 'only_hop1_paragraph': 452, 'neither': 17, 'only_hop2_paragraph': 7, 'unresolved': 1}

**Probe training rows** (one row per labelled hop; the shared probe uses hop-1 and hop-2 rows together):

| probe | train rows | test rows | layer | τ (calibrated) | test AUROC | test precision | test recall | test FPR |
|---|---|---|---|---|---|---|---|---|
| shared (main) | 6325 | 1586 | 12 | 0.650 | 0.954 | 0.899 | 0.803 | 0.063 |
| dedicated hop 1 | 3200 | 798 | 12 | 0.538 | 0.949 | 0.795 | 0.698 | 0.038 |
| dedicated hop 2 | 3125 | 788 | 12 | 0.686 | 0.929 | 0.928 | 0.836 | 0.120 |

## 2. The 500 evaluated 2Wiki questions

- Label source: {'phase1': 476, 'relabeled': 24, 'unavailable': 0}
- First failing hop of the model's first answer: hop 1 86, hop 2 265, no failure 149
- First-stage condition: {'only_hop1_paragraph': 282, 'both_present': 203, 'neither': 9, 'only_hop2_paragraph': 6}

## 3. Failure detection on live trajectories (2Wiki) — confusion matrices

Positive = the hop (or trajectory) failed. TP = failure flagged, FN = failure missed, FP = correct hop flagged, TN = correct hop passed.

| rows | detector | n | failures | TP | FN | FP | TN | precision | recall | F0.5 | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hop1 | probe (shared) | 499 | 85 | 51 | 34 | 10 | 404 | 0.836 | 0.600 | 0.775 | 0.947 |
| hop1 | probe (dedicated) | 499 | 85 | 59 | 26 | 15 | 399 | 0.797 | 0.694 | 0.774 | 0.949 |
| hop1 | judge | 499 | 85 | 26 | 59 | 105 | 309 | 0.198 | 0.306 | 0.213 | 0.569 |
| hop2 | probe (shared) | 492 | 327 | 277 | 50 | 27 | 138 | 0.911 | 0.847 | 0.898 | 0.925 |
| hop2 | probe (dedicated) | 492 | 327 | 269 | 58 | 25 | 140 | 0.915 | 0.823 | 0.895 | 0.922 |
| hop2 | judge | 492 | 327 | 125 | 202 | 32 | 133 | 0.796 | 0.382 | 0.654 | 0.678 |
| trajectory | probe (shared) | 500 | 351 | 291 | 60 | 29 | 120 | 0.909 | 0.829 | 0.892 | – |
| trajectory | probe (dedicated) | 500 | 351 | 292 | 59 | 25 | 124 | 0.921 | 0.832 | 0.902 | – |
| trajectory | judge | 500 | 351 | 182 | 169 | 47 | 102 | 0.795 | 0.519 | 0.718 | – |

First failing hop localized correctly (on 351 failing trajectories): probe 74.4%, judge 24.8%.

## 4. What the model's first answer (no retrieval) looks like

**2wiki, by retrieval condition:**

| retrieval_condition | n | correct | says not in context | confident wrong | no answer (token cap) |
|---|---|---|---|---|---|
| both_present | 203 | 136 (67.0%) | 2 (1.0%) | 54 (26.6%) | 11 (5.4%) |
| neither | 9 | 0 (0.0%) | 5 (55.6%) | 3 (33.3%) | 1 (11.1%) |
| only_hop1_paragraph | 282 | 30 (10.6%) | 74 (26.2%) | 125 (44.3%) | 53 (18.8%) |
| only_hop2_paragraph | 6 | 1 (16.7%) | 3 (50.0%) | 1 (16.7%) | 1 (16.7%) |
| ALL | 500 | 167 (33.4%) | 84 (16.8%) | 183 (36.6%) | 66 (13.2%) |

**hotpotqa, by retrieval condition:**

| retrieval_condition | n | correct | says not in context | confident wrong | no answer (token cap) |
|---|---|---|---|---|---|
| both_present | 273 | 189 (69.2%) | 0 (0.0%) | 72 (26.4%) | 12 (4.4%) |
| neither | 15 | 1 (6.7%) | 3 (20.0%) | 7 (46.7%) | 4 (26.7%) |
| one_present | 212 | 54 (25.5%) | 9 (4.2%) | 126 (59.4%) | 23 (10.8%) |
| ALL | 500 | 244 (48.8%) | 12 (2.4%) | 205 (41.0%) | 39 (7.8%) |

**hotpotqa, by hotpot type:**

| hotpot_type | n | correct | says not in context | confident wrong | no answer (token cap) |
|---|---|---|---|---|---|
| bridge | 404 | 204 (50.5%) | 11 (2.7%) | 158 (39.1%) | 31 (7.7%) |
| comparison | 96 | 40 (41.7%) | 1 (1.0%) | 47 (49.0%) | 8 (8.3%) |
| ALL | 500 | 244 (48.8%) | 12 (2.4%) | 205 (41.0%) | 39 (7.8%) |

## 5. Every system's outcomes

Fixed = wrong without intervention, right with the system; broken = the reverse.

**2wiki (n = 500)**

| system | correct | wrong | fixed | broken | still right | still wrong | restarted | restarted at hop 1 / hop 2 | EM | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 171 (34.2%) | 329 | 0 | 0 | 171 | 329 | 0 (0.0%) | 0 / 0 | 25.8% | 31.9% |
| Always retrieve | 178 (35.6%) | 322 | 30 | 23 | 148 | 299 | 500 (100.0%) | 0 / 0 | 25.8% | 32.5% |
| Judge trigger | 227 (45.4%) | 273 | 63 | 7 | 164 | 266 | 229 (45.8%) | 131 / 98 | 32.6% | 41.6% |
| Judge trigger, ≤ 2 restarts | 235 (47.0%) | 265 | 69 | 5 | 166 | 260 | 229 (45.8%) | 131 / 98 | 34.4% | 43.1% |
| **Probe trigger (shared probe)** | 269 (53.8%) | 231 | 108 | 10 | 161 | 221 | 320 (64.0%) | 61 / 259 | 40.6% | 50.1% |
| **Probe trigger (shared probe), ≤ 2 restarts** | 297 (59.4%) | 203 | 133 | 7 | 164 | 196 | 320 (64.0%) | 61 / 259 | 44.4% | 54.8% |
| Probe trigger, dedicated probes | 265 (53.0%) | 235 | 104 | 10 | 161 | 225 | 317 (63.4%) | 74 / 243 | 39.6% | 49.0% |
| Probe trigger, accuracy-optimal τ | 277 (55.4%) | 223 | 122 | 16 | 155 | 207 | 391 (78.2%) | 28 / 363 | 41.0% | 50.9% |
| Oracle trigger | 275 (55.0%) | 225 | 114 | 10 | 161 | 215 | 345 (69.0%) | 85 / 260 | 41.0% | 51.0% |
| Judge trigger (Qwen2.5-3B-Instruct) | 228 (45.6%) | 272 | 73 | 16 | 155 | 256 | 443 (88.6%) | 372 / 71 | 33.4% | 41.9% |
| Judge trigger (Qwen2.5-7B-Instruct) | 254 (50.8%) | 246 | 91 | 8 | 163 | 238 | 314 (62.8%) | 108 / 206 | 37.0% | 46.3% |
| Judge trigger (Llama-3.1-8B-Instruct) | 205 (41.0%) | 295 | 37 | 3 | 168 | 292 | 133 (26.6%) | 65 / 68 | 30.6% | 38.2% |

**hotpotqa (n = 500)**

| system | correct | wrong | fixed | broken | still right | still wrong | restarted | restarted at hop 1 / hop 2 | EM | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 252 (50.4%) | 248 | 0 | 0 | 252 | 248 | 0 (0.0%) | 0 / 0 | 36.2% | 48.7% |
| Always retrieve | 273 (54.6%) | 227 | 72 | 51 | 201 | 176 | 500 (100.0%) | 0 / 0 | 37.8% | 52.1% |
| Judge trigger | 263 (52.6%) | 237 | 24 | 13 | 239 | 224 | 113 (22.6%) | 44 / 69 | 37.4% | 50.3% |
| Judge trigger, ≤ 2 restarts | 267 (53.4%) | 233 | 26 | 11 | 241 | 222 | 113 (22.6%) | 44 / 69 | 38.2% | 51.1% |
| **Probe trigger (shared probe)** | 285 (57.0%) | 215 | 55 | 22 | 230 | 193 | 293 (58.6%) | 88 / 205 | 39.8% | 54.3% |
| **Probe trigger (shared probe), ≤ 2 restarts** | 282 (56.4%) | 218 | 60 | 30 | 222 | 188 | 293 (58.6%) | 88 / 205 | 39.4% | 53.9% |
| Probe trigger, dedicated probes | 282 (56.4%) | 218 | 57 | 27 | 225 | 191 | 306 (61.2%) | 118 / 188 | 38.8% | 53.5% |
| Probe trigger, accuracy-optimal τ | 290 (58.0%) | 210 | 73 | 35 | 217 | 175 | 388 (77.6%) | 47 / 341 | 39.8% | 54.8% |
| Judge trigger (Qwen2.5-3B-Instruct) | 278 (55.6%) | 222 | 68 | 42 | 210 | 180 | 375 (75.0%) | 256 / 119 | 37.8% | 52.6% |
| Judge trigger (Qwen2.5-7B-Instruct) | 273 (54.6%) | 227 | 49 | 28 | 224 | 199 | 246 (49.2%) | 124 / 122 | 37.4% | 51.8% |
| Judge trigger (Llama-3.1-8B-Instruct) | 260 (52.0%) | 240 | 17 | 9 | 243 | 231 | 78 (15.6%) | 38 / 40 | 36.6% | 49.8% |

## 6. Probe-triggered restarts in detail (shared probe, one restart)

**2wiki — first-answer category → outcome after the probe trigger:**

| first answer | n | probe restarted | → right after | → wrong after |
|---|---|---|---|---|
| correct | 167 | 50 (29.9%) | 158 | 9 |
| says not in context | 84 | 79 (94.0%) | 34 | 50 |
| confident wrong | 183 | 138 (75.4%) | 54 | 129 |
| no answer (token cap) | 66 | 53 (80.3%) | 23 | 43 |

**hotpotqa — first-answer category → outcome after the probe trigger:**

| first answer | n | probe restarted | → right after | → wrong after |
|---|---|---|---|---|
| correct | 244 | 128 (52.5%) | 222 | 22 |
| says not in context | 12 | 10 (83.3%) | 6 | 6 |
| confident wrong | 205 | 125 (61.0%) | 40 | 165 |
| no answer (token cap) | 39 | 30 (76.9%) | 17 | 22 |

**2Wiki — did the repair retrieval bring back the missing gold paragraph?**

| case | restarts | fixed | broken | still wrong | still right |
|---|---|---|---|---|---|
| gold paragraph missing, retrieved | 131 | 86 | 1 | 37 | 7 |
| gold paragraph missing, not retrieved | 118 | 5 | 3 | 104 | 6 |
| both gold paragraphs already in the prompt | 71 | 17 | 6 | 19 | 29 |
| restarted at hop 1 (all) | 61 | 7 | 1 | 51 | 2 |
| restarted at hop 2 (all) | 259 | 101 | 9 | 109 | 40 |

## 7. Cost per question (means)

| system | dataset | generated tokens | processed tokens | tokens before a failure is flagged |
|---|---|---|---|---|
| No intervention | 2wiki | 143.4 | 911 | – |
| Always retrieve | 2wiki | 147.5 | 1209 | 0.0 |
| Judge trigger | 2wiki | 214.8 | 2576 | 169.7 |
| Judge trigger, ≤ 2 restarts | 2wiki | 251.7 | 3620 | 169.7 |
| **Probe trigger (shared probe)** | 2wiki | 160.1 | 1574 | 53.8 |
| No intervention | hotpotqa | 134.3 | 1072 | – |
| Always retrieve | hotpotqa | 136.1 | 1471 | 0.0 |
| Judge trigger | hotpotqa | 164.4 | 2860 | 124.6 |
| Judge trigger, ≤ 2 restarts | hotpotqa | 174.2 | 3456 | 124.6 |
| **Probe trigger (shared probe)** | hotpotqa | 153.7 | 1831 | 52.4 |

Live latency subset (n = 100, sequential): no_intervention 7124 ms mean, probe_restart_2r 9822 ms mean, judge_restart 10178 ms mean, always_retrieve 8308 ms mean; probe check 9.4 ms per hop; judge 738 ms per answer.
