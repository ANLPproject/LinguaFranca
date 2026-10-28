# Complete statistics — Phases 3–5 (`pipeline_tfidf_v2_r2`, main probe: shared)

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

- **all 4,000**: hop-1 failures 782 (19.6%); hop-2 failures 2620 (65.5%); first-stage condition {'both_present': 1599, 'only_p1': 963, 'only_p2': 1346, 'neither': 92}
- **train**: hop-1 failures 641 (20.0%); hop-2 failures 2096 (65.5%); first-stage condition {'both_present': 1276, 'only_p1': 777, 'only_p2': 1072, 'neither': 75}
- **test**: hop-1 failures 141 (17.6%); hop-2 failures 524 (65.5%); first-stage condition {'both_present': 323, 'only_p2': 274, 'only_p1': 186, 'neither': 17}

**Probe training rows** (one row per labelled hop; the shared probe uses hop-1 and hop-2 rows together):

| probe | train rows | test rows | layer | τ (calibrated) | test AUROC | test precision | test recall | test FPR |
|---|---|---|---|---|---|---|---|---|
| shared (main) | 6325 | 1586 | 10 | 0.639 | 0.931 | 0.887 | 0.780 | 0.070 |
| dedicated hop 1 | 3200 | 798 | 10 | 0.698 | 0.933 | 0.832 | 0.604 | 0.026 |
| dedicated hop 2 | 3125 | 788 | 10 | 0.722 | 0.910 | 0.922 | 0.812 | 0.127 |

## 2. The 500 evaluated 2Wiki questions

- Label source: {'phase1': 476, 'relabeled': 24, 'unavailable': 0}
- First failing hop of the model's first answer: hop 1 86, hop 2 265, no failure 149
- First-stage condition: {'only_p2': 165, 'both_present': 203, 'only_p1': 123, 'neither': 9}

## 3. Failure detection on live trajectories (2Wiki) — confusion matrices

Positive = the hop (or trajectory) failed. TP = failure flagged, FN = failure missed, FP = correct hop flagged, TN = correct hop passed.

| rows | detector | n | failures | TP | FN | FP | TN | precision | recall | F0.5 | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hop1 | probe (shared) | 499 | 85 | 61 | 24 | 18 | 396 | 0.772 | 0.718 | 0.761 | 0.918 |
| hop1 | probe (dedicated) | 499 | 85 | 50 | 35 | 9 | 405 | 0.847 | 0.588 | 0.779 | 0.927 |
| hop1 | judge | 499 | 85 | 26 | 59 | 105 | 309 | 0.198 | 0.306 | 0.213 | 0.569 |
| hop2 | probe (shared) | 492 | 327 | 264 | 63 | 17 | 148 | 0.940 | 0.807 | 0.910 | 0.902 |
| hop2 | probe (dedicated) | 492 | 327 | 267 | 60 | 24 | 141 | 0.918 | 0.817 | 0.895 | 0.910 |
| hop2 | judge | 492 | 327 | 125 | 202 | 32 | 133 | 0.796 | 0.382 | 0.654 | 0.678 |
| trajectory | probe (shared) | 500 | 351 | 281 | 70 | 19 | 130 | 0.937 | 0.801 | 0.906 | – |
| trajectory | probe (dedicated) | 500 | 351 | 284 | 67 | 22 | 127 | 0.928 | 0.809 | 0.902 | – |
| trajectory | judge | 500 | 351 | 182 | 169 | 47 | 102 | 0.795 | 0.519 | 0.718 | – |

First failing hop localized correctly (on 351 failing trajectories): probe 72.4%, judge 24.8%.

## 4. What the model's first answer (no retrieval) looks like

**2wiki, by retrieval condition:**

| retrieval_condition | n | correct | says not in context | confident wrong | no answer (token cap) |
|---|---|---|---|---|---|
| both_present | 203 | 136 (67.0%) | 2 (1.0%) | 54 (26.6%) | 11 (5.4%) |
| neither | 9 | 0 (0.0%) | 5 (55.6%) | 3 (33.3%) | 1 (11.1%) |
| only_p1 | 123 | 18 (14.6%) | 32 (26.0%) | 52 (42.3%) | 21 (17.1%) |
| only_p2 | 165 | 13 (7.9%) | 45 (27.3%) | 74 (44.8%) | 33 (20.0%) |
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
| No intervention | 167 (33.4%) | 333 | 0 | 0 | 167 | 333 | 0 (0.0%) | 0 / 0 | 25.8% | 31.3% |
| Always retrieve | 172 (34.4%) | 328 | 29 | 24 | 143 | 304 | 500 (100.0%) | 0 / 0 | 25.8% | 31.5% |
| Judge trigger | 220 (44.0%) | 280 | 62 | 9 | 158 | 271 | 229 (45.8%) | 131 / 98 | 32.2% | 40.4% |
| Judge trigger, ≤ 2 restarts | 228 (45.6%) | 272 | 68 | 7 | 160 | 265 | 229 (45.8%) | 131 / 98 | 33.8% | 41.9% |
| **Probe trigger (shared probe)** | 248 (49.6%) | 252 | 92 | 11 | 156 | 241 | 300 (60.0%) | 79 / 221 | 38.2% | 46.2% |
| Probe trigger, dedicated probes | 255 (51.0%) | 245 | 98 | 10 | 157 | 235 | 306 (61.2%) | 59 / 247 | 39.0% | 47.5% |
| Probe trigger, dedicated probes, ≤ 2 restarts | 282 (56.4%) | 218 | 122 | 7 | 160 | 211 | 306 (61.2%) | 59 / 247 | 42.6% | 52.0% |
| Probe trigger, accuracy-optimal τ | 266 (53.2%) | 234 | 118 | 19 | 148 | 215 | 458 (91.6%) | 20 / 438 | 40.0% | 48.8% |
| Oracle trigger | 270 (54.0%) | 230 | 113 | 10 | 157 | 220 | 345 (69.0%) | 85 / 260 | 40.6% | 49.8% |

**hotpotqa (n = 500)**

| system | correct | wrong | fixed | broken | still right | still wrong | restarted | restarted at hop 1 / hop 2 | EM | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| No intervention | 244 (48.8%) | 256 | 0 | 0 | 244 | 256 | 0 (0.0%) | 0 / 0 | 35.2% | 47.0% |
| Always retrieve | 260 (52.0%) | 240 | 70 | 54 | 190 | 186 | 500 (100.0%) | 0 / 0 | 36.6% | 49.6% |
| Judge trigger | 255 (51.0%) | 245 | 25 | 14 | 230 | 231 | 113 (22.6%) | 44 / 69 | 36.4% | 48.7% |
| Judge trigger, ≤ 2 restarts | 259 (51.8%) | 241 | 27 | 12 | 232 | 229 | 113 (22.6%) | 44 / 69 | 37.2% | 49.4% |
| **Probe trigger (shared probe)** | 269 (53.8%) | 231 | 57 | 32 | 212 | 199 | 296 (59.2%) | 125 / 171 | 37.4% | 51.0% |
| Probe trigger, dedicated probes | 279 (55.8%) | 221 | 64 | 29 | 215 | 192 | 305 (61.0%) | 90 / 215 | 40.0% | 53.0% |
| Probe trigger, dedicated probes, ≤ 2 restarts | 277 (55.4%) | 223 | 66 | 33 | 211 | 190 | 305 (61.0%) | 90 / 215 | 39.2% | 53.1% |
| Probe trigger, accuracy-optimal τ | 281 (56.2%) | 219 | 88 | 51 | 193 | 168 | 487 (97.4%) | 35 / 452 | 38.6% | 53.3% |

The shared probe with ≤ 2 restarts was computed in the final Kaggle run (`outputs/pipeline/tfidf_v2_r2_shared/results.md`): 2Wiki 276/500 correct (55.2%), fixed 117, broken 8; HotpotQA 265/500 (53.0%), fixed 54, broken 33.

## 6. Probe-triggered restarts in detail (shared probe, one restart)

**2wiki — first-answer category → outcome after the probe trigger:**

| first answer | n | probe restarted | → right after | → wrong after |
|---|---|---|---|---|
| correct | 167 | 50 (29.9%) | 156 | 11 |
| says not in context | 84 | 76 (90.5%) | 34 | 50 |
| confident wrong | 183 | 126 (68.9%) | 42 | 141 |
| no answer (token cap) | 66 | 48 (72.7%) | 16 | 50 |

**hotpotqa — first-answer category → outcome after the probe trigger:**

| first answer | n | probe restarted | → right after | → wrong after |
|---|---|---|---|---|
| correct | 244 | 125 (51.2%) | 212 | 32 |
| says not in context | 12 | 11 (91.7%) | 6 | 6 |
| confident wrong | 205 | 137 (66.8%) | 45 | 160 |
| no answer (token cap) | 39 | 23 (59.0%) | 6 | 33 |

**2Wiki — did the repair retrieval bring back the missing gold paragraph?**

| case | restarts | fixed | broken | still wrong | still right |
|---|---|---|---|---|---|
| gold paragraph missing, retrieved | 113 | 76 | 1 | 30 | 6 |
| gold paragraph missing, not retrieved | 115 | 4 | 2 | 105 | 4 |
| both gold paragraphs already in the prompt | 72 | 12 | 8 | 23 | 29 |
| restarted at hop 1 (all) | 79 | 8 | 0 | 67 | 4 |
| restarted at hop 2 (all) | 221 | 84 | 11 | 91 | 35 |

## 7. Cost per question (means)

| system | dataset | generated tokens | processed tokens | tokens before a failure is flagged |
|---|---|---|---|---|
| No intervention | 2wiki | 142.1 | 909 | – |
| Always retrieve | 2wiki | 145.9 | 1207 | 0.0 |
| Judge trigger | 2wiki | 212.4 | 2573 | 167.9 |
| Judge trigger, ≤ 2 restarts | 2wiki | 249.0 | 3618 | 167.9 |
| **Probe trigger (shared probe)** | 2wiki | 157.4 | 1526 | 52.0 |
| No intervention | hotpotqa | 133.6 | 1071 | – |
| Always retrieve | hotpotqa | 135.3 | 1470 | 0.0 |
| Judge trigger | hotpotqa | 163.6 | 2859 | 124.0 |
| Judge trigger, ≤ 2 restarts | hotpotqa | 173.2 | 3455 | 124.0 |
| **Probe trigger (shared probe)** | hotpotqa | 158.1 | 1838 | 48.6 |

Live latency subset (n = 100, sequential): no_intervention 7124 ms mean, probe_restart_2r_alt 9449 ms mean, judge_restart 10096 ms mean, always_retrieve 7938 ms mean; probe check 9.4 ms per hop; judge 738 ms per answer.
