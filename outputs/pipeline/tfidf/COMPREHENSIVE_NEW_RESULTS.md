# TF-IDF setting: results and diagnostics (corrected)

**Setting:** 2WikiMultihopQA, 10-paragraph pool, TF-IDF top-5 context; HotpotQA as out-of-domain. Llama-3.2-3B-Instruct, greedy, fp16.
**Probe:** StandardScaler + logistic regression, layer 10, Platt-calibrated, τ from train-split CV (F0.5). **Main system: one shared probe (τ = 0.639).** Dedicated hop-1/hop-2 probes (τ = 0.698 / 0.722) are an ablation.

**Sources.** Tables 1–6 and 8 are produced by `scripts/generate_diagnostics_report.py` → `diagnostics.md` / `diagnostics.json` (labels = the pipeline labels in `probes/test_predictions.jsonl`, deployed layer-10 probes, one hedge list from `src/pipeline/common.py`). Tables 7 and 9 are from `../tfidf_v2_r2_shared/results.md`. This file replaces an earlier version that mixed a different label file, a layer-14 analysis from an unsaved Kaggle log, and the retracted alphabetical retrieval buckets.

---

## Summary

1. **Detection.** The shared layer-10 probe reaches AUROC 0.931 on hops 1+2 of the held-out split (n = 1,586 hops); dedicated probes 0.933 (hop 1) and 0.910 (hop 2). Baselines on the same rows: hop index only 0.746, TF-IDF on hop text 0.819. On live trajectories: hop-1 AUROC 0.918, hop-2 0.902, versus 0.569 and 0.678 for the zero-shot judge.
2. **Most hop-2 failures are missing evidence, not error compounding.** Only 19% of hop-2 failures (99/512) follow a failed hop 1. When hop 1 succeeded, hop 2 still fails 41.6% of the time with both gold paragraphs present and 82.8% when the hop-2 paragraph is missing.
3. **The probe is weakest where it matters.** On hop-1 confident failures it catches 56% of "dodged" steps, 47% of steps that copy a distractor entity, and 90% (9/10) of pure fabrications, at the deployed τ.
4. **End to end (2Wiki).** 33.4% → 49.6% with the probe trigger (55.2% with up to 2 restarts), versus 44.0% (45.6%) for the judge trigger. On HotpotQA the probe reaches 53.8% (53.0% with 2 restarts), and the difference to the judge is not significant.
5. **Cost.** The probe flags a failure after 52 tokens on average versus 168 for the judge, with about 41% fewer processed tokens. End-to-end wall-clock latency is about the same (9.9 s vs 10.1 s).

---

## 1. Retrieval conditions in true hop order

The earlier `only_p1` / `only_p2` buckets split titles alphabetically and are retracted. `fix_retrieval_conditions.py` identifies the hop-1 title as the one mentioned in the question (3,906 of 4,000 records), falls back to the bridge entity (92), and falls back to alphabetical order for only 2 records.

| condition | all (N=4000) | test (N=800) | eval set (500 questions) |
|---|---|---|---|
| both_present | 1,599 (40.0%) | 323 (40.4%) | 203 |
| only_hop1_passage | 2,263 (56.6%) | 450 (56.2%) | 281 |
| only_hop2_passage | 46 (1.1%) | 10 (1.2%) | 7 |
| neither | 92 (2.3%) | 17 (2.1%) | 9 |

The totals of missing gold paragraphs are unchanged (306 of 1,000 prompt slots in the 500-question eval set). Only their split between hop 1 and hop 2 changes. Since TF-IDF scores paragraphs against the question, which names only the hop-1 entity, the hop-2 paragraph is the one that gets dropped.

## 2. Failure rates and hop-2 decomposition (test split, generated hops only)

| | failures | rate |
|---|---|---|
| Hop 1 | 139 / 798 | 17.4% |
| Hop 2 | 512 / 788 | 65.0% |

12 test questions never closed hop 2 and are excluded from both the rates and the probe evaluation.

* Cascading (hop 1 failed and hop 2 failed): 99 / 512 = **19.3%** of hop-2 failures.
* Hop-2 failure rate given hop 1 succeeded: 413 / 651 = 63.4%.

| condition (hop 1 succeeded) | n | hop-2 fails | rate |
|---|---|---|---|
| both_present | 303 | 126 | 41.6% |
| only_hop1_passage | 343 | 284 | 82.8% |
| only_hop2_passage | 5 | 3 | 60.0% (too few) |

This is an association. The hop-2 paragraph being absent is not randomized.

A manual check of 30 hop-2 failures with both paragraphs present found: wrong attribute (e.g. date of birth instead of place of birth) 26, date-format mismatch 2, hop 2 never completed 2. This hand-check is not reproducible from the repository (the 30 ids are not stored), so quote it as qualitative only.

## 3. Probe detection (test split, layer 10, deployed τ)

"Conf." removes hedging failures ("no information", …) so that confident failures are compared with successes.

| probe | rows | fails | τ | AUROC | AUROC conf. | conf. fails | recall @5% FPR | recall @10% FPR | recall @ τ (conf.) |
|---|---|---|---|---|---|---|---|---|---|
| hop 1 (dedicated) | 798 | 139 | 0.698 | 0.933 | 0.928 | 116 | 67.2% | 81.9% | 55.2% |
| hop 2 (dedicated) | 788 | 512 | 0.722 | 0.910 | 0.908 | 465 | 56.1% | 72.3% | 80.9% |
| hops 1+2 (shared, main) | 1,586 | 651 | 0.639 | 0.931 | 0.928 | 581 | 69.7% | 81.1% | 77.5% |

Baselines on the same hops 1–2 rows: majority-class accuracy 0.580; hop-index-only AUROC 0.746; TF-IDF on hop text AUROC 0.819; shared probe 0.931. The hop-index baseline is high because hop 2 fails 65% of the time versus 17% for hop 1, so pooled numbers partly reflect the base rate. Report the per-hop numbers as the headline.

**AUROC within retrieval condition** (shared probe): both_present 0.921 (hop 1), 0.896 (hop 2); only_hop1_passage 0.902 (hop 1), 0.873 (hop 2). The other groups have too few rows or one class and are not interpretable (e.g. `neither`, hop 1: 16 of 17 are failures). The probe is therefore not simply detecting passage absence, but it is a little worse (not "identical") when a passage is missing.

**Layer profile** (train-split grouped 5-fold CV AUROC, block output L): hop 1 rises from 0.827 (L0) to a plateau of 0.934–0.943 over L10–L15 (best L13, 0.943), then declines to 0.888 (L27). Hop 2 plateaus at 0.920–0.925 over L10–L15. Full table: `diagnostics.md` §7. Layer 10 was fixed in advance and is within 0.01 of the best layer for every task. Layer 0 here is the output of the first block, not the raw embeddings.

## 4. Hop-1 confident failures (N = 116, dedicated hop-1 probe, τ = 0.698)

| category | n | share | flagged | recall | 95% Wilson CI |
|---|---|---|---|---|---|
| Restated the question's own subject ("dodge") | 59 | 50.9% | 33 | 55.9% | [43.3%, 67.8%] |
| Entity copied from a distractor paragraph | 47 | 40.5% | 22 | 46.8% | [33.3%, 60.8%] |
| Pure fabrication (entity not in context or question) | 10 | 8.6% | 9 | 90.0% | [59.6%, 98.2%] |

Categories are assigned from the extracted entity (`bridging_entity_pred`) by string containment, so they are heuristic. The fabrication group is small, so its interval is wide.

## 5. End-to-end accuracy (token F1 ≥ 0.5)

| system | 2Wiki | Δ vs none [95% CI] | fixed / broken | HotpotQA | Δ vs none [95% CI] | fixed / broken |
|---|---|---|---|---|---|---|
| No intervention | 0.334 | – | – | 0.488 | – | – |
| Always retrieve | 0.344 | +0.010 [−0.020, +0.038] | 29 / 24 | 0.520 | +0.032 [−0.012, +0.072] | 70 / 54 |
| Judge trigger | 0.440 | +0.106 [+0.074, +0.138] | 62 / 9 | 0.510 | +0.022 [−0.002, +0.046] | 25 / 14 |
| Judge trigger, ≤ 2 restarts | 0.456 | +0.122 [+0.090, +0.154] | 68 / 7 | 0.518 | +0.030 [+0.006, +0.054] | 27 / 12 |
| **Probe trigger (shared)** | **0.496** | +0.162 [+0.124, +0.200] | 92 / 11 | **0.538** | +0.050 [+0.014, +0.084] | 57 / 32 |
| **Probe trigger, ≤ 2 restarts** | **0.552** | +0.218 [+0.178, +0.258] | 117 / 8 | 0.530 | +0.042 [+0.006, +0.078] | 54 / 33 |
| Oracle trigger (1 restart) | 0.540 | +0.206 [+0.166, +0.244] | 113 / 10 | – | – | – |

Probe minus judge: 2Wiki +5.6 points, CI [+2.2, +9.2] (≤ 2 restarts +9.6, CI [+5.6, +13.4]). HotpotQA +2.8 points, CI [−0.8, +6.4], so **not significant**. On HotpotQA the probe breaks 32 originally correct answers (judge 14).

**By first-stage evidence (2Wiki, 500 questions).** Per-hop rows from the old table are retracted. The combined group "exactly one gold paragraph present" is unaffected, because the two old buckets together contain the same 288 questions as the corrected `only_hop1_passage` + `only_hop2_passage`:

| group | n | none | judge | probe | probe ≤ 2 | oracle |
|---|---|---|---|---|---|---|
| both gold present | 203 | 0.670 | 0.680 | 0.690 | 0.719 | 0.714 |
| exactly one gold present | 288 | 0.108 | 0.285 | 0.375 | 0.434 | 0.434 |
| neither | 9 | 0.000 | 0.000 | 0.000 | 0.556 | 0.000 |

(derived by size-weighting the two old buckets). To get separate hop-1-only and hop-2-only rows run `scripts/remap_eval_conditions.py` and re-run `evaluate` on the saved run outputs. These rows are not reported here.

## 6. Efficiency and localization

| metric | probe (shared) | judge (same 3B model) |
|---|---|---|
| tokens generated before the failure is flagged | 52.0 | 167.9 (3.2× later) |
| processed tokens per question (1 restart) | 1,526 | 2,573 (probe uses 40.7% fewer) |
| scoring cost | 9.45 ms per hop boundary | 737.5 ms per trajectory (two hops, about 369 ms per hop, about 39× slower per hop) |
| end-to-end wall clock (live subset, n = 100) | 9.9 s (≤ 2 restarts) | 10.1 s (1 restart) |
| first-failure localization on failing trajectories | 72.4% | 24.8% |
| exact first-fail hop including "no failure" | 76.8% | 37.8% |

The probe's cost advantage is in tokens and detection time, not in end-to-end latency, because the probe restarts on 60% of questions.

---

## 7. Corrections relative to the previous version of this file

| previous statement | status |
|---|---|
| Headline "dedicated probes at layer 14, AUROC 0.925 / 0.915" | Replaced. Deployed probes are layer 10 (shared main). 0.925 / 0.915 were confident-only AUROCs of the dedicated probes and are now 0.928 / 0.908 with one hedge list. |
| Layer 14 "best single layer", CV AUROC 0.953, shuffled-label 0.464, per-layer accuracy/F1 table | Removed. They came from a Kaggle log not stored in the repository. Use `diagnostics.md` §7. Shuffled-label control for the earlier setting is in `imppoints.md` §2. |
| "Pure fabrications detected 100% (13/13 and 12/12)" | Now 9/10 at τ. Counts differ because the old script used a different hedge list and included rows without predictions. |
| "Distractor grounding 60.2%" | That number pooled 55–59 dodged questions with the 47 distractor copies. Separated in §4: 55.9% and 46.8%. |
| "Recall @ 0.5 cutoff" | Replaced by recall at the deployed τ. |
| Hop failure counts 147 / 533 of 800 | Now 139 / 798 and 512 / 788, on the pipeline labels and generated hops only. |
| Table 8 per-hop retrieval rows | Retracted (§5). |
| "Identical AUROC within every retrieval condition" | Not supported; see §3. |
| "3.2× latency", "78× faster" | Corrected in §6. |
| Localization 74.4% / 77.6% | These belong to the dedicated-probe run. The shared main run gives 72.4% / 76.8%. |

---

## 8. Probe accuracy: all 10 passages (Phase 1/2 setting) vs TF-IDF top-5 (distractor setting)

**Note on "gold only".** The mid-term report says the prompt held only the gold passages. That filter silently failed (`imppoints.md` §1): every Phase 1 prompt contained all 10 passages (2 gold + 8 distractors). A gold-only result was never produced. The valid comparison is therefore *all 10 present* vs *TF-IDF top-5, where a gold paragraph is often missing*. The mid-term numbers (92.5% accuracy, AUROC 0.971, 98.2% confident-wrong) also used the old labels and are retracted.

Held-out test split, layer-wise probe, threshold 0.5 on the balanced logistic regression (same rule in both settings). Sources: `outputs/probing/results.md` (all 10, corrected labels) and `probes/test_predictions.jsonl` (TF-IDF).

| task | setting | layer | n | failure rate | acc | bal. acc | AUROC | TPR | FPR |
|---|---|---|---|---|---|---|---|---|---|
| hop 1 | all 10 passages | 10 | 814 | 8.4% | 0.951 | 0.860 | 0.953 | 0.750 | 0.031 |
| hop 1 | TF-IDF top-5 | 10 | 798 | 17.4% | 0.906 | 0.849 | 0.933 | 0.763 | 0.064 |
| hop 2 | all 10 passages | 13 | 814 | 40.8% | 0.899 | 0.896 | 0.949 | 0.880 | 0.087 |
| hop 2 | TF-IDF top-5 | 10 | 788 | 65.0% | 0.848 | 0.837 | 0.910 | 0.873 | 0.199 |
| hops 1+2 | all 10 passages | 13 | 1,628 | 24.6% | 0.909 | 0.887 | 0.956 | 0.845 | 0.070 |
| hops 1+2 | TF-IDF top-5 (shared) | 10 | 1,586 | 41.0% | 0.866 | 0.863 | 0.931 | 0.842 | 0.117 |

Text-only baselines on hops 1+2 (AUROC): all 10 passages: TF-IDF on hop text 0.770, hop index 0.719; TF-IDF setting: 0.819 and 0.746. The probe's margin over the text baseline is +0.19 (all 10) and +0.11 (TF-IDF).

At the deployed thresholds (TF-IDF setting): hop 1 acc 0.910, TPR 0.604, FPR 0.026; shared probe acc 0.869, TPR 0.780, FPR 0.070.

**Reading it.**
* The signal is still strong with distractors and missing evidence (AUROC 0.91–0.93), but it is lower: −0.02 on hop 1, −0.04 on hop 2, −0.025 pooled. False alarms on hop 2 roughly double (0.087 → 0.199 at 0.5).
* Accuracy is not comparable across settings because the base rate changes (hop-1 failures double, hop 2 goes from 41% to 65%). Use AUROC and balanced accuracy.
* The drop is not only due to missing evidence: inside `both_present` questions the TF-IDF-setting AUROC is 0.921 (hop 1) and 0.896 (hop 2), still below the all-10 numbers. The settings also differ in sample, split and failure mix (here failures include evidence-starved guesses), so the cause cannot be isolated from these runs.
* Layer plateau: all 10: hop-1 CV peak 0.968 at L10–11; TF-IDF: 0.943 at L13, plateau L10–L15.

### Confident failures (hedges removed): all 10 passages vs TF-IDF top-5

Phase 1 check: all 4,078 examples of the Phase 1 dataset (HF revision `ce07fd04`) have exactly 10 `Title:` paragraphs in their context, so the earlier "gold only" setting was never used.

| | all 10 passages | TF-IDF top-5 |
|---|---|---|
| hop-1 confident failures, test | 55 (240 in train CV) | 116 |
| recall at the 0.5 threshold (false-alarm rate) | 0.69 (3.1%) test; 0.76 (2.2%) CV | 0.72 (6.4%) |
| recall at 3% false alarms, hop 1 | ≈ 0.69 | 0.61 |
| hop-1 AUROC, confident only / natural only | 0.942 (natural) | 0.928 (confident) |
| hops 1+2 confident recall (false alarms) | 0.85 (5.3%), hop-specific probes, n = 365 | 0.84 (11.7%) at 0.5; 0.70 at 5% (shared probe, n = 581) |
| hedging failures | only in counterfactuals (13 in test) | natural: 23 (hop 1), 47 (hop 2) in test |

At equal false-alarm rates the all-10 probe is better on confident failures: about 0.69 vs 0.61 on hop 1 (3% false alarms) and 0.85 vs 0.70 on hops 1+2 (5%). At the 0.5 threshold the TF-IDF recall looks equal or higher only because it also raises false alarms. The all-10 hop-1 figure rests on n = 55 test failures, so its interval is wide. The two settings use different probe layers/heads and different samples.
