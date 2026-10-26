# Draft — Experimental setup and results for Phases 3–5

*Source numbers: `outputs/pipeline/tfidf/results.md` (full Kaggle run, 500 2WikiMultihopQA + 500 HotpotQA questions) and `outputs/pipeline/tfidf/probes/probes_meta.json`. Tables in LaTeX form are in `tables.tex` in the run's output zip; the accuracy-vs-cost figure is `sweep.png`.*

---

## Experimental setup

**Retrieval condition.** We simulate a weak first-stage retriever: for each question, TF-IDF ranks the 10 candidate paragraphs (2 gold, 8 distractors) and the model sees the top 5. Both gold paragraphs survive in only 40% of questions (2Wiki), so many reasoning failures are caused by missing evidence, which is the situation a retrieval-based repair is meant to fix. The same procedure is applied to HotpotQA (distractor setting) for out-of-domain evaluation.

**Model and probes.** Llama-3.2-3B-Instruct (fp16, greedy decoding). Two dedicated linear probes (standardized logistic regression) read the mean-pooled residual stream of the hop's tokens at layer 10: one at `</hop1>`, one at `</hop2>`. The proposal specified a single probe applied at every hop boundary. Because hop-1 and hop-2 failures are represented differently (weak cross-hop transfer, Phase 2), we instead train one probe per hop, and report the single-probe design as an ablation. They are trained on the 80% train split of 4,000 questions (grouped split, seed 42). Each trigger threshold τ maximises F0.5 on out-of-fold predictions from 5-fold grouped cross-validation inside the train split, so neither the layer nor τ is tuned on test data. On the held-out split the probes reach AUROC 0.933 (hop 1) and 0.910 (hop 2).

**Systems.** All systems share the model, prompt and repair. The repair queries a FAISS index (all-MiniLM-L6-v2) over a pooled paragraph corpus (all paragraphs of the 4,000 2Wiki dataset questions and the 500 HotpotQA evaluation questions, plus those of 20,000 further random 2Wiki questions, de-duplicated by title) with the question plus the failing hop's text. It prepends the top-3 paragraphs not already in context, flushes the KV cache and regenerates from scratch (*Naive Restart*).
- *No intervention*: one greedy trajectory.
- *Probe trigger (ours)*: generation is halted at the first hop whose probe P(fail) ≥ τ; repair starts immediately.
- *Judge trigger (Doctor-RAG-style)*: after the full trajectory is generated, the same 3B model judges hop 1 and hop 2 (P(no) vs. P(yes) logits); repair starts from the earliest hop judged wrong.
- *Always retrieve*: the repair retrieval is applied before generation, with no trigger.
- *Oracle trigger*: repair at the gold first-failure hop (2Wiki only), an upper bound for any trigger using this repair.

**Evaluation.** Hop labels come from Phase 1 gold-entity matching (24 of 500 regenerated trajectories differed from the stored ones and were relabeled with the same matcher). An answer counts as correct if its token F1 with the gold answer is ≥ 0.5; we also report EM and F1. CIs are 95% paired bootstrap intervals over questions. Every trajectory is generated once with the probe in monitor mode and every restart is pre-computed. Because decoding is greedy, a halted trajectory is an exact prefix of the uninterrupted one, so all systems are evaluated from these cached generations. On a 100-question subset run fully live (halt → retrieve → restart), the cached evaluation agreed with the live system on 100% of triggers and correctness decisions (99% identical answer strings). The live probe features matched the stored training features with mean cosine 0.99999.

---

## RQ1 — Detection and localization

**Table 1.** Failure detection on 2Wiki held-out questions (live trajectories). P/R/F0.5/FPR treat failure as positive; the probe uses its calibrated τ, the judge its own yes/no decision.

| Rows | Trigger | AUROC | Precision | Recall | F0.5 | FPR |
|---|---|---|---|---|---|---|
| Hop 1 (n=499, 85 failures) | Probe | **0.927** | **0.847** | **0.588** | **0.779** | **0.022** |
| | Judge | 0.569 | 0.198 | 0.306 | 0.213 | 0.254 |
| Hop 2 (n=492, 327 failures) | Probe | **0.910** | **0.918** | **0.817** | **0.895** | **0.145** |
| | Judge | 0.678 | 0.796 | 0.382 | 0.654 | 0.194 |
| Trajectory (n=500, 351 failing) | Probe | – | **0.928** | **0.809** | **0.902** | **0.148** |
| | Judge | – | 0.795 | 0.519 | 0.718 | 0.315 |

The internal signal detects failing hops far more reliably than the text-based judge, which is close to chance on hop 1 (AUROC 0.569). The probe also localizes the earliest failing hop correctly in **74.4%** of failing trajectories, against **24.8%** for the judge. Including correctly predicting "no failure", it identifies the exact first-failure hop in 77.6% of all trajectories, vs. 37.8% for the judge. Detection quality is stable from the offline evaluation on stored features (AUROC 0.933 / 0.910) to live generation (0.927 / 0.910).

---

## RQ3 — End-to-end accuracy, cost and latency

**Table 2.** Answer accuracy (F1 ≥ 0.5) with Δ vs. no intervention and its 95% CI; *fixed/broken* = answers turned right/wrong relative to no intervention.

| System | 2Wiki acc | Δ [95% CI] | fixed / broken | HotpotQA acc | Δ [95% CI] | fixed / broken |
|---|---|---|---|---|---|---|
| No intervention | 0.334 | – | – | 0.488 | – | – |
| Always retrieve | 0.346 | +0.012 [−0.020, +0.044] | 34 / 28 | 0.508 | +0.020 [−0.022, +0.062] | 62 / 52 |
| Judge trigger (Doctor-RAG-style) | 0.452 | +0.118 [+0.086, +0.150] | 66 / 7 | 0.516 | +0.028 [+0.006, +0.050] | 24 / 10 |
| **Probe trigger (ours)** | **0.472** | **+0.138 [+0.104, +0.174]** | 80 / 11 | **0.576** | **+0.088 [+0.052, +0.124]** | 69 / 25 |
| Oracle trigger | 0.502 | +0.168 [+0.134, +0.204] | 93 / 9 | – | – | – |

Probe-triggered repair improves accuracy by **+13.8 points** on 2Wiki, recovering **82%** of the oracle trigger's gain (the judge recovers 70%). Out of domain on HotpotQA, without re-tuning τ, it improves by **+8.8 points** and **significantly outperforms the judge trigger** (+6.0 points, paired 95% CI [+2.4, +9.6]). On 2Wiki the probe–judge accuracy difference (+2.0) is not significant (CI [−1.4, +5.2]). Retrieval without a trigger does not help on either dataset: it fixes and breaks answers at similar rates (34/28 and 62/52). *When* to retrieve matters as much as *what* is retrieved.

**Where the gains come from.** On 2Wiki, the gains concentrate where first-stage retrieval dropped a gold paragraph:

| First-stage condition (2Wiki) | n | No intervention | Probe trigger | Judge trigger | Oracle |
|---|---|---|---|---|---|
| Both gold paragraphs retrieved | 203 | 0.670 | 0.680 | 0.675 | 0.714 |
| Only the hop-1 paragraph | 123 | 0.146 | **0.382** | 0.341 | 0.398 |
| Only the hop-2 paragraph | 165 | 0.079 | **0.309** | 0.285 | 0.345 |
| Neither | 9 | 0.000 | 0.000 | 0.000 | 0.000 |

When the evidence is present, every system performs about the same. When a gold paragraph is missing, the probe-triggered restart roughly 2.6–3.9× the accuracy, close to the oracle. HotpotQA shows the same pattern (one gold paragraph present: 0.255 → 0.420).

**Table 3.** Cost per question (means). *Detect tokens* = tokens generated before a failure is flagged (triggered questions only); *processed* = prefill + generated tokens including judge prompts and restarts; latency on the sequential live subset (n=100).

| System | Detect tokens (2Wiki / Hotpot) | Processed tokens (2Wiki / Hotpot) | Generated tokens (2Wiki / Hotpot) | Latency mean / median (ms) |
|---|---|---|---|---|
| No intervention | – | 909 / 1071 | 142 / 134 | 7124 / 4654 |
| Always retrieve | 0 / 0 | 1166 / 1418 | 154 / 132 | 8127 / 5314 |
| Judge trigger | 167.9 / 124.0 | 2567 / 2855 | 211 / 164 | 10411 / 8724 |
| **Probe trigger** | **54.2 / 53.0** | **1535 / 1887** | **165 / 166** | **7844 / 6167** |

The probe flags failures **3.1× earlier** on 2Wiki (2.3× on HotpotQA), before the failing trajectory is completed, and processes **40% / 34% fewer tokens** than the judge pipeline. Its mean latency is **25% lower** (median 29% lower) and only 10% above no intervention, even though it restarts 61% of questions. Scoring a hop costs **9.5 ms** for the probe, against **738 ms** per trajectory for the judge's two forward passes; both costs exclude retrieval (21.5 ms).

**Ablation: dedicated vs. one all-hops probe.** A single probe trained on hop-1 and hop-2 states together (same layer and calibration) yields 0.464 (2Wiki) and 0.556 (HotpotQA), vs. 0.472 and 0.576 for the dedicated probes. The direction favours dedicated probes, consistent with the weak cross-hop transfer observed in Phase 2 (hop 1 and hop 2 are represented differently), but the margin is small and not tested for significance.

**Final-answer failure prediction.** Using the maximum hop P(fail) as a score for "the final answer will be wrong", the probe reaches AUROC 0.824 on 2Wiki (judge 0.686), but only 0.575 on HotpotQA (judge 0.571). Out of domain, the probe is a useful repair trigger even though it is a weak predictor of final-answer failure.

---

## RQ4 — Layers

Train-split CV AUROC peaks in the middle of the network (hop 1: 0.943 at layer 13; hop 2: 0.925 at layer 14), with a broad plateau. Layer 10 is the earliest layer within 0.01 AUROC of the best for hop 1 and the all-hops probe; for hop 2 the plateau starts at layer 9. Early layers carry markedly less signal (hop 1: 0.827 at layer 0; hop 2: 0.726), and AUROC declines again towards the last layers (0.888 at layer 27). Failure becomes linearly decodable from the middle third of the network onward, early enough to act on at the hop boundary.

---

## Limitations (Phases 3–5)

- **Baseline strength.** Our Doctor-RAG-style judge is the same 3B model, zero-shot: it reuses the prompt of our standalone judge baseline, whereas the proposal specified a few-shot prompt. Doctor-RAG itself relies on a fine-tuned diagnosis model, so our baseline is a matched-compute approximation, not a reproduction. The probe is trained on our hop labels and the judge is not, so the comparison measures a trained internal signal against an off-the-shelf text signal.
- **Restart rate.** The calibrated hop-2 probe has a 14.5% false-alarm rate, and 61% of questions are restarted. On HotpotQA the probe trigger breaks 25 correct answers (judge: 10). A cost-sensitive τ is a natural extension; the threshold sweep (`sweep.png`) shows the accuracy/cost trade-off.
- **Accuracy vs. judge on 2Wiki** is not significantly different. The in-domain advantages are detection quality, earliness and cost.
- **Labels.** Hop labels are gold-entity presence checks. A hop can be counted as failing although its reasoning is sound but phrased without the entity, and vice versa.
- **Simulated evaluation.** Systems are evaluated from cached greedy generations. This was verified against the fully live system on 100 questions (100% trigger and correctness agreement), not on all 1,000.
- **Scope.** A single 3B model, a 2-hop dataset, a simulated weak first-stage retriever, and a corpus pooled from the evaluation datasets rather than full Wikipedia.
