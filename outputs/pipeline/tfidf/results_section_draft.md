# Draft — Experimental setup and results for Phases 3–5

*Source numbers: `outputs/pipeline/tfidf_v2_r2_shared/results.md` (main), `outputs/pipeline/tfidf_v2_r2/` (figure `sweep.png`, `tables.tex`, ablations) and `outputs/pipeline/tfidf/results.md` (original Naive Restart, repair v1). Probe metadata: `outputs/pipeline/tfidf/probes/probes_meta.json`. Data and retriever details: `outputs/pipeline/README.md`.*

---

## Experimental setup

**Retrieval condition.** We simulate a weak first-stage retriever. 2WikiMultihopQA provides 10 candidate paragraphs per question (2 gold, 8 distractors); TF-IDF ranks them against the question and the model sees the top 5. On the 500 test questions both gold paragraphs survive in only 41%, one in 58%, neither in 2%. Many reasoning failures are therefore caused by missing evidence, which is what a retrieval-based repair can fix. HotpotQA (distractor setting, 500 questions) is processed the same way for out-of-domain evaluation.

**Model and probe.** Llama-3.2-3B-Instruct (fp16, greedy decoding) writes `<hop1>…</hop1><hop2>…</hop2><answer>…</answer>`. A single linear probe (standardized logistic regression, 3,072 weights), as specified in the proposal, reads the mean-pooled layer-10 residual stream over the tokens of the hop that just closed. The same probe and the same threshold are applied at `</hop1>` and `</hop2>`. It is trained on hop-1 and hop-2 states of the 80% train split of 4,000 questions (grouped split, seed 42). Its scores are Platt-calibrated on out-of-fold predictions from 5-fold grouped cross-validation inside the train split, and the threshold τ = 0.639 maximises F0.5 on those predictions. Neither the layer nor τ is tuned on test data. Held-out test AUROC: 0.931.

**Repair (Naive Restart).** When a trigger fires at hop *k*, generation stops. The question plus the hop-*k* text query a dense retriever: all-MiniLM-L6-v2 embeddings in an exact FAISS inner-product index over 94,031 Wikipedia paragraphs, pooled from the 2Wiki and HotpotQA questions and 20,000 further 2Wiki questions; it contains every gold paragraph the first stage dropped, among 90k+ others. For a hop-2 failure the query is led by the hop-1 bridge entity. The 20 best candidates are re-ranked by a cross-encoder (ms-marco-MiniLM-L-6-v2), and the top 3 not already in the prompt are prepended to its context. The KV cache is flushed and the answer regenerated from scratch. With *up to 2 restarts*, a restart that is flagged again gets one more retrieval and restart.

**Systems.** All share the model, prompt and repair; they differ only in when the repair is triggered.
- *No intervention*: one greedy trajectory.
- *Probe trigger (ours)*: generation halts at the first hop with calibrated P(fail) ≥ τ.
- *Judge trigger (Doctor-RAG-style)*: after the full trajectory, the same 3B model judges hop 1 and hop 2 zero-shot (P(no) vs. P(yes) logits); repair starts from the earliest hop judged wrong. With 2 restarts, the judge re-checks the complete restarted answer.
- *Always retrieve*: the repair retrieval with the question as query, before generating.
- *Oracle trigger*: repair at the gold first-failure hop (2Wiki), an upper bound for any trigger using this repair.

**Evaluation.** Hop labels come from Phase 1 gold-entity matching (24 of 500 regenerated trajectories differed from the stored ones and were relabeled with the same matcher). An answer is correct if its token F1 with the gold answer is ≥ 0.5; EM and F1 are also reported. CIs are 95% paired bootstrap intervals over questions. Every trajectory is generated once with the probe in monitor mode, and all restarts are pre-computed. Because decoding is greedy, a halted trajectory is an exact prefix of the uninterrupted one, so all systems and thresholds are evaluated from these cached generations. On 100 questions run fully live (halt → retrieve → restart, up to 2 times), the cached evaluation agreed on 100% of triggers, restart counts and correctness decisions (95% identical answer strings). The live probe features matched the training features with mean cosine 0.99999.

---

## RQ1 — Detection and localization

**Table 1.** Failure detection on 2Wiki held-out questions (live trajectories). Failure is the positive class; probes use their calibrated τ, the judge its own yes/no decision.

| Rows | Trigger | AUROC | Precision | Recall | F0.5 | FPR |
|---|---|---|---|---|---|---|
| Hop 1 (n=499, 85 fail) | **Shared probe** | **0.918** | 0.772 | **0.718** | 0.761 | 0.043 |
| | Dedicated hop-1 probe | 0.927 | 0.847 | 0.588 | 0.779 | 0.022 |
| | Judge | 0.569 | 0.198 | 0.306 | 0.213 | 0.254 |
| Hop 2 (n=492, 327 fail) | **Shared probe** | **0.902** | **0.940** | 0.807 | **0.910** | **0.103** |
| | Dedicated hop-2 probe | 0.910 | 0.918 | 0.817 | 0.895 | 0.145 |
| | Judge | 0.678 | 0.796 | 0.382 | 0.654 | 0.194 |
| Trajectory (n=500, 351 failing) | **Shared probe** | – | **0.937** | 0.801 | **0.906** | **0.128** |
| | Dedicated probes | – | 0.928 | 0.809 | 0.902 | 0.148 |
| | Judge | – | 0.795 | 0.519 | 0.718 | 0.315 |

The internal signal detects failing hops far more reliably than the text-based judge, which is near chance on hop 1. The single shared probe localizes the earliest failing hop correctly in **72.4%** of failing trajectories, against **24.8%** for the judge. Including correctly predicting "no failure", it identifies the exact first-failure hop in 76.8% of trajectories (judge 37.8%). Hop-specific probes are marginally better per hop (AUROC +0.008–0.009). At the trajectory level the shared probe is slightly better (F0.5 0.906 vs. 0.902).

---

## RQ3 — End-to-end accuracy, cost and latency

**Table 2.** Answer accuracy (F1 ≥ 0.5); Δ vs. no intervention with 95% CI; *fixed / broken* = answers turned right / wrong relative to no intervention.

| System | 2Wiki acc | Δ [95% CI] | fixed / broken | HotpotQA acc | Δ [95% CI] | fixed / broken |
|---|---|---|---|---|---|---|
| No intervention | 0.334 | – | – | 0.488 | – | – |
| Always retrieve | 0.344 | +0.010 [−0.020, +0.038] | 29 / 24 | 0.520 | +0.032 [−0.012, +0.072] | 70 / 54 |
| Judge trigger | 0.440 | +0.106 [+0.074, +0.138] | 62 / 9 | 0.510 | +0.022 [−0.002, +0.046] | 25 / 14 |
| Judge trigger, ≤ 2 restarts | 0.456 | +0.122 [+0.090, +0.154] | 68 / 7 | 0.518 | +0.030 [+0.006, +0.054] | 27 / 12 |
| **Probe trigger (ours)** | **0.496** | **+0.162 [+0.124, +0.200]** | 92 / 11 | **0.538** | **+0.050 [+0.014, +0.084]** | 57 / 32 |
| **Probe trigger, ≤ 2 restarts** | **0.552** | **+0.218 [+0.178, +0.258]** | 117 / 8 | 0.530 | +0.042 [+0.006, +0.078] | 54 / 33 |
| Oracle trigger (1 restart) | 0.540 | +0.206 [+0.166, +0.244] | 113 / 10 | – | – | – |

On 2Wiki, probe-triggered repair raises accuracy by **+16.2 points** with one restart and **+21.8** with up to two, where it exceeds the one-restart oracle (0.540). It **significantly outperforms the judge trigger**: +5.6 points (paired CI [+2.2, +9.2]) with one restart, +9.6 ([+5.6, +13.4]) with two. On HotpotQA, without re-tuning τ, the probe trigger improves accuracy by +5.0 points (CI [+1.4, +8.4]); the judge's +2.2 is not significant. The probe–judge difference there (+2.8) is not significant either (CI [−0.8, +6.4]), and a second restart does not help out of domain. Retrieval without a trigger fixes and breaks answers at similar rates on both datasets: *when* to retrieve matters as much as *what* is retrieved.

**Where the gains come from** (2Wiki, by first-stage condition):

| First-stage condition | n | No intervention | Judge | Probe | Probe, ≤ 2 restarts | Oracle |
|---|---|---|---|---|---|---|
| Both gold paragraphs retrieved | 203 | 0.670 | 0.680 | 0.690 | 0.719 | 0.714 |
| Only the hop-1 paragraph | 123 | 0.146 | 0.301 | **0.415** | **0.488** | 0.463 |
| Only the hop-2 paragraph | 165 | 0.079 | 0.273 | **0.345** | **0.394** | 0.412 |
| Neither | 9 | 0.000 | 0.000 | 0.000 | 0.556 | 0.000 |

With the evidence present all systems are similar. When a gold paragraph is missing, probe-triggered repair multiplies accuracy by 2.8–4.4× (one restart). A second restart is the only system that recovers questions with both paragraphs missing (5 of 9). HotpotQA shows the same pattern (one gold paragraph present: 0.255 → 0.387).

**Table 3.** Cost per question (means, one restart unless noted). *Detect tokens* = tokens generated before a failure is flagged (triggered questions); *processed* = prefill + generated tokens incl. judge prompts and restarts.

| System | Detect tokens (2Wiki / Hotpot) | Processed tokens (2Wiki / Hotpot) | Generated tokens (2Wiki / Hotpot) |
|---|---|---|---|
| No intervention | – | 909 / 1071 | 142 / 134 |
| Always retrieve | 0 / 0 | 1207 / 1470 | 146 / 135 |
| Judge trigger | 167.9 / 124.0 | 2573 / 2859 | 212 / 164 |
| **Probe trigger** | **52.0 / 48.6** | **1526 / 1838** | **157 / 158** |
| Judge trigger, ≤ 2 restarts | 167.9 / 124.0 | 3617 / 3455 | 248 / 173 |
| **Probe trigger, ≤ 2 restarts** | **52.0 / 48.6** | **2007 / 2459** | **171 / 177** |

The probe flags failures **3.2× earlier** on 2Wiki (2.6× on HotpotQA), before the failing trajectory is completed. It processes **41% / 36% fewer tokens** than the judge pipeline, and 45% / 29% fewer with two restarts. Scoring a hop costs **9.5 ms** for the probe, against **738 ms** per trajectory for the judge's two forward passes. Retrieval with re-ranking costs about 128 ms. On a sequential live subset (n = 100), mean latency is 9.9 s for the probe trigger with up to two restarts and 10.1 s for the judge trigger with one; no intervention takes 7.1 s.

**Figure (`tfidf_v2_r2/sweep.png`).** Sweeping the trigger threshold traces accuracy against mean generated tokens. On both datasets the probe-trigger curve lies above the judge-trigger curve at every compute budget.

**Ablations** (accuracy, 2Wiki / HotpotQA):

| Variant | Probe trigger | Judge trigger |
|---|---|---|
| Original Naive Restart (repair v1: question + hop text, dense top-3), shared probe | 0.464 / 0.556 | 0.452 / 0.516 |
| Repair v1, dedicated hop-1 / hop-2 probes | 0.472 / 0.576 | (same) |
| **Repair v2 (bridge-entity query + re-ranking), shared probe** | **0.496 / 0.538** | 0.440 / 0.510 |
| Repair v2, dedicated probes | 0.510 / 0.558 | (same) |
| Repair v2, ≤ 2 restarts, shared probe | 0.552 / 0.530 | 0.456 / 0.518 |
| Repair v2, ≤ 2 restarts, dedicated probes | 0.564 / 0.554 | (same) |
| Repair v2, accuracy-optimal τ (2-fold CV on 2Wiki), shared probe | 0.532 / 0.562, at 92% / 97% restart rate | – |

- **Dedicated vs. shared probe.** Hop-specific probes are 1.2–2.4 points more accurate end to end, consistent with the weak cross-hop transfer in Phase 2 (hop-1 and hop-2 failures are represented differently). The shared probe is the proposal's design and applies unchanged to chains of any length.
- **Repair.** The improved repair adds 3–4 points in domain but loses ~2 points on HotpotQA.
- **Threshold.** Choosing τ for accuracy rather than F0.5 restarts nearly every question at hop 2, for at most +3.6 points; the F0.5 threshold is the better accuracy/cost trade-off.

---

## RQ4 — Layers

Train-split CV AUROC peaks in the middle of the network (hop 1: 0.943 at layer 13; hop 2: 0.925 at layer 14; shared probe: 0.940 at layer 14), with a broad plateau. Layer 10 is within 0.01 AUROC of the best for every probe. Early layers carry markedly less signal (hop 1: 0.827 at layer 0; hop 2: 0.726), and AUROC declines again towards the last layers (0.888 at layer 27). Failure becomes linearly decodable from the middle third of the network onward, early enough to act on at the hop boundary.

---

## Limitations (Phases 3–5)

- **Baseline strength.** Our Doctor-RAG-style judge is the same 3B model, zero-shot: it reuses the prompt of our standalone judge baseline, whereas the proposal specified a few-shot prompt. Doctor-RAG itself relies on a fine-tuned diagnosis model, so our baseline is a matched-compute approximation, not a reproduction. The probe is trained on our hop labels and the judge is not.
- **Design after inspection.** The improved repair (v2) and the second restart were designed after inspecting results of the original repair on the same 2Wiki test questions. HotpotQA, where they bring no gain, is the unbiased check. The original Naive Restart results are reported alongside.
- **Out of domain** the probe's advantage over the judge is not significant (one restart: CI [−0.8, +6.4]). The probe breaks more correct answers there (32 vs. 14), and final-answer failure prediction transfers poorly (AUROC 0.595).
- **Restart rate.** The calibrated probe restarts 60% of questions; a cost-sensitive τ is a natural extension.
- **Labels** are gold-entity presence checks. A hop can be counted as failing although its reasoning is sound but phrased without the entity, and vice versa.
- **Simulated evaluation.** Systems are evaluated from cached greedy generations, verified against the fully live system on 100 questions (100% trigger, restart-count and correctness agreement), not on all 1,000.
- **Scope.** A single 3B model; 2-hop questions only (2WikiMultihopQA compositional, HotpotQA), extending to longer chains (e.g. MuSiQue) is future work; a simulated first-stage retriever; a 94k-paragraph corpus pooled from the evaluation datasets rather than full Wikipedia.
