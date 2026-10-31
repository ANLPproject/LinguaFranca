# LinguaFranca — Final Report Source Document

**Purpose.** Everything the final paper should contain: setting, method, every result with its exact numbers, how to phrase each claim, what must not be claimed, limitations, and where every number comes from. Each result names its source file so it can be checked. All numbers are final (Oct 31, 2026, branch `main`). `imppoints.md` holds the same material in lab-notebook form; this document is organised as the paper.

**Conventions.** "Correct" = token F1 ≥ 0.5 with the gold answer. CIs are 95 %: paired bootstrap over questions (end-to-end) or bootstrap over question groups (detection). n.s. = CI includes 0. "Block L" = output of transformer block L (HF `hidden_states[L+1]`). 2Wiki = 2WikiMultihopQA.

---

## 0. Proposed title and abstract (draft)

**Title (options).** "Reading Hop Failures from Hidden States: Probe-Triggered Restarts for Multi-Hop RAG" · "Your Model Knows When a Hop Fails: Linear Probes as Cheap Repair Triggers for Multi-Hop RAG".

**Abstract (draft).** Multi-hop retrieval-augmented generation fails when an intermediate reasoning step ("hop") does not resolve the entity the next step needs, often because the evidence for it was never retrieved. We ask whether the generator's own hidden states reveal such failures, and whether that signal can drive repair. On 2WikiMultihopQA with Llama-3.2-3B-Instruct writing explicit hops over a weak TF-IDF context, a linear probe on the mean hidden state of a hop detects failing hops with AUROC 0.954, far above zero-shot LLM judges up to 8B (≤ 0.78) and the token-confidence signals used by active-retrieval methods such as FLARE and DRAGIN (≤ 0.68), and on par with a 184M-parameter text diagnoser fine-tuned on the same labels (0.959, n.s.) — while costing one dot product on states the generator already computes. Causal patching shows the probe reads whether the hop is grounded in the context rather than its wording, and follows interventions pair by pair (within-pair r = 0.59). The signal is present after the first ~3 tokens of a hop, while the text is still identical. Used as a live trigger for retrieve-and-restart, the probe raises accuracy from 34.2 % to 53.8 % (59.4 % with two restarts), matches restarting every question with a third fewer restarts, beats a Doctor-RAG-style judge at matched restart budgets of 40–70 %, and flags failures 3× earlier at ~40 % lower token cost. Error analysis shows the remaining errors come from retrieval in domain and from probe transfer out of domain (HotpotQA).

---

## 1. Introduction — points to make

1. Multi-hop RAG fails at intermediate steps; a wrong bridge entity silently derails the rest of the chain. Repairing requires knowing **whether** and **where** a step failed.
2. Existing triggers: (a) post-hoc LLM judges / diagnosers (Doctor-RAG-style) — need the full answer and a second model; (b) token-confidence triggers (FLARE: low token probability; DRAGIN: entropy) — cheap but blind to confident errors.
3. Question: does the generator's internal state already encode hop failure, early and cheaply enough to trigger repair *during* generation?
4. **Contributions** (each backed below):
   - C1. Hop-level failure detection from hidden states: AUROC 0.954; ≫ zero-shot judges and token confidence; = a fine-tuned text diagnoser at ~zero extra inference cost (§5.1).
   - C2. Mechanistic evidence: the probe reads grounding, not wording, and tracks causal interventions pair by pair; causal alignment selects the probe layer when natural accuracy cannot (§5.2).
   - C3. The signal is anticipatory: present after ~3 identical tokens of a hop, and the end of hop 1 predicts a hop-2 failure (§5.3).
   - C4. A probe-triggered restart pipeline with a careful evaluation: fair thresholds, matched budgets, cost, selective answering, out-of-domain transfer, error budget (§5.4–5.6).
   - C5. Honest diagnosis of what limits the system: retrieval in domain, probe transfer out of domain (§5.6).

---

## 2. Related work — what to cite and how we differ

- **Active / adaptive retrieval:** FLARE (Jiang et al., 2023), DRAGIN (Su et al., 2024) trigger retrieval from token probabilities / entropy. We test exactly these signals at the same hop boundary: AUROC ≤ 0.68 vs 0.954 (§5.1).
- **Post-hoc diagnosis and repair:** Doctor-RAG (fine-tuned diagnoser; reports 81.3 % diagnosis accuracy on its own benchmark — *not comparable* to ours: different data, labels and granularity). We reproduce the *approach* on our data: zero-shot judges (4 models) and a fine-tuned DeBERTa diagnoser on our labels (§5.1).
- **Truthfulness / error probes on hidden states:** e.g. SAPLMA (Azaria & Mitchell, 2023) and later probing work: statement-level. Ours is hop-level, inside generation, and used as a live trigger with end-to-end evaluation.
- **Causal tracing / activation patching:** Meng et al. (2022, ROME). We use generation-based patching and logit causal tracing, and test whether the *probe* follows the intervention (§5.2).
- **Naive Restart RAG:** the repair is the proposal's retrieve-and-restart; our contribution is the trigger and its evaluation.

---

## 3. Task, data and labels

### 3.1 Model and output format
- Generator: **Llama-3.2-3B-Instruct**, greedy decoding, ≤ 400 new tokens.
- Strict XML chain of thought: `<hop1>…</hop1> <hop2>…</hop2> … <answer>…</answer>`; gives exact token spans per hop.

### 3.2 Setting: TF-IDF top-5 context (main)
- 2Wiki questions are strictly 2-hop (all gold graphs have 2 edges). Context = the top 5 of each question's 10 paragraphs by TF-IDF, so evidence is often missing — the realistic case a restart can fix.
- **4,000 questions**, grouped 80/20 split: **3,200 train / 800 test** (6,325 / 1,586 labelled hop rows). Data pinned to an HF revision (`src/pipeline/common.py`).
- **Failure rates:** hop 1 19.6 % (782), hop 2 65.5 % (2,620); test 17.6 % / 65.5 %. Hedges ("no information about X") are 10.8 % of failing test hops; the rest are confident wrong or off-target statements.
- **What TF-IDF drops (true hop order):** both gold paragraphs present 1,594; only the hop-1 paragraph 2,269; only the hop-2 paragraph 39; neither 92 (6 unresolved). TF-IDF finds the question's entity page and misses the bridge entity's page.
- ⚠️ The stored `only_p1/only_p2` labels are alphabetical, not hop order; all figures use `hop_order_condition`. Earlier alphabetical figures are retracted.
- Source: `outputs/pipeline/tfidf_final_lrC0.01_L12/full_stats.md`, `imppoints.md` §1b.

### 3.3 Labels
- A hop **fails** if its text does not state that hop's gold bridge entity (string → normalised → SBERT ≥ 0.70 → LLM fallback, the LLM checking every failure). Hops beyond 2 are unlabelled.
- Semantics caveat (state it): "failure" = did not state the bridge entity — includes detours, not only hallucinations.

### 3.4 Counterfactuals (causal analysis only)
- Swap the question's entity for another title (context unchanged); generated for the TF-IDF data (`cf_v1`). Kinds: mostly hedge, also other-wrong-entity and copied-seed-entity.
- **Never used for training** (natural failures are plentiful; §5.2 shows CF-derived directions do not transfer to natural failures). Used for patching and as an evaluation stratum.

### 3.5 Evaluation sets for the pipeline
- **2Wiki:** 500 questions sampled from the test split.
- **HotpotQA (out of domain):** 500 validation questions (distractor setting), same TF-IDF top-5 context; 404 bridge, 96 comparison.

---

## 4. Method

### 4.1 The probe
- Feature: mean of the hop's token states at **block 12**. Model: StandardScaler + logistic regression, **C = 0.01**, balanced class weights. **One shared probe for hop 1 and hop 2.**
- Platt-calibrated on out-of-fold scores; threshold **τ = 0.650** = F0.5-optimal on grouped 5-fold CV **inside the train split** (the test split never chooses anything).
- Layer/C choice: train-split CV AUROC is a flat plateau over blocks 11–15 (0.952–0.954); block 12 within 0.002 of the best. C = 0.01 vs the earlier C = 1 / block 10 probe: test AUROC 0.954 vs 0.931. Independent support from causal alignment (§5.2).
- τ note (footnote): the same weights got τ = 0.585 under sklearn 1.6.1 (different CV folds); all pipeline results use 0.650, and conclusions hold for any τ in 0.30–0.90 (§5.4.4).
- Source: `outputs/pipeline/tfidf/probes_lr_C0.01_L12/{probes_meta.json, layer_check.md}`.

### 4.2 Probe-triggered restart pipeline
1. Generate; at each `</hopK>` score the hop (≈ 9.5 ms incl. pooling).
2. If P(fail) ≥ τ: stop. Repair query (v2): hop 1 → question + hop-1 text; hop 2 → hop-1 bridge entity + question + hop-2 text.
3. Retrieve from a **94,031-paragraph corpus** (all 10 paragraphs of every dataset question + 20k random 2Wiki questions + HotpotQA paragraphs): all-MiniLM-L6-v2 dense top 20 → cross-encoder (ms-marco-MiniLM-L-6-v2) re-rank → top 3 not already in the context.
4. Prepend them to the context and **restart** generation; up to 2 restarts (the second triggered by the probe on the restart).
5. If a generation hits the token cap without `<answer>`, the answer is forced from hops 1–2 (105 first answers, 441 restarts) — applied to every system equally.
- Validity: live features = training features (cosine 0.99999); the cached evaluation equals a fully live run on 100 questions (same trigger 100 %, same answer 98 %).

### 4.3 Baselines
| baseline | what it is | role |
|---|---|---|
| No intervention | first answer | floor |
| Always retrieve | retrieve with the question before generating | blind retrieval |
| Always restart after hop 1 / hop 2 | restart every question | accuracy ceiling of the repair without a trigger |
| Random trigger | probe's per-hop restart rate, 200 seeds | is selection better than chance? |
| Hedge-keyword trigger | restart if the hop hedges | trivial text rule |
| Text trigger | TF-IDF + LR on hop text, same labels | trained bag-of-words |
| **Zero-shot judges** (Doctor-RAG-style, post hoc) | Llama-3.2-3B (the generator itself), Llama-3.1-8B, Qwen2.5-3B, Qwen2.5-7B; sees context + question + hop, P(fail) = P("no") | the off-the-shelf diagnoser |
| **Fine-tuned text diagnoser** | DeBERTa-v3-base cross-encoder (184M), fine-tuned on the probe's exact train rows/labels; input question + step 1 + step to check + context (512 tokens) | the fair stand-in for a trained diagnoser |
| Token confidence | max/mean token NLL (FLARE), max/mean entropy (DRAGIN-style), same hop, same moment | active-RAG signals |
| Oracle trigger | restart at the gold first failing hop | upper bound of any detector |

### 4.4 Metrics
- Detection: AUROC (threshold-free), precision / recall / F0.5 / FPR at each detector's own train-chosen threshold, calibration (ECE, Brier).
- End to end: accuracy, Δ vs no intervention, paired CIs, restart rate, fixed / broken answers, tokens processed per question, detection latency.
- Fairness checks: accuracy-optimal thresholds for every trigger (2-fold on 2Wiki; HotpotQA uses the 2Wiki fit), matched restart budgets, τ sweep.

---

## 5. Results

### 5.1 Detection: probe vs judges vs diagnoser vs token confidence

**Main table — held-out test split, 1,586 hops (Phase 1 trajectories), identical rows for every detector.**

| detector | AUROC [95 % CI] | F0.5 | precision | recall | FPR | extra cost per check |
|---|---|---|---|---|---|---|
| **Probe (block 12)** | **0.954** [0.944, 0.964] | 0.878 | 0.899 | 0.803 | 0.063 | one dot product (9.5 ms incl. pooling) |
| Fine-tuned DeBERTa-v3 diagnoser | **0.959** [0.949, 0.968] | 0.901 | 0.921 | 0.826 | 0.049 | 37.8 ms, separate 184M model, ~370 tokens |
| LR on 5 token-confidence signals | 0.731 [0.707, 0.756] | 0.631 | 0.663 | 0.527 | 0.186 | ~0 |
| mean token entropy | 0.677 | 0.612 | | | | ~0 |
| max entropy (DRAGIN-style) | 0.642 | 0.554 | | | | ~0 |
| mean token NLL | 0.628 | 0.555 | | | | ~0 |
| max token NLL (FLARE) | 0.592 | 0.501 | | | | ~0 |
| probe + all 5 confidence signals | 0.953 | 0.882 | | | | adds nothing |

- Probe − diagnoser AUROC −0.005 [−0.013, +0.004] (n.s.); hop 1 0.941 vs 0.939, hop 2 0.931 vs 0.936. Probe–diagnoser score correlation ρ = 0.85. Confident-failure recall 0.79 vs 0.81.
- Probe vs token NLL: ρ = 0.25. On the 958 hops where FLARE stays silent (the model is confident), the probe still has AUROC 0.958 and flags 259 / 334 of the failures FLARE misses.
- Sources: `uncertainty_early/uncertainty_early.md`, `outputs/pipeline/tfidf/text_diagnoser/text_diagnoser.md`.

**Zero-shot judges — live 2Wiki trajectories of the pipeline run (500 questions).**

| detector | hop-1 AUROC (n = 499, 85 fail) | hop-2 AUROC (n = 492, 327 fail) | ECE | Brier |
|---|---|---|---|---|
| **Probe** | **0.947** [0.923, 0.968] | **0.925** [0.897, 0.948] | **0.024** | **0.080** |
| Fine-tuned diagnoser | 0.933 [0.896, 0.965] | 0.931 [0.901, 0.955] | – | – |
| Judge Qwen2.5-7B | 0.779 | 0.690 | 0.241 | 0.249 |
| Judge Llama-3.1-8B | 0.597 | 0.729 | 0.264 | 0.305 |
| Judge Llama-3.2-3B (the generator) | 0.569 | 0.678 | 0.221 | 0.290 |
| Judge Qwen2.5-3B | 0.558 | 0.562 | 0.482 | 0.487 |
| token confidence (best: mean entropy) | 0.727 | 0.734 | – | – |

- Trajectory level (probe vs Llama-3.2-3B judge): F0.5 0.892 vs 0.718; first-failure localisation 74.4 % vs 24.8 %. Phase-1-only rows give the same AUROCs (relabelling does not matter).
- **In-context examples do not fix it either** (`outputs/pipeline/tfidf_final_lrC0.01_L12/fewshot_judge/fewshot_judge.md`): the Llama-3.1-8B judge with two training-split demonstrations (one failing, one correct hop-2 step) gets slightly *worse*: hop AUROC 0.547 / 0.678 vs 0.597 / 0.729 zero-shot, ECE 0.333 vs 0.264, 2Wiki accuracy 0.396 vs 0.410 at θ = 0.5; 1.2 s and ~1,040 prompt tokens per check. With accuracy-optimal thresholds it collapses to always-restart (0.98) and ties, like every judge; at matched budgets of 40–90 % the probe beats it (+0.05 to +0.09, CIs > 0). Only supervised training closes the gap (diagnoser 0.959).
- Bigger judges do not fix it: Llama-3.1-8B is no better than 3B; several judges are biased rather than informative (Qwen-3B flags 75–79 % of hops, the Llamas 13–32 %).
- Sources: `results.md`, `robustness/robustness.md` §6, §8.

**How to write it.** The failure signal is *in* the text + context (a fine-tuned diagnoser recovers it), but off-the-shelf judges and token-confidence triggers cannot read it, and a linear probe on the generator's own states gets it for free. Earlier full10 study (appendix A): probe 0.958 vs judges ≤ 0.718 on identical rows — same picture.

### 5.2 Mechanism: what the probe reads (causal patching, TF-IDF CFs)

Setup: 219 clean/CF pairs from the test split (211 informative: clean run states the gold hop-1 entity, CF run does not); prompts differ only in the question's entity. Sources: `outputs/patching/` (`patching_tfidf_results`, `patching_tuned_results`, `causal_tracing_tfidf`, `probe_tracing_tfidf/analysis.md`, `extra_analysis.md`); figure `outputs/patching/causal_figure.{png,pdf}`.

**A. Where the entity information lives (generation-based patching; copy clean states into the CF run at block L; does hop 1 name the gold entity?)**

| block | entity span | tail tokens | last prompt token | third-entity control |
|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.10 |
| 8 | 0.90 | 0.35 | 0.00 | 0.15 |
| 10 | 0.73 | 0.40 | 0.00 | 0.14 |
| 12 | 0.45 | 0.50 | 0.01 | 0.14 |
| 13 | 0.38 | 0.48 | 0.03 | 0.14 |
| 14 | 0.19 | 0.15 | 0.07 | 0.12 |
| 24 | 0.08 | 0.04 | 0.02 | 0.06 |

- Entity information sits in the entity tokens through ~block 8, moves into the following tokens over blocks 9–13, and is no longer recoverable from either alone from block 14. Entity-specific (vs control) restoration significant through block 13 (sign test p ≤ 8e-13), small after (+0.07 at 14, n.s. at 24). Fully reproducible (0 / 3,066 outcomes differ between runs).
- **Logit causal tracing** (n = 183, recovery of the clean hop-1 log-prob gap): entity last token peaks 0.63 at block 7; last prompt token reaches 0.29–0.30 at blocks 14–15; MLPs at the entity contribute over blocks 0–13 and nothing from 16; **sharp handoff at block 13 → 14** (entity span 0.48 → 0.08). Replicates the mid-term (full10) pattern.
- **Null result to report:** patching the single state at `</hop1>` restores ≤ 1 % at blocks 8–20 — the probed hop-boundary state alone is not causal. ⚠️ This corrects the mid-term report's claim (which came from a void notebook).

**B. Does the probe follow the intervention?** (hop-1 text teacher-forced from the clean run, so the **words are identical**; only the question entity differs)

| | old probe (C = 1, block 10) | **tuned probe (C = 0.01, block 12)** |
|---|---|---|
| Test A: AUROC clean vs CF, same words | 0.911 | **0.960** (0.91–0.96 over retrains) |
| Test B: within-pair r, behaviour recovery vs probe drop (entity last token) | +0.28 [+0.19, +0.36] | **+0.59** [+0.54, +0.64] (0.55–0.62 over retrains) |
| Specificity: probe drop entity patch − third-entity control | +2.19 log-odds, 99 % of pairs | **+2.80** [+2.67, +2.94], 99 % of pairs |
| Generated hop 1: CF flagged / clean flagged (τ 0.650) | 0.48 / 0.024 | 0.69 / 0.005 (AUROC 0.990) |

- **The probe reads grounding, not wording** (same tokens; the score rises in 99 % of pairs).
- **Alignment:** pair by pair, how much behaviour a patch restores predicts how much it restores the probe (r computed with pair and block means removed; the pooled Spearman in `summary.md` is invalid and must not be quoted).
- **Layer selection by causality:** retrains at blocks 10 / 12 have equal natural AUROC (0.92–0.94) but Test-B r 0.51–0.62 at block 12 vs 0.20–0.46 at block 10; C = 0.01 ≥ C = 1 in 10 / 10 comparisons. Natural accuracy cannot tell the layers apart; causal alignment can.
- The tuned probe also tracks the third-entity control (r = +0.37): it follows how well the context supports the hop under any intervention — specificity is the row above.
- **CFs as training data would not work:** a direction from CF pairs separates held-out CFs perfectly (AUROC 1.00) but natural failures only weakly (0.57–0.72), whereas the natural-trained probe transfers to CFs (0.96). Justifies training on natural failures only.
- Lag: the probe restores later than behaviour (patch at block 10: behaviour 0.78, probe 0.38) because it reads block 12.

### 5.3 The signal is anticipatory

- **Inside a hop** (probe trained on partial hops; thresholds from train OOF only): AUROC **0.834** after ~10 % of the hop (~3 tokens), 0.885 at 50 %, 0.929 at the end. **95 % of failing and 99 % of correct hops open with the identical words "From the context, I find"**, and removing hedges does not change the early signal — the state predicts the failure before the text differs. Token-confidence signals on the same prefixes stay ≤ 0.68. (`outputs/pipeline/tfidf/online_probe/online_probe.md`)
- **Across hops:** on questions whose hop 1 is correct, the state at the end of hop 1 predicts a hop-2 failure with AUROC 0.82–0.83 (blocks 8–16), vs 0.62 for the hop-1 text, 0.68 for a "bridge entity has no page in context" rule, and 0.72 for an oracle rule using gold titles. (`outputs/pipeline/anticipatory_probe.md`)
- Caveat: the fine-tuned text diagnoser also sees the context, so "anticipatory" is about the *generator's state*, not a claim that context alone carries no signal.

### 5.4 End to end: probe-triggered restarts

**5.4.1 Main table** (final run; `outputs/pipeline/tfidf_final_lrC0.01_L12/results.md`, `trigger_baselines.md`, `text_diagnoser.md`)

| system | 2Wiki acc | Δ vs none [CI] | restarts | tokens / q | HotpotQA acc | Δ vs none [CI] | restarts | tokens / q |
|---|---|---|---|---|---|---|---|---|
| No intervention | 0.342 | – | 0 | 911 | 0.504 | – | 0 | 1072 |
| Always retrieve | 0.356 | n.s. | 1.00 | 1209 | 0.546 | n.s. | 1.00 | 1471 |
| Hedge-keyword trigger | 0.364 | | 0.08 | | 0.506 | | 0.00 | |
| Random trigger (probe's rate, 200 seeds) | 0.465 ± 0.013 | | 0.63 | | 0.550 ± 0.012 | | 0.57 | |
| Always restart after hop 1 | 0.448 | | 1.00 | | 0.568 | | 1.00 | |
| Judge Llama-3.2-3B (θ = 0.5) | 0.454 | +0.112 | 0.46 | 2576 | 0.526 | +0.022 (n.s.) | 0.23 | 2860 |
| Judge Llama-3.1-8B | 0.410 | +0.068 | 0.27 | 2359 | 0.520 | n.s. | 0.16 | 2770 |
| Judge Qwen2.5-3B | 0.456 | +0.114 | 0.89 | 3068 | 0.556 | +0.052 | 0.75 | 3606 |
| Judge Qwen2.5-7B | 0.508 | +0.166 | 0.63 | 2785 | 0.546 | +0.042 | 0.49 | 3248 |
| Text trigger (TF-IDF + LR) | 0.516 | | 0.70 | | 0.576 | | 0.82 | |
| Fine-tuned diagnoser (τ = 0.948) | 0.534 | | 0.64 | (+37.8 ms/check) | 0.570 | | 0.72 | |
| **Probe trigger (ours)** | **0.538** | **+0.196** [+0.156, +0.236] | **0.64** | **1574** | **0.570** | **+0.066** [+0.034, +0.098] | **0.59** | **1831** |
| Always restart after hop 2 | 0.554 | | 0.98 | 1968 | 0.586 | | 0.97 | 2351 |
| Oracle trigger (gold labels) | 0.550 | +0.208 | 0.69 | 1619 | – | | | |
| Judge Llama-3.2-3B, ≤ 2 restarts | 0.470 | +0.128 | 0.46 | 3620 | 0.534 | +0.030 | 0.23 | 3456 |
| **Probe, ≤ 2 restarts** | **0.594** | **+0.252** [+0.210, +0.294] | 0.64 | 2049 | 0.564 | +0.060 | 0.59 | 2368 |

- Probe − judge (Llama-3.2-3B): 2Wiki [+0.050, +0.118]; HotpotQA [+0.010, +0.078]; ≤ 2 restarts: 2Wiki [+0.088, +0.162], HotpotQA [−0.006, +0.066].
- Probe vs always-restart-after-hop-2: −0.016 (n.s.) on both datasets, with **34 % fewer restarts** (0.64 vs 0.98) and **half the broken answers** (10 vs 20 on 2Wiki).
- Probe vs fine-tuned diagnoser: +0.004 [−0.016, +0.024] (2Wiki), 0.000 (HotpotQA) — tie; on HotpotQA the diagnoser needs more restarts (0.72 vs 0.59).
- The probe gets **94 % of the oracle trigger's gain** on 2Wiki ((0.538 − 0.342) / (0.550 − 0.342)).
- Probe ≤ 2 restarts (0.594) is the most accurate system on 2Wiki.

**5.4.2 Cost and timing**
- Detection after 54 vs 170 generated tokens (probe vs judge; 3.2× earlier); 9.5 ms per probe check vs ~740 ms per judge call; processed tokens per question 1574 vs 2576 (−39 %).
- Live latency (100 questions, sequential): no intervention 7.1 s, always retrieve 8.3 s, probe ≤ 2 restarts 9.8 s, judge (1 restart) 10.2 s.

**5.4.3 Fair thresholds — the honest version of "probe beats judge"** (`robustness/robustness.md` §1)
- Give every trigger the same accuracy-optimal (θ1, θ2) (2-fold on 2Wiki): judges collapse to restarting 93–99 % of questions and tie the probe (2Wiki: probe 0.554, judges 0.518–0.558; HotpotQA: probe 0.580, judges 0.574–0.586).
- **Phrase as:** the maximum accuracy is set by restarting everything; the probe reaches it with far fewer restarts, and at any fixed budget it chooses better.

**5.4.4 Robustness** (`robustness/robustness.md`)
- **Matched restart budget** (2Wiki): at 40–70 % restarts the probe beats every judge with CIs > 0 (e.g. 60 %: 0.524 vs judges 0.456–0.504); at 10–30 % the differences are small and mostly n.s. (at 10 % Qwen-3B is ahead, −0.036); vs token-confidence triggers +0.03 to +0.09 at 40–90 %; vs the fine-tuned diagnoser: every difference n.s. HotpotQA: mostly ties.
- **τ sensitivity:** probe > judge at every τ from 0.30 to 0.90 on 2Wiki (+0.046 to +0.094, all CIs > 0); HotpotQA +0.020 to +0.054 (CI > 0 up to τ = 0.80).
- **Answer forcing does not matter:** without forced answers probe 0.528 / 0.552 vs judge 0.440 / 0.510; all conclusions hold.
- **Old vs tuned probe on the same generations:** +0.032 [+0.016, +0.050] (2Wiki).
- **Calibration:** probe ECE 0.024 / Brier 0.080 vs judges 0.22–0.48 / 0.25–0.49.
- **Dedicated per-hop probes** (ablation): 0.530 / 0.564 — the single shared probe is as good.
- **Verifier:** restarts the probe still flags are 26 % accurate vs 72 % for restarts it clears (2Wiki). As an answer selector (best of original, hop-1 restart, hop-2 restart): probe 0.550 vs judge 0.506 vs random 0.462 (oracle choice 0.626). (`verifier_selection.md`)

**5.4.5 Selective answering** (`selective/selective_and_cost.md`; AURC = mean error over coverages, lower = better)

| 2Wiki | answer the most confident 50 % | AURC | gap closed (random → perfect) |
|---|---|---|---|
| first answers, ranked by probe | **0.552** | **0.418** | 66 % |
| first answers, best judge | ≤ 0.472 | 0.528–0.615 | ≤ 36 % |
| probe pipeline (restart, then abstain) | **0.748** | **0.272** | – |
| judge pipeline | 0.544 | 0.447 | – |

Random AURC 0.658, perfect 0.292; probe − best judge −0.109 [−0.151, −0.069]. HotpotQA: ties.

### 5.5 Mid-hop stopping (measured end to end)

- Hybrid trigger: the partial-hop probe stops a hop early when very confident (precision ≥ the deployed probe's, chosen on train), otherwise the deployed probe checks at `</hopK>`. Detection: precision 0.884, recall 0.820, F0.5 0.871 (deployed 0.899 / 0.803 / 0.878); 60 % of caught failures stopped mid-hop.
- End to end, exact (re-retrieval with the partial-hop query; parity with the run 1977 / 1977): 2Wiki **0.554** vs 0.538 (+0.016 [−0.004, +0.036]), HotpotQA 0.562 vs 0.570 (n.s.). Generated tokens −7 % on 2Wiki, processed tokens unchanged; out of domain it over-fires (HotpotQA restarts 0.59 → 0.77).
- Why accuracy holds: the partial query finds a missing gold paragraph as often as the full one (58.8 % vs 56.4 %), and finding it decides the answer (0.72 vs 0.07 accuracy).
- **Claim:** mid-generation stopping is feasible without losing accuracy. **Not** a speed-up. (`midhop/midhop_e2e.md`)

### 5.6 Error analysis

**5.6.1 Where the remaining errors come from (2Wiki, share of all 500 questions)** (`error_budget/error_budget.md`)

| first stage that broke | probe | judge | oracle |
|---|---|---|---|
| detector missed the failing hop | 8.6 % | 25.4 % | 0.4 % |
| detector flagged the wrong hop | 4.8 % | 12.6 % | 0 % |
| no hop failed, answer still wrong | 5.6 % | 6.8 % | 7.4 % |
| restarted, evidence was already there | 4.2 % | 1.2 % | 7.2 % |
| restarted, retrieval missed the gold paragraph | 16.2 % | 6.6 % | 21.8 % |
| gold paragraph retrieved, answer still wrong | 6.8 % | 2.0 % | 8.2 % |
| **all errors** | **46.2 %** | **54.6 %** | **45.0 %** |

- Detection costs the probe 13.4 % of questions, the judge 38 %. **In domain, the bottleneck is retrieval**: even a perfect detector loses 21.8 % to it; accuracy is 72 % when the restart retrieves the missing paragraph vs 12–13 % when it does not.
- **Retrieval ablation confirms it** (`outputs/pipeline/tfidf_hybrid_lrC0.01_L12/dense_vs_hybrid.md`): dense + BM25 restart retrieval finds the missing hop-2 paragraph 66 % vs 53 % of the time; probe pipeline 0.538 → 0.558 (≤ 2 restarts 0.594 → 0.604; broken answers 10 → 5) on 2Wiki; no gain on HotpotQA (0.550). Report dense as main (pre-registered, better OOD), hybrid as the ablation.
- The hidden state is a good trigger but not a good query: a ridge map from the hop state to a retrieval query finds the missing paragraph far less often than the text query (hit@3 0.156 vs 0.438; blend +0.025). (`outputs/pipeline/latent_retrieval.md`)

**5.6.2 Out of domain: HotpotQA**
- By question type (`robustness.md` §10): **bridge** questions (n = 404) probe − none +0.079 [+0.042, +0.119], probe − judge +0.052 [+0.015, +0.089]; **comparison** questions (n = 96) ≈ 0 — the probe was trained on 2Wiki bridge chains.
- Automatic hop labels for bridge questions (`hotpot_hops/hotpot_hops.md`; hop 1 must name the bridge entity, hop 2 the answer; both rules validated on 2Wiki against the LLM-checked labels: 97.3 % / 97.0 % agreement). On 234 unambiguous bridge questions:
  - detection degrades: probe hop 1 **0.738** (2Wiki 0.947), still the best signal (judges 0.53–0.64, FLARE 0.52); hop 2 **0.675**, tied with judges (0.60–0.67) and entropy (0.70);
  - accuracy: none 0.479, probe 0.564, judge 0.530, **oracle 0.628** — the probe gets **57 %** of the oracle's gain (2Wiki 94 %);
  - → out of domain the repair works when aimed correctly; what degrades is the detector.

**5.6.3 Case studies** (`error_budget/error_budget.md`, `cases.jsonl`)
- **Main example (Figure):** *"When was the performer of song Doesn't Really Matter born?"* (gold: May 16, 1966). Hop 1 "the performer is Janet Jackson" (probe 0.04, judge 0.00 — correct). Hop 2 "Janet Jackson is an American singer" — fluent, confident, dodges the question: **probe 0.90, judge 0.07**. First answer "American". Restart at hop 2 retrieves the *Janet Jackson* page → "May 16, 1966" ✓. 21 such cases (probe flags, judge does not, restart fixes); most are this nationality-instead-of-date pattern.
- **Show honestly:** restarts that break a correct answer (10; the model had recovered from a failing hop on its own and the restart retrieved the wrong page); probe misses (43; some trajectories write no hop text at all).

---

## 6. Claims for the paper

**Claim (safe, measured):**
1. A linear probe on the generator's hidden states detects failing hops with AUROC 0.954 — far above zero-shot judges up to 8B and token-confidence triggers, and on par with a fine-tuned 184M text diagnoser — at essentially no extra inference cost.
2. The probe reads whether the hop is grounded in the context, not its wording, and tracks causal interventions pair by pair; causal alignment, not natural accuracy, identifies the right layer.
3. The signal appears after ~3 tokens of identical text, and the end of hop 1 predicts a hop-2 failure.
4. As a restart trigger it improves accuracy by +19.6 points on 2Wiki (+25.2 with two restarts) and +6.6 on HotpotQA, matches always-restart with a third fewer restarts and half the broken answers, gets 94 % of an oracle trigger's gain, beats every judge at matched restart budgets of 40–70 % (2Wiki), and detects failures 3× earlier at ~40 % lower token cost.
5. Remaining errors are dominated by retrieval in domain (confirmed by a retrieval ablation) and by probe transfer out of domain.

**Do NOT claim:**
- that text cannot reveal the failure, or that the probe out-detects a *trained* text model (the fine-tuned diagnoser ties it);
- higher maximum accuracy than the judges (with fair thresholds they reach the same accuracy by restarting almost everything);
- that the probed hop-boundary state alone causes the failure (Experiment B is null);
- a compute speed-up from mid-hop stopping;
- any p-value from the pooled Spearman in `probe_tracing_tfidf/summary.md`;
- comparability with Doctor-RAG's 81.3 % (different benchmark and labels).

---

## 7. Limitations (write all of these)

1. **One generator** (Llama-3.2-3B-Instruct); results may differ for other models and sizes.
2. **One in-domain dataset** (2Wiki); out of domain (HotpotQA) detection degrades (hop AUROC 0.74 / 0.68) and gains are small and limited to bridge questions.
3. **Label semantics:** "failure" = the hop did not state the gold bridge entity (includes detours). HotpotQA hop labels are automatic (validated on 2Wiki, bridge questions only).
4. **Baselines:** judges are zero-shot (plus a two-shot Llama-3.1-8B, which is no better); the fine-tuned diagnoser is one model size (DeBERTa-v3-base), one seed.
5. **Causal claims** concern the input-side entity representation and the probe's alignment with it; the probed state alone is not causal. CF pairs are an induced, mostly hedging failure type; confident copied-seed CF failures are caught ≤ 22 %.
6. **Repair design** (query v2, re-ranking) was developed after seeing early 2Wiki results; HotpotQA is the clean check. Answer forcing was added in the final run (applied to every system).
7. **Mid-hop stopping** keeps accuracy but saves no compute overall and over-fires out of domain.
8. Threshold τ depends on CV folds (0.585 vs 0.650 for the same weights); conclusions hold over τ ∈ [0.30, 0.90].

## 8. Future work
- A second generator (e.g. Qwen2.5-3B / Llama-3.1-8B) and a second in-domain dataset (MuSiQue).
- Probes trained on several datasets for out-of-domain transfer.
- Better repair retrieval (the in-domain bottleneck): hybrid retrieval already +2 points; learned queries.
- Combining the probe with a trained diagnoser (they correlate 0.85; complementary errors not yet tested).
- End-to-end mid-hop restarts that reuse the partial generation.

---

## 9. Figures and tables for the paper

**Main text**
1. **Fig. 1 — pipeline diagram** (to draw): generate → probe at `</hopK>` → retrieve → restart.
2. **Fig. 2 — causal figure** `outputs/patching/causal_figure.{png,pdf}`: (a) logit tracing with the block-14 handoff; (b) probe follows the restoration, with third-entity control; (c) pair-level alignment (r = 0.59 vs 0.28).
3. **Fig. 3 — probe vs token confidence** `outputs/pipeline/tfidf_final_lrC0.01_L12/uncertainty_early/uncertainty_early.{png,pdf}`: AUROC dot plot + signal inside the hop. Add the diagnoser point (0.959) and the judges for a single "detectors" figure.
4. **Fig. 4 — accuracy vs cost** `selective/selective_and_cost.{png,pdf}` panels (c)/(d): every system's accuracy against tokens per question.
5. **Table 1** — detection (§5.1 main table + zero-shot judges).
6. **Table 2** — end-to-end main table (§5.4.1), plus fair-threshold / matched-budget summary.
7. **Table 3** — error budget (§5.6.1).
8. **Box** — the Janet Jackson case.

**Appendix:** selective-answering curves (`selective_and_cost.png` (a)/(b)); reliability diagram (`robustness/reliability.png`); τ sweep (`sweep.png`); prefix-probe curve (`outputs/pipeline/tfidf/online_probe/online_probe.png`); `accuracy_vs_restarts.png`; full matched-budget tables; HotpotQA hop-label tables; retrieval ablation; mid-hop table; LaTeX tables `tables.tex`.

---

## Appendix A. Earlier study: full 10-paragraph context (mid-term setting)

Same model, labels and grouped split, but all 10 paragraphs in the context (Phase 1's gold filter silently failed, so this is the "evidence always present" setting). 4,078 examples (4,000 natural + 78 CF), 3,264 / 814 split. Source: `imppoints.md` §1–§6, §7b; `outputs/probing/`, `outputs/llm_judge/`.

| task (test) | layer | AUROC [95 % CI] | recall | FPR |
|---|---|---|---|---|
| hop 1 | 10 | 0.953 [0.918, 0.980] | 0.750 | 0.031 |
| hop 2 | 13 | 0.949 [0.933, 0.963] | 0.880 | 0.087 |
| hops 1+2 | 13 | 0.956 [0.943, 0.966] | 0.845 | 0.070 |

- Baselines without hidden states: TF-IDF on hop text 0.890 (hop 1) / 0.723 (hop 2); hop index alone 0.719. Shuffled labels 0.47–0.51. Layer profile: plateau over blocks 9–14.
- Confident hop-1 failures: recall ~75 % at blocks 10–13 at ~2 % false alarms (44 % at block 0).
- **Judges on identical rows** (1,617 test hops): probe 0.958 vs Qwen2.5-7B 0.718, Llama-3.2-3B 0.606, Llama-3.1-8B 0.604, Qwen2.5-3B 0.597. Confident-failure recall: probe 0.85, Qwen-7B 0.66, Llama-3B 0.18, Llama-8B 0.13.
- Hop-1 failure → wrong final answer is weak in this setting (AUROC 0.59), because the model often recovers from detours when all evidence is present — motivating the TF-IDF setting.
- Full10 patching (51 pairs) shows the same layer story (entity tokens to ~block 8, readout 9–13, nothing from 14); its 51/51 CF detection was inflated (CFs in training, threshold 0.5) — superseded by §5.2.
- ⚠️ Retracted from earlier drafts: "98.2 % confident-wrong" and "91.4 % all-hop" (`FIXES_AND_RERUN.md` §2).

## Appendix B. Reproducibility map

| result | notebook / script | output |
|---|---|---|
| probes | `python -m src.pipeline.probes --C 0.01 …` | `outputs/pipeline/tfidf/probes_lr_C0.01_L12/` |
| pipeline run (final) | `kaggle/pipeline_phase3_5.ipynb`, `src/pipeline/run.py` | `outputs/pipeline/tfidf_final_lrC0.01_L12/` |
| robustness, trivial triggers, verifier | `scripts/eval_robustness.py`, `trigger_baselines.py`, `verifier_selection.py` | same folder |
| selective answering, cost | `scripts/selective_and_cost.py` | `…/selective/` |
| error budget, cases | `scripts/error_budget_cases.py` | `…/error_budget/` |
| HotpotQA hop labels | `scripts/hotpot_hops.py` | `…/hotpot_hops/` |
| token confidence, early signal | `scripts/uncertainty_early.py`, `online_probe.py` | `…/uncertainty_early/`, `outputs/pipeline/tfidf/online_probe/` |
| mid-hop end to end | `scripts/midhop_e2e.py` | `…/midhop/` |
| fine-tuned diagnoser | `scripts/text_diagnoser.py` | `outputs/pipeline/tfidf/text_diagnoser/` |
| retrieval ablation | pipeline with `--retriever hybrid` | `outputs/pipeline/tfidf_hybrid_lrC0.01_L12/` |
| causal patching / tracing | `kaggle/04–10_*.ipynb`, `scripts/probe_tracing_analysis.py`, `make_causal_figures.py` | `outputs/patching/` |
| anticipatory probe, latent retrieval | `scripts/anticipatory_probe.py`, `latent_retrieval.py` | `outputs/pipeline/*.md` |

Not in git (large; regenerable): probe-tracing `features.npz` (94 MB, Kaggle output of notebook 10), the full run folder (331 MB, Kaggle), prefix features (~1 GB). Hidden states and data: HF dataset `AnishRacherla/LinguaFranca-Phase3` (pinned revisions in `src/pipeline/common.py`).

Compute: all runs on Kaggle T4 GPUs; probe training and all analyses on CPU in minutes.
