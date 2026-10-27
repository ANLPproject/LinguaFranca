# Important Discoveries & Insights (LinguaFranca) — v3

> **Oct 2026 revision.** Sections 1–7 are the corrected numbers from the audit on branch `fixes-v2`
> (`FIXES_AND_RERUN.md` §2 lists every error in the earlier version of this file, e.g. the invalid
> "98.2 % confident-wrong" and "91.4 % all-hop" figures). They were recomputed on the canonical dataset
> (4,078 examples: 4,000 natural + 78 counterfactual) with the canonical grouped 80/20 split
> (3,264 train / 814 test). Files referenced in §2–7 (`outputs/probing/`, `outputs/llm_judge/`,
> `outputs/patching/`, `kaggle/0*.ipynb`) live on `fixes-v2` until it is merged.
> Section 8 (Phases 3–5) is new: `src/pipeline/`, results in `outputs/pipeline/tfidf/`.

## 1. Data generation & labeling (Phase 1)
* **Strict XML chain-of-thought.** Llama-3.2-3B-Instruct wraps each step in `<hop1>…</hop1>`, `<hop2>…</hop2>`. This gives exact token boundaries for pooling per hop.
* **Context = all 10 passages (2 gold + 8 distractors), not gold-only.** The gold-title filter in `generate_cot._normalise` silently matched nothing on the HF columnar format, so every Phase 1 prompt contains all 10 paragraphs (verified: 4,000 / 4,000). Sections 1–7 therefore describe this *10-passage* setting; the Phase 3–5 experiments (§8) use the TF-IDF top-5 setting from branch `experiment/distractor-rag`.
* **Every question is strictly 2-hop** (all 4,000 gold reasoning graphs have exactly 2 edges) — but the **model** often writes more steps: 1,329 / 4,000 examples have 3–18 `<hopN>` blocks.
* **Label = "does the hop text state that hop's gold bridging entity?"** (string → Wikidata alias → SBERT → LLM-judge fallback). Hops 3+ have no gold entity and are now **unlabeled (-1)**. The old labeler called them all failures; they were 64 % of all failure labels and made the hop index alone predictive (77.9 % accuracy).
* **Label semantics caveat.** "Failure" means *the hop did not state the bridge entity*, which includes detours, not only hallucinations: 38 % of natural hop-1 failures and 48 % of natural hop-2 failures still reach a correct final answer (the model states the fact in a later step).
* **Class balance.** Hop 1: 3,714 success / 364 failure (8.9 %); hop 2: 2,381 / 1,697 (41.6 %).
* **Counterfactuals (78)** swap the question's entry entity for another title of the same token length. Because the context is unchanged, ~90 % of them make the model say "X is not mentioned in the context" (hedging). They are an *entity-not-in-context* stratum, reported separately. **Natural hop-1 failures contain no hedging at all** — they are confident wrong or off-target statements.

## 2. Probing results (Phases 2–3)
Probe: StandardScaler + logistic regression (balanced class weights) on the mean-pooled hidden state of the hop's tokens. Layer L = output of transformer block L (layer 0 is *after* the first block, not the raw embedding). Layers chosen by 5-fold grouped CV **inside the train split**; numbers are on the held-out test split with 95 % grouped-bootstrap CIs.

| task (test split) | layer | acc | bal. acc | AUROC [95 % CI] | failure recall (TPR) | false-alarm rate (FPR) | majority-class acc |
|---|---|---|---|---|---|---|---|
| Hop 1 | 10 | 0.951 | 0.860 | **0.953** [0.918, 0.980] | 0.750 | 0.031 | 0.916 |
| Hop 1, natural only | 10 | 0.950 | 0.827 | 0.942 [0.895, 0.974] | 0.685 | 0.031 | 0.932 |
| Hop 2 | 13 | 0.899 | 0.896 | 0.949 [0.933, 0.963] | 0.880 | 0.087 | 0.592 |
| Hops 1+2 pooled | 13 | 0.909 | 0.887 | **0.956** [0.943, 0.966] | 0.845 | 0.070 | 0.754 |
| Hops 1+2, natural only | 13 | 0.907 | 0.882 | 0.952 [0.939, 0.964] | 0.834 | 0.070 | 0.767 |

**Baselines that never see hidden states (same test rows):**

| task | TF-IDF on hop text (AUROC / bal. acc) | hop index only | hedge-keyword rule (TPR / FPR) |
|---|---|---|---|
| Hop 1 | 0.890 / 0.783 | — | 0.191 / 0.003 |
| Hop 2 | 0.723 / 0.674 | — | 0.033 / 0.002 |
| Hops 1+2 | 0.770 / 0.692 | 0.719 / 0.719 | 0.060 / 0.002 |

* **Layer profile.** Signal rises from block 0 (hop-1 AUROC 0.83) to a broad plateau over **blocks 9–14** (AUROC 0.95–0.96) and declines slowly to 0.92 at block 27. Layers 10, 11 and 13 are statistically indistinguishable — there is no single "sweet spot" layer.
* **Controls.** Shuffled-label probe: AUROC 0.47–0.51 (chance). Cross-hop transfer (train hop 1 → test hop 2): AUROC 0.66–0.69 at the middle layers, so hop-1 and hop-2 failure directions differ substantially.
* **Text vs hidden state.** On hop 1 a bag-of-words on the hop text already reaches AUROC 0.89, and block 0 reaches 0.83, so much of the hop-1 signal is lexical. The middle layers add a real margin (0.95–0.96), especially on hop 2 (0.95 vs 0.72 for text).
* **Split stability** (`bootstrap_confident_wrong.py`, 5 seeds): hop-1 AUROC 0.961 ± 0.006 (layer 10), 0.960 ± 0.005 (layer 11).

## 3. Hedging vs confident failures (hop 1)
Recall on hop-1 failures at the default threshold, **always with the false-alarm rate on hop-1 successes**:

| layer | hedging failures (all are CFs) | confident failures | false-alarm rate |
|---|---|---|---|
| 0 | 0.98 | 0.44 | 0.038 |
| 5 | 1.00 | 0.60 | 0.025 |
| 10 | 1.00 | **0.76** (CV) / 0.69 (test, n = 55) | 0.022 / 0.031 |
| 13 | 1.00 | 0.74 | 0.021 |
| 20 | 1.00 | 0.62 | 0.026 |
| 27 | 1.00 | 0.53 | 0.034 |

(CV = out-of-fold on the train split, n = 240 confident and 56 hedging failures; test n = 55 / 13.)

* Hedging ("there is no mention of X") is caught at every layer — it is lexically obvious, and in this dataset all hedging failures are counterfactuals.
* Confident failures are the real test: recall climbs from 44 % (block 0) to **~75 % at blocks 10–13** at a ~2 % false-alarm rate, then falls off in late layers.
* ⚠️ The previous table ("100 % hedging / 98.2 % confident-wrong at layer 11") came from probes that were trained on **hop 2** and scored on hop 1. They flagged 89 % of hop-1 *successes* too. It must not be used.

## 4. Qualitative error analysis (hop 1, layer 10, held-out test)
See `outputs/probing/qualitative_hop1.md`. The probe's most confident natural failures fall into two types:
* **Off-target / non-progress steps:** the model restates a fact about the film instead of naming the composer or director. Example: "Daddy-O was released in 1958." (asked for its composer).
* **Confident wrong entity:** "the composer of Astral City is Wagner de Assis" (he is the director); "the composer of Grizzly Man is Werner Herzog" (director, not composer).

## 5. Internal failures → external failures
The hop-1 probe score predicts a wrong *final* answer only weakly (test AUROC 0.59; the gold hop-1 label itself: 0.56), because many hop-1 "failures" are detours the model recovers from. Hop-level detection works well. The link from hop failure to final-answer failure is weak in this setup and should be stated as such.

## 6. LLM-as-judge baseline (Phase 3) — measured on identical rows
Source: `kaggle/02_llm_judge.ipynb` (Oct 2026). 1,617 held-out rows (hops 1–2 of the canonical test split; 11 hops with empty text dropped); the probe is scored on exactly the same rows. The judge sees the context, the question and the hop text and is asked whether the step is correct and grounded; P(failure) = P("no") vs P("yes") from one forward pass (no sampling). Llama models ran through ungated mirrors of the same weights (`unsloth/...`).

| detector | AUROC | bal. acc | TPR (failure recall) | FPR | hop-1 AUROC |
|---|---|---|---|---|---|
| **Probe** (hop-specific, layers 10/13) | **0.958** | **0.900** | 0.853 | 0.053 | **0.955** |
| Qwen2.5-7B judge | 0.718 | 0.656 | 0.627 | 0.314 | 0.709 |
| Llama-3.2-3B judge | 0.606 | 0.546 | 0.175 | 0.083 | 0.674 |
| Llama-3.1-8B judge | 0.604 | 0.531 | 0.129 | 0.067 | 0.597 |
| Qwen2.5-3B judge | 0.597 | 0.552 | 0.794 | 0.690 | 0.613 |

* The probe is far ahead of every judge (AUROC 0.96 vs ≤ 0.72). Judge size does not fix it: the 8B Llama is no better than the 3B.
* Two judges are miscalibrated rather than informative: Qwen-3B answers "no" for 72 % of rows (its 79 % "recall" comes with a 69 % false-alarm rate), and both Llamas answer "yes" for ~90 % of rows (recall 13–18 %). AUROC is the fair, threshold-free comparison; even at the best threshold their balanced accuracy is 0.57–0.66.
* Failure recall on confident failures (n = 365): probe 0.85, Qwen-7B 0.66, Llama-3B 0.18, Llama-8B 0.13. Hedging failures (n = 24, all counterfactuals): probe 0.96, Llama judges 0.04 — they correctly call "X is not mentioned" a grounded statement, which our label calls a failure, so the CF rows (n = 24) are noisy and should not be over-read.
* Caveat on interpretation: the judge is never told the gold bridge entity (neither is the probe). The label is "did the hop name the gold bridge entity", which a grounding judge can only partly infer. This is the fair "text-only diagnoser" comparison the proposal asks for, but it is a weaker test than a fine-tuned diagnoser such as Doctor-RAG's.
* The earlier Llama-3B-judge figures (61.7 % accuracy, 18.4 % confident-failure recall) turn out to be consistent with this re-run (0.184 recall); what was wrong before was the probe side and the comparison setup, not the judge itself.

## 7. Causal patching (Phase 4) — final run (v4, Kaggle, Llama-3.2-3B, 51 token-aligned pairs)
Source: `kaggle/03_causal_patching.ipynb`, results in `outputs/patching/`. All checks passed (hook self-test, all-positions sanity = 51/51 restored, probe-link self-check cosine = 1.0, 0 errors). The clean and counterfactual (CF) prompts differ only in the swapped entity (median 5 tokens, 2–11); the CF run never states the right entity (0/51) and hedges ("not mentioned") in 86 % of pairs. Earlier patching results (v1 notebook, v2.0 run) are void.

**Experiment A — copy the clean run's internal state into the CF run at block L; does hop 1 now name the correct bridge entity?** (n = 51; success = gold entity appears in the generated hop 1)

| block | entity tokens | tail (tokens after the entity) | last prompt token only | donor-entity control |
|---|---|---|---|---|
| 0 | 1.00 | 0.02 | 0.00 | 0.18 |
| 4 | 1.00 | 0.16 | 0.00 | 0.61 |
| 8 | 1.00 | 0.45 | 0.00 | 0.43 |
| 10 | 0.90 | 0.63 | 0.00 | 0.43 |
| 11 | 0.76 | 0.73 | 0.00 | 0.39 |
| 13 | 0.69 | 0.65 | 0.04 | 0.37 |
| 14 | 0.39 | 0.12 | 0.04 | 0.29 |
| 20 | 0.35 | 0.06 | 0.04 | 0.27 |
| 24 | 0.08 | 0.04 | 0.00 | 0.06 |

* **Where the entity information lives:** through block ~8 it sits entirely in the entity tokens (100 % restored). Over blocks ~8–13 it is read out into the tokens that follow (tail restoration rises 0.45 → 0.73 while entity-token restoration falls 1.00 → 0.69). By block 14 neither alone is sufficient (0.39 / 0.12), and the single last prompt token never is (≤ 4 %). At blocks 9–13, patching the entity tokens or the tail restores 84–98 % of pairs. This transition window (blocks ~9–14) coincides with the blocks where the hop-1 probe is strongest (§2), which is suggestive but not proof.
* **The donor control is NOT zero.** Overwriting the entity tokens with a *different* entity's states restores the gold entity in 18–61 % of runs. Most likely, once the question's entity is perturbed to something the model does not recognise it falls back to the only entity in the context (hypothesis, not tested). So the honest effect of "restoring the clean entity" is the paired difference to the donor: significantly positive at blocks 0–13 (exact sign test, every p ≤ 0.0005; +0.27 to +0.82), **but not significant at blocks 14, 16, 20, 24** (+0.02 to +0.10, p ≥ 0.06). Do not claim entity-specific restoration beyond block 13.
* **Experiment B — a single state at `</hop1>`:** restores 0/51 at blocks 8, 11, 14, 20 (95 % CI 0–7 %; baseline correct 1/51). Fixing the one hop-boundary vector does not fix the outcome.
* **Experiment C — does the hop-1 probe follow the intervention?** The probe flags 0/51 clean runs and 51/51 CF runs. When an entity patch restores the entity it flags ≤ 5 % of those runs (0 % at blocks 0–10); when it does not it flags 77–100 %. Overall its verdict matches "entity restored?" in 691/714 patched runs (96.8 %). Caveat: the probe reads the states of the generated hop-1 *text*, so a large part of this agreement is expected; it shows the probe tracks the outcome rather than superficial properties of the prompt (the prompt is the same CF prompt in every run).
* **Limits (state these in the paper):** one model; 51 pairs, all from the counterfactual "entity not in context" type; success is a substring match on the generated hop; the patches restore the *input-side* representation of the entity, not the probed hop-boundary state itself (the latter is Experiment B, which is null). So Phase 4 shows *where the information the hop depends on is processed and that the probe's verdict follows it*; it does not show that the probed hop-1 state alone causes the failure.

## 8. Phases 3–5 — probe-triggered Naive Restart vs. Doctor-RAG-style judge (TF-IDF setting)
Final numbers: `outputs/pipeline/tfidf_v2_r2_shared/results.md`; paper-ready text: `outputs/pipeline/tfidf/results_section_draft.md`; data sources, retrieval corpus and retriever: `outputs/pipeline/README.md`; code: `src/pipeline/`.

**Setting.** A weak first-stage retriever: TF-IDF keeps the top 5 of each question's 10 paragraphs, so both gold paragraphs survive in only 41 % of test questions (306 gold paragraphs missing out of 1,000). **One shared probe** (logistic regression, layer 10, Platt-calibrated, τ = 0.639 = F0.5-optimal on train-split CV; the proposal's design) watches generation at `</hop1>` and `</hop2>`. If P(fail) ≥ τ, generation stops. The question + failing hop text (hop 2: led by the hop-1 bridge entity) retrieve from a **94,031-paragraph corpus** (MiniLM + FAISS, cross-encoder re-ranking of the top 20). The top 3 new paragraphs are prepended and the model restarts, with up to 2 restarts.

**Detection on live 2Wiki trajectories:**

| | shared probe | dedicated hop-1/hop-2 probes | judge (same 3B model, post hoc) |
|---|---|---|---|
| Hop-1 AUROC | **0.918** | 0.927 | 0.569 |
| Hop-2 AUROC | **0.902** | 0.910 | 0.678 |
| Trajectory F0.5 | **0.906** | 0.902 | 0.718 |
| First-failure localization | **72.4 %** | 74.4 % | 24.8 % |

**End-to-end accuracy (F1 ≥ 0.5), Δ vs no intervention [95 % CI]:**

| system | 2Wiki | HotpotQA (OOD) |
|---|---|---|
| No intervention | 0.334 | 0.488 |
| Always retrieve (no trigger) | 0.344 (+0.010, n.s.) | 0.520 (+0.032, n.s.) |
| Judge trigger | 0.440 (+0.106) | 0.510 (+0.022, n.s.) |
| **Probe trigger** | **0.496 (+0.162 [+0.124, +0.200])** | **0.538 (+0.050 [+0.014, +0.084])** |
| **Probe trigger, ≤ 2 restarts** | **0.552 (+0.218 [+0.178, +0.258])** | 0.530 (+0.042) |
| Judge trigger, ≤ 2 restarts | 0.456 | 0.518 |
| Oracle trigger (1 restart) | 0.540 | – |

* **Probe vs judge:** significant on 2Wiki (+5.6, CI [+2.2, +9.2]; ≤ 2 restarts +9.6, CI [+5.6, +13.4]); not significant on HotpotQA (+2.8, CI [−0.8, +6.4]).
* **Gains come from missing evidence:** only hop-1 paragraph present 0.146 → 0.415 (0.488 with 2 restarts); only hop-2 paragraph 0.079 → 0.345 (0.394); both present ~unchanged. Blind always-retrieve fixes and breaks answers at similar rates.
* **Cost:** flags failures after 52 vs 168 generated tokens (3.2× earlier), 41 % fewer processed tokens, one probe check ≈ 9.5 ms vs ≈ 740 ms for the judge; live latency 9.9 s (probe, ≤ 2 restarts) vs 10.1 s (judge, 1 restart).
* **Ablations:** dedicated hop-1/hop-2 probes +1.2–2.4 pts. The original Naive Restart (repair v1) scores 0.464 / 0.556: the improved repair helps in domain (+3.2) but not OOD (−1.8). An accuracy-optimal τ restarts 92–97 % of questions for ≤ +3.6 pts, so F0.5 τ is the better trade-off.
* **Validity checks:** live features = training features (cosine 0.99999); the cached evaluation matches the fully live system on 100 questions (100 % same trigger, restart count and correctness).
* **Caveats:** the judge is the untuned, zero-shot 3B model, not Doctor-RAG's fine-tuned diagnoser; the repair improvements were designed after seeing the first 2Wiki results (HotpotQA is the clean check); 60 % of questions are restarted; on HotpotQA the probe breaks more correct answers than the judge (32 vs 14).

## 9. Positioning vs Doctor-RAG
Doctor-RAG reports 81.3 % diagnosis accuracy with a distilled, fine-tuned diagnoser on its own benchmark; that figure is not directly comparable to ours (different data, labels and granularity). The defensible claims are the matched comparisons:
1. On identical rows, hidden-state probes detect hop failures far better than off-the-shelf text judges up to 8B (§6: AUROC 0.96 vs ≤ 0.72).
2. Used as a live repair trigger with the same retrieval and restart, a single probe is significantly more accurate than the post-hoc judge in domain, and at least as accurate out of domain, while detecting failures ~3× earlier at ~40 % lower token cost (§8).

## 10. Report checklist (from Anish, `experiment/distractor-rag`) — where each item stands
1. Data-imbalance numbers — §1 (class balance) and the TF-IDF setting's hop-1 / hop-2 failure rates in §8.
2. Counterfactuals: how and why — §1 (78 CFs, entity-not-in-context stratum).
3. MLP vs logistic regression vs SVM — not in the v2 numbers; the old comparison script is in `legacy/generate_report_stats.py` (old labels — rerun before using).
4. Doctor-RAG with a fine-tuned diagnoser (~80 %) — not reproduced; see §9 for how to position it.
5. Inference savings vs Doctor-RAG — §8 (3.1× earlier detection, 40 % fewer tokens, 25 % lower latency).
6. "Internal doubt" on the timeline — §8: the probe flags at ~54 generated tokens vs ~168 for the judge.
7. Probe accuracy on hedging vs confident failures — §3.
