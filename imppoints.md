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

## 7. Causal patching (Phase 4) — TF-IDF setting, final (branch `cf-tfidf`, Oct 29)
Pairs: the TF-IDF test-split counterfactuals (`cf_v1`), hedge + other_wrong_entity kinds; the first 219 in a fixed order, **211 informative** (clean run states the gold hop-1 entity, CF run does not). Clean and CF prompts differ only in the entity. Probe = the **tuned shared probe** (LR C = 0.01, block 12; the same model as `main:outputs/pipeline/tfidf/probes_lr_C0.01_L12`, weight cosine 1.0000); the old probe (C = 1, block 10) is reported alongside. Results: `outputs/patching/` — `patching_tfidf_results` (nb 05), `patching_tuned_results` (nb 08), `causal_tracing_tfidf` (nb 09), `probe_tracing_tfidf` (nb 10, + `analysis.md`), `extra_analysis.md`.

**Threshold note.** The pipeline (Phases 3–5 on `main`) uses τ = 0.650 for this probe; the cf-tfidf bundle got τ = 0.585 from the same model (different CV folds; F0.5-optimal τ is flat and fold-sensitive). Flag rates below are given at **τ = 0.650** so the paper uses one threshold; AUROCs and correlations do not depend on τ.

**A. Generation-based patching (nb 05): copy clean states into the CF run at block L; does hop 1 name the gold entity?** (n = 211)

| block | entity span | tail | last prompt token | third-entity control | entity − control (sign test) |
|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.10 | +0.90 (p 1e-57) |
| 8 | 0.90 | 0.35 | 0.00 | 0.15 | +0.74 |
| 10 | 0.73 | 0.40 | 0.00 | 0.14 | +0.59 |
| 11 | 0.46 | 0.46 | 0.01 | 0.16 | +0.30 |
| 12 | 0.45 | 0.50 | 0.01 | 0.14 | +0.31 |
| 13 | 0.38 | 0.48 | 0.03 | 0.14 | +0.24 (p 8e-13) |
| 14 | 0.19 | 0.15 | 0.07 | 0.12 | +0.07 (p 0.006) |
| 24 | 0.08 | 0.04 | 0.02 | 0.06 | +0.02 (n.s.) |

* The entity information sits in the entity tokens through block ~8, moves into the following tokens over blocks 9–13, and neither alone is sufficient from block 14. Entity ∪ tail at blocks 11–13 restores only 0.71–0.77, so part of it is already in later positions. Reproducible: 0 / 3,066 outcomes differ between two runs.
* The third-entity control is 6–16 % (lower than full10's 18–61 %); entity-specific restoration is large through block 13 and small after.
* **Experiment B (single state at `</hop1>`): null**, ≤ 1 % at blocks 8–20. ⚠️ **The mid-report §6 claim that patching the `</hop1>` state restores the answer came from the void v1 notebook and must be corrected.**

**B. Logit-based causal tracing (nb 09; the mid-term PPT's slides 12–14 method), n = 183 pairs with a ≥ 1-nat gap.** Recovery = fraction of the clean-vs-CF log-prob gap of the clean hop 1 (teacher-forced, entity-name tokens not scored). Sanity (all positions, block 0) = 1.000.

| condition | TF-IDF | mid-term (full10, 93 pairs) |
|---|---|---|
| entity last token | peak **0.63 at block 7**; 0.05 at 14 | peak 0.48 at L7 |
| last prompt token | ~0 until 11; **0.29 at 14**, peak 0.30 at 15 | jumps to 0.39 at L14 |
| MLP at entity (±2) | 0.38–0.40 at 0–2, 0.30–0.36 at 7–11, **0 from 16** | peak 0.25 at L10–11 |
| entity span | 1.00 to block 7, 0.82 at 10, 0.48 at 13, **0.08 at 14** | – |
| tail | 0 → **0.66 at 11–12** → 0.35 at 14 | – |

* **The block 13 → 14 handoff (slide 13) replicates, sharper:** entity last 0.31 → 0.05, entity span 0.48 → 0.08, last token 0.13 → 0.29 in one step.
* Difference from the mid-term: the MLP window has no single peak at 10–11; MLPs at the entity contribute over blocks 0–13 and nothing after 14. Phrase it that way.
* The last prompt token reaches 0.30 in log-prob but ≤ 7 % in generation (A): information arrives there but is not sufficient to change the output.

**C. Does the probe follow the intervention? (nb 08, generated hop 1, tuned probe)** CF baseline flagged 0.69 at τ = 0.650 (0.75 at 0.585; old probe 0.48), clean 0.005; AUROC clean vs CF 0.990 (old 0.935). Agreement with "entity restored?" for patches below the probe's block: 0.95 (blocks 0–9: 0.98 vs old 0.93). The jump in false flags exactly at the probe's own block (old 10, new 12) is a reading artefact (the patch overwrites the state the probe reads).

**D. Probe ↔ causal tracing (nb 10 + `analysis.md`), n = 192.** The clean run's hop-1 text is teacher-forced in every run, so **the words are identical; only the question entity differs**.

| | old (C = 1, block 10) | tuned (C = 0.01, block 12) |
|---|---|---|
| Test A: AUROC clean vs CF (same words) | 0.911 | **0.960** (0.91–0.96 over 5 retrain bootstraps) |
| Test A: flagged clean → CF | 0.021 → 0.380 (τ 0.639) | 0.010 → 0.354 (τ 0.650; 0.469 at 0.585) |
| Test B: within-pair r, entity last token [95 % CI] | +0.28 [+0.19, +0.36] | **+0.59 [+0.54, +0.64]** (0.55–0.62 over retrains) |
| specificity: probe drop entity span − third-entity control | +2.19 log-odds, 99 % of pairs | **+2.80** [+2.67, +2.94], 99 % of pairs |

* **The probe reads grounding, not wording:** same tokens, and the score rises in 99 % of pairs. Hidden states also carry the words (block 0 gives hop-1 AUROC 0.83, §2), so this test is what separates the two; notebook 08's higher flag rate includes hedge wording.
* **Alignment:** pair by pair, how much behaviour is restored predicts how much the probe score is restored. Test B r is computed with pair and block means removed and pair-bootstrap CIs; the pooled Spearman in `summary.md` (p ~ 1e-294) is invalid (~2,300 rows from ~190 pairs) and must not be quoted. Entity-span correlations are near zero at blocks 0–7 only because behaviour is at ceiling (SD 0.02); at blocks 9–11 they are +0.42 to +0.51.
* **The tuned probe also tracks the third-entity control (r = +0.37 [+0.26, +0.48]; old ≈ 0):** it follows how well the context supports the hop text under any intervention, not only the true entity. Specificity is the row above.
* **Robustness (retrains at blocks 10/12, C 0.001–1, 5 bootstrap training samples):** natural AUROC is 0.92–0.94 at both blocks, but Test B r is 0.51–0.62 at block 12 vs 0.20–0.46 at block 10, and C = 0.01 ≥ C = 1 in all 10 comparisons. **Natural accuracy cannot tell the layers apart; causal alignment can** — independent support for block 12.
* **Lag:** the probe restores later than behaviour (patch at block 10: behaviour 0.78, probe 0.38) because it reads block 12; the old probe shows the same lag 2 blocks before its layer.
* **Grounding direction** d = mean(F_cf − F_clean): separates held-out CF pairs perfectly (AUROC 1.00) but natural failures only weakly (0.57 at block 10, 0.72 at block 12). The natural-trained probe transfers to CFs (0.96) but not the reverse: CFs are one narrow failure type, which supports **not** training on them. cos(d, probe weights) ≈ 0.05 is not evidence of different mechanisms (LR weights are covariance-adjusted); do not report it as such.

**Limits:** one model; CF pairs are an induced, mostly hedging failure type; confident `copied_seed_entity` failures are caught at ≤ 22 % by every probe (nb 06/07); patches restore the input-side entity representation, the probed state alone (B) is null.

## 7b. Earlier causal patching study — full10 setting (v4, Kaggle, Llama-3.2-3B, 51 token-aligned pairs)
Superseded by §7 (more pairs, TF-IDF setting, no train leak). Note: the old probe's 51/51 CF detection here was inflated: the CFs were in its training data and it used a 0.5 threshold.
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

## 8. Phases 3–5 — final run with the tuned probe (TF-IDF setting, Oct 30)
Run: `pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12` (Havish, on `main`: `outputs/pipeline/tfidf_final_lrC0.01_L12/results.md`). Extra analyses on branch `eval-robustness`, all on the **same cached generations** (only the trigger differs): `robustness/robustness.md`, `selective/`, `error_budget/`, `uncertainty_early/` under `outputs/pipeline/tfidf_final_lrC0.01_L12/`. Scripts: `scripts/eval_robustness.py`, `selective_and_cost.py`, `error_budget_cases.py`, `uncertainty_early.py`.

**Setting.** As in §8b below, but the probe is the tuned one (LR C = 0.01, block 12, shared for both hops, τ = 0.650 = F0.5-optimal on train-split CV). τ was 0.585 on `cf-tfidf` with the same weights: only the CV folds differ (sklearn 1.6 vs 1.8). This run also **forces an answer** from hops 1–2 when generation hits the token cap without `<answer>` (105 first answers, 441 restarts). That moved the baseline (0.334 → 0.342 on 2Wiki), so **never compare this run with §8b across runs**; use A2 below. 500 2Wiki + 500 HotpotQA questions; correct = token F1 ≥ 0.5; CIs are paired bootstraps over questions.

**Headline.**

| system | 2Wiki | HotpotQA (OOD) | tokens / question (2Wiki) |
|---|---|---|---|
| No intervention | 0.342 | 0.504 | 911 |
| Always retrieve | 0.356 (n.s.) | 0.546 (n.s.) | 1209 |
| Judge trigger (same 3B, post hoc) | 0.454 | 0.526 (n.s.) | 2576 |
| Best of 4 judges (Qwen2.5-7B) | 0.508 | 0.546 | 2785 |
| **Probe trigger** | **0.538** (+0.196 [+0.156, +0.236]) | **0.570** (+0.066 [+0.034, +0.098]) | **1574** |
| **Probe, ≤ 2 restarts** | **0.594** (+0.252) — best of all systems | 0.564 | 2049 |
| Always restart after hop 2 | 0.554 | 0.586 | 1968 |
| Oracle trigger (gold labels, upper bound) | 0.550 | – | 1619 |

* Probe − judge: 2Wiki [+0.050, +0.118]; HotpotQA [+0.010, +0.078]. Probe ≈ always-restart-after-hop-2 in accuracy (−0.016, n.s.) with 34 % fewer restarts (0.64 vs 0.98) and half the broken answers (10 vs 20).
* The probe gets **94 % of the oracle's gain** on 2Wiki ((0.538 − 0.342) / (0.550 − 0.342)): a better detector can add ≤ ~1 point with this repair.
* Detection on live 2Wiki hops (AUROC): probe 0.947 / 0.925 (hop 1 / 2) vs judges 0.56–0.78 / 0.56–0.73. First-failure localisation 74.4 % vs 24.8 %. Flags after 54 vs 170 generated tokens; 9.5 ms vs 740 ms per check.

**A. Robustness** (`robustness/robustness.md`, run by Havish with our script)
1. **Fair thresholds — the honest version of "probe beats judge".** Give every judge the same 2-fold accuracy-optimal threshold the probe gets and they all collapse to restarting 93–99 % of questions and tie the probe on accuracy (2Wiki: probe 0.554, judges 0.518–0.558). So **do not claim higher maximum accuracy than the judges**; the maximum is set by always-restart. Claim instead: the probe reaches it with far fewer restarts, and at any fixed budget it chooses better (item 3).
2. **Old vs tuned probe on the same generations** (the clean comparison): +0.032 [+0.016, +0.050] on 2Wiki at train-CV τ; hop AUROC 0.947 / 0.925 vs 0.918 / 0.902. HotpotQA +0.010 (n.s.).
3. **Matched restart budget:** on 2Wiki at 40–70 % restarts the probe beats every judge, all CIs > 0 (e.g. 60 %: 0.524 vs 0.456–0.504). HotpotQA: mostly ties.
4. **τ does not matter:** probe > judge at every τ from 0.30 to 0.90 on 2Wiki (+0.046 to +0.094, all CIs > 0).
5. **Forcing does not matter:** every conclusion holds when forced answers count as wrong.
6. **Calibration:** ECE 0.024 (probe) vs 0.22–0.48 (judges); Brier 0.080 vs 0.25–0.49.
7. **HotpotQA by type explains the weak OOD result:** bridge questions (n = 404) probe − none +0.079 [+0.042, +0.119], probe − judge +0.052 [+0.015, +0.089]; comparison questions (n = 96) ≈ 0 (the probe was trained on 2Wiki bridge chains).
8. **The probe also verifies restarts:** on 2Wiki, restarts it still flags are 26 % accurate vs 72 % for restarts it clears.

**B. The hidden state knows what the output probabilities don't** (`uncertainty_early/`; one teacher-forced pass, Kaggle T4). The standard active-RAG triggers (FLARE: a low-probability token; DRAGIN-style: entropy) computed on the same hops, at the same moment, at the same cost.
* Hop-failure AUROC on the probe's test split (1586 hops): **probe 0.954** vs mean entropy 0.677, max entropy 0.642, mean NLL 0.628, max NLL (FLARE) 0.592; an LR on all five 0.731. Probe + all five = 0.953: confidence adds nothing. Probe vs NLL Spearman ρ = +0.25.
* **Confident failures:** on the 958 hops where FLARE stays silent, the probe still has AUROC 0.958 and flags 259 / 334 of the failures FLARE misses.
* End-to-end (2Wiki): at deployable thresholds the probe beats every confidence trigger by +0.08 (CIs ≈ [+0.05, +0.12]); confidence triggers land at the judges' level (≈ 0.455). Matched budget 40–90 %: +0.03 to +0.09, CIs > 0. Accuracy-optimal: they collapse towards always-restart (as the judges do). HotpotQA: ties.
* Parity with the run: |ΔP(fail)| mean 0.0007 over 1977 hops; sanity: test AUROC 0.9542 = the bundle's.
* **Within the hop:** the probe on the running mean of the first x % of the hop: 0.72 at 20 %, 0.79 at 50 %, 0.95 at the end; confidence signals never exceed 0.68. The signal builds up as the entity is written (consistent with §7). **No mid-hop trigger:** the probe is fitted on full hops and every hop opens with the template "From the context, I find that", so short-prefix scores are off its scale (97 % above τ after 3 tokens). Prefix AUROCs are valid (rank-based), thresholds are not.

**C. Selective answering** (`selective/`). Answer only the questions the detector is most confident about. AURC = mean error over all coverages, lower = better.
* First answers, 2Wiki: **probe AURC 0.418** vs judges 0.528–0.615 (probe − best judge −0.109 [−0.151, −0.069]); random 0.658, perfect ranking 0.292, so the probe closes 66 % of the gap and the best judge 36 %. Answering the most confident 50 %: 0.552 vs ≤ 0.472.
* The whole system (restart, then abstain using the score of the final trajectory), 2Wiki: probe pipeline AURC 0.272, accuracy 0.748 on the half it answers; judge pipeline 0.447 / 0.544.
* HotpotQA: ties everywhere (the probe's final-answer AUROC there is only 0.595).

**D. Where the remaining errors come from** (`error_budget/`, 2Wiki; each wrong answer assigned to the first stage that broke; share of all 500 questions)

| stage | probe | judge | oracle |
|---|---|---|---|
| detector missed the failing hop | 8.6 % | 25.4 % | 0.4 % |
| detector flagged the wrong hop | 4.8 % | 12.6 % | 0 % |
| no hop failed, answer still wrong | 5.6 % | 6.8 % | 7.4 % |
| restarted, but the evidence was already there | 4.2 % | 1.2 % | 7.2 % |
| restarted, retrieval missed the gold paragraph | 16.2 % | 6.6 % | 21.8 % |
| gold paragraph retrieved, answer still wrong | 6.8 % | 2.0 % | 8.2 % |
| **all errors** | **46.2 %** | **54.6 %** | **45.0 %** |

* Detection costs the probe 13.4 % of questions and the judge 38 %. **The bottleneck is now retrieval:** even a perfect detector loses 21.8 % to retrieval. When the restart retrieves the missing gold paragraph accuracy is 72 %, otherwise 12–13 %. → future work: better repair retrieval, not better detection.

**E. Case study for the paper** (`error_budget/error_budget.md`, 12 examples in `cases.jsonl`). *"When was the performer of song Doesn't Really Matter born?"* Hop 1 "the performer is Janet Jackson" (probe 0.04, judge 0.00). Hop 2 "Janet Jackson is an American singer" — fluent, confident and dodging the question: **probe 0.90, judge 0.07**. First answer "American". The probe restarts at hop 2, retrieval brings the *Janet Jackson* page, final answer "May 16, 1966" ✓. 21 such cases (probe flags, judge does not, restart fixes it); most are this nationality-instead-of-date pattern. Counter-examples to show honestly: restarts that break a correct answer (10; the model had recovered from a failing hop on its own and the restart retrieved the wrong page) and probe misses (43; some trajectories write no hop text at all).

**F. Caveats.** One model; the judges are zero-shot, not Doctor-RAG's fine-tuned diagnoser; 24 of 500 2Wiki trajectories differ from Phase 1 and were relabelled with a presence check (detection numbers are unchanged on Phase-1-only rows); HotpotQA has no hop labels (no oracle, no error budget); the repair was designed after seeing early 2Wiki results (HotpotQA is the clean check).

## 8b. Earlier Phases 3–5 run — deployed probe C = 1, block 10 (superseded by §8)
Numbers: `outputs/pipeline/tfidf_v2_r2_shared/results.md`; paper-ready text: `outputs/pipeline/tfidf/results_section_draft.md`; data sources, retrieval corpus and retriever: `outputs/pipeline/README.md`; code: `src/pipeline/`. No answer forcing, so its accuracies are not comparable with §8.

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
2. Used as a live repair trigger with the same retrieval and restart, a single probe is significantly more accurate than the post-hoc judge at its own decision rule and at every matched restart budget in domain, at least as accurate out of domain, and reaches the accuracy of always-restarting with 34 % fewer restarts, while detecting failures ~3× earlier at ~40 % lower token cost (§8). With accuracy-tuned thresholds the judges match that accuracy only by restarting almost everything (§8 A1).
3. The probe sees failures the model's own token probabilities do not (AUROC 0.95 vs ≤ 0.68 for FLARE/DRAGIN-style signals; §8 B), so it is also a better trigger than confidence-based active RAG.

## 10. Report checklist (from Anish, `experiment/distractor-rag`) — where each item stands
1. Data-imbalance numbers — §1 (class balance) and the TF-IDF setting's hop-1 / hop-2 failure rates in §8.
2. Counterfactuals: how and why — §1 (78 CFs, entity-not-in-context stratum).
3. MLP vs logistic regression vs SVM — not in the v2 numbers; the old comparison script is in `legacy/generate_report_stats.py` (old labels — rerun before using).
4. Doctor-RAG with a fine-tuned diagnoser (~80 %) — not reproduced; see §9 for how to position it.
5. Inference savings vs Doctor-RAG — §8 (3.2× earlier detection, 39 % fewer tokens; accuracy-vs-cost figure `selective/selective_and_cost.png`).
6. "Internal doubt" on the timeline — §8: the probe flags at ~54 generated tokens vs ~170 for the judge; within a hop the signal builds up as the entity is written (§8 B).
7. Probe accuracy on hedging vs confident failures — §3.
