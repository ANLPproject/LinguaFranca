# Important Discoveries & Insights (LinguaFranca) — v2, corrected

> **Oct 2026 revision.** Several numbers in the previous version of this file were wrong
> (see `FIXES_AND_RERUN.md` §2 for each one). Everything below was recomputed with the v2
> scripts on the canonical dataset (`data/canonical/augmented_v2.jsonl`, 4,078 examples:
> 4,000 natural + 78 counterfactual) and the canonical grouped 80/20 split
> (3,264 train / 814 test examples). Probing numbers: `outputs/probing/results.md`.
> Section 7 (patching) is PENDING the final v4 run (`kaggle/03`).

## 1. Data generation & labeling (Phase 1)
* **Strict XML chain-of-thought.** Llama-3.2-3B-Instruct wraps each step in `<hop1>…</hop1>`, `<hop2>…</hop2>`. This gives exact token boundaries for pooling per hop.
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

## 7. Causal patching (Phase 4) — final run (v4) PENDING
* **Earlier runs, what survives:**
  * v1 notebook: void (sampled instead of greedy; patched into prompts that already said "not mentioned").
  * v2.0 run (Oct 2026): hook self-test and all-positions sanity passed. But its "entity-span" column is **invalid**: the Llama chat template contains today's date, the clean and CF prompts were generated on different days, and v2.0 treated everything from that date token to the entity (~1,000 tokens) as "the entity". Its last-token column (≈0 % at every layer) and Experiment B (0/51 restored at blocks 8/11/14/20, 95 % CI 0–7 %) were not affected.
* **v4 (final, `kaggle/03_causal_patching.ipynb`)** makes the pairs true minimal pairs: the CF prompt's date token is set to the clean one (verified on the real tokenizer: 51/51 pairs then differ only in the entity, 2–11 tokens). It measures:
  * **A:** restoration of the hop-1 bridge entity when patching the entity tokens, the tail tokens or the last token, by block, plus a donor-entity control and a sanity check;
  * **B:** the `</hop1>` transplant;
  * **C:** whether the hop-1 probe's verdict flips when the patch restores the entity. This links Phase 4 to the probe; a self-check verifies that recomputed Phase-1 states match the stored ones.
* Fill in from `results/patching/summary.md`.
* Data note: in 7 of the 78 CF records, `hops[0].text` comes from a different decoding than `generated_cot` (both are "not mentioned" hedges; labels identical). The hidden states and all Phase-4 inputs use `generated_cot`, so nothing is affected.

## 8. Phase 5 (next) — which probe to use
Use `outputs/probing/probes/hop1_probe_layer10.joblib` (a dict: `["model"]` is the sklearn pipeline). Its input is the mean of the hop-1 token states at block 10, matching `generate_cot.extract_hidden_states`. Pick the alarm threshold on validation data for the false-alarm budget you want: at 0.5 it catches ~75 % of hop-1 failures with ~3 % false alarms.

## 9. Positioning vs Doctor-RAG
Doctor-RAG reports 81.3 % diagnosis accuracy with a distilled, fine-tuned diagnoser on its own benchmark. That figure is not directly comparable to ours (different data, labels and task granularity). The defensible claim is the matched comparison in §6 once it is run: same rows, same labels, hidden-state probe vs off-the-shelf text judges, plus cost (one dot product vs a ~1k-token LLM call per hop).
