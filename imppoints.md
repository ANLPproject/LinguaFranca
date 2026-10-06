# Important Discoveries & Insights (LinguaFranca) — v2, corrected

> **Oct 2026 revision.** Several numbers in the previous version of this file were wrong
> (see `FIXES_AND_RERUN.md` §2 for each one). Everything below was recomputed with the v2
> scripts on the canonical dataset (`data/canonical/augmented_v2.jsonl`, 4,078 examples:
> 4,000 natural + 78 counterfactual) and the canonical grouped 80/20 split
> (3,264 train / 814 test examples). Probing numbers: `outputs/probing/results.md`.
> Sections marked **PENDING** need the Kaggle GPU notebooks (`kaggle/02`, `kaggle/03`).

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

## 6. LLM-as-judge baseline — **PENDING** (`kaggle/02_llm_judge.ipynb`)
* The previous figures (61.7 % accuracy; 5.9 % hedging and 18.4 % confident-failure recall) are **void**. The judge was scored on all generated hops, including the extra hops that were "failure" by construction, on a different data file and split than the probe, and the probe's 98.2 % comparison number was itself invalid.
* v2 scores four judges (Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B) on the **identical 1,628 test rows** as the probe, with AUROC from P(no)/P(yes). Fill in from `outputs/llm_judge/summary.md`.
* Reading guide: for counterfactual rows the hop text is usually a *true* statement ("X is not mentioned"), so a grounding judge saying "yes" is not wrong about grounding. Compare on natural rows.

## 7. Causal patching (Phase 4) — **PENDING** (`kaggle/03_causal_patching.ipynb`)
* The earlier notebook's results are void: it sampled (no `do_sample=False`, Llama samples at T = 0.6 by default) and patched into counterfactual prompts whose hop-1 text already said "not mentioned". The "0 % prompt-end" figure came from an earlier (greedy) version and was never logged.
* v2: classic causal tracing on token-aligned (clean, CF) pairs. It restores the clean residual stream at the entity tokens or at the last prompt token, layer by layer, with a donor-entity control and a hook self-test. Fill in from `outputs/patching/summary.md`.

## 8. Phase 5 (next) — which probe to use
Use `outputs/probing/probes/hop1_probe_layer10.joblib` (a dict: `["model"]` is the sklearn pipeline). Its input is the mean of the hop-1 token states at block 10, matching `generate_cot.extract_hidden_states`. Pick the alarm threshold on validation data for the false-alarm budget you want: at 0.5 it catches ~75 % of hop-1 failures with ~3 % false alarms.

## 9. Positioning vs Doctor-RAG
Doctor-RAG reports 81.3 % diagnosis accuracy with a distilled, fine-tuned diagnoser on its own benchmark. That figure is not directly comparable to ours (different data, labels and task granularity). The defensible claim is the matched comparison in §6 once it is run: same rows, same labels, hidden-state probe vs off-the-shelf text judges, plus cost (one dot product vs a ~1k-token LLM call per hop).
