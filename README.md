# LinguaFranca

> **"Do Internal Failures Predict External Ones?"**  
> Causal Hop Localization for Faithful, Efficient Retrieval-Augmented Multi-Hop QA

---

## Overview

A small LLM answering a two-step question often gets the first step wrong and then confidently builds the rest of its answer on the mistake. Text-based repair systems such as Doctor-RAG only notice after the whole answer is written, by asking a second model to read it.

We test whether a **lightweight probe on the model's hidden states** can detect a failing reasoning step **the moment that step ends**, before the rest of the answer is generated. We also test whether that signal is **causally** tied to the failure (activation patching), and whether using it to **trigger retrieval and restart** beats a text-based judge in accuracy, cost and latency.

> **Start here:** this README explains the whole pipeline. Numbers: `imppoints.md` (all phases) and `outputs/pipeline/tfidf_v2_r2_shared/results.md` (final Phase 3–5 run). Paper text for Phases 3–5: `outputs/pipeline/tfidf/results_section_draft.md`. What was fixed in Oct 2026: `FIXES_AND_RERUN.md`.

### Phases (repo numbering)
| Phase | Description | Code | Kaggle notebook |
|---|---|---|---|
| 1 | Data construction & hop labeling (CoT, hidden states, labels, counterfactuals) | `phase1_dataset/`, `utils/` | `kaggle_phase1.ipynb` (only to regenerate) |
| 1b | Canonical dataset (pinned revision + extra-hop fix) | `scripts/make_canonical_dataset.py` | — (CPU, 1 min) |
| 2–3 | Hidden-state probes, baselines, controls | `advanced_probing.py`, `evaluate_saved_probe.py`, `bootstrap_confident_wrong.py`, `lf_common.py` | `kaggle/01_probing.ipynb` (CPU) |
| 3 | LLM-as-judge text baseline (same rows as the probe) | `evaluate_llm_baseline.py` | `kaggle/02_llm_judge.ipynb` (GPU) |
| 4 | Causal validation (activation patching) | `causal_patching.py`, `lf_models.py` | `kaggle/03_causal_patching.ipynb` (GPU) |
| 2–3 (suite) | Logistic regression vs linear SVM vs MLP on all 28 layers, both settings, one protocol, reproduction check against the old run | `lf_probe_suite.py`, `tests/test_probe_suite.py` | `kaggle/04_probe_suite.ipynb` (CPU; upload `kaggle_upload/probe_suite_bundle.zip` from `scripts/make_probe_suite_bundle.py`) |
| 5 | Probe-triggered Naive-Restart RAG + Doctor-RAG-style baseline + end-to-end evaluation | `src/pipeline/` | `kaggle/pipeline_phase3_5.ipynb` (GPU) |

---

## How the whole pipeline works

### 1. Building the dataset (Phase 1)

**Source.** [2WikiMultihopQA](https://huggingface.co/datasets/framolfese/2WikiMultihopQA): questions that need two reasoning steps. Example: *"Where was the director of film Heads or Tails (1937) born?"*
- **Step 1:** who directed the film? → *Arcady Boytler* (the **bridge entity**)
- **Step 2:** where was Arcady Boytler born? → *Moscow* (the answer)

For every question the dataset provides the **gold entity of each step** (its reasoning graph) and **10 paragraphs**: 2 gold (one per step) and 8 distractors.

**Sampling.** 4,000 questions (seed 42), keeping only the true 2-step chain types (compositional / inference / bridge) and dropping "compare A and B" questions. Every question therefore has exactly **2 gold hops**.

**Which paragraphs the model sees.** Two versions of the data exist:
| Setting | Context in the prompt | Used for |
|---|---|---|
| `full10` (original Phase 1) | all 10 paragraphs. The intended gold-only filter silently failed, so gold-only was never used. | Phases 2–4 |
| **`tfidf`** (branch `experiment/distractor-rag`) | a weak search engine (TF-IDF) scores the 10 paragraphs against the question and keeps the **top 5**. | **Phases 3–5 (main)** |

In the `tfidf` setting a gold paragraph is often dropped: on the 500 test questions both survive in only **41%**, so **306 gold paragraphs are missing**. This is deliberate. It mimics real RAG, where the first search misses things, and creates the failures a repair can fix.

**Generating answers.** Llama-3.2-3B-Instruct (fp16, greedy decoding, ≤ 400 tokens) answers in a strict format:
```
<hop1>From the context, I find that the director of ... is Arcady Boytler.</hop1>
<hop2>From the context, I find that Arcady Boytler was born in Moscow.</hop2>
<answer>Moscow</answer>
```

**Recording the model's "thoughts".** The prompt plus the answer is run back through the model. At all 28 layers we take the hidden state (3,072 numbers per token) and **average it over each hop's tokens**. That gives one vector per hop per layer, stored as `<id>.pt` on Hugging Face (`AnishRacherla/LinguaFranca-Phase3`).

**Labeling each hop.** Does the hop state its gold entity? The check runs from cheapest to most expensive: normalized string match → Wikidata alias → sentence-embedding similarity ≥ 0.70 → an LLM judge as a last resort. **0 = success, 1 = failure.** When the model rambles into `<hop3>`, `<hop4>`…, those steps have no gold entity, so they are unlabeled and ignored. In the `tfidf` data, hop 1 fails ~20% of the time, hop 2 ~65%, and 35% of final answers are correct.

**Counterfactuals.** In the `full10` data, 78 questions with a swapped entry entity were generated (the entity is then not in the context). They are reported as a separate stratum; the `tfidf` data has none.

**Split.** A grouped 80/20 split (seed 42): 3,200 train / 800 test questions. A question and its counterfactual never land on both sides. There is no separate validation split, so every threshold is tuned by cross-validation inside train.

### 2. The probe (Phases 2–3)

A **probe** is a tiny classifier that reads a hop's averaged hidden-state vector and outputs **P(this hop failed)**.
- **Model:** standardize the 3,072 numbers, then logistic regression. One dot product, ≈ 10 ms.
- **Layer 10.** All 28 layers were tested: the signal is weak early, peaks around layers 10–14 and fades at the end. Layer 10 is within 0.01 AUROC of the best.
- **One shared probe** (the proposal's design): the same probe is trained on hop-1 and hop-2 examples together and used at both hop boundaries. Dedicated hop-1 / hop-2 probes are reported as an ablation (slightly better, +1.2–2.4 points).
- **Threshold τ = 0.639.** Chosen on cross-validated training predictions to maximize F0.5, which favours precision to avoid pointless restarts. Scores are Platt-calibrated, so 0.64 really means ≈ 64% chance of failure.
- **Held-out test:** AUROC 0.93.

**Causal check (Phase 4).** Activation patching shows that the information a hop depends on flows through exactly these middle layers (`imppoints.md` §7). The probe is reading something relevant to the failure, not an artefact.

### 3. The live system (Phases 3–5)

```
Question + 5 TF-IDF paragraphs
        │
        ▼
Llama generates token by token ── </hop1> closes ──► PROBE reads layer 10 (avg over hop-1 tokens)
        │                                                │
        │                              P(fail) < 0.639: continue │ P(fail) ≥ 0.639: STOP
        ▼                                                │                 │
   </hop2> closes ──► PROBE again ── same rule ──────────┘                 ▼
        │                                                     RETRIEVE + RESTART (section 4)
        ▼                                                                  │
   <answer> … </answer>                                   the restarted answer is checked again
                                                          → one more retrieve + restart (max 2)
```

The probe catches a failure **at the end of the bad hop**, after ~50 tokens, before the model writes the rest of a doomed answer.

### 4. The retriever (the repair)

**What we search: a 94,031-paragraph corpus** standing in for Wikipedia (`src/pipeline/retriever.py`, `build_corpus`):

| What's in it | Paragraphs |
|---|---|
| All paragraphs of the 4,000 2Wiki dataset questions | 13,719 |
| All paragraphs of the 500 HotpotQA test questions | 4,871 |
| Paragraphs of 20,000 *other* random 2Wiki questions (hard, same-domain distractors) | 75,441 |

Every gold paragraph the TF-IDF stage dropped is in the corpus (306/306), just as real Wikipedia contains the article a question needs. The system still has to **find** it among 94k paragraphs. It never sees labels, gold titles or answers; it only skips paragraphs already in the prompt.

**How a restart works:**
1. **Query.** Hop 1 failed: the question + hop 1's text. Hop 2 failed: **hop 1's entity first**, then the question and hop 2's text, e.g. *"Arcady Boytler. Where was the director of film X born? …"*. A failing hop 2 often just says "there is no information", so the entity tells the search what to look for.
2. **Dense search.** Every paragraph is a 384-number vector from `sentence-transformers/all-MiniLM-L6-v2`. The query is embedded the same way, and a FAISS index (exact inner product = cosine) returns the 20 closest paragraphs by meaning.
3. **Re-rank.** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads each query–paragraph pair and re-orders the 20.
4. **Keep the top 3** paragraphs not already in the prompt.
5. **Naive Restart.** Prepend them to the original context, throw away everything generated so far (including the KV cache), and regenerate the answer from scratch.
6. **Second round.** If the restarted answer is flagged too, search once more (excluding paragraphs already added) and restart a final time.

### 5. Systems compared

All systems share the **same model, prompt and repair**. Only *when* the repair fires differs.

| System | When it repairs |
|---|---|
| No intervention | never |
| **Probe trigger (ours)** | the moment the probe flags a hop |
| Judge trigger (Doctor-RAG-style) | after the **full** answer, the same 3B model is asked "is step 1 correct? step 2?" (zero-shot, yes/no logits); repair from the first "no" |
| Always retrieve | always, before answering, with the question as query |
| Oracle trigger | at the truly failing hop (uses the labels): an upper bound for any trigger |

### 6. Evaluation

- **Questions:** 500 held-out 2Wiki test questions, plus 500 HotpotQA questions the probe never saw (out of domain).
- **Metrics:**
  - detection: AUROC, precision / recall / F0.5, FPR
  - first-failure localization
  - answer accuracy (token F1 ≥ 0.5 = correct), EM, F1
  - generated and processed tokens, wall-clock latency
  - 95% paired-bootstrap confidence intervals
- **Efficient but exact.** Greedy decoding is deterministic, so a halted answer is an exact prefix of the full one. Each answer is generated once with the probe recording its scores, all restarts are pre-computed, and every system and threshold is computed from that cache.
- **Checks.** 100 questions also run **truly live** (halt → retrieve → restart): 100% agreement on triggers, restart counts and correctness. The live probe's features match the training features (cosine 0.99999).

### 7. Results (final run: shared probe, up to 2 restarts)

| | 2Wiki | HotpotQA (OOD) |
|---|---|---|
| No intervention | 33.4% | 48.8% |
| Always retrieve | 34.4% (not significant) | 52.0% (not significant) |
| Judge trigger | 44.0% | 51.0% |
| **Probe trigger** | **49.6%** | **53.8%** |
| **Probe trigger, up to 2 restarts** | **55.2%** | 53.0% |
| Oracle trigger | 54.0% | – |

- **Detection:** probe AUROC ≈ 0.91 vs 0.57–0.68 for the judge. The probe finds the first failing hop 72% of the time; the judge, 25%.
- **Probe vs judge accuracy:** significantly better on 2Wiki (+5.6 points, CI [+2.2, +9.2]; +9.6 with 2 restarts). Not significant on HotpotQA (CI [−0.8, +6.4]).
- **Cost:** flags failures **3× earlier** (52 vs 168 tokens), **~40% fewer tokens**; one probe check ≈ 10 ms vs ≈ 740 ms for the judge.
- **Where it helps:** when exactly one gold paragraph was missing (288 of 500 questions), accuracy rises from 10.8% → 37.5% with one restart (43.4% with two; judge 28.5%); when both are present it is ~unchanged (67% → 69%). Note: the earlier split into "only hop-1" / "only hop-2" used alphabetical title order, not hop order, and is retracted. In true hop order the missing paragraph is almost always the hop-2 one (see `outputs/pipeline/tfidf/diagnostics.md`); per-hop numbers come from re-running `evaluate` after `scripts/remap_eval_conditions.py`.
- **Retrieval without a trigger does not help:** it fixes and breaks answers at about the same rate.

### Why the absolute accuracy looks low

1. **The task is hard on purpose.** TF-IDF removes a gold paragraph in **59%** of questions, and if the fact isn't in the prompt, the model cannot answer from the context.

   | 2Wiki condition | share | baseline accuracy |
   |---|---|---|
   | both gold paragraphs present | 41% | **67%** |
   | a gold paragraph missing | 58% | 8–15% |

   The 3B model answers two-thirds correctly when the evidence is there. In the `full10` setting (all 10 paragraphs) it gets **69%** of final answers right.
2. **The model is small.** Llama-3.2-3B is a weak 2-hop reasoner. The paper compares *triggers* under the same model, so the claim is the gain, not the absolute score.
3. **What we claim is the improvement:**
   - 33.4% → 55.2% is **+65% relative**, matching the oracle trigger (54.0%), so a better trigger could hardly do more with this repair.
   - Failure detection is AUROC ≈ 0.91.
4. **The metric:** "correct" = token F1 ≥ 0.5. Exact match is stricter: 25.8% → 41.8%.

### Honest limitations

- The judge is the untuned, zero-shot 3B model, a matched-compute approximation of Doctor-RAG (which uses a fine-tuned diagnoser). The probe is trained on our labels; the judge is not.
- The improved repair (entity-led query, re-ranking, second restart) was designed after seeing the first results on the same 2Wiki questions. HotpotQA, where it brings no gain, is the unbiased check. The original Naive Restart is reported as an ablation (2Wiki 46.4%, HotpotQA 55.6%).
- 60% of questions are restarted. On HotpotQA the probe breaks more correct answers than the judge (32 vs 14).
- 2-hop questions only, a single 3B model, a simulated first-stage retriever, and a corpus pooled from the evaluation datasets rather than full Wikipedia.

---

## Setup

```bash
pip install -r requirements.txt
```

For Llama-3.2-3B-Instruct you need a Hugging Face token with gated-model access (the Phase 3–5 code falls back to the ungated mirror `unsloth/Llama-3.2-3B-Instruct`, which has the same weights):
```bash
huggingface-cli login
```

---

## Commands

### Phase 1 — regenerate the dataset (not needed: the data is on Hugging Face)
```bash
python phase1_dataset/build_dataset.py --config configs/data_config.yaml            # download → CoT → label → counterfactuals → split
python phase1_dataset/build_dataset.py --config configs/data_config.yaml --dry-run  # validate schema only
python phase1_dataset/generate_cot.py  --config configs/data_config.yaml            # individual steps: download / generate_cot / label_hops / counterfactuals
```

Output record (`*.jsonl`):
```json
{
  "id": "...", "source": "2wikimultihopqa", "question": "...", "gold_answer": "...",
  "reasoning_graph": [{"hop": 1, "gold_entity": "..."}, {"hop": 2, "gold_entity": "..."}],
  "prompt": "...", "context": "Title: ...\n...",
  "generated_cot": "<hop1>...</hop1><hop2>...</hop2><answer>...</answer>",
  "hops": [{"hop_idx": 1, "text": "...", "bridging_entity_gold": "...", "match_method": "normalized_string", "label": 0}],
  "first_fail_hop": null, "final_answer_correct": true,
  "retrieval_condition": "both_present"
}
```

### Phases 3–5 — pipeline
```bash
python tests/test_pipeline.py                 # unit tests incl. live-vs-Phase-1 feature parity (CPU)
python tests/smoke_pipeline_local.py          # every stage end to end with a tiny scripted model (CPU)
python -m src.pipeline.probes --setting tfidf --out outputs/pipeline/tfidf/probes   # probes + 28-layer sweep (CPU, ~10 min)
python -m src.pipeline.run --setting tfidf --out <dir> --probe-dir outputs/pipeline/tfidf/probes   # GPU (~4 h on a T4)
python -m src.pipeline.evaluate --out <dir> --probe-dir outputs/pipeline/tfidf/probes
```
On Kaggle, import `kaggle/pipeline_phase3_5.ipynb` (GPU T4, Internet on, secret `HF_TOKEN`).

| Option | Values |
|---|---|
| `--setting` | `tfidf` (main) · `full10` |
| `--probe-mode` | `shared` (main: one probe for both hops) · `dedicated` (hop-1 + hop-2 probes) |
| `--repair` | `v2` (entity-led hop-2 query + cross-encoder re-ranking, default) · `v1` (question + hop text, dense top-3) |
| `--rounds` | `2` (default) · `1` (original Naive Restart) |
| `--reuse-from <dir>` | reuse an earlier run's trajectories, judge scores, features, corpus and (if compatible) restarts |

Runs, data sources and both retrieval stages in detail: `outputs/pipeline/README.md`.

---

## Project Structure

Data and hidden states live on Hugging Face (`AnishRacherla/LinguaFranca-Phase3`), not in git. The Phase 1 dataset is also on [Kaggle](https://www.kaggle.com/datasets/havishbalaga/linguafranca-phase1-data). Raw Phase 3–5 run outputs (~300 MB per run) are not committed; their summaries are in `outputs/pipeline/`.

```
LinguaFranca/
├── configs/data_config.yaml
├── phase1_dataset/            ← Phase 1 (download, generate_cot, label_hops, hop_rules, counterfactuals, build_dataset)
├── utils/                     ← entity matching + Wikidata aliases
├── src/pipeline/              ← Phases 3–5: probes, retriever, live generation, judge, systems, run, evaluate
├── scripts/                   ← canonical dataset, Kaggle bundle, pipeline-notebook generator
├── lf_common.py, lf_models.py ← shared split / metrics / model helpers (Phases 2–4)
├── advanced_probing.py        ← Phase 2–3 probes (v2); advanced_probing_distractor.py = TF-IDF-branch version
├── evaluate_saved_probe.py, bootstrap_confident_wrong.py
├── evaluate_llm_baseline.py   ← LLM judge baseline (v2)
├── causal_patching.py         ← Phase 4 (v4 run)
├── measure_tfidf_recall.py    ← TF-IDF first-stage recall analysis
├── kaggle/                    ← notebooks to run (01 probing, 02 judge, 03 patching, pipeline_phase3_5)
├── kaggle_phase1.ipynb, kaggle_phase2_probes.ipynb
├── outputs/                   ← probing / llm_judge / patching results; pipeline/ = Phase 3–5 runs + paper draft
├── tests/                     ← test_phase1.py, test_pipeline.py, smoke_pipeline_local.py
├── legacy/                    ← superseded notebooks and scripts (see legacy/README.md)
├── FIXES_AND_RERUN.md, imppoints.md
└── LinguaFranca-Proposal.pdf
```
