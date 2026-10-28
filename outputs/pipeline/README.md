# Phases 3–5 outputs — runs, data and retrieval

Code: `src/pipeline/` · notebook: `kaggle/pipeline_phase3_5.ipynb` · paper text: `tfidf/results_section_draft.md`.

## Runs (all: setting `tfidf`, 500 held-out 2Wiki + 500 HotpotQA questions, Llama-3.2-3B-Instruct, Kaggle T4)

| folder | probe trigger | repair | max restarts | status |
|---|---|---|---|---|
| `tfidf_v2_r2_shared/` | **one shared probe for both hops (main)** | **v2** | **2** | **final: main results** |
| `tfidf_v2_r2/` | dedicated probes, re-evaluated with the shared probe as main (`results.md`; the original report is `results_dedicated_main.md`) | v2 | 2 (dedicated) | figure `sweep.png`, `tables.tex`, `results.json` for the 1-restart rows |
| `tfidf/` | dedicated probes (shared = ablation row) | v1 (proposal's Naive Restart) | 1 | first run; probes in `tfidf/probes/` |

The trajectories, judge scores and first-round restarts are identical across the v2 runs (re-used with `--reuse-from`). Only the probe trigger and the second round differ. The raw outputs (`s0.jsonl`, `s0_feats/`, restarts, …, ~300 MB per run) are not in git.

## Where the evidence comes from

**1. Phase 1 data: the model does not get all gold evidence.**
2WikiMultihopQA gives each question 10 candidate paragraphs: the 2 gold ones (one per hop) plus 8 distractors. In the `tfidf` setting (branch `experiment/distractor-rag`), a weak first-stage retriever (TF-IDF cosine between the question and each of those 10 paragraphs) keeps only the **top 5**. That simulates real RAG, where the first retrieval misses things. In the 2Wiki test questions:

| first-stage condition | share of questions |
|---|---|
| both gold paragraphs kept | 41 % (203 / 500) |
| only the hop-1 paragraph kept | 56 % (281) |
| only the hop-2 paragraph kept | 1 % (7) |
| neither kept | 2 % (9) |

So **306 gold paragraphs are missing** from the 500 prompts. (The earlier "gold only" setting was never used: Phase 1's gold filter silently failed, so the main Phase 1 data had all 10 paragraphs. See `imppoints.md` §1.)

**2. The "extra" evidence for a restart comes from a separate retrieval corpus.** It stands in for Wikipedia and is built once per run by `retriever.build_corpus()`:

| source | paragraphs |
|---|---|
| all 10 paragraphs of each of the 4,000 2Wiki dataset questions (train + test) | 13,719 |
| all 10 paragraphs of each of the 500 HotpotQA evaluation questions | 4,871 |
| all paragraphs of 20,000 further random 2Wiki train questions (hard, same-domain distractors) | 75,441 |
| **total, de-duplicated by title** | **94,031** |

Every paragraph the first stage dropped is in this corpus (306 / 306 for the 2Wiki test questions), just as a real Wikipedia index contains the article a question needs. The system still has to *find* it among 94k paragraphs, without knowing which one is gold. Retrieval never sees labels, gold titles or answers. It only excludes paragraphs already in the prompt.

## The retrievers

| stage | when | how |
|---|---|---|
| **First stage** (data, fixed) | before generation | TF-IDF (scikit-learn, English stop words) over the question's own 10 candidate paragraphs → top 5 go into the prompt |
| **Repair retrieval** (ours, every restart) | after a trigger fires | **Query:** v1 = question + text of the failing hop. v2 = for a hop-2 failure, the hop-1 bridge entity leads the query, e.g. "Arcady Boytler. Where was the director of film X born? <hop-2 text>". **Dense search:** `sentence-transformers/all-MiniLM-L6-v2` embeddings (normalised, so inner product = cosine) in a FAISS `IndexFlatIP` (exact search) over the 94k corpus. **v2 re-ranking:** the 20 best candidates are re-ordered by the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. **Result:** the top 3 paragraphs not already in the prompt are **prepended** to the prompt's context, the KV cache is flushed, and the answer is regenerated from scratch (Naive Restart). A second round (`--rounds 2`) excludes the first round's paragraphs too. |

The probe trigger, the judge trigger, always-retrieve and the oracle all use this same repair. Only *when* (and from which hop) it is triggered differs.
