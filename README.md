# LinguaFranca

> **"Do Internal Failures Predict External Ones?"**  
> Causal Hop Localization for Faithful, Efficient Retrieval-Augmented Multi-Hop QA

---

## Overview

This repository implements the full data and modelling pipeline for the project. We test whether a lightweight probe on an LLM's hidden states can detect and localize a failing reasoning hop **at the hop boundary** — before subsequent steps are generated — and whether that signal is **causally** responsible for the failure (via activation patching).

> **Start here:** `FIXES_AND_RERUN.md` (what was fixed in Oct 2026 and what to run) and `imppoints.md` (current results).

### Phases (repo numbering)
| Phase | Description | Code | Kaggle notebook |
|---|---|---|---|
| 1 | Data construction & hop labeling (CoT, hidden states, labels, counterfactuals) | `phase1_dataset/`, `utils/` | `kaggle_phase1.ipynb` (only to regenerate) |
| 1b | Canonical dataset (pinned revision + extra-hop fix) | `scripts/make_canonical_dataset.py` | — (CPU, 1 min) |
| 2–3 | Hidden-state probes, baselines, controls | `advanced_probing.py`, `evaluate_saved_probe.py`, `bootstrap_confident_wrong.py`, `lf_common.py` | `kaggle/01_probing.ipynb` (CPU) |
| 3 | LLM-as-judge text baseline (same rows as the probe) | `evaluate_llm_baseline.py` | `kaggle/02_llm_judge.ipynb` (GPU) |
| 4 | Causal validation (activation patching) | `causal_patching.py`, `lf_models.py` | `kaggle/03_causal_patching.ipynb` (GPU) |
| 5 | Probe-triggered Naive-Restart RAG + Doctor-RAG-style baseline | *(not started)* | — |

---

## Setup

```bash
pip install -r requirements.txt
```

For Llama-3.2-3B-Instruct you need a Hugging Face token with gated-model access:
```bash
huggingface-cli login
```

---

## Phase 1 — Dataset Construction

### What Happened
In Phase 1, we built the dataset required to train a probe that detects reasoning failures in LLMs based on their hidden states.

### Why Did We Do This?
Large Language Models usually answer questions correctly, meaning natural failures are rare. To train a balanced probe, we need both successful reasoning paths and failed ones, along with the model's internal "brain activity" (hidden states) during both.

### How Was It Done?
1. **Generated Reasoning**: Fed multi-hop questions to the model to generate Chain-of-Thought reasoning.
2. **Recorded States**: Saved the model's internal hidden states while it generated its reasoning.
3. **Labeled Hops**: Programmatically graded each reasoning hop as Success or Failure based on whether it retrieved the correct bridging entity.
4. **Counterfactuals**: Artificially induced failures by swapping entities in the prompt, forcing the model to fail, which balances our dataset.
5. **Split Data**: Separated the data into train, val, and test splits securely.

### What Should We Do Next?
**Phase 2 — Internal-state probing**: We will train a classifier probe on `train.jsonl` using the hidden states saved in `data/hidden_states/` to predict hop-level failures before they fully manifest in text.

### Commands

```bash
# Full pipeline (download → CoT → label → counterfactuals → split)
python phase1_dataset/build_dataset.py --config configs/data_config.yaml

# Dry-run (skips model inference, validates schema only)
python phase1_dataset/build_dataset.py --config configs/data_config.yaml --dry-run

# Individual steps
python phase1_dataset/download.py       --config configs/data_config.yaml
python phase1_dataset/generate_cot.py   --config configs/data_config.yaml
python phase1_dataset/label_hops.py     --config configs/data_config.yaml
python phase1_dataset/counterfactuals.py --config configs/data_config.yaml
```

### Output schema (`data/processed/*.jsonl`)
```json
{
  "id": "2wiki_00421",
  "source": "2wikimultihopqa",
  "question": "...",
  "gold_answer": "...",
  "reasoning_graph": [{"hop": 1, "gold_entity": "..."}],
  "generated_cot": "<hop1>...</hop1><hop2>...</hop2>",
  "hops": [{
    "hop_idx": 1,
    "text": "...",
    "bridging_entity_gold": "...",
    "bridging_entity_pred": "...",
    "match_method": "normalized_string",
    "label": 0
  }],
  "first_fail_hop": null,
  "is_counterfactual": false,
  "final_answer_correct": true
}
```

---

## Project Structure

> **Note:** The `data/` folder is heavily populated during Phase 1. Because of its large size (3+ GB), it is not tracked in this repository locally. The generated data is hosted securely on Kaggle at the [Phase 1 Dataset Link](https://www.kaggle.com/datasets/havishbalaga/linguafranca-phase1-data).

Data and hidden states live on Hugging Face (`AnishRacherla/LinguaFranca-Phase3`, pinned revision
`ce07fd04…` in `lf_common.py`), not in git.

```
LinguaFranca/
├── configs/data_config.yaml
├── phase1_dataset/            ← Phase 1 pipeline (download, generate_cot, label_hops, hop_rules, counterfactuals, build_dataset)
├── utils/                     ← entity matching + Wikidata aliases
├── scripts/
│   ├── make_canonical_dataset.py   ← builds data/canonical/augmented_v2.jsonl
│   └── make_kaggle_bundle.py       ← builds kaggle_upload/linguafranca_bundle.zip
├── lf_common.py, lf_models.py ← shared split / metrics / model helpers
├── advanced_probing.py        ← Phase 2-3 probes (v2)
├── evaluate_saved_probe.py    ← score saved probes (v2)
├── bootstrap_confident_wrong.py
├── evaluate_llm_baseline.py   ← LLM judge baseline (v2)
├── causal_patching.py         ← Phase 4 (v2)
├── kaggle/                    ← the notebooks to run (01 CPU, 02/03 GPU)
├── kaggle_phase1.ipynb        ← regenerate Phase 1 from scratch (not needed now)
├── outputs/probing/           ← v2 probing results
├── legacy/                    ← superseded notebooks (see legacy/README.md)
├── tests/test_phase1.py       ← unit tests (python tests/test_phase1.py)
├── FIXES_AND_RERUN.md, imppoints.md
└── LinguaFranca-Proposal.pdf
```
