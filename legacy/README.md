# Legacy notebooks and scripts (superseded — do not use for results)

Kept for history only. Use the notebooks in `kaggle/` and the v2 scripts at the repo root instead.
See `FIXES_AND_RERUN.md` for the full list of problems.

| file | why it is superseded |
|---|---|
| `create_kaggle_phase4_patching.ipynb` | `generate()` without `do_sample=False` (Llama samples at T=0.6 by default), markdown described the opposite experiment, CF questions are unanswerable so "restoring the clean answer" has no valid target. Replaced by `causal_patching.py` + `kaggle/03_causal_patching.ipynb`. |
| `create_kaggle_phase3_5_advanced_probing.ipynb`, `kaggle_qualitative_report.ipynb` | Ran `evaluate_saved_probe.py` v1 on the HF probes, which were trained on hop 2 (saving bug). Produced the invalid "100 % hedging / 98.2 % confident-wrong" table. |
| `create_kaggle_llm_baseline.ipynb` | LLM judge v1: judged extra hops that were "failure" by construction, on a different data file/split than the probe. |
| `kaggle_relabel.ipynb`, `create_kaggle_relabel.py` | Overwrote the HF data files on 2026-09-30 with a re-generated counterfactual set (115 CFs) whose hidden states were never uploaded (9/115 have `.pt` files); also ran with an empty Wikidata alias table; ID-level (not pair-level) splits. |
| `kaggle_phase3_eval.ipynb`, `create_kaggle_phase3_eval.py` | Probes trained on all hops incl. extra hops; saved only a `.pkl` best-layer probe. |
| `check_blanks.py` | Counted "blank gold" hops, which were really the extra hops beyond the 2-edge gold graph (now labeled -1). |
