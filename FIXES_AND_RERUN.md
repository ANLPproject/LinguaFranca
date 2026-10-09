# Fixes & re-run guide (Oct 2026)

---

## 0. Current status (read this first)

| what | status |
|---|---|
| Phase 1 data + hidden states | fine, no rerun |
| Labels (extra-hop fix), canonical dataset | done |
| Probing (Phase 2–3) | done; numbers in `imppoints.md` §2–5 (run locally, reproducible with `kaggle/01`, optional) |
| LLM-judge baseline (`kaggle/02`) | **done, valid** (probe AUROC 0.958 vs best judge 0.718); `imppoints.md` §6 |
| Causal patching (`kaggle/03`) | **run the final v4 once more** — the previous run had a bug in my entity-span definition (see §2 item 4b) |


This file explains what was wrong in Phases 1–4, what was changed, which results change, and exactly what to run.

**Short version:** Phase 1 generation (CoT + hidden states) does **not** need to be re-run. One labeling rule was wrong; it is fixed by a cheap post-processing step that is already done. Probing (Phase 2–3) has already been re-run with the fixed code (numbers are in `imppoints.md`). You need to run **two GPU notebooks on Kaggle**: the LLM-judge baseline (~1 h) and the causal patching (~1 h). The probing notebook is optional (to reproduce my numbers on Kaggle, ~1 h CPU).

---

## 1. Do I need to re-run from Phase 1?

**No.** Here is why, phase by phase:

| phase | what is wrong | needs re-run? |
|---|---|---|
| 1 — CoT generation + hidden states | nothing; the 4,078 `.pt` files on HF are correct | **no** |
| 1 — hop labels | hops beyond the 2-edge gold graph (hop 3–18) were labeled "failure" | fixed post-hoc, already done (`scripts/make_canonical_dataset.py`, seconds, CPU). Identical to re-running the labeler (proof in `phase1_dataset/hop_rules.py`) |
| 1 — counterfactuals | only 78 were made (the stopping rule counted the fake failures), and they are "entity not in context" questions | **no** — we keep the 78 (they have hidden states) and report them as a separate stratum. Regenerating would only add more trivially detectable hedging examples |
| 2/3 — probes | wrong hop saved, extra hops in the data, no FPR, layer picked on test | **re-run done locally** (`outputs/probing/`); optional Kaggle reproduction: `kaggle/01_probing.ipynb` |
| 3 — LLM judge | scored on different rows/data than the probe, incl. fake-failure hops | **re-run on Kaggle GPU**: `kaggle/02_llm_judge.ipynb` |
| 4 — patching | sampling instead of greedy, invalid design, no recorded result | **re-run on Kaggle GPU**: `kaggle/03_causal_patching.ipynb` |

Only if you later want a *bigger* dataset (more questions or a new counterfactual design) would you run `kaggle_phase1.ipynb` again (~2 h per 2,000 examples on a T4, plus counterfactuals). Not needed now.

---

## 2. What was wrong (verified, with severity)

### Critical / high (changed reported results)
1. **Extra hops labeled as failures.** 1,329 of 4,000 examples have more `<hopN>` blocks than the 2 gold edges. The labeler marked every extra hop "failure", which made up 64 % of all failure labels (3,420 / 5,307). Consequences:
   * the "all-hop" probe accuracy (91.4 %) was partly hop-position detection — the hop index alone scored 77.9 % / AUROC 0.855;
   * the counterfactual loop believed the data was already 44 % failing and stopped after 78 counterfactuals;
   * the LLM judge was scored on thousands of these fake failures.
   **Fix:** `phase1_dataset/hop_rules.py` (label -1), used by `label_hops.py` and `scripts/make_canonical_dataset.py`.
2. **Saved "hop1" probes were hop-2 probes.** v1 `advanced_probing.py` saved probes with `hop_idx == 1`, which is hop 2 after 0-indexing (confirmed: 100 % train accuracy on hop 2, 18–40 % on hop 1). `evaluate_saved_probe.py` v1 scored them on hop 1 and printed only recall, which produced the "100 % hedging / 98.2 % confident-wrong" table. Those probes flag **89 % of hop-1 successes** as failures (layer 11). The correct figure is ~75 % recall at ~2–3 % false alarms. **Fix:** v2 `advanced_probing.py` saves `hop1_/hop2_/hop12_probe_layer{L}.joblib` with metadata; v2 `evaluate_saved_probe.py` always prints FPR and scores both hops.
3. **Probe and LLM judge were compared on different data.** On 2026-09-30 `kaggle_relabel.ipynb` overwrote the HF data files with a new set of 115 counterfactuals whose hidden states were never uploaded (9 of 115 have `.pt` files). The probe numbers came from the earlier file (78 CFs), the judge from the later one, and each script made its own split. The judge also judged the fake-failure extra hops. **Fix:** one canonical file pinned to the pre-overwrite revision (`data/canonical/augmented_v2.jsonl`), and one split function used by every script (`lf_common.py`).
4. **Phase 4 notebook.** `model.generate()` was called without `do_sample=False`, and Llama-3.2-Instruct samples by default (T = 0.6), so baseline and patched answers differed by chance. The markdown described the opposite experiment. ~90 % of CF prompts already contained "X is not mentioned" in hop 1, so "restore the clean answer" had no valid target. No result was ever recorded. **Fix:** new `causal_patching.py`: greedy decoding, causal tracing on token-aligned pairs, controls, and a hook self-test. While building it I also found that HF's last `output_hidden_states` entry is *after* the final norm, so states are now captured with block hooks.
4b. **My own bug in the first `causal_patching.py` run (fixed in v4).** The Llama chat template inserts today's date, and the clean and counterfactual prompts were generated on different days, so they also differ at one date token near the start. v2.0 took "first to last differing token" as the entity span (~1,000 tokens), so its entity-span results are invalid (its last-token column and Experiment B are unaffected). v4 sets the CF's date token to the clean one (true minimal pairs; verified on all 51 pairs), uses only the 2–11 entity tokens, adds a tail condition, a length-matched donor control and a direct probe link (Experiment C), and refuses to run if the spans look wrong.

### Medium (code bugs that crashed or mis-stated things)
5. Two `NameError`s in v1 `advanced_probing.py` (`X_test_text`, `X_test_h1`). The script crashed before the probe-saving block, so the HF probes came from an older run.
6. The shuffled-label control used an MLP (the probe is logistic regression) and an unseeded permutation. Now LR with a seed: AUROC 0.47–0.51.
7. Layers were chosen on the test split ("Layer 11 sweet spot"). They are now chosen by CV on the train split; layers 9–14 form a plateau.
8. No baselines were reported. Now the majority class, hop index, TF-IDF on hop text, and a hedge-keyword rule are reported on identical test rows.
9. The cross-hop test included training examples. It now uses held-out test rows only.
10. `kaggle_phase1.ipynb` and `kaggle_relabel.ipynb` split by raw id, which could separate `q` and `q_cf`. They now use `build_dataset.split_by_id` (the probing never used these splits).

### Low (no effect on results; cleaned up)
11. `tests/test_phase1.py` called `build_prompt` with the wrong number of arguments. Fixed, and tests were added (17 pass).
12. `data/raw/2wikimultihopqa/test.jsonl` was an empty tracked file. Removed.
13. `.gitignore` was UTF-16 (git ignored it), so `__pycache__` files were committed. Rewritten and untracked.
14. `run_counterfactuals(hf_token=…)` is unused. Documented and kept for compatibility.
15. `dataset_offset` (the two generation batches) is now documented in `configs/data_config.yaml`.
16. `counterfactuals.py` required *every* hop label to be 0 to call an example "clean". It now ignores unlabeled hops.

### What the audit files (`phase*_audit.md`) got wrong
* "Blank-gold bug is fixed; HF data is clean" — no: the "blank gold" hops were the extra hops, still labeled failures.
* "91.4 % is inflated by counterfactuals" — CFs are only 1.9 % of examples; the inflation was the extra hops.
* "Layer 0 = raw token embeddings" — no: layer 0 is the output of the first transformer block.
* "The probe-vs-judge comparison is valid" — no (item 3).
* "`dataset_offset: 2000` is in the config file" — it was only set in the notebook.
* "0 % prompt-end patching was never run" — an earlier, greedy version of the notebook did run it (git history), but it was not logged.

---

## 3. What changes in the results

| claim | old | new (v2) |
|---|---|---|
| All-hop probe accuracy | 91.4 % (incl. fake-failure hops) | hops 1+2: **90.9 % acc, AUROC 0.956** (majority 75.4 %, best text baseline 0.77) |
| Hop-1 probe | 95.1 % acc, AUROC 0.953 | same: **95.1 %, AUROC 0.953** — but majority-class acc is 91.6 %; recall 75 %, FPR 3 % |
| Best layer | "Layer 11 is the sweet spot" | plateau at blocks 9–14; CV picks 10 (hop 1) and 13 (hops 1+2) |
| Hedging detection | 100 % | 100 % (all hedging failures are counterfactuals; lexically trivial) |
| Confident-wrong detection | 98.2 % (layer 11) | **~75 %** recall at ~2 % false alarms (layers 10–13) |
| Cross-hop transfer | 62.5 % / 0.661 | AUROC 0.66–0.69 |
| Shuffled labels | 50.6 % | AUROC 0.47–0.51 |
| LLM judge | 61.7 % | **pending** (`kaggle/02`) |
| Patching | 0 % prompt-end (unlogged), no entity-level number | **pending** (`kaggle/03`) |
| Hop failure → final-answer failure | not reported | weak: AUROC 0.59 |

The paper's core claim survives and is now defensible: hidden states detect hop-level failures (AUROC ~0.95), clearly above text-only baselines, especially on hop 2.

---

## 4. Step-by-step: what to run on Kaggle

You need a Kaggle account with **phone verification** (required for Internet and GPUs), and for notebooks 02/03 a **Hugging Face token** from an account that has accepted the **Llama 3.2** (and, for the 8B judge, **Llama 3.1**) licence on huggingface.co.

### Step 0 — make the bundle (already done for you)
`kaggle_upload/linguafranca_bundle.zip` contains the code, the canonical dataset and my probing outputs.
If you change any code later: `python scripts/make_kaggle_bundle.py` rebuilds it.

### Step 1 — upload the bundle as a Kaggle Dataset (once)
1. kaggle.com → **Datasets** → **New Dataset**.
2. Drag in `kaggle_upload/linguafranca_bundle.zip`. Name it e.g. `linguafranca-bundle`. Keep it **Private**. Click **Create**.
   (Kaggle unzips it automatically. The notebooks find it wherever it lands under `/kaggle/input`.)

### Step 2 — add your HF token as a Kaggle secret (once)
In any notebook: **Add-ons → Secrets → Add a new secret**, label `HF_TOKEN`, value = your HF token (read access is enough). Switch it on for each notebook that needs it (02 and 03).

### Step 3 — run `kaggle/02_llm_judge.ipynb` (GPU, ~1 h)
1. kaggle.com → **Code** → **New Notebook** → **File → Import Notebook** → upload `kaggle/02_llm_judge.ipynb`.
2. Right panel: **Add Input** → your `linguafranca-bundle` dataset. **Session options:** Accelerator **GPU T4 x2**, Internet **On**. Secret `HF_TOKEN` on.
3. **Run All** (or "Save Version → Save & Run All" to run it in the background).
4. When done, download `llm_judge_results.zip` from the **Output** section. Send me / paste `summary.md`.
   * If a Llama model fails with a 401/403, the token's account has not accepted that model's licence. The script skips that model and continues.

### Step 4 — run `kaggle/03_causal_patching.ipynb` (GPU, ~2.3 h; use the NEW bundle)
Same steps as Step 3 (GPU T4 x1 is enough). First check that the log prints `HOOK SELF-TEST: … PASS`. Download `patching_results.zip` → `summary.md`.

### Step 5 (optional) — run `kaggle/01_probing.ipynb` (CPU, ~1 h)
Reproduces `outputs/probing/` on Kaggle. Accelerator **None**, Internet **On**, no secret needed. It also re-runs the old HF probes so you can see the hop-2 bug yourself.

### Step 6 — update the write-up
Fill the two PENDING sections of `imppoints.md` from the two `summary.md` files (or send them to me).

### Step 7 (recommended) — publish the fixed artifacts for the team
* Upload `data/canonical/augmented_v2.jsonl` and `outputs/probing/probes/` to the HF repo under new paths (e.g. `v2/augmented_v2.jsonl`, `probes_v2/`). Do **not** overwrite the old files, so the history stays reproducible.
* Commit the code changes to git (see §6) so teammates' notebooks pick them up.

---

## 5. Where every change is

| file | change |
|---|---|
| `phase1_dataset/hop_rules.py` | **new** — extra-hop rule (label -1), back-fill gold entity, recompute `first_fail_hop` |
| `phase1_dataset/label_hops.py` | applies the extra-hop rule at the end of `label_example` |
| `phase1_dataset/counterfactuals.py` | "clean" example = no labeled failure (ignores -1); docstring explains ratio + CF nature |
| `configs/data_config.yaml` | documented `dataset_offset`, LLM-judge budget, failure-ratio semantics (values unchanged) |
| `scripts/make_canonical_dataset.py` | **new** — builds `data/canonical/augmented_v2.jsonl` from the pinned HF revision |
| `scripts/make_kaggle_bundle.py` | **new** — builds `kaggle_upload/linguafranca_bundle.zip` |
| `lf_common.py` | **new** — canonical split, hop-1/2 rows, hidden-state loader, metrics with FPR, grouped bootstrap CI |
| `lf_models.py` | **new** — model loading + residual-stream patch hook (transformers 4.x and 5.x) |
| `advanced_probing.py` | **rewritten** (v2) |
| `evaluate_saved_probe.py` | **rewritten** (v2) — FPR, both hops, `--legacy` mode for the old HF probes |
| `bootstrap_confident_wrong.py` | **rewritten** (v2) — multi-seed stability with FPR |
| `evaluate_llm_baseline.py` | **rewritten** (v2) — identical rows to the probe, logit-based P(fail), strata, probe comparison |
| `causal_patching.py` | **new** — Phase 4 v2 |
| `kaggle/01_probing.ipynb`, `02_llm_judge.ipynb`, `03_causal_patching.ipynb` | **new** Kaggle notebooks |
| `kaggle_phase1.ipynb` | Step 10 split by base id; status note |
| `legacy/` | superseded notebooks/scripts moved here (see `legacy/README.md`) |
| `tests/test_phase1.py` | fixed test 13; added tests 15–17 |
| `imppoints.md`, `README.md` | corrected numbers / structure |
| `.gitignore` | UTF-8; ignores `data/`, bundle, probe binaries |
| `outputs/probing/` | v2 probing results (`results.md`, `results.json`, `qualitative_hop1.md`, `run.log`; probes + predictions are git-ignored but inside the bundle) |

---

## 6. Running things locally (optional, CPU)

```bash
pip install -r requirements.txt scikit-learn joblib
python tests/test_phase1.py                                   # 17 unit tests, ~1 min
python scripts/make_canonical_dataset.py --out data/canonical # ~1 min
# hidden states (2.1 GB):
python -c "from huggingface_hub import snapshot_download as s; s('AnishRacherla/LinguaFranca-Phase3', repo_type='dataset', revision='ce07fd04bf3e39d920e3ccbf769298fa4396eb1f', allow_patterns=['data/hidden_states/*'], local_dir='hf')"
python advanced_probing.py --hs-dir hf/data/hidden_states --out outputs/probing --layer-chunk 6   # ~20-30 min
python scripts/make_kaggle_bundle.py
```
The LLM-judge and patching scripts need a GPU (use the Kaggle notebooks).

Suggested commit (from the repo root):
```bash
git checkout -b fixes-v2
git add -A ':!phase*_audit.md' ':!.agent'
git commit -m "Fix extra-hop labels, probe saving, judge/probe comparison, Phase 4 patching (v2)"
```
