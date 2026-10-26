"""
Generates kaggle/pipeline_phase3_5.ipynb (Phases 3–5 end to end).
Run from the project root:  python scripts/make_pipeline_notebook.py
"""

import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("# LinguaFranca — Phases 3–5: probe-triggered Naive Restart RAG vs. Doctor-RAG-style judge",
       "",
       "Runs the whole pipeline (`src/pipeline/`): live probe-monitored generation, post-hoc LLM judge,",
       "retrieval + restart, the latency subset, feature-parity check, and the final evaluation.",
       "",
       "**Kaggle settings:** Accelerator **GPU T4** (x1 is enough), Internet **On**,",
       "Add-ons → Secrets → `HF_TOKEN` switched on (an account that accepted the Llama 3.2 licence;",
       "otherwise the ungated `unsloth/Llama-3.2-3B-Instruct` mirror with the same weights is used).",
       "",
       "1. First run with `SMOKE = True` interactively (~15 min) to check everything works.",
       "2. Then set `SMOKE = False` and use **Save Version → Save & Run All** (~3–4 h on one T4).",
       "3. Download `pipeline_results.zip` from the Output tab and send it back (or paste `results.md`).",
       "",
       "**Re-running with a new repair / rounds setting:** attach the earlier full run as input",
       "(right panel → *Add Input* → *Your Work* → this notebook's earlier version). The run then reuses its",
       "trajectories, judge scores, features and corpus and only re-does the restarts (~1.5 h instead of ~4 h).",
       "",
       "Every stage checkpoints to jsonl and skips finished items, so re-running the run cell resumes."),
    code("# ===== Setup =====",
         "!pip install -q faiss-cpu",
         "import os, subprocess, sys",
         "REPO_URL = 'https://github.com/ANLPproject/LinguaFranca.git'",
         "WORK = '/kaggle/working/LinguaFranca'",
         "if not os.path.exists(WORK):",
         "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, WORK], check=True)",
         "else:",
         "    subprocess.run(['git', '-C', WORK, 'pull'], check=True)",
         "os.chdir(WORK); sys.path.insert(0, WORK)",
         "print(subprocess.run(['git', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)",
         "try:",
         "    from kaggle_secrets import UserSecretsClient",
         "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')",
         "    print('HF_TOKEN loaded from Kaggle secrets')",
         "except Exception as e:",
         "    print(f'No HF_TOKEN secret ({e.__class__.__name__}); the ungated model mirror will be used')",
         "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'GPU(s)')"),
    code("# ===== Configuration =====",
         "SMOKE = True            # True: tiny end-to-end check (~15 min).  False: the full run.",
         "SETTING = 'tfidf'       # 'tfidf' = TF-IDF top-5 first-stage retrieval (main) | 'full10' = all 10 passages",
         "N_2WIKI, N_HOTPOT, N_LIVE, N_EXTRA_CORPUS = (8, 8, 4, 2000) if SMOKE else (500, 500, 100, 20000)",
         "REPAIR = 'v2'           # 'v1' = original repair | 'v2' = bridge-entity hop-2 query + cross-encoder re-ranking",
         "ROUNDS = 2              # max restarts per question (1 = original Naive Restart)",
         "OUT = f'/kaggle/working/pipeline_{SETTING}_{REPAIR}_r{ROUNDS}' + ('_smoke' if SMOKE else '')",
         "# An attached earlier full run (any /kaggle/input/**/s0.jsonl of this setting) is reused automatically.",
         "import glob",
         "prev = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/s0.jsonl', recursive=True)",
         "        if f'pipeline_{SETTING}' in p and '_smoke' not in p]",
         "REUSE_FROM = prev[0] if prev and not SMOKE else None",
         "print('reusing earlier run:', REUSE_FROM)",
         "# Probes trained from the stored Phase 1 hidden states (committed to the repo; portable .npz).",
         "# If the folder is missing, the run trains them first (CPU, downloads ~2 GB of hidden states).",
         "PROBE_DIR = f'{WORK}/outputs/pipeline/{SETTING}/probes'",
         "print('probes present:', os.path.exists(f'{PROBE_DIR}/probes_meta.json'), '| output dir:', OUT)"),
    code("# ===== Run all stages =====",
         "cmd = [sys.executable, '-u', '-m', 'src.pipeline.run', '--setting', SETTING, '--out', OUT,",
         "       '--probe-dir', PROBE_DIR, '--n-2wiki', str(N_2WIKI), '--n-hotpot', str(N_HOTPOT),",
         "       '--n-live', str(N_LIVE), '--n-extra-corpus', str(N_EXTRA_CORPUS),",
         "       '--repair', REPAIR, '--rounds', str(ROUNDS)] + (['--reuse-from', REUSE_FROM] if REUSE_FROM else [])",
         "print(' '.join(cmd))",
         "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
         "for line in proc.stdout:",
         "    print(line, end='')",
         "assert proc.wait() == 0, 'pipeline failed - see the log above (re-running this cell resumes)'"),
    code("# ===== Results =====",
         "from IPython.display import Markdown, Image, display",
         "display(Markdown(open(f'{OUT}/results.md', encoding='utf-8').read()))",
         "if os.path.exists(f'{OUT}/sweep.png'):",
         "    display(Image(f'{OUT}/sweep.png'))"),
    code("# ===== Package outputs (the corpus embeddings are rebuilt on demand, so they are left out) =====",
         "import shutil, zipfile",
         "zpath = '/kaggle/working/pipeline_results.zip'",
         "with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:",
         "    for root, _, files in os.walk(OUT):",
         "        if '/corpus' in root:",
         "            continue",
         "        for f in files:",
         "            p = os.path.join(root, f)",
         "            z.write(p, os.path.relpath(p, os.path.dirname(OUT)))",
         "print(zpath, round(os.path.getsize(zpath) / 1e6, 1), 'MB')"),
]

nb = {"nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10.0"}},
      "cells": cells}
out = Path("kaggle/pipeline_phase3_5.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Created {out}")
