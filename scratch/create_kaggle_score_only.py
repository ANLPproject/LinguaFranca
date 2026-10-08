import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Score Saved Probe Features\n",
            "Make sure to attach the Kaggle dataset containing your `probe_features.npz` file before running this!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q huggingface_hub joblib scipy numpy"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, joblib, os, glob\n",
            "import numpy as np\n",
            "from scipy.stats import spearmanr\n",
            "from huggingface_hub import hf_hub_download\n",
            "\n",
            "# Automatically find probe_features.npz in the attached Kaggle inputs\n",
            "FEATURES_PATH = None\n",
            "for f in glob.glob('/kaggle/input/**/probe_features.npz', recursive=True):\n",
            "    FEATURES_PATH = f\n",
            "    break\n",
            "\n",
            "if not FEATURES_PATH:\n",
            "    raise FileNotFoundError(\"Could not find probe_features.npz in /kaggle/input/. Did you attach the dataset?\")\n",
            "else:\n",
            "    print(f\"Found features at: {FEATURES_PATH}\")\n",
            "\n",
            "PROBE_LAYERS = [9, 10, 11, 12, 13]\n",
            "REPO = 'AnishRacherla/LinguaFranca-Phase3'\n",
            "\n",
            "Z = np.load(FEATURES_PATH)\n",
            "FL = list(Z['feature_layers']); CL = list(Z['control_layers'])\n",
            "ids, recovery, has_ctrl = Z['ids'], Z['recovery'], Z['has_control']\n",
            "n, num_layers = recovery.shape\n",
            "print(f'Loaded features for {n} pairs, feature layers {FL}')\n",
            "\n",
            "def load_probe(P):\n",
            "    return joblib.load(hf_hub_download(REPO, f'probes/hop1_probe_layer{P}.joblib', repo_type='dataset'))\n",
            "\n",
            "def score(probe, X):\n",
            "    shp = X.shape[:-1]\n",
            "    flat = np.nan_to_num(X.reshape(-1, X.shape[-1]).astype(np.float64))\n",
            "    return probe.decision_function(flat).reshape(shp)\n",
            "\n",
            "train_ids = None\n",
            "try:\n",
            "    tid_path = hf_hub_download(REPO, 'probes/probe_train_ids.json', repo_type='dataset')\n",
            "    train_ids = set(map(str, json.load(open(tid_path))))\n",
            "except Exception:\n",
            "    print('(no probe_train_ids.json found - reporting all pairs only)')\n",
            "\n",
            "sig = lambda x: 1 / (1 + np.exp(-np.asarray(x)))\n",
            "rng = np.random.default_rng(0)\n",
            "def boot(x, B=2000):\n",
            "    x = np.asarray(x); i = rng.integers(0, len(x), (B, len(x))); mm = x[i].mean(1)\n",
            "    return np.percentile(mm, 2.5), np.percentile(mm, 97.5)\n",
            "\n",
            "S = {}\n",
            "for P in PROBE_LAYERS:\n",
            "    k = FL.index(P)\n",
            "    pr = load_probe(P)\n",
            "    S[P] = {'clean': score(pr, Z['F_clean'][:, k]), 'cf': score(pr, Z['F_cf'][:, k]),\n",
            "            'entity': score(pr, Z['F_entity'][:, :, k]), 'control': score(pr, Z['F_control'][:, :, k])}\n",
            "\n",
            "def analyze(sel, title):\n",
            "    print(f'\\n################ {title}: n = {sel.sum()} ################')\n",
            "    for P in PROBE_LAYERS:\n",
            "        cl, co = S[P]['clean'][sel], S[P]['cf'][sel]\n",
            "        ent, ctl, rec = S[P]['entity'][sel], S[P]['control'][sel], recovery[sel]\n",
            "        hc = has_ctrl[sel]\n",
            "        d = co - cl; lo, hi = boot(d)\n",
            "        ok = sig(cl).mean() < 0.3\n",
            "        print(f'================ Probe @ layer {P} ================')\n",
            "        print(f'SANITY: P(fail) on CLEAN correct hops = {sig(cl).mean():.3f} '\n",
            "              f\"{'OK' if ok else 'TOO HIGH - wrong probe or feature mismatch, do not interpret'}\")\n",
            "        print('TEST A - same hop-1 words, only the question entity differs:')\n",
            "        print(f'  P(fail) clean {sig(cl).mean():.3f} -> corrupted {sig(co).mean():.3f} | '\n",
            "              f'log-odds rise {d.mean():+.2f} [95% CI {lo:+.2f}, {hi:+.2f}] | {(d > 0).mean()*100:.0f}% of pairs rise')\n",
            "        print(f'TEST B - patch the entity at layer L (only L < {P} can reach the probe):')\n",
            "        print(f\"  {'L':>3} | behaviour recovery | probe log-odds drop | shuffled-control drop\")\n",
            "        xs, ys = [], []\n",
            "        for l in range(num_layers):\n",
            "            drop = co - ent[:, l]\n",
            "            cs = f'{(co[hc] - ctl[hc, CL.index(l)]).mean():+.2f}' if l in CL and hc.any() else '   -'\n",
            "            if l < P:\n",
            "                xs += list(rec[:, l]); ys += list(drop)\n",
            "            if l <= P + 1:\n",
            "                print(f'  {l:>3} | {rec[:, l].mean():18.2f} | {drop.mean():+19.2f} | {cs:>21}')\n",
            "        rho, pval = spearmanr(xs, ys)\n",
            "        print(f'  Spearman(behaviour recovery, probe drop) over pairs x layers<{P}: rho = {rho:.2f} (p = {pval:.1e})')\n",
            "\n",
            "analyze(np.ones(n, bool), 'ALL valid pairs')\n",
            "if train_ids is not None:\n",
            "    held = np.array([i not in train_ids for i in ids])\n",
            "    if held.sum() >= 10:\n",
            "        analyze(held, 'HELD-OUT pairs only (probe never trained on them)')\n",
            "    else:\n",
            "        print(f'\\nOnly {held.sum()} held-out pairs - too few for a separate analysis.')\n"
        ]
    }
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {},
    "cells": cells
}

with open("create_kaggle_score_only.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Created create_kaggle_score_only.ipynb")
