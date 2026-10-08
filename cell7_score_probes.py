import json, joblib
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download

FEATURES_PATH = "probe_features.npz"      # e.g. "/kaggle/input/<your-notebook-output>/probe_features.npz"
PROBE_LAYERS = [10, 13]                   # any subset of the saved feature layers (9-13)
REPO = "AnishRacherla/LinguaFranca-Phase3"
LOCAL_PROBE_DIR = None                    # or a folder with hop1_probe_layer{L}.joblib to skip HF

Z = np.load(FEATURES_PATH)
FL = list(Z["feature_layers"]); CL = list(Z["control_layers"])
ids, recovery, has_ctrl = Z["ids"], Z["recovery"], Z["has_control"]
n, num_layers = recovery.shape
print(f"Loaded features for {n} pairs, feature layers {FL}")

def load_probe(P):
    path = (f"{LOCAL_PROBE_DIR}/hop1_probe_layer{P}.joblib" if LOCAL_PROBE_DIR else
            hf_hub_download(REPO, f"probes/hop1_probe_layer{P}.joblib", repo_type="dataset"))
    return joblib.load(path)

def score(probe, X):
    \"\"\"Probe log-odds of FAILURE (class 1) for every row of X [..., d].\"\"\"
    shp = X.shape[:-1]
    flat = np.nan_to_num(X.reshape(-1, X.shape[-1]).astype(np.float64))
    return probe.decision_function(flat).reshape(shp)

train_ids = None
try:
    tid_path = (f"{LOCAL_PROBE_DIR}/probe_train_ids.json" if LOCAL_PROBE_DIR else
                hf_hub_download(REPO, "probes/probe_train_ids.json", repo_type="dataset"))
    train_ids = set(map(str, json.load(open(tid_path))))
except Exception:
    print("(no probe_train_ids.json found - reporting all pairs only)")

sig = lambda x: 1 / (1 + np.exp(-np.asarray(x)))
rng = np.random.default_rng(0)
def boot(x, B=2000):
    x = np.asarray(x); i = rng.integers(0, len(x), (B, len(x))); mm = x[i].mean(1)
    return np.percentile(mm, 2.5), np.percentile(mm, 97.5)

S = {}
for P in PROBE_LAYERS:
    k = FL.index(P)
    pr = load_probe(P)
    S[P] = {"clean": score(pr, Z["F_clean"][:, k]), "cf": score(pr, Z["F_cf"][:, k]),
            "entity": score(pr, Z["F_entity"][:, :, k]), "control": score(pr, Z["F_control"][:, :, k])}

def analyze(sel, title):
    print(f"\n################ {title}: n = {sel.sum()} ################")
    for P in PROBE_LAYERS:
        cl, co = S[P]["clean"][sel], S[P]["cf"][sel]
        ent, ctl, rec = S[P]["entity"][sel], S[P]["control"][sel], recovery[sel]
        hc = has_ctrl[sel]
        d = co - cl; lo, hi = boot(d)
        ok = sig(cl).mean() < 0.3
        print(f"================ Probe @ layer {P} ================")
        print(f"SANITY: P(fail) on CLEAN correct hops = {sig(cl).mean():.3f} "
              f"({'OK' if ok else 'TOO HIGH - wrong probe or feature mismatch, do not interpret'})")
        print("TEST A - same hop-1 words, only the question entity differs:")
        print(f"  P(fail) clean {sig(cl).mean():.3f} -> corrupted {sig(co).mean():.3f} | "
              f"log-odds rise {d.mean():+.2f} [95% CI {lo:+.2f}, {hi:+.2f}] | {(d > 0).mean()*100:.0f}% of pairs rise")
        print(f"TEST B - patch the entity at layer L (only L < {P} can reach the probe's layer):")
        print(f"  {'L':>3} | behaviour recovery | probe log-odds drop | shuffled-control drop")
        xs, ys = [], []
        for l in range(num_layers):
            drop = co - ent[:, l]
            cs = f"{(co[hc] - ctl[hc, CL.index(l)]).mean():+.2f}" if l in CL and hc.any() else "   -"
            if l < P:
                xs += list(rec[:, l]); ys += list(drop)
            if l <= P + 1:
                print(f"  {l:>3} | {rec[:, l].mean():18.2f} | {drop.mean():+19.2f} | {cs:>21}")
        rho, pval = spearmanr(xs, ys)
        print(f"  Spearman(behaviour recovery, probe drop) over pairs x layers<{P}: rho = {rho:.2f} (p = {pval:.1e})")

analyze(np.ones(n, bool), "ALL valid pairs")
if train_ids is not None:
    held = np.array([i not in train_ids for i in ids])
    if held.sum() >= 10:
        analyze(held, "HELD-OUT pairs only (probe never trained on them)")
    else:
        print(f"\nOnly {held.sum()} held-out pairs - too few for a separate analysis.")
