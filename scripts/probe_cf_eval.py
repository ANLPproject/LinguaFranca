"""
scripts/probe_cf_eval.py
────────────────────────
CPU: do the deployed probe and the probe-suite models (Anish, probe_suite_v1/tfidf)
detect the TF-IDF counterfactual hop-1 failures (cf_v1)?  Decides whether the
deployed probe should be replaced.

Scored rows (hop 1 only: every CF fails at hop 1; stored hidden states, no GPU)
  cf       the 505 counterfactual hop-1 failures, by failure kind
  seed     their clean seeds (hop 1 correct) - same context, other entity
  natural  natural TF-IDF test-split hop-1 rows: hedging failures, confident
           failures, successes (the reference the probes were tuned for)
Probes: deployed shared hop12 (layer 10, Platt + τ) and dedicated hop1, and the
suite's hop12 / hop1 lr_default, lr (tuned), svm, mlp at their selected layers.
Thresholds: each probe's own CV-F0.5 threshold (the operating point), plus its
default threshold (0.5 / SVM margin 0; deployed: raw P 0.5) to separate the
threshold effect from the probe.

Main numbers use TEST-split CFs (their seeds were never in any probe's training);
train-split CFs are reported separately.

Usage (Kaggle CPU or local; downloads ~1 GB of hidden states):
  python scripts/probe_cf_eval.py --out outputs/probe_cf_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import (CF_V1, HF_REPO, SETTINGS, apply_extra_hop_rule, hf_file, is_hedge,  # noqa: E402
                                 load_setting_examples, read_jsonl)
from src.pipeline.probes import LinearProbe, apply_platt  # noqa: E402

DEPLOYED = ROOT / "outputs/pipeline/tfidf/probes"
SUITE_REV = "e95aeb3f95482538ebf49e948b90d51ee5841b04"     # probe_suite_v1/tfidf upload
SUITE_DIR = "probe_suite_v1/tfidf/probes"
SUITE = ["lr_default_hop12_L14", "lr_hop12_L14", "svm_hop12_L14", "mlp_hop12_L13",
         "lr_default_hop1_L11", "lr_hop1_L13", "svm_hop1_L13", "mlp_hop1_L13"]


# ─────────────────────────────────────────────────────────────────────────────
# Probes: name -> (score(X[n, d]) -> scores, meta with layer / operating threshold / default threshold)
# ─────────────────────────────────────────────────────────────────────────────

def deployed_probes():
    meta = json.loads((DEPLOYED / "probes_meta.json").read_text())
    out = {}
    for task in ("hop12", "hop1"):
        p = meta["probes"][task]
        lp, calib = LinearProbe(DEPLOYED / f"{task}.npz"), p["calibration"]
        # calibrated scale; raw P 0.5 corresponds to calibrated apply_platt(calib, 0.5)
        out[f"deployed_{task}_L{p['layer']}"] = (
            lambda X, lp=lp, c=calib: np.asarray(apply_platt(c, lp.p_fail(X)), dtype=float),
            dict(layer=p["layer"], thr=p["tau"], thr_default=float(apply_platt(calib, 0.5)), kind="lr (calibrated)"))
    return out


def suite_probes():
    out = {}
    for name in SUITE:
        meta = json.loads(Path(hf_file(f"{SUITE_DIR}/{name}.json", SUITE_REV)).read_text())
        m = dict(layer=meta["layer"], thr=meta["tau_cv_f05"], thr_default=meta["decision_threshold"], kind=meta["kind"])
        if meta["kind"] in ("lr", "svm"):
            z = np.load(hf_file(f"{SUITE_DIR}/{name}.npz", SUITE_REV))
            mean, scale = z["mean"].astype(np.float64), z["scale"].astype(np.float64)
            coef, b = z["coef"].astype(np.float64), float(np.ravel(z["intercept"])[0])

            def score(X, mean=mean, scale=scale, coef=coef, b=b, svm=meta["kind"] == "svm"):
                s = ((np.nan_to_num(np.asarray(X, dtype=np.float64)) - mean) / scale) @ coef + b
                return s if svm else 1.0 / (1.0 + np.exp(-s))
        else:
            import joblib
            try:
                est = joblib.load(hf_file(f"{SUITE_DIR}/{name}.joblib", SUITE_REV))
            except Exception as e:                                   # sklearn pickle from another version
                print(f"[skip] {name}: cannot unpickle ({e.__class__.__name__}: {e})")
                continue

            def score(X, est=est):
                X = np.nan_to_num(np.asarray(X, dtype=np.float32))
                return est["mlp"].predict_proba(est["scaler"].transform(X))[:, 1]
        out[name] = (score, m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rows and features
# ─────────────────────────────────────────────────────────────────────────────

def hop1(e):
    h = next((h for h in e.get("hops", []) if h.get("hop_idx") == 1 and h.get("label") in (0, 1)), None)
    order = [sp["hop_idx"] for sp in e.get("hop_spans") or []]
    return (h, order.index(1)) if h is not None and 1 in order else (None, None)


def build_rows():
    natural = {e["id"]: e for e in load_setting_examples("tfidf")}
    test_ids = set(json.loads((DEPLOYED / "probes_meta.json").read_text())["test_ids"])
    cfs = read_jsonl(hf_file(CF_V1["file"], CF_V1["revision"]))
    rows = []
    for cf in cfs:
        apply_extra_hop_rule(cf)
        sw = cf["counterfactual_swap"]
        h, pos = hop1(cf)
        seed = natural.get(cf["clean_pair_id"])
        hs, ps = hop1(seed) if seed else (None, None)
        if h is None or hs is None:
            continue
        rows.append(dict(group="cf", id=cf["id"], cf=True, pos=pos, y=h["label"], kind=sw["failure_kind"], split=sw["split"]))
        rows.append(dict(group="seed", id=seed["id"], cf=False, pos=ps, y=hs["label"], kind="seed", split=sw["split"]))
    for i in sorted(test_ids):
        e = natural.get(i)
        h, pos = hop1(e) if e else (None, None)
        if h is None:
            continue
        kind = "success" if h["label"] == 0 else ("hedge" if is_hedge(h.get("text", "")) else "confident")
        rows.append(dict(group="natural", id=i, cf=False, pos=pos, y=h["label"], kind=kind, split="test"))
    return rows


def load_features(rows, layers):
    """{layer: float32 [n_rows, d]} from the stored pooled states (natural at the pinned revision, cf_v1)."""
    import torch
    from huggingface_hub import snapshot_download
    s = SETTINGS["tfidf"]
    nat = sorted({r["id"] for r in rows if not r["cf"]})
    cf = sorted({r["id"] for r in rows if r["cf"]})
    print(f"downloading {len(nat)} natural + {len(cf)} counterfactual hidden-state files ...", flush=True)
    root_n = Path(snapshot_download(HF_REPO, repo_type="dataset", revision=s["revision"], max_workers=16,
                                   allow_patterns=[f"{s['hs_prefix']}/{i}.pt" for i in nat])) / s["hs_prefix"]
    root_c = Path(snapshot_download(HF_REPO, repo_type="dataset", revision=CF_V1["revision"], max_workers=16,
                                   allow_patterns=[f"{CF_V1['hs_prefix']}/{i}.pt" for i in cf])) / CF_V1["hs_prefix"]
    X = {L: None for L in layers}
    cache = {}
    for ri, r in enumerate(rows):
        key = (r["cf"], r["id"])
        if key not in cache:
            d = torch.load((root_c if r["cf"] else root_n) / f"{r['id']}.pt", map_location="cpu", weights_only=False)
            li = list(d.get("layer_indices", range(d["pooled"].shape[0])))
            cache = {key: (d["pooled"], li)}                  # rows of one example are adjacent; keep one file
        pooled, li = cache[key]
        for L in layers:
            if X[L] is None:
                X[L] = np.zeros((len(rows), pooled.shape[-1]), dtype=np.float32)
            X[L][ri] = pooled[li.index(L), r["pos"]].float().numpy()
    return X


# ─────────────────────────────────────────────────────────────────────────────

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def evaluate(rows, X, probes):
    A = {k: np.array([r[k] for r in rows]) for k in ("group", "kind", "split", "y")}
    R = {}
    for name, (score, m) in probes.items():
        s = score(X[m["layer"]])
        res = {"layer": m["layer"], "kind": m["kind"], "thr": m["thr"], "thr_default": m["thr_default"]}
        for tname, t in (("op", m["thr"]), ("default", m["thr_default"])):
            flag = s >= t
            d = {}
            for split in ("test", "train"):
                cf = (A["group"] == "cf") & (A["split"] == split)
                seed = (A["group"] == "seed") & (A["split"] == split)
                d[f"cf_{split}"] = float(flag[cf].mean())
                d[f"seed_{split}_false_alarm"] = float(flag[seed].mean())
                for k in ("hedge", "other_wrong_entity", "copied_seed_entity"):
                    mk = cf & (A["kind"] == k)
                    d[f"cf_{split}_{k}"] = float(flag[mk].mean()) if mk.any() else float("nan")
            for k in ("hedge", "confident", "success"):
                mk = (A["group"] == "natural") & (A["kind"] == k)
                d[f"natural_{k}"] = float(flag[mk].mean())
            res[tname] = d
        cf_t, seed_t = (A["group"] == "cf") & (A["split"] == "test"), (A["group"] == "seed") & (A["split"] == "test")
        m2 = cf_t | seed_t
        res["auroc_cf_vs_seed_test"] = auroc(A["y"][m2], s[m2])
        nat = A["group"] == "natural"
        res["auroc_natural_test"] = auroc(A["y"][nat], s[nat])
        res["mean_score"] = {"cf_test": float(s[cf_t].mean()), "seed_test": float(s[seed_t].mean())}
        R[name] = res
    n = Counter((r["group"], r["split"], r["kind"]) for r in rows)
    return R, {"/".join(k): v for k, v in sorted(n.items())}


def markdown(R, counts):
    f = lambda x: "–" if x != x else f"{x:.3f}"  # noqa: E731
    L = ["# Do the probes detect the TF-IDF counterfactual hop-1 failures?", "",
         f"Rows: {counts}", "",
         "Hop-1 rows only. `cf` = cf_v1 counterfactuals (all hop-1 failures); `seed` = their clean seeds (hop 1 correct, "
         "same context). Main columns = TEST split (seeds never in any probe's training). Natural = TF-IDF test split.", "",
         "## At each probe's operating threshold (CV-F0.5; deployed: calibrated τ)", "",
         "| probe | layer | thr | CF flagged | hedge | other | copied | seed false alarm | AUROC CF vs seed | "
         "natural hedge recall | natural confident recall | natural false alarm | natural AUROC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in R.items():
        d = r["op"]
        L.append(f"| {name} | {r['layer']} | {r['thr']:.3f} | **{f(d['cf_test'])}** | {f(d['cf_test_hedge'])} | "
                 f"{f(d['cf_test_other_wrong_entity'])} | {f(d['cf_test_copied_seed_entity'])} | "
                 f"{f(d['seed_test_false_alarm'])} | {f(r['auroc_cf_vs_seed_test'])} | {f(d['natural_hedge'])} | "
                 f"{f(d['natural_confident'])} | {f(d['natural_success'])} | {f(r['auroc_natural_test'])} |")
    L += ["", "## At the default threshold (0.5 / SVM margin 0; deployed: raw P 0.5) - the threshold effect", "",
          "| probe | thr | CF flagged | hedge | other | copied | seed false alarm | natural hedge | natural confident | "
          "natural false alarm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in R.items():
        d = r["default"]
        L.append(f"| {name} | {r['thr_default']:.3f} | **{f(d['cf_test'])}** | {f(d['cf_test_hedge'])} | "
                 f"{f(d['cf_test_other_wrong_entity'])} | {f(d['cf_test_copied_seed_entity'])} | "
                 f"{f(d['seed_test_false_alarm'])} | {f(d['natural_hedge'])} | {f(d['natural_confident'])} | "
                 f"{f(d['natural_success'])} |")
    L += ["", "## Train-split CFs (seeds were in the probes' training; for reference)", "",
          "| probe | CF flagged (op) | seed false alarm (op) | CF flagged (default) |", "|---|---|---|---|"]
    for name, r in R.items():
        L.append(f"| {name} | {f(r['op']['cf_train'])} | {f(r['op']['seed_train_false_alarm'])} | "
                 f"{f(r['default']['cf_train'])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/probe_cf_eval")
    args = ap.parse_args()
    probes = {**deployed_probes(), **suite_probes()}
    print("probes:", {k: v[1]["layer"] for k, v in probes.items()})
    rows = build_rows()
    X = load_features(rows, sorted({m["layer"] for _, m in probes.values()}))
    R, counts = evaluate(rows, X, probes)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"counts": counts, "probes": R}, indent=1))
    md = markdown(R, counts)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
