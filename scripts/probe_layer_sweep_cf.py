"""
scripts/probe_layer_sweep_cf.py
───────────────────────────────
CPU: which layer (and which LR regularisation) detects the TF-IDF counterfactual
hop-1 failures best, compared at a MATCHED false-alarm rate?

probe_cf_eval.py compared the deployed probe (layer 10) with suite probes
(layer 14) at their own thresholds, which mixes the effect of layer, C and
threshold.  Here the same shared hop12 logistic regression (the deployed recipe,
src.pipeline.probes.make_probe) is trained on the natural train split at EVERY
layer, with C = 1 (deployed) and C = 0.01 (suite-tuned), and every probe is read
at the threshold that gives 5 % false alarms on natural test hop-1 successes.

Per layer and C:
  natural   test hop-1 AUROC, recall on natural hedges / confident failures @5% FPR
  cf        test-split cf_v1 hop-1 failures: AUROC CF vs their seeds, recall @5% FPR
            per failure kind (hedge / other_wrong_entity / copied_seed_entity)
  cf_train  the same recall on train-split CFs (independent check of the ranking)

Usage (Kaggle CPU; downloads ~2.5 GB of hidden states, ~20-30 min):
  python scripts/probe_layer_sweep_cf.py --out outputs/probe_layer_sweep_cf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_cf_eval import DEPLOYED, auroc, build_rows, load_features  # noqa: E402
from src.pipeline.common import download_hidden_states, load_setting_examples  # noqa: E402
from src.pipeline.probes import hop_rows, load_pooled, make_probe  # noqa: E402

N_LAYERS = 28
CS = {"C=1 (deployed)": 1.0, "C=0.01 (tuned)": 0.01}
FPR = 0.05


def fit(X, y, C):
    m = make_probe()
    m.set_params(logisticregression__C=C)
    return m.fit(X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/probe_layer_sweep_cf")
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()
    layers = list(range(N_LAYERS)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    # training rows: natural train split, hops 1+2 (exactly the deployed shared probe's rows)
    ex = load_setting_examples("tfidf")
    train_ids = set(json.loads((DEPLOYED / "probes_meta.json").read_text())["train_ids"])
    tr = hop_rows(ex, [i for i, e in enumerate(ex) if e["id"] in train_ids])
    hs_dir = download_hidden_states("tfidf", ids=sorted({r["id"] for r in tr}))
    Xtr = load_pooled(ex, tr, hs_dir, layers)                       # fp16 [n, len(layers), d]
    ytr = np.array([r["y"] for r in tr])
    print(f"train rows {len(tr)} (fail {ytr.sum()})", flush=True)

    # evaluation rows: CFs + seeds + natural test hop 1 (same rows as probe_cf_eval.py)
    rows = build_rows()
    Xev = load_features(rows, layers)
    g = np.array([r["group"] for r in rows]); k = np.array([r["kind"] for r in rows])
    sp = np.array([r["split"] for r in rows]); y = np.array([r["y"] for r in rows])
    nat, succ = g == "natural", (g == "natural") & (k == "success")
    cf_t, seed_t = (g == "cf") & (sp == "test"), (g == "seed") & (sp == "test")

    R = {}
    t0 = time.time()
    for li, L in enumerate(layers):
        for cname, C in CS.items():
            m = fit(Xtr[:, li].astype(np.float32), ytr, C)
            s = m.predict_proba(np.nan_to_num(Xev[L]))[:, 1]
            thr = float(np.quantile(s[succ], 1 - FPR))               # 5 % false alarms on natural successes
            flag = s > thr
            rec = lambda mask: float(flag[mask].mean()) if mask.any() else float("nan")  # noqa: E731
            r = {"natural_auroc": auroc(y[nat], s[nat]), "cf_vs_seed_auroc": auroc(y[cf_t | seed_t], s[cf_t | seed_t]),
                 "natural_hedge": rec(nat & (k == "hedge")), "natural_confident": rec(nat & (k == "confident")),
                 "seed_false_alarm": rec(seed_t), "fpr_natural": rec(succ)}
            for kind in ("hedge", "other_wrong_entity", "copied_seed_entity"):
                r[f"cf_{kind}"] = rec(cf_t & (k == kind))
                r[f"cf_train_{kind}"] = rec((g == "cf") & (sp == "train") & (k == kind))
            R.setdefault(cname, {})[L] = r
        print(f"layer {L} done ({time.time() - t0:.0f}s): " + "  ".join(
            f"{c}: nat {R[c][L]['natural_auroc']:.3f} cf-hedge {R[c][L]['cf_hedge']:.2f}" for c in CS), flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    f = lambda x: "–" if x != x else f"{x:.3f}"  # noqa: E731
    md = ["# Layer sweep: shared hop12 LR on natural train, read at 5 % false alarms on natural hop-1 successes", "",
          "CF columns = test-split cf_v1 counterfactuals (seeds never trained on); cf_train = train-split CFs.", ""]
    for cname in CS:
        md += [f"## {cname}", "",
               "| layer | natural AUROC | natural hedge | natural confident | AUROC CF vs seed | CF hedge | CF other | "
               "CF copied | seed false alarm | CF-train hedge |", "|---|---|---|---|---|---|---|---|---|---|"]
        for L, r in R[cname].items():
            md.append(f"| {L} | {f(r['natural_auroc'])} | {f(r['natural_hedge'])} | {f(r['natural_confident'])} | "
                      f"{f(r['cf_vs_seed_auroc'])} | **{f(r['cf_hedge'])}** | {f(r['cf_other_wrong_entity'])} | "
                      f"{f(r['cf_copied_seed_entity'])} | {f(r['seed_false_alarm'])} | {f(r['cf_train_hedge'])} |")
        md.append("")
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
