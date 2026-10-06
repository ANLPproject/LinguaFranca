"""
bootstrap_confident_wrong.py  (v2)
──────────────────────────────────
Split-stability check for the hop-1 probe: repeat the grouped 80/20 split with
several seeds, retrain, and report mean ± std of AUROC, recall on confident
(non-hedging) failures, recall on natural failures, and FPR.

This is "repeated random sub-sampling", not k-fold CV or a bootstrap; the
per-split grouped bootstrap CIs are in advanced_probing.py's results.

v1 reported only recall on confident failures (no FPR) — see FIXES_AND_RERUN.md.

Usage
─────
  python bootstrap_confident_wrong.py --hs-dir data/hidden_states --layers 8,9,10,11,12,13,14
"""

from __future__ import annotations

import argparse

import numpy as np

from lf_common import (DEFAULT_DATA, binary_metrics, canonical_split, hop_rows, load_examples, load_hidden,
                       make_probe, rows_arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_data", nargs="?", default=None)
    ap.add_argument("legacy_hs", nargs="?", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--hs-dir", default=None)
    ap.add_argument("--layers", default="8,9,10,11,12,13,14")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    data = args.data or args.legacy_data or DEFAULT_DATA
    hs_dir = args.hs_dir or args.legacy_hs or "data/hidden_states"

    examples = load_examples(data, hs_dir=hs_dir)
    rows = hop_rows(examples, hops=(1,))
    A = rows_arrays(rows)
    layers = [int(x) for x in args.layers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    X = load_hidden(examples, rows, hs_dir, layers)

    print(f"Hop-1 probe stability over {len(seeds)} grouped 80/20 splits (seeds {seeds})")
    print(f"{'layer':>5} | {'AUROC':>13} | {'conf-fail recall':>16} | {'natural recall':>14} | {'FPR':>13} | n conf/hedge (test, avg)")
    for L in layers:
        rec = {k: [] for k in ("auroc", "conf", "nat", "fpr", "n_conf", "n_hedge")}
        for s in seeds:
            tr_idx, _ = canonical_split(examples, seed=s)
            is_tr = np.zeros(len(examples), bool); is_tr[tr_idx] = True
            trm = is_tr[A["ei"]]; tem = ~trm
            clf = make_probe().fit(X[L][trm], A["y"][trm])
            p = clf.predict_proba(X[L][tem])[:, 1]
            y = A["y"][tem]; pred = p >= 0.5
            hedge = A["hedge"][tem]; cf = A["cf"][tem]
            m = binary_metrics(y, p)
            rec["auroc"].append(m["auroc"]); rec["fpr"].append(m["fpr"])
            rec["conf"].append(pred[(y == 1) & ~hedge].mean())
            rec["nat"].append(pred[(y == 1) & ~cf].mean())
            rec["n_conf"].append(((y == 1) & ~hedge).sum()); rec["n_hedge"].append(((y == 1) & hedge).sum())
        f = lambda k: f"{np.mean(rec[k]):.3f}±{np.std(rec[k]):.3f}"
        print(f"{L:>5} | {f('auroc'):>13} | {f('conf'):>16} | {f('nat'):>14} | {f('fpr'):>13} | "
              f"~{np.mean(rec['n_conf']):.0f} / {np.mean(rec['n_hedge']):.0f}")


if __name__ == "__main__":
    main()
