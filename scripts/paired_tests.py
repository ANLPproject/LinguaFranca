"""
scripts/paired_tests.py
───────────────────────
Paired bootstrap comparisons between the probes of the probe suite (kaggle/04_probe_suite.ipynb).

Why paired: two models scored on the same test rows make correlated errors, so comparing their separate confidence
intervals says little.  Here the SAME resampled questions (whole question groups, 2,000 resamples) are used for both
models and the DIFFERENCE is reported with its 95% interval.  If the interval excludes 0 the gap is supported.

Inputs (from probe_suite_results.zip): <suite>/<setting>/{cells.jsonl, rows.jsonl, selected.json}
The test split is recomputed with the project's canonical split; the recomputed AUROC of every model is checked
against the AUROC stored in cells.jsonl, so a misaligned row order would be caught.

    python scripts/paired_tests.py --suite outputs/probing/probe_suite
Writes <suite>/paired_tests.md and paired_tests.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

TASKS = {"hop1": (1,), "hop2": (2,), "hop12": (1, 2)}
PAIRS = [("lr", "lr_default"), ("svm", "lr"), ("mlp", "lr"), ("mlp", "lr_default"), ("mlp", "svm")]
NAMES = {"lr_default": "default LR", "lr": "tuned LR", "svm": "SVM", "mlp": "MLP"}
B = 2000


def auroc(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def recall_at_fpr(y, s, fpr=0.03):
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


def group_resamples(groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]


def summarize(d):
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi), float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def paired(y, sa, sb, resamples):
    out = {}
    for name, fn in (("auroc", auroc), ("recall@3%FPR", recall_at_fpr)):
        point = fn(y, sa) - fn(y, sb)
        boot = np.array([fn(y[i], sa[i]) - fn(y[i], sb[i]) for i in resamples])
        _, lo, hi, p = summarize(boot)
        out[name] = dict(a=float(fn(y, sa)), b=float(fn(y, sb)), diff=float(point), lo=lo, hi=hi, p=p)
    return out


def test_ids(setting):
    from src.pipeline.common import canonical_split, load_setting_examples
    ex = load_setting_examples(setting)
    _, te = canonical_split(ex)
    return {ex[i]["id"] for i in te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="outputs/probing/probe_suite")
    ap.add_argument("--fixed-layer", type=int, default=13)
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()
    suite = Path(args.suite)
    md = ["# Paired bootstrap tests between probes (test split)\n",
          f"Each row: difference in test AUROC (and recall at 3% false alarms) between two models on the SAME rows, "
          f"{args.boot} resamples of whole question groups, 95% interval of the difference. **Supported** = the interval "
          f"excludes 0. Many comparisons are listed, so treat a lone borderline interval with caution.\n"]
    store, data = {}, {}
    for setting in ("full10", "tfidf"):
        rows = [json.loads(l) for l in open(suite / setting / "rows.jsonl", encoding="utf-8")]
        sel = {(r["model"], r["task"]): r["layer"] for r in json.load(open(suite / setting / "selected.json"))}
        cells = {}
        for l in open(suite / setting / "cells.jsonl", encoding="utf-8"):
            c = json.loads(l)
            cells[(c["model"], c["task"], c["layer"])] = c
        tids = test_ids(setting)
        per_task = {}
        for task, hops in TASKS.items():
            sub = [r for r in rows if r["id"] in tids and r["hop"] in hops]
            y = np.array([r["y"] for r in sub])
            g = np.array([r["group"] for r in sub])
            n_cells = {len(c["test_scores"]) for (m, t, L), c in cells.items() if t == task}
            assert n_cells == {len(sub)}, f"{setting}/{task}: recomputed {len(sub)} test rows, cells have {n_cells}"
            per_task[task] = dict(y=y, g=g, ids=[(r["id"], r["hop"]) for r in sub], res=group_resamples(g, args.boot, 7))
        # alignment check: recomputed AUROC == stored AUROC for every cell
        # Scores in cells.jsonl were rounded to 5 decimals, so saturated probabilities (default LR, MLP) tie and the
        # recomputed AUROC can sit a little below the stored one.  Cells with few ties must match exactly; every cell
        # must match within 0.015 and must be far above the AUROC obtained with a shuffled row order.
        worst_clean, worst_tied, worst_cell = 0.0, 0.0, None
        rng = np.random.default_rng(0)
        for (m, t, L), c in cells.items():
            s = np.array(c["test_scores"])
            y_t = per_task[t]["y"]
            d = abs(auroc(y_t, s) - c["test"]["all"]["auroc"])
            tied = 1 - len(np.unique(s)) / len(s)
            if tied < 0.05:
                worst_clean = max(worst_clean, d)
            else:
                worst_tied = max(worst_tied, d)
            if d > worst_tied and tied >= 0.05:
                worst_cell = (m, t, L)
            if c["test"]["all"]["auroc"] > 0.7:
                assert auroc(y_t, rng.permutation(s)) < c["test"]["all"]["auroc"] - 0.15, f"{setting}/{m}/{t}/{L}: order looks wrong"
        assert worst_clean < 2e-3, f"{setting}: row alignment check failed on tie-free cells (max AUROC mismatch {worst_clean})"
        assert worst_tied < 0.015, f"{setting}: row alignment check failed (max AUROC mismatch {worst_tied})"
        print(f"[{setting}] alignment OK: tie-free cells max diff {worst_clean:.5f}; cells with rounding ties max diff {worst_tied:.5f}")
        md.append(f"## {setting}\n\nAlignment check: recomputed AUROC equals the stored AUROC for the {len(cells)} cells "
                  f"(tie-free cells: max difference {worst_clean:.5f}; default LR / MLP cells have rounding ties from the 5-decimal "
                  f"score storage: max difference {worst_tied:.4f}). Shuffling the row order drops every AUROC by more than 0.15, "
                  f"so rows are aligned.\n")
        exact_p = suite / setting / "exact_scores.npz"
        exact = dict(np.load(exact_p)) if exact_p.exists() else {}

        def scores(model, task, layer, variant, _c=cells, _sel=sel, _ex=exact):
            """Exact (unrounded) scores for the CV-selected layers when available, else the stored rounded scores."""
            if variant == "selected" and f"{model}/{task}" in _ex and _sel[(model, task)] == layer:
                return _ex[f"{model}/{task}"]
            return np.array(_c[(model, task, layer)]["test_scores"])

        data[setting] = dict(per_task=per_task, cells=cells, sel=sel, exact=exact, scores=scores)

        for variant in ("selected", f"layer {args.fixed_layer}"):
            note = (" (exact scores recomputed from the saved probes)" if variant == "selected" and exact else
                    " (stored scores rounded to 5 decimals: default-LR / MLP AUROC can be off by up to ~0.004)")
            md.append(f"### Models compared at their {'CV-selected layers' if variant == 'selected' else variant}{note}\n")
            md.append("| task | A vs B | layers | AUROC A / B | AUROC diff [95% CI] | supported? | recall@3%FPR diff [95% CI] | supported? |"
                      "\n|---|---|---|---|---|---|---|---|")
            for task in TASKS:
                pt = per_task[task]
                for a, b in PAIRS:
                    La = sel[(a, task)] if variant == "selected" else args.fixed_layer
                    Lb = sel[(b, task)] if variant == "selected" else args.fixed_layer
                    sa, sb = scores(a, task, La, variant), scores(b, task, Lb, variant)
                    r = paired(pt["y"], sa, sb, pt["res"])
                    store[f"{setting}/{variant}/{task}/{a}-vs-{b}"] = dict(layers=[La, Lb], **r)
                    A, R = r["auroc"], r["recall@3%FPR"]
                    sup = lambda x: "yes" if (x["lo"] > 0 or x["hi"] < 0) else "no"
                    md.append(f"| {task} | {NAMES[a]} vs {NAMES[b]} | {La} / {Lb} | {A['a']:.3f} / {A['b']:.3f} | "
                              f"{A['diff']:+.3f} [{A['lo']:+.3f}, {A['hi']:+.3f}] | {sup(A)} | "
                              f"{R['diff']:+.3f} [{R['lo']:+.3f}, {R['hi']:+.3f}] | {sup(R)} |")
            md.append("")

    # all-10 vs TF-IDF: the settings share the same 4,000 questions but the test splits differ -> grouped bootstrap
    # of each set independently (unpaired), and a paired version on the (question, hop) rows that are test rows in both
    md.append("## all-10 passages vs TF-IDF top-5 (same model, selected layers)\n")
    shared_note = {}
    for task in TASKS:
        fa, tf = data["full10"]["per_task"][task], data["tfidf"]["per_task"][task]
        shared_note[task] = len(set(fa["ids"]) & set(tf["ids"]))
    md.append("The two settings use the same 4,000 questions, but their test splits are different draws, so the comparison is "
              "**unpaired**: each side is resampled independently. Rows that are test rows in both settings: "
              + ", ".join(f"{t}: {n}" for t, n in shared_note.items()) + ".\n")
    md.append("| task | model | full10 AUROC | TF-IDF AUROC | diff (full10 - TF-IDF) [95% CI] | supported? |\n|---|---|---|---|---|---|")
    for task in TASKS:
        fa, tf = data["full10"]["per_task"][task], data["tfidf"]["per_task"][task]
        for m in ("lr_default", "lr", "svm", "mlp"):
            sa = data["full10"]["scores"](m, task, data["full10"]["sel"][(m, task)], "selected")
            sb = data["tfidf"]["scores"](m, task, data["tfidf"]["sel"][(m, task)], "selected")
            pa = np.array([auroc(fa["y"][i], sa[i]) for i in fa["res"]])
            pb = np.array([auroc(tf["y"][i], sb[i]) for i in tf["res"][:len(fa["res"])]])
            d = pa - pb
            _, lo, hi, _ = summarize(d)
            pt = auroc(fa["y"], sa) - auroc(tf["y"], sb)
            store[f"cross/{task}/{m}"] = dict(full10=float(auroc(fa["y"], sa)), tfidf=float(auroc(tf["y"], sb)), diff=float(pt), lo=lo, hi=hi)
            md.append(f"| {task} | {NAMES[m]} | {auroc(fa['y'], sa):.3f} | {auroc(tf['y'], sb):.3f} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                      f"{'yes' if (lo > 0 or hi < 0) else 'no'} |")
    (suite / "paired_tests.md").write_text("\n".join(md), encoding="utf-8")
    (suite / "paired_tests.json").write_text(json.dumps(store, indent=1), encoding="utf-8")
    print("wrote", suite / "paired_tests.md")


if __name__ == "__main__":
    main()
