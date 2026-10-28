"""
lf_probe_suite.py
─────────────────
One script that trains and evaluates hop-failure probes — logistic regression, linear SVM and MLP — on
EVERY layer, for the all-10-passage setting (`full10`, Phase 1/2) and the TF-IDF top-5 setting (`tfidf`),
with a single protocol and every metric computed in one place.

Protocol (identical to advanced_probing.py v2 so the numbers are comparable)
  * rows      hops 1 and 2 with a 0/1 label; counterfactual rows are included in train and test
  * split     lf_common.canonical_split (GroupShuffleSplit 80/20, seed 42, grouped by clean_pair_id)
  * features  mean-pooled hop states, layer L = output of block L
  * CV        5-fold GroupKFold on the TRAIN split only -> layer selection (AUROC on the natural rows)
  * test      fitted on the full train split, scored once on the held-out split; decision threshold 0.5
              (probability models) or 0 (SVM margin)
  * the logistic regression `lr_default` is lf_common.make_probe() unchanged  => must reproduce the old numbers
New here
  * lr (tuned C), svm (tuned C), mlp (tuned) — hyper-parameters chosen by train-split CV at --tune-layers,
    never on test
  * rows whose hop was never generated (labelled, but no span in the hidden-state file) are dropped and
    counted instead of being scored on a zero vector
  * extra metrics for every cell: precision, F1(fail), recall at 3% / 5% false alarms, confident-only
    (hedges removed) AUROC / recall, natural vs counterfactual strata
  * selected-layer report: grouped bootstrap CIs, shuffled-label control, operating point at the CV-F0.5
    threshold, text / hop-index / majority baselines on the same rows
  * results are appended to cells.jsonl after every cell, so a timed-out session resumes where it stopped

Usage (Kaggle CPU; ~2-3 h for everything):
    python lf_probe_suite.py --setting full10,tfidf --out /kaggle/working/probe_suite
Quick check (~10 min):
    python lf_probe_suite.py --setting full10 --fast --out /kaggle/working/probe_suite_fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lf_common import binary_metrics, canonical_split, group_id, grouped_bootstrap_ci, make_probe   # noqa: E402

GOLD_HOPS = (1, 2)
TASKS = {"hop1": (1,), "hop2": (2,), "hop12": (1, 2)}
HEDGE_KEYWORDS = ["no information", "no mention", "not mentioned", "no relevant information",
                  "not found", "cannot find", "does not mention"]
ALL_MODELS = ["lr_default", "lr", "svm", "mlp"]
GRIDS = {
    "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0)],
    "svm": [dict(C=c) for c in (1e-4, 1e-3, 1e-2, 1e-1)],
    "mlp": [dict(hidden=h, alpha=a) for h in (64, 256) for a in (1e-2, 1e-1)],
}
FAST_GRIDS = {"lr": [dict(C=0.01), dict(C=1.0)], "svm": [dict(C=1e-3), dict(C=1e-2)],
              "mlp": [dict(hidden=64, alpha=1e-2), dict(hidden=64, alpha=1e-1)]}
KIND = {"lr_default": "lr", "lr": "lr", "svm": "svm", "mlp": "mlp"}


def is_hedge(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in HEDGE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Models: fit(X, y) / score(X) (higher = more likely a failing hop), thr = decision threshold
# ─────────────────────────────────────────────────────────────────────────────

class Model:
    def __init__(self, kind: str, params: dict, seed: int = 42, mlp_epochs: int = 25):
        self.kind, self.p, self.seed, self.mlp_epochs = kind, dict(params), seed, mlp_epochs
        self.thr = 0.0 if kind == "svm" else 0.5

    def fit(self, X, y):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "lr":
                if self.p.get("C", 1.0) == 1.0:
                    self.m = make_probe(self.seed)                       # == the old probe, bit for bit
                else:
                    from sklearn.linear_model import LogisticRegression
                    self.m = make_pipeline(StandardScaler(), LogisticRegression(
                        C=self.p["C"], max_iter=1000, class_weight="balanced", random_state=self.seed))
                self.m.fit(X, y)
            elif self.kind == "svm":
                from sklearn.svm import LinearSVC
                self.m = make_pipeline(StandardScaler(), LinearSVC(
                    C=self.p["C"], class_weight="balanced", max_iter=20000, tol=1e-3, random_state=self.seed))
                self.m.fit(X, y)
            else:
                from sklearn.neural_network import MLPClassifier
                self.sc = StandardScaler().fit(X)
                Z = self.sc.transform(X)
                rng = np.random.RandomState(self.seed)
                pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                idx = np.arange(len(y))
                if len(pos) and len(neg):                                # balance classes by oversampling
                    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
                    idx = np.concatenate([idx, rng.choice(small, len(big) - len(small), replace=True)])
                rng.shuffle(idx)
                self.m = MLPClassifier(hidden_layer_sizes=(self.p["hidden"],), alpha=self.p["alpha"],
                                       max_iter=self.mlp_epochs, batch_size=128, learning_rate_init=1e-3,
                                       random_state=self.seed)
                self.m.fit(Z[idx], np.asarray(y)[idx])
        return self

    def score(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        if self.kind == "lr":
            return self.m.predict_proba(X)[:, 1]
        if self.kind == "svm":
            return self.m.decision_function(X)
        return self.m.predict_proba(self.sc.transform(X))[:, 1]


def oof_scores(kind, params, X, y, groups, n_jobs, mlp_epochs):
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    def one(tr, te):
        if len(set(y[tr].tolist())) < 2:
            return te, np.full(len(te), np.nan)
        return te, Model(kind, params, mlp_epochs=mlp_epochs).fit(X[tr], y[tr]).score(X[te])

    out = np.full(len(y), np.nan)
    splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    for te, s in Parallel(n_jobs=n_jobs)(delayed(one)(tr, te) for tr, te in splits):
        out[te] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_fpr(y, s, target):
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= target].max())


def metrics(y, s, thr, extra=True):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    m = binary_metrics(y, prob=s, pred=(s >= thr).astype(int))
    if extra and len(y):
        pred = s >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        tpr = m.get("tpr", float("nan"))
        m["precision"] = prec
        m["f1_fail"] = 2 * prec * tpr / (prec + tpr) if prec == prec and tpr == tpr and prec + tpr else float("nan")
        m["recall_at_fpr3"], m["recall_at_fpr5"] = recall_at_fpr(y, s, 0.03), recall_at_fpr(y, s, 0.05)
    return m


def strata(A, base):
    return {"all": base, "natural": base & ~A["cf"], "cf": base & A["cf"]}


# ─────────────────────────────────────────────────────────────────────────────
# Rows and features
# ─────────────────────────────────────────────────────────────────────────────

def build_rows(examples):
    rows = []
    for ei, e in enumerate(examples):
        order = [h["hop_idx"] for h in e.get("hop_spans", [])]
        for h in e.get("hops", []):
            k = h.get("hop_idx")
            if k not in GOLD_HOPS or h.get("label") not in (0, 1):
                continue
            rows.append(dict(ei=ei, id=e["id"], hop=k, y=int(h["label"]), group=group_id(e),
                             cf=bool(e.get("is_counterfactual", False)), text=h.get("text", "") or "",
                             hedge=is_hedge(h.get("text", "")), order=order,
                             final_correct=bool(e.get("final_answer_correct", False))))
    return rows


def load_features(examples, rows, hs_dir, legacy=False):
    """X [n_kept_rows, n_layers, d] and the kept rows.  A labelled hop with no hidden-state column is dropped.
    legacy=True reproduces the old scripts: every labelled hop is kept, its column is hop_idx-1, and a missing
    column stays an all-zero vector (this only exists to verify that the old numbers can be reproduced)."""
    import torch
    hs_dir = Path(hs_dir)
    by_ex: dict[int, list[int]] = {}
    for ri, r in enumerate(rows):
        by_ex.setdefault(r["ei"], []).append(ri)
    X, keep, dropped = None, np.zeros(len(rows), bool), {"no_span": 0, "no_column": 0}
    for n, (ei, ris) in enumerate(by_ex.items()):
        d = torch.load(hs_dir / f"{examples[ei]['id']}.pt", map_location="cpu", weights_only=False)
        pooled = d["pooled"]
        if X is None:
            dt = np.float16 if pooled.dtype == torch.float16 else np.float32
            X = np.zeros((len(rows), pooled.shape[0], pooled.shape[-1]), dtype=dt)
        for ri in ris:
            r = rows[ri]
            if legacy:
                pos = r["hop"] - 1
                keep[ri] = True
                if pos < pooled.shape[1]:
                    X[ri] = pooled[:, pos].to(torch.float16 if dt == np.float16 else torch.float32).numpy()
                continue
            if r["order"]:
                if r["hop"] not in r["order"]:
                    dropped["no_span"] += 1
                    continue
                pos = r["order"].index(r["hop"])
            else:
                pos = r["hop"] - 1
            if pos >= pooled.shape[1]:
                dropped["no_column"] += 1
                continue
            X[ri] = pooled[:, pos].to(torch.float16 if dt == np.float16 else torch.float32).numpy()
            keep[ri] = True
        if n % 500 == 0:
            print(f"  loaded {n}/{len(by_ex)} hidden-state files", flush=True)
    return np.nan_to_num(X[keep]), [r for r, k in zip(rows, keep) if k], dropped


def arrays(rows):
    A = {k: np.array([r[k] for r in rows]) for k in ("ei", "hop", "y", "cf", "hedge", "final_correct")}
    A["group"] = np.array([r["group"] for r in rows])
    return A


# ─────────────────────────────────────────────────────────────────────────────
# One setting
# ─────────────────────────────────────────────────────────────────────────────

def jdump(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_setting(setting, examples, hs_dir, args, out_root):
    out = Path(out_root) / setting
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr_idx, te_idx = canonical_split(examples)
    rows0 = build_rows(examples)
    X, rows, dropped = load_features(examples, rows0, hs_dir, legacy=getattr(args, "legacy_rows", False))
    A = arrays(rows)
    n_layers = X.shape[1]
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    is_tr_ex = np.zeros(len(examples), bool)
    is_tr_ex[tr_idx] = True
    tr_rows = is_tr_ex[A["ei"]]
    te_rows = ~tr_rows
    assert not (set(A["group"][tr_rows]) & set(A["group"][te_rows])), "train/test groups overlap"
    masks = {t: (tr_rows & np.isin(A["hop"], h), te_rows & np.isin(A["hop"], h)) for t, h in TASKS.items()}
    info = dict(setting=setting, legacy_rows=bool(getattr(args, "legacy_rows", False)), n_examples=len(examples), n_train_examples=int(len(tr_idx)),
                n_test_examples=int(len(te_idx)), rows_labelled=len(rows0), rows_used=len(rows), dropped=dropped,
                d=int(X.shape[-1]), n_layers=n_layers, tasks={})
    print("=" * 100)
    print(f"[{setting}] examples {len(examples)} (train {len(tr_idx)} / test {len(te_idx)}), labelled hop rows "
          f"{len(rows0)}, used {len(rows)}, dropped {dropped}")
    for t, (trm, tem) in masks.items():
        info["tasks"][t] = dict(n_train=int(trm.sum()), n_test=int(tem.sum()), fail_train=int(A["y"][trm].sum()),
                                fail_test=int(A["y"][tem].sum()), cf_test=int(A["cf"][tem].sum()),
                                hedge_fail_test=int((A["hedge"] & (A["y"] == 1))[tem].sum()))
        print(f"  {t:5s} train n={trm.sum():5d} fail={A['y'][trm].sum():5d} | test n={tem.sum():5d} "
              f"fail={A['y'][tem].sum():5d} ({100 * A['y'][tem].mean():.1f}%) cf={A['cf'][tem].sum()}")
    (out / "info.json").write_text(json.dumps(info, indent=1, default=jdump))
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "hop", "y", "cf", "hedge", "group")}) + "\n")

    n_jobs = args.jobs or min(5, os.cpu_count() or 1)
    grids = FAST_GRIDS if args.fast else GRIDS

    # ── hyper-parameter tuning (train-split CV, at the tuning layers only) ──────────────────────────
    tune_path = out / "tuning.json"
    tuning = json.loads(tune_path.read_text()) if tune_path.exists() else {}
    tune_layers = [L for L in (int(x) for x in args.tune_layers.split(",")) if L < n_layers] or [layers[0]]
    for mname in [m for m in args.models if m != "lr_default"]:
        for task, (trm, _) in masks.items():
            key = f"{mname}/{task}"
            if key in tuning:
                continue
            best, board = None, []
            for hp in grids[mname]:
                aucs = []
                for L in tune_layers:
                    p = oof_scores(KIND[mname], hp, X[trm][:, L].astype(np.float32), A["y"][trm], A["group"][trm],
                                   n_jobs, args.mlp_epochs)
                    nat = ~A["cf"][trm]
                    aucs.append(metrics(A["y"][trm][nat], p[nat], 0.0, extra=False).get("auroc", float("nan")))
                board.append((float(np.nanmean(aucs)), hp))
                print(f"  [tune {setting}/{key}] {hp} -> CV AUROC {board[-1][0]:.4f} (layers {tune_layers})", flush=True)
            best = max(board, key=lambda b: b[0])
            tuning[key] = dict(params=best[1], cv_auroc=best[0], board=board, layers=tune_layers)
            tune_path.write_text(json.dumps(tuning, indent=1, default=jdump))

    def hp_of(mname, task):
        return {} if mname == "lr_default" else tuning[f"{mname}/{task}"]["params"]

    # ── all cells: model x task x layer ───────────────────────────────────────────────────────────────
    cells_path = out / "cells.jsonl"
    done = set()
    cells = []
    if cells_path.exists():
        for l in open(cells_path, encoding="utf-8"):
            c = json.loads(l)
            done.add((c["model"], c["task"], c["layer"]))
            cells.append(c)
    for L in layers:
        XL = X[:, L].astype(np.float32)
        for mname in args.models:
            for task, (trm, tem) in masks.items():
                if (mname, task, L) in done:
                    continue
                hp = hp_of(mname, task)
                kind = KIND[mname]
                ytr, gtr = A["y"][trm], A["group"][trm]
                oof = oof_scores(kind, hp, XL[trm], ytr, gtr, n_jobs, args.mlp_epochs)
                mdl = Model(kind, hp, mlp_epochs=args.mlp_epochs).fit(XL[trm], ytr)
                s_te = mdl.score(XL[tem])
                thr = mdl.thr
                Atr = {k: v[trm] for k, v in A.items()}
                Ate = {k: v[tem] for k, v in A.items()}
                cv = {s: metrics(ytr[m], oof[m], thr, extra=False) for s, m in strata(Atr, np.ones(len(ytr), bool)).items()
                      if m.sum()}
                test = {s: metrics(Ate["y"][m], s_te[m], thr) for s, m in strata(Ate, np.ones(len(s_te), bool)).items()
                        if m.sum()}
                conf = ~(Ate["hedge"] & (Ate["y"] == 1))
                test["confident_only"] = metrics(Ate["y"][conf], s_te[conf], thr)
                c = dict(setting=setting, model=mname, task=task, layer=L, params=hp, thr=thr, cv=cv, test=test,
                         oof=[round(float(x), 5) for x in oof], test_scores=[round(float(x), 5) for x in s_te])
                with open(cells_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(c, default=jdump) + "\n")
                cells.append(c)
                done.add((mname, task, L))
        print(f"[{setting}] layer {L} done ({time.time() - t0:.0f}s)", flush=True)
        del XL
    return dict(out=out, info=info, rows=rows, A=A, X=X, masks=masks, cells=cells, tuning=tuning, layers=layers)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def best_f05(y, s):
    y = np.asarray(y).astype(int)
    best_t, best_f = float(np.min(s)), -1.0
    for t in np.unique(s):
        pred = s >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
        if f > best_f or (f == best_f and t > best_t):
            best_t, best_f = float(t), f
    return best_t


def baselines(A, rows, tr_rows, te_rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    txt = np.array([r["text"] for r in rows], dtype=object)
    out = {}
    for task, hops in TASKS.items():
        trm, tem = tr_rows & np.isin(A["hop"], hops), te_rows & np.isin(A["hop"], hops)
        ytr, yte = A["y"][trm], A["y"][tem]
        res = {"majority": metrics(yte, np.full(len(yte), float(round(ytr.mean()))), 0.5, extra=False),
               "hedge_keyword_rule": metrics(yte, A["hedge"][tem].astype(float), 0.5, extra=False)}
        tf = TfidfVectorizer(max_features=1000)
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(tf.fit_transform(txt[trm]), ytr)
        res["tfidf_text"] = metrics(yte, c.predict_proba(tf.transform(txt[tem]))[:, 1], 0.5, extra=False)
        if len(hops) > 1:
            oh = np.eye(3)[A["hop"]]
            c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(oh[trm], ytr)
            res["hop_index_only"] = metrics(yte, c.predict_proba(oh[tem])[:, 1], 0.5, extra=False)
        out[task] = res
    return out


def f(x, n=3):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def report(R, args, old_json=None):
    out, A, X, masks, cells, tuning, layers = R["out"], R["A"], R["X"], R["masks"], R["cells"], R["tuning"], R["layers"]
    setting, info = out.name, R["info"]
    md = [f"# Probe suite — setting `{setting}`"
          + (" (LEGACY ROW HANDLING: reproduces the old scripts, not for reporting)" if info.get("legacy_rows") else "")
          + "\n",
          f"Examples {info['n_examples']} (train {info['n_train_examples']} / test {info['n_test_examples']}); "
          f"labelled hop rows {info['rows_labelled']}, used {info['rows_used']}, dropped (hop never generated) "
          f"{info['dropped']}. Layer L = output of block L. Test metrics at the model's default threshold "
          f"(0.5 for probability models, 0 for the SVM margin); CV on the train split only.\n"]
    tr_rows = np.zeros(len(A["y"]), bool)
    for t in TASKS:
        tr_rows |= masks[t][0]
    te_rows = ~tr_rows
    by = {(c["model"], c["task"], c["layer"]): c for c in cells}
    models = [m for m in args.models if any(k[0] == m for k in by)]

    # tuning
    md.append("## Tuned hyper-parameters (train-split CV AUROC at layers " + args.tune_layers + ")\n")
    md.append("| model | task | chosen | CV AUROC |\n|---|---|---|---|")
    for k, v in sorted(tuning.items()):
        md.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | {v['params']} | {v['cv_auroc']:.4f} |")

    # baselines
    base = baselines(A, R["rows"], tr_rows, te_rows)
    md.append("\n## Baselines on the same test rows (no hidden states)\n")
    md.append("| task | baseline | acc | bal. acc | AUROC | TPR | FPR | majority |\n|---|---|---|---|---|---|---|---|")
    for t, res in base.items():
        for n, x in res.items():
            md.append(f"| {t} | {n} | {f(x.get('acc'))} | {f(x.get('bacc'))} | {f(x.get('auroc'))} | {f(x.get('tpr'))} | "
                      f"{f(x.get('fpr'))} | {f(x.get('majority'))} |")

    # per-layer tables
    for task in TASKS:
        md.append(f"\n## {task}: test AUROC / CV AUROC by layer\n")
        md.append("| layer | " + " | ".join(f"{m} test | {m} CV" for m in models) + " |")
        md.append("|---|" + "---|---|" * len(models))
        for L in layers:
            cs = [by.get((m, task, L)) for m in models]
            md.append(f"| {L} | " + " | ".join(
                (f"{f(c['test']['all'].get('auroc'))} | {f(c['cv'].get('natural', c['cv']['all']).get('auroc'))}" if c else "– | –")
                for c in cs) + " |")

    # selected layers
    md.append("\n## Selected layer per model (max CV AUROC on natural train rows) and held-out test results\n")
    md.append("| task | model | layer | acc | bal. acc | AUROC [95% CI] | TPR | FPR | prec. | F1(fail) | recall@3%FPR | "
              "recall@5%FPR | conf. AUROC | conf. recall | shuffled AUROC (mean of 5) |\n|" + "---|" * 15)
    selected, final_rows = {}, []
    for task in TASKS:
        trm, tem = masks[task]
        for m in models:
            cand = [(L, by[(m, task, L)]) for L in layers if (m, task, L) in by]
            if not cand:
                continue
            L, c = max(cand, key=lambda x: x[1]["cv"].get("natural", x[1]["cv"]["all"]).get("auroc", -1))
            selected[f"{m}/{task}"] = L
            s_te = np.array(c["test_scores"])
            yte = A["y"][tem]
            ci = grouped_bootstrap_ci(yte, s_te, A["group"][tem], "auroc", n_boot=args.n_boot)
            # shuffled-label control at the selected layer
            kind, hp = KIND[m], c["params"]
            shuf_runs = []
            for seed in range(5):                                          # mean of 5 permutations (one is noisy)
                ys = np.random.RandomState(seed).permutation(A["y"][trm])
                ctrl = Model(kind, hp, mlp_epochs=args.mlp_epochs).fit(X[trm][:, L].astype(np.float32), ys)
                shuf_runs.append(metrics(yte, ctrl.score(X[tem][:, L].astype(np.float32)), ctrl.thr,
                                         extra=False).get("auroc"))
            shuf = float(np.nanmean(shuf_runs))
            # operating point at the CV-F0.5 threshold (OOF scores of the train split)
            oof = np.array(c["oof"])
            tau = best_f05(A["y"][trm], oof)
            at_tau = metrics(yte, s_te, tau)
            t = c["test"]["all"]
            tc = c["test"]["confident_only"]
            md.append(f"| {task} | {m} | {L} | {f(t['acc'])} | {f(t['bacc'])} | {f(t['auroc'])} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                      f"{f(t['tpr'])} | {f(t['fpr'])} | {f(t.get('precision'))} | {f(t.get('f1_fail'))} | "
                      f"{f(t.get('recall_at_fpr3'))} | {f(t.get('recall_at_fpr5'))} | {f(tc.get('auroc'))} | "
                      f"{f(tc.get('tpr'))} | {f(shuf)} |")
            final_rows.append(dict(setting=setting, task=task, model=m, layer=L, tau_cv_f05=tau, auroc_ci95=ci,
                                   shuffled_auroc=shuf, test=c["test"], at_tau=at_tau, params=hp))
    md.append("\n### Operating point at the CV-F0.5 threshold (selected layers)\n")
    md.append("| task | model | threshold | acc | TPR | FPR | precision | F1(fail) |\n|---|---|---|---|---|---|---|---|")
    for r in final_rows:
        a = r["at_tau"]
        md.append(f"| {r['task']} | {r['model']} | {r['tau_cv_f05']:.3f} | {f(a.get('acc'))} | {f(a.get('tpr'))} | "
                  f"{f(a.get('fpr'))} | {f(a.get('precision'))} | {f(a.get('f1_fail'))} |")

    # strata at selected layers
    md.append("\n### Natural vs counterfactual (selected layers, AUROC)\n")
    md.append("| task | model | all | natural | cf |\n|---|---|---|---|---|")
    for r in final_rows:
        tt = r["test"]
        md.append(f"| {r['task']} | {r['model']} | {f(tt['all'].get('auroc'))} | {f(tt.get('natural', {}).get('auroc'))} | "
                  f"{f(tt.get('cf', {}).get('auroc'))} |")

    # old-protocol check
    if old_json and setting == "full10" and "lr_default" in models and Path(old_json).exists():
        old = json.loads(Path(old_json).read_text())
        md.append("\n## Reproduction check vs the previous run (`lr_default`, test, all rows)\n")
        md.append("| task | layer | n new / old | AUROC new / old | acc new / old | max abs diff |\n|---|---|---|---|---|---|")
        worst = 0.0
        for task in TASKS:
            for L in layers:
                c, o = by.get(("lr_default", task, L)), old.get("layers", {}).get(str(L), {}).get(task, {}).get("test", {}).get("all")
                if not c or not o:
                    continue
                t = c["test"]["all"]
                d = max(abs(t["auroc"] - o["auroc"]), abs(t["acc"] - o["acc"]))
                worst = max(worst, d)
                md.append(f"| {task} | {L} | {t['n']} / {o['n']} | {t['auroc']:.4f} / {o['auroc']:.4f} | "
                          f"{t['acc']:.4f} / {o['acc']:.4f} | {d:.4f} |")
        md.append(f"\n**Largest difference: {worst:.4f}** ({'MATCH' if worst < 0.005 else 'DIFFERENT — read the row counts above'}).")
        print(f"[{setting}] reproduction check vs old run: largest |diff| = {worst:.4f}")

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "selected.json").write_text(json.dumps(final_rows, indent=1, default=jdump))
    flat = ["setting,model,task,layer,split,n,acc,bacc,auroc,tpr,fpr,precision,f1_fail,recall_at_fpr3,recall_at_fpr5,cv_auroc_natural"]
    for c in cells:
        cvn = c["cv"].get("natural", c["cv"]["all"]).get("auroc")
        for sp in ("all", "natural", "cf", "confident_only"):
            t = c["test"].get(sp)
            if t:
                flat.append(",".join(str(x) for x in (c["setting"], c["model"], c["task"], c["layer"], sp, t.get("n"),
                                                       t.get("acc"), t.get("bacc"), t.get("auroc"), t.get("tpr"),
                                                       t.get("fpr"), t.get("precision"), t.get("f1_fail"),
                                                       t.get("recall_at_fpr3"), t.get("recall_at_fpr5"), cvn)))
    (out / "cells.csv").write_text("\n".join(flat), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
        for ax, task in zip(axs, TASKS):
            for m in models:
                xs = [L for L in layers if (m, task, L) in by]
                ax.plot(xs, [by[(m, task, L)]["test"]["all"].get("auroc") for L in xs], marker="o", ms=3, label=m)
            ax.set_title(f"{setting}: {task}")
            ax.set_xlabel("layer")
            ax.grid(alpha=.3)
        axs[0].set_ylabel("test AUROC")
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(out / "auroc_by_layer.png", dpi=130)
    except Exception as e:                                                 # noqa: BLE001
        print("[plot skipped]", e)
    print(f"[{setting}] wrote {out}/summary.md, selected.json, cells.csv")
    return final_rows


# ─────────────────────────────────────────────────────────────────────────────

def load_setting(setting, args):
    """(examples, hs_dir) for `full10` or `tfidf`, from the Hugging Face repo (or local overrides)."""
    from src.pipeline.common import download_hidden_states, hidden_state_ids, load_setting_examples
    hs_dir = getattr(args, f"hs_dir_{setting}") or str(download_hidden_states(setting))
    if setting == "full10" and args.full10_data:
        from lf_common import load_examples
        return load_examples(args.full10_data, hs_dir=hs_dir), hs_dir
    ids = {p.stem for p in Path(hs_dir).glob("*.pt")}
    return load_setting_examples(setting, ids), hs_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="full10,tfidf", help="comma list of: full10, tfidf")
    ap.add_argument("--out", default="outputs/probe_suite")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--layers", default="all", help="'all' or e.g. 8,10,13")
    ap.add_argument("--tune-layers", default="10,13")
    ap.add_argument("--jobs", type=int, default=0, help="parallel CV folds (default min(5, cpus))")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--mlp-epochs", type=int, default=25)
    ap.add_argument("--fast", action="store_true", help="2-point grids, layers 8,10,13, 200 bootstraps")
    ap.add_argument("--legacy-rows", action="store_true",
                    help="keep every labelled hop and zero-fill missing columns, exactly like the old scripts "
                         "(reproduction check only, not for reporting)")
    ap.add_argument("--old-json", default="outputs/probing/results.json", help="previous results for the reproduction check")
    ap.add_argument("--full10-data", default=None, help="canonical augmented_v2.jsonl (default: rebuilt from the HF revision)")
    ap.add_argument("--hs-dir-full10", default=None)
    ap.add_argument("--hs-dir-tfidf", default=None)
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    assert all(m in ALL_MODELS for m in args.models), f"models must be among {ALL_MODELS}"
    if args.fast:
        args.layers = "8,10,13" if args.layers == "all" else args.layers
        args.n_boot = min(args.n_boot, 200)
    for setting in [s.strip() for s in args.setting.split(",") if s.strip()]:
        examples, hs_dir = load_setting(setting, args)
        R = run_setting(setting, examples, hs_dir, args, args.out)
        report(R, args, args.old_json)
    print("[done]")


if __name__ == "__main__":
    main()
