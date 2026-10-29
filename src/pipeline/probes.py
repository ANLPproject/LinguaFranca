"""
src/pipeline/probes.py
──────────────────────
Train the probes the live system uses and calibrate their trigger thresholds.

  hop1  — StandardScaler + LogisticRegression on mean-pooled hop-1 states
  hop2  — same, hop-2 states
  hop12 — one probe on hop-1 and hop-2 rows together (ablation)

Thresholds τ maximise F0.5 on out-of-fold predictions from grouped 5-fold CV
inside the train split, so neither the layer nor τ ever sees the test split.

Usage (CPU is fine; downloads ~2 GB of hidden states):
    python -m src.pipeline.probes --setting tfidf --out outputs/pipeline/tfidf/probes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.pipeline.common import (GOLD_HOPS, best_f05_threshold, canonical_split, detection_metrics,
                                 download_hidden_states, group_id, hidden_state_ids, is_hedge,
                                 load_setting_examples)

N_LAYERS = 28  # Llama-3.2-3B


def make_probe(seed: int = 42, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed))


# ─────────────────────────────────────────────────────────────────────────────
# Rows and features
# ─────────────────────────────────────────────────────────────────────────────

def hop_rows(examples: list[dict], idx) -> list[dict]:
    """One row per labeled gold hop that the model actually generated."""
    rows = []
    for ei in idx:
        e = examples[ei]
        order = [h["hop_idx"] for h in e.get("hop_spans", [])]
        for h in e.get("hops", []):
            k = h.get("hop_idx")
            if k not in GOLD_HOPS or h.get("label") not in (0, 1) or k not in order:
                continue
            rows.append(dict(ei=int(ei), id=e["id"], hop=k, pos=order.index(k), y=int(h["label"]),
                             group=group_id(e), cf=bool(e.get("is_counterfactual", False)),
                             hedge=is_hedge(h.get("text", ""))))
    return rows


def load_pooled(examples: list[dict], rows: list[dict], hs_dir, layers) -> np.ndarray:
    """float16 [n_rows, len(layers), d]; each .pt is read once."""
    import torch
    hs_dir = Path(hs_dir)
    by_ex: dict[int, list[int]] = {}
    for ri, r in enumerate(rows):
        by_ex.setdefault(r["ei"], []).append(ri)
    X = None
    for n, (ei, ris) in enumerate(by_ex.items()):
        d = torch.load(hs_dir / f"{examples[ei]['id']}.pt", map_location="cpu", weights_only=False)
        pooled = d["pooled"]
        li = list(d.get("layer_indices", range(pooled.shape[0])))
        if X is None:
            X = np.zeros((len(rows), len(layers), pooled.shape[-1]), dtype=np.float16)
        sel = [li.index(L) for L in layers]
        for ri in ris:
            X[ri] = pooled[sel, rows[ri]["pos"]].to(torch.float16).numpy()
        if n % 500 == 0:
            print(f"  loaded {n}/{len(by_ex)} hidden-state files", flush=True)
    return np.nan_to_num(X)


def oof_probs(X, y, groups, n_folds: int = 5, n_jobs: int | None = None, C: float = 1.0) -> np.ndarray:
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    def fold(tr, te):
        m = make_probe(C=C).fit(X[tr], y[tr])
        return te, m.predict_proba(X[te])[:, 1]

    out = np.full(len(y), np.nan)
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    for te, p in Parallel(n_jobs=n_jobs or n_folds)(delayed(fold)(tr, te) for tr, te in splits):
        out[te] = p
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_platt(y, p) -> dict:
    """1-D logistic regression on logit(p): calibrated = sigmoid(a * logit(p) + b), with a > 0."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], np.asarray(y).astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    assert a > 0, "Platt slope must be positive (calibration must be monotone)"
    return dict(a=a, b=b)


def apply_platt(calib: dict | None, p):
    if not calib:
        return p
    out = 1.0 / (1.0 + np.exp(-(calib["a"] * _logit(p) + calib["b"])))
    return float(out) if np.ndim(out) == 0 else out


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def _bootstrap_auroc_ci(y, p, groups, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(set(y[idx].tolist())) > 1:
            vals.append(_auroc(y[idx], p[idx]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [float("nan")] * 2


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_probes(setting: str, out_dir, hs_dir=None, hop1_layer="10", hop2_layer="10", hop12_layer="10",
                 sweep="all", n_folds: int = 5, C: float = 1.0) -> dict:
    import joblib
    import sklearn

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hs_ids = hidden_state_ids(setting)
    examples = load_setting_examples(setting, hs_ids)
    tr, te = canonical_split(examples)
    print(f"[probes] setting={setting}: {len(examples)} examples with hidden states "
          f"(train {len(tr)} / test {len(te)})")
    if hs_dir is None:
        hs_dir = download_hidden_states(setting)

    fixed = {k: (None if str(v) == "cv" else int(v)) for k, v in
             (("hop1", hop1_layer), ("hop2", hop2_layer), ("hop12", hop12_layer))}
    if sweep == "all":
        layers = list(range(N_LAYERS))
    elif sweep == "none":
        layers = sorted({v for v in fixed.values() if v is not None} | {10, 13})
    else:
        layers = sorted({int(x) for x in str(sweep).split(",")} | {v for v in fixed.values() if v is not None})

    rows_tr, rows_te = hop_rows(examples, tr), hop_rows(examples, te)
    print(f"[probes] loading layers {layers} for {len(rows_tr)} train / {len(rows_te)} test hop rows")
    X_tr, X_te = load_pooled(examples, rows_tr, hs_dir, layers), load_pooled(examples, rows_te, hs_dir, layers)

    def arr(rows, key):
        return np.array([r[key] for r in rows])

    tasks = {"hop1": (1,), "hop2": (2,), "hop12": (1, 2)}
    cv_table = {t: {} for t in tasks}
    oof_cache = {}
    for t, hops in tasks.items():
        m_tr = np.isin(arr(rows_tr, "hop"), hops)
        y, g = arr(rows_tr, "y")[m_tr], arr(rows_tr, "group")[m_tr]
        for li, L in enumerate(layers):
            if fixed[t] is not None and L != fixed[t] and sweep == "none":
                continue
            p = oof_probs(X_tr[m_tr, li].astype(np.float32), y, g, n_folds, C=C)
            cv_table[t][L] = _auroc(y, p)
            oof_cache[(t, L)] = p
        print(f"[probes] {t} CV AUROC by layer: " +
              " ".join(f"{L}:{a:.3f}" for L, a in cv_table[t].items()), flush=True)

    meta = dict(setting=setting, C=C, n_examples=len(examples), n_train=len(tr), n_test=len(te),
                train_ids=[examples[i]["id"] for i in tr], test_ids=[examples[i]["id"] for i in te],
                sklearn_version=sklearn.__version__, cv_auroc=cv_table, probes={})
    preds = []
    for t, hops in tasks.items():
        L = fixed[t] if fixed[t] is not None else max(cv_table[t], key=cv_table[t].get)
        li = layers.index(L)
        m_tr = np.isin(arr(rows_tr, "hop"), hops)
        m_te = np.isin(arr(rows_te, "hop"), hops)
        y_tr, y_te = arr(rows_tr, "y")[m_tr], arr(rows_te, "y")[m_te]
        tau, f_cv = best_f05_threshold(y_tr, oof_cache[(t, L)])
        # Platt calibration on the out-of-fold scores: a monotone map, so every decision at the mapped
        # threshold is unchanged, but P(fail) becomes a real probability (τ ≈ 0.99 sat on a cliff).
        calib = fit_platt(y_tr, oof_cache[(t, L)])
        model = make_probe(C=C).fit(X_tr[m_tr, li].astype(np.float32), y_tr)
        p_te = model.predict_proba(X_te[m_te, li].astype(np.float32))[:, 1]
        g_te = arr(rows_te, "group")[m_te]
        test = detection_metrics(y_te, p_te >= tau, p_te)
        test["auroc_ci95"] = _bootstrap_auroc_ci(y_te, p_te, g_te)
        cf_te, hedge_te = arr(rows_te, "cf")[m_te], arr(rows_te, "hedge")[m_te]
        strata = {}
        if cf_te.any():
            strata["natural"] = detection_metrics(y_te[~cf_te], p_te[~cf_te] >= tau, p_te[~cf_te])
            strata["counterfactual"] = detection_metrics(y_te[cf_te], p_te[cf_te] >= tau, p_te[cf_te])
        fail = y_te == 1
        strata["hedge_fail_recall"] = float((p_te[fail & hedge_te] >= tau).mean()) if (fail & hedge_te).any() else None
        strata["confident_fail_recall"] = float((p_te[fail & ~hedge_te] >= tau).mean()) if (fail & ~hedge_te).any() else None
        joblib.dump(model, out_dir / f"{t}.joblib")
        export_portable(model, out_dir / f"{t}.npz")
        port = LinearProbe(out_dir / f"{t}.npz").p_fail(X_te[m_te, li].astype(np.float32))
        assert np.allclose(port, p_te, atol=1e-6), "portable probe disagrees with sklearn"
        pc_te = apply_platt(calib, p_te)
        test["ece_raw"], test["ece_calibrated"] = ece(y_te, p_te), ece(y_te, pc_te)
        # τ is stored on the calibrated scale (what ProbeBundle.p_fail returns); tau_raw on the LR scale
        meta["probes"][t] = dict(layer=int(L), hops=list(hops), tau=float(apply_platt(calib, tau)), tau_raw=float(tau),
                                 calibration=calib, cv_f05=float(f_cv),
                                 cv_auroc=float(cv_table[t][L]), test=test, test_strata=strata,
                                 n_train_rows=int(m_tr.sum()), n_test_rows=int(m_te.sum()))
        print(f"[probes] {t}: layer {L}  tau={meta['probes'][t]['tau']:.3f} (raw {tau:.3f})  CV F0.5={f_cv:.3f}  "
              f"TEST auroc={test.get('auroc', float('nan')):.3f} P={test['precision']:.3f} R={test['recall']:.3f} "
              f"F0.5={test['f05']:.3f} FPR={test['fpr']:.3f}  ECE raw {test['ece_raw']:.3f} -> calibrated "
              f"{test['ece_calibrated']:.3f}", flush=True)
        for r, p, pc in zip([r for r, m in zip(rows_te, m_te) if m], p_te, pc_te):
            preds.append(dict(task=t, id=r["id"], hop=r["hop"], y=r["y"], p_fail=float(pc), p_fail_raw=float(p)))

    (out_dir / "probes_meta.json").write_text(json.dumps(meta, indent=1))
    with open(out_dir / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"[probes] saved to {out_dir} ({time.time() - t0:.0f}s)")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Runtime bundle
# ─────────────────────────────────────────────────────────────────────────────

def export_portable(model, path) -> None:
    """StandardScaler + LogisticRegression as plain arrays (no sklearn-version pickle issues)."""
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    np.savez(path, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0], intercept=lr.intercept_)


class LinearProbe:
    """P(fail) = sigmoid(((x - mean) / scale) · coef + intercept) — identical to the pipeline's predict_proba."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale = z["mean"].astype(np.float64), z["scale"].astype(np.float64)
        self.coef, self.intercept = z["coef"].astype(np.float64), float(z["intercept"][0])

    def p_fail(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        return 1.0 / (1.0 + np.exp(-(((x - self.mean) / self.scale) @ self.coef + self.intercept)))


PROBE_MODES = {"shared": {1: "hop12", 2: "hop12"},       # one probe + one τ for every hop boundary (main)
               "dedicated": {1: "hop1", 2: "hop2"}}      # a separate probe + τ per hop (ablation)


class ProbeBundle:
    """The probes + thresholds used by the live generation loop."""

    def __init__(self, probe_dir, hop_task: dict | None = None, mode: str = "shared"):
        probe_dir = Path(probe_dir)
        self.meta = json.loads((probe_dir / "probes_meta.json").read_text())
        # which trained probe scores each boundary
        self.mode = mode if hop_task is None else "custom"
        self.hop_task = hop_task or PROBE_MODES[mode]
        self.models = {t: self._load(probe_dir, t) for t in set(self.hop_task.values())}
        self.layer = {k: self.meta["probes"][t]["layer"] for k, t in self.hop_task.items()}
        self.tau = {k: self.meta["probes"][t]["tau"] for k, t in self.hop_task.items()}
        self.calib = {k: self.meta["probes"][t].get("calibration") for k, t in self.hop_task.items()}

    @staticmethod
    def _load(probe_dir: Path, task: str):
        if (probe_dir / f"{task}.npz").exists():
            return LinearProbe(probe_dir / f"{task}.npz")
        import joblib
        m = joblib.load(probe_dir / f"{task}.joblib")
        m.p_fail = lambda x, m=m: m.predict_proba(np.nan_to_num(np.atleast_2d(x)))[:, 1]
        return m

    def p_fail_raw(self, k: int, pooled_all_layers: np.ndarray) -> float:
        """Uncalibrated LR output.  pooled_all_layers: [n_layers, d] mean-pooled hop-k states (float16)."""
        x = pooled_all_layers[self.layer[k]].astype(np.float16).astype(np.float32)
        return float(self.models[self.hop_task[k]].p_fail(x)[0])

    def p_fail(self, k: int, pooled_all_layers: np.ndarray) -> float:
        """Calibrated P(fail) (same scale as self.tau); raw LR output if the bundle has no calibration."""
        return float(apply_platt(self.calib[k], self.p_fail_raw(k, pooled_all_layers)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="tfidf", choices=["tfidf", "full10"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hs-dir", default=None, help="local dir with <id>.pt files (default: download from HF)")
    ap.add_argument("--hop1-layer", default="10", help="int, or 'cv' to pick by train-split CV")
    ap.add_argument("--hop2-layer", default="10")
    ap.add_argument("--hop12-layer", default="10")
    ap.add_argument("--sweep", default="all", help="'all' (28 layers, RQ4 table), 'none', or e.g. '8,10,13'")
    ap.add_argument("--C", type=float, default=1.0, help="LR inverse regularisation (1.0 = the deployed probe)")
    args = ap.parse_args()
    train_probes(args.setting, args.out, args.hs_dir, args.hop1_layer, args.hop2_layer, args.hop12_layer, args.sweep,
                 C=args.C)


if __name__ == "__main__":
    main()
