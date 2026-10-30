"""
scripts/probe_tracing_analysis.py
─────────────────────────────────
CPU analyses on probe_tracing.py's features.npz (no model).

Part 1 (seconds, needs only features.npz + the two probe bundles):
  * Test A per failure kind (hedge / other_wrong_entity)
  * Test B done properly.  probe_tracing.py pooled pairs x blocks into one Spearman, which
    (a) treats ~2,300 rows as independent although there are ~190 pairs (p-values far too small) and
    (b) mixes block trends and pair differences into the correlation.  Here:
      - Spearman across pairs at each block separately
      - within-pair, within-block correlation: recovery and probe drop with pair and block means
        removed (two-way demeaned), 95% CI from a bootstrap over PAIRS
      - the pooled Spearman with a pair-bootstrap CI instead of a p-value
  * specificity: probe drop under the entity patch minus under the third-entity control, per pair
  * lag: probe drop as a fraction of the Test A rise, next to behavioural recovery, per block

Part 2 (--natural; Kaggle CPU, downloads the natural TF-IDF hidden states, ~10-20 min):
  * robustness (mid-term cell 9): the shared hop12 LR retrained at blocks 10 / 12 with
    C in {0.001, 0.01, 0.1, 1} and on 5 bootstrap resamples of the training questions;
    Test A AUROC and the within-pair Test B correlation for every retrain
  * grounding direction (mid-term cell 10): d = mean(F_cf - F_clean) over half of the pairs;
    does the projection on d separate clean vs CF on the other half, and natural failures from
    natural successes (seed questions of the pairs excluded)?  cosine(d, probe weights).

Usage:
  python scripts/probe_tracing_analysis.py
  python scripts/probe_tracing_analysis.py --natural          # + Part 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.probes import LinearProbe, apply_platt  # noqa: E402

PROBES = {"old (C=1, block 10)": "outputs/pipeline/tfidf/probes",
          "tuned (C=0.01, block 12)": "outputs/patching/patching_tuned_results/probes_c001_L12"}
N_BOOT = 2000


def logodds(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def auroc(neg, pos):
    return float(roc_auc_score([0] * len(neg) + [1] * len(pos), np.r_[neg, pos]))


def ci(boot):
    return f"[{np.nanpercentile(boot, 2.5):+.2f}, {np.nanpercentile(boot, 97.5):+.2f}]"


def within_r(x, y):
    """Pearson of two-way demeaned x, y ([pairs, blocks], complete rows only)."""
    def dm(a):
        return a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
    a, b = dm(x).ravel(), dm(y).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_b_stats(rec, drop, rng):
    """rec, drop: [pairs, blocks] for blocks below the probe layer."""
    ok = ~(np.isnan(rec).any(1) | np.isnan(drop).any(1))
    x, y = rec[ok], drop[ok]
    n = len(x)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    w = within_r(x, y)
    w_b = [within_r(x[i], y[i]) for i in idx]
    pooled = spearmanr(x.ravel(), y.ravel())[0]
    pooled_b = [spearmanr(x[i].ravel(), y[i].ravel())[0] for i in idx[:500]]
    per_block = [(spearmanr(x[:, b], y[:, b])[0], np.std(x[:, b])) for b in range(x.shape[1])]
    return dict(n=n, within=w, within_ci=ci(w_b), pooled=pooled, pooled_ci=ci(pooled_b), per_block=per_block)


def bundle(d):
    meta = json.loads((ROOT / d / "probes_meta.json").read_text())["probes"]["hop12"]
    lp = LinearProbe(ROOT / d / "hop12.npz")
    return dict(layer=meta["layer"], tau=meta["tau"], lp=lp,
                score=lambda X: logodds(apply_platt(meta["calibration"], lp.p_fail(np.asarray(X, np.float32)))))


def scores(Z, keep, li, score):
    """log-odds of clean, CF and every patched run at feature layer index li."""
    s_clean, s_cf = score(Z["F_clean"][keep, li]), score(Z["F_cf"][keep, li])
    F = Z["F_patch"][keep][:, :, :, li]                                  # [n, conds, 28, d]
    s_p = np.full(F.shape[:3], np.nan)
    ok = ~np.isnan(F[..., 0].astype(np.float32))
    s_p[ok] = score(F[ok])
    return s_clean, s_cf, s_p


def part1(Z, keep, patch, rng):
    rec, kinds = Z["recovery"][keep], Z["kinds"][keep]
    L = []
    for name, d in PROBES.items():
        pr = bundle(d)
        li, P = list(Z["feat_layers"]).index(pr["layer"]), pr["layer"]
        s_clean, s_cf, s_p = scores(Z, keep, li, pr["score"])
        rise = s_cf - s_clean
        L += [f"## {name}", "", "### Test A by failure kind (same hop-1 words)", "",
              "| CF kind | n | flagged clean | flagged CF | AUROC clean vs CF | mean log-odds rise |", "|---|---|---|---|---|---|"]
        tau = logodds(np.array(pr["tau"]))
        for k in ["all"] + sorted(set(kinds)):
            m = np.ones(len(kinds), bool) if k == "all" else kinds == k
            L.append(f"| {k} | {m.sum()} | {(s_clean[m] >= tau).mean():.3f} | {(s_cf[m] >= tau).mean():.3f} | "
                     f"{auroc(s_clean[m], s_cf[m]):.3f} | {rise[m].mean():+.2f} |")

        drop = {c: s_cf[:, None] - s_p[:, k, :P] for k, c in enumerate(patch)}
        L += ["", f"### Test B (patches at blocks < {P}), correlation of behavioural recovery with probe drop", "",
              "| patch | pairs | within-pair, within-block r [95% CI, pair bootstrap] | pooled Spearman [95% CI] |",
              "|---|---|---|---|"]
        st = {c: test_b_stats(rec[:, k, :P], drop[c], rng) for k, c in enumerate(patch)}
        for c in patch:
            s = st[c]
            L.append(f"| {c} | {s['n']} | {s['within']:+.2f} {s['within_ci']} | {s['pooled']:+.2f} {s['pooled_ci']} |")
        L += ["", "Spearman across pairs at each block (SD of behavioural recovery in brackets; near-zero SD = ceiling, rho uninformative):", "",
              "| block | " + " | ".join(patch) + " |", "|---" * (len(patch) + 1) + "|"]
        for b in range(P):
            L.append(f"| {b} | " + " | ".join(f"{st[c]['per_block'][b][0]:+.2f} ({st[c]['per_block'][b][1]:.2f})" for c in patch) + " |")

        if "control" in patch:
            k_s, k_c = patch.index("entity_span"), patch.index("control")
            ok = ~np.isnan(s_p[:, k_c, :P]).any(1)
            spec = (s_p[ok, k_c, :P] - s_p[ok, k_s, :P]).mean(1)          # drop(span) - drop(control), per pair
            b = [spec[rng.integers(0, len(spec), len(spec))].mean() for _ in range(N_BOOT)]
            L += ["", "### Specificity: probe drop (entity span) − probe drop (third-entity control), mean over blocks < "
                  f"{P}", "", f"- {spec.mean():+.2f} log-odds {ci(b)}; larger for the entity patch in "
                  f"{(spec > 0).mean() * 100:.0f}% of {ok.sum()} pairs"]

        L += ["", "### Lag: probe drop as a fraction of the Test A rise vs behavioural recovery (entity span)", "",
              "| block | behavioural recovery | probe restoration |", "|---|---|---|"]
        k_s = patch.index("entity_span")
        for b in range(P):
            L.append(f"| {b} | {np.nanmean(rec[:, k_s, b]):.2f} | {np.nanmean(drop['entity_span'][:, b]) / rise.mean():.2f} |")
        L.append("")
    return L


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: needs the natural hidden states
# ─────────────────────────────────────────────────────────────────────────────

def part2(Z, keep, patch, rng):
    from src.pipeline.common import CF_V1, download_hidden_states, hf_file, load_setting_examples, read_jsonl
    from src.pipeline.probes import hop_rows, load_pooled, make_probe
    layers = [int(x) for x in Z["feat_layers"]]
    ex = load_setting_examples("tfidf")
    meta = json.loads((ROOT / PROBES["old (C=1, block 10)"] / "probes_meta.json").read_text())
    tr_ids, te_ids = set(meta["train_ids"]), set(meta["test_ids"])
    seeds = {c["id"]: c["clean_pair_id"] for c in read_jsonl(hf_file(CF_V1["file"], CF_V1["revision"]))}
    seed_ids = {seeds.get(i) for i in Z["ids"]}
    tr = hop_rows(ex, [i for i, e in enumerate(ex) if e["id"] in tr_ids])
    te = hop_rows(ex, [i for i, e in enumerate(ex) if e["id"] in te_ids and e["id"] not in seed_ids])
    hs = download_hidden_states("tfidf", ids=sorted({r["id"] for r in tr + te}))
    Xtr = np.nan_to_num(load_pooled(ex, tr, hs, layers).astype(np.float32))
    Xte = np.nan_to_num(load_pooled(ex, te, hs, layers).astype(np.float32))
    ytr, yte = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
    q = np.array([r["id"] for r in tr]); uq = np.unique(q)
    rec = Z["recovery"][keep]
    k_l = patch.index("entity_last")

    def evaluate(m, li, P):
        sc = lambda X: m.decision_function(np.nan_to_num(np.asarray(X, np.float32)))  # noqa: E731
        s_clean, s_cf, s_p = scores(Z, keep, li, sc)
        st = test_b_stats(rec[:, k_l, :P], s_cf[:, None] - s_p[:, k_l, :P], rng)
        return (auroc(sc(Xte[yte == 0, li]), sc(Xte[yte == 1, li])),
                auroc(s_clean, s_cf), st["within"], st["within_ci"])

    L = ["## Part 2a: robustness of Tests A/B to probe retrains (shared hop12 LR, natural train split)", "",
         "Natural AUROC on test hops 1+2 (seed questions of the pairs excluded). Test B = within-pair, within-block r, entity last token.", "",
         "| block | C | training sample | natural AUROC | Test A AUROC | Test B r [95% CI] |", "|---|---|---|---|---|---|"]
    global N_BOOT
    nb, N_BOOT = N_BOOT, 500
    for P in layers:
        li = layers.index(P)
        for C in (0.001, 0.01, 0.1, 1.0):
            m = make_probe(C=C).fit(Xtr[:, li], ytr)
            a, b, w, wc = evaluate(m, li, P)
            L.append(f"| {P} | {C:g} | full | {a:.3f} | {b:.3f} | {w:+.2f} {wc} |")
        for s in range(5):
            pick = set(rng.choice(uq, len(uq), replace=True))
            m_ = np.isin(q, list(pick))
            for C in (0.01, 1.0):
                m = make_probe(seed=s, C=C).fit(Xtr[m_, li], ytr[m_])
                a, b, w, wc = evaluate(m, li, P)
                L.append(f"| {P} | {C:g} | bootstrap {s} | {a:.3f} | {b:.3f} | {w:+.2f} {wc} |")
        print(f"robustness block {P} done", flush=True)
    N_BOOT = nb

    L += ["", "## Part 2b: grounding direction d = mean(F_cf − F_clean) (2-fold over pairs)", "",
          "| block | held-out pairs: AUROC clean vs CF | natural test: AUROC failure vs success | cos(d / scale, probe weights) |",
          "|---|---|---|---|"]
    n = keep.sum()
    fold = np.arange(n) % 2
    for P in layers:
        li = layers.index(P)
        Fc, Ff = Z["F_clean"][keep, li].astype(np.float32), Z["F_cf"][keep, li].astype(np.float32)
        a_held, nat = [], []
        for f in (0, 1):
            d = (Ff[fold == f] - Fc[fold == f]).mean(0)
            a_held.append(auroc(Fc[fold != f] @ d, Ff[fold != f] @ d))
            nat.append(auroc(Xte[yte == 0, li] @ d, Xte[yte == 1, li] @ d))
        d = (Ff - Fc).mean(0)
        cos = []
        for name, dd in PROBES.items():
            pr = bundle(dd)
            if pr["layer"] == P:
                v = d / pr["lp"].scale                                     # d in the probe's standardized space
                cos.append(f"{name}: {v @ pr['lp'].coef / (np.linalg.norm(v) * np.linalg.norm(pr['lp'].coef)):+.2f}")
        L.append(f"| {P} | {np.mean(a_held):.3f} | {np.mean(nat):.3f} | {'; '.join(cos) or '–'} |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="outputs/patching/probe_tracing_tfidf/probe_tracing_tfidf/features.npz")
    ap.add_argument("--out", default="outputs/patching/probe_tracing_tfidf/analysis.md")
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--natural", action="store_true", help="Part 2 (downloads the natural hidden states)")
    args = ap.parse_args()
    Z = np.load(ROOT / args.features, allow_pickle=True)
    Z = {k: Z[k] for k in Z.files}
    keep = Z["gap"] >= args.min_gap
    patch = [str(x) for x in Z["patch"]]
    rng = np.random.default_rng(0)
    L = ["# Probe <-> causal tracing: corrected statistics", "",
         f"Pairs with a hop-1 log-prob gap >= {args.min_gap} nat: {keep.sum()} of {len(keep)}. "
         "Probe scores are calibrated log-odds; CIs are 95% bootstraps over pairs.", ""]
    L += part1(Z, keep, patch, rng)
    if args.natural:
        L += part2(Z, keep, patch, rng)
    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
