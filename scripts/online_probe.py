"""
scripts/online_probe.py
───────────────────────
Can the probe stop a failing hop BEFORE `</hopK>`?  The deployed probe is fitted on full-hop means, so its
scores on a partial hop are off-scale (uncertainty_early.md, B).  Here a probe is trained on partial hops.

  features   running mean of the block-12 states after each token t of a hop (the deployed feature at t = n)
             + log(t), the number of hop tokens so far (known online; the hop length is not)
  probe      StandardScaler + LR (C = 0.01, the deployed recipe) on every prefix t >= MIN_T of the train hops,
             label = the hop's label
  online     fire at the first t >= MIN_T with P(fail) >= θ; θ = F0.5-optimal on grouped 5-fold out-of-fold
             predictions inside the train split (the test split is never used to choose anything)
  hybrid     the online trigger at a stricter θ (train-OOF precision >= the deployed probe's), otherwise the
             deployed probe at `</hopK>` as now

Stage 1 (GPU, ~20 min on a T4): one teacher-forced pass per Phase 1 trajectory of the probe's split, the same
extraction as uncertainty_early.hop_signals; saves the per-token running means (~1 GB, float16).
Stage 2 (CPU, ~10-20 min): training, OOF thresholds, test metrics, figure.

Usage (Kaggle GPU, HF_TOKEN in the environment):
    python scripts/online_probe.py --stage all --out /kaggle/working/online_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import (best_f05_threshold, detection_metrics, fbeta, hop_match,  # noqa: E402
                                 load_setting_examples)
from src.pipeline.probes import LinearProbe, _auroc, _bootstrap_auroc_ci, _logit, hop_rows, make_probe  # noqa: E402

PROBE_DIR = "outputs/pipeline/tfidf/probes_lr_C0.01_L12"
TASK = "hop12"
MIN_T = 3
FRACS = np.round(np.arange(0.1, 1.01, 0.1), 1)
C_REG = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: per-token running means (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def extract(out: Path, probe_dir: Path, limit: int | None):
    import torch
    from src.pipeline.generation import _device, char_span_to_tokens, load_model

    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    layer = meta["probes"][TASK]["layer"]
    split = {i: "train" for i in meta["train_ids"]} | {i: "test" for i in meta["test_ids"]}
    examples = [e for e in load_setting_examples("tfidf") if e["id"] in split]
    rows = [r for r in hop_rows(examples, range(len(examples))) if not r["cf"]]
    by_ex: dict[int, list[dict]] = {}
    for r in rows:
        by_ex.setdefault(r["ei"], []).append(r)
    todo = list(by_ex.items())[:limit] if limit else list(by_ex.items())

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    dev = _device(model)
    feats = {"train": [], "test": []}
    index = {"train": [], "test": []}
    t0, n_skip = time.time(), 0
    with torch.inference_mode():
        for n, (ei, rs) in enumerate(todo):
            e = examples[ei]
            prompt, cot = e["prompt"], e.get("generated_cot", "")
            full = tok(prompt + cot, return_tensors="pt").input_ids[0]
            p_len = len(tok(prompt, return_tensors="pt").input_ids[0])
            enc = tok(cot, return_offsets_mapping=True, add_special_tokens=False)
            if not cot or len(full) != p_len + len(enc["input_ids"]):
                n_skip += 1
                continue
            offs = enc["offset_mapping"]
            hs = model.model(full.unsqueeze(0).to(dev), output_hidden_states=True).hidden_states[layer + 1][0]
            for r in rs:
                m = hop_match(cot, r["hop"])
                if m is None:
                    continue
                ts, te = char_span_to_tokens([x[0] for x in offs], [x[1] for x in offs], m.start(1), m.end(1))
                if ts is None or te is None:
                    continue
                a, b = p_len + ts, p_len + te + 1
                H = hs[a:b].float()
                cum = H.cumsum(0) / torch.arange(1, b - a + 1, device=dev)[:, None]
                cum[-1] = H.mean(0).to(torch.float16).float()        # the pipeline's fp16 pooled feature
                sp = split[r["id"]]
                feats[sp].append(cum.to(torch.float16).cpu().numpy())
                index[sp].append(dict(id=r["id"], hop=r["hop"], y=r["y"], group=r["group"], hedge=r["hedge"],
                                      n=int(b - a)))
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} trajectories, {time.time() - t0:.0f}s", flush=True)
    for sp in feats:
        np.save(out / f"prefix_{sp}.npy", np.concatenate(feats[sp]))
        (out / f"prefix_{sp}.json").write_text(json.dumps(index[sp]))
    print(f"done; {n_skip} trajectories skipped (prompt/generation boundary not token-aligned)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: training and evaluation (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def load(out: Path, sp: str):
    X = np.load(out / f"prefix_{sp}.npy", mmap_mode="r")
    idx = json.loads((out / f"prefix_{sp}.json").read_text())
    start = np.concatenate([[0], np.cumsum([r["n"] for r in idx])])
    return X, idx, start


def design(X, idx, start, rows=None):
    """Prefix rows t >= MIN_T (and t = n for hops shorter than MIN_T): (features with log t, hop index, t)."""
    rows = range(len(idx)) if rows is None else rows
    sel, hop, t = [], [], []
    for i in rows:
        n = idx[i]["n"]
        ts = np.arange(min(MIN_T, n), n + 1)
        sel.append(start[i] + ts - 1); hop.append(np.full(len(ts), i)); t.append(ts)
    sel, hop, t = np.concatenate(sel), np.concatenate(hop), np.concatenate(t)
    F = np.nan_to_num(np.asarray(X[sel], dtype=np.float32))
    return np.column_stack([F, np.log(t)]), hop, t


def online(p, hop, t, theta, n_hops):
    """First t with p >= θ per hop (None if never)."""
    fire = [None] * n_hops
    order = np.lexsort((t, hop))
    for j in order:
        h = hop[j]
        if fire[h] is None and p[j] >= theta:
            fire[h] = int(t[j])
    return fire


def best_online_theta(p, hop, t, y, n_hops, min_precision=None):
    """θ maximising hop-level F0.5 of the online trigger (or the lowest θ with precision >= min_precision)."""
    # a hop fires at θ iff max_t p >= θ, so detection depends on the per-hop max only
    mx = np.full(n_hops, -np.inf)
    np.maximum.at(mx, hop, p)
    best, best_f = 1.0, -1.0
    for th in np.unique(mx):
        flag = mx >= th
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn = int((~flag & (y == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        if min_precision is not None:
            if pr >= min_precision:
                return float(th)
            continue
        f = fbeta(pr, rc)
        if f > best_f:
            best, best_f = float(th), f
    return best


def analyze(out: Path, probe_dir: Path):
    from sklearn.model_selection import GroupKFold
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    pm = meta["probes"][TASK]
    dep = LinearProbe(probe_dir / f"{TASK}.npz")
    z_tau = float(_logit(pm["tau_raw"]))

    Xtr, itr, str_ = load(out, "train")
    Xte, ite, ste = load(out, "test")
    y_tr = np.array([r["y"] for r in itr]); y_te = np.array([r["y"] for r in ite])
    g_tr = np.array([r["group"] for r in itr]); g_te = np.array([r["group"] for r in ite])
    n_te = np.array([r["n"] for r in ite])
    t0 = time.time()

    # out-of-fold scores inside the train split (grouped by question pair): the prefix probe on every prefix, and
    # the full-hop recipe on full hops (its OOF precision at its F0.5 threshold = the bar for the strict θ)
    full_tr = np.nan_to_num(np.asarray(Xtr[str_[1:] - 1], dtype=np.float32))
    parts, oof_full = [], np.zeros(len(itr))
    for f, (a, b) in enumerate(GroupKFold(5).split(np.zeros(len(itr)), y_tr, g_tr)):
        Fa, ha, _ = design(Xtr, itr, str_, a)
        m = make_probe(C=C_REG).fit(Fa, y_tr[ha])
        Fb, hb, tb = design(Xtr, itr, str_, b)
        parts.append((m.predict_proba(Fb)[:, 1], hb, tb))
        oof_full[b] = make_probe(C=C_REG).fit(full_tr[a], y_tr[a]).predict_proba(full_tr[b])[:, 1]
        print(f"  fold {f}: {time.time() - t0:.0f}s", flush=True)
    oof_p, oof_hop, oof_t = (np.concatenate(x) for x in zip(*parts))
    th_f05 = best_online_theta(oof_p, oof_hop, oof_t, y_tr, len(itr))
    t_full = best_f05_threshold(y_tr, oof_full)[0]
    dep_prec = detection_metrics(y_tr, oof_full >= t_full)["precision"]
    th_strict = best_online_theta(oof_p, oof_hop, oof_t, y_tr, len(itr), min_precision=dep_prec)

    # final prefix probe on all train hops
    F, h, _ = design(Xtr, itr, str_)
    model = make_probe(C=C_REG).fit(F, y_tr[h])
    Fte, hte, tte = design(Xte, ite, ste)
    p_te = model.predict_proba(Fte)[:, 1]
    print(f"  trained: {time.time() - t0:.0f}s", flush=True)

    # deployed probe, end of hop
    full_te = np.nan_to_num(np.asarray(Xte[ste[1:] - 1], dtype=np.float64))
    z_end = _logit(dep.p_fail(full_te))
    end_flag = z_end >= z_tau

    L = [f"# A probe that stops a failing hop before it ends (TF-IDF Phase 1 split, block {pm['layer']})", "",
         f"Train {len(itr)} hops, test {len(ite)} hops (fail {y_te.mean():.2f}); hop length median {int(np.median(n_te))} "
         f"tokens. Prefix probe: StandardScaler + LR (C = {C_REG}) on the running mean of the hop's block-{pm['layer']} "
         f"states after each token t ≥ {MIN_T}, plus log t; label = the hop's label. Thresholds from grouped 5-fold "
         f"out-of-fold predictions inside the train split only. Deployed probe: τ = {pm['tau']:.3f} at `</hopK>`.", "",
         f"Sanity: deployed probe on these features, test AUROC {_auroc(y_te, z_end):.4f} (bundle "
         f"{pm['test']['auroc']:.4f}).", "",
         "## 1. Ranking quality by how much of the hop has been generated (test AUROC)", "",
         "| prefix | prefix probe [95% CI] | deployed probe (full-hop fit) |", "|---|---|---|"]
    curve = {"prefix": [], "deployed": []}
    t_min = np.minimum(MIN_T, n_te)                                 # first prefix the prefix probe scores
    first_row = np.concatenate([[0], np.cumsum(n_te - t_min + 1)])[:-1]   # where each hop starts in p_te
    for fr in FRACS:
        ts = np.maximum(t_min, np.ceil(fr * n_te).astype(int))     # same prefix length for both probes
        sp = p_te[first_row + ts - t_min]
        zd = _logit(dep.p_fail(np.nan_to_num(np.asarray(Xte[ste[:-1] + ts - 1], dtype=np.float64))))
        a, ad = _auroc(y_te, sp), _auroc(y_te, zd)
        lo, hi = _bootstrap_auroc_ci(y_te, sp, g_te)
        curve["prefix"].append(a); curve["deployed"].append(ad)
        L.append(f"| {int(fr * 100)} % | {a:.3f} [{lo:.3f}, {hi:.3f}] | {ad:.3f} |")

    L += ["", "## 2. Online triggers on the test split (hop level)", "",
          f"θ (train OOF): F0.5-optimal {th_f05:.3f}; strict {th_strict:.3f} (precision ≥ the deployed probe's train "
          f"precision {dep_prec:.3f}).", "",
          "| trigger | precision | recall | F0.5 | FPR | fires before `</hopK>` (true failures) | median share of the hop at firing | hop tokens saved per caught failure |",
          "|---|---|---|---|---|---|---|---|"]

    def row(name, fire_t, flag):
        dm = detection_metrics(y_te, flag)
        tp = np.flatnonzero(flag & (y_te == 1))
        ft = np.array([fire_t[i] if fire_t[i] is not None else n_te[i] for i in tp])
        early = np.mean(ft < n_te[tp]) if len(tp) else np.nan
        share = np.median(ft / n_te[tp]) if len(tp) else np.nan
        saved = np.mean(n_te[tp] - ft) if len(tp) else np.nan
        L.append(f"| {name} | {dm['precision']:.3f} | {dm['recall']:.3f} | {dm['f05']:.3f} | {dm['fpr']:.3f} | "
                 f"{early:.0%} | {share:.0%} | {saved:.1f} |")
        return ft
    row("Deployed probe at `</hopK>` (now)", [int(n) if f else None for n, f in zip(n_te, end_flag)], end_flag)
    fire = online(p_te, hte, tte, th_f05, len(ite))
    row("Prefix probe, online, F0.5 θ", fire, np.array([f is not None for f in fire]))
    fire_s = online(p_te, hte, tte, th_strict, len(ite))
    on_s = np.array([f is not None for f in fire_s])
    row("Prefix probe, online, strict θ", fire_s, on_s)
    hyb_t = [fs if fs is not None else (int(n) if e else None) for fs, n, e in zip(fire_s, n_te, end_flag)]
    row("**Hybrid**: strict online, else deployed at `</hopK>`", hyb_t, on_s | end_flag)
    L += ["", "Tokens saved are hop tokens not generated before the restart (the pipeline's restart cost is "
              "unchanged). The end-to-end effect is not simulated: every cached restart started from a complete hop.", ""]

    (out / "online_probe.md").write_text("\n".join(L), encoding="utf-8")
    plot(out, curve, n_te, y_te, fire_s)
    print("\n".join(L))


def plot(out, curve, n_te, y_te, fire_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#6b7280", "#e5e7eb"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax = axes[0]
    x = FRACS * 100
    ax.plot(x, curve["prefix"], color=BLUE, lw=2, marker="o", ms=3.5, label="prefix-trained probe")
    ax.plot(x, curve["deployed"], color=ORANGE, lw=2, ls="--", marker="o", ms=3.5, label="deployed probe (full-hop fit)")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("share of the hop generated (%)", fontsize=9, color=INK)
    ax.set_ylabel("hop-failure AUROC (test)", fontsize=9, color=INK)
    ax.set_title("(a) Ranking quality inside the hop", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axes[1]
    tp = [i for i, f in enumerate(fire_s) if f is not None and y_te[i] == 1]
    share = np.array([fire_s[i] / n_te[i] for i in tp]) * 100
    ax.hist(share, bins=np.arange(0, 105, 5), color=BLUE, edgecolor="white", linewidth=1)
    ax.set_xlabel("share of the hop generated when the online trigger fires (%)", fontsize=9, color=INK)
    ax.set_ylabel("caught failing hops", fontsize=9, color=INK)
    ax.set_title("(b) When the strict online trigger fires", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "online_probe.png", dpi=220)
    fig.savefig(out / "online_probe.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
    ap.add_argument("--out", default="outputs/pipeline/tfidf/online_probe")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories only (smoke test)")
    args = ap.parse_args()
    out, probe_dir = Path(args.out), ROOT / args.probe_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("extract", "all"):
        extract(out, probe_dir, args.limit)
    if args.stage in ("analyze", "all"):
        analyze(out, probe_dir)


if __name__ == "__main__":
    main()
