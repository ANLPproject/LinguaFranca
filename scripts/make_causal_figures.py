"""
scripts/make_causal_figures.py
──────────────────────────────
Paper figure for the TF-IDF causal results (CPU, seconds):

  (a) logit causal tracing (nb 09): recovery of the clean hop-1 log-prob by patched block, 5 conditions
  (b) probe vs behaviour (nb 10): entity-span patches — behavioural recovery and the probes' restoration
      (fraction of the Test A rise), old probe (block 10) and tuned probe (block 12); third-entity control
  (c) Test B pair by pair (nb 10): entity last-token patches, recovery vs tuned-probe drop, one point per pair
      (mean over blocks < 12), with the within-pair / within-block r from analysis.md

Usage:
  python scripts/make_causal_figures.py --out outputs/patching/causal_figure.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_tracing_analysis import PROBES, bundle, scores, within_r  # noqa: E402

TRACE = "outputs/patching/causal_tracing_tfidf/causal_tracing_tfidf/summary.json"
FEATS = "outputs/patching/probe_tracing_tfidf/probe_tracing_tfidf/features.npz"
# reference categorical palette, fixed order (identity also carried by line style and direct labels)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/patching/causal_figure.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = json.loads((ROOT / TRACE).read_text())["conditions"]
    Z = np.load(ROOT / FEATS, allow_pickle=True)
    Z = {k: Z[k] for k in Z.files}
    keep = Z["gap"] >= 1.0
    patch = [str(x) for x in Z["patch"]]
    rec = Z["recovery"][keep]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, 0.9]))

    # (a) logit tracing
    ax = axes[0]; style(ax)
    names = [("entity_span", "entity span", "-"), ("entity_last", "entity last token", "--"),
             ("tail", "tokens after entity", "-."), ("mlp_window", "MLP at entity (±2)", ":"),
             ("last_token", "last prompt token", (0, (5, 1, 1, 1)))]
    x = np.arange(28)
    ax.axvspan(13.5, 14.5, color=GRID, lw=0)
    ax.text(14, 1.04, "handoff", ha="center", fontsize=7.5, color=MUTED)
    for (key, lab, ls), col in zip(names, C):
        m, ci = np.array(T[key]["mean"]), np.array(T[key]["ci95"])
        ax.fill_between(x, m - ci, m + ci, color=col, alpha=0.12, lw=0)
        ax.plot(x, m, ls=ls, color=col, lw=2, label=lab)
    ax.set_xlim(0, 27); ax.set_ylim(-0.08, 1.1)
    ax.set_xlabel("patched block", fontsize=9, color=INK)
    ax.set_ylabel("recovery of the clean hop-1 log-prob", fontsize=9, color=INK)
    ax.set_title("(a) Where the entity information lives", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.97))

    # (b) probe vs behaviour, entity-span patches
    ax = axes[1]; style(ax)
    k_s, k_c = patch.index("entity_span"), patch.index("control")
    xb = np.arange(14)
    ax.plot(xb, np.nanmean(rec[:, k_s, :14], 0), color=INK, lw=2.2, label="behaviour (log-prob recovery)")
    for (name, d), col, ls in zip(PROBES.items(), (C[1], C[0]), ("--", "-")):
        pr = bundle(d)
        li, P = list(Z["feat_layers"]).index(pr["layer"]), pr["layer"]
        s_clean, s_cf, s_p = scores(Z, keep, li, pr["score"])
        rise = (s_cf - s_clean).mean()
        frac = (s_cf[:, None] - s_p[:, k_s, :P]).mean(0) / rise
        ctrl = np.nanmean(s_cf[:, None] - s_p[:, k_c, :P], 0) / rise
        short = "tuned probe, block 12" if "tuned" in name else "old probe, block 10"
        ax.plot(np.arange(P), frac, ls=ls, color=col, lw=2, marker="o", ms=3.5, label=short)
        ax.plot(np.arange(P), ctrl, ls=":", color=col, lw=1.4)
        ax.axvline(P, color=col, lw=0.8, ls=(0, (2, 2)))
        ax.text(P - 0.15, 1.06, f"reads {P}", ha="right", fontsize=7, color=MUTED)
    ax.text(0.3, 0.08, "third-entity control (dotted)", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 13); ax.set_ylim(-0.08, 1.1)
    ax.set_xlabel("patched block (entity span)", fontsize=9, color=INK)
    ax.set_ylabel("fraction restored", fontsize=9, color=INK)
    ax.set_title("(b) The probe follows the restoration", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.14))

    # (c) pair-level Test B, tuned probe, entity last token
    ax = axes[2]; style(ax)
    pr = bundle(PROBES["tuned (C=0.01, block 12)"])
    li, P = list(Z["feat_layers"]).index(pr["layer"]), pr["layer"]
    s_clean, s_cf, s_p = scores(Z, keep, li, pr["score"])
    k_l = patch.index("entity_last")
    r_l, d_l = rec[:, k_l, :P], s_cf[:, None] - s_p[:, k_l, :P]
    ok = ~(np.isnan(r_l).any(1) | np.isnan(d_l).any(1))
    ax.scatter(r_l[ok].mean(1), d_l[ok].mean(1), s=12, color=C[0], alpha=0.75, edgecolor="white", linewidth=0.4)
    rw = within_r(r_l[ok], d_l[ok])
    ax.text(0.03, 0.95, f"within-pair r = {rw:+.2f}\n(old probe: +0.28)", transform=ax.transAxes, va="top",
            fontsize=8, color=INK)
    ax.set_xlabel("behavioural recovery (mean, blocks 0–11)", fontsize=9, color=INK)
    ax.set_ylabel("tuned-probe log-odds drop", fontsize=9, color=INK)
    ax.set_title("(c) Pair by pair (entity last token)", fontsize=10, color=INK, loc="left")

    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
