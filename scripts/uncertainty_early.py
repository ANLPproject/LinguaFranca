"""
scripts/uncertainty_early.py
────────────────────────────
Two questions the pipeline results leave open, on the probe's own Phase 1 split (TF-IDF setting):

  A. Is the probe more than the model's token confidence?  Active-RAG triggers (FLARE: retrieve when a
     token's probability drops below θ; DRAGIN-style entropy) use exactly that signal.  Hop-level AUROC
     and F0.5 of mean / max token NLL and entropy vs the probe, an LR on all of them, probe + them, and
     the failures the model is *confident* about.
  B. How early inside a hop does the probe know?  The probe applied to the running mean of the block-12
     states after the first m tokens of the hop (no retraining), vs the confidence signals on the same
     prefix; and an online trigger that fires at the first prefix ≥ τ.

Stage 1 (GPU, ~20 min on a T4): one teacher-forced forward pass per Phase 1 trajectory (train + test ids
of the probe bundle), the same extraction as generation.phase1_features / score_generation.
Stage 2 (CPU, seconds): tables + figure.

Usage (Kaggle GPU, HF_TOKEN in the environment):
    python scripts/uncertainty_early.py --stage extract --out /kaggle/working/uncertainty_early
    python scripts/uncertainty_early.py --stage analyze --out /kaggle/working/uncertainty_early
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

from src.pipeline.common import (best_f05_threshold, detection_metrics, hop_match,  # noqa: E402
                                 load_setting_examples, read_jsonl)
from src.pipeline.probes import LinearProbe, _auroc, _bootstrap_auroc_ci, _logit, hop_rows  # noqa: E402

PROBE_DIR = "outputs/pipeline/tfidf/probes_lr_C0.01_L12"
TASK = "hop12"                          # the shared probe the pipeline deploys
SIGNALS = {"mean_nll": "mean token NLL", "max_nll": "max token NLL (FLARE: min token prob)",
           "mean_ent": "mean token entropy", "max_ent": "max token entropy (DRAGIN-style)",
           "n_tok": "hop length (tokens)"}
FRACS = np.round(np.arange(0.1, 1.01, 0.1), 1)
FIRST_M = [1, 2, 3, 5, 8, 12]
MIN_ONLINE = 3                          # the online trigger may fire from the 3rd hop token on


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: teacher-forced pass
# ─────────────────────────────────────────────────────────────────────────────

def hop_signals(model, tok, prompt: str, cot: str, hops, layer: int, probe):
    """
    One teacher-forced pass (Phase 1's tokenization, as generation.phase1_features).  Per hop k:
    nll / ent = per-token NLL and predictive entropy of the hop's tokens; z = raw probe logit of the running
    mean of the block-`layer` states (z[-1] = the full hop, pooled in fp16 as the pipeline does).
    None if the prompt/generation boundary is not token-aligned.
    """
    import torch
    from src.pipeline.generation import _device, char_span_to_tokens
    dev = _device(model)
    full = tok(prompt + cot, return_tensors="pt").input_ids[0]
    p_len = len(tok(prompt, return_tensors="pt").input_ids[0])
    enc = tok(cot, return_offsets_mapping=True, add_special_tokens=False)
    if not cot or len(full) != p_len + len(enc["input_ids"]):
        return None
    offs = enc["offset_mapping"]
    with torch.inference_mode():
        o = model.model(full.unsqueeze(0).to(dev), output_hidden_states=True)
        hs, last = o.hidden_states[layer + 1][0], o.last_hidden_state[0]
        out = {}
        for k in hops:
            m = hop_match(cot, k)
            if m is None:
                continue
            ts, te = char_span_to_tokens([x[0] for x in offs], [x[1] for x in offs], m.start(1), m.end(1))
            if ts is None or te is None:
                continue
            a, b = p_len + ts, p_len + te + 1
            logp = torch.log_softmax(model.lm_head(last[a - 1:b - 1]).float(), -1)
            nll = -logp.gather(1, full[a:b].to(dev)[:, None])[:, 0]
            ent = -(logp.exp() * logp).sum(-1)
            H = hs[a:b].float()
            cum = (H.cumsum(0) / torch.arange(1, b - a + 1, device=dev)[:, None]).cpu().numpy()
            cum[-1] = H.mean(0).to(torch.float16).float().cpu().numpy()
            out[k] = dict(nll=np.round(nll.cpu().numpy(), 4).tolist(), ent=np.round(ent.cpu().numpy(), 4).tolist(),
                          z=np.round(_logit(probe.p_fail(cum)), 4).tolist())
    return out


def extract(out: Path, probe_dir: Path, limit: int | None):
    from src.pipeline.generation import load_model

    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    layer = meta["probes"][TASK]["layer"]
    probe = LinearProbe(probe_dir / f"{TASK}.npz")
    split = {i: "train" for i in meta["train_ids"]} | {i: "test" for i in meta["test_ids"]}
    examples = [e for e in load_setting_examples("tfidf") if e["id"] in split]
    rows = [r for r in hop_rows(examples, range(len(examples))) if not r["cf"]]
    by_ex: dict[int, list[dict]] = {}
    for r in rows:
        by_ex.setdefault(r["ei"], []).append(r)
    todo = list(by_ex.items())[:limit] if limit else list(by_ex.items())

    path = out / "rows.jsonl"
    done = {(r["id"], r["hop"]) for r in read_jsonl(path)} if path.exists() else set()
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    t0, n_fail = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for n, (ei, rs) in enumerate(todo):
            rs = [r for r in rs if (r["id"], r["hop"]) not in done]
            if not rs:
                continue
            e = examples[ei]
            sig = hop_signals(model, tok, e["prompt"], e.get("generated_cot", ""), [r["hop"] for r in rs],
                              layer, probe)
            if sig is None:
                n_fail += 1
                continue
            for r in rs:
                if r["hop"] not in sig:
                    continue
                rec = dict(id=r["id"], hop=r["hop"], y=r["y"], group=r["group"], hedge=r["hedge"],
                           split=split[r["id"]], **sig[r["hop"]])
                f.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"  {n}/{len(todo)} trajectories, {time.time() - t0:.0f}s", flush=True)
    print(f"done; {n_fail} trajectories skipped (prompt/generation boundary not token-aligned)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: analysis
# ─────────────────────────────────────────────────────────────────────────────

def scalar(r: dict, key: str, m: int | None = None) -> float:
    nll, ent = np.array(r["nll"][:m]), np.array(r["ent"][:m])
    return dict(mean_nll=nll.mean(), max_nll=nll.max(), mean_ent=ent.mean(), max_ent=ent.max(),
                n_tok=float(len(r["nll"])))[key]


def paired_auroc_diff(y, a, b, groups, n_boot: int = 2000, seed: int = 0):
    """AUROC(a) − AUROC(b) with a grouped bootstrap CI."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(set(y[idx].tolist())) > 1:
            vals.append(_auroc(y[idx], a[idx]) - _auroc(y[idx], b[idx]))
    return _auroc(y, a) - _auroc(y, b), np.quantile(vals, [0.025, 0.975])


def f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def analyze(out: Path, probe_dir: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    pm = meta["probes"][TASK]
    z_tau = float(_logit(pm["tau_raw"]))            # calibrated P ≥ τ  ⇔  raw logit ≥ logit(τ_raw)
    R = read_jsonl(out / "rows.jsonl")
    tr = [r for r in R if r["split"] == "train"]
    te = [r for r in R if r["split"] == "test"]
    y_tr, y_te = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
    g_te = np.array([r["group"] for r in te])
    hop_te = np.array([r["hop"] for r in te])
    P_tr, P_te = np.array([r["z"][-1] for r in tr]), np.array([r["z"][-1] for r in te])
    S_tr = {k: np.array([scalar(r, k) for r in tr]) for k in SIGNALS}
    S_te = {k: np.array([scalar(r, k) for r in te]) for k in SIGNALS}
    U_tr, U_te = np.column_stack([S_tr[k] for k in SIGNALS]), np.column_stack([S_te[k] for k in SIGNALS])

    def lr(X, y):
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                  class_weight="balanced")).fit(X, y)
    unc = lr(U_tr, y_tr)
    both = lr(np.column_stack([P_tr, U_tr]), y_tr)
    score_te = {"probe": P_te, **S_te, "unc_lr": unc.decision_function(U_te),
                "probe_unc_lr": both.decision_function(np.column_stack([P_te, U_te]))}
    score_tr = {"probe": P_tr, **S_tr, "unc_lr": unc.decision_function(U_tr),
                "probe_unc_lr": both.decision_function(np.column_stack([P_tr, U_tr]))}
    names = {"probe": f"**Probe** (hidden state, block {pm['layer']})", **SIGNALS,
             "unc_lr": "LR on all five confidence signals", "probe_unc_lr": "Probe + all five confidence signals (LR)"}
    # thresholds: the probe's deployed τ; every other signal its own F0.5-optimal threshold on the train rows
    thr = {k: (z_tau if k == "probe" else best_f05_threshold(y_tr, score_tr[k])[0]) for k in score_te}

    L = [f"# Probe vs token confidence, and how early the probe knows (TF-IDF Phase 1, probe `{probe_dir.name}`)\n",
         f"Rows: labelled hops 1–2 of the probe's split, no counterfactuals. Train {len(tr)} hops "
         f"(fail {y_tr.mean():.2f}), test {len(te)} hops (fail {y_te.mean():.2f}). Confidence signals come from the "
         f"same teacher-forced pass as the probe features, over the hop's tokens. CIs: bootstrap over question "
         f"groups. Thresholds: the probe uses its deployed τ = {pm['tau']:.3f}; every other signal gets its own "
         f"F0.5-optimal threshold on the train rows.\n",
         f"Sanity: probe test AUROC from this pass {_auroc(y_te, P_te):.4f} vs the bundle's "
         f"{pm['test']['auroc']:.4f}.\n"]

    # ── A. detection ──
    L.append("## A. Hop-failure detection on the test split\n")
    for hsel, lab in ((None, "hops 1–2"), (1, "hop 1"), (2, "hop 2")):
        s = np.ones(len(te), bool) if hsel is None else hop_te == hsel
        L.append(f"### {lab} (n = {s.sum()}, fail {y_te[s].mean():.2f})\n")
        L.append("| signal | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − signal, AUROC [95% CI] |")
        L.append("|---|---|---|---|---|---|---|")
        for k in score_te:
            sc = score_te[k][s]
            dm = detection_metrics(y_te[s], sc >= thr[k], sc)
            d = "–" if k == "probe" else \
                (lambda v: f"{v[0]:+.3f} [{v[1][0]:+.3f}, {v[1][1]:+.3f}]")(
                    paired_auroc_diff(y_te[s], score_te["probe"][s], sc, g_te[s]))
            L.append(f"| {names[k]} | {f(_auroc(y_te[s], sc))} {ci(_bootstrap_auroc_ci(y_te[s], sc, g_te[s]))} | "
                     f"{f(dm['precision'])} | {f(dm['recall'])} | {f(dm['f05'])} | {f(dm['fpr'])} | {d} |")
        L.append("")

    # failures the model is confident about: FLARE's threshold would not fire
    flare = S_te["max_nll"] >= thr["max_nll"]
    conf = ~flare
    L.append("### Failures the model is confident about\n")
    L.append(f"Hops where FLARE's rule (max token NLL ≥ {thr['max_nll']:.2f}, i.e. some token prob < "
             f"{np.exp(-thr['max_nll']):.3f}) does **not** fire: n = {conf.sum()}, of which "
             f"{int(y_te[conf].sum())} fail ({y_te[conf].mean():.2f}).\n")
    L.append("| subset | n | fail | probe AUROC [95% CI] | probe recall at τ | probe FPR at τ | mean-NLL AUROC |")
    L.append("|---|---|---|---|---|---|---|")
    for sub, lab in ((conf, "FLARE silent (model confident)"), (flare, "FLARE fires")):
        dm = detection_metrics(y_te[sub], P_te[sub] >= z_tau, P_te[sub])
        L.append(f"| {lab} | {sub.sum()} | {int(y_te[sub].sum())} | {f(_auroc(y_te[sub], P_te[sub]))} "
                 f"{ci(_bootstrap_auroc_ci(y_te[sub], P_te[sub], g_te[sub]))} | {f(dm['recall'])} | {f(dm['fpr'])} | "
                 f"{f(_auroc(y_te[sub], S_te['mean_nll'][sub]))} |")
    caught = conf & (y_te == 1) & (P_te >= z_tau)
    L.append(f"\nOf the {int((conf & (y_te == 1)).sum())} failing hops FLARE misses, the probe flags "
             f"{int(caught.sum())}. Probe-flagged hops that FLARE also flags: "
             f"{int(((P_te >= z_tau) & flare).sum())} / {int((P_te >= z_tau).sum())}.\n")
    rho = np.corrcoef(np.argsort(np.argsort(P_te)), np.argsort(np.argsort(S_te["mean_nll"])))[0, 1]
    L.append(f"Spearman ρ between the probe score and mean token NLL on the test hops: {rho:+.2f}.\n")

    # ── B. early detection ──
    L.append("## B. How early inside a hop the probe knows\n")
    L.append("The deployed probe (no retraining) on the running mean of the hop's block-12 states after the first "
             "m tokens; the confidence signals on the same prefix. Test split, hops 1–2. "
             f"Hop length: median {int(np.median(S_te['n_tok']))} tokens "
             f"(IQR {int(np.quantile(S_te['n_tok'], .25))}–{int(np.quantile(S_te['n_tok'], .75))}).\n")
    L.append("| prefix | probe AUROC [95% CI] | mean NLL | max NLL | mean entropy |")
    L.append("|---|---|---|---|---|")
    curve = {k: [] for k in ("probe", "mean_nll", "max_nll", "mean_ent")}

    def prefix_row(lab, ms):
        zp = np.array([r["z"][m - 1] for r, m in zip(te, ms)])
        vals = dict(probe=_auroc(y_te, zp), **{k: _auroc(y_te, np.array([scalar(r, k, m) for r, m in zip(te, ms)]))
                                               for k in ("mean_nll", "max_nll", "mean_ent")})
        L.append(f"| {lab} | {f(vals['probe'])} {ci(_bootstrap_auroc_ci(y_te, zp, g_te))} | {f(vals['mean_nll'])} | "
                 f"{f(vals['max_nll'])} | {f(vals['mean_ent'])} |")
        return vals
    for m in FIRST_M:
        prefix_row(f"first {m} token{'s' * (m > 1)}", [min(m, len(r["z"])) for r in te])
    for fr in FRACS:
        v = prefix_row(f"{int(fr * 100)} % of the hop", [max(1, int(np.ceil(fr * len(r["z"])))) for r in te])
        for k in curve:
            curve[k].append(v[k])
    L.append("")

    # online trigger: fire at the first prefix (≥ MIN_ONLINE tokens) whose calibrated P(fail) ≥ τ
    fire_at = []
    for r in te:
        z = np.array(r["z"])
        hit = np.flatnonzero(z[MIN_ONLINE - 1:] >= z_tau) if len(z) >= MIN_ONLINE else np.array([])
        fire_at.append(int(hit[0]) + MIN_ONLINE if len(hit) else (len(z) if z[-1] >= z_tau else None))
    on = np.array([t is not None for t in fire_at])
    end = P_te >= z_tau
    tp = on & (y_te == 1)
    frac_tp = np.array([fire_at[i] / len(te[i]["z"]) for i in np.flatnonzero(tp)])
    saved = np.array([len(te[i]["z"]) - fire_at[i] for i in np.flatnonzero(tp)])
    dm_on, dm_end = detection_metrics(y_te, on), detection_metrics(y_te, end)
    L.append(f"Online trigger (fires at the first prefix of ≥ {MIN_ONLINE} tokens with P(fail) ≥ τ, same τ):\n")
    L.append("| trigger | precision | recall | F0.5 | FPR |")
    L.append("|---|---|---|---|---|")
    for lab, dm in (("end of hop (deployed)", dm_end), ("online, within the hop", dm_on)):
        L.append(f"| {lab} | {f(dm['precision'])} | {f(dm['recall'])} | {f(dm['f05'])} | {f(dm['fpr'])} |")
    L.append(f"\nOn true failures the online trigger fires after {np.median(frac_tp) * 100:.0f} % of the hop "
             f"(median; {np.mean(frac_tp <= 0.5) * 100:.0f} % fire within the first half), saving "
             f"{saved.mean():.1f} hop tokens on average.\n")

    (out / "uncertainty_early.md").write_text("\n".join(L), encoding="utf-8")
    plot(out, score_te, names, y_te, g_te, curve)
    print("\n".join(L))


def plot(out: Path, score_te, names, y_te, g_te, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_axisbelow(True)

    ax = axes[0]
    keys = ["probe", "max_nll", "mean_nll", "max_ent", "mean_ent", "unc_lr", "n_tok"]
    short = {"probe": "probe (block 12)", "max_nll": "max token NLL (FLARE)", "mean_nll": "mean token NLL",
             "max_ent": "max entropy (DRAGIN-style)", "mean_ent": "mean entropy", "unc_lr": "LR on all 5 signals",
             "n_tok": "hop length"}
    for i, k in enumerate(keys):
        a = _auroc(y_te, score_te[k])
        lo, hi = _bootstrap_auroc_ci(y_te, score_te[k], g_te)
        col = C[0] if k == "probe" else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(a, i, "o", color=col, ms=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(hi + 0.01, i, f"{a:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(keys)), [short[k] for k in keys], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0.5, color=GRID, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("hop-failure AUROC (test split, 95% CI)", fontsize=9, color=INK)
    ax.set_title("(a) Hidden state vs token confidence", fontsize=10, color=INK, loc="left")

    ax = axes[1]
    x = FRACS * 100
    for (k, lab, ls), col in zip((("probe", "probe (running mean)", "-"), ("max_nll", "max token NLL", "--"),
                                  ("mean_nll", "mean token NLL", "-."), ("mean_ent", "mean entropy", ":")), C):
        ax.plot(x, curve[k], ls=ls, color=col, lw=2, marker="o", ms=3.5, label=lab)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("share of the hop generated (%)", fontsize=9, color=INK)
    ax.set_ylabel("hop-failure AUROC", fontsize=9, color=INK)
    ax.set_title("(b) How early the signal appears", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "uncertainty_early.png", dpi=220)
    fig.savefig(out / "uncertainty_early.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 (needs a finished Phase 3-5 run): confidence signals as end-to-end triggers
# ─────────────────────────────────────────────────────────────────────────────

E2E = ("max_nll", "mean_nll", "max_ent", "mean_ent")


def run_extract(out: Path, probe_dir: Path, run: Path):
    """Confidence signals of the run's own first-stage trajectories (s0), both datasets (GPU)."""
    from src.pipeline.evaluate import _merged
    from src.pipeline.generation import load_model
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    layer, probe = meta["probes"][TASK]["layer"], LinearProbe(probe_dir / f"{TASK}.npz")
    items = {it["uid"]: it for it in read_jsonl(run / "eval_set.jsonl")}
    s0 = [r for r in _merged(run, "s0") if r["uid"] in items]
    path = out / "run_signals.jsonl"
    done = {r["uid"] for r in read_jsonl(path)} if path.exists() else set()
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for n, r in enumerate(s0):
            if r["uid"] in done:
                continue
            sig = hop_signals(model, tok, items[r["uid"]]["prompt"], r.get("text", ""), (1, 2), layer, probe) or {}
            f.write(json.dumps(dict(uid=r["uid"], hops={str(k): {s: scalar(v, s) for s in SIGNALS} | dict(z=v["z"][-1])
                                                          for k, v in sig.items()})) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"  {n}/{len(s0)} trajectories, {time.time() - t0:.0f}s", flush=True)


def e2e(out: Path, probe_dir: Path, run: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_robustness import DATASETS, BUDGETS, Run, arr, auroc_ci, budget_threshold, row
    from src.pipeline.evaluate import _paired_ci, _relabel
    from src.pipeline.probes import ProbeBundle

    R = Run(run, force=True)
    _relabel(R.items, R.s0)
    by_ds = {ds: [it for it in R.items if it["dataset"] == ds] for ds in DATASETS}
    base = {ds: arr(R.simple("no_intervention"), by_ds[ds]) for ds in DATASETS}
    pb = ProbeBundle(probe_dir, mode="shared")
    P = R.probe_scores(pb)
    sig = {r["uid"]: r["hops"] for r in read_jsonl(out / "run_signals.jsonl")}
    S = {s: {it["uid"]: {int(k): v[s] for k, v in sig.get(it["uid"], {}).items()} for it in R.items} for s in E2E}
    # parity: the probe on this pass's features vs the run's stored s0 features (same trajectories?)
    from src.pipeline.probes import apply_platt
    dp = [abs(float(apply_platt(pb.calib[int(k)], 1 / (1 + np.exp(-v["z"])))) - P[u][int(k)])
          for u, h in sig.items() if u in P for k, v in h.items() if int(k) in P[u]]
    n_miss = sum(1 for it in R.items for k in R.eligible(it["uid"]) if k not in S["max_nll"][it["uid"]])
    # deployable thresholds: each signal's F0.5-optimal threshold on the probe's train split (Phase 1 rows)
    tr = [r for r in read_jsonl(out / "rows.jsonl") if r["split"] == "train"]
    y_tr = np.array([r["y"] for r in tr])
    thr = {s: best_f05_threshold(y_tr, np.array([scalar(r, s) for r in tr]))[0] for s in E2E}
    names = {s: SIGNALS[s] for s in E2E}
    probe = R.trigger(P, pb.tau)
    dep = {s: R.trigger(S[s], {1: thr[s], 2: thr[s]}) for s in E2E}
    acc_opt = {}
    for s in E2E:
        per, _ = R.acc_optimal(S[s])
        acc_opt[s] = R.trigger_per_item(S[s], per)
    per_p, _ = R.acc_optimal(P)
    probe_acc = R.trigger_per_item(P, per_p)

    head = ["| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |", "|---|---|---|---|---|---|"]
    L = [f"# End-to-end: probe vs token-confidence triggers (`{run.name}`)\n",
         "Same cached generations and restarts as every other system; one restart; correct = token F1 ≥ 0.5; CIs: paired "
         "bootstrap over questions. Confidence signals are computed at the same hop boundary as the probe (as cheap: no "
         "extra model call), from one teacher-forced pass over each first-stage trajectory.\n",
         f"Parity: the probe on this pass's features vs the run's stored features: |ΔP(fail)| mean {np.mean(dp):.4f}, "
         f"max {np.max(dp):.4f} over {len(dp)} hops. Restartable hops without confidence signals (never trigger): "
         f"{n_miss}.\n",
         "## 1. Deployable thresholds (each fitted on the probe's train split only)\n",
         "Probe: its deployed τ. Each confidence trigger: its own F0.5-optimal threshold on the train hops "
         "(" + ", ".join(f"{s} {thr[s]:.2f}" for s in E2E) + "). Last column vs the probe.\n"]
    for ds in DATASETS:
        its = by_ds[ds]
        ref = arr(probe, its)
        L += [f"### {ds} (n = {len(its)})", ""] + head
        L.append(row("**Probe trigger (ours)**", probe, its, base[ds]))
        for s in E2E:
            L.append(row(f"{names[s]}", dep[s], its, base[ds], ref))
        L.append(row("Always restart after hop 2", R.always_hop(2), its, base[ds], ref))
        L.append("")
    L += ["## 2. Accuracy-optimal thresholds (2-fold on 2Wiki; HotpotQA uses all 2Wiki), last column vs the "
          "accuracy-optimal probe\n"]
    for ds in DATASETS:
        its = by_ds[ds]
        ref = arr(probe_acc, its)
        L += [f"### {ds}", ""] + head
        L.append(row("**Probe, accuracy-optimal**", probe_acc, its, base[ds]))
        for s in E2E:
            L.append(row(f"{names[s]}, accuracy-optimal", acc_opt[s], its, base[ds], ref))
        L.append("")
    L += ["## 3. Matched restart budget. Cells: accuracy (probe − trigger [95% CI])\n"]
    scorers = {"Probe": P, **{names[s]: S[s] for s in E2E}}
    for ds in DATASETS:
        its = by_ds[ds]
        el = [R.eligible(it["uid"]) for it in its]
        L += [f"### {ds}", "", "| budget | " + " | ".join(scorers) + " |", "|---" * (len(scorers) + 1) + "|"]
        for b in BUDGETS:
            res = {}
            for n, sc in scorers.items():
                t = budget_threshold([sc[it["uid"]] for it in its], el, b)
                res[n] = arr(R.trigger(sc, {1: t, 2: t}), its)
            ref = res["Probe"]
            cells = [f"**{ref.mean():.3f}**"] + ["{:.3f} ({:+.3f} [{:+.3f}, {:+.3f}])".format(
                c.mean(), ref.mean() - c.mean(), *_paired_ci(ref, c)) for n, c in res.items() if n != "Probe"]
            L.append(f"| {int(b * 100)}% | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 4. Hop-level detection on live 2Wiki hops, AUROC [95% CI]\n",
          "| hop | n | fail | probe | " + " | ".join(names[s] for s in E2E) + " |", "|---" * (4 + len(E2E)) + "|"]
    for k in (1, 2):
        rk = [it for it in by_ds["2wiki"] if it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]]
              and all(k in S[s][it["uid"]] for s in E2E)]
        y = [it["labels"][str(k)] for it in rk]
        cells = [auroc_ci(y, [P[it["uid"]][k] for it in rk])] + [auroc_ci(y, [S[s][it["uid"]][k] for it in rk]) for s in E2E]
        L.append(f"| {k} | {len(rk)} | {sum(y)} | " + " | ".join(f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in cells) + " |")
    (out / "e2e_confidence.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "analyze", "run-extract", "e2e", "all"], default="all",
                    help="'all' = extract + analyze, plus run-extract + e2e when --run is given")
    ap.add_argument("--out", default="outputs/pipeline/tfidf/uncertainty_early")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    ap.add_argument("--run", default=None, help="finished Phase 3-5 run folder (eval_set.jsonl, s0*.jsonl, restarts...)")
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories only (smoke test)")
    args = ap.parse_args()
    out, probe_dir = Path(args.out), ROOT / args.probe_dir
    run = Path(args.run) if args.run else None
    out.mkdir(parents=True, exist_ok=True)
    st = args.stage
    if st in ("extract", "all"):
        extract(out, probe_dir, args.limit)
    if st in ("analyze", "all"):
        analyze(out, probe_dir)
    if run is not None and st in ("run-extract", "all"):
        run_extract(out, probe_dir, run)
    if run is not None and st in ("e2e", "all"):
        e2e(out, probe_dir, run)


if __name__ == "__main__":
    main()
