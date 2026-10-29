"""
scripts/eval_robustness.py
──────────────────────────
Robustness checks and extra analyses for a finished Phase 3-5 run, all on the SAME cached generations
(CPU only; nothing is re-generated).  Every system acts on the same s0 trajectories and restart outputs,
so differences come from the trigger alone.

  1  fair thresholds     every judge gets the same 2-fold accuracy-optimal (θ1, θ2) the probe gets
  2  old vs tuned probe  the deployed C=1 / block-10 probe re-scored inside this run (same generations,
                         same answer forcing), so the probe change is not confounded with other changes
  3  answer forcing      accuracy of every system with and without the forced answers, and how many
                         final answers each system takes from forcing
  4  matched budget      accuracy at a fixed restart rate for every trigger, with paired-bootstrap CIs
  5  τ sensitivity       end-to-end accuracy of the probe over a range of τ (shared by both hops)
  6  detection CIs       hop-level AUROC with 95% CIs, all rows and Phase-1-labelled rows only
                         (the relabeled trajectories used a weaker presence check)
  7  repair verifier     does the probe still work on the RESTARTED trajectories (extra evidence in the
                         prompt), and does its score on the restart say whether the repair worked?
  8  reliability         calibration of the probe and the judges on live 2Wiki hops (ECE, Brier, figure)
  9  restart yield       restarts by hop: fixed / broken; 2Wiki: did the repair retrieve the missing gold
                         paragraph, and how accurate is the restart when it did / did not
  10 HotpotQA by type    bridge vs comparison questions (the probe was trained on 2Wiki bridge chains)
  11 probe errors        what the probe misses and its false alarms

Usage (on Kaggle CPU, with the run's output folder attached):
    python scripts/eval_robustness.py --run /kaggle/input/<...>/pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12 \
        --probe-dir outputs/pipeline/tfidf/probes_lr_C0.01_L12 --old-probe-dir outputs/pipeline/tfidf/probes \
        --out outputs/pipeline/tfidf_final_lrC0.01_L12/robustness
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.pipeline.common import (answer_correct, detection_metrics, hop_order_condition, is_hedge,  # noqa: E402
                                 load_setting_examples, read_jsonl)
from src.pipeline.evaluate import _acc_optimal_tau, _fold, _merged, _paired_ci, _relabel, _safe  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import outcome  # noqa: E402

DATASETS = ("2wiki", "hotpotqa")
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TAUS = np.round(np.arange(0.30, 0.91, 0.05), 2)
MAIN_JUDGE = "Llama-3.2-3B-Instruct"          # judge.jsonl = the Doctor-RAG-style judge with the base model


# ─────────────────────────────────────────────────────────────────────────────
# Loading (same rules as src/pipeline/evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run: Path, force: bool):
    items_all = read_jsonl(run / "eval_set.jsonl")
    s0 = {r["uid"]: r for r in _merged(run, "s0")}
    rs = defaultdict(dict)
    for r in _merged(run, "restarts") + _merged(run, "restarts2"):
        rs[r["uid"]][r["key"]] = r
    for sc in _merged(run, "rescore"):
        if sc["key"] in rs[sc["uid"]]:
            rs[sc["uid"]][sc["key"]]["rescore"] = sc
    if force:
        for f in _merged(run, "forced"):
            rec = s0.get(f["uid"]) if f["key"] == "s0" else rs[f["uid"]].get(f["key"])
            if rec is not None and not rec["answer"]:
                rec["answer"], rec["answer_forced"] = f["answer"], True
                rec["n_gen"] += f["n_gen"]
    items = [it for it in items_all if it["uid"] in s0 and "q" in rs[it["uid"]]
             and all(f"h{k}" in rs[it["uid"]] for k in ("1", "2")
                     if k in s0[it["uid"]]["boundaries"] and s0[it["uid"]]["hop_text"][k])]
    judges = {MAIN_JUDGE: {r["uid"]: r for r in _merged(run, "judge")}}
    for f in sorted(run.glob("judge__*.jsonl")):
        for r in read_jsonl(f):
            judges.setdefault(r["model"].split("/")[-1], {})[r["uid"]] = r
    return items, s0, rs, judges


class Run:
    def __init__(self, run: Path, force: bool = True):
        self.items, self.s0, self.rs, self.judges = load_run(run, force)
        self.run = run

    def eligible(self, uid):
        """Hops with a closed boundary, non-empty text and a restart record (where a restart can happen)."""
        s = self.s0[uid]
        return {k for k in (1, 2) if str(k) in s["boundaries"] and s["hop_text"].get(str(k)) and f"h{k}" in self.rs[uid]}

    def probe_scores(self, bundle):
        out = {}
        for it in self.items:
            z = np.load(self.run / "s0_feats" / f"{_safe(it['uid'])}.npz")
            out[it["uid"]] = {k: bundle.p_fail(k, z[f"h{k}"]) for k in (1, 2) if f"h{k}" in z.files}
        return out

    def judge_scores(self, name):
        J = self.judges[name]
        return {it["uid"]: {int(k): v["p_fail"] for k, v in (J.get(it["uid"]) or {}).items() if k in ("1", "2")}
                for it in self.items}

    # systems ------------------------------------------------------------
    def trigger(self, scores, tau):
        """Restart at the earliest eligible hop with score >= tau[k] (accounting of the probe system)."""
        return {it["uid"]: outcome("probe_restart", it, self.s0[it["uid"]], self.rs[it["uid"]],
                                   probe_scores={k: v for k, v in scores[it["uid"]].items() if k in self.eligible(it["uid"])},
                                   tau=tau) for it in self.items}

    def judge_system(self, name, theta=None):
        """The judge system (post-hoc, judge-token accounting); theta = {k: θk} per item uid, or None = 0.5."""
        J = self.judges[name]
        out = {}
        for it in self.items:
            u, rec = it["uid"], J.get(it["uid"])
            if rec is not None and theta is not None:
                el = self.eligible(u)
                rec = {k: dict(v, p_fail=float(int(k) in el and v["p_fail"] >= theta[u][int(k)])) if k in ("1", "2") else v
                       for k, v in rec.items()}
            out[u] = outcome("judge_restart", it, self.s0[u], self.rs[u], judge_rec=rec)
        return out

    def simple(self, system):
        return {it["uid"]: outcome(system, it, self.s0[it["uid"]], self.rs[it["uid"]]) for it in self.items}

    def always_hop(self, k):
        return self.trigger({it["uid"]: {k: 1.0} for it in self.items}, {1: 0.5, 2: 0.5})

    def acc_arrays(self, its, scores):
        def corr(it, key):
            r = self.rs[it["uid"]].get(key)
            return answer_correct(r["answer"], it["gold_answer"]) if r else None
        c0 = np.array([answer_correct(self.s0[it["uid"]]["answer"], it["gold_answer"]) for it in its], float)
        a = dict(c0=c0)
        for k in (1, 2):
            ck = [corr(it, f"h{k}") if k in self.eligible(it["uid"]) else None for it in its]
            a[f"c{k}"] = np.array([c0[i] if v is None else v for i, v in enumerate(ck)], float)
            a[f"p{k}"] = np.array([scores[it["uid"]].get(k, -1.0) if ck[i] is not None else -1.0
                                   for i, it in enumerate(its)], float)
        return a

    def acc_optimal(self, scores):
        """Per-item (θ1, θ2): 2-fold CV over 2Wiki questions; HotpotQA uses the fit on all 2Wiki (as evaluate.py)."""
        wiki = [it for it in self.items if it["dataset"] == "2wiki"]
        folds = {f: [it for it in wiki if _fold(it["uid"]) == f] for f in (0, 1)}
        fit = {f: _acc_optimal_tau(self.acc_arrays(folds[1 - f], scores)) for f in (0, 1) if folds[1 - f]}
        fit_all = _acc_optimal_tau(self.acc_arrays(wiki, scores))
        per = {}
        for it in self.items:
            t = fit.get(_fold(it["uid"]), fit_all) if it["dataset"] == "2wiki" else fit_all
            per[it["uid"]] = {1: t[0], 2: t[1]}
        return per, dict(fold0=fit.get(0), fold1=fit.get(1), all_2wiki=fit_all)

    def trigger_per_item(self, scores, per_tau):
        return {it["uid"]: outcome("probe_restart", it, self.s0[it["uid"]], self.rs[it["uid"]],
                                   probe_scores={k: v for k, v in scores[it["uid"]].items() if k in self.eligible(it["uid"])},
                                   tau=per_tau[it["uid"]]) for it in self.items}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def arr(o, its, key="correct"):
    return np.array([float(o[it["uid"]][key] or 0) for it in its])


def row(name, o, its, base, ref=None):
    c = arr(o, its)
    fired = np.mean([o[it["uid"]]["fired"] for it in its])
    fixed, broken = int(((c == 1) & (base == 0)).sum()), int(((c == 0) & (base == 1)).sum())
    d = "–" if ref is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(ref.mean() - c.mean(), *_paired_ci(ref, c))
    return f"| {name} | {c.mean():.3f} | {fired:.2f} | {fixed} | {broken} | {d} |"


def auroc_ci(y, s, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) == 2:
            b.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((b, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return float(e), float(np.mean((p - y) ** 2)), rows


def budget_threshold(scores, eligible, b):
    """Threshold giving a restart rate as close to b as possible (restart if any eligible hop >= t)."""
    mx = np.array([max([v for k, v in s.items() if k in el], default=-np.inf) for s, el in zip(scores, eligible)])
    srt = np.sort(mx[np.isfinite(mx)])[::-1]
    if not len(srt):
        return np.inf
    n = int(round(b * len(mx)))
    return float(srt[min(max(n, 1), len(srt)) - 1])


def matcher():
    try:
        from utils.matching import EntityMatcher
        return EntityMatcher(aliases={}, sbert_threshold=0.70)
    except Exception as e:                                         # noqa: BLE001
        print(f"[robustness] no entity matcher ({e.__class__.__name__}); restart-hop labels skipped")
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run's output folder (eval_set.jsonl, s0.jsonl, s0_feats/, ...)")
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes_lr_C0.01_L12", help="the run's probe")
    ap.add_argument("--old-probe-dir", default="outputs/pipeline/tfidf/probes", help="deployed C=1 / block-10 probe")
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/robustness")
    ap.add_argument("--no-text-trigger", action="store_true", help="skip training the TF-IDF text trigger")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    R = Run(Path(args.run), force=True)
    items, s0, rs = R.items, R.s0, R.rs
    _relabel(items, s0)
    by_ds = {ds: [it for it in items if it["dataset"] == ds] for ds in DATASETS}
    base = {ds: arr(R.simple("no_intervention"), by_ds[ds]) for ds in DATASETS}
    pb = ProbeBundle(ROOT / args.probe_dir, mode="shared")
    pb_old = ProbeBundle(ROOT / args.old_probe_dir, mode="shared")
    P, P_old = R.probe_scores(pb), R.probe_scores(pb_old)
    J = {n: R.judge_scores(n) for n in R.judges}
    print(f"[robustness] {len(items)} items; judges: {list(J)}", flush=True)

    probe = R.trigger(P, pb.tau)
    L = [f"# Robustness checks and extra analyses (`{Path(args.run).name}`)", "",
         f"Items: {len(items)} ({', '.join(f'{ds} {len(v)}' for ds, v in by_ds.items())}). All systems act on the same "
         f"cached generations; correct = token F1 ≥ 0.5; CIs are paired bootstraps over questions (2,000 resamples). "
         f"Probe: `{args.probe_dir}` (τ = {pb.tau[1]:.3f}); old probe: `{args.old_probe_dir}` (layer {pb_old.layer[1]}, "
         f"τ = {pb_old.tau[1]:.3f}).", ""]
    head = ["| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |", "|---|---|---|---|---|---|"]

    # ── 1. fair thresholds ───────────────────────────────────────────────────
    per_p, fit_p = R.acc_optimal(P)
    probe_acc = R.trigger_per_item(P, per_p)
    judge_def = {n: R.judge_system(n) for n in J}
    judge_acc, fits = {}, {"probe": fit_p}
    for n in J:
        per_j, fits[n] = R.acc_optimal(J[n])
        judge_acc[n] = R.judge_system(n, per_j)
    L += ["## 1. Fair thresholds: every trigger with the same accuracy-optimal (θ1, θ2)", "",
          "The headline compares the probe at its train-CV τ with judges at their own yes/no decision (θ = 0.5). Here every "
          "trigger also gets the procedure the probe's 'accuracy-optimal τ' row uses: (θ1, θ2) maximising accuracy on one "
          "half of the 2Wiki questions, applied to the other half; HotpotQA uses the fit on all 2Wiki. The last column "
          "compares with the accuracy-optimal probe.", "",
          "Fitted thresholds (fold 0 / fold 1 / all 2Wiki): " + "; ".join(f"{k}: {v['fold0']} / {v['fold1']} / {v['all_2wiki']}"
                                                                            for k, v in fits.items()), ""]
    for ds in DATASETS:
        its = by_ds[ds]
        ref = arr(probe_acc, its)
        L += [f"### {ds} (n = {len(its)})", ""] + head
        L.append(row("Probe, train-CV τ (headline)", probe, its, base[ds], ref))
        L.append(row("**Probe, accuracy-optimal**", probe_acc, its, base[ds]))
        for n in J:
            L.append(row(f"Judge {n}, θ = 0.5 (headline)", judge_def[n], its, base[ds], ref))
            L.append(row(f"Judge {n}, accuracy-optimal", judge_acc[n], its, base[ds], ref))
        L.append(row("Always restart after hop 2", R.always_hop(2), its, base[ds], ref))
        L.append("")

    # ── 2. old vs tuned probe in the same run ────────────────────────────────
    old = R.trigger(P_old, pb_old.tau)
    per_o, fit_o = R.acc_optimal(P_old)
    old_acc = R.trigger_per_item(P_old, per_o)
    L += ["## 2. Old (C = 1, block 10) vs tuned (C = 0.01, block 12) probe on the same generations", "",
          f"Old probe accuracy-optimal (θ1, θ2): {fit_o}. Last column: tuned − old.", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        L += [f"### {ds}", "", "| threshold | tuned | old | tuned − old [95% CI] | restart rate tuned / old |", "|---|---|---|---|---|"]
        for name, a, b in (("train-CV τ", probe, old), ("accuracy-optimal", probe_acc, old_acc)):
            ca, cb = arr(a, its), arr(b, its)
            L.append(f"| {name} | {ca.mean():.3f} | {cb.mean():.3f} | {ca.mean() - cb.mean():+.3f} "
                     f"[{_paired_ci(ca, cb)[0]:+.3f}, {_paired_ci(ca, cb)[1]:+.3f}] | "
                     f"{np.mean([a[it['uid']]['fired'] for it in its]):.2f} / {np.mean([b[it['uid']]['fired'] for it in its]):.2f} |")
        L.append("")
    wiki = by_ds["2wiki"]
    L += ["Hop-level detection on live 2Wiki hops (AUROC [95% CI]):", "", "| hop | n | fail | tuned | old |", "|---|---|---|---|---|"]
    for k in (1, 2):
        rows_k = [it for it in wiki if it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]] and k in P_old[it["uid"]]]
        y = [it["labels"][str(k)] for it in rows_k]
        a, b = auroc_ci(y, [P[it["uid"]][k] for it in rows_k]), auroc_ci(y, [P_old[it["uid"]][k] for it in rows_k])
        L.append(f"| {k} | {len(rows_k)} | {sum(y)} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] |")
    L.append("")

    # ── 3. answer forcing ────────────────────────────────────────────────────
    R0 = Run(Path(args.run), force=False)
    systems = {"No intervention": lambda r: r.simple("no_intervention"),
               "Probe trigger (ours)": lambda r: r.trigger(P, pb.tau),
               "Always retrieve": lambda r: r.simple("always_retrieve"),
               "Always restart after hop 2": lambda r: r.always_hop(2),
               **{f"Judge {n}": (lambda r, n=n: r.judge_system(n)) for n in J}}
    L +=["## 3. Answer forcing: accuracy with and without forced answers", "",
          "Generations that hit the token cap without `<answer>` get an answer forced from hops 1–2 (stage `force`). "
          "'without' scores those as empty (wrong). 'forced' = how many final answers of the system come from forcing.", ""]
    with_f = {n: f(R) for n, f in systems.items()}
    no_f = {n: f(R0) for n, f in systems.items()}
    for ds in DATASETS:
        its = by_ds[ds]
        pr_w, pr_n = arr(with_f["Probe trigger (ours)"], its), arr(no_f["Probe trigger (ours)"], its)
        L += [f"### {ds}", "", "| system | with forcing | without | forced answers | probe − system, without forcing [95% CI] |",
              "|---|---|---|---|---|"]
        for n in systems:
            cw, cn = arr(with_f[n], its), arr(no_f[n], its)
            nf = sum(with_f[n][it["uid"]]["answer"] != no_f[n][it["uid"]]["answer"] for it in its)
            d = "–" if n.startswith("Probe") else "{:+.3f} [{:+.3f}, {:+.3f}]".format(pr_n.mean() - cn.mean(), *_paired_ci(pr_n, cn))
            L.append(f"| {n} | {cw.mean():.3f} | {cn.mean():.3f} | {nf} | {d} |")
        L.append("")

    # ── 4. matched budget with CIs ───────────────────────────────────────────
    scorers = {"Probe (tuned)": P, "Probe (old)": P_old, **{f"Judge {n}": J[n] for n in J}}
    if not args.no_text_trigger:
        try:
            from trigger_baselines import train_text_trigger
            meta = json.loads((ROOT / args.probe_dir / "probes_meta.json").read_text())
            tm, _, _ = train_text_trigger(meta["setting"])
            scorers["Text trigger (TF-IDF+LR)"] = {
                it["uid"]: {k: float(tm.predict_proba([f"hop{k} {s0[it['uid']]['hop_text'][str(k)]}"])[0, 1])
                            for k in R.eligible(it["uid"])} for it in items}
        except Exception as e:                                     # noqa: BLE001
            print(f"[robustness] text trigger skipped ({e.__class__.__name__}: {e})")
    L += ["## 4. Accuracy at a matched restart budget (paired-bootstrap CIs)", "",
          "Each trigger restarts at its earliest hop whose score exceeds the threshold that gives the budget's restart rate "
          "(one threshold for both hops). Judges' token cost is ignored here (they need the full answer first): only their "
          "choice of questions and hops is compared. Cells: accuracy (probe − trigger [95% CI]).", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        el = [R.eligible(it["uid"]) for it in its]
        names = list(scorers)
        L += [f"### {ds}", "", "| budget | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for b in BUDGETS:
            res = {}
            for n, S in scorers.items():
                t = budget_threshold([S[it["uid"]] for it in its], el, b)
                o = R.trigger(S, {1: t, 2: t})
                res[n] = (arr(o, its), np.mean([o[it["uid"]]["fired"] for it in its]))
            ref = res["Probe (tuned)"][0]
            cells = []
            for n in names:
                c, rate = res[n]
                if n == "Probe (tuned)":
                    cells.append(f"**{c.mean():.3f}** (rate {rate:.2f})")
                else:
                    lo, hi = _paired_ci(ref, c)
                    cells.append(f"{c.mean():.3f} ({ref.mean() - c.mean():+.3f} [{lo:+.3f}, {hi:+.3f}])")
            L.append(f"| {int(b * 100)}% | " + " | ".join(cells) + " |")
        L.append("")

    # ── 5. τ sensitivity ─────────────────────────────────────────────────────
    jd = judge_def[MAIN_JUDGE]
    L += ["## 5. Sensitivity of the probe's end-to-end accuracy to τ (one τ for both hops)", "",
          f"The same model gave τ = 0.585 or 0.650 depending only on the CV folds, so the result should not hinge on one τ. "
          f"Last columns: probe − {MAIN_JUDGE} judge (θ = 0.5).", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        cj = arr(jd, its)
        L += [f"### {ds}", "", "| τ | accuracy | restart rate | broken | probe − judge [95% CI] |", "|---|---|---|---|---|"]
        for t in TAUS:
            o = R.trigger(P, {1: float(t), 2: float(t)})
            c = arr(o, its)
            lo, hi = _paired_ci(c, cj)
            L.append(f"| {t:.2f} | {c.mean():.3f} | {np.mean([o[it['uid']]['fired'] for it in its]):.2f} | "
                     f"{int(((c == 0) & (base[ds] == 1)).sum())} | {c.mean() - cj.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        L.append("")

    # ── 6. detection with CIs, Phase-1 labels only ───────────────────────────
    L += ["## 6. Hop-level detection on live 2Wiki hops, AUROC [95% CI]", "",
          "'phase1 only' drops the trajectories that did not reproduce the stored Phase 1 text and were relabeled with a "
          "presence check (no LLM tier).", "",
          "| hop | rows | n | fail | probe (tuned) | " + " | ".join(f"judge {n}" for n in J) + " |", "|---" * (5 + len(J)) + "|"]
    for k in (1, 2):
        for sub, keep in (("all", lambda it: True), ("phase1 only", lambda it: it.get("label_source") == "phase1")):
            rows_k = [it for it in wiki if keep(it) and it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]]
                      and all(k in J[n][it["uid"]] for n in J)]
            y = [it["labels"][str(k)] for it in rows_k]
            cells = []
            for S in [P] + [J[n] for n in J]:
                a = auroc_ci(y, [S[it["uid"]][k] for it in rows_k])
                cells.append(f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]")
            L.append(f"| {k} | {sub} | {len(rows_k)} | {sum(y)} | " + " | ".join(cells) + " |")
    L.append("")

    # ── 7. repair verifier ───────────────────────────────────────────────────
    M = matcher()
    L += ["## 7. The probe on restarted trajectories (repair verifier)", "",
          "Every hop-1 / hop-2 restart was re-scored by the probe during the run (stage `rescore`; the restart prompt "
          "contains the retrieved evidence, so this is a distribution shift for the probe).", ""]
    for ds in DATASETS:
        y_ans, p_ans, y_s0, p_s0 = [], [], [], []
        for it in by_ds[ds]:
            u = it["uid"]
            for key in ("h1", "h2"):
                r = rs[u].get(key)
                sc = ((r or {}).get("rescore") or {}).get("probe") or {}
                if r is None or not sc:
                    continue
                y_ans.append(int(not answer_correct(r["answer"], it["gold_answer"])))
                p_ans.append(max(v["p_fail"] for v in sc.values()))
            if P[u]:
                y_s0.append(int(not answer_correct(s0[u]["answer"], it["gold_answer"])))
                p_s0.append(max(P[u].values()))
        a, b = auroc_ci(y_ans, p_ans), auroc_ci(y_s0, p_s0)
        L.append(f"- **{ds}**: max hop P(fail) predicts a wrong final answer — first trajectories AUROC {b[0]:.3f} "
                 f"[{b[1]:.3f}, {b[2]:.3f}] (n = {len(y_s0)}); restarted trajectories {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] (n = {len(y_ans)})")
    # verifier: among probe-triggered restarts, accuracy when the probe still flags the restart vs when it does not
    L += ["", "Probe-triggered restarts (one round), split by whether the probe still flags the restarted trajectory "
          "(any hop ≥ τ): if the probe is a useful verifier, flagged restarts should be much less accurate.", "",
          "| dataset | restarts | still flagged | accuracy, still flagged | accuracy, cleared | answer fixed (cleared / flagged) |",
          "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        fl, cl = [], []
        for it in by_ds[ds]:
            o = probe[it["uid"]]
            if not o["fired"]:
                continue
            r = rs[it["uid"]].get(f"h{o['hop']}")
            sc = ((r or {}).get("rescore") or {}).get("probe") or {}
            if not sc:
                continue
            ok = answer_correct(r["answer"], it["gold_answer"])
            was = answer_correct(s0[it["uid"]]["answer"], it["gold_answer"])
            (fl if any(v["p_fail"] >= pb.tau[int(k)] for k, v in sc.items()) else cl).append((ok, ok and not was))
        f_ = lambda x, i: np.mean([v[i] for v in x]) if x else float("nan")  # noqa: E731
        L.append(f"| {ds} | {len(fl) + len(cl)} | {len(fl)} | {f_(fl, 0):.3f} | {f_(cl, 0):.3f} | "
                 f"{sum(v[1] for v in cl)} / {sum(v[1] for v in fl)} |")
    if M is not None:
        y, p = defaultdict(list), defaultdict(list)
        for it in wiki:
            for key in ("h1", "h2"):
                r = rs[it["uid"]].get(key)
                sc = ((r or {}).get("rescore") or {}).get("probe") or {}
                hops = ((r or {}).get("rescore") or {}).get("hop_text") or {}
                for k in ("1", "2"):
                    g, t = it["gold_entities"].get(k, ""), hops.get(k, "")
                    if g.strip() and t and k in sc:
                        y[k].append(0 if M.match(g, t).matched else 1); p[k].append(sc[k]["p_fail"])
        L += ["", "Hop-level detection on the restarted 2Wiki trajectories (labels: presence check of the gold bridge entity, "
              "no LLM tier):", "", "| hop | n | fail | probe AUROC [95% CI] | recall | FPR |", "|---|---|---|---|---|---|"]
        for k in ("1", "2"):
            if y[k]:
                a = auroc_ci(y[k], p[k])
                dm = detection_metrics(y[k], np.array(p[k]) >= pb.tau[int(k)], p[k])
                L.append(f"| {k} | {len(y[k])} | {sum(y[k])} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | "
                         f"{dm['recall']:.3f} | {dm['fpr']:.3f} |")
    L.append("")

    # ── 8. reliability ───────────────────────────────────────────────────────
    L += ["## 8. Calibration on live 2Wiki hops 1–2", "", "| trigger | n | ECE | Brier |", "|---|---|---|---|"]
    rel = {}
    for name, S in [("Probe (tuned)", P), ("Probe (old)", P_old)] + [(f"Judge {n}", J[n]) for n in J]:
        y, p = [], []
        for it in wiki:
            for k in (1, 2):
                if it["labels"].get(str(k)) in (0, 1) and k in S[it["uid"]]:
                    y.append(it["labels"][str(k)]); p.append(S[it["uid"]][k])
        e, br, rows_b = ece(y, p)
        rel[name] = rows_b
        L.append(f"| {name} | {len(y)} | {e:.3f} | {br:.3f} |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1)
        for name, rows_b in rel.items():
            ax.plot([r[2] for r in rows_b], [r[3] for r in rows_b], "-o", ms=3, lw=1.5,
                    label=name, color="#111827" if name == "Probe (tuned)" else None)
        ax.set_xlabel("predicted P(fail)"); ax.set_ylabel("observed failure rate")
        ax.set_title("Reliability on live 2Wiki hops"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=200); plt.close(fig)
        L += ["", "Figure: `reliability.png`."]
    except Exception as e:                                         # noqa: BLE001
        print(f"[robustness] reliability figure skipped ({e})")
    L.append("")

    # ── 9. restart yield and retrieval hits ──────────────────────────────────
    meta = json.loads((ROOT / args.probe_dir / "probes_meta.json").read_text())
    try:
        ex = {e["id"]: e for e in load_setting_examples(meta["setting"])}
    except Exception as e:                                         # noqa: BLE001
        print(f"[robustness] dataset not loadable ({e.__class__.__name__}); retrieval hits skipped")
        ex = {}
    L += ["## 9. Where the probe restarts and what the repair retrieves", "",
          "| dataset | restart at | n | accuracy before | after | fixed | broken |", "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for k in (1, 2):
            its = [it for it in by_ds[ds] if probe[it["uid"]]["hop"] == k]
            if not its:
                continue
            c0 = np.array([answer_correct(s0[it["uid"]]["answer"], it["gold_answer"]) for it in its], float)
            c1 = arr(probe, its)
            L.append(f"| {ds} | hop {k} | {len(its)} | {c0.mean():.3f} | {c1.mean():.3f} | "
                     f"{int(((c1 == 1) & (c0 == 0)).sum())} | {int(((c1 == 0) & (c0 == 1)).sum())} |")
    if ex:
        L += ["", "2Wiki: the gold paragraph(s) missing from the first-stage context, and whether the restart's retrieval "
              "brought one back (all hop-1 / hop-2 restarts, whatever the trigger):", "",
              "| restart at | n (≥ 1 gold paragraph missing) | retrieval found a missing gold paragraph | accuracy if found | if not found |",
              "|---|---|---|---|---|"]
        for k in (1, 2):
            found, miss = [], []
            for it in wiki:
                e, r = ex.get(it["id"]), rs[it["uid"]].get(f"h{k}")
                if e is None or r is None:
                    continue
                missing = set(e.get("gold_titles") or []) - set(it["context_titles"])
                if not missing:
                    continue
                ok = answer_correct(r["answer"], it["gold_answer"])
                (found if missing & set(r.get("retrieved_titles") or []) else miss).append(ok)
            n = len(found) + len(miss)
            if n:
                L.append(f"| hop {k} | {n} | {len(found) / n:.3f} | {np.mean(found) if found else float('nan'):.3f} | "
                         f"{np.mean(miss) if miss else float('nan'):.3f} |")
    L.append("")

    # ── 10. HotpotQA by question type ────────────────────────────────────────
    hot = by_ds["hotpotqa"]
    if hot and any(it.get("hotpot_type") for it in hot):
        L += ["## 10. HotpotQA by question type (the probe was trained on 2Wiki bridge chains)", "",
              "| type | n | none | probe | judge (θ 0.5) | always hop 2 | probe − none [95% CI] | probe − judge [95% CI] | probe restart rate |",
              "|---|---|---|---|---|---|---|---|---|"]
        ah2 = R.always_hop(2)
        for t in sorted({it.get("hotpot_type") for it in hot if it.get("hotpot_type")}):
            its = [it for it in hot if it.get("hotpot_type") == t]
            b0, cp, cj, ca = arr(R.simple("no_intervention"), its), arr(probe, its), arr(jd, its), arr(ah2, its)
            L.append(f"| {t} | {len(its)} | {b0.mean():.3f} | {cp.mean():.3f} | {cj.mean():.3f} | {ca.mean():.3f} | "
                     "{:+.3f} [{:+.3f}, {:+.3f}] | {:+.3f} [{:+.3f}, {:+.3f}] | ".format(
                         cp.mean() - b0.mean(), *_paired_ci(cp, b0), cp.mean() - cj.mean(), *_paired_ci(cp, cj))
                     + f"{np.mean([probe[it['uid']]['fired'] for it in its]):.2f} |")
        L.append("")

    # ── 11. probe errors ─────────────────────────────────────────────────────
    L += ["## 11. What the probe misses and its false alarms (live 2Wiki hops, τ = train-CV)", "",
          "| hop | failing hops | missed | missed: hedge | missed: final answer still correct | false alarms | "
          "false alarms: final answer wrong anyway |", "|---|---|---|---|---|---|---|"]
    cond = Counter()
    for k in (1, 2):
        rows_k = [it for it in wiki if it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]]]
        fail = [it for it in rows_k if it["labels"][str(k)] == 1]
        succ = [it for it in rows_k if it["labels"][str(k)] == 0]
        miss = [it for it in fail if P[it["uid"]][k] < pb.tau[k]]
        fa = [it for it in succ if P[it["uid"]][k] >= pb.tau[k]]
        # first-stage condition in TRUE hop order (the stored labels are alphabetical; see common.hop_order_condition)
        cond.update((k, hop_order_condition(ex[it["id"]], it["context_titles"]) if it["id"] in ex
                     else it.get("retrieval_condition")) for it in miss)
        ok = lambda it: answer_correct(s0[it["uid"]]["answer"], it["gold_answer"])  # noqa: E731
        pct = lambda x, n: f"{x} ({x / n:.0%})" if n else "0"  # noqa: E731
        L.append(f"| {k} | {len(fail)} | {pct(len(miss), len(fail))} | "
                 f"{pct(sum(is_hedge(s0[it['uid']]['hop_text'][str(k)]) for it in miss), len(miss))} | "
                 f"{pct(sum(ok(it) for it in miss), len(miss))} | {pct(len(fa), len(succ))} | "
                 f"{pct(sum(not ok(it) for it in fa), len(fa))} |")
    L += ["", "Missed failing hops by first-stage retrieval condition: "
          + ", ".join(f"hop {k} {c}: {n}" for (k, c), n in sorted(cond.items(), key=lambda x: (x[0][0], -x[1]))), ""]

    md = "\n".join(L) + "\n"
    (out / "robustness.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
