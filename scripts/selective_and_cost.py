"""
scripts/selective_and_cost.py
─────────────────────────────
Two more views of a finished Phase 3-5 run (CPU, ~1 min; same cached generations as every other table):

  1  selective answering   rank questions by the trigger's own failure score (max hop P(fail)) and answer only
                           the most confident share: accuracy at 50 / 80 % coverage and the area under the
                           risk-coverage curve (AURC, lower = better), probe vs every judge, paired-bootstrap CIs.
                           (a) on the first answers, (b) on each system's final answers, scored on the final
                           trajectory (the restart's own rescore when it restarted).
  2  accuracy vs cost      accuracy against tokens processed per question for every system (figure + table).

Usage:
    python scripts/selective_and_cost.py --run <run folder> --out outputs/pipeline/tfidf_final_lrC0.01_L12/selective
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_robustness import DATASETS, MAIN_JUDGE, Run, arr  # noqa: E402
from src.pipeline.evaluate import _relabel  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import RESTART_KEY, outcome  # noqa: E402

COVERAGES = (0.5, 0.8)
# label offsets (points) where labels would collide in the cost panels
NUDGE = {("2wiki", "Oracle trigger"): (-6, 4), ("2wiki", "Probe (ours)"): (6, -9),
         ("2wiki", "Judge Llama-3.2-3B-Instruct"): (-6, -10), ("2wiki", "Judge Qwen2.5-3B-Instruct"): (6, -10)}
N_BOOT = 2000


def max_score(d: dict) -> float:
    return max(d.values()) if d else 0.0


def aurc(correct: np.ndarray, score: np.ndarray) -> float:
    """Mean selective risk over all coverages, answering the lowest failure scores first."""
    err = 1 - correct[np.argsort(score, kind="stable")]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def sel_acc(correct, score, cov):
    n = max(1, int(round(cov * len(correct))))
    return float(correct[np.argsort(score, kind="stable")[:n]].mean())


def jitter(score, seed=0):
    """Random tie-break (judges' P(fail) has many ties); fixed per question so bootstrap pairs stay paired."""
    return np.asarray(score, float) + np.random.default_rng(seed).uniform(0, 1e-6, len(score))


def boot_diff(c_a, s_a, c_b, s_b, fn, seed=0):
    rng = np.random.default_rng(seed)
    n, v = len(c_a), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        v.append(fn(c_a[i], s_a[i]) - fn(c_b[i], s_b[i]))
    return np.quantile(v, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes_lr_C0.01_L12")
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/selective")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    R = Run(Path(args.run), force=True)
    _relabel(R.items, R.s0)
    pb = ProbeBundle(ROOT / args.probe_dir, mode="shared")
    P = R.probe_scores(pb)
    J = {n: R.judge_scores(n) for n in R.judges}
    by_ds = {ds: [it for it in R.items if it["dataset"] == ds] for ds in DATASETS}

    def el(u, sc):
        return {k: v for k, v in sc[u].items() if k in R.eligible(u)}

    # systems, with each system's failure score on its FINAL trajectory
    def probe_sys(rounds):
        return {it["uid"]: outcome("probe_restart", it, R.s0[it["uid"]], R.rs[it["uid"]], probe_scores=el(it["uid"], P),
                                   tau=pb.tau, rounds=rounds) for it in R.items}

    def judge_sys(n, rounds):
        return {it["uid"]: outcome("judge_restart", it, R.s0[it["uid"]], R.rs[it["uid"]],
                                   judge_rec=R.judges[n].get(it["uid"]), rounds=rounds) for it in R.items}

    def final_score(o, u, first, field, key="p_fail"):
        """Failure score of the trajectory whose answer the system returns (round-1 restart's rescore if restarted)."""
        if not o["fired"]:
            return max_score(first[u])
        if o["n_restarts"] == 2:
            return np.nan                                   # round-2 restarts were not rescored
        sc = (R.rs[u].get(RESTART_KEY[o["hop"]]) or {}).get("rescore", {}).get(field) or {}
        return max([v[key] for v in sc.values()], default=np.nan)

    probe1 = probe_sys(1)
    judge1 = {n: judge_sys(n, 1) for n in J}
    L = [f"# Selective answering and accuracy vs cost (`{Path(args.run).name}`)", "",
         f"Items: {len(R.items)}. Correct = token F1 ≥ 0.5. Failure score of a trajectory = max over its hops of P(fail) "
         f"(probe: calibrated, τ = {pb.tau[1]:.3f}; judges: their P(fail)). Questions are answered from the lowest score "
         f"up; AURC = mean error over all coverages (lower = better; 'oracle' = perfect ranking, 'random' = error rate). "
         f"Ties broken at random. CIs: paired bootstrap over questions ({N_BOOT} resamples).", ""]

    # ── 1a. first answers ────────────────────────────────────────────────────
    L += ["## 1a. Abstaining on the first answers (no restart)", "",
          "Every signal ranks the same answers: the model's uninterrupted trajectories.", ""]
    rows_fig = {}
    for ds in DATASETS:
        its = by_ds[ds]
        c = arr(R.simple("no_intervention"), its)
        sig = {"Probe (ours)": jitter([max_score(P[it["uid"]]) for it in its])}
        for n in J:
            sig[f"Judge {n}"] = jitter([max_score(J[n][it["uid"]]) for it in its], 1)
        oracle = aurc(c, 1 - c + np.random.default_rng(2).uniform(0, 1e-6, len(c)))
        L += [f"### {ds} (n = {len(its)}, accuracy at full coverage {c.mean():.3f})", "",
              "| signal | acc @ 50 % | acc @ 80 % | AURC | probe − signal, AURC [95% CI] |", "|---|---|---|---|---|"]
        for n, s in sig.items():
            d = "–" if n.startswith("Probe") else "{:+.3f} [{:+.3f}, {:+.3f}]".format(
                aurc(c, sig["Probe (ours)"]) - aurc(c, s), *boot_diff(c, sig["Probe (ours)"], c, s, aurc))
            L.append(f"| {n} | {sel_acc(c, s, .5):.3f} | {sel_acc(c, s, .8):.3f} | {aurc(c, s):.3f} | {d} |")
        L += [f"| *random* | {c.mean():.3f} | {c.mean():.3f} | {1 - c.mean():.3f} | |",
              f"| *oracle* | {sel_acc(c, 1 - c, .5):.3f} | {sel_acc(c, 1 - c, .8):.3f} | {oracle:.3f} | |", ""]
        rows_fig[ds] = (c, sig)

    # ── 1b. final answers of each pipeline ──────────────────────────────────
    L += ["## 1b. Restart, then abstain: each system's final answers, ranked by its own detector", "",
          "Probe pipeline: one probe restart, then the probe's score on the final trajectory. Judge pipelines: one judge "
          "restart, then that judge's score on the final trajectory. Accuracy and abstention both differ here, so this "
          "is the whole system: how accurate is it on the questions it chooses to answer?", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        sysm = {"Probe (ours)": (probe1, P, "probe")}
        sysm.update({f"Judge {n}": (judge1[n], J[n], "judge") for n in [MAIN_JUDGE]})
        L += [f"### {ds}", "", "| system | accuracy | acc @ 50 % | acc @ 80 % | AURC | probe − system, AURC [95% CI] |",
              "|---|---|---|---|---|---|"]
        cs = {}
        for n, (o, first, field) in sysm.items():
            c = arr(o, its)
            s = np.array([final_score(o[it["uid"]], it["uid"], first, field) for it in its])
            s = jitter(np.nan_to_num(s, nan=np.nanmean(s)), 3)
            cs[n] = (c, s)
        pc, ps = cs["Probe (ours)"]
        for n, (c, s) in cs.items():
            d = "–" if n.startswith("Probe") else "{:+.3f} [{:+.3f}, {:+.3f}]".format(
                aurc(pc, ps) - aurc(c, s), *boot_diff(pc, ps, c, s, aurc))
            L.append(f"| {n} | {c.mean():.3f} | {sel_acc(c, s, .5):.3f} | {sel_acc(c, s, .8):.3f} | {aurc(c, s):.3f} | {d} |")
        L.append("")
    L += ["Judges rescored only the main judge's restarts, so 1b compares with the main judge "
          f"({MAIN_JUDGE}) only.", ""]

    # ── 2. accuracy vs cost ─────────────────────────────────────────────────
    systems = {"No intervention": R.simple("no_intervention"), "Always retrieve": R.simple("always_retrieve"),
               "Always restart after hop 2": R.always_hop(2), "Probe (ours)": probe1, "Probe, up to 2 restarts": probe_sys(2),
               f"Judge {MAIN_JUDGE}": judge1[MAIN_JUDGE], f"Judge {MAIN_JUDGE}, up to 2 restarts": judge_sys(MAIN_JUDGE, 2),
               **{f"Judge {n}": judge1[n] for n in J if n != MAIN_JUDGE},
               "Oracle trigger": {it["uid"]: outcome("oracle_restart", it, R.s0[it["uid"]], R.rs[it["uid"]]) for it in R.items}}
    L += ["## 2. Accuracy vs cost", "",
          "Tokens processed per question = prompt + generated tokens of every forward pass, plus the judge's own tokens "
          "(probe cost is one dot product per hop boundary). Oracle: 2Wiki only (needs gold hop labels).", ""]
    cost = {}
    for ds in DATASETS:
        its = by_ds[ds]
        L += [f"### {ds}", "", "| system | accuracy | tokens processed / question | restart rate |", "|---|---|---|---|"]
        cost[ds] = {}
        for n, o in systems.items():
            if n == "Oracle trigger" and ds != "2wiki":
                continue
            acc = arr(o, its).mean()
            tok = np.mean([o[it["uid"]]["processed_tokens"] for it in its])
            fired = np.mean([o[it["uid"]]["fired"] for it in its])
            cost[ds][n] = (tok, acc)
            L.append(f"| {n} | {acc:.3f} | {tok:.0f} | {fired:.2f} |")
        L.append("")
    (out / "selective_and_cost.md").write_text("\n".join(L), encoding="utf-8")
    plot(out, rows_fig, cost)
    print("\n".join(L))


def plot(out, rows_fig, cost):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#6b7280", "#e5e7eb"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax in axes.flat:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for j, ds in enumerate(DATASETS):
        ax = axes[0, j]
        c, sig = rows_fig[ds]
        cov = np.arange(1, len(c) + 1) / len(c)
        for n, s in sig.items():
            acc = np.cumsum(c[np.argsort(s, kind="stable")]) / np.arange(1, len(c) + 1)
            probe = n.startswith("Probe")
            ax.plot(cov * 100, acc, color=BLUE if probe else MUTED, lw=2.2 if probe else 1.1,
                    ls="-" if probe else "--", alpha=1 if probe else 0.8, label=n if probe else None)
        ax.plot([], [], color=MUTED, ls="--", lw=1.1, label="the 4 judges")
        ax.axhline(c.mean(), color=INK, lw=0.8, ls=":")
        ax.text(6, c.mean() - 0.035, "answer everything", ha="left", fontsize=7, color=MUTED)
        ax.set_xlim(5, 100); ax.set_ylim(0, 1.02)
        ax.set_xlabel("questions answered (%)", fontsize=9, color=INK)
        ax.set_ylabel("accuracy on answered questions", fontsize=9, color=INK)
        ax.set_title(f"({'ab'[j]}) Abstaining on first answers, {ds}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

        ax = axes[1, j]
        for n, (tok, acc) in cost[ds].items():
            col = BLUE if n.startswith("Probe") else ORANGE if n.startswith("Judge") else MUTED
            mk = "o" if n.startswith("Probe") else "s" if n.startswith("Judge") else "D"
            ax.scatter(tok, acc, s=55, color=col, marker=mk, edgecolor="white", linewidth=1.5, zorder=3)
            lab = n.replace("Judge ", "J: ").replace("-Instruct", "").replace(", up to 2 restarts", ",\n2 restarts")
            ax.annotate(lab, (tok, acc), xytext=NUDGE.get((ds, n), (6, 3)), textcoords="offset points",
                        fontsize=6.5, color=INK, ha="right" if NUDGE.get((ds, n), (6, 3))[0] < 0 else "left")
        ax.margins(x=0.12, y=0.08)
        for lab, col, mk in (("probe", BLUE, "o"), ("judge", ORANGE, "s"), ("baseline", MUTED, "D")):
            ax.scatter([], [], color=col, marker=mk, s=40, label=lab)
        ax.set_xlabel("tokens processed per question", fontsize=9, color=INK)
        ax.set_ylabel("accuracy", fontsize=9, color=INK)
        ax.set_title(f"({'cd'[j]}) Accuracy vs cost, {ds}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "selective_and_cost.png", dpi=220)
    fig.savefig(out / "selective_and_cost.pdf")


if __name__ == "__main__":
    main()
