"""
The probe as a VERIFIER: pick which answer to keep among the candidates a question already has
(the original trajectory and its restarts), using the model's own hidden states, vs a text judge.

Every candidate trajectory was scored during the run:
  original      probe P(fail) from the live features (s0_feats), judge P(fail) from judge.jsonl
  restarts      probe and judge P(fail) on the restarted trajectory (stage `rescore`, keys h1 / h2)
A candidate's failure score = max over its hops 1-2.  Selection keeps the candidate with the lowest score
(ties -> the original).  Everything is computed from cached outputs (CPU).

Systems (all with the same candidates; correct = token F1 >= 0.5):
  original only / always restart after hop 2       (no verification)
  probe-verified / judge-verified                  keep the hop-2 restart only if the verifier scores it lower
  best-of-3 (original, hop-1 restart, hop-2 restart) chosen by the probe / by the judge / at random / by an oracle
  probe trigger + probe verification              restart only where the probe fires, keep the restart only
                                                   if the probe rates it better than the original

Usage:
    python scripts/verifier_selection.py --run pipeline_results/<run> \
        --probe-dir outputs/pipeline/tfidf/probes_lr_C0.01_L12 --out outputs/pipeline/<dir>/verifier_selection.md
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.pipeline.common import answer_correct, read_jsonl  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import probe_trigger  # noqa: E402


def safe(uid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def paired_ci(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    items = read_jsonl(run / "eval_set.jsonl")
    s0 = {r["uid"]: r for r in read_jsonl(run / "s0.jsonl")}
    judge = {r["uid"]: r for r in read_jsonl(run / "judge.jsonl")}
    rs = defaultdict(dict)
    for r in read_jsonl(run / "restarts.jsonl"):
        rs[r["uid"]][r["key"]] = r
    for f in read_jsonl(run / "forced.jsonl"):                          # same answer forcing as evaluate.py
        rec = s0.get(f["uid"]) if f["key"] == "s0" else rs[f["uid"]].get(f["key"])
        if rec is not None and not rec["answer"]:
            rec["answer"] = f["answer"]
    resc = {(r["uid"], r["key"]): r for r in read_jsonl(run / "rescore.jsonl")}
    pb = ProbeBundle(args.probe_dir, mode="shared")

    rows = []
    for it in items:
        u = it["uid"]
        if u not in s0:
            continue
        z = np.load(run / "s0_feats" / f"{safe(u)}.npz")
        p0 = {k: pb.p_fail(k, z[f"h{k}"]) for k in (1, 2) if f"h{k}" in z.files}
        j0 = {int(k): v["p_fail"] for k, v in judge.get(u, {}).items() if k in ("1", "2")}
        cand = {"orig": dict(ok=answer_correct(s0[u]["answer"], it["gold_answer"]),
                             probe=max(p0.values(), default=np.nan), judge=max(j0.values(), default=np.nan))}
        for key in ("h1", "h2"):
            r, sc = rs[u].get(key), resc.get((u, key))
            if r is None or sc is None:
                continue
            pp = [v["p_fail"] for v in sc["probe"].values()]
            jj = [v["p_fail"] for v in sc["judge"].values()]
            cand[key] = dict(ok=answer_correct(r["answer"], it["gold_answer"]),
                             probe=max(pp) if pp else np.nan, judge=max(jj) if jj else np.nan)
        rows.append(dict(uid=u, ds=it["dataset"], cand=cand, p0=p0))

    def pick(c, keys, by):
        """Keep the candidate with the lowest failure score (missing score = worst; ties -> earlier key)."""
        keys = [k for k in keys if k in c]
        sc = [c[k][by] if np.isfinite(c[k][by]) else np.inf for k in keys]
        return keys[int(np.argmin(sc))]

    rng = np.random.default_rng(0)
    systems = {
        "Original answer only (no intervention)": lambda r: "orig",
        "Always restart after hop 2 (no verification)": lambda r: "h2" if "h2" in r["cand"] else "orig",
        "**Always restart after hop 2 + probe verification**": lambda r: pick(r["cand"], ["orig", "h2"], "probe"),
        "Always restart after hop 2 + judge verification (Llama-3.2-3B)": lambda r: pick(r["cand"], ["orig", "h2"], "judge"),
        "**Probe trigger (τ = 0.650) + probe verification**": lambda r: (
            "orig" if probe_trigger(r["p0"], pb.tau) is None or f"h{probe_trigger(r['p0'], pb.tau)}" not in r["cand"]
            else pick(r["cand"], ["orig", f"h{probe_trigger(r['p0'], pb.tau)}"], "probe")),
        "**Best of 3 (original, hop-1 restart, hop-2 restart) chosen by the probe**":
            lambda r: pick(r["cand"], ["orig", "h1", "h2"], "probe"),
        "Best of 3 chosen by the judge (Llama-3.2-3B)": lambda r: pick(r["cand"], ["orig", "h1", "h2"], "judge"),
        "Best of 3 chosen at random": lambda r: rng.choice([k for k in ("orig", "h1", "h2") if k in r["cand"]]),
        "Best of 3, oracle choice (upper bound)": lambda r: next((k for k in ("orig", "h1", "h2")
                                                                 if k in r["cand"] and r["cand"][k]["ok"]), "orig"),
    }
    L = ["# The probe as a verifier: which answer to keep (`" + run.name + "`)", "",
         "Each candidate trajectory's failure score = max hop P(fail), from the probe (hidden states) or the text judge "
         "(Llama-3.2-3B, the pipeline's judge); both scored every candidate during the run. The candidate with the lower "
         "score is kept (ties → the original). Same candidates for every row; one restart per candidate. "
         "Correct = token F1 ≥ 0.5; CIs: paired bootstrap vs 'always restart after hop 2'.", ""]
    for ds in ("2wiki", "hotpotqa"):
        R = [r for r in rows if r["ds"] == ds]
        base = np.array([r["cand"]["orig"]["ok"] for r in R], float)
        ah2 = np.array([r["cand"]["h2" if "h2" in r["cand"] else "orig"]["ok"] for r in R], float)
        L += [f"## {ds} (n = {len(R)})", "",
              "| system | accuracy | fixed | broken | vs always-restart-after-hop-2 [95% CI] |", "|---|---|---|---|---|"]
        for name, f in systems.items():
            ch = [f(r) for r in R]
            c = np.array([r["cand"][k]["ok"] for r, k in zip(R, ch)], float)
            ci = "–" if name.startswith("Always restart after hop 2 (no") else \
                "{:+.3f} [{:+.3f}, {:+.3f}]".format(c.mean() - ah2.mean(), *paired_ci(c, ah2))
            L.append(f"| {name} | {c.mean():.3f} | {int(((c == 1) & (base == 0)).sum())} | "
                     f"{int(((c == 0) & (base == 1)).sum())} | {ci} |")
        L.append("")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
