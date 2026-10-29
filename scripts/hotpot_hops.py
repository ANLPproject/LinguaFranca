"""
scripts/hotpot_hops.py
──────────────────────
Hop labels for HotpotQA bridge questions, so the out-of-domain result can be split into detection vs repair
(CPU, ~5 min; a finished Phase 3-5 run + the HotpotQA distractor validation parquet).

Labels (same presence check as the pipeline's relabelling of 2Wiki, utils.matching.EntityMatcher, SBERT 0.70):
  hop 1 fails if it does not name the bridge entity; bridge entity = the supporting paragraph whose title is
        NOT in the question (questions where this is ambiguous are dropped)
  hop 2 fails if it does not contain the gold answer
Both rules are first checked on 2Wiki against the Phase 1 labels (LLM-checked) to show they are trustworthy.

  1  validation on 2Wiki   bridge-title rule vs the gold hop-1 entity; presence labels vs Phase 1 labels
  2  detection             hop-level AUROC on HotpotQA bridge: probe vs 4 judges vs token confidence
  3  error budget          as for 2Wiki, plus the oracle trigger (restart at the labelled first failing hop)

Usage:
    python scripts/hotpot_hops.py --run <run folder> --out outputs/pipeline/tfidf_final_lrC0.01_L12/hotpot_hops
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from error_budget_cases import STAGES, stage  # noqa: E402
from eval_robustness import MAIN_JUDGE, Run, arr, auroc_ci  # noqa: E402
from src.pipeline.common import load_setting_examples, normalize_answer, read_jsonl  # noqa: E402
from src.pipeline.evaluate import _paired_ci, _relabel  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import outcome  # noqa: E402

SIGNALS = "outputs/pipeline/tfidf_final_lrC0.01_L12/uncertainty_early/run_signals.jsonl"
CONF = {"max_nll": "max token NLL (FLARE)", "mean_ent": "mean token entropy"}


def base(t: str) -> str:
    return normalize_answer(re.sub(r"\s*\(.*?\)\s*", " ", t))


def bridge_title(titles, question):
    """The supporting title NOT in the question (None if not exactly one)."""
    q = normalize_answer(question)
    out = [t for t in dict.fromkeys(titles) if base(t) and base(t) not in q]
    return out[0] if len(out) == 1 and len(dict.fromkeys(titles)) == 2 else None


def hotpot_records(ids):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet",
                                         repo_type="dataset"))
    df = df[df["id"].isin(ids)]
    return {r["id"]: dict(type=r["type"], titles=list(r["supporting_facts"]["title"]), answer=r["answer"])
            for _, r in df.iterrows()}


def presence_labels(matcher, gold: dict, hops: dict) -> dict:
    return {k: 0 if (hops.get(k) and matcher.match(g, hops[k]).matched) else 1 for k, g in gold.items() if g.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes_lr_C0.01_L12")
    ap.add_argument("--signals", default=SIGNALS, help="run_signals.jsonl from uncertainty_early.py (optional)")
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/hotpot_hops")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from utils.matching import EntityMatcher
    matcher = EntityMatcher(aliases={}, sbert_threshold=0.70)

    R = Run(Path(args.run), force=True)
    _relabel(R.items, R.s0)
    pb = ProbeBundle(ROOT / args.probe_dir, mode="shared")
    P = R.probe_scores(pb)
    J = {n: R.judge_scores(n) for n in R.judges}
    sig_path = ROOT / args.signals
    C = {}
    if sig_path.exists():
        sig = {r["uid"]: r["hops"] for r in read_jsonl(sig_path)}
        C = {s: {u: {int(k): v[s] for k, v in h.items()} for u, h in sig.items()} for s in CONF}
    wiki = [it for it in R.items if it["dataset"] == "2wiki"]
    hot = [it for it in R.items if it["dataset"] == "hotpotqa"]

    # ── 1. validate both rules on 2Wiki ──────────────────────────────────────
    ex = {e["id"]: e for e in load_setting_examples("tfidf")}
    n_ok = n_amb = n_wrong = 0
    for it in wiki:
        b = bridge_title(ex[it["id"]].get("gold_titles") or [], it["question"])
        g = it["gold_entities"].get("1", "")
        if b is None:
            n_amb += 1
        elif base(b) == base(g) or base(b) in base(g) or base(g) in base(b):
            n_ok += 1
        else:
            n_wrong += 1
    agree, n_lab = Counter(), 0
    for it in wiki:
        if it.get("label_source") != "phase1":
            continue
        pl = presence_labels(matcher, it["gold_entities"], R.s0[it["uid"]]["hop_text"])
        for k, y in it["labels"].items():
            if k in pl and y in (0, 1):
                agree[(y, pl[k])] += 1
    n_lab = sum(agree.values())
    acc_lab = (agree[(0, 0)] + agree[(1, 1)]) / n_lab
    L = [f"# HotpotQA hop labels: detection vs repair out of domain (`{Path(args.run).name}`)", "",
         "Labels: hop 1 fails if it does not name the bridge entity (the supporting paragraph whose title is not in the "
         "question), hop 2 fails if it does not contain the gold answer; presence check = the pipeline's relabelling "
         "matcher (exact / normalised / SBERT ≥ 0.70, no LLM tier). Bridge questions only. Correct = token F1 ≥ 0.5.", "",
         "## 1. Are the automatic rules trustworthy? (checked on 2Wiki)", "",
         f"* Bridge-title rule vs the gold hop-1 entity, 500 questions: {n_ok} agree, {n_wrong} disagree, {n_amb} "
         f"ambiguous (dropped) → {n_ok / max(1, n_ok + n_wrong):.1%} of resolved questions agree.",
         f"* Presence labels vs Phase 1 labels (LLM-checked), {n_lab} hops: agreement {acc_lab:.1%} "
         f"(Phase 1 fail → presence fail {agree[(1, 1)]}, → pass {agree[(1, 0)]}; Phase 1 pass → presence pass "
         f"{agree[(0, 0)]}, → fail {agree[(0, 1)]}).", ""]

    # ── 2. HotpotQA labels + detection ───────────────────────────────────────
    recs = hotpot_records({it["id"] for it in hot})
    lab = []
    for it in hot:
        r = recs.get(it["id"])
        if r is None or r["type"] != "bridge":
            continue
        b = bridge_title(r["titles"], it["question"])
        if b is None:
            continue
        gold = {"1": re.sub(r"\s*\(.*?\)\s*$", "", b), "2": it["gold_answer"]}
        it["labels"] = presence_labels(matcher, gold, R.s0[it["uid"]]["hop_text"])
        fails = [int(k) for k, v in it["labels"].items() if v == 1]
        it["first_fail"] = min(fails) if fails else None
        it["gold_titles"] = list(dict.fromkeys(r["titles"]))
        lab.append(it)
    n_bridge = sum(1 for it in hot if (recs.get(it["id"]) or {}).get("type") == "bridge")
    L += ["## 2. Hop-failure detection on HotpotQA bridge questions", "",
          f"{n_bridge} bridge questions; {len(lab)} with an unambiguous bridge title. AUROC [95% CI]; 2Wiki (in domain) "
          f"for reference.", ""]
    sigs = {"probe": P, **{f"judge {n}": J[n] for n in J}, **{CONF[s]: C[s] for s in C}}
    L += ["| hop | dataset | n | fail | " + " | ".join(sigs) + " |", "|---" * (4 + len(sigs)) + "|"]
    for k in (1, 2):
        for ds, its in (("HotpotQA", lab), ("2Wiki", wiki)):
            rows = [it for it in its if it["labels"].get(str(k)) in (0, 1) and all(k in S.get(it["uid"], {}) for S in sigs.values())]
            y = [it["labels"][str(k)] for it in rows]
            cells = [auroc_ci(y, [S[it["uid"]][k] for it in rows]) for S in sigs.values()]
            L.append(f"| {k} | {ds} | {len(rows)} | {sum(y)} | " +
                     " | ".join(f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in cells) + " |")
    L.append("")

    # ── 3. error budget + oracle on HotpotQA bridge ──────────────────────────
    def el(u):
        return {k: v for k, v in P[u].items() if k in R.eligible(u)}
    systems = {
        "None": {it["uid"]: outcome("no_intervention", it, R.s0[it["uid"]], R.rs[it["uid"]]) for it in lab},
        "Probe (ours)": {it["uid"]: outcome("probe_restart", it, R.s0[it["uid"]], R.rs[it["uid"]],
                                            probe_scores=el(it["uid"]), tau=pb.tau) for it in lab},
        f"Judge {MAIN_JUDGE}": {it["uid"]: outcome("judge_restart", it, R.s0[it["uid"]], R.rs[it["uid"]],
                                                   judge_rec=R.judges[MAIN_JUDGE].get(it["uid"])) for it in lab},
        "Oracle trigger": {it["uid"]: outcome("oracle_restart", it, R.s0[it["uid"]], R.rs[it["uid"]]) for it in lab}}
    n = len(lab)
    acc = {nm: arr(o, lab) for nm, o in systems.items()}
    gain = acc["Probe (ours)"].mean() - acc["None"].mean()
    ogain = acc["Oracle trigger"].mean() - acc["None"].mean()
    L += ["## 3. Accuracy, oracle and error budget on HotpotQA bridge questions", "",
          "| system | accuracy | vs probe [95% CI] |", "|---|---|---|"]
    for nm, c in acc.items():
        d = "–" if nm.startswith("Probe") else "{:+.3f} [{:+.3f}, {:+.3f}]".format(
            acc["Probe (ours)"].mean() - c.mean(), *_paired_ci(acc["Probe (ours)"], c))
        L.append(f"| {nm} | {c.mean():.3f} | {d} |")
    L += ["", f"The probe gets {gain / ogain:.0%} of the oracle trigger's gain over no intervention "
              f"(2Wiki: 94 %)." if ogain > 0 else "", "",
          "| stage | " + " | ".join(nm for nm in systems if nm != "None") + " |", "|---" * len(systems) + "|"]
    budget = {nm: Counter(stage(it, O[it["uid"]], R.rs[it["uid"]], it["gold_titles"]) for it in lab
                          if not O[it["uid"]]["correct"]) for nm, O in systems.items() if nm != "None"}
    for s in STAGES:
        L.append(f"| {s} | " + " | ".join(f"{b[s]} ({b[s] / n:.1%})" for b in budget.values()) + " |")
    L.append("| **all errors** | " + " | ".join(f"**{sum(b.values())} ({sum(b.values()) / n:.1%})**"
                                                for b in budget.values()) + " |")
    (out / "hotpot_hops.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
