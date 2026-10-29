"""
scripts/error_budget_cases.py
─────────────────────────────
Where the remaining errors come from, and real examples for the paper (CPU, ~1 min, 2Wiki: needs gold hop
labels and gold paragraph titles).

  1  error budget   every wrong final answer of a pipeline assigned to the first stage that broke:
                      no hop failed (the hops were right, the answer is still wrong)
                      detector missed the failing hop (no restart)
                      detector flagged the wrong hop (restart at a hop after / before the first failing one)
                      restart at the failing hop, but the evidence was already in the context
                      restart at the failing hop, retrieval did not bring the missing gold paragraph
                      restart at the failing hop, retrieval brought it, the answer is still wrong
                    for the probe pipeline, the main judge pipeline and the oracle trigger
  2  case studies   candidates per category, printed in full (question, hops, scores, retrieval, answers)

Usage:
    python scripts/error_budget_cases.py --run <run folder> --out outputs/pipeline/tfidf_final_lrC0.01_L12/error_budget
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_robustness import MAIN_JUDGE, Run  # noqa: E402
from src.pipeline.common import answer_correct, is_hedge, load_setting_examples  # noqa: E402
from src.pipeline.evaluate import _relabel  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import RESTART_KEY, outcome  # noqa: E402

STAGES = ["no hop failed", "detector missed the failing hop", "detector flagged the wrong hop",
          "restarted: evidence was already in the context", "restarted: retrieval missed the gold paragraph",
          "restarted: gold paragraph retrieved, answer still wrong"]
N_CASES = 3


def first_fail(it):
    lab = it["labels"]
    return next((k for k in (1, 2) if lab.get(str(k)) == 1), None)


def stage(it, o, rs, gold_titles):
    ff = first_fail(it)
    if ff is None:
        return STAGES[0]
    if not o["fired"]:
        return STAGES[1]
    if o["hop"] != ff:
        return STAGES[2]
    missing = set(gold_titles) - set(it["context_titles"])
    if not missing:
        return STAGES[3]
    r = rs.get(RESTART_KEY[o["hop"]]) or {}
    return STAGES[5] if missing & set(r.get("retrieved_titles") or []) else STAGES[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes_lr_C0.01_L12")
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/error_budget")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    R = Run(Path(args.run), force=True)
    _relabel(R.items, R.s0)
    pb = ProbeBundle(ROOT / args.probe_dir, mode="shared")
    P = R.probe_scores(pb)
    JD = R.judges[MAIN_JUDGE]
    ex = {e["id"]: e for e in load_setting_examples("tfidf")}
    wiki = [it for it in R.items if it["dataset"] == "2wiki"]
    gold = {it["uid"]: ex[it["id"]].get("gold_titles") or [] for it in wiki}

    def el(u):
        return {k: v for k, v in P[u].items() if k in R.eligible(u)}
    systems = {
        "Probe (ours)": {it["uid"]: outcome("probe_restart", it, R.s0[it["uid"]], R.rs[it["uid"]], probe_scores=el(it["uid"]),
                                            tau=pb.tau) for it in wiki},
        f"Judge {MAIN_JUDGE}": {it["uid"]: outcome("judge_restart", it, R.s0[it["uid"]], R.rs[it["uid"]],
                                                   judge_rec=JD.get(it["uid"])) for it in wiki},
        "Oracle trigger": {it["uid"]: outcome("oracle_restart", it, R.s0[it["uid"]], R.rs[it["uid"]]) for it in wiki}}

    n = len(wiki)
    L = [f"# Error budget and case studies, 2Wiki (`{Path(args.run).name}`)", "",
         f"n = {n} questions; one restart; correct = token F1 ≥ 0.5. Each wrong final answer is assigned to the first "
         f"stage that broke. 'Failing hop' = the first hop labelled as failing (Phase 1 labels; relabeled where the "
         f"trajectory changed). 'Missing gold paragraph' = a gold title of the question not in the first-stage context.", "",
         "## 1. Error budget (share of all questions)", "",
         "| stage | " + " | ".join(systems) + " |", "|---" * (len(systems) + 1) + "|"]
    budget = {}
    for name, O in systems.items():
        budget[name] = Counter(stage(it, O[it["uid"]], R.rs[it["uid"]], gold[it["uid"]])
                               for it in wiki if not O[it["uid"]]["correct"])
    for s in STAGES:
        L.append(f"| {s} | " + " | ".join(f"{budget[nm][s]} ({budget[nm][s] / n:.1%})" for nm in systems) + " |")
    L.append("| **all errors** | " + " | ".join(f"**{sum(budget[nm].values())} ({sum(budget[nm].values()) / n:.1%})**"
                                                for nm in systems) + " |")
    rescuable = Counter()
    for it in wiki:
        if first_fail(it) is not None:
            missing = set(gold[it["uid"]]) - set(it["context_titles"])
            rescuable["failing hop, gold paragraph missing" if missing else "failing hop, evidence present"] += 1
    L += ["", f"Questions with a failing hop: {sum(rescuable.values())} ({dict(rescuable)}).", ""]

    # ── 2. case studies ──────────────────────────────────────────────────────
    probe, judge = systems["Probe (ours)"], systems[f"Judge {MAIN_JUDGE}"]

    def c0(it):
        return answer_correct(R.s0[it["uid"]]["answer"], it["gold_answer"])

    def jflag(u):
        return any(v["p_fail"] >= 0.5 for k, v in (JD.get(u) or {}).items() if k in ("1", "2"))

    cats = {
        "A. Confident wrong hop: the probe flags it, the judge does not, the restart retrieves the missing paragraph "
        "and fixes the answer":
            lambda it, u: (not c0(it) and probe[u]["correct"] and probe[u]["fired"] and not jflag(u)
                           and first_fail(it) == probe[u]["hop"]
                           and not is_hedge(R.s0[u]["hop_text"].get(str(probe[u]["hop"]), ""))
                           and stage(it, dict(probe[u], correct=False), R.rs[u], gold[u]) == STAGES[5]),
        "B. The probe misses a failing hop":
            lambda it, u: not probe[u]["correct"] and stage(it, probe[u], R.rs[u], gold[u]) == STAGES[1],
        "C. A restart that breaks a correct first answer (check the hop labels: often a failing hop the model "
        "recovered from on its own)":
            lambda it, u: c0(it) and not probe[u]["correct"] and probe[u]["fired"],
        "D. Correct detection, retrieval fails":
            lambda it, u: not probe[u]["correct"] and stage(it, probe[u], R.rs[u], gold[u]) == STAGES[4]}
    cases = []
    L += ["## 2. Case studies", "",
          f"Up to {N_CASES} candidates per category (shortest trajectories first). Probe P(fail): calibrated, "
          f"τ = {pb.tau[1]:.3f}; judge P(fail): {MAIN_JUDGE}, θ = 0.5.", ""]
    for cat, pred in cats.items():
        cand = [it for it in wiki if pred(it, it["uid"])]
        cand.sort(key=lambda it: len(R.s0[it["uid"]].get("text", "")))
        L += [f"### {cat}", "", f"{len(cand)} questions qualify.", ""]
        for it in cand[:N_CASES]:
            u, s0, o = it["uid"], R.s0[it["uid"]], probe[it["uid"]]
            r = R.rs[u].get(RESTART_KEY[o["hop"]]) if o["fired"] else None
            missing = sorted(set(gold[u]) - set(it["context_titles"]))
            rec = dict(category=cat, uid=u, question=it["question"], gold_answer=it["gold_answer"],
                       gold_entities=it["gold_entities"], labels=it["labels"], context_titles=it["context_titles"],
                       missing_gold=missing, hops={k: s0["hop_text"].get(k, "") for k in ("1", "2")},
                       probe={k: round(v, 3) for k, v in P[u].items()},
                       judge={k: round(v["p_fail"], 3) for k, v in (JD.get(u) or {}).items() if k in ("1", "2")},
                       first_answer=s0["answer"], restart_hop=o["hop"],
                       retrieved=(r or {}).get("retrieved_titles"), restart_hops=(r or {}).get("hop_text"),
                       final_answer=o["answer"], final_correct=bool(o["correct"]))
            cases.append(rec)
            L += [f"**Q:** {it['question']}  ", f"**Gold:** {it['gold_answer']} (bridge: {it['gold_entities'].get('1')}); "
                  f"hop labels {it['labels']}  ",
                  f"**Context (top-5 TF-IDF):** {', '.join(it['context_titles'])}; missing gold: {', '.join(missing) or '–'}  "]
            for k in ("1", "2"):
                if s0["hop_text"].get(k):
                    pk = P[u].get(int(k)); jk = (JD.get(u) or {}).get(k, {}).get("p_fail")
                    L.append(f"- hop {k}: \"{s0['hop_text'][k]}\" — probe {pk:.2f}" if pk is not None else
                             f"- hop {k}: \"{s0['hop_text'][k]}\"")
                    if jk is not None:
                        L[-1] += f", judge {jk:.2f}"
            L.append(f"- first answer: **{s0['answer'] or '∅'}**")
            if r is not None:
                L.append(f"- restart at hop {o['hop']}; retrieved: {', '.join(r.get('retrieved_titles') or [])}")
                for k, t in (r.get("hop_text") or {}).items():
                    if t:
                        L.append(f"  - restart hop {k}: \"{t}\"")
            L += [f"- final answer: **{o['answer'] or '∅'}** ({'correct' if o['correct'] else 'wrong'})", ""]
    (out / "error_budget.md").write_text("\n".join(L), encoding="utf-8")
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
