"""
hop_success_after_retrieval.py
Hop-level effect of the repair: for each hop k that failed in the original trajectory (S0), does the
restart triggered at that hop (key h1 / h2) state the gold entity of hop k?  And how often does a hop that
succeeded in S0 stop succeeding (broken)?

Uses only files of a pipeline run (CPU):  eval_set.jsonl, s0*.jsonl, restarts*.jsonl.
Success = the normalised gold entity occurs in the normalised hop text, applied identically to S0 and the
restart.  This is stricter than the Phase 1 labeler (no alias / SBERT / LLM tiers), so absolute rates are
lower bounds; the before/after comparison is like for like.  Only 2Wiki items are used (HotpotQA has no
hop labels).  Writes <run>/hop_success.md.

    python scripts/hop_success_after_retrieval.py --run C:\\path\\to\\pipeline_run
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline.common import hop_text, normalize_answer, read_jsonl      # noqa: E402


def merged(run: Path, stem: str):
    recs = []
    for p in sorted([*run.glob(f"{stem}.jsonl"), *run.glob(f"{stem}.shard*.jsonl")]):
        recs += read_jsonl(p)
    return recs


def stated(gold: str, text: str) -> bool:
    g, t = normalize_answer(gold), normalize_answer(text)
    return bool(g) and bool(t) and g in t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    items = {i["uid"]: i for i in read_jsonl(run / "eval_set.jsonl") if i["dataset"] == "2wiki"}
    s0 = {r["uid"]: r for r in merged(run, "s0")}
    rs = {(r["uid"], r["key"]): r for r in merged(run, "restarts")}

    # [hop][group] -> counts; group = condition (corrected conditions if remap_eval_conditions.py was run)
    cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for uid, it in items.items():
        if uid not in s0:
            continue
        for k in (1, 2):
            gold = it["gold_entities"].get(str(k), "")
            r = rs.get((uid, f"h{k}"))
            base_text = hop_text(s0[uid]["text"], k)
            if not gold.strip() or r is None or not base_text:
                continue
            before = stated(gold, base_text)
            after = stated(gold, hop_text(r["text"], k))
            for g in ("all", it.get("retrieval_condition", "?")):
                c = cnt[k][g]
                c["n"] += 1
                c["before_ok"] += before
                if not before:
                    c["failed_before"] += 1
                    c["fixed"] += after
                else:
                    c["broken"] += (not after)

    lines = ["# Hop success before / after the restart (2Wiki eval set)\n",
             "Restart = the h1 / h2 restart (one round). Success = gold entity of the hop appears in the hop text "
             "(strict string match, same rule before and after).\n",
             "| hop | group | n | success before | failed before | of those, success after restart | "
             "succeeded before | of those, failed after restart |", "|---|---|---|---|---|---|---|---|"]
    for k in (1, 2):
        for g in sorted(cnt[k], key=lambda x: (x != "all", x)):
            c = cnt[k][g]
            fb, sb = c["failed_before"], c["n"] - c["failed_before"]
            lines.append(f"| {k} | {g} | {c['n']} | {c['before_ok'] / c['n']:.1%} | {fb} | "
                         f"{(c['fixed'] / fb if fb else float('nan')):.1%} ({c['fixed']}/{fb}) | {sb} | "
                         f"{(c['broken'] / sb if sb else float('nan')):.1%} ({c['broken']}/{sb}) |")
    out = "\n".join(lines)
    (run / "hop_success.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
