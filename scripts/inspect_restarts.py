"""
What does the model output after retrieval?  Builds a readable report of real restart examples
from a pipeline run, plus how often the repair retrieval finds the gold paragraph the first stage dropped.

Usage (from the repo root, on an extracted run folder):
    python scripts/inspect_restarts.py --run pipeline_results/pipeline_tfidf_v2_r2 \
        --probe-dir outputs/pipeline/tfidf/probes --out outputs/pipeline/tfidf_v2_r2/retrieval_samples.md
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.pipeline.common import answer_correct, hop_text, load_setting_examples, read_jsonl  # noqa: E402
from src.pipeline.probes import ProbeBundle  # noqa: E402


def safe(uid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def short(s, n=220):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)

    items = {it["uid"]: it for it in read_jsonl(run / "eval_set.jsonl")}
    s0 = {r["uid"]: r for r in read_jsonl(run / "s0.jsonl")}
    rs = {}
    for r in read_jsonl(run / "restarts.jsonl"):
        rs[(r["uid"], r["key"])] = r
    per = {r["uid"]: r for r in read_jsonl(run / "per_item.jsonl")}
    gold_titles = {e["id"]: e.get("gold_titles", []) for e in load_setting_examples("tfidf")}
    pb = ProbeBundle(args.probe_dir, mode="shared")

    def pfail(uid):
        z = np.load(run / "s0_feats" / f"{safe(uid)}.npz")
        return {k: pb.p_fail(k, z[f"h{k}"]) for k in (1, 2) if f"h{k}" in z.files}

    # ── retrieval quality on 2Wiki: does the repair find the gold paragraph the first stage dropped? ──
    rows = []
    for uid, it in items.items():
        if it["dataset"] != "2wiki":
            continue
        o = per[uid]["probe_restart"]
        if not o["fired"]:
            continue
        k = o["hop"]
        r = rs[(uid, f"h{k}")]
        gold = gold_titles.get(it["id"], [])
        missing = [t for t in gold if t not in it["context_titles"]]
        hit = bool(set(missing) & set(r["retrieved_titles"]))
        before = per[uid]["no_intervention"]["correct"]
        rows.append(dict(uid=uid, hop=k, missing=missing, hit=hit, before=before, after=o["correct"], r=r))

    with_missing = [x for x in rows if x["missing"]]
    lines = [f"# What comes out after retrieval — real examples (`{run.name}`)", "",
             "Main system: one shared probe, repair v2 (entity-led hop-2 query + cross-encoder re-ranking), one restart. "
             "Each example shows the model's first answer, where the probe stopped it, what the retriever returned "
             "(★ = a gold paragraph), and the restarted answer.", "",
             "## How often does the repair retrieval find the missing gold paragraph? (2Wiki, probe-triggered restarts)", ""]
    n_hit = sum(x["hit"] for x in with_missing)
    lines += [f"- Restarts triggered: **{len(rows)}** of 500 questions.",
              f"- Restarts where a gold paragraph was missing from the prompt: **{len(with_missing)}**; "
              f"the top-3 retrieved paragraphs contain a missing gold paragraph in **{n_hit} ({n_hit / max(len(with_missing), 1):.0%})**.",
              ""]
    tab = Counter((x["hit"], x["before"], x["after"]) for x in with_missing)
    lines += ["| gold paragraph retrieved? | n | answer fixed (wrong → right) | broken (right → wrong) | still wrong | still right |",
              "|---|---|---|---|---|---|"]
    for hit in (True, False):
        sub = [x for x in with_missing if x["hit"] == hit]
        c = lambda b, a: sum(1 for x in sub if x["before"] == b and x["after"] == a)  # noqa: E731
        lines.append(f"| {'yes' if hit else 'no'} | {len(sub)} | {c(False, True)} | {c(True, False)} | "
                     f"{c(False, False)} | {c(True, True)} |")
    no_miss = [x for x in rows if not x["missing"]]
    lines += ["", f"Restarts where both gold paragraphs were already in the prompt: {len(no_miss)} "
                  f"(fixed {sum(1 for x in no_miss if not x['before'] and x['after'])}, "
                  f"broken {sum(1 for x in no_miss if x['before'] and not x['after'])}).", ""]
    no_ans = sum(1 for r in read_jsonl(run / "restarts.jsonl") if not r["answer"])
    lines += [f"Note: {no_ans} of {len(rs)} restarts ({no_ans / len(rs):.0%}) — and a similar share of first answers — "
              "never write `<answer>`: the model keeps adding `<hopN>` steps until the 400-token cap. They count as wrong.", ""]

    # ── examples ─────────────────────────────────────────────────────────────
    rng = random.Random(args.seed)

    def pick(cond, n):
        c = [x for x in rows if cond(x)]
        rng.shuffle(c)
        return c[:n]

    groups = [
        ("Fixed — gold paragraph retrieved, hop-1 failure", pick(lambda x: x["hit"] and not x["before"] and x["after"] and x["hop"] == 1, 2)),
        ("Fixed — gold paragraph retrieved, hop-2 failure", pick(lambda x: x["hit"] and not x["before"] and x["after"] and x["hop"] == 2, 2)),
        ("Gold paragraph retrieved, answer still wrong", pick(lambda x: x["hit"] and not x["after"], 2)),
        ("Gold paragraph NOT retrieved", pick(lambda x: x["missing"] and not x["hit"], 2)),
        ("Broken — right before the restart, wrong after", pick(lambda x: x["before"] and not x["after"], 2)),
    ]
    for title, exs in groups:
        lines += [f"## {title}", ""]
        for x in exs:
            lines += example(items[x["uid"]], s0[x["uid"]], x, pfail(x["uid"]), gold_titles.get(items[x["uid"]]["id"], []))
    hot = [uid for uid, it in items.items() if it["dataset"] == "hotpotqa" and per[uid]["probe_restart"]["fired"]
           and not per[uid]["no_intervention"]["correct"] and per[uid]["probe_restart"]["correct"]]
    rng.shuffle(hot)
    lines += ["## HotpotQA (out of domain) — fixed", ""]
    for uid in hot[:2]:
        o = per[uid]["probe_restart"]
        x = dict(hop=o["hop"], r=rs[(uid, f"h{o['hop']}")], before=False, after=True)
        lines += example(items[uid], s0[uid], x, pfail(uid), [])
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} triggered restarts; gold found in {n_hit}/{len(with_missing)}")


def example(it, s0, x, p, gold):
    r, k = x["r"], x["hop"]
    star = lambda t: f"★ {t}" if t in gold else t  # noqa: E731
    L = [f"**Q:** {it['question']}  ", f"**Gold answer:** {it['gold_answer']}" +
         (f"  · gold paragraphs: {', '.join(gold)}" if gold else ""), "",
         f"*First-stage context (TF-IDF top-5):* {', '.join(star(t) for t in it['context_titles'])}", "",
         "*First answer (no retrieval):*", "```",
         f"<hop1>{short(hop_text(s0['text'], 1))}</hop1>",
         f"<hop2>{short(hop_text(s0['text'], 2))}</hop2>",
         f"<answer>{s0['answer'] or '(none — ran to the token cap)'}</answer>   → {'✅' if x['before'] else '❌'}", "```",
         f"*Probe:* P(fail) hop 1 = {p.get(1, float('nan')):.2f}, hop 2 = {p.get(2, float('nan')):.2f} "
         f"(τ = 0.64) → **stopped at `</hop{k}>`**", "",
         f"*Retrieval query:* `{short(r['query'], 200)}`", "", "*Retrieved (top 3, prepended to the context):*"]
    for p_ in r["retrieved"]:
        L.append(f"- **{star(p_['title'])}** — {short(p_['text'], 160)}")
    L += ["", "*Restarted answer:*", "```",
          f"<hop1>{short(hop_text(r['text'], 1))}</hop1>",
          f"<hop2>{short(hop_text(r['text'], 2))}</hop2>",
          f"<answer>{r['answer'] or '(none — ran to the token cap)'}</answer>   → "
          f"{'✅' if answer_correct(r['answer'], it['gold_answer']) else '❌'}", "```", "", "---", ""]
    return L


if __name__ == "__main__":
    main()
