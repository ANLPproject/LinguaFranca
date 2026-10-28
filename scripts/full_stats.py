"""
Complete statistics for a pipeline run: data and splits, label distributions, confusion matrices,
first-answer categories, every system's outcomes (fixed / broken / still right / still wrong),
restarts by hop, retrieval hit rates, HotpotQA by question type.

Usage (from the repo root, on an extracted and evaluated run folder):
    python scripts/full_stats.py --run pipeline_results/pipeline_tfidf_v2_r2 \
        --probe-dir outputs/pipeline/tfidf/probes --out outputs/pipeline/full_stats.md
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.pipeline.common import (answer_correct, canonical_split, hidden_state_ids,  # noqa: E402
                                 load_setting_examples, read_jsonl)

HEDGE = re.compile(r"not (mentioned|specified|provided|given|available|stated|found|known)|no (information|mention|data)"
                   r"|unknown|cannot (be )?determin|can't|not enough|does not (mention|say|provide|specify)|n/a|^none$"
                   r"|insufficient", re.I)
SYSTEMS = [("no_intervention", "No intervention"), ("always_retrieve", "Always retrieve"),
           ("judge_restart", "Judge trigger"), ("judge_restart_2r", "Judge trigger, ≤ 2 restarts"),
           ("probe_restart", "**Probe trigger (shared probe)**"),
           ("probe_restart_alt", "Probe trigger, dedicated probes"),
           ("probe_restart_2r_alt", "Probe trigger, dedicated probes, ≤ 2 restarts"),
           ("probe_restart_acc", "Probe trigger, accuracy-optimal τ"), ("oracle_restart", "Oracle trigger")]
FIRST = ["correct", "says not in context", "confident wrong", "no answer (token cap)"]


def pct(a, b):
    return f"{a} ({a / b:.1%})" if b else "0"


def first_kind(answer, gold):
    a = (answer or "").strip()
    if not a:
        return "no answer (token cap)"
    if answer_correct(a, gold):
        return "correct"
    return "says not in context" if HEDGE.search(a) else "confident wrong"


def confusion(m):
    n, nf = m["n"], m["n_fail"]
    tp = round(m["recall"] * nf)
    fp = round(m["fpr"] * (n - nf)) if m["fpr"] == m["fpr"] else 0
    return tp, nf - tp, fp, n - nf - fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", default="outputs/pipeline/tfidf/probes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    R = json.loads((run / "results.json").read_text())
    meta = json.loads((Path(args.probe_dir) / "probes_meta.json").read_text())
    items = {it["uid"]: it for it in read_jsonl(run / "eval_set.jsonl")}
    s0 = {r["uid"]: r for r in read_jsonl(run / "s0.jsonl")}
    per = {r["uid"]: r for r in read_jsonl(run / "per_item.jsonl")}
    rs = {(r["uid"], r["key"]): r for r in read_jsonl(run / "restarts.jsonl")}
    ex_all = load_setting_examples("tfidf")
    ex = load_setting_examples("tfidf", hidden_state_ids("tfidf"))
    tr, te = canonical_split(ex)
    gold_titles = {e["id"]: e.get("gold_titles", []) for e in ex_all}
    L = [f"# Complete statistics — Phases 3–5 (`{run.name}`, main probe: {R['probe_mode']})", "",
         "Every number below is computed from the run's output files (`scripts/full_stats.py`). "
         "Correct = token F1 ≥ 0.5 with the gold answer.", ""]

    # ── 1. data ──────────────────────────────────────────────────────────────
    hop = lambda e, k: next((h["label"] for h in e["hops"] if h["hop_idx"] == k and h["label"] in (0, 1)), None)  # noqa: E731
    L += ["## 1. Data and splits", "",
          "| | count |", "|---|---|",
          f"| 2WikiMultihopQA questions (TF-IDF top-5 setting, all with hidden states) | {len(ex)} |",
          f"| → train split (probe training + threshold CV) | {len(tr)} |",
          f"| → test split (held out) | {len(te)} |",
          f"| 2Wiki questions evaluated end to end (random sample of the test split) | {sum(it['dataset'] == '2wiki' for it in items.values())} |",
          f"| HotpotQA questions evaluated (validation, distractor setting, out of domain) | {sum(it['dataset'] == 'hotpotqa' for it in items.values())} |",
          f"| → HotpotQA bridge / comparison questions | {sum(it.get('hotpot_type') == 'bridge' for it in items.values())} / "
          f"{sum(it.get('hotpot_type') == 'comparison' for it in items.values())} |",
          f"| Retrieval corpus (paragraphs) | 94,031 |", ""]
    for name, idx in (("all 4,000", range(len(ex))), ("train", tr), ("test", te)):
        h1 = Counter(hop(ex[i], 1) for i in idx); h2 = Counter(hop(ex[i], 2) for i in idx)
        cond = Counter(ex[i].get("retrieval_condition") for i in idx)
        L.append(f"- **{name}**: hop-1 failures {pct(h1[1], h1[0] + h1[1])}; hop-2 failures {pct(h2[1], h2[0] + h2[1])}; "
                 f"first-stage condition {dict(cond)}")
    L += ["", "**Probe training rows** (one row per labelled hop; the shared probe uses hop-1 and hop-2 rows together):", "",
          "| probe | train rows | test rows | layer | τ (calibrated) | test AUROC | test precision | test recall | test FPR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, name in (("hop12", "shared (main)"), ("hop1", "dedicated hop 1"), ("hop2", "dedicated hop 2")):
        p = meta["probes"][t]
        L.append(f"| {name} | {p['n_train_rows']} | {p['n_test_rows']} | {p['layer']} | {p['tau']:.3f} | "
                 f"{p['test']['auroc']:.3f} | {p['test']['precision']:.3f} | {p['test']['recall']:.3f} | {p['test']['fpr']:.3f} |")

    # ── 2. evaluation set labels ─────────────────────────────────────────────
    wiki = [it for it in items.values() if it["dataset"] == "2wiki"]
    ff = Counter(per[it["uid"]]["first_fail"] for it in wiki)
    L += ["", "## 2. The 500 evaluated 2Wiki questions", "",
          f"- Label source: {R['label_sources']}",
          f"- First failing hop of the model's first answer: hop 1 {ff[1]}, hop 2 {ff[2]}, no failure {ff[None]}",
          f"- First-stage condition: {dict(Counter(it['retrieval_condition'] for it in wiki))}", ""]

    # ── 3. detection ─────────────────────────────────────────────────────────
    L += ["## 3. Failure detection on live trajectories (2Wiki) — confusion matrices", "",
          "Positive = the hop (or trajectory) failed. TP = failure flagged, FN = failure missed, "
          "FP = correct hop flagged, TN = correct hop passed.", "",
          "| rows | detector | n | failures | TP | FN | FP | TN | precision | recall | F0.5 | AUROC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rowname in ("hop1", "hop2", "trajectory"):
        for trig, name in (("probe", "probe (shared)"), ("probe_dedicated", "probe (dedicated)"), ("judge", "judge")):
            m = R["detection_2wiki"].get(rowname, {}).get(trig)
            if not m:
                continue
            tp, fn, fp, tn = confusion(m)
            au = f"{m['auroc']:.3f}" if "auroc" in m else "–"
            L.append(f"| {rowname} | {name} | {m['n']} | {m['n_fail']} | {tp} | {fn} | {fp} | {tn} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f05']:.3f} | {au} |")
    t = R["detection_2wiki"]["trajectory"]
    L += ["", f"First failing hop localized correctly (on {t['probe']['n_fail']} failing trajectories): "
              f"probe {t['probe']['localization_acc_on_failures']:.1%}, judge {t['judge']['localization_acc_on_failures']:.1%}.", ""]

    # ── 4. first answers ─────────────────────────────────────────────────────
    L += ["## 4. What the model's first answer (no retrieval) looks like", ""]
    for ds, key in (("2wiki", "retrieval_condition"), ("hotpotqa", "retrieval_condition"), ("hotpotqa", "hotpot_type")):
        grp = [it for it in items.values() if it["dataset"] == ds]
        L += [f"**{ds}, by {key.replace('_', ' ')}:**", "",
              f"| {key} | n | " + " | ".join(FIRST) + " |", "|---|---|" + "---|" * len(FIRST)]
        for g in sorted({it.get(key) for it in grp}, key=str) + ["ALL"]:
            sub = grp if g == "ALL" else [it for it in grp if it.get(key) == g]
            c = Counter(first_kind(s0[it["uid"]]["answer"], it["gold_answer"]) for it in sub)
            L.append(f"| {g} | {len(sub)} | " + " | ".join(pct(c[k], len(sub)) for k in FIRST) + " |")
        L.append("")

    # ── 5. systems ───────────────────────────────────────────────────────────
    L += ["## 5. Every system's outcomes", "",
          "Fixed = wrong without intervention, right with the system; broken = the reverse.", ""]
    for ds in ("2wiki", "hotpotqa"):
        grp = [it for it in items.values() if it["dataset"] == ds]
        base = {it["uid"]: per[it["uid"]]["no_intervention"]["correct"] for it in grp}
        L += [f"**{ds} (n = {len(grp)})**", "",
              "| system | correct | wrong | fixed | broken | still right | still wrong | restarted | restarted at hop 1 / hop 2 | EM | F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, name in SYSTEMS:
            if key not in per[grp[0]["uid"]]:
                continue
            o = {it["uid"]: per[it["uid"]][key] for it in grp}
            c = sum(v["correct"] for v in o.values())
            fx = sum(v["correct"] and not base[u] for u, v in o.items())
            br = sum(base[u] and not v["correct"] for u, v in o.items())
            sr = sum(base[u] and v["correct"] for u, v in o.items())
            sw = sum(not base[u] and not v["correct"] for u, v in o.items())
            fired = sum(v["fired"] for v in o.values())
            h = Counter(v["hop"] for v in o.values() if v["fired"])
            em = sum(v["em"] for v in o.values()) / len(o); f1 = sum(v["f1"] for v in o.values()) / len(o)
            L.append(f"| {name} | {pct(c, len(grp))} | {len(grp) - c} | {fx} | {br} | {sr} | {sw} | {pct(fired, len(grp))} | "
                     f"{h[1]} / {h[2]} | {em:.1%} | {f1:.1%} |")
        L.append("")
    L += ["The shared probe with ≤ 2 restarts was computed in the final Kaggle run "
          "(`outputs/pipeline/tfidf_v2_r2_shared/results.md`): 2Wiki 276/500 correct (55.2%), fixed 117, broken 8; "
          "HotpotQA 265/500 (53.0%), fixed 54, broken 33.", ""]

    # ── 6. probe restarts in detail ──────────────────────────────────────────
    L += ["## 6. Probe-triggered restarts in detail (shared probe, one restart)", ""]
    for ds in ("2wiki", "hotpotqa"):
        grp = [it for it in items.values() if it["dataset"] == ds]
        L += [f"**{ds} — first-answer category → outcome after the probe trigger:**", "",
              "| first answer | n | probe restarted | → right after | → wrong after |", "|---|---|---|---|---|"]
        for k in FIRST:
            sub = [it for it in grp if first_kind(s0[it["uid"]]["answer"], it["gold_answer"]) == k]
            if not sub:
                continue
            o = [per[it["uid"]]["probe_restart"] for it in sub]
            L.append(f"| {k} | {len(sub)} | {pct(sum(x['fired'] for x in o), len(sub))} | "
                     f"{sum(x['correct'] for x in o)} | {sum(not x['correct'] for x in o)} |")
        L.append("")
    rows = []
    for it in wiki:
        o = per[it["uid"]]["probe_restart"]
        if not o["fired"]:
            continue
        r = rs[(it["uid"], f"h{o['hop']}")]
        missing = [g for g in gold_titles.get(it["id"], []) if g not in it["context_titles"]]
        rows.append(dict(hit=bool(set(missing) & set(r["retrieved_titles"])), missing=bool(missing),
                         before=per[it["uid"]]["no_intervention"]["correct"], after=o["correct"], hop=o["hop"]))
    L += ["**2Wiki — did the repair retrieval bring back the missing gold paragraph?**", "",
          "| case | restarts | fixed | broken | still wrong | still right |", "|---|---|---|---|---|---|"]
    for name, cond in (("gold paragraph missing, retrieved", lambda x: x["missing"] and x["hit"]),
                       ("gold paragraph missing, not retrieved", lambda x: x["missing"] and not x["hit"]),
                       ("both gold paragraphs already in the prompt", lambda x: not x["missing"])):
        sub = [x for x in rows if cond(x)]
        L.append(f"| {name} | {len(sub)} | {sum(not x['before'] and x['after'] for x in sub)} | "
                 f"{sum(x['before'] and not x['after'] for x in sub)} | {sum(not x['before'] and not x['after'] for x in sub)} | "
                 f"{sum(x['before'] and x['after'] for x in sub)} |")
    for k in (1, 2):
        sub = [x for x in rows if x["hop"] == k]
        L.append(f"| restarted at hop {k} (all) | {len(sub)} | {sum(not x['before'] and x['after'] for x in sub)} | "
                 f"{sum(x['before'] and not x['after'] for x in sub)} | {sum(not x['before'] and not x['after'] for x in sub)} | "
                 f"{sum(x['before'] and x['after'] for x in sub)} |")

    # ── 7. costs ─────────────────────────────────────────────────────────────
    L += ["", "## 7. Cost per question (means)", "",
          "| system | dataset | generated tokens | processed tokens | tokens before a failure is flagged |", "|---|---|---|---|---|"]
    for ds in ("2wiki", "hotpotqa"):
        for key, name in SYSTEMS[:5]:
            m = R["end_to_end"][ds].get(key)
            if m:
                dt = f"{m['mean_detect_tokens']:.1f}" if m.get("mean_detect_tokens") is not None else "–"
                L.append(f"| {name} | {ds} | {m['gen_tokens']:.1f} | {m['processed_tokens']:.0f} | {dt} |")
    lat = R.get("latency_live_subset", {})
    if lat:
        live_name = lambda k: lat.get("live_probe_system", k) if k == "probe_restart" else k  # noqa: E731
        L += ["", f"Live latency subset (n = {lat['n']}, sequential): " +
              ", ".join(f"{live_name(k)} {v['mean']:.0f} ms mean" for k, v in lat["wall_ms"].items()) +
              f"; probe check {lat['probe_ms_per_boundary']:.1f} ms per hop; judge {lat['judge_ms_per_trajectory']:.0f} ms per answer."]
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
