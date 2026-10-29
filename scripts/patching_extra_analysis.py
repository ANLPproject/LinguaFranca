"""
scripts/patching_extra_analysis.py
──────────────────────────────────
CPU analyses on the TF-IDF patching runs (no model):

  1. reproducibility: Experiment A entity-span outcomes of the C-only re-run vs the full run, pair by pair
  2. Experiment C, old probe (C=1, layer 10) vs tuned probe (C=0.01, layer 12) on the same pairs:
     baseline flag rates by CF failure kind, AUROC clean vs CF, agreement below / at-or-above the probe layer
  3. entity span OR tail restores (union), per block
  4. donor control restricted to third entities from the same question template (type-matched)

Usage:
  python scripts/patching_extra_analysis.py \
      --full outputs/patching/patching_tfidf_results/patching_tfidf \
      --c-tuned outputs/patching/patching_tuned_results/patching_tfidf_c_tuned
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import CF_V1, hf_file, read_jsonl  # noqa: E402


def load(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    recs = [json.loads(l) for l in open(d / "pairs.jsonl", encoding="utf-8") if l.strip()]
    recs = {r["cf_id"]: r for r in recs if not r.get("error")}
    return meta, recs


def valid(r):
    return r["baseline_clean_states_gold"] and not r["baseline_cf_states_gold"]


def sign_test(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([0] * len(neg) + [1] * len(pos), list(neg) + list(pos)))


def probe_stats(meta, recs, ids):
    tau, pl = meta["probe"]["tau"], meta["probe"]["layer"]
    rs = [recs[i] for i in ids]
    out = {"tau": tau, "layer": pl,
           "clean_flagged": float(np.mean([r["C_clean"] >= tau for r in rs])),
           "cf_flagged": float(np.mean([r["C_cf"] >= tau for r in rs])),
           "auroc_clean_vs_cf": auroc([r["C_clean"] for r in rs], [r["C_cf"] for r in rs])}
    for k in ("hedge", "other_wrong_entity"):
        sub = [r for r in rs if r["failure_kind"] == k]
        out[f"cf_flagged_{k}"] = (float(np.mean([r["C_cf"] >= tau for r in sub])), len(sub))
    for side, keep in (("upstream", lambda L: L < pl), ("downstream", lambda L: L >= pl)):
        a = t = 0
        for r in rs:
            for L, pf in r["C_entity_span"].items():
                if pf is None or not keep(int(L)):
                    continue
                a += (pf >= tau) != r["A_by_layer"][L]["entity_span"]
                t += 1
        out[f"agreement_{side}"] = (a, t, a / t if t else float("nan"))
    # the same block range for both probes: patches below block 10 (upstream of both)
    a = t = 0
    for r in rs:
        for L, pf in r["C_entity_span"].items():
            if pf is not None and int(L) < 10:
                a += (pf >= tau) != r["A_by_layer"][L]["entity_span"]
                t += 1
    out["agreement_blocks_lt10"] = (a, t, a / t if t else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="outputs/patching/patching_tfidf_results/patching_tfidf")
    ap.add_argument("--c-tuned", default="outputs/patching/patching_tuned_results/patching_tfidf_c_tuned")
    ap.add_argument("--out", default="outputs/patching/extra_analysis.md")
    args = ap.parse_args()
    m_full, full = load(args.full)
    m_ct, ct = load(args.c_tuned)
    L = []

    # 1. reproducibility
    common = sorted(set(full) & set(ct))
    diff = [(i, Lb) for i in common for Lb in full[i]["A_by_layer"]
            if full[i]["A_by_layer"][Lb]["entity_span"] != ct[i]["A_by_layer"].get(Lb, {}).get("entity_span")]
    base_diff = sum(full[i]["baseline_cf_states_gold"] != ct[i]["baseline_cf_states_gold"] or
                    full[i]["baseline_clean_states_gold"] != ct[i]["baseline_clean_states_gold"] for i in common)
    text_diff = sum(full[i]["cf_baseline_text"] != ct[i]["cf_baseline_text"] for i in common)
    L += ["# Patching (TF-IDF): extra analyses", "",
          "## 1. Reproducibility (full run vs C-only re-run, same pairs)", "",
          f"- pairs in both runs: {len(common)}; entity-span outcomes that differ: {len(diff)} of "
          f"{sum(len(full[i]['A_by_layer']) for i in common)}; baseline labels that differ: {base_diff}; "
          f"CF baseline texts that differ: {text_diff}", ""]

    # 2. Experiment C old vs tuned probe on the same valid pairs
    ids = [i for i in common if valid(full[i]) and valid(ct[i])
           and full[i].get("C_cf") is not None and ct[i].get("C_cf") is not None]
    so, sn = probe_stats(m_full, full, ids), probe_stats(m_ct, ct, ids)
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    fa = lambda x: f"{x[0]}/{x[1]} ({x[2]:.3f})"  # noqa: E731
    fk = lambda x: f"{x[0]:.3f} (n={x[1]})"  # noqa: E731
    L += [f"## 2. Experiment C: old vs tuned probe (same {len(ids)} informative pairs)", "",
          "| | old: C=1, block 10 | tuned: C=0.01, block 12 |", "|---|---|---|"]
    for key, name, fmt in (("tau", "τ (calibrated)", f3), ("clean_flagged", "clean baseline flagged", f3),
                           ("cf_flagged", "CF baseline flagged", f3), ("cf_flagged_hedge", "  CF hedge flagged", fk),
                           ("cf_flagged_other_wrong_entity", "  CF other wrong entity flagged", fk),
                           ("auroc_clean_vs_cf", "AUROC clean vs CF baseline", f3),
                           ("agreement_upstream", "agreement, patches below the probe's layer", fa),
                           ("agreement_downstream", "agreement, patches at/above the probe's layer", fa),
                           ("agreement_blocks_lt10", "agreement, patches at blocks < 10 (same range)", fa)):
        L.append(f"| {name} | {fmt(so[key])} | {fmt(sn[key])} |")
    L.append("")

    # 3. entity OR tail
    vf = [full[i] for i in full if valid(full[i])]
    L += ["## 3. Entity span OR tail restores (full run)", "",
          "| block | entity | tail | entity OR tail | entity AND tail |", "|---|---|---|---|---|"]
    for Lb in sorted({int(x) for r in vf for x in r["A_by_layer"]}):
        rows = [r["A_by_layer"][str(Lb)] for r in vf]
        e = np.array([x["entity_span"] for x in rows]); t = np.array([x["tail"] for x in rows])
        L.append(f"| {Lb} | {e.mean():.2f} | {t.mean():.2f} | {(e | t).mean():.2f} | {(e & t).mean():.2f} |")
    L.append("")

    # 4. donor control, same-template third entities only
    tmpl = {c["clean_pair_id"]: c["counterfactual_swap"]["template"] for c in read_jsonl(hf_file(CF_V1["file"], CF_V1["revision"]))}
    same = [r for r in vf if r.get("donor_clean_id") and tmpl.get(r["donor_clean_id"]) == tmpl.get(r["clean_id"])]
    L += ["## 4. Donor control with type-matched third entities (same question template)", "",
          f"Pairs whose third entity comes from the same template: {len(same)} of {len(vf)} "
          f"(templates unknown for {sum(1 for r in vf if r.get('donor_clean_id') and r['donor_clean_id'] not in tmpl)}).", "",
          "| block | n | entity | donor | entity − donor | sign test p | all pairs: donor |", "|---|---|---|---|---|---|---|"]
    for Lb in sorted({int(x) for r in vf for x in r["A_by_layer"]}):
        rows = [r["A_by_layer"][str(Lb)] for r in same if "donor_control" in r["A_by_layer"][str(Lb)]]
        alld = [r["A_by_layer"][str(Lb)]["donor_control"] for r in vf if "donor_control" in r["A_by_layer"][str(Lb)]]
        if not rows:
            continue
        e = np.array([x["entity_span"] for x in rows]); d = np.array([x["donor_control"] for x in rows])
        b, c = int((e & ~d).sum()), int((d & ~e).sum())
        L.append(f"| {Lb} | {len(rows)} | {e.mean():.2f} | {d.mean():.2f} | {e.mean() - d.mean():+.2f} | "
                 f"{sign_test(b, c):.2g} | {np.mean(alld):.2f} |")
    md = "\n".join(L) + "\n"
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
