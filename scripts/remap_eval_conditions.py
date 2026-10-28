"""
remap_eval_conditions.py
Rewrite `retrieval_condition` of the 2Wiki items in a pipeline run's eval_set.jsonl with the corrected
hop-order conditions (see fix_retrieval_conditions.py).  Run this before re-running evaluate, so the
"accuracy by first-stage retrieval condition" tables use only_hop1_passage / only_hop2_passage in true
hop order instead of the old alphabetical only_p1 / only_p2 buckets.  The generations are untouched.

    python scripts/remap_eval_conditions.py --run <run_dir> \
        [--labeled data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl]
    python -m src.pipeline.evaluate --out <run_dir> --probe-dir outputs/pipeline/tfidf/probes
    (add --check to only print the old -> new mapping without writing)
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pipeline run dir containing eval_set.jsonl")
    ap.add_argument("--labeled", default=str(ROOT / "data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cond = {}
    with open(args.labeled, encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            cond[d["id"]] = d["retrieval_condition"]

    path = Path(args.run) / "eval_set.jsonl"
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    moves, missing = Counter(), 0
    for it in items:
        if it["dataset"] != "2wiki":
            continue
        new = cond.get(it["id"])
        if new is None:
            missing += 1
            continue
        moves[(it.get("retrieval_condition"), new)] += 1
        it["retrieval_condition"] = new
    for (a, b), n in sorted(moves.items(), key=lambda x: -x[1]):
        print(f"{a:>12} -> {b:<18} {n}")
    print("new distribution:", dict(Counter(it["retrieval_condition"] for it in items if it["dataset"] == "2wiki")),
          "| ids not found:", missing)
    if args.check:
        return
    bak = path.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"rewrote {path} (backup: {bak})")


if __name__ == "__main__":
    main()
