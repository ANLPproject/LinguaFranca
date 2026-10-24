"""
measure_tfidf_recall.py
───────────────────────
Measures TF-IDF recall@5 on 2WikiMultihopQA to determine whether
TF-IDF over the 10 provided passages creates realistic retrieval failures.

Outputs per example:
  - Is gold passage 1 in top-5?
  - Is gold passage 2 in top-5?
  - Condition: both_present / only_p1 / only_p2 / neither

Run:
    python measure_tfidf_recall.py --raw_dir data/raw --n 2000
"""

import argparse
import json
import logging
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ─── helpers ────────────────────────────────────────────────────────────────

def extract_passages(rec: dict) -> list:
    """Return list of {title, text} for all 10 passages in a 2Wiki record."""
    ctx = rec.get("context", [])
    passages = []
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents_list = ctx.get("sentences", [])
        for t, sents in zip(titles, sents_list):
            text = " ".join(str(s) for s in sents) if isinstance(sents, list) else str(sents)
            passages.append({"title": t, "text": text})
    else:
        for item in ctx:
            if isinstance(item, (list, tuple)):
                t = item[0] if item else ""
                sents = item[1] if len(item) > 1 else []
            elif isinstance(item, dict):
                t = item.get("title", item.get("key", ""))
                sents = item.get("sentences", item.get("value", []))
            else:
                continue
            text = " ".join(str(s) for s in sents) if isinstance(sents, list) else str(sents)
            passages.append({"title": t, "text": text})
    return passages


def extract_gold_titles(rec: dict) -> list:
    """Return the gold supporting-fact passage titles."""
    sfs = rec.get("supporting_facts", [])
    titles = set()
    # HF columnar format: {"title": [...], "sent_id": [...]}
    if isinstance(sfs, dict):
        for t in sfs.get("title", []):
            if t:
                titles.add(t)
    else:
        # list-of-tuples or list-of-dicts
        for sf in sfs:
            if isinstance(sf, (list, tuple)):
                t = sf[0] if sf else ""
            elif isinstance(sf, dict):
                t = sf.get("title", sf.get("key", ""))
            else:
                t = ""
            if t:
                titles.add(t)
    return sorted(titles)


def tfidf_top_k(question: str, passages: list, k: int = 5) -> list:
    """
    Rank passages by TF-IDF cosine similarity to the question.
    Returns indices of top-k passages (sorted by descending score).
    """
    docs = [p["text"] for p in passages]
    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vec.fit_transform(docs + [question])
        doc_vecs  = tfidf_matrix[:-1]
        query_vec = tfidf_matrix[-1]
        scores    = cosine_similarity(query_vec, doc_vecs)[0]
        top_k_idx = np.argsort(scores)[::-1][:k]
        return top_k_idx.tolist()
    except ValueError:
        # Empty vocabulary edge case
        return list(range(min(k, len(passages))))


# ─── main analysis ──────────────────────────────────────────────────────────

def run(raw_dir: str, n: int, k: int, out_file: str):
    src = Path(raw_dir) / "2wikimultihopqa" / "train.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"Not found: {src}  -- run phase1_dataset/download.py first"
        )

    with open(src, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d raw records, analysing first %d", len(records), n)
    records = records[:n]

    results = []

    for rec in records:
        passages    = extract_passages(rec)
        gold_titles = set(extract_gold_titles(rec))
        question    = rec.get("question", "")

        if not passages or not gold_titles:
            continue

        top_k_idx        = tfidf_top_k(question, passages, k=k)
        retrieved_titles = {passages[i]["title"] for i in top_k_idx}

        # 2WikiMultihopQA has exactly 2 gold passages per example.
        # Determine hop order by checking which gold title appears in the question
        # (the entry entity is hop-1). Fall back to evidences[0] subject, then alpha.
        question_lower = question.lower()
        hop1_title = None
        for t in gold_titles:
            if t.lower() in question_lower:
                hop1_title = t
                break

        if hop1_title is None:
            evidences = rec.get("evidences", [])
            ev0_subject = None
            if isinstance(evidences, dict):
                subj_list = evidences.get("subject") or evidences.get("sub") or []
                ev0_subject = subj_list[0] if subj_list else None
            elif isinstance(evidences, (list, tuple)) and evidences:
                ev0 = evidences[0]
                if isinstance(ev0, (list, tuple)) and ev0:
                    ev0_subject = str(ev0[0])
                elif isinstance(ev0, dict):
                    ev0_subject = ev0.get("subject") or ev0.get("sub") or ev0.get("page")
            if ev0_subject:
                ev0_lower = ev0_subject.lower()
                for t in gold_titles:
                    if (t.lower() == ev0_lower or t.lower().startswith(ev0_lower)
                            or ev0_lower.startswith(t.lower())):
                        hop1_title = t
                        break

        if hop1_title is None:
            gold_list = sorted(gold_titles)
            hop1_title = gold_list[0] if gold_list else None

        remaining = [t for t in gold_titles if t != hop1_title]
        hop2_title = remaining[0] if remaining else None

        hop1_hit = (hop1_title in retrieved_titles) if hop1_title else None
        hop2_hit = (hop2_title in retrieved_titles) if hop2_title else None

        if hop1_hit and hop2_hit:
            condition = "both_present"
        elif hop1_hit and not hop2_hit:
            condition = "only_hop1_passage"
        elif not hop1_hit and hop2_hit:
            condition = "only_hop2_passage"
        else:
            condition = "neither"

        # Store titles in hop order (hop-1 first)
        gold_titles_ordered = ([hop1_title] if hop1_title else []) + ([hop2_title] if hop2_title else [])

        results.append({
            "id":               rec.get("_id") or rec.get("id") or f"unk_{uuid.uuid4().hex[:6]}",
            "question":         question,
            "gold_titles":      gold_titles_ordered,
            "retrieved_titles": sorted(retrieved_titles),
            "n_passages":       len(passages),
            "hop1_hit":         hop1_hit,
            "hop2_hit":         hop2_hit,
            "condition":        condition,
        })

    # ─── aggregate stats ────────────────────────────────────────────────────
    n_total     = len(results)
    cond_counts = Counter(r["condition"] for r in results)

    hop1_hits  = [r for r in results if r["hop1_hit"] is True]
    hop2_hits  = [r for r in results if r["hop2_hit"] is True]
    hop2_total = [r for r in results if r["hop2_hit"] is not None]

    recall_hop1 = len(hop1_hits) / n_total         if n_total    else 0.0
    recall_hop2 = len(hop2_hits) / len(hop2_total) if hop2_total else 0.0
    recall_both = cond_counts["both_present"] / n_total if n_total else 0.0

    print("\n" + "=" * 60)
    print(f"  TF-IDF Recall@{k} Analysis  --  2WikiMultihopQA")
    print("=" * 60)
    print(f"  Examples analysed     : {n_total}")
    print(f"  Passages per example  : ~10  (top-{k} retrieved)")
    print()
    print(f"  Recall@{k} -- hop-1 passage : {recall_hop1*100:.1f}%")
    print(f"  Recall@{k} -- hop-2 passage : {recall_hop2*100:.1f}%  (of examples with hop-2)")
    print(f"  Both gold retrieved          : {recall_both*100:.1f}%")
    print()
    print("  Condition breakdown:")
    for cond, count in sorted(cond_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_total
        bar = "#" * int(pct / 2)
        print(f"    {cond:<20} {count:>5}  ({pct:5.1f}%)  {bar}")
    print()

    miss_rate = 1.0 - recall_both
    print("  -- Interpretation ----------------------------------------")
    if miss_rate < 0.10:
        print("  WARNING: TF-IDF recall is very high (>90%).")
        print("  The gold passages are keyword-dominant, so TF-IDF almost")
        print("  always retrieves them. This will NOT produce enough")
        print("  retrieval failures for your probe to learn from.")
        print("  Options: drop 2nd-hop passage explicitly, or add unrelated")
        print("  passages from a different index to increase difficulty.")
    elif miss_rate < 0.25:
        print("  OK: Moderate miss rate. TF-IDF creates some failures.")
        print("  Focus on 'only_hop2_passage' and 'neither' subsets for probe training.")
        print("  Consider also reporting recall@3 for a harder cut.")
    else:
        print("  GOOD: TF-IDF miss rate is substantial.")
        print("  Proceed with TF-IDF as the weak retriever in v2 dataset.")
    print("=" * 60)

    # Save per-example results for deeper analysis
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Per-example results saved to: %s", out_file)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure TF-IDF recall@K on 2WikiMultihopQA"
    )
    parser.add_argument(
        "--raw_dir", default="data/raw",
        help="Root data directory (expects 2wikimultihopqa/train.jsonl inside)"
    )
    parser.add_argument(
        "--n",  type=int, default=2000,
        help="Number of examples to analyse"
    )
    parser.add_argument(
        "--k",  type=int, default=5,
        help="Top-K passages retrieved per example"
    )
    parser.add_argument(
        "--out", default="data/analysis/tfidf_recall.jsonl",
        help="Path to save per-example JSONL results"
    )
    args = parser.parse_args()
    run(args.raw_dir, args.n, args.k, args.out)
