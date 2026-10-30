"""
scripts/make_cf_candidates.py
─────────────────────────────
CPU step 1 of the TF-IDF counterfactuals: choose (seed, donor) minimal pairs and
build the Wikidata alias table.  The GPU step is generate_counterfactuals_tfidf.py.

Design
──────
Seed   a TF-IDF question whose hop 1 succeeded.  Its entry entity is the gold
       paragraph title that appears in the question ("Heads Or Tails (1937 Film)").
Donor  another TF-IDF question with the SAME question template (entity masked:
       "Where was the director of film {X} born?"), so the swap keeps the entity
       type (film -> film, person -> person), and in the SAME split as the seed.
CF     the seed's stored prompt with only the entry entity replaced by the
       donor's.  System prompt, date token and TF-IDF context are the seed's, so
       the CF is a real 2Wiki question (the donor's) whose evidence is missing
       from the context.  Its gold chain is the donor's reasoning graph.

A donor is kept only if
  * the donor entity, its gold titles and its hop-1 gold entity are absent
    from the seed's context (so the CF evidence really is missing),
  * its hop-1 gold entity differs from the seed's (else "failure" is ambiguous),
  * the two prompts tokenize to the same length and every differing token lies
    inside the entity's token span (exact positional alignment for patching).

Outputs (data/cf_tfidf/):
  candidates.jsonl   one row per seed: up to --donors-per-seed aligned donors, test split first
  aliases.json       Wikidata aliases for every gold entity the GPU step labels against
  candidates.stats.json

Usage (kaggle/04_counterfactuals_tfidf.ipynb runs the first form; the GPU step then
fetches aliases only for the donors it actually tries):
  python scripts/make_cf_candidates.py --skip-aliases       # candidates only (~2 min)
  python scripts/make_cf_candidates.py                      # + aliases for every donor (~30-60 min, Wikidata)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common import (SETTINGS, apply_extra_hop_rule, hf_file, prompt_from_template,  # noqa: E402
                                 read_jsonl)

TOKENIZER = "unsloth/Llama-3.2-3B-Instruct"     # same tokenizer as meta-llama/Llama-3.2-3B-Instruct, ungated
PROBE_META = ROOT / "outputs/pipeline/tfidf/probes/probes_meta.json"
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def entry_entity(e: dict):
    """(start, end) of the longest gold title (or title without its "(… film)" suffix) in the question."""
    q = e["question"].lower()
    best = None
    for t in e.get("gold_titles") or []:
        for v in {t, _PAREN.sub("", t)}:
            v = v.strip().lower()
            if len(v) < 2 or q.count(v) != 1:
                continue
            i = q.index(v)
            if best is None or len(v) > best[1] - best[0]:
                best = (i, i + len(v))
    return best


def gold(e: dict, k: int) -> str:
    return next((n.get("gold_entity", "") for n in e.get("reasoning_graph", []) if n.get("hop") == k), "")


def overlaps(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return bool(a) and bool(b) and (a in b or b in a)


def question_offset(prompt: str) -> int:
    return prompt.rindex("\n\nQuestion: ") + len("\n\nQuestion: ")


def entity_token_range(enc, c0: int, c1: int):
    """Indices of tokens whose characters overlap [c0, c1)."""
    idx = [i for i, (s, t) in enumerate(enc["offset_mapping"]) if s < c1 and t > c0]
    return (idx[0], idx[-1] + 1) if idx else None


def aligned(tok, p_seed: str, p_cf: str, c0: int, c1_seed: int, c1_cf: int):
    """Token-level check: same length, all differences inside the entity span (same span in both)."""
    es = tok(p_seed, add_special_tokens=False, return_offsets_mapping=True)
    ec = tok(p_cf, add_special_tokens=False, return_offsets_mapping=True)
    a, b = es["input_ids"], ec["input_ids"]
    if len(a) != len(b):
        return None
    rs, rc = entity_token_range(es, c0, c1_seed), entity_token_range(ec, c0, c1_cf)
    if rs is None or rs != rc:
        return None
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not diff or min(diff) < rs[0] or max(diff) >= rs[1]:
        return None
    return dict(span=list(rs), n_entity_tokens=rs[1] - rs[0], n_diff=len(diff))


def build(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    s = SETTINGS["tfidf"]
    ex = read_jsonl(hf_file(s["data_file"], s["revision"]))
    for e in ex:
        apply_extra_hop_rule(e)
    test_ids = set(json.loads(PROBE_META.read_text())["test_ids"])
    print(f"{len(ex)} TF-IDF examples (revision {s['revision'][:10]}), {len(test_ids)} in the probe's test split")

    info = {}
    for e in ex:
        sp = entry_entity(e)
        if not sp:
            continue
        q = e["question"]
        info[e["id"]] = dict(span=sp, surface=q[sp[0]:sp[1]], template=(q[:sp[0]] + "{X}" + q[sp[1]:]).lower())
    by_t = defaultdict(list)
    for e in ex:
        if e["id"] in info:
            by_t[(info[e["id"]]["template"], e["id"] in test_ids)].append(e)

    def hop1_ok(e):
        return any(h["hop_idx"] == 1 and h["label"] == 0 for h in e["hops"]) and \
            any(h["hop_idx"] == 1 for h in e.get("hop_spans") or [])

    seeds = [e for e in ex if e["id"] in info and hop1_ok(e) and len(e.get("reasoning_graph") or []) == 2]
    rng = random.Random(args.seed)
    test_seeds = [e for e in seeds if e["id"] in test_ids]
    train_seeds = [e for e in seeds if e["id"] not in test_ids]
    rng.shuffle(test_seeds)
    rng.shuffle(train_seeds)
    print(f"entry entity found: {len(info)}/{len(ex)}; seeds (hop 1 correct): {len(seeds)} "
          f"(test {len(test_seeds)}, train {len(train_seeds)})")

    rows, why = [], Counter()
    for rank, e in enumerate(test_seeds + train_seeds):
        I, is_test = info[e["id"]], e["id"] in test_ids
        ctx = e["context"].lower()
        retrieved = {t.lower() for t in e.get("retrieved_titles") or []}
        q, (i0, i1) = e["question"], I["span"]
        g1 = gold(e, 1)
        pool = [d for d in by_t[(I["template"], is_test)] if d["id"] != e["id"]]
        rng.shuffle(pool)
        donors = []
        for d in pool:
            D = info[d["id"]]
            dg1 = gold(d, 1)
            if overlaps(D["surface"], I["surface"]):
                why["same_entity"] += 1; continue
            if not dg1 or len(d.get("reasoning_graph") or []) != 2:
                why["donor_graph"] += 1; continue
            if overlaps(dg1, g1):
                why["same_hop1_gold"] += 1; continue
            if D["surface"].lower() in ctx or _PAREN.sub("", D["surface"]).lower().strip() in ctx \
                    or any(t.lower() in retrieved for t in d.get("gold_titles") or []) or dg1.lower() in ctx:
                why["donor_evidence_in_context"] += 1; continue
            cf_q = q[:i0] + D["surface"] + q[i1:]
            p_cf = prompt_from_template(e["prompt"], e["context"], cf_q)
            c0 = question_offset(e["prompt"]) + i0
            al = aligned(tok, e["prompt"], p_cf, c0, c0 + (i1 - i0), c0 + len(D["surface"]))
            if al is None:
                why["not_token_aligned"] += 1; continue
            donors.append(dict(donor_id=d["id"], donor_entity=D["surface"], cf_question=cf_q, cf_prompt=p_cf,
                               reasoning_graph=[{**n, "in_context": False} for n in d["reasoning_graph"]],
                               gold_answer=d["gold_answer"], gold_titles=d.get("gold_titles"), **al))
            if len(donors) >= args.donors_per_seed:
                break
        if not donors:
            why["seed_without_donor"] += 1
            continue
        rows.append(dict(rank=len(rows), seed_id=e["id"], split="test" if is_test else "train",
                         template=I["template"], seed_entity=I["surface"], seed_gold=[g1, gold(e, 2)],
                         seed_retrieval_condition=e.get("retrieval_condition"), donors=donors))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(revision=s["revision"], n_examples=len(ex), n_entry_found=len(info), n_seeds=len(seeds),
                 n_seeds_with_donor=len(rows), by_split=Counter(r["split"] for r in rows),
                 donors_per_seed=Counter(len(r["donors"]) for r in rows), rejected_donors=why,
                 entity_tokens=Counter(d["n_entity_tokens"] for r in rows for d in r["donors"]),
                 templates=len({r["template"] for r in rows}))
    (out / "candidates.stats.json").write_text(json.dumps(stats, indent=1, default=dict))
    print(json.dumps(stats, indent=1, default=dict))
    return rows


def build_aliases(rows, path: Path, sleep: float):
    """Wikidata aliases for the donors' gold entities (+ seed hop-1 gold); saved every 100 queries, resumable."""
    from utils.wikidata_aliases import _query_wikidata
    ents = sorted({n["gold_entity"] for r in rows for d in r["donors"] for n in d["reasoning_graph"]
                   if n.get("gold_entity")} | {r["seed_gold"][0] for r in rows if r["seed_gold"][0]})
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [x for x in ents if x not in table]
    print(f"aliases: {len(ents)} entities, {len(todo)} to query")
    # Fail loudly instead of silently writing an empty table (what happened in the TF-IDF Phase 1 run).
    if todo and not _query_wikidata("United States"):
        raise SystemExit("Wikidata returns nothing for 'United States' - network/User-Agent problem; not writing aliases")
    t0 = time.time()
    for i, x in enumerate(todo, 1):
        table[x] = _query_wikidata(x)
        time.sleep(sleep)
        if i % 100 == 0 or i == len(todo):
            path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
            print(f"  {i}/{len(todo)}  ({time.time() - t0:.0f}s)  with aliases: "
                  f"{sum(bool(v) for v in table.values())}/{len(table)}", flush=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/cf_tfidf")
    ap.add_argument("--donors-per-seed", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-aliases", action="store_true")
    ap.add_argument("--alias-sleep", type=float, default=0.25)
    args = ap.parse_args()
    rows = build(args)
    if not args.skip_aliases:
        build_aliases(rows, ROOT / args.out / "aliases.json", args.alias_sleep)


if __name__ == "__main__":
    main()
