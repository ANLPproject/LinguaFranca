"""
scripts/alias_audit.py
──────────────────────
How many hop labels are false failures that a Wikidata alias table would have fixed ("NYC" vs "New York City")?

The Phase 1 matcher consults aliases[gold_entity].  In the released runs that table was empty or near-empty
(all-10: 27 alias hits; TF-IDF: 0), and utils/wikidata_aliases._query_wikidata returns [] on ANY error, which
build_alias_table then caches as "no aliases" — a failed lookup is indistinguishable from a missing one.

This audit
  * takes the rows labelled failure by `unmatched_gold` (generated hops 1-2) whose gold is not a date and whose hop
    text is not a hedge ("no information"), for both settings;
  * looks every gold entity up on Wikidata (also with a trailing "(…)" disambiguation removed), recording errors
    separately and caching only successful lookups, so it can be re-run to resume;
  * counts a row as a false failure when an alias occurs in the hop text as a whole-word phrase (>= 3 characters).
Nothing is written back to the data; the output is a report plus the list of proposed flips for manual review.

    python scripts/alias_audit.py [--workers 4]          # ~15-30 min for ~1,400 entities (resumable)
Writes outputs/alias_audit/{aliases.json, report.md, flips_<setting>.jsonl}
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from relabel_dates import gold_date                                   # noqa: E402

OUT = ROOT / "outputs" / "alias_audit"
ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"Accept": "application/sparql-results+json",
           "User-Agent": "LinguaFranca-research/1.0 (https://github.com/ANLPproject/LinguaFranca; student NLP project)"}
QUERY = ('SELECT ?alias WHERE {{ ?item rdfs:label "{e}"@en . ?item skos:altLabel ?alias . '
         'FILTER(LANG(?alias) = "en") }} LIMIT 50')


def lookup(entity: str, retries: int = 4):
    """(status, aliases): status 'ok' (even if no aliases) or 'error'."""
    q = QUERY.format(e=entity.replace("\\", "\\\\").replace('"', '\\"'))
    for a in range(retries):
        try:
            r = requests.get(ENDPOINT, params={"query": q, "format": "json"}, headers=HEADERS, timeout=25)
            if r.status_code in (429, 503):
                time.sleep(3 * (a + 1))
                continue
            r.raise_for_status()
            return "ok", [b["alias"]["value"] for b in r.json()["results"]["bindings"]]
        except Exception:                                           # noqa: BLE001
            time.sleep(2 * (a + 1))
    return "error", []


def norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def contains_phrase(text_norm: str, alias_norm: str) -> bool:
    return len(alias_norm) >= 3 and f" {alias_norm} " in f" {text_norm} "


def candidates():
    from src.pipeline.common import is_hedge, load_setting_examples
    rows = []
    for setting in ("full10", "tfidf"):
        for e in load_setting_examples(setting):
            order = [h["hop_idx"] for h in e.get("hop_spans", [])]
            for h in e["hops"]:
                k = h["hop_idx"]
                if k in (1, 2) and h.get("label") == 1 and h.get("match_method") == "unmatched_gold" and k in order:
                    g = h.get("bridging_entity_gold") or next(
                        (n["gold_entity"] for n in e["reasoning_graph"] if n["hop"] == k), "")
                    if g and gold_date(g) is None and not is_hedge(h.get("text", "")):
                        rows.append(dict(setting=setting, id=e["id"], hop=k, gold=g, text=h.get("text", ""),
                                         question=e.get("question", "")))
    return rows


def variants(g: str):
    base = re.sub(r"\s*\([^)]*\)\s*$", "", g).strip()
    return [g] + ([base] if base and base != g else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-entities", type=int, default=0, help="debug: only the first N entities")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = candidates()
    ents = sorted({v for r in rows for v in variants(r["gold"])})
    if args.max_entities:
        ents = ents[:args.max_entities]
    cache_p = OUT / "aliases.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    todo = [e for e in ents if cache.get(e, {}).get("status") != "ok"]
    print(f"candidate rows: {len(rows)} | lookups needed: {len(todo)} of {len(ents)} (cached ok: {len(ents) - len(todo)})", flush=True)

    t0 = time.time()

    def work(e):
        s, al = lookup(e)
        time.sleep(0.2)
        return e, s, al

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for n, (e, s, al) in enumerate(ex.map(work, todo), 1):
            cache[e] = {"status": s, "aliases": al}
            if n % 50 == 0 or n == len(todo):
                cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                ok = sum(v["status"] == "ok" for v in cache.values())
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s) ok so far {ok}", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    st = [cache.get(e, {}).get("status", "missing") for e in ents]
    n_ok, n_err = st.count("ok"), st.count("error") + st.count("missing")
    with_alias = sum(1 for e in ents if cache.get(e, {}).get("aliases"))
    md = ["# Alias audit: false failures a Wikidata alias table would fix (report only)\n",
          f"Rows audited: failure labels from `unmatched_gold`, gold not a date, hop text not a hedge. Gold entities looked up: "
          f"{len(ents)} (successful lookups {n_ok}, errors {n_err}); entities with at least one alias: {with_alias}.\n"]
    summary = {}
    for setting in ("full10", "tfidf"):
        rs = [r for r in rows if r["setting"] == setting]
        flips = []
        for r in rs:
            tn = norm(r["text"])
            qn = norm(r["question"])
            for v in variants(r["gold"]):
                hit = next((a for a in cache.get(v, {}).get("aliases", [])
                            if contains_phrase(tn, norm(a)) and not contains_phrase(qn, norm(a))), None)
                if hit:
                    flips.append(dict(r, alias=hit))
                    break
        with open(OUT / f"flips_{setting}.jsonl", "w", encoding="utf-8") as f:
            for x in flips:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary[setting] = (len(rs), len(flips))
        md += [f"## {setting}\n", f"* audited failure rows: {len(rs)}", f"* **proposed flips (alias found in hop text): {len(flips)}**",
               "", "Proposed flips (review by hand):\n"]
        md += [f"- hop {x['hop']} | gold `{x['gold']}` ~ alias `{x['alias']}` | {x['text'][:130]}" for x in flips[:30]]
        md.append("")
        print(f"[{setting}] audited {len(rs)} rows -> {len(flips)} proposed alias flips", flush=True)
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "report.md", "| lookups ok/error:", n_ok, n_err)


if __name__ == "__main__":
    main()
