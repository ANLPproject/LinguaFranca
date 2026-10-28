"""
scripts/relabel_dates.py
────────────────────────
How many hop labels are false failures caused only by DATE FORMATTING?

The Phase 1 labeler (normalized string -> Wikidata alias -> SBERT -> LLM) marks a hop as a failure when the gold
entity is not found in the hop text.  When the gold answer is a date, "10 January 2016" and "January 10, 2016"
share no normalized substring, so a correct step is labelled a failure.  The alias table cannot fix this (it was
empty / near-empty, see the report).

This script re-checks ONLY rows labelled `unmatched_gold` (generated hops 1-2) with one extra rule, identical for
both settings: the gold is parsed as a date (day / month / year, any common format) and the hop text counts as
correct if it contains a date with the same year AND the same month AND the same day (whichever the gold
specifies).  A year-only statement for a full-date gold stays a failure (it is incomplete); those rows are only
counted.  Nothing is written back to the data: the output is a report and a list of proposed flips.

    python scripts/relabel_dates.py            # both settings; needs the Hugging Face label files (small)
Writes outputs/relabel_dates/report.md and flips_<setting>.jsonl
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_M = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_D = r"(\d{1,2})(?:st|nd|rd|th)?"
_Y = r"(\d{4})"
_PATTERNS = [
    (re.compile(rf"\b{_D}\s+(?:of\s+)?{_M},?\s+{_Y}\b", re.I), ("d", "m", "y")),       # 10 January 2016
    (re.compile(rf"\b{_M}\s+{_D},?\s+{_Y}\b", re.I), ("m", "d", "y")),                 # January 10, 2016
    (re.compile(rf"\b{_Y}-(\d{{2}})-(\d{{2}})\b"), ("y", "mn", "d")),                  # 2016-01-10
    (re.compile(rf"\b{_M},?\s+{_Y}\b", re.I), ("m", "y")),                             # January 2016
    (re.compile(rf"\b{_Y}\b"), ("y",)),                                                # 2016
]


def find_dates(text: str):
    """All (year, month|None, day|None) found in `text`, most specific pattern first (spans do not overlap)."""
    out, taken = [], []
    for rx, order in _PATTERNS:
        for m in rx.finditer(text or ""):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            g = dict(zip(order, m.groups()))
            if "mn" in g:
                g["m"] = int(g.pop("mn"))
            elif "m" in g:
                g["m"] = MONTHS[g["m"].lower().rstrip(".")]
            y = int(g["y"])
            mo = g.get("m")
            d = int(g["d"]) if "d" in g else None
            if not (1000 <= y <= 2100) or (mo is not None and not 1 <= mo <= 12) or (d is not None and not 1 <= d <= 31):
                continue
            taken.append((m.start(), m.end()))
            out.append((y, mo, d))
    return out


def gold_date(gold: str):
    """The gold as a date, only if the WHOLE string is a date (so 'Seti I' or 'Rome 1990' are not dates)."""
    g = (gold or "").strip().rstrip(".")
    for rx, order in _PATTERNS:
        m = rx.fullmatch(g)
        if m:
            ds = find_dates(g)
            return ds[0] if ds else None
    return None


def date_matches(gold, text_dates, strict=True):
    y, mo, d = gold
    for ty, tm, td in text_dates:
        if ty != y:
            continue
        if strict and ((mo is not None and tm != mo) or (d is not None and td != d)):
            continue
        return True
    return False


def selftest():
    assert gold_date("10 January 2016") == (2016, 1, 10) and gold_date("September 2, 1965") == (1965, 9, 2)
    assert gold_date("1887") == (1887, None, None) and gold_date("Seti I") is None and gold_date("Rome 1990") is None
    assert date_matches((2016, 1, 10), find_dates("He died on January 10, 2016 in NYC"))
    assert date_matches((1965, 9, 2), find_dates("born on 2nd September 1965"))
    assert not date_matches((2004, 9, 27), find_dates("Tsai Wan-lin died in 2004"))            # year only: stays a failure
    assert date_matches((2004, 9, 27), find_dates("Tsai Wan-lin died in 2004"), strict=False)  # ...but counted as year-only
    assert not date_matches((1987, 8, 11), find_dates("died on August 12, 1987"))
    assert date_matches((1887, None, None), find_dates("was born in 1887"))


def main():
    selftest()
    from src.pipeline.common import canonical_split, load_setting_examples
    out = ROOT / "outputs" / "relabel_dates"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Date-format relabel check (report only; no labels were changed)\n",
          "Rule (same for both settings): re-check only rows labelled `unmatched_gold` among generated hops 1-2. If the gold "
          "is a date, the hop is counted correct when the text contains a date with the same year and the same month and "
          "day (as far as the gold specifies them). Year-only statements for a full-date gold stay failures.\n"]
    for setting in ("full10", "tfidf"):
        ex = load_setting_examples(setting)
        tr, te = canonical_split(ex)
        te_set = set(te.tolist())
        n_rows = n_fail = n_unm = n_dategold = 0
        flips, year_only = [], 0
        fail_before = collections.Counter()
        tot_hop = collections.Counter()
        for i, e in enumerate(ex):
            order = [h["hop_idx"] for h in e.get("hop_spans", [])]
            for h in e["hops"]:
                k = h["hop_idx"]
                if k not in (1, 2) or h.get("label") not in (0, 1) or k not in order:
                    continue
                n_rows += 1
                tot_hop[k] += 1
                fail_before[k] += h["label"] == 1
                if h["label"] == 1:
                    n_fail += 1
                if h.get("match_method") != "unmatched_gold":
                    continue
                n_unm += 1
                gold = h.get("bridging_entity_gold") or next((n["gold_entity"] for n in e["reasoning_graph"] if n["hop"] == k), "")
                gd = gold_date(gold)
                if gd is None:
                    continue
                n_dategold += 1
                td = find_dates(h.get("text", ""))
                if date_matches(gd, td, strict=True):
                    flips.append(dict(id=e["id"], hop=k, gold=gold, text=h.get("text", ""), in_test=i in te_set))
                elif date_matches(gd, td, strict=False):
                    year_only += 1
        fail_after = collections.Counter(fail_before)
        for f_ in flips:
            fail_after[f_["hop"]] -= 1
        ff_changes = 0
        flip_ids = collections.defaultdict(set)
        for f_ in flips:
            flip_ids[f_["id"]].add(f_["hop"])
        for e in ex:
            if e["id"] in flip_ids:
                fails = [h["hop_idx"] for h in e["hops"] if h["hop_idx"] in (1, 2) and h.get("label") == 1]
                new = [k for k in fails if k not in flip_ids[e["id"]]]
                ff_changes += (min(fails) if fails else None) != (min(new) if new else None)
        with open(out / f"flips_{setting}.jsonl", "w", encoding="utf-8") as f:
            for f_ in flips:
                f.write(json.dumps(f_, ensure_ascii=False) + "\n")
        n_te = sum(f_["in_test"] for f_ in flips)
        md += [f"## {setting}\n",
               f"* labelled generated hop rows: {n_rows}; labelled failure: {n_fail}; `unmatched_gold`: {n_unm}",
               f"* of those, gold is a date: {n_dategold}",
               f"* **false failures fixed by the date rule: {len(flips)}** ({100 * len(flips) / n_rows:.2f}% of all rows; "
               f"{100 * len(flips) / max(n_fail, 1):.1f}% of all failure labels); in the test split: {n_te}",
               f"* year-only statements for a full-date gold (left as failures): {year_only}",
               f"* hop failure rate before -> after: hop 1 {100 * fail_before[1] / tot_hop[1]:.1f}% -> {100 * fail_after[1] / tot_hop[1]:.1f}%"
               f", hop 2 {100 * fail_before[2] / tot_hop[2]:.1f}% -> {100 * fail_after[2] / tot_hop[2]:.1f}%",
               f"* questions whose first-failure hop would change: {ff_changes}", "",
               "Examples of proposed flips:\n"]
        for f_ in flips[:8]:
            md.append(f"- hop {f_['hop']} | gold: `{f_['gold']}` | text: {f_['text'][:140]}")
        md.append("")
        print(f"[{setting}] flips {len(flips)} / {n_rows} rows ({100 * len(flips) / n_rows:.2f}%), test {n_te}, "
              f"year-only left {year_only}, first-fail changes {ff_changes}")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", out / "report.md")


if __name__ == "__main__":
    main()
