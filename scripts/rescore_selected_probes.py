"""
scripts/rescore_selected_probes.py
──────────────────────────────────
Recompute the EXACT (unrounded) test scores of the saved selected-layer probes of the probe suite.

cells.jsonl stored scores rounded to 5 decimals, which creates ties for saturated probabilities (default LR, MLP) and
shifts their AUROC by up to ~0.004 at the selected layers.  The probes saved by the suite
(<suite>/<setting>/probes/<model>_<task>_L<layer>.joblib) are reloaded and applied to the real hidden states of the
test split (only the ~800 test files per setting are downloaded from Hugging Face).  Each recomputed AUROC is checked
against the stored value, which also validates that the saved probes are the ones that were evaluated.

    python scripts/rescore_selected_probes.py --suite outputs/probing/probe_suite
Writes <suite>/<setting>/exact_scores.npz (keys "<model>/<task>") and <suite>/rescore_check.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import lf_probe_suite as S                                            # noqa: E402
from paired_tests import TASKS, auroc, test_ids                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="outputs/probing/probe_suite")
    args = ap.parse_args()
    suite = Path(args.suite)
    from src.pipeline.common import download_hidden_states, load_setting_examples
    md = ["# Exact re-scoring of the saved selected-layer probes\n",
          "Probes reloaded from `probes/*.joblib` and applied to the real test-split hidden states. `stored` = AUROC in "
          "cells.jsonl (computed from unrounded scores during the run); `exact` = recomputed now.\n",
          "| setting | model | task | layer | stored AUROC | exact AUROC | |diff| |\n|---|---|---|---|---|---|---|"]
    worst = 0.0
    for setting in ("full10", "tfidf"):
        sdir = suite / setting
        sel = json.load(open(sdir / "selected.json", encoding="utf-8"))
        stored = {}
        for l in open(sdir / "cells.jsonl", encoding="utf-8"):
            c = json.loads(l)
            stored[(c["model"], c["task"], c["layer"])] = c["test"]["all"]["auroc"]
        tids = test_ids(setting)
        ex = [e for e in load_setting_examples(setting) if e["id"] in tids]            # order preserved
        print(f"[{setting}] downloading hidden states of {len(ex)} test examples ...", flush=True)
        hs = download_hidden_states(setting, ids=[e["id"] for e in ex])
        X, rows, dropped = S.load_features(ex, S.build_rows(ex), hs)
        print(f"[{setting}] test rows {len(rows)} (dropped {dropped})", flush=True)
        # the row order must equal the order of the suite's test rows
        suite_rows = [json.loads(l) for l in open(sdir / "rows.jsonl", encoding="utf-8")]
        want = [(r["id"], r["hop"]) for r in suite_rows if r["id"] in tids]
        assert want == [(r["id"], r["hop"]) for r in rows], f"{setting}: test row order differs from the suite's"
        y_all = np.array([r["y"] for r in rows])
        hop_all = np.array([r["hop"] for r in rows])
        out = {}
        for s in sel:
            m, task, L = s["model"], s["task"], s["layer"]
            score, meta = S.load_probe(sdir / "probes" / f"{m}_{task}_L{L}.joblib")
            mask = np.isin(hop_all, TASKS[task])
            sc = np.asarray(score(X[np.flatnonzero(mask), L].astype(np.float32)), dtype=float)
            a = auroc(y_all[mask], sc)
            d = abs(a - stored[(m, task, L)])
            worst = max(worst, d)
            out[f"{m}/{task}"] = sc
            md.append(f"| {setting} | {m} | {task} | {L} | {stored[(m, task, L)]:.4f} | {a:.4f} | {d:.4f} |")
        np.savez(sdir / "exact_scores.npz", **out)
        print(f"[{setting}] wrote {sdir / 'exact_scores.npz'}", flush=True)
    md.append(f"\nLargest |stored - exact| AUROC over all selected probes: **{worst:.5f}**.")
    (suite / "rescore_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[-16:]))


if __name__ == "__main__":
    main()
