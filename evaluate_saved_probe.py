"""
evaluate_saved_probe.py  (v2)
─────────────────────────────
Score saved probes on the canonical held-out test split.

Why v2
──────
The v1 script evaluated the HF probes `probes/hop1_probe_layer*.joblib`.  Those
were trained on HOP 2 (bug in v1 advanced_probing.py: `hop_idx == 1` is hop 2
after 0-indexing).  On hop-1 data they flag ~90 % of SUCCESSES as failures, and
v1 only printed recall on failures, so the "100 % hedging / 98.2 % confident
wrong" table looked great.  v2 always prints recall (TPR) next to FPR and
scores every probe on both hop-1 and hop-2 rows, so a hop mismatch is obvious.

Usage
─────
  # probes written by advanced_probing.py v2
  python evaluate_saved_probe.py --probes-dir outputs/probing/probes --hs-dir data/hidden_states
  # the OLD probes on Hugging Face (to see the bug)
  python evaluate_saved_probe.py --hf-subdir probes --legacy --hs-dir data/hidden_states
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from lf_common import (DEFAULT_DATA, HF_REPO, HIDDEN_STATES_REVISION, binary_metrics, canonical_split,
                       fmt_metrics, hop_rows, load_examples, load_hidden, rows_arrays)


def load_probe(args, task, layer):
    """Returns (model, metadata-or-None)."""
    if args.legacy:
        name = f"hop1_probe_layer{layer}.joblib"      # v1 naming (actually hop-2 probes)
    else:
        name = f"{task}_probe_layer{layer}.joblib"
    if args.probes_dir:
        path = Path(args.probes_dir) / name
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, f"{args.hf_subdir}/{name}", repo_type="dataset",
                               revision=args.hf_revision)
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj
    return obj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--hs-dir", default="data/hidden_states")
    ap.add_argument("--probes-dir", default=None, help="local folder of *.joblib (else download from HF)")
    ap.add_argument("--hf-subdir", default="probes_v2")
    ap.add_argument("--hf-revision", default=None, help="HF revision (default: latest; legacy uses pinned)")
    ap.add_argument("--legacy", action="store_true", help="evaluate v1 'hop1_probe_layer*.joblib' files")
    ap.add_argument("--task", default="hop1", choices=["hop1", "hop2", "hop12"])
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()
    if args.legacy and args.hf_subdir == "probes_v2":
        args.hf_subdir = "probes"
    if args.legacy and args.hf_revision is None:
        args.hf_revision = HIDDEN_STATES_REVISION

    examples = load_examples(args.data, hs_dir=args.hs_dir)
    _, te_idx = canonical_split(examples)
    test_ex = [examples[i] for i in te_idx]
    rows = hop_rows(test_ex)
    A = rows_arrays(rows)
    layers = list(range(28)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"test examples={len(test_ex)}  hop-1 rows={int((A['hop'] == 1).sum())}  hop-2 rows={int((A['hop'] == 2).sum())}")
    print("Each probe is scored on hop-1 AND hop-2 test rows.  A real hop-1 probe should be best on hop-1 rows,")
    print("with low FPR.  TPR without FPR is meaningless.\n")

    X_all = load_hidden(test_ex, rows, args.hs_dir, layers)
    hdr = f"{'layer':>5} | {'rows':5} | {'acc':>5} {'bacc':>5} {'auroc':>5} {'TPR':>5} {'FPR':>5} | hedge-recall conf-recall natural-recall (hop-1 failures)"
    print(hdr); print("-" * len(hdr))
    for L in layers:
        try:
            model, meta = load_probe(args, args.task, L)
        except Exception as e:
            print(f"{L:>5} | probe not found ({e.__class__.__name__})")
            continue
        if meta is not None and not args.legacy and meta.get("task") != args.task:
            print(f"  [warn] layer {L}: file metadata says task={meta.get('task')}")
        X = X_all[L]
        for hop in (1, 2):
            m = A["hop"] == hop
            p = model.predict_proba(X[m])[:, 1]
            r = binary_metrics(A["y"][m], p)
            extra = ""
            if hop == 1:
                pred = p >= 0.5
                f = A["y"][m] == 1
                def rec(mm):
                    return f"{pred[mm].mean():.3f}(n={mm.sum()})" if mm.sum() else "n/a"
                extra = f"{rec(f & A['hedge'][m])}  {rec(f & ~A['hedge'][m])}  {rec(f & ~A['cf'][m])}"
            print(f"{L:>5} | hop{hop}  | {r['acc']:.3f} {r['bacc']:.3f} {r.get('auroc', float('nan')):.3f} "
                  f"{r['tpr']:.3f} {r['fpr']:.3f} | {extra}")
    if args.legacy:
        print("\nLegacy v1 probes: if hop-2 rows score far better than hop-1 rows and hop-1 FPR is high,")
        print("the probe was trained on hop 2 (the v1 saving bug).")


if __name__ == "__main__":
    main()
