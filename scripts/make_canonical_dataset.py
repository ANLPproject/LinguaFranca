"""
scripts/make_canonical_dataset.py
─────────────────────────────────
Build the ONE dataset file every Phase 2-4 script should use:
    augmented_v2.jsonl

Why this exists
───────────────
* The HF repo's current data/2wikimultihopqa/augmented.jsonl was overwritten on
  2026-09-30 20:15 UTC by kaggle_relabel.ipynb.  That run regenerated the
  counterfactuals (115 new ones) but never uploaded their hidden states, so only
  9 of its 115 CFs have a .pt file.  All probe results were computed on the
  EARLIER file (78 CFs, every one with hidden states).  We pin that revision.
* Applies the extra-hop rule (phase1_dataset/hop_rules.py): hops beyond the
  2-edge gold graph become label -1 instead of failures.
* Keeps only examples that have a hidden-state file at the pinned revision, so
  the train/test split is identical no matter which script computes it.

No GPU, no model.  Takes ~1 minute (mostly the download).

Usage
─────
    python scripts/make_canonical_dataset.py --out data/canonical
    python scripts/make_canonical_dataset.py --input path/to/old_augmented.jsonl --out data/canonical
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_dataset.hop_rules import apply_extra_hop_rule  # noqa: E402

HF_REPO = "AnishRacherla/LinguaFranca-Phase3"
# Last revision before kaggle_relabel.ipynb overwrote the data files
# (commit "Upload probes/hop1_probe_layer27.joblib", 2026-09-30 05:26 UTC).
PINNED_REVISION = "ce07fd04bf3e39d920e3ccbf769298fa4396eb1f"
SRC_FILE = "data/2wikimultihopqa/augmented.jsonl"


def download_source() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, SRC_FILE, repo_type="dataset", revision=PINNED_REVISION))


def hidden_state_ids() -> set[str] | None:
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(HF_REPO, repo_type="dataset", revision=PINNED_REVISION)
    except Exception as e:  # offline: skip the check
        print(f"[warn] could not list HF files ({e}); skipping hidden-state check")
        return None
    return {f.rsplit("/", 1)[-1][:-3] for f in files if f.startswith("data/hidden_states/") and f.endswith(".pt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="local copy of the pinned augmented.jsonl (else downloaded)")
    ap.add_argument("--out", default="data/canonical")
    args = ap.parse_args()

    src = Path(args.input) if args.input else download_source()
    examples = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    print(f"source: {src}  ({len(examples)} examples)")

    pts = hidden_state_ids()
    if pts is not None:
        before = len(examples)
        examples = [e for e in examples if e["id"] in pts]
        print(f"kept {len(examples)}/{before} examples that have hidden states")

    relabeled = collections.Counter()
    for e in examples:
        old = [h.get("label") for h in e.get("hops", [])]
        apply_extra_hop_rule(e)
        relabeled["extra_hops_unlabeled"] += sum(1 for o, h in zip(old, e["hops"]) if o in (0, 1) and h["label"] == -1)
        e["dataset_version"] = "v2"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "augmented_v2.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # stats
    st = collections.Counter()
    for e in examples:
        kind = "cf" if e.get("is_counterfactual") else "natural"
        st[f"examples_{kind}"] += 1
        for h in e["hops"]:
            pos = f"hop{h['hop_idx']}" if h["hop_idx"] <= 2 else "hop_extra"
            st[f"{pos}_{kind}_label{h['label']}"] += 1
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stats = {"source_revision": PINNED_REVISION, "n_examples": len(examples), "sha256": sha,
             **relabeled, **dict(sorted(st.items()))}
    (out / "augmented_v2.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
