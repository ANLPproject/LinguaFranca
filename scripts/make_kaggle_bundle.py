"""
scripts/make_kaggle_bundle.py
─────────────────────────────
Zip everything the Kaggle notebooks in kaggle/ need into
    kaggle_upload/linguafranca_bundle.zip
Upload that zip once as a Kaggle Dataset (see FIXES_AND_RERUN.md).

Contents: the Phase 2-4 scripts, the canonical dataset, and (if present) the
probing outputs from outputs/probing (so the LLM-judge notebook can compare
against the probe without re-running probing).

Usage:  python scripts/make_kaggle_bundle.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "lf_common.py", "lf_models.py", "kaggle_helpers.py",
    "advanced_probing.py", "evaluate_saved_probe.py", "bootstrap_confident_wrong.py",
    "evaluate_llm_baseline.py", "causal_patching.py",
    "scripts/make_canonical_dataset.py",
    "phase1_dataset/__init__.py", "phase1_dataset/hop_rules.py",
    "data/canonical/augmented_v2.jsonl", "data/canonical/augmented_v2.stats.json",
    "data/canonical/hop1_probe_layer10.npz", "data/canonical/phase4_reference_hop1_L10.npz",
]
OPTIONAL_DIRS = ["outputs/probing"]


def main():
    out = ROOT / "kaggle_upload" / "linguafranca_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing files (run scripts/make_canonical_dataset.py first?): {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, f)
        for d in OPTIONAL_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("BUNDLE_README.txt",
                   "LinguaFranca Kaggle bundle. Upload this zip as a Kaggle Dataset and attach it to the\n"
                   "notebooks in kaggle/ (01_probing, 02_llm_judge, 03_causal_patching).\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
