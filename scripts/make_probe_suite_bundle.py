"""
scripts/make_probe_suite_bundle.py
──────────────────────────────────
Zip the (small) code kaggle/04_probe_suite.ipynb needs into kaggle_upload/probe_suite_bundle.zip.
Upload it as its own Kaggle Dataset.  It does NOT need data/canonical (the notebook rebuilds it from Hugging Face).
Includes outputs/probing/results.json (the previous run) for the reproduction check, if present.

    python scripts/make_probe_suite_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["lf_common.py", "lf_probe_suite.py", "kaggle_helpers.py",
         "scripts/make_canonical_dataset.py",
         "phase1_dataset/__init__.py", "phase1_dataset/hop_rules.py",
         "src/__init__.py", "src/pipeline/__init__.py", "src/pipeline/common.py"]
OPTIONAL = ["outputs/probing/results.json"]


def main():
    out = ROOT / "kaggle_upload" / "probe_suite_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES + [o for o in OPTIONAL if (ROOT / o).exists()]:
            z.write(ROOT / f, f)
        z.writestr("BUNDLE_README.txt", "Code bundle for kaggle/04_probe_suite.ipynb. Attach ONLY this dataset to that notebook.\n")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB):", FILES + [o for o in OPTIONAL if (ROOT / o).exists()])


if __name__ == "__main__":
    main()
