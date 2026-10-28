"""
scripts/upload_probes_to_hf.py
──────────────────────────────
Upload trained probes (and their metrics) to the project's Hugging Face dataset repo under a NEW folder.

Safety
  * DRY RUN by default: prints exactly what would be uploaded; nothing leaves the machine without --yes.
  * Never deletes anything and never touches existing data: it refuses a prefix that already contains files
    (unless --allow-existing) and refuses the repo's existing top-level folders (probes/, data/, distractor_rag/,
    tfidf_distractor_rag/).  The old `probes/` folder holds the superseded hop-1/hop-2 probes (FIXES_AND_RERUN.md §2).
  * The token is taken from HF_TOKEN or your local `huggingface-cli login`; it must have WRITE access.

Examples
  # the deployed TF-IDF probes used by src/pipeline (shared + dedicated), with their thresholds and predictions
  python scripts/upload_probes_to_hf.py --src outputs/pipeline/tfidf/probes --prefix probes_deployed_tfidf_v2 \
      --include hop1.npz hop2.npz hop12.npz probes_meta.json test_predictions.jsonl --note "..." [--yes]
  # the probe-suite probes (after kaggle/04_probe_suite.ipynb)
  python scripts/upload_probes_to_hf.py --src <out>/full10 --prefix probe_suite_v1/full10 \
      --include "probes/*" summary.md selected.json cells.csv info.json tuning.json rows.jsonl auroc_by_layer.png [--yes]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
import tempfile
from pathlib import Path

HF_REPO = "AnishRacherla/LinguaFranca-Phase3"
PROTECTED_TOP = {"probes", "data", "distractor_rag", "tfidf_distractor_rag", "best_probe_layer_10.pkl",
                 "best_probe_layer_11.pkl", ".gitattributes"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="local folder to upload from")
    ap.add_argument("--prefix", required=True, help="new folder inside the repo, e.g. probe_suite_v1/full10")
    ap.add_argument("--include", nargs="*", default=["*"], help="glob patterns relative to --src (default: everything)")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--note", default="", help="free text added to the README.md uploaded next to the files")
    ap.add_argument("--yes", action="store_true", help="really upload (default: dry run)")
    ap.add_argument("--allow-existing", action="store_true", help="allow a prefix that already has files (overwrites same names)")
    args = ap.parse_args()

    src = Path(args.src)
    assert src.is_dir(), f"not a folder: {src}"
    prefix = args.prefix.strip("/")
    top = prefix.split("/")[0]
    if top in PROTECTED_TOP:
        sys.exit(f"refusing prefix '{prefix}': '{top}' is an existing top-level path of {args.repo}. Choose a new folder name.")

    files = sorted(p for p in src.rglob("*") if p.is_file()
                   and any(fnmatch.fnmatch(p.relative_to(src).as_posix(), pat) for pat in args.include))
    if not files:
        sys.exit(f"no files in {src} match {args.include}")
    total = sum(p.stat().st_size for p in files)
    print(f"repo: {args.repo} (dataset)\ndestination folder: {prefix}/\n{len(files)} files, {total / 1e6:.2f} MB:")
    for p in files[:60]:
        print(f"  {p.relative_to(src).as_posix():60s} {p.stat().st_size / 1e3:9.1f} KB")
    if len(files) > 60:
        print(f"  ... and {len(files) - 60} more")

    from huggingface_hub import HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("no Hugging Face token (set HF_TOKEN or run `huggingface-cli login`)")
    api = HfApi(token=token)
    who = api.whoami().get("name")
    existing = [f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith(prefix + "/")]
    print(f"logged in as: {who} | files already under {prefix}/: {len(existing)}")
    if existing and not args.allow_existing:
        sys.exit(f"{prefix}/ already contains {len(existing)} files; choose a new --prefix (or --allow-existing).")
    if not args.yes:
        print("\nDRY RUN: nothing uploaded. Re-run with --yes to upload.")
        return

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for p in files:
            dst = stage / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        readme = [f"# {prefix}", "", "Probes trained for LinguaFranca (hop-level failure detection on Llama-3.2-3B-Instruct "
                  "hidden states). Uploaded by `scripts/upload_probes_to_hf.py`; nothing else in this repo was modified.", "",
                  args.note.strip(), "",
                  "Feature = mean-pooled hidden state of the hop's tokens at the output of transformer block `layer` "
                  "(3072 dims). `.npz` files hold StandardScaler mean/scale and the linear weights; `.joblib` files hold the "
                  "sklearn objects and a `meta` dict (see `.json`). Load suite probes with `lf_probe_suite.load_probe`.", ""]
        (stage / "README.md").write_text("\n".join(readme), encoding="utf-8")
        api.upload_folder(folder_path=str(stage), path_in_repo=prefix, repo_id=args.repo, repo_type="dataset",
                          commit_message=f"Add {prefix} (probes + metrics; no existing files changed)")
    print(f"\nuploaded {len(files) + 1} files to https://huggingface.co/datasets/{args.repo}/tree/main/{prefix}")


if __name__ == "__main__":
    main()
