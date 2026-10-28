"""
scripts/make_probe_suite_notebook.py
Writes kaggle/04_probe_suite.ipynb (so the notebook is generated from reviewed text, like make_pipeline_notebook.py).

    python scripts/make_probe_suite_notebook.py
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(t):
    return dict(cell_type="markdown", metadata={}, source=t.strip("\n").splitlines(True))


def code(t):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t.strip("\n").splitlines(True))


CELLS = [
    md(r"""
# 04 - Probe suite: logistic regression vs linear SVM vs MLP, all layers, both settings

**Settings:** Accelerator **None** | Internet **On** | No input dataset and no secret needed: STEP 1 clones the code from GitHub, from the branch named in `BRANCH` (never `main`). **Push your branch before running**, otherwise the clone will not contain `lf_probe_suite.py`.
**Runtime:** about 4 GB of downloads, then roughly 2-4 h on a Kaggle CPU session (`MODELS = "lr_default,lr,svm"` drops the MLP and roughly halves it).

What it does (`lf_probe_suite.py`): trains hop-1 / hop-2 / hop-1+2 probes on **every layer** for the all-10-passage setting (`full10`) and the TF-IDF top-5 setting (`tfidf`) with the SAME protocol as the earlier probing run, then reports every metric in one place. `lr_default` is the old probe unchanged and must reproduce the old numbers (see the *Reproduction check* in `full10/summary.md`).

Safety: a **preflight** (STEP 4, a few minutes) runs both settings on one layer first. The main run appends every finished cell to `cells.jsonl`, so if the session stops, run the main cell again and it resumes. The last cell packages whatever exists.
"""),
    code(r"""
# STEP 1 - clone the code from GitHub (YOUR branch, never main), then rebuild the canonical dataset.
REPO = "https://github.com/ANLPproject/LinguaFranca"
BRANCH = "experiment/distractor-rag"          # <- change this if you pushed to another branch (e.g. "anish-dev")
assert BRANCH not in ("main", "master"), "BRANCH must be your own branch, not main."
import os, subprocess, sys, shutil
from pathlib import Path
KW = Path(os.environ.get("LF_KAGGLE_WORK", "/kaggle/working"))
WORK = KW / "lf"                              # kaggle_helpers runs everything from this folder
def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)
if (WORK / ".git").exists():                  # re-run of the cell: update the existing clone to the branch tip
    r = sh("git", "-C", str(WORK), "fetch", "--depth", "1", "origin", BRANCH)
    if r.returncode == 0:
        r = sh("git", "-C", str(WORK), "reset", "--hard", "FETCH_HEAD")
else:
    if WORK.exists():
        shutil.rmtree(WORK)
    r = sh("git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1", REPO, str(WORK))
assert r.returncode == 0, f"git failed (is Internet ON? did you push branch '{BRANCH}'?):\n{r.stderr}"
head = sh("git", "-C", str(WORK), "log", "-1", "--format=%h %s").stdout.strip()
print(f"cloned {REPO} @ branch {BRANCH}\nHEAD: {head}")
assert Path(WORK / "lf_probe_suite.py").exists(), \
    f"lf_probe_suite.py is not on branch '{BRANCH}': commit and push it first (git push origin {BRANCH}), then re-run this cell."
os.chdir(WORK)
sys.path.insert(0, str(WORK))
import kaggle_helpers as kh
kh.RESULTS.mkdir(parents=True, exist_ok=True)
for mod, pip_name in (("sklearn", "scikit-learn"), ("joblib", "joblib"), ("huggingface_hub", "huggingface_hub")):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)
assert Path("src/pipeline/common.py").exists() and Path("lf_common.py").exists(), "branch is missing required files"
# the same canonical file the old probing run used (pinned HF revision, about 1 minute)
rc = kh.run([sys.executable, "scripts/make_canonical_dataset.py", "--out", "data/canonical"], "build_canonical")
assert rc == 0 and Path("data/canonical/augmented_v2.jsonl").exists(), "could not build data/canonical/augmented_v2.jsonl (see log)"
"""),
    code(r"""
# STEP 2 - download the hidden states for both settings (about 2 GB each; retried and resumable).
from lf_common import HF_REPO, HIDDEN_STATES_REVISION
from huggingface_hub import snapshot_download
kh.download_with_retries(lambda: snapshot_download(
    repo_id=HF_REPO, repo_type="dataset", revision=HIDDEN_STATES_REVISION,
    allow_patterns=["data/hidden_states/*"], local_dir="/kaggle/working/hf_full10", max_workers=8), what="full10 hidden states")
HS_FULL10 = "/kaggle/working/hf_full10/data/hidden_states"
from src.pipeline.common import download_hidden_states
kh.download_with_retries(lambda: download_hidden_states("tfidf", local_dir="/kaggle/working/hf_tfidf"), what="tfidf hidden states")
HS_TFIDF = "/kaggle/working/hf_tfidf/tfidf_distractor_rag/data/hidden_states"
n1 = len([f for f in os.listdir(HS_FULL10) if f.endswith(".pt")])
n2 = len([f for f in os.listdir(HS_TFIDF) if f.endswith(".pt")])
print("hidden-state files: full10", n1, "(expected 4078) | tfidf", n2, "(expected about 4000)")
assert n1 >= 4000 and n2 >= 3900, "download incomplete - run this cell again (it resumes)"
"""),
    code(r"""
# STEP 3 - settings for the runs.
MODELS = "lr_default,lr,svm,mlp"      # drop ",mlp" for a faster run
COMMON = ["--full10-data", "data/canonical/augmented_v2.jsonl", "--hs-dir-full10", HS_FULL10, "--hs-dir-tfidf", HS_TFIDF,
          "--old-json", "outputs/probing/results.json"]
OUT = kh.RESULTS / "probe_suite"
"""),
    code(r"""
# STEP 4 - PREFLIGHT: both settings, one layer, logistic regression only. Stops loudly here if anything is wrong.
PRE = kh.RESULTS / "probe_suite_preflight"
rc = kh.run([sys.executable, "lf_probe_suite.py", "--setting", "full10,tfidf", "--layers", "10", "--models", "lr_default,lr",
             "--n-boot", "50", "--fast", "--out", str(PRE)] + COMMON, "preflight_probe_suite")
ok = rc == 0 and (PRE / "full10" / "summary.md").exists() and (PRE / "tfidf" / "summary.md").exists()
assert ok, "PREFLIGHT FAILED - read the log above. Nothing long was started."
kh.show(PRE / "full10" / "summary.md")
# Reproduction of the OLD numbers: same rows as the old scripts (labelled hops that were never generated stay as zero vectors).
PRE_LEG = kh.RESULTS / "probe_suite_legacy_check"
rc = kh.run([sys.executable, "lf_probe_suite.py", "--setting", "full10", "--layers", "9,10,11,13", "--models", "lr_default",
             "--n-boot", "50", "--fast", "--legacy-rows", "--out", str(PRE_LEG)] + COMMON, "legacy_check_probe_suite")
kh.show(PRE_LEG / "full10" / "summary.md")
print("\nPREFLIGHT PASSED.")
print("Look at the 'Reproduction check' block of the LEGACY run: layers 9/10/11/13 should say MATCH (diff < 0.005).")
print("The main run uses corrected rows (never-generated hops dropped), so it can differ from the old numbers by a hair.")
"""),
    code(r"""
# STEP 5 - MAIN RUN (resumable: if the session stops, run this cell again).
rc = kh.run_with_retries([sys.executable, "lf_probe_suite.py", "--setting", "full10,tfidf", "--models", MODELS,
                          "--out", str(OUT)] + COMMON, "probe_suite", attempts=3)
print("main run exit code:", rc)
"""),
    code(r"""
# STEP 6 - FINAL (always runs): show and package whatever exists.
for s in ("full10", "tfidf"):
    kh.show(OUT / s / "summary.md")
kh.package(OUT, "probe_suite_results")
print("\nFiles:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()) if OUT.exists() else "(none)")
print("Download probe_suite_results.zip from the Output tab (or take files from results/probe_suite/).")
"""),
]


def main():
    for c in CELLS:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))                    # every code cell must parse
    nb = dict(cells=CELLS, nbformat=4, nbformat_minor=5,
              metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"},
                        "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}})
    out = ROOT / "kaggle" / "04_probe_suite.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out, f"({len(CELLS)} cells, all code cells parse)")


if __name__ == "__main__":
    main()
