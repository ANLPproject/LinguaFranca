"""
kaggle_helpers.py
─────────────────
Helpers used by the notebooks in kaggle/.  All the logic lives here (not in
notebook cells) so it is unit-testable and cells stay tiny.

Design rules
────────────
* `run()` NEVER raises: it streams a subprocess's output, writes a log, and
  returns the exit code, so one failing step cannot abort the notebook.
* Preflight steps DO raise, but only at the very start (minutes in), with an
  explicit message — never after hours of work.
* Everything is written under RESULTS (= /kaggle/working/results/...), so it
  appears in the notebook's Output tab even if the final zip cell never runs.

Test hooks (used by my local end-to-end tests; harmless on Kaggle):
  LF_KAGGLE_INPUT, LF_KAGGLE_WORK : replace /kaggle/input and /kaggle/working
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

KAGGLE_INPUT = Path(os.environ.get("LF_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORK = Path(os.environ.get("LF_KAGGLE_WORK", "/kaggle/working"))
WORK = KAGGLE_WORK / "lf"
RESULTS = KAGGLE_WORK / "results"


def locate_bundle() -> Path:
    hits = sorted(KAGGLE_INPUT.rglob("lf_common.py"))
    if not hits:
        raise RuntimeError(
            "The bundle was not found under /kaggle/input.\n"
            "Fix: in the right-hand panel click 'Add Input' and add the Kaggle Dataset you created from "
            "linguafranca_bundle.zip, then run this cell again.")
    return hits[0].parent


def banner(msg: str) -> None:
    print("\n" + "=" * 90 + f"\n{msg}\n" + "=" * 90, flush=True)


def setup() -> Path:
    """Copy the bundle to a writable folder, cd into it, make it importable, check dependencies."""
    src = locate_bundle()
    if not WORK.exists():
        shutil.copytree(src, WORK)
    os.chdir(WORK)
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = Path("/kaggle/temp")
    if temp.exists() and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(temp / "hf")          # big model downloads go to non-persisted disk
    print(f"bundle: {src}\nworking dir: {WORK}\nresults dir: {RESULTS}")
    print(f"python {sys.version.split()[0]}; free disk: {shutil.disk_usage('/').free / 1e9:.0f} GB")
    for mod, pip_name in (("accelerate", "accelerate"), ("sklearn", "scikit-learn"), ("joblib", "joblib"),
                          ("huggingface_hub", "huggingface_hub")):
        try:
            __import__(mod)
        except ImportError:
            print(f"installing missing dependency: {pip_name}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)
    try:
        import torch
        import transformers
        print(f"torch {torch.__version__}, transformers {transformers.__version__}, cuda={torch.cuda.is_available()}")
    except Exception as e:
        print("[warn] could not import torch/transformers:", e)
    return WORK


def hf_login() -> str:
    """Log in with the Kaggle secret HF_TOKEN (or env). Returns the token ('' if none)."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = ""
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            os.environ["HF_TOKEN"] = token
            print("Logged in to Hugging Face.")
        except Exception as e:
            print(f"[warn] HF login failed ({e}); continuing without a token")
            token = ""
    else:
        print("No HF_TOKEN secret found. Ungated models and the ungated Llama mirrors will still work.")
    return token


def require_gpu(min_total_gb: float = 14.0, allow_cpu: bool = False) -> float:
    import torch
    if not torch.cuda.is_available():
        if allow_cpu or os.environ.get("LF_ALLOW_CPU"):
            print("[test mode] no GPU; continuing on CPU")
            return 0.0
        raise RuntimeError("No GPU detected. In the right panel: Session options -> Accelerator -> 'GPU T4 x2' "
                           "(or 'GPU T4 x1' for notebook 03), then restart and Run All.")
    n = torch.cuda.device_count()
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
    names = ", ".join(torch.cuda.get_device_properties(i).name for i in range(n))
    print(f"GPUs: {n} x [{names}], total memory {total:.0f} GB")
    if total < min_total_gb:
        raise RuntimeError(f"Only {total:.0f} GB of GPU memory; this notebook needs >= {min_total_gb:.0f} GB.")
    return total


def run(cmd: list[str], log_name: str, env_extra: dict | None = None) -> int:
    """Stream a subprocess to the notebook and to a log file. Never raises. Returns the exit code."""
    log = RESULTS / "logs"; log.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env_extra or {})}
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        with open(log / f"{log_name}.log", "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 cwd=str(WORK), env=env, encoding="utf-8", errors="replace")
            for line in p.stdout:
                lf.write(line)
                if "Loading weights" in line or "it/s]" in line or "Fetching" in line:
                    continue                              # skip progress-bar spam in the notebook
                print(line, end="")
            rc = p.wait()
    except Exception as e:
        print(f"[run] could not execute: {e}")
        return 1
    print(f"[run] exit code {rc} after {time.time() - t0:.0f}s", flush=True)
    return rc


def run_with_retries(cmd: list[str], log_name: str, attempts: int = 3, env_extra: dict | None = None) -> int:
    """Re-run a RESUMABLE command until it succeeds (the scripts skip finished work)."""
    rc = 1
    for a in range(1, attempts + 1):
        if a > 1:
            print(f"\n[retry {a}/{attempts}] resuming (finished work is skipped)...")
        rc = run(cmd, f"{log_name}_try{a}", env_extra)
        if rc == 0:
            break
    return rc


def check_models(models: list[str], hf_token: str) -> list[tuple[str, str, str]]:
    """Return [(requested, usable_repo_or_'', message)], accessibility checked via config.json."""
    from lf_models import resolve_model
    out = []
    for m in models:
        repo, msg = resolve_model(m, hf_token or None)
        print(("  OK   " if repo else "  FAIL ") + msg)
        out.append((m, repo or "", msg))
    return out


def download_with_retries(fn, attempts: int = 4, what: str = "download"):
    for a in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{what}] attempt {a}/{attempts} failed: {e.__class__.__name__}: {str(e)[:150]}")
            time.sleep(5 * a)
    raise RuntimeError(f"{what} failed after {attempts} attempts; check that Internet is ON for this notebook "
                       "(right panel -> Session options -> Internet).")


def package(src_dir: Path, zip_name: str) -> Path | None:
    """Zip a results folder into /kaggle/working/<zip_name>.zip (never raises)."""
    try:
        src_dir = Path(src_dir)
        out = KAGGLE_WORK / f"{zip_name}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(src_dir.rglob("*")):
                if p.is_file() and "partial_" not in p.name:
                    z.write(p, p.relative_to(src_dir.parent).as_posix())
        print(f"packaged {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    except Exception as e:
        print(f"[package] failed: {e} (the files are still in {src_dir}, visible in the Output tab)")
        return None


def show(path: Path) -> None:
    path = Path(path)
    print(path.read_text(encoding="utf-8") if path.exists() else f"(missing: {path})")
