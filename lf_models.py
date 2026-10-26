"""
lf_models.py
────────────
Model-loading / hooking helpers shared by evaluate_llm_baseline.py and
causal_patching.py.  Works with transformers 4.4x - 5.x.

Robustness features (these exist so a long Kaggle run does not die late):
  * dtype kwarg handled across transformers versions;
  * automatic fallback from a gated repo (meta-llama/...) to an ungated mirror
    with the same weights (unsloth/...) when the HF token lacks access;
  * `resolve_model()` checks access BEFORE any download;
  * `free_model_cache()` deletes a model's downloaded files to keep Kaggle's
    disk from filling when several large models are run in sequence.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

import torch

# Same weights, no licence click-through needed.  Used only if the official repo is inaccessible.
MIRRORS = {
    "meta-llama/Llama-3.2-3B-Instruct": ["unsloth/Llama-3.2-3B-Instruct"],
    "meta-llama/Llama-3.1-8B-Instruct": ["unsloth/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"],
}
# Rough fp16 download+load footprint in GB (disk), used for the free-space check.
APPROX_GB = {"3b": 8, "7b": 17, "8b": 18}


def _can_fetch(repo: str, token: str | None) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo, "config.json", token=token)
        return True
    except Exception:
        return False


def resolve_model(name: str, token: str | None = None) -> tuple[str | None, str]:
    """Return (usable_repo_or_None, message).  Tries the official repo, then mirrors."""
    if _can_fetch(name, token):
        return name, f"{name}: accessible"
    for alt in MIRRORS.get(name, []):
        if _can_fetch(alt, token):
            return alt, f"{name}: NOT accessible with this token -> using ungated mirror {alt} (same weights)"
    return None, (f"{name}: NOT accessible (gated or network problem) and no mirror available. "
                  f"Accept the licence at https://huggingface.co/{name} with the account that owns HF_TOKEN.")


def approx_download_gb(name: str) -> int:
    n = name.lower()
    for k, v in APPROX_GB.items():
        if k in n:
            return v
    return 10


def free_disk_gb(path: str = "/") -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 1e9


def free_model_cache(name: str) -> None:
    """Delete the downloaded files of one model from the HF cache (frees GBs on Kaggle)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    d = Path(HF_HUB_CACHE) / ("models--" + name.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dtype_kwarg() -> str:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9


def load_causal_lm(name: str, dtype: str = "float16", device_map: str | None = "auto", token: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:                                   # silence per-call warnings that flood long logs
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if not torch.cuda.is_available():
        td, device_map = torch.float32, None          # CPU smoke tests
    tok = AutoTokenizer.from_pretrained(name, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, token=token, **{_dtype_kwarg(): td})
    model.eval()
    # Never sample, whatever the repo's generation_config says (Llama-3.2-Instruct ships do_sample=True, T=0.6).
    gc_ = model.generation_config
    gc_.do_sample = False
    gc_.temperature = None
    gc_.top_p = None
    gc_.top_k = None
    return model, tok


def input_device(model):
    return next(model.parameters()).device


def decoder_layers(model):
    """The list of transformer blocks (Llama / Qwen / Mistral layout)."""
    return model.model.layers


def replace_positions_hook(positions, vectors):
    """
    Forward hook for a decoder block: overwrite the block OUTPUT (residual
    stream after block L == HF hidden_states[L+1]) at `positions` with
    `vectors` [len(positions), d].  Only fires on a pass that covers those
    positions (the prefill), never on 1-token decode steps.
    """
    pos = torch.as_tensor(positions, dtype=torch.long)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] <= int(pos.max()):
            return output
        hs = hs.clone()
        hs[0, pos.to(hs.device), :] = vectors.to(device=hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    return hook
