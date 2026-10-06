"""
lf_models.py
────────────
Small model-loading / hooking helpers shared by evaluate_llm_baseline.py and
causal_patching.py.  Works with transformers 4.4x and 5.x.
"""

from __future__ import annotations

import torch


def load_causal_lm(name: str, dtype: str = "float16", device_map: str | None = "auto", token: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if not torch.cuda.is_available():
        td, device_map = torch.float32, None          # CPU smoke tests
    tok = AutoTokenizer.from_pretrained(name, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map=device_map, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=td, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=td, **kw)
    model.eval()
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
