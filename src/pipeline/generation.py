"""
src/pipeline/generation.py
──────────────────────────
Live generation with hop-boundary probing (Phase 3, steps 1–3 of the proposal).

generate_monitored() runs greedy decoding token by token with model.forward()
and a KV cache.  Forward hooks copy every decoder block's output for each
generated token into a GPU buffer.  As soon as the decoded text contains a
complete <hopK>…</hopK> (K = 1, 2), the hop's tokens are mean-pooled per layer
and scored by the hop-K probe.  With halt=True generation stops when
P(fail) >= τ_K; otherwise the scores are only recorded (monitor mode).

Feature parity with Phase 1 (the probes were trained on its features)
─────────────────────────────────────────────────────────────────────
* the prompt is tokenized exactly like Phase 1 (`tokenizer(prompt)`, which
  adds a second <|begin_of_text|> in front of the chat-template string);
* the pooled span is the hop CONTENT (between the tags), mapped from
  characters to tokens with Phase 1's rule (first token containing the first
  char … token containing the last char);
* layer L = output of block L; for the last block the final RMSNorm is applied,
  matching HF hidden_states[-1] which Phase 1 stored for layer 27.
phase1_features() re-implements the Phase 1 extraction, for parity tests.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.pipeline.common import MODEL_MIRRORS, MODEL_NAME, hop_match, parse_answer


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _can_fetch(repo: str, token) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo, "config.json", token=token)
        return True
    except Exception:
        return False


def load_model(name: str = MODEL_NAME, token: str | None = None):
    """Greedy-only fp16 model on one GPU (or fp32 on CPU).  Falls back to an ungated mirror."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = name
    if not _can_fetch(name, token):
        repo = next((m for m in MODEL_MIRRORS if _can_fetch(m, token)), None)
        if repo is None:
            raise RuntimeError(f"Cannot access {name} (accept the licence on huggingface.co) or any mirror.")
        print(f"[model] {name} not accessible with this token -> using mirror {repo} (same weights)")
    cuda = torch.cuda.is_available()
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(repo, token=token, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, token=token, device_map={"": 0} if cuda else None,
                                                 **{dkw: torch.float16 if cuda else torch.float32})
    model.eval()
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    return model, tok


def eos_ids(model, tok) -> set[int]:
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def _device(model):
    return next(model.parameters()).device


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture
# ─────────────────────────────────────────────────────────────────────────────

class _Capture:
    """Forward hooks that store each block's output for the current 1-token decode step."""

    def __init__(self, model, max_tokens: int):
        layers = model.model.layers
        self.n_layers = len(layers)
        self.buf = torch.zeros(self.n_layers, max_tokens, model.config.hidden_size,
                               dtype=next(model.parameters()).dtype, device=_device(model))
        self.pos = -1
        self.norm = model.model.norm
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]

    def _hook(self, i):
        last = i == self.n_layers - 1

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if self.pos >= 0 and hs.shape[1] == 1:
                h = hs[0, -1]
                self.buf[i, self.pos] = self.norm(h) if last else h
        return hook

    def pooled(self, t_start: int, t_end_incl: int) -> np.ndarray:
        """[n_layers, d] float16 mean over tokens t_start..t_end_incl (mean in fp32, stored as fp16 like Phase 1)."""
        return self.buf[:, t_start:t_end_incl + 1].float().mean(1).to(torch.float16).cpu().numpy()

    def remove(self):
        for h in self.handles:
            h.remove()


def char_span_to_tokens(starts, ends, c_start: int, c_end: int):
    """Phase 1's rule: first token with cs <= c_start < ce, first token with cs < c_end <= ce."""
    ts = next((i for i, (cs, ce) in enumerate(zip(starts, ends)) if cs <= c_start < ce), None)
    te = next((i for i, (cs, ce) in enumerate(zip(starts, ends)) if cs < c_end <= ce), None)
    return ts, te


# ─────────────────────────────────────────────────────────────────────────────
# Live generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_monitored(model, tok, prompt: str, probes=None, halt: bool = False,
                       max_new_tokens: int = 400, hops=(1, 2), keep_pooled: bool = True) -> dict:
    """
    Greedy generation with probing at </hop1> and </hop2>.

    Returns a dict with the generated text and answer, token counts and timings,
    and per boundary K: n_tokens (generated so far), t_ms (since start), p_fail,
    probe_ms; `stopped_at` = K if the probe halted generation.  If keep_pooled,
    "_pooled" maps K -> [n_layers, d] float16 features (not JSON-serialisable).
    """
    dev = _device(model)
    stop = eos_ids(model, tok)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)          # Phase 1 tokenization (double BOS)
    cap = _Capture(model, max_new_tokens)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = model(input_ids=ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        t_prefill = (time.perf_counter() - t0) * 1000
        gen, ends, text = [], [], ""
        boundaries, pooled, stopped = {}, {}, None
        pending = list(hops)
        for _ in range(max_new_tokens):
            nxt = int(torch.argmax(logits))
            if nxt in stop:
                break
            gen.append(nxt)
            cap.pos = len(gen) - 1                                       # feed it: its hidden state is now captured
            out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
            cap.pos = -1
            text = tok.decode(gen, skip_special_tokens=True)
            ends.append(len(text))
            while pending:
                k = pending[0]
                m = hop_match(text, k)
                if m is None:
                    break
                pending.pop(0)
                starts = [0] + ends[:-1]
                ts, te = char_span_to_tokens(starts, ends, m.start(1), m.end(1))
                b = dict(n_tokens=len(gen), t_ms=(time.perf_counter() - t0) * 1000, p_fail=None, probe_ms=0.0,
                         span=[ts, te])
                if ts is not None and te is not None and te >= ts:
                    tp = time.perf_counter()
                    feats = cap.pooled(ts, te)
                    if probes is not None and k in probes.layer:
                        b["p_fail"] = probes.p_fail(k, feats)
                    b["probe_ms"] = (time.perf_counter() - tp) * 1000
                    if keep_pooled:
                        pooled[k] = feats
                boundaries[k] = b
                if halt and b["p_fail"] is not None and b["p_fail"] >= probes.tau[k]:
                    stopped = k
                    break
            if stopped is not None:
                break
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res = dict(text=text, answer=parse_answer(text), n_prompt=int(ids.shape[1]), n_gen=len(gen),
                   t_prefill_ms=t_prefill, t_total_ms=(time.perf_counter() - t0) * 1000,
                   boundaries={str(k): v for k, v in boundaries.items()}, stopped_at=stopped)
        if keep_pooled:
            res["_pooled"] = pooled
        return res
    finally:
        cap.remove()
        del cap


@torch.inference_mode()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 400, batch_size: int = 8) -> list[dict]:
    """Plain batched greedy generation (restarts).  Prompts are length-sorted; OOM halves the batch."""
    dev = _device(model)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[dict | None] = [None] * len(prompts)
    i, bs = 0, batch_size
    while i < len(order):
        idx = order[i:i + bs]
        try:
            enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(dev)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[generate_batch] OOM -> batch size {bs}")
            continue
        plen = enc.input_ids.shape[1]
        stop = eos_ids(model, tok) | {tok.pad_token_id}
        for row, j in enumerate(idx):
            new = out[row, plen:].tolist()
            n_gen = next((t for t, x in enumerate(new) if x in stop), len(new))
            text = tok.decode(new, skip_special_tokens=True)
            results[j] = dict(text=text, answer=parse_answer(text), n_gen=n_gen,
                              n_prompt=int(enc.attention_mask[row].sum()))
        i += len(idx)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 feature extraction (reference implementation for parity tests)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def phase1_features(model, tok, prompt: str, cot: str, hops=(1, 2)) -> dict:
    """Same computation as phase1_dataset/generate_cot.extract_hidden_states: K -> [n_layers, d] float16."""
    dev = _device(model)
    full = tok(prompt + cot, return_tensors="pt").input_ids[0]
    p_len = len(tok(prompt, return_tensors="pt").input_ids[0])
    enc = tok(cot, return_offsets_mapping=True, add_special_tokens=False)
    if len(full) != p_len + len(enc["input_ids"]):
        raise ValueError("prompt/generation boundary is not token-aligned")
    offs = enc["offset_mapping"]
    hs = model(full.unsqueeze(0).to(dev), output_hidden_states=True).hidden_states
    out = {}
    for k in hops:
        m = hop_match(cot, k)
        if m is None:
            continue
        ts, te = char_span_to_tokens([o[0] for o in offs], [o[1] for o in offs], m.start(1), m.end(1))
        if ts is None or te is None:
            continue
        a, b = p_len + ts, p_len + te + 1
        out[k] = torch.stack([hs[L + 1][0, a:b].mean(0) for L in range(len(hs) - 1)]).to(torch.float16).cpu().numpy()
    return out
