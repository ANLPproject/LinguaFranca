"""
src/pipeline/judge.py
─────────────────────
Text-based failure trigger in the style of Doctor-RAG (Phase 4 baseline).

After the COMPLETE trajectory has been generated, the same base model is asked,
for hop 1 and hop 2, whether the step is correct and grounded in the context.
P(fail) = sigmoid(logit("no") - logit("yes")) from one forward pass per hop.
The earliest hop with P(fail) >= θ (default 0.5, i.e. the judge's own yes/no
decision) is the localized failure; retrieval + restart then proceed exactly as
for the probe trigger.

The prompt is the team's existing judge prompt (fixes-v2 evaluate_llm_baseline.py),
so detection numbers here are comparable to the standalone judge baseline.
"""

from __future__ import annotations

import time

import torch

SYSTEM = "You are a precise binary judge. Answer with exactly 'yes' or 'no' — no other text."
USER_TMPL = (
    "Context Information:\n{context}\n\n"
    "Question: {question}\n\n"
    "Task: Based ONLY on the context above, is the following reasoning step factually "
    "correct and properly grounded in the context? Answer 'no' if the step states "
    "something not supported by the context, even if it sounds plausible.\n"
    "Reasoning step: \"{hop}\"\n"
    "Answer:"
)


def _first_token_ids(tok, words) -> list[int]:
    ids = set()
    for w in words:
        for v in (w, " " + w):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


class Judge:
    def __init__(self, model, tok, date_string: str | None = None):
        self.model, self.tok = model, tok
        self.yes = _first_token_ids(tok, ["yes", "Yes", "YES"])
        self.no = _first_token_ids(tok, ["no", "No", "NO"])
        self.kw = {"date_string": date_string} if date_string else {}

    @torch.inference_mode()
    def p_fail(self, context: str, question: str, hop: str) -> dict:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(context=context, question=question, hop=hop)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.kw)
        dev = next(self.model.parameters()).device
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.model(**enc).logits[0, -1].float()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        diff = torch.logsumexp(logits[self.no], 0) - torch.logsumexp(logits[self.yes], 0)
        return dict(p_fail=float(torch.sigmoid(diff)), ntok=int(enc.input_ids.shape[1]), ms=ms)
