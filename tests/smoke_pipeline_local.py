"""
CPU smoke test of the full src.pipeline.run plumbing (no GPU, no 3B model).
Run from project root:  python tests/smoke_pipeline_local.py [out_dir]

The 3B model is replaced by a tiny random Llama whose logits are scripted: for
a 2Wiki prompt it emits that question's stored Phase 1 CoT, for any other
prompt a fixed 2-hop CoT.  Probes are random linear probes on the tiny model's
64-dim states.  The retrieval corpus is built from a small synthetic raw file.
Every stage runs except `parity` (it needs the real model's 3072-dim states).
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import src.pipeline.generation as generation
import src.pipeline.retriever as retriever
from src.pipeline.common import canonical_split, hidden_state_ids, load_setting_examples, read_jsonl

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="lf_smoke_"))
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
N = 4

tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct", padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
EOS = tok.convert_tokens_to_ids("<|eot_id|>")
DEFAULT_COT = ("<hop1>From the context, I find that the entity is Alpha Beta.</hop1>\n"
               "<hop2>From the context, I find that Alpha Beta was born in Gamma.</hop2>\n<answer>Gamma</answer>")

# ── data: real tfidf examples, canonical split ─────────────────────────────
examples = load_setting_examples("tfidf", hidden_state_ids("tfidf"))
tr, te = canonical_split(examples)
cot_by_question = {e["question"]: e["generated_cot"] for e in examples}

# ── fake probes on the tiny model (64-dim, 4 layers) ───────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.probes import export_portable

pdir = OUT / "probes"
pdir.mkdir()
rng = np.random.default_rng(0)
for t in ("hop1", "hop2", "hop12"):
    clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(rng.normal(size=(40, 64)), np.r_[np.zeros(20), np.ones(20)])
    export_portable(clf, pdir / f"{t}.npz")
meta = dict(setting="tfidf", test_ids=[examples[i]["id"] for i in te], train_ids=[examples[i]["id"] for i in tr],
            cv_auroc={"hop1": {str(L): 0.8 + 0.01 * L for L in range(4)}},
            probes={"hop1": dict(layer=2, tau=0.5), "hop2": dict(layer=3, tau=0.5), "hop12": dict(layer=3, tau=0.5)})
(pdir / "probes_meta.json").write_text(json.dumps(meta))


# ── scripted tiny model ────────────────────────────────────────────────────
def tiny_scripted_model(name=None, token=None):
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = EOS
    state = {"scripts": [], "step": []}

    def pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[1] > 1:                    # prefill: one script per row
            scripts = []
            for row in ids:
                text = tok.decode(row, skip_special_tokens=False)
                q = text.rsplit("Question: ", 1)[-1].split("<|eot_id|>")[0]
                scripts.append(tok(cot_by_question.get(q, DEFAULT_COT), add_special_tokens=False).input_ids)
            state["scripts"], state["step"] = scripts, [0] * len(scripts)

    def post(module, args, out):
        out = out.clone()
        for b in range(out.shape[0]):
            s, i = state["scripts"][b], state["step"][b]
            nxt = s[i] if i < len(s) else EOS
            state["step"][b] += 1
            out[b, -1, :] = -1e4
            out[b, -1, nxt] = 1e4
        return out

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.lm_head.register_forward_hook(post)
    return model, tok


generation.load_model = tiny_scripted_model


# ── small raw 2Wiki file for the corpus ────────────────────────────────────
def fake_raw_file(path=None):
    raw = OUT / "raw_2wiki.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        for e in examples[:300]:
            blocks = [b for b in e["context"].split("\n\nTitle: ")]
            titles, sents = [], []
            for b in blocks:
                b = b[len("Title: "):] if b.startswith("Title: ") else b
                t, _, txt = b.partition("\n")
                titles.append(t); sents.append([txt])
            f.write(json.dumps({"_id": e["id"], "context": {"title": titles, "sentences": sents}}) + "\n")
    return str(raw)


retriever.hf_file = fake_raw_file

# ── run every stage except parity ──────────────────────────────────────────
from src.pipeline import run

sys.argv = ["run", "--setting", "tfidf", "--out", str(OUT), "--probe-dir", str(pdir),
            "--stages", "evalset,corpus,s0,judge,restarts,live,evaluate",
            "--n-2wiki", str(N), "--n-hotpot", str(N), "--n-live", "4", "--n-extra-corpus", "50",
            "--max-new-tokens", "400", "--batch-size", "3"]
run.main()

# ── checks ─────────────────────────────────────────────────────────────────
items = read_jsonl(OUT / "eval_set.jsonl")
s0 = {r["uid"]: r for r in read_jsonl(OUT / "s0.jsonl")}
assert len(items) == 2 * N and len(s0) == 2 * N
wiki = [it for it in items if it["dataset"] == "2wiki"]
assert all(s0[it["uid"]]["reproduced"] for it in wiki), "scripted model must reproduce the stored CoT"
assert all("1" in s0[it["uid"]]["boundaries"] for it in items)
rs = read_jsonl(OUT / "restarts.jsonl")
assert {r["key"] for r in rs} == {"q", "h1", "h2"} and all(len(r["retrieved_titles"]) == 3 for r in rs)
for r in rs:
    it = next(i for i in items if i["uid"] == r["uid"])
    assert not set(r["retrieved_titles"]) & set(it["context_titles"]), "restart must retrieve passages not in context"
res = json.loads((OUT / "results.json").read_text())
assert set(res["end_to_end"]) == {"2wiki", "hotpotqa"}
eq = res["latency_live_subset"]["equivalence"]
assert eq["first_pass_is_prefix"] == 1.0 and eq["same_trigger"] == 1.0, eq
for f in ("results.md", "tables.tex", "sweep.csv", "per_item.jsonl"):
    assert (OUT / f).exists(), f
print(f"\n[SMOKE PASS] all stages ran; outputs in {OUT}")
