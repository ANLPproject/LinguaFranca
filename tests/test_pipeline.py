"""
Phase 3–5 pipeline unit tests — no GPU required.
Run from project root: python tests/test_pipeline.py

The generation tests use a tiny random Llama with scripted logits and the real
Llama-3 tokenizer (ungated mirror; skipped if it cannot be downloaded).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

print("=" * 55)
print("  LinguaFranca Phase 3–5 pipeline — Unit Tests")
print("=" * 55)

from src.pipeline.common import (answer_correct, answer_em, answer_f1, apply_extra_hop_rule, best_f05_threshold,
                                 context_titles, detection_metrics, format_passages, hop_text, parse_answer,
                                 prompt_from_template, prompt_with_evidence)
from src.pipeline.systems import judge_trigger, outcome, probe_trigger

# A prompt in the exact shape of a stored Phase 1 prompt (system few-shot example also has "Context:\n")
SYS = ("You are a careful multi-hop reasoning assistant.\n\nExample:\nContext:\nTitle: Inception\nInception is a film.\n\n"
       "Question: What nationality is the director of Inception?\n<hop1>…</hop1>")
PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
          "Today Date: 21 Oct 2026\n\n" + SYS + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
          "Context:\nTitle: A\nA text.\n\nTitle: B\nB text.\n\nQuestion: Who is X?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")

# ── Test 1: prompt surgery ─────────────────────────────────────
p2 = prompt_from_template(PROMPT, "Title: C\nC text.", "Where is Y?")
assert p2.startswith(PROMPT[:PROMPT.index("<|start_header_id|>user")]), "system part must be untouched"
assert "Context:\nTitle: C\nC text.\n\nQuestion: Where is Y?<|eot_id|><|start_header_id|>assistant" in p2
assert "Title: A" not in p2 and "Inception is a film." in p2
p3 = prompt_with_evidence(PROMPT, [{"title": "R1", "text": "r one"}, {"title": "R2", "text": "r two"}])
assert "Context:\nTitle: R1\nr one\n\nTitle: R2\nr two\n\nTitle: A\nA text." in p3
assert p3.count("Title: Inception") == 1 and prompt_with_evidence(PROMPT, []) == PROMPT
assert context_titles(p3[p3.index("<|start_header_id|>user"):]) == ["R1", "R2", "A", "B"]
print("[PASS] prompt_from_template / prompt_with_evidence (user-turn Context only)")

# ── Test 2: parsing + answer metrics ───────────────────────────
cot = "<hop1>From the context, X is Ana.</hop1>\n<hop2>Ana was born in Lima.</hop2>\n<answer>Lima, Peru</answer>"
assert hop_text(cot, 1) == "From the context, X is Ana." and hop_text(cot, 2) == "Ana was born in Lima."
assert hop_text(cot, 3) == "" and parse_answer(cot) == "Lima, Peru" and parse_answer("no tags") == ""
assert answer_em("The Beatles", "beatles") == 1.0 and answer_f1("Lima, Peru", "Lima") == 2 / 3
assert answer_correct("Lima, Peru", "Lima") and not answer_correct("Paris", "Lima")
assert format_passages([{"title": "T", "text": "x"}]) == "Title: T\nx"
print("[PASS] hop_text / parse_answer / EM / F1")

# ── Test 3: extra-hop rule ─────────────────────────────────────
ex = {"reasoning_graph": [{"hop": 1}, {"hop": 2}],
      "hops": [{"hop_idx": 1, "label": 0}, {"hop_idx": 2, "label": 0}, {"hop_idx": 3, "label": 1}]}
apply_extra_hop_rule(ex)
assert ex["hops"][2]["label"] == -1 and ex["first_fail_hop"] is None
ex["hops"][1]["label"] = 1
assert apply_extra_hop_rule(ex)["first_fail_hop"] == 2
print("[PASS] apply_extra_hop_rule (hops beyond the gold graph are unlabeled)")

# ── Test 4: F0.5 threshold + detection metrics ─────────────────
y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.2, 0.7, 0.6, 0.8, 0.9])
t, f = best_f05_threshold(y, p)
assert t == 0.8 and abs(f - 1.25 * (2 / 3) / (0.25 + 2 / 3)) < 1e-9, (t, f)   # P=1, R=2/3 beats P=.75, R=1
m = detection_metrics(y, p >= 0.6, p)
assert m["precision"] == 0.75 and m["recall"] == 1.0 and abs(m["fpr"] - 1 / 3) < 1e-9 and 0 < m["auroc"] <= 1
print("[PASS] best_f05_threshold / detection_metrics")

# ── Test 5: systems & token accounting ─────────────────────────
item = {"gold_answer": "Lima", "first_fail": 2}
s0 = {"n_prompt": 1000, "n_gen": 60, "answer": "Paris",
      "boundaries": {"1": {"n_tokens": 20, "p_fail": 0.1}, "2": {"n_tokens": 40, "p_fail": 0.9}}}
rs = {"q": {"answer": "Lima", "n_gen": 55, "n_prompt": 1300},
      "h1": {"answer": "Rome", "n_gen": 50, "n_prompt": 1310},
      "h2": {"answer": "Lima", "n_gen": 45, "n_prompt": 1320}}
tau = {1: 0.5, 2: 0.5}
assert probe_trigger({1: 0.1, 2: 0.9}, tau) == 2 and probe_trigger({1: 0.6, 2: 0.9}, tau) == 1
assert probe_trigger({1: 0.1}, tau) is None
jrec = {"1": {"p_fail": 0.7, "ntok": 900}, "2": {"p_fail": 0.2, "ntok": 910}}
assert judge_trigger(jrec) == 1 and judge_trigger(jrec, theta=0.8) is None and judge_trigger(None) is None
o = outcome("no_intervention", item, s0, rs)
assert not o["correct"] and o["gen_tokens"] == 60 and o["processed_tokens"] == 1060 and not o["fired"]
o = outcome("probe_restart", item, s0, rs, probe_scores={1: 0.1, 2: 0.9}, tau=tau)
assert o["correct"] and o["hop"] == 2 and o["gen_tokens"] == 40 + 45 and o["detect_tokens"] == 40
assert o["processed_tokens"] == 1000 + 40 + 1320 + 45
o = outcome("judge_restart", item, s0, rs, judge_rec=jrec)
assert o["hop"] == 1 and o["answer"] == "Rome" and o["gen_tokens"] == 60 + 50 and o["detect_tokens"] == 60
assert o["processed_tokens"] == 1000 + 60 + 1810 + 1310 + 50
o = outcome("judge_restart", item, s0, rs, judge_rec=jrec, theta=0.8)
assert not o["fired"] and o["processed_tokens"] == 1060 + 1810, "judge cost is paid even without a restart"
o = outcome("always_retrieve", item, s0, rs)
assert o["correct"] and o["gen_tokens"] == 55 and o["processed_tokens"] == 1355
o = outcome("oracle_restart", item, s0, rs)
assert o["hop"] == 2 and o["correct"]
o = outcome("oracle_restart", {"gold_answer": "Lima", "first_fail": 2}, dict(s0, boundaries={"1": s0["boundaries"]["1"]}), rs)
assert not o["fired"], "oracle cannot act on a hop that never closed"
print("[PASS] systems: triggers, answers and token accounting")

# ── Test 5b: second restart round ──────────────────────────────
rs2 = dict(rs, h2=dict(rs["h2"], answer="Paris", rescore={
    "probe": {"1": {"n_tokens": 18, "p_fail": 0.2}, "2": {"n_tokens": 35, "p_fail": 0.95}},
    "judge": {"1": {"p_fail": 0.1, "ntok": 950}, "2": {"p_fail": 0.9, "ntok": 960}}}))
rs2["h2>2"] = {"answer": "Lima", "n_gen": 40, "n_prompt": 1500}
o1 = outcome("probe_restart", item, s0, rs2, probe_scores={1: 0.1, 2: 0.9}, tau=tau, rounds=1)
o2 = outcome("probe_restart", item, s0, rs2, probe_scores={1: 0.1, 2: 0.9}, tau=tau, rounds=2)
assert not o1["correct"] and o1["n_restarts"] == 1
assert o2["correct"] and o2["n_restarts"] == 2 and o2["gen_tokens"] == 40 + 35 + 40
assert o2["processed_tokens"] == 1000 + 40 + 1320 + 35 + 1500 + 40
jr2 = {"1": {"p_fail": 0.1, "ntok": 900}, "2": {"p_fail": 0.8, "ntok": 910}}
o3 = outcome("judge_restart", item, s0, rs2, judge_rec=jr2, rounds=2)
assert o3["correct"] and o3["n_restarts"] == 2 and o3["gen_tokens"] == 60 + 45 + 40
assert o3["processed_tokens"] == 1000 + 60 + 1810 + 1320 + 45 + 1910 + 1500 + 40, "judge pays for re-checking restart 1"
print("[PASS] second restart round: probe halts restart 1 at </hop2>; judge re-checks the full restart")

# ── Test 5c: bridge entity + repair queries ────────────────────
from src.pipeline.common import bridge_entity, repair_query
assert bridge_entity("From the context, I find that the director of film X is Arcady Boytler.") == "Arcady Boytler"
assert bridge_entity("From the context, I find that Stewart Raffill is the director of The Tender Warrior.") == "Stewart Raffill"
assert bridge_entity("From the context, I find that there is no mention of the composer.") == ""
hops = {"1": "From the context, I find that the director of film X is Arcady Boytler.",
        "2": "There is no information about where he was born."}
assert repair_query("Where was the director of X born?", hops, 0) == "Where was the director of X born?"
assert repair_query("Q?", hops, 1, "v2") == "Q? " + hops["1"] == repair_query("Q?", hops, 1, "v1")
assert repair_query("Q?", hops, 2, "v1") == "Q? " + hops["2"]
assert repair_query("Q?", hops, 2, "v2") == "Arcady Boytler. Q? " + hops["2"]
print("[PASS] bridge_entity / repair_query (v2 leads hop-2 queries with the hop-1 entity)")

# ── Test 5d: Platt calibration is monotone and keeps decisions ─
from src.pipeline.probes import apply_platt, ece, fit_platt
rng0 = np.random.default_rng(1)
yy0 = rng0.integers(0, 2, 400)
pp0 = np.clip(0.7 + 0.29 * yy0 + rng0.normal(0, 0.05, 400), 0.01, 0.999)    # over-confident scores
cal = fit_platt(yy0, pp0)
pc0 = apply_platt(cal, pp0)
assert cal["a"] > 0 and np.all(np.diff(pc0[np.argsort(pp0)]) >= -1e-12), "calibration must be monotone"
t_raw = 0.9
assert np.array_equal(pp0 >= t_raw, pc0 >= apply_platt(cal, t_raw)), "same decisions at the mapped threshold"
assert ece(yy0, pc0) < ece(yy0, pp0)
print("[PASS] Platt calibration: monotone, decision-preserving, lower ECE")

# ── Tests 6-8: live generation with a tiny scripted model ──────
try:
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
except Exception as e:  # noqa: BLE001
    tok = None
    print(f"[SKIP] generation tests (tokenizer unavailable: {e.__class__.__name__})")

if tok is not None:
    from src.pipeline.generation import char_span_to_tokens, generate_monitored, phase1_features
    from src.pipeline.probes import ProbeBundle

    torch.manual_seed(0)
    EOS = tok.convert_tokens_to_ids("<|eot_id|>")
    COT = ("<hop1>From the context, I find that the director of Águila o sol is Arcady Boytler.</hop1>\n"
           "<hop2>From the context, I find that Arcady Boytler was born in Moscow, Russia.</hop2>\n"
           "<answer>Moscow, Russia</answer>")
    script = tok(COT, add_special_tokens=False).input_ids

    def scripted_model():
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
        model = LlamaForCausalLM(cfg).eval()
        model.generation_config.eos_token_id = EOS
        state = {"i": 0}

        def hook(module, args, out):
            i = state["i"]
            state["i"] += 1
            nxt = script[i] if i < len(script) else EOS
            out = out.clone()
            out[:, -1, :] = -1e4
            out[:, -1, nxt] = 1e4
            return out

        model.lm_head.register_forward_hook(hook)
        return model, state

    model, state = scripted_model()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "Context:\nTitle: A\nA text.\n\nQuestion: Q?"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026")

    # Test 6: monitor mode reproduces the scripted text and finds both boundaries
    r = generate_monitored(model, tok, prompt, probes=None, halt=False, max_new_tokens=120)
    assert r["text"] == COT, r["text"]
    assert r["answer"] == "Moscow, Russia" and r["n_gen"] == len(script) and r["stopped_at"] is None
    b1, b2 = r["boundaries"]["1"], r["boundaries"]["2"]
    # the closing tag can end inside a token such as ">\n"
    assert tok.decode(script[:b1["n_tokens"]]).rstrip().endswith("</hop1>") and b1["n_tokens"] < b2["n_tokens"] < r["n_gen"]
    assert not tok.decode(script[:b1["n_tokens"] - 1]).rstrip().endswith("</hop1>"), "boundary detected at the first possible token"
    assert set(r["_pooled"]) == {1, 2} and r["_pooled"][1].shape == (4, 64) and r["_pooled"][1].dtype == np.float16
    print("[PASS] generate_monitored: scripted text, answer, boundaries at </hop1> and </hop2>")

    # Test 7: live KV-cache features == Phase 1 full-forward features (all layers incl. the final-norm layer)
    ref = phase1_features(model, tok, prompt, COT)
    for k in (1, 2):
        a, b = r["_pooled"][k].astype(np.float32), ref[k].astype(np.float32)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        assert cos.min() > 0.9999 and np.abs(a - b).max() < 2e-2, (k, cos, np.abs(a - b).max())
    starts, ends = [0, 3, 7], [3, 7, 10]
    assert char_span_to_tokens(starts, ends, 3, 9) == (1, 2) and char_span_to_tokens(starts, ends, 0, 3) == (0, 0)
    print("[PASS] live hop features match the Phase 1 extraction (cosine > 0.9999 at every layer)")

    # Test 8: a real ProbeBundle halts generation at </hop1>
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 64)); yy = np.r_[np.zeros(20), np.ones(20)]
        from src.pipeline.probes import LinearProbe, export_portable
        for t in ("hop1", "hop2"):
            clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, yy)
            joblib.dump(clf, Path(d) / f"{t}.joblib")
        export_portable(clf, Path(d) / "check.npz")
        assert np.allclose(LinearProbe(Path(d) / "check.npz").p_fail(X), clf.predict_proba(X)[:, 1], atol=1e-9)
        meta = {"probes": {"hop1": {"layer": 2, "tau": 0.0}, "hop2": {"layer": 3, "tau": 1.01}}}
        Path(d, "probes_meta.json").write_text(json.dumps(meta))
        pb = ProbeBundle(d, mode="dedicated")
        state["i"] = 0
        h = generate_monitored(model, tok, prompt, probes=pb, halt=True, max_new_tokens=120)
        assert h["stopped_at"] == 1 and h["n_gen"] == b1["n_tokens"] and h["text"].rstrip().endswith("</hop1>")
        assert r["text"].startswith(h["text"]) and h["boundaries"]["1"]["p_fail"] is not None
        meta["probes"]["hop1"]["tau"] = 1.01
        Path(d, "probes_meta.json").write_text(json.dumps(meta))
        state["i"] = 0
        h2 = generate_monitored(model, tok, prompt, probes=ProbeBundle(d, mode="dedicated"), halt=True, max_new_tokens=120)
        assert h2["stopped_at"] is None and h2["text"] == COT
    print("[PASS] halting: probe stops at </hop1> (prefix of the full trajectory); τ > 1 never halts")

print("=" * 55)
print("  All pipeline tests passed")
print("=" * 55)
