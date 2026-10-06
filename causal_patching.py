"""
causal_patching.py  (Phase 4, v2)
─────────────────────────────────
Activation patching on (clean, counterfactual) minimal pairs.

Why a rewrite (see FIXES_AND_RERUN.md)
──────────────────────────────────────
The v1 notebook (now legacy/create_kaggle_phase4_patching.ipynb):
  * called model.generate() without do_sample=False.  Llama-3.2-Instruct's
    default generation config samples (temperature 0.6), so "baseline" and
    "patched" answers differed by chance;
  * patched one vector at </hop1> after the CF's own hop-1 text, which in ~90 %
    of pairs already says "X is not mentioned in the context", and then asked
    whether the model produced the answer to a DIFFERENT question;
  * never recorded its result.

Design (classic causal tracing, Meng et al. 2022)
─────────────────────────────────────────────────
Pairs: CF question = clean question with the entry entity swapped for another
entity of the same token length.  We keep pairs whose two prompts tokenize to
the same length, so every position lines up; the differing positions are the
entity span.

Experiment A — where does the hop-1 information live?
  For each layer L and position set S, run the CF prompt but overwrite the
  residual stream (output of block L) at S with the clean run's states, then
  greedily generate hop 1.  Success = the generated hop 1 states the CLEAN
  pair's gold hop-1 bridge entity (normalized substring).
    S = entity span        (the swapped tokens)
    S = last prompt token  (the "prompt-end" patch from v1's earlier version)
    S = donor entity span  (control: clean states from a DIFFERENT pair; should
                            not restore THIS pair's entity)
    S = all prompt tokens at layer 0 (sanity upper bound; should ≈ clean run)
  Expected shape: entity-span patching restores in early/middle layers and
  fades once the information has moved downstream; last-token patching rises
  in later layers.  That curve is the causal localization result.

Experiment B — hop-boundary transplant (the v1 idea, done deterministically)
  Feed CF prompt + the CF's own hop-1 text; overwrite the state at the final
  "</hop1>" token (layer L) with the clean run's state there; generate the rest.
  Success = final answer contains the clean gold answer.  Reported with the
  unpatched baseline so the (expected small) effect size is honest.

Usage (GPU, ~1-1.5 h on a Kaggle T4 for the default settings)
─────
  python causal_patching.py --data data/canonical/augmented_v2.jsonl --out outputs/patching
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from lf_common import DEFAULT_DATA, load_examples, normalize_text
from lf_models import decoder_layers, input_device, load_causal_lm, replace_positions_hook

HOP1_RE = re.compile(r"<hop1>.*?</hop1>", re.DOTALL | re.IGNORECASE)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
HEDGE = ["no information", "no mention", "not mentioned", "not found", "cannot find", "does not mention",
         "cannot be determined", "not provided"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def contains(hay, needle):
    n = normalize_text(needle)
    return bool(n) and n in normalize_text(hay)


def hop1_text(gen: str) -> str:
    i = gen.lower().find("</hop1>")
    return gen[: i] if i >= 0 else gen


def build_pairs(examples):
    by_id = {e["id"]: e for e in examples}
    pairs = []
    for cf in examples:
        if not cf.get("is_counterfactual"):
            continue
        clean = by_id.get(cf.get("clean_pair_id"))
        if clean is None:
            continue
        h_c = next((h for h in clean["hops"] if h["hop_idx"] == 1), None)
        h_f = next((h for h in cf["hops"] if h["hop_idx"] == 1), None)
        if h_c and h_f and h_c["label"] == 0 and h_f["label"] == 1:
            pairs.append((clean, cf))
    return pairs


class Runner:
    def __init__(self, model, tok, max_new_tokens):
        self.model, self.tok, self.max_new = model, tok, max_new_tokens
        self.dev = input_device(model)
        self.layers = decoder_layers(model)

    def ids(self, text):
        # Same tokenization as Phase 1 generation (tokenizer default special tokens).
        return self.tok(text, return_tensors="pt").input_ids[0]

    @torch.no_grad()
    def states(self, ids, positions, layers):
        """{layer: tensor[len(positions), d]} of block OUTPUTS at positions (CPU fp32).

        Captured with hooks on the blocks themselves, i.e. exactly the tensor the
        replace hook overwrites.  (HF output_hidden_states[-1] is AFTER the final
        norm, so it cannot be used for the last layer.)"""
        caps, handles = {}, []
        pos = torch.as_tensor(positions, dtype=torch.long)

        def grab(L):
            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                caps[L] = hs[0, pos.to(hs.device)].float().cpu()
            return hook

        for L in layers:
            handles.append(self.layers[L].register_forward_hook(grab(L)))
        try:
            self.model(input_ids=ids[None].to(self.dev))
        finally:
            for h in handles:
                h.remove()
        return caps

    @torch.no_grad()
    def hook_self_test(self, ids_c, ids_f):
        """Patch EVERY position at the LAST block with clean states: the CF run's
        final logits must then equal the clean run's.  Returns max |diff|."""
        L = len(self.layers) - 1
        P = len(ids_c)
        st = self.states(ids_c, list(range(P)), [L])[L]
        clean = self.model(input_ids=ids_c[None].to(self.dev)).logits[0, -1].float()
        h = self.layers[L].register_forward_hook(replace_positions_hook(list(range(P)), st))
        try:
            patched = self.model(input_ids=ids_f[None].to(self.dev)).logits[0, -1].float()
        finally:
            h.remove()
        unpatched = self.model(input_ids=ids_f[None].to(self.dev)).logits[0, -1].float()
        return (patched - clean).abs().max().item(), (unpatched - clean).abs().max().item()

    @torch.no_grad()
    def generate(self, ids, layer=None, positions=None, vectors=None, max_new=None, stop="</hop1>"):
        handle = None
        if layer is not None:
            handle = self.layers[layer].register_forward_hook(replace_positions_hook(positions, vectors))
        try:
            kw = dict(input_ids=ids[None].to(self.dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=self.dev),
                      max_new_tokens=max_new or self.max_new, do_sample=False, temperature=None, top_p=None,
                      pad_token_id=self.tok.pad_token_id)
            try:
                out = self.model.generate(**kw, stop_strings=[stop] if stop else None, tokenizer=self.tok)
            except (TypeError, ValueError):
                out = self.model.generate(**kw)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0][len(ids):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/patching")
    ap.add_argument("--layers", default="0,2,4,6,8,9,10,11,12,13,14,16,18,20,22,24,27",
                    help="layers for Experiment A ('all' for every layer)")
    ap.add_argument("--boundary-layers", default="8,11,14,20", help="layers for Experiment B")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48, help="hop-1 generation budget")
    ap.add_argument("--answer-tokens", type=int, default=160, help="Experiment B generation budget")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.data)
    pairs = build_pairs(examples)
    print(f"minimal pairs (clean hop1 OK, CF hop1 failed): {len(pairs)}")

    model, tok = load_causal_lm(args.model, args.dtype, token=args.hf_token)
    n_layers = len(decoder_layers(model))
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",") if int(x) < n_layers]
    b_layers = [int(x) for x in args.boundary_layers.split(",") if int(x) < n_layers]
    R = Runner(model, tok, args.max_new_tokens)

    # ── pass 1: alignment + clean states ───────────────────────────────────────
    prepared, skipped = [], defaultdict(int)
    for clean, cf in pairs:
        ids_c, ids_f = R.ids(clean["prompt"]), R.ids(cf["prompt"])
        if len(ids_c) != len(ids_f):
            skipped["prompt_lengths_differ"] += 1; continue
        diff = (ids_c != ids_f).nonzero().flatten().tolist()
        if not diff:
            skipped["prompts_identical"] += 1; continue
        span = list(range(min(diff), max(diff) + 1))
        P = len(ids_c)
        st = R.states(ids_c, span + [P - 1], layers)
        prepared.append(dict(clean=clean, cf=cf, ids_c=ids_c, ids_f=ids_f, span=span, P=P,
                             span_states={L: v[:-1] for L, v in st.items()},
                             last_states={L: v[-1:] for L, v in st.items()},
                             gold1=clean["reasoning_graph"][0]["gold_entity"],
                             entry=cf.get("counterfactual_swap", {}).get("original_entity", "")))
        if args.max_pairs and len(prepared) >= args.max_pairs:
            break
    print(f"aligned pairs used: {len(prepared)}  skipped: {dict(skipped)}")
    if not prepared:
        raise SystemExit("no token-aligned pairs — wrong tokenizer/model for this dataset?")
    d_patch, d_none = R.hook_self_test(prepared[0]["ids_c"], prepared[0]["ids_f"])
    ok = d_patch < 1e-2 * max(d_none, 1e-6) or d_patch < 1e-3
    print(f"HOOK SELF-TEST: |patched - clean| = {d_patch:.2e}  vs  |unpatched - clean| = {d_none:.2e}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("hook self-test failed: patching is not taking effect; results would be meaningless")
    by_len = defaultdict(list)
    for i, p in enumerate(prepared):
        by_len[len(p["span"])].append(i)

    # ── Experiment A ───────────────────────────────────────────────────────────
    t0 = time.time()
    recs = []
    for i, p in enumerate(prepared):
        rec = {"clean_id": p["clean"]["id"], "cf_id": p["cf"]["id"], "gold_hop1": p["gold1"],
               "clean_q": p["clean"]["question"], "cf_q": p["cf"]["question"], "span_len": len(p["span"])}
        g_clean = hop1_text(R.generate(p["ids_c"]))
        g_cf = hop1_text(R.generate(p["ids_f"]))
        rec["baseline_clean_states_gold"] = contains(g_clean, p["gold1"])
        rec["baseline_cf_states_gold"] = contains(g_cf, p["gold1"])
        rec["baseline_cf_hedges"] = any(h in g_cf.lower() for h in HEDGE)
        rec["cf_baseline_text"] = g_cf.strip()[:200]
        # sanity: all prompt positions at layer 0
        st0 = R.states(p["ids_c"], list(range(p["P"])), [0])[0]
        g = hop1_text(R.generate(p["ids_f"], 0, list(range(p["P"])), st0))
        rec["sanity_all_positions_L0"] = contains(g, p["gold1"])
        rec["sanity_text_equals_clean"] = g.strip() == g_clean.strip()
        donors = [j for j in by_len[len(p["span"])] if j != i]
        donor = prepared[donors[i % len(donors)]] if donors else None
        rec["by_layer"] = {}
        for L in layers:
            r = {}
            g = hop1_text(R.generate(p["ids_f"], L, p["span"], p["span_states"][L]))
            r["entity_span"] = contains(g, p["gold1"])
            g = hop1_text(R.generate(p["ids_f"], L, [p["P"] - 1], p["last_states"][L]))
            r["last_token"] = contains(g, p["gold1"])
            if donor is not None:
                g = hop1_text(R.generate(p["ids_f"], L, p["span"], donor["span_states"][L]))
                r["donor_control"] = contains(g, p["gold1"])
                r["donor_control_states_donor_gold"] = contains(g, donor["gold1"])
            rec["by_layer"][L] = r
        recs.append(rec)
        print(f"[A {i + 1}/{len(prepared)} {time.time() - t0:.0f}s] clean_ok={rec['baseline_clean_states_gold']} "
              f"cf_ok={rec['baseline_cf_states_gold']} sanity={rec['sanity_all_positions_L0']} "
              f"entity-span restores at layers {[L for L in layers if rec['by_layer'][L]['entity_span']]}", flush=True)

    # Only pairs where the clean run states the gold entity and the CF run does not are informative.
    valid = [r for r in recs if r["baseline_clean_states_gold"] and not r["baseline_cf_states_gold"]]
    summary = {"model": args.model, "n_pairs_total": len(pairs), "n_aligned": len(prepared),
               "skipped": dict(skipped), "n_valid": len(valid),
               "baseline_clean_states_gold_rate": sum(r["baseline_clean_states_gold"] for r in recs) / max(len(recs), 1),
               "baseline_cf_states_gold_rate": sum(r["baseline_cf_states_gold"] for r in recs) / max(len(recs), 1),
               "baseline_cf_hedge_rate": sum(r["baseline_cf_hedges"] for r in recs) / max(len(recs), 1),
               "sanity_all_positions_L0_rate": sum(r["sanity_all_positions_L0"] for r in valid) / max(len(valid), 1),
               "sanity_text_equals_clean_rate": sum(r["sanity_text_equals_clean"] for r in recs) / max(len(recs), 1),
               "hook_self_test": {"max_abs_diff_patched": d_patch, "max_abs_diff_unpatched": d_none},
               "experiment_A": {}}
    for L in layers:
        row = {}
        for cond in ("entity_span", "last_token", "donor_control"):
            vals = [r["by_layer"][L][cond] for r in valid if cond in r["by_layer"][L]]
            k, n = sum(vals), len(vals)
            row[cond] = {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95": wilson(k, n)}
        summary["experiment_A"][L] = row

    # ── Experiment B ───────────────────────────────────────────────────────────
    if not args.skip_b:
        b = {L: {"patched": 0, "baseline": 0, "restored": 0, "n": 0} for L in b_layers}
        b_recs = []
        for i, p in enumerate(prepared):
            m_c = HOP1_RE.search(p["clean"]["generated_cot"]); m_f = HOP1_RE.search(p["cf"]["generated_cot"])
            if not (m_c and m_f):
                continue
            seq_c = R.ids(p["clean"]["prompt"] + p["clean"]["generated_cot"][: m_c.end()])
            seq_f = R.ids(p["cf"]["prompt"] + p["cf"]["generated_cot"][: m_f.end()])
            gold_ans = p["clean"]["gold_answer"]
            base = R.generate(seq_f, max_new=args.answer_tokens, stop="</answer>")
            m = ANS_RE.search(base)
            base_ok = contains(m.group(1) if m else base, gold_ans)
            st = R.states(seq_c, [len(seq_c) - 1], b_layers)
            rr = {"cf_id": p["cf"]["id"], "baseline_ok": base_ok, "by_layer": {}}
            for L in b_layers:
                g = R.generate(seq_f, L, [len(seq_f) - 1], st[L], max_new=args.answer_tokens, stop="</answer>")
                m = ANS_RE.search(g)
                ok = contains(m.group(1) if m else g, gold_ans)
                b[L]["n"] += 1; b[L]["patched"] += ok; b[L]["baseline"] += base_ok; b[L]["restored"] += ok and not base_ok
                rr["by_layer"][L] = ok
            b_recs.append(rr)
            print(f"[B {i + 1}/{len(prepared)} {time.time() - t0:.0f}s] baseline_ok={base_ok} patched={rr['by_layer']}", flush=True)
        summary["experiment_B"] = {L: {**v, "restored_rate": v["restored"] / v["n"] if v["n"] else float("nan"),
                                       "restored_ci95": wilson(v["restored"], v["n"])} for L, v in b.items()}
        with open(out / "experiment_B_pairs.jsonl", "w", encoding="utf-8") as f:
            for r in b_recs:
                f.write(json.dumps(r) + "\n")

    with open(out / "experiment_A_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_markdown(summary, out / "summary.md")
    print((out / "summary.md").read_text(encoding="utf-8"))


def write_markdown(S, path):
    L = [f"# Phase 4 — activation patching ({S['model']})\n",
         f"Pairs: {S['n_pairs_total']} total, {S['n_aligned']} token-aligned, skipped {S['skipped']}. "
         f"Informative pairs (clean run states the gold hop-1 entity, CF run does not): **{S['n_valid']}**.\n",
         f"- clean baseline states gold hop-1 entity: {S['baseline_clean_states_gold_rate']:.3f}",
         f"- CF baseline states it: {S['baseline_cf_states_gold_rate']:.3f}; CF baseline hedges: {S['baseline_cf_hedge_rate']:.3f}",
         f"- hook self-test (all positions, last block): max |logit diff| {S['hook_self_test']['max_abs_diff_patched']:.2e} "
         f"(unpatched {S['hook_self_test']['max_abs_diff_unpatched']:.2e})",
         f"- sanity (patch ALL prompt positions at layer 0): {S['sanity_all_positions_L0_rate']:.3f} restored (should be ≈1); "
         f"generated text identical to clean run in {S['sanity_text_equals_clean_rate']:.3f} of pairs\n",
         "## Experiment A — restoration of the hop-1 bridge entity, by layer\n",
         "| layer | entity-span patch | last-prompt-token patch | donor control |", "|---|---|---|---|"]
    for layer, row in S["experiment_A"].items():
        cells = []
        for c in ("entity_span", "last_token", "donor_control"):
            x = row[c]
            cells.append(f"{x['rate']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] (n={x['n']})" if x["n"] else "n/a")
        L.append(f"| {layer} | " + " | ".join(cells) + " |")
    if "experiment_B" in S:
        L += ["\n## Experiment B — hop-boundary (</hop1>) transplant, final answer restored\n",
              "| layer | baseline correct | patched correct | restored (patched ok & baseline wrong) |", "|---|---|---|---|"]
        for layer, v in S["experiment_B"].items():
            L.append(f"| {layer} | {v['baseline']}/{v['n']} | {v['patched']}/{v['n']} | "
                     f"{v['restored_rate']:.2f} [{v['restored_ci95'][0]:.2f}, {v['restored_ci95'][1]:.2f}] |")
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
