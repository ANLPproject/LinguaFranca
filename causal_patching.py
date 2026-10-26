"""
causal_patching.py  (Phase 4, v3)
─────────────────────────────────
Activation patching on (clean, counterfactual) minimal pairs, plus a direct
link to the hop-1 probe.

History
───────
* v1 notebook (legacy/create_kaggle_phase4_patching.ipynb): sampled instead of
  greedy decoding, patched into prompts whose hop-1 text already said "not
  mentioned", never recorded a result.
* v2.0 of this script: the "entity span" was first-to-last differing token.
  Because the Llama chat template contains "Today Date: <date>" and the clean
  and counterfactual prompts were generated on different days, the prompts
  also differ at ONE token near the start, so that "span" covered ~1,000
  tokens (nearly the whole prompt).  Its Experiment-A numbers are invalid.
* v3 (this file): the date token of the CF prompt is set to the clean prompt's
  date token, so the two prompts differ ONLY in the swapped entity (a true
  minimal pair); the entity span is those 2-11 tokens; a "tail" condition, a
  length-matched donor control and a probe link (Experiment C) were added.

Design (classic causal tracing, Meng et al. 2022)
─────────────────────────────────────────────────
Pairs: CF question = clean question with the entry entity swapped for another
entity of the same token length.  Only pairs whose prompts tokenize to the same
length are used (51 of 78 for Llama-3.2), so every position lines up.

Experiment A — where does the information needed for hop 1 live?
  Run the CF prompt, overwrite the residual stream (output of block L) at
  positions S with the clean run's states, greedily generate hop 1.
  Success = generated hop 1 states the CLEAN pair's gold hop-1 bridge entity.
    S = entity span   : the swapped entity tokens only (2-11 tokens)
    S = tail          : every token after the entity (rest of the question +
                        chat-template tokens, incl. the last prompt token; 5-12)
    S = last token    : only the last prompt token
    S = donor control : entity span filled with a DIFFERENT pair's clean entity
                        states (closest span length) — should NOT restore
    sanity            : ALL prompt positions at block 0 — should restore ~100 %
  Because attention is causal, only the entity span and the tail can carry
  entity information; together they are everything that differs.

Experiment B — hop-boundary transplant (the v1 idea, done properly)
  CF prompt + CF's own hop-1 text; overwrite the state at the final "</hop1>"
  token (block L) with the clean run's state there; generate the rest.
  Success = final answer contains the clean gold answer.

Experiment C — does patching also flip the PROBE's verdict?
  For the clean baseline, the CF baseline and every entity-span patched run,
  the generated hop-1 text is fed back (with the same patch active), its block-10
  states are mean-pooled exactly as in Phase 1, and the hop-1 probe
  (data/canonical/hop1_probe_layer10.npz, = outputs/probing/probes/
  hop1_probe_layer10.joblib) gives P(failure).  A self-check first recomputes
  Phase-1 states for a few stored examples and compares them with the stored
  reference vectors (cosine should be ~1).  Experiment C can never affect A/B:
  any error there is recorded and skipped.

Robustness
──────────
  * Entity-span guard (aborts if spans look wrong) + hook self-test (aborts if
    patching has no effect) run before any long work.
  * Each pair is appended to pairs.jsonl when done; re-running resumes.
    Records from older versions are ignored (RESULT_VERSION).
  * An error on one pair is logged and skipped.  summary.* rewritten every 5 pairs.

Usage (GPU; ~2.3 h on a Kaggle T4 with the defaults)
─────
  python causal_patching.py --data data/canonical/augmented_v2.jsonl --out outputs/patching
  python causal_patching.py --out outputs/patching --summarize-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lf_common import DEFAULT_DATA, load_examples, normalize_text
from lf_models import (cleanup_gpu, decoder_layers, input_device, load_causal_lm, replace_positions_hook,
                       resolve_model)

RESULT_VERSION = 4          # v4 = date-harmonized minimal pairs (+ tail, donor, probe link)
HOP1_RE = re.compile(r"<hop1>.*?</hop1>", re.DOTALL | re.IGNORECASE)
HOP1_TEXT_RE = re.compile(r"<hop1>(.*?)(?:</hop1>|$)", re.DOTALL | re.IGNORECASE)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
HEDGE = ["no information", "no mention", "not mentioned", "not found", "cannot find", "does not mention",
         "cannot be determined", "not provided"]
PROBE_FILE = "data/canonical/hop1_probe_layer10.npz"
REF_FILE = "data/canonical/phase4_reference_hop1_L10.npz"
CONDITIONS = ("entity_span", "tail", "last_token", "donor_control")


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


def clusters(positions, gap=30):
    """Split sorted positions into runs whose consecutive gaps are <= gap."""
    runs = [[positions[0]]]
    for p in positions[1:]:
        if p - runs[-1][-1] <= gap:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


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
        # Same tokenization as Phase 1 (tokenizer default special tokens).
        return self.tok(text, return_tensors="pt").input_ids[0]

    @torch.no_grad()
    def states(self, ids, positions, layers):
        """{layer: [len(positions), d]} block OUTPUTS at positions (CPU fp32), via hooks on the
        blocks themselves (HF output_hidden_states[-1] is after the final norm)."""
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
        """Patch EVERY position at the LAST block with clean states: the CF run's final
        logits must then equal the clean run's.  Returns (max|diff| patched, unpatched)."""
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
        """Greedy continuation of `ids` (optionally with a patch at block `layer`). Returns new text."""
        handle = None
        if layer is not None:
            handle = self.layers[layer].register_forward_hook(replace_positions_hook(positions, vectors))
        try:
            kw = dict(input_ids=ids[None].to(self.dev),
                      attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=self.dev),
                      max_new_tokens=max_new or self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            try:                                              # stop at the closing tag (fast path)
                out = self.model.generate(**kw, stop_strings=[stop] if stop else None, tokenizer=self.tok)
            except Exception:                                 # plain greedy if stop_strings unsupported
                out = self.model.generate(**kw)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0][len(ids):], skip_special_tokens=True)


def hop1_token_span(tok, gen_text):
    """Tokenize gen_text (no special tokens) and return (encoding, first, last) token indices covering
    the hop-1 TEXT between <hop1> and </hop1> — the same char->token rule as Phase 1
    (generate_cot.extract_hidden_states).  None if there is no hop-1 text."""
    m = HOP1_TEXT_RE.search(gen_text)
    if not m or not m.group(1).strip():
        return None
    c0, c1 = m.start(1), m.end(1)
    enc = tok(gen_text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    ts = next((i for i, (a, b) in enumerate(offs) if a <= c0 < b), None)
    te = next((i for i, (a, b) in enumerate(offs) if a < c1 <= b), None)
    if ts is None or te is None or te < ts:
        return None
    return enc, ts, te


class ProbeLink:
    """Experiment C: hop-1 probe applied to block-10 mean-pooled hop-1 text states (as in Phase 1)."""

    def __init__(self, R: Runner, probe_file: str = PROBE_FILE, ref_file: str = REF_FILE):
        self.R = R
        self.ok = False
        self.msg = ""
        try:
            d = np.load(probe_file)
            self.mean, self.scale, self.coef, self.b = d["mean"], d["scale"], d["coef"], float(d["intercept"])
            self.layer = int(d["layer"])
            ref = np.load(ref_file)
            self.ref = {str(i): v for i, v in zip(ref["ids"], ref["vecs"])}
            hidden = R.model.config.hidden_size
            if hidden != len(self.coef):
                self.msg = f"probe has {len(self.coef)} features but the model has hidden size {hidden}; disabled"
                return
            self.ok = True
        except Exception as e:
            self.msg = f"probe link disabled ({e.__class__.__name__}: {e})"

    def p_fail(self, vec):
        z = (np.nan_to_num(vec.astype(np.float64)) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.coef + self.b))))

    @torch.no_grad()
    def pooled(self, prefix_ids, gen_text, layer=None, positions=None, vectors=None):
        """Mean of block-`self.layer` outputs over the hop-1 TEXT tokens (tags excluded), computed
        on prefix_ids + tokens(gen_text), with the same patch active.  None if no hop-1 text."""
        span = hop1_token_span(self.R.tok, gen_text)
        if span is None:
            return None
        enc, ts, te = span
        P = len(prefix_ids)
        ids = torch.cat([prefix_ids, torch.tensor(enc["input_ids"], dtype=prefix_ids.dtype)])
        pos = torch.arange(P + ts, P + te + 1)
        caps, handles = {}, []
        if layer is not None:
            handles.append(self.R.layers[layer].register_forward_hook(replace_positions_hook(positions, vectors)))

        def grab(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            caps["v"] = hs[0, pos.to(hs.device)].float().mean(0).cpu().numpy()

        handles.append(self.R.layers[self.layer].register_forward_hook(grab))
        try:
            self.R.model(input_ids=ids[None].to(self.R.dev))
        finally:
            for h in handles:
                h.remove()
        return caps.get("v")

    def self_check(self, prepared, n=3):
        """Recompute Phase-1 pooled states for stored texts; compare with reference vectors."""
        cos = []
        for p in prepared[:n]:
            for ex, ids in ((p["clean"], p["ids_c"]), (p["cf"], p["ids_f_orig"])):
                ref = self.ref.get(ex["id"])
                gm = HOP1_RE.search(ex["generated_cot"])
                if ref is None or gm is None:
                    continue
                v = self.pooled(ids, ex["generated_cot"][: gm.end()])
                if v is not None:
                    cos.append(float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9)))
        return cos


# ─────────────────────────────────────────────────────────────────────────────
# Summaries (computed from pairs.jsonl so a partial run still yields results)
# ─────────────────────────────────────────────────────────────────────────────

def summarize(out: Path, meta: dict) -> dict:
    recs = []
    p = out / "pairs.jsonl"
    if p.exists():
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if not r.get("error") and r.get("v") == RESULT_VERSION]
    layers = sorted({int(L) for r in recs for L in r.get("A_by_layer", {})})
    valid = [r for r in recs if r["baseline_clean_states_gold"] and not r["baseline_cf_states_gold"]]
    n = max(len(recs), 1)
    S = {**meta, "n_pairs_done": len(recs), "n_valid": len(valid),
         "baseline_clean_states_gold_rate": sum(r["baseline_clean_states_gold"] for r in recs) / n,
         "baseline_cf_states_gold_rate": sum(r["baseline_cf_states_gold"] for r in recs) / n,
         "baseline_cf_hedge_rate": sum(r["baseline_cf_hedges"] for r in recs) / n,
         "sanity_all_positions_L0_rate": sum(r["sanity_all_positions_L0"] for r in valid) / max(len(valid), 1),
         "sanity_text_equals_clean_rate": sum(r["sanity_text_equals_clean"] for r in recs) / n,
         "experiment_A": {}, "experiment_B": {}, "experiment_C": {}}
    for L in layers:
        row = {}
        for cond in CONDITIONS:
            vals = [r["A_by_layer"][str(L)][cond] for r in valid if cond in r["A_by_layer"].get(str(L), {})]
            k, m = sum(vals), len(vals)
            row[cond] = {"rate": k / m if m else float("nan"), "k": k, "n": m, "ci95": wilson(k, m)}
        S["experiment_A"][L] = row
    for L in sorted({int(L) for r in recs for L in r.get("B_by_layer", {})}):
        have = [r for r in recs if str(L) in r.get("B_by_layer", {})]
        base = sum(r["B_baseline_ok"] for r in have)
        patched = sum(r["B_by_layer"][str(L)] for r in have)
        restored = sum(r["B_by_layer"][str(L)] and not r["B_baseline_ok"] for r in have)
        S["experiment_B"][L] = {"n": len(have), "baseline": base, "patched": patched, "restored": restored,
                                "restored_rate": restored / len(have) if have else float("nan"),
                                "restored_ci95": wilson(restored, len(have))}
    # Experiment C (probe link)
    C = {}
    def flag_stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"n": 0}
        k = sum(v >= 0.5 for v in vals)
        return {"n": len(vals), "mean_p_fail": float(np.mean(vals)), "flag_rate": k / len(vals), "ci95": wilson(k, len(vals))}
    C["clean_baseline"] = flag_stats([r.get("C_clean") for r in valid])
    C["cf_baseline"] = flag_stats([r.get("C_cf") for r in valid])
    C["by_layer"] = {}
    for L in layers:
        restored, not_restored = [], []
        for r in valid:
            pf = r.get("C_entity_span", {}).get(str(L))
            if pf is None:
                continue
            (restored if r["A_by_layer"][str(L)]["entity_span"] else not_restored).append(pf)
        C["by_layer"][L] = {"restored": flag_stats(restored), "not_restored": flag_stats(not_restored)}
    C["self_check_cosines"] = meta.get("probe_self_check_cosines")
    S["experiment_C"] = C
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    write_markdown(S, out / "summary.md")
    return S


def _fmt_rate(x):
    return f"{x['rate']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] (n={x['n']})" if x["n"] else "n/a"


def _fmt_flag(x):
    return f"{x['flag_rate']:.2f} (mean p={x['mean_p_fail']:.2f}, n={x['n']})" if x.get("n") else "n/a"


def write_markdown(S, path):
    nan = float("nan")
    hs = S.get("hook_self_test", {})
    L = [f"# Phase 4 — activation patching ({S.get('model')}, via {S.get('used_repo')})\n",
         f"Pairs: {S.get('n_pairs_total')} total, {S.get('n_aligned')} token-aligned (skipped {S.get('skipped')}); "
         f"{S['n_pairs_done']} completed. Informative pairs (clean run states the gold hop-1 entity, CF run does not): "
         f"**{S['n_valid']}**. Prompts are date-harmonized: clean and CF differ only in the entity "
         f"(entity span median {S.get('entity_span_median')} tokens, tail median {S.get('tail_median')}).\n",
         "## Checks",
         f"- hook self-test (all positions, last block): max |logit diff| {hs.get('max_abs_diff_patched', nan):.2e} "
         f"(unpatched {hs.get('max_abs_diff_unpatched', nan):.2e}) — must be ~0 vs large",
         f"- sanity (patch ALL prompt positions at block 0): {S['sanity_all_positions_L0_rate']:.3f} restored, generated "
         f"text identical to the clean run in {S['sanity_text_equals_clean_rate']:.3f} of pairs — both should be ~1",
         f"- baselines: clean run states the gold hop-1 entity {S['baseline_clean_states_gold_rate']:.3f}; "
         f"CF run {S['baseline_cf_states_gold_rate']:.3f}; CF run hedges ('not mentioned') {S['baseline_cf_hedge_rate']:.3f}",
         f"- probe-link self-check (recomputed vs stored Phase-1 states, cosine): {S['experiment_C'].get('self_check_cosines')}\n",
         "## Experiment A — hop-1 bridge entity restored, by patched block\n",
         "| block | entity span | tail (tokens after entity) | last prompt token | donor-entity control |",
         "|---|---|---|---|---|"]
    for layer, row in S["experiment_A"].items():
        L.append(f"| {layer} | " + " | ".join(_fmt_rate(row[c]) for c in CONDITIONS) + " |")
    if S["experiment_B"]:
        L += ["\n## Experiment B — single-state transplant at `</hop1>`, final answer restored\n",
              "| block | baseline correct | patched correct | restored (patched ok & baseline wrong) |", "|---|---|---|---|"]
        for layer, v in S["experiment_B"].items():
            L.append(f"| {layer} | {v['baseline']}/{v['n']} | {v['patched']}/{v['n']} | "
                     f"{v['restored_rate']:.2f} [{v['restored_ci95'][0]:.2f}, {v['restored_ci95'][1]:.2f}] |")
    C = S["experiment_C"]
    L += ["\n## Experiment C — hop-1 probe (block 10) on the generated hop 1: fraction flagged as failure\n",
          f"- clean baseline: {_fmt_flag(C['clean_baseline'])}",
          f"- CF baseline: {_fmt_flag(C['cf_baseline'])}\n",
          "| block patched (entity span) | runs where the entity WAS restored | runs where it was NOT restored |",
          "|---|---|---|"]
    for layer, v in C["by_layer"].items():
        L.append(f"| {layer} | {_fmt_flag(v['restored'])} | {_fmt_flag(v['not_restored'])} |")
    L.append("\nReading C: if the probe tracks the causal state, it should flag few restored runs and most "
             "non-restored runs.")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/patching")
    ap.add_argument("--layers", default="0,2,4,6,8,9,10,11,12,13,14,16,20,24",
                    help="blocks for Experiments A/C ('all' for every block)")
    ap.add_argument("--boundary-layers", default="8,11,14,20", help="blocks for Experiment B")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48, help="hop-1 generation budget")
    ap.add_argument("--answer-tokens", type=int, default=160, help="Experiment B generation budget")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--probe-file", default=PROBE_FILE)
    ap.add_argument("--ref-file", default=REF_FILE)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if args.summarize_only:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summarize(out, meta)
        print((out / "summary.md").read_text(encoding="utf-8"))
        return

    examples = load_examples(args.data)
    pairs = build_pairs(examples)
    print(f"minimal pairs (clean hop1 OK, CF hop1 failed): {len(pairs)}")

    repo, msg = resolve_model(args.model, args.hf_token)
    print(msg)
    if repo is None:
        raise SystemExit(f"cannot access {args.model}; nothing to do")
    model, tok = load_causal_lm(repo, args.dtype, token=args.hf_token)
    n_layers = len(decoder_layers(model))
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",") if int(x) < n_layers]
    b_layers = [int(x) for x in args.boundary_layers.split(",") if int(x) < n_layers]
    R = Runner(model, tok, args.max_new_tokens)

    # ── prepare: token alignment, date harmonization, clean states ─────────────
    prepared, skipped = [], defaultdict(int)
    for clean, cf in pairs:
        ids_c, ids_f_orig = R.ids(clean["prompt"]), R.ids(cf["prompt"])
        if len(ids_c) != len(ids_f_orig):
            skipped["prompt_lengths_differ"] += 1; continue
        diff = (ids_c != ids_f_orig).nonzero().flatten().tolist()
        if not diff:
            skipped["prompts_identical"] += 1; continue
        runs = clusters(diff)
        ent, nuisance = runs[-1], [q for run in runs[:-1] for q in run]
        ids_f = ids_f_orig.clone()
        if nuisance:                               # e.g. the "Today Date" token: make CF identical to clean there
            ids_f[nuisance] = ids_c[nuisance]
        span = list(range(min(ent), max(ent) + 1))
        P = len(ids_c)
        tail = list(range(max(ent) + 1, P)) or [P - 1]
        st = R.states(ids_c, span + tail, layers)
        k = len(span)
        prepared.append(dict(clean=clean, cf=cf, ids_c=ids_c, ids_f=ids_f, ids_f_orig=ids_f_orig,
                             span=span, tail=tail, P=P, n_nuisance=len(nuisance),
                             nuisance_text_cf=tok.decode(ids_f_orig[nuisance]) if nuisance else "",
                             nuisance_text_clean=tok.decode(ids_c[nuisance]) if nuisance else "",
                             span_states={L: v[:k] for L, v in st.items()},
                             tail_states={L: v[k:] for L, v in st.items()},
                             last_states={L: v[-1:] for L, v in st.items()},
                             span_text_cf=tok.decode(ids_f[span]), span_text_clean=tok.decode(ids_c[span]),
                             gold1=clean["reasoning_graph"][0]["gold_entity"]))
        if args.max_pairs and len(prepared) >= args.max_pairs:
            break
    print(f"aligned pairs used: {len(prepared)}  skipped: {dict(skipped)}")
    if not prepared:
        raise SystemExit("no token-aligned pairs — wrong tokenizer/model for this dataset?")

    # ── guards (abort early, never after hours) ─────────────────────────────────
    span_lens = [len(p["span"]) for p in prepared]; tail_lens = [len(p["tail"]) for p in prepared]
    med_span, med_tail = statistics.median(span_lens), statistics.median(tail_lens)
    print(f"entity span length: median {med_span} (min {min(span_lens)}, max {max(span_lens)}); "
          f"tail length: median {med_tail} (min {min(tail_lens)}, max {max(tail_lens)})")
    print(f"harmonized non-entity differences: {sum(p['n_nuisance'] for p in prepared)} token(s) in "
          f"{sum(1 for p in prepared if p['n_nuisance'])} pair(s), e.g. "
          f"{next((repr(p['nuisance_text_cf']) + ' -> ' + repr(p['nuisance_text_clean']) for p in prepared if p['n_nuisance']), 'none')}")
    for p in prepared[:3]:
        print(f"  swapped entity tokens: {p['span_text_clean']!r} -> {p['span_text_cf']!r}")
    if med_span > 40 or max(span_lens) > 80:
        raise SystemExit("entity-span detection looks broken (span too long); refusing to run")
    remaining = sum(int((p["ids_c"] != p["ids_f"]).sum()) - sum(1 for q in p["span"] if p["ids_c"][q] != p["ids_f"][q])
                    for p in prepared)
    if remaining:
        raise SystemExit(f"{remaining} differing token(s) outside the entity span remain after harmonization; refusing")
    d_patch, d_none = R.hook_self_test(prepared[0]["ids_c"], prepared[0]["ids_f"])
    ok = d_patch < 1e-2 * max(d_none, 1e-6) or d_patch < 1e-3
    print(f"HOOK SELF-TEST: |patched - clean| = {d_patch:.2e}  vs  |unpatched - clean| = {d_none:.2e}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("hook self-test failed: patching is not taking effect; results would be meaningless")

    PL = None
    cos = None
    if not args.skip_c:
        PL = ProbeLink(R, args.probe_file, args.ref_file)
        if PL.ok:
            try:
                cos = [round(c, 4) for c in PL.self_check(prepared)]
                verdict = "OK" if cos and min(cos) > 0.98 else "WARNING (probe link may be unreliable)"
                print(f"PROBE-LINK SELF-CHECK: cosine(recomputed, stored Phase-1 state) = {cos} -> {verdict}")
            except Exception as e:
                print(f"PROBE-LINK SELF-CHECK failed ({e.__class__.__name__}: {e}); Experiment C disabled")
                PL = None
        else:
            print(f"[info] {PL.msg}")
            PL = None

    meta = {"model": args.model, "used_repo": repo, "n_pairs_total": len(pairs), "n_aligned": len(prepared),
            "skipped": dict(skipped), "layers_A": layers, "layers_B": b_layers,
            "entity_span_median": med_span, "tail_median": med_tail,
            "hook_self_test": {"max_abs_diff_patched": d_patch, "max_abs_diff_unpatched": d_none},
            "probe_self_check_cosines": cos}
    meta_path.write_text(json.dumps(meta, indent=1))

    # donor = the other pair with the closest entity-span length (deterministic)
    donors = {}
    for i, p in enumerate(prepared):
        others = [j for j in range(len(prepared)) if j != i]
        donors[i] = min(others, key=lambda j: (abs(len(prepared[j]["span"]) - len(p["span"])), j)) if others else None

    pairs_path = out / "pairs.jsonl"
    done = set()
    if pairs_path.exists():
        done = {r["cf_id"] for r in (json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip())
                if r.get("v") == RESULT_VERSION and not r.get("error")}
        print(f"resuming: {len(done)} pairs already done")
    fh = open(pairs_path, "a", encoding="utf-8")
    t0 = time.time()
    n_new = 0
    for i, p in enumerate(prepared):
        cf_id = p["cf"]["id"]
        if cf_id in done:
            continue
        try:
            donor = prepared[donors[i]] if donors[i] is not None else None
            rec = run_pair(R, p, donor, layers, b_layers, args, PL)
        except Exception as e:
            traceback.print_exc()
            cleanup_gpu()
            rec = {"v": RESULT_VERSION, "cf_id": cf_id, "error": f"{e.__class__.__name__}: {str(e)[:200]}"}
            print(f"[pair {i + 1}/{len(prepared)}] ERROR, skipped")
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        n_new += 1
        if not rec.get("error"):
            print(f"[{i + 1}/{len(prepared)} {time.time() - t0:.0f}s] clean_ok={rec['baseline_clean_states_gold']} "
                  f"cf_ok={rec['baseline_cf_states_gold']} sanity={rec['sanity_all_positions_L0']} | restored at blocks: "
                  f"entity={[int(L) for L, v in rec['A_by_layer'].items() if v['entity_span']]} "
                  f"tail={[int(L) for L, v in rec['A_by_layer'].items() if v['tail']]}", flush=True)
        if n_new % 5 == 0:
            summarize(out, meta)
    fh.close()
    S = summarize(out, meta)
    print((out / "summary.md").read_text(encoding="utf-8"))
    n_err = sum(1 for l in open(pairs_path, encoding="utf-8") if l.strip() and json.loads(l).get("error"))
    print(f"DONE: {S['n_pairs_done']} pairs ok, {n_err} errors.")


def run_pair(R, p, donor, layers, b_layers, args, PL):
    rec = {"v": RESULT_VERSION, "clean_id": p["clean"]["id"], "cf_id": p["cf"]["id"], "gold_hop1": p["gold1"],
           "clean_q": p["clean"]["question"], "cf_q": p["cf"]["question"],
           "span_len": len(p["span"]), "tail_len": len(p["tail"]), "n_nuisance_harmonized": p["n_nuisance"],
           "span_text_clean": p["span_text_clean"], "span_text_cf": p["span_text_cf"],
           "donor_cf_id": donor["cf"]["id"] if donor else None}

    def probe(prefix, gen, layer=None, positions=None, vectors=None):
        if PL is None:
            return None
        try:
            v = PL.pooled(prefix, gen, layer, positions, vectors)
            return None if v is None else round(PL.p_fail(v), 5)
        except Exception as e:                    # Experiment C must never break A/B
            rec.setdefault("C_errors", []).append(f"{e.__class__.__name__}: {str(e)[:80]}")
            return None

    gen_clean = R.generate(p["ids_c"])
    gen_cf = R.generate(p["ids_f"])
    g_clean, g_cf = hop1_text(gen_clean), hop1_text(gen_cf)
    rec["baseline_clean_states_gold"] = contains(g_clean, p["gold1"])
    rec["baseline_cf_states_gold"] = contains(g_cf, p["gold1"])
    rec["baseline_cf_hedges"] = any(h in g_cf.lower() for h in HEDGE)
    rec["cf_baseline_text"] = g_cf.strip()[:200]
    rec["C_clean"] = probe(p["ids_c"], gen_clean)
    rec["C_cf"] = probe(p["ids_f"], gen_cf)

    st0 = R.states(p["ids_c"], list(range(p["P"])), [0])[0]       # sanity: all positions at block 0
    g = hop1_text(R.generate(p["ids_f"], 0, list(range(p["P"])), st0))
    rec["sanity_all_positions_L0"] = contains(g, p["gold1"])
    rec["sanity_text_equals_clean"] = g.strip() == g_clean.strip()
    del st0

    rec["A_by_layer"], rec["C_entity_span"] = {}, {}
    for L in layers:
        r = {}
        gen = R.generate(p["ids_f"], L, p["span"], p["span_states"][L])
        r["entity_span"] = contains(hop1_text(gen), p["gold1"])
        rec["C_entity_span"][str(L)] = probe(p["ids_f"], gen, L, p["span"], p["span_states"][L])
        r["tail"] = contains(hop1_text(R.generate(p["ids_f"], L, p["tail"], p["tail_states"][L])), p["gold1"])
        r["last_token"] = contains(hop1_text(R.generate(p["ids_f"], L, [p["P"] - 1], p["last_states"][L])), p["gold1"])
        if donor is not None:                       # different pair's clean entity states, length-matched
            d = donor["span_states"][L]; k = len(p["span"])
            d = d[:k] if len(d) >= k else torch.cat([d, d[-1:].repeat(k - len(d), 1)])
            r["donor_control"] = contains(hop1_text(R.generate(p["ids_f"], L, p["span"], d)), p["gold1"])
        rec["A_by_layer"][str(L)] = r

    if not args.skip_b and b_layers:
        m_c = HOP1_RE.search(p["clean"]["generated_cot"]); m_f = HOP1_RE.search(p["cf"]["generated_cot"])
        if m_c and m_f:
            add = lambda t: torch.tensor(R.tok(t, add_special_tokens=False).input_ids, dtype=p["ids_c"].dtype)
            seq_c = torch.cat([p["ids_c"], add(p["clean"]["generated_cot"][: m_c.end()])])
            seq_f = torch.cat([p["ids_f"], add(p["cf"]["generated_cot"][: m_f.end()])])   # date-harmonized prompt
            gold_ans = p["clean"]["gold_answer"]
            base = R.generate(seq_f, max_new=args.answer_tokens, stop="</answer>")
            m = ANS_RE.search(base)
            rec["B_baseline_ok"] = contains(m.group(1) if m else base, gold_ans)
            st = R.states(seq_c, [len(seq_c) - 1], b_layers)
            rec["B_by_layer"] = {}
            for L in b_layers:
                g = R.generate(seq_f, L, [len(seq_f) - 1], st[L], max_new=args.answer_tokens, stop="</answer>")
                m = ANS_RE.search(g)
                rec["B_by_layer"][str(L)] = contains(m.group(1) if m else g, gold_ans)
    return rec


if __name__ == "__main__":
    main()
