"""
causal_patching.py  (Phase 4, v5 — TF-IDF setting)
──────────────────────────────────────────────────
Activation patching on (clean, counterfactual) minimal pairs, plus a direct
link to the deployed hop-failure probe.

History
───────
* v1 notebook (legacy/create_kaggle_phase4_patching.ipynb): sampled instead of
  greedy decoding, patched into prompts whose hop-1 text already said "not
  mentioned", never recorded a result.
* v2.0: the "entity span" was first-to-last differing token, which included a
  differing chat-template date token (~1,000-token "span").  Invalid.
* v4 (git history; results in outputs/patching/): full-10-passage setting, 51
  date-harmonized pairs, old hop-1 probe.  All checks passed.
* v5 (this file): TF-IDF setting, to match the deployed probe and pipeline.
    - pairs = natural TF-IDF data (pinned revision) + the TF-IDF counterfactuals
      cf_v1 (generate_counterfactuals_tfidf.py).  The CF prompt is the seed's
      stored prompt with only the entry entity swapped (same date token, same
      context), so no harmonization is needed; the recorded entity token span
      is used and every differing token must lie inside it.
    - default pairs: test split, failure kinds hedge + other_wrong_entity
      (copied_seed_entity CFs already state the clean gold, so nothing to restore).
    - donor control: a THIRD entity (another pair's clean entity, same token
      length, different hop-1 gold, not in this context) spliced into THIS
      prompt and run through the model, so its states carry this context.
      v4 took the donor's states from the donor's own prompt (other context)
      and did not exclude donors with the same hop-1 gold.
    - success = the clean hop-1 gold entity or a Wikidata alias of it appears
      in the generated hop 1 as whole words (v4: raw substring, no aliases).
    - Experiment C uses the deployed TF-IDF probe bundle (shared hop12 probe,
      block 10, Platt-calibrated, trigger τ); its self-check compares against
      the stored Phase-1 / cf_v1 hidden-state files.
    - summary adds the paired sign test entity-span vs donor control, the probe
      verdict / restoration agreement, and a breakdown by CF failure kind.

Design (classic causal tracing, Meng et al. 2022)
─────────────────────────────────────────────────
Experiment A — where does the information needed for hop 1 live?
  Run the CF prompt, overwrite the residual stream (output of block L) at
  positions S with the clean run's states, greedily generate hop 1.
  Success = generated hop 1 states the CLEAN pair's gold hop-1 bridge entity.
    S = entity span   : the swapped entity tokens only
    S = tail          : every token after the entity (rest of the question +
                        chat-template tokens, incl. the last prompt token)
    S = last token    : only the last prompt token
    S = donor control : entity span filled with a third entity's states
                        (computed in this prompt) — should NOT restore
    sanity            : ALL prompt positions at block 0 — should restore ~100 %
  Because attention is causal, only the entity span and the tail can carry
  entity information; together they are everything that differs.

Experiment B — hop-boundary transplant
  CF prompt + CF's own hop-1 text; overwrite the state at the final "</hop1>"
  token (block L) with the clean run's state there; generate the rest.
  Success = final answer contains the clean gold answer.

Experiment C — does patching also flip the PROBE's verdict?
  For the clean baseline, the CF baseline and every entity-span patched run,
  the generated hop-1 text is fed back (with the same patch active), its
  block-10 states are mean-pooled exactly as in Phase 1, and the deployed probe
  gives calibrated P(failure); flagged = P >= τ (the live trigger rule).
  Experiment C can never affect A/B: any error there is recorded and skipped.

Robustness
──────────
  * Entity-span guard + hook self-test run before any long work (abort if wrong).
  * Each pair is appended to pairs.jsonl when done; re-running resumes.
    Records from older versions are ignored (RESULT_VERSION).
  * An error on one pair is logged and skipped.  summary.* rewritten every 5 pairs.
  * --max-minutes stops cleanly between pairs (pairs are in a fixed random
    order, so a partial run is a random sample).

Usage (GPU; ~2 min per pair on a Kaggle T4 with the defaults)
─────
  python causal_patching.py --out outputs/patching_tfidf
  python causal_patching.py --out outputs/patching_tfidf --summarize-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from lf_common import normalize_text
from lf_models import cleanup_gpu, decoder_layers, input_device, load_causal_lm, replace_positions_hook, resolve_model
from src.pipeline.common import CF_V1, SETTINGS, apply_extra_hop_rule, hf_file, load_setting_examples, read_jsonl
from src.pipeline.probes import LinearProbe, apply_platt

RESULT_VERSION = 5          # v5 = TF-IDF pairs, in-context donor control, deployed probe
HOP1_RE = re.compile(r"<hop1>.*?</hop1>", re.DOTALL | re.IGNORECASE)
HOP1_TEXT_RE = re.compile(r"<hop1>(.*?)(?:</hop1>|$)", re.DOTALL | re.IGNORECASE)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
HEDGE = ["no information", "no mention", "not mentioned", "not found", "cannot find", "does not mention",
         "cannot be determined", "not provided"]
PROBE_DIR = "outputs/pipeline/tfidf/probes"
PROBE_TASK = "hop12"        # the shared probe the live system deploys
CONDITIONS = ("entity_span", "tail", "last_token", "donor_control")
KINDS = ("hedge", "other_wrong_entity")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(b, c):
    """Exact two-sided binomial (sign) test on discordant pairs b vs c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def name_forms(gold, aliases):
    """Normalized gold entity + its Wikidata aliases (aliases shorter than 4 characters are dropped)."""
    forms = {normalize_text(gold)}
    forms |= {a for a in (normalize_text(x) for x in aliases.get(gold, [])) if len(a) >= 4 and not a.isdigit()}
    return sorted(f for f in forms if f)


def mentions(hay, forms):
    """Any form occurs in hay as whole words (after normalization)."""
    h = f" {normalize_text(hay)} "
    return any(f" {f} " in h for f in forms)


def overlaps(a, b):
    a, b = normalize_text(a), normalize_text(b)
    return bool(a) and bool(b) and (a in b or b in a)


def hop1_text(gen: str) -> str:
    i = gen.lower().find("</hop1>")
    return gen[: i] if i >= 0 else gen


def gold_of(e, k):
    return next((n.get("gold_entity", "") for n in e.get("reasoning_graph", []) if n.get("hop") == k), "")


def load_pairs(split="test", kinds=KINDS, seed=42):
    """(clean, cf) pairs from the TF-IDF data + cf_v1, in a fixed random order; plus the alias table."""
    natural = {e["id"]: e for e in load_setting_examples("tfidf")}
    cfs = read_jsonl(hf_file(CF_V1["file"], CF_V1["revision"]))
    aliases = json.loads(Path(hf_file(CF_V1["aliases"], CF_V1["revision"])).read_text(encoding="utf-8"))
    pairs, why = [], Counter()
    for cf in cfs:
        apply_extra_hop_rule(cf)
        sw = cf["counterfactual_swap"]
        clean = natural.get(cf.get("clean_pair_id"))
        if clean is None:
            why["seed_missing"] += 1; continue
        if split != "all" and sw["split"] != split:
            why[f"not_{split}"] += 1; continue
        if sw["failure_kind"] not in kinds:
            why[f"kind_{sw['failure_kind']}"] += 1; continue
        h_c = next((h for h in clean["hops"] if h["hop_idx"] == 1), None)
        h_f = next((h for h in cf["hops"] if h["hop_idx"] == 1), None)
        if not (h_c and h_f and h_c["label"] == 0 and h_f["label"] == 1):
            why["labels"] += 1; continue
        if normalize_text(gold_of(clean, 1)) != normalize_text(sw["seed_gold"][0]):
            why["seed_gold_mismatch"] += 1; continue
        pairs.append((clean, cf))
    pairs.sort(key=lambda p: p[1]["id"])
    random.Random(seed).shuffle(pairs)
    return pairs, dict(why), aliases


def stored_hop1_state(e, layer):
    """Stored Phase-1 (natural) or cf_v1 pooled hop-1 state at `layer`, float32 [d]."""
    if e.get("is_counterfactual"):
        path = hf_file(f"{CF_V1['hs_prefix']}/{e['id']}.pt", CF_V1["revision"])
    else:
        s = SETTINGS["tfidf"]
        path = hf_file(f"{s['hs_prefix']}/{e['id']}.pt", s["revision"])
    d = torch.load(path, map_location="cpu", weights_only=False)
    li = list(d.get("layer_indices", range(d["pooled"].shape[0])))
    pos = [sp["hop_idx"] for sp in e["hop_spans"]].index(1)
    return d["pooled"][li.index(layer), pos].float().numpy()


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
    """Experiment C: the deployed probe on block-L mean-pooled hop-1 text states (as in Phase 1 / the pipeline)."""

    def __init__(self, R: Runner, probe_dir: str = PROBE_DIR, task: str = PROBE_TASK):
        self.R = R
        self.ok = False
        self.msg = ""
        try:
            meta = json.loads((Path(probe_dir) / "probes_meta.json").read_text())
            if meta.get("setting") != "tfidf":
                self.msg = f"probe bundle is for setting {meta.get('setting')!r}, not tfidf; disabled"
                return
            p = meta["probes"][task]
            self.lp = LinearProbe(Path(probe_dir) / f"{task}.npz")
            self.layer, self.tau, self.calib = int(p["layer"]), float(p["tau"]), p.get("calibration")
            hidden = R.model.config.hidden_size
            if hidden != len(self.lp.coef):
                self.msg = f"probe has {len(self.lp.coef)} features but the model has hidden size {hidden}; disabled"
                return
            self.ok = True
        except Exception as e:
            self.msg = f"probe link disabled ({e.__class__.__name__}: {e})"

    def p_fail(self, vec):
        """Calibrated P(fail), on the pipeline's scale (features rounded to fp16 like the stored states)."""
        x = np.asarray(vec, dtype=np.float32).astype(np.float16).astype(np.float32)
        return float(apply_platt(self.calib, self.lp.p_fail(x)[0]))

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
        """Recompute Phase-1 pooled states for stored texts; compare with the stored hidden-state files.
        Returns [(id, cosine, stored P(fail), recomputed P(fail))]."""
        out = []
        for p in prepared[:n]:
            for ex, ids in ((p["clean"], p["ids_c"]), (p["cf"], p["ids_f"])):
                gm = HOP1_RE.search(ex["generated_cot"])
                if gm is None:
                    continue
                ref = stored_hop1_state(ex, self.layer)
                v = self.pooled(ids, ex["generated_cot"][: gm.end()])
                if v is not None:
                    cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
                    out.append((ex["id"], round(cos, 4), round(self.p_fail(ref), 4), round(self.p_fail(v), 4)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Summaries (computed from pairs.jsonl so a partial run still yields results)
# ─────────────────────────────────────────────────────────────────────────────

def _rate(vals):
    k, m = sum(vals), len(vals)
    return {"rate": k / m if m else float("nan"), "k": k, "n": m, "ci95": wilson(k, m)}


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
         "n_valid_by_kind": dict(Counter(r["failure_kind"] for r in valid)),
         "baseline_clean_states_gold_rate": sum(r["baseline_clean_states_gold"] for r in recs) / n,
         "baseline_cf_states_gold_rate": sum(r["baseline_cf_states_gold"] for r in recs) / n,
         "baseline_cf_hedge_rate": sum(r["baseline_cf_hedges"] for r in recs) / n,
         "sanity_all_positions_L0_rate": sum(r["sanity_all_positions_L0"] for r in valid) / max(len(valid), 1),
         "sanity_text_equals_clean_rate": sum(r["sanity_text_equals_clean"] for r in recs) / n,
         "experiment_A": {}, "A_entity_vs_donor": {}, "A_entity_span_by_kind": {},
         "experiment_B": {}, "experiment_C": {}}
    for L in layers:
        rows = [r["A_by_layer"][str(L)] for r in valid if str(L) in r["A_by_layer"]]
        S["experiment_A"][L] = {c: _rate([x[c] for x in rows if c in x]) for c in CONDITIONS}
        # paired: entity span (clean entity) vs donor control (third entity), same pairs
        both = [x for x in rows if "donor_control" in x]
        b = sum(x["entity_span"] and not x["donor_control"] for x in both)
        c = sum(x["donor_control"] and not x["entity_span"] for x in both)
        S["A_entity_vs_donor"][L] = {"n": len(both), "diff": (b - c) / len(both) if both else float("nan"),
                                     "entity_only": b, "donor_only": c, "sign_test_p": sign_test(b, c)}
        S["A_entity_span_by_kind"][L] = {k: _rate([r["A_by_layer"][str(L)]["entity_span"] for r in valid
                                                   if r["failure_kind"] == k and str(L) in r["A_by_layer"]])
                                         for k in KINDS}
    for L in sorted({int(L) for r in valid for L in r.get("B_by_layer", {})}):
        have = [r for r in valid if str(L) in r.get("B_by_layer", {})]
        base = sum(r["B_baseline_ok"] for r in have)
        patched = sum(r["B_by_layer"][str(L)] for r in have)
        restored = sum(r["B_by_layer"][str(L)] and not r["B_baseline_ok"] for r in have)
        S["experiment_B"][L] = {"n": len(have), "baseline": base, "patched": patched, "restored": restored,
                                "restored_rate": restored / len(have) if have else float("nan"),
                                "restored_ci95": wilson(restored, len(have))}
    # Experiment C (probe link): flagged = calibrated P(fail) >= τ, the live trigger rule
    tau = (meta.get("probe") or {}).get("tau", 0.5)
    C = {"tau": tau}

    def flag_stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"n": 0}
        k = sum(v >= tau for v in vals)
        return {"n": len(vals), "mean_p_fail": float(np.mean(vals)), "flag_rate": k / len(vals), "ci95": wilson(k, len(vals))}

    C["clean_baseline"] = flag_stats([r.get("C_clean") for r in valid])
    C["cf_baseline"] = flag_stats([r.get("C_cf") for r in valid])
    C["by_layer"] = {}
    agree = total = 0
    for L in layers:
        restored, not_restored = [], []
        for r in valid:
            pf = r.get("C_entity_span", {}).get(str(L))
            if pf is None:
                continue
            ok = r["A_by_layer"][str(L)]["entity_span"]
            (restored if ok else not_restored).append(pf)
            agree += (pf >= tau) != ok
            total += 1
        C["by_layer"][L] = {"restored": flag_stats(restored), "not_restored": flag_stats(not_restored)}
    C["agreement"] = {"k": agree, "n": total, "rate": agree / total if total else float("nan")}
    # A patch at block L >= the probe's layer cannot change what the probe reads (the hop-text states at the
    # probe's layer only see prompt states from blocks below L), so split the agreement there.
    pl = (meta.get("probe") or {}).get("layer")
    if pl is not None:
        for name, keep in (("upstream", lambda L: L < pl), ("downstream", lambda L: L >= pl)):
            a = t = 0
            for L in layers:
                if not keep(L):
                    continue
                for r in valid:
                    pf = r.get("C_entity_span", {}).get(str(L))
                    if pf is not None:
                        a += (pf >= tau) != r["A_by_layer"][str(L)]["entity_span"]
                        t += 1
            C[f"agreement_{name}"] = {"k": a, "n": t, "rate": a / t if t else float("nan"), "probe_layer": pl}
    pairs_c = [(r["C_clean"], r["C_cf"]) for r in valid if r.get("C_clean") is not None and r.get("C_cf") is not None]
    if pairs_c:
        from sklearn.metrics import roc_auc_score
        C["auroc_clean_vs_cf"] = float(roc_auc_score([0] * len(pairs_c) + [1] * len(pairs_c),
                                                     [c for c, _ in pairs_c] + [f for _, f in pairs_c]))
    C["self_check"] = meta.get("probe_self_check")
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
    C = S["experiment_C"]
    L = [f"# Phase 4 — activation patching, TF-IDF setting ({S.get('model')}, via {S.get('used_repo')})\n",
         f"Pairs: {S.get('n_pairs_total')} selected ({S.get('pair_filter')}), {S.get('n_aligned')} usable "
         f"(skipped {S.get('skipped')}); {S['n_pairs_done']} completed. Informative pairs (clean run states the gold "
         f"hop-1 entity, CF run does not): **{S['n_valid']}** {S.get('n_valid_by_kind')}. Clean and CF prompts differ "
         f"only in the entity (entity span median {S.get('entity_span_median')} tokens, tail median {S.get('tail_median')}). "
         f"Success = gold hop-1 entity or a Wikidata alias, whole words.\n",
         "## Checks",
         f"- hook self-test (all positions, last block): max |logit diff| {hs.get('max_abs_diff_patched', nan):.2e} "
         f"(unpatched {hs.get('max_abs_diff_unpatched', nan):.2e}) — must be ~0 vs large",
         f"- sanity (patch ALL prompt positions at block 0): {S['sanity_all_positions_L0_rate']:.3f} restored, generated "
         f"text identical to the clean run in {S['sanity_text_equals_clean_rate']:.3f} of pairs — both should be ~1",
         f"- baselines: clean run states the gold hop-1 entity {S['baseline_clean_states_gold_rate']:.3f}; "
         f"CF run {S['baseline_cf_states_gold_rate']:.3f}; CF run hedges ('not mentioned') {S['baseline_cf_hedge_rate']:.3f}",
         f"- probe-link self-check (id, cosine recomputed vs stored state, stored P, recomputed P): {C.get('self_check')}\n",
         "## Experiment A — hop-1 bridge entity restored, by patched block\n",
         "| block | entity span | tail (tokens after entity) | last prompt token | donor control (third entity) |",
         "|---|---|---|---|---|"]
    for layer, row in S["experiment_A"].items():
        L.append(f"| {layer} | " + " | ".join(_fmt_rate(row[c]) for c in CONDITIONS) + " |")
    L += ["\n### Entity span vs donor control (paired, same pairs)\n",
          "| block | n | entity − donor | entity only | donor only | sign test p |", "|---|---|---|---|---|---|"]
    for layer, v in S["A_entity_vs_donor"].items():
        L.append(f"| {layer} | {v['n']} | {v['diff']:+.2f} | {v['entity_only']} | {v['donor_only']} | {v['sign_test_p']:.2g} |")
    L += ["\n### Entity span restoration by CF failure kind\n", "| block | " + " | ".join(KINDS) + " |",
          "|---" * (len(KINDS) + 1) + "|"]
    for layer, v in S["A_entity_span_by_kind"].items():
        L.append(f"| {layer} | " + " | ".join(_fmt_rate(v[k]) for k in KINDS) + " |")
    if S["experiment_B"]:
        L += ["\n## Experiment B — single-state transplant at `</hop1>`, final answer restored\n",
              "| block | baseline correct | patched correct | restored (patched ok & baseline wrong) |", "|---|---|---|---|"]
        for layer, v in S["experiment_B"].items():
            L.append(f"| {layer} | {v['baseline']}/{v['n']} | {v['patched']}/{v['n']} | "
                     f"{v['restored_rate']:.2f} [{v['restored_ci95'][0]:.2f}, {v['restored_ci95'][1]:.2f}] |")
    ag = C["agreement"]
    pr = S.get("probe") or {}
    L += [f"\n## Experiment C — probe `{pr.get('dir')}` ({pr.get('task')}, block {pr.get('layer')}, calibrated, "
          f"flagged = P ≥ τ = {C['tau']:.3f}) on the generated hop 1\n",
          f"- clean baseline: {_fmt_flag(C['clean_baseline'])}",
          f"- CF baseline: {_fmt_flag(C['cf_baseline'])}",
          f"- AUROC clean vs CF baseline: {C.get('auroc_clean_vs_cf', float('nan')):.3f}",
          f"- verdict matches 'entity NOT restored' in {ag['k']}/{ag['n']} patched runs ({ag['rate']:.3f})"]
    for name in ("upstream", "downstream"):
        a = C.get(f"agreement_{name}")
        if a:
            side = "below" if name == "upstream" else "at or above"
            L.append(f"  - patches {side} the probe layer ({a['probe_layer']}): {a['k']}/{a['n']} ({a['rate']:.3f})")
    L += ["",
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
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/patching_tfidf")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"],
                    help="probe split of the pairs (test: the probe never saw the clean seeds)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="CF failure kinds to use")
    ap.add_argument("--layers", default="0,2,4,6,8,9,10,11,12,13,14,16,20,24",
                    help="blocks for Experiments A/C ('all' for every block)")
    ap.add_argument("--boundary-layers", default="8,10,11,14,20", help="blocks for Experiment B")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly between pairs after this long")
    ap.add_argument("--max-new-tokens", type=int, default=48, help="hop-1 generation budget")
    ap.add_argument("--answer-tokens", type=int, default=160, help="Experiment B generation budget")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--c-only", action="store_true",
                    help="Experiment C only: baselines, sanity and entity-span patches (+ probe); no tail / last "
                         "token / donor / Experiment B (~1/3 of the time; for re-running C with another probe)")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if args.summarize_only:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summarize(out, meta)
        print((out / "summary.md").read_text(encoding="utf-8"))
        return

    kinds = tuple(k for k in args.kinds.split(",") if k)
    pairs, why, aliases = load_pairs(args.split, kinds)
    print(f"minimal pairs (split={args.split}, kinds={kinds}; clean hop1 OK, CF hop1 failed): {len(pairs)}  "
          f"excluded: {why}")

    repo, msg = resolve_model(args.model, args.hf_token)
    print(msg)
    if repo is None:
        raise SystemExit(f"cannot access {args.model}; nothing to do")
    model, tok = load_causal_lm(repo, args.dtype, token=args.hf_token)
    n_layers = len(decoder_layers(model))
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",") if int(x) < n_layers]
    b_layers = [int(x) for x in args.boundary_layers.split(",") if int(x) < n_layers]
    R = Runner(model, tok, args.max_new_tokens)

    # ── prepare: token alignment against the recorded entity span, clean states ──
    prepared, skipped = [], defaultdict(int)
    for clean, cf in pairs:
        sw = cf["counterfactual_swap"]
        ids_c, ids_f = R.ids(clean["prompt"]), R.ids(cf["prompt"])
        if len(ids_c) != len(ids_f):
            skipped["prompt_lengths_differ"] += 1; continue
        diff = (ids_c != ids_f).nonzero().flatten().tolist()
        if not diff:
            skipped["prompts_identical"] += 1; continue
        off = len(ids_c) - len(tok(clean["prompt"], add_special_tokens=False).input_ids)   # special tokens added
        s0, s1 = sw["entity_token_span"]
        span = list(range(s0 + off, s1 + off))
        if min(diff) < span[0] or max(diff) > span[-1]:
            skipped["diff_outside_entity_span"] += 1; continue
        P = len(ids_c)
        tail = list(range(span[-1] + 1, P)) or [P - 1]
        st = R.states(ids_c, span + tail, layers)
        k = len(span)
        gold1 = gold_of(clean, 1)
        prepared.append(dict(clean=clean, cf=cf, ids_c=ids_c, ids_f=ids_f, span=span, tail=tail, P=P,
                             span_states={L: v[:k] for L, v in st.items()},
                             tail_states={L: v[k:] for L, v in st.items()},
                             last_states={L: v[-1:] for L, v in st.items()},
                             span_text_cf=tok.decode(ids_f[span]), span_text_clean=tok.decode(ids_c[span]),
                             gold1=gold1, forms=name_forms(gold1, aliases), kind=sw["failure_kind"]))
        if args.max_pairs and len(prepared) >= args.max_pairs:
            break
    print(f"usable pairs: {len(prepared)}  skipped: {dict(skipped)}")
    if not prepared:
        raise SystemExit("no token-aligned pairs — wrong tokenizer/model for this dataset?")

    # ── donor control: a third entity (another pair's clean entity tokens, same length) in THIS prompt ──
    rng = random.Random(0)
    n_ctrl = 0
    for i, p in enumerate(prepared):
        ctx = p["clean"]["context"].lower()
        cands = [j for j, q in enumerate(prepared) if j != i and len(q["span"]) == len(p["span"])
                 and q["clean"]["id"] != p["clean"]["id"] and not overlaps(q["gold1"], p["gold1"])
                 and normalize_text(q["span_text_clean"]) not in normalize_text(ctx)
                 and not overlaps(q["span_text_clean"], p["span_text_clean"])
                 and not overlaps(q["span_text_clean"], p["span_text_cf"])]
        p["donor"] = rng.choice(cands) if cands else None
        n_ctrl += p["donor"] is not None

    # ── guards (abort early, never after hours) ─────────────────────────────────
    span_lens = [len(p["span"]) for p in prepared]; tail_lens = [len(p["tail"]) for p in prepared]
    med_span, med_tail = statistics.median(span_lens), statistics.median(tail_lens)
    print(f"entity span length: median {med_span} (min {min(span_lens)}, max {max(span_lens)}); "
          f"tail length: median {med_tail} (min {min(tail_lens)}, max {max(tail_lens)})")
    print(f"donor control available for {n_ctrl}/{len(prepared)} pairs (same span length, different hop-1 gold)")
    print(f"gold forms with aliases: {sum(len(p['forms']) > 1 for p in prepared)}/{len(prepared)}")
    for p in prepared[:3]:
        d = prepared[p["donor"]]["span_text_clean"] if p["donor"] is not None else None
        print(f"  entity tokens: clean {p['span_text_clean']!r} -> CF {p['span_text_cf']!r} | control {d!r} | "
              f"gold hop 1 {p['gold1']!r}")
    if med_span > 40 or max(span_lens) > 80:
        raise SystemExit("entity-span detection looks broken (span too long); refusing to run")
    d_patch, d_none = R.hook_self_test(prepared[0]["ids_c"], prepared[0]["ids_f"])
    ok = d_patch < 1e-2 * max(d_none, 1e-6) or d_patch < 1e-3
    print(f"HOOK SELF-TEST: |patched - clean| = {d_patch:.2e}  vs  |unpatched - clean| = {d_none:.2e}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("hook self-test failed: patching is not taking effect; results would be meaningless")

    PL, chk = None, None
    if not args.skip_c:
        PL = ProbeLink(R, args.probe_dir)
        if PL.ok:
            try:
                chk = PL.self_check(prepared)
                cos = [c for _, c, _, _ in chk]
                verdict = "OK" if cos and min(cos) > 0.98 else "WARNING (probe link may be unreliable)"
                print(f"PROBE-LINK SELF-CHECK (id, cosine, stored P, recomputed P): {chk} -> {verdict}")
            except Exception as e:
                print(f"PROBE-LINK SELF-CHECK failed ({e.__class__.__name__}: {e}); Experiment C disabled")
                PL = None
        else:
            print(f"[info] {PL.msg}")
            PL = None
    if args.c_only and PL is None:
        raise SystemExit("--c-only needs a working probe link (see the message above); refusing to run")

    meta = {"c_only": args.c_only,"model": args.model, "used_repo": repo, "setting": "tfidf", "cf_revision": CF_V1["revision"],
            "data_revision": SETTINGS["tfidf"]["revision"],
            "pair_filter": f"split={args.split}, kinds={','.join(kinds)}", "n_pairs_total": len(pairs),
            "excluded": why, "n_aligned": len(prepared), "skipped": dict(skipped), "n_donor_control": n_ctrl,
            "layers_A": layers, "layers_B": b_layers, "entity_span_median": med_span, "tail_median": med_tail,
            "hook_self_test": {"max_abs_diff_patched": d_patch, "max_abs_diff_unpatched": d_none},
            "probe": {"dir": args.probe_dir, "task": PROBE_TASK, "layer": PL.layer, "tau": PL.tau} if PL else None,
            "probe_self_check": chk}
    meta_path.write_text(json.dumps(meta, indent=1))

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
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            print(f"time limit ({args.max_minutes} min) reached; stopping cleanly. Re-run to continue.")
            break
        try:
            donor = prepared[p["donor"]] if p["donor"] is not None else None
            rec = run_pair(R, p, donor, layers, b_layers, args, PL)
        except Exception as e:
            traceback.print_exc()
            cleanup_gpu()
            rec = {"v": RESULT_VERSION, "cf_id": cf_id, "error": f"{e.__class__.__name__}: {str(e)[:200]}"}
            print(f"[pair {i + 1}/{len(prepared)}] ERROR, skipped")
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        n_new += 1
        if not rec.get("error"):
            print(f"[{i + 1}/{len(prepared)} {time.time() - t0:.0f}s] {p['kind']} clean_ok={rec['baseline_clean_states_gold']} "
                  f"cf_ok={rec['baseline_cf_states_gold']} sanity={rec['sanity_all_positions_L0']} | restored at blocks: "
                  f"entity={[int(L) for L, v in rec['A_by_layer'].items() if v['entity_span']]} "
                  f"tail={[int(L) for L, v in rec['A_by_layer'].items() if v.get('tail')]} "
                  f"donor={[int(L) for L, v in rec['A_by_layer'].items() if v.get('donor_control')]}", flush=True)
        if n_new % 5 == 0:
            summarize(out, meta)
    fh.close()
    S = summarize(out, meta)
    print((out / "summary.md").read_text(encoding="utf-8"))
    n_err = sum(1 for l in open(pairs_path, encoding="utf-8") if l.strip() and json.loads(l).get("error"))
    print(f"DONE: {S['n_pairs_done']} pairs ok, {n_err} errors.")


def run_pair(R, p, donor, layers, b_layers, args, PL):
    rec = {"v": RESULT_VERSION, "clean_id": p["clean"]["id"], "cf_id": p["cf"]["id"], "gold_hop1": p["gold1"],
           "gold_forms": p["forms"], "failure_kind": p["kind"],
           "clean_q": p["clean"]["question"], "cf_q": p["cf"]["question"],
           "span_len": len(p["span"]), "tail_len": len(p["tail"]),
           "span_text_clean": p["span_text_clean"], "span_text_cf": p["span_text_cf"],
           "donor_clean_id": donor["clean"]["id"] if donor else None,
           "donor_span_text": donor["span_text_clean"] if donor else None}
    hit = lambda text: mentions(hop1_text(text), p["forms"])   # noqa: E731

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
    rec["baseline_clean_states_gold"] = hit(gen_clean)
    rec["baseline_cf_states_gold"] = hit(gen_cf)
    rec["baseline_cf_hedges"] = any(h in g_cf.lower() for h in HEDGE)
    rec["clean_baseline_text"] = g_clean.strip()[:200]
    rec["cf_baseline_text"] = g_cf.strip()[:200]
    rec["C_clean"] = probe(p["ids_c"], gen_clean)
    rec["C_cf"] = probe(p["ids_f"], gen_cf)

    st0 = R.states(p["ids_c"], list(range(p["P"])), [0])[0]       # sanity: all positions at block 0
    g = R.generate(p["ids_f"], 0, list(range(p["P"])), st0)
    rec["sanity_all_positions_L0"] = hit(g)
    rec["sanity_text_equals_clean"] = hop1_text(g).strip() == g_clean.strip()
    del st0

    ctrl = None
    if donor is not None and not args.c_only:     # third entity's tokens spliced into THIS prompt
        ids_x = p["ids_f"].clone()
        ids_x[p["span"]] = donor["ids_c"][donor["span"]]
        ctrl = R.states(ids_x, p["span"], layers)

    rec["A_by_layer"], rec["C_entity_span"] = {}, {}
    for L in layers:
        r = {}
        gen = R.generate(p["ids_f"], L, p["span"], p["span_states"][L])
        r["entity_span"] = hit(gen)
        rec["C_entity_span"][str(L)] = probe(p["ids_f"], gen, L, p["span"], p["span_states"][L])
        if not args.c_only:
            r["tail"] = hit(R.generate(p["ids_f"], L, p["tail"], p["tail_states"][L]))
            r["last_token"] = hit(R.generate(p["ids_f"], L, [p["P"] - 1], p["last_states"][L]))
        if ctrl is not None:
            r["donor_control"] = hit(R.generate(p["ids_f"], L, p["span"], ctrl[L]))
        rec["A_by_layer"][str(L)] = r

    if not args.skip_b and not args.c_only and b_layers:
        m_c = HOP1_RE.search(p["clean"]["generated_cot"]); m_f = HOP1_RE.search(p["cf"]["generated_cot"])
        if m_c and m_f:
            add = lambda t: torch.tensor(R.tok(t, add_special_tokens=False).input_ids, dtype=p["ids_c"].dtype)  # noqa: E731
            seq_c = torch.cat([p["ids_c"], add(p["clean"]["generated_cot"][: m_c.end()])])
            seq_f = torch.cat([p["ids_f"], add(p["cf"]["generated_cot"][: m_f.end()])])
            gold_ans = p["clean"]["gold_answer"]
            base = R.generate(seq_f, max_new=args.answer_tokens, stop="</answer>")
            m = ANS_RE.search(base)
            rec["B_baseline_ok"] = mentions(m.group(1) if m else base, [normalize_text(gold_ans)])
            st = R.states(seq_c, [len(seq_c) - 1], b_layers)
            rec["B_by_layer"] = {}
            for L in b_layers:
                g = R.generate(seq_f, L, [len(seq_f) - 1], st[L], max_new=args.answer_tokens, stop="</answer>")
                m = ANS_RE.search(g)
                rec["B_by_layer"][str(L)] = mentions(m.group(1) if m else g, [normalize_text(gold_ans)])
    return rec


if __name__ == "__main__":
    main()
