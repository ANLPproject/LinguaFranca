"""
probe_tracing.py  (Phase 4: probe <-> causal tracing, TF-IDF setting)
─────────────────────────────────────────────────────────────────────
The probe-vs-causal-tracing experiments of the mid-term work (cell7_score_probes.py,
create_kaggle_robustness_check / grounding_check), on the TF-IDF pairs, saving FEATURES
so every probe (old: C=1 block 10; tuned: C=0.01 block 12; any retrain) is scored on CPU.

Per (clean, CF) pair the clean run's own generated text up to </hop1> is teacher-forced
(exactly the Phase-1 extraction, so the clean features equal the stored hidden states) after
  clean        the clean prompt
  cf           the CF prompt                         (same hop-1 words, only the entity differs)
  entity_last  CF prompt, clean state at the last entity token, block l   (l = 0..27)
  entity_span  CF prompt, clean states at all entity tokens, block l
  control      CF prompt, entity tokens' states from a THIRD entity spliced into this prompt
and records for each run
  * log P(hop-1 tokens, entity-name tokens excluded)  -> behavioural recovery
    (lp - lp_cf) / (lp_clean - lp_cf), as causal_tracing_logit.py
  * the hop-1 TEXT states (tags excluded) mean-pooled at blocks 10 and 12 (the probes' layers)

Test A  same words, entity differs: does the probe score rise from clean to CF?
Test B  entity patched at block l < probe layer: does the probe score drop track the
        behavioural recovery (Spearman over pairs x blocks)?  The control should give neither.
The summary scores the old and the tuned probe; features.npz allows the robustness
(probe retrains / C) and grounding-direction analyses offline.

Usage (GPU, ~15-20 s per pair on a T4):
  python probe_tracing.py --out outputs/probe_tracing_tfidf --max-pairs 219
  python probe_tracing.py --out outputs/probe_tracing_tfidf --summarize-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_patching import HOP1_RE, hop1_token_span, load_pairs, overlaps
from lf_common import normalize_text
from lf_models import decoder_layers, load_causal_lm, resolve_model
from src.pipeline.probes import LinearProbe, apply_platt

FEAT_LAYERS = (10, 12)
PATCH = ("entity_last", "entity_span", "control")
PROBES = {"old (C=1, block 10)": "outputs/pipeline/tfidf/probes",
          "tuned (C=0.01, block 12)": "outputs/patching/patching_tuned_results/probes_c001_L12"}


@torch.no_grad()
def run_batch(model, layers, input_ids, P, mask, hop_pos, configs, batch):
    """configs: one list per run of (block, positions, vectors) patches.  Returns
    (summed log P of the masked target tokens, pooled hop states [runs, len(FEAT_LAYERS), d])."""
    dev = next(model.parameters()).device
    tgt = input_ids[P:].to(dev)
    lps, feats = [], []
    for s in range(0, len(configs), batch):
        chunk = configs[s:s + batch]
        by_mod, caps, handles = {}, {}, []
        for b, cfg in enumerate(chunk):
            for mod, pos, vecs in cfg:
                by_mod.setdefault(mod, []).append((b, pos, vecs))
        for mod, items in by_mod.items():
            def patch(module, args, output, items=items):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                for b, pos, vecs in items:
                    h[b, pos.to(h.device), :] = vecs.to(h.device, h.dtype)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            handles.append(mod.register_forward_hook(patch))
        for L in FEAT_LAYERS:                     # registered after the patches: sees the patched output
            def grab(module, args, output, L=L):
                h = output[0] if isinstance(output, tuple) else output
                caps[L] = h[:, hop_pos.to(h.device)].float().mean(1).cpu()
            handles.append(layers[L].register_forward_hook(grab))
        try:
            ids = input_ids.unsqueeze(0).repeat(len(chunk), 1).to(dev)
            hid = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, P - 1:-1]
            lp = F.log_softmax(model.lm_head(hid).float(), dim=-1)
        finally:
            for h in handles:
                h.remove()
        lp = lp.gather(2, tgt.view(1, -1, 1).expand(len(chunk), -1, 1)).squeeze(2)
        lps += lp[:, mask.to(dev)].sum(1).tolist()
        feats.append(torch.stack([caps[L] for L in FEAT_LAYERS], 1))
    return lps, torch.cat(feats).numpy().astype(np.float16)


@torch.no_grad()
def harvest(model, layers, input_ids):
    store, hs = {}, []
    for l, layer in enumerate(layers):
        def hook(module, args, output, l=l):
            h = output[0] if isinstance(output, tuple) else output
            store[l] = h[0].detach().to("cpu")
        hs.append(layer.register_forward_hook(hook))
    try:
        model.model(input_ids=input_ids.unsqueeze(0).to(next(model.parameters()).device), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return store


def prepare(tok, clean, cf):
    """Token-aligned ids, entity span, teacher-forced target (generated text up to </hop1>), masks."""
    m = HOP1_RE.search(clean["generated_cot"])
    if not m:
        return None, "no_hop1"
    tgt_text = clean["generated_cot"][: m.end()]
    c_ids = tok(clean["prompt"], return_tensors="pt").input_ids[0]
    f_ids = tok(cf["prompt"], return_tensors="pt").input_ids[0]
    if len(c_ids) != len(f_ids):
        return None, "length_differs"
    off = len(c_ids) - len(tok(clean["prompt"], add_special_tokens=False).input_ids)
    s0, s1 = cf["counterfactual_swap"]["entity_token_span"]
    span = list(range(s0 + off, s1 + off))
    diff = (c_ids != f_ids).nonzero().flatten().tolist()
    if not diff or min(diff) < span[0] or max(diff) > span[-1]:
        return None, "diff_outside_entity_span"
    hs = hop1_token_span(tok, tgt_text)                    # Phase-1 char->token rule
    if hs is None:
        return None, "no_hop1_text"
    enc, ts, te = hs
    t_ids = torch.tensor(enc["input_ids"], dtype=c_ids.dtype)
    P = len(c_ids)
    mask = torch.zeros(len(t_ids), dtype=torch.bool)
    mask[ts:te + 1] = True                                 # score the hop-1 text tokens ...
    mask &= ~torch.isin(t_ids, torch.cat([c_ids[span], f_ids[span]]))   # ... except entity-name tokens
    if mask.sum() == 0:
        return None, "no_scored_tokens"
    return dict(c_ids=c_ids, f_ids=f_ids, t_ids=t_ids, span=span, P=P, mask=mask,
                hop_pos=torch.arange(P + ts, P + te + 1), clean=clean, cf=cf,
                gold1=next((n["gold_entity"] for n in clean["reasoning_graph"] if n.get("hop") == 1), ""),
                ent_text=tok.decode(c_ids[span])), None


def pick_controls(prep, seed=0):
    """Third entity for each pair: another pair's clean entity tokens, same span length, different
    hop-1 gold, not in this context (as causal_patching.py's donor control)."""
    rng = random.Random(seed)
    for i, p in enumerate(prep):
        ctx = normalize_text(p["clean"]["context"])
        c = [j for j, q in enumerate(prep) if j != i and len(q["span"]) == len(p["span"])
             and not overlaps(q["gold1"], p["gold1"]) and normalize_text(q["ent_text"]) not in ctx
             and not overlaps(q["ent_text"], p["ent_text"])]
        p["control_tokens"] = None
        if c:
            j = rng.choice(c)
            p["control_tokens"] = prep[j]["c_ids"][prep[j]["span"]]


def trace(model, layers, p, batch):
    P, span = p["P"], p["span"]
    clean_in = torch.cat([p["c_ids"], p["t_ids"]])
    cf_in = torch.cat([p["f_ids"], p["t_ids"]])
    store = harvest(model, layers, clean_in)
    (lp_clean,), f_clean = run_batch(model, layers, clean_in, P, p["mask"], p["hop_pos"], [[]], batch)
    configs, keys = [[]], [("cf", -1)]
    last_e, span_t = torch.tensor([span[-1]]), torch.tensor(span)
    ctrl = None
    if p["control_tokens"] is not None:
        x_in = cf_in.clone()
        x_in[span_t] = p["control_tokens"]
        ctrl = harvest(model, layers, x_in)
    for l in range(len(layers)):
        configs.append([(layers[l], last_e, store[l][last_e])]); keys.append(("entity_last", l))
        configs.append([(layers[l], span_t, store[l][span_t])]); keys.append(("entity_span", l))
        if ctrl is not None:
            configs.append([(layers[l], span_t, ctrl[l][span_t])]); keys.append(("control", l))
    lps, feats = run_batch(model, layers, cf_in, P, p["mask"], p["hop_pos"], configs, batch)
    lp_cf = lps[0]
    n_l = len(layers)
    rec = np.full((len(PATCH), n_l), np.nan, dtype=np.float32)
    fp = np.full((len(PATCH), n_l, len(FEAT_LAYERS), feats.shape[-1]), np.nan, dtype=np.float16)
    gap = lp_clean - lp_cf
    for (k, l), lp, f in zip(keys, lps, feats):
        if k == "cf":
            continue
        rec[PATCH.index(k), l] = (lp - lp_cf) / gap if gap != 0 else np.nan
        fp[PATCH.index(k), l] = f
    return dict(gap=gap, recovery=rec, F_clean=f_clean[0], F_cf=feats[0], F_patch=fp)


# ─────────────────────────────────────────────────────────────────────────────

def load_probe(d):
    meta = json.loads((Path(d) / "probes_meta.json").read_text())["probes"]["hop12"]
    lp = LinearProbe(Path(d) / "hop12.npz")
    raw = lambda X: lp.p_fail(np.asarray(X, np.float32))                  # noqa: E731
    return dict(layer=meta["layer"], tau=meta["tau"], calib=meta["calibration"], raw=raw)


def logodds(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def summarize(out: Path, min_gap: float):
    from scipy.stats import spearmanr
    Z = np.load(out / "features.npz", allow_pickle=True)
    gap = Z["gap"]
    keep = gap >= min_gap
    L = ["# Probe <-> causal tracing, TF-IDF pairs", "",
         f"Pairs traced: {len(gap)}; with a clean-vs-CF hop-1 log-prob gap >= {min_gap} nat: {keep.sum()}. "
         "Hop 1 = the clean run's own text, teacher-forced in every run (same words); only the entity differs.", ""]
    rec = Z["recovery"][keep]
    L += ["## Behavioural recovery (fraction of the log-prob gap), mean by block", "",
          "| block | " + " | ".join(PATCH) + " |", "|---" * (len(PATCH) + 1) + "|"]
    for l in range(rec.shape[2]):
        L.append(f"| {l} | " + " | ".join(f"{np.nanmean(rec[:, k, l]):.2f}" for k in range(len(PATCH))) + " |")
    rng = np.random.default_rng(0)
    for name, d in PROBES.items():
        if not (Path(d) / "probes_meta.json").exists():
            L += ["", f"## {name}: bundle not found at {d}"]; continue
        pr = load_probe(d)
        li = FEAT_LAYERS.index(pr["layer"])
        cal = lambda X: apply_platt(pr["calib"], pr["raw"](X))           # noqa: E731
        pc, pf = cal(Z["F_clean"][keep, li]), cal(Z["F_cf"][keep, li])
        d_lo = logodds(pf) - logodds(pc)
        boot = [d_lo[rng.integers(0, len(d_lo), len(d_lo))].mean() for _ in range(2000)]
        from sklearn.metrics import roc_auc_score
        L += ["", f"## {name}", "",
              "**Test A — same hop-1 words, only the question entity differs:**",
              f"- P(fail) clean {pc.mean():.3f} -> CF {pf.mean():.3f}; flagged at τ={pr['tau']:.3f}: "
              f"clean {(pc >= pr['tau']).mean():.3f}, CF {(pf >= pr['tau']).mean():.3f}",
              f"- log-odds rise {d_lo.mean():+.2f} [95% CI {np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]; "
              f"rises in {(d_lo > 0).mean() * 100:.0f}% of pairs; AUROC clean vs CF "
              f"{roc_auc_score([0] * len(pc) + [1] * len(pf), np.r_[pc, pf]):.3f}", "",
              f"**Test B — patch at block l < {pr['layer']} (only those can reach the probe's block):**", "",
              "| block | recovery (entity last / span / control) | probe log-odds drop (entity last / span / control) |",
              "|---|---|---|"]
        lo_cf = logodds(pf)
        drops = {}
        for k, c in enumerate(PATCH):
            fpk = Z["F_patch"][keep, k, :, li]                            # [n, 28, d]
            sc = np.full(fpk.shape[:2], np.nan)
            ok = ~np.isnan(fpk[..., 0].astype(np.float32))
            sc[ok] = logodds(cal(fpk[ok]))
            drops[c] = lo_cf[:, None] - sc
        for l in range(pr["layer"] + 2):
            L.append(f"| {l} | " + " / ".join(f"{np.nanmean(rec[:, k, l]):.2f}" for k in range(len(PATCH))) + " | "
                     + " / ".join(f"{np.nanmean(drops[c][:, l]):+.2f}" for c in PATCH) + " |")
        for c in PATCH:
            x = rec[:, PATCH.index(c), :pr["layer"]].ravel(); y = drops[c][:, :pr["layer"]].ravel()
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() > 10:
                rho, pv = spearmanr(x[ok], y[ok])
                L.append(f"- Spearman(recovery, probe drop), {c}, blocks < {pr['layer']}: rho = {rho:.2f} (p = {pv:.1e}, n = {ok.sum()})")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/probe_tracing_tfidf")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(out, args.min_gap)); return

    pairs, why, _ = load_pairs("test")
    pairs = pairs[:args.max_pairs] if args.max_pairs else pairs
    repo, msg = resolve_model(args.model, args.hf_token)
    print(msg)
    model, tok = load_causal_lm(repo, "float16", token=args.hf_token)
    layers = decoder_layers(model)
    prep, skipped = [], {}
    for clean, cf in pairs:
        p, reason = prepare(tok, clean, cf)
        if p is None:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            prep.append(p)
    pick_controls(prep)
    print(f"pairs: {len(pairs)}  usable: {len(prep)}  skipped: {skipped}  "
          f"with control: {sum(p['control_tokens'] is not None for p in prep)}")

    # self-check: clean teacher-forced features must equal the stored Phase-1 states
    from causal_patching import stored_hop1_state
    p0 = prep[0]
    r0 = trace(model, layers, p0, args.batch)
    for L in FEAT_LAYERS:
        ref = stored_hop1_state(p0["clean"], L)
        v = r0["F_clean"][FEAT_LAYERS.index(L)].astype(np.float32)
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
        print(f"SELF-CHECK block {L}: cosine(teacher-forced clean, stored Phase-1 state) = {cos:.4f}")
        if cos < 0.99:
            raise SystemExit("feature mismatch with the stored states: probe scores would be meaningless")

    res, ids, t0 = [r0], [p0["cf"]["id"]], time.time()
    for i, p in enumerate(prep[1:], 2):
        try:
            res.append(trace(model, layers, p, args.batch)); ids.append(p["cf"]["id"])
        except torch.cuda.OutOfMemoryError:
            raise SystemExit("CUDA OOM: re-run with a smaller --batch")
        r = res[-1]
        print(f"[{i}/{len(prep)} {time.time() - t0:.0f}s] gap {r['gap']:.1f} | entity_span recovery "
              f"L0 {r['recovery'][1, 0]:.2f} L10 {r['recovery'][1, 10]:.2f} L14 {r['recovery'][1, 14]:.2f}", flush=True)
        if i % 10 == 0:
            torch.cuda.empty_cache()
    np.savez_compressed(out / "features.npz", ids=np.array(ids), gap=np.array([r["gap"] for r in res]),
                        recovery=np.stack([r["recovery"] for r in res]),
                        F_clean=np.stack([r["F_clean"] for r in res]), F_cf=np.stack([r["F_cf"] for r in res]),
                        F_patch=np.stack([r["F_patch"] for r in res]), feat_layers=np.array(FEAT_LAYERS),
                        patch=np.array(PATCH), kinds=np.array([p["cf"]["counterfactual_swap"]["failure_kind"] for p in prep]))
    print(summarize(out, args.min_gap))


if __name__ == "__main__":
    main()
