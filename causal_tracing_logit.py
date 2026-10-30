"""
causal_tracing_logit.py  (Phase 4, logit-based causal tracing — TF-IDF setting)
────────────────────────────────────────────────────────────────────────────────
The method of the mid-term slides (create_kaggle_phase4_patching.ipynb v3, commit dbcfd74),
run on the TF-IDF counterfactual pairs that causal_patching.py v5 uses.

Per (clean, CF) pair: teacher-force the CLEAN run's own hop 1 ("<hop1>…</hop1>") after
  clean prompt  -> log P_clean(hop 1)
  CF prompt     -> log P_cf(hop 1)
  CF prompt with one clean state copied in at block l -> log P_patched(hop 1)
and score  recovery = (log P_patched - log P_cf) / (log P_clean - log P_cf)
(0 = no effect, 1 = fully restored).  Hop-1 tokens that are entity-name tokens are not
scored (the copied name would trivially favour the clean run).  Pairs with a gap < 1 nat
are skipped (nothing to restore).

Conditions, every block 0-27:
  entity_last  residual stream at the LAST token of the swapped entity (Meng et al.'s
               "last subject token")                                         [slides]
  last_token   residual stream at the last prompt token                      [slides]
  mlp_window   MLP outputs at the last entity token, blocks l-2 .. l+2       [slides]
  entity_span  residual stream at ALL entity tokens        (as Experiment A of causal_patching.py)
  tail         residual stream at all tokens after the entity              (as Experiment A)
plus a sanity check: all prompt positions at block 0 (recovery close to 1).

Differences from the v3 notebook (method unchanged): TF-IDF data + cf_v1 pairs, which are
exactly token-aligned (v3 had to align 45/93 full-10 pairs of different length); the entity
span is the recorded one; prompts are tokenized as in Phase 1 (default special tokens);
entity_span / tail / sanity added; per-pair records saved, resumable.

Usage (GPU, ~15-20 s per pair on a T4):
  python causal_tracing_logit.py --out outputs/causal_tracing_tfidf --max-pairs 219
  python causal_tracing_logit.py --out outputs/causal_tracing_tfidf --summarize-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from causal_patching import load_pairs
from lf_models import decoder_layers, load_causal_lm, resolve_model

N_LAYERS = 28
WINDOW = 5
CONDITIONS = ("entity_last", "last_token", "mlp_window", "entity_span", "tail")
LABEL = {"entity_last": "Entity token (last)", "last_token": "Last prompt token",
         "mlp_window": f"MLP at entity (±{WINDOW // 2} blocks)", "entity_span": "Entity span (all tokens)",
         "tail": "Tail (tokens after entity)"}
VERSION = 1


def get_hop1(cot):
    m = re.search(r"(<hop1>.*?</hop1>)", cot, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


class Tracer:
    def __init__(self, model, tok, batch):
        self.model, self.tok, self.batch = model, tok, batch
        self.layers = decoder_layers(model)
        self.dev = next(model.parameters()).device

    @torch.no_grad()
    def run(self, input_ids, P, mask, configs):
        """configs: one list per run of (module, positions, vectors).  Returns summed log P of the
        target tokens (input_ids[P:]) under `mask` for each run (the v3 notebook's run_batch)."""
        out, tgt = [], input_ids[P:].to(self.dev)
        for s in range(0, len(configs), self.batch):
            chunk = configs[s:s + self.batch]
            by_mod = {}
            for b, cfg in enumerate(chunk):
                for mod, pos, vecs in cfg:
                    by_mod.setdefault(mod, []).append((b, pos, vecs))
            handles = []
            for mod, items in by_mod.items():
                def hook(module, args, output, items=items):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    for b, pos, vecs in items:
                        h[b, pos.to(h.device), :] = vecs.to(h.device, h.dtype)
                    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
                handles.append(mod.register_forward_hook(hook))
            try:
                ids = input_ids.unsqueeze(0).repeat(len(chunk), 1).to(self.dev)
                hid = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, P - 1:-1]
                lp = F.log_softmax(self.model.lm_head(hid).float(), dim=-1)
            finally:
                for hd in handles:
                    hd.remove()
            lp = lp.gather(2, tgt.view(1, -1, 1).expand(len(chunk), -1, 1)).squeeze(2)
            out += lp[:, mask.to(self.dev)].sum(1).tolist()
        return out

    @torch.no_grad()
    def harvest(self, input_ids):
        """Clean run: residual stream + MLP outputs of every block, [seq, d] each (CPU)."""
        store, hs = {}, []

        def mk(key):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                store[key] = h[0].detach().to("cpu")
            return hook
        for l, layer in enumerate(self.layers):
            hs.append(layer.register_forward_hook(mk(("res", l))))
            hs.append(layer.mlp.register_forward_hook(mk(("mlp", l))))
        try:
            self.model.model(input_ids=input_ids.unsqueeze(0).to(self.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return store


def trace_pair(T, tok, clean, cf, min_gap):
    sw = cf["counterfactual_swap"]
    tgt = get_hop1(clean["generated_cot"])
    if not tgt:
        return {"skip": "no_hop1"}
    c_ids = tok(clean["prompt"], return_tensors="pt").input_ids[0]          # Phase 1 tokenization
    f_ids = tok(cf["prompt"], return_tensors="pt").input_ids[0]
    if len(c_ids) != len(f_ids):
        return {"skip": "length_differs"}
    off = len(c_ids) - len(tok(clean["prompt"], add_special_tokens=False).input_ids)
    s0, s1 = sw["entity_token_span"]
    span = list(range(s0 + off, s1 + off))
    diff = (c_ids != f_ids).nonzero().flatten().tolist()
    if not diff or min(diff) < span[0] or max(diff) > span[-1]:
        return {"skip": "diff_outside_entity_span"}
    t_ids = torch.tensor(tok(tgt, add_special_tokens=False).input_ids, dtype=c_ids.dtype)
    ent_tok = torch.cat([c_ids[span], f_ids[span]])
    mask = ~torch.isin(t_ids, ent_tok)
    if mask.sum() == 0:
        return {"skip": "no_scored_tokens"}
    P = len(c_ids)
    clean_in, cf_in = torch.cat([c_ids, t_ids]), torch.cat([f_ids, t_ids])
    store = T.harvest(clean_in)
    lp_clean = T.run(clean_in, P, mask, [[]])[0]
    lp_cf = T.run(cf_in, P, mask, [[]])[0]
    gap = lp_clean - lp_cf
    if gap < min_gap:
        return {"skip": "no_gap", "gap": gap}

    L = T.layers
    last_e = torch.tensor([span[-1]])
    span_t = torch.tensor(span)
    tail_t = torch.tensor(list(range(span[-1] + 1, P)))
    last_p = torch.tensor([P - 1])
    allp = torch.arange(P)
    configs, keys = [[(L[0], allp, store[("res", 0)][allp])]], [("sanity", 0)]
    for l in range(len(L)):
        res = store[("res", l)]
        configs.append([(L[l], last_e, res[last_e])]); keys.append(("entity_last", l))
        configs.append([(L[l], last_p, res[last_p])]); keys.append(("last_token", l))
        win = range(max(0, l - WINDOW // 2), min(len(L), l + WINDOW // 2 + 1))
        configs.append([(L[k].mlp, last_e, store[("mlp", k)][last_e]) for k in win]); keys.append(("mlp_window", l))
        configs.append([(L[l], span_t, res[span_t])]); keys.append(("entity_span", l))
        configs.append([(L[l], tail_t, res[tail_t])]); keys.append(("tail", l))
    scores = T.run(cf_in, P, mask, configs)
    rec = {c: [None] * len(L) for c in CONDITIONS}
    for (c, l), lp in zip(keys, scores):
        if c == "sanity":
            rec["sanity_all_positions_L0"] = (lp - lp_cf) / gap
        else:
            rec[c][l] = (lp - lp_cf) / gap
    rec.update(gap=gap, n_scored=int(mask.sum()), span_len=len(span), tail_len=len(tail_t),
               entity_clean=tok.decode(c_ids[span]), entity_cf=tok.decode(f_ids[span]))
    return rec


def summarize(out: Path):
    recs = [json.loads(l) for l in open(out / "pairs.jsonl", encoding="utf-8") if l.strip()]
    ok = [r for r in recs if r.get("v") == VERSION and "skip" not in r and "error" not in r]
    skips = {}
    for r in recs:
        if "skip" in r:
            skips[r["skip"]] = skips.get(r["skip"], 0) + 1
    n = len(ok)
    S = {"n_pairs_traced": n, "skipped": skips, "conditions": {}}
    if not n:
        (out / "summary.json").write_text(json.dumps(S, indent=1))
        return S
    S["sanity_all_positions_L0_mean"] = float(np.mean([r["sanity_all_positions_L0"] for r in ok]))
    S["gap_median"] = float(np.median([r["gap"] for r in ok]))
    for c in CONDITIONS:
        A = np.array([r[c] for r in ok], dtype=float)                      # [n, 28]
        m, ci = A.mean(0), 1.96 * A.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(A.shape[1])
        S["conditions"][c] = {"mean": m.tolist(), "ci95": ci.tolist(), "frac_gt_half": (A > 0.5).mean(0).tolist(),
                              "peak_layer": int(m.argmax()), "peak": float(m.max())}
    (out / "summary.json").write_text(json.dumps(S, indent=1))

    md = [f"# Logit-based causal tracing of hop 1 — TF-IDF counterfactual pairs (n = {n})", "",
          "Recovery = (log P_patched − log P_cf) / (log P_clean − log P_cf) of the clean run's hop 1 "
          "(teacher-forced, entity-name tokens not scored); 0 = no effect, 1 = fully restored. "
          f"Skipped: {skips}. Median gap {S['gap_median']:.1f} nats.", "",
          f"Sanity (all prompt positions at block 0): mean recovery {S['sanity_all_positions_L0_mean']:.3f} (should be close to 1: later tokens still see the CF entity in block-0 attention).", "",
          "Peaks: " + "; ".join(f"{LABEL[c]} {S['conditions'][c]['peak']:.2f} at block {S['conditions'][c]['peak_layer']}"
                                for c in CONDITIONS), "",
          "| block | " + " | ".join(LABEL[c] for c in CONDITIONS) + " |", "|---" * (len(CONDITIONS) + 1) + "|"]
    for l in range(N_LAYERS):
        md.append(f"| {l} | " + " | ".join(f"{S['conditions'][c]['mean'][l]:.2f} ± {S['conditions'][c]['ci95'][l]:.2f}"
                                          for c in CONDITIONS) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        for c in ("entity_last", "last_token", "mlp_window"):
            m, ci = np.array(S["conditions"][c]["mean"]), np.array(S["conditions"][c]["ci95"])
            plt.plot(range(N_LAYERS), m, marker="o", label=LABEL[c])
            plt.fill_between(range(N_LAYERS), m - ci, m + ci, alpha=0.2)
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel("Block patched"); plt.ylabel("Fraction of clean-vs-CF gap recovered")
        plt.title(f"Causal tracing of hop 1, TF-IDF (n={n} pairs)"); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(out / "causal_trace.png", dpi=200, bbox_inches="tight")
    except Exception as e:                                               # plotting is optional
        print(f"[plot skipped] {e}")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/causal_tracing_tfidf")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--max-pairs", type=int, default=None, help="219 = the pairs of the patching runs")
    ap.add_argument("--batch", type=int, default=7, help="patched runs per forward pass (lower on OOM)")
    ap.add_argument("--min-gap", type=float, default=1.0, help="nats; pairs below have nothing to restore")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out)
        print((out / "summary.md").read_text(encoding="utf-8"))
        return

    pairs, why, _ = load_pairs(args.split)
    pairs = pairs[:args.max_pairs] if args.max_pairs else pairs
    print(f"pairs: {len(pairs)} (same fixed order as causal_patching.py)  excluded: {why}")
    repo, msg = resolve_model(args.model, args.hf_token)
    print(msg)
    if repo is None:
        raise SystemExit(f"cannot access {args.model}")
    model, tok = load_causal_lm(repo, "float16", token=args.hf_token)
    T = Tracer(model, tok, args.batch)

    path = out / "pairs.jsonl"
    done = set()
    if path.exists():
        done = {r["cf_id"] for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())
                if r.get("v") == VERSION and "error" not in r}
        print(f"resuming: {len(done)} pairs done")
    t0, fh = time.time(), open(path, "a", encoding="utf-8")
    for i, (clean, cf) in enumerate(pairs):
        if cf["id"] in done:
            continue
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            print("time limit reached; re-run to continue"); break
        try:
            rec = trace_pair(T, tok, clean, cf, args.min_gap)
        except torch.cuda.OutOfMemoryError:
            raise SystemExit("CUDA OOM: re-run with a smaller --batch (finished pairs are kept)")
        except Exception as e:
            rec = {"error": f"{e.__class__.__name__}: {str(e)[:200]}"}
        rec.update(v=VERSION, cf_id=cf["id"], clean_id=clean["id"], failure_kind=cf["counterfactual_swap"]["failure_kind"])
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        if "skip" in rec or "error" in rec:
            print(f"[{i + 1}/{len(pairs)}] {rec.get('skip') or rec.get('error')}")
        else:
            el, lt = np.array(rec["entity_last"]), np.array(rec["last_token"])
            print(f"[{i + 1}/{len(pairs)} {time.time() - t0:.0f}s] gap {rec['gap']:.1f} sanity {rec['sanity_all_positions_L0']:.2f} | "
                  f"entity_last peak {el.max():.2f}@{el.argmax()} last_token peak {lt.max():.2f}@{lt.argmax()}", flush=True)
        if i % 10 == 0:
            torch.cuda.empty_cache()
    fh.close()
    summarize(out)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
