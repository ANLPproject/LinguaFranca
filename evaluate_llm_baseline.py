"""
evaluate_llm_baseline.py  (v2)
──────────────────────────────
Text-only "LLM-as-judge" baseline for hop-failure detection, scored on EXACTLY
the same held-out rows as the hidden-state probe (lf_common: canonical split,
hops 1-2, label 0/1).

What changed vs v1 (see FIXES_AND_RERUN.md):
  * v1 judged every generated hop, including the model's extra hops 3..18 that
    the old labeler marked "failure" by default (64 % of all failure labels).
    The judge (correctly) called many of those grounded, which pushed its
    accuracy down to 61.7 %.  v2 judges hops 1-2 only.
  * v1 ran on the later 115-CF data file and a different split than the probe.
    v2 uses the canonical file and split.
  * v2 reads P(no) vs P(yes) from the next-token logits (one forward pass, no
    sampling), so the judge gets an AUROC directly comparable to the probe.
  * Reports TPR with FPR, per stratum (hop 1 / hop 2 / natural / CF, hedging vs
    confident failures).
  * If --probe-predictions is given (outputs/probing/test_predictions.csv), the
    probe is scored on the identical rows next to each judge.

The prompt is unchanged from v1.  Note what it asks: "is this step correct and
grounded in the context?"  For counterfactual rows the hop text is usually a
TRUE statement ("X is not mentioned in the context") that our label calls a
failure; read the CF/hedging rows with that in mind.

Usage (GPU; Llama models need an HF token with Llama access)
─────
  python evaluate_llm_baseline.py --data data/canonical/augmented_v2.jsonl \
      --models meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen2.5-7B-Instruct \
      --probe-predictions outputs/probing/test_predictions.csv --out outputs/llm_judge
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from lf_common import DEFAULT_DATA, binary_metrics, canonical_split, fmt_metrics, hop_rows, load_examples
from lf_models import input_device, load_causal_lm

DEFAULT_MODELS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                  "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]

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


def context_str(ctx) -> str:
    if isinstance(ctx, list):
        return "".join(c[1] if isinstance(c, (list, tuple)) and len(c) > 1 else str(c) for c in ctx)
    return str(ctx or "")


def answer_token_ids(tok, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            enc = tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def strata_report(rows, prob, pred):
    prob = np.asarray(prob, dtype=float); pred = np.asarray(pred)
    ok = np.isfinite(prob)
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"  [warn] {n_bad} rows had non-finite logits and are excluded from this model's metrics")
    rows = [r for r, k in zip(rows, ok) if k]; prob = prob[ok]; pred = pred[ok]
    y = np.array([r["y"] for r in rows]); hop = np.array([r["hop"] for r in rows])
    cf = np.array([r["cf"] for r in rows]); hedge = np.array([r["hedge"] for r in rows])
    masks = {"all": np.ones(len(y), bool), "hop1": hop == 1, "hop2": hop == 2,
             "natural": ~cf, "cf": cf, "hop1_natural": (hop == 1) & ~cf}
    rep = {k: binary_metrics(y[m], prob[m], pred[m]) for k, m in masks.items() if m.sum()}
    rep["n_nonfinite_excluded"] = n_bad
    f = y == 1
    for name, m in {"hedge_fail_recall": f & hedge, "confident_fail_recall": f & ~hedge,
                    "hop1_confident_fail_recall": f & ~hedge & (hop == 1)}.items():
        rep[name] = {"value": float(pred[m].mean()) if m.sum() else float("nan"), "n": int(m.sum())}
    return rep


def load_probe_predictions(path, results_json=None):
    """Map (id, hop) -> probe probability using the hop-specific probe at the selected layers."""
    sel = None
    rj = Path(results_json) if results_json else Path(path).with_name("results.json")
    if rj.exists():
        sel = json.loads(rj.read_text())["selected_layers"]
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hop = int(r["hop_idx"])
            task = f"hop{hop}"
            L = sel[task] if sel else 11
            v = r.get(f"{task}_L{L}", "")
            if v != "":
                out[(r["id"], hop)] = float(v)
    return out, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "--labels", dest="data", default=DEFAULT_DATA)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="outputs/llm_judge")
    ap.add_argument("--hops", default="1,2")
    ap.add_argument("--max-items", type=int, default=None, help="for quick tests")
    ap.add_argument("--probe-predictions", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--hs-dir", default=None, help="ignored (kept for old notebook calls)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.data)
    _, te_idx = canonical_split(examples)
    test_ex = [examples[i] for i in te_idx]
    hops = tuple(int(h) for h in args.hops.split(","))
    rows = hop_rows(test_ex, hops=hops, require_text=True)
    if args.max_items:
        rows = rows[:args.max_items]
    y = np.array([r["y"] for r in rows])
    print(f"test examples={len(test_ex)}  rows judged={len(rows)}  failures={y.sum()} ({100 * y.mean():.1f}%)  "
          f"CF rows={sum(r['cf'] for r in rows)}")

    summary = {"n_rows": len(rows), "models": {}}
    if args.probe_predictions:
        pp, sel = load_probe_predictions(args.probe_predictions)
        keep = [i for i, r in enumerate(rows) if (r["id"], r["hop"]) in pp]
        if len(keep) == len(rows):
            pr = np.array([pp[(r["id"], r["hop"])] for r in rows])
            summary["probe"] = {"selected_layers": sel, **strata_report(rows, pr, (pr >= 0.5).astype(int))}
            print(f"\nPROBE on identical rows (layers {sel}): {fmt_metrics(summary['probe']['all'])}")
        else:
            print(f"[warn] probe predictions cover {len(keep)}/{len(rows)} rows — skipping probe comparison")

    for model_name in args.models.split(","):
        model_name = model_name.strip()
        print(f"\n{'=' * 80}\nLoading {model_name} ...")
        try:
            model, tok = load_causal_lm(model_name, args.dtype, token=args.hf_token)
        except Exception as e:
            print(f"[skip] could not load {model_name}: {e}")
            continue
        yes_ids = answer_token_ids(tok, ["yes", "Yes", "YES"])
        no_ids = answer_token_ids(tok, ["no", "No", "NO"])
        dev = input_device(model)
        p_fail, preds, lat, ntok, top = [], [], [], [], []
        t_start = time.time()
        id2ex = {x["id"]: x for x in test_ex}
        for i, r in enumerate(rows):
            ex = id2ex[r["id"]]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(context=context_str(ex.get("context")),
                                                                 question=ex["question"], hop=r["text"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                logits = model(**enc).logits[0, -1].float()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.time() - t0) * 1000); ntok.append(enc.input_ids.shape[1])
            l_yes = torch.logsumexp(logits[yes_ids], 0); l_no = torch.logsumexp(logits[no_ids], 0)
            if not torch.isfinite(l_no - l_yes):              # fp16 overflow (e.g. Qwen on T4)
                p_fail.append(float("nan")); preds.append(-1); top.append("<nonfinite>"); continue
            pf = torch.sigmoid(l_no - l_yes).item()          # P(no | yes-or-no) = P(failure)
            p_fail.append(pf); preds.append(int(l_no > l_yes))
            top.append(tok.decode([int(logits.argmax())]))
            if i < 10:
                print(f"  [{i}] top-token={top[-1]!r:8} p_fail={pf:.3f} label={r['y']} hop{r['hop']} :: {r['text'][:90]}")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(rows)}  ({time.time() - t_start:.0f}s)", flush=True)

        rep = strata_report(rows, p_fail, preds)
        frac_yesno = float(np.mean([t.strip().lower() in ("yes", "no") for t in top]))
        rep["latency_ms_mean"] = float(np.mean(lat)); rep["prompt_tokens_mean"] = float(np.mean(ntok))
        rep["top_token_is_yes_or_no"] = frac_yesno
        summary["models"][model_name] = rep
        print(f"\nRESULTS {model_name}  (top token was yes/no for {100 * frac_yesno:.1f}% of rows)")
        for k in ("all", "hop1", "hop2", "natural", "cf", "hop1_natural"):
            if k in rep:
                print(f"  {k:13s} {fmt_metrics(rep[k])}")
        for k in ("hedge_fail_recall", "confident_fail_recall", "hop1_confident_fail_recall"):
            print(f"  {k:27s} {rep[k]['value']:.3f} (n={rep[k]['n']})")
        print(f"  latency {rep['latency_ms_mean']:.0f} ms/hop, {rep['prompt_tokens_mean']:.0f} prompt tokens/hop")

        safe = model_name.replace("/", "__")
        with open(out / f"judge_predictions_{safe}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "hop_idx", "label", "is_cf", "is_hedge", "p_fail", "pred", "top_token", "latency_ms"])
            for r, pf, pd, tt, la in zip(rows, p_fail, preds, top, lat):
                w.writerow([r["id"], r["hop"], r["y"], int(r["cf"]), int(r["hedge"]), f"{pf:.5f}", pd, tt, f"{la:.1f}"])
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_markdown(summary, out / "summary.md")
    print(f"\nwrote {out}/summary.json, summary.md, judge_predictions_*.csv")


def write_markdown(summary, path):
    L = ["# LLM-as-judge vs probe (identical held-out rows: hops 1-2, canonical split)\n",
         f"Rows: {summary['n_rows']}\n",
         "| detector | stratum | acc | bal. acc | AUROC | TPR | FPR | n |", "|---|---|---|---|---|---|---|---|"]
    entries = []
    if "probe" in summary:
        entries.append((f"probe (layers {summary['probe'].get('selected_layers')})", summary["probe"]))
    entries += list(summary["models"].items())
    for name, rep in entries:
        for k in ("all", "hop1", "hop2", "natural", "cf"):
            if k in rep:
                m = rep[k]
                L.append(f"| {name} | {k} | {m.get('acc', float('nan')):.3f} | {m.get('bacc', float('nan')):.3f} | "
                         f"{m.get('auroc', float('nan')):.3f} | {m.get('tpr', float('nan')):.3f} | "
                         f"{m.get('fpr', float('nan')):.3f} | {m.get('n')} |")
    L.append("\nFailure recall by type:\n")
    L.append("| detector | hedging fails | confident fails | hop-1 confident fails |\n|---|---|---|---|")
    for name, rep in entries:
        L.append(f"| {name} | " + " | ".join(f"{rep[k]['value']:.3f} (n={rep[k]['n']})" for k in
                                            ("hedge_fail_recall", "confident_fail_recall", "hop1_confident_fail_recall")) + " |")
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
