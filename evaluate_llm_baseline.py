"""
evaluate_llm_baseline.py  (v2)
──────────────────────────────
Text-only "LLM-as-judge" baseline for hop-failure detection, scored on EXACTLY
the same held-out rows as the hidden-state probe (lf_common: canonical split,
hops 1-2, label 0/1).

What changed vs v1 (see FIXES_AND_RERUN.md):
  * v1 judged every generated hop, including the model's extra hops 3..18 that
    the old labeler marked "failure" by default (64 % of all failure labels).
    v2 judges hops 1-2 only.
  * v1 ran on the later 115-CF data file and a different split than the probe.
    v2 uses the canonical file and split.
  * v2 reads P(no) vs P(yes) from the next-token logits (one forward pass, no
    sampling), so the judge gets an AUROC directly comparable to the probe.
  * Reports TPR with FPR, per stratum (hop 1 / hop 2 / natural / CF, hedging vs
    confident failures).  If --probe-predictions is given, the probe is scored
    on the identical rows next to each judge.

The prompt is unchanged from v1.  For counterfactual rows the hop text is
usually a TRUE statement ("X is not mentioned in the context") that our label
calls a failure; read the CF/hedging rows with that in mind.

Robustness (so a long Kaggle run cannot be lost late):
  * one result file per model (model_<name>.json); finished models are skipped
    on re-run, a crashed model never affects the others;
  * within a model, progress is checkpointed every 100 rows and resumed;
  * per-row errors (e.g. a CUDA OOM on one long prompt) are retried, then
    recorded as missing rather than aborting;
  * gated models fall back to an ungated mirror with the same weights;
  * disk / GPU-memory checks skip a model that cannot fit instead of crashing;
  * summary.json / summary.md are rewritten after every model.

Usage (GPU; Llama models need an HF token with Llama access, or the mirror is used)
─────
  python evaluate_llm_baseline.py --data data/canonical/augmented_v2.jsonl \
      --models meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen2.5-7B-Instruct \
      --probe-predictions outputs/probing/test_predictions.csv --out outputs/llm_judge
  python evaluate_llm_baseline.py --out outputs/llm_judge --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from lf_common import DEFAULT_DATA, binary_metrics, canonical_split, fmt_metrics, hop_rows, load_examples
from lf_models import (approx_download_gb, cleanup_gpu, free_disk_gb, free_model_cache, gpu_total_gb,
                       input_device, load_causal_lm, resolve_model)

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


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "__")


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
        print(f"  [warn] {n_bad} rows had no valid prediction and are excluded from this model's metrics")
    rows = [r for r, k in zip(rows, ok) if k]; prob = prob[ok]; pred = pred[ok]
    y = np.array([r["y"] for r in rows]); hop = np.array([r["hop"] for r in rows])
    cf = np.array([r["cf"] for r in rows]); hedge = np.array([r["hedge"] for r in rows])
    masks = {"all": np.ones(len(y), bool), "hop1": hop == 1, "hop2": hop == 2,
             "natural": ~cf, "cf": cf, "hop1_natural": (hop == 1) & ~cf}
    rep = {k: binary_metrics(y[m], prob[m], pred[m]) for k, m in masks.items() if m.sum()}
    rep["n_missing_excluded"] = n_bad
    f = y == 1
    for name, m in {"hedge_fail_recall": f & hedge, "confident_fail_recall": f & ~hedge,
                    "hop1_confident_fail_recall": f & ~hedge & (hop == 1)}.items():
        rep[name] = {"value": float(pred[m].mean()) if m.sum() else float("nan"), "n": int(m.sum())}
    return rep


def load_probe_predictions(path):
    """Map (id, hop) -> probe probability using the hop-specific probe at the selected layers."""
    rj = Path(path).with_name("results.json")
    sel = json.loads(rj.read_text())["selected_layers"] if rj.exists() else None
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


# ─────────────────────────────────────────────────────────────────────────────
# One model
# ─────────────────────────────────────────────────────────────────────────────

def judge_one_model(repo, rows, id2ex, out, tag, hf_token, dtype):
    """Judge all rows with `repo`; checkpoint to out/partial_<tag>.jsonl; returns the report dict."""
    model, tok = load_causal_lm(repo, dtype, token=hf_token)
    try:
        yes_ids = answer_token_ids(tok, ["yes", "Yes", "YES"])
        no_ids = answer_token_ids(tok, ["no", "No", "NO"])
        dev = input_device(model)
        partial = out / f"partial_{tag}.jsonl"
        recs = []
        if partial.exists():
            recs = [json.loads(l) for l in open(partial, encoding="utf-8") if l.strip()]
            print(f"  resuming from {len(recs)}/{len(rows)} rows")
        pf = open(partial, "a", encoding="utf-8")
        t_start = time.time()
        n_err = 0
        for i in range(len(recs), len(rows)):
            r = rows[i]; ex = id2ex[r["id"]]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(context=context_str(ex.get("context")),
                                                                 question=ex["question"], hop=r["text"])}]
            rec = {"p_fail": float("nan"), "pred": -1, "top": "<error>", "lat": float("nan"), "ntok": 0}
            for attempt in range(2):
                try:
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        logits = model(**enc).logits[0, -1].float()
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    lat = (time.time() - t0) * 1000
                    l_yes = torch.logsumexp(logits[yes_ids], 0); l_no = torch.logsumexp(logits[no_ids], 0)
                    if torch.isfinite(l_no - l_yes):
                        rec = {"p_fail": torch.sigmoid(l_no - l_yes).item(), "pred": int(l_no > l_yes),
                               "top": tok.decode([int(logits.argmax())]), "lat": lat, "ntok": int(enc.input_ids.shape[1])}
                    else:
                        rec["top"] = "<nonfinite>"
                    break
                except Exception as e:                     # OOM etc.: free memory, retry once, then record missing
                    cleanup_gpu()
                    if attempt == 1:
                        n_err += 1
                        print(f"  [row {i}] failed twice ({e.__class__.__name__}: {str(e)[:80]}) -> recorded as missing")
            recs.append(rec)
            pf.write(json.dumps(rec) + "\n")
            if i < 8:
                print(f"  [{i}] top={rec['top']!r:8} p_fail={rec['p_fail']:.3f} label={r['y']} hop{r['hop']} :: {r['text'][:80]}")
            if (i + 1) % 100 == 0:
                pf.flush()
                print(f"  {i + 1}/{len(rows)}  ({time.time() - t_start:.0f}s)", flush=True)
        pf.close()
        if len(rows) and sum(np.isnan(x["p_fail"]) for x in recs) > 0.10 * len(rows):
            raise RuntimeError(f"more than 10% of rows produced no valid prediction ({n_err} hard errors)")
        return recs
    finally:
        del model, tok
        cleanup_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (also runnable alone: --aggregate-only)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(out: Path):
    summary = {"models": {}, "failed": {}}
    pj = out / "probe_summary.json"
    if pj.exists():
        summary["probe"] = json.loads(pj.read_text())
    meta = out / "meta.json"
    if meta.exists():
        summary.update(json.loads(meta.read_text()))
    for p in sorted(out.glob("model_*.json")):
        d = json.loads(p.read_text())
        summary["models"][d["requested_model"]] = d
    for p in sorted(out.glob("failed_*.json")):
        d = json.loads(p.read_text())
        summary["failed"][d["requested_model"]] = d["reason"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_markdown(summary, out / "summary.md")
    return summary


def write_markdown(summary, path):
    L = ["# LLM-as-judge vs probe (identical held-out rows: hops 1-2, canonical split)\n",
         f"Rows: {summary.get('n_rows', '?')}\n",
         "| detector | stratum | acc | bal. acc | AUROC | TPR | FPR | n |", "|---|---|---|---|---|---|---|---|"]
    entries = []
    if "probe" in summary:
        entries.append((f"probe (layers {summary['probe'].get('selected_layers')})", summary["probe"]))
    for name, d in summary["models"].items():
        label = name if d.get("used_repo") == name else f"{name} (via {d.get('used_repo')})"
        entries.append((label, d["report"]))
    nan = float("nan")
    for name, rep in entries:
        for k in ("all", "hop1", "hop2", "natural", "cf"):
            if k in rep:
                m = rep[k]
                L.append(f"| {name} | {k} | {m.get('acc', nan):.3f} | {m.get('bacc', nan):.3f} | "
                         f"{m.get('auroc', nan):.3f} | {m.get('tpr', nan):.3f} | {m.get('fpr', nan):.3f} | {m.get('n')} |")
    L.append("\nFailure recall by type:\n")
    L.append("| detector | hedging fails | confident fails | hop-1 confident fails |\n|---|---|---|---|")
    for name, rep in entries:
        L.append(f"| {name} | " + " | ".join(f"{rep[k]['value']:.3f} (n={rep[k]['n']})" for k in
                                            ("hedge_fail_recall", "confident_fail_recall", "hop1_confident_fail_recall")) + " |")
    if summary.get("failed"):
        L.append("\n## Models that did NOT run\n")
        for k, v in summary["failed"].items():
            L.append(f"- **{k}**: {v}")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "--labels", dest="data", default=DEFAULT_DATA)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="outputs/llm_judge")
    ap.add_argument("--hops", default="1,2")
    ap.add_argument("--max-items", type=int, default=None, help="quick test on the first N rows")
    ap.add_argument("--probe-predictions", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--force", action="store_true", help="redo models that already have a result file")
    ap.add_argument("--free-disk", action="store_true", help="delete each model's download after use")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--hs-dir", default=None, help="ignored (kept for old notebook calls)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        s = aggregate(out)
        print(f"aggregated {len(s['models'])} model(s), {len(s['failed'])} failed -> {out}/summary.md")
        return

    examples = load_examples(args.data)
    _, te_idx = canonical_split(examples)
    test_ex = [examples[i] for i in te_idx]
    id2ex = {x["id"]: x for x in test_ex}
    hops = tuple(int(h) for h in args.hops.split(","))
    rows = hop_rows(test_ex, hops=hops, require_text=True)
    if args.max_items:
        rows = rows[:args.max_items]
    y = np.array([r["y"] for r in rows])
    print(f"test examples={len(test_ex)}  rows judged={len(rows)}  failures={y.sum()} ({100 * y.mean():.1f}%)  "
          f"CF rows={sum(r['cf'] for r in rows)}")
    (out / "meta.json").write_text(json.dumps({"n_rows": len(rows)}))

    if args.probe_predictions and Path(args.probe_predictions).exists():
        pp, sel = load_probe_predictions(args.probe_predictions)
        if all((r["id"], r["hop"]) in pp for r in rows):
            pr = np.array([pp[(r["id"], r["hop"])] for r in rows])
            ps = {"selected_layers": sel, **strata_report(rows, pr, (pr >= 0.5).astype(int))}
            (out / "probe_summary.json").write_text(json.dumps(ps, indent=1))
            print(f"\nPROBE on identical rows (layers {sel}): {fmt_metrics(ps['all'])}")
        else:
            print("[warn] probe predictions do not cover all judged rows - skipping probe comparison")

    for requested in [m.strip() for m in args.models.split(",") if m.strip()]:
        tag = safe_name(requested)
        done_p, fail_p = out / f"model_{tag}.json", out / f"failed_{tag}.json"
        print(f"\n{'=' * 80}\n{requested}")
        if done_p.exists() and not args.force:
            print("  already done - skipping (use --force to redo)")
            continue
        fail_p.unlink(missing_ok=True)

        def fail(reason):
            print(f"  [SKIP] {reason}")
            fail_p.write_text(json.dumps({"requested_model": requested, "reason": reason}))
            aggregate(out)

        repo, msg = resolve_model(requested, args.hf_token)
        print(f"  {msg}")
        if repo is None:
            fail(msg); continue
        need_disk = approx_download_gb(repo)
        if free_disk_gb() < need_disk + 3:
            fail(f"not enough free disk ({free_disk_gb():.0f} GB free, ~{need_disk} GB needed)"); continue
        if torch.cuda.is_available():
            need_gpu = need_disk * 0.95            # fp16 weights ~ download size
            if gpu_total_gb() < need_gpu * 1.1:
                fail(f"not enough GPU memory ({gpu_total_gb():.0f} GB total, ~{need_gpu:.0f} GB needed). "
                     "Choose 'GPU T4 x2' in the notebook settings."); continue
        t0 = time.time()
        try:
            recs = judge_one_model(repo, rows, id2ex, out, tag, args.hf_token, args.dtype)
        except Exception as e:
            traceback.print_exc()
            fail(f"{e.__class__.__name__}: {str(e)[:300]}")
            cleanup_gpu()
            if args.free_disk:
                free_model_cache(repo)
            continue

        p_fail = [x["p_fail"] for x in recs]; preds = [x["pred"] for x in recs]
        rep = strata_report(rows, p_fail, preds)
        lat = [x["lat"] for x in recs if np.isfinite(x["lat"])]; ntok = [x["ntok"] for x in recs if x["ntok"]]
        rep["latency_ms_mean"] = float(np.mean(lat)) if lat else float("nan")
        rep["prompt_tokens_mean"] = float(np.mean(ntok)) if ntok else float("nan")
        rep["top_token_is_yes_or_no"] = float(np.mean([x["top"].strip().lower() in ("yes", "no") for x in recs]))
        print(f"\nRESULTS {requested}  (top token was yes/no for {100 * rep['top_token_is_yes_or_no']:.1f}% of rows, "
              f"{time.time() - t0:.0f}s)")
        for k in ("all", "hop1", "hop2", "natural", "cf", "hop1_natural"):
            if k in rep:
                print(f"  {k:13s} {fmt_metrics(rep[k])}")
        for k in ("hedge_fail_recall", "confident_fail_recall", "hop1_confident_fail_recall"):
            print(f"  {k:27s} {rep[k]['value']:.3f} (n={rep[k]['n']})")
        print(f"  latency {rep['latency_ms_mean']:.0f} ms/hop, {rep['prompt_tokens_mean']:.0f} prompt tokens/hop")

        with open(out / f"judge_predictions_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "hop_idx", "label", "is_cf", "is_hedge", "p_fail", "pred", "top_token", "latency_ms"])
            for r, x in zip(rows, recs):
                w.writerow([r["id"], r["hop"], r["y"], int(r["cf"]), int(r["hedge"]),
                            f"{x['p_fail']:.5f}", x["pred"], x["top"], f"{x['lat']:.1f}"])
        done_p.write_text(json.dumps({"requested_model": requested, "used_repo": repo, "report": rep}, indent=1))
        (out / f"partial_{tag}.jsonl").unlink(missing_ok=True)
        if args.free_disk:
            free_model_cache(repo)
        aggregate(out)

    s = aggregate(out)
    print(f"\nDONE: {len(s['models'])} model(s) finished, {len(s['failed'])} skipped/failed -> {out}/summary.md")


if __name__ == "__main__":
    main()
