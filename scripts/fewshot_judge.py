"""
scripts/fewshot_judge.py
────────────────────────
The post-hoc judge with in-context examples: does a worked demonstration close the gap between a zero-shot
judge and the probe?  Same judge code, prompt and P(fail) read-out as the pipeline's judges
(src/pipeline/judge.py); the only change is that the chat starts with demonstration turns.

  demonstrations  from the probe's TRAIN split only (never test / evaluation questions): with --shots 2 (default)
                  one failing hop answered "no" (a confident, non-hedge failure) and one correct hop answered
                  "yes" — a balanced pair, since a single example of one class biases a yes/no judge; --shots 1
                  uses the failing example only.  Chosen deterministically: the shortest contexts, ties by id.
  scoring         hop 1 and hop 2 of the final run's first-stage trajectories (both datasets), as for the
                  pipeline's judges; resumable
  report          live 2Wiki hop AUROC [CI], end-to-end accuracy at θ = 0.5 and accuracy-optimal thresholds,
                  matched restart budgets vs the probe, calibration and cost — next to the probe and the
                  zero-shot judge of the same model

Kaggle: GPU T4 x2 (an 8B judge is split over both GPUs), HF_TOKEN in the environment, the run folder attached.
    python scripts/fewshot_judge.py --run /kaggle/input/<...>/pipeline_..._L12 --out /kaggle/working/fewshot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.pipeline.common import hop_text, is_hedge, load_setting_examples, read_jsonl  # noqa: E402
from src.pipeline.judge import SYSTEM, USER_TMPL, Judge  # noqa: E402

PROBE_DIR = "outputs/pipeline/tfidf/probes_lr_C0.01_L12"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# Demonstrations (train split only)
# ─────────────────────────────────────────────────────────────────────────────

def pick_demos(probe_dir: Path, shots: int) -> list[dict]:
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    train = set(meta["train_ids"])
    cands = {0: [], 1: []}
    for e in load_setting_examples("tfidf"):
        if e["id"] not in train or e.get("is_counterfactual"):
            continue
        for h in e.get("hops", []):
            if h.get("hop_idx") != 2 or h.get("label") not in (0, 1):
                continue
            t = hop_text(e.get("generated_cot", ""), 2)
            if not t or is_hedge(t):                       # a confident step, the hard case for a judge
                continue
            cands[h["label"]].append(dict(id=e["id"], context=e["context"], question=e["question"], hop=t,
                                          answer="no" if h["label"] == 1 else "yes"))
    for y in cands:
        cands[y].sort(key=lambda d: (len(d["context"]), d["id"]))
    demos = [cands[1][0]] + ([cands[0][0]] if shots >= 2 else [])
    return demos


class FewShotJudge(Judge):
    """src.pipeline.judge.Judge with demonstration turns before the question."""

    def __init__(self, model, tok, demos: list[dict], date_string: str | None = None):
        super().__init__(model, tok, date_string)
        self.demos = demos

    def p_fail(self, context: str, question: str, hop: str) -> dict:
        import torch
        msgs = [{"role": "system", "content": SYSTEM}]
        for d in self.demos:
            msgs += [{"role": "user", "content": USER_TMPL.format(context=d["context"], question=d["question"], hop=d["hop"])},
                     {"role": "assistant", "content": d["answer"]}]
        msgs.append({"role": "user", "content": USER_TMPL.format(context=context, question=question, hop=hop)})
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.kw)
        dev = next(self.model.parameters()).device
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.inference_mode():
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(**enc).logits[0, -1].float()
            if dev.type == "cuda":
                torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        diff = torch.logsumexp(logits[self.no], 0) - torch.logsumexp(logits[self.yes], 0)
        return dict(p_fail=float(torch.sigmoid(diff)), ntok=int(enc.input_ids.shape[1]), ms=ms)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def score(out: Path, run: Path, name: str, demos: list[dict]):
    from src.pipeline.evaluate import _merged
    from src.pipeline.generation import load_model
    from src.pipeline.run import prompt_date
    items = read_jsonl(run / "eval_set.jsonl")
    s0 = {r["uid"]: r for r in _merged(run, "s0")}
    path = out / "judge_fewshot.jsonl"
    done = {r["uid"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [it for it in items if it["uid"] in s0 and it["uid"] not in done]
    print(f"[fewshot] {len(todo)} trajectories to score with {name}", flush=True)
    if not todo:
        return
    model, tok = load_model(name, os.environ.get("HF_TOKEN"), spread=True)
    judge = FewShotJudge(model, tok, demos, prompt_date(items[0]["prompt"]) if "llama" in name.lower() else None)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            s = s0[it["uid"]]
            rec = {"uid": it["uid"], "model": name}
            for k in ("1", "2"):
                if k in s["boundaries"] and s["hop_text"][k]:
                    rec[k] = judge.p_fail(it["context"], it["question"], s["hop_text"][k])
            f.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                f.flush()
                print(f"[fewshot] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Report (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def report(out: Path, run: Path, probe_dir: Path, name: str, demos: list[dict], shots: int):
    from eval_robustness import BUDGETS, DATASETS, Run, arr, auroc_ci, budget_threshold, ece, row
    from src.pipeline.evaluate import _paired_ci, _relabel
    from src.pipeline.probes import ProbeBundle
    R = Run(run, force=True)
    _relabel(R.items, R.s0)
    pb = ProbeBundle(probe_dir, mode="shared")
    P = R.probe_scores(pb)
    zs = name.split("/")[-1]                                   # the run's zero-shot judge of the same model
    fs = f"{zs} (few-shot)"
    R.judges[fs] = {r["uid"]: r for r in read_jsonl(out / "judge_fewshot.jsonl")}
    J = {n: R.judge_scores(n) for n in (zs, fs) if n in R.judges}
    by_ds = {ds: [it for it in R.items if it["dataset"] == ds] for ds in DATASETS}
    base = {ds: arr(R.simple("no_intervention"), by_ds[ds]) for ds in DATASETS}
    probe = R.trigger(P, pb.tau)
    per_p, _ = R.acc_optimal(P)
    probe_acc = R.trigger_per_item(P, per_p)
    fs_fit = R.acc_optimal(J[fs])
    sysd = {n: R.judge_system(n) for n in J}
    sysa = {n: R.judge_system(n, R.acc_optimal(J[n])[0]) for n in J}

    ms = [v["ms"] for r in R.judges[fs].values() for k, v in r.items() if k in ("1", "2")]
    nt = [v["ntok"] for r in R.judges[fs].values() for k, v in r.items() if k in ("1", "2")]
    L = [f"# Few-shot judge (`{name}`, {len(demos)} demonstration{'s' * (len(demos) > 1)}) — `{run.name}`", "",
         "Same judge prompt and P(fail) read-out as the pipeline's zero-shot judges; the chat starts with "
         f"demonstration turns taken from the probe's TRAIN split (shortest contexts, confident non-hedge hop-2 "
         f"steps): {', '.join(d['answer'] + ' (' + d['id'][:8] + ')' for d in demos)}. Correct = token F1 ≥ 0.5; "
         "CIs: paired bootstrap over questions.", "",
         f"Cost: {np.mean(ms):.0f} ms and {np.mean(nt):.0f} prompt tokens per check (mean), after the full answer "
         "is generated; the probe: ~9.5 ms at the hop boundary.", "",
         "## Hop-level detection on live 2Wiki hops, AUROC [95% CI]", "",
         "| hop | n | fail | probe | " + " | ".join(f"judge {n}" for n in J) + " |", "|---" * (4 + len(J)) + "|"]
    for k in (1, 2):
        rk = [it for it in by_ds["2wiki"] if it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]]
              and all(k in J[n].get(it["uid"], {}) for n in J)]
        y = [it["labels"][str(k)] for it in rk]
        cells = [auroc_ci(y, [P[it["uid"]][k] for it in rk])] + [auroc_ci(y, [J[n][it["uid"]][k] for it in rk]) for n in J]
        L.append(f"| {k} | {len(rk)} | {sum(y)} | " + " | ".join(f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in cells) + " |")
    yl, pl = {n: [] for n in J}, {n: [] for n in J}
    for it in by_ds["2wiki"]:
        for k in (1, 2):
            if it["labels"].get(str(k)) in (0, 1):
                for n in J:
                    if k in J[n].get(it["uid"], {}):
                        yl[n].append(it["labels"][str(k)]); pl[n].append(J[n][it["uid"]][k])
    L += ["", "Calibration on live 2Wiki hops: " + "; ".join(
        f"judge {n} ECE {ece(yl[n], pl[n])[0]:.3f}, Brier {ece(yl[n], pl[n])[1]:.3f}" for n in J) +
        " (probe: ECE 0.024, Brier 0.080).", ""]

    head = ["| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |", "|---|---|---|---|---|---|"]
    L += ["## End to end (one restart, same cached generations)", "",
          f"Few-shot judge accuracy-optimal (θ1, θ2): {fs_fit[1]}.", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        ref, ref_a = arr(probe, its), arr(probe_acc, its)
        L += [f"### {ds} (n = {len(its)})", "", "The judges' own decision (θ = 0.5) vs the probe's train-CV τ:", ""] + head
        L.append(row("**Probe trigger (ours)**", probe, its, base[ds]))
        for n in J:
            L.append(row(f"Judge {n}, θ = 0.5", sysd[n], its, base[ds], ref))
        L += ["", "Accuracy-optimal thresholds for every trigger:", ""] + head
        L.append(row("**Probe, accuracy-optimal**", probe_acc, its, base[ds]))
        for n in J:
            L.append(row(f"Judge {n}, accuracy-optimal", sysa[n], its, base[ds], ref_a))
        L += ["", "Matched restart budget — accuracy (probe − judge [95% CI]):", "",
              "| budget | probe | " + " | ".join(f"judge {n}" for n in J) + " |", "|---" * (2 + len(J)) + "|"]
        el = [R.eligible(it["uid"]) for it in its]
        for b in BUDGETS:
            tp = budget_threshold([P[it["uid"]] for it in its], el, b)
            cp = arr(R.trigger(P, {1: tp, 2: tp}), its)
            cells = []
            for n in J:
                tj = budget_threshold([J[n].get(it["uid"], {}) for it in its], el, b)
                cj = arr(R.trigger(J[n], {1: tj, 2: tj}), its)
                lo, hi = _paired_ci(cp, cj)
                cells.append(f"{cj.mean():.3f} ({cp.mean() - cj.mean():+.3f} [{lo:+.3f}, {hi:+.3f}])")
            L.append(f"| {int(b * 100)}% | **{cp.mean():.3f}** | " + " | ".join(cells) + " |")
        L.append("")
    (out / "fewshot_judge.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the final run's folder (eval_set.jsonl, s0.jsonl, judge__*.jsonl, ...)")
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/fewshot_judge")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shots", type=int, default=2, choices=[1, 2],
                    help="2 = one failing + one correct demonstration (default); 1 = the failing one only")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    ap.add_argument("--stage", choices=["score", "report", "all"], default="all")
    args = ap.parse_args()
    out, run, probe_dir = Path(args.out), Path(args.run), ROOT / args.probe_dir
    out.mkdir(parents=True, exist_ok=True)
    demos = pick_demos(probe_dir, args.shots)
    (out / "demos.json").write_text(json.dumps(demos, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[fewshot] demonstrations: " + "; ".join(f"{d['answer']} <- {d['hop']!r}" for d in demos), flush=True)
    if args.stage in ("score", "all"):
        score(out, run, args.model, demos)
    if args.stage in ("report", "all"):
        report(out, run, probe_dir, args.model, demos, args.shots)


if __name__ == "__main__":
    main()
