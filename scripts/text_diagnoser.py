"""
scripts/text_diagnoser.py
─────────────────────────
A TRAINED text diagnoser: the fair stand-in for Doctor-RAG's fine-tuned diagnoser.  Same rows, same labels and
same split as the probe; the only difference is that it reads text (context, question, hop) instead of the hidden
state.  Answers "would a model trained on our labels see what the hidden state sees?"

  model      cross-encoder (default microsoft/deberta-v3-base), one logit, BCE with balanced class weight (as the
             probe's class_weight="balanced"); input A = question + earlier step + the step to check,
             input B = the context (the longer side is truncated, in practice the context), max 512 tokens
  selection  10 % of the TRAIN questions (grouped) held out: best epoch by validation AUROC, τ = F0.5-optimal on
             validation (the probe's τ also comes from the train split only); the test split is never used to choose
  test       hop-level AUROC [grouped bootstrap CI], precision / recall / F0.5 / FPR at τ, paired AUROC difference
             vs the probe on the identical test rows, by hop, hedge vs confident failures, latency per check
  e2e        (--run) the diagnoser as the restart trigger on the final run: deployable τ, accuracy-optimal
             thresholds, matched restart budget, live 2Wiki hop AUROC — same machinery as eval_robustness.py

One Kaggle GPU session (T4), ~30-60 min; HF_TOKEN in the environment; the run folder attached for --run.
Usage:
    python scripts/text_diagnoser.py --out /kaggle/working/diagnoser --run /kaggle/input/<...>/pipeline_..._L12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.pipeline.common import (best_f05_threshold, detection_metrics, hop_text, load_setting_examples,  # noqa: E402
                                 read_jsonl)
from src.pipeline.probes import _auroc, _bootstrap_auroc_ci, hop_rows  # noqa: E402

PROBE_DIR = "outputs/pipeline/tfidf/probes_lr_C0.01_L12"
MODEL = "microsoft/deberta-v3-base"
MAX_LEN, EPOCHS, LR, BATCH, ACCUM, WARMUP, VAL_FRAC, SEED = 512, 3, 2e-5, 8, 2, 0.1, 0.1, 42


def make_input(question: str, hops: dict, k: int, context: str) -> tuple[str, str]:
    """(A, B) for the cross-encoder. hops: {"1": text, "2": text}; k = the step being checked."""
    a = f"Question: {question}"
    if k == 2 and hops.get("1"):
        a += f"\nStep 1: {hops['1']}"
    a += f"\nStep {k} (check this step): {hops.get(str(k)) or ''}"
    return a, context


def paired_auroc_diff(y, a, b, groups, n_boot: int = 2000, seed: int = 0):
    """AUROC(a) − AUROC(b) with a grouped bootstrap CI."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(set(y[idx].tolist())) > 1:
            vals.append(_auroc(y[idx], a[idx]) - _auroc(y[idx], b[idx]))
    return _auroc(y, a) - _auroc(y, b), np.quantile(vals, [0.025, 0.975])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class Diagnoser:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(self.dev)

    def encode(self, pairs):
        # longest_first trims the longer side (the context) and never fails if A alone is long
        return self.tok([a for a, _ in pairs], [b for _, b in pairs], truncation="longest_first",
                        max_length=MAX_LEN, padding=True, return_tensors="pt")

    def logits(self, pairs, batch: int = 32) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(pairs), batch):
                enc = self.encode(pairs[s:s + batch]).to(self.dev)
                with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
                    out.append(self.model(**enc).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, tr_pairs, y_tr, va_pairs, y_va):
        """Returns per-epoch validation AUROCs; keeps the best epoch's weights."""
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], device=self.dev)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * int(np.ceil(len(tr_pairs) / (BATCH * ACCUM)))
        sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=self.dev == "cuda")
        best, best_state, hist = -1.0, None, []
        order = list(range(len(tr_pairs)))
        for ep in range(EPOCHS):
            self.model.train()
            random.Random(SEED + ep).shuffle(order)
            t0, n_bad = time.time(), 0
            for i, s in enumerate(range(0, len(order), BATCH)):
                idx = order[s:s + BATCH]
                enc = self.encode([tr_pairs[j] for j in idx]).to(self.dev)
                yb = torch.tensor(y_tr[idx], dtype=torch.float32, device=self.dev)
                with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
                    logit = self.model(**enc).logits[:, 0]
                loss = loss_fn(logit.float(), yb) / ACCUM
                if not torch.isfinite(loss):
                    n_bad += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                if (i + 1) % ACCUM == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                if i % 200 == 0:
                    print(f"  epoch {ep} batch {i}/{len(order) // BATCH} loss {loss.item() * ACCUM:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            a = _auroc(y_va, self.logits(va_pairs))
            hist.append(a)
            print(f"[diagnoser] epoch {ep}: validation AUROC {a:.4f} (non-finite losses skipped: {n_bad})", flush=True)
            if a > best:
                best = a
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return hist


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 data: train / validation / test, aligned with the probe
# ─────────────────────────────────────────────────────────────────────────────

def phase1_rows(probe_dir: Path):
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    split = {i: "train" for i in meta["train_ids"]} | {i: "test" for i in meta["test_ids"]}
    examples = [e for e in load_setting_examples("tfidf") if e["id"] in split]
    rows = [r for r in hop_rows(examples, range(len(examples))) if not r["cf"]]
    for r in rows:
        e = examples[r["ei"]]
        hops = {k: hop_text(e.get("generated_cot", ""), int(k)) for k in ("1", "2")}
        r["pair"] = make_input(e["question"], hops, r["hop"], e["context"])
        r["split"] = split[r["id"]]
    # validation: 10 % of the train question groups
    groups = sorted({r["group"] for r in rows if r["split"] == "train"})
    random.Random(SEED).shuffle(groups)
    val_groups = set(groups[:int(VAL_FRAC * len(groups))])
    for r in rows:
        if r["split"] == "train" and r["group"] in val_groups:
            r["split"] = "val"
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/pipeline/tfidf/text_diagnoser")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--run", default=None, help="finished Phase 3-5 run folder: also evaluate as the restart trigger")
    args = ap.parse_args()
    out, probe_dir = Path(args.out), ROOT / args.probe_dir
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    rows, meta = phase1_rows(probe_dir)
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    y = {s: np.array([r["y"] for r in sp[s]]) for s in sp}
    print(f"[diagnoser] rows: " + ", ".join(f"{s} {len(v)} (fail {y[s].mean():.2f})" for s, v in sp.items()), flush=True)

    D = Diagnoser(args.model)
    hist = D.fit([r["pair"] for r in sp["train"]], y["train"], [r["pair"] for r in sp["val"]], y["val"])
    sig = lambda z: 1 / (1 + np.exp(-z))                                   # noqa: E731
    p_val = sig(D.logits([r["pair"] for r in sp["val"]]))
    tau = best_f05_threshold(y["val"], p_val)[0]
    p_te = sig(D.logits([r["pair"] for r in sp["test"]]))
    with open(out / "diagnoser_test.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(sp["test"], p_te):
            f.write(json.dumps(dict(id=r["id"], hop=r["hop"], y=r["y"], p_fail=float(p))) + "\n")

    # the probe on the identical test rows (the shared probe's stored test predictions)
    probe = {(r["id"], r["hop"]): r["p_fail"] for r in read_jsonl(probe_dir / "test_predictions.jsonl")
             if r["task"] == "hop12"}
    keep = np.array([(r["id"], r["hop"]) in probe for r in sp["test"]])
    te = [r for r, k in zip(sp["test"], keep) if k]
    yt, pt = y["test"][keep], p_te[keep]
    pp = np.array([probe[(r["id"], r["hop"])] for r in te])
    gt = np.array([r["group"] for r in te])
    hop = np.array([r["hop"] for r in te])
    hedge = np.array([r["hedge"] for r in te])
    ptau = meta["probes"]["hop12"]["tau"]

    # latency of one check (batch 1, the trigger's situation)
    D.model.eval()
    t0 = time.time()
    for r in te[:200]:
        D.logits([r["pair"]], batch=1)
    ms = (time.time() - t0) / min(200, len(te)) * 1000
    n_tok = np.mean([len(D.encode([r["pair"]])["input_ids"][0]) for r in te[:200]])

    L = [f"# Trained text diagnoser vs the probe (`{args.model}`)", "",
         f"Same TF-IDF Phase 1 rows, labels and split as the probe (counterfactuals excluded). Train {len(sp['train'])} / "
         f"validation {len(sp['val'])} (10 % of the train question groups) / test {len(te)} hop rows. Input: question, "
         f"step 1 (when checking step 2), the step to check, and the context (truncated first) — max {MAX_LEN} tokens "
         f"(mean {n_tok:.0f} used). {EPOCHS} epochs, lr {LR}, batch {BATCH}×{ACCUM}, balanced BCE; best epoch by "
         f"validation AUROC ({', '.join(f'{a:.4f}' for a in hist)}); τ = {tau:.3f} (F0.5-optimal on validation). "
         f"Probe: the deployed shared probe (τ = {ptau:.3f}), stored test predictions on the identical rows. CIs: "
         f"bootstrap over question groups.", "",
         "## Hop-failure detection, test split", "",
         "| rows | n | fail | detector | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − diagnoser, AUROC [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, m in (("hops 1–2", np.ones(len(te), bool)), ("hop 1", hop == 1), ("hop 2", hop == 2)):
        dif, (lo, hi) = paired_auroc_diff(yt[m], pp[m], pt[m], gt[m])
        for name, s, th in (("**probe**", pp, ptau), ("trained text diagnoser", pt, tau)):
            dm = detection_metrics(yt[m], s[m] >= th)
            c = _bootstrap_auroc_ci(yt[m], s[m], gt[m])
            d = f"{dif:+.3f} [{lo:+.3f}, {hi:+.3f}]" if name.startswith("trained") else ""
            L.append(f"| {lab} | {m.sum()} | {int(yt[m].sum())} | {name} | {_auroc(yt[m], s[m]):.3f} [{c[0]:.3f}, {c[1]:.3f}] | "
                     f"{dm['precision']:.3f} | {dm['recall']:.3f} | {dm['f05']:.3f} | {dm['fpr']:.3f} | {d} |")
    L += ["", "Recall on failing hops by kind (at each detector's τ):", "",
          "| failure kind | n | probe | diagnoser |", "|---|---|---|---|"]
    for lab, m in (("hedge", (yt == 1) & hedge), ("confident (no hedge)", (yt == 1) & ~hedge)):
        L.append(f"| {lab} | {m.sum()} | {(pp[m] >= ptau).mean():.3f} | {(pt[m] >= tau).mean():.3f} |")
    both = np.corrcoef(np.argsort(np.argsort(pp)), np.argsort(np.argsort(pt)))[0, 1]
    L += ["", f"Spearman ρ between probe and diagnoser scores: {both:+.2f}. Cost of one check: diagnoser {ms:.1f} ms "
              f"(batch 1, {n_tok:.0f} tokens, a separate {args.model.split('/')[-1]} forward pass) vs the probe's one dot "
              f"product on states the generator already computed (pipeline: 9.5 ms incl. pooling).", ""]

    if args.run:
        L += e2e(Path(args.run), probe_dir, D, tau, sig)
    (out / "text_diagnoser.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ─────────────────────────────────────────────────────────────────────────────
# End to end on the final run
# ─────────────────────────────────────────────────────────────────────────────

def e2e(run: Path, probe_dir: Path, D: Diagnoser, tau: float, sig) -> list[str]:
    from eval_robustness import BUDGETS, DATASETS, MAIN_JUDGE, Run, arr, auroc_ci, budget_threshold, row
    from src.pipeline.evaluate import _paired_ci, _relabel
    from src.pipeline.probes import ProbeBundle
    R = Run(run, force=True)
    _relabel(R.items, R.s0)
    pb = ProbeBundle(probe_dir, mode="shared")
    P = R.probe_scores(pb)
    pairs, keys = [], []
    for it in R.items:
        u, s0 = it["uid"], R.s0[it["uid"]]
        for k in sorted(R.eligible(u)):
            pairs.append(make_input(it["question"], s0["hop_text"], k, it["context"]))
            keys.append((u, k))
    z = sig(D.logits(pairs))
    T = {it["uid"]: {} for it in R.items}
    for (u, k), p in zip(keys, z):
        T[u][k] = float(p)
    by_ds = {ds: [it for it in R.items if it["dataset"] == ds] for ds in DATASETS}
    base = {ds: arr(R.simple("no_intervention"), by_ds[ds]) for ds in DATASETS}
    probe, diag = R.trigger(P, pb.tau), R.trigger(T, {1: tau, 2: tau})
    per_p, _ = R.acc_optimal(P)
    per_d, fit_d = R.acc_optimal(T)
    probe_acc, diag_acc = R.trigger_per_item(P, per_p), R.trigger_per_item(T, per_d)
    judge = R.judge_system(MAIN_JUDGE)
    head = ["| system | accuracy | restart rate | fixed | broken | probe − system [95% CI] |", "|---|---|---|---|---|---|"]
    L = [f"## End to end on `{run.name}` (one restart, same cached generations)", "",
         f"Diagnoser accuracy-optimal (θ1, θ2): {fit_d}.", ""]
    for ds in DATASETS:
        its = by_ds[ds]
        ref, ref_acc = arr(probe, its), arr(probe_acc, its)
        L += [f"### {ds} (n = {len(its)})", "", "Deployable thresholds (fitted on training data only):", ""] + head
        L += [row("**Probe trigger (ours)**", probe, its, base[ds]),
              row(f"Trained text diagnoser (τ = {tau:.3f})", diag, its, base[ds], ref),
              row(f"Judge {MAIN_JUDGE}, θ = 0.5", judge, its, base[ds], ref),
              row("Always restart after hop 2", R.always_hop(2), its, base[ds], ref), "",
              "Accuracy-optimal thresholds (2-fold on 2Wiki; HotpotQA uses all 2Wiki):", ""] + head
        L += [row("**Probe, accuracy-optimal**", probe_acc, its, base[ds]),
              row("Trained text diagnoser, accuracy-optimal", diag_acc, its, base[ds], ref_acc), "",
              "Matched restart budget — accuracy, probe vs diagnoser (probe − diagnoser [95% CI]):", "",
              "| budget | probe | diagnoser | difference |", "|---|---|---|---|"]
        el = [R.eligible(it["uid"]) for it in its]
        for b in BUDGETS:
            tp = budget_threshold([P[it["uid"]] for it in its], el, b)
            td = budget_threshold([T[it["uid"]] for it in its], el, b)
            cp, cd = arr(R.trigger(P, {1: tp, 2: tp}), its), arr(R.trigger(T, {1: td, 2: td}), its)
            lo, hi = _paired_ci(cp, cd)
            L.append(f"| {int(b * 100)}% | {cp.mean():.3f} | {cd.mean():.3f} | {cp.mean() - cd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        L.append("")
    L += ["### Live 2Wiki hops, AUROC [95% CI]", "", "| hop | n | fail | probe | diagnoser |", "|---|---|---|---|---|"]
    for k in (1, 2):
        rk = [it for it in by_ds["2wiki"] if it["labels"].get(str(k)) in (0, 1) and k in P[it["uid"]] and k in T[it["uid"]]]
        yk = [it["labels"][str(k)] for it in rk]
        a, b = auroc_ci(yk, [P[it["uid"]][k] for it in rk]), auroc_ci(yk, [T[it["uid"]][k] for it in rk])
        L.append(f"| {k} | {len(rk)} | {sum(yk)} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] |")
    L.append("")
    return L


if __name__ == "__main__":
    main()
