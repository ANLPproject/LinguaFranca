"""
Is the probe needed?  Compare the probe trigger with trivial triggers, all on the SAME cached generations
(same model, prompt, repair and restart outputs; only the trigger differs), one restart.

  always_hop2   restart after </hop2> whenever hop 2 closes
  always_hop1   restart after </hop1> whenever hop 1 closes
  hedge         restart at the first hop whose text hedges ("not mentioned", "no information", ...)
  random        restart at hop k with the probe's per-hop restart rate (mean of 200 seeds)
  text_lr       a TRAINED text trigger: TF-IDF + logistic regression on the hop text, trained on the same
                train-split hop labels as the probe, threshold = F0.5-optimal on out-of-fold train predictions

Usage:
    python scripts/trigger_baselines.py --run pipeline_results/<run> \
        --probe-dir outputs/pipeline/tfidf/probes_lr_C0.01_L12 --out outputs/pipeline/<dir>/trigger_baselines.md
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.pipeline.common import (best_f05_threshold, canonical_split, group_id, hidden_state_ids,  # noqa: E402
                                 hop_text, is_hedge, load_setting_examples, read_jsonl)
from src.pipeline.probes import ProbeBundle  # noqa: E402
from src.pipeline.systems import outcome  # noqa: E402


def safe(uid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def paired_ci(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def train_text_trigger(setting: str):
    """TF-IDF + LR on hop texts of the probe's TRAIN split (hops 1-2), threshold by out-of-fold F0.5."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    ex = load_setting_examples(setting, hidden_state_ids(setting))
    tr, _ = canonical_split(ex)
    X, y, g = [], [], []
    for i in tr:
        e = ex[i]
        for h in e["hops"]:
            if h["hop_idx"] in (1, 2) and h.get("label") in (0, 1) and h.get("text"):
                X.append(f"hop{h['hop_idx']} {h['text']}"); y.append(h["label"]); g.append(group_id(e))
    X, y, g = np.array(X, dtype=object), np.array(y), np.array(g)
    make = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),  # noqa: E731
                                 LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, g):
        oof[b] = make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    tau, _ = best_f05_threshold(y, oof)
    model = make().fit(X, y)
    return model, tau, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    items = [it for it in read_jsonl(run / "eval_set.jsonl")]
    s0 = {r["uid"]: r for r in read_jsonl(run / "s0.jsonl")}
    rs = defaultdict(dict)
    for r in read_jsonl(run / "restarts.jsonl"):
        rs[r["uid"]][r["key"]] = r
    for f in read_jsonl(run / "forced.jsonl"):                     # same answer forcing as evaluate.py
        rec = s0.get(f["uid"]) if f["key"] == "s0" else rs[f["uid"]].get(f["key"])
        if rec is not None and not rec["answer"]:
            rec["answer"] = f["answer"]
            rec["n_gen"] += f["n_gen"]
    items = [it for it in items if it["uid"] in s0 and "q" in rs[it["uid"]]]
    meta = json.loads((Path(args.probe_dir) / "probes_meta.json").read_text())
    pb = ProbeBundle(args.probe_dir, mode="shared")

    def closed(uid):
        return {k for k in (1, 2) if str(k) in s0[uid]["boundaries"] and s0[uid]["hop_text"].get(str(k))
                and f"h{k}" in rs[uid]}

    probe = {}
    for it in items:
        z = np.load(run / "s0_feats" / f"{safe(it['uid'])}.npz")
        probe[it["uid"]] = {k: pb.p_fail(k, z[f"h{k}"]) for k in (1, 2) if f"h{k}" in z.files}
    text_model, text_tau, n_text = train_text_trigger(meta["setting"])
    text = {}
    for it in items:
        u = it["uid"]
        text[u] = {k: float(text_model.predict_proba([f"hop{k} {s0[u]['hop_text'][str(k)]}"])[0, 1])
                   for k in closed(u)}

    def run_trigger(scores_of, tau):
        return {it["uid"]: outcome("probe_restart", it, s0[it["uid"]], rs[it["uid"]],
                                   probe_scores=scores_of(it["uid"]), tau=tau) for it in items}

    one = {1: 0.5, 2: 0.5}
    systems = {
        "No intervention": {it["uid"]: outcome("no_intervention", it, s0[it["uid"]], rs[it["uid"]]) for it in items},
        "**Probe trigger (ours)**": run_trigger(lambda u: probe[u], pb.tau),
        "Always restart after hop 2": run_trigger(lambda u: {2: 1.0} if 2 in closed(u) else {}, one),
        "Always restart after hop 1": run_trigger(lambda u: {1: 1.0} if 1 in closed(u) else {}, one),
        "Hedge-keyword trigger": run_trigger(
            lambda u: {k: float(is_hedge(s0[u]["hop_text"][str(k)])) for k in closed(u)}, one),
        f"Trained text trigger (TF-IDF+LR on hop text, τ={text_tau:.2f})": run_trigger(
            lambda u: text[u], {1: text_tau, 2: text_tau}),
    }
    # random trigger at the probe's per-hop restart rate, per dataset
    rnd = {}
    for ds in ("2wiki", "hotpotqa"):
        its = [it for it in items if it["dataset"] == ds]
        po = systems["**Probe trigger (ours)**"]
        rate = {k: np.mean([po[it["uid"]]["hop"] == k for it in its]) for k in (1, 2)}
        acc, gen, fixed, broken, fire = [], [], [], [], []
        base = np.array([systems["No intervention"][it["uid"]]["correct"] for it in its], float)
        per_seed = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            o = []
            for it in its:
                u = it["uid"]; r = rng.random()
                k = 1 if r < rate[1] else (2 if r < rate[1] + rate[2] else None)
                sc = {k: 1.0} if k in closed(u) else {}
                o.append(outcome("probe_restart", it, s0[u], rs[u], probe_scores=sc, tau=one))
            c = np.array([x["correct"] for x in o], float)
            per_seed.append(c)
            acc.append(c.mean()); gen.append(np.mean([x["gen_tokens"] for x in o]))
            fixed.append(((c == 1) & (base == 0)).sum()); broken.append(((c == 0) & (base == 1)).sum())
            fire.append(np.mean([x["fired"] for x in o]))
        rnd[ds] = dict(acc=float(np.mean(acc)), acc_sd=float(np.std(acc)), gen=float(np.mean(gen)),
                       fixed=float(np.mean(fixed)), broken=float(np.mean(broken)), fire=float(np.mean(fire)),
                       seeds=per_seed)

    L = [f"# Is the probe needed? Trivial triggers vs the probe (`{run.name}`)", "",
         "All triggers act on the same cached generations: same model, prompt, repair and restart outputs, one "
         "restart. Only the decision *whether and where to restart* differs. Correct = token F1 ≥ 0.5. "
         "CIs: paired bootstrap of probe − trigger accuracy.", "",
         f"Trained text trigger: TF-IDF (1–2-grams) + logistic regression on the hop text, trained on the "
         f"probe's train split ({n_text} labelled hops 1–2), threshold F0.5-optimal on out-of-fold predictions.", ""]
    for ds in ("2wiki", "hotpotqa"):
        its = [it for it in items if it["dataset"] == ds]
        base = np.array([systems["No intervention"][it["uid"]]["correct"] for it in its], float)
        pr = np.array([systems["**Probe trigger (ours)**"][it["uid"]]["correct"] for it in its], float)
        L += [f"## {ds} (n = {len(its)})", "",
              "| trigger | accuracy | restart rate | fixed | broken | gen tokens | probe − trigger [95% CI] |",
              "|---|---|---|---|---|---|---|"]
        for name, o in systems.items():
            oo = [o[it["uid"]] for it in its]
            c = np.array([x["correct"] for x in oo], float)
            ci = "–" if "Probe trigger" in name else "{:+.3f} [{:+.3f}, {:+.3f}]".format(pr.mean() - c.mean(), *paired_ci(pr, c))
            L.append(f"| {name} | {c.mean():.3f} | {np.mean([x['fired'] for x in oo]):.2f} | "
                     f"{int(((c == 1) & (base == 0)).sum())} | {int(((c == 0) & (base == 1)).sum())} | "
                     f"{np.mean([x['gen_tokens'] for x in oo]):.1f} | {ci} |")
        r = rnd[ds]
        diffs = [pr.mean() - s.mean() for s in r["seeds"]]
        L.append(f"| Random trigger at the probe's per-hop restart rate (200 seeds) | {r['acc']:.3f} ± {r['acc_sd']:.3f} | "
                 f"{r['fire']:.2f} | {r['fixed']:.1f} | {r['broken']:.1f} | {r['gen']:.1f} | "
                 f"{np.mean(diffs):+.3f} (all 200 seeds below the probe: {all(d > 0 for d in diffs)}) |")
        L.append("")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
