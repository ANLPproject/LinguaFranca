"""
advanced_probing.py  (v2)
─────────────────────────
Hop-level failure probes on pooled hidden states, with honest baselines.

What changed vs v1 (see FIXES_AND_RERUN.md):
  * Only hops 1-2 are used.  v1 also used the model's extra hops (hop 3..18),
    which have no gold entity and were all labeled "failure" by the old
    labeler — 1/3 of v1's rows and 64 % of its failure labels.
  * Layer selection uses 5-fold grouped CV on the TRAIN split only; the test
    split is touched once, for the final numbers.
  * Every recall (TPR) is reported with its false-positive rate.
  * Saved probes are trained on the right hop (v1 saved Hop-2 probes under the
    name hop1_probe_*), and carry metadata so evaluate_saved_probe.py can check.
  * Natural examples and counterfactuals (CF) are reported separately.  CFs are
    "entity not in context" questions; ~90 % of their hop-1 failures are the
    model saying so ("hedging").  Natural hop-1 failures contain no hedging.
  * Fixed: NameErrors in the hedging block, unseeded/MLP shuffled-label control,
    cross-hop test that included training examples.

Usage
─────
  python advanced_probing.py --data data/canonical/augmented_v2.jsonl \
                             --hs-dir data/hidden_states --out outputs/probing
  (legacy form also works:  python advanced_probing.py <data.jsonl> <hs_dir>)

Runtime: CPU only.  ~25 min for 28 layers on a 16-core laptop; ~45-60 min on a
Kaggle CPU session.  RAM: ~100 MB per layer in a chunk (--layer-chunk).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from lf_common import (DEFAULT_DATA, binary_metrics, canonical_split, fmt_metrics, grouped_bootstrap_ci,
                       hop_rows, load_examples, load_hidden, make_probe, rows_arrays)

TASKS = {  # task name -> which hops it trains/tests on
    "hop1": (1,),
    "hop2": (2,),
    "hop12": (1, 2),
}


def oof_cv(X, y, groups, n_splits=5, seed=42):
    """Out-of-fold probabilities with GroupKFold."""
    prob = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = make_probe(seed).fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    return prob


def strata(A, mask_extra=None):
    """Named boolean masks over rows for reporting."""
    base = np.ones(len(A["y"]), bool) if mask_extra is None else mask_extra
    return {"all": base, "natural": base & ~A["cf"], "cf": base & A["cf"]}


def failure_breakdown(A, prob, mask):
    """Hop-1 failure recall split by hedging / confident, plus FPR on successes."""
    pred = prob >= 0.5
    f = mask & (A["y"] == 1)
    s = mask & (A["y"] == 0)
    def rate(m):
        return (float(pred[m].mean()), int(m.sum())) if m.sum() else (float("nan"), 0)
    return {
        "hedge_recall": rate(f & A["hedge"]),
        "confident_recall": rate(f & ~A["hedge"]),
        "natural_fail_recall": rate(f & ~A["cf"]),
        "cf_fail_recall": rate(f & A["cf"]),
        "fpr_on_successes": rate(s),
    }


def text_baselines(A, rows, tr_rows, te_rows):
    """Baselines that never look at hidden states (train split -> test split)."""
    out = {}
    txt = np.array([r["text"] for r in rows], dtype=object)
    for task, hops in TASKS.items():
        trm = tr_rows & np.isin(A["hop"], hops)
        tem = te_rows & np.isin(A["hop"], hops)
        y_tr, y_te = A["y"][trm], A["y"][tem]
        res = {}
        maj = int(round(y_tr.mean()))
        res["majority"] = binary_metrics(y_te, prob=np.full(len(y_te), float(maj)), pred=np.full(len(y_te), maj))
        res["hedge_keyword_rule"] = binary_metrics(y_te, prob=A["hedge"][tem].astype(float),
                                                   pred=A["hedge"][tem].astype(int))
        tf = TfidfVectorizer(max_features=1000)
        Xtr = tf.fit_transform(txt[trm]); Xte = tf.transform(txt[tem])
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y_tr)
        res["tfidf_text"] = binary_metrics(y_te, prob=c.predict_proba(Xte)[:, 1])
        if len(hops) > 1:
            oh_tr = np.eye(3)[A["hop"][trm]]; oh_te = np.eye(3)[A["hop"][tem]]
            c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(oh_tr, y_tr)
            res["hop_index_only"] = binary_metrics(y_te, prob=c.predict_proba(oh_te)[:, 1])
        out[task] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_data", nargs="?", default=None)
    ap.add_argument("legacy_hs", nargs="?", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--hs-dir", default=None)
    ap.add_argument("--out", default="outputs/probing")
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 8,10,11,12")
    ap.add_argument("--layer-chunk", type=int, default=7, help="layers loaded into RAM at once")
    ap.add_argument("--exclude-cf", action="store_true", help="train/test on natural examples only")
    args = ap.parse_args()

    data = args.data or args.legacy_data or DEFAULT_DATA
    hs_dir = Path(args.hs_dir or args.legacy_hs or "data/hidden_states")
    out = Path(args.out); (out / "probes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    examples = load_examples(data, hs_dir=hs_dir)
    if args.exclude_cf:
        examples = [e for e in examples if not e.get("is_counterfactual")]
    tr_idx, te_idx = canonical_split(examples)
    rows = hop_rows(examples)
    A = rows_arrays(rows)
    is_tr_ex = np.zeros(len(examples), bool); is_tr_ex[tr_idx] = True
    tr_rows = is_tr_ex[A["ei"]]; te_rows = ~tr_rows
    data_sha = hashlib.sha256(Path(data).read_bytes()).hexdigest()[:16]

    import torch
    sample = torch.load(hs_dir / f"{examples[0]['id']}.pt", map_location="cpu", weights_only=False)
    n_layers = sample["pooled"].shape[0]
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    print("=" * 100)
    print(f"data={data} (sha256 {data_sha}…)  examples={len(examples)}  "
          f"train/test examples={len(tr_idx)}/{len(te_idx)}  rows(hops 1-2)={len(rows)}")
    for task, hops in TASKS.items():
        for name, m in (("train", tr_rows), ("test", te_rows)):
            mm = m & np.isin(A["hop"], hops)
            print(f"  {task:5s} {name:5s}: n={mm.sum():5d}  fail={A['y'][mm].sum():5d} "
                  f"({100 * A['y'][mm].mean():.1f}%)  cf_rows={A['cf'][mm].sum()}  hedging_fails={(A['hedge'] & (A['y'] == 1))[mm].sum()}")

    results = {"meta": dict(data=str(data), data_sha256_16=data_sha, n_examples=len(examples),
                            n_train_examples=int(len(tr_idx)), n_test_examples=int(len(te_idx)),
                            exclude_cf=args.exclude_cf, layers=layers),
               "text_baselines": text_baselines(A, rows, tr_rows, te_rows),
               "layers": {}}
    print("\nTEXT-ONLY BASELINES (train split -> test split, all test rows):")
    for task, res in results["text_baselines"].items():
        for name, m in res.items():
            print(f"  {task:5s} {name:20s} {fmt_metrics(m)}")

    test_probs = {}  # column name -> array over all rows (nan where not applicable)
    for c0 in range(0, len(layers), args.layer_chunk):
        chunk = layers[c0:c0 + args.layer_chunk]
        X = load_hidden(examples, rows, hs_dir, chunk)
        print(f"\n[{time.time() - t0:.0f}s] loaded layers {chunk}")
        for L in chunk:
            XL = X[L]
            lr = {}
            fitted = {}
            for task, hops in TASKS.items():
                tm = np.isin(A["hop"], hops)
                trm = tr_rows & tm; tem = te_rows & tm
                # (a) 5-fold grouped CV inside the train split -> layer selection
                p_oof = np.full(len(rows), np.nan)
                p_oof[trm] = oof_cv(XL[trm], A["y"][trm], A["group"][trm])
                cv = {s: binary_metrics(A["y"][m], p_oof[m]) for s, m in strata(A, trm).items() if m.sum()}
                # (b) fit on full train split -> held-out test
                clf = make_probe().fit(XL[trm], A["y"][trm])
                fitted[task] = clf
                p_te = np.full(len(rows), np.nan)
                p_te[tem] = clf.predict_proba(XL[tem])[:, 1]
                test = {s: binary_metrics(A["y"][m], p_te[m]) for s, m in strata(A, tem).items() if m.sum()}
                lr[task] = {"cv_train": cv, "test": test}
                if task == "hop1":
                    lr[task]["breakdown_cv_train"] = failure_breakdown(A, p_oof, trm)
                    lr[task]["breakdown_test"] = failure_breakdown(A, p_te, tem)
                    # internal -> external: does the hop-1 score predict a wrong FINAL answer?
                    m = tem
                    lr[task]["test_predicts_final_wrong"] = {
                        "probe_auroc": binary_metrics(~A["final_correct"][m], p_te[m]).get("auroc"),
                        "gold_hop1_label_auroc": binary_metrics(~A["final_correct"][m], A["y"][m].astype(float)).get("auroc"),
                    }
                test_probs[f"{task}_L{L}"] = p_te
                joblib.dump({"model": clf, "task": task, "hops": list(hops), "layer": L,
                             "trained_on": "canonical train split (GroupShuffleSplit seed 42, 80%)",
                             "data_sha256_16": data_sha, "n_train": int(trm.sum()),
                             "exclude_cf": args.exclude_cf},
                            out / "probes" / f"{task}_probe_layer{L}.joblib")
            # cross-hop generalisation: hop-1 probe applied to held-out hop-2 rows
            m2 = te_rows & (A["hop"] == 2)
            lr["cross_hop1_to_hop2_test"] = binary_metrics(A["y"][m2], fitted["hop1"].predict_proba(XL[m2])[:, 1])
            results["layers"][L] = lr
            h1 = lr["hop1"]
            print(f"[{time.time() - t0:.0f}s] L{L:2d} | hop1 CV(train,natural) auroc={h1['cv_train']['natural'].get('auroc', np.nan):.3f} "
                  f"| hop1 TEST all: {fmt_metrics(h1['test']['all'], ('acc', 'bacc', 'auroc', 'tpr', 'fpr'))} "
                  f"| hop2 TEST auroc={lr['hop2']['test']['all'].get('auroc', np.nan):.3f} "
                  f"| hop12 TEST natural auroc={lr['hop12']['test']['natural'].get('auroc', np.nan):.3f}", flush=True)
        del X
        with open(out / "results_partial.json", "w") as f:        # safety net if a later step fails
            json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ── layer selection (train-CV only) + final held-out report ───────────────
    def best_layer(task):
        return max(layers, key=lambda L: results["layers"][L][task]["cv_train"]["natural"].get("auroc", -1))

    sel = {task: best_layer(task) for task in TASKS}
    results["selected_layers"] = sel
    X = load_hidden(examples, rows, hs_dir, sorted(set(sel.values())))
    print("\n" + "=" * 100)
    print("SELECTED LAYERS (by 5-fold CV AUROC on the TRAIN split, natural rows):", sel)
    final = {}
    for task, L in sel.items():
        tem = te_rows & np.isin(A["hop"], TASKS[task])
        p = test_probs[f"{task}_L{L}"]
        rep = {}
        for s, m in strata(A, tem).items():
            if not m.sum():
                continue
            rep[s] = binary_metrics(A["y"][m], p[m])
            if s != "cf":
                rep[s]["ci95"] = {k: grouped_bootstrap_ci(A["y"][m], p[m], A["group"][m], k)
                                  for k in ("auroc", "bacc", "tpr", "fpr")}
        # shuffled-label control: same probe, same split, permuted train labels
        trm = tr_rows & np.isin(A["hop"], TASKS[task])
        y_shuf = np.random.RandomState(0).permutation(A["y"][trm])
        ctrl = make_probe().fit(X[L][trm], y_shuf)
        rep["shuffled_label_control"] = binary_metrics(A["y"][tem], ctrl.predict_proba(X[L][tem])[:, 1])
        final[task] = {"layer": L, **rep}
        print(f"\n[{task}] layer {L} — HELD-OUT TEST")
        for s in ("all", "natural", "cf"):
            if s in rep:
                ci = rep[s].get("ci95", {})
                ci_s = "  ".join(f"{k}95%=[{v[0]:.3f},{v[1]:.3f}]" for k, v in ci.items())
                print(f"  {s:8s} {fmt_metrics(rep[s])}  {ci_s}")
        print(f"  shuffled-label control: {fmt_metrics(rep['shuffled_label_control'], ('acc', 'bacc', 'auroc'))}")
    results["final_test"] = final
    bd = results["layers"][sel["hop1"]]["hop1"]
    print(f"\n[hop1] failure breakdown at layer {sel['hop1']} (recall at threshold 0.5; value, n):")
    for split in ("breakdown_cv_train", "breakdown_test"):
        print(f"  {split}: " + "  ".join(f"{k}={v[0]:.3f}(n={v[1]})" for k, v in bd[split].items()))
    print(f"  hop-1 score -> wrong final answer (test): {bd['test_predicts_final_wrong']}")

    # ── write outputs ──────────────────────────────────────────────────────────
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_markdown(results, out / "results.md")
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = sorted(test_probs)
        w.writerow(["id", "hop_idx", "label", "is_cf", "is_hedge", "final_answer_correct"] + cols)
        for i in np.flatnonzero(te_rows):
            r = rows[i]
            w.writerow([r["id"], r["hop"], r["y"], int(r["cf"]), int(r["hedge"]), int(r["final_correct"])]
                       + [("" if np.isnan(test_probs[c][i]) else f"{test_probs[c][i]:.5f}") for c in cols])
    write_qualitative(examples, rows, A, test_probs[f"hop1_L{sel['hop1']}"], te_rows, sel["hop1"],
                      out / "qualitative_hop1.md")
    print(f"\n[done in {time.time() - t0:.0f}s] outputs in {out}/ : results.json, results.md, "
          f"test_predictions.csv, qualitative_hop1.md, probes/*.joblib")


def write_markdown(R, path):
    L = []
    L.append("# Probing results (v2: hops 1-2 only, canonical split)\n")
    m = R["meta"]
    L.append(f"Examples: {m['n_examples']} (train {m['n_train_examples']} / test {m['n_test_examples']}); "
             f"data sha256 prefix `{m['data_sha256_16']}`.\n")
    L.append("## Text-only baselines (test split)\n")
    L.append("| task | baseline | acc | bal. acc | AUROC | TPR | FPR | majority |\n|---|---|---|---|---|---|---|---|")
    for task, res in R["text_baselines"].items():
        for name, x in res.items():
            L.append(f"| {task} | {name} | {x.get('acc', float('nan')):.3f} | {x.get('bacc', float('nan')):.3f} | "
                     f"{x.get('auroc', float('nan')):.3f} | {x.get('tpr', float('nan')):.3f} | {x.get('fpr', float('nan')):.3f} | {x.get('majority', float('nan')):.3f} |")
    L.append("\n## Per-layer probes\n")
    L.append("CV = 5-fold grouped CV inside the train split (natural rows). TEST = held-out test split (all rows).\n")
    L.append("| layer | hop1 CV AUROC | hop1 TEST acc | bal.acc | AUROC | TPR | FPR | hop1 confident-fail recall (CV) | hop2 TEST AUROC | hop1+2 TEST natural AUROC | hop1→hop2 AUROC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for Lr, x in sorted(R["layers"].items(), key=lambda kv: int(kv[0])):
        h1 = x["hop1"]; t = h1["test"]["all"]
        L.append(f"| {Lr} | {h1['cv_train']['natural'].get('auroc', float('nan')):.3f} | {t['acc']:.3f} | {t['bacc']:.3f} | "
                 f"{t.get('auroc', float('nan')):.3f} | {t['tpr']:.3f} | {t['fpr']:.3f} | "
                 f"{h1['breakdown_cv_train']['confident_recall'][0]:.3f} | "
                 f"{x['hop2']['test']['all'].get('auroc', float('nan')):.3f} | "
                 f"{x['hop12']['test']['natural'].get('auroc', float('nan')):.3f} | "
                 f"{x['cross_hop1_to_hop2_test'].get('auroc', float('nan')):.3f} |")
    L.append(f"\n## Selected layers (by train-split CV): {R['selected_layers']}\n")
    for task, x in R["final_test"].items():
        L.append(f"### {task} @ layer {x['layer']} (held-out test)\n")
        for s in ("all", "natural", "cf"):
            if s in x:
                ci = x[s].get("ci95", {})
                ci_s = ", ".join(f"{k} 95% CI [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci.items())
                L.append(f"- **{s}**: {fmt_metrics(x[s])}" + (f" — {ci_s}" if ci_s else ""))
        L.append(f"- shuffled-label control: {fmt_metrics(x['shuffled_label_control'], ('acc', 'bacc', 'auroc'))}\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def write_qualitative(examples, rows, A, prob, te_rows, layer, path):
    m = te_rows & (A["hop"] == 1)
    idx = np.flatnonzero(m)
    def block(title, sel, k):
        out = [f"\n## {title}\n"]
        for i in sel[:k]:
            r = rows[i]; e = examples[r["ei"]]
            gold = e["reasoning_graph"][0]["gold_entity"]
            out.append(f"- p(fail)={prob[i]:.3f} | label={r['y']} | {'CF' if r['cf'] else 'natural'}\n"
                       f"  - Q: {e['question']}\n  - gold hop-1 entity: {gold}\n  - hop 1: {r['text']}")
        return out
    order = idx[np.argsort(-prob[idx])]
    L = [f"# Qualitative hop-1 analysis (layer {layer}, held-out test split)"]
    L += block("Most confident failure predictions — natural examples (true failures)",
               [i for i in order if not A["cf"][i] and A["y"][i] == 1], 10)
    L += block("False alarms — natural successes the probe flags as failures",
               [i for i in order if A["y"][i] == 0], 10)
    L += block("Missed failures — natural failures with the lowest p(fail)",
               [i for i in order[::-1] if not A["cf"][i] and A["y"][i] == 1], 10)
    L += block("Counterfactual (entity-not-in-context) failures",
               [i for i in order if A["cf"][i]], 5)
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
