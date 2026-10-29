"""
Idea 2 — "retrieve with the model's thoughts" (latent retrieval), offline test on CPU.

A ridge regression maps the hop's hidden state (mean-pooled hop tokens at layer L, as the probe reads it) to
the retriever's embedding of the paragraph that hop needs (all-MiniLM-L6-v2 space).  At test time the mapped
vector is used as a search query, alone or blended with the text query.

Train: probe TRAIN split (all labelled hops 1-2 whose needed paragraph is in the corpus).  Ridge alpha and the
blend weight are chosen by grouped CV inside train.  Test: probe TEST split, hops that FAILED and whose needed
paragraph was MISSING from the prompt (the cases a repair has to fix).  Metric: the needed paragraph is in the
top-k results (paragraphs already in the prompt excluded).

Usage:
    python scripts/latent_retrieval.py --hs-dir <tfidf hidden_states dir> --run <pipeline run dir> \
        --out outputs/pipeline/latent_retrieval.md [--layer 12] [--rerank]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.pipeline.common import (canonical_split, context_titles, gold_titles_in_hop_order, group_id,  # noqa: E402
                                 hidden_state_ids, hop_text, load_setting_examples, read_jsonl, repair_query)
from src.pipeline.retriever import EMBED_MODEL, RERANK_MODEL, build_corpus  # noqa: E402


def rows_for(examples, idx, hs_dir, layer, title_idx, only_failed_missing=False):
    X, T, meta = [], [], []
    for i in idx:
        e = examples[i]
        order = gold_titles_in_hop_order(e)
        if order is None:
            continue
        spans = [h["hop_idx"] for h in e.get("hop_spans", [])]
        ctx = set(context_titles(e["context"]))
        pooled = None
        for h in e["hops"]:
            k = h["hop_idx"]
            if k not in (1, 2) or h.get("label") not in (0, 1) or k not in spans:
                continue
            need = order[k - 1]
            if need not in title_idx:
                continue
            if only_failed_missing and not (h["label"] == 1 and need not in ctx):
                continue
            if pooled is None:
                d = torch.load(Path(hs_dir) / f"{e['id']}.pt", map_location="cpu", weights_only=False)
                li = list(d.get("layer_indices", range(d["pooled"].shape[0])))
                pooled = d["pooled"][li.index(layer)].float().numpy()
            X.append(pooled[spans.index(k)])
            T.append(title_idx[need])
            hops = {"1": hop_text(e["generated_cot"], 1), "2": hop_text(e["generated_cot"], 2)}
            meta.append(dict(hop=k, need=need, excl=ctx, group=group_id(e),
                             q=repair_query(e["question"], hops, k, "v2")))
    return np.nan_to_num(np.array(X, dtype=np.float32)), np.array(T), meta


def fit_ridge(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ Y)
    b = Y.mean(0) - 0.0
    return lambda x: ((np.atleast_2d(x) - mu) / sd) @ W + b


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def hits_at_k(Q, E, targets, excl_sets, titles, k=3, pool=None):
    """Q: [n, d] queries (normalised).  Returns hit@k list and the candidate pools (for re-ranking)."""
    S = Q @ E.T
    hits, pools = [], []
    for i in range(len(Q)):
        top = np.argpartition(-S[i], 80)[:80]
        top = top[np.argsort(-S[i][top])]
        cand = [j for j in top if titles[j] not in excl_sets[i]]
        pools.append(cand[: (pool or k)])
        hits.append(targets[i] in cand[:k])
    return np.array(hits), pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs-dir", required=True)
    ap.add_argument("--run", required=True, help="pipeline run dir (for hotpot_passages.jsonl, to rebuild the corpus)")
    ap.add_argument("--emb-cache", default=None, help="corpus embeddings .npy (same corpus order) to skip encoding")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    ex = load_setting_examples("tfidf", hidden_state_ids("tfidf"))
    tr, te = canonical_split(ex)
    corpus = build_corpus({e["id"] for e in load_setting_examples("tfidf")},
                          read_jsonl(Path(args.run) / "hotpot_passages.jsonl"), 20000, 42)
    titles = [p["title"] for p in corpus]
    title_idx = {t: i for i, t in enumerate(titles)}
    from sentence_transformers import CrossEncoder, SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    if args.emb_cache and Path(args.emb_cache).exists():
        E = np.load(args.emb_cache)
        assert len(E) == len(corpus), "embedding cache does not match the corpus"
    else:
        E = enc.encode([f"{p['title']}: {p['text']}" for p in corpus], batch_size=256,
                       normalize_embeddings=True).astype(np.float32)
    print(f"corpus {len(corpus)} ({time.time() - t0:.0f}s)", flush=True)

    Xtr, Ttr, Mtr = rows_for(ex, tr, args.hs_dir, args.layer, title_idx)
    Xte, Tte, Mte = rows_for(ex, te, args.hs_dir, args.layer, title_idx, only_failed_missing=True)
    print(f"train rows {len(Xtr)}; test rows (failed hop, needed paragraph missing) {len(Xte)} "
          f"(hop 1: {sum(m['hop'] == 1 for m in Mte)}, hop 2: {sum(m['hop'] == 2 for m in Mte)}) ({time.time() - t0:.0f}s)",
          flush=True)
    Ytr = E[Ttr]

    # ── choose ridge alpha and blend weight by grouped CV inside train (on failed+missing train rows) ──
    from sklearn.model_selection import GroupKFold
    groups = np.array([m["group"] for m in Mtr])
    fm = np.array([i for i, m in enumerate(Mtr) if m["need"] not in m["excl"]])     # needed paragraph missing
    Qtext_tr = norm(enc.encode([Mtr[i]["q"] for i in fm], normalize_embeddings=True))
    best = None
    for alpha in (1e2, 1e3, 1e4, 3e4, 1e5):
        lat = np.zeros((len(fm), E.shape[1]), dtype=np.float32)
        for a, b in GroupKFold(5).split(Xtr, groups=groups):
            f = fit_ridge(Xtr[a], Ytr[a], alpha)
            sel = np.isin(fm, b)
            lat[sel] = f(Xtr[fm[sel]])
        lat = norm(lat)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):                               # w = weight on the latent query
            Q = norm(w * lat + (1 - w) * Qtext_tr)
            h, _ = hits_at_k(Q, E, Ttr[fm], [Mtr[i]["excl"] for i in fm], titles)
            if best is None or h.mean() > best[0]:
                best = (h.mean(), alpha, w)
        print(f"  alpha {alpha:g}: best so far hit@3 {best[0]:.3f} (alpha {best[1]:g}, w {best[2]})", flush=True)
    _, alpha, w_best = best
    f = fit_ridge(Xtr, Ytr, alpha)

    # ── test ─────────────────────────────────────────────────────────────────
    Qtext = norm(enc.encode([m["q"] for m in Mte], normalize_embeddings=True))
    Qlat = norm(f(Xte))
    excl = [m["excl"] for m in Mte]
    variants = {"text query (current)": Qtext, "hidden-state query": Qlat,
                f"blend (w = {w_best} on hidden state, chosen on train)": norm(w_best * Qlat + (1 - w_best) * Qtext),
                "blend (w = 0.5)": norm(0.5 * Qlat + 0.5 * Qtext)}
    res = {}
    ce = CrossEncoder(RERANK_MODEL, device="cpu") if args.rerank else None
    for name, Q in variants.items():
        h3, pools = hits_at_k(Q, E, Tte, excl, titles, 3, pool=20)
        h10, _ = hits_at_k(Q, E, Tte, excl, titles, 10)
        r = dict(h3=h3, h10=h10)
        if ce is not None:
            rr = []
            for m, pool, t in zip(Mte, pools, Tte):
                sc = ce.predict([(m["q"], f"{titles[j]}: {corpus[j]['text']}") for j in pool], show_progress_bar=False)
                rr.append(t in [pool[i] for i in np.argsort(-sc)][:3])
            r["h3_rerank"] = np.array(rr)
        res[name] = r
        print(f"{name}: hit@3 {h3.mean():.3f}" + (f", +rerank {r['h3_rerank'].mean():.3f}" if ce else ""), flush=True)

    hop = np.array([m["hop"] for m in Mte])
    L = [f"# Latent retrieval: search with the model's hidden state (layer {args.layer})", "",
         f"Ridge map (alpha = {alpha:g}, chosen by grouped 5-fold CV on the train split) from the mean-pooled hop state to the "
         "all-MiniLM-L6-v2 embedding of the paragraph that hop needs. Trained on all labelled hops 1–2 of the probe's "
         f"train split ({len(Xtr)} rows). Tested on the probe's **test** split: hops that failed and whose needed "
         f"paragraph was missing from the prompt ({len(Mte)} rows; hop 1: {(hop == 1).sum()}, hop 2: {(hop == 2).sum()}). "
         "A hit = the needed paragraph is among the top results (paragraphs already in the prompt excluded). "
         "The text query is the pipeline's current repair query (repair v2). Corpus: the pipeline's 94,031 paragraphs.", "",
         "| query | hit@3 | hit@3 hop 1 | hit@3 hop 2 | hit@10 |" + (" hit@3 after cross-encoder re-rank of top 20 |" if ce else ""),
         "|---|---|---|---|---|" + ("---|" if ce else "")]
    for name, r in res.items():
        L.append(f"| {name} | {r['h3'].mean():.3f} | {r['h3'][hop == 1].mean():.3f} | {r['h3'][hop == 2].mean():.3f} | "
                 f"{r['h10'].mean():.3f} |" + (f" {r['h3_rerank'].mean():.3f} |" if ce else ""))
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
