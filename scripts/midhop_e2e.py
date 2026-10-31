"""
scripts/midhop_e2e.py
─────────────────────
End-to-end accuracy of stopping a failing hop MID-HOP (the prefix probe of online_probe.py), on the final
Phase 3-5 run.  Exact, not approximate: a mid-hop stop changes only the restart's retrieval query (the hop text
is shorter), so the restart is re-retrieved with the real retriever; if it gets the same paragraphs as the cached
restart, the cached answer is the exact answer (same prompt, greedy decoding), otherwise the restart is generated
(and its answer forced if it hits the token cap, as in the pipeline).

Stages (one Kaggle GPU session, the run folder attached; --stage all):
  prefix    online_probe.extract + fit_prefix: prefix probe + thresholds (train split only)   (GPU ~20 min + CPU)
  score     per-token prefix-probe scores on the run's first-stage trajectories              (GPU ~5 min)
  restart   build the retriever (same corpus/models as the run); parity check on every cached
            restart's stored query; retrieve for every mid-hop stop; generate the changed ones (GPU)
  report    accuracy, restart rate, tokens, paired CIs vs the deployed probe                  (CPU)

Systems (one restart, as the headline "Probe trigger" row):
  deployed   the run's probe at `</hopK>` (τ = 0.650)
  hybrid     prefix probe at the strict θ during hop k; otherwise the deployed probe at `</hopK>`
  online     prefix probe at the F0.5 θ only (no end-of-hop check)

Usage:
    python scripts/midhop_e2e.py --stage all --run /kaggle/input/<...>/pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12 \
        --out /kaggle/working/midhop
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

from src.pipeline.common import (answer_correct, hop_match, prompt_with_evidence, read_jsonl,  # noqa: E402
                                 repair_query)

PROBE_DIR = "outputs/pipeline/tfidf/probes_lr_C0.01_L12"
TOP_K, RERANK_POOL, N_EXTRA, SEED, MAX_NEW = 3, 20, 20000, 42, 400          # the run's settings (run_args.json)


# ─────────────────────────────────────────────────────────────────────────────
# score: per-token prefix-probe scores on the run's trajectories (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def score(out: Path, probe_dir: Path, run: Path):
    import torch
    from eval_robustness import Run
    from src.pipeline.generation import _device, char_span_to_tokens, load_model
    from src.pipeline.probes import LinearProbe
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    layer = meta["probes"]["hop12"]["layer"]
    prefix = LinearProbe(out / "prefix_probe.npz")
    min_t = json.loads((out / "prefix_meta.json").read_text())["min_t"]
    R = Run(run, force=True)
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    dev = _device(model)
    path, t0, n_skip = out / "midhop_signals.jsonl", time.time(), 0
    with open(path, "w", encoding="utf-8") as f, torch.inference_mode():
        for n, it in enumerate(R.items):
            u, s0 = it["uid"], R.s0[it["uid"]]
            prompt, cot = it["prompt"], s0.get("text", "")
            full = tok(prompt + cot, return_tensors="pt").input_ids[0]
            p_len = len(tok(prompt, return_tensors="pt").input_ids[0])
            enc = tok(cot, return_offsets_mapping=True, add_special_tokens=False)
            rec = dict(uid=u, hops={})
            if cot and len(full) == p_len + len(enc["input_ids"]):
                offs = enc["offset_mapping"]
                hs = model.model(full.unsqueeze(0).to(dev), output_hidden_states=True).hidden_states[layer + 1][0]
                for k in sorted(R.eligible(u)):
                    m = hop_match(cot, k)
                    if m is None:
                        continue
                    ts, te = char_span_to_tokens([x[0] for x in offs], [x[1] for x in offs], m.start(1), m.end(1))
                    if ts is None or te is None:
                        continue
                    a, b = p_len + ts, p_len + te + 1
                    H = hs[a:b].float()
                    cum = (H.cumsum(0) / torch.arange(1, b - a + 1, device=dev)[:, None]).cpu().numpy()
                    nh = b - a
                    t = np.arange(min(min_t, nh), nh + 1)
                    p = prefix.p_fail(np.column_stack([cum[t - 1], np.log(t)]))
                    rec["hops"][str(k)] = dict(
                        n=int(nh), t_min=int(t[0]), p=np.round(p, 5).tolist(), gen_before=int(ts),
                        # character end of each hop token, relative to the hop content (partial text = content[:cend])
                        cend=[int(min(offs[ts + i][1], m.end(1)) - m.start(1)) for i in range(nh)])
            else:
                n_skip += 1
            f.write(json.dumps(rec) + "\n")
            if n % 100 == 0:
                print(f"  [score] {n}/{len(R.items)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[score] done; {n_skip} trajectories not token-aligned (they keep the deployed decision)")


# ─────────────────────────────────────────────────────────────────────────────
# decisions
# ─────────────────────────────────────────────────────────────────────────────

def decide(system, u, R, P, tau, sig, pmeta):
    """(k, t) — restart at hop k after t hop tokens (t = None: at `</hopK>`, the cached restart) — or None."""
    th = pmeta["th_strict"] if system == "hybrid" else pmeta["th_f05"]
    for k in sorted(R.eligible(u)):
        h = sig.get(u, {}).get(str(k))
        if system != "deployed" and h is not None:
            hit = [i for i, p in enumerate(h["p"]) if p >= th]
            if hit:
                t = h["t_min"] + hit[0]
                return (k, None if t >= h["n"] else t)
        if system != "online" or h is None:                       # online falls back only if it has no scores
            if P[u].get(k, -1.0) >= tau[k]:
                return (k, None)
    return None


def partial_hops(R, u, k, t, sig):
    s0 = R.s0[u]
    content = hop_match(s0["text"], k).group(1)
    part = content[:sig[u][str(k)]["cend"][t - 1]].strip()
    return {"1": s0["hop_text"]["1"] if k == 2 else part, "2": part if k == 2 else ""}


# ─────────────────────────────────────────────────────────────────────────────
# restart: retrieval (+ parity) and new generations (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def restart(out: Path, probe_dir: Path, run: Path):
    from eval_robustness import Run
    from src.pipeline.common import load_setting_examples
    from src.pipeline.generation import force_answers, generate_batch, load_model
    from src.pipeline.probes import ProbeBundle
    from src.pipeline.retriever import DenseRetriever, build_corpus
    R = Run(run, force=True)
    pb = ProbeBundle(probe_dir, mode="shared")
    P = R.probe_scores(pb)
    pmeta = json.loads((out / "prefix_meta.json").read_text())
    sig = {r["uid"]: r["hops"] for r in read_jsonl(out / "midhop_signals.jsonl")}
    by_uid = {it["uid"]: it for it in R.items}

    cdir = out / "corpus"
    if (cdir / "embeddings.npy").exists():
        ret = DenseRetriever.load(cdir)
    else:
        ids = {e["id"] for e in load_setting_examples("tfidf")}
        ret = DenseRetriever(build_corpus(ids, read_jsonl(run / "hotpot_passages.jsonl"), N_EXTRA, SEED))
        ret.save(cdir)

    def search(it, query):
        hits, _ = ret.search(query, TOP_K, exclude_titles=it["context_titles"], rerank_pool=RERANK_POOL)
        return hits

    # parity: the rebuilt retriever on every cached hop restart's stored query
    cached = [(u, key, r) for u, rs in R.rs.items() if u in by_uid for key, r in rs.items() if key in ("h1", "h2")]
    same = sum(1 for u, key, r in cached if [h["title"] for h in search(by_uid[u], r["query"])] == r["retrieved_titles"])
    parity = dict(n=len(cached), identical=same)
    print(f"[restart] retrieval parity on cached restarts: {same}/{len(cached)}", flush=True)

    # every mid-hop stop of every system
    todo = {}
    for system in ("hybrid", "online"):
        for it in R.items:
            d = decide(system, it["uid"], R, P, pb.tau, sig, pmeta)
            if d is not None and d[1] is not None:
                todo[(it["uid"], d[0], d[1])] = None
    recs, pending = [], []
    for (u, k, t) in todo:
        it = by_uid[u]
        query = repair_query(it["question"], partial_hops(R, u, k, t, sig), k, "v2")
        hits = search(it, query)
        titles = [h["title"] for h in hits]
        rec = dict(uid=u, k=k, t=t, query=query, retrieved_titles=titles,
                   same_as_cached=titles == R.rs[u][f"h{k}"]["retrieved_titles"])
        if not rec["same_as_cached"]:
            rec["prompt"] = prompt_with_evidence(it["prompt"], hits)
            pending.append(rec)
        recs.append(rec)
    print(f"[restart] {len(recs)} mid-hop stops; {len(pending)} retrieve different paragraphs -> generate", flush=True)
    if pending:
        model, tok = load_model(token=os.environ.get("HF_TOKEN"))
        outs = generate_batch(model, tok, [r["prompt"] for r in pending], MAX_NEW, 8)
        for r, o in zip(pending, outs):
            r.update(text=o["text"], answer=o["answer"], n_gen=o["n_gen"], n_prompt=o["n_prompt"], forced=False)
        need = [r for r in pending if not r["answer"]]
        for r, o in zip(need, force_answers(model, tok, [(r["prompt"], r["text"]) for r in need], 8)):
            r.update(answer=o["answer"], n_gen=r["n_gen"] + o["n_gen"], forced=True)
    with open(out / "midhop_restarts.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    (out / "parity.json").write_text(json.dumps(parity))


# ─────────────────────────────────────────────────────────────────────────────
# report (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def report(out: Path, probe_dir: Path, run: Path):
    from eval_robustness import DATASETS, Run, arr
    from src.pipeline.evaluate import _paired_ci
    from src.pipeline.probes import ProbeBundle
    from src.pipeline.systems import outcome
    R = Run(run, force=True)
    pb = ProbeBundle(probe_dir, mode="shared")
    P = R.probe_scores(pb)
    pmeta = json.loads((out / "prefix_meta.json").read_text())
    sig = {r["uid"]: r["hops"] for r in read_jsonl(out / "midhop_signals.jsonl")}
    mid = {(r["uid"], r["k"], r["t"]): r for r in read_jsonl(out / "midhop_restarts.jsonl")}
    parity = json.loads((out / "parity.json").read_text())

    def run_system(system):
        res = {}
        for it in R.items:
            u, s0 = it["uid"], R.s0[it["uid"]]
            d = decide(system, u, R, P, pb.tau, sig, pmeta)
            base = dict(answer=s0["answer"], fired=False, mid=False, changed=False,
                        gen=s0["n_gen"], processed=s0["n_prompt"] + s0["n_gen"])
            if d is not None:
                k, t = d
                cached = R.rs[u][f"h{k}"]
                if t is None:
                    r, gen_stop, changed = cached, s0["boundaries"][str(k)]["n_tokens"], False
                else:
                    m = mid[(u, k, t)]
                    r = cached if m["same_as_cached"] else m
                    gen_stop, changed = sig[u][str(k)]["gen_before"] + t, not m["same_as_cached"]
                base = dict(answer=r["answer"], fired=True, mid=t is not None, changed=changed, gen=gen_stop + r["n_gen"],
                            processed=s0["n_prompt"] + gen_stop + r["n_prompt"] + r["n_gen"])
            base["correct"] = answer_correct(base["answer"], it["gold_answer"])
            res[u] = base
        return res

    S = {"deployed": run_system("deployed"), "hybrid": run_system("hybrid"), "online": run_system("online")}
    ref = {it["uid"]: outcome("probe_restart", it, R.s0[it["uid"]], R.rs[it["uid"]],
                              probe_scores={k: v for k, v in P[it["uid"]].items() if k in R.eligible(it["uid"])},
                              tau=pb.tau) for it in R.items}
    agree = np.mean([S["deployed"][u]["correct"] == bool(ref[u]["correct"]) for u in ref])
    none = {it["uid"]: answer_correct(R.s0[it["uid"]]["answer"], it["gold_answer"]) for it in R.items}
    names = {"deployed": "Deployed probe at `</hopK>` (headline)",
             "hybrid": "**Hybrid**: prefix probe (strict θ) mid-hop, else deployed at `</hopK>`",
             "online": f"Prefix probe only (F0.5 θ = {pmeta['th_f05']:.3f})"}
    L = [f"# Mid-hop stopping, end to end (`{run.name}`)", "",
         "One restart, correct = token F1 ≥ 0.5, paired bootstrap CIs over questions. A mid-hop stop restarts with the "
         "partial hop text in the retrieval query; restarts that retrieve the same paragraphs as the cached restart reuse "
         "its answer (identical prompt, greedy decoding), the others were generated (answers forced at the token cap, as "
         "in the pipeline).", "",
         f"Checks: retrieval parity on the run's cached hop restarts {parity['identical']}/{parity['n']} identical; "
         f"the deployed system here vs the pipeline's own accounting: {agree:.1%} of questions identical.", ""]
    for ds in DATASETS:
        its = [it for it in R.items if it["dataset"] == ds]
        c0 = np.array([none[it["uid"]] for it in its], float)
        cd = arr({u: dict(correct=v["correct"]) for u, v in S["deployed"].items()}, its)
        L += [f"## {ds} (n = {len(its)})", "",
              "| system | accuracy | Δ vs deployed [95% CI] | restart rate | of which mid-hop | mid-hop retrieval changed | "
              "fixed | broken | generated tokens / q | processed tokens / q |", "|---|---|---|---|---|---|---|---|---|---|"]
        for sname, res in S.items():
            c = np.array([res[it["uid"]]["correct"] for it in its], float)
            fired = [res[it["uid"]] for it in its if res[it["uid"]]["fired"]]
            midn = [x for x in fired if x["mid"]]
            d = "–" if sname == "deployed" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(c.mean() - cd.mean(), *_paired_ci(c, cd))
            L.append(f"| {names[sname]} | {c.mean():.3f} | {d} | {len(fired) / len(its):.2f} | "
                     f"{len(midn) / max(1, len(fired)):.0%} | {np.mean([x['changed'] for x in midn]) if midn else 0:.0%} | "
                     f"{int(((c == 1) & (c0 == 0)).sum())} | {int(((c == 0) & (c0 == 1)).sum())} | "
                     f"{np.mean([res[it['uid']]['gen'] for it in its]):.1f} | "
                     f"{np.mean([res[it['uid']]['processed'] for it in its]):.0f} |")
        L.append(f"| No intervention | {c0.mean():.3f} | | 0.00 | | | 0 | 0 | | |")
        L.append("")

    # does the shorter (partial-hop) query bring back the missing gold paragraph? 2Wiki only (gold titles known)
    from src.pipeline.common import load_setting_examples
    ex = {e["id"]: e for e in load_setting_examples("tfidf")}
    wiki = [it for it in R.items if it["dataset"] == "2wiki"]
    L += ["## Retrieval with the partial-hop query (2Wiki)", "",
          "Mid-hop stops of each system where at least one gold paragraph is missing from the first-stage context. "
          "'Full' = the cached restart's query (complete hop text, what the deployed probe would have used); "
          "'partial' = the query actually used after the mid-hop stop. Found = a missing gold paragraph is among the 3 "
          "retrieved.", "",
          "| system | mid-hop stops (gold missing) | full query finds it | partial query finds it | both | only full | only partial | "
          "accuracy if partial found / not found |", "|---|---|---|---|---|---|---|---|"]
    for sname in ("hybrid", "online"):
        full_f, part_f, corr = [], [], []
        for it in wiki:
            u = it["uid"]
            d = decide(sname, u, R, P, pb.tau, sig, pmeta)
            if d is None or d[1] is None:
                continue
            k, t = d
            missing = set(ex[it["id"]].get("gold_titles") or []) - set(it["context_titles"])
            if not missing:
                continue
            full_f.append(bool(missing & set(R.rs[u][f"h{k}"]["retrieved_titles"])))
            part_f.append(bool(missing & set(mid[(u, k, t)]["retrieved_titles"])))
            corr.append(bool(S[sname][u]["correct"]))
        if not full_f:
            continue
        full_f, part_f, corr = (np.array(x) for x in (full_f, part_f, corr))
        acc_f = corr[part_f].mean() if part_f.any() else float("nan")
        acc_n = corr[~part_f].mean() if (~part_f).any() else float("nan")
        L.append(f"| {names[sname]} | {len(full_f)} | {full_f.mean():.1%} | {part_f.mean():.1%} | "
                 f"{int((full_f & part_f).sum())} | {int((full_f & ~part_f).sum())} | {int((~full_f & part_f).sum())} | "
                 f"{acc_f:.3f} / {acc_n:.3f} |")
    L.append("")
    (out / "midhop_e2e.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prefix", "score", "restart", "report", "all"], default="all")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="outputs/pipeline/tfidf_final_lrC0.01_L12/midhop")
    ap.add_argument("--probe-dir", default=PROBE_DIR)
    args = ap.parse_args()
    out, probe_dir, run = Path(args.out), ROOT / args.probe_dir, Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    st = args.stage

    def free_gpu():                                           # each GPU stage loads its own model copy
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    if st in ("prefix", "all"):
        import online_probe
        if not (out / "prefix_train.npy").exists():
            online_probe.extract(out, probe_dir, None)
            free_gpu()
        online_probe.analyze(out, probe_dir)                  # fits + saves the prefix probe, writes online_probe.md
    if st in ("score", "all"):
        score(out, probe_dir, run)
        free_gpu()
    if st in ("restart", "all"):
        restart(out, probe_dir, run)
        free_gpu()
    if st in ("report", "all"):
        report(out, probe_dir, run)


if __name__ == "__main__":
    main()
