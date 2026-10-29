"""
src/pipeline/run.py
───────────────────
Runs the Phase 3–5 pipeline stage by stage.  Every stage appends to its own
jsonl file and skips items already done, so a crashed or timed-out Kaggle
session resumes where it stopped.  --shard/--num-shards splits the GPU stages
across parallel sessions (copy all shard outputs into one --out dir before
`evaluate`).

Stages (default: all, in this order)
──────
probes    train hop-1/hop-2 probes + CV thresholds (CPU; skipped if --probe-dir has them)
evalset   2Wiki held-out test questions (natural) + HotpotQA OOD questions
corpus    retrieval corpus + embeddings
s0        uninterrupted trajectory per question, probe in monitor mode   (GPU)
judge     post-hoc LLM judge on hop 1 / hop 2                             (GPU)
restarts  retrieval + restart for keys q / h1 / h2, batched               (GPU)
rescore   probe + judge scores on the h1 / h2 restarts (second-round triggers)   (GPU, --rounds 2)
restarts2 second retrieval + restart where a restart fails again, keys "h1>2" … (GPU, --rounds 2)
force     an answer for every generation that hit the token cap without <answer> (GPU)
judges    the same post-hoc judge with other models (--judge-models), on the same hops (GPU, T4 x2 for 7-8B)
live      latency subset: truly live probe-halt-retrieve-restart, etc.   (GPU)
parity    live probe features vs the stored Phase 1 features             (CPU)
evaluate  metrics, tables, figures                                        (CPU)

Repair versions (--repair)
    v1  query = question + failing hop text; dense top-3
    v2  hop-2 queries lead with the hop-1 bridge entity; dense top-20 re-ranked by a cross-encoder -> top-3
--reuse-from DIR copies the eval set, trajectories (s0), judge scores and features of an earlier run,
so a new repair / rounds setting only re-runs the restarts.

Example (Kaggle, T4):
    python -m src.pipeline.run --setting tfidf --out /kaggle/working/pipeline_tfidf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np

from src.pipeline.common import (EXTRA_JUDGES, MIRRORS, MODEL_NAME, SETTINGS, append_jsonl, context_titles, download_hidden_states,
                                 format_passages, hop_label, hop_text, load_setting_examples,
                                 prompt_from_template, prompt_with_evidence, raw_passages, read_jsonl,
                                 repair_query, tfidf_rank, write_jsonl, hop_order_condition)

ALL_STAGES = ["probes", "evalset", "corpus", "s0", "judge", "restarts", "rescore", "restarts2", "force", "live",
              "parity", "judges", "evaluate"]
REUSABLE = ["eval_set.jsonl", "hotpot_passages.jsonl", "s0.jsonl", "s0.shard*.jsonl", "judge.jsonl",
            "judge.shard*.jsonl", "judge__*.jsonl", "s0_feats", "parity.json", "corpus"]


def safe(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def done_uids(path: Path, key=lambda r: r["uid"]) -> set:
    return {key(r) for r in read_jsonl(path)}


def merged(out: Path, stem: str) -> list[dict]:
    """All records of one stage: <stem>.jsonl plus <stem>.shardN.jsonl (not other stems sharing a prefix)."""
    recs = []
    for p in sorted([*out.glob(f"{stem}.jsonl"), *out.glob(f"{stem}.shard*.jsonl")]):
        recs += read_jsonl(p)
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Eval set
# ─────────────────────────────────────────────────────────────────────────────

def load_hotpot(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = None
    for name in ("hotpotqa/hotpot_qa", "hotpot_qa"):
        try:
            ds = load_dataset(name, "distractor", split="validation")
            break
        except Exception as e:                                   # noqa: BLE001
            print(f"[evalset] could not load {name}: {e.__class__.__name__}: {str(e)[:120]}")
    if ds is None:
        raise RuntimeError("HotpotQA could not be loaded")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def build_eval_set(setting: str, out: Path, meta: dict, n_2wiki: int, n_hotpot: int, seed: int = 42):
    top_k = SETTINGS[setting]["context_top_k"]
    examples = load_setting_examples(setting)
    test_ids = set(meta["test_ids"])
    pool = sorted((e for e in examples if e["id"] in test_ids and not e.get("is_counterfactual")),
                  key=lambda e: e["id"])
    random.Random(seed).shuffle(pool)
    items = []
    for e in pool[:n_2wiki]:
        graph = {n["hop"]: n.get("gold_entity", "") for n in e.get("reasoning_graph", [])}
        labels = {str(k): hop_label(e, k) for k in (1, 2)}
        items.append(dict(uid=f"2wiki:{e['id']}", dataset="2wiki", id=e["id"], question=e["question"],
                          gold_answer=e["gold_answer"], prompt=e["prompt"], context=e["context"],
                          context_titles=context_titles(e["context"]), labels=labels,
                          gold_entities={str(k): graph.get(k, "") for k in (1, 2)},
                          first_fail=e.get("first_fail_hop"), stored_cot=e.get("generated_cot", ""),
                          retrieval_condition=(hop_order_condition(e, context_titles(e["context"]))
                                               if e.get("gold_titles") else "all_passages")))
    template = pool[0]["prompt"]
    hot_passages = []
    for rec in load_hotpot(n_hotpot, seed):
        passages = raw_passages(rec)
        sel = passages if top_k >= len(passages) else \
            [passages[i] for i in tfidf_rank(rec["question"], [p["text"] for p in passages], top_k)]
        gold_titles = set(rec["supporting_facts"]["title"]) if isinstance(rec.get("supporting_facts"), dict) else set()
        sel_titles = {p["title"] for p in sel}
        n_gold_in = len(gold_titles & sel_titles)
        cond = "both_present" if n_gold_in >= 2 else ("one_present" if n_gold_in == 1 else "neither")
        ctx = format_passages(sel)
        items.append(dict(uid=f"hotpot:{rec['id']}", dataset="hotpotqa", id=rec["id"], question=rec["question"],
                          gold_answer=rec["answer"], prompt=prompt_from_template(template, ctx, rec["question"]),
                          context=ctx, context_titles=[p["title"] for p in sel], labels={}, gold_entities={},
                          first_fail=None, stored_cot="", retrieval_condition=cond,
                          hotpot_type=rec.get("type"), hotpot_level=rec.get("level")))
        hot_passages.append(dict(uid=f"hotpot:{rec['id']}", all_passages=passages))
    write_jsonl(out / "eval_set.jsonl", items)
    write_jsonl(out / "hotpot_passages.jsonl", hot_passages)
    print(f"[evalset] {sum(i['dataset'] == '2wiki' for i in items)} 2Wiki test + "
          f"{sum(i['dataset'] == 'hotpotqa' for i in items)} HotpotQA items")
    return items


def prompt_date(prompt: str) -> str | None:
    m = re.search(r"Today Date: (.+)\n", prompt)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="tfidf", choices=list(SETTINGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--probe-dir", default=None, help="default: <out>/probes")
    ap.add_argument("--probe-sweep", default="none", help="layer sweep when training probes here: none | all")
    ap.add_argument("--n-2wiki", type=int, default=500)
    ap.add_argument("--n-hotpot", type=int, default=500)
    ap.add_argument("--n-live", type=int, default=100, help="latency/equivalence subset size (half per dataset)")
    ap.add_argument("--n-extra-corpus", type=int, default=20000, help="extra random 2Wiki questions in the corpus")
    ap.add_argument("--top-k", type=int, default=3, help="passages retrieved per restart")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repair", default="v2", choices=["v1", "v2"])
    ap.add_argument("--rerank-pool", type=int, default=20, help="candidates re-ranked by the cross-encoder (v2)")
    ap.add_argument("--retriever", default="dense", choices=["dense", "hybrid"],
                    help="hybrid = dense + BM25 keyword search fused by reciprocal rank (finds exact names)")
    ap.add_argument("--rounds", type=int, default=2, choices=[1, 2], help="max restarts per question")
    ap.add_argument("--reuse-from", default=None, help="earlier run dir: reuse its eval set, s0, judge, features")
    ap.add_argument("--judge-models", default=",".join(EXTRA_JUDGES),
                    help="extra post-hoc judges (comma-separated HF ids; '' = none)")
    ap.add_argument("--probe-mode", default="shared", choices=["shared", "dedicated"],
                    help="shared: one probe + one τ for both hops (main); dedicated: hop-1 and hop-2 probes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir) if args.probe_dir else out / "probes"
    # fingerprint of the probe bundle: second-round decisions are only reusable with the same probes
    meta_file = probe_dir / "probes_meta.json"
    args.probe_id = hashlib.sha1(meta_file.read_bytes()).hexdigest()[:12] if meta_file.exists() else None
    if args.reuse_from:
        src = Path(args.reuse_from)
        prev = json.loads((src / "run_args.json").read_text()) if (src / "run_args.json").exists() else {}
        pats = list(REUSABLE)
        same_repair = ((prev.get("repair", "v1"), prev.get("retriever", "dense"), prev.get("top_k", 3))
                       == (args.repair, args.retriever, args.top_k))
        # runs made before probe fingerprints existed used the default bundle at outputs/pipeline/<setting>/probes
        same_probe = (prev.get("probe_id") == args.probe_id if "probe_id" in prev
                      else Path(prev.get("probe_dir", "")).name == probe_dir.name == "probes")
        if same_repair:                                             # restarts depend on the repair only
            pats += ["restarts.jsonl", "restarts.shard*.jsonl"]
            if prev.get("probe_mode", "dedicated") == args.probe_mode and same_probe:  # round 2 needs the same probe
                pats += ["rescore.jsonl", "rescore.shard*.jsonl", "restarts2.jsonl", "restarts2.shard*.jsonl"]
        for pat in pats:
            for p in src.glob(pat):
                dst = out / p.name
                if dst.exists():
                    continue
                (shutil.copytree if p.is_dir() else shutil.copy2)(p, dst)
                print(f"[reuse] {p} -> {dst}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.rounds == 1:
        stages = [s for s in stages if s not in ("rescore", "restarts2")]
    pool = args.rerank_pool if args.repair == "v2" else 0
    sfx = f".shard{args.shard}" if args.num_shards > 1 else ""
    token = os.environ.get("HF_TOKEN")
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=1))

    # ── probes ────────────────────────────────────────────────────────────────
    if "probes" in stages and not (probe_dir / "probes_meta.json").exists():
        from src.pipeline.probes import train_probes
        train_probes(args.setting, probe_dir, sweep=args.probe_sweep)
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    if meta["setting"] != args.setting:
        raise SystemExit(f"probes in {probe_dir} were trained for setting {meta['setting']!r}, not {args.setting!r}")

    # ── eval set ─────────────────────────────────────────────────────────────
    if "evalset" in stages and not (out / "eval_set.jsonl").exists():
        build_eval_set(args.setting, out, meta, args.n_2wiki, args.n_hotpot, args.seed)
    items = read_jsonl(out / "eval_set.jsonl")
    mine = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    by_uid = {it["uid"]: it for it in items}
    print(f"[run] setting={args.setting} items={len(items)} this shard={len(mine)}")

    # ── corpus ───────────────────────────────────────────────────────────────
    retriever = None

    def get_retriever():
        nonlocal retriever
        if retriever is None:
            from src.pipeline.retriever import DenseRetriever, build_corpus
            cdir = out / "corpus"
            if (cdir / "embeddings.npy").exists():
                retriever = DenseRetriever.load(cdir)
            else:
                ids = {e["id"] for e in load_setting_examples(args.setting)}
                corpus = build_corpus(ids, read_jsonl(out / "hotpot_passages.jsonl"), args.n_extra_corpus, args.seed)
                retriever = DenseRetriever(corpus)
                retriever.save(cdir)
        return retriever

    if "corpus" in stages:
        get_retriever()

    # ── model (GPU stages) ───────────────────────────────────────────────────
    gpu_stages = {"s0", "judge", "restarts", "rescore", "restarts2", "force", "live"} & set(stages)
    model = tok = probes = None
    if gpu_stages:
        from src.pipeline.generation import load_model
        from src.pipeline.probes import ProbeBundle
        model, tok = load_model(args.model, token)
        probes = ProbeBundle(probe_dir, mode=args.probe_mode)
        print(f"[run] probe mode {args.probe_mode}: layers {probes.layer}  thresholds {probes.tau}")

    # ── s0: monitored trajectories ───────────────────────────────────────────
    s0_path = out / f"s0{sfx}.jsonl"
    if "s0" in stages:
        from src.pipeline.generation import generate_monitored
        feats_dir = out / "s0_feats"
        feats_dir.mkdir(exist_ok=True)
        done = done_uids(s0_path)
        todo = [it for it in mine if it["uid"] not in done]
        t0 = time.time()
        for n, it in enumerate(todo):
            r = generate_monitored(model, tok, it["prompt"], probes, halt=False, max_new_tokens=args.max_new_tokens)
            pooled = r.pop("_pooled")
            np.savez_compressed(feats_dir / f"{safe(it['uid'])}.npz", **{f"h{k}": v for k, v in pooled.items()})
            r.update(uid=it["uid"], hop_text={str(k): hop_text(r["text"], k) for k in (1, 2)},
                     reproduced=(r["text"].strip() == it["stored_cot"].strip()) if it["stored_cot"] else None)
            append_jsonl(s0_path, r)
            if n % 25 == 0:
                pf = {k: round(v["p_fail"], 3) for k, v in r["boundaries"].items() if v["p_fail"] is not None}
                print(f"[s0] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s) n_gen={r['n_gen']} p_fail={pf}", flush=True)

    s0_all = {r["uid"]: r for r in merged(out, "s0")}

    # ── judge ────────────────────────────────────────────────────────────────
    if "judge" in stages:
        from src.pipeline.judge import Judge
        judge = Judge(model, tok, prompt_date(items[0]["prompt"]))
        j_path = out / f"judge{sfx}.jsonl"
        done = done_uids(j_path)
        for n, it in enumerate([it for it in mine if it["uid"] in s0_all and it["uid"] not in done]):
            s0 = s0_all[it["uid"]]
            rec = {"uid": it["uid"]}
            for k in ("1", "2"):
                if k in s0["boundaries"] and s0["hop_text"][k]:
                    rec[k] = judge.p_fail(it["context"], it["question"], s0["hop_text"][k])
            append_jsonl(j_path, rec)
            if n % 100 == 0:
                print(f"[judge] {n + 1} done", flush=True)

    # ── restarts ─────────────────────────────────────────────────────────────
    def restart_requests(it, s0):
        reqs = [("q", repair_query(it["question"], {}, 0))]
        for k in ("1", "2"):
            if k in s0["boundaries"] and s0["hop_text"][k]:
                reqs.append((f"h{k}", repair_query(it["question"], s0["hop_text"], int(k), args.repair)))
        return reqs

    def run_restarts(pending, path, tag):
        from src.pipeline.generation import generate_batch
        print(f"[{tag}] {len(pending)} generations to run")
        t0 = time.time()
        chunk = 8 * args.batch_size
        for s in range(0, len(pending), chunk):
            part = pending[s:s + chunk]
            outs = generate_batch(model, tok, [p["prompt"] for p in part], args.max_new_tokens, args.batch_size)
            for p, o in zip(part, outs):
                p = {k: v for k, v in p.items() if k != "prompt"}
                append_jsonl(path, {**p, **o})
            print(f"[{tag}] {min(s + chunk, len(pending))}/{len(pending)} ({time.time() - t0:.0f}s)", flush=True)

    def retrieve(it, query, exclude):
        hits, ms = get_retriever().search(query, args.top_k, exclude_titles=exclude, rerank_pool=pool,
                                          hybrid=args.retriever == "hybrid")
        return hits, ms

    if "restarts" in stages:
        r_path = out / f"restarts{sfx}.jsonl"
        done = done_uids(r_path, key=lambda r: (r["uid"], r["key"]))
        pending = []
        for it in mine:
            if it["uid"] not in s0_all:
                continue
            for key, query in restart_requests(it, s0_all[it["uid"]]):
                if (it["uid"], key) not in done:
                    hits, ms = retrieve(it, query, it["context_titles"])
                    pending.append(dict(uid=it["uid"], key=key, query=query, retrieved_titles=[h["title"] for h in hits],
                                        retrieved=[{"title": h["title"], "text": h["text"]} for h in hits],
                                        retrieval_ms=ms, repair=args.repair, prompt=prompt_with_evidence(it["prompt"], hits)))
        run_restarts(pending, r_path, "restarts")

    def round1_prompt(it, r):
        return prompt_with_evidence(it["prompt"], r["retrieved"])

    # ── rescore: second-round trigger scores on the h1 / h2 restarts ─────────
    if "rescore" in stages:
        from src.pipeline.generation import score_generation
        from src.pipeline.judge import Judge
        judge = Judge(model, tok, prompt_date(items[0]["prompt"]))
        rs_path = out / f"rescore{sfx}.jsonl"
        done = done_uids(rs_path, key=lambda r: (r["uid"], r["key"]))
        mine_uids = {it["uid"] for it in mine}
        todo = [r for r in merged(out, "restarts") if r["key"] in ("h1", "h2") and r["uid"] in mine_uids
                and (r["uid"], r["key"]) not in done and "retrieved" in r]
        print(f"[rescore] {len(todo)} restarts to score")
        t0 = time.time()
        for n, r in enumerate(todo):
            it = by_uid[r["uid"]]
            hops = {k: hop_text(r["text"], int(k)) for k in ("1", "2")}
            rec = dict(uid=r["uid"], key=r["key"], hop_text=hops,
                       probe=score_generation(model, tok, round1_prompt(it, r), r["text"], probes), judge={})
            ctx = format_passages(r["retrieved"]) + "\n\n" + it["context"]
            for k in ("1", "2"):
                if hops[k]:
                    rec["judge"][k] = judge.p_fail(ctx, it["question"], hops[k])
            append_jsonl(rs_path, rec)
            if n % 200 == 0:
                print(f"[rescore] {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    # ── restarts2: a second retrieval + restart where the first restart fails again ──
    if "restarts2" in stages:
        from src.pipeline.systems import judge_trigger, probe_trigger
        r1 = {(r["uid"], r["key"]): r for r in merged(out, "restarts")}
        r2_path = out / f"restarts2{sfx}.jsonl"
        done = done_uids(r2_path, key=lambda r: (r["uid"], r["key"]))
        mine_uids = {it["uid"] for it in mine}
        pending = []
        for sc in merged(out, "rescore"):
            if sc["uid"] not in mine_uids:
                continue
            ks = {probe_trigger({int(k): v["p_fail"] for k, v in sc["probe"].items()}, probes.tau),
                  judge_trigger(sc["judge"])} - {None}
            for j in sorted(ks):
                key = f"{sc['key']}>{j}"
                if (sc["uid"], key) in done:
                    continue
                it, parent = by_uid[sc["uid"]], r1[(sc["uid"], sc["key"])]
                query = repair_query(it["question"], sc["hop_text"], j, args.repair)
                hits, ms = retrieve(it, query, it["context_titles"] + parent["retrieved_titles"])
                pending.append(dict(uid=sc["uid"], key=key, query=query, retrieved_titles=[h["title"] for h in hits],
                                    retrieved=[{"title": h["title"], "text": h["text"]} for h in hits],
                                    retrieval_ms=ms, repair=args.repair,
                                    prompt=prompt_with_evidence(round1_prompt(it, parent), hits)))
        run_restarts(pending, r2_path, "restarts2")

    # ── force: answers for generations that hit the token cap without <answer> ──
    if "force" in stages:
        from src.pipeline.generation import force_answers
        f_path = out / f"forced{sfx}.jsonl"
        done = done_uids(f_path, key=lambda r: (r["uid"], r["key"]))
        mine_uids = {it["uid"] for it in mine}
        r1 = {(r["uid"], r["key"]): r for r in merged(out, "restarts")}
        todo = []                                            # (uid, key, generation prompt, text)
        for uid in mine_uids & set(s0_all):
            if not s0_all[uid]["answer"] and (uid, "s0") not in done:
                todo.append((uid, "s0", by_uid[uid]["prompt"], s0_all[uid]["text"]))
        for r in merged(out, "restarts") + merged(out, "restarts2"):
            if r["uid"] not in mine_uids or r["answer"] or (r["uid"], r["key"]) in done or "retrieved" not in r:
                continue
            it = by_uid[r["uid"]]
            base = it["prompt"] if ">" not in r["key"] else round1_prompt(it, r1[(r["uid"], r["key"].split(">")[0])])
            todo.append((r["uid"], r["key"], prompt_with_evidence(base, r["retrieved"]), r["text"]))
        print(f"[force] {len(todo)} generations without an <answer>")
        for s in range(0, len(todo), 64):
            part = todo[s:s + 64]
            outs = force_answers(model, tok, [(p, t) for _, _, p, t in part], args.batch_size)
            for (uid, key, _, _), o in zip(part, outs):
                append_jsonl(f_path, dict(uid=uid, key=key, **o))

    # ── live latency / equivalence subset ────────────────────────────────────
    if "live" in stages:
        from src.pipeline.generation import force_answers, generate_monitored
        from src.pipeline.systems import judge_trigger
        ret = get_retriever()
        j_all = {r["uid"]: r for r in merged(out, "judge")}
        l_path = out / f"live{sfx}.jsonl"
        done = done_uids(l_path)
        subset = []
        for ds in ("2wiki", "hotpotqa"):
            subset += [it for it in mine if it["dataset"] == ds and it["uid"] in s0_all][: args.n_live // 2]

        def finish(prompt, g):
            """The final answer, forced from hops 1-2 if the generation never wrote <answer> (as in `force`)."""
            if g["answer"] or "force" not in stages:
                return g["answer"]
            return force_answers(model, tok, [(prompt, g["text"])], 1)[0]["answer"]

        def seq_restart(it, query):
            t = time.perf_counter()
            hits, ms = retrieve(it, query, it["context_titles"])
            prompt_r = prompt_with_evidence(it["prompt"], hits)
            g = generate_monitored(model, tok, prompt_r, None, max_new_tokens=args.max_new_tokens, keep_pooled=False)
            return dict(answer=finish(prompt_r, g), text=g["text"], n_gen=g["n_gen"], retrieval_ms=ms,
                        wall_ms=(time.perf_counter() - t) * 1000)

        for it in [it for it in subset if it["uid"] not in done]:
            rec = {"uid": it["uid"]}
            # ours, truly live: generate -> probe halts at </hopK> -> retrieve -> flush -> restart (up to --rounds)
            t = time.perf_counter()
            prompt, exclude = it["prompt"], list(it["context_titles"])
            g = generate_monitored(model, tok, prompt, probes, halt=True,
                                   max_new_tokens=args.max_new_tokens, keep_pooled=False)
            first, stops, n_gen = g, [g["stopped_at"]], g["n_gen"]
            for r in range(1, args.rounds + 1):
                if not g["stopped_at"]:
                    break
                hops = {k: hop_text(g["text"], int(k)) for k in ("1", "2")}
                hits, _ = retrieve(it, repair_query(it["question"], hops, g["stopped_at"], args.repair), exclude)
                exclude += [h["title"] for h in hits]
                prompt = prompt_with_evidence(prompt, hits)
                last = r == args.rounds
                g = generate_monitored(model, tok, prompt, None if last else probes, halt=not last,
                                       max_new_tokens=args.max_new_tokens, keep_pooled=False)
                stops.append(g["stopped_at"])
                n_gen += g["n_gen"]
            rec["probe_restart"] = dict(stopped_at=first["stopped_at"], stops=stops, rounds=args.rounds,
                                        first_pass_text=first["text"], first_pass_n_gen=first["n_gen"],
                                        answer=finish(prompt, g), n_gen=n_gen, wall_ms=(time.perf_counter() - t) * 1000)
            rec["always_retrieve"] = seq_restart(it, it["question"])
            jk = judge_trigger(j_all.get(it["uid"]))
            if jk:
                rec["judge_restart"] = seq_restart(it, repair_query(it["question"], s0_all[it["uid"]]["hop_text"],
                                                                    jk, args.repair))
            append_jsonl(l_path, rec)
        print(f"[live] {len(read_jsonl(l_path))} items in latency subset")

    # ── parity: live features vs stored Phase 1 features ─────────────────────
    if "parity" in stages:
        import torch
        from src.pipeline.probes import ProbeBundle
        pb = probes or ProbeBundle(probe_dir, mode=args.probe_mode)
        repro = [it for it in items if it["dataset"] == "2wiki" and s0_all.get(it["uid"], {}).get("reproduced")]
        repro = repro[:200]
        res = dict(n_2wiki_s0=sum(1 for it in items if it["dataset"] == "2wiki" and it["uid"] in s0_all),
                   n_reproduced=sum(1 for it in items if s0_all.get(it["uid"], {}).get("reproduced")), rows=[])
        if repro:
            hs_dir = download_hidden_states(args.setting, [it["id"] for it in repro])
            ex_by_id = {e["id"]: e for e in load_setting_examples(args.setting)}
            for it in repro:
                f = out / "s0_feats" / f"{safe(it['uid'])}.npz"
                if not f.exists():
                    continue
                live = np.load(f)
                stored = torch.load(hs_dir / f"{it['id']}.pt", map_location="cpu", weights_only=False)["pooled"]
                order = [h["hop_idx"] for h in ex_by_id[it["id"]]["hop_spans"]]
                for k in (1, 2):
                    if f"h{k}" not in live.files or k not in order:
                        continue
                    a = live[f"h{k}"].astype(np.float32)
                    b = stored[:, order.index(k)].float().numpy()
                    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                    L = pb.layer[k]
                    res["rows"].append(dict(uid=it["uid"], hop=k, cos_probe_layer=float(cos[L]),
                                            cos_min=float(cos.min()), p_live=pb.p_fail(k, live[f"h{k}"]),
                                            p_stored=pb.p_fail(k, b.astype(np.float16))))
        (out / "parity.json").write_text(json.dumps(res, indent=1))
        if res["rows"]:
            dp = np.array([abs(r["p_live"] - r["p_stored"]) for r in res["rows"]])
            print(f"[parity] {len(res['rows'])} hops: mean cos@probe layer "
                  f"{np.mean([r['cos_probe_layer'] for r in res['rows']]):.5f}, |Δp| mean {dp.mean():.4f} max {dp.max():.4f}")
        print(f"[parity] reproduced Phase 1 trajectory exactly: {res['n_reproduced']}/{res['n_2wiki_s0']}")

    # ── judges: the post-hoc judge with other (larger) models, same hop texts ──
    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    if "judges" in stages and judge_models:
        import gc
        import shutil as _sh
        import torch
        from huggingface_hub.constants import HF_HUB_CACHE
        from src.pipeline.generation import load_model
        from src.pipeline.judge import Judge
        model = tok = None                       # free the generator; judges load one at a time
        gc.collect()
        torch.cuda.empty_cache()
        for name in judge_models:
            j_path = out / f"judge__{safe(name)}{sfx}.jsonl"
            done = done_uids(j_path)
            todo = [it for it in mine if it["uid"] in s0_all and it["uid"] not in done]
            if not todo:
                continue
            print(f"[judges] {name}: {len(todo)} items", flush=True)
            try:
                jm, jt = load_model(name, token, spread=True)
            except Exception as e:                                         # noqa: BLE001
                print(f"[judges] skipping {name}: {e.__class__.__name__}: {str(e)[:200]}")
                continue
            judge = Judge(jm, jt, prompt_date(items[0]["prompt"]) if "llama" in name.lower() else None)
            t0 = time.time()
            for n, it in enumerate(todo):
                s0 = s0_all[it["uid"]]
                rec = {"uid": it["uid"], "model": name}
                for k in ("1", "2"):
                    if k in s0["boundaries"] and s0["hop_text"][k]:
                        rec[k] = judge.p_fail(it["context"], it["question"], s0["hop_text"][k])
                append_jsonl(j_path, rec)
                if n % 200 == 0:
                    print(f"[judges] {name}: {n + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
            del judge, jm, jt
            gc.collect()
            torch.cuda.empty_cache()
            for repo in {name, *MIRRORS.get(name, [])}:                # free Kaggle disk for the next judge
                _sh.rmtree(Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")), ignore_errors=True)

    # ── evaluate ─────────────────────────────────────────────────────────────
    if "evaluate" in stages:
        from src.pipeline.evaluate import evaluate
        evaluate(out, probe_dir, args.probe_mode)


if __name__ == "__main__":
    main()
