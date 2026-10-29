"""
generate_counterfactuals_tfidf.py
─────────────────────────────────
GPU step 2 of the TF-IDF counterfactuals (step 1: scripts/make_cf_candidates.py).
Replaces generate_500_counterfactuals.py, whose output (augmented_500cf.jsonl) is
not usable: hop text never updated, hop-2 labels copied from the seed, memory
check read a missing field, entity type ignored, full10 context, no hidden states.

For each (seed, donor) candidate, in order (test-split seeds first):
  1. generate from the CF prompt exactly as Phase 1 did (batched, left padding,
     greedy, 400 new tokens, decode without special tokens);
  2. label every hop with the Phase 1 labeler (phase1_dataset.label_hops.label_example)
     against the DONOR's gold chain, with the TF-IDF run's matcher settings
     (SBERT 0.70, LLM fallback = the same model, budget 1.0) plus Wikidata aliases;
  3. accept only an induced hop-1 failure:
       reject  no <hop1> tag                       (format)
       reject  hop 1 states the donor's hop-1 gold  (model knows the answer: not a failure)
       reject  final answer correct for the donor   (parametric memory)
     a rejected seed retries with its next donor;
  4. for accepted CFs, extract hidden states with Phase 1's own
     phase1_dataset.generate_cot.extract_hidden_states (28 layers, mean-pooled per
     hop, same .pt format as tfidf_distractor_rag/data/hidden_states).

Self-checks before any long work (abort on failure):
  * hidden-state parity: recompute the stored .pt of a few natural seeds with the
    same function, cosine must be >= 0.999 at layers 0 / 10 / 27;
  * token alignment of every CF prompt with its seed prompt (same length).
Reported, not enforced: how often re-generating a seed prompt reproduces its stored
trajectory (batching changes fp16 numerics slightly; Phase 3-5 saw 476/500).

Outputs (--out):
  counterfactuals.jsonl  accepted CF records (same schema as the TF-IDF augmented.jsonl, plus
                         is_counterfactual / clean_pair_id / counterfactual_swap)
  hidden_states/<id>.pt  one file per accepted CF
  attempts.jsonl         every attempt with its outcome (resume log)
  aliases.json           Wikidata alias cache
  summary.json / summary.md

Resumable: re-running skips seeds that are finished.

Usage (Kaggle GPU):
  python generate_counterfactuals_tfidf.py --candidates data/cf_tfidf/candidates.jsonl --out results/cf_tfidf
"""

from __future__ import annotations

import argparse
import json
import time
import types
from collections import Counter, deque
from pathlib import Path

import torch

from lf_models import input_device, load_causal_lm, resolve_model
from phase1_dataset.generate_cot import extract_hidden_states, parse_hop_spans, parse_predicted_answer
from phase1_dataset.label_hops import label_example, make_llm_judge
from src.pipeline.common import SETTINGS, hf_file, is_hedge, read_jsonl
from utils.matching import EntityMatcher

N_LAYERS = 28
CHECK_LAYERS = (0, 10, 27)
HS_CFG = {"hidden_states": {"save_single_token": False}}     # as in the TF-IDF Phase 1 run


def jl_append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Generation (Phase 1 generate_cot_batch, but with the prompts given verbatim:
# re-applying the chat template would change the "Today Date" token)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(input_device(model))
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True) for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# Self-checks
# ─────────────────────────────────────────────────────────────────────────────

def hs_parity_check(model, tok, seeds: list[dict], tmp: Path) -> list[float]:
    """Recompute stored Phase 1 .pt files with the same function; return min cosine per seed."""
    s = SETTINGS["tfidf"]
    tmp.mkdir(parents=True, exist_ok=True)
    mins = []
    for e in seeds:
        ref = torch.load(hf_file(f"{s['hs_prefix']}/{e['id']}.pt", s["revision"]), map_location="cpu",
                         weights_only=False)["pooled"].float()
        (tmp / f"{e['id']}.pt").unlink(missing_ok=True)
        extract_hidden_states(e, types.SimpleNamespace(model=model), tok, HS_CFG, tmp, list(range(N_LAYERS)))
        new = torch.load(tmp / f"{e['id']}.pt", map_location="cpu", weights_only=False)["pooled"].float()
        assert new.shape == ref.shape, f"shape {tuple(new.shape)} vs stored {tuple(ref.shape)}"
        cos = [float(torch.nn.functional.cosine_similarity(new[L, h], ref[L, h], dim=0))
               for L in CHECK_LAYERS for h in range(ref.shape[1]) if ref[L, h].abs().sum() > 0]
        mins.append(min(cos))
    return mins


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/cf_tfidf/candidates.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=500, help="stop after this many accepted CFs")
    ap.add_argument("--max-tries-per-seed", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--no-aliases", action="store_true", help="do not query Wikidata (labels without aliases)")
    ap.add_argument("--n-parity", type=int, default=3, help="seeds for the hidden-state parity check")
    ap.add_argument("--n-repro", type=int, default=8, help="seeds for the generation-reproduction report")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    hs_dir = out / "hidden_states"
    hs_dir.mkdir(parents=True, exist_ok=True)
    att_path, cf_path, alias_path = out / "attempts.jsonl", out / "counterfactuals.jsonl", out / "aliases.json"

    cands = read_jsonl(args.candidates)
    s = SETTINGS["tfidf"]
    natural = {e["id"]: e for e in read_jsonl(hf_file(s["data_file"], s["revision"]))}
    print(f"candidates: {len(cands)} seeds ({Counter(c['split'] for c in cands)}); natural examples: {len(natural)}")

    repo, msg = resolve_model(args.model, args.hf_token)
    print(msg)
    if repo is None:
        raise SystemExit("model not reachable")
    model, tok = load_causal_lm(repo, "float16", token=args.hf_token)
    tok.padding_side = "left"                                  # as Phase 1
    assert model.config.num_hidden_layers == N_LAYERS

    # ── self-check 1: hidden-state parity with the stored Phase 1 files ───────
    par = hs_parity_check(model, tok, [natural[c["seed_id"]] for c in cands[:args.n_parity]], out / "_parity")
    print(f"HIDDEN-STATE PARITY (min cosine per seed, layers {CHECK_LAYERS}): {[round(x, 5) for x in par]}")
    if min(par) < 0.999:
        raise SystemExit("hidden-state parity FAILED: this environment does not reproduce the stored Phase 1 states")

    # ── self-check 2: every CF prompt token-aligned with its seed prompt ──────
    bad = sum(len(tok(d["cf_prompt"]).input_ids) != len(tok(natural[c["seed_id"]]["prompt"]).input_ids)
              for c in cands for d in c["donors"])
    print(f"TOKEN ALIGNMENT: {bad} misaligned CF prompts out of {sum(len(c['donors']) for c in cands)}")
    if bad:
        raise SystemExit("token alignment FAILED (candidates built with a different tokenizer?)")

    # ── report: does re-generating a seed reproduce its stored trajectory? ────
    rs = [natural[c["seed_id"]] for c in cands[:args.n_repro]]
    regen = generate(model, tok, [e["prompt"] for e in rs], args.max_new_tokens)
    n_same = sum(g.strip() == e["generated_cot"].strip() for g, e in zip(regen, rs))
    print(f"GENERATION REPRODUCTION: {n_same}/{len(rs)} seed trajectories identical to the stored ones "
          f"(informational; fp16 batching can change a few)")

    # ── matcher: Phase 1 TF-IDF settings + aliases fetched for the donors actually tried ──
    aliases = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    use_aliases = not args.no_aliases
    if use_aliases:
        from utils.wikidata_aliases import _query_wikidata
        if not _query_wikidata("United States"):
            print("!" * 90 + "\nWARNING: Wikidata unreachable from here -> labeling WITHOUT aliases "
                  "(as the TF-IDF Phase 1 run did)\n" + "!" * 90)
            use_aliases = False
    matcher = EntityMatcher(aliases=aliases, sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
                            sbert_threshold=0.70, llm_judge=make_llm_judge(model, tok, input_device(model)),
                            llm_budget=1.0)

    def ensure_aliases(names):
        new = [x for x in names if x and x not in aliases]
        for x in new:
            aliases[x] = _query_wikidata(x)
            time.sleep(0.25)
        if new:
            alias_path.write_text(json.dumps(aliases, ensure_ascii=False), encoding="utf-8")

    # ── resume ────────────────────────────────────────────────────────────────
    tries, done_seeds = Counter(), set()
    n_acc = 0
    for a in read_jsonl(att_path):
        tries[a["seed_id"]] += 1
        if a["status"] == "accepted":
            done_seeds.add(a["seed_id"]); n_acc += 1
    for c in cands:
        if tries[c["seed_id"]] >= min(args.max_tries_per_seed, len(c["donors"])):
            done_seeds.add(c["seed_id"])
    queue = deque(c for c in cands if c["seed_id"] not in done_seeds)
    print(f"resume: {n_acc} accepted so far, {len(queue)} seeds left; target {args.target}")

    deadline = t0 + args.max_minutes * 60
    while queue and n_acc < args.target and time.time() < deadline:
        batch = [queue.popleft() for _ in range(min(args.batch_size, len(queue)))]
        jobs = [(c, c["donors"][tries[c["seed_id"]]]) for c in batch]
        if use_aliases:
            ensure_aliases([n["gold_entity"] for _, d in jobs for n in d["reasoning_graph"]] +
                           [c["seed_gold"][0] for c, _ in jobs])
        gens = generate(model, tok, [d["cf_prompt"] for _, d in jobs], args.max_new_tokens)
        retry = []
        for (c, d), gen in zip(jobs, gens):
            seed = natural[c["seed_id"]]
            k = tries[c["seed_id"]]
            tries[c["seed_id"]] += 1
            rec = dict(
                id=f"{seed['id']}_cf", source=seed.get("source", "2wikimultihopqa"), question=d["cf_question"],
                gold_answer=d["gold_answer"], reasoning_graph=d["reasoning_graph"], gold_titles=d["gold_titles"],
                context=seed["context"], retrieved_titles=seed.get("retrieved_titles"),
                retrieval_condition="neither", gold_p1_retrieved=False, gold_p2_retrieved=False,
                gold_entities_in_context=[False, False],
                gold_answer_in_context=d["gold_answer"].lower() in seed["context"].lower(),
                prompt=d["cf_prompt"], generated_cot=gen, hop_spans=parse_hop_spans(gen),
                predicted_answer=parse_predicted_answer(gen) or "",
                is_counterfactual=True, clean_pair_id=seed["id"])
            rec = label_example(rec, matcher)
            h1 = next((h for h in rec["hops"] if h["hop_idx"] == 1), None)
            h1_text = h1["text"] if h1 else ""
            if not any(sp["hop_idx"] == 1 for sp in rec["hop_spans"]):
                status, reason = "rejected", "no_hop1_tag"
            elif h1["label"] == 0:
                status, reason = "rejected", "hop1_states_donor_gold"
            elif rec["final_answer_correct"]:
                status, reason = "rejected", "answer_correct_from_memory"
            else:
                status, reason = "accepted", ""
            kind = None
            if status == "accepted":
                kind = ("hedge" if is_hedge(h1_text) else
                        "copied_seed_entity" if matcher.score(c["seed_gold"][0], h1_text) >= 0.70 else
                        "other_wrong_entity")
                rec["counterfactual_swap"] = dict(
                    seed_entity=c["seed_entity"], donor_entity=d["donor_entity"], donor_id=d["donor_id"],
                    template=c["template"], split=c["split"], seed_gold=c["seed_gold"],
                    seed_retrieval_condition=c["seed_retrieval_condition"], target_fail_hop=1,
                    n_entity_tokens=d["n_entity_tokens"], entity_token_span=d["span"], failure_kind=kind,
                    donor_rank=k, aliases_used=use_aliases)
                try:
                    (hs_dir / f"{rec['id']}.pt").unlink(missing_ok=True)
                    extract_hidden_states(rec, types.SimpleNamespace(model=model), tok, HS_CFG, hs_dir,
                                          list(range(N_LAYERS)))
                except Exception as ex:                          # e.g. prompt/generation tokenization seam
                    status, reason, kind = "rejected", f"hidden_state_error: {str(ex)[:120]}", None
            if status == "accepted":
                jl_append(cf_path, rec)
                n_acc += 1
            elif tries[c["seed_id"]] < min(args.max_tries_per_seed, len(c["donors"])):
                retry.append(c)
            jl_append(att_path, dict(seed_id=c["seed_id"], split=c["split"], donor_rank=k, donor_id=d["donor_id"],
                                     status=status, reason=reason, failure_kind=kind, hop1_text=h1_text[:300],
                                     hop_labels={h["hop_idx"]: h["label"] for h in rec["hops"]},
                                     predicted_answer=rec["predicted_answer"][:120]))
        queue.extendleft(reversed(retry))                        # retry a rejected seed right away
        print(f"[{(time.time() - t0) / 60:5.1f} min] accepted {n_acc}/{args.target} | seeds left {len(queue)}",
              flush=True)

    summarize(out, dict(parity_min_cos=par, repro=f"{n_same}/{len(rs)}", aliases_used=use_aliases,
                        matcher=matcher.stats(), minutes=round((time.time() - t0) / 60, 1),
                        stopped=("target" if n_acc >= args.target else "time" if time.time() >= deadline
                                 else "candidates_exhausted")))


def summarize(out: Path, run_meta: dict) -> dict:
    att = read_jsonl(out / "attempts.jsonl")
    cfs = read_jsonl(out / "counterfactuals.jsonl")
    hs = {p.stem for p in (out / "hidden_states").glob("*.pt")}
    sw = [c["counterfactual_swap"] for c in cfs]
    h2 = Counter(next((h["label"] for h in c["hops"] if h["hop_idx"] == 2), "none") for c in cfs)
    S = dict(run=run_meta, attempts=len(att), accepted=len(cfs), with_hidden_states=sum(c["id"] in hs for c in cfs),
             acceptance_rate=round(len(cfs) / max(len(att), 1), 3),
             outcomes=Counter(a["reason"] or "accepted" for a in att),
             accepted_by_split=Counter(x["split"] for x in sw), failure_kind=Counter(x["failure_kind"] for x in sw),
             failure_kind_by_split={sp: Counter(x["failure_kind"] for x in sw if x["split"] == sp)
                                    for sp in ("test", "train")},
             hop2_label=h2, entity_tokens=Counter(x["n_entity_tokens"] for x in sw),
             templates=len({x["template"] for x in sw}))
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=dict))
    L = ["# TF-IDF counterfactuals", "", "```", json.dumps(S, indent=1, default=dict), "```", "", "## Examples", ""]
    for kind in ("hedge", "copied_seed_entity", "other_wrong_entity"):
        for c in [c for c in cfs if c["counterfactual_swap"]["failure_kind"] == kind][:3]:
            x = c["counterfactual_swap"]
            h1 = next((h["text"] for h in c["hops"] if h["hop_idx"] == 1), "")
            L += [f"* **{kind}** — {x['seed_entity']} → {x['donor_entity']}: *{c['question']}*  ",
                  f"  donor hop-1 gold: {c['reasoning_graph'][0]['gold_entity']} | seed hop-1 gold: {x['seed_gold'][0]}  ",
                  f"  hop 1: {h1[:200]}"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return S


if __name__ == "__main__":
    main()
