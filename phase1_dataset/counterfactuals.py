"""
phase1_dataset/counterfactuals.py
────────────────────────────
Construct minimal counterfactual questions to balance the failure class.

Why we need this
────────────────
A competent 3B model gets ~87 % of hops correct, so the natural failure rate
is low.  A probe trained on imbalanced data trivially predicts "success" for
everything.  We manufacture failures by substituting one surface entity in the
question with another that leads the model to the *wrong* bridging entity.

Why token-length matching is mandatory
──────────────────────────────────────
In Phase 2 (activation patching) we need the clean and corrupted prompts to
be token-for-token identical *up to* the substituted entity, so the hidden-
state patch lands at the exact same position.  If lengths differ, all
subsequent token positions shift and the patch hits the wrong place.

Algorithm
─────────
For each clean example where *all* hops succeed:
  1. Identify the "entry entity" — the surface form in the question that
     triggers hop 1 (e.g., "Inception" triggers retrieval of the director).
  2. Find candidate substitutes from the dataset vocabulary with the same
     token count as the original entity.
  3. Replace, re-run the model (or predict failure from the reasoning graph),
     validate that hop 1 is now labeled FAILURE.
  4. Discard any substitution where the model accidentally still gets it right.

Dual purpose
────────────
The (clean, corrupted) pairs produced here are stored as minimal pairs and
reused in Phase 2 (causal patching) without extra data construction.

Output
──────
Appends counterfactual records to the labeled pool.  Each record includes:
  "is_counterfactual": true
  "counterfactual_swap": {
      "original_entity": ...,
      "substituted_entity": ...,
      "token_count": ...,
      "target_fail_hop": ...
  }
  "clean_pair_id": <id of the original clean example>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
import contextlib
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from phase1_dataset.label_hops import label_example, make_llm_judge
from utils.matching import EntityMatcher
from utils.wikidata_aliases import load_alias_table

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Entity vocabulary builder
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_vocab(examples: list[dict], tokenizer=None) -> dict[int, list[str]]:
    """
    Collect entry entities from the dataset to use as substitutions.
    Like-for-like swaps (titles for titles), bucketed by REAL token count.

    NOTE: make_counterfactual() looks this dict up with the tokenizer's token
    count, so the buckets must be keyed the same way.  (The previous version
    keyed by whitespace word count, so the lookup hit the wrong bucket and most
    examples found few or no candidates.)
    """
    vocab: dict[int, list[str]] = {}
    for ex in examples:
        e = find_entry_entity(ex["question"], ex.get("reasoning_graph", []))
        if not e:
            continue
        if tokenizer is not None:
            n = len(tokenizer.encode(e, add_special_tokens=False))
        else:
            n = len(e.split())
        vocab.setdefault(n, []).append(e)
    return {k: sorted(set(v)) for k, v in vocab.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Token-length checker
# ─────────────────────────────────────────────────────────────────────────────

def same_token_length(
    orig: str,
    subst: str,
    tokenizer,
) -> bool:
    """
    Check that `orig` and `subst` tokenize to the exact same number of tokens.

    This guarantees prefix-token alignment in the (clean, corrupted) pair:
    every token position before and after the substituted entity is identical,
    so an activation patch at position P in the corrupted prompt goes to the
    same semantic location as in the clean prompt.
    """
    orig_ids  = tokenizer.encode(orig,  add_special_tokens=False)
    subst_ids = tokenizer.encode(subst, add_special_tokens=False)
    return len(orig_ids) == len(subst_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-entity extractor
# ─────────────────────────────────────────────────────────────────────────────

def find_entry_entity(question: str, reasoning_graph: list[dict]) -> Optional[str]:
    """
    Identify the surface-form entity in the question that "triggers" hop 1.

    Strategy:
      1. Check if any entity from the FULL reasoning graph appears verbatim in
         the question (case-insensitive).  Skip the hop-1 gold entity itself
         (that is the *object* we expect to retrieve, not the trigger).
      2. Fall back to a Title-Case NP regex over the question, filtering out
         stop-words and the hop-1 gold entity.

    The multi-word regex is anchored to word boundaries so it handles entities
    like "The Lord of the Rings" and also works on lowercase questions by
    accepting any run of \\S+ characters anchored around capitalized tokens.

    Returns the longest matching surface form, or None if heuristics fail.
    """
    import re
    if not reasoning_graph:
        return None

    hop1_gold  = reasoning_graph[0].get("gold_entity", "").lower()
    q_lower    = question.lower()

    # ── Pass 1: graph entities that appear in the question ─────────────────────
    # Other hops' gold entities often ARE the entry entity for the next hop, so
    # searching the full graph gives a much higher hit rate.
    candidates_from_graph: list[str] = []
    for node in reasoning_graph:
        ent = node.get("gold_entity", "").strip()
        if not ent:
            continue
        ent_lower = ent.lower()
        if ent_lower == hop1_gold:   # skip the hop-1 object itself
            continue
        # Check verbatim presence (word-boundary to avoid partial matches)
        if re.search(r"\b" + re.escape(ent_lower) + r"\b", q_lower):
            candidates_from_graph.append(ent)

    if candidates_from_graph:
        return max(candidates_from_graph, key=len)

    # ── Pass 2: Title-Case NP regex fallback ───────────────────────────────────
    # Matches runs of capitalised tokens (handles multi-word proper nouns) plus optional parentheticals.
    pattern    = r"\b[A-Z][a-zA-Z'-]*(?:\s+(?:[A-Z][a-zA-Z'-]*|of|the|and|de|van|von))*\b(?:\s*\([^)]*\))?"
    np_matches = re.findall(pattern, question)

    STOP_WORDS = {"what", "where", "who", "when", "why", "how", "which", "the", "a", "an", "is", "are", "was", "were", "do", "does", "did"}

    filtered = [
        c for c in np_matches
        if c.lower() not in hop1_gold
        and hop1_gold not in c.lower()
        and len(c) > 1                # skip single-letter hits
        and c.lower().strip() not in STOP_WORDS
    ]

    if not filtered:
        return None

    return max(filtered, key=len)


# ─────────────────────────────────────────────────────────────────────────────
# Per-example counterfactual generator
# ─────────────────────────────────────────────────────────────────────────────

def make_counterfactual(
    clean_example: dict,
    entity_vocab: dict[int, list[str]],
    tokenizer,
    matcher: EntityMatcher,
    model,
    cfg: dict,
    rng: random.Random,
) -> Optional[dict]:
    """
    Attempt to build a counterfactual for a single clean example.

    Returns a new example dict (with is_counterfactual=True) or None if no
    valid substitution could be found within `max_substitution_attempts`.
    """
    cf_cfg        = cfg["counterfactuals"]
    max_attempts  = cf_cfg.get("max_substitution_attempts", 25)
    validate_fail = cf_cfg.get("validate_induced_failure", True)

    # SAFETY: validate_induced_failure must stay True.
    # If disabled, cf["prompt"]/generated_cot are never set, so extract_hidden_states()
    # would silently extract activations from the *clean* example's prompt — corrupting
    # every counterfactual hidden-state vector with no error thrown.
    if not validate_fail:
        raise ValueError(
            "validate_induced_failure: false is not supported. "
            "It would silently write stale clean-example prompts into counterfactual "
            "records, corrupting hidden-state extraction. Keep it true."
        )

    question    = clean_example["question"]
    graph       = clean_example.get("reasoning_graph", [])
    entry_ent   = find_entry_entity(question, graph)

    if not entry_ent:
        return None

    # Find token-length-matched substitutes
    token_count = len(tokenizer.encode(entry_ent, add_special_tokens=False))
    same_len    = entity_vocab.get(token_count, [])
    candidates  = [
        e for e in same_len
        if e != entry_ent                         # not the same entity
        and e.lower() not in question.lower()     # not already in question
    ]
    rng.shuffle(candidates)

    for substitute in candidates[:max_attempts]:
        # Verify token-length match with the actual tokenizer (not the proxy)
        if not same_token_length(entry_ent, substitute, tokenizer):
            continue

        # Build counterfactual question
        cf_question = question.replace(entry_ent, substitute, 1)
        if cf_question == question:
            continue

        # If validation is enabled, run the model and check that hop 1 fails.
        # If validation is disabled (dry run), assume the substitution works.
        if validate_fail:
            induced, cf_prompt, cf_gen_text, cf_hop_spans, cf_pred_ans = _validate_induces_failure(
                cf_question, clean_example, matcher, model, tokenizer, cfg
            )
            if not induced:
                continue

        # Build the counterfactual record
        cf = copy.deepcopy(clean_example)
        cf["id"]               = clean_example["id"] + "_cf"
        cf["question"]         = cf_question
        if validate_fail:
            cf["prompt"]           = cf_prompt
            cf["generated_cot"]    = cf_gen_text
            cf["hop_spans"]        = cf_hop_spans
            cf["predicted_answer"] = cf_pred_ans
        cf["is_counterfactual"] = True
        cf["clean_pair_id"]    = clean_example["id"]
        cf["counterfactual_swap"] = {
            "original_entity":    entry_ent,
            "substituted_entity": substitute,
            "token_count":        token_count,
            "target_fail_hop":    1,
        }
        # The gold answer and reasoning graph stay the same
        # (model is expected to fail to reach the correct bridging entity)
        return cf

    return None


def _validate_induces_failure(
    cf_question: str,
    clean_example: dict,
    matcher: EntityMatcher,
    model,
    tokenizer,
    cfg: dict,
) -> tuple:
    """
    Run the model on the counterfactual question and check that hop 1 fails.
    Returns True if hop 1 is labeled failure (substitution is valid).
    """
    import torch
    from phase1_dataset.generate_cot import build_prompt, parse_hop_spans, parse_predicted_answer

    sys_prompt = cfg["generation"]["system_prompt"]
    prompt     = build_prompt(cf_question, clean_example.get("context", ""), sys_prompt, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_text = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    hop_spans = parse_hop_spans(gen_text)
    pred_ans = parse_predicted_answer(gen_text)
    if not hop_spans:
        return True, prompt, gen_text, hop_spans, pred_ans   # No hop generated → counts as failure

    # Check hop 1
    hop1_text   = hop_spans[0]["text"] if hop_spans else ""
    gold_entity = (
        clean_example.get("reasoning_graph", [{}])[0].get("gold_entity", "")
    )
    result = matcher.match(gold_entity, hop1_text)
    induced = not result.matched
    return induced, prompt, gen_text, hop_spans, pred_ans


# ─────────────────────────────────────────────────────────────────────────────
# Batched candidate generation (speed)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _no_judge(matcher):
    """Temporarily disable the LLM judge (fast tiers only)."""
    saved = matcher.llm_judge
    matcher.llm_judge = None
    try:
        yield
    finally:
        matcher.llm_judge = saved


def batched_generate(prompts, model, tokenizer, max_new_tokens=160, bs=8):
    """
    Greedy generation for many prompts at once (left-padded, length-sorted so
    batches waste little padding).  Returns decoded new text, in input order.
    """
    import torch
    res = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        plen = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            res[i] = tokenizer.decode(out[j][plen:], skip_special_tokens=True)
    return res


def propose_candidates(clean_example, entity_vocab, tokenizer, rng, max_attempts, banned_questions):
    """
    Build up to `max_attempts` token-length-matched substitution candidates for
    one clean example.  No model calls here.  Each candidate is a dict.
    """
    question  = clean_example["question"]
    graph     = clean_example.get("reasoning_graph", [])
    entry_ent = find_entry_entity(question, graph)
    if not entry_ent:
        return []

    token_count = len(tokenizer.encode(entry_ent, add_special_tokens=False))
    q_lower     = question.lower()
    pool = [
        e for e in entity_vocab.get(token_count, [])
        if e.lower() != entry_ent.lower() and e.lower() not in q_lower
    ]
    rng.shuffle(pool)

    cands = []
    for sub in pool:
        if len(cands) >= max_attempts:
            break
        if not same_token_length(entry_ent, sub, tokenizer):
            continue
        cf_q = question.replace(entry_ent, sub, 1)
        if cf_q == question or cf_q.lower() in banned_questions:
            continue
        cands.append({
            "cf_question": cf_q, "substitute": sub,
            "entry_entity": entry_ent, "token_count": token_count,
        })
    return cands


def _build_cf_record(clean_example, cand, prompt, gen_text, hop_spans, pred_ans, tokenizer):
    from phase1_dataset.generate_cot import parse_predicted_answer  # noqa: F401
    cf = copy.deepcopy(clean_example)
    cf["id"]               = clean_example["id"] + "_cf"
    cf["question"]         = cand["cf_question"]
    cf["prompt"]           = prompt
    cf["generated_cot"]    = gen_text
    cf["hop_spans"]        = hop_spans
    cf["predicted_answer"] = pred_ans
    cf["is_counterfactual"] = True
    cf["clean_pair_id"]    = clean_example["id"]
    aligned = None
    if clean_example.get("prompt"):
        aligned = (len(tokenizer(clean_example["prompt"]).input_ids)
                   == len(tokenizer(prompt).input_ids))
    cf["counterfactual_swap"] = {
        "original_entity":    cand["entry_entity"],
        "substituted_entity": cand["substitute"],
        "token_count":        cand["token_count"],
        "target_fail_hop":    1,
        # Phase 2 (activation patching) should keep only pairs where this is True
        "prompt_token_aligned": aligned,
    }
    return cf


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_counterfactuals(cfg: dict, model=None, tokenizer=None, hf_token=None) -> Path:
    """
    Generate counterfactual examples until the HOP-level failure ratio reaches
    cf_cfg["target_failure_ratio"].  Returns path to augmented.jsonl.

    Speed design: candidates for `chunk_size` examples are validated together
    with batched generation, one candidate per example per round; only examples
    still unresolved go on to the next round.  The run is resumable (records
    are appended to augmented.jsonl as they are accepted) and can stop itself
    after cf_cfg["max_minutes"] so a Kaggle session is never lost.
    """
    from phase1_dataset.generate_cot import (
        build_prompt, parse_hop_spans, parse_predicted_answer, extract_hidden_states,
    )

    raw_dir   = Path(cfg["data"]["raw_dir"])
    match_cfg = cfg["matching"]
    cf_cfg    = cfg["counterfactuals"]
    rng       = random.Random(cfg.get("seed", 42))

    max_attempts = cf_cfg.get("max_substitution_attempts", 3)
    chunk_size   = cf_cfg.get("chunk_size", 32)
    gen_bs       = cf_cfg.get("gen_batch_size", 8)
    max_new      = cf_cfg.get("max_new_tokens", 160)
    max_minutes  = cf_cfg.get("max_minutes")
    deadline     = time.time() + 60 * max_minutes if max_minutes else None
    target_ratio = cf_cfg.get("target_failure_ratio", 0.45)

    labeled_path   = raw_dir / "2wikimultihopqa" / "labeled.jsonl"
    augmented_path = raw_dir / "2wikimultihopqa" / "augmented.jsonl"
    if not labeled_path.exists():
        raise FileNotFoundError(f"{labeled_path} not found — run label_hops.py first.")

    with open(labeled_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    clean_examples = [
        ex for ex in examples
        if ex.get("first_fail_hop") is None
        and all(h["label"] == 0 for h in ex.get("hops", []))
    ]

    existing_cfs = []
    if augmented_path.exists():
        with open(augmented_path, "r", encoding="utf-8") as f:
            existing_cfs = [j for j in (json.loads(l) for l in f) if j.get("is_counterfactual", False)]
        logger.info("Resuming: %d counterfactuals already in augmented.jsonl.", len(existing_cfs))
    else:
        with open(augmented_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def _count(data):
        fh = sum(1 for e in data for h in e.get("hops", []) if h.get("label") == 1)
        ch = sum(1 for e in data for h in e.get("hops", []) if h.get("label") == 0)
        return fh, ch

    fail_h, clean_h = _count(examples + existing_cfs)
    ratio = lambda: fail_h / max(fail_h + clean_h, 1)
    logger.info("Starting hop ratio: %.1f%% (%d fail / %d clean). Target: %.0f%%",
                100 * ratio(), fail_h, clean_h, 100 * target_ratio)
    if ratio() >= target_ratio:
        logger.info("Target already met. Nothing to generate.")
        return augmented_path

    # Model / tokenizer (tokenizer is needed for the vocab, so load first)
    if model is None:
        from phase1_dataset.generate_cot import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)
    tokenizer.padding_side = "left"

    aliases = load_alias_table(match_cfg["wikidata_aliases_path"])
    matcher = EntityMatcher(
        aliases=aliases,
        sbert_model_name=match_cfg["sbert_model"],
        sbert_threshold=match_cfg["sbert_threshold"],
        # Same judge policy as run_labeling in the Phase 1 notebook, so CF labels
        # follow the same rules as the original labels.
        llm_judge=make_llm_judge(model, tokenizer, device=model.device)
                  if match_cfg.get("use_llm_fallback", False) else None,
        llm_budget=match_cfg.get("llm_fallback_budget", 0.05),
    )

    entity_vocab = build_entity_vocab(examples, tokenizer)
    logger.info("Entity vocab: %d entities across %d token-length buckets.",
                sum(len(v) for v in entity_vocab.values()), len(entity_vocab))

    # nnsight is not needed on this path (extraction uses native HF hidden states)
    hs_model = SimpleNamespace(model=model)
    n_layers = model.config.num_hidden_layers
    layers_cfg = cfg["hidden_states"].get("layers", "all")
    layer_indices = list(range(n_layers)) if layers_cfg == "all" else layers_cfg
    hs_dir = Path(cfg["data"]["hidden_states_dir"])

    used_clean = {c["clean_pair_id"] for c in existing_cfs}
    available = [ex for ex in clean_examples if ex["id"] not in used_clean]
    rng.shuffle(available)
    existing_q = {e["question"].lower() for e in examples} | {c["question"].lower() for c in existing_cfs}
    sys_prompt = cfg["generation"]["system_prompt"]

    stats = {"chunks": 0, "no_cands": 0, "gen": 0, "no_hops": 0, "not_induced": 0,
             "label_rejected": 0, "accepted": 0}
    t0 = time.time()

    for c0 in tqdm(range(0, len(available), chunk_size), desc="Processing Chunks"):
        if ratio() >= target_ratio:
            logger.info("Target ratio met. Stopping.")
            break
        if deadline and time.time() > deadline:
            logger.info("max_minutes reached — stopping cleanly (run is resumable).")
            break

        chunk = available[c0:c0 + chunk_size]
        pending = []
        for ex in chunk:
            cands = propose_candidates(ex, entity_vocab, tokenizer, rng, max_attempts, existing_q)
            if cands:
                pending.append((ex, cands))
            else:
                stats["no_cands"] += 1

        for attempt in range(max_attempts):
            if not pending or ratio() >= target_ratio:
                break
            jobs = [(ex, cands[attempt]) for ex, cands in pending if attempt < len(cands)]
            if not jobs:
                break
            prompts = [build_prompt(c["cf_question"], ex.get("context", ""), sys_prompt, tokenizer)
                       for ex, c in jobs]
            texts = batched_generate(prompts, model, tokenizer, max_new, gen_bs)
            stats["gen"] += len(jobs)

            done_ids = set()
            for (ex, cand), prompt, text in zip(jobs, prompts, texts):
                if ratio() >= target_ratio:
                    break
                hop_spans = parse_hop_spans(text)
                if not hop_spans:                      # format failure, not a reasoning failure
                    stats["no_hops"] += 1
                    continue
                gold0 = (ex.get("reasoning_graph") or [{}])[0].get("gold_entity", "")
                with _no_judge(matcher):               # cheap first screen: fast tiers only
                    induced = not matcher.match(gold0, hop_spans[0]["text"]).matched
                if not induced:
                    stats["not_induced"] += 1
                    continue
                if cand["cf_question"].lower() in existing_q:
                    continue

                cf = _build_cf_record(ex, cand, prompt, text, hop_spans,
                                      parse_predicted_answer(text), tokenizer)
                cf.update(label_example(cf, matcher))  # full labeling, with the judge
                if cf.get("first_fail_hop") != 1:      # judge/bipartite says hop 1 was fine
                    stats["label_rejected"] += 1
                    continue

                extract_hidden_states(cf, hs_model, tokenizer, cfg, hs_dir, layer_indices)
                with open(augmented_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cf, ensure_ascii=False) + "\n")

                existing_q.add(cf["question"].lower())
                fail_h  += sum(1 for h in cf["hops"] if h["label"] == 1)
                clean_h += sum(1 for h in cf["hops"] if h["label"] == 0)
                stats["accepted"] += 1
                done_ids.add(ex["id"])
            pending = [(ex, c) for ex, c in pending if ex["id"] not in done_ids]

        stats["chunks"] += 1
        el = time.time() - t0
        print(
            f"chunk {stats['chunks']} | accepted {stats['accepted']} ({100 * stats['accepted'] / max(stats['gen'], 1):.0f}% of {stats['gen']} generated) | hop ratio {100 * ratio():.1f}% | {el / 60:.1f} min | "
            f"rejects: no_cands={stats['no_cands']} no_hops={stats['no_hops']} not_induced={stats['not_induced']} label={stats['label_rejected']}"
        )

    logger.info("Done. Added %d counterfactuals. Final hop failure ratio: %.1f%% (%d fail / %d clean).",
                stats["accepted"], 100 * ratio(), fail_h, clean_h)
    return augmented_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Generate counterfactual examples")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_counterfactuals(config)
