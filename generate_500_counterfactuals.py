"""
generate_500_counterfactuals.py
───────────────────────────────────────────────────────────────────────────────
Generates counterfactual minimal pairs until 500 total are accumulated.

Designed to run on Kaggle (T4 x2).  Will resume automatically from an existing
augmented.jsonl if it already contains counterfactuals.

Usage (Kaggle cell) — paste your HF token directly:
    !python generate_500_counterfactuals.py \
        --hf_token  hf_XXXXXXXXXXXXXXXXXXXX \
        --input_repo AnishRacherla/LinguaFranca-Phase3 \
        --output /kaggle/working/augmented_500cf.jsonl \
        --target 500

Or, if you already have the file locally:
    !python generate_500_counterfactuals.py \
        --input  /path/to/augmented.jsonl \
        --output /kaggle/working/augmented_500cf.jsonl \
        --target 500

Expected runtime: ~2-4 hours on T4x2.
"""

from __future__ import annotations

import huggingface_hub

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID         = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS   = 160
MAX_ATTEMPTS     = 5       # candidate substitutions to try per seed example
MAX_MINUTES      = 600     # stop gracefully before Kaggle 12h timeout

SYSTEM_PROMPT = """\
You are a careful multi-hop reasoning assistant. You will be given a Context
(a set of passages) and a Question. Base EVERY reasoning step strictly on
facts stated in the Context — do NOT use outside knowledge or memorized
facts, and do NOT guess if the Context does not support a step.

For each reasoning step, identify the key entity or fact FROM THE CONTEXT
that advances toward the answer. State that entity or fact explicitly.
Wrap each step in numbered XML hop tags, then give the final answer.

Example:
Context:
Title: Inception
Inception is a 2010 science fiction film directed by Christopher Nolan.
Title: Christopher Nolan
Christopher Edward Nolan is a British and American filmmaker.

Question: What nationality is the director of Inception?
<hop1>From the context, I find that the director of Inception is Christopher Nolan.</hop1>
<hop2>From the context, I find that Christopher Nolan is British and American.</hop2>
<answer>British and American</answer>

REQUIRED OUTPUT FORMAT (follow exactly):
<hop1>From the context, I find that [key entity or fact for step 1].</hop1>
<hop2>From the context, I find that [key entity or fact for step 2].</hop2>
<answer>Final answer here.</answer>

Do not skip steps. Do not add any text outside the tags."""


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {MODEL_ID}")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    mdl.eval()
    print(f"Model loaded on: {mdl.device}")
    return mdl, tok


def build_prompt(question: str, context: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_single(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entity / counterfactual helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_hop_spans(text: str) -> list[dict]:
    spans = []
    for m in re.finditer(r"<hop(\d+)>(.*?)</hop\1>", text, re.DOTALL):
        spans.append({"hop_idx": int(m.group(1)), "text": m.group(2).strip()})
    return spans


def find_entry_entity(question: str, reasoning_graph: list) -> str | None:
    """Identify the surface entity in the question that triggers hop 1."""
    if not reasoning_graph:
        return None
    hop1_gold = reasoning_graph[0].get("gold_entity", "").lower()
    q_lower   = question.lower()

    # Pass 1: graph entities that appear in the question
    candidates: list[str] = []
    for node in reasoning_graph:
        ent = node.get("gold_entity", "").strip()
        if not ent or ent.lower() == hop1_gold:
            continue
        if re.search(r"\b" + re.escape(ent.lower()) + r"\b", q_lower):
            candidates.append(ent)
    if candidates:
        return max(candidates, key=len)

    # Pass 2: Title-Case NP regex fallback
    pattern   = r"\b[A-Z][a-zA-Z'-]*(?:\s+(?:[A-Z][a-zA-Z'-]*|of|the|and|de|van|von))*\b(?:\s*\([^)]*\))?"
    np_matches = re.findall(pattern, question)
    STOP = {"what","where","who","when","why","how","which","the","a","an",
            "is","are","was","were","do","does","did"}
    filtered = [
        c for c in np_matches
        if c.lower() not in hop1_gold
        and hop1_gold not in c.lower()
        and len(c) > 1
        and c.lower().strip() not in STOP
    ]
    return max(filtered, key=len) if filtered else None


def same_token_length(orig: str, subst: str, tokenizer) -> bool:
    """Token-length parity is mandatory for activation-patching alignment."""
    n_orig  = len(tokenizer.encode(orig,  add_special_tokens=False))
    n_subst = len(tokenizer.encode(subst, add_special_tokens=False))
    return n_orig == n_subst


def build_entity_vocab(examples: list, tokenizer) -> dict[int, list[str]]:
    """Bucket all entry entities by their token count."""
    vocab: dict[int, list[str]] = {}
    for ex in examples:
        ent = find_entry_entity(ex["question"], ex.get("reasoning_graph", []))
        if not ent:
            continue
        n = len(tokenizer.encode(ent, add_special_tokens=False))
        vocab.setdefault(n, []).append(ent)
    return {k: sorted(set(v)) for k, v in vocab.items()}


def failure_is_induced(hop1_text: str, gold_entity: str,
                       gen_text: str, gold_answer: str) -> bool:
    """
    Return True if the counterfactual generation has FAILED on hop 1.
    Two conditions:
      (a) gold_entity does NOT appear in hop1_text, AND
      (b) (optional) the final answer is NOT the gold answer
          (rules out parametric-memory cheat answers).
    """
    if gold_entity.lower() in hop1_text.lower():
        return False   # model still found the correct bridging entity

    if gold_answer:
        ans_match = re.search(r"<answer>(.*?)</answer>", gen_text, re.DOTALL)
        final_ans = ans_match.group(1).strip().lower() if ans_match else ""
        if gold_answer.lower() in final_ans:
            return False   # model arrived at correct answer from memory

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── HuggingFace login ─────────────────────────────────────────────────────
    if args.hf_token:
        try:
            huggingface_hub.login(token=args.hf_token)
            print("Logged in to HuggingFace.")
        except Exception as e:
            print(f"HF login failed: {e}")
    else:
        # Try Kaggle Secrets as fallback
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            huggingface_hub.login(token=token)
            print("Logged in via Kaggle Secrets.")
        except Exception:
            print("No HF token provided and Kaggle Secrets unavailable. "
                  "Private model/dataset access may fail.")

    # ── Resolve input file ────────────────────────────────────────────────────
    if args.input_repo and not args.input:
        from huggingface_hub import hf_hub_download
        print(f"Downloading augmented.jsonl from {args.input_repo} ...")
        downloaded = hf_hub_download(
            repo_id=args.input_repo,
            filename="data/2wikimultihopqa/augmented.jsonl",
            repo_type="dataset",
        )
        input_path = Path(downloaded)
    elif args.input:
        input_path = Path(args.input)
    else:
        raise ValueError("Provide either --input (local path) or --input_repo (HuggingFace repo id).")

    output_path = Path(args.output)
    target      = args.target

    print(f"Loading {input_path} …")
    with open(input_path, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    existing_cfs    = [d for d in all_data if d.get("is_counterfactual")]
    clean_examples  = [d for d in all_data if not d.get("is_counterfactual")]

    print(f"  Clean examples    : {len(clean_examples)}")
    print(f"  Existing CFs      : {len(existing_cfs)}")
    print(f"  Need to generate  : {max(0, target - len(existing_cfs))} more")

    if len(existing_cfs) >= target:
        print("Target already met. Nothing to do.")
        # Write to output anyway so output file always exists
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in all_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return

    model, tokenizer = load_model_and_tokenizer()

    # Eligible seeds: ALL hops succeeded
    all_success = [
        ex for ex in clean_examples
        if ex.get("first_fail_hop") is None
        and all(h.get("label") == 0 for h in ex.get("hops", []))
        and ex.get("hops")
    ]
    print(f"\nFully-successful seeds: {len(all_success)}")

    used_ids = {cf["clean_pair_id"] for cf in existing_cfs}
    available = [ex for ex in all_success if ex["id"] not in used_ids]

    rng = random.Random(args.seed)
    rng.shuffle(available)
    print(f"Unused seeds (available): {len(available)}")

    print("Building entity vocab …")
    entity_vocab = build_entity_vocab(all_success, tokenizer)
    total_ents = sum(len(v) for v in entity_vocab.values())
    print(f"  {total_ents} entities across {len(entity_vocab)} token-length buckets")

    # Track all known questions to avoid duplicates
    existing_questions = (
        {ex["question"].lower() for ex in clean_examples}
        | {cf["question"].lower() for cf in existing_cfs}
    )

    # Write existing records to output file
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    accepted = len(existing_cfs)
    stats = {"tried": 0, "no_ent": 0, "no_cands": 0, "not_induced": 0, "accepted": 0}
    t0 = time.time()
    deadline = t0 + MAX_MINUTES * 60

    print(f"\nRunning generation loop … (target={target}, max_minutes={MAX_MINUTES})")
    print("─" * 72)

    for seed_ex in tqdm(available, desc="Seed examples"):
        if accepted >= target:
            print(f"\n✓ Reached target of {target} counterfactuals.")
            break
        if time.time() > deadline:
            print(f"\n⏰ Time limit reached. Accepted so far: {accepted}")
            break

        question    = seed_ex["question"]
        graph       = seed_ex.get("reasoning_graph", [])
        entry_ent   = find_entry_entity(question, graph)
        if not entry_ent:
            stats["no_ent"] += 1
            continue

        tok_count = len(tokenizer.encode(entry_ent, add_special_tokens=False))
        pool = entity_vocab.get(tok_count, [])
        q_lower = question.lower()
        candidates = [
            e for e in pool
            if e.lower() != entry_ent.lower()
            and e.lower() not in q_lower
            and same_token_length(entry_ent, e, tokenizer)
        ]
        rng.shuffle(candidates)

        if not candidates:
            stats["no_cands"] += 1
            continue

        gold_entity = (graph[0].get("gold_entity", "") if graph else "")
        gold_answer = seed_ex.get("answer", "")

        for substitute in candidates[:MAX_ATTEMPTS]:
            stats["tried"] += 1
            cf_question = question.replace(entry_ent, substitute, 1)
            if cf_question == question or cf_question.lower() in existing_questions:
                continue

            context  = seed_ex.get("context", "")
            prompt   = build_prompt(cf_question, context, tokenizer)
            gen_text = generate_single(prompt, model, tokenizer)

            hop_spans = parse_hop_spans(gen_text)
            if not hop_spans:
                continue   # format failure — skip

            hop1_text = hop_spans[0]["text"]
            if not failure_is_induced(hop1_text, gold_entity, gen_text, gold_answer):
                stats["not_induced"] += 1
                continue

            # ── Build and save the counterfactual record ──────────────────────
            cf = copy.deepcopy(seed_ex)
            cf["id"]               = seed_ex["id"] + "_cf"
            cf["question"]         = cf_question
            cf["prompt"]           = prompt
            cf["generated_cot"]    = gen_text
            cf["hop_spans"]        = hop_spans
            cf["is_counterfactual"] = True
            cf["clean_pair_id"]    = seed_ex["id"]
            cf["counterfactual_swap"] = {
                "original_entity":    entry_ent,
                "substituted_entity": substitute,
                "token_count":        tok_count,
                "target_fail_hop":    1,
                "prompt_token_aligned": True,   # guaranteed by same_token_length
            }
            # Mark hop 1 as failure in the hops list
            cf["hops"] = copy.deepcopy(seed_ex.get("hops", []))
            for h in cf["hops"]:
                if h.get("hop_idx") == 1:
                    h["label"]          = 1
                    h["generated_text"] = hop1_text

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cf, ensure_ascii=False) + "\n")

            existing_questions.add(cf_question.lower())
            used_ids.add(seed_ex["id"])
            accepted += 1
            stats["accepted"] += 1

            elapsed = (time.time() - t0) / 60
            rate = stats["accepted"] / max(stats["tried"], 1) * 100
            print(
                f"[{accepted:4d}/{target}] '{entry_ent}' → '{substitute}' | "
                f"tried={stats['tried']} accept_rate={rate:.0f}% | {elapsed:.1f}min"
            )
            break   # one CF per seed example

    elapsed_total = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print(f"DONE. Total counterfactuals: {accepted}")
    print(f"Stats: {stats}")
    print(f"Total time: {elapsed_total:.1f} minutes")
    print(f"Output: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 500 counterfactuals for causal patching")

    # ── Input source (one of these two is required) ───────────────────────────
    parser.add_argument(
        "--input", default=None,
        help="Local path to augmented.jsonl (existing data from Phase 1). "
             "Use this OR --input_repo."
    )
    parser.add_argument(
        "--input_repo", default=None,
        help="HuggingFace dataset repo ID to download augmented.jsonl from "
             "(e.g. 'AnishRacherla/LinguaFranca-Phase3'). Use this OR --input."
    )

    # ── HuggingFace authentication ────────────────────────────────────────────
    parser.add_argument(
        "--hf_token", default=None,
        help="HuggingFace access token (paste your token here, e.g. hf_XXXX). "
             "Required for private repos / gated models. Falls back to "
             "Kaggle Secrets (HF_TOKEN) if not provided."
    )

    # ── Output & generation settings ─────────────────────────────────────────
    parser.add_argument(
        "--output", required=True,
        help="Output path for the enlarged file (e.g. /kaggle/working/augmented_500cf.jsonl)"
    )
    parser.add_argument(
        "--target", type=int, default=500,
        help="Total counterfactuals to accumulate (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--upload_repo", default=None,
        help="(Optional) HuggingFace dataset repo ID to upload the output file to "
             "after generation completes (e.g. 'AnishRacherla/LinguaFranca-Phase3')."
    )
    args = parser.parse_args()
    run(args)

    # ── Optional upload ───────────────────────────────────────────────────────
    if args.upload_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"\nUploading {args.output} to {args.upload_repo} ...")
        api.upload_file(
            path_or_fileobj=args.output,
            path_in_repo="data/2wikimultihopqa/augmented_500cf.jsonl",
            repo_id=args.upload_repo,
            repo_type="dataset",
            commit_message=f"Add augmented_500cf.jsonl",
        )
        print("Upload complete!")
