"""
phase1_dataset/generate_cot.py
─────────────────────────
Run the LLM on 2WikiMultihopQA questions to produce XML-tagged CoT trajectories
and extract per-hop hidden states.

Two-pass design
───────────────
Pass 1 — CoT Generation (batched, standard HF generation)
    Feed the question with a prompt template.
    Model produces <hop1>…</hop1> … <hopN>…</hopN> <answer>…</answer>.
    This pass is batched for throughput.

Pass 2 — Hidden State Extraction (per-example, nnsight forward pass)
    Feed the complete prompt + generated text.
    Hook all transformer layers (or a selected subset).
    For each <hopN> span: mean-pool hidden states across its token range.
    Also save the single last-token hidden state of each bridging-entity span
    (needed for causal patching in Phase 2).

Outputs
───────
• data/raw/2wikimultihopqa/generated_cot.jsonl
    One record per example: original fields + generated_cot string.
• data/hidden_states/<id>.pt
    torch.Tensor of shape [n_layers, n_hops, hidden_dim]  (mean-pooled)
    plus a "single_token" sub-dict for causal patching (if configured).

Usage (standalone):
    python phase1_dataset/generate_cot.py --config configs/data_config.yaml
    python phase1_dataset/generate_cot.py --config configs/data_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, system_prompt: str, tokenizer) -> str:
    """
    Build the full prompt string using the tokenizer's own chat template.

    Uses ``tokenizer.apply_chat_template()`` so the format is automatically
    correct for whichever model is loaded (Qwen2.5-Instruct, Llama-3.2-Instruct,
    or anything else) — no hardcoded special tokens.

    The system prompt instructs the model to wrap each reasoning step in
    <hopN>…</hopN> XML tags.  This gives us:
      • Deterministic token boundaries for hidden-state pooling
      • A simple parser for entity extraction in label_hops.py
    """
    user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # appends the assistant-turn opener
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hop span parser (character-level, tokenizer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

_HOP_TAG_RE = re.compile(r"<hop(\d+)>(.*?)</hop\1>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_hop_spans(generated_text: str) -> list[dict]:
    """
    Extract all <hopN>…</hopN> blocks from a generated CoT string.

    Returns a list of dicts:
        {"hop_idx": int, "text": str, "char_start": int, "char_end": int}

    Note: char positions refer to positions within `generated_text` only
    (not the full prompt).  Token positions are computed in the extraction
    pass where the full sequence is available.
    """
    hops = []
    for m in _HOP_TAG_RE.finditer(generated_text):
        hops.append({
            "hop_idx":    int(m.group(1)),
            "text":       m.group(2).strip(),
            "char_start": m.start(2),
            "char_end":   m.end(2),
        })
    hops.sort(key=lambda h: h["hop_idx"])
    return hops


def parse_predicted_answer(generated_text: str) -> Optional[str]:
    m = _ANSWER_TAG_RE.search(generated_text)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """Load the base model and tokenizer from HuggingFace."""
    model_name = cfg["model"]["name"]
    dtype_map  = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype      = dtype_map.get(cfg["model"]["dtype"], torch.float16)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s  (dtype=%s)", model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — CoT generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_cot_batch(
    examples: list[dict],
    model,
    tokenizer,
    cfg: dict,
) -> list[dict]:
    """
    Run batched greedy generation on a list of examples.

    Each example must have: id, question, gold_answer, reasoning_graph.
    Returns list of enriched dicts with 'generated_cot', 'predicted_answer',
    'hop_spans' (character-level), and 'prompt' fields added.
    """
    gen_cfg    = cfg["generation"]
    batch_size = gen_cfg["batch_size"]
    sys_prompt = gen_cfg["system_prompt"]
    results    = []

    for batch_start in tqdm(
        range(0, len(examples), batch_size),
        desc="Generating CoT",
        unit="batch",
    ):
        batch = examples[batch_start : batch_start + batch_size]
        # Pass tokenizer so apply_chat_template uses the right format
        prompts = [build_prompt(ex["question"], ex.get("context", ""), sys_prompt, tokenizer) for ex in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg.get("do_sample", False),
                temperature=gen_cfg.get("temperature", 0.0) or None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        for ex, gen_ids in zip(batch, output_ids):
            new_ids   = gen_ids[prompt_len:]
            gen_text  = tokenizer.decode(new_ids, skip_special_tokens=True)
            hop_spans = parse_hop_spans(gen_text)
            pred_ans  = parse_predicted_answer(gen_text)

            results.append({
                **ex,
                "prompt":          build_prompt(ex["question"], ex.get("context", ""), sys_prompt, tokenizer),
                "generated_cot":   gen_text,
                "hop_spans":       hop_spans,
                "predicted_answer": pred_ans,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — Hidden state extraction (nnsight)
# ─────────────────────────────────────────────────────────────────────────────

def extract_hidden_states(
    example: dict,
    model,
    tokenizer,
    cfg: dict,
    hs_dir: Path,
) -> None:
    """
    Run a single forward pass with nnsight hooks on the full prompt+generation.

    Saves a .pt file with:
        "pooled": Tensor [n_layers, n_hops, hidden_dim]  — mean-pooled per hop
        "single_token": dict  hop_idx → Tensor [n_layers, hidden_dim]
                              (last token of bridging entity mention, for patching)
    """
    try:
        from nnsight import LanguageModel as NNsightLM
    except ImportError:
        logger.warning("nnsight not installed — skipping hidden state extraction.")
        return

    out_path = hs_dir / f"{example['id']}.pt"
    if out_path.exists():
        return

    full_text = example["prompt"] + example["generated_cot"]
    token_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    prompt_token_len = len(
        tokenizer(example["prompt"], return_tensors="pt").input_ids[0]
    )

    # Find token-level boundaries for each hop span
    # We re-tokenize the generated text to map char offsets → token offsets
    gen_encoding = tokenizer(
        example["generated_cot"],
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    expected_len = prompt_token_len + len(gen_encoding["input_ids"])
    assert len(token_ids) == expected_len, (
        f"Tokenization mismatch: joint={len(token_ids)} vs separate={expected_len}. "
        f"Prompt/generation boundary may not be token-aligned for id={example['id']}."
    )
    offset_map = gen_encoding["offset_mapping"]  # list of (char_start, char_end)

    hop_token_spans: list[tuple[int, int]] = []  # (t_start, t_end) in full-seq space
    for hop in example.get("hop_spans", []):
        c_start = hop["char_start"]
        c_end   = hop["char_end"]
        t_start = next(
            (i for i, (cs, ce) in enumerate(offset_map) if cs <= c_start < ce),
            None,
        )
        t_end = next(
            (i for i, (cs, ce) in enumerate(offset_map) if cs < c_end <= ce),
            None,
        )
        if t_start is not None and t_end is not None:
            # Shift by prompt length (token offsets are in generated-text space)
            hop_token_spans.append((
                prompt_token_len + t_start,
                prompt_token_len + t_end + 1,  # exclusive
            ))
        else:
            hop_token_spans.append((None, None))

    # Determine which layers to hook
    n_layers      = model.config.num_hidden_layers
    layers_cfg    = cfg["hidden_states"].get("layers", "all")
    layer_indices = list(range(n_layers)) if layers_cfg == "all" else layers_cfg

    # Run nnsight forward pass with saved layer outputs
    nn_model = NNsightLM(model, tokenizer=tokenizer)
    saved_states = {}  # layer_idx → Tensor [seq_len, hidden_dim]

    with nn_model.trace(full_text):
        for li in layer_indices:
            saved_states[li] = nn_model.model.layers[li].output[0].save()

    # Mean-pool per hop span, per layer
    n_hops   = len(hop_token_spans)
    hid_dim  = model.config.hidden_size
    pooled   = torch.zeros(len(layer_indices), n_hops, hid_dim, dtype=model.dtype)

    for li_idx, li in enumerate(layer_indices):
        hs_val = saved_states[li]
        hs = hs_val.value if hasattr(hs_val, "value") else hs_val  # [1, seq_len, hid_dim] or [seq_len, hid_dim]
        if hs.dim() == 3:
            hs = hs.squeeze(0)
        for hop_idx, (t_start, t_end) in enumerate(hop_token_spans):
            if t_start is None:
                continue
            pooled[li_idx, hop_idx] = hs[t_start:t_end].mean(dim=0).cpu()

    save_dict: dict = {"pooled": pooled, "layer_indices": layer_indices}

    # Single-token (last token of each hop) for causal patching
    if cfg["hidden_states"].get("save_single_token", False):
        single = {}
        for li_idx, li in enumerate(layer_indices):
            hs_val = saved_states[li]
            hs = hs_val.value if hasattr(hs_val, "value") else hs_val
            if hs.dim() == 3:
                hs = hs.squeeze(0)
            for hop_idx, (t_start, t_end) in enumerate(hop_token_spans):
                if t_start is None:
                    continue
                key = f"hop{hop_idx + 1}"
                if key not in single:
                    single[key] = torch.zeros(len(layer_indices), hid_dim, dtype=model.dtype)
                # Last token of the hop span
                single[key][li_idx] = hs[t_end - 1].cpu()
        save_dict["single_token"] = single

    hs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(cfg: dict, dry_run: bool = False) -> Path:
    """
    Generate CoT for all sampled 2WikiMultihopQA examples.
    Returns path to generated_cot.jsonl.
    """
    import random
    raw_dir  = Path(cfg["data"]["raw_dir"])
    hs_dir   = Path(cfg["data"]["hidden_states_dir"])
    n_sample = cfg["data"]["n_source_examples"]
    seed     = cfg.get("seed", 42)

    # Load sampled examples from the raw 2Wiki train JSONL
    src_path = raw_dir / "2wikimultihopqa" / "train.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} not found — run download.py first."
        )

    # Oversample the raw pool by 3x, filter to compositional questions only,
    # then trim to n_sample. This happens before any model work — it's free.
    OVERSAMPLE = 3
    rng = random.Random(seed)
    with open(src_path, encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]
    rng.shuffle(all_records)
    raw_pool = all_records[: n_sample * OVERSAMPLE]
    compositional_types = {"bridge", "compositional", "inference"}  # not "comparison"

    def _is_compositional(rec: dict) -> bool:
        q_type = rec.get("type", "").lower().strip()
        if not q_type:
            return True   # unknown type: include rather than exclude
        return q_type in compositional_types

    filtered = [r for r in raw_pool if _is_compositional(r)]
    sampled = filtered[:n_sample]
    if len(sampled) < n_sample:
        logger.warning(
            "Only found %d compositional examples in pool of %d raw records "
            "(multiplier=%dx). Consider increasing OVERSAMPLE.",
            len(sampled), len(raw_pool), OVERSAMPLE,
        )
    else:
        yield_pct = 100 * len(filtered) / len(raw_pool)
        logger.info(
            "Compositional yield: %d/%d (%.0f%%) — drew %d examples for generation.",
            len(filtered), len(raw_pool), yield_pct, len(sampled),
        )

    # Normalise reasoning graph into our schema
    def _normalise(rec: dict) -> dict:
        """
        Extract the reasoning graph from 2Wiki's `evidences` field.
        Handles both list-of-lists and list-of-dicts schemas from different HF download paths.
        """
        evidences = rec.get("evidences", [])
        graph = []
        for i, ev in enumerate(evidences):
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                gold_ent = ev[2]
            elif isinstance(ev, dict):
                gold_ent = ev.get("val") or ev.get("object") or ev.get("value") or ""
            else:
                gold_ent = ""
            graph.append({"hop": i + 1, "gold_entity": gold_ent})

        # supporting_facts: [[title, sent_id], ...] or [{"title": ..., "sent_id": ...}, ...]
        def _sf_title(sf):
            if isinstance(sf, (list, tuple)):
                return sf[0] if sf else ""
            elif isinstance(sf, dict):
                return sf.get("title") or sf.get("key") or ""
            return ""

        gold_titles = {_sf_title(sf) for sf in rec.get("supporting_facts", [])} - {""}

        # Handle Hugging Face columnar dictionary format
        ctx_obj = rec.get("context", [])
        if isinstance(ctx_obj, dict):
            titles = ctx_obj.get("title", [])
            sents_list = ctx_obj.get("sentences", [])
            context_list = list(zip(titles, sents_list))
        else:
            context_list = ctx_obj
            
        # Filter context to only gold supporting-fact passages.
        # We now correctly handle the HF columnar dict format, so this works.
        # Fallback to first 4 paragraphs if gold_titles is empty (malformed record).
        gold_context = ""
        matched_titles = set()
        for ctx in context_list:
            if isinstance(ctx, (list, tuple)):
                t = ctx[0] if ctx else ""
                sents = ctx[1] if len(ctx) > 1 else []
            elif isinstance(ctx, dict):
                t = ctx.get("title") or ctx.get("key") or ""
                sents = ctx.get("sentences") or ctx.get("value") or []
            else:
                continue

            if gold_titles and t not in gold_titles:
                continue   # skip distractor passages

            if isinstance(sents, (list, tuple)):
                sents = " ".join(str(s) for s in sents)
            gold_context += f"Title: {t}\n{sents}\n\n"
            matched_titles.add(t)

        # Fallback: if gold_titles filtering gave nothing (empty record), use first 4
        if not gold_context.strip():
            for ctx in context_list[:4]:
                if isinstance(ctx, (list, tuple)):
                    t = ctx[0] if ctx else ""
                    sents = ctx[1] if len(ctx) > 1 else []
                elif isinstance(ctx, dict):
                    t = ctx.get("title") or ctx.get("key") or ""
                    sents = ctx.get("sentences") or ctx.get("value") or []
                else:
                    continue
                if isinstance(sents, (list, tuple)):
                    sents = " ".join(str(s) for s in sents)
                gold_context += f"Title: {t}\n{sents}\n\n"

        return {
            "id":             rec.get("_id") or rec.get("id") or rec.get("qid") or f"unknown_{uuid.uuid4().hex[:8]}",
            "source":         "2wikimultihopqa",
            "question":       rec["question"],
            "context":        gold_context.strip(),
            "gold_answer":    rec["answer"],
            "reasoning_graph": graph,
            "raw":            rec,
        }

    examples = [_normalise(r) for r in sampled]

    out_path = raw_dir / "2wikimultihopqa" / "generated_cot.jsonl"

    if dry_run:
        logger.info("[DRY RUN] Would process %d examples → %s", len(examples), out_path)
        # Write schema-only stubs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples[:5]:
                stub = {**ex, "generated_cot": "<hop1>DRY RUN</hop1>",
                        "hop_spans": [{"hop_idx": 1, "text": "DRY RUN",
                                       "char_start": 6, "char_end": 13}],
                        "predicted_answer": "DRY RUN", "prompt": "DRY RUN"}
                f.write(json.dumps(stub, ensure_ascii=False) + "\n")
        return out_path

    # ── Checkpoint logic ──────────────────────────────────────────────────────
    processed_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                processed_ids.add(json.loads(line)["id"])
        logger.info("Found %d already generated examples in %s", len(processed_ids), out_path)

    examples_to_run = [ex for ex in examples if ex["id"] not in processed_ids]

    if not examples_to_run:
        logger.info("All %d examples already generated.", len(examples))
        return out_path

    model, tokenizer = load_model_and_tokenizer(cfg)
    
    chunk_size = 100
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open in append mode
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(examples_to_run), chunk_size):
            chunk = examples_to_run[i : i + chunk_size]
            logger.info("Processing chunk %d / %d", (i // chunk_size) + 1, (len(examples_to_run) + chunk_size - 1) // chunk_size)
            
            enriched = generate_cot_batch(chunk, model, tokenizer, cfg)
            
            # Hidden state extraction (if configured)
            if cfg["hidden_states"].get("extract", True):
                for ex in tqdm(enriched, desc="Hidden states", leave=False):
                    extract_hidden_states(ex, model, tokenizer, cfg, hs_dir)
            
            for ex in enriched:
                ex.pop("raw", None)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                
            # Automatically upload checkpoint to HuggingFace
            if os.environ.get("HF_REPO_ID") and os.environ.get("HF_TOKEN"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=os.environ.get("HF_TOKEN"))
                    api.upload_file(
                        path_or_fileobj=str(out_path),
                        path_in_repo="2wikimultihopqa/generated_cot.jsonl",
                        repo_id=os.environ.get("HF_REPO_ID"),
                        repo_type="dataset",
                        commit_message=f"Checkpoint: {min(i + chunk_size, len(examples_to_run))} examples processed"
                    )
                    logger.info("Successfully uploaded checkpoint to HF: %s", os.environ.get("HF_REPO_ID"))
                except Exception as e:
                    logger.warning("Failed to upload checkpoint to HF: %s", e)

    logger.info("Finished processing. Saved CoT records to %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Generate XML-tagged CoT trajectories")
    parser.add_argument("--config",  default="configs/data_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate schema without running the model")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_generation(config, dry_run=args.dry_run)
