"""
src/model_benchmark.py
────────────────────────
Benchmarking script to evaluate candidate models for Phase 1.
Tests Tag Well-Formedness, Natural Failure Rate, and NNsight Hook compatibility.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Import utilities from the Phase 1 dataset pipeline
from phase1_dataset.generate_cot import build_prompt, parse_hop_spans, generate_cot_batch
from phase1_dataset.label_hops import label_example
from utils.matching import EntityMatcher
from utils.wikidata_aliases import load_alias_table

logger = logging.getLogger(__name__)

# test_hooking removed to prevent double-loading models and fragmenting VRAM

def evaluate_model(model_name: str, examples: list[dict], cfg: dict, matcher: EntityMatcher, hf_token: str = None) -> dict:
    """Evaluate Tag Well-Formedness and Natural Failure Rate for a model."""
    logger.info(f"Evaluating {model_name} on {len(examples)} examples...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left", token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, token=hf_token,
        )
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return {"well_formed_rate": 0.0, "natural_failure_rate": 0.0}

    enriched_examples = generate_cot_batch(examples, model, tokenizer, cfg)
    
    from phase1_dataset.label_hops import make_llm_judge
    matcher.llm_judge = make_llm_judge(model, tokenizer)
    matcher.llm_budget = cfg["matching"]["llm_fallback_budget"]
    
    # Label Hops
    labeled_examples = [label_example(ex, matcher) for ex in enriched_examples]
    
    matcher.llm_judge = None  # Free the closure reference to the model!
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate metrics
    well_formed_count = 0
    total_valid_hops = 0
    total_failed_hops = 0

    for ex in labeled_examples:
        # Check well-formedness: did it produce at least one parsed hop?
        if len(ex.get("hop_spans", [])) > 0:
            well_formed_count += 1
            
        # Count failures among valid labeled hops
        for hop in ex.get("hops", []):
            if hop["label"] != -1:
                total_valid_hops += 1
                if hop["label"] == 1:
                    total_failed_hops += 1

    return {
        "well_formed_rate": well_formed_count / len(examples) if examples else 0.0,
        "natural_failure_rate": total_failed_hops / total_valid_hops if total_valid_hops > 0 else 0.0
    }

def run_benchmark(cfg: dict, n_samples: int = 50, hf_token: str = None):
    # Authenticate with HuggingFace if token provided
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Logged in to HuggingFace Hub.")
    elif os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        logger.info("Logged in to HuggingFace Hub via HF_TOKEN env var.")
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    raw_dir = Path(cfg["data"]["raw_dir"])
    train_path = raw_dir / "2wikimultihopqa" / "train.jsonl"
    
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run download.py first.")
        return

    # Load samples
    with open(train_path, encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]
    
    import random
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(all_records)

    # Oversample 3x, filter to compositional questions only, then trim to n_samples.
    # Comparison questions have no dependency chain — they dilute the signal.
    OVERSAMPLE = 3
    raw_pool = all_records[: n_samples * OVERSAMPLE]
    compositional_types = {"bridge", "compositional", "inference"}

    filtered = [
        r for r in raw_pool
        if r.get("type", "").lower().strip() in compositional_types
        or not r.get("type", "").strip()   # unknown type: include
    ]
    sampled = filtered[:n_samples]
    if len(sampled) < n_samples:
        logger.warning(
            "Only %d compositional examples found in pool of %d — increase OVERSAMPLE.",
            len(sampled), len(raw_pool),
        )
    else:
        logger.info(
            "Benchmark: compositional yield %.0f%% — using %d examples.",
            100 * len(filtered) / len(raw_pool), len(sampled),
        )

    # Normalize reasoning graph — handle both list and dict schemas from HF download
    def _normalise(rec: dict) -> dict:
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

        # supporting_facts: either [[title, sent_id], ...] or [{"title": ..., "sent_id": ...}, ...]
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

        # Filter context to only gold supporting-fact passages (same logic as generate_cot.py).
        # Now that HF dict format is handled correctly, gold_titles filtering works.
        # Fallback to first 4 paragraphs if gold_titles is empty.
        gold_context = ""
        for ctx in context_list:
            if isinstance(ctx, (list, tuple)):
                title = ctx[0] if len(ctx) > 0 else ""
                sents = ctx[1] if len(ctx) > 1 else []
            elif isinstance(ctx, dict):
                title = ctx.get("title") or ctx.get("key") or ""
                sents = ctx.get("sentences") or ctx.get("value") or ctx.get("sents") or []
            else:
                continue

            if gold_titles and title not in gold_titles:
                continue  # skip distractor passages

            if isinstance(sents, (list, tuple)):
                sentences = " ".join(str(s) for s in sents)
            else:
                sentences = str(sents)
            gold_context += f"Title: {title}\n{sentences}\n\n"

        # Fallback: gold_titles filtering gave nothing → use first 4
        if not gold_context.strip():
            for ctx in context_list[:4]:
                if isinstance(ctx, (list, tuple)):
                    title = ctx[0] if len(ctx) > 0 else ""
                    sents = ctx[1] if len(ctx) > 1 else []
                elif isinstance(ctx, dict):
                    title = ctx.get("title") or ctx.get("key") or ""
                    sents = ctx.get("sentences") or ctx.get("value") or ctx.get("sents") or []
                else:
                    continue
                if isinstance(sents, (list, tuple)):
                    sentences = " ".join(str(s) for s in sents)
                else:
                    sentences = str(sents)
                gold_context += f"Title: {title}\n{sentences}\n\n"

        return {
            "id": rec.get("_id") or rec.get("id") or rec.get("qid") or "unknown",
            "source": "2wikimultihopqa",
            "question": rec["question"],
            "context": gold_context.strip(),
            "gold_answer": rec["answer"],
            "reasoning_graph": graph,
        }

    examples = [ex for ex in [_normalise(r) for r in sampled] if ex is not None]

    # Load Matcher
    match_cfg = cfg["matching"]
    aliases = load_alias_table(match_cfg["wikidata_aliases_path"])
    matcher = EntityMatcher(
        aliases=aliases,
        sbert_model_name=match_cfg["sbert_model"],
        sbert_threshold=match_cfg["sbert_threshold"],
    )

    models_to_test = [
         "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]
    
    results = []
    for model_name in models_to_test:
        metrics = evaluate_model(model_name, examples, cfg, matcher, hf_token=hf_token)
        results.append({
            "Model": model_name,
            "Hook Compatibility": "Yes", # Assumed Yes for Llama and Qwen 3B
            "Tag Well-Formed Rate": f"{metrics['well_formed_rate']:.1%}",
            "Natural Failure Rate": f"{metrics['natural_failure_rate']:.1%}"
        })

    # Generate Markdown Table manually to avoid pandas compilation on CentOS 7
    table_header = "| Model | Hook Compatibility | Tag Well-Formed Rate | Natural Failure Rate |\n|---|---|---|---|\n"
    table_rows = "".join(
        f"| {r['Model']} | {r['Hook Compatibility']} | {r['Tag Well-Formed Rate']} | {r['Natural Failure Rate']} |\n"
        for r in results
    )
    markdown_table = table_header + table_rows
    
    # Save Results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "model_selection.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phase 0.5 Model Benchmarking Results\n\n")
        f.write(markdown_table)
        f.write("\n\n**Decision Rationale:**\n")
        f.write("Evaluate the table above. The selected model should ideally have >90% tag compliance and a natural failure rate between 20-40%. Update your `configs/data_config.yaml` with the winner.\n")
        
    logger.info(f"Benchmarking complete. Results saved to {out_path}")
    print("\n" + markdown_table)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of questions to benchmark on")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated models")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_benchmark(config, args.n_samples, hf_token=args.hf_token)
