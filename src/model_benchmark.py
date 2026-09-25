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

def test_hooking(model_name: str, hf_token: str = None) -> bool:
    """Test if nnsight can trace and overwrite hidden states on this model."""
    try:
        from nnsight import LanguageModel
        logger.info(f"Testing nnsight hooking for {model_name}...")
        nn_model = LanguageModel(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token,
        )
        # Try common layer attribute names across architectures
        inner = nn_model.model
        layer = None
        for attr in ("layers", "decoder", "h", "blocks"):
            if hasattr(inner, attr):
                layer = getattr(inner, attr)[0]
                break
        if layer is None:
            raise ValueError(f"Cannot find transformer layer block on {type(inner).__name__}")
        with nn_model.trace("The capital of France is"):
            hidden = layer.output[0].save()
            layer.output[0] = hidden
        logger.info(f"Hooking successful for {model_name}.")
        del nn_model
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.warning(f"Hooking failed for {model_name}: {e}")
        return False

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

    # Generate CoT
    enriched_examples = generate_cot_batch(examples, model, tokenizer, cfg)
    
    # Label Hops
    labeled_examples = [label_example(ex, matcher) for ex in enriched_examples]
    
    del model
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
    sampled = all_records[:n_samples]

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

        # context: either [[title, [sent, ...]], ...] or [{"title": ..., "sentences": [...]}, ...]
        def _ctx_parts(ctx):
            if isinstance(ctx, (list, tuple)):
                title = ctx[0] if len(ctx) > 0 else ""
                sents = ctx[1] if len(ctx) > 1 else []
            elif isinstance(ctx, dict):
                title = ctx.get("title") or ctx.get("key") or ""
                sents = ctx.get("sentences") or ctx.get("value") or ctx.get("sents") or []
            else:
                return "", []
            return title, sents

        gold_context = ""
        for ctx in rec.get("context", []):
            title, sents = _ctx_parts(ctx)
            if title in gold_titles:
                if isinstance(sents, (list, tuple)):
                    sentences = " ".join(str(s) for s in sents)
                else:
                    sentences = str(sents)
                gold_context += f"Title: {title}\n{sentences}\n\n"

        return {
            "id": rec.get("_id") or rec.get("id") or f"2wiki_{i}",
            "source": "2wikimultihopqa",
            "question": rec["question"],
            "context": gold_context.strip(),
            "gold_answer": rec["answer"],
            "reasoning_graph": graph,
        }

    examples = [_normalise(r) for r in sampled]

    # Load Matcher
    match_cfg = cfg["matching"]
    aliases = load_alias_table(match_cfg["wikidata_aliases_path"])
    matcher = EntityMatcher(
        aliases=aliases,
        sbert_model_name=match_cfg["sbert_model"],
        sbert_threshold=match_cfg["sbert_threshold"],
    )

    models_to_test = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct"
    ]
    
    results = []
    for model_name in models_to_test:
        # Hook test and CoT evaluation are INDEPENDENT.
        # Always run evaluation; hook compatibility is just metadata.
        hook_success = test_hooking(model_name, hf_token=hf_token)
        metrics = evaluate_model(model_name, examples, cfg, matcher, hf_token=hf_token)
        results.append({
            "Model": model_name,
            "Hook Compatibility": "Yes" if hook_success else "No",
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
