import json
import time
import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_llm_baseline(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    with open(labels_file, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]

    # Filter out valid examples exactly like advanced_probing.py
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    
    # 1. GroupShuffleSplit to avoid train/test leak (same as advanced_probing.py)
    groups = [ex.get("clean_pair_id", ex["id"]) for ex in valid_ex]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(valid_ex, groups=groups))
    test_ex = [valid_ex[i] for i in test_idx]
    
    # Extract hop-level test data
    test_data = []
    for ex in test_ex:
        for hop in ex.get('hops', []):
            if hop.get('label') in (0, 1):
                gold_ent = hop.get("bridging_entity_gold", "")
                if not gold_ent:
                    # Fallback to the reasoning graph for unmatched hops
                    hop_idx = hop.get("hop_idx", -1)
                    for g in ex.get("reasoning_graph", []):
                        if g.get("hop") == hop_idx:
                            gold_ent = g.get("gold_entity", "")
                            break
                            
                ctx = ex.get("context", "")
                if isinstance(ctx, list):
                    ctx_str = "".join([c[1] if isinstance(c, (list, tuple)) and len(c) > 1 else str(c) for c in ctx])
                else:
                    ctx_str = str(ctx)
                    
                test_data.append({
                    "context": ctx_str,
                    "gold_entity": gold_ent,
                    "hop_text": hop["text"],
                    "label": hop["label"]
                })
                
    print(f"Total test hops to evaluate: {len(test_data)}")
    
    models = [
        "meta-llama/Llama-3.2-3B-Instruct", 
        "Qwen/Qwen2.5-3B-Instruct"
    ]
    
    for model_name in models:
        print(f"\n========================================")
        print(f"Loading {model_name} ...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        model.eval()
        
        y_true = []
        y_pred = []
        latencies = []
        token_costs = []
        
        SYSTEM = "You are a precise binary judge. Answer with exactly 'yes' or 'no' — no other text."
        USER_TMPL = (
            "Context Information:\n{context}\n\n"
            "Task: Based on the context above, does the following reasoning step correctly identify the entity '{entity}'?\n"
            "Reasoning step: \"{hop}\"\n"
            "Answer:"
        )

        print(f"Running evaluation...")
        y_fail_hedge_true, y_fail_hedge_pred = [], []
        y_fail_conf_true, y_fail_conf_pred = [], []
        
        for item in test_data:
            context_text = item["context"]
            gold_entity = item["gold_entity"]
            hop_text = item["hop_text"]
            true_label = item["label"]
            
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": USER_TMPL.format(context=context_text, entity=gold_entity, hop=hop_text)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = enc.input_ids.shape[1]
            
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc, 
                    max_new_tokens=3, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
                )
            t1 = time.time()
            
            reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip().lower()
            pred_label = 0 if reply.startswith("yes") else 1
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            latencies.append((t1 - t0) * 1000)  # ms
            token_costs.append(input_len + 3) # approx total tokens per hop decision
            
            # Track hedging vs confident
            is_hedge = any(k in hop_text.lower() for k in ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"])
            if true_label == 1:
                if is_hedge:
                    y_fail_hedge_true.append(1)
                    y_fail_hedge_pred.append(pred_label)
                else:
                    y_fail_conf_true.append(1)
                    y_fail_conf_pred.append(pred_label)
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_lat = np.mean(latencies)
        avg_toks = np.mean(token_costs)
        
        print(f"\nRESULTS FOR {model_name}:")
        print(f"Overall Accuracy: {acc:.3f} | F1: {f1:.3f}")
        
        hedge_acc = accuracy_score(y_fail_hedge_true, y_fail_hedge_pred) if y_fail_hedge_true else 0.0
        conf_acc = accuracy_score(y_fail_conf_true, y_fail_conf_pred) if y_fail_conf_true else 0.0
        print(f"Hedging Failure Acc: {hedge_acc:.3f}")
        print(f"Confident Failure Acc: {conf_acc:.3f}")
        
        print(f"Average Latency per hop: {avg_lat:.1f} ms")
        print(f"Average Token Cost per hop: {avg_toks:.1f} tokens")
        
        # Free up memory before loading the next model
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/2wikimultihopqa/labeled.jsonl")
    parser.add_argument("--hs-dir", default="data/hidden_states")
    args = parser.parse_args()
    
    run_llm_baseline(args.labels, args.hs_dir)
