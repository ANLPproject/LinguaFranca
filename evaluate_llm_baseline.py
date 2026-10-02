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

    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]

    groups = [ex.get("clean_pair_id", ex["id"]) for ex in valid_ex]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(valid_ex, groups=groups))
    test_ex = [valid_ex[i] for i in test_idx]

    # Extract hop-level test data. No gold-entity lookup, no positional fallback —
    # the judge is asked about correctness given the question and context directly.
    test_data = []
    n_skipped_empty = 0
    for ex in test_ex:
        for hop in ex.get("hops", []):
            if hop.get("label") not in (0, 1):
                continue
            hop_text = hop.get("text", "").strip()
            if not hop_text:
                # Phase B "missing" hops: the model never generated this hop at all.
                # There is nothing to judge — count it as an automatic failure
                # prediction without a model call, rather than asking the judge
                # about an empty string.
                n_skipped_empty += 1
                test_data.append({
                    "context": None, "question": ex["question"], "hop_text": "",
                    "label": hop["label"], "auto_fail": True,
                })
                continue

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
                ctx_str = "\n".join(
                    (c[0] + ": " + "".join(c[1])) if isinstance(c, (list, tuple)) and len(c) > 1 else str(c)
                    for c in ctx
                )
            else:
                ctx_str = str(ctx)

            test_data.append({
                "context": ctx_str, "question": ex["question"], "gold_entity": gold_ent,
                "hop_text": hop_text, "label": hop["label"], "auto_fail": False,
            })

    print(f"Total test hops to evaluate: {len(test_data)}  "
          f"({n_skipped_empty} are empty 'missing' hops, auto-scored as failure)")

    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
<<<<<<< HEAD
        "Qwen/Qwen2.5-7B-Instruct",
        
=======
>>>>>>> 7aa5ba846d77adebf41fc69723f9561d226c86d5
    ]

    SYSTEM = (
        "You are a precise binary judge evaluating a single reasoning step from a "
        "multi-hop question-answering system."
    )
    USER_TMPL = (
        "Context Information:\n{context}\n\n"
        "Question: {question}\n\n"
        "Task: Does this reasoning step explicitly and correctly identify the entity '{entity}'? "
        "Answer 'No' if the step fails to name the entity, says the information is missing, "
        "says it cannot be determined, or identifies a different entity — even if the step's "
        "wording is not factually false.\n"
        "Reasoning step: \"{hop}\"\n"
        "Answer with exactly one word, Yes or No."
    )

    for model_name in models:
        print(f"\n{'=' * 40}\nLoading {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        # Resolve the Yes/No token ids actually produced by this tokenizer/template.
        # Try a couple of common variants and pick whichever is a single token.
        def first_token_id(word):
            for cand in (word, " " + word):
                ids = tokenizer.encode(cand, add_special_tokens=False)
                if len(ids) == 1:
                    return ids[0]
            return tokenizer.encode(word, add_special_tokens=False)[0]

        yes_id = first_token_id("Yes")
        no_id = first_token_id("No")
        print(f"Yes token id: {yes_id} ({tokenizer.decode([yes_id])!r})  |  "
              f"No token id: {no_id} ({tokenizer.decode([no_id])!r})")

        y_true, y_pred = [], []
        latencies, token_costs = [], []
        y_fail_hedge_true, y_fail_hedge_pred = [], []
        y_fail_conf_true, y_fail_conf_pred = [], []
        raw_samples = []

        for i, item in enumerate(test_data):
            true_label = item["label"]

            if item["auto_fail"]:
                pred_label = 1
                latencies.append(0.0)
                token_costs.append(0)
            else:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(
                        context=item["context"], question=item["question"], entity=item["gold_entity"], hop=item["hop_text"]
                    )},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = enc.input_ids.shape[1]

                t0 = time.time()
                with torch.no_grad():
                    out = model(**enc)
                    next_logits = out.logits[0, -1]
                t1 = time.time()

                pred_label = 0 if next_logits[yes_id] > next_logits[no_id] else 1
                latencies.append((t1 - t0) * 1000)
                token_costs.append(input_len)  # single forward pass, no generated tokens

                if len(raw_samples) < 20:
                    raw_samples.append((
                        "yes" if pred_label == 0 else "no", true_label,
                        float(next_logits[yes_id]), float(next_logits[no_id]),
                    ))

            y_true.append(true_label)
            y_pred.append(pred_label)

            is_hedge = any(
                k in item["hop_text"].lower()
                for k in ["no information", "no mention", "not mentioned",
                          "no relevant information", "not found", "cannot find",
                          "does not mention"]
            ) or item["auto_fail"]
            if true_label == 1:
                if is_hedge:
                    y_fail_hedge_true.append(1); y_fail_hedge_pred.append(pred_label)
                else:
                    y_fail_conf_true.append(1); y_fail_conf_pred.append(pred_label)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\nRaw judge decisions (first 20, non-auto-fail only):")
        for pred, true, yl, nl in raw_samples:
            print(f"  pred={pred:<3} true={true}  yes_logit={yl:.2f}  no_logit={nl:.2f}")

        print(f"\nRESULTS FOR {model_name}:")
        print(f"Overall Accuracy: {acc:.3f} | F1: {f1:.3f}")
        hedge_acc = accuracy_score(y_fail_hedge_true, y_fail_hedge_pred) if y_fail_hedge_true else 0.0
        conf_acc = accuracy_score(y_fail_conf_true, y_fail_conf_pred) if y_fail_conf_true else 0.0
        print(f"Hedging Failure Acc: {hedge_acc:.3f}  (n={len(y_fail_hedge_true)})")
        print(f"Confident Failure Acc: {conf_acc:.3f}  (n={len(y_fail_conf_true)})")
        print(f"Average Latency per hop: {np.mean(latencies):.1f} ms")
        print(f"Average Token Cost per hop: {np.mean(token_costs):.1f} tokens")

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
