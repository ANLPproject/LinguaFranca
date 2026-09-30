import json
import torch
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from huggingface_hub import hf_hub_download

def run_evaluation(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    print(f"[*] Loading dataset from {labels_file}...")
    with open(labels_file, encoding='utf-8') as f:
        examples = [json.loads(l) for l in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    
    # Deterministic split to match the training script exactly (random_state=42)
    groups = [ex.get("clean_pair_id", ex["id"]) for ex in valid_ex]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(valid_ex, groups=groups))
    test_ex = [valid_ex[i] for i in test_idx]
    
    print(f"[*] Test set size: {len(test_ex)} examples")
    
    # HEDGING VS CONFIDENTLY WRONG ANALYSIS
    hedging_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
    
    print("\n" + "="*80)
    print("HEDGING VS CONFIDENTLY WRONG FAILURES (All Layers, Hop 1 Test Set)")
    print("="*80)
    print(f"{'Layer':<7} | {'Hedging Acc (%)':<20} | {'Confident Wrong Acc (%)':<25}")
    print("-" * 80)

    # We will do qualitative analysis ONLY on the best layer (Layer 10) to avoid spamming the screen
    qual_results_layer10 = []

    for layer_idx in range(28):
        try:
            probe_path = hf_hub_download(
                repo_id="AnishRacherla/LinguaFranca-Phase3",
                filename=f"probes/hop1_probe_layer{layer_idx}.joblib",
                repo_type="dataset"
            )
            clf_h1 = joblib.load(probe_path)
        except Exception as e:
            print(f"Layer {layer_idx:<5} | Failed to download probe")
            continue

        # Extract Hop-1 hidden states for Test Set
        X_test_hs, X_test_text, y_test = [], [], []
        for ex in test_ex:
            pt_path = hs_dir / f"{ex['id']}.pt"
            data = torch.load(pt_path)
            pooled = data['pooled']
            
            for hop in ex.get('hops', []):
                h_idx = hop['hop_idx'] - 1
                if hop.get('label') in (0, 1) and h_idx == 0 and h_idx < pooled.shape[1]:
                    X_test_hs.append(pooled[layer_idx, h_idx].detach().numpy())
                    X_test_text.append(hop['text'])
                    y_test.append(hop['label'])
                    
                    if layer_idx == 10:
                        vec = pooled[layer_idx, h_idx].detach().numpy()
                        vec = np.nan_to_num(vec)
                        prob_fail = clf_h1.predict_proba([vec])[0][1]
                        qual_results_layer10.append({
                            "prob_fail": prob_fail,
                            "true_label": hop['label'],
                            "question": ex['question'],
                            "gold_entity": hop.get("bridging_entity_gold", hop.get('wiki_links', ['Unknown'])[0] if hop.get('wiki_links') else 'Unknown'),
                            "generated_text": hop['text'],
                            "is_cf": ex.get('is_counterfactual', False)
                        })
                    
        X_test_hs = np.array(X_test_hs)
        y_test = np.array(y_test)
        X_test_text = np.array(X_test_text)
        
        fail_mask = y_test == 1
        X_fail_text = X_test_text[fail_mask]
        X_fail_hs = X_test_hs[fail_mask]
        y_fail = y_test[fail_mask]
        
        y_fail_pred = clf_h1.predict(X_fail_hs)
        is_hedging = np.array([any(k in t.lower() for k in hedging_keywords) for t in X_fail_text])
        
        hedge_acc = np.mean(y_fail_pred[is_hedging] == y_fail[is_hedging]) if np.sum(is_hedging) > 0 else 0
        wrong_acc = np.mean(y_fail_pred[~is_hedging] == y_fail[~is_hedging]) if np.sum(~is_hedging) > 0 else 0
        
        print(f"Layer {layer_idx:<5} | {hedge_acc*100:<18.1f} | {wrong_acc*100:<23.1f}")
    
    # QUALITATIVE ERROR ANALYSIS (Layer 10)
    print("\n" + "="*80)
    print("QUALITATIVE PROBE ANALYSIS (Layer 10)")
    print("="*80)
    
    qual_results_layer10.sort(key=lambda x: x["prob_fail"], reverse=True)
    
    cf_failures = [r for r in qual_results_layer10 if r["is_cf"]][:5]
    natural_failures = [r for r in qual_results_layer10 if not r["is_cf"]][:5]
    
    print("\n--- TOP 5 COUNTERFACTUAL FAILURES ---")
    for i, res in enumerate(cf_failures):
        print(f"\nSample {i+1}")
        print(f"Question:       {res['question']}")
        print(f"Gold Entity:    {res['gold_entity']}")
        print(f"Generated Hop:  {res['generated_text']}")
        print(f"True Label:     {'1 (Hallucination)' if res['true_label'] == 1 else '0 (Success)'}")
        print(f"Probe Predicts: {res['prob_fail']*100:.1f}% chance of Failure")

    print("\n\n--- TOP 5 NATURAL FAILURES (No Swapped Entities) ---")
    for i, res in enumerate(natural_failures):
        print(f"\nSample {i+1}")
        print(f"Question:       {res['question']}")
        print(f"Gold Entity:    {res['gold_entity']}")
        print(f"Generated Hop:  {res['generated_text']}")
        print(f"True Label:     {'1 (Hallucination)' if res['true_label'] == 1 else '0 (Success)'}")
        print(f"Probe Predicts: {res['prob_fail']*100:.1f}% chance of Failure")

if __name__ == "__main__":
    labels_file = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/2wikimultihopqa-phase2-labels/labeled_dataset.jsonl"
    hs_dir = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/input/linguafranca-hidden-states/hidden_states"
    run_evaluation(labels_file, hs_dir)
