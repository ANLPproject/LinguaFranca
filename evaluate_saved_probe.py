import json
import torch
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from huggingface_hub import hf_hub_download

def run_evaluation(labels_file, hs_dir, layer_idx=10):
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
    
    print(f"[*] Downloading Layer {layer_idx} probe from Hugging Face...")
    try:
        probe_path = hf_hub_download(
            repo_id="AnishRacherla/LinguaFranca-Phase3",
            filename=f"probes/hop1_probe_layer{layer_idx}.joblib",
            repo_type="dataset"
        )
        clf_h1 = joblib.load(probe_path)
        print("[+] Successfully loaded trained probe!")
    except Exception as e:
        print(f"[-] Failed to download probe from HF: {e}")
        return

    # Extract Hop-1 hidden states for Test Set
    X_test_hs, X_test_text, y_test, is_cf_test = [], [], [], []
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
                is_cf_test.append(ex.get("is_counterfactual", False))
                
    X_test_hs = np.array(X_test_hs)
    y_test = np.array(y_test)
    X_test_text = np.array(X_test_text)
    
    # HEDGING VS CONFIDENTLY WRONG ANALYSIS
    hedging_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
    
    fail_mask = y_test == 1
    X_fail_text = X_test_text[fail_mask]
    X_fail_hs = X_test_hs[fail_mask]
    y_fail = y_test[fail_mask]
    
    y_fail_pred = clf_h1.predict(X_fail_hs)
    is_hedging = np.array([any(k in t.lower() for k in hedging_keywords) for t in X_fail_text])
    
    hedge_acc = np.mean(y_fail_pred[is_hedging] == y_fail[is_hedging]) if np.sum(is_hedging) > 0 else 0
    wrong_acc = np.mean(y_fail_pred[~is_hedging] == y_fail[~is_hedging]) if np.sum(~is_hedging) > 0 else 0
    
    print("\n" + "="*80)
    print(f"HEDGING VS CONFIDENTLY WRONG FAILURES (Layer {layer_idx}, Hop 1 Test Set)")
    print("="*80)
    print(f"Total Failures in Test Set: {len(y_fail)}")
    print(f"Hedging Failures: {np.sum(is_hedging)} | Probe Accuracy on Hedging: {hedge_acc:.3f}")
    print(f"Confident Wrong Failures: {np.sum(~is_hedging)} | Probe Accuracy on Confident Wrong: {wrong_acc:.3f}")
    
    # QUALITATIVE ERROR ANALYSIS
    print("\n" + "="*80)
    print("QUALITATIVE PROBE ANALYSIS (Top 10 Most Confident Hop-1 Failure Predictions)")
    print("="*80)
    
    qual_results = []
    # Re-extract all test hops to get probabilities and metadata
    for ex in test_ex:
        pt_path = hs_dir / f"{ex['id']}.pt"
        data = torch.load(pt_path)
        pooled = data['pooled']
        for hop in ex.get('hops', []):
            h_idx = hop['hop_idx'] - 1
            if h_idx == 0 and hop.get('label') in (0, 1) and h_idx < pooled.shape[1]:
                vec = pooled[layer_idx, h_idx].detach().numpy()
                vec = np.nan_to_num(vec)
                prob_fail = clf_h1.predict_proba([vec])[0][1]
                
                qual_results.append({
                    "prob_fail": prob_fail,
                    "true_label": hop['label'],
                    "question": ex['question'],
                    "gold_entity": hop.get('wiki_links', ['Unknown'])[0] if hop.get('wiki_links') else 'Unknown',
                    "generated_text": hop['text']
                })
                
    qual_results.sort(key=lambda x: x["prob_fail"], reverse=True)
    for i, res in enumerate(qual_results[:10]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Question:       {res['question']}")
        print(f"Gold Entity:    {res['gold_entity']}")
        print(f"Generated Hop:  {res['generated_text']}")
        print(f"True Label:     {'1 (Hallucination)' if res['true_label'] == 1 else '0 (Success)'}")
        print(f"Probe Predicts: {res['prob_fail']*100:.1f}% chance of Failure")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_saved_probe.py <labels_jsonl> <hs_dir> [layer_idx]")
    else:
        layer_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run_evaluation(sys.argv[1], sys.argv[2], layer_idx)
