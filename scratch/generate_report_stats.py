import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def print_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def analyze_dataset(labels_file):
    print_header("1. DATASET STATISTICS (IMBALANCES & COUNTERFACTUALS)")
    with open(labels_file, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    
    total_examples = len(examples)
    total_hops = 0
    natural_success = 0
    natural_failures = 0
    counterfactuals = 0
    hedging_count = 0
    confident_count = 0
    
    hedge_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
    
    for ex in examples:
        is_counterfactual = "clean_pair_id" in ex and ex["clean_pair_id"] != ex["id"]
        if is_counterfactual:
            counterfactuals += 1
            
        for hop in ex.get("hops", []):
            if hop.get("label") not in (0, 1): continue
            total_hops += 1
            
            label = hop["label"]
            text = hop.get("text", "").lower()
            
            if not is_counterfactual:
                if label == 0: natural_success += 1
                if label == 1: natural_failures += 1
                
            if label == 1:
                is_hedge = not text or any(k in text for k in hedge_keywords)
                if is_hedge: hedging_count += 1
                else: confident_count += 1
                
    print(f"Total Unique Examples: {total_examples}")
    print(f"Total Valid Hops Evaluated: {total_hops}")
    print(f"Natural Successes (Label 0): {natural_success}")
    print(f"Natural Failures (Label 1): {natural_failures}")
    print(f"Natural Failure Rate: {(natural_failures / (natural_success + natural_failures + 1e-9))*100:.2f}%")
    print(f"Counterfactuals Injected: {counterfactuals}")
    print(f"Total Failures Broken Down -> Hedging: {hedging_count} | Confident Hallucinations: {confident_count}")


def evaluate_probes(labels_file, hs_dir, layer=11):
    print_header(f"2. PROBE COMPARISON (LAYER {layer}) - LR vs SVM vs MLP")
    hs_dir = Path(hs_dir)
    
    with open(labels_file, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    
    X, y, groups, is_hedge = [], [], [], []
    hedge_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
    
    for ex in valid_ex:
        hs = torch.load(hs_dir / f"{ex['id']}.pt", map_location="cpu", weights_only=True)
        if layer >= hs.shape[0]: continue
        
        # Assume hs is shape [layers, hops, dim]
        for hop_idx, hop in enumerate(ex.get("hops", [])):
            if hop.get("label") not in (0, 1): continue
            if hop_idx >= hs.shape[1]: continue
            
            vec = hs[layer, hop_idx].numpy()
            X.append(vec)
            y.append(hop["label"])
            groups.append(ex.get("clean_pair_id", ex["id"]))
            
            text = hop.get("text", "").lower()
            hedge = not text or any(k in text for k in hedge_keywords)
            is_hedge.append(hedge)
            
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    is_hedge = np.array(is_hedge)
    
    print(f"Loaded {len(X)} hidden states of dimension {X.shape[1]} for Layer {layer}.")
    
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=2000, dual=False),
        "MLP (1 Hidden Layer)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
    }
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        accs, aurocs = [], []
        hedge_accs, conf_accs = [], []
        inf_times = []
        
        for train_idx, test_idx in gss.split(X, y, groups=groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            hedge_test = is_hedge[test_idx]
            
            model.fit(X_train, y_train)
            
            t0 = time.perf_counter()
            preds = model.predict(X_test)
            t1 = time.perf_counter()
            inf_times.append((t1 - t0) / len(X_test) * 1000) # ms per sample
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.decision_function(X_test)
                
            accs.append(accuracy_score(y_test, preds))
            aurocs.append(roc_auc_score(y_test, probs))
            
            # Hedging vs Confident
            fail_idx = (y_test == 1)
            if np.sum(fail_idx) > 0:
                h_idx = fail_idx & hedge_test
                c_idx = fail_idx & ~hedge_test
                if np.sum(h_idx) > 0:
                    hedge_accs.append(accuracy_score(y_test[h_idx], preds[h_idx]))
                if np.sum(c_idx) > 0:
                    conf_accs.append(accuracy_score(y_test[c_idx], preds[c_idx]))
                    
        print(f"Accuracy:        {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"AUROC:           {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
        print(f"Hedging Acc:     {np.mean(hedge_accs):.3f}")
        print(f"Confident Acc:   {np.mean(conf_accs):.3f}")
        print(f"Inference Time:  {np.mean(inf_times):.4f} ms per hop")

if __name__ == "__main__":
    # Point these to the kaggle downloaded dataset
    LABELS_FILE = "data/2wikimultihopqa/augmented.jsonl"
    HS_DIR = "data/hidden_states"
    
    import os
    if not os.path.exists(LABELS_FILE):
        print(f"Could not find {LABELS_FILE}. Please update the path.")
    else:
        analyze_dataset(LABELS_FILE)
        evaluate_probes(LABELS_FILE, HS_DIR, layer=11)
