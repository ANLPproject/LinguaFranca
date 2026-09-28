import json
import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import sys

def extract_X_y_hop_specific(ex_list, hs_dir, layer_idx, target_hop=None):
    """Extracts X and y, optionally filtering for a specific hop index (0-indexed)."""
    X, y = [], []
    for ex in ex_list:
        pt_path = hs_dir / f"{ex['id']}.pt"
        if not pt_path.exists():
            continue
            
        data = torch.load(pt_path)
        pooled = data['pooled']
        
        for hop in ex.get('hops', []):
            h_idx = hop['hop_idx'] - 1
            if hop.get('label') in (0, 1) and h_idx < pooled.shape[1]:
                if target_hop is None or h_idx == target_hop:
                    X.append(pooled[layer_idx, h_idx].detach().numpy())
                    y.append(hop['label'])
    return np.array(X), np.array(y)

def run_advanced_probing_all_layers(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    with open(labels_file) as f:
        examples = [json.loads(l) for l in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    train_ex, test_ex = train_test_split(valid_ex, test_size=0.2, random_state=42)
    
    # Get available layers
    sample = torch.load(hs_dir / f"{valid_ex[0]['id']}.pt")
    layer_indices = sample.get('layer_indices', list(range(sample['pooled'].shape[0])))
    
    print("=" * 80)
    print("ADVANCED PROBING RESULTS (ALL LAYERS)")
    print("=" * 80)
    
    # Headers
    print(f"{'Layer':<7} | {'Logistic':<10} | {'Linear SVM':<10} | {'Small MLP':<10} | {'Hop 1 -> Hop 2 (Logistic)':<25}")
    print("-" * 80)
    
    for li in layer_indices:
        # 1. Base Extraction
        X_train, y_train = extract_X_y_hop_specific(train_ex, hs_dir, li)
        X_test, y_test = extract_X_y_hop_specific(test_ex, hs_dir, li)
        X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)
        
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            print(f"{li:<7} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<25}")
            continue
            
        # 2. Architectures
        lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        acc_lr = accuracy_score(y_test, lr.predict(X_test))
        
        svm = SVC(kernel='linear').fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, svm.predict(X_test))
        
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500).fit(X_train, y_train)
        acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
        
        # 3. Cross Hop Generalization (Train Hop 1 -> Test Hop 2)
        X_train_h1, y_train_h1 = extract_X_y_hop_specific(train_ex, hs_dir, li, target_hop=0)
        X_test_h2, y_test_h2   = extract_X_y_hop_specific(test_ex, hs_dir, li, target_hop=1)
        
        acc_cross_hop = "N/A"
        if len(X_train_h1) > 0 and len(X_test_h2) > 0 and len(np.unique(y_train_h1)) > 1:
            X_train_h1, X_test_h2 = np.nan_to_num(X_train_h1), np.nan_to_num(X_test_h2)
            lr_h1 = LogisticRegression(max_iter=1000).fit(X_train_h1, y_train_h1)
            acc_cross_hop = f"{accuracy_score(y_test_h2, lr_h1.predict(X_test_h2)):.3f}"
            
        print(f"Layer {li:<1} | {acc_lr:.3f}      | {acc_svm:.3f}      | {acc_mlp:.3f}      | {acc_cross_hop}")
        
    # We can print calibration for just the best layer (11) as an example to not spam output
    print("\n" + "=" * 80)
    print("CALIBRATION CURVE FOR BEST LAYER (11) using Logistic Regression")
    print("=" * 80)
    X_train_11, y_train_11 = extract_X_y_hop_specific(train_ex, hs_dir, 11)
    X_test_11, y_test_11 = extract_X_y_hop_specific(test_ex, hs_dir, 11)
    
    if len(X_train_11) > 0:
        lr_11 = LogisticRegression(max_iter=1000).fit(np.nan_to_num(X_train_11), y_train_11)
        probs = lr_11.predict_proba(np.nan_to_num(X_test_11))[:, 1]
        prob_true, prob_pred = calibration_curve(y_test_11, probs, n_bins=5)
        
        print(f"{'Predicted Probability Bucket':<30} | {'Actual Observed Failure Rate'}")
        print("-" * 60)
        for pred, true in zip(prob_pred, prob_true):
            print(f"~ {pred*100:05.2f}% chance of failure       | {true*100:05.2f}% actually failed")
            
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python advanced_probing.py <labels_jsonl> <hs_dir>")
    else:
        run_advanced_probing_all_layers(sys.argv[1], sys.argv[2])
