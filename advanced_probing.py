import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
import sys

def extract_data(ex_list, hs_dir, layer_idx=None):
    """Extracts X (hidden states), X_text (raw text), y (labels), and is_counterfactual flag."""
    X_hs, X_text, y, is_cf, hop_idxs = [], [], [], [], []
    for ex in ex_list:
        pt_path = hs_dir / f"{ex['id']}.pt"
        if not pt_path.exists():
            continue
            
        data = torch.load(pt_path)
        pooled = data['pooled']
        
        for hop in ex.get('hops', []):
            h_idx = hop['hop_idx'] - 1
            if hop.get('label') in (0, 1) and h_idx < pooled.shape[1]:
                if layer_idx is not None:
                    X_hs.append(pooled[layer_idx, h_idx].detach().numpy())
                else:
                    X_hs.append(None)
                X_text.append(hop['text'])
                y.append(hop['label'])
                is_cf.append(ex.get("is_counterfactual", False))
                hop_idxs.append(h_idx)
                
    return np.array(X_hs) if layer_idx is not None else None, X_text, np.array(y), np.array(is_cf), np.array(hop_idxs)

def run_advanced_probing_all_layers(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    with open(labels_file) as f:
        examples = [json.loads(l) for l in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    
    # 6. Label Noise Check on Natural Hops
    natural_failures = [h["match_method"] for e in valid_ex if not e.get("is_counterfactual") 
                       for h in e.get("hops", []) if h.get("label") == 1]
    print("=" * 140)
    print("LABEL NOISE CHECK (Natural failures match method):")
    print(Counter(natural_failures))
    print("=" * 140)

    # 3. GroupShuffleSplit to avoid train/test leak
    groups = [ex.get("clean_pair_id", ex["id"]) for ex in valid_ex]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(valid_ex, groups=groups))
    train_ex = [valid_ex[i] for i in train_idx]
    test_ex = [valid_ex[i] for i in test_idx]
    
    # Get available layers
    sample = torch.load(hs_dir / f"{valid_ex[0]['id']}.pt")
    layer_indices = sample.get('layer_indices', list(range(sample['pooled'].shape[0])))
    
    # Extract base features
    _, X_train_txt, y_train, _, train_hop_idx = extract_data(train_ex, hs_dir, layer_idx=layer_indices[0])
    _, X_test_txt, y_test, is_cf_test, test_hop_idx = extract_data(test_ex, hs_dir, layer_idx=layer_indices[0])
    
    majority_class_rate = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print(f"MAJORITY CLASS BASELINE: {majority_class_rate:.3f}")
    
    # TF-IDF Text Baseline
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train_txt)
    X_test_tfidf = tfidf.transform(X_test_txt)
    
    clf_text = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_text.fit(X_train_tfidf, y_train)
    y_pred_txt = clf_text.predict(X_test_tfidf)
    y_prob_txt = clf_text.predict_proba(X_test_tfidf)[:, 1]
    print(f"TF-IDF TEXT BASELINE    -> Acc: {accuracy_score(y_test, y_pred_txt):.3f} | F1: {f1_score(y_test, y_pred_txt):.3f} | AUROC: {roc_auc_score(y_test, y_prob_txt):.3f}")
    
    # Hop Index Baseline
    max_hops = max(np.max(train_hop_idx), np.max(test_hop_idx)) + 1
    X_train_hop = np.eye(max_hops)[train_hop_idx]
    X_test_hop = np.eye(max_hops)[test_hop_idx]
    clf_hop = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_hop.fit(X_train_hop, y_train)
    y_pred_hop = clf_hop.predict(X_test_hop)
    y_prob_hop = clf_hop.predict_proba(X_test_hop)[:, 1]
    print(f"HOP-IDX ONLY BASELINE   -> Acc: {accuracy_score(y_test, y_pred_hop):.3f} | F1: {f1_score(y_test, y_pred_hop):.3f} | AUROC: {roc_auc_score(y_test, y_prob_hop):.3f}")
    print("=" * 140)
    
    # Headers
    print(f"{'Layer':<7} | {'Logistic All (Acc|F1|AUC)':<27} | {'Logistic Natural-Only (Acc|AUC)':<33} | {'Linear SVM (Acc)':<18} | {'Small MLP (Acc)':<18}")
    print("-" * 140)
    
    natural_mask = ~is_cf_test
    
    for li in layer_indices:
        X_train, _, _, _, _ = extract_data(train_ex, hs_dir, layer_idx=li)
        X_test, _, _, _, _ = extract_data(test_ex, hs_dir, layer_idx=li)
        X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)
        
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            continue
            
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        
        # Mixed metrics
        acc_lr, f1_lr, auc_lr = accuracy_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr), roc_auc_score(y_test, y_prob_lr)
        
        # Natural-only metrics
        if np.sum(natural_mask) > 0 and len(np.unique(y_test[natural_mask])) > 1:
            acc_nat = accuracy_score(y_test[natural_mask], y_pred_lr[natural_mask])
            auc_nat = roc_auc_score(y_test[natural_mask], y_prob_lr[natural_mask])
        else:
            acc_nat, auc_nat = 0.0, 0.0
            
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        svm.fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, svm.predict(X_test))
        
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500))
        mlp.fit(X_train, y_train)
        acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
            
        print(f"Layer {li:<1} | {acc_lr:.3f} | {f1_lr:.3f} | {auc_lr:.3f}          | {acc_nat:.3f} | {auc_nat:.3f}                       | {acc_svm:.3f}              | {acc_mlp:.3f}")
        
    print("=" * 140)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python advanced_probing.py <labels_jsonl> <hs_dir>")
    else:
        run_advanced_probing_all_layers(sys.argv[1], sys.argv[2])
