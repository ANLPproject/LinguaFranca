import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys

def extract_data(ex_list, hs_dir, layer_idx=None, target_hop=None):
    """Extracts X (hidden states), X_text (raw text), and y (labels)."""
    X_hs, X_text, y = [], [], []
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
                    if layer_idx is not None:
                        X_hs.append(pooled[layer_idx, h_idx].detach().numpy())
                    else:
                        X_hs.append(None)
                    X_text.append(hop['text'])
                    y.append(hop['label'])
    return np.array(X_hs) if layer_idx is not None else None, X_text, np.array(y)

def run_advanced_probing_all_layers(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    with open(labels_file) as f:
        examples = [json.loads(l) for l in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    train_ex, test_ex = train_test_split(valid_ex, test_size=0.2, random_state=42)
    
    # Get available layers
    sample = torch.load(hs_dir / f"{valid_ex[0]['id']}.pt")
    layer_indices = sample.get('layer_indices', list(range(sample['pooled'].shape[0])))
    
    # 1. Base Extraction to calculate Majority Class & TF-IDF
    _, X_train_txt, y_train = extract_data(train_ex, hs_dir, layer_idx=layer_indices[0])
    _, X_test_txt, y_test = extract_data(test_ex, hs_dir, layer_idx=layer_indices[0])
    
    majority_class_rate = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print("=" * 110)
    print(f"MAJORITY CLASS BASELINE: {majority_class_rate:.3f} (This is why Layer 0 was at 0.82!)")
    print("=" * 110)
    
    # TF-IDF Text Baseline
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train_txt)
    X_test_tfidf = tfidf.transform(X_test_txt)
    
    clf_text = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_text.fit(X_train_tfidf, y_train)
    y_pred_txt = clf_text.predict(X_test_tfidf)
    y_prob_txt = clf_text.predict_proba(X_test_tfidf)[:, 1]
    
    print(f"TF-IDF TEXT BASELINE -> Acc: {accuracy_score(y_test, y_pred_txt):.3f} | F1: {f1_score(y_test, y_pred_txt):.3f} | AUROC: {roc_auc_score(y_test, y_prob_txt):.3f}")
    print("=" * 110)
    
    # Headers
    print(f"{'Layer':<7} | {'Logistic (Acc | F1 | AUC)':<30} | {'Linear SVM (Acc)':<18} | {'Small MLP (Acc)':<18}")
    print("-" * 110)
    
    for li in layer_indices:
        X_train, _, _ = extract_data(train_ex, hs_dir, layer_idx=li)
        X_test, _, _ = extract_data(test_ex, hs_dir, layer_idx=li)
        
        X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)
        
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            continue
            
        # 2. Architectures (WITH STANDARD SCALER!)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        acc_lr, f1_lr, auc_lr = accuracy_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr), roc_auc_score(y_test, y_prob_lr)
        
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        svm.fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, svm.predict(X_test))
        
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500))
        mlp.fit(X_train, y_train)
        acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
            
        print(f"Layer {li:<1} | {acc_lr:.3f}  | {f1_lr:.3f} | {auc_lr:.3f}          | {acc_svm:.3f}              | {acc_mlp:.3f}")
        
    print("=" * 110)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python advanced_probing.py <labels_jsonl> <hs_dir>")
    else:
        run_advanced_probing_all_layers(sys.argv[1], sys.argv[2])
