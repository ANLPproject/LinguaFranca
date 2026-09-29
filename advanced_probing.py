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
    """Extracts X (hidden states), X_text (raw text), y (labels), is_counterfactual, hop_idxs, and group_ids."""
    X_hs, X_text, y, is_cf, hop_idxs, group_ids = [], [], [], [], [], []
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
                group_ids.append(ex.get("clean_pair_id", ex["id"]))
                
    return np.array(X_hs) if layer_idx is not None else None, X_text, np.array(y), np.array(is_cf), np.array(hop_idxs), np.array(group_ids)

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
    _, X_train_txt, y_train, _, train_hop_idx, _ = extract_data(train_ex, hs_dir, layer_idx=layer_indices[0])
    _, X_test_txt, y_test, is_cf_test, test_hop_idx, _ = extract_data(test_ex, hs_dir, layer_idx=layer_indices[0])
    
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
    # print(f"{'Layer':<7} | {'Logistic All (Acc|F1|AUC)':<27} | {'Logistic Natural-Only (Acc|AUC)':<33} | {'Linear SVM (Acc)':<18} | {'Small MLP (Acc)':<18}")
    # print("-" * 140)
    
    natural_mask = ~is_cf_test
    
    # Skipping the first massive table as we already have this data!
    # for li in layer_indices:
        X_train, _, _, _, _, _ = extract_data(train_ex, hs_dir, layer_idx=li)
        X_test, _, _, _, _, _ = extract_data(test_ex, hs_dir, layer_idx=li)
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
        

    # =====================================================================
    # 5-FOLD GROUPED CROSS-VALIDATION
    # =====================================================================
    print("\n" + "=" * 140)
    print("5-FOLD GROUPED CROSS-VALIDATION (Logistic Regression)")
    print("=" * 140)
    from sklearn.model_selection import GroupKFold
    
    gkf = GroupKFold(n_splits=5)
    print(f"{'Layer':<7} | {'Mean Acc':<10} | {'Std Acc':<10} | {'Mean AUROC':<10} | {'Std AUROC':<10}")
    print("-" * 65)
    
    for li in layer_indices:
        X_all, _, y_all, _, _, groups_all = extract_data(valid_ex, hs_dir, layer_idx=li)
        X_all = np.nan_to_num(X_all)
        
        if len(X_all) == 0:
            continue
            
        acc_scores = []
        auc_scores = []
        
        for tr_idx, te_idx in gkf.split(X_all, y_all, groups=groups_all):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]
            
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
                
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
            
            acc_scores.append(accuracy_score(y_te, y_pred))
            auc_scores.append(roc_auc_score(y_te, y_prob))
            
        if acc_scores:
            print(f"Layer {li:<1} | {np.mean(acc_scores):.3f}      | ±{np.std(acc_scores):.3f}   | {np.mean(auc_scores):.3f}       | ±{np.std(auc_scores):.3f}")
    
    print("=" * 140)

    # =====================================================================
    # EXPERIMENTAL CONTROLS (Hop-1 Only & MLP Shuffled Label)
    # =====================================================================
    print("\n" + "=" * 140)
    print("EXPERIMENTAL CONTROLS")
    print("=" * 140)
    
    # We will pick the best layer based on logistic accuracy on all test data (usually layer 10)
    best_layer_idx = 10 if 10 in layer_indices else layer_indices[len(layer_indices)//2]
    
    print(f"Running controls on Layer {best_layer_idx}")
    X_train_hs, _, _, _, _, _ = extract_data(train_ex, hs_dir, layer_idx=best_layer_idx)
    X_test_hs, _, _, _, _, _ = extract_data(test_ex, hs_dir, layer_idx=best_layer_idx)
    
    # 1. Shuffled-Label Control for MLP
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
    
    # Shuffle labels
    y_train_shuffled = np.random.permutation(y_train)
    mlp.fit(X_train_hs, y_train_shuffled)
    y_pred_mlp_shuf = mlp.predict(X_test_hs)
    shuf_acc = accuracy_score(y_test, y_pred_mlp_shuf)
    print(f"[Control] MLP with SHUFFLED LABELS -> Acc: {shuf_acc:.3f} (Should be near chance/majority class)")
    
    # 2. Hop-1 Only Check
    # Filter train and test sets for only hop_idx == 0
    train_h1_mask = train_hop_idx == 0
    test_h1_mask = test_hop_idx == 0
    
    if np.sum(train_h1_mask) > 0 and np.sum(test_h1_mask) > 0:
        X_train_hs_h1 = X_train_hs[train_h1_mask]
        y_train_h1 = y_train[train_h1_mask]
        X_test_hs_h1 = X_test_hs[test_h1_mask]
        y_test_h1 = y_test[test_h1_mask]
        
        # Train and eval Logistic probe on Hop 1 only
        clf_h1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf_h1.fit(X_train_hs_h1, y_train_h1)
        y_pred_h1 = clf_h1.predict(X_test_hs_h1)
        y_prob_h1 = clf_h1.predict_proba(X_test_hs_h1)[:, 1]
        
        print(f"\n[Control] HOP-1 ONLY LOGISTIC PROBE (Layer {best_layer_idx})")
        print(f"Train N={len(y_train_h1)}, Test N={len(y_test_h1)}")
        print(f"Majority Class Rate: {max(np.mean(y_test_h1 == 0), np.mean(y_test_h1 == 1)):.3f}")
        print(f"Acc: {accuracy_score(y_test_h1, y_pred_h1):.3f} | AUROC: {roc_auc_score(y_test_h1, y_prob_h1):.3f}")
    else:
        print("\n[Control] Not enough Hop-1 examples to run Hop-1 Only check.")

    # 3. Cross-Hop Generalization (Train Hop 1, Test Hop 2)
    train_h2_mask = train_hop_idx == 1
    test_h2_mask = test_hop_idx == 1
    
    # We already have Hop 1 logistic regression trained (clf_h1). Let's test it on Hop 2.
    if np.sum(train_h1_mask) > 0 and (np.sum(train_h2_mask) > 0 or np.sum(test_h2_mask) > 0):
        # Pool all hop 2 examples we have
        h2_mask = (train_hop_idx == 1) | (test_hop_idx == 1)
        # We need to extract them from the original train/test mix, let's just grab them:
        X_all_hs, _, y_all, _, hop_all, _ = extract_data(valid_ex, hs_dir, layer_idx=best_layer_idx)
        
        h2_mask_all = hop_all == 1
        X_all_h2 = X_all_hs[h2_mask_all]
        y_all_h2 = y_all[h2_mask_all]
        
        if len(y_all_h2) > 0:
            y_pred_h2 = clf_h1.predict(X_all_h2)
            y_prob_h2 = clf_h1.predict_proba(X_all_h2)[:, 1]
            
            print(f"\n[Control] CROSS-HOP GENERALIZATION (Train Hop 1 -> Test Hop 2, Layer {best_layer_idx})")
            print(f"Test N={len(y_all_h2)}")
            print(f"Acc: {accuracy_score(y_all_h2, y_pred_h2):.3f} | AUROC: {roc_auc_score(y_all_h2, y_prob_h2):.3f}")
    
    # 4. Bootstrap/Multiple-Seed Error Bars (Layers 9-13)
    print(f"\n[Control] MULTIPLE-SEED BOOTSTRAP (Layers 8-14)")
    print(f"{'Layer':<7} | {'Mean Acc':<10} | {'Std Dev':<10}")
    print("-" * 40)
    
    seeds = [42, 1337, 2026, 9999, 12345]
    layers_to_test = [l for l in layer_indices if 8 <= l <= 14]
    
    for li in layers_to_test:
        X_tr, _, _, _, _, _ = extract_data(train_ex, hs_dir, layer_idx=li)
        X_te, _, _, _, _, _ = extract_data(test_ex, hs_dir, layer_idx=li)
        
        accs = []
        for s in seeds:
            clf_boot = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=s))
            # Resample train data
            np.random.seed(s)
            indices = np.random.choice(len(X_tr), len(X_tr), replace=True)
            clf_boot.fit(X_tr[indices], y_train[indices])
            accs.append(accuracy_score(y_test, clf_boot.predict(X_te)))
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"Layer {li:<1} | {mean_acc:.3f}      | ±{std_acc:.3f}")
    
    print("=" * 140)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python advanced_probing.py <labels_jsonl> <hs_dir>")
    else:
        run_advanced_probing_all_layers(sys.argv[1], sys.argv[2])
