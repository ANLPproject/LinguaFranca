import json
import torch
import numpy as np
import joblib
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
    #     pass

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
        # HEDGING VS CONFIDENTLY WRONG ANALYSIS
        hedging_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
        X_test_text_h1 = np.array(X_test_text)[test_h1_mask]
        
        fail_mask = y_test_h1 == 1
        X_fail_text = X_test_text_h1[fail_mask]
        X_fail_hs = X_test_h1[fail_mask]
        y_fail = y_test_h1[fail_mask]  # all 1s
        
        y_fail_pred = clf_h1.predict(X_fail_hs)
        
        is_hedging = np.array([any(k in t.lower() for k in hedging_keywords) for t in X_fail_text])
        
        hedge_acc = np.mean(y_fail_pred[is_hedging] == y_fail[is_hedging]) if np.sum(is_hedging) > 0 else 0
        wrong_acc = np.mean(y_fail_pred[~is_hedging] == y_fail[~is_hedging]) if np.sum(~is_hedging) > 0 else 0
        
        print(f"\n[Control] HEDGING VS CONFIDENTLY WRONG FAILURES (Layer {best_layer_idx}, Hop 1 Test Set)")
        print(f"Total Failures in Test Set: {len(y_fail)}")
        print(f"Hedging Failures: {np.sum(is_hedging)} | Probe Accuracy on Hedging: {hedge_acc:.3f}")
        print(f"Confident Wrong Failures: {np.sum(~is_hedging)} | Probe Accuracy on Confident Wrong: {wrong_acc:.3f}")

    else:
        print("\n[Control] Not enough Hop-1 examples to run Hop-1 Only check.")

    # 3. Cross-Hop Generalization (Train Hop 1, Test Hop 2)
    train_h2_mask = train_hop_idx == 1
    test_h2_mask = test_hop_idx == 1
    
    # We already have Hop 1 logistic regression trained (clf_h1). Let's test it on Hop 2.
    if np.sum(train_h1_mask) > 0 and (np.sum(train_h2_mask) > 0 or np.sum(test_h2_mask) > 0):
        # Pool all hop 2 examples we have
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


    # =====================================================================
    
    # Save the trained Hop-1 probe for ALL layers for Phase 5 (RAG System)
    print("\n[+] Saving trained Hop-1 probes for all layers to disk...")
    for li in layer_indices:
        X_tr, _, y_tr, _, hop_idx_tr, _ = extract_data(train_ex, hs_dir, layer_idx=li)
        h1_mask_tr = hop_idx_tr == 1
        X_tr_h1 = X_tr[h1_mask_tr]
        y_tr_h1 = y_tr[h1_mask_tr]
        
        if len(y_tr_h1) > 0:
            clf_li = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
            clf_li.fit(X_tr_h1, y_tr_h1)
            probe_save_path = hs_dir.parent / f"hop1_probe_layer{li}.joblib"
            joblib.dump(clf_li, probe_save_path)
    print(f"[+] Saved {len(layer_indices)} layer probes to {hs_dir.parent}")

    # QUALITATIVE ERROR ANALYSIS
    # =====================================================================
    print("\n" + "=" * 140)
    print("QUALITATIVE PROBE ANALYSIS (Top 10 Most Confident Hop-1 Failure Predictions)")
    print("=" * 140)
    
    if np.sum(test_h1_mask) > 0:
        qual_results = []
        # We need the original text. Let's zip X_test_txt with the predictions.
        # test_h1_mask masks the flattened array. 
        # But we want the Question and Gold Entity from test_ex.
        # Let's just manually re-evaluate test_ex for Hop 1 to have all metadata.
        
        for ex in test_ex:
            pt_path = hs_dir / f"{ex['id']}.pt"
            if not pt_path.exists(): continue
            
            data = torch.load(pt_path)
            pooled = data['pooled']
            
            for hop in ex.get('hops', []):
                h_idx = hop['hop_idx'] - 1
                if h_idx == 0 and hop.get('label') in (0, 1) and h_idx < pooled.shape[1]:
                    vec = pooled[best_layer_idx, h_idx].detach().numpy()
                    vec = np.nan_to_num(vec)
                    
                    # Predict probability of failure (label=1)
                    prob_fail = clf_h1.predict_proba([vec])[0][1]
                    
                    qual_results.append({
                        "prob_fail": prob_fail,
                        "true_label": hop['label'],
                        "question": ex['question'],
                        "gold_entity": hop.get('wiki_links', ['Unknown'])[0] if hop.get('wiki_links') else 'Unknown',
                        "generated_text": hop['text']
                    })
                    
        # Sort by most confident failure predictions
        qual_results.sort(key=lambda x: x["prob_fail"], reverse=True)
        
        for i, res in enumerate(qual_results[:10]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Question:       {res['question']}")
            print(f"Gold Entity:    {res['gold_entity']}")
            print(f"Generated Hop:  {res['generated_text']}")
            print(f"True Label:     {'1 (Hallucination)' if res['true_label'] == 1 else '0 (Success)'}")
            print(f"Probe Predicts: {res['prob_fail']*100:.1f}% chance of Failure")
            
    print("\n" + "=" * 140)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python advanced_probing.py <labels_jsonl> <hs_dir>")
    else:
        run_advanced_probing_all_layers(sys.argv[1], sys.argv[2])
