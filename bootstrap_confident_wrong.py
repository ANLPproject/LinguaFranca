import json
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
import sys

def run_bootstrap(labels_file, hs_dir):
    hs_dir = Path(hs_dir)
    print(f"[*] Loading dataset from {labels_file}...")
    with open(labels_file, encoding='utf-8') as f:
        examples = [json.loads(l) for l in f]
        
    valid_ex = [ex for ex in examples if (hs_dir / f"{ex['id']}.pt").exists()]
    
    layer_indices = range(8, 15)
    random_states = [42, 43, 44, 45, 46]
    hedging_keywords = ["no information", "no mention", "not mentioned", "no relevant information", "not found", "cannot find", "does not mention"]
    
    print("\n" + "="*80)
    print("BOOTSTRAP ERROR BARS: CONFIDENTLY WRONG FAILURES (Layers 8-14, 5-Fold)")
    print("="*80)
    print(f"{'Layer':<7} | {'Mean Confident Wrong Acc (%)':<30} | {'Std Dev (%)':<12} | {'N (Wrong/Hedge)':<15}")
    print("-" * 80)
    
    for layer_idx in layer_indices:
        layer_wrong_accs = []
        n_wrong_list = []
        n_hedge_list = []
        
        for seed in random_states:
            groups = [ex.get("clean_pair_id", ex["id"]) for ex in valid_ex]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(gss.split(valid_ex, groups=groups))
            train_ex = [valid_ex[i] for i in train_idx]
            test_ex = [valid_ex[i] for i in test_idx]
            
            # Extract Train
            X_tr, y_tr = [], []
            for ex in train_ex:
                pt_path = hs_dir / f"{ex['id']}.pt"
                data = torch.load(pt_path)
                pooled = data['pooled']
                for hop in ex.get('hops', []):
                    h_idx = hop['hop_idx'] - 1
                    if hop.get('label') in (0, 1) and h_idx == 0 and h_idx < pooled.shape[1]:
                        X_tr.append(pooled[layer_idx, h_idx].detach().numpy())
                        y_tr.append(hop['label'])
            
            X_tr = np.array(X_tr)
            y_tr = np.array(y_tr)
            
            # Train model
            if len(y_tr) > 0:
                clf_h1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
                clf_h1.fit(X_tr, y_tr)
            else:
                continue
            
            # Extract Test
            X_te, X_te_txt, y_te = [], [], []
            for ex in test_ex:
                pt_path = hs_dir / f"{ex['id']}.pt"
                data = torch.load(pt_path)
                pooled = data['pooled']
                for hop in ex.get('hops', []):
                    h_idx = hop['hop_idx'] - 1
                    if hop.get('label') in (0, 1) and h_idx == 0 and h_idx < pooled.shape[1]:
                        X_te.append(pooled[layer_idx, h_idx].detach().numpy())
                        X_te_txt.append(hop['text'])
                        y_te.append(hop['label'])
                        
            X_te = np.array(X_te)
            y_te = np.array(y_te)
            X_te_txt = np.array(X_te_txt)
            
            fail_mask = y_te == 1
            X_fail_hs = X_te[fail_mask]
            X_fail_text = X_te_txt[fail_mask]
            y_fail = y_te[fail_mask]
            
            y_fail_pred = clf_h1.predict(X_fail_hs)
            is_hedging = np.array([any(k in t.lower() for k in hedging_keywords) for t in X_fail_text])
            
            n_wrong = np.sum(~is_hedging)
            n_hedge = np.sum(is_hedging)
            wrong_acc = np.mean(y_fail_pred[~is_hedging] == y_fail[~is_hedging]) if n_wrong > 0 else 0
            
            layer_wrong_accs.append(wrong_acc)
            n_wrong_list.append(n_wrong)
            n_hedge_list.append(n_hedge)
            
        mean_acc = np.mean(layer_wrong_accs) * 100
        std_acc = np.std(layer_wrong_accs) * 100
        avg_n_wrong = np.mean(n_wrong_list)
        avg_n_hedge = np.mean(n_hedge_list)
        
        print(f"Layer {layer_idx:<5} | {mean_acc:<30.1f} | ±{std_acc:<10.1f} | ~{int(avg_n_wrong)} / {int(avg_n_hedge)}")

if __name__ == "__main__":
    labels_file = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/2wikimultihopqa-phase2-labels/labeled_dataset.jsonl"
    hs_dir = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/input/linguafranca-hidden-states/hidden_states"
    run_bootstrap(labels_file, hs_dir)
