import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

def roc_auc_score(y_true, y_score):
    # Mann-Whitney U statistic for AUROC
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    rank_sum = sum(i + 1 for i, (s, label) in enumerate(ranked) if label == 1)
    u = rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return u / (n_pos * n_neg)

def roc_curve(y_true, y_score):
    # Sort descending
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    fpr, tpr, thresh = [0.0], [0.0], [float('inf')]
    tp, fp = 0, 0
    for s, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / n_neg if n_neg else 0)
        tpr.append(tp / n_pos if n_pos else 0)
        thresh.append(s)
    return fpr, tpr, thresh

# Load probes_meta to get test_ids
with open('outputs/pipeline/tfidf/probes/probes_meta.json', encoding='utf-8') as f:
    meta = json.load(f)
test_ids = set(meta['test_ids'])

# Load labeled examples for test set
labeled_map = {}
with open('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d['id'] in test_ids:
            labeled_map[d['id']] = d

print(f"Total test examples loaded: {len(labeled_map)}")

preds_by_task = {'hop1': {}, 'hop2': {}, 'hop12': {}}
with open('outputs/pipeline/tfidf/probes/test_predictions.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        task = d['task']
        hop = d['hop']
        ex_id = d['id']
        if task in preds_by_task:
            preds_by_task[task][(ex_id, hop)] = d

hedging_keywords = [
    "no information", "no mention", "not mentioned", "no relevant information",
    "not found", "cannot find", "does not mention", "unclear", "no context",
    "the context does not", "insufficient information", "not specified"
]

def analyze_hop(hop_num, task_name="hop1"):
    y_list, p_list, is_hedge_list, ent_in_ctx_list = [], [], [], []
    
    for ex_id, ex in labeled_map.items():
        key = (ex_id, hop_num)
        pred_entry = preds_by_task[task_name].get(key)
        if not pred_entry:
            continue
        
        hops = ex.get('hops', [])
        target_hop = None
        for h in hops:
            if h.get('hop_idx') == hop_num and h.get('label') in (0, 1):
                target_hop = h
                break
        if not target_hop:
            continue
        
        y = target_hop['label']
        p = pred_entry['p_fail']
        text = target_hop.get('text', '')
        
        is_hedge = any(k in text.lower() for k in hedging_keywords)
        pred_ent = target_hop.get('bridging_entity_pred', '')
        context_str = ex.get('context', '')
        if isinstance(context_str, list):
            flat_ctx = " ".join([f"{p[0]} {' '.join(p[1])}" if isinstance(p, (list, tuple)) and len(p)>1 else str(p) for p in context_str]).lower()
        else:
            flat_ctx = str(context_str).lower()
        
        ent_in_ctx = bool(pred_ent and pred_ent.lower() in flat_ctx)
        
        y_list.append(y)
        p_list.append(p)
        is_hedge_list.append(is_hedge)
        ent_in_ctx_list.append(ent_in_ctx)
        
    n_total = len(y_list)
    n_fail = sum(1 for y in y_list if y == 1)
    n_succ = sum(1 for y in y_list if y == 0)
    n_hedge = sum(1 for y, h in zip(y_list, is_hedge_list) if y == 1 and h)
    n_conf = sum(1 for y, h in zip(y_list, is_hedge_list) if y == 1 and not h)
    
    print("=" * 80)
    print(f"EVALUATION: Hop {hop_num} (Task probe: '{task_name}') | Total N = {n_total}")
    print(f"Total Failures: {n_fail} ({n_fail/n_total*100:.1f}%) | Successes: {n_succ} ({n_succ/n_total*100:.1f}%)")
    print(f"Hedges: {n_hedge} | Confident Failures: {n_conf}")
    print("=" * 80)
    
    # Filter for conf-wrong vs success
    m_indices = [i for i, (y, h) in enumerate(zip(y_list, is_hedge_list)) if (y == 1 and not h) or y == 0]
    y_conf_vs_succ = [1 if y_list[i] == 1 else 0 for i in m_indices]
    p_conf_vs_succ = [p_list[i] for i in m_indices]
    
    auc_conf = roc_auc_score(y_conf_vs_succ, p_conf_vs_succ)
    print(f"AUROC conf-wrong vs success: {auc_conf:.3f}")
    
    fpr, tpr, _ = roc_curve(y_conf_vs_succ, p_conf_vs_succ)
    for target_fpr in (0.05, 0.10):
        # find closest tpr where fpr <= target_fpr
        best_tpr = max([tp for fp, tp in zip(fpr, tpr) if fp <= target_fpr] or [0.0])
        print(f"recall @ FPR {target_fpr:.2f}: {best_tpr:.3f}")
        
    # Recall @ 0.5 threshold breakdown: entity IN context vs NOT in context
    conf_in_ctx = [p_list[i] for i in range(len(y_list)) if y_list[i] == 1 and not is_hedge_list[i] and ent_in_ctx_list[i]]
    conf_not_in_ctx = [p_list[i] for i in range(len(y_list)) if y_list[i] == 1 and not is_hedge_list[i] and not ent_in_ctx_list[i]]
    
    rec_in = sum(1 for p in conf_in_ctx if p > 0.5) / len(conf_in_ctx) if conf_in_ctx else 0.0
    rec_not = sum(1 for p in conf_not_in_ctx if p > 0.5) / len(conf_not_in_ctx) if conf_not_in_ctx else 0.0
    
    print(f"{'entity IN ctx (distractor grounding)':<38} N={len(conf_in_ctx):>3} | recall@0.5: {rec_in:.3f}")
    print(f"{'entity NOT in ctx (pure fabrication)':<38} N={len(conf_not_in_ctx):>3} | recall@0.5: {rec_not:.3f}")
    print()

# 1. Hop 1 (dedicated hop1 probe)
analyze_hop(1, task_name="hop1")

# 2. Hop 2 (dedicated hop2 probe)
analyze_hop(2, task_name="hop2")

# 3. All Hops (hop12 probe)
print("=" * 80)
print("ALL HOPS COMBINED (Evaluated with 'hop12' probe)")
print("=" * 80)
y_all, p_all, is_hedge_all, ent_in_ctx_all = [], [], [], []
for h_num in (1, 2):
    for ex_id, ex in labeled_map.items():
        key = (ex_id, h_num)
        pred_entry = preds_by_task['hop12'].get(key)
        if not pred_entry: continue
        hops = ex.get('hops', [])
        target_hop = None
        for h in hops:
            if h.get('hop_idx') == h_num and h.get('label') in (0, 1):
                target_hop = h; break
        if not target_hop: continue
        
        y = target_hop['label']
        p = pred_entry['p_fail']
        text = target_hop.get('text', '')
        is_hedge = any(k in text.lower() for k in hedging_keywords)
        pred_ent = target_hop.get('bridging_entity_pred', '')
        context_str = ex.get('context', '')
        if isinstance(context_str, list):
            flat_ctx = " ".join([f"{p[0]} {' '.join(p[1])}" if isinstance(p, (list, tuple)) and len(p)>1 else str(p) for p in context_str]).lower()
        else:
            flat_ctx = str(context_str).lower()
        ent_in_ctx = bool(pred_ent and pred_ent.lower() in flat_ctx)
        
        y_all.append(y)
        p_all.append(p)
        is_hedge_all.append(is_hedge)
        ent_in_ctx_all.append(ent_in_ctx)

m_all_indices = [i for i, (y, h) in enumerate(zip(y_all, is_hedge_all)) if (y == 1 and not h) or y == 0]
y_conf_all = [1 if y_all[i] == 1 else 0 for i in m_all_indices]
p_conf_all = [p_all[i] for i in m_all_indices]

auc_all = roc_auc_score(y_conf_all, p_conf_all)
print(f"All-hop AUROC conf-wrong vs success: {auc_all:.3f}")
fpr_a, tpr_a, _ = roc_curve(y_conf_all, p_conf_all)
for target_fpr in (0.05, 0.10):
    best_tpr = max([tp for fp, tp in zip(fpr_a, tpr_a) if fp <= target_fpr] or [0.0])
    print(f"All-hop recall @ FPR {target_fpr:.2f}: {best_tpr:.3f}")

conf_all_in = [p_all[i] for i in range(len(y_all)) if y_all[i] == 1 and not is_hedge_all[i] and ent_in_ctx_all[i]]
conf_all_not = [p_all[i] for i in range(len(y_all)) if y_all[i] == 1 and not is_hedge_all[i] and not ent_in_ctx_all[i]]

rec_all_in = sum(1 for p in conf_all_in if p > 0.5) / len(conf_all_in) if conf_all_in else 0.0
rec_all_not = sum(1 for p in conf_all_not if p > 0.5) / len(conf_all_not) if conf_all_not else 0.0
print(f"{'entity IN ctx (distractor grounding)':<38} N={len(conf_all_in):>3} | recall@0.5: {rec_all_in:.3f}")
print(f"{'entity NOT in ctx (pure fabrication)':<38} N={len(conf_all_not):>3} | recall@0.5: {rec_all_not:.3f}")

n_conf_all = sum(1 for y, h in zip(y_all, is_hedge_all) if y == 1 and not h)
conf_rec_overall = sum(1 for y, h, p in zip(y_all, is_hedge_all, p_all) if y == 1 and not h and p > 0.5) / n_conf_all
print(f"\nOVERALL ALL-HOP CONFIDENT WRONG RECALL @ 0.5: {conf_rec_overall*100:.1f}% (N={n_conf_all})")
