import json
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

meta = json.load(open('outputs/pipeline/tfidf/probes/probes_meta.json'))
test_ids = set(meta['test_ids'])

with open('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if json.loads(l)['id'] in test_ids]

print(f"Loaded {len(records)} test records.")

# Helper to recompute true hop 1 and hop 2 titles
def get_ordered_gold_titles(r):
    q = r['question']
    g_titles = r.get('gold_titles', [])
    if len(g_titles) < 2:
        return g_titles, None, None
    
    # Check which title is in question
    t0_in_q = g_titles[0].split(' (')[0].lower() in q.lower() or g_titles[0].lower() in q.lower()
    t1_in_q = g_titles[1].split(' (')[0].lower() in q.lower() or g_titles[1].lower() in q.lower()
    
    if t0_in_q and not t1_in_q:
        h1_title, h2_title = g_titles[0], g_titles[1]
    elif t1_in_q and not t0_in_q:
        h1_title, h2_title = g_titles[1], g_titles[0]
    else:
        # Fallback: check gold bridging entity in hops if available
        h1_be = ""
        for h in r.get('hops', []):
            if h.get('hop_idx') == 1:
                h1_be = h.get('bridging_entity_gold', '')
        if h1_be and (h1_be.lower() in g_titles[1].lower()):
            h1_title, h2_title = g_titles[0], g_titles[1]
        elif h1_be and (h1_be.lower() in g_titles[0].lower()):
            h1_title, h2_title = g_titles[1], g_titles[0]
        else:
            h1_title, h2_title = g_titles[0], g_titles[1]
            
    return g_titles, h1_title, h2_title

recomputed = []
for r in records:
    g_titles, h1_t, h2_t = get_ordered_gold_titles(r)
    ret_titles = set(r.get('retrieved_titles', []))
    
    h1_ret = (h1_t in ret_titles) if h1_t else False
    h2_ret = (h2_t in ret_titles) if h2_t else False
    
    if h1_ret and h2_ret:
        cond = 'both_present'
    elif h1_ret and not h2_ret:
        cond = 'only_hop1_present'
    elif not h1_ret and h2_ret:
        cond = 'only_hop2_present'
    else:
        cond = 'neither_present'
        
    h1_label = None
    h2_label = None
    h1_pred_ans = ""
    h2_pred_ans = ""
    h1_gold_ans = ""
    h2_gold_ans = ""
    
    for h in r.get('hops', []):
        if h.get('hop_idx') == 1:
            h1_label = h.get('label')
            h1_pred_ans = h.get('bridging_entity_pred', '')
            h1_gold_ans = h.get('bridging_entity_gold', '')
        elif h.get('hop_idx') == 2:
            h2_label = h.get('label')
            h2_pred_ans = h.get('bridging_entity_pred', '') # or answer
            h2_gold_ans = h.get('bridging_entity_gold', '')
            
    recomputed.append({
        'id': r['id'],
        'question': r['question'],
        'h1_title': h1_t,
        'h2_title': h2_t,
        'h1_ret': h1_ret,
        'h2_ret': h2_ret,
        'cond': cond,
        'h1_label': h1_label,
        'h2_label': h2_label,
        'h1_pred': h1_pred_ans,
        'h1_gold': h1_gold_ans,
        'h2_pred': h2_pred_ans,
        'h2_gold': h2_gold_ans,
        'gold_ans': r.get('answer', ''),
        'hops': r.get('hops', []),
        'context': r.get('context', [])
    })

print("\n--- RECOMPUTED RETRIEVAL CONDITION COUNTS ---")
cond_counts = Counter(x['cond'] for x in recomputed)
for k, v in cond_counts.items():
    print(f"  {k:20s}: {v:4d} ({v/len(recomputed)*100:5.1f}%)")

print("\n--- HOP FAILURE RATES OVERALL ---")
h1_fails = sum(1 for x in recomputed if x['h1_label'] == 1)
h1_total = sum(1 for x in recomputed if x['h1_label'] is not None)
h2_fails = sum(1 for x in recomputed if x['h2_label'] == 1)
h2_total = sum(1 for x in recomputed if x['h2_label'] is not None)
print(f"Hop 1: {h1_fails}/{h1_total} ({h1_fails/h1_total*100:.1f}%)")
print(f"Hop 2: {h2_fails}/{h2_total} ({h2_fails/h2_total*100:.1f}%)")

print("\n--- HOP 2 FAILURE RATE GIVEN HOP 1 SUCCEEDED (N=648) ---")
h1_succ = [x for x in recomputed if x['h1_label'] == 0 and x['h2_label'] is not None]
h2_fails_given_h1_succ = sum(1 for x in h1_succ if x['h2_label'] == 1)
print(f"Hop 2 fails after Hop 1 succeeded: {h2_fails_given_h1_succ}/{len(h1_succ)} ({h2_fails_given_h1_succ/len(h1_succ)*100:.1f}%)")

print("\n--- HOP 2 FAILURE BREAKDOWN BY RECOMPUTED RETRIEVAL CONDITION ---")
for cond in ['both_present', 'only_hop1_present', 'only_hop2_present', 'neither_present']:
    subset = [x for x in recomputed if x['cond'] == cond and x['h2_label'] is not None]
    fails = sum(1 for x in subset if x['h2_label'] == 1)
    tot = len(subset)
    # also when hop 1 succ
    subset_h1_succ = [x for x in subset if x['h1_label'] == 0]
    fails_h1_succ = sum(1 for x in subset_h1_succ if x['h2_label'] == 1)
    tot_h1_succ = len(subset_h1_succ)
    print(f"Condition: {cond}")
    print(f"  All:            {fails}/{tot} ({fails/tot*100:.1f}%)" if tot > 0 else "  All: 0/0")
    print(f"  When Hop 1 OK:  {fails_h1_succ}/{tot_h1_succ} ({fails_h1_succ/tot_h1_succ*100:.1f}%)" if tot_h1_succ > 0 else "  When Hop 1 OK: 0/0")
