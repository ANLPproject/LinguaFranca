"""
fix_retrieval_conditions.py
Fixes the alphabetical p1/p2 bug in labeled.jsonl by recomputing gold passage
order using the true multi-hop reasoning sequence:
  - Hop 1 title: The gold passage entity introduced in the question prompt.
  - Hop 2 title: The bridge passage entity connected via Hop 1 reasoning graph.
Updates:
  - gold_titles: ordered chronologically [hop1_title, hop2_title]
  - gold_p1_retrieved: whether hop1_title was retrieved
  - gold_p2_retrieved: whether hop2_title was retrieved
  - retrieval_condition: both_present | only_hop1_passage | only_hop2_passage | neither
"""

import json
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

LABELED_PATH = Path('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl')
BACKUP_PATH = Path('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl.bak')

if not LABELED_PATH.exists():
    print(f"Error: {LABELED_PATH} does not exist.")
    sys.exit(1)

if not BACKUP_PATH.exists():
    print(f"Creating backup: {BACKUP_PATH}")
    shutil.copyfile(LABELED_PATH, BACKUP_PATH)
else:
    print(f"Backup already exists at: {BACKUP_PATH}")

old_conds = Counter()
new_conds = Counter()
updated_records = []

resolved_by_q = 0
resolved_by_be = 0
resolved_by_fallback = 0

with open(BACKUP_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        old_conds[r.get('retrieval_condition', 'unknown')] += 1
        
        q = r.get('question', '').lower()
        g_titles = r.get('gold_titles', [])
        ret_titles = set(r.get('retrieved_titles', []))
        
        if len(g_titles) >= 2:
            t0 = g_titles[0]
            t1 = g_titles[1]
            t0_clean = t0.split(' (')[0].lower()
            t1_clean = t1.split(' (')[0].lower()
            
            t0_in_q = (t0_clean in q) or (t0.lower() in q)
            t1_in_q = (t1_clean in q) or (t1.lower() in q)
            
            if t0_in_q and not t1_in_q:
                h1, h2 = t0, t1
                resolved_by_q += 1
            elif t1_in_q and not t0_in_q:
                h1, h2 = t1, t0
                resolved_by_q += 1
            else:
                # Check bridging entity in reasoning graph
                be = ""
                for node in r.get('reasoning_graph', []):
                    if node.get('hop') == 1:
                        be = node.get('gold_entity', '').lower()
                if be and (be in t1.lower() or t1.lower() in be):
                    h1, h2 = t0, t1
                    resolved_by_be += 1
                elif be and (be in t0.lower() or t0.lower() in be):
                    h1, h2 = t1, t0
                    resolved_by_be += 1
                else:
                    h1, h2 = t0, t1
                    resolved_by_fallback += 1
                    
            g1_hit = h1 in ret_titles
            g2_hit = h2 in ret_titles
            
            if g1_hit and g2_hit:
                cond = "both_present"
            elif g1_hit and not g2_hit:
                cond = "only_hop1_passage"
            elif not g1_hit and g2_hit:
                cond = "only_hop2_passage"
            else:
                cond = "neither"
                
            r['gold_titles'] = [h1, h2]
            r['gold_p1_retrieved'] = g1_hit
            r['gold_p2_retrieved'] = g2_hit
            r['retrieval_condition'] = cond
        else:
            cond = r.get('retrieval_condition', 'unknown')
            
        new_conds[cond] += 1
        updated_records.append(r)

# Write updated records back to labeled.jsonl
with open(LABELED_PATH, 'w', encoding='utf-8') as f:
    for r in updated_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Successfully processed and updated {len(updated_records)} records in {LABELED_PATH}!")
print(f"  Resolved by question entity match:  {resolved_by_q} ({resolved_by_q/len(updated_records)*100:.1f}%)")
print(f"  Resolved by bridging entity match:  {resolved_by_be} ({resolved_by_be/len(updated_records)*100:.1f}%)")
print(f"  Fallback:                           {resolved_by_fallback} ({resolved_by_fallback/len(updated_records)*100:.1f}%)")

print("\n" + "="*60)
print("RETRIEVAL CONDITION COMPARISON (OLD vs NEW)")
print("="*60)
print(f"{'Condition':<25} | {'Old Alphabetical':<16} | {'New Hop-Order':<16}")
print("-" * 60)
all_keys = set(old_conds.keys()) | set(new_conds.keys())
for k in sorted(all_keys):
    print(f"{k:<25} | {old_conds.get(k, 0):>16} | {new_conds.get(k, 0):>16}")
print("="*60)
