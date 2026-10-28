import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

count = 0
with open('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('retrieval_condition') == 'only_p2':
            q = d['question']
            g_titles = d.get('gold_titles', [])
            ret_titles = d.get('retrieved_titles', [])
            
            # Find which title is mentioned in the question (True Hop 1 passage)
            h1 = None
            for t in g_titles:
                clean_t = t.split(' (')[0].lower()
                if clean_t in q.lower() or t.lower() in q.lower():
                    h1 = t
                    break
            
            print(f"Example {count+1}:")
            print(f"  Q: {q}")
            print(f"  Stored gold_titles: {g_titles}")
            print(f"  Retrieved titles:   {ret_titles}")
            print(f"  True Hop 1 title:   '{h1}'")
            if len(g_titles) >= 2:
                print(f"  Stored p1: '{g_titles[0]}' | Stored p2: '{g_titles[1]}'")
                is_p1_true_h1 = (g_titles[0] == h1)
                print(f"  Is Stored p1 the True Hop 1? {is_p1_true_h1}")
                print(f"  Was True Hop 1 actually retrieved? {h1 in ret_titles if h1 else 'Unknown'}")
            print("-" * 60)
            count += 1
            if count >= 10:
                break
