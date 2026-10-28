import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

meta = json.load(open('outputs/pipeline/tfidf/probes/probes_meta.json'))
test_ids = set(meta['test_ids'])

with open('data_hf_cache/tfidf_distractor_rag/data/raw/2wikimultihopqa/labeled.jsonl', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if json.loads(l)['id'] in test_ids]

hedging_keywords = [
    'no information', 'no mention', 'not mentioned', 'no relevant information',
    'not found', 'cannot find', 'does not mention', 'unclear', 'no context'
]

reasons = {
    'restated_question_subject': [],
    'genuine_distractor_copy': [],
    'pure_fabrication': []
}

for r in records:
    for h in r.get('hops', []):
        if h.get('hop_idx') == 1 and h.get('label') == 1:
            txt = h.get('text', '')
            if not any(k in txt.lower() for k in hedging_keywords):
                pe = h.get('bridging_entity_pred', '').strip()
                q = r['question']
                ge = h.get('bridging_entity_gold', '')
                
                context_str = r.get('context', '')
                if isinstance(context_str, list):
                    flat_ctx = " ".join([f"{p[0]} {' '.join(p[1])}" if isinstance(p, (list, tuple)) and len(p)>1 else str(p) for p in context_str]).lower()
                else:
                    flat_ctx = str(context_str).lower()
                
                if pe and pe.lower() in q.lower():
                    reasons['restated_question_subject'].append((q, txt, pe, ge))
                elif pe and pe.lower() in flat_ctx:
                    reasons['genuine_distractor_copy'].append((q, txt, pe, ge))
                else:
                    reasons['pure_fabrication'].append((q, txt, pe, ge))

print(f"Total Confident Hop-1 Failures: {sum(len(v) for v in reasons.values())}")
print(f"1. Restated Question Subject (Dodged Question): {len(reasons['restated_question_subject'])} ({len(reasons['restated_question_subject'])/119*100:.1f}%)")
print(f"2. Genuine Distractor Copy (Copied from Context): {len(reasons['genuine_distractor_copy'])} ({len(reasons['genuine_distractor_copy'])/119*100:.1f}%)")
print(f"3. Pure Fabrication (Invented/Parametric):        {len(reasons['pure_fabrication'])} ({len(reasons['pure_fabrication'])/119*100:.1f}%)")

print("\n--- SAMPLE 3 RESTATED QUESTION SUBJECTS ---")
for q, txt, pe, ge in reasons['restated_question_subject'][:3]:
    print(f"Q:    {q}")
    print(f"Hop:  {txt}")
    print(f"Extracted: '{pe}' | Gold was: '{ge}'\n")

print("--- SAMPLE 3 GENUINE DISTRACTOR COPIES ---")
for q, txt, pe, ge in reasons['genuine_distractor_copy'][:3]:
    print(f"Q:    {q}")
    print(f"Hop:  {txt}")
    print(f"Extracted: '{pe}' | Gold was: '{ge}'\n")

print("--- SAMPLE 3 PURE FABRICATIONS ---")
for q, txt, pe, ge in reasons['pure_fabrication'][:3]:
    print(f"Q:    {q}")
    print(f"Hop:  {txt}")
    print(f"Extracted: '{pe}' | Gold was: '{ge}'\n")
