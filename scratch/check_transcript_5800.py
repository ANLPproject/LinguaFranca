import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Anish\.gemini\antigravity-ide\brain\0250fd25-cdcc-452a-beef-030fcc5da843\.system_generated\logs\transcript.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index', 0)
        if idx >= 5835:
            src = d.get('source')
            if src == 'USER_EXPLICIT':
                print(f"\n==================== USER STEP {idx} ====================")
                print(d.get('content', ''))
            elif src == 'MODEL':
                c = d.get('content', '')
                if c and ('mentor' in c.lower() or 'combined' in c.lower() or 'review' in c.lower()):
                    print(f"\n--- MODEL STEP {idx} ---")
                    print(c[:500] + '...')
