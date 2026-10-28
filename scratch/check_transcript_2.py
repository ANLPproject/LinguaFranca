import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Anish\.gemini\antigravity-ide\brain\0250fd25-cdcc-452a-beef-030fcc5da843\.system_generated\logs\transcript.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        idx = data.get('step_index')
        if idx and 5567 <= idx <= 5600:
            src = data.get('source')
            typ = data.get('type')
            content = str(data.get('content', ''))
            if src in ('USER_EXPLICIT', 'MODEL') and content:
                print(f'=== [{idx}] {src} ===')
                print(content)
