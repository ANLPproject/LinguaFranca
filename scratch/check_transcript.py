import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Anish\.gemini\antigravity-ide\brain\0250fd25-cdcc-452a-beef-030fcc5da843\.system_generated\logs\transcript.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        idx = data.get('step_index')
        if idx and idx >= 5565:
            src = data.get('source')
            typ = data.get('type')
            content = str(data.get('content', ''))
            calls = data.get('tool_calls', [])
            if src == 'USER_EXPLICIT' or (src == 'MODEL' and (content or calls)):
                print(f'=== [{idx}] {src} | {typ} ===')
                if content:
                    print(content[:600])
                if calls:
                    print('TOOL CALLS:', [c.get('name') for c in calls])
