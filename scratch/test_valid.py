import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('kaggle_phase2_probes.ipynb', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    first_line = ''.join(c['source']).strip().split('\n')[0]
    print(f"Cell {i:02d} [{ctype}]: {first_line[:80]}")
