import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'c:\Users\Anish\OneDrive\Desktop\anlp_project\LinguaFranca\kaggle_phase2_probes.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb[ cells])}')
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell['source'])
    first_line = src.strip().split('\n')[0] if src.strip() else '(empty)'
    print(f'Cell {i:02d} [{ctype}]: {first_line[:80]}')
