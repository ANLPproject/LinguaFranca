"""
LinguaFranca — Phase 1: Dataset Construction
Kaggle Run Script

HOW TO USE:
  1. In Kaggle: New Notebook → paste each CELL below into separate code cells
  2. Set Accelerator = GPU T4 x2
  3. Turn Internet ON
  4. Add your 'linguafranca-src' dataset
  5. Run All
"""

# ════════════════════════════════════════════════════════════
# CELL 1 — Check GPU
# ════════════════════════════════════════════════════════════
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                    '--format=csv,noheader'], capture_output=True, text=True)
print('GPU:', r.stdout.strip() or '❌ NOT FOUND — enable GPU in Settings!')


# ════════════════════════════════════════════════════════════
# CELL 2 — Install dependencies
# ════════════════════════════════════════════════════════════
# (run as a shell cell in Kaggle: prefix with !)
# !pip install nnsight sentence-transformers SPARQLWrapper datasets accelerate -q


# ════════════════════════════════════════════════════════════
# CELL 3 — Unpack source code
# ════════════════════════════════════════════════════════════
import zipfile, os, sys

ZIP_PATH = '/kaggle/input/linguafranca-src/linguafranca_src.zip'
WORK_DIR = '/kaggle/working/LinguaFranca'

os.makedirs(WORK_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(WORK_DIR)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
print('Working dir:', os.getcwd())
print('Contents:',    os.listdir('.'))


# ════════════════════════════════════════════════════════════
# CELL 4 — Configure (Qwen, Kaggle paths, scale)
# ════════════════════════════════════════════════════════════
import yaml

with open('configs/data_config.yaml') as f:
    cfg = yaml.safe_load(f)

# Model — Qwen needs no HF token
cfg['model']['name']              = 'Qwen/Qwen2.5-3B-Instruct'
cfg['model']['trust_remote_code'] = True
cfg['model']['dtype']             = 'bfloat16'

# Paths → Kaggle working dir (persistent for this session)
cfg['data']['raw_dir']            = '/kaggle/working/data/raw'
cfg['data']['processed_dir']      = '/kaggle/working/data/processed'
cfg['data']['hidden_states_dir']  = '/kaggle/working/data/hidden_states'
cfg['matching']['wikidata_aliases_path'] = '/kaggle/working/data/raw/wikidata_aliases.json'

# ─── Scale ──────────────────────────────────────────────────────
# 8000 examples ≈ 4–5 hrs on T4  (Kaggle limit: 12 hrs) ← FINE
# 2000 examples ≈ 1–2 hrs        ← quick test / first run
# Change this to 8000 for the full dataset once you've tested:
cfg['data']['n_source_examples']  = 8000   # ← FULL SCALE
# cfg['data']['n_source_examples'] = 2000  # ← quick test

cfg['generation']['batch_size']   = 8      # T4 can handle 8

with open('configs/data_config.yaml', 'w') as f:
    yaml.dump(cfg, f)

print(f"Model      : {cfg['model']['name']}")
print(f"n_examples : {cfg['data']['n_source_examples']}")
print(f"batch_size : {cfg['generation']['batch_size']}")
print(f"raw_dir    : {cfg['data']['raw_dir']}")


# ════════════════════════════════════════════════════════════
# CELL 5 — Download datasets
# ════════════════════════════════════════════════════════════
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s')

from src.data.download import download_all
paths = download_all(cfg)
for ds, splits in paths.items():
    for split, p in splits.items():
        print(f'  {ds}/{split} → {p}')


# ════════════════════════════════════════════════════════════
# CELL 6 — Generate CoT + extract hidden states
#          ⏱ ~4-5 hrs for 8000 examples on T4
# ════════════════════════════════════════════════════════════
from src.data.generate_cot import run_generation
cot_path = run_generation(cfg, dry_run=False)
print(f'\nCoT saved → {cot_path}')


# ════════════════════════════════════════════════════════════
# CELL 7 — Label hops
# ════════════════════════════════════════════════════════════
from src.data.label_hops import run_labeling
labeled_path = run_labeling(cfg)

import json
with open(labeled_path) as f:
    examples = [json.loads(l) for l in f]

total_hops = sum(len(e['hops']) for e in examples)
fail_hops  = sum(1 for e in examples for h in e['hops'] if h['label'] == 1)
print(f'\nTotal hops : {total_hops}')
print(f'Failures   : {fail_hops}  ({100*fail_hops/total_hops:.1f}%)')
print(f'(Target after counterfactuals: ~45-50%)')


# ════════════════════════════════════════════════════════════
# CELL 8 — Build counterfactuals
# ════════════════════════════════════════════════════════════
from src.data.counterfactuals import run_counterfactuals
augmented_path = run_counterfactuals(cfg)
print(f'\nAugmented data → {augmented_path}')


# ════════════════════════════════════════════════════════════
# CELL 9 — Split and write final JSONL
# ════════════════════════════════════════════════════════════
import random
from pathlib import Path
from src.data.build_dataset import (
    split_by_id, check_class_balance,
    check_no_leakage, save_jsonl, sample_hotpotqa
)

with open(augmented_path) as f:
    all_examples = [json.loads(l) for l in f]

rng    = random.Random(cfg.get('seed', 42))
splits = cfg['data']['splits']
train, val, test = split_by_id(all_examples, splits['train'], splits['val'], rng)

print('Class balance:')
check_class_balance('train', train)
check_class_balance('val',   val)
check_class_balance('test',  test)
check_no_leakage(train, val, test)

processed = Path(cfg['data']['processed_dir'])
save_jsonl(train, processed / 'train.jsonl')
save_jsonl(val,   processed / 'val.jsonl')
save_jsonl(test,  processed / 'test.jsonl')

hotpot = sample_hotpotqa(cfg, rng)
if hotpot:
    save_jsonl(hotpot, processed / 'hotpotqa_test.jsonl')

print(f'\n✅  Done!  train={len(train)} | val={len(val)} | test={len(test)}')


# ════════════════════════════════════════════════════════════
# CELL 10 — Inspect a sample
# ════════════════════════════════════════════════════════════
import random, json
from pathlib import Path

processed = Path(cfg['data']['processed_dir'])
with open(processed / 'train.jsonl') as f:
    train_data = [json.loads(l) for l in f]

for ex in random.sample(train_data, 3):
    print('─' * 60)
    print(f"Q : {ex['question']}")
    print(f"A : {ex['gold_answer']}   |   CF={ex['is_counterfactual']}")
    for h in ex['hops']:
        sym = '✓' if h['label'] == 0 else '✗'
        print(f"  hop{h['hop_idx']} [{sym}] via {h['match_method']}  gold={h['bridging_entity_gold']!r}")
    print(f"  first_fail_hop: {ex['first_fail_hop']}")

print('\nOutput files are in /kaggle/working/data/processed/')
print('Download via: Notebook → Data tab → Output')
