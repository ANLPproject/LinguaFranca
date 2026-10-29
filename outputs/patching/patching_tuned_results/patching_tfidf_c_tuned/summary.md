# Phase 4 — activation patching, TF-IDF setting (meta-llama/Llama-3.2-3B-Instruct, via meta-llama/Llama-3.2-3B-Instruct)

Pairs: 259 selected (split=test, kinds=hedge,other_wrong_entity), 219 usable (skipped {}); 219 completed. Informative pairs (clean run states the gold hop-1 entity, CF run does not): **211** {'hedge': 188, 'other_wrong_entity': 23}. Clean and CF prompts differ only in the entity (entity span median 5 tokens, tail median 6). Success = gold hop-1 entity or a Wikidata alias, whole words.

## Checks
- hook self-test (all positions, last block): max |logit diff| 0.00e+00 (unpatched 7.55e+00) — must be ~0 vs large
- sanity (patch ALL prompt positions at block 0): 1.000 restored, generated text identical to the clean run in 0.986 of pairs — both should be ~1
- baselines: clean run states the gold hop-1 entity 0.973; CF run 0.009; CF run hedges ('not mentioned') 0.886
- probe-link self-check (id, cosine recomputed vs stored state, stored P, recomputed P): [['ec7da1b20bda11eba7f7acde48001122', 1.0, 0.3337, 0.3348], ['ec7da1b20bda11eba7f7acde48001122_cf', 1.0, 0.979, 0.979], ['089c29860bdb11eba7f7acde48001122', 1.0, 0.0438, 0.0438], ['089c29860bdb11eba7f7acde48001122_cf', 1.0, 0.192, 0.192], ['5ba8a10a0bda11eba7f7acde48001122', 1.0, 0.0371, 0.0371], ['5ba8a10a0bda11eba7f7acde48001122_cf', 1.0, 0.8134, 0.8134]]

## Experiment A — hop-1 bridge entity restored, by patched block

| block | entity span | tail (tokens after entity) | last prompt token | donor control (third entity) |
|---|---|---|---|---|
| 0 | 1.00 [0.98, 1.00] (n=211) | n/a | n/a | n/a |
| 2 | 0.97 [0.94, 0.99] (n=211) | n/a | n/a | n/a |
| 4 | 0.93 [0.89, 0.96] (n=211) | n/a | n/a | n/a |
| 6 | 0.93 [0.89, 0.96] (n=211) | n/a | n/a | n/a |
| 8 | 0.90 [0.85, 0.93] (n=211) | n/a | n/a | n/a |
| 9 | 0.78 [0.72, 0.83] (n=211) | n/a | n/a | n/a |
| 10 | 0.73 [0.67, 0.79] (n=211) | n/a | n/a | n/a |
| 11 | 0.46 [0.40, 0.53] (n=211) | n/a | n/a | n/a |
| 12 | 0.45 [0.39, 0.52] (n=211) | n/a | n/a | n/a |
| 13 | 0.38 [0.32, 0.45] (n=211) | n/a | n/a | n/a |
| 14 | 0.19 [0.15, 0.25] (n=211) | n/a | n/a | n/a |
| 16 | 0.18 [0.13, 0.24] (n=211) | n/a | n/a | n/a |
| 20 | 0.18 [0.13, 0.23] (n=211) | n/a | n/a | n/a |
| 24 | 0.08 [0.05, 0.13] (n=211) | n/a | n/a | n/a |

### Entity span vs donor control (paired, same pairs)

| block | n | entity − donor | entity only | donor only | sign test p |
|---|---|---|---|---|---|
| 0 | 0 | +nan | 0 | 0 | 1 |
| 2 | 0 | +nan | 0 | 0 | 1 |
| 4 | 0 | +nan | 0 | 0 | 1 |
| 6 | 0 | +nan | 0 | 0 | 1 |
| 8 | 0 | +nan | 0 | 0 | 1 |
| 9 | 0 | +nan | 0 | 0 | 1 |
| 10 | 0 | +nan | 0 | 0 | 1 |
| 11 | 0 | +nan | 0 | 0 | 1 |
| 12 | 0 | +nan | 0 | 0 | 1 |
| 13 | 0 | +nan | 0 | 0 | 1 |
| 14 | 0 | +nan | 0 | 0 | 1 |
| 16 | 0 | +nan | 0 | 0 | 1 |
| 20 | 0 | +nan | 0 | 0 | 1 |
| 24 | 0 | +nan | 0 | 0 | 1 |

### Entity span restoration by CF failure kind

| block | hedge | other_wrong_entity |
|---|---|---|
| 0 | 1.00 [0.98, 1.00] (n=188) | 1.00 [0.86, 1.00] (n=23) |
| 2 | 0.97 [0.93, 0.99] (n=188) | 1.00 [0.86, 1.00] (n=23) |
| 4 | 0.92 [0.87, 0.95] (n=188) | 1.00 [0.86, 1.00] (n=23) |
| 6 | 0.93 [0.89, 0.96] (n=188) | 0.91 [0.73, 0.98] (n=23) |
| 8 | 0.89 [0.84, 0.93] (n=188) | 0.96 [0.79, 0.99] (n=23) |
| 9 | 0.76 [0.69, 0.82] (n=188) | 0.96 [0.79, 0.99] (n=23) |
| 10 | 0.71 [0.64, 0.77] (n=188) | 0.91 [0.73, 0.98] (n=23) |
| 11 | 0.43 [0.36, 0.50] (n=188) | 0.74 [0.54, 0.87] (n=23) |
| 12 | 0.42 [0.35, 0.49] (n=188) | 0.74 [0.54, 0.87] (n=23) |
| 13 | 0.39 [0.32, 0.46] (n=188) | 0.35 [0.19, 0.55] (n=23) |
| 14 | 0.19 [0.14, 0.25] (n=188) | 0.26 [0.13, 0.46] (n=23) |
| 16 | 0.19 [0.14, 0.25] (n=188) | 0.13 [0.05, 0.32] (n=23) |
| 20 | 0.18 [0.13, 0.24] (n=188) | 0.13 [0.05, 0.32] (n=23) |
| 24 | 0.09 [0.05, 0.13] (n=188) | 0.04 [0.01, 0.21] (n=23) |

## Experiment C — probe `/kaggle/working/probes_c001_L12` (hop12, block 12, calibrated, flagged = P ≥ τ = 0.585) on the generated hop 1

- clean baseline: 0.00 (mean p=0.05, n=211)
- CF baseline: 0.75 (mean p=0.75, n=211)
- AUROC clean vs CF baseline: 0.990
- verdict matches 'entity NOT restored' in 2555/2954 patched runs (0.865)
  - patches below the probe layer (12): 1611/1688 (0.954)
  - patches at or above the probe layer (12): 944/1266 (0.746)

| block patched (entity span) | runs where the entity WAS restored | runs where it was NOT restored |
|---|---|---|
| 0 | 0.00 (mean p=0.05, n=211) | n/a |
| 2 | 0.00 (mean p=0.06, n=205) | 0.83 (mean p=0.83, n=6) |
| 4 | 0.00 (mean p=0.06, n=196) | 0.80 (mean p=0.81, n=15) |
| 6 | 0.01 (mean p=0.07, n=196) | 0.73 (mean p=0.68, n=15) |
| 8 | 0.01 (mean p=0.10, n=189) | 0.86 (mean p=0.82, n=22) |
| 9 | 0.02 (mean p=0.14, n=165) | 0.87 (mean p=0.86, n=46) |
| 10 | 0.11 (mean p=0.26, n=155) | 0.91 (mean p=0.88, n=56) |
| 11 | 0.18 (mean p=0.35, n=98) | 0.89 (mean p=0.84, n=113) |
| 12 | 0.47 (mean p=0.54, n=96) | 0.90 (mean p=0.86, n=115) |
| 13 | 0.38 (mean p=0.49, n=81) | 0.86 (mean p=0.85, n=130) |
| 14 | 0.39 (mean p=0.51, n=41) | 0.84 (mean p=0.82, n=170) |
| 16 | 0.34 (mean p=0.48, n=38) | 0.77 (mean p=0.78, n=173) |
| 20 | 0.35 (mean p=0.50, n=37) | 0.74 (mean p=0.75, n=174) |
| 24 | 0.29 (mean p=0.42, n=17) | 0.71 (mean p=0.72, n=194) |

Reading C: if the probe tracks the causal state, it should flag few restored runs and most non-restored runs.