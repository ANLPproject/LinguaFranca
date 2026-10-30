# Phase 4 — activation patching, TF-IDF setting (meta-llama/Llama-3.2-3B-Instruct, via meta-llama/Llama-3.2-3B-Instruct)

Pairs: 259 selected (split=test, kinds=hedge,other_wrong_entity), 259 usable (skipped {}); 219 completed. Informative pairs (clean run states the gold hop-1 entity, CF run does not): **211** {'hedge': 188, 'other_wrong_entity': 23}. Clean and CF prompts differ only in the entity (entity span median 5 tokens, tail median 6). Success = gold hop-1 entity or a Wikidata alias, whole words.

## Checks
- hook self-test (all positions, last block): max |logit diff| 0.00e+00 (unpatched 7.55e+00) — must be ~0 vs large
- sanity (patch ALL prompt positions at block 0): 1.000 restored, generated text identical to the clean run in 0.986 of pairs — both should be ~1
- baselines: clean run states the gold hop-1 entity 0.973; CF run 0.009; CF run hedges ('not mentioned') 0.886
- probe-link self-check (id, cosine recomputed vs stored state, stored P, recomputed P): [['ec7da1b20bda11eba7f7acde48001122', 1.0, 0.2073, 0.2079], ['ec7da1b20bda11eba7f7acde48001122_cf', 1.0, 0.968, 0.9675], ['089c29860bdb11eba7f7acde48001122', 1.0, 0.0543, 0.0543], ['089c29860bdb11eba7f7acde48001122_cf', 1.0, 0.1918, 0.1918], ['5ba8a10a0bda11eba7f7acde48001122', 1.0, 0.0625, 0.0625], ['5ba8a10a0bda11eba7f7acde48001122_cf', 1.0, 0.5764, 0.5764]]

## Experiment A — hop-1 bridge entity restored, by patched block

| block | entity span | tail (tokens after entity) | last prompt token | donor control (third entity) |
|---|---|---|---|---|
| 0 | 1.00 [0.98, 1.00] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.10 [0.06, 0.14] (n=210) |
| 2 | 0.97 [0.94, 0.99] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.00 [0.00, 0.03] (n=211) | 0.09 [0.06, 0.14] (n=210) |
| 4 | 0.93 [0.89, 0.96] (n=211) | 0.06 [0.04, 0.10] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.12 [0.08, 0.17] (n=210) |
| 6 | 0.93 [0.89, 0.96] (n=211) | 0.15 [0.11, 0.21] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.13 [0.09, 0.18] (n=210) |
| 8 | 0.90 [0.85, 0.93] (n=211) | 0.35 [0.29, 0.42] (n=211) | 0.00 [0.00, 0.02] (n=211) | 0.15 [0.11, 0.21] (n=210) |
| 9 | 0.78 [0.72, 0.83] (n=211) | 0.39 [0.33, 0.46] (n=211) | 0.00 [0.00, 0.03] (n=211) | 0.14 [0.10, 0.19] (n=210) |
| 10 | 0.73 [0.67, 0.79] (n=211) | 0.40 [0.33, 0.47] (n=211) | 0.00 [0.00, 0.03] (n=211) | 0.14 [0.10, 0.20] (n=210) |
| 11 | 0.46 [0.40, 0.53] (n=211) | 0.46 [0.39, 0.53] (n=211) | 0.01 [0.00, 0.04] (n=211) | 0.16 [0.11, 0.21] (n=210) |
| 12 | 0.45 [0.39, 0.52] (n=211) | 0.50 [0.44, 0.57] (n=211) | 0.01 [0.00, 0.04] (n=211) | 0.14 [0.10, 0.20] (n=210) |
| 13 | 0.38 [0.32, 0.45] (n=211) | 0.48 [0.41, 0.55] (n=211) | 0.03 [0.01, 0.06] (n=211) | 0.14 [0.10, 0.20] (n=210) |
| 14 | 0.19 [0.15, 0.25] (n=211) | 0.15 [0.11, 0.20] (n=211) | 0.07 [0.04, 0.11] (n=211) | 0.12 [0.09, 0.18] (n=210) |
| 16 | 0.18 [0.13, 0.24] (n=211) | 0.07 [0.04, 0.11] (n=211) | 0.06 [0.04, 0.10] (n=211) | 0.12 [0.08, 0.17] (n=210) |
| 20 | 0.18 [0.13, 0.23] (n=211) | 0.07 [0.04, 0.11] (n=211) | 0.05 [0.03, 0.09] (n=211) | 0.11 [0.08, 0.16] (n=210) |
| 24 | 0.08 [0.05, 0.13] (n=211) | 0.04 [0.02, 0.08] (n=211) | 0.02 [0.01, 0.05] (n=211) | 0.06 [0.03, 0.10] (n=210) |

### Entity span vs donor control (paired, same pairs)

| block | n | entity − donor | entity only | donor only | sign test p |
|---|---|---|---|---|---|
| 0 | 210 | +0.90 | 190 | 0 | 1.3e-57 |
| 2 | 210 | +0.88 | 185 | 0 | 4.1e-56 |
| 4 | 210 | +0.81 | 170 | 0 | 1.3e-51 |
| 6 | 210 | +0.80 | 168 | 0 | 5.3e-51 |
| 8 | 210 | +0.74 | 156 | 0 | 2.2e-47 |
| 9 | 210 | +0.64 | 135 | 0 | 4.6e-41 |
| 10 | 210 | +0.59 | 124 | 0 | 9.4e-38 |
| 11 | 210 | +0.30 | 68 | 4 | 4.6e-16 |
| 12 | 210 | +0.31 | 68 | 3 | 5.1e-17 |
| 13 | 210 | +0.24 | 53 | 3 | 8.1e-13 |
| 14 | 210 | +0.07 | 21 | 6 | 0.0059 |
| 16 | 210 | +0.06 | 22 | 9 | 0.029 |
| 20 | 210 | +0.06 | 19 | 6 | 0.015 |
| 24 | 210 | +0.02 | 5 | 0 | 0.062 |

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

## Experiment B — single-state transplant at `</hop1>`, final answer restored

| block | baseline correct | patched correct | restored (patched ok & baseline wrong) |
|---|---|---|---|
| 8 | 0/211 | 0/211 | 0.00 [0.00, 0.02] |
| 10 | 0/211 | 0/211 | 0.00 [0.00, 0.02] |
| 11 | 0/211 | 0/211 | 0.00 [0.00, 0.02] |
| 14 | 0/211 | 2/211 | 0.01 [0.00, 0.03] |
| 20 | 0/211 | 2/211 | 0.01 [0.00, 0.03] |

## Experiment C — deployed probe (calibrated, flagged = P ≥ τ = 0.639) on the generated hop 1

- clean baseline: 0.02 (mean p=0.10, n=211)
- CF baseline: 0.48 (mean p=0.60, n=211)
- verdict matches 'entity NOT restored' in 2369/2954 patched runs (0.802)

| block patched (entity span) | runs where the entity WAS restored | runs where it was NOT restored |
|---|---|---|
| 0 | 0.02 (mean p=0.10, n=211) | n/a |
| 2 | 0.02 (mean p=0.12, n=205) | 0.83 (mean p=0.82, n=6) |
| 4 | 0.03 (mean p=0.13, n=196) | 0.80 (mean p=0.80, n=15) |
| 6 | 0.07 (mean p=0.20, n=196) | 0.73 (mean p=0.75, n=15) |
| 8 | 0.12 (mean p=0.30, n=189) | 0.82 (mean p=0.80, n=22) |
| 9 | 0.13 (mean p=0.31, n=165) | 0.80 (mean p=0.82, n=46) |
| 10 | 0.37 (mean p=0.50, n=155) | 0.93 (mean p=0.90, n=56) |
| 11 | 0.42 (mean p=0.53, n=98) | 0.86 (mean p=0.84, n=113) |
| 12 | 0.41 (mean p=0.51, n=96) | 0.83 (mean p=0.83, n=115) |
| 13 | 0.36 (mean p=0.49, n=81) | 0.85 (mean p=0.84, n=130) |
| 14 | 0.39 (mean p=0.53, n=41) | 0.79 (mean p=0.80, n=170) |
| 16 | 0.34 (mean p=0.48, n=38) | 0.77 (mean p=0.79, n=173) |
| 20 | 0.38 (mean p=0.50, n=37) | 0.73 (mean p=0.77, n=174) |
| 24 | 0.18 (mean p=0.35, n=17) | 0.49 (mean p=0.59, n=194) |

Reading C: if the probe tracks the causal state, it should flag few restored runs and most non-restored runs.