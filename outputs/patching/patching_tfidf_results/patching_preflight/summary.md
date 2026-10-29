# Phase 4 — activation patching, TF-IDF setting (meta-llama/Llama-3.2-3B-Instruct, via meta-llama/Llama-3.2-3B-Instruct)

Pairs: 259 selected (split=test, kinds=hedge,other_wrong_entity), 2 usable (skipped {}); 2 completed. Informative pairs (clean run states the gold hop-1 entity, CF run does not): **2** {'hedge': 2}. Clean and CF prompts differ only in the entity (entity span median 5.5 tokens, tail median 6.5). Success = gold hop-1 entity or a Wikidata alias, whole words.

## Checks
- hook self-test (all positions, last block): max |logit diff| 0.00e+00 (unpatched 7.55e+00) — must be ~0 vs large
- sanity (patch ALL prompt positions at block 0): 1.000 restored, generated text identical to the clean run in 1.000 of pairs — both should be ~1
- baselines: clean run states the gold hop-1 entity 1.000; CF run 0.000; CF run hedges ('not mentioned') 1.000
- probe-link self-check (id, cosine recomputed vs stored state, stored P, recomputed P): [('ec7da1b20bda11eba7f7acde48001122', 1.0, 0.2073, 0.2079), ('ec7da1b20bda11eba7f7acde48001122_cf', 1.0, 0.968, 0.9675), ('089c29860bdb11eba7f7acde48001122', 1.0, 0.0543, 0.0543), ('089c29860bdb11eba7f7acde48001122_cf', 1.0, 0.1918, 0.1918)]

## Experiment A — hop-1 bridge entity restored, by patched block

| block | entity span | tail (tokens after entity) | last prompt token | donor control (third entity) |
|---|---|---|---|---|
| 0 | 1.00 [0.34, 1.00] (n=2) | 0.00 [0.00, 0.66] (n=2) | 0.00 [0.00, 0.66] (n=2) | n/a |
| 10 | 0.50 [0.09, 0.91] (n=2) | 0.00 [0.00, 0.66] (n=2) | 0.00 [0.00, 0.66] (n=2) | n/a |

### Entity span vs donor control (paired, same pairs)

| block | n | entity − donor | entity only | donor only | sign test p |
|---|---|---|---|---|---|
| 0 | 0 | +nan | 0 | 0 | 1 |
| 10 | 0 | +nan | 0 | 0 | 1 |

### Entity span restoration by CF failure kind

| block | hedge | other_wrong_entity |
|---|---|---|
| 0 | 1.00 [0.34, 1.00] (n=2) | n/a |
| 10 | 0.50 [0.09, 0.91] (n=2) | n/a |

## Experiment B — single-state transplant at `</hop1>`, final answer restored

| block | baseline correct | patched correct | restored (patched ok & baseline wrong) |
|---|---|---|---|
| 10 | 0/2 | 0/2 | 0.00 [0.00, 0.66] |

## Experiment C — deployed probe (calibrated, flagged = P ≥ τ = 0.639) on the generated hop 1

- clean baseline: 0.00 (mean p=0.13, n=2)
- CF baseline: 0.50 (mean p=0.58, n=2)
- verdict matches 'entity NOT restored' in 3/4 patched runs (0.750)

| block patched (entity span) | runs where the entity WAS restored | runs where it was NOT restored |
|---|---|---|
| 0 | 0.00 (mean p=0.12, n=2) | n/a |
| 10 | 1.00 (mean p=0.86, n=1) | 1.00 (mean p=0.99, n=1) |

Reading C: if the probe tracks the causal state, it should flag few restored runs and most non-restored runs.