# Phase 4 — activation patching (meta-llama/Llama-3.2-3B-Instruct, via unsloth/Llama-3.2-3B-Instruct)

Pairs: 78 total, 51 token-aligned (skipped {'prompt_lengths_differ': 27}); 51 completed. Informative pairs (clean run states the gold hop-1 entity, CF run does not): **51**. Prompts are date-harmonized: clean and CF differ only in the entity (entity span median 5 tokens, tail median 7).

## Checks
- hook self-test (all positions, last block): max |logit diff| 0.00e+00 (unpatched 7.91e+00) — must be ~0 vs large
- sanity (patch ALL prompt positions at block 0): 1.000 restored, generated text identical to the clean run in 1.000 of pairs — both should be ~1
- baselines: clean run states the gold hop-1 entity 1.000; CF run 0.000; CF run hedges ('not mentioned') 0.863
- probe-link self-check (recomputed vs stored Phase-1 states, cosine): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

## Experiment A — hop-1 bridge entity restored, by patched block

| block | entity span | tail (tokens after entity) | last prompt token | donor-entity control |
|---|---|---|---|---|
| 0 | 1.00 [0.93, 1.00] (n=51) | 0.02 [0.00, 0.10] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.18 [0.10, 0.30] (n=51) |
| 2 | 1.00 [0.93, 1.00] (n=51) | 0.06 [0.02, 0.16] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.45 [0.32, 0.59] (n=51) |
| 4 | 1.00 [0.93, 1.00] (n=51) | 0.16 [0.08, 0.28] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.61 [0.47, 0.73] (n=51) |
| 6 | 0.98 [0.90, 1.00] (n=51) | 0.24 [0.14, 0.37] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.55 [0.41, 0.68] (n=51) |
| 8 | 1.00 [0.93, 1.00] (n=51) | 0.45 [0.32, 0.59] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.43 [0.31, 0.57] (n=51) |
| 9 | 0.90 [0.79, 0.96] (n=51) | 0.63 [0.49, 0.75] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.41 [0.29, 0.55] (n=51) |
| 10 | 0.90 [0.79, 0.96] (n=51) | 0.63 [0.49, 0.75] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.43 [0.31, 0.57] (n=51) |
| 11 | 0.76 [0.63, 0.86] (n=51) | 0.73 [0.59, 0.83] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.39 [0.27, 0.53] (n=51) |
| 12 | 0.67 [0.53, 0.78] (n=51) | 0.71 [0.57, 0.81] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.39 [0.27, 0.53] (n=51) |
| 13 | 0.69 [0.55, 0.80] (n=51) | 0.65 [0.51, 0.76] (n=51) | 0.04 [0.01, 0.13] (n=51) | 0.37 [0.25, 0.51] (n=51) |
| 14 | 0.39 [0.27, 0.53] (n=51) | 0.12 [0.06, 0.23] (n=51) | 0.04 [0.01, 0.13] (n=51) | 0.29 [0.19, 0.43] (n=51) |
| 16 | 0.39 [0.27, 0.53] (n=51) | 0.08 [0.03, 0.19] (n=51) | 0.04 [0.01, 0.13] (n=51) | 0.29 [0.19, 0.43] (n=51) |
| 20 | 0.35 [0.24, 0.49] (n=51) | 0.06 [0.02, 0.16] (n=51) | 0.04 [0.01, 0.13] (n=51) | 0.27 [0.17, 0.41] (n=51) |
| 24 | 0.08 [0.03, 0.19] (n=51) | 0.04 [0.01, 0.13] (n=51) | 0.00 [0.00, 0.07] (n=51) | 0.06 [0.02, 0.16] (n=51) |

## Experiment B — single-state transplant at `</hop1>`, final answer restored

| block | baseline correct | patched correct | restored (patched ok & baseline wrong) |
|---|---|---|---|
| 8 | 1/51 | 0/51 | 0.00 [0.00, 0.07] |
| 11 | 1/51 | 1/51 | 0.00 [0.00, 0.07] |
| 14 | 1/51 | 1/51 | 0.00 [0.00, 0.07] |
| 20 | 1/51 | 1/51 | 0.00 [0.00, 0.07] |

## Experiment C — hop-1 probe (block 10) on the generated hop 1: fraction flagged as failure

- clean baseline: 0.00 (mean p=0.00, n=51)
- CF baseline: 1.00 (mean p=1.00, n=51)

| block patched (entity span) | runs where the entity WAS restored | runs where it was NOT restored |
|---|---|---|
| 0 | 0.00 (mean p=0.00, n=51) | n/a |
| 2 | 0.00 (mean p=0.01, n=51) | n/a |
| 4 | 0.00 (mean p=0.01, n=51) | n/a |
| 6 | 0.00 (mean p=0.01, n=50) | 1.00 (mean p=1.00, n=1) |
| 8 | 0.00 (mean p=0.01, n=51) | n/a |
| 9 | 0.00 (mean p=0.00, n=46) | 1.00 (mean p=1.00, n=5) |
| 10 | 0.00 (mean p=0.01, n=46) | 1.00 (mean p=0.94, n=5) |
| 11 | 0.03 (mean p=0.02, n=39) | 0.83 (mean p=0.81, n=12) |
| 12 | 0.00 (mean p=0.00, n=34) | 0.88 (mean p=0.86, n=17) |
| 13 | 0.00 (mean p=0.00, n=35) | 0.94 (mean p=0.90, n=16) |
| 14 | 0.05 (mean p=0.04, n=20) | 0.90 (mean p=0.91, n=31) |
| 16 | 0.15 (mean p=0.12, n=20) | 0.77 (mean p=0.76, n=31) |
| 20 | 0.06 (mean p=0.04, n=18) | 0.94 (mean p=0.92, n=33) |
| 24 | 0.00 (mean p=0.00, n=4) | 1.00 (mean p=1.00, n=47) |

Reading C: if the probe tracks the causal state, it should flag few restored runs and most non-restored runs.