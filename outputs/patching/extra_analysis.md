# Patching (TF-IDF): extra analyses

## 1. Reproducibility (full run vs C-only re-run, same pairs)

- pairs in both runs: 219; entity-span outcomes that differ: 0 of 3066; baseline labels that differ: 0; CF baseline texts that differ: 0

## 2. Experiment C: old vs tuned probe (same 211 informative pairs)

| | old: C=1, block 10 | tuned: C=0.01, block 12 |
|---|---|---|
| τ (calibrated) | 0.639 | 0.585 |
| clean baseline flagged | 0.024 | 0.005 |
| CF baseline flagged | 0.483 | 0.749 |
|   CF hedge flagged | 0.441 (n=188) | 0.734 (n=188) |
|   CF other wrong entity flagged | 0.826 (n=23) | 0.870 (n=23) |
| AUROC clean vs CF baseline | 0.935 | 0.990 |
| agreement, patches below the probe's layer | 1174/1266 (0.927) | 1611/1688 (0.954) |
| agreement, patches at/above the probe's layer | 1195/1688 (0.708) | 944/1266 (0.746) |
| agreement, patches at blocks < 10 (same range) | 1174/1266 (0.927) | 1241/1266 (0.980) |

## 3. Entity span OR tail restores (full run)

| block | entity | tail | entity OR tail | entity AND tail |
|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 1.00 | 0.00 |
| 2 | 0.97 | 0.00 | 0.97 | 0.00 |
| 4 | 0.93 | 0.06 | 0.93 | 0.06 |
| 6 | 0.93 | 0.15 | 0.94 | 0.14 |
| 8 | 0.90 | 0.35 | 0.92 | 0.33 |
| 9 | 0.78 | 0.39 | 0.88 | 0.29 |
| 10 | 0.73 | 0.40 | 0.83 | 0.30 |
| 11 | 0.46 | 0.46 | 0.75 | 0.18 |
| 12 | 0.45 | 0.50 | 0.77 | 0.19 |
| 13 | 0.38 | 0.48 | 0.71 | 0.15 |
| 14 | 0.19 | 0.15 | 0.30 | 0.04 |
| 16 | 0.18 | 0.07 | 0.23 | 0.02 |
| 20 | 0.18 | 0.07 | 0.22 | 0.02 |
| 24 | 0.08 | 0.04 | 0.11 | 0.01 |

## 4. Donor control with type-matched third entities (same question template)

Pairs whose third entity comes from the same template: 14 of 211 (templates unknown for 0).

| block | n | entity | donor | entity − donor | sign test p | all pairs: donor |
|---|---|---|---|---|---|---|
| 0 | 14 | 1.00 | 0.14 | +0.86 | 0.00049 | 0.10 |
| 2 | 14 | 0.93 | 0.14 | +0.79 | 0.00098 | 0.09 |
| 4 | 14 | 0.93 | 0.21 | +0.71 | 0.002 | 0.12 |
| 6 | 14 | 0.93 | 0.14 | +0.79 | 0.00098 | 0.13 |
| 8 | 14 | 0.86 | 0.21 | +0.64 | 0.0039 | 0.15 |
| 9 | 14 | 0.71 | 0.21 | +0.50 | 0.016 | 0.14 |
| 10 | 14 | 0.71 | 0.21 | +0.50 | 0.016 | 0.14 |
| 11 | 14 | 0.43 | 0.00 | +0.43 | 0.031 | 0.16 |
| 12 | 14 | 0.43 | 0.00 | +0.43 | 0.031 | 0.14 |
| 13 | 14 | 0.36 | 0.00 | +0.36 | 0.062 | 0.14 |
| 14 | 14 | 0.07 | 0.00 | +0.07 | 1 | 0.12 |
| 16 | 14 | 0.00 | 0.00 | +0.00 | 1 | 0.12 |
| 20 | 14 | 0.07 | 0.00 | +0.07 | 1 | 0.11 |
| 24 | 14 | 0.00 | 0.00 | +0.00 | 1 | 0.06 |
