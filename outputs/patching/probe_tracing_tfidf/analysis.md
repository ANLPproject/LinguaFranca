# Probe <-> causal tracing: corrected statistics

Pairs with a hop-1 log-prob gap >= 1.0 nat: 192 of 219. Probe scores are calibrated log-odds; CIs are 95% bootstraps over pairs.

## old (C=1, block 10)

### Test A by failure kind (same hop-1 words)

| CF kind | n | flagged clean | flagged CF | AUROC clean vs CF | mean log-odds rise |
|---|---|---|---|---|---|
| all | 192 | 0.021 | 0.380 | 0.911 | +2.94 |
| hedge | 175 | 0.017 | 0.366 | 0.908 | +2.92 |
| other_wrong_entity | 17 | 0.059 | 0.529 | 0.969 | +3.16 |

### Test B (patches at blocks < 10), correlation of behavioural recovery with probe drop

| patch | pairs | within-pair, within-block r [95% CI, pair bootstrap] | pooled Spearman [95% CI] |
|---|---|---|---|
| entity_last | 192 | +0.28 [+0.19, +0.36] | +0.50 [+0.42, +0.58] |
| entity_span | 192 | -0.07 [-0.15, +0.01] | +0.16 [+0.09, +0.25] |
| control | 191 | -0.00 [-0.11, +0.10] | +0.04 [-0.07, +0.17] |

Spearman across pairs at each block (SD of behavioural recovery in brackets; near-zero SD = ceiling, rho uninformative):

| block | entity_last | entity_span | control |
|---|---|---|---|
| 0 | +0.56 (0.36) | -0.06 (0.02) | +0.04 (0.72) |
| 1 | +0.62 (0.37) | -0.10 (0.03) | +0.02 (0.72) |
| 2 | +0.65 (0.38) | -0.01 (0.04) | -0.01 (0.72) |
| 3 | +0.66 (0.39) | +0.05 (0.05) | +0.01 (0.69) |
| 4 | +0.58 (0.41) | +0.06 (0.07) | +0.09 (0.68) |
| 5 | +0.60 (0.41) | +0.10 (0.08) | +0.08 (0.68) |
| 6 | +0.27 (0.37) | +0.04 (0.10) | +0.05 (0.63) |
| 7 | +0.24 (0.37) | -0.00 (0.14) | +0.09 (0.62) |
| 8 | +0.17 (0.33) | +0.14 (0.17) | +0.05 (0.58) |
| 9 | +0.03 (0.29) | +0.13 (0.24) | +0.09 (0.52) |

### Specificity: probe drop (entity span) − probe drop (third-entity control), mean over blocks < 10

- +2.19 log-odds [+2.05, +2.32]; larger for the entity patch in 99% of 191 pairs

### Lag: probe drop as a fraction of the Test A rise vs behavioural recovery (entity span)

| block | behavioural recovery | probe restoration |
|---|---|---|
| 0 | 1.00 | 0.99 |
| 1 | 1.00 | 0.96 |
| 2 | 1.00 | 0.90 |
| 3 | 1.00 | 0.90 |
| 4 | 1.00 | 0.85 |
| 5 | 1.00 | 0.77 |
| 6 | 1.00 | 0.67 |
| 7 | 0.99 | 0.54 |
| 8 | 0.96 | 0.40 |
| 9 | 0.83 | 0.35 |

## tuned (C=0.01, block 12)

### Test A by failure kind (same hop-1 words)

| CF kind | n | flagged clean | flagged CF | AUROC clean vs CF | mean log-odds rise |
|---|---|---|---|---|---|
| all | 192 | 0.010 | 0.469 | 0.960 | +3.65 |
| hedge | 175 | 0.011 | 0.463 | 0.956 | +3.64 |
| other_wrong_entity | 17 | 0.000 | 0.529 | 1.000 | +3.74 |

### Test B (patches at blocks < 12), correlation of behavioural recovery with probe drop

| patch | pairs | within-pair, within-block r [95% CI, pair bootstrap] | pooled Spearman [95% CI] |
|---|---|---|---|
| entity_last | 192 | +0.59 [+0.54, +0.64] | +0.67 [+0.62, +0.71] |
| entity_span | 192 | +0.14 [+0.04, +0.24] | +0.37 [+0.30, +0.44] |
| control | 191 | +0.37 [+0.26, +0.48] | +0.37 [+0.25, +0.46] |

Spearman across pairs at each block (SD of behavioural recovery in brackets; near-zero SD = ceiling, rho uninformative):

| block | entity_last | entity_span | control |
|---|---|---|---|
| 0 | +0.68 (0.36) | -0.08 (0.02) | +0.39 (0.72) |
| 1 | +0.74 (0.37) | -0.11 (0.03) | +0.39 (0.72) |
| 2 | +0.76 (0.38) | -0.07 (0.04) | +0.38 (0.72) |
| 3 | +0.78 (0.39) | -0.02 (0.05) | +0.37 (0.69) |
| 4 | +0.74 (0.41) | -0.01 (0.07) | +0.41 (0.68) |
| 5 | +0.74 (0.41) | -0.04 (0.08) | +0.39 (0.68) |
| 6 | +0.64 (0.37) | +0.00 (0.10) | +0.35 (0.63) |
| 7 | +0.58 (0.37) | +0.05 (0.14) | +0.35 (0.62) |
| 8 | +0.41 (0.33) | +0.16 (0.17) | +0.26 (0.58) |
| 9 | +0.27 (0.29) | +0.42 (0.24) | +0.30 (0.52) |
| 10 | +0.28 (0.29) | +0.50 (0.28) | +0.36 (0.52) |
| 11 | +0.40 (0.31) | +0.51 (0.35) | +0.41 (0.42) |

### Specificity: probe drop (entity span) − probe drop (third-entity control), mean over blocks < 12

- +2.80 log-odds [+2.67, +2.94]; larger for the entity patch in 99% of 191 pairs

### Lag: probe drop as a fraction of the Test A rise vs behavioural recovery (entity span)

| block | behavioural recovery | probe restoration |
|---|---|---|
| 0 | 1.00 | 1.00 |
| 1 | 1.00 | 0.99 |
| 2 | 1.00 | 0.95 |
| 3 | 1.00 | 0.96 |
| 4 | 1.00 | 0.96 |
| 5 | 1.00 | 0.91 |
| 6 | 1.00 | 0.90 |
| 7 | 0.99 | 0.88 |
| 8 | 0.96 | 0.78 |
| 9 | 0.83 | 0.62 |
| 10 | 0.78 | 0.38 |
| 11 | 0.51 | 0.23 |

## Part 2a: robustness of Tests A/B to probe retrains (shared hop12 LR, natural train split)

Natural AUROC on test hops 1+2 (seed questions of the pairs excluded). Test B = within-pair, within-block r, entity last token.

| block | C | training sample | natural AUROC | Test A AUROC | Test B r [95% CI] |
|---|---|---|---|---|---|
| 10 | 0.001 | full | 0.936 | 0.897 | +0.46 [+0.38, +0.54] |
| 10 | 0.01 | full | 0.942 | 0.928 | +0.38 [+0.30, +0.46] |
| 10 | 0.1 | full | 0.929 | 0.919 | +0.30 [+0.22, +0.38] |
| 10 | 1 | full | 0.921 | 0.911 | +0.28 [+0.19, +0.36] |
| 10 | 0.01 | bootstrap 0 | 0.936 | 0.855 | +0.32 [+0.24, +0.40] |
| 10 | 1 | bootstrap 0 | 0.922 | 0.847 | +0.20 [+0.12, +0.28] |
| 10 | 0.01 | bootstrap 1 | 0.938 | 0.883 | +0.40 [+0.33, +0.47] |
| 10 | 1 | bootstrap 1 | 0.925 | 0.867 | +0.31 [+0.24, +0.38] |
| 10 | 0.01 | bootstrap 2 | 0.935 | 0.914 | +0.43 [+0.36, +0.50] |
| 10 | 1 | bootstrap 2 | 0.920 | 0.896 | +0.40 [+0.32, +0.46] |
| 10 | 0.01 | bootstrap 3 | 0.938 | 0.904 | +0.38 [+0.29, +0.45] |
| 10 | 1 | bootstrap 3 | 0.922 | 0.897 | +0.31 [+0.23, +0.39] |
| 10 | 0.01 | bootstrap 4 | 0.939 | 0.871 | +0.37 [+0.31, +0.44] |
| 10 | 1 | bootstrap 4 | 0.920 | 0.846 | +0.22 [+0.14, +0.29] |
| 12 | 0.001 | full | 0.937 | 0.949 | +0.61 [+0.56, +0.66] |
| 12 | 0.01 | full | 0.943 | 0.960 | +0.59 [+0.54, +0.64] |
| 12 | 0.1 | full | 0.933 | 0.950 | +0.57 [+0.51, +0.62] |
| 12 | 1 | full | 0.926 | 0.935 | +0.56 [+0.51, +0.61] |
| 12 | 0.01 | bootstrap 0 | 0.932 | 0.958 | +0.59 [+0.54, +0.64] |
| 12 | 1 | bootstrap 0 | 0.916 | 0.947 | +0.57 [+0.51, +0.62] |
| 12 | 0.01 | bootstrap 1 | 0.936 | 0.913 | +0.56 [+0.50, +0.61] |
| 12 | 1 | bootstrap 1 | 0.923 | 0.907 | +0.51 [+0.45, +0.57] |
| 12 | 0.01 | bootstrap 2 | 0.937 | 0.940 | +0.57 [+0.52, +0.63] |
| 12 | 1 | bootstrap 2 | 0.923 | 0.910 | +0.53 [+0.47, +0.58] |
| 12 | 0.01 | bootstrap 3 | 0.931 | 0.911 | +0.55 [+0.49, +0.61] |
| 12 | 1 | bootstrap 3 | 0.918 | 0.880 | +0.53 [+0.47, +0.59] |
| 12 | 0.01 | bootstrap 4 | 0.940 | 0.942 | +0.62 [+0.56, +0.67] |
| 12 | 1 | bootstrap 4 | 0.927 | 0.930 | +0.60 [+0.54, +0.65] |

## Part 2b: grounding direction d = mean(F_cf − F_clean) (2-fold over pairs)

| block | held-out pairs: AUROC clean vs CF | natural test: AUROC failure vs success | cos(d / scale, probe weights) |
|---|---|---|---|
| 10 | 0.997 | 0.573 | old (C=1, block 10): +0.04 |
| 12 | 1.000 | 0.717 | tuned (C=0.01, block 12): +0.07 |
