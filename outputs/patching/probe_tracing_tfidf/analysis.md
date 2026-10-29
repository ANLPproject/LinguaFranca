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

