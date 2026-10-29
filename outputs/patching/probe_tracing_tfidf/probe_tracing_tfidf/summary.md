# Probe <-> causal tracing, TF-IDF pairs

Pairs traced: 219; with a clean-vs-CF hop-1 log-prob gap >= 1.0 nat: 192. Hop 1 = the clean run's own text, teacher-forced in every run (same words); only the entity differs.

## Behavioural recovery (fraction of the log-prob gap), mean by block

| block | entity_last | entity_span | control |
|---|---|---|---|
| 0 | 0.33 | 1.00 | -0.04 |
| 1 | 0.39 | 1.00 | -0.05 |
| 2 | 0.40 | 1.00 | -0.05 |
| 3 | 0.40 | 1.00 | -0.04 |
| 4 | 0.42 | 1.00 | -0.03 |
| 5 | 0.44 | 1.00 | -0.03 |
| 6 | 0.52 | 1.00 | -0.00 |
| 7 | 0.56 | 0.99 | -0.01 |
| 8 | 0.50 | 0.96 | -0.00 |
| 9 | 0.44 | 0.83 | 0.03 |
| 10 | 0.40 | 0.78 | 0.03 |
| 11 | 0.30 | 0.51 | 0.07 |
| 12 | 0.30 | 0.50 | 0.08 |
| 13 | 0.26 | 0.41 | 0.05 |
| 14 | 0.04 | 0.07 | 0.01 |
| 15 | 0.02 | 0.03 | 0.00 |
| 16 | 0.01 | 0.02 | 0.01 |
| 17 | 0.00 | -0.01 | 0.01 |
| 18 | 0.00 | -0.02 | 0.01 |
| 19 | 0.00 | -0.04 | 0.01 |
| 20 | 0.00 | -0.04 | 0.01 |
| 21 | 0.00 | -0.01 | 0.02 |
| 22 | 0.00 | -0.01 | 0.02 |
| 23 | -0.00 | -0.02 | 0.02 |
| 24 | 0.00 | -0.01 | 0.01 |
| 25 | -0.00 | -0.01 | 0.01 |
| 26 | -0.00 | -0.00 | 0.00 |
| 27 | -0.00 | -0.00 | -0.00 |

## old (C=1, block 10)

**Test A — same hop-1 words, only the question entity differs:**
- P(fail) clean 0.098 -> CF 0.523; flagged at τ=0.639: clean 0.021, CF 0.380
- log-odds rise +2.94 [95% CI +2.74, +3.14]; rises in 98% of pairs; AUROC clean vs CF 0.911

**Test B — patch at block l < 10 (only those can reach the probe's block):**

| block | recovery (entity last / span / control) | probe log-odds drop (entity last / span / control) |
|---|---|---|
| 0 | 0.33 / 1.00 / -0.04 | +0.62 / +2.92 / -0.01 |
| 1 | 0.39 / 1.00 / -0.05 | +0.70 / +2.82 / -0.01 |
| 2 | 0.40 / 1.00 / -0.05 | +0.66 / +2.65 / -0.01 |
| 3 | 0.40 / 1.00 / -0.04 | +0.70 / +2.66 / -0.01 |
| 4 | 0.42 / 1.00 / -0.03 | +0.72 / +2.51 / -0.05 |
| 5 | 0.44 / 1.00 / -0.03 | +0.74 / +2.27 / -0.05 |
| 6 | 0.52 / 1.00 / -0.00 | +0.83 / +1.97 / -0.06 |
| 7 | 0.56 / 0.99 / -0.01 | +0.71 / +1.59 / -0.06 |
| 8 | 0.50 / 0.96 / -0.00 | +0.60 / +1.18 / -0.03 |
| 9 | 0.44 / 0.83 / 0.03 | +0.50 / +1.02 / -0.03 |
| 10 | 0.40 / 0.78 / 0.03 | +0.00 / +0.00 / +0.00 |
| 11 | 0.30 / 0.51 / 0.07 | +0.00 / +0.00 / +0.00 |
- Spearman(recovery, probe drop), entity_last, blocks < 10: rho = 0.50 (p = 1.4e-122, n = 1920)
- Spearman(recovery, probe drop), entity_span, blocks < 10: rho = 0.16 (p = 9.1e-13, n = 1920)
- Spearman(recovery, probe drop), control, blocks < 10: rho = 0.04 (p = 7.0e-02, n = 1910)

## tuned (C=0.01, block 12)

**Test A — same hop-1 words, only the question entity differs:**
- P(fail) clean 0.061 -> CF 0.530; flagged at τ=0.585: clean 0.010, CF 0.469
- log-odds rise +3.65 [95% CI +3.47, +3.83]; rises in 99% of pairs; AUROC clean vs CF 0.960

**Test B — patch at block l < 12 (only those can reach the probe's block):**

| block | recovery (entity last / span / control) | probe log-odds drop (entity last / span / control) |
|---|---|---|
| 0 | 0.33 / 1.00 / -0.04 | +0.90 / +3.64 / +0.09 |
| 1 | 0.39 / 1.00 / -0.05 | +1.01 / +3.61 / +0.08 |
| 2 | 0.40 / 1.00 / -0.05 | +0.99 / +3.48 / +0.08 |
| 3 | 0.40 / 1.00 / -0.04 | +1.03 / +3.50 / +0.07 |
| 4 | 0.42 / 1.00 / -0.03 | +1.10 / +3.49 / +0.10 |
| 5 | 0.44 / 1.00 / -0.03 | +1.17 / +3.33 / +0.10 |
| 6 | 0.52 / 1.00 / -0.00 | +1.40 / +3.27 / +0.12 |
| 7 | 0.56 / 0.99 / -0.01 | +1.56 / +3.20 / +0.14 |
| 8 | 0.50 / 0.96 / -0.00 | +1.38 / +2.85 / +0.15 |
| 9 | 0.44 / 0.83 / 0.03 | +1.06 / +2.24 / +0.12 |
| 10 | 0.40 / 0.78 / 0.03 | +0.60 / +1.39 / +0.08 |
| 11 | 0.30 / 0.51 / 0.07 | +0.41 / +0.84 / +0.06 |
| 12 | 0.30 / 0.50 / 0.08 | +0.00 / +0.00 / +0.00 |
| 13 | 0.26 / 0.41 / 0.05 | +0.00 / +0.00 / +0.00 |
- Spearman(recovery, probe drop), entity_last, blocks < 12: rho = 0.67 (p = 2.5e-294, n = 2304)
- Spearman(recovery, probe drop), entity_span, blocks < 12: rho = 0.37 (p = 1.4e-75, n = 2304)
- Spearman(recovery, probe drop), control, blocks < 12: rho = 0.37 (p = 2.6e-73, n = 2292)
