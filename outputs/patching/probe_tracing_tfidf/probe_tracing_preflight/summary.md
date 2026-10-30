# Probe <-> causal tracing, TF-IDF pairs

Pairs traced: 2; with a clean-vs-CF hop-1 log-prob gap >= 1.0 nat: 2. Hop 1 = the clean run's own text, teacher-forced in every run (same words); only the entity differs.

## Behavioural recovery (fraction of the log-prob gap), mean by block

| block | entity_last | entity_span | control |
|---|---|---|---|
| 0 | 0.19 | 1.00 | nan |
| 1 | 0.54 | 1.00 | nan |
| 2 | 0.56 | 1.00 | nan |
| 3 | 0.56 | 0.98 | nan |
| 4 | 0.65 | 1.06 | nan |
| 5 | 0.67 | 1.05 | nan |
| 6 | 0.72 | 0.99 | nan |
| 7 | 0.71 | 1.03 | nan |
| 8 | 0.68 | 1.01 | nan |
| 9 | 0.54 | 0.79 | nan |
| 10 | 0.48 | 0.72 | nan |
| 11 | 0.45 | 0.57 | nan |
| 12 | 0.44 | 0.48 | nan |
| 13 | 0.44 | 0.42 | nan |
| 14 | 0.02 | -0.02 | nan |
| 15 | 0.03 | -0.05 | nan |
| 16 | 0.04 | -0.04 | nan |
| 17 | 0.01 | -0.11 | nan |
| 18 | -0.01 | -0.10 | nan |
| 19 | -0.01 | -0.15 | nan |
| 20 | -0.01 | -0.15 | nan |
| 21 | -0.01 | -0.12 | nan |
| 22 | -0.00 | -0.11 | nan |
| 23 | -0.01 | -0.13 | nan |
| 24 | -0.01 | -0.07 | nan |
| 25 | -0.00 | -0.07 | nan |
| 26 | -0.01 | -0.03 | nan |
| 27 | -0.00 | -0.00 | nan |

## old (C=1, block 10)

**Test A — same hop-1 words, only the question entity differs:**
- P(fail) clean 0.131 -> CF 0.655; flagged at τ=0.639: clean 0.000, CF 0.500
- log-odds rise +2.90 [95% CI +1.14, +4.66]; rises in 100% of pairs; AUROC clean vs CF 1.000

**Test B — patch at block l < 10 (only those can reach the probe's block):**

| block | recovery (entity last / span / control) | probe log-odds drop (entity last / span / control) |
|---|---|---|
| 0 | 0.19 / 1.00 / nan | +0.61 / +2.98 / +nan |
| 1 | 0.54 / 1.00 / nan | +1.18 / +3.09 / +nan |
| 2 | 0.56 / 1.00 / nan | +1.17 / +3.34 / +nan |
| 3 | 0.56 / 0.98 / nan | +1.19 / +3.36 / +nan |
| 4 | 0.65 / 1.06 / nan | +1.23 / +3.51 / +nan |
| 5 | 0.67 / 1.05 / nan | +1.20 / +3.41 / +nan |
| 6 | 0.72 / 0.99 / nan | +1.37 / +3.09 / +nan |
| 7 | 0.71 / 1.03 / nan | +0.96 / +2.37 / +nan |
| 8 | 0.68 / 1.01 / nan | +0.69 / +1.70 / +nan |
| 9 | 0.54 / 0.79 / nan | +0.60 / +1.37 / +nan |
| 10 | 0.48 / 0.72 / nan | -0.00 / -0.00 / +nan |
| 11 | 0.45 / 0.57 / nan | -0.00 / -0.00 / +nan |
- Spearman(recovery, probe drop), entity_last, blocks < 10: rho = 0.88 (p = 3.2e-07, n = 20)
- Spearman(recovery, probe drop), entity_span, blocks < 10: rho = 0.12 (p = 6.0e-01, n = 20)

## tuned (C=0.01, block 12)

**Test A — same hop-1 words, only the question entity differs:**
- P(fail) clean 0.189 -> CF 0.945; flagged at τ=0.585: clean 0.000, CF 1.000
- log-odds rise +4.75 [95% CI +3.76, +5.75]; rises in 100% of pairs; AUROC clean vs CF 1.000

**Test B — patch at block l < 12 (only those can reach the probe's block):**

| block | recovery (entity last / span / control) | probe log-odds drop (entity last / span / control) |
|---|---|---|
| 0 | 0.19 / 1.00 / nan | +1.06 / +4.74 / +nan |
| 1 | 0.54 / 1.00 / nan | +1.37 / +4.70 / +nan |
| 2 | 0.56 / 1.00 / nan | +1.37 / +4.70 / +nan |
| 3 | 0.56 / 0.98 / nan | +1.37 / +4.75 / +nan |
| 4 | 0.65 / 1.06 / nan | +1.26 / +4.76 / +nan |
| 5 | 0.67 / 1.05 / nan | +1.18 / +4.49 / +nan |
| 6 | 0.72 / 0.99 / nan | +1.21 / +4.37 / +nan |
| 7 | 0.71 / 1.03 / nan | +1.00 / +4.07 / +nan |
| 8 | 0.68 / 1.01 / nan | +1.13 / +3.66 / +nan |
| 9 | 0.54 / 0.79 / nan | +1.14 / +2.87 / +nan |
| 10 | 0.48 / 0.72 / nan | +0.43 / +1.49 / +nan |
| 11 | 0.45 / 0.57 / nan | +0.57 / +0.99 / +nan |
| 12 | 0.44 / 0.48 / nan | -0.00 / -0.00 / +nan |
| 13 | 0.44 / 0.42 / nan | -0.00 / -0.00 / +nan |
- Spearman(recovery, probe drop), entity_last, blocks < 12: rho = 0.90 (p = 2.6e-09, n = 24)
- Spearman(recovery, probe drop), entity_span, blocks < 12: rho = 0.29 (p = 1.6e-01, n = 24)
