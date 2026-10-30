# Probe vs token confidence, and how early the probe knows (TF-IDF Phase 1, probe `probes_lr_C0.01_L12`)

Rows: labelled hops 1–2 of the probe's split, no counterfactuals. Train 6325 hops (fail 0.42), test 1586 hops (fail 0.41). Confidence signals come from the same teacher-forced pass as the probe features, over the hop's tokens. CIs: bootstrap over question groups. Thresholds: the probe uses its deployed τ = 0.650; every other signal gets its own F0.5-optimal threshold on the train rows.

Sanity: probe test AUROC from this pass 0.9542 vs the bundle's 0.9542.

## A. Hop-failure detection on the test split

### hops 1–2 (n = 1586, fail 0.41)

| signal | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − signal, AUROC [95% CI] |
|---|---|---|---|---|---|---|
| **Probe** (hidden state, block 12) | 0.954 [0.944, 0.964] | 0.899 | 0.803 | 0.878 | 0.063 | – |
| mean token NLL | 0.628 [0.601, 0.655] | 0.593 | 0.442 | 0.555 | 0.212 | +0.326 [+0.299, +0.354] |
| max token NLL (FLARE: min token prob) | 0.592 [0.562, 0.620] | 0.505 | 0.487 | 0.501 | 0.333 | +0.363 [+0.333, +0.392] |
| mean token entropy | 0.677 [0.651, 0.705] | 0.686 | 0.427 | 0.612 | 0.136 | +0.277 [+0.250, +0.304] |
| max token entropy (DRAGIN-style) | 0.642 [0.614, 0.670] | 0.589 | 0.449 | 0.554 | 0.218 | +0.312 [+0.283, +0.341] |
| hop length (tokens) | 0.417 [0.389, 0.447] | 0.410 | 1.000 | 0.465 | 1.000 | +0.537 [+0.508, +0.567] |
| LR on all five confidence signals | 0.731 [0.707, 0.756] | 0.663 | 0.527 | 0.631 | 0.186 | +0.223 [+0.198, +0.249] |
| Probe + all five confidence signals (LR) | 0.953 [0.943, 0.964] | 0.903 | 0.805 | 0.882 | 0.060 | +0.001 [+0.000, +0.001] |

### hop 1 (n = 798, fail 0.17)

| signal | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − signal, AUROC [95% CI] |
|---|---|---|---|---|---|---|
| **Probe** (hidden state, block 12) | 0.941 [0.919, 0.960] | 0.832 | 0.604 | 0.773 | 0.026 | – |
| mean token NLL | 0.644 [0.591, 0.698] | 0.306 | 0.475 | 0.329 | 0.228 | +0.297 [+0.240, +0.353] |
| max token NLL (FLARE: min token prob) | 0.591 [0.535, 0.644] | 0.239 | 0.532 | 0.268 | 0.358 | +0.351 [+0.295, +0.407] |
| mean token entropy | 0.684 [0.630, 0.738] | 0.429 | 0.410 | 0.425 | 0.115 | +0.257 [+0.203, +0.314] |
| max token entropy (DRAGIN-style) | 0.648 [0.596, 0.701] | 0.279 | 0.381 | 0.295 | 0.208 | +0.293 [+0.240, +0.347] |
| hop length (tokens) | 0.449 [0.388, 0.507] | 0.174 | 1.000 | 0.209 | 1.000 | +0.492 [+0.437, +0.551] |
| LR on all five confidence signals | 0.695 [0.647, 0.745] | 0.372 | 0.460 | 0.387 | 0.164 | +0.246 [+0.191, +0.299] |
| Probe + all five confidence signals (LR) | 0.940 [0.918, 0.959] | 0.833 | 0.612 | 0.777 | 0.026 | +0.001 [-0.000, +0.002] |

### hop 2 (n = 788, fail 0.65)

| signal | AUROC [95% CI] | precision | recall | F0.5 | FPR | probe − signal, AUROC [95% CI] |
|---|---|---|---|---|---|---|
| **Probe** (hidden state, block 12) | 0.931 [0.912, 0.951] | 0.913 | 0.857 | 0.901 | 0.152 | – |
| mean token NLL | 0.704 [0.666, 0.744] | 0.822 | 0.434 | 0.697 | 0.174 | +0.227 [+0.189, +0.264] |
| max token NLL (FLARE: min token prob) | 0.655 [0.615, 0.693] | 0.764 | 0.475 | 0.681 | 0.272 | +0.277 [+0.239, +0.318] |
| mean token entropy | 0.733 [0.694, 0.772] | 0.812 | 0.432 | 0.691 | 0.185 | +0.199 [+0.161, +0.235] |
| max token entropy (DRAGIN-style) | 0.681 [0.641, 0.721] | 0.781 | 0.467 | 0.688 | 0.243 | +0.250 [+0.211, +0.291] |
| hop length (tokens) | 0.519 [0.480, 0.561] | 0.650 | 1.000 | 0.699 | 1.000 | +0.412 [+0.366, +0.458] |
| LR on all five confidence signals | 0.724 [0.687, 0.765] | 0.809 | 0.545 | 0.737 | 0.239 | +0.207 [+0.168, +0.246] |
| Probe + all five confidence signals (LR) | 0.931 [0.911, 0.950] | 0.918 | 0.857 | 0.906 | 0.141 | +0.001 [-0.000, +0.002] |

### Failures the model is confident about

Hops where FLARE's rule (max token NLL ≥ 0.77, i.e. some token prob < 0.463) does **not** fire: n = 958, of which 334 fail (0.35).

| subset | n | fail | probe AUROC [95% CI] | probe recall at τ | probe FPR at τ | mean-NLL AUROC |
|---|---|---|---|---|---|---|
| FLARE silent (model confident) | 958 | 334 | 0.958 [0.945, 0.970] | 0.775 | 0.053 | 0.528 |
| FLARE fires | 628 | 317 | 0.943 [0.924, 0.961] | 0.833 | 0.084 | 0.705 |

Of the 334 failing hops FLARE misses, the probe flags 259. Probe-flagged hops that FLARE also flags: 290 / 582.

Spearman ρ between the probe score and mean token NLL on the test hops: +0.25.

## B. How early inside a hop the probe knows

The deployed probe (no retraining) on the running mean of the hop's block-12 states after the first m tokens; the confidence signals on the same prefix. Test split, hops 1–2. Hop length: median 21 tokens (IQR 18–25).

| prefix | probe AUROC [95% CI] | mean NLL | max NLL | mean entropy |
|---|---|---|---|---|
| first 1 token | 0.749 [0.726, 0.773] | 0.727 | 0.727 | 0.737 |
| first 2 tokens | 0.553 [0.523, 0.582] | 0.729 | 0.727 | 0.739 |
| first 3 tokens | 0.707 [0.682, 0.731] | 0.557 | 0.576 | 0.594 |
| first 5 tokens | 0.766 [0.743, 0.789] | 0.615 | 0.603 | 0.652 |
| first 8 tokens | 0.788 [0.766, 0.810] | 0.481 | 0.458 | 0.551 |
| first 12 tokens | 0.787 [0.767, 0.809] | 0.473 | 0.490 | 0.520 |
| 10 % of the hop | 0.585 [0.556, 0.614] | 0.619 | 0.620 | 0.648 |
| 20 % of the hop | 0.722 [0.698, 0.746] | 0.596 | 0.588 | 0.628 |
| 30 % of the hop | 0.757 [0.733, 0.781] | 0.558 | 0.536 | 0.583 |
| 40 % of the hop | 0.789 [0.768, 0.809] | 0.441 | 0.427 | 0.477 |
| 50 % of the hop | 0.792 [0.773, 0.814] | 0.446 | 0.453 | 0.487 |
| 60 % of the hop | 0.815 [0.796, 0.836] | 0.467 | 0.485 | 0.517 |
| 70 % of the hop | 0.860 [0.843, 0.878] | 0.516 | 0.523 | 0.575 |
| 80 % of the hop | 0.908 [0.895, 0.923] | 0.568 | 0.564 | 0.625 |
| 90 % of the hop | 0.943 [0.932, 0.953] | 0.597 | 0.578 | 0.648 |
| 100 % of the hop | 0.954 [0.944, 0.964] | 0.628 | 0.592 | 0.677 |

The probe's scale is fitted on full hops, so prefix scores are rank-comparable (AUROC) but not threshold-comparable: after 3 tokens (mostly the template 'From the context, I find that') the median raw logit is +4.75 vs -1.24 at the end of the hop, and 97% of hops are above τ. A mid-hop trigger would need a probe trained on prefixes.
