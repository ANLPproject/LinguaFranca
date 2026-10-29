# The probe as a verifier: which answer to keep (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

Each candidate trajectory's failure score = max hop P(fail), from the probe (hidden states) or the text judge (Llama-3.2-3B, the pipeline's judge); both scored every candidate during the run. The candidate with the lower score is kept (ties → the original). Same candidates for every row; one restart per candidate. Correct = token F1 ≥ 0.5; CIs: paired bootstrap vs 'always restart after hop 2'.

## 2wiki (n = 500)

| system | accuracy | fixed | broken | vs always-restart-after-hop-2 [95% CI] |
|---|---|---|---|---|
| Original answer only (no intervention) | 0.342 | 0 | 0 | -0.212 [-0.254, -0.168] |
| Always restart after hop 2 (no verification) | 0.554 | 126 | 20 | – |
| **Always restart after hop 2 + probe verification** | 0.542 | 110 | 10 | -0.012 [-0.032, +0.008] |
| Always restart after hop 2 + judge verification (Llama-3.2-3B) | 0.500 | 90 | 11 | -0.054 [-0.080, -0.028] |
| **Probe trigger (τ = 0.650) + probe verification** | 0.532 | 104 | 9 | -0.022 [-0.046, +0.000] |
| **Best of 3 (original, hop-1 restart, hop-2 restart) chosen by the probe** | 0.550 | 113 | 9 | -0.004 [-0.030, +0.022] |
| Best of 3 chosen by the judge (Llama-3.2-3B) | 0.506 | 94 | 12 | -0.048 [-0.078, -0.020] |
| Best of 3 chosen at random | 0.462 | 72 | 12 | -0.092 [-0.128, -0.058] |
| Best of 3, oracle choice (upper bound) | 0.626 | 142 | 0 | +0.072 [+0.050, +0.096] |

## hotpotqa (n = 500)

| system | accuracy | fixed | broken | vs always-restart-after-hop-2 [95% CI] |
|---|---|---|---|---|
| Original answer only (no intervention) | 0.504 | 0 | 0 | -0.082 [-0.124, -0.038] |
| Always restart after hop 2 (no verification) | 0.586 | 89 | 48 | – |
| **Always restart after hop 2 + probe verification** | 0.566 | 64 | 33 | -0.020 [-0.044, +0.004] |
| Always restart after hop 2 + judge verification (Llama-3.2-3B) | 0.592 | 70 | 26 | +0.006 [-0.018, +0.030] |
| **Probe trigger (τ = 0.650) + probe verification** | 0.554 | 44 | 19 | -0.032 [-0.066, +0.004] |
| **Best of 3 (original, hop-1 restart, hop-2 restart) chosen by the probe** | 0.562 | 67 | 38 | -0.024 [-0.054, +0.008] |
| Best of 3 chosen by the judge (Llama-3.2-3B) | 0.574 | 72 | 37 | -0.012 [-0.046, +0.018] |
| Best of 3 chosen at random | 0.544 | 62 | 42 | -0.042 [-0.076, -0.010] |
| Best of 3, oracle choice (upper bound) | 0.730 | 113 | 0 | +0.144 [+0.114, +0.176] |


## With a learned margin (fair version)

The rule above keeps whichever candidate has the lower score. Restarts are scored on a shifted input (extra evidence in the
prompt), so a fairer rule keeps the hop-2 restart unless its score exceeds the original's by more than a margin δ, with δ
chosen by 2-fold cross-validation over the 2Wiki questions (HotpotQA: δ fitted on all of 2Wiki). Scores: max over hops or the hop-2 score.

| dataset | verifier | score | accuracy | vs always restart after hop 2 [95% CI] |
|---|---|---|---|---|
| 2Wiki (n = 492) | probe | max | 0.559 | +0.004 [−0.004, +0.012] |
| 2Wiki | probe | hop 2 | 0.557 | +0.002 [−0.006, +0.012] |
| 2Wiki | judge (Llama-3.2-3B) | max | 0.553 | −0.002 [−0.012, +0.008] |
| 2Wiki | judge | hop 2 | 0.549 | −0.006 [−0.018, +0.006] |
| HotpotQA (n = 486) | probe | max | 0.580 | −0.006 [−0.014, +0.000] |
| HotpotQA | probe | hop 2 | 0.580 | −0.006 [−0.016, +0.002] |
| HotpotQA | judge | max | 0.597 | +0.010 [+0.000, +0.021] |
| HotpotQA | judge | hop 2 | 0.593 | +0.006 [+0.000, +0.014] |

## Reading

With a learned margin, every verifier keeps the restart almost always and lands on 'always restart after hop 2'.
Choosing between the original and the restarted answer adds no accuracy in this setting, although an oracle choice among
three candidates would add 7 points (2Wiki 0.626). Without a margin, the probe is the much better verifier on 2Wiki
(best of 3: 0.550 vs 0.506 for the judge) and halves the broken answers. In this setting the repair, not the trigger or
the verifier, bounds end-to-end accuracy.
