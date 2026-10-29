# Selective answering and accuracy vs cost (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

Items: 1000. Correct = token F1 ≥ 0.5. Failure score of a trajectory = max over its hops of P(fail) (probe: calibrated, τ = 0.650; judges: their P(fail)). Questions are answered from the lowest score up; AURC = mean error over all coverages (lower = better; 'oracle' = perfect ranking, 'random' = error rate). Ties broken at random. CIs: paired bootstrap over questions (2000 resamples).

## 1a. Abstaining on the first answers (no restart)

Every signal ranks the same answers: the model's uninterrupted trajectories.

### 2wiki (n = 500, accuracy at full coverage 0.342)

| signal | acc @ 50 % | acc @ 80 % | AURC | probe − signal, AURC [95% CI] |
|---|---|---|---|---|
| Probe (ours) | 0.552 | 0.422 | 0.418 | – |
| Judge Llama-3.2-3B-Instruct | 0.472 | 0.385 | 0.528 | -0.109 [-0.151, -0.069] |
| Judge Llama-3.1-8B-Instruct | 0.432 | 0.385 | 0.569 | -0.151 [-0.198, -0.103] |
| Judge Qwen2.5-3B-Instruct | 0.404 | 0.350 | 0.615 | -0.197 [-0.244, -0.149] |
| Judge Qwen2.5-7B-Instruct | 0.448 | 0.367 | 0.550 | -0.132 [-0.177, -0.090] |
| *random* | 0.342 | 0.342 | 0.658 | |
| *oracle* | 0.684 | 0.427 | 0.292 | |

### hotpotqa (n = 500, accuracy at full coverage 0.504)

| signal | acc @ 50 % | acc @ 80 % | AURC | probe − signal, AURC [95% CI] |
|---|---|---|---|---|
| Probe (ours) | 0.576 | 0.547 | 0.433 | – |
| Judge Llama-3.2-3B-Instruct | 0.544 | 0.505 | 0.433 | +0.000 [-0.057, +0.056] |
| Judge Llama-3.1-8B-Instruct | 0.564 | 0.527 | 0.431 | +0.002 [-0.055, +0.059] |
| Judge Qwen2.5-3B-Instruct | 0.520 | 0.502 | 0.461 | -0.028 [-0.084, +0.026] |
| Judge Qwen2.5-7B-Instruct | 0.556 | 0.527 | 0.430 | +0.003 [-0.047, +0.053] |
| *random* | 0.504 | 0.504 | 0.496 | |
| *oracle* | 1.000 | 0.630 | 0.151 | |

## 1b. Restart, then abstain: each system's final answers, ranked by its own detector

Probe pipeline: one probe restart, then the probe's score on the final trajectory. Judge pipelines: one judge restart, then that judge's score on the final trajectory. Accuracy and abstention both differ here, so this is the whole system: how accurate is it on the questions it chooses to answer?

### 2wiki

| system | accuracy | acc @ 50 % | acc @ 80 % | AURC | probe − system, AURC [95% CI] |
|---|---|---|---|---|---|
| Probe (ours) | 0.538 | 0.748 | 0.620 | 0.272 | – |
| Judge Llama-3.2-3B-Instruct | 0.454 | 0.544 | 0.502 | 0.447 | -0.175 [-0.229, -0.122] |

### hotpotqa

| system | accuracy | acc @ 50 % | acc @ 80 % | AURC | probe − system, AURC [95% CI] |
|---|---|---|---|---|---|
| Probe (ours) | 0.570 | 0.620 | 0.583 | 0.394 | – |
| Judge Llama-3.2-3B-Instruct | 0.526 | 0.592 | 0.555 | 0.390 | +0.004 [-0.061, +0.068] |

Judges rescored only the main judge's restarts, so 1b compares with the main judge (Llama-3.2-3B-Instruct) only.

## 2. Accuracy vs cost

Tokens processed per question = prompt + generated tokens of every forward pass, plus the judge's own tokens (probe cost is one dot product per hop boundary). Oracle: 2Wiki only (needs gold hop labels).

### 2wiki

| system | accuracy | tokens processed / question | restart rate |
|---|---|---|---|
| No intervention | 0.342 | 911 | 0.00 |
| Always retrieve | 0.356 | 1209 | 1.00 |
| Always restart after hop 2 | 0.554 | 1968 | 0.98 |
| Probe (ours) | 0.538 | 1574 | 0.64 |
| Probe, up to 2 restarts | 0.594 | 2049 | 0.64 |
| Judge Llama-3.2-3B-Instruct | 0.454 | 2576 | 0.46 |
| Judge Llama-3.2-3B-Instruct, up to 2 restarts | 0.470 | 3620 | 0.46 |
| Judge Llama-3.1-8B-Instruct | 0.410 | 2359 | 0.27 |
| Judge Qwen2.5-3B-Instruct | 0.456 | 3068 | 0.89 |
| Judge Qwen2.5-7B-Instruct | 0.508 | 2785 | 0.63 |
| Oracle trigger | 0.550 | 1619 | 0.69 |

### hotpotqa

| system | accuracy | tokens processed / question | restart rate |
|---|---|---|---|
| No intervention | 0.504 | 1072 | 0.00 |
| Always retrieve | 0.546 | 1471 | 1.00 |
| Always restart after hop 2 | 0.586 | 2351 | 0.97 |
| Probe (ours) | 0.570 | 1831 | 0.59 |
| Probe, up to 2 restarts | 0.564 | 2368 | 0.59 |
| Judge Llama-3.2-3B-Instruct | 0.526 | 2860 | 0.23 |
| Judge Llama-3.2-3B-Instruct, up to 2 restarts | 0.534 | 3456 | 0.23 |
| Judge Llama-3.1-8B-Instruct | 0.520 | 2770 | 0.16 |
| Judge Qwen2.5-3B-Instruct | 0.556 | 3606 | 0.75 |
| Judge Qwen2.5-7B-Instruct | 0.546 | 3248 | 0.49 |
