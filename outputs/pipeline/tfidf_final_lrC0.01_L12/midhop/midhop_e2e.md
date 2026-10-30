# Mid-hop stopping, end to end (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

One restart, correct = token F1 ≥ 0.5, paired bootstrap CIs over questions. A mid-hop stop restarts with the partial hop text in the retrieval query; restarts that retrieve the same paragraphs as the cached restart reuse its answer (identical prompt, greedy decoding), the others were generated (answers forced at the token cap, as in the pipeline).

Checks: retrieval parity on the run's cached hop restarts 1977/1977 identical; the deployed system here vs the pipeline's own accounting: 100.0% of questions identical.

## 2wiki (n = 500)

| system | accuracy | Δ vs deployed [95% CI] | restart rate | of which mid-hop | mid-hop retrieval changed | fixed | broken | generated tokens / q | processed tokens / q |
|---|---|---|---|---|---|---|---|---|---|
| Deployed probe at `</hopK>` (headline) | 0.538 | – | 0.64 | 0% | 0% | 108 | 10 | 160.1 | 1574 |
| **Hybrid**: prefix probe (strict θ) mid-hop, else deployed at `</hopK>` | 0.554 | +0.016 [-0.004, +0.036] | 0.67 | 58% | 80% | 115 | 9 | 149.6 | 1604 |
| Prefix probe only (F0.5 θ = 0.962) | 0.532 | -0.006 [-0.036, +0.026] | 0.63 | 95% | 86% | 102 | 7 | 144.9 | 1573 |
| No intervention | 0.342 | | 0.00 | | | 0 | 0 | | |

## hotpotqa (n = 500)

| system | accuracy | Δ vs deployed [95% CI] | restart rate | of which mid-hop | mid-hop retrieval changed | fixed | broken | generated tokens / q | processed tokens / q |
|---|---|---|---|---|---|---|---|---|---|
| Deployed probe at `</hopK>` (headline) | 0.570 | – | 0.59 | 0% | 0% | 55 | 22 | 153.7 | 1831 |
| **Hybrid**: prefix probe (strict θ) mid-hop, else deployed at `</hopK>` | 0.562 | -0.008 [-0.036, +0.020] | 0.77 | 65% | 78% | 67 | 38 | 158.1 | 2080 |
| Prefix probe only (F0.5 θ = 0.962) | 0.562 | -0.008 [-0.046, +0.028] | 0.77 | 99% | 83% | 67 | 38 | 153.7 | 2094 |
| No intervention | 0.504 | | 0.00 | | | 0 | 0 | | |

## Retrieval with the partial-hop query (2Wiki)

Mid-hop stops of each system where at least one gold paragraph is missing from the first-stage context. 'Full' = the cached restart's query (complete hop text, what the deployed probe would have used); 'partial' = the query actually used after the mid-hop stop. Found = a missing gold paragraph is among the 3 retrieved.

| system | mid-hop stops (gold missing) | full query finds it | partial query finds it | both | only full | only partial | accuracy if partial found / not found |
|---|---|---|---|---|---|---|---|
| **Hybrid**: prefix probe (strict θ) mid-hop, else deployed at `</hopK>` | 165 | 56.4% | 58.8% | 88 | 5 | 9 | 0.722 / 0.074 |
| Prefix probe only (F0.5 θ = 0.962) | 237 | 53.2% | 54.0% | 117 | 9 | 11 | 0.750 / 0.092 |

(Kaggle runs of `scripts/midhop_e2e.py`: `--stage all`, then `--stage report` for the retrieval section; the accuracy tables reproduced exactly. The prefix probe's own report is `outputs/pipeline/tfidf/online_probe/online_probe.md`.)
