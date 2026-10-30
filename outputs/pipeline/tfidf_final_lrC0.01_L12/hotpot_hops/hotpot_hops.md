# HotpotQA hop labels: detection vs repair out of domain (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

Labels: hop 1 fails if it does not name the bridge entity (the supporting paragraph whose title is not in the question), hop 2 fails if it does not contain the gold answer; presence check = the pipeline's relabelling matcher (exact / normalised / SBERT ≥ 0.70, no LLM tier). Bridge questions only. Correct = token F1 ≥ 0.5.

## 1. Are the automatic rules trustworthy? (checked on 2Wiki)

* Bridge-title rule vs the gold hop-1 entity, 500 questions: 473 agree, 13 disagree, 14 ambiguous (dropped) → 97.3% of resolved questions agree.
* Presence labels vs Phase 1 labels (LLM-checked), 952 hops: agreement 97.0% (Phase 1 fail → presence fail 396, → pass 0; Phase 1 pass → presence pass 527, → fail 29).

## 2. Hop-failure detection on HotpotQA bridge questions

404 bridge questions; 234 with an unambiguous bridge title. AUROC [95% CI]; 2Wiki (in domain) for reference.

| hop | dataset | n | fail | probe | judge Llama-3.2-3B-Instruct | judge Llama-3.1-8B-Instruct | judge Qwen2.5-3B-Instruct | judge Qwen2.5-7B-Instruct | max token NLL (FLARE) | mean token entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HotpotQA | 234 | 82 | 0.738 [0.675, 0.802] | 0.527 [0.450, 0.604] | 0.606 [0.532, 0.678] | 0.561 [0.485, 0.638] | 0.642 [0.573, 0.715] | 0.521 [0.448, 0.600] | 0.595 [0.525, 0.672] |
| 1 | 2Wiki | 499 | 85 | 0.947 [0.923, 0.968] | 0.569 [0.501, 0.628] | 0.597 [0.532, 0.660] | 0.558 [0.493, 0.619] | 0.779 [0.723, 0.831] | 0.620 [0.551, 0.686] | 0.727 [0.658, 0.782] |
| 2 | HotpotQA | 225 | 127 | 0.675 [0.602, 0.744] | 0.601 [0.521, 0.679] | 0.667 [0.595, 0.737] | 0.597 [0.528, 0.675] | 0.633 [0.559, 0.709] | 0.631 [0.559, 0.701] | 0.697 [0.629, 0.767] |
| 2 | 2Wiki | 492 | 327 | 0.925 [0.897, 0.948] | 0.678 [0.626, 0.727] | 0.729 [0.677, 0.777] | 0.562 [0.506, 0.618] | 0.690 [0.640, 0.735] | 0.651 [0.600, 0.699] | 0.734 [0.687, 0.783] |

## 3. Accuracy, oracle and error budget on HotpotQA bridge questions

| system | accuracy | vs probe [95% CI] |
|---|---|---|
| None | 0.479 | +0.085 [+0.030, +0.141] |
| Probe (ours) | 0.564 | – |
| Judge Llama-3.2-3B-Instruct | 0.530 | +0.034 [-0.017, +0.085] |
| Oracle trigger | 0.628 | -0.064 [-0.107, -0.021] |

The probe gets 57% of the oracle trigger's gain over no intervention (2Wiki: 94 %).

| stage | Probe (ours) | Judge Llama-3.2-3B-Instruct | Oracle trigger |
|---|---|---|---|
| no hop failed | 19 (8.1%) | 16 (6.8%) | 16 (6.8%) |
| detector missed the failing hop | 31 (13.2%) | 74 (31.6%) | 3 (1.3%) |
| detector flagged the wrong hop | 20 (8.5%) | 10 (4.3%) | 0 (0.0%) |
| restarted: evidence was already in the context | 8 (3.4%) | 2 (0.9%) | 17 (7.3%) |
| restarted: retrieval missed the gold paragraph | 11 (4.7%) | 3 (1.3%) | 27 (11.5%) |
| restarted: gold paragraph retrieved, answer still wrong | 13 (5.6%) | 5 (2.1%) | 24 (10.3%) |
| **all errors** | **102 (43.6%)** | **110 (47.0%)** | **87 (37.2%)** |