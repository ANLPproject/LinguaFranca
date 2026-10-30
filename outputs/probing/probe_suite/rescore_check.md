# Exact re-scoring of the saved selected-layer probes

Probes reloaded from `probes/*.joblib` and applied to the real test-split hidden states. `stored` = AUROC in cells.jsonl (computed from unrounded scores during the run); `exact` = recomputed now.

| setting | model | task | layer | stored AUROC | exact AUROC | |diff| |
|---|---|---|---|---|---|---|
| full10 | lr_default | hop1 | 11 | 0.9547 | 0.9547 | 0.0000 |
| full10 | lr | hop1 | 13 | 0.9650 | 0.9650 | 0.0000 |
| full10 | svm | hop1 | 12 | 0.9597 | 0.9597 | 0.0000 |
| full10 | mlp | hop1 | 11 | 0.9552 | 0.9552 | 0.0000 |
| full10 | lr_default | hop2 | 13 | 0.9465 | 0.9465 | 0.0000 |
| full10 | lr | hop2 | 13 | 0.9552 | 0.9552 | 0.0000 |
| full10 | svm | hop2 | 13 | 0.9539 | 0.9539 | 0.0000 |
| full10 | mlp | hop2 | 13 | 0.9691 | 0.9691 | 0.0000 |
| full10 | lr_default | hop12 | 13 | 0.9544 | 0.9544 | 0.0000 |
| full10 | lr | hop12 | 13 | 0.9643 | 0.9643 | 0.0000 |
| full10 | svm | hop12 | 13 | 0.9626 | 0.9626 | 0.0000 |
| full10 | mlp | hop12 | 13 | 0.9633 | 0.9633 | 0.0000 |
| tfidf | lr_default | hop1 | 11 | 0.9308 | 0.9308 | 0.0000 |
| tfidf | lr | hop1 | 13 | 0.9460 | 0.9460 | 0.0000 |
| tfidf | svm | hop1 | 13 | 0.9521 | 0.9521 | 0.0000 |
| tfidf | mlp | hop1 | 13 | 0.9588 | 0.9588 | 0.0000 |
| tfidf | lr_default | hop2 | 14 | 0.9158 | 0.9158 | 0.0000 |
| tfidf | lr | hop2 | 13 | 0.9238 | 0.9238 | 0.0000 |
| tfidf | svm | hop2 | 13 | 0.9241 | 0.9241 | 0.0000 |
| tfidf | mlp | hop2 | 13 | 0.9397 | 0.9397 | 0.0000 |
| tfidf | lr_default | hop12 | 14 | 0.9346 | 0.9346 | 0.0000 |
| tfidf | lr | hop12 | 14 | 0.9501 | 0.9501 | 0.0000 |
| tfidf | svm | hop12 | 14 | 0.9500 | 0.9500 | 0.0000 |
| tfidf | mlp | hop12 | 13 | 0.9636 | 0.9636 | 0.0000 |

Largest |stored - exact| AUROC over all selected probes: **0.00000**.