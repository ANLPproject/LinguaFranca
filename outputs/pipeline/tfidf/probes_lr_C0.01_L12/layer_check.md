# Tuned LR probe (C = 0.01) — layer check on the TRAIN split only

Layer 12 was proposed from the counterfactual layer sweep on branch `cf-tfidf` (notebook 07), which ranks layers on
test-split counterfactuals and sets its threshold on natural test hop-1 rows. To make sure the choice does not
depend on test data, the same probe recipe (StandardScaler + LogisticRegression, C = 0.01, class_weight balanced)
was cross-validated inside the train split only (5-fold GroupKFold, `python -m src.pipeline.probes --C 0.01 --sweep ...`):

| layer | 6 | 8 | 9 | 10 | 11 | **12** | 13 | 14 | 15 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared (hop 1+2) CV AUROC | 0.924 | 0.940 | 0.943 | 0.950 | 0.953 | **0.952** | 0.953 | 0.954 | 0.952 | 0.951 | 0.946 | 0.942 |
| hop 1 CV AUROC | 0.906 | 0.925 | 0.934 | 0.944 | 0.949 | **0.947** | 0.951 | 0.949 | 0.949 | 0.945 | 0.938 | 0.930 |
| hop 2 CV AUROC | 0.911 | 0.926 | 0.931 | 0.931 | 0.931 | **0.934** | 0.936 | 0.934 | 0.933 | 0.931 | 0.929 | 0.931 |

Layer 12 is within 0.002 of the best train-CV layer (14) on a flat plateau (layers 11–15), so the choice is
supported by training data alone.

The same model is also in `cf-tfidf:outputs/patching/patching_tuned_results/probes_c001_L12` (identical rows,
weight cosine 1.0000, test AUROC 0.954), but its threshold τ = 0.585 differs from this bundle's τ = 0.650 because
scikit-learn 1.6.1 (Kaggle) and 1.8.0 (here) assign the cross-validation folds differently, giving different
out-of-fold scores. All Phase 3–5 results in `outputs/pipeline/tfidf_final_lrC0.01_L12/` use this bundle (τ = 0.650).
