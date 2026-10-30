# Paired bootstrap tests between probes (test split)

Each row: difference in test AUROC (and recall at 3% false alarms) between two models on the SAME rows, 2000 resamples of whole question groups, 95% interval of the difference. **Supported** = the interval excludes 0. Many comparisons are listed, so treat a lone borderline interval with caution.

## full10

Alignment check: recomputed AUROC equals the stored AUROC for the 336 cells (tie-free cells: max difference 0.00001; default LR / MLP cells have rounding ties from the 5-decimal score storage: max difference 0.0088). Shuffling the row order drops every AUROC by more than 0.15, so rows are aligned.

### Models compared at their CV-selected layers (exact scores recomputed from the saved probes)

| task | A vs B | layers | AUROC A / B | AUROC diff [95% CI] | supported? | recall@3%FPR diff [95% CI] | supported? |
|---|---|---|---|---|---|---|---|
| hop1 | tuned LR vs default LR | 13 / 11 | 0.965 / 0.955 | +0.010 [-0.005, +0.028] | no | +0.000 [-0.118, +0.078] | no |
| hop1 | SVM vs tuned LR | 12 / 13 | 0.960 / 0.965 | -0.005 [-0.017, +0.006] | no | -0.029 [-0.116, +0.074] | no |
| hop1 | MLP vs tuned LR | 11 / 13 | 0.955 / 0.965 | -0.010 [-0.029, +0.004] | no | -0.059 [-0.145, +0.064] | no |
| hop1 | MLP vs default LR | 11 / 11 | 0.955 / 0.955 | +0.000 [-0.014, +0.016] | no | -0.059 [-0.153, +0.034] | no |
| hop1 | MLP vs SVM | 11 / 12 | 0.955 / 0.960 | -0.005 [-0.022, +0.008] | no | -0.029 [-0.139, +0.086] | no |
| hop2 | tuned LR vs default LR | 13 / 13 | 0.955 / 0.946 | +0.009 [+0.004, +0.014] | yes | +0.028 [-0.006, +0.103] | no |
| hop2 | SVM vs tuned LR | 13 / 13 | 0.954 / 0.955 | -0.001 [-0.003, +0.001] | no | +0.028 [-0.025, +0.042] | no |
| hop2 | MLP vs tuned LR | 13 / 13 | 0.969 / 0.955 | +0.014 [+0.007, +0.021] | yes | +0.072 [-0.013, +0.121] | no |
| hop2 | MLP vs default LR | 13 / 13 | 0.969 / 0.946 | +0.023 [+0.013, +0.033] | yes | +0.100 [+0.023, +0.181] | yes |
| hop2 | MLP vs SVM | 13 / 13 | 0.969 / 0.954 | +0.015 [+0.007, +0.024] | yes | +0.044 [-0.031, +0.110] | no |
| hop12 | tuned LR vs default LR | 13 / 13 | 0.964 / 0.954 | +0.010 [+0.006, +0.014] | yes | +0.026 [-0.019, +0.076] | no |
| hop12 | SVM vs tuned LR | 13 / 13 | 0.963 / 0.964 | -0.002 [-0.004, +0.000] | no | +0.008 [-0.036, +0.034] | no |
| hop12 | MLP vs tuned LR | 13 / 13 | 0.963 / 0.964 | -0.001 [-0.009, +0.007] | no | +0.005 [-0.059, +0.076] | no |
| hop12 | MLP vs default LR | 13 / 13 | 0.963 / 0.954 | +0.009 [-0.000, +0.019] | no | +0.031 [-0.031, +0.107] | no |
| hop12 | MLP vs SVM | 13 / 13 | 0.963 / 0.963 | +0.001 [-0.008, +0.009] | no | -0.003 [-0.060, +0.079] | no |

### Models compared at their layer 13 (stored scores rounded to 5 decimals: default-LR / MLP AUROC can be off by up to ~0.004)

| task | A vs B | layers | AUROC A / B | AUROC diff [95% CI] | supported? | recall@3%FPR diff [95% CI] | supported? |
|---|---|---|---|---|---|---|---|
| hop1 | tuned LR vs default LR | 13 / 13 | 0.965 / 0.961 | +0.004 [-0.001, +0.010] | no | +0.000 [-0.079, +0.059] | no |
| hop1 | SVM vs tuned LR | 13 / 13 | 0.966 / 0.965 | +0.001 [-0.005, +0.008] | no | -0.029 [-0.136, +0.070] | no |
| hop1 | MLP vs tuned LR | 13 / 13 | 0.966 / 0.965 | +0.001 [-0.003, +0.005] | no | -0.044 [-0.090, +0.071] | no |
| hop1 | MLP vs default LR | 13 / 13 | 0.966 / 0.961 | +0.005 [-0.001, +0.013] | no | -0.044 [-0.099, +0.079] | no |
| hop1 | MLP vs SVM | 13 / 13 | 0.966 / 0.966 | +0.000 [-0.006, +0.006] | no | -0.015 [-0.082, +0.131] | no |
| hop2 | tuned LR vs default LR | 13 / 13 | 0.955 / 0.946 | +0.009 [+0.004, +0.015] | yes | +0.028 [-0.006, +0.103] | no |
| hop2 | SVM vs tuned LR | 13 / 13 | 0.954 / 0.955 | -0.001 [-0.003, +0.001] | no | +0.028 [-0.025, +0.042] | no |
| hop2 | MLP vs tuned LR | 13 / 13 | 0.969 / 0.955 | +0.014 [+0.007, +0.021] | yes | +0.072 [-0.013, +0.121] | no |
| hop2 | MLP vs default LR | 13 / 13 | 0.969 / 0.946 | +0.023 [+0.014, +0.034] | yes | +0.100 [+0.023, +0.181] | yes |
| hop2 | MLP vs SVM | 13 / 13 | 0.969 / 0.954 | +0.015 [+0.007, +0.024] | yes | +0.044 [-0.031, +0.110] | no |
| hop12 | tuned LR vs default LR | 13 / 13 | 0.964 / 0.954 | +0.010 [+0.006, +0.015] | yes | +0.026 [-0.019, +0.076] | no |
| hop12 | SVM vs tuned LR | 13 / 13 | 0.963 / 0.964 | -0.002 [-0.004, +0.000] | no | +0.008 [-0.036, +0.034] | no |
| hop12 | MLP vs tuned LR | 13 / 13 | 0.962 / 0.964 | -0.003 [-0.011, +0.005] | no | +0.005 [-0.059, +0.076] | no |
| hop12 | MLP vs default LR | 13 / 13 | 0.962 / 0.954 | +0.007 [-0.003, +0.017] | no | +0.031 [-0.031, +0.107] | no |
| hop12 | MLP vs SVM | 13 / 13 | 0.962 / 0.963 | -0.001 [-0.010, +0.008] | no | -0.003 [-0.060, +0.079] | no |

## tfidf

Alignment check: recomputed AUROC equals the stored AUROC for the 336 cells (tie-free cells: max difference 0.00001; default LR / MLP cells have rounding ties from the 5-decimal score storage: max difference 0.0048). Shuffling the row order drops every AUROC by more than 0.15, so rows are aligned.

### Models compared at their CV-selected layers (exact scores recomputed from the saved probes)

| task | A vs B | layers | AUROC A / B | AUROC diff [95% CI] | supported? | recall@3%FPR diff [95% CI] | supported? |
|---|---|---|---|---|---|---|---|
| hop1 | tuned LR vs default LR | 13 / 11 | 0.946 / 0.931 | +0.015 [+0.001, +0.030] | yes | +0.058 [-0.064, +0.153] | no |
| hop1 | SVM vs tuned LR | 13 / 13 | 0.952 / 0.946 | +0.006 [-0.004, +0.018] | no | -0.036 [-0.121, +0.111] | no |
| hop1 | MLP vs tuned LR | 13 / 13 | 0.959 / 0.946 | +0.013 [+0.004, +0.024] | yes | -0.036 [-0.097, +0.130] | no |
| hop1 | MLP vs default LR | 13 / 11 | 0.959 / 0.931 | +0.028 [+0.011, +0.049] | yes | +0.022 [-0.053, +0.201] | no |
| hop1 | MLP vs SVM | 13 / 13 | 0.959 / 0.952 | +0.007 [-0.003, +0.017] | no | +0.000 [-0.109, +0.145] | no |
| hop2 | tuned LR vs default LR | 13 / 14 | 0.924 / 0.916 | +0.008 [-0.007, +0.022] | no | +0.070 [-0.081, +0.277] | no |
| hop2 | SVM vs tuned LR | 13 / 13 | 0.924 / 0.924 | +0.000 [-0.002, +0.002] | no | -0.025 [-0.066, +0.031] | no |
| hop2 | MLP vs tuned LR | 13 / 13 | 0.940 / 0.924 | +0.016 [+0.006, +0.026] | yes | +0.174 [+0.060, +0.281] | yes |
| hop2 | MLP vs default LR | 13 / 14 | 0.940 / 0.916 | +0.024 [+0.010, +0.038] | yes | +0.244 [+0.100, +0.401] | yes |
| hop2 | MLP vs SVM | 13 / 13 | 0.940 / 0.924 | +0.016 [+0.005, +0.026] | yes | +0.199 [+0.059, +0.289] | yes |
| hop12 | tuned LR vs default LR | 14 / 14 | 0.950 / 0.935 | +0.015 [+0.011, +0.021] | yes | +0.094 [+0.014, +0.156] | yes |
| hop12 | SVM vs tuned LR | 14 / 14 | 0.950 / 0.950 | -0.000 [-0.001, +0.001] | no | -0.009 [-0.032, +0.036] | no |
| hop12 | MLP vs tuned LR | 13 / 14 | 0.964 / 0.950 | +0.014 [+0.008, +0.019] | yes | +0.089 [+0.014, +0.177] | yes |
| hop12 | MLP vs default LR | 13 / 14 | 0.964 / 0.935 | +0.029 [+0.021, +0.037] | yes | +0.183 [+0.090, +0.254] | yes |
| hop12 | MLP vs SVM | 13 / 14 | 0.964 / 0.950 | +0.014 [+0.008, +0.020] | yes | +0.098 [+0.009, +0.170] | yes |

### Models compared at their layer 13 (stored scores rounded to 5 decimals: default-LR / MLP AUROC can be off by up to ~0.004)

| task | A vs B | layers | AUROC A / B | AUROC diff [95% CI] | supported? | recall@3%FPR diff [95% CI] | supported? |
|---|---|---|---|---|---|---|---|
| hop1 | tuned LR vs default LR | 13 / 13 | 0.946 / 0.931 | +0.015 [+0.007, +0.025] | yes | +0.036 [-0.063, +0.108] | no |
| hop1 | SVM vs tuned LR | 13 / 13 | 0.952 / 0.946 | +0.006 [-0.004, +0.018] | no | -0.036 [-0.121, +0.111] | no |
| hop1 | MLP vs tuned LR | 13 / 13 | 0.959 / 0.946 | +0.013 [+0.004, +0.024] | yes | -0.036 [-0.097, +0.130] | no |
| hop1 | MLP vs default LR | 13 / 13 | 0.959 / 0.931 | +0.028 [+0.014, +0.046] | yes | +0.000 [-0.071, +0.176] | no |
| hop1 | MLP vs SVM | 13 / 13 | 0.959 / 0.952 | +0.007 [-0.003, +0.017] | no | +0.000 [-0.109, +0.145] | no |
| hop2 | tuned LR vs default LR | 13 / 13 | 0.924 / 0.913 | +0.010 [-0.002, +0.023] | no | -0.035 [-0.160, +0.140] | no |
| hop2 | SVM vs tuned LR | 13 / 13 | 0.924 / 0.924 | +0.000 [-0.002, +0.002] | no | -0.025 [-0.066, +0.031] | no |
| hop2 | MLP vs tuned LR | 13 / 13 | 0.940 / 0.924 | +0.016 [+0.006, +0.026] | yes | +0.174 [+0.060, +0.281] | yes |
| hop2 | MLP vs default LR | 13 / 13 | 0.940 / 0.913 | +0.026 [+0.015, +0.038] | yes | +0.139 [+0.012, +0.308] | yes |
| hop2 | MLP vs SVM | 13 / 13 | 0.940 / 0.924 | +0.016 [+0.006, +0.026] | yes | +0.199 [+0.059, +0.289] | yes |
| hop12 | tuned LR vs default LR | 13 / 13 | 0.951 / 0.934 | +0.017 [+0.012, +0.023] | yes | +0.077 [-0.035, +0.147] | no |
| hop12 | SVM vs tuned LR | 13 / 13 | 0.952 / 0.951 | +0.000 [-0.001, +0.001] | no | +0.023 [-0.017, +0.066] | no |
| hop12 | MLP vs tuned LR | 13 / 13 | 0.963 / 0.951 | +0.012 [+0.007, +0.017] | yes | +0.081 [-0.002, +0.161] | no |
| hop12 | MLP vs default LR | 13 / 13 | 0.963 / 0.934 | +0.029 [+0.021, +0.037] | yes | +0.158 [+0.046, +0.235] | yes |
| hop12 | MLP vs SVM | 13 / 13 | 0.963 / 0.952 | +0.012 [+0.006, +0.017] | yes | +0.058 [-0.014, +0.148] | no |

## all-10 passages vs TF-IDF top-5 (same model, selected layers)

The two settings use the same 4,000 questions, but their test splits are different draws, so the comparison is **unpaired**: each side is resampled independently. Rows that are test rows in both settings: hop1: 798, hop2: 781, hop12: 1579.

| task | model | full10 AUROC | TF-IDF AUROC | diff (full10 - TF-IDF) [95% CI] | supported? |
|---|---|---|---|---|---|
| hop1 | default LR | 0.955 | 0.931 | +0.024 [-0.020, +0.064] | no |
| hop1 | tuned LR | 0.965 | 0.946 | +0.019 [-0.012, +0.049] | no |
| hop1 | SVM | 0.960 | 0.952 | +0.008 [-0.028, +0.040] | no |
| hop1 | MLP | 0.955 | 0.959 | -0.004 [-0.042, +0.027] | no |
| hop2 | default LR | 0.946 | 0.916 | +0.031 [+0.003, +0.058] | yes |
| hop2 | tuned LR | 0.955 | 0.924 | +0.031 [+0.007, +0.057] | yes |
| hop2 | SVM | 0.954 | 0.924 | +0.030 [+0.005, +0.056] | yes |
| hop2 | MLP | 0.969 | 0.940 | +0.029 [+0.010, +0.048] | yes |
| hop12 | default LR | 0.954 | 0.935 | +0.020 [+0.003, +0.036] | yes |
| hop12 | tuned LR | 0.964 | 0.950 | +0.014 [+0.000, +0.028] | yes |
| hop12 | SVM | 0.963 | 0.950 | +0.013 [-0.002, +0.027] | no |
| hop12 | MLP | 0.963 | 0.964 | -0.000 [-0.013, +0.012] | no |