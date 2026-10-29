# Does the model know hop 2 will fail before writing it? (anticipatory probe)

Rows: questions whose hop 1 is **correct** and hop 2 is labelled. Train split 2509 rows (hop-2 failure rate 0.64); evaluated on the held-out test split, 651 rows (failure rate 0.63). Probes are trained on the train split only. All signals are available at the end of hop 1, before hop 2 is generated (except the reference row).

| signal (end of hop 1) | test AUROC for 'hop 2 will fail' |
|---|---|
| **Hidden state at the end of hop 1** (layer 8, probe recipe C = 0.01) | 0.829 |
| **Hidden state at the end of hop 1** (layer 10, probe recipe C = 0.01) | 0.825 |
| **Hidden state at the end of hop 1** (layer 12, probe recipe C = 0.01) | 0.821 |
| **Hidden state at the end of hop 1** (layer 14, probe recipe C = 0.01) | 0.817 |
| **Hidden state at the end of hop 1** (layer 16, probe recipe C = 0.01) | 0.820 |
| Hop-1 text (TF-IDF + LR, same rows and labels) | 0.623 |
| Rule: the bridge entity named in hop 1 has no page in the context | 0.683 |
| Oracle rule (uses gold titles): the gold hop-2 paragraph is missing | 0.722 |
| *Reference: hidden state at the end of hop 2 (after it is written), layer 12* | 0.936 |

The hop-1 hidden state anticipates the hop-2 failure (AUROC ≈ 0.82 at every layer tested) far better than the hop-1 text, and better than rules about missing evidence, including one that uses gold titles. The model's state encodes more than whether the bridge entity's page is present.
