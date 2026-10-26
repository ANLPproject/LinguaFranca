# LLM-as-judge vs probe (identical held-out rows: hops 1-2, canonical split)

Rows: 1617

| detector | stratum | acc | bal. acc | AUROC | TPR | FPR | n |
|---|---|---|---|---|---|---|---|
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | all | 0.925 | 0.900 | 0.958 | 0.853 | 0.053 | 1617 |
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | hop1 | 0.951 | 0.860 | 0.955 | 0.750 | 0.031 | 814 |
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | hop2 | 0.898 | 0.894 | 0.946 | 0.875 | 0.087 | 803 |
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | natural | 0.923 | 0.896 | 0.955 | 0.844 | 0.053 | 1593 |
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | cf | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 24 |
| Qwen/Qwen2.5-3B-Instruct | all | 0.427 | 0.552 | 0.597 | 0.794 | 0.690 | 1617 |
| Qwen/Qwen2.5-3B-Instruct | hop1 | 0.385 | 0.544 | 0.613 | 0.735 | 0.647 | 814 |
| Qwen/Qwen2.5-3B-Instruct | hop2 | 0.469 | 0.526 | 0.546 | 0.807 | 0.755 | 803 |
| Qwen/Qwen2.5-3B-Instruct | natural | 0.420 | 0.550 | 0.600 | 0.790 | 0.690 | 1593 |
| Qwen/Qwen2.5-3B-Instruct | cf | 0.875 | 0.935 | 0.913 | 0.870 | 0.000 | 24 |
| Qwen/Qwen2.5-7B-Instruct | all | 0.672 | 0.656 | 0.718 | 0.627 | 0.314 | 1617 |
| Qwen/Qwen2.5-7B-Instruct | hop1 | 0.768 | 0.646 | 0.709 | 0.500 | 0.208 | 814 |
| Qwen/Qwen2.5-7B-Instruct | hop2 | 0.574 | 0.587 | 0.642 | 0.654 | 0.479 | 803 |
| Qwen/Qwen2.5-7B-Instruct | natural | 0.680 | 0.673 | 0.737 | 0.661 | 0.315 | 1593 |
| Qwen/Qwen2.5-7B-Instruct | cf | 0.125 | 0.543 | 0.130 | 0.087 | 0.000 | 24 |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | all | 0.740 | 0.531 | 0.604 | 0.129 | 0.067 | 1617 |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | hop1 | 0.853 | 0.539 | 0.597 | 0.162 | 0.084 | 814 |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | hop2 | 0.625 | 0.541 | 0.711 | 0.121 | 0.039 | 803 |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | natural | 0.750 | 0.534 | 0.604 | 0.134 | 0.067 | 1593 |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | cf | 0.083 | 0.522 | 0.826 | 0.043 | 0.000 | 24 |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | all | 0.738 | 0.546 | 0.606 | 0.175 | 0.083 | 1617 |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | hop1 | 0.846 | 0.582 | 0.674 | 0.265 | 0.101 | 814 |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | hop2 | 0.629 | 0.550 | 0.620 | 0.156 | 0.056 | 803 |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | natural | 0.748 | 0.550 | 0.612 | 0.183 | 0.083 | 1593 |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | cf | 0.083 | 0.522 | 1.000 | 0.043 | 0.000 | 24 |

Failure recall by type:

| detector | hedging fails | confident fails | hop-1 confident fails |
|---|---|---|---|
| probe (layers {'hop1': 10, 'hop2': 13, 'hop12': 13}) | 0.958 (n=24) | 0.847 (n=365) | 0.691 (n=55) |
| Qwen/Qwen2.5-3B-Instruct | 0.875 (n=24) | 0.789 (n=365) | 0.709 (n=55) |
| Qwen/Qwen2.5-7B-Instruct | 0.167 (n=24) | 0.658 (n=365) | 0.618 (n=55) |
| meta-llama/Llama-3.1-8B-Instruct (via unsloth/Llama-3.1-8B-Instruct) | 0.042 (n=24) | 0.134 (n=365) | 0.200 (n=55) |
| meta-llama/Llama-3.2-3B-Instruct (via unsloth/Llama-3.2-3B-Instruct) | 0.042 (n=24) | 0.184 (n=365) | 0.327 (n=55) |