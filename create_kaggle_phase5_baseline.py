import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LinguaFranca - Phase 5 (LLM Judge Baseline)\n",
            "This notebook evaluates Llama-3.2 and Qwen-2.5 as text-based judges against the linear probe.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "!pip install huggingface_hub scikit-learn accelerate transformers -q"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os, subprocess, sys\n",
            "REPO_URL = 'https://github.com/ANLPproject/LinguaFranca.git'\n",
            "WORK_DIR = '/kaggle/working/LinguaFranca'\n",
            "if not os.path.exists(WORK_DIR):\n",
            "    print('Cloning repository...')\n",
            "    subprocess.run(['git', 'clone', REPO_URL, WORK_DIR], check=True)\n",
            "else:\n",
            "    print('Pulling latest changes...')\n",
            "    subprocess.run(['git', '-C', WORK_DIR, 'pull'], check=True)\n",
            "os.chdir(WORK_DIR)\n",
            "sys.path.insert(0, WORK_DIR)\n",
            "print('Working directory:', os.getcwd())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os\n",
            "from huggingface_hub import login\n",
            "HF_TOKEN = 'YOUR_HF_TOKEN_HERE'  # REPLACE WITH YOUR TOKEN\n",
            "os.environ['HF_TOKEN'] = HF_TOKEN\n",
            "# login(token=HF_TOKEN) # Uncomment to login so Llama-3.2 can download"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Download Dataset & Hidden States from HF\n",
            "We pull the data generated in Phase 3 directly from the Hugging Face Hub."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "from huggingface_hub import snapshot_download\n",
            "HF_REPO_ID = 'AnishRacherla/LinguaFranca-Phase3' # REPLACE IF NEEDED\n",
            "print(f\"Downloading dataset from {HF_REPO_ID}...\")\n",
            "snapshot_download(repo_id=HF_REPO_ID, repo_type=\"dataset\", local_dir=\"/kaggle/working\", allow_patterns=\"data/*\")\n",
            "print(\"Download complete!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Run LLM Baseline Evaluation\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "!python evaluate_llm_baseline.py --labels /kaggle/working/data/2wikimultihopqa/augmented.jsonl --hs-dir /kaggle/working/data/hidden_states"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('kaggle_phase5_baseline.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Successfully generated kaggle_phase5_baseline.ipynb")
