import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LinguaFranca - Phase 3 (Merge, Probe, Counterfactuals & Hugging Face)\n",
            "Upload your `data_merged.zip` as a dataset to this notebook.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "!pip install huggingface_hub scikit-learn nnsight pyyaml accelerate datasets sentence-transformers SPARQLWrapper -q"
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
            "    subprocess.run(['git', 'clone', '-b', 'anish-dev', REPO_URL, WORK_DIR], check=True)\n",
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
            "HF_REPO_ID = 'AnishRacherla/LinguaFranca-Phase3' # REPLACE WITH YOUR REPO ID\n",
            "os.environ['HF_TOKEN'] = HF_TOKEN\n",
            "os.environ['HF_REPO_ID'] = HF_REPO_ID\n",
            "# login(token=HF_TOKEN) # Uncomment when you put your token in"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Smart Data Setup (Pull from HF or Extract & Relabel)\n",
            "If the relabeled dataset is already on Hugging Face, we download it directly. Otherwise, we extract the Kaggle input, run the full relabeling pipeline, and push a checkpoint."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os, shutil, yaml\n",
            "from pathlib import Path\n",
            "from huggingface_hub import snapshot_download, HfApi\n",
            "from phase1_dataset.label_hops import run_labeling\n",
            "\n",
            "repo_id = os.environ.get('HF_REPO_ID')\n",
            "working_data_dir = Path('/kaggle/working/data')\n",
            "downloaded_from_hf = False\n",
            "\n",
            "print(f\"Checking if {repo_id} has our data on Hugging Face...\")\n",
            "try:\n",
            "    snapshot_download(repo_id=repo_id, repo_type=\"dataset\", local_dir=\"/kaggle/working\", allow_patterns=\"data/*\")\n",
            "    if (working_data_dir / '2wikimultihopqa' / 'labeled.jsonl').exists():\n",
            "        print(\"Successfully downloaded existing checkpoint from Hugging Face!\")\n",
            "        downloaded_from_hf = True\n",
            "    else:\n",
            "        print(\"Repo exists but labeled.jsonl not found inside.\")\n",
            "except Exception as e:\n",
            "    print(f\"Could not download from Hugging Face: {e}\")\n",
            "\n",
            "if not downloaded_from_hf:\n",
            "    print(\"Falling back to local Kaggle /input zip...\")\n",
            "    input_dir = Path('/kaggle/input')\n",
            "    merged_dirs = list(input_dir.rglob('labeled.jsonl')) + list(input_dir.rglob('generated_cot.jsonl'))\n",
            "    if merged_dirs:\n",
            "        source_data_dir = merged_dirs[0].parent\n",
            "        if not working_data_dir.exists():\n",
            "            shutil.copytree(source_data_dir, working_data_dir)\n",
            "    \n",
            "    with open('configs/data_config.yaml') as f:\n",
            "        cfg = yaml.safe_load(f)\n",
            "    cfg['data']['raw_dir'] = '/kaggle/working/data'\n",
            "    cfg['data']['processed_dir'] = '/kaggle/working/data/processed'\n",
            "    cfg['data']['hidden_states_dir'] = '/kaggle/working/data/hidden_states'\n",
            "    with open('configs/data_config.yaml', 'w') as f:\n",
            "        yaml.dump(cfg, f)\n",
            "        \n",
            "    two_wiki_dir = working_data_dir / '2wikimultihopqa'\n",
            "    two_wiki_dir.mkdir(parents=True, exist_ok=True)\n",
            "    if (working_data_dir / 'generated_cot.jsonl').exists():\n",
            "        shutil.move(str(working_data_dir / 'generated_cot.jsonl'), str(two_wiki_dir / 'generated_cot.jsonl'))\n",
            "    \n",
            "    print('Relabeling dataset from scratch...')\n",
            "    labeled_path = run_labeling(cfg)\n",
            "    print(f'Labeled dataset saved to: {labeled_path}')\n",
            "    \n",
            "    api = HfApi()\n",
            "    if 'YOUR_HF_TOKEN_HERE' not in os.environ.get('HF_TOKEN', ''):\n",
            "        print(f\"Uploading fresh relabeled dataset to {repo_id} as a checkpoint...\")\n",
            "        api.create_repo(repo_id=repo_id, repo_type=\"dataset\", exist_ok=True)\n",
            "        api.upload_folder(folder_path=\"/kaggle/working/data\", repo_id=repo_id, repo_type=\"dataset\", path_in_repo=\"data\")\n",
            "        print(\"Intermediate upload complete!\")\n",
            "    else:\n",
            "        print(\"HF Token not provided. Skipping intermediate upload.\")\n",
            "else:\n",
            "    with open('configs/data_config.yaml') as f:\n",
            "        cfg = yaml.safe_load(f)\n",
            "    cfg['data']['raw_dir'] = '/kaggle/working/data'\n",
            "    cfg['data']['processed_dir'] = '/kaggle/working/data/processed'\n",
            "    cfg['data']['hidden_states_dir'] = '/kaggle/working/data/hidden_states'\n",
            "    with open('configs/data_config.yaml', 'w') as f:\n",
            "        yaml.dump(cfg, f)\n",
            "    print(\"Config updated for downloaded HF dataset.\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Copy Hidden States from Kaggle Zip\n",
            "We unconditionally copy the original hidden states so that they are present before Counterfactual Generation starts."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "# Find hidden_states in Kaggle input\n",
            "input_dir = Path('/kaggle/input')\n",
            "hidden_states_dirs = list(input_dir.rglob('hidden_states'))\n",
            "\n",
            "if hidden_states_dirs:\n",
            "    src_hs = hidden_states_dirs[0]\n",
            "    dest_hs = Path('/kaggle/working/data/hidden_states')\n",
            "    \n",
            "    print(f\"Found hidden states at {src_hs}\")\n",
            "    print(f\"Copying to {dest_hs} (overwriting if needed)...\")\n",
            "    shutil.copytree(src_hs, dest_hs, dirs_exist_ok=True)\n",
            "    print(\"Done copying original hidden states!\")\n",
            "    print(f\"Total .pt files copied: {len(list(dest_hs.glob('*.pt')))}\")\n",
            "    \n",
            "    # Upload original hidden states to Hugging Face IMMEDIATELY so they are safe\n",
            "    if 'YOUR_HF_TOKEN_HERE' not in os.environ.get('HF_TOKEN', ''):\n",
            "        from huggingface_hub import HfApi\n",
            "        api = HfApi()\n",
            "        repo_id = os.environ.get('HF_REPO_ID')\n",
            "        print(f\"Backing up original hidden states to {repo_id} before CF generation...\")\n",
            "        try:\n",
            "            api.upload_folder(folder_path='/kaggle/working/data/hidden_states', repo_id=repo_id, repo_type='dataset', path_in_repo='data/hidden_states')\n",
            "            print(\"Original hidden states safely backed up to Hugging Face!\")\n",
            "        except Exception as e:\n",
            "            print(f\"Could not upload hidden states: {e}\")\n",
            "else:\n",
            "    print(\"Could not find hidden_states in /kaggle/input! Did you include it in your zip?\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Generate Counterfactuals\n",
            "Generate counterfactuals to balance the dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import json\n",
            "from phase1_dataset.counterfactuals import run_counterfactuals\n",
            "from phase1_dataset.generate_cot import load_model_and_tokenizer\n",
            "\n",
            "print(\"Loading model for counterfactual generation...\")\n",
            "model, tokenizer = load_model_and_tokenizer(cfg)\n",
            "\n",
            "print(\"Running counterfactual generation with checkpoints...\")\n",
            "augmented_path = run_counterfactuals(cfg, model=model, tokenizer=tokenizer)\n",
            "\n",
            "with open(augmented_path) as f:\n",
            "    all_ex = [json.loads(l) for l in f]\n",
            "n_orig = sum(1 for e in all_ex if not e.get('is_counterfactual', False))\n",
            "n_cf   = sum(1 for e in all_ex if e.get('is_counterfactual', False))\n",
            "print('Original:', n_orig, '| Counterfactual:', n_cf, '| Total:', len(all_ex))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Layer Probing (Fixing the Data Leak & Per-Hop granularity)\n",
            "We train logistic regression probes to predict hop-level success/failure using the hidden states."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import json, torch, pickle, numpy as np\n",
            "from pathlib import Path\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.pipeline import make_pipeline\n",
            "from sklearn.metrics import accuracy_score\n",
            "from sklearn.model_selection import GroupShuffleSplit\n",
            "\n",
            "hs_dir = Path(cfg['data']['hidden_states_dir'])\n",
            "labels_file = augmented_path # Use the augmented dataset for training probes\n",
            "\n",
            "with open(labels_file) as f:\n",
            "    examples = [json.loads(l) for l in f]\n",
            "\n",
            "pt_files = list(hs_dir.glob('*.pt'))\n",
            "if not pt_files:\n",
            "    print('No .pt files found in', hs_dir)\n",
            "else:\n",
            "    sample = torch.load(pt_files[0])\n",
            "    layer_indices = sample.get('layer_indices', list(range(sample['pooled'].shape[0])))\n",
            "    \n",
            "    # Prepare dataset\n",
            "    valid_ex = []\n",
            "    for ex in examples:\n",
            "        if (hs_dir / (ex['id'] + '.pt')).exists():\n",
            "            valid_ex.append(ex)\n",
            "            \n",
            "    print(f\"Found {len(valid_ex)} examples with hidden states.\")\n",
            "    \n",
            "    # 1. SPLIT BY CLEAN PAIR ID FIRST (Fixes Train/Test Data Leak)\n",
            "    groups = [ex.get('clean_pair_id', ex['id']) for ex in valid_ex]\n",
            "    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
            "    train_idx, test_idx = next(gss.split(valid_ex, groups=groups))\n",
            "    train_ex = [valid_ex[i] for i in train_idx]\n",
            "    test_ex = [valid_ex[i] for i in test_idx]\n",
            "    \n",
            "    def extract_X_y(ex_list, li):\n",
            "        X, y = [], []\n",
            "        for ex in ex_list:\n",
            "            data = torch.load(hs_dir / (ex['id'] + '.pt'))\n",
            "            pooled = data['pooled']\n",
            "            li_list = data.get('layer_indices', list(range(pooled.shape[0])))\n",
            "            if li not in li_list:\n",
            "                continue\n",
            "            idx = li_list.index(li)\n",
            "            for hop in ex.get('hops', []):\n",
            "                hop_idx = hop['hop_idx'] - 1  # 0-indexed for pooled\n",
            "                if hop.get('label') in (0, 1) and hop_idx < pooled.shape[1]:\n",
            "                    X.append(pooled[idx, hop_idx].detach().numpy())\n",
            "                    y.append(hop['label'])\n",
            "        return np.array(X), np.array(y)\n",
            "    \n",
            "    layer_accs = {}\n",
            "    best_clf = None\n",
            "    best_li = None\n",
            "    best_acc = 0\n",
            "    \n",
            "    for li in layer_indices:\n",
            "        X_train, y_train = extract_X_y(train_ex, li)\n",
            "        X_test, y_test = extract_X_y(test_ex, li)\n",
            "        \n",
            "        if len(X_train) == 0 or len(np.unique(y_train)) < 2: \n",
            "            print(f\"Skipping layer {li}: not enough classes.\")\n",
            "            continue\n",
            "        \n",
            "        X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)\n",
            "        # Use StandardScaler and balanced class_weight!\n",
            "        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))\n",
            "        clf.fit(X_train, y_train)\n",
            "        acc = accuracy_score(y_test, clf.predict(X_test))\n",
            "        layer_accs[li] = acc\n",
            "        \n",
            "        if acc > best_acc:\n",
            "            best_acc = acc\n",
            "            best_clf = clf\n",
            "            best_li = li\n",
            "            \n",
            "    print('\\nLayer probe accuracy (on unseen test set):')\n",
            "    for li, acc in sorted(layer_accs.items()):\n",
            "        print('  Layer', str(li).rjust(2), ':', round(acc, 3), chr(0x2588)*int(acc*40))\n",
            "        \n",
            "    print(f'\\n-> Best Layer: {best_li} with {best_acc:.3f} accuracy')\n",
            "    \n",
            "    # Save the best probe\n",
            "    if best_li is not None:\n",
            "        with open(f'/kaggle/working/best_probe_layer_{best_li}.pkl', 'wb') as f:\n",
            "            pickle.dump(best_clf, f)\n",
            "        print(f\"Probe saved to /kaggle/working/best_probe_layer_{best_li}.pkl\")\n",
            "    else:\n",
            "        print(\"No valid layer probes were trained.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Advanced Probing Metrics\n",
            "Run the advanced probing script to extract TF-IDF baselines, natural-only metrics, AUROC/F1, and Hop Index baseline."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "!wget -q -O advanced_probing.py https://raw.githubusercontent.com/ANLPproject/LinguaFranca/anish-dev/advanced_probing.py\n",
            "!python advanced_probing.py /kaggle/working/data/2wikimultihopqa/augmented.jsonl /kaggle/working/data/hidden_states"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Push to Hugging Face\n",
            "Uploads the final dataset and probes to your Hugging Face Hub dataset repository."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "from huggingface_hub import HfApi\n",
            "import glob\n",
            "\n",
            "api = HfApi()\n",
            "repo_id = os.environ.get('HF_REPO_ID')\n",
            "\n",
            "if 'YOUR_HF_TOKEN_HERE' in os.environ.get('HF_TOKEN', ''):\n",
            "    print(\"Please provide your HF Token above to push to Hub.\")\n",
            "else:\n",
            "    print(f\"Creating/verifying dataset repo: {repo_id}\")\n",
            "    api.create_repo(repo_id=repo_id, repo_type=\"dataset\", exist_ok=True)\n",
            "    \n",
            "    print(\"Uploading data folder to Hugging Face...\")\n",
            "    api.upload_folder(\n",
            "        folder_path=\"/kaggle/working/data\",\n",
            "        repo_id=repo_id,\n",
            "        repo_type=\"dataset\",\n",
            "        path_in_repo=\"data\"\n",
            "    )\n",
            "    \n",
            "    for pkl_file in glob.glob('/kaggle/working/*.pkl'):\n",
            "        print(f\"Uploading {pkl_file}...\")\n",
            "        api.upload_file(\n",
            "            path_or_fileobj=pkl_file,\n",
            "            path_in_repo=os.path.basename(pkl_file),\n",
            "            repo_id=repo_id,\n",
            "            repo_type=\"dataset\"\n",
            "        )\n",
            "    \n",
            "    print(\"Upload complete! You can view your dataset at: https://huggingface.co/datasets/\" + repo_id)"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('kaggle_phase3_eval.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Successfully generated kaggle_phase3_eval.ipynb")
