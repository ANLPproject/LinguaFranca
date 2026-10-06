"""
lf_common.py
────────────
Shared data/split/metric helpers for the Phase 2-4 scripts
(advanced_probing.py, evaluate_saved_probe.py, bootstrap_confident_wrong.py,
evaluate_llm_baseline.py, causal_patching.py).

Everything that decides WHICH rows are evaluated lives here, so the probe and
the LLM judge are always scored on identical (example, hop) rows.

Conventions
───────────
* Dataset: data/canonical/augmented_v2.jsonl (scripts/make_canonical_dataset.py).
* Rows: hops 1 and 2 only (the 2-edge gold graph), label in {0, 1}.
  Label 1 = the hop text does not state that hop's gold bridging entity.
* Split: GroupShuffleSplit(test_size=0.2, random_state=42) grouped by
  clean_pair_id, over examples in file order — the same split the original
  Phase 3 numbers used.
* Hidden-state "layer L" = pooled[L] = output of transformer block L
  (HF hidden_states[L + 1]); layer 0 is NOT the raw embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HF_REPO = "AnishRacherla/LinguaFranca-Phase3"
HIDDEN_STATES_REVISION = "ce07fd04bf3e39d920e3ccbf769298fa4396eb1f"
DEFAULT_DATA = "data/canonical/augmented_v2.jsonl"
GOLD_HOPS = (1, 2)
SPLIT_SEED = 42

HEDGE_KEYWORDS = ["no information", "no mention", "not mentioned", "no relevant information",
                  "not found", "cannot find", "does not mention"]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_examples(path: str | Path = DEFAULT_DATA, hs_dir: str | Path | None = None) -> list[dict]:
    """Load the canonical jsonl.  If hs_dir is given, keep only examples with a .pt file."""
    with open(path, encoding="utf-8") as f:
        ex = [json.loads(l) for l in f if l.strip()]
    if hs_dir is not None:
        hs_dir = Path(hs_dir)
        kept = [e for e in ex if (hs_dir / f"{e['id']}.pt").exists()]
        if len(kept) != len(ex):
            print(f"[lf_common] WARNING: {len(ex) - len(kept)} examples have no hidden-state file in {hs_dir}; "
                  "the split will differ from the canonical one.")
        ex = kept
    return ex


def group_id(e: dict) -> str:
    return e.get("clean_pair_id", e["id"])


def canonical_split(examples: list[dict], seed: int = SPLIT_SEED, test_size: float = 0.2):
    """Return (train_idx, test_idx) arrays over `examples` (grouped by clean_pair_id)."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(examples)), groups=[group_id(e) for e in examples]))
    return tr, te


def is_hedge(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in HEDGE_KEYWORDS)


def hop_rows(examples: list[dict], hops=GOLD_HOPS, require_text: bool = False) -> list[dict]:
    """One dict per labeled gold hop.  `ei` indexes into `examples`."""
    rows = []
    for ei, e in enumerate(examples):
        for h in e.get("hops", []):
            if h.get("label") not in (0, 1) or h.get("hop_idx") not in hops:
                continue
            if require_text and not (h.get("text") or "").strip():
                continue
            rows.append(dict(
                ei=ei, id=e["id"], hop=h["hop_idx"], y=int(h["label"]),
                cf=bool(e.get("is_counterfactual", False)), group=group_id(e),
                text=h.get("text", ""), hedge=is_hedge(h.get("text", "")),
                final_correct=bool(e.get("final_answer_correct", False)),
            ))
    return rows


def rows_arrays(rows: list[dict]) -> dict:
    keys = ("ei", "hop", "y", "cf", "hedge", "final_correct")
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out["group"] = np.array([r["group"] for r in rows])
    return out


def load_hidden(examples: list[dict], rows: list[dict], hs_dir: str | Path, layers: list[int]) -> dict:
    """{layer: float32 [n_rows, d]} for the given rows (pooled hop states)."""
    import torch
    hs_dir = Path(hs_dir)
    by_ex: dict[int, list[int]] = {}
    for ri, r in enumerate(rows):
        by_ex.setdefault(r["ei"], []).append(ri)
    out = None
    for ei, ris in by_ex.items():
        d = torch.load(hs_dir / f"{examples[ei]['id']}.pt", map_location="cpu", weights_only=False)
        pooled = d["pooled"]
        li_list = list(d.get("layer_indices", range(pooled.shape[0])))
        if out is None:
            out = {L: np.zeros((len(rows), pooled.shape[-1]), dtype=np.float32) for L in layers}
        for ri in ris:
            hi = rows[ri]["hop"] - 1
            if hi >= pooled.shape[1]:
                continue
            for L in layers:
                out[L][ri] = pooled[li_list.index(L), hi].float().numpy()
    for L in layers:
        np.nan_to_num(out[L], copy=False)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Models & metrics
# ─────────────────────────────────────────────────────────────────────────────

def make_probe(seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def binary_metrics(y, prob=None, pred=None, threshold: float = 0.5) -> dict:
    """acc, balanced acc, AUROC, AUPRC, TPR (= failure recall), FPR, majority rate, n.

    Always report TPR together with FPR: a detector that flags everything has
    TPR = 1.0 (this is how the old 98 % "confident-wrong" number arose)."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, roc_auc_score)
    y = np.asarray(y).astype(int)
    if pred is None:
        pred = (np.asarray(prob) >= threshold).astype(int)
    pred = np.asarray(pred).astype(int)
    n = len(y)
    out = dict(n=int(n), n_fail=int(y.sum()))
    if n == 0:
        return out
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    both = len(set(y.tolist())) > 1
    out.update(
        acc=float(accuracy_score(y, pred)),
        bacc=float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        tpr=tp / (tp + fn) if tp + fn else float("nan"),
        fpr=fp / (fp + tn) if fp + tn else float("nan"),
        majority=float(max(y.mean(), 1 - y.mean())),
    )
    if prob is not None and both:
        out["auroc"] = float(roc_auc_score(y, prob))
        out["auprc"] = float(average_precision_score(y, prob))
    return out


def grouped_bootstrap_ci(y, prob, groups, metric: str = "auroc", n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05):
    """Percentile CI for a metric, resampling whole groups (examples / clean pairs)."""
    rng = np.random.default_rng(seed)
    y, prob, groups = np.asarray(y), np.asarray(prob), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        m = binary_metrics(y[idx], prob[idx]).get(metric)
        if m is not None and not np.isnan(m):
            vals.append(m)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def fmt_metrics(m: dict, keys=("acc", "bacc", "auroc", "auprc", "tpr", "fpr", "majority")) -> str:
    return "  ".join(f"{k}={m[k]:.3f}" if isinstance(m.get(k), float) else f"{k}=n/a" for k in keys) + f"  n={m.get('n')}"


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (same as utils.matching.normalize)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
