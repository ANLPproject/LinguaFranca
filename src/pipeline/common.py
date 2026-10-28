"""
src/pipeline/common.py
──────────────────────
Settings, data loading, the canonical split, prompt surgery and answer metrics
shared by every pipeline stage.

Conventions (identical to fixes-v2 lf_common.py so numbers line up)
───────────────────────────────────────────────────────────────────
* Rows: hops 1 and 2 only (2WikiMultihopQA compositional = 2 gold edges).
  Hops 3+ have no gold entity and are excluded (the "extra hop" rule).
  Label 1 = the hop text does not state that hop's gold bridging entity.
* Split: GroupShuffleSplit(test_size=0.2, random_state=42), grouped by
  clean_pair_id, over the examples that have a hidden-state file, in file order.
* "Layer L" = output of transformer block L = HF hidden_states[L + 1].
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "AnishRacherla/LinguaFranca-Phase3"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_MIRRORS = ["unsloth/Llama-3.2-3B-Instruct"]      # same weights, ungated
# Ungated mirrors (same weights) for every gated model the pipeline may load
MIRRORS = {MODEL_NAME: MODEL_MIRRORS,
           "meta-llama/Llama-3.1-8B-Instruct": ["unsloth/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]}
# The off-the-shelf judges of the standalone judge baseline (outputs/llm_judge), rerun inside the pipeline
EXTRA_JUDGES = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
GOLD_HOPS = (1, 2)
SPLIT_SEED = 42

# Raw 2WikiMultihopQA train split (the pool Phase 1 sampled from) — used to
# recover all 10 paragraphs per question for the retrieval corpus.
RAW_2WIKI_FILE = "tfidf_distractor_rag/data/raw/2wikimultihopqa/train.jsonl"

SETTINGS = {
    # Main setting: a weak first-stage retriever (TF-IDF top-5 of the 10 2Wiki
    # passages).  Both gold passages survive only ~40 % of the time, so a
    # failing hop is often caused by missing evidence that retrieval can supply.
    "tfidf": dict(
        data_file="tfidf_distractor_rag/data/raw/2wikimultihopqa/augmented.jsonl",
        hs_prefix="tfidf_distractor_rag/data/hidden_states",
        revision=None,
        context_top_k=5,
    ),
    # Original Phase 1 setting: all 10 passages (2 gold + 8 distractors), pinned
    # to the pre-overwrite revision whose hidden states match the labels.
    "full10": dict(
        data_file="data/2wikimultihopqa/augmented.jsonl",
        hs_prefix="data/hidden_states",
        revision="ce07fd04bf3e39d920e3ccbf769298fa4396eb1f",
        context_top_k=10,
    ),
}

HEDGE_KEYWORDS = ["no information", "no mention", "not mentioned", "no relevant information",
                  "not found", "cannot find", "does not mention"]

_HOP_RE = {k: re.compile(rf"<hop{k}>(.*?)</hop{k}>", re.DOTALL | re.IGNORECASE) for k in range(1, 20)}
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"^Title: (.*)$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path, recs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hf_file(path: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, path, repo_type="dataset", revision=revision)


def hidden_state_ids(setting: str) -> set[str]:
    """Ids that have a hidden-state file on HF (without downloading them)."""
    from huggingface_hub import HfApi
    s = SETTINGS[setting]
    files = HfApi().list_repo_files(HF_REPO, repo_type="dataset", revision=s["revision"])
    pre = s["hs_prefix"] + "/"
    return {f[len(pre):-3] for f in files if f.startswith(pre) and f.endswith(".pt")}


def download_hidden_states(setting: str, ids=None, local_dir: str | None = None) -> Path:
    """Download hidden-state .pt files (all, or only `ids`).  Returns the directory holding them."""
    from huggingface_hub import snapshot_download
    s = SETTINGS[setting]
    patterns = [f"{s['hs_prefix']}/*"] if ids is None else [f"{s['hs_prefix']}/{i}.pt" for i in ids]
    root = snapshot_download(HF_REPO, repo_type="dataset", revision=s["revision"],
                             allow_patterns=patterns, local_dir=local_dir, max_workers=16)
    return Path(root) / s["hs_prefix"]


def load_setting_examples(setting: str, hs_ids: set[str] | None = None) -> list[dict]:
    """The setting's labeled examples, restricted to those with hidden states, in file order."""
    s = SETTINGS[setting]
    ex = read_jsonl(hf_file(s["data_file"], s["revision"]))
    if hs_ids is not None:
        ex = [e for e in ex if e["id"] in hs_ids]
    for e in ex:
        apply_extra_hop_rule(e)
    return ex


# ─────────────────────────────────────────────────────────────────────────────
# Labels and split
# ─────────────────────────────────────────────────────────────────────────────

def apply_extra_hop_rule(example: dict) -> dict:
    """Hops beyond the gold graph get label -1; first_fail_hop is recomputed over hops 1..n_gold."""
    n_gold = len(example.get("reasoning_graph") or [])
    for h in example.get("hops", []):
        if h.get("hop_idx", 0) > n_gold:
            h["label"] = -1
    fails = [h["hop_idx"] for h in example.get("hops", []) if h.get("label") == 1]
    example["first_fail_hop"] = min(fails) if fails else None
    return example


def hop_label(example: dict, k: int):
    for h in example.get("hops", []):
        if h.get("hop_idx") == k and h.get("label") in (0, 1):
            return int(h["label"])
    return None


def group_id(e: dict) -> str:
    return e.get("clean_pair_id", e["id"])


def canonical_split(examples: list[dict], seed: int = SPLIT_SEED, test_size: float = 0.2):
    """(train_idx, test_idx) over `examples`, grouped by clean_pair_id."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(examples)), groups=[group_id(e) for e in examples]))
    return tr, te


def is_hedge(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in HEDGE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing
# ─────────────────────────────────────────────────────────────────────────────

def hop_match(text: str, k: int):
    """The regex match for <hopk>…</hopk> (group 2 = content, as in Phase 1), or None."""
    return _HOP_RE[k].search(text or "")


def hop_text(text: str, k: int) -> str:
    m = hop_match(text, k)
    return m.group(1).strip() if m else ""


_CAPS_RE = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+(?:of|de|da|von|van|the|del|la|[A-Z][\w'’.-]*))*")


def bridge_entity(hop1: str) -> str:
    """
    The entity a hop-1 step resolves to, e.g. "… the director of film X is Arcady Boytler." -> "Arcady Boytler".
    Takes what follows the last " is / was / are / were "; falls back to the last capitalised phrase.
    Returns "" for hedging steps ("there is no information about …").
    """
    t = (hop1 or "").strip()
    if not t or is_hedge(t):
        return ""
    t = re.sub(r"^From the context, I find that\s+", "", t)
    parts = re.split(r"\s(?:is|was|are|were)\s", t)
    tail = parts[-1] if len(parts) > 1 else ""
    if re.match(r"(?:the|a|an)\s", tail):                      # "X is the director of Y": X is the entity
        head = re.split(r"\s(?:is|was|are|were)\s", t)[0]
        head = re.sub(r"^(?:the\s)?[^A-Z]*", "", head).strip(" ,.")
        if head and len(head) <= 80:
            return head
    tail = re.split(r"[,;(]|\s(?:and|who|which|born)\s", tail)[0].strip().strip(".\"'“”‘’ ")
    if tail and len(tail) <= 80 and tail[:1].isupper() and tail.split()[0] not in ("American", "British", "German",
                                                                                  "French", "Italian", "Indian"):
        return tail
    caps = [m.group(0).strip(". ") for m in _CAPS_RE.finditer(t) if m.group(0) not in ("From", "I")]
    return caps[-1] if caps else ""


def repair_query(question: str, hops: dict, k: int, repair: str = "v2") -> str:
    """
    Retrieval query for a restart triggered at hop k (k = 0: retrieve before generating).
      v1: question + failing hop text
      v2: as v1 for hop 1; for hop 2 the hop-1 bridge entity leads the query
          ("Arcady Boytler. Where was the director of film X born? <hop-2 text>"),
          because a failing hop-2 step is often a hedge that does not name what to look up.
    hops: {"1": hop-1 text, "2": hop-2 text}.
    """
    if k == 0:
        return question
    h = (hops.get(str(k)) or "").strip()
    if repair == "v2" and k == 2:
        lead = bridge_entity(hops.get("1", "")) or (hops.get("1") or "").strip()
        return " ".join(x for x in (f"{lead}." if lead else "", question, h) if x)
    return f"{question} {h}".strip()


def gold_titles_in_hop_order(example: dict):
    """
    (hop-1 title, hop-2 title) of a 2Wiki example, or None if it cannot be resolved.

    The stored `gold_titles` / `retrieval_condition` of the TF-IDF data were generated before the
    hop-order fix (commit 687b983) and are in ALPHABETICAL order, so `only_p1` / `only_p2` do not
    mean hop 1 / hop 2.  Here the hop-2 paragraph is the one about the bridge entity (the gold
    entity of hop 1, e.g. "Wallace Fox"); the hop-1 paragraph is the other one (e.g. "The Great Mike").
    Fallback: the hop-1 paragraph is the gold title that appears in the question.
    """
    g = list(dict.fromkeys(example.get("gold_titles") or []))
    graph = example.get("reasoning_graph") or []
    if len(g) != 2 or not graph:
        return None
    base = lambda t: normalize_answer(re.sub(r"\s*\(.*?\)\s*", " ", t))  # noqa: E731  "X (director)" -> "x"
    bridge = normalize_answer(graph[0].get("gold_entity", ""))
    hit = [t for t in g if base(t) and bridge and (base(t) == bridge or base(t) in bridge or bridge in base(t))]
    if len(hit) == 1:
        return (g[1] if hit[0] == g[0] else g[0]), hit[0]
    q = normalize_answer(example.get("question", ""))
    in_q = [t for t in g if base(t) and base(t) in q]
    if len(in_q) == 1:
        return in_q[0], (g[1] if in_q[0] == g[0] else g[0])
    return None


def hop_order_condition(example: dict, titles_in_prompt) -> str:
    """First-stage retrieval condition in true hop order."""
    order = gold_titles_in_hop_order(example)
    if order is None:
        return "unresolved"
    present = set(titles_in_prompt)
    return {(True, True): "both_present", (True, False): "only_hop1_paragraph",
            (False, True): "only_hop2_paragraph", (False, False): "neither"}[(order[0] in present, order[1] in present)]


def parse_answer(text: str) -> str:
    m = _ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def context_titles(context: str) -> list[str]:
    return [t.strip() for t in _TITLE_RE.findall(context or "")]


def format_passages(passages) -> str:
    return "".join(f"Title: {p['title']}\n{p['text']}\n\n" for p in passages).strip()


_USER_HEADER = "<|start_header_id|>user<|end_header_id|>"


def _user_context_start(prompt: str) -> int:
    """Index of the user turn's "Context:\n" (the system prompt's few-shot example has one too)."""
    u = prompt.index(_USER_HEADER)
    return prompt.index("Context:\n", u)


def prompt_from_template(template_prompt: str, context: str, question: str) -> str:
    """
    Build a prompt with the exact system prompt / chat template / date of a stored
    Phase 1 prompt, swapping in a new context and question.  (Re-applying the chat
    template today would change the "Today Date" line, a known confound.)
    """
    i = _user_context_start(template_prompt)
    j = template_prompt.rindex("\n\nQuestion: ")
    k = template_prompt.index("<|eot_id|>", j)
    return template_prompt[:i] + "Context:\n" + context + "\n\nQuestion: " + question + template_prompt[k:]


def prompt_with_evidence(prompt: str, passages) -> str:
    """Naive Restart: prepend retrieved passages to the original context."""
    if not passages:
        return prompt
    i = _user_context_start(prompt) + len("Context:\n")
    return prompt[:i] + format_passages(passages) + "\n\n" + prompt[i:]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF first-stage retriever (same as experiment/distractor-rag generate_cot.py)
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_rank(question: str, passage_texts: list[str], top_k: int) -> list[int]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not passage_texts:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(passage_texts + [question])
        scores = cosine_similarity(mat[-1], mat[:-1])[0]
        return np.argsort(scores)[::-1][:top_k].tolist()
    except ValueError:
        return list(range(min(top_k, len(passage_texts))))


def raw_passages(rec: dict) -> list[dict]:
    """All (title, text) paragraphs of a raw 2Wiki / HotpotQA record (list or HF columnar format)."""
    ctx = rec.get("context", [])
    if isinstance(ctx, dict):
        ctx = list(zip(ctx.get("title", []), ctx.get("sentences", [])))
    out = []
    for c in ctx:
        if isinstance(c, (list, tuple)):
            t, sents = (c[0] if c else ""), (c[1] if len(c) > 1 else [])
        elif isinstance(c, dict):
            t, sents = c.get("title") or "", c.get("sentences") or []
        else:
            continue
        if isinstance(sents, (list, tuple)):
            sents = " ".join(str(s).strip() for s in sents)
        out.append({"title": str(t), "text": str(sents)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Answer metrics (SQuAD / HotpotQA normalisation)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def answer_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_correct(pred: str, gold: str) -> bool:
    """Phase 1 convention: correct iff token F1 >= 0.5."""
    return answer_f1(pred, gold) >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Detection metrics
# ─────────────────────────────────────────────────────────────────────────────

def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def detection_metrics(y, flag, score=None) -> dict:
    """Precision / recall / F0.5 / FPR (failure = positive class), plus AUROC if scores are given."""
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(int)
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = dict(n=int(len(y)), n_fail=int(y.sum()), precision=prec, recall=rec, f05=fbeta(prec, rec),
               fpr=fp / (fp + tn) if fp + tn else float("nan"), flag_rate=float(flag.mean()) if len(flag) else 0.0)
    if score is not None and len(set(y.tolist())) > 1:
        from sklearn.metrics import roc_auc_score
        s = np.asarray(score, dtype=float)
        ok = np.isfinite(s)
        out["auroc"] = float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok].tolist())) > 1 else float("nan")
    return out


def best_f05_threshold(y, prob) -> tuple[float, float]:
    """Threshold maximising F0.5 over the observed scores (ties -> higher threshold)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    best_t, best_f = 0.5, -1.0
    for t in np.unique(prob):
        flag = prob >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = fbeta(p, r)
        if f > best_f or (f == best_f and t > best_t):
            best_t, best_f = float(t), f
    return best_t, best_f
