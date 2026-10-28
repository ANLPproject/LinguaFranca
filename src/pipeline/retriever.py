"""
src/pipeline/retriever.py
─────────────────────────
Second-stage retriever used by every restart: all-MiniLM-L6-v2 embeddings
in a FAISS inner-product index (cosine on normalised vectors) over a pooled
Wikipedia-paragraph corpus:

  * all 10 paragraphs of every question in the setting's dataset
    (so a gold passage the first-stage TF-IDF retriever dropped is findable),
  * all 10 paragraphs of every HotpotQA evaluation question,
  * all paragraphs of `n_extra` further random 2Wiki train questions
    (hard same-distribution distractors).

Paragraphs are de-duplicated by title.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from src.pipeline.common import RAW_2WIKI_FILE, hf_file, raw_passages

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # repair v2: re-orders the dense top-20


def build_corpus(dataset_ids: set[str], hotpot_records: list[dict], n_extra: int = 20000,
                 seed: int = 42, raw_path: str | None = None) -> list[dict]:
    raw_path = raw_path or hf_file(RAW_2WIKI_FILE)
    rng = random.Random(seed)
    seen, corpus = set(), []

    def add(passages, source):
        for p in passages:
            if p["title"] and p["title"] not in seen and p["text"].strip():
                seen.add(p["title"])
                corpus.append(dict(title=p["title"], text=p["text"], source=source))

    extra_pool, n_other, n_hit = [], 0, 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rid = rec.get("_id") or rec.get("id")
            if rid in dataset_ids:
                add(raw_passages(rec), "2wiki_dataset")
                n_hit += 1
                continue
            # uniform reservoir sample of the other questions
            if n_other < n_extra:
                extra_pool.append(rec)
            else:
                j = rng.randint(0, n_other)
                if j < n_extra:
                    extra_pool[j] = rec
            n_other += 1
    for rec in hotpot_records:
        add(rec["all_passages"], "hotpotqa")
    for rec in extra_pool:
        add(raw_passages(rec), "2wiki_extra")
    print(f"[corpus] {len(corpus)} unique paragraphs (dataset questions found in raw file: {n_hit}/{len(dataset_ids)})")
    return corpus


class DenseRetriever:
    def __init__(self, passages: list[dict], embeddings: np.ndarray | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.passages = passages
        self.encoder = SentenceTransformer(EMBED_MODEL, device=device)
        if embeddings is None:
            t0 = time.time()
            embeddings = self.encoder.encode([f"{p['title']}: {p['text']}" for p in passages], batch_size=256,
                                             convert_to_numpy=True, normalize_embeddings=True,
                                             show_progress_bar=True).astype(np.float32)
            print(f"[retriever] embedded {len(passages)} paragraphs in {time.time() - t0:.0f}s")
        self.emb = embeddings
        try:
            import faiss
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        except ImportError:
            print("[retriever] faiss not installed -> exact numpy search (same results)")
            self.index = None

    def save(self, d) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.emb)
        with open(d / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, d, device: str | None = None) -> "DenseRetriever":
        d = Path(d)
        with open(d / "passages.jsonl", encoding="utf-8") as f:
            passages = [json.loads(l) for l in f]
        return cls(passages, np.load(d / "embeddings.npy"), device)

    def search(self, query: str, k: int = 3, exclude_titles=(), rerank_pool: int = 0,
               hybrid: bool = False) -> tuple[list[dict], float]:
        """
        Top-k passages not already in the context.  Returns (passages, latency_ms).
        hybrid: fuse the dense ranking with a BM25 keyword ranking (reciprocal-rank fusion), which
                finds exact names ("William Franklyn") that embedding search confuses with similar ones.
        rerank_pool > 0: take that many candidates and re-order them with a cross-encoder first.
        """
        t0 = time.perf_counter()
        q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        excl = set(exclude_titles)
        n = max(k, rerank_pool, 50 if hybrid else 0) + len(excl) + 5
        if self.index is not None:
            _, idx = self.index.search(q, n)
            idx = idx[0]
        else:
            idx = np.argsort(-(self.emb @ q[0]))[:n]
        dense = [int(i) for i in idx if i >= 0 and self.passages[i]["title"] not in excl]
        if hybrid:
            scores = self._bm25().scores(query)
            top = np.argpartition(-scores, n)[:n]
            lexical = [int(i) for i in top[np.argsort(-scores[top])] if self.passages[i]["title"] not in excl]
            fused: dict[int, float] = {}
            for ranking in (dense[:50], lexical[:50]):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
            dense = sorted(fused, key=fused.get, reverse=True)
        hits = [self.passages[i] for i in dense]
        if rerank_pool > 0:
            pool = hits[:rerank_pool]
            scores = self._reranker().predict([(query, f"{p['title']}: {p['text']}") for p in pool],
                                              show_progress_bar=False)
            hits = [pool[i] for i in np.argsort(-np.asarray(scores))]
        return hits[:k], (time.perf_counter() - t0) * 1000

    def _reranker(self):
        if getattr(self, "_ce", None) is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(RERANK_MODEL, device=str(self.encoder.device))
        return self._ce

    def _bm25(self):
        if getattr(self, "_lex", None) is None:
            self._lex = BM25([f"{p['title']}: {p['text']}" for p in self.passages])
        return self._lex


class BM25:
    """Okapi BM25 (k1 = 1.2, b = 0.75) as a sparse matrix: scores = W[:, query terms] @ idf."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        t0 = time.time()
        self.cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words="english")
        tf = self.cv.fit_transform(docs).astype(np.float32).tocsr()
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf.data = tf.data * (k1 + 1) / (tf.data + np.repeat(norm, np.diff(tf.indptr)))
        self.w = tf.tocsc()
        print(f"[retriever] BM25 index over {n_docs} paragraphs in {time.time() - t0:.0f}s")

    def scores(self, query: str) -> np.ndarray:
        terms = self.cv.transform([query]).indices
        if len(terms) == 0:
            return np.zeros(self.w.shape[0], dtype=np.float32)
        return np.asarray(self.w[:, terms] @ self.idf[terms]).ravel()
