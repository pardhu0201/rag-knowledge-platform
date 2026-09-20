"""In-process BM25 over the chunk table, rebuilt whenever the corpus changes.

A document platform's corpus mutates constantly (uploads, deletes), unlike a
fixed policy corpus, so the index is keyed by a cheap version fingerprint
(row count + latest timestamp) and rebuilt lazily on the first search after a
change rather than on every request.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.embeddings import tokenize
from app.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class BM25Index:
    chunk_ids: list[str]
    doc_tokens: list[list[str]]
    doc_freq: Counter
    avg_len: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, rows: list[Chunk]) -> BM25Index:
        chunk_ids = [c.id for c in rows]
        doc_tokens = [tokenize(c.content) for c in rows]
        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))
        avg_len = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(
            chunk_ids=chunk_ids, doc_tokens=doc_tokens, doc_freq=doc_freq, avg_len=avg_len or 1.0
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunk_ids:
            return []
        n = len(self.chunk_ids)
        scores = np.zeros(n, dtype=np.float32)
        for term in set(q_tokens):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                norm = 1.0 - self.b + self.b * (len(tokens) / self.avg_len)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        order = np.argsort(-scores)[:limit]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]


_cache: dict[str, BM25Index] = {}


def _corpus_version(db: Session) -> str:
    count = db.execute(select(func.count(Chunk.id))).scalar() or 0
    latest = db.execute(select(func.max(Document.created_at))).scalar()
    return f"{count}:{latest}"


def get_index(db: Session) -> BM25Index:
    version = _corpus_version(db)
    cached = _cache.get(version)
    if cached is None:
        _cache.clear()
        rows = list(db.execute(select(Chunk)).scalars())
        cached = BM25Index.build(rows)
        _cache[version] = cached
        log.info("Built BM25 index over %d chunks (version=%s)", len(rows), version)
    return cached


def invalidate() -> None:
    _cache.clear()
