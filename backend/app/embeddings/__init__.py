"""Pluggable embedding backends.

``hashing`` (default): a dependency-free hashed bag-of-features encoder
(stemmed words + bigrams + character 4-grams, sub-linear term frequency,
feature-family weighting, L2 normalised). Costs nothing, needs no model
download, deterministic - which is what keeps the public demo free.

``sentence-transformers`` (opt-in via ``EMBEDDING_PROVIDER``): real dense
embeddings for when retrieval quality matters more than install size.

Both satisfy the same interface, so FAISS indexing and the hybrid retriever
are unaware of which one is active.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Light suffix normalisation so inflections match ("expenses" ~ "expense")."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, stemmed word tokenizer shared by the embedder and BM25."""
    return [stem(w) for w in _WORD_RE.findall(text.lower())]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing over words, bigrams and character 4-grams."""

    name = "hashing"

    # Character n-grams vastly outnumber words, so without down-weighting
    # they swamp word identity and every document ends up looking alike.
    FEATURE_WEIGHTS = {"w": 1.0, "b": 0.7, "c": 0.3}

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> Counter[str]:
        words = tokenize(text)
        feats: Counter[str] = Counter()
        for w in words:
            feats[f"w:{w}"] += 1
            padded = f"^{w}$"
            for i in range(len(padded) - 3):
                feats[f"c:{padded[i : i + 4]}"] += 1
        for a, b in zip(words, words[1:], strict=False):
            feats[f"b:{a}_{b}"] += 1
        return feats

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % dim, sign

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, count in self._features(text).items():
            idx, sign = self._bucket(feature, self.dim)
            weight = self.FEATURE_WEIGHTS.get(feature[0], 1.0)
            vec[idx] += sign * weight * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_one(text)


class SentenceTransformerEmbedder:
    """Dense transformer embeddings - opt-in, requires `sentence-transformers`."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"sentence-transformers", "st", "minilm"}:
        try:
            embedder = SentenceTransformerEmbedder()
            log.info("Embeddings: sentence-transformers (dim=%d)", embedder.dim)
            return embedder
        except Exception as exc:
            log.warning("sentence-transformers unavailable (%s); falling back to hashing", exc)
    embedder = HashingEmbedder(dim=settings.embedding_dim)
    log.info("Embeddings: hashing (dim=%d)", embedder.dim)
    return embedder
