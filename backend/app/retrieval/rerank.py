"""Reranking: a second, finer pass over the hybrid retriever's candidates.

Hybrid fusion is a good coarse filter but a poor final ranker - it never looks
at the query and a chunk *together*, only at each independently. Reranking
closes that gap.

``heuristic`` (default, zero-dependency): scores each candidate on term
coverage, exact phrase/bigram matches, and how close together the query's
terms appear in the text (proximity) - a cheap proxy for "this passage is
actually about the query" that a bag-of-words score cannot express.

``cross-encoder`` (opt-in via ``RERANK_PROVIDER``): a real
``sentence-transformers`` cross-encoder that scores (query, passage) pairs
jointly - meaningfully better, at the cost of a heavier dependency and slower
inference, so it stays opt-in rather than the default.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.embeddings import tokenize
from app.logging_config import get_logger
from app.retrieval.hybrid import RetrievedChunk

log = get_logger(__name__)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "is",
    "are",
    "do",
    "does",
    "did",
    "how",
    "what",
    "when",
    "where",
    "who",
    "why",
    "can",
    "i",
    "my",
    "me",
    "we",
    "you",
    "your",
    "was",
    "were",
    "be",
    "with",
}


def _heuristic_score(query: str, chunk: RetrievedChunk) -> float:
    q_tokens = [t for t in tokenize(query) if t not in _STOPWORDS]
    if not q_tokens:
        return chunk.score

    body = chunk.content.split("]\n", 1)[-1] if chunk.content.startswith("[") else chunk.content
    body_tokens = tokenize(body)
    body_set = set(body_tokens)
    q_set = set(q_tokens)

    coverage = len(q_set & body_set) / len(q_set)

    # Exact bigram matches reward passages that use the query's own phrasing,
    # not just its individual words.
    q_bigrams = {(a, b) for a, b in zip(q_tokens, q_tokens[1:], strict=False)}
    body_bigrams = {(a, b) for a, b in zip(body_tokens, body_tokens[1:], strict=False)}
    bigram_hits = len(q_bigrams & body_bigrams) / max(1, len(q_bigrams))

    # Proximity: the smallest window of body tokens that contains every
    # matched query term, normalised so a tightly-clustered match scores near
    # 1.0 and a match scattered across a long chunk scores near 0.
    positions = [i for i, t in enumerate(body_tokens) if t in q_set]
    if len(positions) >= 2:
        window = positions[-1] - positions[0] + 1
        proximity = min(1.0, len(positions) * 6 / window)
    else:
        proximity = 0.5 if positions else 0.0

    length_penalty = 1.0 / (1.0 + abs(len(body) - 400) / 800)

    return 0.45 * coverage + 0.25 * bigram_hits + 0.20 * proximity + 0.10 * length_penalty


@lru_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score and re-order candidates; returns the top `top_k`."""
    if not chunks:
        return []

    provider = settings.rerank_provider.strip().lower()
    if provider in {"cross-encoder", "cross_encoder", "crossencoder"}:
        try:
            model = _cross_encoder()
            pairs = [(query, c.content) for c in chunks]
            scores = model.predict(pairs)
            ordered = sorted(zip(chunks, scores, strict=True), key=lambda p: -p[1])
            result = [c for c, _ in ordered[:top_k]]
            for position, chunk in enumerate(result, start=1):
                chunk.rank = position
            return result
        except Exception as exc:
            log.warning("Cross-encoder rerank unavailable (%s); using heuristic", exc)

    scored = [(chunk, _heuristic_score(query, chunk)) for chunk in chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    result = [chunk for chunk, _ in scored[:top_k]]
    for position, chunk in enumerate(result, start=1):
        chunk.rank = position
    return result
