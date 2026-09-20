"""Hybrid retrieval: BM25 + FAISS vectors, fused by score and rank together.

Why hybrid: a query like "Q2 2024 revenue growth rate" mixes a rare exact
token (a quarter label, a product name) where lexical matching wins, with a
paraphrase ("how fast did revenue grow" -> "revenue growth rate") where dense
vectors win. Neither leg is reliable alone, and the two scores are on
incomparable scales, so `_fuse` blends a min-max normalised score with a
normalised reciprocal rank from each leg - pure rank fusion (RRF) throws away
a decisive BM25 score margin on a corpus this size; pure score fusion lets one
leg's scale dominate.

A near-duplicate suppression pass (Jaccard over token sets, engaged only
above a similarity threshold) removes the overlapping windows chunking
produces, without demoting genuinely distinct passages the way classic MMR's
smooth penalty would.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk, Document
from app.embeddings import get_embedder, tokenize
from app.retrieval.bm25 import get_index
from app.retrieval.vector_store import get_vector_store

RRF_K = 12
NEAR_DUPLICATE_JACCARD = 0.6


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int | None
    heading: str
    content: str
    score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rank: int = 0

    @property
    def snippet(self) -> str:
        body = self.content.split("]\n", 1)[-1] if self.content.startswith("[") else self.content
        body = " ".join(body.split())
        return body[:420] + ("..." if len(body) > 420 else "")

    def to_citation(self, index: int) -> dict:
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_number": self.page_number,
            "heading": self.heading,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "lexical_score": round(self.lexical_score, 4),
        }


def _fuse(legs: list[tuple[list[tuple[str, float]], float]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for hits, weight in legs:
        if not hits:
            continue
        scores = [s for _, s in hits]
        low, high = min(scores), max(scores)
        span = (high - low) or 1.0
        for rank, (chunk_id, score) in enumerate(hits):
            normalised = (score - low) / span
            reciprocal = (RRF_K + 1) / (RRF_K + rank + 1)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * (
                0.6 * normalised + 0.4 * reciprocal
            )
    return fused


def _near_duplicate_suppress(
    candidates: list[RetrievedChunk], top_k: int, penalty: float = 0.5
) -> list[RetrievedChunk]:
    if len(candidates) <= top_k:
        return candidates
    token_sets = [set(tokenize(c.content)) for c in candidates]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_value = None, -1e9
        for i in remaining:
            redundancy = 0.0
            for j in selected:
                union = token_sets[i] | token_sets[j]
                if union:
                    redundancy = max(redundancy, len(token_sets[i] & token_sets[j]) / len(union))
            excess = max(0.0, redundancy - NEAR_DUPLICATE_JACCARD) / (1.0 - NEAR_DUPLICATE_JACCARD)
            value = candidates[i].score - penalty * excess
            if value > best_value:
                best_idx, best_value = i, value
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)  # type: ignore[arg-type]
    return [candidates[i] for i in selected]


def retrieve(
    db: Session,
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    candidates = candidates or settings.retrieval_candidates

    bm25_index = get_index(db)
    store = get_vector_store()
    if not bm25_index.chunk_ids or store.count == 0:
        return []

    embedder = get_embedder()
    dense_weight = 0.55 if embedder.name == "hashing" else 1.0

    # Dense leg: FAISS returns faiss_id, so resolve to chunk_id via one query.
    dense_hits_raw = store.search(embedder.embed_query(query), candidates)
    faiss_ids = [fid for fid, _ in dense_hits_raw]
    faiss_to_chunk: dict[int, Chunk] = {}
    if faiss_ids:
        rows = db.execute(select(Chunk).where(Chunk.faiss_id.in_(faiss_ids))).scalars()
        faiss_to_chunk = {c.faiss_id: c for c in rows}
    dense_hits = [
        (faiss_to_chunk[fid].id, score) for fid, score in dense_hits_raw if fid in faiss_to_chunk
    ]

    lexical_hits = bm25_index.search(query, candidates)

    fused = _fuse([(dense_hits, dense_weight), (lexical_hits, 1.0)])
    if not fused:
        return []

    # Resolve every fused chunk id to its full row + document title, once.
    chunk_ids = list(fused.keys())
    rows = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()

    dense_by_id = dict(dense_hits)
    lexical_by_id = dict(lexical_hits)
    built: list[RetrievedChunk] = []
    for chunk, document in rows:
        built.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                document_title=document.title,
                page_number=chunk.page_number,
                heading=chunk.heading,
                content=chunk.content,
                dense_score=dense_by_id.get(chunk.id, 0.0),
                lexical_score=lexical_by_id.get(chunk.id, 0.0),
            )
        )

    best = max(fused.values()) or 1.0
    for chunk in built:
        chunk.score = fused.get(chunk.chunk_id, 0.0) / best
    built.sort(key=lambda c: c.score, reverse=True)

    diversified = _near_duplicate_suppress(built[: max(top_k * 3, top_k)], top_k)
    for position, chunk in enumerate(diversified, start=1):
        chunk.rank = position
    return diversified


def build_context_block(chunks: list[RetrievedChunk], token_budget: int = 3200) -> str:
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        body = chunk.content
        cost = max(1, len(body) // 4)
        if used + cost > token_budget and parts:
            break
        used += cost
        page = f"p.{chunk.page_number}" if chunk.page_number else "n/a"
        parts.append(f"[{i}] source={chunk.document_title} | page={page}\n{body}")
    return "\n\n---\n\n".join(parts)
