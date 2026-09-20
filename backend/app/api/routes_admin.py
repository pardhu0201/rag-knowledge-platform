"""Health and the observability dashboard the frontend reads."""

from __future__ import annotations

from collections import Counter

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__
from app.config import settings
from app.db.base import get_session
from app.db.models import Chunk, Document, QueryLog
from app.embeddings import get_embedder
from app.generation.llm_client import get_llm
from app.retrieval.vector_store import get_vector_store
from app.schemas import DashboardResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_session)):
    documents = db.execute(select(func.count(Document.id))).scalar() or 0
    chunks = db.execute(select(func.count(Chunk.id))).scalar() or 0
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_mode=get_llm().mode,
        llm_model=settings.anthropic_model if get_llm().available else "deterministic-fallback",
        embedding_provider=get_embedder().name,
        rerank_provider=settings.rerank_provider,
        vector_store="faiss",
        documents=documents,
        chunks=chunks,
        vectors=get_vector_store().count,
    )


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(limit: int = 500, db: Session = Depends(get_session)):
    rows = list(
        db.execute(select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)).scalars()
    )
    if not rows:
        return DashboardResponse(
            queries_total=0,
            blocked_total=0,
            average_groundedness=0.0,
            low_groundedness_rate=0.0,
            average_latency_ms=0.0,
            p95_latency_ms=0.0,
            average_retrieval_ms=0.0,
            average_rerank_ms=0.0,
            average_generation_ms=0.0,
            total_estimated_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            flag_counts={},
            recent_queries=[],
        )

    answered = [r for r in rows if not r.blocked]
    latencies = [r.total_ms for r in answered] or [0]
    groundedness = [r.groundedness_score for r in answered] or [0.0]

    flag_counter: Counter = Counter()
    for r in rows:
        flag_counter.update(r.flags or [])
        if r.blocked:
            flag_counter[f"blocked:{r.block_reason}"] += 1

    return DashboardResponse(
        queries_total=len(rows),
        blocked_total=sum(1 for r in rows if r.blocked),
        average_groundedness=round(sum(groundedness) / len(groundedness), 3),
        low_groundedness_rate=round(
            sum(1 for g in groundedness if g < settings.groundedness_threshold) / len(groundedness),
            3,
        ),
        average_latency_ms=round(sum(latencies) / len(latencies), 1),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 1),
        average_retrieval_ms=round(
            sum(r.retrieval_ms for r in answered) / max(1, len(answered)), 1
        ),
        average_rerank_ms=round(sum(r.rerank_ms for r in answered) / max(1, len(answered)), 1),
        average_generation_ms=round(
            sum(r.generation_ms for r in answered) / max(1, len(answered)), 1
        ),
        total_estimated_cost_usd=round(sum(r.estimated_cost_usd for r in rows), 4),
        total_input_tokens=sum(r.input_tokens for r in rows),
        total_output_tokens=sum(r.output_tokens for r in rows),
        flag_counts=dict(flag_counter),
        recent_queries=[
            {
                "id": r.id,
                "query": r.query[:140],
                "llm_mode": r.llm_mode,
                "blocked": r.blocked,
                "groundedness_score": r.groundedness_score,
                "total_ms": r.total_ms,
                "estimated_cost_usd": r.estimated_cost_usd,
                "flags": r.flags,
                "created_at": r.created_at,
            }
            for r in rows[:20]
        ],
    )
