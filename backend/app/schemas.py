"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int | None = None
    heading: str
    snippet: str
    score: float
    dense_score: float
    lexical_score: float


class ContextInjection(BaseModel):
    patterns: list[str]
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    blocked: bool
    block_reason: str
    citations: list[Citation] = Field(default_factory=list)
    used_citations: list[int] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    follow_up_question: str = ""
    groundedness: dict[str, Any] = Field(default_factory=dict)
    context_injections: dict[int, ContextInjection] = Field(default_factory=dict)
    llm_mode: str = "demo"
    latency_ms: int = 0
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    log_id: str = ""


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    doc_type: str
    page_count: int
    chunk_count: int
    created_at: datetime


class IngestResponse(BaseModel):
    document_id: str
    title: str
    filename: str
    chunks: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_mode: str
    llm_model: str
    embedding_provider: str
    rerank_provider: str
    vector_store: str
    documents: int
    chunks: int
    vectors: int


class DashboardResponse(BaseModel):
    queries_total: int
    blocked_total: int
    average_groundedness: float
    low_groundedness_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    average_retrieval_ms: float
    average_rerank_ms: float
    average_generation_ms: float
    total_estimated_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    flag_counts: dict[str, int]
    recent_queries: list[dict[str, Any]]
