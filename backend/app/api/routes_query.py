"""Query endpoint - the whole retrieve/rerank/guardrail/generate pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.pipeline import answer_query
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_session)) -> QueryResponse:
    result = answer_query(db, payload.query)
    return QueryResponse(**result)
