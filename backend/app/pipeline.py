"""Top-level query pipeline: retrieve -> rerank -> guardrails -> generate.

    query
      |
      v
    [prompt-injection guardrail: query]  -- match --> blocked, logged, END
      |
      v
    hybrid retrieve (BM25 + FAISS, fused)
      |
      v
    rerank (heuristic or cross-encoder)
      |
      v
    [prompt-injection guardrail: passages]  -- flag only, never blocks
      |
      v
    generate (Claude, or extractive fallback)
      |
      v
    [groundedness guardrail]  -- low score --> prepend a warning banner
      |
      v
    answer + citations + guardrail flags + latency/cost, logged to QueryLog

Every stage is timed and every generation call's tokens are converted to an
estimated cost, which is exactly what the observability dashboard reads.
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import QueryLog
from app.generation.answer import generate_answer
from app.guardrails.groundedness import check_groundedness
from app.guardrails.prompt_injection import scan_passages, scan_query
from app.logging_config import get_logger
from app.metrics.cost import estimate_cost_usd
from app.retrieval.hybrid import build_context_block, retrieve
from app.retrieval.rerank import rerank as rerank_chunks

log = get_logger(__name__)

LOW_GROUNDEDNESS_BANNER = (
    "> ⚠️ **Low confidence** - this answer may not be fully supported by the "
    "retrieved documents. Treat it as a starting point, not a final answer.\n\n"
)


def answer_query(db: Session, query: str, employee_context: str = "") -> dict:
    started = time.perf_counter()
    flags: list[str] = []

    # --- guardrail: prompt injection in the query itself -------------------
    query_scan = scan_query(query)
    if query_scan.matched and settings.block_on_prompt_injection:
        log_row = QueryLog(
            query=query,
            answer="",
            llm_mode="blocked",
            blocked=True,
            block_reason="prompt_injection_in_query",
            flags=[f"prompt_injection:{p}" for p in query_scan.patterns],
            total_ms=int((time.perf_counter() - started) * 1000),
        )
        db.add(log_row)
        db.commit()
        return {
            "answer": (
                "I can't process that request - it looks like an attempt to override "
                "my instructions rather than a genuine question. Please rephrase."
            ),
            "blocked": True,
            "block_reason": "prompt_injection_in_query",
            "citations": [],
            "used_citations": [],
            "flags": ["prompt_injection_in_query"],
            "groundedness_score": 0.0,
            "llm_mode": "blocked",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "log_id": log_row.id,
        }

    # --- retrieval -----------------------------------------------------------
    t0 = time.perf_counter()
    broad_k = min(settings.retrieval_candidates, settings.retrieval_top_k * 3)
    candidates = retrieve(db, query, top_k=broad_k)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # --- rerank ----------------------------------------------------------------
    t0 = time.perf_counter()
    reranked = rerank_chunks(query, candidates, top_k=settings.retrieval_top_k)
    rerank_ms = int((time.perf_counter() - t0) * 1000)

    citations = [c.to_citation(i) for i, c in enumerate(reranked, start=1)]
    retrieved_dicts = [
        {**citation, "content": chunk.content}
        for citation, chunk in zip(citations, reranked, strict=True)
    ]
    context_block = build_context_block(reranked)

    # --- guardrail: prompt injection inside retrieved passages --------------
    passage_hits = scan_passages(retrieved_dicts)
    if passage_hits:
        flags.append("context_injection_detected")

    # --- generation ------------------------------------------------------------
    t0 = time.perf_counter()
    result = generate_answer(query, retrieved_dicts, context_block)
    generation_ms = int((time.perf_counter() - t0) * 1000)

    output = result.value
    valid_indices = {item["index"] for item in retrieved_dicts}
    used_citations = [c for c in output.used_citations if c in valid_indices]

    # --- guardrail: groundedness ---------------------------------------------
    checks = check_groundedness(query, output.answer, retrieved_dicts)
    answer_text = output.answer.strip()
    if checks["invalid_citations"]:
        flags.append("invalid_citation")
    if checks["ungrounded_numbers"]:
        flags.append("ungrounded_numbers")
    if output.insufficient_evidence or not retrieved_dicts:
        flags.append("insufficient_evidence")
    if checks["query_coverage"] < 0.4:
        flags.append("question_not_covered_by_corpus")
    if checks["groundedness_score"] < settings.groundedness_threshold:
        flags.append("low_groundedness_warning")
        answer_text = LOW_GROUNDEDNESS_BANNER + answer_text

    cost = estimate_cost_usd(
        result.model or settings.anthropic_model, result.input_tokens, result.output_tokens
    )
    total_ms = int((time.perf_counter() - started) * 1000)

    log_row = QueryLog(
        query=query,
        answer=answer_text,
        llm_mode=result.mode,
        retrieved_count=len(retrieved_dicts),
        top_score=reranked[0].score if reranked else 0.0,
        citations_used=len(used_citations),
        groundedness_score=checks["groundedness_score"],
        blocked=False,
        block_reason="",
        flags=flags,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost_usd=cost,
    )
    db.add(log_row)
    db.commit()
    db.refresh(log_row)

    return {
        "answer": answer_text,
        "blocked": False,
        "block_reason": "",
        "citations": citations,
        "used_citations": used_citations,
        "flags": flags,
        "follow_up_question": output.follow_up_question,
        "groundedness": checks,
        "context_injections": {
            idx: {"patterns": scan.patterns, "snippet": scan.snippet}
            for idx, scan in passage_hits.items()
        },
        "llm_mode": result.mode,
        "latency_ms": total_ms,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "generation_ms": generation_ms,
        "token_usage": {"input": result.input_tokens, "output": result.output_tokens},
        "estimated_cost_usd": cost,
        "log_id": log_row.id,
    }
