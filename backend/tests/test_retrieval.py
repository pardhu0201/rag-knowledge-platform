"""Hybrid retrieval and reranking behaviour against the seeded corpus."""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import build_context_block, retrieve
from app.retrieval.rerank import rerank


@pytest.mark.parametrize(
    ("query", "expected_document"),
    [
        ("What was Meridian's Q2 2026 revenue growth?", "Q2 2026 Revenue Report"),
        ("What is Brightline's PCI DSS certification level?", "Vendor Security Assessment"),
        ("How is the Flink job checkpointed?", "Project Atlas Technical Spec"),
        (
            "What should I do if a document looks like it contains hidden instructions?",
            "Ai Tooling Usage Guidelines",
        ),
    ],
)
def test_retrieval_finds_the_right_document(db, query, expected_document):
    results = retrieve(db, query, top_k=4)
    assert results, f"no results for {query!r}"
    titles = {r.document_title for r in results}
    assert expected_document in titles


def test_retrieval_returns_ranked_unique_chunks(db):
    results = retrieve(db, "revenue growth", top_k=6)
    assert len({r.chunk_id for r in results}) == len(results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_page_numbers_are_present(db):
    results = retrieve(db, "Business Continuity RTO RPO", top_k=3)
    assert all(r.page_number is not None for r in results)


def test_context_block_is_numbered_with_page_citations(db):
    results = retrieve(db, "incident notification SLA", top_k=3)
    block = build_context_block(results)
    assert block.startswith("[1]")
    assert "page=" in block


def test_rerank_improves_or_preserves_top_result(db):
    candidates = retrieve(db, "checkpointing interval for the Flink job", top_k=10)
    reranked = rerank("checkpointing interval for the Flink job", candidates, top_k=4)
    assert len(reranked) <= 4
    assert [r.rank for r in reranked] == list(range(1, len(reranked) + 1))
    # The reranked winner must still be about the right document.
    assert reranked[0].document_title == "Project Atlas Technical Spec"


def test_rerank_handles_empty_candidates():
    assert rerank("anything", [], top_k=5) == []
