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


def test_lost_vector_index_is_rebuilt_on_boot(db):
    """The FAISS file is lost while the database survives (corrupt file,
    dimension-mismatch rebuild). Seeding skips unchanged documents, so without
    reconciliation dense search stayed empty and every query returned nothing.
    """
    from app.db.init_db import initialise
    from app.retrieval.vector_store import get_vector_store

    store = get_vector_store()
    before = store.count
    assert before > 0
    store.reset()  # simulate the lost index

    # Even before any repair, lexical search keeps answering.
    assert retrieve(db, "Q2 2026 revenue growth"), "empty vector index must not disable BM25"

    initialise(seed=True)  # a reboot
    assert store.count == before
    assert retrieve(db, "Q2 2026 revenue growth")


def test_corpus_version_changes_on_in_place_edit(db):
    from app.ingestion.pipeline import delete_document, ingest_bytes
    from app.retrieval.bm25 import _corpus_version

    v0 = _corpus_version(db)
    doc = ingest_bytes(
        db, data=b"Bike racks are in the basement level two garage.", filename="bikes.txt"
    )
    v1 = _corpus_version(db)
    ingest_bytes(
        db,
        data=b"Bike racks are on the rooftop terrace near the cafe.",
        filename="bikes.txt",
        replace=True,
    )
    v2 = _corpus_version(db)
    delete_document(db, doc.document_id)
    assert len({v0, v1, v2}) == 3
    assert _corpus_version(db) == v0


def test_failed_commit_leaves_no_orphan_vectors(db, monkeypatch):
    from app.ingestion.pipeline import ingest_bytes
    from app.retrieval.vector_store import get_vector_store

    store = get_vector_store()
    before = store.count

    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "commit", boom)
    with pytest.raises(RuntimeError):
        ingest_bytes(
            db, data=b"This upload never commits to the database at all.", filename="ghost.txt"
        )
    assert store.count == before
