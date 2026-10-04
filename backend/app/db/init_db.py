"""Schema creation and seed-corpus ingestion.

Runs on every boot and is idempotent (ingestion skips unchanged documents by
checksum), which is what lets a free container come up cold with a working,
demo-able corpus and no manual setup step.
"""

from __future__ import annotations

from app.db.base import Base, engine, session_scope
from app.ingestion.pipeline import ingest_seed_corpus, reconcile_vector_store
from app.logging_config import get_logger

log = get_logger(__name__)


def create_schema() -> None:
    Base.metadata.create_all(bind=engine)
    log.info("Database schema ready (%s)", engine.url.render_as_string(hide_password=True))


def seed_corpus() -> int:
    with session_scope() as db:
        results = ingest_seed_corpus(db)
    changed = sum(1 for r in results if r.status != "unchanged")
    log.info(
        "Seed corpus ready: %d documents, %d chunks (%d changed)",
        len(results),
        sum(r.chunks for r in results),
        changed,
    )
    return len(results)


def initialise(seed: bool = True) -> None:
    create_schema()
    if seed:
        seed_corpus()
    # Always - uploaded documents need their vectors too, not just the seed set.
    with session_scope() as db:
        reconcile_vector_store(db)
