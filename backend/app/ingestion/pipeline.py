"""Ingestion: extract -> chunk -> embed -> FAISS + SQLite, idempotently.

A document is keyed by its filename and skipped when the content checksum is
unchanged, so re-running the seeder on every container boot never duplicates
the corpus. `faiss_id` is derived deterministically from the chunk's own uuid
(via a 63-bit hash) rather than an auto-increment counter, so it survives a
process restart without needing a separate persisted counter file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import SEED_CORPUS_DIR
from app.db.models import Chunk, Document
from app.embeddings import get_embedder
from app.ingestion.chunking import chunk_document
from app.ingestion.extract import SUPPORTED_SUFFIXES, extract
from app.logging_config import get_logger
from app.retrieval.bm25 import invalidate as invalidate_bm25
from app.retrieval.vector_store import get_vector_store

log = get_logger(__name__)


@dataclass
class IngestResult:
    document_id: str
    title: str
    filename: str
    chunks: int
    status: str  # created | updated | unchanged


def _checksum(text: str) -> str:
    """Content hash, salted with the embedder's identity.

    Skipping re-ingestion on an unchanged checksum means switching embedder
    would otherwise silently leave stale vectors in the index, so the
    embedder's name+dim is folded into the key.
    """
    embedder = get_embedder()
    fingerprint = f"{text}\x00{embedder.name}:{embedder.dim}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _faiss_id_for(chunk_id: str) -> int:
    """Deterministic 63-bit int id derived from a chunk's uuid hex."""
    digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


class DocumentConflict(ValueError):
    """A *different* document already exists under this filename."""


def ingest_bytes(
    db: Session,
    *,
    data: bytes,
    filename: str,
    title: str | None = None,
    replace: bool = False,
) -> IngestResult:
    """Ingest one file.

    Documents are keyed by filename. Re-uploading identical content is a
    no-op; uploading *different* content under an existing name raises
    `DocumentConflict` unless `replace=True` - otherwise one user's
    "report.pdf" would silently overwrite another's.
    """
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}")

    extracted = extract(data, suffix)
    full_text = extracted.full_text
    if not full_text.strip():
        raise ValueError("No extractable text in that file")

    checksum = _checksum(full_text)
    display_title = title or filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()

    document = db.execute(
        select(Document).where(Document.filename == filename)
    ).scalar_one_or_none()
    if document is not None and document.checksum == checksum:
        return IngestResult(
            document.id, document.title, filename, document.chunk_count, "unchanged"
        )

    if document is not None and not replace:
        raise DocumentConflict(
            f"A different document named '{filename}' already exists. "
            "Rename the file, or re-upload with replace=true to update it."
        )

    status = "updated" if document is not None else "created"
    store = get_vector_store()

    if document is not None:
        # Remove the old vectors before re-chunking, so a re-upload never
        # leaves orphaned vectors pointing at deleted SQL rows.
        old_ids = [c.faiss_id for c in document.chunks]
        store.remove(np.array(old_ids, dtype=np.int64))
        db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    else:
        document = Document(filename=filename)
        db.add(document)

    document.title = display_title
    document.doc_type = suffix.lstrip(".")
    document.page_count = extracted.page_count
    document.checksum = checksum
    db.flush()

    pieces = chunk_document(display_title, extracted)
    embedder = get_embedder()
    vectors = embedder.embed_documents([p.content for p in pieces])

    faiss_ids = np.array(
        [_faiss_id_for(f"{document.id}:{p.ordinal}") for p in pieces], dtype=np.int64
    )
    if len(pieces):
        store.add(faiss_ids, vectors)

    for piece, faiss_id in zip(pieces, faiss_ids, strict=True):
        db.add(
            Chunk(
                document_id=document.id,
                ordinal=piece.ordinal,
                page_number=piece.page_number,
                heading=piece.heading,
                content=piece.content,
                token_estimate=piece.token_estimate,
                faiss_id=int(faiss_id),
            )
        )
    document.chunk_count = len(pieces)
    try:
        db.commit()
    except Exception:
        # Keep FAISS and the database in step: vectors for rows that never
        # committed would be orphans. (Old vectors of a replaced document are
        # restored by reconcile_vector_store() on the next boot.)
        db.rollback()
        store.remove(faiss_ids)
        raise
    invalidate_bm25()

    log.info("Ingested %-30s %-9s %3d chunks", filename, status, len(pieces))
    return IngestResult(document.id, document.title, filename, len(pieces), status)


def reconcile_vector_store(db: Session) -> int:
    """Re-embed any chunk whose vector is missing from the FAISS index.

    The index is a separate file from the database. If it is lost, corrupted
    or rebuilt empty (e.g. a dimension mismatch on load) while the database
    survives, ingestion's checksum skip treats every document as "unchanged"
    and never re-adds the vectors - dense search would stay empty forever.
    Re-embedding from the stored chunk text restores it without re-uploading.
    Returns the number of vectors restored.
    """
    store = get_vector_store()
    rows = list(db.execute(select(Chunk.faiss_id, Chunk.content)).all())
    if not rows:
        return 0
    present = store.ids()
    missing = [(fid, content) for fid, content in rows if fid not in present]
    if not missing:
        return 0
    vectors = get_embedder().embed_documents([content for _, content in missing])
    store.add(np.array([fid for fid, _ in missing], dtype=np.int64), vectors)
    log.warning("Restored %d missing vectors into the FAISS index", len(missing))
    return len(missing)


def ingest_file(db: Session, path: Path) -> IngestResult:
    # The seed corpus is the source of truth for its own files, so an edited
    # seed document replaces its previous version.
    return ingest_bytes(db, data=path.read_bytes(), filename=path.name, replace=True)


def ingest_seed_corpus(db: Session) -> list[IngestResult]:
    if not SEED_CORPUS_DIR.exists():
        log.warning("Seed corpus directory %s does not exist", SEED_CORPUS_DIR)
        return []
    results = []
    for path in sorted(SEED_CORPUS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            results.append(ingest_file(db, path))
    return results


def delete_document(db: Session, document_id: str) -> bool:
    document = db.get(Document, document_id)
    if document is None:
        return False
    old_ids = [c.faiss_id for c in document.chunks]
    # Commit the rows first: if that fails nothing changes. A vector left
    # behind after a failed removal is harmless - retrieval drops FAISS hits
    # with no matching chunk row.
    db.delete(document)
    db.commit()
    if old_ids:
        get_vector_store().remove(np.array(old_ids, dtype=np.int64))
    invalidate_bm25()
    return True
