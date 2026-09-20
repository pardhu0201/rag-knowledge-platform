"""Document management: list, upload, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.db.models import Document
from app.ingestion.extract import SUPPORTED_SUFFIXES
from app.ingestion.pipeline import delete_document, ingest_bytes
from app.logging_config import get_logger
from app.schemas import DocumentOut, IngestResponse

log = get_logger(__name__)
router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB - generous for a real PDF report


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_session)):
    rows = db.execute(select(Document).order_by(Document.created_at.desc())).scalars()
    return [
        DocumentOut(
            id=d.id,
            title=d.title,
            filename=d.filename,
            doc_type=d.doc_type,
            page_count=d.page_count,
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in rows
    ]


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_session)):
    name = file.filename or "upload.txt"
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 15 MB")

    try:
        result = ingest_bytes(db, data=data, filename=name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(
        document_id=result.document_id,
        title=result.title,
        filename=result.filename,
        chunks=result.chunks,
        status=result.status,
    )


@router.delete("/documents/{document_id}")
def remove_document(document_id: str, db: Session = Depends(get_session)):
    if not delete_document(db, document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}
