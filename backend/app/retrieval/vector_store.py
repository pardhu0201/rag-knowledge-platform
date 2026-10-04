"""FAISS-backed vector index with a persisted id map.

FAISS's `IndexIDMap2` lets us attach an arbitrary 63-bit integer id to each
vector instead of relying on insertion order, which is what makes deletion
(removing a document's chunks) and incremental re-indexing safe. Chunk rows
in SQLite store that same integer in `Chunk.faiss_id`, so a search result
(a FAISS id) maps straight back to a chunk row with one lookup.

The index is flat (`IndexFlatIP`, inner product over L2-normalised vectors =
cosine similarity) rather than an approximate index (IVF/HNSW): at the scale
a demo document platform actually holds (thousands of chunks, not millions),
exact search is fast enough that approximate search would only add tuning
surface for no real benefit.
"""

from __future__ import annotations

import threading

import faiss
import numpy as np

from app.config import FAISS_INDEX_DIR
from app.embeddings import get_embedder
from app.logging_config import get_logger

log = get_logger(__name__)

_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"


class VectorStore:
    """Thread-safe wrapper around one FAISS IndexIDMap2(IndexFlatIP)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._lock = threading.Lock()
        self._index = self._load_or_create()

    def _load_or_create(self) -> faiss.IndexIDMap2:
        if _INDEX_FILE.exists():
            try:
                index = faiss.read_index(str(_INDEX_FILE))
                if index.d == self.dim:
                    log.info("Loaded FAISS index (%d vectors)", index.ntotal)
                    return index
                log.warning(
                    "FAISS index dimension mismatch (%d != %d); rebuilding", index.d, self.dim
                )
            except Exception as exc:  # pragma: no cover - corrupt file on disk
                log.warning("Could not load FAISS index (%s); rebuilding", exc)
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def save(self) -> None:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(_INDEX_FILE))

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        with self._lock:
            self._index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))
        self.save()

    def remove(self, ids: np.ndarray) -> None:
        if len(ids) == 0:
            return
        with self._lock:
            self._index.remove_ids(ids.astype(np.int64))
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return `(faiss_id, cosine_similarity)` pairs, best first."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            top_k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(query_vector.astype(np.float32).reshape(1, -1), top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    def ids(self) -> set[int]:
        """Every id currently in the index (for reconciliation with the DB)."""
        with self._lock:
            if self._index.ntotal == 0:
                return set()
            return {int(i) for i in faiss.vector_to_array(self._index.id_map)}

    def reset(self) -> None:
        with self._lock:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.save()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Singleton store, dimensioned from the *actual* active embedder.

    Reading the dimension off the embedder instance (rather than a separate
    settings field) is what keeps the index and the embedder in sync even if
    `EMBEDDING_PROVIDER` is switched to one with a different dimension.
    """
    global _store
    if _store is None:
        _store = VectorStore(dim=get_embedder().dim)
    return _store
