"""Test fixtures.

The app builds its settings/engine at import time from env vars, so the test
database and a scratch FAISS index directory must be set *before* any `app`
module is imported.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest

TEST_ROOT = Path(tempfile.gettempdir()) / "rag_platform_test"
TEST_DB = TEST_ROOT / "app.db"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["ANTHROPIC_API_KEY"] = ""  # force deterministic demo mode in CI
os.environ["SEED_ON_STARTUP"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["EMBEDDING_PROVIDER"] = "hashing"

# Point the FAISS index at the same scratch directory before app.config is
# ever imported, by monkeypatching the module-level constant post-import is
# unsafe (other modules cache it) - instead we override via env-independent
# path by ensuring RUNTIME/FAISS dirs are unique to this test root.
os.environ.setdefault("_RAG_TEST_ROOT", str(TEST_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _database():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT, ignore_errors=True)

    # Import after the reset above (some test modules import app.config at
    # collection time, which lazily creates these directories once via the
    # engine's own startup - the rmtree just removed them again).
    from app.config import FAISS_INDEX_DIR, RUNTIME_DIR
    from app.db.init_db import initialise

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    initialise(seed=True)
    yield
    with contextlib.suppress(PermissionError):
        shutil.rmtree(TEST_ROOT, ignore_errors=True)


@pytest.fixture
def db():
    from app.db.base import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
