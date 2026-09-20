"""Extraction, chunking and ingestion behaviour."""

from __future__ import annotations

import pytest

from app.ingestion.chunking import _is_heading, chunk_document
from app.ingestion.extract import ExtractedDocument, ExtractedPage, extract

SAMPLE_PAGE = """Executive Summary

Revenue grew 22% year over year to $18.4 million, driven by strong
enterprise expansion across all regions.

Risks and Considerations

Foreign exchange movements created a modest headwind to reported
international revenue this quarter.
"""


def test_txt_extraction_is_single_page():
    doc = extract(b"hello world", ".txt")
    assert doc.page_count == 1
    assert doc.pages[0].text == "hello world"


def test_unsupported_suffix_raises():
    with pytest.raises(ValueError):
        extract(b"data", ".exe")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Executive Summary", True),
        ("Risks and Considerations", True),
        ("Revenue grew 22% year over year to $18.4 million.", False),
        ("A", False),
        ("This heading is deliberately far too long to be a real heading line", False),
    ],
)
def test_heading_detection(line, expected):
    assert _is_heading(line) is expected


def test_chunking_carries_page_and_heading():
    extracted = ExtractedDocument(
        pages=[ExtractedPage(page_number=3, text=SAMPLE_PAGE)], page_count=5
    )
    chunks = chunk_document("Test Report", extracted)
    assert chunks, "expected at least one chunk"
    assert all(c.page_number == 3 for c in chunks)
    headings = {c.heading for c in chunks}
    assert "Executive Summary" in headings
    assert "Risks and Considerations" in headings
    for chunk in chunks:
        assert chunk.content.startswith("[Test Report > p.3")


def test_short_pieces_are_dropped():
    extracted = ExtractedDocument(
        pages=[ExtractedPage(page_number=1, text="Heading\n\nToo short.")], page_count=1
    )
    chunks = chunk_document("Doc", extracted)
    assert all(len(c.content) > 40 for c in chunks)
