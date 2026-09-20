"""Page-aware, heading-aware chunking for generic uploaded documents.

Uploaded reports rarely have Markdown structure, but they almost always have
*visual* structure: short, standalone, title-cased lines acting as section
headings ("Executive Summary", "Risks and Considerations"), surrounded by
blank lines. This treats each page as a sequence of paragraphs, tracks the
most recent heading-like paragraph as running context, and only then splits
the accumulated body text to a target chunk size - so a chunk about
"consecutive days" late in a section still carries that section's heading,
the same way a human reader would.

That heading (plus the page number extraction already gives) is what makes a
citation like "Business Continuity, p. 4" possible instead of just "p. 4".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.extract import ExtractedDocument

# A heading candidate: a short, standalone paragraph, title-cased-ish, with no
# terminal sentence punctuation - "Business Continuity" qualifies,
# "Brightline maintains an active-active architecture." does not.
_HEADING_RE = re.compile(r"^(?=.{3,70}$)(?!.*[.!?]$)[A-Z][A-Za-z0-9 ,&/'()-]*$")
_HEADING_MAX_WORDS = 9


@dataclass
class TextChunk:
    ordinal: int
    page_number: int
    heading: str
    content: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _is_heading(paragraph: str) -> bool:
    line = paragraph.strip()
    if not line or "\n" in line:
        return False
    if len(line.split()) > _HEADING_MAX_WORDS:
        return False
    return bool(_HEADING_RE.match(line))


def _sections(page_text: str) -> list[tuple[str, str]]:
    """Split one page's text into `(heading, body)` sections."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))
        buffer.clear()

    for paragraph in paragraphs:
        if _is_heading(paragraph):
            flush()
            current_heading = paragraph
        else:
            buffer.append(paragraph)
    flush()
    return sections or [("", page_text.strip())]


def chunk_document(
    title: str,
    extracted: ExtractedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[TextChunk] = []
    ordinal = 0
    for page in extracted.pages:
        text = page.text.strip()
        if not text:
            continue
        for heading, body in _sections(text):
            crumb_parts = [title, f"p.{page.page_number}", *([heading] if heading else [])]
            crumb = " > ".join(crumb_parts)
            for piece in splitter.split_text(body):
                piece = piece.strip()
                if len(piece) < 40:
                    continue
                content = f"[{crumb}]\n{piece}"
                chunks.append(
                    TextChunk(
                        ordinal=ordinal,
                        page_number=page.page_number,
                        heading=heading,
                        content=content,
                        token_estimate=estimate_tokens(content),
                    )
                )
                ordinal += 1
    return chunks
