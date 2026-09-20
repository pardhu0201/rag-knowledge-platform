"""Prompt-injection guardrail.

A document-upload RAG platform has two distinct injection surfaces, and they
call for different responses:

1. **The user's own query** tries to hijack the assistant directly
   ("ignore your instructions and...", "reveal your system prompt"). This is
   unambiguous - there is no legitimate reason a real question needs that
   phrasing - so a match here **blocks** the request outright before
   retrieval even runs, when ``BLOCK_ON_PROMPT_INJECTION`` is on.

2. **Retrieved document content** contains the same kind of phrasing, planted
   by whoever uploaded the document, hoping the assistant will obey it when
   the passage is fed back in in a future answer. This is the classic
   indirect-injection attack on RAG systems - and it must **not** hard-block**,
   because plenty of entirely legitimate documents *discuss* instruction
   phrasing (a security-awareness policy that quotes a phishing email, this
   very platform's own documentation). Blocking on mention would make the
   product unusable. Instead this is a **detect-and-flag** guardrail: matches
   are surfaced on the observability dashboard, and the actual mitigation is
   structural - the generation system prompt (see `generation/prompts.py`)
   explicitly instructs the model to treat retrieved text as data, never as
   instructions, regardless of what it contains.

Detection here is deliberately a fast, transparent regex scan rather than a
second LLM call - a guardrail that itself depends on a model call is a
guardrail that fails exactly when the model is being attacked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Each pattern is a (name, compiled regex) pair. Patterns target *instruction*
# phrasing aimed at an AI assistant, not general suspicious words, to keep the
# false-positive rate low on ordinary documents and questions.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all)\b"
            r"[^.\n]{0,40}\b(instructions?|prompt|rules?|guidelines?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(reveal|show|print|output|repeat|what (is|are))\b[^.\n]{0,30}"
            r"\b(your\s+)?(system\s+prompt|system\s+message|instructions?|initial\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"\byou\s+are\s+now\b|\bact\s+as\s+(if\s+you\s+(are|were)|an?\s)\b"
            r"|\bnew\s+persona\b|\bpretend\s+(you|to\s+be)\b|\bDAN\s+mode\b|\bjailbreak\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_turn_injection",
        re.compile(
            r"^\s*(system|assistant|human)\s*:\s*",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "restriction_removal",
        re.compile(
            r"\bwithout\s+(any\s+)?(restrictions?|limitations?|filters?|guardrails?)\b"
            r"|\bno\s+longer\s+(bound|restricted|limited)\b|\bunrestricted\s+mode\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class InjectionScan:
    matched: bool
    patterns: list[str] = field(default_factory=list)
    snippet: str = ""


def scan(text: str) -> InjectionScan:
    """Run every pattern against `text`, returning the first few matches."""
    hits: list[str] = []
    first_span: tuple[int, int] | None = None
    for name, pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(name)
            if first_span is None:
                first_span = match.span()
    snippet = ""
    if first_span:
        start = max(0, first_span[0] - 20)
        end = min(len(text), first_span[1] + 20)
        snippet = text[start:end].strip()
    return InjectionScan(matched=bool(hits), patterns=hits, snippet=snippet)


def scan_query(query: str) -> InjectionScan:
    return scan(query)


def scan_passages(retrieved: list[dict]) -> dict[int, InjectionScan]:
    """Scan each retrieved passage independently, keyed by citation index."""
    flagged: dict[int, InjectionScan] = {}
    for item in retrieved:
        result = scan(item["content"])
        if result.matched:
            flagged[item["index"]] = result
    return flagged
