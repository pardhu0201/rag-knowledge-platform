"""Groundedness guardrail - catches an unsupported answer before it ships.

Deliberately does **not** trust an LLM to grade its own answer. Every check
here is deterministic and runs identically whether the answer came from
Claude or from the extractive fallback:

* **citation validity**  - every ``[n]`` marker points at a real passage
* **citation coverage**  - share of factual sentences that carry a citation
  at all (lead-ins, headings and meta-notes are excluded from the count)
* **evidence support**   - IDF-weighted token overlap between each cited
  sentence and the passage it cites
* **numeric grounding**  - every number in the answer must appear somewhere
  in the evidence; a fabricated figure is the highest-signal failure mode in
  RAG and is penalised harder than any other defect
* **question coverage**  - retrieval always returns *something*, so an
  out-of-scope question can still produce a fully-cited, well-supported
  *non-answer*. This compares the question's own vocabulary against the
  evidence and gates the whole score by it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.embeddings import tokenize
from app.generation.answer import split_sentences

STRONG_MATCH_BM25 = 12.0
RELEVANCE_FLOOR = 0.15

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_NUMBER_RE = re.compile(r"(?<![\w\[])(\d+(?:[.,]\d+)?)\s*(%|percent|days?|hours?|weeks?)?")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "is",
    "are",
    "be",
    "with",
    "that",
    "this",
    "it",
    "as",
    "at",
    "by",
    "from",
    "you",
    "your",
    "will",
    "must",
    "may",
    "can",
    "not",
    "if",
    "any",
    "all",
    "has",
    "have",
    "how",
    "what",
    "when",
    "where",
    "who",
    "why",
    "do",
    "does",
    "did",
    "need",
    "much",
    "many",
    "long",
    "i",
    "my",
    "me",
    "we",
    "our",
    "there",
    "about",
    "next",
    "per",
    "each",
    "get",
    "am",
    "was",
    "were",
    "should",
    "would",
}


def _normalise_number(raw: str) -> str:
    value = raw.replace(",", "")
    return value[:-2] if value.endswith(".0") else value


def extract_numbers(text: str) -> set[str]:
    stripped = _CITATION_RE.sub(" ", text)
    return {_normalise_number(m.group(1)) for m in _NUMBER_RE.finditer(stripped)}


def _is_claim_sentence(sentence: str) -> bool:
    cleaned = sentence.strip()
    if len(cleaned) < 30 or cleaned.endswith(("?", ":")):
        return False
    if cleaned.startswith(("_", "#", "|", ">")):
        return False
    words = [w for w in tokenize(cleaned) if w not in _STOPWORDS]
    return len(words) >= 5


def lexical_support(sentence: str, passage: str, idf: dict[str, float]) -> float:
    s_tokens = [t for t in tokenize(sentence) if t not in _STOPWORDS and len(t) > 2]
    if not s_tokens:
        return 1.0
    p_tokens = set(tokenize(passage))
    total = sum(idf.get(t, 1.0) for t in s_tokens)
    matched = sum(idf.get(t, 1.0) for t in s_tokens if t in p_tokens)
    return matched / total if total else 0.0


def query_coverage(query: str, evidence_text: str) -> float:
    terms = {t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 2}
    if not terms:
        return 1.0
    evidence_terms = set(tokenize(evidence_text))
    return len(terms & evidence_terms) / len(terms)


def check_groundedness(query: str, answer: str, retrieved: list[dict]) -> dict:
    passages = {item["index"]: item["content"] for item in retrieved}
    evidence_text = "\n".join(passages.values())

    df: Counter = Counter()
    for content in passages.values():
        df.update(set(tokenize(content)))
    n_docs = max(1, len(passages))
    idf = {term: math.log(1 + n_docs / (1 + count)) for term, count in df.items()}

    citations = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
    invalid = sorted({c for c in citations if c not in passages})

    sentences = [s for s in split_sentences(answer) if _is_claim_sentence(s)]
    cited_sentences = [s for s in sentences if _CITATION_RE.search(s)]
    coverage = (len(cited_sentences) / len(sentences)) if sentences else 1.0

    support_scores: list[float] = []
    weak_sentences: list[str] = []
    for sentence in cited_sentences:
        refs = [int(m.group(1)) for m in _CITATION_RE.finditer(sentence)]
        texts = [passages[r] for r in refs if r in passages]
        if not texts:
            support_scores.append(0.0)
            weak_sentences.append(sentence)
            continue
        best = max(lexical_support(sentence, text, idf) for text in texts)
        support_scores.append(best)
        if best < 0.45:
            weak_sentences.append(sentence)
    support = sum(support_scores) / len(support_scores) if support_scores else 0.0

    evidence_numbers = extract_numbers(evidence_text)
    answer_numbers = extract_numbers(answer)
    ungrounded_numbers = sorted(
        n for n in answer_numbers - evidence_numbers if len(n) > 1 or int(float(n)) > 3
    )

    term_coverage = query_coverage(query, evidence_text) if retrieved else 0.0
    lexical_scores = [i.get("lexical_score") for i in retrieved]
    if any(score is not None for score in lexical_scores):
        top_lexical = max(float(score or 0.0) for score in lexical_scores)
        retrieval_strength = min(1.0, top_lexical / STRONG_MATCH_BM25)
        relevance = min(term_coverage, retrieval_strength)
    else:
        retrieval_strength = term_coverage
        relevance = term_coverage

    penalties = 0.0
    penalties += 0.25 if invalid else 0.0
    penalties += 0.40 if ungrounded_numbers else 0.0

    quality = 0.30 * coverage + 0.45 * support + 0.25 * (1.0 if retrieved else 0.0)
    score = quality * (RELEVANCE_FLOOR + (1.0 - RELEVANCE_FLOOR) * relevance)
    score = max(0.0, min(1.0, score - penalties))

    return {
        "citation_count": len(citations),
        "invalid_citations": invalid,
        "claim_sentences": len(sentences),
        "cited_sentences": len(cited_sentences),
        "citation_coverage": round(coverage, 3),
        "lexical_support": round(support, 3),
        "weak_sentences": weak_sentences[:5],
        "ungrounded_numbers": ungrounded_numbers,
        "query_coverage": round(relevance, 3),
        "retrieval_strength": round(retrieval_strength, 3),
        "groundedness_score": round(score, 3),
    }
