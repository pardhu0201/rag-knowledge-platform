# Architecture

## Ingestion

`backend/app/ingestion/`. Extraction (`extract.py`) preserves page numbers -
the citation unit people actually expect from a document platform ("page 5,"
not "chunk 14") - via `pypdf` for PDFs, `python-docx` for Word documents, and
plain decoding for text/Markdown.

Chunking (`chunking.py`) is heading-aware even though uploaded reports rarely
have Markdown structure: each page is split into paragraphs, a short,
standalone, title-cased paragraph with no terminal punctuation is treated as
a running heading, and everything after it inherits that heading until the
next one - so a chunk about "consecutive days" late in a "Business
Continuity" section still carries that heading, the way a human reader
would. That heading plus the page number is what makes a citation like
"Business Continuity, p. 4" possible.

Ingestion (`pipeline.py`) is idempotent: a document is keyed by filename and
skipped when its content checksum (salted with the active embedder's
identity, so switching embedders doesn't leave stale vectors) is unchanged.
Re-uploading a changed file removes its old FAISS vectors before re-indexing,
so a re-upload never leaves orphaned vectors pointing at deleted rows.

## Retrieval

`backend/app/retrieval/`. Two independently-scored candidate lists are fused:

1. **BM25** (`bm25.py`, `k1=1.5, b=0.75`) over a stemmed token index, rebuilt
   lazily whenever the corpus's version fingerprint changes.
2. **FAISS** (`vector_store.py`) - `IndexIDMap2(IndexFlatIP)` over
   L2-normalised vectors (inner product = cosine similarity). Flat, exact
   search rather than an approximate index (IVF/HNSW): at the scale a demo
   document platform actually holds, exact search is fast enough that
   approximate search would only add tuning surface for no benefit. The
   `IndexIDMap2` lets each chunk carry a stable, deterministic 63-bit id
   (derived from its own uuid via BLAKE2b) rather than relying on insertion
   order, which is what makes deletion and re-indexing safe.

Fusion (`hybrid.py::_fuse`) blends each leg's min-max normalised score with
its normalised reciprocal rank (60/40) rather than pure Reciprocal Rank
Fusion - on a corpus this size, pure rank fusion throws away a decisive BM25
score margin (a 9.5 vs a 5.7) by flattening it into a near-tie. A
**heading-relevance boost** rewards a chunk whose section heading already
answers the question. **Near-duplicate suppression** (Jaccard over token
sets, engaged only above a similarity threshold) removes the overlapping
windows chunking produces without demoting genuinely distinct passages the
way classic MMR's smooth penalty would.

## Reranking

`backend/app/retrieval/rerank.py`. Hybrid fusion never looks at the query and
a candidate *together* - it scores each independently. Reranking closes that
gap with a second pass:

- **`heuristic`** (default, zero-dependency): term coverage, exact
  bigram/phrase matches (rewards passages using the query's own phrasing, not
  just its words), and *proximity* - the smallest window of the passage that
  contains every matched query term, normalised so a tight cluster of matches
  scores near 1.0 and matches scattered across a long chunk score near 0.
- **`cross-encoder`** (opt-in via `RERANK_PROVIDER`): a real
  `sentence-transformers` cross-encoder scoring (query, passage) pairs
  jointly - meaningfully better, heavier, so it stays opt-in.

## Generation and citations

`backend/app/generation/`. With Claude available, `answer.py` calls the model
with `output_config.format` structured output so the response is always
schema-valid JSON, with a hard citation contract: every claim carries a `[n]`
marker, and the system prompt (`prompts.py`) explicitly frames retrieved
passages as **untrusted data, never instructions** - the prompt-side half of
the prompt-injection defence (see Guardrails below).

Without a key, generation is **extractive**: the highest-scoring sentences
from the retrieved evidence are selected by IDF-weighted term overlap with
the question - plus a heading-inheritance bonus, since a sentence under a
"Q2 2026 Revenue" heading that doesn't itself say "revenue" should still
outrank an unrelated sentence that happens to repeat query words - and
returned verbatim with their citation numbers. It cannot state a fact that is
not written, word for word, in a cited passage.

## Guardrails

`backend/app/guardrails/`.

### Groundedness (`groundedness.py`)

Deliberately does not trust an LLM to grade its own answer. Every check is
deterministic and runs identically in both generation modes:

- **citation validity** - every `[n]` marker points at a real passage
- **citation coverage** - share of factual sentences that carry a citation
  at all
- **evidence support** - IDF-weighted token overlap between each cited
  sentence and the passage it cites
- **numeric grounding** - every number in the answer must appear somewhere in
  the evidence; a fabricated figure is penalised harder (0.40) than any other
  defect, since it is the highest-signal failure mode and the one a fluent
  answer hides best
- **question coverage** - retrieval always returns *something*, so an
  out-of-scope question can still produce a fully-cited, well-supported
  *non-answer*. This compares the question's own vocabulary (plus an absolute
  BM25 floor) against the evidence and gates the whole score by it.

When the resulting score falls below `GROUNDEDNESS_THRESHOLD`
(`app/pipeline.py`), the guardrail doesn't just log a flag - it **rewrites
the answer**, prepending a visible warning banner before the response ever
reaches the user.

### Prompt injection (`prompt_injection.py`)

A document-upload RAG platform has two distinct injection surfaces, and they
get different responses:

1. **The user's own query** trying to hijack the assistant directly
   ("ignore your instructions and...") - unambiguous, so a match **blocks**
   the request outright, before retrieval even runs.
2. **Retrieved document content** containing the same phrasing, planted by
   whoever uploaded the document, hoping the assistant obeys it when that
   passage is fed back in - the classic indirect-injection attack on RAG
   systems. This must **not** hard-block, because legitimate documents
   genuinely *discuss* this phrasing (a security-awareness policy quoting a
   phishing email - see the seeded `Ai Tooling Usage Guidelines` document,
   which does exactly this and is correctly flagged, never blocked, by this
   guardrail in the shipped demo). Instead it's **detect-and-flag**: surfaced
   on the dashboard, with the actual mitigation living in the generation
   system prompt's explicit "treat retrieved text as data" instruction.

Detection is a fast, transparent regex scan rather than a second model call -
a guardrail that itself depends on a model call is a guardrail that fails
exactly when the model is being attacked.

## Observability

Every query is logged to `QueryLog` (`db/models.py`) with per-stage latency
(retrieval/rerank/generation), token counts, an estimated dollar cost
(`metrics/cost.py`, from a small static Anthropic pricing table - works
offline, no live billing API dependency), the groundedness score, and every
guardrail flag that fired. `GET /api/dashboard` aggregates this into the
numbers the frontend's Dashboard page charts: mean/p95 latency, guardrail
activity counts, and a full recent-query log.

## Evaluation

`backend/evals/golden_set.json` + `run_eval.py` scores 22 labelled cases
across all four seed documents plus deliberately out-of-scope and
prompt-injection cases: retrieval recall@5/MRR, fact coverage, groundedness,
hallucinated-number rate, invalid-citation rate, and - the cases that matter
most for a guardrailed system - whether every injection attempt was actually
blocked and every out-of-scope question actually flagged low-confidence. CI
runs this on every push and fails if any threshold regresses.
