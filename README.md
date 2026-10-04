# RAG Knowledge Platform

A production-shaped document intelligence platform: upload PDFs, reports and
technical documents, ask questions in plain language, and get answers with
inline citations — backed by hybrid retrieval, reranking, and guardrails
that catch unsupported answers and prompt-injection attempts before they
reach you.

**[Live demo →](https://rag-knowledge-platform-tgzo.onrender.com)** &nbsp;·&nbsp;
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pardhu0201/rag-knowledge-platform)

Runs for free (Render's free web-service tier — sleeps after 15 minutes
idle, ~50s to wake on the next request) in deterministic extractive mode
with no API key, and upgrades in place to Claude Opus 5 generation when one
is supplied. Click **Deploy to Render** to spin up your own copy.

![Ask page — cited answer, live pipeline metrics, groundedness guardrail](docs/screenshots/01-ask.png)

## Why this exists

A portfolio project demonstrating, with working and tested code, the parts of
a RAG system that separate a demo from something production-shaped:

- **Real document ingestion** — PDF (page-accurate), DOCX and plain text,
  chunked with heading-and-page tracking so a citation reads "Business
  Continuity, p. 4," not just "chunk 14."
- **Hybrid retrieval done properly** — BM25 + FAISS vector search, fused by
  blended score-and-rank (not naive RRF, which throws away a decisive lexical
  score margin at this corpus scale), with near-duplicate suppression and a
  heading-relevance boost.
- **A real reranking stage** — a second, query-aware pass over retrieval's
  candidates (term coverage, phrase/bigram matches, token proximity), with a
  cross-encoder swap-in for when quality matters more than latency.
- **Guardrails that actually guard, not just log** — a deterministic
  groundedness checker (citation validity, evidence support, numeric
  hallucination detection, out-of-scope detection) that **rewrites the
  answer** with a visible warning banner when confidence is low; and a
  prompt-injection scanner that **blocks outright** when the attack is in the
  user's own query, but only **flags** — never blocks — when it's embedded in
  a retrieved document, because a security-awareness policy that quotes a
  phishing email should never make the platform unusable. Patterns target
  instructions aimed at the assistant ("ignore *your* instructions"), not
  topic words, so "what are the instructions for deploying X?" is answered;
  leetspeak, zero-width characters and Cyrillic look-alikes are normalised
  away before matching. Both directions are tested — attacks blocked *and*
  legitimate trigger-word questions let through.
- **Evaluation against a curated dataset, not vibes** — `backend/evals/` is a
  32-case golden set scoring retrieval recall/MRR, fact coverage, groundedness,
  hallucination rate, and — critically — whether prompt-injection attempts
  were actually blocked, legitimate questions that *mention* injection
  vocabulary were never blocked, and out-of-scope questions were flagged. CI runs
  it on every push and fails on regression.
- **A real cost/latency dashboard** — every query is logged with per-stage
  timing (retrieval / rerank / generation) and an estimated dollar cost from
  actual token counts, because "it works" and "it's affordable at scale" are
  different questions.

## Architecture

```mermaid
flowchart LR
    U([User]) -->|query| G1{{"Prompt-injection\nguardrail (query)"}}
    G1 -- match --> BLOCK[Blocked + logged]
    G1 -- clean --> R[Hybrid retrieve\nBM25 + FAISS]
    R --> RR[Rerank]
    RR --> G2{{"Prompt-injection\nscan (passages)"}}
    G2 -->|flag only| GEN[Generate\nClaude or extractive]
    GEN --> G3{{"Groundedness\nguardrail"}}
    G3 -- low score --> WARN["Answer + warning banner"]
    G3 -- high score --> OK[Answer]
    WARN --> U
    OK --> U
    BLOCK --> U
```

| Stage | What it does |
|---|---|
| **Ingestion** | PDF/DOCX/TXT → page-tracked extraction → heading-aware chunking → embed → FAISS + SQLite |
| **Retrieval** | BM25 + FAISS, fused by normalised score + reciprocal rank, near-duplicate suppression |
| **Rerank** | Query-aware re-scoring of retrieval's candidates (heuristic default, cross-encoder opt-in) |
| **Generation** | Claude Opus 5 with structured citations, or a strictly extractive fallback with zero API cost |
| **Guardrails** | Groundedness (rewrites low-confidence answers with a warning) + prompt injection (blocks on query, flags on passage) |
| **Observability** | Every query logged: latency per stage, token usage, estimated cost, groundedness score, guardrail flags |

Full design notes — the fusion formula, the reranker's scoring, the
groundedness math, why blocking differs between query and passage injection —
are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Choice |
|---|---|
| API | Python, **FastAPI**, Pydantic v2 |
| Retrieval | **LangChain** text splitters, hashed/sentence-transformer embeddings, hybrid BM25 |
| Vector store | **FAISS** (in-process, `IndexIDMap2` + flat cosine — genuinely free forever, no hosted service) |
| Reranking | Heuristic (zero-dependency default) or `sentence-transformers` cross-encoder |
| Generation | **Claude Opus 5** (Anthropic API), optional — deterministic extractive fallback needs no key |
| Guardrails | Deterministic groundedness scoring + regex-based prompt-injection detection |
| Database | SQLite (documents, chunks metadata, query log) |
| Frontend | **React 19**, TypeScript, Tailwind v4, Vite |
| Tests / evals | pytest (68 tests), a 32-case golden-set eval harness, ruff |
| CI/CD | GitHub Actions — lint, tests, evals, full Docker boot-and-healthcheck |
| Deployment | Single Docker image — free-tier ready on Render |

## Screenshots

| | |
|---|---|
| ![Prompt injection blocked](docs/screenshots/02-ask-blocked.png) | ![Dashboard](docs/screenshots/04-dashboard.png) |
| A direct prompt-injection attempt is blocked before retrieval even runs. | Every guardrail firing, per-stage latency, and full query log. |

## Quick start

### Docker (recommended)

```bash
git clone https://github.com/pardhu0201/rag-knowledge-platform.git
cd rag-knowledge-platform
docker build -t rag-knowledge-platform .
docker run -p 7860:7860 rag-knowledge-platform
# open http://localhost:7860
```

No environment variables required. Add `-e ANTHROPIC_API_KEY=sk-...` to
switch on Claude-generated answers.

### Local dev (hot reload)

```bash
# Backend
cd backend
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev    # http://localhost:5173, proxies /api to :8000
```

### Tests and evaluation

```bash
cd backend
pytest -q                     # 68 tests: ingestion, retrieval, guardrails, API
python -m evals.run_eval      # retrieval/groundedness/guardrail scorecard
```

## How the free demo works

With no `ANTHROPIC_API_KEY`, generation is **extractive**: the highest-scoring
sentences from the retrieved evidence are selected by IDF-weighted term
overlap (with a heading-inheritance bonus) and returned verbatim with their
citation numbers — it is structurally incapable of stating a fact that isn't
written, word for word, in a cited document. Retrieval, reranking, both
guardrails, and the observability dashboard are **identical** in both modes;
supplying an API key swaps only the generation step.

## Project layout

```
backend/
  app/
    ingestion/      # extraction (PDF page tracking), heading-aware chunking, pipeline
    embeddings/      # hashing (zero-dep) + sentence-transformers embedders
    retrieval/      # BM25, FAISS vector store, hybrid fusion, reranking
    generation/      # Anthropic client wrapper, prompts, extractive fallback
    guardrails/      # groundedness scoring, prompt-injection detection
    metrics/         # token-cost estimation
    api/             # FastAPI routers
    pipeline.py      # ties every stage together, timed and logged
  data/seed_corpus/  # 4 seed documents (revenue report, tech spec, vendor assessment, security guideline)
  evals/             # 32-case golden-set evaluation harness
  tests/             # 68 pytest tests
frontend/
  src/
    pages/           # Ask, Documents, Dashboard
    components/      # AnswerBody (citations), SourcePanel
    lib/api.ts        # typed client
docs/                # architecture notes, deployment guide, screenshots
```

## License

MIT — see [LICENSE](LICENSE).
