# Deployment

The whole app ships as **one Docker image** - the React bundle is built in a
first stage and served by FastAPI as static files in the second.

## Recommended: Render (free web service, SQLite + FAISS on disk)

1. Click **[Deploy to Render](https://render.com/deploy?repo=https://github.com/pardhu0201/rag-knowledge-platform)**
   (also linked from the README), or in the Render dashboard choose
   **New → Blueprint** and point it at this repo - it reads `render.yaml` at
   the repo root automatically.
2. No managed database is provisioned. Documents, chunks and the query log
   live in SQLite; the vector index is FAISS-on-disk. Nothing expires,
   unlike Render's free Postgres (30-day expiry on new instances), which
   this project has no dependency on in the first place - but the free
   tier's disk is **ephemeral**: it is wiped on every restart, redeploy or
   wake from sleep. The seed corpus is re-ingested on boot, so the demo
   always comes back working; documents you uploaded yourself do not.
3. Optionally add an `ANTHROPIC_API_KEY` secret in the service's
   **Environment** tab to switch on Claude-generated answers. Nothing else is
   required.
4. Render's free tier **sleeps after 15 minutes of inactivity** and takes
   roughly 50 seconds to wake on the next request - expected for a portfolio
   demo. Health check is `/api/health`.

## Alternative: Vercel (frontend) + Render (backend only)

Only needed if you want the frontend and backend on separate hosts.

1. Deploy the same root `Dockerfile` to Render (via the blueprint above), but
   set `SERVE_FRONTEND=false` in the service's environment - the built
   frontend is still baked into the image, but FastAPI stops serving it and
   only answers under `/api/*`.
2. Deploy `frontend/` to Vercel using the build settings in
   `frontend/vercel.json` (already committed), with:
   ```
   VITE_API_BASE=https://<your-render-service>.onrender.com
   ```
3. Allow your frontend's origin on the backend: set `CORS_ORIGINS` to your
   production URL and, optionally, `CORS_ORIGIN_REGEX` for your own preview
   deployments (e.g. `^https://rag-knowledge-platform(-[a-z0-9-]+)?\.vercel\.app$`).
   The backend no longer trusts every `*.vercel.app` origin by default - that
   would let any Vercel-hosted site call the API from a visitor's browser.

## Local: Docker Compose

```bash
docker compose up --build
# open http://localhost:7860
```

State (SQLite + the FAISS index) persists in a named volume across
`down`/`up` cycles.

## Environment variables

Every variable has a working default - see `.env.example` at the repo root.
The ones that matter for a deployment:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Blank = deterministic extractive mode (free). Set to enable Claude generation. |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (zero-dependency) or `sentence-transformers` (better recall, heavier image). |
| `RERANK_PROVIDER` | `heuristic` | `heuristic` (zero-dependency) or `cross-encoder` (better, heavier). |
| `GROUNDEDNESS_THRESHOLD` | `0.5` | Below this, the answer is prefixed with a visible low-confidence warning. |
| `BLOCK_ON_PROMPT_INJECTION` | `true` | Whether a query-level injection match blocks the request outright. |
| `CORS_ORIGINS` | `localhost:5173` | Comma-separated list; add your frontend's production origin. |
| `CORS_ORIGIN_REGEX` | *(empty)* | Optional regex for extra origins, e.g. your own Vercel previews. |
| `ADMIN_TOKEN` | *(empty)* | When set, document upload/delete require an `X-Admin-Token` header (the UI prompts once). Querying stays open. Blank = open demo. |
| `SEED_ON_STARTUP` | `true` | Re-ingests the bundled demo corpus on every boot (idempotent). |

## Verifying a deployment

```bash
curl https://<your-host>/api/health
# {"status":"ok","llm_mode":"demo"|"claude","documents":4,"chunks":37,"vectors":37,...}
```

`vectors` should equal `chunks` - if it doesn't, the FAISS index and the
SQLite chunk table have drifted and a re-seed (`SEED_ON_STARTUP=true` on next
boot, or delete `backend/data/faiss_index` and `backend/data/runtime`) will
resync them.
