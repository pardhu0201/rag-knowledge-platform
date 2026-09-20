/** Typed client for the RAG Knowledge Platform API. */

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
export const apiUrl = (path: string) => `${BASE}/api${path}`;

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_number: number | null;
  heading: string;
  snippet: string;
  score: number;
  dense_score: number;
  lexical_score: number;
}

export interface ContextInjection {
  patterns: string[];
  snippet: string;
}

export interface Groundedness {
  citation_coverage?: number;
  lexical_support?: number;
  query_coverage?: number;
  retrieval_strength?: number;
  invalid_citations?: number[];
  ungrounded_numbers?: string[];
  weak_sentences?: string[];
  groundedness_score?: number;
  [key: string]: unknown;
}

export interface QueryResponse {
  answer: string;
  blocked: boolean;
  block_reason: string;
  citations: Citation[];
  used_citations: number[];
  flags: string[];
  follow_up_question: string;
  groundedness: Groundedness;
  context_injections: Record<number, ContextInjection>;
  llm_mode: string;
  latency_ms: number;
  retrieval_ms: number;
  rerank_ms: number;
  generation_ms: number;
  token_usage: { input: number; output: number };
  estimated_cost_usd: number;
  log_id: string;
}

export interface DocumentSummary {
  id: string;
  title: string;
  filename: string;
  doc_type: string;
  page_count: number;
  chunk_count: number;
  created_at: string;
}

export interface Health {
  status: string;
  version: string;
  llm_mode: string;
  llm_model: string;
  embedding_provider: string;
  rerank_provider: string;
  vector_store: string;
  documents: number;
  chunks: number;
  vectors: number;
}

export interface Dashboard {
  queries_total: number;
  blocked_total: number;
  average_groundedness: number;
  low_groundedness_rate: number;
  average_latency_ms: number;
  p95_latency_ms: number;
  average_retrieval_ms: number;
  average_rerank_ms: number;
  average_generation_ms: number;
  total_estimated_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
  flag_counts: Record<string, number>;
  recent_queries: {
    id: string;
    query: string;
    llm_mode: string;
    blocked: boolean;
    groundedness_score: number;
    total_ms: number;
    estimated_cost_usd: number;
    flags: string[];
    created_at: string;
  }[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* no JSON body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  dashboard: () => request<Dashboard>("/dashboard"),

  documents: () => request<DocumentSummary[]>("/documents"),
  deleteDocument: (id: string) =>
    request<{ deleted: string }>(`/documents/${id}`, { method: "DELETE" }),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ title: string; chunks: number; status: string }>("/documents/upload", {
      method: "POST",
      body: form,
    });
  },

  query: (query: string) =>
    request<QueryResponse>("/query", { method: "POST", body: JSON.stringify({ query }) }),
};
