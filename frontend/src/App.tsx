import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Gauge, MessagesSquare, Search, UploadCloud } from "lucide-react";

import { api, type Health } from "./lib/api";
import AskPage from "./pages/AskPage";
import DocumentsPage from "./pages/DocumentsPage";
import DashboardPage from "./pages/DashboardPage";

const NAV = [
  { to: "/ask", label: "Ask", icon: MessagesSquare },
  { to: "/documents", label: "Documents", icon: UploadCloud },
  { to: "/dashboard", label: "Dashboard", icon: Gauge },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="flex min-h-full flex-col lg:flex-row">
      <aside className="shrink-0 border-b border-[var(--color-line)] bg-[rgba(14,19,34,0.72)] backdrop-blur lg:sticky lg:top-0 lg:h-screen lg:w-64 lg:border-r lg:border-b-0">
        <div className="flex items-center gap-3 px-5 py-5">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-[var(--color-brand)] to-[var(--color-brand-2)] text-[#04231a]">
            <Search size={18} strokeWidth={2.4} />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">RAG Knowledge Platform</div>
            <div className="truncate text-[11px] text-[var(--color-ink-3)]">
              Retrieval &middot; Evaluation &middot; Guardrails
            </div>
          </div>
        </div>

        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible lg:pb-0">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                [
                  "flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-[var(--color-surface-3)] font-medium text-[var(--color-ink)]"
                    : "text-[var(--color-ink-2)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-ink)]",
                ].join(" ")
              }
            >
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <SystemCard health={health} />
      </aside>

      <main className="min-w-0 flex-1">
        <Routes>
          <Route path="/" element={<Navigate to="/ask" replace />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="*" element={<Navigate to="/ask" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function SystemCard({ health }: { health: Health | null }) {
  const live = health?.status === "ok";
  const claude = health?.llm_mode === "claude";

  return (
    <div className="hidden px-4 pb-5 lg:block">
      <div className="panel-2 space-y-2.5 p-3.5 text-[11px]">
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${live ? "bg-[var(--color-ok)]" : "bg-[var(--color-danger)]"}`}
          />
          <span className="label">{live ? "System online" : "API unreachable"}</span>
        </div>

        {health && (
          <dl className="space-y-1.5 text-[var(--color-ink-3)]">
            <Row label="Generation" value={claude ? health.llm_model : "Extractive (demo)"} accent={claude} />
            <Row label="Vector store" value={health.vector_store} />
            <Row label="Embeddings" value={health.embedding_provider} />
            <Row label="Rerank" value={health.rerank_provider} />
            <Row label="Corpus" value={`${health.documents} docs / ${health.chunks} chunks`} />
          </dl>
        )}

        {health && !claude && (
          <p className="border-t border-[var(--color-line)] pt-2.5 leading-relaxed text-[var(--color-ink-3)]">
            Running without an API key: answers are extracted verbatim from
            the cited passages. Set <span className="font-mono">ANTHROPIC_API_KEY</span> for
            generative reasoning.
          </p>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="shrink-0">{label}</dt>
      <dd
        className={`truncate text-right font-mono ${accent ? "text-[var(--color-brand)]" : "text-[var(--color-ink-2)]"}`}
      >
        {value}
      </dd>
    </div>
  );
}
