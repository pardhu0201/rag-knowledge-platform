import { AlertTriangle, FileText } from "lucide-react";

import type { Citation, ContextInjection } from "../lib/api";

/** The evidence an answer was built from, with retrieval scores and any
 *  prompt-injection flags surfaced per passage. */
export default function SourcePanel({
  citations,
  used,
  injections,
  highlighted,
}: {
  citations: Citation[];
  used: number[];
  injections: Record<number, ContextInjection>;
  highlighted?: number | null;
}) {
  if (!citations.length) return null;
  const usedSet = new Set(used);

  return (
    <div className="space-y-2">
      {citations.map((citation) => {
        const isUsed = usedSet.has(citation.index);
        const isActive = highlighted === citation.index;
        const injection = injections[citation.index];
        return (
          <article
            key={citation.chunk_id}
            id={`source-${citation.index}`}
            className={[
              "rounded-lg border p-3 transition-colors",
              isActive
                ? "border-[var(--color-brand)] bg-[rgba(45,212,191,0.08)]"
                : "border-[var(--color-line)] bg-[var(--color-surface-2)]",
              isUsed ? "" : "opacity-60",
            ].join(" ")}
          >
            <div className="flex items-start gap-2.5">
              <span
                className={[
                  "mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded font-mono text-[11px] font-semibold",
                  isUsed
                    ? "bg-[rgba(45,212,191,0.2)] text-[var(--color-brand)]"
                    : "bg-[var(--color-surface-3)] text-[var(--color-ink-3)]",
                ].join(" ")}
              >
                {citation.index}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 text-[13px] font-medium">
                  <FileText size={12} className="shrink-0 text-[var(--color-ink-3)]" />
                  <span className="truncate">{citation.document_title}</span>
                </div>
                <div className="truncate text-[11px] text-[var(--color-ink-3)]">
                  {citation.page_number ? `p. ${citation.page_number}` : ""}
                  {citation.heading ? ` · ${citation.heading}` : ""}
                </div>
              </div>
            </div>

            <p className="mt-2 text-[12.5px] leading-relaxed text-[var(--color-ink-2)]">
              {citation.snippet}
            </p>

            {injection && (
              <div className="mt-2 flex items-start gap-1.5 rounded-md border border-[rgba(225,29,72,0.35)] bg-[rgba(225,29,72,0.08)] px-2.5 py-1.5 text-[11px] text-[var(--color-danger)]">
                <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                <span>
                  Injection-style phrasing detected ({injection.patterns.join(", ")}) - not
                  followed, flagged for review only.
                </span>
              </div>
            )}

            <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[10.5px] text-[var(--color-ink-3)]">
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                fused {citation.score.toFixed(2)}
              </span>
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                bm25 {citation.lexical_score.toFixed(1)}
              </span>
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                vec {citation.dense_score.toFixed(2)}
              </span>
              {!isUsed && <span className="italic">retrieved, not cited</span>}
            </div>
          </article>
        );
      })}
    </div>
  );
}
