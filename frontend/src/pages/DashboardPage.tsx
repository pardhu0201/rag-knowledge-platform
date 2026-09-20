import { useEffect, useState } from "react";
import { AlertTriangle, Clock, DollarSign, Gauge, Loader2, ShieldCheck } from "lucide-react";

import { EmptyState, GroundednessMeter, PageHeader } from "../components/primitives";
import { api, type Dashboard } from "../lib/api";

/* --------------------------------------------------------------------------
   Chart palette: one hue (sequential job) for the flag-count magnitude bars -
   length carries the value. Latency/cost use plain KPI tiles (a single
   current value each), which per the form heuristic is a stat tile, not a
   chart. No categorical identity encoding is needed anywhere on this page.
   -------------------------------------------------------------------------- */
const SEQUENTIAL = "#38bdf8";

export default function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .dashboard()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-[var(--color-ink-3)]">
        <Loader2 size={22} className="animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Dashboard"
        subtitle="Retrieval quality, latency and estimated token cost across every answered query."
      />

      <div className="space-y-7 px-5 py-6 sm:px-7">
        {!data || data.queries_total === 0 ? (
          <EmptyState
            icon={<Gauge size={18} />}
            title="No queries yet"
            body="Ask a few questions on the Ask page and quality, latency and cost statistics will appear here."
          />
        ) : (
          <>
            <StatRow data={data} />

            <div className="grid items-start gap-5 lg:grid-cols-2">
              <FlagBreakdown flags={data.flag_counts} />
              <LatencyBreakdown data={data} />
            </div>

            <RecentQueries rows={data.recent_queries} />
          </>
        )}
      </div>
    </div>
  );
}

function StatRow({ data }: { data: Dashboard }) {
  const tiles = [
    { label: "Queries answered", value: String(data.queries_total), icon: Gauge },
    {
      label: "Mean groundedness",
      value: `${Math.round(data.average_groundedness * 100)}%`,
      icon: ShieldCheck,
      note: `${Math.round(data.low_groundedness_rate * 100)}% below threshold`,
    },
    {
      label: "p95 latency",
      value: `${Math.round(data.p95_latency_ms)}ms`,
      icon: Clock,
      note: `mean ${Math.round(data.average_latency_ms)}ms`,
    },
    {
      label: "Estimated cost",
      value: `$${data.total_estimated_cost_usd.toFixed(4)}`,
      icon: DollarSign,
      note: `${data.total_input_tokens + data.total_output_tokens} tokens total`,
    },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {tiles.map(({ label, value, icon: Icon, note }) => (
        <div key={label} className="panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-ink-3)]">
            <Icon size={13} />
            <span className="label">{label}</span>
          </div>
          <div className="mt-2 font-mono text-[26px] leading-none font-semibold tracking-tight">
            {value}
          </div>
          {note && <div className="mt-1.5 text-[11.5px] text-[var(--color-ink-3)]">{note}</div>}
        </div>
      ))}
    </div>
  );
}

function FlagBreakdown({ flags }: { flags: Record<string, number> }) {
  const rows = Object.entries(flags).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, n]) => n));

  return (
    <section className="panel p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Guardrail activity</h2>
      <p className="mt-0.5 mb-4 text-[12.5px] text-[var(--color-ink-3)]">
        How often each guardrail fired, most common first
      </p>

      {rows.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-[var(--color-ink-3)]">
          No flags recorded - every query answered cleanly.
        </p>
      ) : (
        <ul className="space-y-3">
          {rows.map(([flag, count]) => (
            <li key={flag} title={`${flag}: ${count}`}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className="truncate text-[12.5px] text-[var(--color-ink-2)]">
                  {flag.replace(/_/g, " ")}
                </span>
                <span className="shrink-0 font-mono text-[12.5px] text-[var(--color-ink)]">
                  {count}
                </span>
              </div>
              <div className="h-2.5 w-full rounded-[4px] bg-[var(--color-surface-2)]">
                <div
                  className="h-full rounded-[4px] transition-[width] duration-500"
                  style={{ width: `${Math.max((count / max) * 100, 3)}%`, background: SEQUENTIAL }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function LatencyBreakdown({ data }: { data: Dashboard }) {
  const stages = [
    { label: "Retrieval", ms: data.average_retrieval_ms },
    { label: "Rerank", ms: data.average_rerank_ms },
    { label: "Generation", ms: data.average_generation_ms },
  ];
  const max = Math.max(1, ...stages.map((s) => s.ms));

  return (
    <section className="panel p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Mean latency by pipeline stage</h2>
      <p className="mt-0.5 mb-4 text-[12.5px] text-[var(--color-ink-3)]">
        Where time goes on an average answered query
      </p>
      <ul className="space-y-3">
        {stages.map((stage) => (
          <li key={stage.label}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="text-[12.5px] text-[var(--color-ink-2)]">{stage.label}</span>
              <span className="font-mono text-[12.5px] text-[var(--color-ink)]">
                {stage.ms.toFixed(1)}ms
              </span>
            </div>
            <div className="h-2.5 w-full rounded-[4px] bg-[var(--color-surface-2)]">
              <div
                className="h-full rounded-[4px] transition-[width] duration-500"
                style={{ width: `${Math.max((stage.ms / max) * 100, 3)}%`, background: SEQUENTIAL }}
              />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function RecentQueries({ rows }: { rows: Dashboard["recent_queries"] }) {
  if (!rows.length) return null;
  return (
    <section className="panel overflow-hidden">
      <div className="border-b border-[var(--color-line)] px-4 py-3.5 sm:px-5">
        <h2 className="text-sm font-semibold">Recent queries</h2>
        <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-3)]">
          Every row is a logged, replayable query
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-[13px]">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-[var(--color-ink-3)]">
              <th className="px-4 py-2.5 font-medium sm:px-5">Query</th>
              <th className="px-3 py-2.5 font-medium">Mode</th>
              <th className="px-3 py-2.5 font-medium">Groundedness</th>
              <th className="px-3 py-2.5 font-medium">Flags</th>
              <th className="px-3 py-2.5 text-right font-medium sm:px-5">Latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.id}
                className="border-b border-[var(--color-line)] last:border-0 hover:bg-[var(--color-surface-2)]"
              >
                <td className="max-w-[320px] truncate px-4 py-2.5 sm:px-5">{row.query}</td>
                <td className="px-3 py-2.5 text-[var(--color-ink-3)]">
                  {row.blocked ? (
                    <span className="inline-flex items-center gap-1 text-[var(--color-danger)]">
                      <AlertTriangle size={11} />
                      blocked
                    </span>
                  ) : (
                    row.llm_mode
                  )}
                </td>
                <td className="px-3 py-2.5">
                  <GroundednessMeter value={row.groundedness_score} compact />
                </td>
                <td className="px-3 py-2.5 text-[11px] text-[var(--color-ink-3)]">
                  {row.flags.length ? row.flags.join(", ") : "-"}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-[var(--color-ink-3)] sm:px-5">
                  {row.total_ms}ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
