import type { ReactNode } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle: string;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-[var(--color-line)] px-5 py-5 sm:px-7">
      <div className="min-w-0">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        <p className="mt-0.5 max-w-2xl text-sm text-[var(--color-ink-3)]">{subtitle}</p>
      </div>
      {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
    </header>
  );
}

const BANDS = [
  { min: 0.75, label: "High confidence", color: "var(--color-ok)" },
  { min: 0.5, label: "Moderate confidence", color: "var(--color-warn)" },
  { min: 0, label: "Low confidence", color: "var(--color-danger)" },
];

export function groundednessBand(value: number) {
  return BANDS.find((band) => value >= band.min) ?? BANDS[2];
}

export function GroundednessMeter({ value, compact }: { value: number; compact?: boolean }) {
  const band = groundednessBand(value);
  const pct = Math.round(value * 100);

  if (compact) {
    return (
      <span className="chip" style={{ color: band.color, borderColor: `${band.color}55` }}>
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: band.color }} />
        {pct}%
      </span>
    );
  }

  return (
    <div className="min-w-[168px]">
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="label">{band.label}</span>
        <span className="font-mono text-sm font-semibold" style={{ color: band.color }}>
          {pct}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-3)]">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.max(pct, 2)}%`, background: band.color }}
        />
      </div>
    </div>
  );
}

const FLAG_LABELS: Record<string, string> = {
  invalid_citation: "Invalid citation",
  ungrounded_numbers: "Ungrounded figures",
  insufficient_evidence: "Insufficient evidence",
  question_not_covered_by_corpus: "Outside the knowledge base",
  low_groundedness_warning: "Low groundedness warning",
  context_injection_detected: "Context injection detected",
  prompt_injection_in_query: "Prompt injection blocked",
};

export function FlagList({ flags }: { flags: string[] }) {
  if (!flags.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {flags.map((flag) => {
        const danger = flag.includes("injection") || flag.includes("invalid");
        const color = danger ? "var(--color-danger)" : "var(--color-warn)";
        return (
          <span
            key={flag}
            className="chip"
            style={{ color, borderColor: `${color}55` }}
            title={flag}
          >
            {FLAG_LABELS[flag] ?? flag.replace(/_/g, " ")}
          </span>
        );
      })}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  body,
}: {
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      <div className="grid h-11 w-11 place-items-center rounded-xl bg-[var(--color-surface-2)] text-[var(--color-ink-3)]">
        {icon}
      </div>
      <div>
        <div className="text-sm font-medium">{title}</div>
        <p className="mx-auto mt-1 max-w-sm text-sm text-[var(--color-ink-3)]">{body}</p>
      </div>
    </div>
  );
}
