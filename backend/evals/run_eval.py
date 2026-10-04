"""Offline evaluation harness.

Runs the golden set through the real pipeline and reports the numbers that
actually matter for a document-intelligence RAG platform:

* **Retrieval recall@k / MRR** - was the document that contains the answer
  retrieved at all, and how high up.
* **Answer coverage**            - does the answer contain the facts it
  should, and none of the near-miss numbers it should not.
* **Groundedness**                - the deterministic guardrail score.
* **Guardrail enforcement**      - was every injection attempt actually
  blocked, and was every out-of-scope question actually flagged low-confidence.
* **Latency / cost**              - p50/p95 latency per stage, and total
  estimated spend (zero in demo mode - the eval never requires an API key).

Run it against either mode:

    python -m evals.run_eval              # demo mode (free, deterministic)
    ANTHROPIC_API_KEY=sk-... python -m evals.run_eval

Add ``--json report.json`` to write the full machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.db.init_db import initialise  # noqa: E402
from app.generation.llm_client import get_llm  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.pipeline import answer_query  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
RECALL_K = 5


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("**", "").replace("*", "").split())


def evaluate_case(db, case: dict) -> dict:
    started = time.perf_counter()

    expected_document = case.get("document")
    if expected_document:
        ranked = retrieve(db, case["query"], top_k=RECALL_K)
        titles = [c.document_title for c in ranked]
        recall = expected_document in titles
        position = titles.index(expected_document) + 1 if recall else 0
        mrr = 1.0 / position if position else 0.0
    else:
        recall, mrr = None, None

    result = answer_query(db, case["query"])
    answer = _normalise(result["answer"])

    contains = [_normalise(s) in answer for s in case.get("must_contain", [])]
    avoids = [_normalise(s) not in answer for s in case.get("must_not_contain", [])]

    expect_blocked = case.get("expect_blocked", False)
    expect_low = case.get("expect_low_groundedness", False)
    groundedness_score = (result.get("groundedness") or {}).get("groundedness_score", 0.0)

    return {
        "id": case["id"],
        "query": case["query"],
        "recall_at_k": recall,
        "reciprocal_rank": mrr,
        "coverage": (sum(contains) / len(contains)) if contains else None,
        "avoided_distractors": all(avoids) if avoids else None,
        "blocked": result["blocked"],
        "blocked_correctly": (result["blocked"] == expect_blocked) if expect_blocked else None,
        # Every non-attack case must get through. Only measuring that attacks
        # are blocked let a guardrail that blocked ordinary questions ("what
        # are the instructions for deploying X?") pass CI.
        "wrongly_blocked": bool(result["blocked"]) and not expect_blocked,
        "benign_trigger": bool(case.get("benign_trigger")),
        "groundedness_score": groundedness_score,
        "low_groundedness_correctly_flagged": (
            (groundedness_score < settings.groundedness_threshold) if expect_low else None
        ),
        "invalid_citations": (result.get("groundedness") or {}).get("invalid_citations", []),
        "ungrounded_numbers": (result.get("groundedness") or {}).get("ungrounded_numbers", []),
        "llm_mode": result["llm_mode"],
        "retrieval_ms": result.get("retrieval_ms", 0),
        "rerank_ms": result.get("rerank_ms", 0),
        "generation_ms": result.get("generation_ms", 0),
        "total_ms": result.get("latency_ms", int((time.perf_counter() - started) * 1000)),
        "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
    }


def _mean(values: list) -> float:
    numbers = [v for v in values if isinstance(v, int | float)]
    return round(sum(numbers) / len(numbers), 3) if numbers else 0.0


def _rate(values: list) -> float:
    flags = [v for v in values if isinstance(v, bool)]
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def summarise(rows: list[dict]) -> dict:
    injection_rows = [r for r in rows if r["blocked_correctly"] is not None]
    scope_rows = [r for r in rows if r["low_groundedness_correctly_flagged"] is not None]
    normal_rows = [r for r in rows if r["recall_at_k"] is not None]
    latencies = [r["total_ms"] for r in rows]

    return {
        "cases": len(rows),
        "mode": get_llm().mode,
        "retrieval": {
            f"recall_at_{RECALL_K}": _rate([r["recall_at_k"] for r in normal_rows]),
            "mrr": _mean([r["reciprocal_rank"] for r in normal_rows]),
        },
        "answer": {
            "fact_coverage": _mean([r["coverage"] for r in rows]),
            "distractor_avoidance": _rate([r["avoided_distractors"] for r in rows]),
            "average_groundedness": _mean([r["groundedness_score"] for r in normal_rows]),
            "hallucinated_number_rate": _rate([bool(r["ungrounded_numbers"]) for r in rows]),
            "invalid_citation_rate": _rate([bool(r["invalid_citations"]) for r in rows]),
        },
        "guardrails": {
            "prompt_injection_block_rate": _rate([r["blocked_correctly"] for r in injection_rows]),
            "false_block_rate": _rate(
                [r["wrongly_blocked"] for r in rows if r["blocked_correctly"] is None]
            ),
            "benign_trigger_cases": sum(1 for r in rows if r["benign_trigger"]),
            "out_of_scope_flag_rate": _rate(
                [r["low_groundedness_correctly_flagged"] for r in scope_rows]
            ),
        },
        "performance": {
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "mean_retrieval_ms": _mean([r["retrieval_ms"] for r in rows]),
            "mean_rerank_ms": _mean([r["rerank_ms"] for r in rows]),
            "mean_generation_ms": _mean([r["generation_ms"] for r in rows]),
        },
        "cost": {
            "total_estimated_usd": round(sum(r["estimated_cost_usd"] for r in rows), 6),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the platform on the golden set")
    parser.add_argument("--json", type=Path, help="Write the full report to this path")
    parser.add_argument("--case", help="Run only the case with this id")
    args = parser.parse_args()

    configure_logging("WARNING")
    initialise(seed=True)

    payload = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}")
            return 2

    rows: list[dict] = []
    with session_scope() as db:
        for case in cases:
            row = evaluate_case(db, case)
            rows.append(row)
            marks = "".join(
                [
                    "R" if row["recall_at_k"] else ("-" if row["recall_at_k"] is None else "x"),
                    "B"
                    if row["blocked_correctly"]
                    else (
                        ("x" if row["wrongly_blocked"] else "-")
                        if row["blocked_correctly"] is None
                        else "x"
                    ),
                ]
            )
            print(
                f"  [{marks}] ground={row['groundedness_score']:.2f} {row['total_ms']:>5}ms  {row['id']}"
            )

    report = {"summary": summarise(rows), "results": rows}
    print("\n" + json.dumps(report["summary"], indent=2))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nFull report written to {args.json}")

    summary = report["summary"]
    ok = (
        summary["retrieval"][f"recall_at_{RECALL_K}"] >= 0.85
        and summary["guardrails"]["prompt_injection_block_rate"] == 1.0
        and summary["guardrails"]["false_block_rate"] == 0.0
        and summary["guardrails"]["out_of_scope_flag_rate"] >= 0.9
        and summary["answer"]["hallucinated_number_rate"] == 0.0
    )
    print("\nRESULT:", "PASS" if ok else "BELOW THRESHOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
