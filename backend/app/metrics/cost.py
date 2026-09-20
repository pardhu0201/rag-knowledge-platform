"""Token-cost estimation.

Anthropic pricing, per million tokens (input/output), current as of this
project's build. Kept as a small table rather than a live lookup so the cost
dashboard works offline and in demo mode with zero API calls.
"""

from __future__ import annotations

# (input $/MTok, output $/MTok)
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_PRICING = (5.00, 25.00)  # Opus-tier, used if the model isn't in the table


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICING.get(model, DEFAULT_PRICING)
    return round(
        (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price, 6
    )
