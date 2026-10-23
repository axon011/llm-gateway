"""Per-model list prices, dated, so a cost is always traceable to a source.

Cost = provider-reported tokens x the list price below. Nothing is estimated: a model missing
from the table gets cost None, never a guess. On a free tier the real bill is 0; the cost here is
what the same traffic would cost on the paid tier, which is the number that matters for scaling.

Update a price by adding a new entry with a later `since` date; old calls keep the price that was
valid when they ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Price:
    since: date
    input_per_m: float  # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens
    source: str


# checked 2026-10-22
PRICES: dict[str, list[Price]] = {
    "openai/gpt-oss-120b": [Price(date(2026, 10, 22), 0.15, 0.60, "https://console.groq.com/docs/models")],
    "openai/gpt-oss-20b": [Price(date(2026, 10, 22), 0.075, 0.30, "https://console.groq.com/docs/models")],
    "gemini-2.5-flash": [Price(date(2026, 10, 22), 0.30, 2.50, "https://ai.google.dev/gemini-api/docs/pricing")],
    "fake-1": [Price(date(2026, 1, 1), 1.0, 2.0, "test fixture")],
}


def price_for(model: str, on: date) -> Price | None:
    """The price valid on `on`. Unknown model or a date before the first entry -> None."""
    entries = PRICES.get(model)
    if entries is None and "/" not in model:
        # Gemini reports versioned names (gemini-2.5-flash-001); fall back to the family name.
        entries = next((v for k, v in PRICES.items() if model.startswith(k)), None)
    if not entries:
        return None
    valid = [p for p in entries if p.since <= on]
    return max(valid, key=lambda p: p.since) if valid else None


def cost_usd(model: str, input_tokens: int, output_tokens: int, on: date) -> float | None:
    p = price_for(model, on)
    if p is None:
        return None
    return round(input_tokens * p.input_per_m / 1e6 + output_tokens * p.output_per_m / 1e6, 8)
