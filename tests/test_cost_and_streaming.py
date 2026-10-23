"""Module 2: dated prices, Groq SSE parsing with real usage, and the 429 wait-and-retry. No network."""

import json
from datetime import date

import httpx
import pytest

from app.pricing import cost_usd, price_for
from app.providers import Completion, GroqProvider
from app.schemas import GenerationOptions, Message

TODAY = date(2026, 10, 22)


def test_cost_is_tokens_times_list_price():
    # gpt-oss-120b: 0.15 in / 0.60 out per 1M
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000, TODAY) == pytest.approx(0.75)
    assert cost_usd("openai/gpt-oss-120b", 1200, 300, TODAY) == pytest.approx(0.00036)


def test_unknown_model_or_date_before_first_price_is_none_not_a_guess():
    assert cost_usd("some/new-model", 100, 100, TODAY) is None
    assert cost_usd("openai/gpt-oss-120b", 100, 100, date(2020, 1, 1)) is None


def test_gemini_versioned_name_falls_back_to_family_price():
    assert price_for("gemini-2.5-flash-001", TODAY) is not None


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"


def _chunk(text: str) -> dict:
    return {"model": "openai/gpt-oss-120b", "choices": [{"delta": {"content": text}}]}


async def _collect(p: GroqProvider, opts: GenerationOptions | None = None):
    items = [i async for i in p.stream([Message(role="user", content="Hi")], opts)]
    return [i for i in items if isinstance(i, str)], items[-1]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "final",
    [
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9}},  # stream_options.include_usage
        {"choices": [], "x_groq": {"usage": {"prompt_tokens": 40, "completion_tokens": 9}}},  # Groq's own field
    ],
)
async def test_groq_stream_yields_chunks_then_completion_with_provider_usage(final):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.read()))
        return httpx.Response(200, content=_sse(_chunk("Guten "), _chunk("Tag"), final))

    p = GroqProvider("k", "openai/gpt-oss-20b", 5.0, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    chunks, done = await _collect(p, GenerationOptions(model="openai/gpt-oss-120b", reasoning_effort="low"))

    assert chunks == ["Guten ", "Tag"]
    assert isinstance(done, Completion)
    assert (done.text, done.model, done.input_tokens, done.output_tokens) == ("Guten Tag", "openai/gpt-oss-120b", 40, 9)
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["reasoning_effort"] == "low"
    assert sent["stream_options"] == {"include_usage": True}
    assert "temperature" not in sent  # unset options keep the provider default


@pytest.mark.anyio
async def test_groq_stream_waits_out_a_short_429_once(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("app.providers.asyncio.sleep", fake_sleep)
    calls = iter(
        [
            httpx.Response(429, headers={"retry-after": "2"}, text="rate limited"),
            httpx.Response(200, content=_sse(_chunk("ok"), {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})),
        ]
    )
    p = GroqProvider("k", "m", 5.0, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: next(calls))))
    chunks, done = await _collect(p)
    assert chunks == ["ok"]
    assert slept == [pytest.approx(2.25)]


@pytest.mark.anyio
async def test_groq_stream_raises_on_a_long_429():
    p = GroqProvider(
        "k", "m", 5.0,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "60"}))),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await _collect(p)
