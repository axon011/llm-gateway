"""GeminiProvider parses the REST response and takes usage ONLY from usageMetadata."""

import httpx
import pytest

from app.providers import GeminiProvider
from app.schemas import Message


@pytest.mark.anyio
async def test_gemini_uses_provider_usage_not_estimates():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = request.read()
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Guten "}, {"text": "Tag"}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 2},
                "modelVersion": "gemini-2.5-flash-001",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p = GeminiProvider("k", "gemini-2.5-flash", 5.0, client=client)
    c = await p.complete([Message(role="system", content="Du bist Tutor."), Message(role="user", content="Hi")])

    assert c.text == "Guten Tag"
    assert (c.input_tokens, c.output_tokens) == (12, 2)
    assert c.model == "gemini-2.5-flash-001"
    assert captured["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert b"systemInstruction" in captured["json"]


def test_gemini_refuses_empty_key():
    with pytest.raises(ValueError):
        GeminiProvider("", "m", 5.0)


@pytest.mark.anyio
async def test_groq_uses_provider_usage():
    from app.providers import GroqProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"role": "assistant", "content": "Guten Tag"}}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25},
            },
        )

    p = GroqProvider("k", "llama-3.3-70b-versatile", 5.0, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    c = await p.complete([Message(role="user", content="Hi")])
    assert c.text == "Guten Tag"
    assert (c.input_tokens, c.output_tokens) == (21, 4)
