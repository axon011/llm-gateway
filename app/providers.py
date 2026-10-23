"""LLM providers. Each returns the text plus the provider's OWN token usage.

The usage numbers here are the only numbers the rest of the gateway is allowed to trust:
no len(prompt)//4 estimates anywhere.

`stream()` yields text chunks and ends with exactly one Completion carrying the totals, so the
caller can forward chunks as they arrive and still price the whole call afterwards.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import httpx

from .schemas import GenerationOptions, Message

# Longest Groq `retry-after` worth waiting out before giving up, in seconds.
MAX_RETRY_AFTER_S = 10.0


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str

    async def complete(self, messages: list[Message], opts: GenerationOptions | None = None) -> Completion: ...

    def stream(
        self, messages: list[Message], opts: GenerationOptions | None = None
    ) -> AsyncIterator[str | Completion]: ...


class FakeProvider:
    """Deterministic provider for tests and offline runs. Reports fixed usage."""

    name = "fake"

    async def complete(self, messages: list[Message], opts: GenerationOptions | None = None) -> Completion:
        last = messages[-1].content if messages else ""
        return Completion(text=f"echo: {last}", model="fake-1", input_tokens=7, output_tokens=3)

    async def stream(self, messages: list[Message], opts: GenerationOptions | None = None):
        done = await self.complete(messages, opts)
        for part in ("echo: ", done.text[len("echo: "):]):
            if part:
                yield part
        yield done


class GeminiProvider:
    """Google Generative Language API, REST, async. Usage comes from `usageMetadata`."""

    name = "gemini"
    _base = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, timeout_s: float, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        self._key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def complete(self, messages: list[Message], opts: GenerationOptions | None = None) -> Completion:
        model = (opts and opts.model) or self._model
        system = [m.content for m in messages if m.role == "system"]
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        if opts and opts.temperature is not None:
            body["generationConfig"] = {"temperature": opts.temperature}

        resp = await self._client.post(
            f"{self._base}/{model}:generateContent",
            headers={"x-goog-api-key": self._key},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"].get("parts", [])
        )
        usage = data.get("usageMetadata", {})
        return Completion(
            text=text,
            model=data.get("modelVersion", model),
            input_tokens=int(usage.get("promptTokenCount", 0)),
            # thinking tokens are billed as output
            output_tokens=int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)),
        )

    async def stream(self, messages: list[Message], opts: GenerationOptions | None = None):
        # Gemini is a fallback: one chunk is enough, the totals are what matter.
        done = await self.complete(messages, opts)
        if done.text:
            yield done.text
        yield done

    async def aclose(self) -> None:
        await self._client.aclose()


class GroqProvider:
    """Groq's OpenAI-compatible chat completions API. Usage comes from `usage`."""

    name = "groq"
    _url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout_s: float, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty")
        self._key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    def _body(self, messages: list[Message], opts: GenerationOptions | None, stream: bool) -> dict:
        body: dict = {
            "model": (opts and opts.model) or self._model,
            "messages": [m.model_dump() for m in messages],
        }
        if opts and opts.temperature is not None:
            body["temperature"] = opts.temperature
        if opts and opts.reasoning_effort:
            body["reasoning_effort"] = opts.reasoning_effort
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    async def complete(self, messages: list[Message], opts: GenerationOptions | None = None) -> Completion:
        resp = await self._client.post(self._url, headers=self._headers(), json=self._body(messages, opts, False))
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self._model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def stream(self, messages: list[Message], opts: GenerationOptions | None = None):
        body = self._body(messages, opts, True)
        for attempt in (1, 2):
            async with self._client.stream("POST", self._url, headers=self._headers(), json=body) as resp:
                if resp.status_code == 429 and attempt == 1:
                    # The free tier's per-minute budget: a short wait is normal, a long one is an error.
                    try:
                        wait = float(resp.headers.get("retry-after", ""))
                    except ValueError:
                        wait = -1.0
                    if 0 <= wait <= MAX_RETRY_AFTER_S:
                        await resp.aread()
                        await asyncio.sleep(wait + 0.25)
                        continue
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                async for item in _parse_openai_sse(resp.aiter_lines(), body["model"]):
                    yield item
                return

    async def aclose(self) -> None:
        await self._client.aclose()


async def _parse_openai_sse(lines: AsyncIterator[str], requested_model: str):
    """OpenAI-style SSE: `data: {json}` lines ending in `data: [DONE]`.

    Usage arrives on the last chunk, either as top-level `usage` (stream_options.include_usage)
    or Groq's own `x_groq.usage`. If neither ever arrives the tokens are reported as 0 rather
    than estimated, so a missing price shows up as zero usage, never as a made-up number.
    """
    parts: list[str] = []
    model = requested_model
    usage: dict = {}
    async for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        model = ev.get("model") or model
        usage = ev.get("usage") or (ev.get("x_groq") or {}).get("usage") or usage
        for choice in ev.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                parts.append(text)
                yield text
    yield Completion(
        text="".join(parts),
        model=model,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
    )
