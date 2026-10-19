"""LLM providers. Each returns the text plus the provider's OWN token usage.

The usage numbers here are the only numbers the rest of the gateway is allowed to trust:
no len(prompt)//4 estimates anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from .schemas import Message


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str

    async def complete(self, messages: list[Message]) -> Completion: ...


class FakeProvider:
    """Deterministic provider for tests and offline runs. Reports fixed usage."""

    name = "fake"

    async def complete(self, messages: list[Message]) -> Completion:
        last = messages[-1].content if messages else ""
        return Completion(text=f"echo: {last}", model="fake-1", input_tokens=7, output_tokens=3)


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

    async def complete(self, messages: list[Message]) -> Completion:
        system = [m.content for m in messages if m.role == "system"]
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}

        resp = await self._client.post(
            f"{self._base}/{self._model}:generateContent",
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
            model=data.get("modelVersion", self._model),
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
        )

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

    async def complete(self, messages: list[Message]) -> Completion:
        resp = await self._client.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "messages": [m.model_dump() for m in messages]},
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self._model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
