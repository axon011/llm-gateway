"""llm-gateway: one HTTP front for LLM calls that traces every call to Langfuse.

Module 0 scope: /health and POST /v1/chat with one provider and one trace per call.
Cost, guardrails, A/B variants and the judge land in later modules.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from .config import settings
from .providers import FakeProvider, GeminiProvider, GroqProvider, Provider
from .schemas import ChatRequest, ChatResponse, Usage
from .tracing import LangfuseTracer, NoopTracer, Tracer


def build_provider() -> Provider:
    if settings.llm_provider == "fake":
        return FakeProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.provider_timeout_s)
    if settings.llm_provider == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model, settings.provider_timeout_s)
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")


def build_tracer() -> Tracer:
    if settings.tracing_enabled:
        return LangfuseTracer(settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host)
    return NoopTracer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = build_provider()
    app.state.tracer = build_tracer()
    yield
    app.state.tracer.flush()
    aclose = getattr(app.state.provider, "aclose", None)
    if aclose:
        await aclose()


app = FastAPI(title="llm-gateway", version="0.0.1", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "provider": app.state.provider.name,
        "tracing": settings.tracing_enabled,
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    provider: Provider = app.state.provider
    tracer: Tracer = app.state.tracer
    with tracer.chat(req) as trace:
        try:
            completion = await provider.complete(req.messages)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"provider {provider.name}: {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=504, detail=f"provider {provider.name}: {type(e).__name__}") from e
        trace.record(provider.name, completion)
        return ChatResponse(
            text=completion.text,
            model=completion.model,
            provider=provider.name,
            usage=Usage(input_tokens=completion.input_tokens, output_tokens=completion.output_tokens),
            trace_id=trace.trace_id,
        )
