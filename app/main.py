"""llm-gateway: one HTTP front for LLM calls that traces and prices every call.

Module 2 scope: /health, POST /v1/chat, POST /v1/chat/stream (SSE), GET /v1/usage.
Every completed call gets a cost from the dated price table (app/pricing.py), lands in Langfuse
and in the SQLite usage ledger. Guardrails, A/B variants and the judge land in later modules.
"""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from .config import settings
from .pricing import cost_usd
from .providers import Completion, FakeProvider, GeminiProvider, GroqProvider, Provider
from .schemas import ChatRequest, ChatResponse, Usage
from .tracing import LangfuseTracer, NoopTracer, Tracer
from .usage import UsageStore


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
    app.state.usage = UsageStore(settings.usage_db_path)
    yield
    app.state.tracer.flush()
    aclose = getattr(app.state.provider, "aclose", None)
    if aclose:
        await aclose()


app = FastAPI(title="llm-gateway", version="0.2.0", lifespan=lifespan)


def require_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.gateway_api_key:
        return
    # compare_digest takes the same time wherever the strings differ, so response timing
    # does not leak how many leading characters of a guessed key were right.
    expected = f"Bearer {settings.gateway_api_key}".encode()
    if not hmac.compare_digest((authorization or "").encode(), expected):
        raise HTTPException(status_code=401, detail="missing or wrong gateway key")


def check_model(req: ChatRequest) -> None:
    if not settings.model_allowed(req.model):
        raise HTTPException(status_code=400, detail=f"model {req.model!r} is not allowed on this gateway")


async def _finish(req: ChatRequest, provider: Provider, trace, completion: Completion) -> Usage:
    """Price, trace and ledger one completed call. Shared by the plain and streaming paths."""
    cost = cost_usd(completion.model, completion.input_tokens, completion.output_tokens, datetime.now(timezone.utc).date())
    trace.record(provider.name, completion, cost)
    await app.state.usage.record(
        route=req.route,
        session_id=req.session_id,
        user_id=req.user_id,
        provider=provider.name,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=cost,
        trace_id=trace.trace_id,
    )
    return Usage(input_tokens=completion.input_tokens, output_tokens=completion.output_tokens, cost_usd=cost)


def _provider_error(provider: Provider, e: httpx.HTTPError) -> HTTPException:
    if isinstance(e, httpx.HTTPStatusError):
        return HTTPException(status_code=502, detail=f"provider {provider.name}: {e.response.status_code}")
    return HTTPException(status_code=504, detail=f"provider {provider.name}: {type(e).__name__}")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "provider": app.state.provider.name,
        "tracing": settings.tracing_enabled,
    }


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_key)])
async def chat(req: ChatRequest) -> ChatResponse:
    check_model(req)
    provider: Provider = app.state.provider
    tracer: Tracer = app.state.tracer
    with tracer.chat(req) as trace:
        try:
            completion = await provider.complete(req.messages, req)
        except httpx.HTTPError as e:
            raise _provider_error(provider, e) from e
        usage = await _finish(req, provider, trace, completion)
        return ChatResponse(
            text=completion.text,
            model=completion.model,
            provider=provider.name,
            usage=usage,
            trace_id=trace.trace_id,
        )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/stream", dependencies=[Depends(require_key)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE: `delta` events carry text chunks; one final `done` (usage, cost, trace) or `error`."""
    check_model(req)  # before the stream opens, so a bad model is a plain 400, not an SSE error
    provider: Provider = app.state.provider
    tracer: Tracer = app.state.tracer

    async def events():
        with tracer.chat(req) as trace:
            try:
                async for item in provider.stream(req.messages, req):
                    if isinstance(item, Completion):
                        usage = await _finish(req, provider, trace, item)
                        yield _sse("done", {"model": item.model, "provider": provider.name,
                                            "usage": usage.model_dump(), "trace_id": trace.trace_id})
                    else:
                        yield _sse("delta", {"text": item})
            except httpx.HTTPError as e:
                err = _provider_error(provider, e)
                yield _sse("error", {"status": err.status_code, "detail": err.detail})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"cache-control": "no-cache"})


@app.get("/v1/usage", dependencies=[Depends(require_key)])
async def usage(days: int = Query(default=7, ge=1, le=366)) -> dict:
    """Totals and per day/route/model groups from the usage ledger."""
    return app.state.usage.summary(days)
