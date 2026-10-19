"""Langfuse tracing behind a tiny interface so tests and key-less runs use a no-op.

Langfuse Python SDK v3+/v4: one client (`Langfuse(...)`), observations opened with
`start_as_current_observation(as_type="span" | "generation")`, trace attributes set with
`update_trace(...)`. Usage is written as `usage_details` from the provider's real counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol

from .providers import Completion
from .schemas import ChatRequest


class Tracer(Protocol):
    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator["ChatTrace"]: ...

    def flush(self) -> None: ...


class ChatTrace(Protocol):
    trace_id: str | None

    def record(self, model: str, completion: Completion) -> None: ...


class _NoopChatTrace:
    trace_id = None

    def record(self, model: str, completion: Completion) -> None:
        return None


class NoopTracer:
    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator[_NoopChatTrace]:
        yield _NoopChatTrace()

    def flush(self) -> None:
        return None


class LangfuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str):
        from langfuse import Langfuse  # imported lazily so the no-op path needs no SDK

        self._lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator["_LangfuseChatTrace"]:
        with self._lf.start_as_current_observation(as_type="span", name=f"chat:{req.route}") as span:
            span.update_trace(
                session_id=req.session_id,
                user_id=req.user_id,
                input=[m.model_dump() for m in req.messages],
                metadata={"route": req.route},
                tags=[req.route],
            )
            yield _LangfuseChatTrace(self._lf, span, req)

    def flush(self) -> None:
        self._lf.flush()


class _LangfuseChatTrace:
    def __init__(self, lf, span, req: ChatRequest):
        self._lf = lf
        self._span = span
        self._req = req
        self.trace_id = span.trace_id

    def record(self, model: str, completion: Completion) -> None:
        with self._lf.start_as_current_observation(
            as_type="generation",
            name=f"{model}.generate",
            model=completion.model,
            input=[m.model_dump() for m in self._req.messages],
        ) as gen:
            gen.update(
                output=completion.text,
                usage_details={"input": completion.input_tokens, "output": completion.output_tokens},
            )
        self._span.update_trace(output=completion.text)
