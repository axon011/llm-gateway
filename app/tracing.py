"""Langfuse tracing behind a tiny interface so tests and key-less runs use a no-op.

Langfuse Python SDK v4 (OpenTelemetry-based):
- `Langfuse(public_key, secret_key, host)` is the client.
- `propagate_attributes(session_id=, user_id=, tags=, metadata=)` sets trace-level attributes for
  everything opened inside it (v3's `update_trace` is gone in v4).
- `start_as_current_observation(as_type="span" | "generation", ...)` opens an observation.
- The root observation's input/output is the trace's input/output (set_trace_io is deprecated).
- Usage is written as `usage_details` from the provider's real counts.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator, Protocol

from .providers import Completion
from .schemas import ChatRequest


class ChatTrace(Protocol):
    trace_id: str | None

    def record(self, provider_name: str, completion: Completion) -> None: ...


class Tracer(Protocol):
    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator[ChatTrace]: ...

    def flush(self) -> None: ...


class _NoopChatTrace:
    trace_id = None

    def record(self, provider_name: str, completion: Completion) -> None:
        return None


class NoopTracer:
    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator[_NoopChatTrace]:
        yield _NoopChatTrace()

    def flush(self) -> None:
        return None


class LangfuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str, *, client=None):
        if client is None:
            from langfuse import Langfuse  # lazy: the no-op path needs no SDK

            client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._lf = client

    @contextmanager
    def chat(self, req: ChatRequest) -> Iterator["_LangfuseChatTrace"]:
        from langfuse import propagate_attributes

        with ExitStack() as stack:
            stack.enter_context(
                propagate_attributes(
                    session_id=req.session_id,
                    user_id=req.user_id,
                    tags=[req.route],
                    metadata={"route": req.route},
                    trace_name=f"chat:{req.route}",
                )
            )
            span = stack.enter_context(
                self._lf.start_as_current_observation(
                    as_type="span",
                    name=f"chat:{req.route}",
                    input=[m.model_dump() for m in req.messages],
                )
            )
            # v4 deprecates set_trace_io; the root observation's input/output is what the trace shows.
            yield _LangfuseChatTrace(self._lf, span, req)

    def flush(self) -> None:
        self._lf.flush()


class _LangfuseChatTrace:
    def __init__(self, lf, span, req: ChatRequest):
        self._lf = lf
        self._span = span
        self._req = req
        self.trace_id = lf.get_current_trace_id()

    def record(self, provider_name: str, completion: Completion) -> None:
        with self._lf.start_as_current_observation(
            as_type="generation",
            name=f"{provider_name}.generate",
            model=completion.model,
            input=[m.model_dump() for m in self._req.messages],
        ) as gen:
            gen.update(
                output=completion.text,
                usage_details={"input": completion.input_tokens, "output": completion.output_tokens},
            )
        self._span.update(output=completion.text)
