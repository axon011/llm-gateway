"""Drive the real LangfuseTracer against the SDK with the OTel exporter swapped for an in-memory one.

Guards the v4 API surface we depend on (propagate_attributes, start_as_current_observation,
get_current_trace_id) AND asserts what actually lands on the exported spans:
session/user attributes, the generation's model and real usage. No network.
"""

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.providers import Completion
from app.schemas import ChatRequest, Message
from app.tracing import LangfuseTracer


@pytest.fixture()
def tracer_and_exporter():
    from langfuse import Langfuse

    client = Langfuse(public_key="pk-test", secret_key="sk-test", host="http://127.0.0.1:9", tracing_enabled=True)
    exporter = InMemorySpanExporter()
    # Replace the HTTP exporter inside the SDK's own span processor. `span_exporter` is a read-only
    # property, and the attribute that holds the exporter differs across OTel SDK versions
    # (BatchSpanProcessor.span_exporter vs ._batch_processor._exporter), so swap by type.
    processor = client._resources.tracer_provider._active_span_processor._span_processors[0]
    replaced = _swap_exporter(processor, exporter)
    assert replaced, "could not find the OTLP exporter inside LangfuseSpanProcessor"
    client._resources.span_exporter = exporter
    yield LangfuseTracer("pk-test", "sk-test", "http://127.0.0.1:9", client=client), exporter
    exporter.clear()


def _swap_exporter(obj, exporter, depth=0) -> bool:
    from opentelemetry.sdk.trace.export import SpanExporter

    if depth > 3:
        return False
    done = False
    for key, val in list(vars(obj).items()):
        if isinstance(val, SpanExporter) and not isinstance(val, InMemorySpanExporter):
            setattr(obj, key, exporter)
            done = True
        elif hasattr(val, "__dict__") and key.startswith("_") and not isinstance(val, (str, bytes)):
            done = _swap_exporter(val, exporter, depth + 1) or done
    return done


def _attrs(span) -> dict:
    return dict(span.attributes or {})


def test_trace_carries_session_user_route_and_real_usage(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    req = ChatRequest(
        messages=[Message(role="user", content="Hallo")],
        session_id="s1",
        user_id="u1",
        route="tutor.reply",
    )
    with tracer.chat(req) as t:
        assert isinstance(t.trace_id, str) and len(t.trace_id) == 32
        t.record("fake", Completion(text="Guten Tag", model="fake-1", input_tokens=7, output_tokens=3), 0.000013)
    tracer.flush()

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert set(by_name) == {"chat:tutor.reply", "fake.generate"}

    gen = _attrs(by_name["fake.generate"])
    assert gen["langfuse.observation.type"] == "generation"
    assert gen["langfuse.observation.model.name"] == "fake-1"
    usage = json.loads(gen["langfuse.observation.usage_details"])
    assert usage == {"input": 7, "output": 3}
    assert json.loads(gen["langfuse.observation.cost_details"]) == {"total": 0.000013}

    for s in spans:  # propagate_attributes puts trace attrs on every span
        a = _attrs(s)
        assert a["session.id"] == "s1"
        assert a["user.id"] == "u1"
        assert "tutor.reply" in a["langfuse.trace.tags"]
        assert a["langfuse.trace.metadata.route"] == "tutor.reply"
        assert a["langfuse.trace.name"] == "chat:tutor.reply"

    root = _attrs(by_name["chat:tutor.reply"])
    assert root["langfuse.observation.output"] == "Guten Tag"  # plain strings are stored unquoted
