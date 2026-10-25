"""Gateway tests: run with the fake provider and no Langfuse keys. No network."""

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ["LLM_PROVIDER"] = "fake"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

from app.main import app  # noqa: E402  (env must be set before import)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_provider_and_tracing_off(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "provider": "fake", "tracing": False}


def test_chat_returns_text_and_provider_usage(client):
    r = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "Hallo"}], "session_id": "s1", "user_id": "u1", "route": "tutor.reply"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "echo: Hallo"
    assert body["provider"] == "fake"
    # fake-1 is priced at 1.0 / 2.0 USD per 1M tokens: 7 * 1e-6 + 3 * 2e-6
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3, "cost_usd": 0.000013}
    assert body["trace_id"] is None  # tracing off -> no trace id, never a fake one


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_sends_deltas_then_one_priced_done(client):
    r = client.post(
        "/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "Hallo"}], "route": "tutor.reply", "session_id": "s1"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r.text)
    deltas = [d["text"] for e, d in ev if e == "delta"]
    assert "".join(deltas) == "echo: Hallo"
    assert [e for e, _ in ev][-1] == "done"
    done = ev[-1][1]
    assert done["usage"] == {"input_tokens": 7, "output_tokens": 3, "cost_usd": 0.000013}
    assert done["model"] == "fake-1"


def test_usage_sums_calls_per_route_and_session(client):
    for route, session in (("tutor.reply", "s1"), ("tutor.reply", "s1"), ("tutor.correct", "s2")):
        client.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}], "route": route, "session_id": session})
    client.post("/v1/chat/stream", json={"messages": [{"role": "user", "content": "x"}], "route": "tutor.reply"})

    u = client.get("/v1/usage", params={"days": 1}).json()
    assert u["calls"] == 4
    assert u["sessions"] == 2
    assert (u["input_tokens"], u["output_tokens"]) == (28, 12)
    assert u["cost_usd"] == pytest.approx(4 * 0.000013)
    assert u["unpriced_calls"] == 0
    by_route = {g["route"]: g["calls"] for g in u["groups"]}
    assert by_route == {"tutor.reply": 3, "tutor.correct": 1}


def test_gateway_key_required_when_configured(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gateway_api_key", "secret")
    body = {"messages": [{"role": "user", "content": "x"}]}
    assert client.post("/v1/chat", json=body).status_code == 401
    assert client.get("/v1/usage").status_code == 401
    assert client.post("/v1/chat", json=body, headers={"authorization": "Bearer secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for uptime checks


def test_chat_rejects_empty_messages(client):
    r = client.post("/v1/chat", json={"messages": []})
    assert r.status_code == 422


def test_chat_rejects_unknown_role(client):
    r = client.post("/v1/chat", json={"messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 422


def test_wrong_key_of_any_length_is_rejected(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gateway_api_key", "secret")
    body = {"messages": [{"role": "user", "content": "x"}]}
    for auth in ("Bearer secre", "Bearer secretX", "secret", "Bearer "):
        assert client.post("/v1/chat", json=body, headers={"authorization": auth}).status_code == 401


def test_model_outside_allowlist_is_refused(client):
    body = {"messages": [{"role": "user", "content": "x"}], "model": "some/expensive-model"}
    assert client.post("/v1/chat", json=body).status_code == 400
    assert client.post("/v1/chat/stream", json=body).status_code == 400
    body["model"] = "openai/gpt-oss-120b"  # what the tutor sends
    assert client.post("/v1/chat", json=body).status_code == 200


def test_oversized_requests_are_refused(client):
    from app.schemas import MAX_MESSAGE_CHARS, MAX_MESSAGES

    big = {"messages": [{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS + 1)}]}
    assert client.post("/v1/chat", json=big).status_code == 422
    many = {"messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1)}
    assert client.post("/v1/chat", json=many).status_code == 422
