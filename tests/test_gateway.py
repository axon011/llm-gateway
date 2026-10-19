"""Module 0 tests: run with the fake provider and no Langfuse keys. No network."""

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
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert body["trace_id"] is None  # tracing off -> no trace id, never a fake one


def test_chat_rejects_empty_messages(client):
    r = client.post("/v1/chat", json={"messages": []})
    assert r.status_code == 422


def test_chat_rejects_unknown_role(client):
    r = client.post("/v1/chat", json={"messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 422
