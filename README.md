# llm-gateway

An async FastAPI service that sits between an application and its LLM providers and traces every call
to a self-hosted Langfuse. Built to put real tracing, cost tracking, guardrails, prompt A/B tests and an
LLM judge in front of one live app (a German tutor) instead of a demo.

Status: module 0 of 8 done (2026-10-19). What exists: `/health`, `POST /v1/chat` with Groq and Gemini
providers, one Langfuse trace per call (root span + generation) carrying the provider's own token
counts, session, user and route tags. Langfuse is provisioned headlessly from `.env`
(`LANGFUSE_INIT_*`), so `docker compose up` yields a working project with no clicking.
Nothing here is estimated. Cost per call is 0 until module 2 adds the pricing table.

## Run

```bash
cp .env.example .env            # add GROQ_API_KEY (or GEMINI_API_KEY + LLM_PROVIDER=gemini)
# generate LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (any pk-lf-... / sk-lf-... strings) and put the
# same values in LANGFUSE_INIT_PROJECT_PUBLIC_KEY / _SECRET_KEY plus an INIT user email + password:
# Langfuse creates org, project, keys and login on first start.
docker compose up -d            # Langfuse (web, worker, postgres, clickhouse, redis, minio) + gateway
curl -s localhost:8080/health   # {"status":"ok","provider":"groq","tracing":true}
curl -s localhost:8080/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Korrigiere: Ich habe gestern ins Kino gegangen."}],"session_id":"demo","user_id":"u-demo","route":"tutor.correct"}'
```

The response includes `trace_id`; open it in the Langfuse UI.

Local, no Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
pytest -q                        # fake provider, tracing off, no network
```

## Design rules

- Token counts and cost come only from the provider's usage fields. No `len(prompt) // 4`.
- With no Langfuse keys the gateway runs with a no-op tracer and returns `trace_id: null`, never a fake id.
- `user_id` is a hash the caller computes. Raw emails never reach the gateway.

## Roadmap

1. Tracing: session/user grouping, streaming spans
2. Cost: dated per-model pricing, per-request cost, daily rollups, per-user budget (429)
3. Guardrails: PII masking, injection screen, output schema and language checks, trips recorded as scores
4. A/B prompt variants from Langfuse prompt management, sticky split, report with confidence intervals
5. LLM-as-judge on a sample of live turns, rubric scores written back, kappa against hand labels
6. CI regression gate on a 50-case dataset
7. Wire the tutor (`lib/llm` http provider), then the multi-agent pipeline
