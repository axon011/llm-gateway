# llm-gateway

An async FastAPI service that sits between an application and its LLM providers and traces every call
to a self-hosted Langfuse. Built to put real tracing, cost tracking, guardrails, prompt A/B tests and an
LLM judge in front of one live app (a German tutor) instead of a demo.

Status: module 0 of 8. What exists: `/health`, `POST /v1/chat` with a Gemini provider, one Langfuse
trace per call carrying the provider's own token counts. Nothing here is estimated.

## Run

```bash
cp .env.example .env            # add GEMINI_API_KEY
docker compose up -d            # Langfuse (web, worker, postgres, clickhouse, redis, minio) + gateway
# http://localhost:3000 -> create org, project, Settings -> API Keys -> put both keys in .env
docker compose up -d gateway    # restart the gateway with the keys
curl -s localhost:8080/health
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
