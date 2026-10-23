"""Usage ledger: one SQLite row per completed call, so cost questions don't depend on Langfuse.

Langfuse shows each trace; this answers "what did the tutor cost this week, per route and model"
with one GET. Writes are tiny and run in a worker thread so the event loop never blocks on disk.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    ts TEXT NOT NULL,             -- UTC ISO-8601
    route TEXT NOT NULL,
    session_id TEXT,
    user_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL,                -- NULL when the model has no price entry
    trace_id TEXT
);
CREATE INDEX IF NOT EXISTS calls_ts ON calls(ts);
"""


class UsageStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def _insert(self, row: tuple) -> None:
        with self._lock:
            self._db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)", row)
            self._db.commit()

    async def record(
        self,
        *,
        route: str,
        session_id: str | None,
        user_id: str | None,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        trace_id: str | None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = (ts, route, session_id, user_id, provider, model, input_tokens, output_tokens, cost_usd, trace_id)
        await asyncio.to_thread(self._insert, row)

    def summary(self, days: int) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        q = """
            SELECT substr(ts, 1, 10) AS day, route, model, COUNT(*), SUM(input_tokens), SUM(output_tokens),
                   SUM(cost_usd), SUM(cost_usd IS NULL)
            FROM calls WHERE ts >= ? GROUP BY day, route, model ORDER BY day, route, model
        """
        with self._lock:
            rows = self._db.execute(q, (since,)).fetchall()
            sessions = self._db.execute(
                "SELECT COUNT(DISTINCT session_id) FROM calls WHERE ts >= ? AND session_id IS NOT NULL", (since,)
            ).fetchone()[0]
        groups = [
            {
                "day": d,
                "route": r,
                "model": m,
                "calls": n,
                "input_tokens": i,
                "output_tokens": o,
                "cost_usd": round(c or 0.0, 6),
                "unpriced_calls": u,
            }
            for d, r, m, n, i, o, c, u in rows
        ]
        calls = sum(g["calls"] for g in groups)
        total = round(sum(g["cost_usd"] for g in groups), 6)
        return {
            "days": days,
            "since": since,
            "calls": calls,
            "sessions": sessions,
            "input_tokens": sum(g["input_tokens"] for g in groups),
            "output_tokens": sum(g["output_tokens"] for g in groups),
            "cost_usd": total,
            "cost_per_call_usd": round(total / calls, 8) if calls else None,
            "cost_per_session_usd": round(total / sessions, 8) if sessions else None,
            "unpriced_calls": sum(g["unpriced_calls"] for g in groups),
            "groups": groups,
        }
