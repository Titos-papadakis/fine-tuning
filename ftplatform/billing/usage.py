"""
CRUD against the `usage_counters` table -- one row per (customer, calendar
month), incremented on every served request. See ftplatform/billing/__init__.py.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def current_period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def record(conn: sqlite3.Connection, customer_id: str, prompt_tokens: int,
           completion_tokens: int, cache_hit: bool = False, period: str | None = None) -> None:
    """Increments this customer's counters for `period` (default: the
    current calendar month), creating the row if this is its first request
    this period."""
    period = period or current_period()
    conn.execute(
        "INSERT INTO usage_counters (customer_id, period, requests, prompt_tokens, "
        "completion_tokens, cache_hits) VALUES (?, ?, 1, ?, ?, ?) "
        "ON CONFLICT(customer_id, period) DO UPDATE SET "
        "requests = requests + 1, prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
        "completion_tokens = completion_tokens + excluded.completion_tokens, "
        "cache_hits = cache_hits + excluded.cache_hits",
        (customer_id, period, prompt_tokens, completion_tokens, int(cache_hit)))
    conn.commit()


def report(conn: sqlite3.Connection, customer_id: str, period: str | None = None) -> dict:
    """This customer's counters for `period` (default: current month). All
    zero, not an error, for a customer with no usage yet this period."""
    period = period or current_period()
    row = conn.execute(
        "SELECT * FROM usage_counters WHERE customer_id = ? AND period = ?",
        (customer_id, period)).fetchone()
    if row is None:
        return {"customer_id": customer_id, "period": period, "requests": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}
    return dict(row)


def report_all(conn: sqlite3.Connection, period: str | None = None) -> list[dict]:
    """Every customer's counters for `period` (default: current month) --
    for `ftplatform usage report-all`, the input to actually invoicing
    people."""
    period = period or current_period()
    rows = conn.execute(
        "SELECT * FROM usage_counters WHERE period = ? ORDER BY customer_id",
        (period,)).fetchall()
    return [dict(r) for r in rows]


def build_hook(conn: sqlite3.Connection):
    """A `ServerState.on_response`-shaped callable (see
    ftplatform.monitoring.capture.build_hook() for the sibling and
    ftplatform.serving.hooks.compose() for running both) that records usage
    against the auth-resolved customer_id. A request with no resolved
    customer_id (auth disabled) is not recorded -- there is no customer to
    bill it against. generate_once()'s cache-hit path always reports
    elapsed_ms=0.0 (see serve.py), which is how a cache hit is distinguished
    here without serve.py needing to know usage_counters exists."""
    def on_response(request, text, prompt_tokens, completion_tokens, elapsed_ms,
                     customer_id=None):
        if customer_id is None:
            return
        record(conn, customer_id, prompt_tokens, completion_tokens, cache_hit=elapsed_ms == 0.0)

    return on_response
