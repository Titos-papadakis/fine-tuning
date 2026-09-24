"""
Phase 7 -- a filesystem/sqlite job queue.

"Automation" here means a poll loop plus a chain of function calls with one
required human click (see jobs/approvals.py) -- not a distributed workflow
engine. No retry-with-backoff beyond a simple staleness check
(`requeue_stale`), no multi-worker coordination, and this assumes exactly
one worker process at a time, same as the single free GPU it runs training
on. Adequate for a solo operator's pilot customers, not a production job
scheduler.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

DEFAULT_STALE_AFTER_S = 3600  # a job "running" longer than this gets requeued


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue(conn: sqlite3.Connection, customer_id: str, kind: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:16]
    conn.execute(
        "INSERT INTO jobs (id, customer_id, kind, status, payload_json, created_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (job_id, customer_id, kind, json.dumps(payload, ensure_ascii=False), _now()))
    conn.commit()
    return job_id


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json"))
    result_json = d.pop("result_json")
    d["result"] = json.loads(result_json) if result_json else None
    return d


def get(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(conn: sqlite3.Connection, customer_id: str | None = None) -> list[dict]:
    if customer_id:
        rows = conn.execute("SELECT * FROM jobs WHERE customer_id = ? ORDER BY created_at",
                             (customer_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
    return [_row_to_dict(r) for r in rows]


def pop_next_pending(conn: sqlite3.Connection) -> dict | None:
    """Claims the oldest pending job by marking it running. Not safe against
    two workers racing this at once -- see the module docstring."""
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1").fetchone()
    if row is None:
        return None
    job_id = row["id"]
    conn.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                 (_now(), job_id))
    conn.commit()
    return get(conn, job_id)


def claim(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Claims one specific pending job (the pipeline driver runs its own
    customer's jobs, not whatever happens to be oldest). None if it is no
    longer pending."""
    cur = conn.execute("UPDATE jobs SET status = 'running', started_at = ? "
                       "WHERE id = ? AND status = 'pending'", (_now(), job_id))
    conn.commit()
    return get(conn, job_id) if cur.rowcount else None


def mark_done(conn: sqlite3.Connection, job_id: str, result: dict) -> None:
    conn.execute("UPDATE jobs SET status = 'done', result_json = ?, finished_at = ? WHERE id = ?",
                 (json.dumps(result, ensure_ascii=False), _now(), job_id))
    conn.commit()


def mark_failed(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    conn.execute("UPDATE jobs SET status = 'failed', result_json = ?, finished_at = ? WHERE id = ?",
                 (json.dumps({"error": error}, ensure_ascii=False), _now(), job_id))
    conn.commit()


def requeue_stale(conn: sqlite3.Connection, stale_after_s: int = DEFAULT_STALE_AFTER_S) -> list[str]:
    """A job stuck 'running' longer than `stale_after_s` -- its worker
    likely died (a killed Colab session, say) -- gets marked failed instead
    of blocking the queue forever. Returns the ids it failed. Re-enqueueing
    (a fresh job with the same payload) is a deliberate separate step, not
    automatic, since a job that died mid-training may have left partial
    state worth looking at first."""
    now = datetime.now(timezone.utc)
    rows = conn.execute("SELECT id, started_at FROM jobs WHERE status = 'running'").fetchall()
    stale_ids = []
    for row in rows:
        started = datetime.fromisoformat(row["started_at"])
        if (now - started).total_seconds() > stale_after_s:
            stale_ids.append(row["id"])
    for job_id in stale_ids:
        mark_failed(conn, job_id, f"stale: still 'running' after {stale_after_s}s, worker likely died")
    return stale_ids
