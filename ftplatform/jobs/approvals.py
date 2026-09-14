"""
Phase 7's human-approval gate.

A customer's very first deployment must be explicitly approved
(`ftplatform approve <customer_id>`) before the job worker will execute a
'deploy' job for them -- see worker.py's `_dispatch`. Subsequent retrain
cycles (Phase 6) need no re-approval: they're already gated by
`deploy.maybe_deploy()`'s own statistical checks, and requiring a human
click on every automatic retrain would defeat the point of Phase 6.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def approve(conn: sqlite3.Connection, customer_id: str) -> None:
    conn.execute(
        "INSERT INTO approvals (customer_id, approved_at) VALUES (?, ?) "
        "ON CONFLICT(customer_id) DO UPDATE SET approved_at = excluded.approved_at",
        (customer_id, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def is_approved(conn: sqlite3.Connection, customer_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM approvals WHERE customer_id = ?", (customer_id,)).fetchone()
    return row is not None
