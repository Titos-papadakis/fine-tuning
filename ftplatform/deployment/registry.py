"""CRUD against the `deployments` table -- the audit trail of every
promotion (and rollback) to production/. Raw data and model weights never
appear here, only pointers and the metrics snapshot that justified the
promotion (see ftplatform/db.py's module docstring for the general split).

History is append-only: a rollback is a new row pointing at an older
candidate, never an edit or delete of a prior row, so "what was live on
date X" stays answerable.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


def record(conn: sqlite3.Connection, customer_id: str, workload: str, candidate_id: str,
           metrics: dict, is_rollback: bool = False) -> None:
    conn.execute(
        "INSERT INTO deployments (customer_id, workload, candidate_id, deployed_at, "
        "is_rollback, metrics_json) VALUES (?, ?, ?, ?, ?, ?)",
        (customer_id, workload, candidate_id,
         datetime.now(timezone.utc).isoformat(timespec="seconds"),
         int(is_rollback), json.dumps(metrics, ensure_ascii=False)))
    conn.commit()


def history(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM deployments WHERE customer_id = ? ORDER BY deployed_at",
        (customer_id,)).fetchall()
    return [dict(r) for r in rows]


def current(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    rows = history(conn, customer_id)
    return rows[-1] if rows else None
