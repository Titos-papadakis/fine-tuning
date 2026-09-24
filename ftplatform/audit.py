"""
Append-only record of every action that changes what a customer gets served,
who can reach it, or what data is held about them: who did it, when, to whom.

Details carry identifiers and counts only -- never ticket text, labels or
keys -- so the log itself is safe to hand to a customer's security reviewer.
Rows deliberately survive `customer delete` (customer_id has no foreign key):
"when was our data erased, and by whom" is exactly what an auditor asks
afterwards.
"""
from __future__ import annotations

import getpass
import json
import os
import sqlite3
from datetime import datetime, timezone


def actor() -> str:
    """FTPLATFORM_ACTOR when set (a worker, a CI job), else the OS user."""
    env = os.environ.get("FTPLATFORM_ACTOR")
    if env:
        return env
    try:
        return getpass.getuser()
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def record(conn: sqlite3.Connection, action: str, customer_id: str | None = None,
           detail: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log (at, actor, customer_id, action, detail_json) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), actor(), customer_id, action,
         json.dumps(detail or {}, ensure_ascii=False, sort_keys=True)))
    conn.commit()


def entries(conn: sqlite3.Connection, customer_id: str | None = None,
            limit: int | None = None) -> list[dict]:
    sql, args = "SELECT * FROM audit_log", []
    if customer_id:
        sql += " WHERE customer_id = ?"
        args.append(customer_id)
    sql += " ORDER BY id"
    rows = [dict(r) for r in conn.execute(sql, args)]
    for r in rows:
        r["detail"] = json.loads(r.pop("detail_json"))
    return rows[-limit:] if limit else rows
