"""
The platform's shared sqlite store.

Holds only metadata and pointers: customer identity and, from later
milestones, job status and deployment records. Raw customer text, model
weights, and predictions never touch this database -- they live under
customers/<id>/ on disk, reached only through CustomerContext. This split
is the entire persistence half of customer isolation (see
ftplatform/customers/context.py for the filesystem half).

WAL mode is used so a long-running write (e.g. a training job's manifest
update) doesn't block a concurrent `customer list`. It does not make this
safe for multiple simultaneous writers -- this platform assumes one
operator/worker process at a time, same as the single free GPU it runs
training on.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ftspec.config import REPO_ROOT

DB_PATH = REPO_ROOT / "customers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    workload    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL
);

-- One row per successful promotion to production/, plus rollbacks (a
-- rollback is itself a new row pointing at an older candidate -- history is
-- never overwritten, only appended to, so "what was live on date X" stays
-- answerable).
CREATE TABLE IF NOT EXISTS deployments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    workload      TEXT NOT NULL,
    candidate_id  TEXT NOT NULL,
    deployed_at   TEXT NOT NULL,
    is_rollback   INTEGER NOT NULL DEFAULT 0,
    metrics_json  TEXT NOT NULL
);

-- Phase 7's job queue index. The job itself is just a row here (a JSON
-- payload/result column, not a separate file) -- with expected volume this
-- small, a second on-disk format would only be more to keep in sync.
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    payload_json  TEXT NOT NULL,
    result_json   TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);

-- Phase 7's human-approval gate: a customer's very first deployment must be
-- approved here before the worker will run a 'deploy' job for them.
CREATE TABLE IF NOT EXISTS approvals (
    customer_id   TEXT PRIMARY KEY REFERENCES customers(id),
    approved_at   TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the platform database, creating its schema if needed."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn
