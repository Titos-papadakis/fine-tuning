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

-- Phase 8 -- cross-customer learning, deliberately narrow: one row per
-- (candidate, system) every time a leaderboard is built for any customer.
-- candidate_id already *is* a technique descriptor by construction (see
-- candidates/generator.py: "stageA-qwen25-3b", "stageB-r16-a16", ...), so
-- this table never carries a customer's actual data, corpus, or model
-- weights -- only which technique was tried, for which workload, and the
-- same composite score the leaderboard itself already computed.
-- customer_id is kept for audit traceability only. See ftplatform/learning/.
CREATE TABLE IF NOT EXISTS candidate_stats (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    workload      TEXT NOT NULL,
    stage         TEXT NOT NULL,
    candidate_id  TEXT NOT NULL,
    system        TEXT NOT NULL,
    score         REAL,
    gated         INTEGER NOT NULL DEFAULT 0,
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    recorded_at   TEXT NOT NULL
);

-- Serving-layer auth: a request must carry a valid, non-revoked key to be
-- served (see ftplatform/auth/keys.py + ftspec.serving.serve's
-- api_key_resolver). Only the sha256 hash is ever stored -- the plaintext
-- key is shown once, at creation, and never again.
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    label         TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    revoked_at    TEXT
);

-- Per-customer metering, one row per (customer, calendar month) --
-- aggregate counters rather than one row per request, since the point is
-- billing/usage reporting, not a request-level audit trail (that's
-- memory/production_log.jsonl, per customer, see ftplatform/monitoring/).
CREATE TABLE IF NOT EXISTS usage_counters (
    customer_id        TEXT NOT NULL REFERENCES customers(id),
    period              TEXT NOT NULL,
    requests            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_hits          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (customer_id, period)
);

-- One onboarding pipeline per customer (baseline -> Stage A -> Stage B ->
-- optional Stage C -> deploy), advanced by ftplatform/jobs/pipeline.py.
-- state_json holds the options it was started with, the job ids of each
-- stage, and each stage's chosen winner, so a restarted driver resumes
-- exactly where the last one stopped instead of re-running finished stages.
CREATE TABLE IF NOT EXISTS pipelines (
    customer_id   TEXT PRIMARY KEY REFERENCES customers(id),
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    state_json    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- Every private dataset/kernel a job created on Kaggle, recorded at push
-- time so `customer delete` can remove them there too -- they hold a copy
-- of the customer's data (see docs/security-and-data-handling.md §4).
CREATE TABLE IF NOT EXISTS kaggle_artifacts (
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    job_id        TEXT NOT NULL,
    dataset_id    TEXT NOT NULL,
    kernel_id     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (customer_id, job_id)
);

-- Append-only; see ftplatform/audit.py. No FK on customer_id on purpose: the
-- record of a customer's deletion must outlive the customer row.
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    actor         TEXT NOT NULL,
    customer_id   TEXT,
    action        TEXT NOT NULL,
    detail_json   TEXT NOT NULL
);

-- One row per customer that has been linked to Stripe -- created the moment
-- a customer is ready to actually be charged, not at `customer add` time (a
-- customer can exist and be trained/evaluated for a good while before any
-- money changes hands). Never holds card/payment details -- that stays in
-- Stripe; this is only enough to know who a customer is over there and
-- whether their subscription is currently active. status mirrors Stripe's
-- own subscription status vocabulary ('active', 'past_due', 'canceled', ...)
-- plus 'unlinked'/'linked' for the two states before a subscription exists.
CREATE TABLE IF NOT EXISTS billing_accounts (
    customer_id             TEXT PRIMARY KEY REFERENCES customers(id),
    email                   TEXT NOT NULL,
    stripe_customer_id      TEXT,
    stripe_subscription_id  TEXT,
    status                  TEXT NOT NULL DEFAULT 'unlinked',
    updated_at              TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the platform database, creating its schema if needed."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def backup(conn: sqlite3.Connection, dest_path: Path) -> Path:
    """Consistent snapshot of the whole platform database to `dest_path`.

    Uses sqlite3's own online backup API rather than copying the file --
    this database runs in WAL mode (see the module docstring) specifically
    so reads/writes aren't blocked, which means a plain file copy can catch
    it mid-write and produce a corrupt snapshot. The backup API takes a
    consistent snapshot regardless of what's concurrently writing."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_conn = sqlite3.connect(dest_path)
    try:
        conn.backup(dest_conn)
    finally:
        dest_conn.close()
    return dest_path
