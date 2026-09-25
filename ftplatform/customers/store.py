"""CRUD against the `customers` table.

Only identity and workload live here -- no raw data, no model artefacts. See
ftplatform/db.py's module docstring for why that split matters.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ftplatform.customers.models import Customer


def create(conn: sqlite3.Connection, customer_id: str, name: str, workload: str) -> Customer:
    customer = Customer(id=customer_id, name=name, workload=workload)  # validates id shape
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        conn.execute(
            "INSERT INTO customers (id, name, workload, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (customer.id, customer.name, customer.workload, customer.status, now),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError(f"customer {customer_id!r} already exists") from e
    return customer.model_copy(update={"created_at": now})


def get(conn: sqlite3.Connection, customer_id: str) -> Customer | None:
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return Customer(**dict(row)) if row else None


def list_all(conn: sqlite3.Connection) -> list[Customer]:
    rows = conn.execute("SELECT * FROM customers ORDER BY created_at").fetchall()
    return [Customer(**dict(r)) for r in rows]


# Every table carrying a customer_id, children before `customers` itself.
CUSTOMER_TABLES = ("deployments", "jobs", "approvals", "candidate_stats", "api_keys",
                   "usage_counters", "billing_accounts", "pipelines", "kaggle_artifacts")
BILLABLE_STATUSES = {"active", "trialing", "past_due", "unpaid"}


class CustomerHasLiveSubscriptionError(ValueError):
    pass


def billable(conn: sqlite3.Connection, customer_id: str) -> bool:
    row = conn.execute("SELECT status FROM billing_accounts WHERE customer_id = ?",
                       (customer_id,)).fetchone()
    return row is not None and row["status"] in BILLABLE_STATUSES


def delete(conn: sqlite3.Connection, customer_id: str, repo_root: Path | None = None,
           force: bool = False) -> dict:
    """Remove a customer entirely: every row in every table, its
    customers/<id>/ tree, and the local Kaggle staging dirs of its jobs.

    Refuses a customer whose Stripe subscription is still billable unless
    `force` -- deleting the records would not stop Stripe charging them, it
    would only remove this side's knowledge that it is. Cancel in Stripe
    first. Returns {"rows": {table: n}, "paths": [removed paths]}."""
    import ftspec.config as config_mod

    if get(conn, customer_id) is None:
        raise ValueError(f"no such customer: {customer_id!r}")
    if billable(conn, customer_id) and not force:
        raise CustomerHasLiveSubscriptionError(
            f"{customer_id!r} has a billable Stripe subscription -- cancel it in Stripe first "
            f"(deleting here would not stop the charges), or pass force.")

    root = Path(repo_root) if repo_root is not None else config_mod.REPO_ROOT
    job_ids = [r["id"] for r in conn.execute("SELECT id FROM jobs WHERE customer_id = ?",
                                             (customer_id,))]
    rows = {}
    for table in CUSTOMER_TABLES:
        rows[table] = conn.execute(f"DELETE FROM {table} WHERE customer_id = ?",  # noqa: S608
                                   (customer_id,)).rowcount
    rows["customers"] = conn.execute("DELETE FROM customers WHERE id = ?", (customer_id,)).rowcount
    conn.commit()

    removed = []
    for path in [root / "customers" / customer_id,
                 *(root / ".kaggle_work" / j for j in job_ids)]:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))

    from ftplatform import audit
    audit.record(conn, "customer.delete", customer_id,
                 {"rows": rows, "directories_removed": len(removed), "forced": force})
    return {"rows": rows, "paths": removed}
