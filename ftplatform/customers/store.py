"""CRUD against the `customers` table.

Only identity and workload live here -- no raw data, no model artefacts. See
ftplatform/db.py's module docstring for why that split matters.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

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
