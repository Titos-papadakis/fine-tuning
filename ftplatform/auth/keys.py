"""
CRUD against the `api_keys` table. See ftplatform/auth/__init__.py for why
only a hash is ever persisted.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone

KEY_PREFIX = "sk-ftplatform-"


def _hash(plaintext_key: str) -> str:
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def create_key(conn: sqlite3.Connection, customer_id: str, label: str = "") -> str:
    """Creates and stores a new key for `customer_id`, returning the
    plaintext -- the only time it is ever available. Losing it means
    revoking and creating a new one, not recovering the old one."""
    plaintext = generate_key()
    conn.execute(
        "INSERT INTO api_keys (key_hash, customer_id, label, created_at) VALUES (?, ?, ?, ?)",
        (_hash(plaintext), customer_id, label,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()
    return plaintext


def resolve_key(conn: sqlite3.Connection, plaintext_key: str) -> str | None:
    """The customer_id a valid, non-revoked key belongs to, or None if the
    key is unknown or has been revoked. Never raises on a bad key -- an
    invalid credential is an expected input from the network, not a bug."""
    row = conn.execute(
        "SELECT customer_id FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
        (_hash(plaintext_key),)).fetchone()
    return row["customer_id"] if row else None


def revoke_key(conn: sqlite3.Connection, key_hash_prefix: str) -> int:
    """Revokes every non-revoked key whose hash starts with
    `key_hash_prefix` (see list_keys() -- a caller only ever sees the hash,
    never the plaintext, so this is how a key gets identified for
    revocation). Returns how many keys were revoked; 0 means the prefix
    matched nothing live, not necessarily an error."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor = conn.execute(
        "UPDATE api_keys SET revoked_at = ? "
        "WHERE key_hash LIKE ? || '%' AND revoked_at IS NULL", (now, key_hash_prefix))
    conn.commit()
    return cursor.rowcount


def list_keys(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    """Metadata only -- key_hash (never the plaintext), label, created_at,
    revoked_at -- for `ftplatform api-key list`."""
    rows = conn.execute(
        "SELECT key_hash, customer_id, label, created_at, revoked_at FROM api_keys "
        "WHERE customer_id = ? ORDER BY created_at", (customer_id,)).fetchall()
    return [dict(r) for r in rows]


def build_resolver(conn: sqlite3.Connection, customer_ids: set[str] | None = None):
    """A `(plaintext_key) -> customer_id | None` closure for
    ftspec.serving.serve.STATE.api_key_resolver, snapshotting every active
    key's hash at call time rather than querying sqlite on every request.
    `customer_ids`, when given, restricts the snapshot to those customers
    (e.g. only the ones actually registered on a shared server) -- a valid
    key for a customer not being served here resolves to None, not a
    surprising cross-deployment success."""
    query = "SELECT key_hash, customer_id FROM api_keys WHERE revoked_at IS NULL"
    rows = conn.execute(query).fetchall()
    by_hash = {r["key_hash"]: r["customer_id"] for r in rows
               if customer_ids is None or r["customer_id"] in customer_ids}

    def resolver(plaintext_key: str) -> str | None:
        return by_hash.get(_hash(plaintext_key))

    return resolver
