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


MIN_REVOKE_PREFIX_LEN = 8


class AmbiguousKeyPrefixError(ValueError):
    pass


def revoke_key(conn: sqlite3.Connection, key_hash_prefix: str) -> int:
    """Revokes every non-revoked key whose hash starts with
    `key_hash_prefix` (see list_keys() -- a caller only ever sees the hash,
    never the plaintext, so this is how a key gets identified for
    revocation). Returns how many keys were revoked; 0 means the prefix
    matched nothing live, not necessarily an error.

    Refuses a prefix shorter than MIN_REVOKE_PREFIX_LEN -- in particular an
    empty string, which would otherwise match (and revoke) every key for
    every customer in one call. `%` and `_` in the prefix are escaped so
    they match themselves literally rather than acting as SQL LIKE
    wildcards, for the same reason: a key_hash is always hex and never
    legitimately contains either, so only a mistaken/malicious prefix would
    include one, and it should narrow the match, never broaden it."""
    if len(key_hash_prefix) < MIN_REVOKE_PREFIX_LEN:
        raise AmbiguousKeyPrefixError(
            f"key_hash_prefix must be at least {MIN_REVOKE_PREFIX_LEN} characters "
            f"(got {len(key_hash_prefix)!r}) -- too short a prefix risks revoking more "
            f"keys than intended")
    escaped = key_hash_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cursor = conn.execute(
        "UPDATE api_keys SET revoked_at = ? "
        "WHERE key_hash LIKE ? || '%' ESCAPE '\\' AND revoked_at IS NULL", (now, escaped))
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
    ftspec.serving.serve.STATE.api_key_resolver. Queries sqlite live on
    every call rather than snapshotting once at build time -- a key
    revoked (or created) after the server started takes effect on its very
    next request. An earlier version snapshotted the active-key set once,
    which meant revoking a leaked key had no effect on an already-running
    server until it was restarted, defeating the point of an emergency
    revocation. `customer_ids`, when given, restricts resolution to those
    customers (e.g. only the ones actually registered on a shared server)
    -- a valid key for a customer not being served here resolves to None,
    not a surprising cross-deployment success."""
    def resolver(plaintext_key: str) -> str | None:
        customer_id = resolve_key(conn, plaintext_key)
        if customer_id is None or (customer_ids is not None and customer_id not in customer_ids):
            return None
        return customer_id

    return resolver
