"""
Per-customer data-handling policy: what gets redacted from captured
production traffic before it is written, and how long captured traffic is
kept. Stored in the customer's own tree (customers/<id>/policy.json) so it
is as isolated as their data and travels with it.

Scope, stated plainly: this governs *captured production traffic* (the
production log and the review queue built from it). A customer's imported
training corpus and the corrections folded into training are their
training data -- kept until they are deleted with the customer, and
described that way in docs/security-and-data-handling.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ftspec.core.redaction import PATTERNS

DEFAULT_REDACT = ("pan", "ssn", "email", "phone", "iban")
DEFAULT_RETENTION_DAYS = 90
CAPTURE_FILES = ("production_log.jsonl", "error_queue.jsonl")


def _path(ctx):
    return ctx.customer_root() / "policy.json"


def load(ctx) -> dict:
    p = _path(ctx)
    stored = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"redact": tuple(stored.get("redact", DEFAULT_REDACT)),
            "retention_days": int(stored.get("retention_days", DEFAULT_RETENTION_DAYS))}


def save(ctx, redact: tuple | None = None, retention_days: int | None = None) -> dict:
    policy = load(ctx)
    if redact is not None:
        unknown = [r for r in redact if r not in PATTERNS]
        if unknown:
            raise ValueError(f"unknown redaction rule(s) {unknown}; available: {sorted(PATTERNS)}")
        policy["redact"] = tuple(redact)
    if retention_days is not None:
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        policy["retention_days"] = retention_days
    _path(ctx).parent.mkdir(parents=True, exist_ok=True)
    _path(ctx).write_text(json.dumps({"redact": list(policy["redact"]),
                                      "retention_days": policy["retention_days"]}, indent=2),
                          encoding="utf-8")
    return policy


def _parse(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def enforce_retention(ctx, now: datetime | None = None) -> dict:
    """Drop captured rows older than the customer's retention window. A row
    with no parseable timestamp is dropped too -- keeping data whose age
    can't be proven is the unsafe direction. Returns {file: rows removed}."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=load(ctx)["retention_days"])
    removed = {}
    for name in CAPTURE_FILES:
        path = ctx.memory_dir() / name
        if not path.exists():
            continue
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keep = []
        for ln in lines:
            ts = _parse(json.loads(ln).get("timestamp"))
            if ts is not None and ts >= cutoff:
                keep.append(ln)
        removed[name] = len(lines) - len(keep)
        if removed[name]:
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
            tmp.replace(path)
    return removed
