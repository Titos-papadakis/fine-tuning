"""
ftplatform/auth/keys.py (API-key CRUD) and ftplatform/billing/usage.py
(per-customer monthly usage counters). Pure sqlite logic, no GPU/serving --
tests/test_serving.py covers the request-auth wiring with a stubbed engine.
"""
from __future__ import annotations

import pytest

from ftplatform.auth import keys
from ftplatform.billing import usage
from ftplatform.customers import store
from ftplatform.db import connect


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    store.create(c, "acme", "Acme Inc", "saas_support")
    store.create(c, "globex", "Globex Corp", "saas_support")
    yield c
    c.close()


# --- keys ----------------------------------------------------------------

def test_create_key_returns_a_usable_plaintext_key(conn):
    key = keys.create_key(conn, "acme")
    assert key.startswith(keys.KEY_PREFIX)
    assert keys.resolve_key(conn, key) == "acme"


def test_the_plaintext_key_is_never_stored(conn):
    key = keys.create_key(conn, "acme")
    row = conn.execute("SELECT * FROM api_keys").fetchone()
    assert key not in row["key_hash"]
    assert row["key_hash"] != key


def test_an_unknown_key_resolves_to_none(conn):
    assert keys.resolve_key(conn, "sk-ftplatform-not-a-real-key") is None


def test_two_customers_keys_resolve_to_their_own_customer(conn):
    acme_key = keys.create_key(conn, "acme")
    globex_key = keys.create_key(conn, "globex")
    assert keys.resolve_key(conn, acme_key) == "acme"
    assert keys.resolve_key(conn, globex_key) == "globex"


def test_a_revoked_key_no_longer_resolves(conn):
    key = keys.create_key(conn, "acme")
    key_hash = conn.execute("SELECT key_hash FROM api_keys").fetchone()["key_hash"]
    keys.revoke_key(conn, key_hash)
    assert keys.resolve_key(conn, key) is None


def test_revoke_key_returns_the_number_revoked(conn):
    keys.create_key(conn, "acme")
    key_hash = conn.execute("SELECT key_hash FROM api_keys").fetchone()["key_hash"]
    assert keys.revoke_key(conn, key_hash) == 1
    assert keys.revoke_key(conn, key_hash) == 0  # already revoked, not double-counted


def test_revoke_key_accepts_a_hash_prefix(conn):
    keys.create_key(conn, "acme")
    key_hash = conn.execute("SELECT key_hash FROM api_keys").fetchone()["key_hash"]
    assert keys.revoke_key(conn, key_hash[:12]) == 1


def test_list_keys_never_exposes_plaintext(conn):
    key = keys.create_key(conn, "acme", label="prod")
    rows = keys.list_keys(conn, "acme")
    assert len(rows) == 1
    assert rows[0]["label"] == "prod"
    assert rows[0]["revoked_at"] is None
    assert all(key not in str(v) for v in rows[0].values())


def test_list_keys_is_scoped_to_the_customer(conn):
    keys.create_key(conn, "acme")
    keys.create_key(conn, "globex")
    assert len(keys.list_keys(conn, "acme")) == 1


def test_build_resolver_matches_create_key_resolve_key(conn):
    key = keys.create_key(conn, "acme")
    resolver = keys.build_resolver(conn)
    assert resolver(key) == "acme"
    assert resolver("garbage") is None


def test_build_resolver_can_be_scoped_to_a_customer_set(conn):
    acme_key = keys.create_key(conn, "acme")
    globex_key = keys.create_key(conn, "globex")
    resolver = keys.build_resolver(conn, customer_ids={"acme"})
    assert resolver(acme_key) == "acme"
    assert resolver(globex_key) is None  # valid key, but not in this server's set


def test_build_resolver_excludes_revoked_keys(conn):
    key = keys.create_key(conn, "acme")
    key_hash = conn.execute("SELECT key_hash FROM api_keys").fetchone()["key_hash"]
    keys.revoke_key(conn, key_hash)
    resolver = keys.build_resolver(conn)
    assert resolver(key) is None


# --- usage -----------------------------------------------------------------

def test_report_for_a_customer_with_no_usage_is_all_zero(conn):
    report = usage.report(conn, "acme")
    assert report == {"customer_id": "acme", "period": usage.current_period(),
                        "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                        "cache_hits": 0}


def test_record_creates_the_first_row(conn):
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50)
    report = usage.report(conn, "acme")
    assert report["requests"] == 1
    assert report["prompt_tokens"] == 100
    assert report["completion_tokens"] == 50
    assert report["cache_hits"] == 0


def test_record_accumulates_across_calls(conn):
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50)
    usage.record(conn, "acme", prompt_tokens=20, completion_tokens=10)
    report = usage.report(conn, "acme")
    assert report["requests"] == 2
    assert report["prompt_tokens"] == 120
    assert report["completion_tokens"] == 60


def test_record_counts_cache_hits_separately(conn):
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50, cache_hit=False)
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50, cache_hit=True)
    report = usage.report(conn, "acme")
    assert report["requests"] == 2
    assert report["cache_hits"] == 1


def test_usage_is_isolated_per_customer(conn):
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50)
    usage.record(conn, "globex", prompt_tokens=5, completion_tokens=5)
    assert usage.report(conn, "acme")["requests"] == 1
    assert usage.report(conn, "globex")["requests"] == 1
    assert usage.report(conn, "acme")["prompt_tokens"] == 100


def test_usage_is_isolated_per_period(conn):
    usage.record(conn, "acme", prompt_tokens=100, completion_tokens=50, period="2026-01")
    usage.record(conn, "acme", prompt_tokens=1, completion_tokens=1, period="2026-02")
    assert usage.report(conn, "acme", period="2026-01")["prompt_tokens"] == 100
    assert usage.report(conn, "acme", period="2026-02")["prompt_tokens"] == 1


def test_report_all_lists_every_customer_for_a_period(conn):
    usage.record(conn, "acme", prompt_tokens=1, completion_tokens=1, period="2026-01")
    usage.record(conn, "globex", prompt_tokens=2, completion_tokens=2, period="2026-01")
    rows = usage.report_all(conn, period="2026-01")
    assert {r["customer_id"] for r in rows} == {"acme", "globex"}


def test_report_all_does_not_include_other_periods(conn):
    usage.record(conn, "acme", prompt_tokens=1, completion_tokens=1, period="2026-01")
    rows = usage.report_all(conn, period="2026-02")
    assert rows == []


# --- build_hook --------------------------------------------------------------

class _FakeMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _FakeRequest:
    def __init__(self):
        self.messages = [_FakeMessage("user", "hi")]


def test_build_hook_records_usage_for_the_resolved_customer(conn):
    hook = usage.build_hook(conn)
    hook(_FakeRequest(), "output text", 100, 50, 12.0, "acme")
    report = usage.report(conn, "acme")
    assert report["requests"] == 1
    assert report["prompt_tokens"] == 100
    assert report["completion_tokens"] == 50


def test_build_hook_skips_requests_with_no_resolved_customer(conn):
    hook = usage.build_hook(conn)
    hook(_FakeRequest(), "output text", 100, 50, 12.0, None)  # must not raise
    assert usage.report_all(conn) == []


def test_build_hook_treats_zero_elapsed_ms_as_a_cache_hit(conn):
    hook = usage.build_hook(conn)
    hook(_FakeRequest(), "output text", 100, 50, 0.0, "acme")
    assert usage.report(conn, "acme")["cache_hits"] == 1
