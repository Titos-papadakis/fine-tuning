"""
Milestone 4, capture + review: turning served requests into a short,
human-reviewable list of rows worth a second look.

`capture.attach()` is tested by grabbing the closure it installs on
`ServerState.on_response` and calling it directly with a hand-built fake
request -- lighter than a full HTTP round-trip (that's
test_serve_on_response_hook.py's job) and lets these tests focus on what
ftplatform does with the hook, not on serve.py's wiring.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.monitoring import capture, review_queue


class _FakeMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _FakeRequest:
    def __init__(self, user_text):
        self.messages = [_FakeMessage("user", user_text)]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _reset_serve_state():
    """capture.attach() mutates the real ftspec.serving.serve.STATE global --
    tests here don't spin up a server, so nothing else resets it between
    tests the way test_edge_cases.py's/test_serve_on_response_hook.py's
    fixtures do."""
    from ftspec.serving import serve as S
    yield
    S.STATE.on_response = None


def _ctx(monkeypatch, tmp_path, conn, customer_id, workload):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, customer_id, customer_id, workload)
    return CustomerContext(conn, customer_id)


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- capture -------------------------------------------------------------------

def test_attach_installs_the_hook(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    assert S.STATE.on_response is None
    capture.attach(ctx)
    assert S.STATE.on_response is not None


def test_capture_records_text_for_an_unregulated_profile(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    capture.attach(ctx)

    S.STATE.on_response(_FakeRequest("customer said the app crashed"), "not json", 10, 20, 5.5)

    rows = _read_jsonl(ctx.memory_dir() / "production_log.jsonl")
    assert len(rows) == 1
    assert rows[0]["text_captured"] is True
    assert rows[0]["input"] == "customer said the app crashed"
    assert rows[0]["contract_valid"] is False  # "not json" cannot parse
    assert rows[0]["prompt_tokens"] == 10 and rows[0]["completion_tokens"] == 20


def test_capture_omits_text_for_a_regulated_profile(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    ctx = _ctx(monkeypatch, tmp_path, conn, "regco", "healthcare_clinical")
    capture.attach(ctx)

    S.STATE.on_response(_FakeRequest("patient details here"), "not json", 10, 20, 5.5)

    rows = _read_jsonl(ctx.memory_dir() / "production_log.jsonl")
    assert rows[0]["text_captured"] is False
    assert rows[0]["input"] is None
    assert rows[0]["output"] is None
    # Structured metadata is still captured even without text.
    assert rows[0]["contract_valid"] is False


def test_captured_rows_are_isolated_per_customer(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    a = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    capture.attach(a)
    S.STATE.on_response(_FakeRequest("acme's data"), "x", 1, 1, 1.0)

    b = _ctx(monkeypatch, tmp_path, conn, "globex", "saas_support")
    capture.attach(b)
    S.STATE.on_response(_FakeRequest("globex's data"), "x", 1, 1, 1.0)

    a_rows = _read_jsonl(a.memory_dir() / "production_log.jsonl")
    b_rows = _read_jsonl(b.memory_dir() / "production_log.jsonl")
    assert len(a_rows) == 1 and a_rows[0]["input"] == "acme's data"
    assert len(b_rows) == 1 and b_rows[0]["input"] == "globex's data"


def test_capture_ignores_a_trailing_customer_id_in_single_customer_mode(monkeypatch, tmp_path, conn):
    """attach()'s ctx is fixed -- a customer_id argument (as serve.py always
    passes now) must not change which log the row lands in."""
    from ftspec.serving import serve as S

    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    capture.attach(ctx)

    S.STATE.on_response(_FakeRequest("hello"), "x", 1, 1, 1.0, "some-other-id")

    rows = _read_jsonl(ctx.memory_dir() / "production_log.jsonl")
    assert len(rows) == 1


# --- attach_shared -----------------------------------------------------------

def test_attach_shared_refuses_without_auth_enabled(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    S.STATE.api_key_resolver = None
    with pytest.raises(RuntimeError, match="requires auth"):
        capture.attach_shared(conn)


def test_attach_shared_writes_to_the_resolved_customers_own_log(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    a = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    b = _ctx(monkeypatch, tmp_path, conn, "globex", "saas_support")
    S.STATE.api_key_resolver = {"key": "acme"}.get  # just needs to be non-None
    try:
        capture.attach_shared(conn)
        S.STATE.on_response(_FakeRequest("acme's traffic"), "x", 1, 1, 1.0, "acme")
        S.STATE.on_response(_FakeRequest("globex's traffic"), "x", 1, 1, 1.0, "globex")

        a_rows = _read_jsonl(a.memory_dir() / "production_log.jsonl")
        b_rows = _read_jsonl(b.memory_dir() / "production_log.jsonl")
        assert len(a_rows) == 1 and a_rows[0]["input"] == "acme's traffic"
        assert len(b_rows) == 1 and b_rows[0]["input"] == "globex's traffic"
    finally:
        S.STATE.api_key_resolver = None


def test_attach_shared_skips_a_request_with_no_resolved_customer_id(monkeypatch, tmp_path, conn):
    from ftspec.serving import serve as S

    a = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    S.STATE.api_key_resolver = {"key": "acme"}.get
    try:
        capture.attach_shared(conn)
        S.STATE.on_response(_FakeRequest("no identity"), "x", 1, 1, 1.0, None)  # not raised
        assert _read_jsonl(a.memory_dir() / "production_log.jsonl") == []
    finally:
        S.STATE.api_key_resolver = None


# --- review_queue ----------------------------------------------------------------

def _write_production_log(ctx, rows):
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(ctx.memory_dir() / "production_log.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(request_id, latency_ms=50.0, contract_valid=True, **extra):
    return {"request_id": request_id, "latency_ms": latency_ms,
            "contract_valid": contract_valid, "input": f"in-{request_id}",
            "output": {"x": 1}, "timestamp": "2026-01-01T00:00:00", **extra}


def test_scan_with_no_log_flags_nothing(monkeypatch, tmp_path, conn):
    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    assert review_queue.scan(ctx) == []


def test_scan_flags_schema_invalid_rows(monkeypatch, tmp_path, conn):
    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    _write_production_log(ctx, [_row("r1", contract_valid=True), _row("r2", contract_valid=False)])

    flagged = review_queue.scan(ctx)

    assert [r["request_id"] for r in flagged] == ["r2"]
    assert "schema_invalid" in flagged[0]["reasons"]


def test_scan_flags_latency_outliers(monkeypatch, tmp_path, conn):
    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    normal = [_row(f"r{i}", latency_ms=50.0) for i in range(10)]
    outlier = _row("slow", latency_ms=5000.0)
    _write_production_log(ctx, [*normal, outlier])

    flagged = review_queue.scan(ctx)

    assert any(r["request_id"] == "slow" and "latency_outlier" in r["reasons"] for r in flagged)
    assert not any(r["request_id"] != "slow" for r in flagged)


def test_scan_does_not_reflag_already_pending_rows(monkeypatch, tmp_path, conn):
    ctx = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    _write_production_log(ctx, [_row("r1", contract_valid=False)])

    first = review_queue.scan(ctx)
    second = review_queue.scan(ctx)  # same log, nothing new

    assert len(first) == 1
    assert second == []


def test_scan_is_isolated_per_customer(monkeypatch, tmp_path, conn):
    a = _ctx(monkeypatch, tmp_path, conn, "acme", "saas_support")
    _write_production_log(a, [_row("r1", contract_valid=False)])
    b = _ctx(monkeypatch, tmp_path, conn, "globex", "saas_support")
    _write_production_log(b, [_row("r2", contract_valid=False), _row("r3", contract_valid=False)])

    assert len(review_queue.scan(a)) == 1
    assert len(review_queue.scan(b)) == 2
