"""
Milestone 0 of the multi-customer platform: registering customers and
proving `CustomerContext` never lets two customers' paths collide.

This is the property the rest of the platform (candidate sweeps, deployment,
the self-improving loop) leans on without re-checking it: every module reaches
`ftspec` only through a `CustomerContext`, so if isolation holds here, it holds
everywhere that respects the convention.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext, UnknownCustomerError
from ftplatform.customers.models import Customer
from ftplatform.db import connect


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


def test_create_and_list_round_trip(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")

    customers = store.list_all(conn)
    assert {c.id for c in customers} == {"acme", "globex"}


def test_duplicate_id_is_a_clean_error(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    with pytest.raises(ValueError, match="already exists"):
        store.create(conn, "acme", "Acme Again", "saas_support")


@pytest.mark.parametrize("bad_id", ["", "Acme", "acme customer", "a" * 33, "../etc", "acme/../x"])
def test_unsafe_ids_are_rejected_before_they_ever_become_a_path(conn, bad_id):
    with pytest.raises(ValidationError):
        store.create(conn, bad_id, "Acme Inc", "saas_support")


def test_unknown_customer_context_fails_closed(conn):
    with pytest.raises(UnknownCustomerError):
        CustomerContext(conn, "does-not-exist")


def test_two_customers_share_no_path(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")

    a = CustomerContext(conn, "acme")
    b = CustomerContext(conn, "globex")

    pairs = [
        (a.data_dir(), b.data_dir()),
        (a.candidate_dir("c1"), b.candidate_dir("c1")),
        (a.adapter_dir("c1"), b.adapter_dir("c1")),
        (a.reports_dir("c1"), b.reports_dir("c1")),
        (a.manifests_dir("c1"), b.manifests_dir("c1")),
        (a.production_dir(), b.production_dir()),
    ]
    for path_a, path_b in pairs:
        assert path_a != path_b
        assert "acme" in str(path_a) and "acme" not in str(path_b)
        assert "globex" in str(path_b) and "globex" not in str(path_a)
        # Neither directory is an ancestor of the other.
        assert path_a not in (path_b, *path_b.parents)
        assert path_b not in (path_a, *path_a.parents)


def test_same_customer_different_candidates_do_not_collide(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    ctx = CustomerContext(conn, "acme")

    assert ctx.candidate_dir("c1") != ctx.candidate_dir("c2")
    assert ctx.production_dir() not in (ctx.candidate_dir("c1"), ctx.candidate_dir("c2"))
    # production/ sits beside candidates/, not inside it: candidate_dir's
    # grandparent (.../candidates/c1 -> .../candidates -> .../<workload>) is
    # production_dir's direct parent.
    assert ctx.production_dir().parent == ctx.candidate_dir("c1").parent.parent


def test_context_reuses_ftspec_config_path_methods(conn):
    """Isolation should come from Config's own path methods (data_dir,
    adapter_dir, reports_dir, manifests_dir), not a parallel implementation --
    otherwise the two could silently drift apart, the exact bug the merge/eval
    adapter-path mismatch already taught this project once."""
    store.create(conn, "acme", "Acme Inc", "saas_support")
    ctx = CustomerContext(conn, "acme")

    assert ctx.adapter_dir("c1") == ctx.config.adapter_dir(ctx.candidate_key("c1"))
    assert ctx.reports_dir("c1") == ctx.config.reports_dir(ctx.candidate_key("c1"))
    assert ctx.manifests_dir("c1") == ctx.config.manifests_dir(ctx.candidate_key("c1"))


def test_db_schema_is_created_on_first_connect(tmp_path):
    fresh_path = tmp_path / "fresh.db"
    assert not fresh_path.exists()
    conn = connect(fresh_path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "customers" in tables
    finally:
        conn.close()


def test_pydantic_model_still_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Customer(id="acme", name="Acme", workload="saas_support", unexpected="x")
