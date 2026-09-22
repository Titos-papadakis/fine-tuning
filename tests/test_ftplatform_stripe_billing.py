"""
ftplatform/billing/stripe_billing.py. Every Stripe-facing test passes a
FakeStripe in place of the real `stripe` module -- nothing here makes a
real network call. See test_ftplatform_auth_and_billing.py for the
usage-counter half of billing (what a customer consumed, not whether
they're a paying subscriber).
"""
from __future__ import annotations

import pytest

from ftplatform.billing import stripe_billing as sb
from ftplatform.customers import store
from ftplatform.db import connect


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    store.create(c, "acme", "Acme Inc", "saas_support")
    yield c
    c.close()


class FakeCustomers:
    def __init__(self):
        self.created = []

    def create(self, email, name, metadata):
        self.created.append({"email": email, "name": name, "metadata": metadata})
        return {"id": f"cus_{len(self.created)}"}


class FakeSubscriptions:
    def __init__(self, status="active"):
        self.created = []
        self.status = status

    def create(self, customer, items):
        self.created.append({"customer": customer, "items": items})
        return {"id": f"sub_{len(self.created)}", "status": self.status}

    def retrieve(self, subscription_id):
        return {"id": subscription_id, "status": self.status}


class FakeStripe:
    def __init__(self, status="active"):
        self.Customer = FakeCustomers()
        self.Subscription = FakeSubscriptions(status=status)


# --- talking to "Stripe" -------------------------------------------------

def test_create_stripe_customer_tags_our_own_customer_id():
    fake = FakeStripe()
    stripe_customer_id = sb.create_stripe_customer("acme", "billing@acme.example",
                                                     "Acme Inc", client=fake)
    assert stripe_customer_id == "cus_1"
    assert fake.Customer.created[0]["metadata"] == {"ftplatform_customer_id": "acme"}


def test_create_subscription_targets_the_given_price():
    fake = FakeStripe()
    sub_id = sb.create_subscription("cus_1", "price_flat_3k", client=fake)
    assert sub_id == "sub_1"
    assert fake.Subscription.created[0] == {
        "customer": "cus_1", "items": [{"price": "price_flat_3k"}]}


def test_fetch_subscription_status_reflects_stripe():
    fake = FakeStripe(status="past_due")
    assert sb.fetch_subscription_status("sub_1", client=fake) == "past_due"


# --- local record ----------------------------------------------------------

def test_billing_status_for_an_unlinked_customer_is_not_an_error(conn):
    status = sb.billing_status(conn, "acme")
    assert status["status"] == "unlinked"
    assert status["stripe_customer_id"] is None


def test_link_customer_records_the_stripe_customer_id(conn):
    sb.link_customer(conn, "acme", "billing@acme.example", "cus_1")
    status = sb.billing_status(conn, "acme")
    assert status["status"] == "linked"
    assert status["stripe_customer_id"] == "cus_1"
    assert status["email"] == "billing@acme.example"


def test_link_customer_is_idempotent_and_updates_in_place(conn):
    sb.link_customer(conn, "acme", "old@acme.example", "cus_1")
    sb.link_customer(conn, "acme", "new@acme.example", "cus_1")
    rows = conn.execute("SELECT COUNT(*) AS n FROM billing_accounts").fetchone()
    assert rows["n"] == 1
    assert sb.billing_status(conn, "acme")["email"] == "new@acme.example"


def test_record_subscription_requires_an_existing_billing_account(conn):
    with pytest.raises(ValueError):
        sb.record_subscription(conn, "acme", "sub_1", "active")


def test_record_subscription_updates_status(conn):
    sb.link_customer(conn, "acme", "billing@acme.example", "cus_1")
    sb.record_subscription(conn, "acme", "sub_1", "active")
    status = sb.billing_status(conn, "acme")
    assert status["stripe_subscription_id"] == "sub_1"
    assert status["status"] == "active"


def test_is_active_false_when_unlinked(conn):
    assert sb.is_active(conn, "acme") is False


def test_is_active_true_only_when_status_is_active(conn):
    sb.link_customer(conn, "acme", "billing@acme.example", "cus_1")
    sb.record_subscription(conn, "acme", "sub_1", "past_due")
    assert sb.is_active(conn, "acme") is False
    sb.record_subscription(conn, "acme", "sub_1", "active")
    assert sb.is_active(conn, "acme") is True


# --- orchestration -----------------------------------------------------------

def test_subscribe_customer_goes_from_nothing_to_active(conn):
    fake = FakeStripe(status="active")
    result = sb.subscribe_customer(conn, "acme", "billing@acme.example", "Acme Inc",
                                    "price_flat_3k", client=fake)
    assert result["status"] == "active"
    assert result["stripe_customer_id"] == "cus_1"
    assert result["stripe_subscription_id"] == "sub_1"
    assert fake.Subscription.created[0]["items"] == [{"price": "price_flat_3k"}]


def test_refresh_subscription_status_picks_up_a_stripe_side_change(conn):
    fake = FakeStripe(status="active")
    sb.subscribe_customer(conn, "acme", "billing@acme.example", "Acme Inc",
                           "price_flat_3k", client=fake)

    fake.Subscription.status = "past_due"  # e.g. a card declined, from Stripe's side
    result = sb.refresh_subscription_status(conn, "acme", client=fake)

    assert result["status"] == "past_due"


def test_refresh_subscription_status_is_a_noop_before_any_subscription(conn):
    fake = FakeStripe()
    result = sb.refresh_subscription_status(conn, "acme", client=fake)
    assert result["status"] == "unlinked"
    assert fake.Subscription.created == []  # never called Stripe at all
