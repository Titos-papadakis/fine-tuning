"""
Actually charging a customer -- CRUD against `billing_accounts` (pure sqlite,
no network) plus thin, dependency-injected wrappers around the Stripe SDK
(same pattern as ftplatform.remote.kaggle_ops's `run=subprocess.run`: every
network-touching function takes a `client`, defaulting to a lazy import of
the real `stripe` module, so tests exercise this module's own logic against
a fake client rather than a real Stripe account).

Billing model: one flat-fee monthly subscription per customer (Stripe
Price/Product configured once, by hand, in the Stripe dashboard -- this
module never creates prices, only subscribes a customer to one that already
exists). ftplatform.billing.usage's per-request counters remain the
usage/cost *visibility* layer (what a customer actually consumed); this
module is the separate question of whether they're a paying, active
subscriber at all. Nothing here gates serving on subscription status --
see is_active() if a future caller wants that.

Requires the `stripe` package (`pip install ftspec[billing]`) only when a
`client` isn't supplied -- importable and testable without it installed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stripe(client=None):
    if client is not None:
        return client
    import stripe  # lazy: only needed when actually talking to Stripe
    return stripe


# --- talking to Stripe ---------------------------------------------------

def create_stripe_customer(customer_id: str, email: str, name: str, client=None) -> str:
    """Creates a Stripe Customer, tagged with our own customer_id so a
    support agent looking at the Stripe dashboard can trace it back.
    Returns the Stripe customer id -- callers persist it via link_customer()."""
    stripe = _stripe(client)
    customer = stripe.Customer.create(
        email=email, name=name, metadata={"ftplatform_customer_id": customer_id})
    return customer["id"]


def create_subscription(stripe_customer_id: str, price_id: str, client=None) -> str:
    """Subscribes an existing Stripe customer to a pre-configured Price
    (the flat monthly fee). Returns the Stripe subscription id."""
    stripe = _stripe(client)
    subscription = stripe.Subscription.create(
        customer=stripe_customer_id, items=[{"price": price_id}])
    return subscription["id"]


def fetch_subscription_status(stripe_subscription_id: str, client=None) -> str:
    """Stripe's own live status for this subscription -- 'active',
    'past_due', 'canceled', 'unpaid', etc. Never cached beyond one call;
    refresh_subscription_status() is what re-checks and persists it."""
    stripe = _stripe(client)
    subscription = stripe.Subscription.retrieve(stripe_subscription_id)
    return subscription["status"]


# --- local record (pure sqlite, no network) -------------------------------

def link_customer(conn: sqlite3.Connection, customer_id: str, email: str,
                   stripe_customer_id: str) -> None:
    """Records that `customer_id` now has a Stripe customer, before any
    subscription exists yet -- status starts at 'linked'."""
    conn.execute(
        "INSERT INTO billing_accounts (customer_id, email, stripe_customer_id, status, "
        "updated_at) VALUES (?, ?, ?, 'linked', ?) "
        "ON CONFLICT(customer_id) DO UPDATE SET "
        "email = excluded.email, stripe_customer_id = excluded.stripe_customer_id, "
        "status = 'linked', updated_at = excluded.updated_at",
        (customer_id, email, stripe_customer_id, _now()))
    conn.commit()


def record_subscription(conn: sqlite3.Connection, customer_id: str,
                         stripe_subscription_id: str, status: str) -> None:
    """Persists a subscription id and its current status against an
    already-linked customer. Raises if the customer was never linked --
    a subscription cannot exist without a Stripe customer to own it."""
    cursor = conn.execute(
        "UPDATE billing_accounts SET stripe_subscription_id = ?, status = ?, updated_at = ? "
        "WHERE customer_id = ?",
        (stripe_subscription_id, status, _now(), customer_id))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError(f"customer {customer_id!r} has no billing account -- "
                          f"call link_customer() (or subscribe_customer()) first")


def billing_status(conn: sqlite3.Connection, customer_id: str) -> dict:
    """This customer's billing account, or an all-None 'unlinked' shape if
    they've never been linked to Stripe at all -- not an error, since most
    customers spend a while being trained/evaluated before any money is
    involved."""
    row = conn.execute(
        "SELECT * FROM billing_accounts WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        return {"customer_id": customer_id, "email": None, "stripe_customer_id": None,
                 "stripe_subscription_id": None, "status": "unlinked", "updated_at": None}
    return dict(row)


def is_active(conn: sqlite3.Connection, customer_id: str) -> bool:
    return billing_status(conn, customer_id)["status"] == "active"


# --- orchestration: the one call each CLI command needs -------------------

def subscribe_customer(conn: sqlite3.Connection, customer_id: str, email: str, name: str,
                        price_id: str, client=None) -> dict:
    """Creates a Stripe customer, links it locally, subscribes it to
    `price_id`, and records the resulting status -- the full path from
    'we have a customer' to 'we can charge them $X/month'."""
    stripe_customer_id = create_stripe_customer(customer_id, email, name, client=client)
    link_customer(conn, customer_id, email, stripe_customer_id)
    stripe_subscription_id = create_subscription(stripe_customer_id, price_id, client=client)
    status = fetch_subscription_status(stripe_subscription_id, client=client)
    record_subscription(conn, customer_id, stripe_subscription_id, status)
    return billing_status(conn, customer_id)


def refresh_subscription_status(conn: sqlite3.Connection, customer_id: str, client=None) -> dict:
    """Re-checks Stripe for this customer's current subscription status
    (a payment failed, a subscription was canceled from the Stripe
    dashboard, ...) and updates the local record to match. A customer with
    no subscription yet is returned unchanged -- nothing to refresh."""
    account = billing_status(conn, customer_id)
    if account["stripe_subscription_id"] is None:
        return account
    status = fetch_subscription_status(account["stripe_subscription_id"], client=client)
    record_subscription(conn, customer_id, account["stripe_subscription_id"], status)
    return billing_status(conn, customer_id)
