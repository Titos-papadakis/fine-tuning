"""
Multi-tenant serving: one shared vLLM engine, one base model, every
customer's currently-deployed LoRA adapter registered against it and
selected per request by customer id (see ftspec.serving.serve's
`_select_lora_request()` / multi-adapter `load_engine()`/`serve()`).

Instead of one model process per customer, N customers on the same workload
share one GPU's worth of base-model weights and pay only the (much smaller)
per-adapter cost -- the "N customers, one GPU" side of the research behind
ftplatform/learning/ (that module shares technique *statistics* across
customers; this shares serving *infrastructure*, never any customer's data
or weights with another -- each adapter is only ever selected for its own
requests, never mixed into another customer's).

Scoped to exactly one workload at a time: customers on different workloads
have different schemas and, per group_by_base_model() below, possibly
different base models too -- there is no guarantee every customer on one
workload converged on the same Stage-A winner. Run one shared server per
(workload, base_model) group, not one for the whole platform.
"""
from __future__ import annotations

import sqlite3

from ftplatform.customers.context import PRODUCTION, CustomerContext
from ftplatform.customers.store import list_all
from ftplatform.deployment import registry
from ftplatform.deployment.deploy import infer_base_model
from ftspec.run import get_logger

log = get_logger("ftplatform.serving.shared_server")


def discover_production_adapters(conn: sqlite3.Connection, workload: str) -> dict[str, dict]:
    """{customer_id: {"adapter_path": str, "base_model": str | None}} for
    every active customer on `workload` with a current production
    deployment whose adapter directory exists on disk. A customer with no
    deployment yet (still onboarding, or gated out by deploy.maybe_deploy())
    is silently skipped rather than an error -- call this again as more
    customers go live rather than caching its result once. `base_model` is
    None when it can't be inferred from the candidate's manifests (see
    deploy.infer_base_model) -- such a customer is discoverable but
    group_by_base_model() below will drop them, since which engine to put
    them on can't be determined."""
    found: dict[str, dict] = {}
    for customer in list_all(conn):
        if customer.workload != workload or customer.status != "active":
            continue
        current = registry.current(conn, customer.id)
        if current is None:
            continue
        ctx = CustomerContext(conn, customer.id)
        adapter_path = ctx.adapter_dir(PRODUCTION)
        if not adapter_path.exists():
            continue
        found[customer.id] = {
            "adapter_path": str(adapter_path),
            "base_model": infer_base_model(ctx, current["candidate_id"]),
        }
    return found


def group_by_base_model(adapters: dict[str, dict]) -> dict[str, dict[str, str]]:
    """{base_model: {customer_id: adapter_path}}, from
    discover_production_adapters()'s output. vLLM's multi-adapter engine can
    only serve adapters that share one base model -- a real possibility on
    the same workload, since different customers' Stage-A sweeps can
    legitimately pick different winners for their own data. A customer
    whose base_model is None (undeterminable) is dropped, logged, not
    silently placed in some group."""
    groups: dict[str, dict[str, str]] = {}
    for customer_id, info in adapters.items():
        base_model = info["base_model"]
        if base_model is None:
            log.warning("skipping %s: could not determine its base model from its "
                        "candidate manifests", customer_id)
            continue
        groups.setdefault(base_model, {})[customer_id] = info["adapter_path"]
    return groups
