"""
The single choke point through which every orchestrator call reaches a
customer's data and model artefacts.

No other module in `ftplatform` is allowed to construct a `Config`/`Profile`
pair directly. Isolation is a convention enforced by funneling every path
through here -- one sqlite row plus one directory subtree per customer, not
OS-level sandboxing. That's an honest, appropriate level of isolation for a
solo operator's pilot customers; it would not satisfy an enterprise security
review that expects tenant isolation enforced below the application layer.

The mechanism reuses `ftspec.config.Config` unmodified: `data.root` (already
configurable) and `outputs_root` (added for this purpose) are pointed at
`customers/<id>/...` instead of the repo-wide `data/`/`outputs/` trees, so
every existing `Config` path method -- `data_dir`, `outputs_dir`,
`adapter_dir`, `reports_dir`, `manifests_dir` -- comes along for free,
already isolated, with no changes to those methods themselves.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ftplatform.customers import store
from ftplatform.customers.models import Customer
from ftspec.config import Config
from ftspec.core.profile import Profile
from ftspec.core.registry import load_profile

PRODUCTION = "production"


class UnknownCustomerError(Exception):
    pass


class CustomerContext:
    """Resolves a customer id to an isolated `Config`/`Profile` pair."""

    def __init__(self, conn: sqlite3.Connection, customer_id: str):
        customer = store.get(conn, customer_id)
        if customer is None:
            raise UnknownCustomerError(f"no such customer: {customer_id!r}")
        self.customer: Customer = customer
        self.profile: Profile = load_profile(customer.workload)

        cfg = Config(profile=customer.workload)
        cfg.data.root = f"customers/{customer.id}/data"
        cfg.outputs_root = f"customers/{customer.id}/model/{customer.workload}"
        self.config = cfg

    # --- data ------------------------------------------------------------

    def data_dir(self) -> Path:
        return self.config.data_dir(self.profile.name)

    # --- model artefacts, keyed by candidate id ---------------------------
    # "production" is just another key -- the currently-deployed candidate's
    # artefacts live at the same path shape as any other candidate's, one
    # level up from the candidates/ subtree (see the plan's directory layout).

    def _key(self, candidate_id: str) -> str:
        return candidate_id if candidate_id == PRODUCTION else f"candidates/{candidate_id}"

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.config.outputs_dir(self._key(candidate_id))

    def adapter_dir(self, candidate_id: str) -> Path:
        return self.config.adapter_dir(self._key(candidate_id))

    def reports_dir(self, candidate_id: str) -> Path:
        return self.config.reports_dir(self._key(candidate_id))

    def manifests_dir(self, candidate_id: str) -> Path:
        return self.config.manifests_dir(self._key(candidate_id))

    def production_dir(self) -> Path:
        return self.candidate_dir(PRODUCTION)

    # --- tenant-level state, not tied to any one candidate -----------------
    # Baseline/deployment bookkeeping and (from Phase 6) captured production
    # traffic. Sits beside model/, not under it -- it outlives any single
    # candidate.

    def memory_dir(self) -> Path:
        return self.config.resolve(f"customers/{self.customer.id}/memory")
