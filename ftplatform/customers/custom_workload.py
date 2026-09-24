"""
A customer's own schema as their workload -- no profile code per customer.

`ftspec` already builds a working profile from a bare JSON Schema
(registry.CustomProfile: generic extraction prompts, contract validation,
field-by-field scoring). This stores that schema, plus the few choices it
cannot infer (headline field, free-text fields, compliance regime), inside
the customer's own tree at customers/<id>/workload/, so it is isolated like
everything else of theirs and travels with them in a Kaggle job packet.

A custom workload cannot generate synthetic data -- the engine has no idea
what a plausible document in an unseen domain looks like -- so its corpus
always comes from `ftplatform customer import`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ftspec.core.contract import Contract
from ftspec.core.profile import Compliance
from ftspec.core.registry import CustomProfile

CUSTOM = "custom"
REGIMES = ("none", "GDPR", "HIPAA", "PCI-DSS")
# Regimes whose data must not leave the boundary by default -- matches the
# shipped fintech (PCI-DSS) and healthcare (HIPAA) profiles.
NO_EGRESS_BY_DEFAULT = {"HIPAA", "PCI-DSS"}


def workload_dir(customer_root: Path) -> Path:
    return Path(customer_root) / "workload"


def build_profile(schema_path: Path, headline_field: str | None = None,
                  free_text_fields: tuple = (), regime: str = "none",
                  allows_external_api: bool | None = None) -> CustomProfile:
    """Validates everything a later GPU run would otherwise trip over hours in:
    the schema parses, and every named field actually exists in it."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    contract = Contract.from_file(Path(schema_path))
    profile = CustomProfile(contract, headline_field=headline_field,
                            free_text_fields=tuple(free_text_fields))
    paths = set(profile.scoring_plan().paths)
    missing = [f for f in (headline_field, *free_text_fields) if f and f not in paths]
    if missing:
        raise ValueError(f"field(s) {missing} are not in the schema; available: {sorted(paths)}")
    if allows_external_api is None:
        allows_external_api = regime not in NO_EGRESS_BY_DEFAULT
    profile.compliance = Compliance(
        regime=regime, allows_external_api=allows_external_api,
        notes=f"Customer-declared regime {regime}.")
    return profile


def install(customer_root: Path, schema_path: Path, **settings) -> CustomProfile:
    profile = build_profile(schema_path, **settings)
    d = workload_dir(customer_root)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(schema_path, d / "schema.json")
    stored = {"headline_field": settings.get("headline_field"),
              "free_text_fields": list(settings.get("free_text_fields", ())),
              "regime": profile.compliance.regime,
              "allows_external_api": profile.compliance.allows_external_api}
    (d / "workload.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return profile


def load(customer_root: Path) -> CustomProfile:
    d = workload_dir(customer_root)
    if not (d / "schema.json").exists():
        raise FileNotFoundError(f"custom workload has no schema at {d / 'schema.json'}")
    s = json.loads((d / "workload.json").read_text(encoding="utf-8"))
    return build_profile(d / "schema.json", headline_field=s.get("headline_field"),
                         free_text_fields=tuple(s.get("free_text_fields", ())),
                         regime=s.get("regime", "none"),
                         allows_external_api=s.get("allows_external_api"))
