"""
Profile discovery.

Three ways to get a profile, in ascending order of customer investment:

    --profile saas_support                     a shipped vertical
    --profile custom --schema mine.json        their schema, engine-supplied everything else
    a package exposing an `ftspec.profiles` entry point   their own installable vertical

The third path matters commercially: a customer can keep a proprietary schema,
rubric and data generator in a private package and still run this engine
unmodified. Nothing about their domain has to be contributed upstream.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from ftspec.core.contract import Contract
from ftspec.core.profile import Compliance, Profile, PromptSet
from ftspec.run import get_logger

log = get_logger("ftspec.registry")

BUILTIN = {
    "saas_support": "profiles.saas_support",
    "fintech_disputes": "profiles.fintech_disputes",
    "healthcare_clinical": "profiles.healthcare_clinical",
}

ALIASES = {
    "saas": "saas_support",
    "support": "saas_support",
    "fintech": "fintech_disputes",
    "disputes": "fintech_disputes",
    "healthcare": "healthcare_clinical",
    "clinical": "healthcare_clinical",
}

GENERIC_INSTRUCTION = (
    "You are a backend extraction engine. Given a raw, unstructured document, "
    "output ONLY a single valid JSON object that strictly matches the schema below. "
    "Do not include markdown code fences, explanations, or any text outside the JSON object. "
    "Every field in the schema is required. Use null where the schema allows it and no "
    "value is present in the document."
)


class CustomProfile(Profile):
    """A customer's own JSON Schema, with the engine supplying everything else.

    Generation is unavailable: the engine has no idea what a plausible document
    in an unseen domain looks like, and inventing one would produce a corpus
    that trains the model on fiction. Customers on this path bring their own
    data via `ftspec prepare --from-jsonl`.
    """

    def __init__(self, contract: Contract, name: str = "custom",
                  headline_field: str | None = None,
                  free_text_fields: tuple = ()):
        self.name = name
        self.title = f"Custom contract ({contract.name})"
        self.description = "Customer-supplied JSON Schema."
        self.compliance = Compliance(
            regime="none",
            allows_external_api=True,
            notes="No regulatory envelope declared. Set one in a profile subclass "
                   "if this data is governed.",
        )
        self.headline_field = headline_field
        self.free_text_fields = free_text_fields
        self._contract = contract

    @property
    def contract(self) -> Contract:
        return self._contract

    @property
    def prompts(self) -> PromptSet:
        schema_prompt = f"{GENERIC_INSTRUCTION}\n\nJSON Schema:\n{self._contract.schema_text()}"
        return PromptSet(
            short="Extract the record as JSON.",
            schema=schema_prompt,
            # With no declared business policy the rubric variant equals the
            # schema variant; the benchmark will show zero prompt-tax difference
            # between them, which is the honest result for this profile.
            schema_rubric=schema_prompt,
        )

    def generate(self, n: int, rng) -> list:
        raise NotImplementedError(
            "Custom profiles do not generate synthetic data. Supply your own corpus:\n"
            "  ftspec prepare --profile custom --schema mine.json --from-jsonl mydata.jsonl"
        )

    def supports_generation(self) -> bool:
        return False


def _load_entry_point(name: str) -> Profile | None:
    """Look for a third-party profile registered under the `ftspec.profiles` group."""
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="ftspec.profiles"):
            if ep.name == name:
                factory = ep.load()
                return factory() if callable(factory) else factory
    except Exception as e:
        log.debug("entry-point lookup failed for %s: %s", name, e)
    return None


def available() -> list:
    names = list(BUILTIN)
    try:
        from importlib.metadata import entry_points
        names += [ep.name for ep in entry_points(group="ftspec.profiles")]
    except Exception:
        pass
    return sorted(set(names))


def load_profile(name: str, schema_path: Path | None = None,
                  headline_field: str | None = None) -> Profile:
    """Resolve a profile by name, or build one from a supplied JSON Schema."""
    key = ALIASES.get(name, name)

    if key == "custom":
        if schema_path is None:
            raise ValueError("--profile custom requires --schema pointing at a JSON Schema file")
        contract = Contract.from_file(Path(schema_path))
        log.info("custom contract loaded from %s (%d required fields)",
                  schema_path, len(contract.required_fields()))
        return CustomProfile(contract, headline_field=headline_field)

    if schema_path is not None:
        raise ValueError(f"--schema is only valid with --profile custom, not '{name}'")

    module_path = BUILTIN.get(key)
    if module_path:
        module = importlib.import_module(module_path)
        profile = module.get_profile()
        log.info("profile '%s' loaded (%s, external API %s)", profile.name,
                  profile.compliance.regime,
                  "allowed" if profile.compliance.allows_external_api else "PROHIBITED")
        return profile

    external = _load_entry_point(key)
    if external is not None:
        log.info("third-party profile '%s' loaded", key)
        return external

    raise ValueError(f"Unknown profile '{name}'. Available: {available() + ['custom']}")
