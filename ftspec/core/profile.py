"""
The plug-and-play unit: a domain profile.

A profile supplies everything domain-specific and nothing else:

    contract    the output record schema
    prompts     short (fine-tuned) / schema / schema+policy variants
    generate    synthetic corpus for the domain
    audit       domain rules asserting each label is derivable from its input
    compliance  the regulatory envelope the data lives in

The engine supplies everything else: training, validation, constrained
decoding, serving, metrics, benchmarking, cost modelling. Adding a vertical is
therefore a data-and-schema exercise, not an engineering one.

On compliance
-------------
`Compliance.allows_external_api` is load-bearing, not documentation. For a
PCI-DSS or HIPAA workload, sending records to a third-party API is not a cost
tradeoff to be weighed — it is frequently prohibited outright. The benchmark
refuses to run hosted-API baselines for such profiles unless the operator
passes an explicit acknowledgement flag, and the refusal is recorded in the run
manifest.

That inverts the usual sales argument. For regulated verticals a specialized
self-hosted model is not the cheaper option; it is the only admissible one, and
the frontier-model column of the comparison table cannot legally be filled in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random

from ftspec.core.contract import Contract
from ftspec.core.fields import ScoringPlan, derive_scoring_plan


@dataclass(frozen=True)
class PromptSet:
    """Prompt variants, and the cost argument they encode.

    A fine-tuned model needs neither the schema nor the business policy in its
    prompt — both live in its weights. A prompted model pays for both on every
    request, forever. Keeping the variants side by side makes that difference
    measurable instead of rhetorical.
    """
    short: str            # fine-tuned: task name only, ~20 tokens
    schema: str           # baseline: knows the shape, must guess the policy
    schema_rubric: str    # baseline: knows shape and policy, pays for both

    def variants(self) -> dict:
        return {"short": self.short, "schema": self.schema, "schema+rubric": self.schema_rubric}


@dataclass(frozen=True)
class Compliance:
    regime: str = "none"                     # "PCI-DSS", "HIPAA", "GDPR", "none"
    allows_external_api: bool = True         # may source text leave the boundary?
    residency_note: str = ""
    forbidden_patterns: tuple = ()           # redaction rule names from core.redaction
    notes: str = ""

    @property
    def is_regulated(self) -> bool:
        return self.regime != "none"

    def egress_refusal(self) -> str:
        return (
            f"Profile is governed by {self.regime} and declares "
            f"allows_external_api=False. Sending source records to a hosted API "
            f"would move regulated data outside the compliance boundary.\n"
            f"{self.residency_note}\n"
            f"Re-run with --acknowledge-egress only if your DPA and controls genuinely "
            f"permit it; the acknowledgement is recorded in the run manifest."
        )


@dataclass
class Sample:
    """One generated training example."""
    source_text: str        # the unstructured input
    record: dict            # ground-truth structured output
    meta: dict = field(default_factory=dict)   # signals for analysis; stripped before training


class Profile(ABC):
    """Base class for a domain vertical."""

    name: str = ""
    title: str = ""
    description: str = ""
    compliance: Compliance = Compliance()

    # Metric hints; the engine infers the rest from the schema.
    free_text_fields: tuple = ()
    headline_field: str | None = None
    rubric_input_fields: tuple = ()
    field_labels: dict = {}

    # --- required ------------------------------------------------------------

    @property
    @abstractmethod
    def contract(self) -> Contract:
        """The output record contract."""

    @property
    @abstractmethod
    def prompts(self) -> PromptSet:
        """Prompt variants for the fine-tuned model and the prompted baselines."""

    @abstractmethod
    def generate(self, n: int, rng: Random) -> list:
        """Produce `n` synthetic samples. Must be deterministic given `rng`."""

    # --- optional ------------------------------------------------------------

    def audit_sample(self, sample: Sample) -> list:
        """Domain rules proving each label is recoverable from `source_text`.

        A label the input never mentions is unpredictable by construction: pure
        noise that silently caps the achievable score and makes the benchmark
        unfalsifiable. Profiles that generate such a label should say so here.
        """
        return []

    def scoring_plan(self) -> ScoringPlan:
        return derive_scoring_plan(
            self.contract.json_schema(),
            free_text=self.free_text_fields,
            headline=self.headline_field,
            rubric_inputs=self.rubric_input_fields,
            labels=self.field_labels,
        )

    def supports_generation(self) -> bool:
        """False for profiles that only wrap a customer's own schema and data."""
        return True

    def summary(self) -> dict:
        plan = self.scoring_plan()
        return {
            "name": self.name,
            "title": self.title,
            "regime": self.compliance.regime,
            "allows_external_api": self.compliance.allows_external_api,
            "scored_fields": len(plan.fields),
            "headline_field": self.headline_field,
            "supports_generation": self.supports_generation(),
        }
