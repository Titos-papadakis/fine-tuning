"""Healthcare vertical: clinical encounter note -> ICD-10 / FHIR record."""
from __future__ import annotations

from random import Random

from ftspec.core.contract import Contract
from ftspec.core.profile import Compliance, Profile, PromptSet, Sample
from ftspec.core.redaction import scan
from profiles.healthcare_clinical import generate as gen
from profiles.healthcare_clinical.rubric import ACUITY_RUBRIC_TEXT
from profiles.healthcare_clinical.schema import ClinicalRecord

INSTRUCTION = (
    "You are a backend extraction engine for a clinical documentation system. "
    "Given a raw clinical encounter note, output ONLY a single valid JSON object "
    "that strictly matches the schema below. "
    "Do not include markdown code fences, explanations, or any text outside the JSON object. "
    "Every field in the schema is required. Use null where the schema allows it and no value "
    "was recorded. Never output patient names, medical record numbers, addresses or dates of "
    "birth; the schema has no field for them."
)


class HealthcareClinicalProfile(Profile):
    name = "healthcare_clinical"
    title = "Clinical Notes — free text to ICD-10 / FHIR entities"
    description = (
        "Converts free-text clinical encounter notes into FHIR-aligned structured "
        "records with ICD-10-CM coding, assigning encounter acuity via the "
        "institution's own triage protocol."
    )

    # As with the fintech profile, this is enforcement rather than a label.
    # PHI cannot be sent to a hosted inference API without a BAA, so the engine
    # refuses those baselines unless the operator explicitly acknowledges it.
    compliance = Compliance(
        regime="HIPAA",
        allows_external_api=False,
        residency_note=(
            "Encounter notes are PHI. Disclosure to a hosted inference provider requires "
            "a Business Associate Agreement under 45 CFR 164.502(e); absent one, the "
            "disclosure is impermissible regardless of cost."
        ),
        forbidden_patterns=("ssn", "email", "phone", "nhs_mrn", "dob"),
        notes=(
            "Schema follows Safe Harbor by omission: no name, MRN, address or date of "
            "birth field exists, and age_years is capped at 89. The triage protocol in "
            "rubric.py is illustrative scaffolding for benchmarking, not clinical guidance."
        ),
    )

    free_text_fields = ("chief_complaint",)
    headline_field = "encounter.acuity"
    rubric_input_fields = ("encounter.encounter_class",)
    field_labels = {"encounter.acuity": "acuity (protocol-derived)"}

    @property
    def contract(self) -> Contract:
        return Contract.from_model(ClinicalRecord, name="ClinicalRecord")

    @property
    def prompts(self) -> PromptSet:
        schema_prompt = f"{INSTRUCTION}\n\nJSON Schema:\n{self.contract.schema_text()}"
        return PromptSet(
            short="Extract the clinical record as JSON.",
            schema=schema_prompt,
            schema_rubric=f"{schema_prompt}\n\n{ACUITY_RUBRIC_TEXT}",
        )

    def generate(self, n: int, rng: Random) -> list:
        return gen.generate(n, rng)

    def audit_sample(self, sample: Sample) -> list:
        """Groundedness plus the Safe Harbor controls that make this corpus publishable."""
        text = sample.source_text
        record = sample.record
        problems: list = []

        primary = [c for c in record["conditions"] if c["is_primary"]]
        if len(primary) != 1:
            problems.append(f"expected exactly one primary condition, found {len(primary)}")
        for condition in record["conditions"]:
            if condition["icd10_code"] not in text:
                problems.append(f"ICD-10 {condition['icd10_code']} absent from the note")

        for obs in record["observations"]:
            if str(obs["value"]) not in text:
                problems.append(f"observation {obs['code']}={obs['value']} absent from the note")

        for med in record["medications"]:
            if med["name"] not in text:
                problems.append(f"medication {med['name']} absent from the note")

        for allergy in record["allergies"]:
            if allergy not in text:
                problems.append(f"allergy {allergy!r} absent from the note")

        # Safe Harbor control: an age above 89 is itself an identifier.
        age = record["patient"]["age_years"]
        if age is not None and age > 89:
            problems.append(f"age_years {age} exceeds the Safe Harbor cap of 89")

        # Safe Harbor control: no direct identifiers anywhere in a published corpus.
        problems += [f"source text: {f}" for f in
                      scan(text, self.compliance.forbidden_patterns, context="source_text")]

        return problems


def get_profile() -> Profile:
    return HealthcareClinicalProfile()
