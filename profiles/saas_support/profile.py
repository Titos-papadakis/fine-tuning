"""SaaS support vertical: support transcripts -> structured CRM records."""
from __future__ import annotations

import re
from random import Random

from ftspec.core.contract import Contract
from ftspec.core.profile import Compliance, Profile, PromptSet, Sample
from profiles.saas_support import generate as gen
from profiles.saas_support import scenarios as sc
from profiles.saas_support.rubric import PRIORITY_RUBRIC_TEXT
from profiles.saas_support.schema import SupportTicket


def _action_evidence() -> dict:
    """Map each action label to the utterances that can evidence it.

    The label and the spoken line are deliberately different wordings
    ("Reviewed billing history" vs "I've gone through your billing history"),
    so matching on the label text would flag correct data. The generator pairs
    them, and the audit checks that pairing rather than guessing at it.
    """
    evidence: dict = {}
    for pairs in sc.ACTION_UTTERANCES.values():
        for label, utterance in pairs:
            evidence.setdefault(label, set()).add(utterance)
    for label, en, el in sc.GENERIC_ACTIONS_EL:
        evidence.setdefault(label, set()).update({en, el})
    return evidence


ACTION_EVIDENCE = _action_evidence()

# Tier is stated in either language; the Greek corpus never uses the English word.
TIER_SURFACE_FORMS = {
    "free": ("free", "δωρεάν"),
    "pro": ("pro", "pro"),
    "enterprise": ("enterprise",),
}

INSTRUCTION = (
    "You are a backend extraction engine for a customer support platform. "
    "Given a raw, unstructured customer support transcript, output ONLY a single "
    "valid JSON object that strictly matches the schema below. "
    "Do not include markdown code fences, explanations, or any text outside the JSON object. "
    "Every field in the schema is required. Use null where the schema allows it and no value "
    "was mentioned in the transcript."
)


class SaasSupportProfile(Profile):
    name = "saas_support"
    title = "SaaS Support — transcripts to CRM actions"
    description = (
        "Converts unstructured customer support conversations into structured CRM "
        "records, assigning ticket priority via an internal triage policy that a "
        "general-purpose model cannot know."
    )

    # Ordinary business data: no regulatory bar to a hosted-API comparison, so
    # the full benchmark matrix is available for this profile.
    compliance = Compliance(
        regime="none",
        allows_external_api=True,
        notes="Support transcripts may contain customer PII; production deployments "
               "should still scrub before any third-party call.",
    )

    free_text_fields = ("ticket_summary",)
    headline_field = "issue.priority"
    rubric_input_fields = (
        "customer.tier", "customer.churn_threat",
        "issue.is_repeat_contact", "extracted_entities.amount",
    )
    field_labels = {"issue.priority": "priority (rubric-derived)"}

    @property
    def contract(self) -> Contract:
        return Contract.from_model(SupportTicket, name="SupportTicket")

    @property
    def prompts(self) -> PromptSet:
        schema_text = self.contract.schema_text()
        schema_prompt = f"{INSTRUCTION}\n\nJSON Schema:\n{schema_text}"
        return PromptSet(
            short="Extract the support ticket record as JSON.",
            schema=schema_prompt,
            schema_rubric=f"{schema_prompt}\n\n{PRIORITY_RUBRIC_TEXT}",
        )

    def generate(self, n: int, rng: Random) -> list:
        return gen.generate(n, rng)

    def audit_sample(self, sample: Sample) -> list:
        """Every label must be recoverable from the transcript.

        A label the conversation never mentions cannot be predicted by any model.
        Shipping such a label would cap the achievable score at an arbitrary
        ceiling and quietly make the benchmark unfalsifiable.
        """
        text = sample.source_text
        record = sample.record
        problems: list = []

        entities = record["extracted_entities"]
        if entities["order_id"] and entities["order_id"] not in text:
            problems.append(f"order_id {entities['order_id']} absent from transcript")
        if entities["product_name"] and entities["product_name"] not in text:
            problems.append(f"product_name {entities['product_name']} absent from transcript")
        if entities["amount"] is not None and f"{entities['amount']:.2f}" not in text:
            problems.append(f"amount {entities['amount']} absent from transcript")

        followup = record["resolution"]["followup_date"]
        if followup and followup not in text:
            problems.append(f"followup_date {followup} absent from transcript")

        # Actions are spoken by the agent, so each label needs its paired line present.
        for action in record["actions_taken"]:
            utterances = ACTION_EVIDENCE.get(action)
            if utterances is None:
                problems.append(f"action {action!r} is not in any action pool")
            elif not any(u in text for u in utterances):
                problems.append(f"action {action!r} has no supporting line in the transcript")

        tier = record["customer"]["tier"]
        if tier != "unknown":
            forms = TIER_SURFACE_FORMS.get(tier, (tier,))
            if not any(f.lower() in text.lower() for f in forms):
                problems.append(f"tier {tier!r} never stated in the transcript")

        # A decoy identifier must never be the answer.
        for decoy in re.findall(r"\b(?:TKT|INV|CASE)-\d+\b", text):
            if entities["order_id"] == decoy:
                problems.append(f"decoy reference {decoy} was extracted as order_id")

        return problems


def get_profile() -> Profile:
    return SaasSupportProfile()
