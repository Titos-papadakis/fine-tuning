"""Fintech vertical: cardholder dispute intake -> Reg E / chargeback record."""
from __future__ import annotations

from random import Random

from ftspec.core.contract import Contract
from ftspec.core.profile import Compliance, Profile, PromptSet, Sample
from ftspec.core.redaction import assert_masked, scan
from profiles.fintech_disputes import generate as gen
from profiles.fintech_disputes.rubric import RISK_RUBRIC_TEXT
from profiles.fintech_disputes.schema import DisputeRecord

INSTRUCTION = (
    "You are a backend extraction engine for a card issuer's dispute system. "
    "Given a raw cardholder dispute intake, output ONLY a single valid JSON object "
    "that strictly matches the schema below. "
    "Do not include markdown code fences, explanations, or any text outside the JSON object. "
    "Every field in the schema is required. Use null where the schema allows it and no value "
    "was stated. Never output a full card number: card_last4 is exactly four digits and "
    "card_bin is exactly six."
)


class FintechDisputesProfile(Profile):
    name = "fintech_disputes"
    title = "Fintech Disputes — cardholder intake to chargeback record"
    description = (
        "Converts cardholder dispute intakes into structured Reg E / chargeback "
        "records, assigning a risk band via the issuer's own dispute policy."
    )

    # This is the load-bearing part. For a PCI-DSS workload, the hosted-API
    # baselines in the benchmark are not merely more expensive -- routing
    # cardholder data to a third party takes it outside the compliance boundary.
    # The engine refuses those baselines unless explicitly acknowledged.
    compliance = Compliance(
        regime="PCI-DSS",
        allows_external_api=False,
        residency_note=(
            "Cardholder dispute intakes are CHD-adjacent. Under PCI-DSS v4.0 they must "
            "remain inside the assessed cardholder data environment; a hosted inference "
            "API is a third-party service provider requiring its own attestation."
        ),
        forbidden_patterns=("pan", "ssn", "iban"),
        notes=(
            "The contract has no field able to hold a PAN: card_last4 is four digits, "
            "card_bin six. Synthetic data uses reserved test BINs only."
        ),
    )

    free_text_fields = ("case_summary",)
    headline_field = "assessment.risk_band"
    rubric_input_fields = (
        "transaction.amount", "assessment.reason_code",
        "cardholder.prior_disputes_12m", "cardholder.account_tenure_months",
        "assessment.within_reg_e_window",
    )
    field_labels = {"assessment.risk_band": "risk_band (policy-derived)"}

    @property
    def contract(self) -> Contract:
        return Contract.from_model(DisputeRecord, name="DisputeRecord")

    @property
    def prompts(self) -> PromptSet:
        schema_prompt = f"{INSTRUCTION}\n\nJSON Schema:\n{self.contract.schema_text()}"
        return PromptSet(
            short="Extract the dispute record as JSON.",
            schema=schema_prompt,
            schema_rubric=f"{schema_prompt}\n\n{RISK_RUBRIC_TEXT}",
        )

    def generate(self, n: int, rng: Random) -> list:
        return gen.generate(n, rng)

    def audit_sample(self, sample: Sample) -> list:
        """Groundedness plus the PCI controls that make this corpus publishable."""
        text = sample.source_text
        record = sample.record
        problems: list = []

        txn = record["transaction"]
        if txn["amount"] is not None and f"{txn['amount']:.2f}" not in text:
            problems.append(f"amount {txn['amount']} absent from the intake")
        if txn["merchant_name"] and txn["merchant_name"] not in text:
            problems.append(f"merchant {txn['merchant_name']} absent from the intake")
        if txn["transaction_date"] and txn["transaction_date"] not in text:
            problems.append(f"transaction_date {txn['transaction_date']} absent from the intake")

        holder = record["cardholder"]
        if holder["card_last4"] and holder["card_last4"] not in text:
            problems.append("card_last4 not derivable from the intake")

        # PCI control 1: the label must be masked, whatever the source contained.
        problems += [str(f) for f in assert_masked(holder["card_last4"], 4, "cardholder.card_last4")]
        problems += [str(f) for f in assert_masked(holder["card_bin"], 6, "cardholder.card_bin")]

        # PCI control 2: no live-looking PAN anywhere in a corpus we publish.
        problems += [f"source text: {f}" for f in
                      scan(text, self.compliance.forbidden_patterns, context="source_text")]

        sla = record["resolution"]["sla_due_date"]
        if sla and sla not in text:
            problems.append(f"sla_due_date {sla} absent from the intake")

        return problems


def get_profile() -> Profile:
    return FintechDisputesProfile()
