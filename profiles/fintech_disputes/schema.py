"""
Fintech dispute contract: cardholder dispute intake -> Reg E / chargeback record.

PCI-DSS shapes this schema directly rather than decorating it. The contract has
no field capable of holding a Primary Account Number: `card_last4` is four
digits and `card_bin` is six, so a model cannot store a full PAN even if the
source document contains one. Schema design is the cheapest possible control —
a field that cannot exist cannot leak.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STRICT = ConfigDict(extra="forbid")

DisputeReason = Literal[
    "unauthorized_transaction", "duplicate_charge", "goods_not_received",
    "goods_not_as_described", "incorrect_amount", "subscription_not_cancelled",
    "atm_dispensing_error", "credit_not_processed",
]
Network = Literal["visa", "mastercard", "amex", "discover", "unknown"]
Channel = Literal["card_present", "ecommerce", "atm", "recurring", "unknown"]
Outcome = Literal[
    "provisional_credit_issued", "chargeback_filed", "representment_expected",
    "denied_insufficient_evidence", "pending_documentation", "withdrawn_by_customer",
]
RiskBand = Literal["low", "standard", "elevated", "critical"]


class Cardholder(BaseModel):
    model_config = STRICT

    # Deliberately masked. There is no field here that can hold a full PAN.
    card_last4: str | None = Field(
        default=None, min_length=4, max_length=4,
        description="Exactly the last four digits of the card. Never the full number.")
    card_bin: str | None = Field(
        default=None, min_length=6, max_length=6,
        description="Issuer BIN, first six digits only.")
    network: Network
    account_tenure_months: int | None = Field(
        default=None, ge=0, description="Months the account has been open, if stated.")
    prior_disputes_12m: int | None = Field(
        default=None, ge=0, description="Disputes filed by this cardholder in the last 12 months.")


class Transaction(BaseModel):
    model_config = STRICT

    merchant_name: str | None = None
    merchant_category: str | None = Field(
        default=None, description="Merchant category as described, e.g. 'airline', 'grocery'.")
    amount: float | None = Field(
        default=None, description="The disputed amount, not incidental figures.")
    currency: str = Field(description="ISO 4217 code, e.g. 'USD', 'EUR'.")
    transaction_date: str | None = Field(default=None, description="ISO 8601 date or null.")
    channel: Channel
    is_recurring: bool


class DisputeAssessment(BaseModel):
    model_config = STRICT

    reason_code: DisputeReason
    # Derived by executing the dispute policy, not by intuition.
    risk_band: RiskBand = Field(description="Assigned per the internal dispute policy.")
    within_reg_e_window: bool = Field(
        description="True if reported within 60 days of the statement date.")
    requires_manual_review: bool
    fraud_indicators: list[str] = Field(
        description="Distinct fraud signals stated in the intake, e.g. 'card_in_possession'.")


class Resolution(BaseModel):
    model_config = STRICT

    outcome: Outcome
    provisional_credit_amount: float | None = Field(
        default=None, description="Provisional credit issued, or null.")
    sla_due_date: str | None = Field(default=None, description="ISO 8601 date or null.")
    documents_requested: list[str]


class DisputeRecord(BaseModel):
    """A structured dispute intake record."""
    model_config = STRICT

    case_summary: str = Field(min_length=1, description="One-sentence summary of the dispute.")
    cardholder: Cardholder
    transaction: Transaction
    assessment: DisputeAssessment
    resolution: Resolution
