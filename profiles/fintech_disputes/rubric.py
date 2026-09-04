"""
Dispute triage policy.

`risk_band` is the field that makes this vertical worth specializing. It is not
a judgement a general model can make, because it encodes an issuer's own risk
appetite: thresholds, repeat-filer tolerance, and which channel/reason
combinations get escalated. Two banks with identical data assign different
bands.

That produces the three-way tradeoff the benchmark measures:

  * a frontier model without the policy guesses, and is wrong systematically
  * a frontier model with the policy in-prompt is right, and pays ~250 extra
    tokens on every request forever
  * a fine-tuned model has the policy in its weights, and pays nothing

For PCI-DSS workloads there is a fourth consideration that dominates all three:
the first two options require sending cardholder dispute data to a third party.
"""
from __future__ import annotations

from dataclasses import dataclass

RISK_RUBRIC_TEXT = """Assign risk_band using this internal dispute policy, evaluated top-down (first match wins):

1. CRITICAL if any of:
   - the disputed amount exceeds 5000 in any currency
   - the reason is unauthorized_transaction AND the cardholder reports the card was in their possession
   - the cardholder has filed more than 4 disputes in the last 12 months
2. ELEVATED if any of:
   - the disputed amount exceeds 1000
   - the reason is unauthorized_transaction on a card_present or atm channel
   - the account has been open for fewer than 6 months
   - the cardholder has filed 2 or more disputes in the last 12 months
3. LOW if all of:
   - the disputed amount is under 100
   - the reason is duplicate_charge, incorrect_amount or credit_not_processed
   - no fraud indicators are present
4. STANDARD otherwise.

requires_manual_review is true when risk_band is elevated or critical, or when the
report falls outside the 60-day Regulation E window."""

HIGH_RISK_REASONS = {"unauthorized_transaction"}
LOW_RISK_REASONS = {"duplicate_charge", "incorrect_amount", "credit_not_processed"}
CARD_PRESENT_CHANNELS = {"card_present", "atm"}


@dataclass
class DisputeSignals:
    """Facts the policy operates on, extracted from the intake document."""
    amount: float | None
    reason_code: str
    channel: str
    card_in_possession: bool
    account_tenure_months: int | None
    prior_disputes_12m: int | None
    fraud_indicators: tuple
    within_reg_e_window: bool


def derive_risk_band(s: DisputeSignals) -> str:
    """Deterministic execution of RISK_RUBRIC_TEXT.

    Kept in lockstep with the prose above: if one changes and the other does
    not, the in-prompt baseline is being handed a policy the labels do not
    follow, and the comparison becomes dishonest in our favour.
    """
    amount = s.amount or 0.0
    priors = s.prior_disputes_12m or 0

    # 1. CRITICAL
    if amount > 5000:
        return "critical"
    if s.reason_code in HIGH_RISK_REASONS and s.card_in_possession:
        return "critical"
    if priors > 4:
        return "critical"

    # 2. ELEVATED
    if amount > 1000:
        return "elevated"
    if s.reason_code in HIGH_RISK_REASONS and s.channel in CARD_PRESENT_CHANNELS:
        return "elevated"
    if s.account_tenure_months is not None and s.account_tenure_months < 6:
        return "elevated"
    if priors >= 2:
        return "elevated"

    # 3. LOW
    if amount < 100 and s.reason_code in LOW_RISK_REASONS and not s.fraud_indicators:
        return "low"

    # 4. STANDARD
    return "standard"


def derive_manual_review(risk_band: str, within_reg_e_window: bool) -> bool:
    return risk_band in ("elevated", "critical") or not within_reg_e_window
