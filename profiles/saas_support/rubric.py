"""
The company-specific triage rubric.

THIS FILE IS THE POINT OF THE ENTIRE PROOF-OF-CONCEPT.

`priority` is not a field a model can guess from general knowledge -- it is
determined by an internal business rule that combines several signals scattered
across the conversation (customer tier, whether this is a repeat contact,
whether the customer threatened to churn, order value, issue category).

That creates a genuine three-way tradeoff which is what we actually benchmark:

  1. A frontier model WITHOUT the rubric in its prompt has to guess.
     -> cheap-ish prompt, but wrong `priority` on a large fraction of tickets.
  2. A frontier model WITH the rubric in its prompt gets it right.
     -> but pays for ~200 extra prompt tokens on EVERY request, forever.
  3. A fine-tuned small model has the rubric baked into its weights.
     -> right answer, ~20-token prompt, self-hosted cost.

Only (3) gets both. That is the value of fine-tuning, and it is a value that
constrained decoding / structured outputs / prompt engineering cannot replicate,
because the problem was never syntax -- it was encoding proprietary judgment.

Swap this file (plus data/schema.json) for your own domain rules and the rest of
the pipeline is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

# Human-readable form of the rubric, injected into prompts for the
# "frontier model WITH rubric" baseline. Kept deliberately terse -- this is a
# realistic, already-compressed version of a real internal policy doc.
PRIORITY_RUBRIC_TEXT = """Priority must be assigned using this internal triage policy, evaluated top-down (first match wins):

1. URGENT if any of:
   - the customer threatens to cancel/churn AND their tier is pro or enterprise
   - tier is enterprise AND the issue is a failed payment, a broken integration, or a sync failure
   - there is a risk of permanent data loss
2. HIGH if any of:
   - the customer threatens to cancel/churn (any tier)
   - this is a repeat contact about the same issue AND sentiment is negative or frustrated
   - tier is enterprise AND category is billing, technical, or refund
   - a disputed monetary amount above 300 is mentioned
   - the customer is reporting a phishing/security concern
3. MEDIUM if any of:
   - sentiment is negative or frustrated
   - category is billing, refund, shipping, technical, or account
4. LOW otherwise."""

# Subcategories that, for an enterprise account, are treated as revenue- or
# operations-blocking and jump straight to urgent.
CRITICAL_ENTERPRISE_SUBCATS = {"payment_failed", "integration_bug", "sync_error"}


@dataclass
class TriageSignals:
    """The extracted facts the rubric operates on."""
    category: str
    subcategory: str
    sentiment: str
    tier: str                      # free | pro | enterprise | unknown
    is_repeat_contact: bool
    churn_threat: bool
    data_loss_risk: bool
    amount: float | None


def derive_priority(s: TriageSignals) -> str:
    """Deterministic application of PRIORITY_RUBRIC_TEXT.

    Kept in lockstep with the prose version above -- if you change one, change
    both, or the "frontier model WITH rubric" baseline becomes unfair.
    """
    # 1. URGENT
    if s.churn_threat and s.tier in ("pro", "enterprise"):
        return "urgent"
    if s.tier == "enterprise" and s.subcategory in CRITICAL_ENTERPRISE_SUBCATS:
        return "urgent"
    if s.data_loss_risk:
        return "urgent"

    # 2. HIGH
    if s.churn_threat:
        return "high"
    if s.is_repeat_contact and s.sentiment in ("negative", "frustrated"):
        return "high"
    if s.tier == "enterprise" and s.category in ("billing", "technical", "refund"):
        return "high"
    if s.amount is not None and s.amount > 300:
        return "high"
    if s.subcategory == "spam_report":
        return "high"

    # 3. MEDIUM
    if s.sentiment in ("negative", "frustrated"):
        return "medium"
    if s.category in ("billing", "refund", "shipping", "technical", "account"):
        return "medium"

    # 4. LOW
    return "low"
