"""
SaaS support contract: transcript -> structured CRM record.

This one Pydantic model is consumed by four independent subsystems, which is
what keeps them from drifting apart:

    1. data validation      SupportTicket.model_validate_json(...)  [strict]
    2. prompt construction  SupportTicket.model_json_schema()
    3. local decoding       Outlines grammar built from the same schema
    4. production serving   vLLM `guided_json` on every request

In a hand-maintained setup these are four copies of the same schema that
silently diverge: the prompt says one thing, the validator enforces another,
and the server guarantees a third. Here, changing a field changes all four.

Validation is STRICT on purpose. A model that emits `"amount": "253.30"` where
the contract says number has violated the contract, even though a lenient
parser would coerce it. Downstream systems that assume a number will break on
a string, so the benchmark must count it as a failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Extra keys are a contract violation, not a curiosity: a consumer that
# iterates fields will choke on an invented one.
STRICT = ConfigDict(extra="forbid")

Sentiment = Literal["positive", "neutral", "negative", "frustrated"]
Tier = Literal["free", "pro", "enterprise", "unknown"]
Category = Literal["billing", "technical", "shipping", "account", "product", "refund", "other"]
Priority = Literal["low", "medium", "high", "urgent"]
Status = Literal["resolved", "pending", "escalated", "unresolved"]


class Customer(BaseModel):
    model_config = STRICT

    sentiment: Sentiment
    language: str = Field(description="ISO 639-1 language code, e.g. 'en', 'el'.")
    tier: Tier = Field(description="Subscription tier if stated anywhere, else 'unknown'.")
    churn_threat: bool = Field(description="True if the customer threatens to cancel or leave.")


class Issue(BaseModel):
    model_config = STRICT

    category: Category
    subcategory: str = Field(min_length=1)
    priority: Priority = Field(description="Assigned per the internal triage policy, not by intuition.")
    is_repeat_contact: bool = Field(description="True if the customer has contacted support about this before.")


class Resolution(BaseModel):
    model_config = STRICT

    status: Status
    requires_followup: bool
    followup_date: str | None = Field(description="ISO 8601 date (YYYY-MM-DD) or null.")


class ExtractedEntities(BaseModel):
    model_config = STRICT

    order_id: str | None = Field(
        description="Order reference in ORD-XXXXX form only. Ticket refs, invoice numbers "
                     "and phone numbers are NOT order ids."
    )
    product_name: str | None = None
    amount: float | None = Field(
        description="The disputed or charged amount central to the issue, not incidental figures."
    )


class SupportTicket(BaseModel):
    """The record extracted from a raw customer-support transcript."""
    model_config = STRICT

    ticket_summary: str = Field(min_length=1, description="One-sentence summary of the conversation.")
    customer: Customer
    issue: Issue
    actions_taken: list[str]
    resolution: Resolution
    extracted_entities: ExtractedEntities
