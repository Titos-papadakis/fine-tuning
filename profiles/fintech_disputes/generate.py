"""
Synthetic dispute intakes.

Same signal-first discipline as the other verticals: sample the facts, execute
the policy to get `risk_band`, then render an intake document that encodes those
facts indirectly, with decoys.

PCI-DSS note: card numbers appearing in generated text are drawn exclusively
from reserved test BIN ranges (4111 1111..., 5555 5555...). Those ranges are
never issued, so this corpus cannot collide with a live PAN — which matters
because it is committed to a public repository. The corresponding label still
stores only `card_last4`, so the model is trained to mask rather than to echo.
"""
from __future__ import annotations

import hashlib
import random

from ftspec.core.profile import Sample
from ftspec.core.redaction import SAFE_TEST_PANS
from profiles.fintech_disputes.rubric import (
    DisputeSignals,
    derive_manual_review,
    derive_risk_band,
)

MERCHANTS = [
    ("SkyLine Airways", "airline"), ("Bytemart Online", "ecommerce"),
    ("GreenGrocer Ltd", "grocery"), ("UrbanFuel Stations", "fuel"),
    ("Nimbus Hosting", "cloud_services"), ("CineMax Theatres", "entertainment"),
    ("PrimeFit Gyms", "fitness"), ("Continental Hotels", "lodging"),
    ("QuickCab", "transport"), ("Vantage Electronics", "electronics"),
]

NETWORKS = ["visa", "mastercard", "amex", "discover"]
CURRENCIES = ["USD", "EUR", "GBP"]

REASON_NARRATIVES = {
    "unauthorized_transaction": [
        "I never made this transaction at {merchant}. I don't recognise it at all.",
        "There's a charge from {merchant} on my statement that I did not authorise.",
    ],
    "duplicate_charge": [
        "{merchant} charged me twice for the same purchase on the same day.",
        "I see two identical {merchant} charges and I only bought once.",
    ],
    "goods_not_received": [
        "I paid {merchant} but the order never arrived, it's been over a month.",
        "{merchant} took the payment and nothing was ever delivered.",
    ],
    "goods_not_as_described": [
        "What {merchant} sent is completely different from what was advertised.",
        "The item from {merchant} is not what the listing described at all.",
    ],
    "incorrect_amount": [
        "{merchant} charged more than the price I agreed to at checkout.",
        "The amount {merchant} took doesn't match my receipt.",
    ],
    "subscription_not_cancelled": [
        "I cancelled my {merchant} subscription weeks ago and they billed me again.",
        "{merchant} kept charging me after I cancelled.",
    ],
    "atm_dispensing_error": [
        "The {merchant} ATM debited my account but never dispensed the cash.",
        "I requested a withdrawal at {merchant} and no money came out, but I was charged.",
    ],
    "credit_not_processed": [
        "{merchant} agreed to refund me and the credit has never appeared.",
        "I returned the item to {merchant} and the promised refund never came.",
    ],
}

POSSESSION_LINES = [
    "The card has been in my wallet the whole time, I still have it.",
    "I never lost the card, it's right here with me.",
    "I've had the card in my possession the entire period.",
]

TENURE_LINES = [
    "I've banked with you for about {months} months.",
    "My account was opened roughly {months} months ago.",
]

PRIOR_DISPUTE_LINES = [
    "I've had to raise {n} disputes this year already.",
    "This is my {n}th dispute in the past twelve months.",
]

FRAUD_INDICATOR_LINES = {
    "card_in_possession": POSSESSION_LINES,
    "multiple_rapid_attempts": [
        "There were several attempted charges within a few minutes of each other.",
        "I can see a run of attempts back to back on the same day.",
    ],
    "foreign_geography": [
        "The transaction shows a country I have never travelled to.",
        "It was posted from overseas and I haven't left the country.",
    ],
    "unrecognised_device": [
        "Your app says it was approved on a device I don't own.",
        "The confirmation went to a device that isn't mine.",
    ],
}

DOCUMENT_OPTIONS = [
    "cardholder_statement", "merchant_receipt", "proof_of_return",
    "police_report", "cancellation_confirmation", "delivery_tracking",
]

# Decoys: numbers a careless extractor might mistake for the disputed amount.
DECOY_LINES = [
    "My monthly account fee of {decoy:.2f} {currency} is separate and correct.",
    "There's also a {decoy:.2f} {currency} charge from a different merchant which is fine.",
    "I did spend {decoy:.2f} {currency} at another shop that day, that one is legitimate.",
]

CASE_REF_LINES = [
    "Your reference for this is CASE-{ref}.",
    "The agent gave me ticket number TKT-{ref}.",
]


def _pan_for(network: str, rng: random.Random) -> str:
    """A reserved test PAN, never a live one."""
    visa = [p for p in SAFE_TEST_PANS if p.startswith("4")]
    mc = [p for p in SAFE_TEST_PANS if p.startswith("5")]
    amex = [p for p in SAFE_TEST_PANS if p.startswith("3")]
    pool = {"visa": visa, "mastercard": mc, "amex": amex}.get(network, visa) or visa
    return rng.choice(pool)


def build_signals(rng: random.Random) -> dict:
    reason = rng.choice(list(REASON_NARRATIVES))
    merchant, category = rng.choice(MERCHANTS)
    network = rng.choice(NETWORKS)
    pan = _pan_for(network, rng)

    channel = rng.choice(["card_present", "ecommerce", "atm", "recurring"]) \
        if reason != "atm_dispensing_error" else "atm"
    if reason == "subscription_not_cancelled":
        channel = "recurring"

    # Amounts straddle the 100 / 1000 / 5000 policy thresholds so those
    # boundaries genuinely have to be learned rather than guessed. The weights
    # are chosen to keep the four risk bands reasonably balanced: an unweighted
    # draw sends over half the corpus to `critical`, which would make accuracy
    # on the headline field almost uninterpretable.
    band = rng.choices([(5, 99), (100, 999), (1000, 4999), (5000, 12000)],
                        weights=[0.34, 0.38, 0.21, 0.07], k=1)[0]
    amount = round(rng.uniform(*band), 2)

    indicators: list = []
    card_in_possession = False
    if reason == "unauthorized_transaction":
        card_in_possession = rng.random() < 0.35
        if card_in_possession:
            indicators.append("card_in_possession")
        for name in ("multiple_rapid_attempts", "foreign_geography", "unrecognised_device"):
            if rng.random() < 0.25:
                indicators.append(name)

    return {
        "reason": reason,
        "merchant": merchant,
        "merchant_category": category,
        "network": network,
        "pan": pan,
        "channel": channel,
        "amount": amount,
        "currency": rng.choice(CURRENCIES),
        "card_in_possession": card_in_possession,
        "fraud_indicators": sorted(set(indicators)),
        # Both of these are ELEVATED triggers, so they are sampled sparsely
        # enough to leave room for the LOW band to occur at all.
        "account_tenure_months": rng.choices(
            [None, rng.randint(1, 5), rng.randint(6, 120)], weights=[0.34, 0.18, 0.48], k=1)[0],
        "prior_disputes_12m": rng.choices(
            [None, 0, 1, 2, 3, 5], weights=[0.28, 0.30, 0.20, 0.10, 0.07, 0.05], k=1)[0],
        "within_reg_e_window": rng.random() < 0.78,
        "transaction_date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "decoy": round(rng.uniform(3, 60), 2),
        "case_ref": rng.randint(1000, 9999),
    }


def render_intake(s: dict, rng: random.Random) -> str:
    lines = [
        "Channel: Cardholder dispute intake (recorded call transcript)",
        f"Agent: Thank you for calling. I can help you dispute a transaction. "
        f"Can you confirm the card ending {s['pan'][-4:]}?",
        f"Cardholder: Yes, the {s['network'].title()} card {s['pan']}.",
        f"Cardholder: {rng.choice(REASON_NARRATIVES[s['reason']]).format(merchant=s['merchant'])}",
        f"Cardholder: The amount was {s['amount']:.2f} {s['currency']}, "
        f"posted on {s['transaction_date']}.",
    ]

    middle: list = []
    for indicator in s["fraud_indicators"]:
        middle.append(f"Cardholder: {rng.choice(FRAUD_INDICATOR_LINES[indicator])}")
    if s["account_tenure_months"] is not None:
        middle.append("Cardholder: " + rng.choice(TENURE_LINES).format(
            months=s["account_tenure_months"]))
    if s["prior_disputes_12m"]:
        middle.append("Cardholder: " + rng.choice(PRIOR_DISPUTE_LINES).format(
            n=s["prior_disputes_12m"]))
    if not s["within_reg_e_window"]:
        middle.append("Agent: I should note this was reported more than 60 days after "
                       "the statement date.")
    else:
        middle.append("Agent: This falls within the 60-day reporting window.")

    # Decoys: a competing amount and a reference number that is not the case id.
    if rng.random() < 0.45:
        middle.append("Cardholder: " + rng.choice(DECOY_LINES).format(
            decoy=s["decoy"], currency=s["currency"]))
    if rng.random() < 0.40:
        middle.append("Agent: " + rng.choice(CASE_REF_LINES).format(ref=s["case_ref"]))

    rng.shuffle(middle)
    return "\n".join(lines + middle)


def make_sample(rng: random.Random) -> Sample:
    s = build_signals(rng)

    signals = DisputeSignals(
        amount=s["amount"], reason_code=s["reason"], channel=s["channel"],
        card_in_possession=s["card_in_possession"],
        account_tenure_months=s["account_tenure_months"],
        prior_disputes_12m=s["prior_disputes_12m"],
        fraud_indicators=tuple(s["fraud_indicators"]),
        within_reg_e_window=s["within_reg_e_window"],
    )
    risk_band = derive_risk_band(signals)
    manual_review = derive_manual_review(risk_band, s["within_reg_e_window"])

    if risk_band == "critical":
        outcome = "chargeback_filed" if s["within_reg_e_window"] else "pending_documentation"
    elif not s["within_reg_e_window"]:
        outcome = "denied_insufficient_evidence"
    elif risk_band == "elevated":
        outcome = rng.choice(["provisional_credit_issued", "chargeback_filed",
                               "pending_documentation"])
    else:
        outcome = rng.choice(["provisional_credit_issued", "representment_expected",
                               "pending_documentation"])

    provisional = round(s["amount"], 2) if outcome == "provisional_credit_issued" else None
    documents = sorted(rng.sample(DOCUMENT_OPTIONS, k=rng.randint(0, 3)))

    sla_due = None
    if manual_review or outcome == "pending_documentation":
        sla_due = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    transcript = render_intake(s, rng)
    if provisional is not None:
        transcript += (f"\nAgent: I've issued a provisional credit of "
                        f"{provisional:.2f} {s['currency']} while we investigate.")
    if documents:
        transcript += f"\nAgent: Please send us: {', '.join(documents)}."
    if sla_due:
        transcript += f"\nAgent: Our decision is due by {sla_due}."
    transcript += f"\nAgent: This case is logged as {outcome.replace('_', ' ')}."

    record = {
        "case_summary": (f"Cardholder disputes a {s['amount']:.2f} {s['currency']} "
                          f"{s['reason'].replace('_', ' ')} at {s['merchant']}."),
        "cardholder": {
            "card_last4": s["pan"][-4:],      # masked by contract, never the full PAN
            "card_bin": s["pan"][:6],
            "network": s["network"],
            "account_tenure_months": s["account_tenure_months"],
            "prior_disputes_12m": s["prior_disputes_12m"],
        },
        "transaction": {
            "merchant_name": s["merchant"],
            "merchant_category": s["merchant_category"],
            "amount": s["amount"],
            "currency": s["currency"],
            "transaction_date": s["transaction_date"],
            "channel": s["channel"],
            "is_recurring": s["channel"] == "recurring",
        },
        "assessment": {
            "reason_code": s["reason"],
            "risk_band": risk_band,
            "within_reg_e_window": s["within_reg_e_window"],
            "requires_manual_review": manual_review,
            "fraud_indicators": s["fraud_indicators"],
        },
        "resolution": {
            "outcome": outcome,
            "provisional_credit_amount": provisional,
            "sla_due_date": sla_due,
            "documents_requested": documents,
        },
    }

    return Sample(source_text=transcript, record=record,
                   meta={k: v for k, v in s.items() if k not in ("pan", "decoy")})


def generate(n: int, rng: random.Random) -> list:
    out, seen, attempts = [], set(), 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        sample = make_sample(rng)
        key = hashlib.sha256(sample.source_text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    if len(out) < n:
        raise RuntimeError(f"could only generate {len(out)}/{n} unique samples")
    return out
