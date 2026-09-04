"""
Synthetic corpus for the SaaS-support vertical.

Generation is signal-first, not template-first:

    1. sample the underlying facts (tier, repeat contact, churn threat, entities)
    2. derive `priority` by executing the triage rubric on those facts
    3. render a noisy, realistic transcript that *encodes* those facts indirectly

That ordering is what makes the task non-trivial. The label is a deterministic
function of facts scattered across the conversation in varied surface forms and
mixed with decoys, so a model must extract and combine signals rather than
pattern-match a template.

Difficulty features built in on purpose:
  - decoy identifiers (TKT-/INV-/CASE-, phone numbers) that must not become `order_id`
  - decoy amounts (plan price) alongside the real disputed amount
  - incidental prices in non-billing tickets, where `amount` must stay null
  - rubric signals placed anywhere in the conversation, never in a fixed slot
  - character-level typos on customer turns, never on entity tokens
  - filler turns, so conversation length carries no information
  - ~15% Greek-language conversations
"""
from __future__ import annotations

import hashlib
import random

from ftspec.core.profile import Sample
from profiles.saas_support import scenarios as sc
from profiles.saas_support.rubric import TriageSignals, derive_priority

# Typos are applied to customer prose only, and never to tokens that carry
# ground-truth values -- corrupting "ORD-12345" would make the label
# unrecoverable and the benchmark meaningless.
TYPO_RATE = 0.035


def is_protected_token(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok) or "$" in tok or tok.isupper()


def inject_typos(text: str, rng: random.Random, rate: float = TYPO_RATE) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) > 3 and not is_protected_token(w) and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            mode = rng.choice(("swap", "drop", "double"))
            if mode == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif mode == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def typo_customer_lines(lines: list, rng: random.Random, protected: set) -> list:
    """Add typos to customer prose, but never to lines carrying a rubric signal.

    A typo in "this is the third time" would corrupt the evidence for
    `is_repeat_contact`, making the label unrecoverable -- the same label-noise
    failure this generator exists to avoid.
    """
    out = []
    for line in lines:
        if line.startswith("Customer:") and line not in protected and rng.random() < 0.45:
            prefix, _, body = line.partition(": ")
            line = f"{prefix}: {inject_typos(body, rng)}"
        out.append(line)
    return out


def choose_weighted(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def rand_order_id(rng: random.Random) -> str:
    return f"ORD-{rng.randint(10000, 99999)}"


def rand_ticket_ref(rng: random.Random) -> str:
    prefix = rng.choice(("TKT", "INV", "CASE"))
    return f"{prefix}-{rng.randint(1000, 9999)}"


def rand_amount(rng: random.Random) -> float:
    """Straddles the rubric's 300 threshold so that boundary genuinely matters."""
    return round(rng.uniform(25.0, 620.0), 2)


def rand_plan_amount(rng: random.Random) -> float:
    return round(rng.uniform(9.0, 99.0), 2)


def rand_date(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def derive_status(sentiment: str, priority: str, rng: random.Random) -> str:
    if priority == "urgent":
        return choose_weighted(rng, {"escalated": 0.55, "pending": 0.28, "resolved": 0.12, "unresolved": 0.05})
    if priority == "high":
        return choose_weighted(rng, {"escalated": 0.32, "pending": 0.30, "resolved": 0.33, "unresolved": 0.05})
    if sentiment in ("positive", "neutral"):
        return choose_weighted(rng, {"resolved": 0.70, "pending": 0.20, "escalated": 0.05, "unresolved": 0.05})
    return choose_weighted(rng, {"resolved": 0.45, "pending": 0.30, "escalated": 0.20, "unresolved": 0.05})


def build_signals(category: str, rng: random.Random) -> dict:
    """Sample the underlying facts of a ticket, before any text exists."""
    subcategory = rng.choice(sc.CATEGORY_SUBCATS[category])
    sentiment = choose_weighted(rng, sc.SENTIMENT_WEIGHTS[category])

    # Pick the phrasing up front: whether this ticket *has* an order id at all
    # depends on whether the chosen phrasing actually mentions one. Assigning an
    # entity the transcript never states would be an unrecoverable label.
    template_idx = rng.randrange(len(sc.ISSUE_TEMPLATES[subcategory]))
    template_text = sc.ISSUE_TEMPLATES[subcategory][template_idx][0]
    template_has_order = "{order_id}" in template_text
    template_has_amount = "{amount}" in template_text

    tier = choose_weighted(rng, {"unknown": 0.30, "free": 0.20, "pro": 0.28, "enterprise": 0.22})
    is_repeat_contact = rng.random() < 0.28
    # Churn threats cluster with anger, but are not implied by it.
    churn_p = {"frustrated": 0.45, "negative": 0.18, "neutral": 0.04, "positive": 0.0}[sentiment]
    churn_threat = rng.random() < churn_p
    data_loss_risk = (subcategory in ("sync_error", "app_crash", "integration_bug")) and rng.random() < 0.30

    # Greek conversations use a per-category template, so entity availability
    # follows the category rather than the English phrasing.
    language = "el" if rng.random() < 0.15 else "en"
    if language == "el":
        el_text = sc.EL_ISSUE_BY_CATEGORY[category][0]
        template_has_order = "{order_id}" in el_text
        template_has_amount = "{amount}" in el_text

    order_id = rand_order_id(rng) if (template_has_order and rng.random() < 0.85) else None
    amount = rand_amount(rng) if template_has_amount else None
    plan_amount = rand_plan_amount(rng)
    product = rng.choice(sc.PRODUCTS)

    return {
        "category": category,
        "subcategory": subcategory,
        "template_idx": template_idx,
        "sentiment": sentiment,
        "tier": tier,
        "is_repeat_contact": is_repeat_contact,
        "churn_threat": churn_threat,
        "data_loss_risk": data_loss_risk,
        "order_id": order_id,
        "amount": amount,
        "plan_amount": plan_amount,
        "product": product,
        "language": language,
    }


def fill(template: str, s: dict, ticket_ref: str) -> str:
    """Fill template slots, degrading gracefully when an entity is absent."""
    text = template
    if s["order_id"] is None:
        text = text.replace("order {order_id}", "my recent order").replace("{order_id}", "that order")
    return text.format(
        product=s["product"],
        order_id=s["order_id"] or "",
        amount=f"{s['amount']:.2f}" if s["amount"] is not None else "",
        plan_amount=f"{s['plan_amount']:.2f}",
        ticket_ref=ticket_ref,
        agent="",
    )


def render_en(s: dict, status: str, action_lines: list, rng: random.Random) -> str:
    agent = rng.choice(sc.AGENT_NAMES)
    ticket_ref = rand_ticket_ref(rng)

    customer_line, agent_probe = sc.ISSUE_TEMPLATES[s["subcategory"]][s["template_idx"]]

    head = [
        "Agent: " + rng.choice(sc.GREETINGS).format(product=s["product"], agent=agent),
        f"Customer: {rng.choice(sc.OPENERS[s['sentiment']])} {fill(customer_line, s, ticket_ref)}",
        agent_probe,
    ]

    # Rubric signals, each rendered in a varied surface form. These lines are
    # protected from typo injection so their evidence stays intact.
    middle, protected = [], set()
    for present, pool in (
        (s["tier"] != "unknown", sc.TIER_MENTIONS.get(s["tier"], [])),
        (s["is_repeat_contact"], sc.REPEAT_CONTACT_LINES),
        (s["data_loss_risk"], sc.DATA_LOSS_LINES),
        (s["churn_threat"], sc.CHURN_LINES),
    ):
        if present and pool:
            line = rng.choice(pool)
            middle.append(line)
            protected.add(line)

    # Decoys: identifiers and prices that must not reach extracted_entities.
    if rng.random() < 0.45:
        middle.append(fill(rng.choice(sc.DISTRACTOR_TURNS), s, ticket_ref))
    if s["amount"] is None and rng.random() < 0.30:
        middle.append(f"Customer: We pay ${s['plan_amount']:.2f} a month for this, by the way.")

    # Filler carries no signal, so conversation length stays uninformative.
    middle.extend(rng.sample(sc.FILLER_TURNS, k=rng.randint(0, 3)))

    rng.shuffle(middle)

    # Actions are spoken by the agent, so `actions_taken` is genuinely
    # recoverable from the transcript rather than being unpredictable noise.
    tail = action_lines + [
        rng.choice(sc.RESOLUTION_LINES[status]),
        f"Customer: {rng.choice(sc.CLOSERS[s['sentiment']])}",
    ]

    return "\n".join(typo_customer_lines(head + middle + tail, rng, protected))


def render_el(s: dict, status: str, action_lines: list, rng: random.Random) -> str:
    agent = rng.choice(sc.AGENT_NAMES)
    ticket_ref = rand_ticket_ref(rng)
    customer_line, agent_probe = sc.EL_ISSUE_BY_CATEGORY[s["category"]]

    head = [
        rng.choice(sc.EL_GREETINGS).format(product=s["product"], agent=agent),
        rng.choice(sc.EL_OPENERS[s["sentiment"]]),
        fill(customer_line, s, ticket_ref),
        agent_probe,
    ]

    middle = []
    if s["tier"] != "unknown":
        middle.append(rng.choice(sc.EL_TIER_MENTIONS[s["tier"]]))
    if s["is_repeat_contact"]:
        middle.append(rng.choice(sc.EL_REPEAT_LINES))
    if s["data_loss_risk"]:
        middle.append(rng.choice(sc.EL_DATA_LOSS_LINES))
    if s["churn_threat"]:
        middle.append(rng.choice(sc.EL_CHURN_LINES))
    if rng.random() < 0.35:
        middle.append(f"Agent: Ο αριθμός αιτήματός σας είναι {ticket_ref}.")
    rng.shuffle(middle)

    tail = action_lines + [
        rng.choice(sc.EL_RESOLUTION_LINES[status]),
        rng.choice(sc.EL_CLOSERS[s["sentiment"]]),
    ]
    return "\n".join(head + middle + tail)


def make_sample(category: str, rng: random.Random) -> dict:
    s = build_signals(category, rng)

    priority = derive_priority(TriageSignals(
        category=s["category"], subcategory=s["subcategory"], sentiment=s["sentiment"],
        tier=s["tier"], is_repeat_contact=s["is_repeat_contact"],
        churn_threat=s["churn_threat"], data_loss_risk=s["data_loss_risk"], amount=s["amount"],
    ))
    status = derive_status(s["sentiment"], priority, rng)

    # Pick actions together with the utterance that states them, so the label
    # and its evidence are generated as one unit and cannot drift apart.
    n_actions = {"resolved": (1, 3), "pending": (0, 2), "escalated": (0, 2), "unresolved": (0, 1)}[status]
    is_greek = s["language"] == "el"
    pool = [(lbl, el) for lbl, _, el in sc.GENERIC_ACTIONS_EL] if is_greek \
        else sc.ACTION_UTTERANCES[s["category"]]
    chosen = rng.sample(pool, k=min(rng.randint(*n_actions), len(pool)))
    actions = [label for label, _ in chosen]
    closing_lines = [utterance for _, utterance in chosen]

    requires_followup = status in ("pending", "escalated", "unresolved") or \
        (status == "resolved" and rng.random() < 0.15)

    # The follow-up date must be spoken too, for the same reason actions are:
    # a date the transcript never mentions is unpredictable by construction.
    followup_date = rand_date(rng) if requires_followup else None
    if followup_date:
        closing_lines.append(
            f"Agent: Θα επικοινωνήσω ξανά μαζί σας στις {followup_date}." if is_greek
            else f"Agent: I'll follow up with you on {followup_date}."
        )

    transcript = render_el(s, status, closing_lines, rng) if is_greek \
        else render_en(s, status, closing_lines, rng)

    summary_tail = f" for order {s['order_id']}" if s["order_id"] else ""
    ground_truth = {
        "ticket_summary": (f"Customer contacted support regarding "
                            f"{sc.SUBCATEGORY_SUMMARIES[s['subcategory']]}{summary_tail}."),
        "customer": {
            "sentiment": s["sentiment"],
            "language": s["language"],
            "tier": s["tier"],
            "churn_threat": s["churn_threat"],
        },
        "issue": {
            "category": s["category"],
            "subcategory": s["subcategory"],
            "priority": priority,
            "is_repeat_contact": s["is_repeat_contact"],
        },
        "actions_taken": actions,
        "resolution": {
            "status": status,
            "requires_followup": requires_followup,
            "followup_date": followup_date,
        },
        "extracted_entities": {
            "order_id": s["order_id"],
            "product_name": s["product"],
            "amount": s["amount"],
        },
    }

    return Sample(
        source_text=transcript,
        record=ground_truth,
        meta={k: v for k, v in s.items() if k != "plan_amount"},
    )


def generate(n: int, rng: random.Random) -> list:
    """Produce `n` unique samples, round-robin over issue categories."""
    categories = list(sc.CATEGORY_SUBCATS.keys())
    out, seen, attempts, i = [], set(), 0, 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        sample = make_sample(categories[i % len(categories)], rng)
        key = hashlib.sha256(sample.source_text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
        i += 1
    if len(out) < n:
        raise RuntimeError(f"could only generate {len(out)}/{n} unique samples")
    return out
