"""
Greek saas_support transcripts must carry their subcategory in the text.

They used to share one template per category, so the subcategory label of a
Greek ticket was unrecoverable from its transcript: a real 10k-example run
scored 125/125 on English eval tickets and 6/25 on Greek ones.
"""
from __future__ import annotations

import random

from profiles.saas_support import generate
from profiles.saas_support import scenarios as sc


def test_every_subcategory_has_greek_templates():
    assert set(sc.EL_ISSUE_TEMPLATES) == set(sc.ISSUE_TEMPLATES)
    assert all(len(v) >= 1 for v in sc.EL_ISSUE_TEMPLATES.values())


def test_no_two_subcategories_share_a_greek_customer_line():
    owner = {}
    for sub, variants in sc.EL_ISSUE_TEMPLATES.items():
        for line, _probe in variants:
            assert line not in owner, f"{sub} and {owner.get(line)} share a Greek line"
            owner[line] = sub


def test_greek_entity_slots_match_the_english_template_family():
    # A Greek variant should ask for an order id / amount exactly when the
    # English templates for that subcategory can -- otherwise extracted_entities
    # gold would depend on the language rather than on the ticket.
    for sub in sc.ISSUE_TEMPLATES:
        en_order = any("{order_id}" in t for t, _ in sc.ISSUE_TEMPLATES[sub])
        el_order = any("{order_id}" in t for t, _ in sc.EL_ISSUE_TEMPLATES[sub])
        en_amount = any("{amount}" in t for t, _ in sc.ISSUE_TEMPLATES[sub])
        el_amount = any("{amount}" in t for t, _ in sc.EL_ISSUE_TEMPLATES[sub])
        assert (en_order, en_amount) == (el_order, el_amount), sub


def test_a_generated_greek_transcript_contains_its_subcategory_line():
    rng = random.Random(7)
    seen = 0
    for _ in range(400):
        category = rng.choice(sorted(sc.CATEGORY_SUBCATS))
        s = generate.build_signals(category, rng)
        if s["language"] != "el":
            continue
        seen += 1
        line = sc.EL_ISSUE_TEMPLATES[s["subcategory"]][s["el_template_idx"]][0]
        text = generate.render_el(s, "resolved", [], rng)
        assert generate.fill(line, s, "unused") in text
        assert "that order" not in text and "my recent order" not in text
    assert seen > 20
