"""
Core engine tests: contract, metric derivation, registry, compliance scanners.

These cover the invariants that would make every downstream number wrong while
everything still appeared to run. No GPU, no API key, no network.
"""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict

from ftspec.core import redaction
from ftspec.core.contract import Contract
from ftspec.core.fields import derive_scoring_plan
from ftspec.core.registry import available, load_profile

RAW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["vendor", "total", "tags"],
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "due": {"type": ["string", "null"]},
    },
}


class Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    score: float


class Doc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    nested: Nested
    items: list[str]


# --- Contract: Pydantic backend ----------------------------------------------

def test_pydantic_contract_accepts_valid():
    c = Contract.from_model(Doc)
    record, err = c.validate(json.dumps(
        {"title": "x", "nested": {"label": "a", "score": 1.5}, "items": ["p"]}))
    assert record is not None, err
    assert record["nested"]["score"] == 1.5


def test_pydantic_contract_rejects_extra_key():
    c = Contract.from_model(Doc)
    record, err = c.validate(json.dumps(
        {"title": "x", "nested": {"label": "a", "score": 1.0}, "items": [], "surprise": 1}))
    assert record is None and err


def test_pydantic_contract_rejects_stringified_number():
    """Strict on purpose: a consumer expecting a number breaks on a string."""
    c = Contract.from_model(Doc)
    record, _ = c.validate(json.dumps(
        {"title": "x", "nested": {"label": "a", "score": "1.0"}, "items": []}))
    assert record is None


def test_contract_tolerates_code_fence_only():
    c = Contract.from_model(Doc)
    body = json.dumps({"title": "x", "nested": {"label": "a", "score": 1.0}, "items": []})
    assert c.validate(f"```json\n{body}\n```")[0] is not None
    assert c.validate(body + "\n\nHope that helps!")[0] is None


# --- Contract: raw JSON Schema backend ---------------------------------------

def test_raw_schema_contract_round_trips():
    c = Contract.from_json_schema(RAW_SCHEMA)
    record, err = c.validate(json.dumps(
        {"vendor": "Acme", "total": 12.5, "tags": ["a"], "due": None}))
    assert record is not None, err


def test_raw_schema_contract_rejects_wrong_type():
    c = Contract.from_json_schema(RAW_SCHEMA)
    record, _ = c.validate(json.dumps({"vendor": "Acme", "total": "12.5", "tags": []}))
    assert record is None


def test_raw_schema_gets_additional_properties_closed():
    """Without this an invented key would pass, weakening the advertised guarantee."""
    schema = {k: v for k, v in RAW_SCHEMA.items() if k != "additionalProperties"}
    c = Contract.from_json_schema(schema)
    assert c.json_schema()["additionalProperties"] is False
    assert c.validate(json.dumps(
        {"vendor": "A", "total": 1.0, "tags": [], "invented": 2}))[0] is None


def test_non_object_schema_is_rejected():
    with pytest.raises(ValueError):
        Contract.from_json_schema({"type": "array"})


# --- Metric derivation -------------------------------------------------------

def test_scoring_plan_infers_kinds_from_types():
    plan = derive_scoring_plan(Contract.from_json_schema(RAW_SCHEMA).json_schema())
    kinds = {f.path: f.kind for f in plan.fields}
    assert kinds["vendor"] == "exact"
    assert kinds["total"] == "numeric"
    assert kinds["tags"] == "set_f1"
    assert kinds["due"] == "exact"      # Optional[str] unwraps to string


def test_scoring_plan_recurses_into_nested_objects():
    plan = derive_scoring_plan(Contract.from_model(Doc).json_schema())
    paths = plan.paths
    assert "nested.label" in paths and "nested.score" in paths
    assert "nested" not in paths, "should score leaves, not whole subtrees"


def test_free_text_is_excluded_from_record_match():
    plan = derive_scoring_plan(Contract.from_model(Doc).json_schema(),
                                free_text=("title",))
    assert plan.kind_of("title") == "token_f1"
    assert "title" not in plan.record_match_paths
    assert "nested.label" in plan.record_match_paths


def test_headline_and_rubric_inputs_are_marked_and_ordered():
    plan = derive_scoring_plan(Contract.from_model(Doc).json_schema(),
                                headline="nested.label",
                                rubric_inputs=("nested.score",))
    assert plan.fields[0].path == "nested.label"
    assert plan.headline().path == "nested.label"
    assert plan.rubric_input_paths == ["nested.score"]


# --- Registry ----------------------------------------------------------------

def test_all_builtin_profiles_load():
    assert set(available()) >= {"saas_support", "fintech_disputes", "healthcare_clinical"}
    for name in available():
        profile = load_profile(name)
        assert profile.name == name
        assert profile.contract.json_schema()["type"] == "object"
        assert profile.prompts.short


def test_profile_aliases_resolve():
    assert load_profile("fintech").name == "fintech_disputes"
    assert load_profile("healthcare").name == "healthcare_clinical"
    assert load_profile("saas").name == "saas_support"


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        load_profile("does_not_exist")


def test_custom_profile_requires_a_schema():
    with pytest.raises(ValueError):
        load_profile("custom")


def test_schema_flag_is_rejected_for_builtin_profiles(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(RAW_SCHEMA), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile("saas_support", schema_path=path)


def test_custom_profile_builds_from_file_and_refuses_generation(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(RAW_SCHEMA), encoding="utf-8")
    profile = load_profile("custom", schema_path=path)
    assert not profile.supports_generation()
    assert len(profile.scoring_plan().fields) == 4
    with pytest.raises(NotImplementedError):
        profile.generate(1, None)


# --- Compliance scanners -----------------------------------------------------

def test_luhn_valid_pan_is_flagged():
    text = "The card number is 4539578763621486."
    assert redaction.scan(text, ("pan",))


def test_reserved_test_pan_is_not_flagged():
    """Synthetic corpora use reserved BINs; flagging them would be pure noise."""
    assert not redaction.scan("Card 4111111111111111 on file.", ("pan",))


def test_long_non_luhn_number_is_not_flagged():
    """An order reference must not be mistaken for a card number."""
    assert not redaction.scan("Reference 1234567890123456 attached.", ("pan",))


@pytest.mark.parametrize("rule,text", [
    ("ssn", "SSN 123-45-6789 on record."),
    ("email", "Contact jane.doe@example.com please."),
    ("phone", "Call (555) 123-4567 today."),
    ("nhs_mrn", "MRN: 4471209 in the chart."),
    ("dob", "DOB 1984-03-11 confirmed."),
])
def test_identifier_rules_fire(rule, text):
    assert redaction.scan(text, (rule,))


def test_scan_record_walks_nested_structures():
    record = {"a": {"b": ["contact me at x@y.com"]}}
    findings = redaction.scan_record(record, ("email",))
    assert findings and "a.b[0]" in findings[0].context


def test_assert_masked_catches_an_unmasked_identifier():
    assert redaction.assert_masked("4111111111111111", 4, "card_last4")
    assert not redaction.assert_masked("1486", 4, "card_last4")
    assert not redaction.assert_masked(None, 4, "card_last4")
