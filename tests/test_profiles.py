"""
Profile tests, parametrized across every shipped vertical.

The same contract is demanded of each: deterministic generation, a label
distribution no single class dominates, every label recoverable from its source
document, and no leakage. A new vertical that satisfies these tests is ready to
train; one that does not would produce a benchmark nobody should believe.

The per-domain rubric tests are separate, because a business policy is the one
thing here that cannot be checked generically.
"""
from __future__ import annotations

import json
import random

import pytest

from ftspec.config import Config
from ftspec.core.registry import load_profile
from ftspec.data import audit, build, validate
from ftspec.evaluation import metrics as M

PROFILE_NAMES = ["saas_support", "fintech_disputes", "healthcare_clinical"]


@pytest.fixture(scope="module", params=PROFILE_NAMES)
def built(request, tmp_path_factory):
    """Build a small corpus once per profile."""
    profile = load_profile(request.param)
    out = tmp_path_factory.mktemp(request.param)
    metrics = build.run(profile, out, n_train=40, n_val=10, n_eval=20)
    return profile, out, metrics


def cfg_for() -> Config:
    return Config()


# --- generation --------------------------------------------------------------

def test_generation_is_deterministic(built):
    profile, _, _ = built
    a = profile.generate(15, random.Random(7))
    b = profile.generate(15, random.Random(7))
    assert [s.source_text for s in a] == [s.source_text for s in b]
    assert [s.record for s in a] == [s.record for s in b]


def test_generated_records_satisfy_the_contract(built):
    profile, _, _ = built
    for sample in profile.generate(25, random.Random(3)):
        record, err = profile.contract.validate(
            json.dumps(sample.record, ensure_ascii=False))
        assert record is not None, f"{profile.name}: {err}"


def test_no_class_dominates_the_headline_field(built):
    """A degenerate distribution makes accuracy uninterpretable."""
    _, _, metrics = built
    baseline = metrics["train_distribution"]["majority_class_baseline"]
    assert baseline < 70.0, f"majority-class baseline {baseline}% is too high"


def test_headline_field_uses_more_than_one_value(built):
    _, _, metrics = built
    assert len(metrics["train_distribution"]["counts"]) >= 3


def test_documents_are_unique_across_all_splits(built):
    _, out, _ = built
    texts = []
    for name in ("train", "val", "eval"):
        for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["messages"][1]["content"])
    assert len(texts) == len(set(texts))


def test_eval_does_not_leak_into_training(built):
    _, out, _ = built

    def texts(name):
        return {json.loads(line)["messages"][1]["content"]
                for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}

    assert not (texts("train") | texts("val")) & texts("eval")


# --- audit and gates ---------------------------------------------------------

def test_generated_corpus_passes_its_own_audit(built):
    profile, out, _ = built
    passed, report, metrics = audit.run(profile, out)
    assert passed, report
    assert metrics["violations"] == 0


def test_corpus_passes_the_preflight_gate(built):
    profile, out, _ = built
    passed, report, _ = validate.run(cfg_for(), profile, out, approx=True)
    assert passed, report


def test_preflight_rejects_a_contract_violation(built, tmp_path):
    profile, out, _ = built
    for name in ("train", "val", "eval"):
        (tmp_path / f"{name}.jsonl").write_text(
            (out / f"{name}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    record = json.loads(row["messages"][2]["content"])
    record["__not_in_schema__"] = True
    row["messages"][2]["content"] = json.dumps(record)
    lines[0] = json.dumps(row)
    (tmp_path / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed, _, _ = validate.run(cfg_for(), profile, tmp_path, approx=True)
    assert not passed


def test_preflight_rejects_an_oversized_document(built, tmp_path):
    """The defect this gate exists for: silent truncation of the completion."""
    profile, out, _ = built
    for name in ("train", "val", "eval"):
        (tmp_path / f"{name}.jsonl").write_text(
            (out / f"{name}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["messages"][1]["content"] += " padding" * 5000
    lines[0] = json.dumps(row)
    (tmp_path / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed, report, _ = validate.run(cfg_for(), profile, tmp_path, approx=True)
    assert not passed
    assert "truncated" in report


def test_preflight_detects_leakage(built, tmp_path):
    profile, out, _ = built
    for name in ("train", "val", "eval"):
        (tmp_path / f"{name}.jsonl").write_text(
            (out / f"{name}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    eval_line = (out / "eval.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with open(tmp_path / "train.jsonl", "a", encoding="utf-8") as f:
        f.write(eval_line + "\n")

    passed, report, _ = validate.run(cfg_for(), profile, tmp_path, approx=True)
    assert not passed
    assert "LEAKAGE" in report


# --- metrics wiring ----------------------------------------------------------

def test_scoring_plan_covers_every_required_top_level_field(built):
    profile, _, _ = built
    plan = profile.scoring_plan()
    roots = {p.split(".")[0] for p in plan.paths}
    assert set(profile.contract.required_fields()) <= roots


def test_self_scoring_is_perfect_and_violation_is_zero(built):
    profile, out, _ = built
    plan = profile.scoring_plan()
    for line in (out / "eval.jsonl").read_text(encoding="utf-8").splitlines()[:10]:
        gold = json.loads(json.loads(line)["messages"][2]["content"])
        assert M.score_record(gold, gold, plan)[M.RECORD_EXACT] == 1.0
        assert M.score_record(None, gold, plan)[M.RECORD_EXACT] == 0.0


def test_prompt_variants_differ_in_length(built):
    """The prompt-tax argument depends on these actually differing."""
    profile, _, _ = built
    p = profile.prompts
    assert len(p.short) < len(p.schema) <= len(p.schema_rubric)


# --- compliance --------------------------------------------------------------

@pytest.mark.parametrize("name,regime,egress", [
    ("saas_support", "none", True),
    ("fintech_disputes", "PCI-DSS", False),
    ("healthcare_clinical", "HIPAA", False),
])
def test_compliance_envelopes_are_declared(name, regime, egress):
    profile = load_profile(name)
    assert profile.compliance.regime == regime
    assert profile.compliance.allows_external_api is egress
    if not egress:
        assert profile.compliance.residency_note
        assert "acknowledge-egress" in profile.compliance.egress_refusal()


def test_fintech_never_emits_a_full_pan():
    """The contract has no field capable of holding one; verify the data agrees."""
    profile = load_profile("fintech_disputes")
    for sample in profile.generate(30, random.Random(11)):
        holder = sample.record["cardholder"]
        assert len(holder["card_last4"]) == 4
        assert len(holder["card_bin"]) == 6
        assert not profile.audit_sample(sample)


def test_healthcare_respects_safe_harbor_age_cap():
    profile = load_profile("healthcare_clinical")
    for sample in profile.generate(40, random.Random(13)):
        age = sample.record["patient"]["age_years"]
        assert age is None or age <= 89
        assert not profile.audit_sample(sample)


# --- domain rubrics ----------------------------------------------------------

def test_saas_rubric_cases():
    from profiles.saas_support.rubric import TriageSignals, derive_priority

    def s(**kw):
        base = dict(category="billing", subcategory="duplicate_charge", sentiment="neutral",
                     tier="unknown", is_repeat_contact=False, churn_threat=False,
                     data_loss_risk=False, amount=None)
        base.update(kw)
        return TriageSignals(**base)

    assert derive_priority(s(churn_threat=True, tier="enterprise")) == "urgent"
    assert derive_priority(s(churn_threat=True, tier="free")) == "high"
    assert derive_priority(s(data_loss_risk=True)) == "urgent"
    assert derive_priority(s(amount=301.0)) == "high"
    assert derive_priority(s(amount=299.0)) == "medium"
    assert derive_priority(s(category="product", subcategory="feature_request",
                              sentiment="positive")) == "low"


def test_fintech_rubric_cases():
    from profiles.fintech_disputes.rubric import (
        DisputeSignals,
        derive_manual_review,
        derive_risk_band,
    )

    def s(**kw):
        base = dict(amount=50.0, reason_code="duplicate_charge", channel="ecommerce",
                     card_in_possession=False, account_tenure_months=24,
                     prior_disputes_12m=0, fraud_indicators=(), within_reg_e_window=True)
        base.update(kw)
        return DisputeSignals(**base)

    assert derive_risk_band(s(amount=5001)) == "critical"
    assert derive_risk_band(s(reason_code="unauthorized_transaction",
                               card_in_possession=True)) == "critical"
    assert derive_risk_band(s(prior_disputes_12m=5)) == "critical"
    assert derive_risk_band(s(amount=1500)) == "elevated"
    assert derive_risk_band(s(account_tenure_months=3)) == "elevated"
    assert derive_risk_band(s()) == "low"
    assert derive_risk_band(s(amount=500)) == "standard"
    assert derive_manual_review("low", within_reg_e_window=False) is True
    assert derive_manual_review("low", within_reg_e_window=True) is False


def test_healthcare_rubric_cases():
    from profiles.healthcare_clinical.rubric import (
        TriageSignals,
        VitalSigns,
        derive_acuity,
    )

    def s(vitals=None, **kw):
        base = dict(abnormal_count=0, encounter_class="ambulatory",
                     primary_severity="mild", primary_unconfirmed=False)
        base.update(kw)
        return TriageSignals(vitals=vitals or VitalSigns(), **base)

    assert derive_acuity(s(vitals=VitalSigns(spo2=88))) == "critical"
    assert derive_acuity(s(vitals=VitalSigns(systolic_bp=85))) == "critical"
    assert derive_acuity(s(vitals=VitalSigns(gcs=11))) == "critical"
    assert derive_acuity(s(abnormal_count=2)) == "emergent"
    assert derive_acuity(s(vitals=VitalSigns(heart_rate=130))) == "emergent"
    assert derive_acuity(s(abnormal_count=1)) == "urgent"
    assert derive_acuity(s(encounter_class="emergency")) == "urgent"
    assert derive_acuity(s(primary_unconfirmed=True)) == "urgent"
    assert derive_acuity(s()) == "routine"
