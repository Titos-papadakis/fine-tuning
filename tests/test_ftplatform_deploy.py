"""
Milestone 3, deployment wiring: deploy-if-better and its manual rollback
escape hatch.

Unlike Stage A/B/C's tests, `evaluate_gates()`'s McNemar check is exercised
with *real* `ftspec.evaluation.metrics.mcnemar_exact` arithmetic over
hand-built paired scores -- it's pure computation, cheap to run for real,
and the whole point of this gate is that it must actually reject a
statistically real regression, not just call a mocked-out function.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.deployment import deploy, registry, rollback


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture
def customer_ctx(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    return CustomerContext(conn, "acme")


def _metrics(adherence=100.0, headline=90.0, record_exact=90.0, cost=1.0, p99=100.0):
    return {"schema_adherence_pct": adherence, "headline_accuracy_pct": headline,
            "record_exact_pct": record_exact, "cost_per_100k_usd": cost, "p99_ms": p99,
            "headline_field": "issue.priority", "p50_ms": p99 / 2, "mean_prompt_tokens": 40.0}


def _write_candidate(ctx, candidate_id, system, metrics, raw_scores=None, params=None):
    """Puts a candidate's evaluate manifest, raw per-record scores, and a
    (dummy) model directory on disk -- exactly what a real Stage A/B/C run
    would have left behind, so deploy.py's read-back path is exercised for
    real rather than mocked away."""
    manifests = ctx.manifests_dir(candidate_id)
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params or {}, "metrics": {"systems": {system: metrics}}}
    (manifests / "evaluate.json").write_text(json.dumps(manifest), encoding="utf-8")

    reports = ctx.reports_dir(candidate_id)
    reports.mkdir(parents=True, exist_ok=True)
    if raw_scores is not None:
        with open(reports / f"raw_{system}.jsonl", "w", encoding="utf-8") as f:
            for s in raw_scores:
                f.write(json.dumps({"output": "{}", "scores": {"_record_exact": s},
                                     "latency_ms": 10.0}) + "\n")

    candidate_dir = ctx.candidate_dir(candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "lora_adapter").mkdir(exist_ok=True)
    (candidate_dir / "lora_adapter" / "adapter_model.bin").write_text("fake", encoding="utf-8")


def _write_baseline(ctx, candidate_id="baseline", system="base-rubric", **metric_kwargs):
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    snapshot = {"kind": "baseline", "candidate_id": candidate_id,
                "systems": {system: _metrics(**metric_kwargs)}}
    (ctx.memory_dir() / "deployed.json").write_text(json.dumps(snapshot), encoding="utf-8")


# --- gates -------------------------------------------------------------------

def test_missing_baseline_is_a_clean_error(customer_ctx):
    _write_candidate(customer_ctx, "stageB-r16-a16", "finetuned", _metrics())
    with pytest.raises(FileNotFoundError, match="baseline run"):
        deploy.evaluate_gates(customer_ctx, "stageB-r16-a16")


def test_adherence_gate_rejects_below_the_floor(customer_ctx):
    _write_baseline(customer_ctx)
    _write_candidate(customer_ctx, "c1", "finetuned", _metrics(adherence=90.0))

    gates = deploy.evaluate_gates(customer_ctx, "c1", min_adherence_pct=98.0)

    assert gates["passed"] is False
    assert "schema_adherence_pct" in gates["reason"]


def test_mcnemar_gate_rejects_a_real_significant_regression(customer_ctx):
    _write_baseline(customer_ctx, record_exact=90.0)
    _write_candidate(customer_ctx, "baseline", "base-rubric",
                      _metrics(record_exact=90.0),
                      raw_scores=[1.0] * 15 + [0.0] * 5)  # current: 15/20 correct
    # new: correct on none of the 15 the current got right, and 0 more --
    # 15 discordant pairs all favoring current, 0 favoring new.
    _write_candidate(customer_ctx, "c1", "finetuned",
                      _metrics(record_exact=25.0),
                      raw_scores=[0.0] * 15 + [0.0] * 5)

    gates = deploy.evaluate_gates(customer_ctx, "c1")

    assert gates["passed"] is False
    assert "McNemar" in gates["reason"]


def test_mcnemar_gate_does_not_reject_when_new_has_no_fewer_wins(customer_ctx):
    _write_baseline(customer_ctx, record_exact=75.0)
    _write_candidate(customer_ctx, "baseline", "base-rubric",
                      _metrics(record_exact=75.0),
                      raw_scores=[1.0] * 15 + [0.0] * 5)
    # new: correct everywhere, including where current was wrong -- 5
    # discordant pairs favoring new, 0 favoring current.
    _write_candidate(customer_ctx, "c1", "finetuned",
                      _metrics(record_exact=100.0),
                      raw_scores=[1.0] * 20)

    gates = deploy.evaluate_gates(customer_ctx, "c1")

    assert gates["passed"] is True


def test_composite_gate_rejects_a_much_worse_candidate_even_without_mcnemar_data(customer_ctx):
    _write_baseline(customer_ctx, headline=95.0, record_exact=95.0, cost=1.0, p99=50.0)
    # No raw scores written -> McNemar gate is skipped (nothing to pair), so
    # this isolates the composite-score gate specifically.
    _write_candidate(customer_ctx, "c1", "finetuned",
                      _metrics(headline=20.0, record_exact=20.0, cost=50.0, p99=5000.0))

    gates = deploy.evaluate_gates(customer_ctx, "c1", epsilon=0.02)

    assert gates["passed"] is False
    assert "composite score" in gates["reason"]


def test_a_clearly_better_candidate_passes_every_gate(customer_ctx):
    _write_baseline(customer_ctx, headline=60.0, record_exact=60.0, cost=5.0, p99=200.0)
    _write_candidate(customer_ctx, "c1", "finetuned",
                      _metrics(headline=99.0, record_exact=99.0, cost=1.0, p99=20.0))

    gates = deploy.evaluate_gates(customer_ctx, "c1")

    assert gates["passed"] is True
    assert gates["reason"] is None


def test_unknown_system_for_the_candidate_is_a_clean_error(customer_ctx):
    _write_baseline(customer_ctx)
    _write_candidate(customer_ctx, "c1", "finetuned", _metrics())
    with pytest.raises(ValueError, match="not found"):
        deploy.evaluate_gates(customer_ctx, "c1", new_system="does-not-exist")


# --- maybe_deploy / production/ -----------------------------------------------

def test_maybe_deploy_promotes_and_records_when_gates_pass(customer_ctx, conn):
    _write_baseline(customer_ctx, headline=60.0, record_exact=60.0)
    _write_candidate(customer_ctx, "c1", "finetuned",
                      _metrics(headline=99.0, record_exact=99.0),
                      params={"base_model": "org/model-x"})

    result = deploy.maybe_deploy(customer_ctx, "c1", conn=conn)

    assert result["deployed"] is True
    assert (customer_ctx.production_dir() / "lora_adapter" / "adapter_model.bin").exists()
    meta = json.loads((customer_ctx.production_dir() / "candidate_meta.json")
                       .read_text(encoding="utf-8"))
    assert meta["base_model"] == "org/model-x"
    deployed = json.loads((customer_ctx.memory_dir() / "deployed.json").read_text(encoding="utf-8"))
    assert deployed["candidate_id"] == "c1"
    history = registry.history(conn, "acme")
    assert len(history) == 1 and history[0]["candidate_id"] == "c1"


def test_maybe_deploy_leaves_production_untouched_when_gates_fail(customer_ctx, conn):
    _write_baseline(customer_ctx, headline=90.0, record_exact=90.0)
    _write_candidate(customer_ctx, "c1", "finetuned", _metrics(adherence=50.0))

    result = deploy.maybe_deploy(customer_ctx, "c1", conn=conn)

    assert result["deployed"] is False
    assert not customer_ctx.production_dir().exists()
    assert registry.history(conn, "acme") == []


def test_infer_base_model_resolves_through_a_stage_c_winner_reference(customer_ctx):
    _write_candidate(customer_ctx, "stageB-r16-a16", "finetuned", _metrics(),
                      params={"base_model": "org/winner-model"})
    _write_candidate(customer_ctx, "stageC-fp16-constrained", "finetuned", _metrics(),
                      params={"winner_candidate_id": "stageB-r16-a16"})

    assert deploy._infer_base_model(customer_ctx, "stageC-fp16-constrained") == "org/winner-model"


# --- rollback ------------------------------------------------------------------

def test_rollback_bypasses_every_gate(customer_ctx, conn):
    _write_baseline(customer_ctx, headline=99.0, record_exact=99.0)  # a strong incumbent
    # A candidate that would fail deploy-if-better outright:
    _write_candidate(customer_ctx, "bad", "finetuned", _metrics(adherence=10.0))

    result = rollback.rollback_to(customer_ctx, "bad", conn=conn)

    assert result["deployed"] is True
    assert (customer_ctx.production_dir() / "lora_adapter").exists()
    history = registry.history(conn, "acme")
    assert history[-1]["candidate_id"] == "bad"
    assert history[-1]["is_rollback"] == 1


def test_rollback_to_unknown_system_is_a_clean_error(customer_ctx):
    _write_candidate(customer_ctx, "c1", "finetuned", _metrics())
    with pytest.raises(ValueError, match="not found"):
        rollback.rollback_to(customer_ctx, "c1", system="does-not-exist")


# --- isolation -----------------------------------------------------------------

def test_two_customers_deployments_do_not_collide(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    a, b = CustomerContext(conn, "acme"), CustomerContext(conn, "globex")

    for ctx, cost in ((a, 1.0), (b, 2.0)):
        _write_baseline(ctx, headline=50.0, record_exact=50.0)
        _write_candidate(ctx, "c1", "finetuned",
                          _metrics(headline=99.0, record_exact=99.0, cost=cost))
        deploy.maybe_deploy(ctx, "c1", conn=conn)

    assert a.production_dir() != b.production_dir()
    dep_a = json.loads((a.memory_dir() / "deployed.json").read_text(encoding="utf-8"))
    dep_b = json.loads((b.memory_dir() / "deployed.json").read_text(encoding="utf-8"))
    assert dep_a["systems"]["finetuned"]["cost_per_100k_usd"] == 1.0
    assert dep_b["systems"]["finetuned"]["cost_per_100k_usd"] == 2.0
    assert len(registry.history(conn, "acme")) == 1
    assert len(registry.history(conn, "globex")) == 1


# --- registry ------------------------------------------------------------------

def test_registry_current_returns_the_latest_row(conn):
    registry.record(conn, "acme", "saas_support", "c1", {"a": 1})
    registry.record(conn, "acme", "saas_support", "c2", {"a": 2})
    assert registry.current(conn, "acme")["candidate_id"] == "c2"


def test_registry_current_is_none_with_no_history(conn):
    assert registry.current(conn, "acme") is None
