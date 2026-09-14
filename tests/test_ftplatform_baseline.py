"""
Milestone 1: Phase 2 baseline capture.

`run_baseline()` should look, to any caller, like `ftspec evaluate` scoped to
one customer -- same return shape, same manifest pattern -- plus one new
artefact: memory/deployed.json, the number a later fine-tuned candidate has
to beat. Tested by monkeypatching the underlying ftspec stage functions
(build.run / validate.run / benchmark.run), the same technique
tests/test_evaluate_combine.py uses for benchmark.py itself, so no GPU or
corpus generation is needed to verify the orchestration and isolation.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import runner
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect


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


def _fake_benchmark_result(n_eval=150):
    return {
        "profile": "saas_support",
        "regime": "none",
        "n_eval": n_eval,
        "report_path": "unused",
        "refused_for_compliance": [],
        "failed_systems": {},
        "egress_acknowledged": False,
        "systems": {
            "base-rubric": {"schema_adherence_pct": 100.0, "record_exact_pct": 66.7,
                             "headline_field": "issue.priority", "headline_accuracy_pct": 66.7,
                             "p50_ms": 12.0, "p99_ms": 14.0, "mean_prompt_tokens": 43.0,
                             "cost_per_100k_usd": 0.12},
        },
    }


def _stub_pipeline(monkeypatch, benchmark_result, validate_result=(True, "ok", {"passed": True})):
    monkeypatch.setattr(runner.build, "run", lambda *a, **k: {"train": 200, "val": 40, "eval": 150})
    monkeypatch.setattr(runner.validate_mod, "run", lambda *a, **k: validate_result)
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: benchmark_result)


def test_run_baseline_writes_deployed_json_under_the_customer_tree(monkeypatch, customer_ctx):
    result = _fake_benchmark_result()
    _stub_pipeline(monkeypatch, result)

    returned = runner.run_baseline(customer_ctx)

    assert returned == result
    deployed = customer_ctx.memory_dir() / "deployed.json"
    assert deployed.exists()
    snapshot = json.loads(deployed.read_text(encoding="utf-8"))
    assert snapshot["kind"] == "baseline"
    assert snapshot["systems"] == result["systems"]
    assert snapshot["n_eval"] == 150
    assert "acme" in str(deployed)  # lives under this customer's own subtree


def test_run_baseline_writes_manifests_per_stage(monkeypatch, customer_ctx):
    _stub_pipeline(monkeypatch, _fake_benchmark_result())
    runner.run_baseline(customer_ctx)

    manifests = customer_ctx.manifests_dir(runner.BASELINE_CANDIDATE_ID)
    for stage in ("prepare", "validate", "evaluate"):
        assert (manifests / f"{stage}.json").exists()


def test_failed_validation_does_not_block_baseline_measurement(monkeypatch, customer_ctx):
    """Baseline systems are prompted, not trained -- no GPU-hour is at risk,
    so a corpus that fails validation still gets measured (with a warning),
    unlike `ftspec train`, which refuses outright on a failed gate."""
    _stub_pipeline(monkeypatch, _fake_benchmark_result(),
                    validate_result=(False, "leakage found", {"passed": False}))

    result = runner.run_baseline(customer_ctx)
    assert result["systems"]  # still ran and returned data despite the failed gate


def test_two_customers_baselines_do_not_collide(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    a = CustomerContext(conn, "acme")
    b = CustomerContext(conn, "globex")

    _stub_pipeline(monkeypatch, _fake_benchmark_result(n_eval=10))
    runner.run_baseline(a)
    _stub_pipeline(monkeypatch, _fake_benchmark_result(n_eval=99))
    runner.run_baseline(b)

    snap_a = json.loads((a.memory_dir() / "deployed.json").read_text(encoding="utf-8"))
    snap_b = json.loads((b.memory_dir() / "deployed.json").read_text(encoding="utf-8"))
    assert snap_a["n_eval"] == 10
    assert snap_b["n_eval"] == 99
