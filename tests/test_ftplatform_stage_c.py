"""
Milestone 3, Stage C: inference-time quantization/constrained-decoding
toggles for an already-trained Stage-B adapter. No training -- one
`benchmark.run()` call per combo, reusing its additive `load_in_4bit`/
`constrained` overrides (see tests/test_benchmark_overrides.py for those
directly).
"""
from __future__ import annotations

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import runner
from ftplatform.candidates.generator import (
    DEFAULT_STAGE_C_COMBOS,
    StageCCandidate,
    stage_c_candidates,
)
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
    ctx = CustomerContext(conn, "acme")
    ctx.data_dir().mkdir(parents=True, exist_ok=True)
    (ctx.data_dir() / "eval.jsonl").write_text("", encoding="utf-8")
    ctx.adapter_dir("stageB-r16-a16").mkdir(parents=True, exist_ok=True)  # a "trained" winner
    return ctx


def _fake_eval_result(cost=1.0):
    return {
        "profile": "saas_support", "regime": "none", "n_eval": 150, "report_path": "unused",
        "refused_for_compliance": [], "failed_systems": {}, "egress_acknowledged": False,
        "systems": {"finetuned": {"schema_adherence_pct": 100.0, "record_exact_pct": 91.0,
                                    "headline_field": "issue.priority", "headline_accuracy_pct": 91.0,
                                    "p50_ms": 8.0, "p99_ms": 9.0, "mean_prompt_tokens": 43.0,
                                    "cost_per_100k_usd": cost}},
    }


def test_stage_c_combos_are_unique_and_match_defaults():
    candidates = stage_c_candidates()
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids)) == len(DEFAULT_STAGE_C_COMBOS)


def test_load_in_4bit_property_matches_quantization_string():
    assert StageCCandidate("x", "4bit", False).load_in_4bit is True
    assert StageCCandidate("x", "fp16", False).load_in_4bit is False


def test_stage_c_refuses_to_run_without_a_trained_winner(customer_ctx):
    candidate = StageCCandidate("stageC-fp16-unconstrained", "fp16", False)
    with pytest.raises(FileNotFoundError, match="stage-b"):
        runner.run_stage_c_candidate(customer_ctx, "never-trained", candidate)


def test_stage_c_passes_the_winners_adapter_and_toggles_through(monkeypatch, customer_ctx):
    captured = {}

    def fake_benchmark_run(**kwargs):
        captured.update(kwargs)
        return _fake_eval_result()

    monkeypatch.setattr(runner.benchmark, "run", fake_benchmark_run)
    candidate = StageCCandidate("stageC-fp16-constrained", "fp16", True)

    runner.run_stage_c_candidate(customer_ctx, "stageB-r16-a16", candidate)

    assert captured["finetuned_model"] == str(customer_ctx.adapter_dir("stageB-r16-a16"))
    assert captured["load_in_4bit"] is False
    assert captured["constrained"] is True
    assert captured["systems"] == "finetuned"


def test_stage_c_writes_its_own_manifest_separate_from_the_winners(monkeypatch, customer_ctx):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_eval_result())
    candidate = StageCCandidate("stageC-4bit-constrained", "4bit", True)

    runner.run_stage_c_candidate(customer_ctx, "stageB-r16-a16", candidate)

    assert (customer_ctx.manifests_dir("stageC-4bit-constrained") / "evaluate.json").exists()
    assert customer_ctx.manifests_dir("stageC-4bit-constrained") != \
        customer_ctx.manifests_dir("stageB-r16-a16")


def test_stage_c_result_feeds_the_leaderboard(monkeypatch, customer_ctx):
    from ftplatform.candidates import leaderboard

    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_eval_result(cost=2.0))
    candidate = StageCCandidate("stageC-fp16-unconstrained", "fp16", False)
    runner.run_stage_c_candidate(customer_ctx, "stageB-r16-a16", candidate)

    rows = leaderboard.load_candidate_systems(customer_ctx, "stageC-fp16-unconstrained")
    assert len(rows) == 1
    assert rows[0].cost_per_100k_usd == 2.0
