"""
Milestone 2: Stage-A candidate runs plus the leaderboard built from them.

`run_stage_a_candidate()` is tested the same way M1's `run_baseline()` was --
monkeypatching `ftspec.evaluation.benchmark.run` so no GPU/model is needed --
and `leaderboard.build()` is tested against the real manifest files that
produces, so the read-back path is exercised for real, not just mocked.
"""
from __future__ import annotations

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import leaderboard, runner
from ftplatform.candidates.generator import StageACandidate, slugify_model
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
    (ctx.data_dir()).mkdir(parents=True, exist_ok=True)
    (ctx.data_dir() / "eval.jsonl").write_text("", encoding="utf-8")  # existence is all that's checked
    return ctx


def _fake_result(adherence=100.0, headline=90.0, record_exact=80.0, cost=1.0, p99=100.0):
    return {
        "profile": "saas_support", "regime": "none", "n_eval": 150, "report_path": "unused",
        "refused_for_compliance": [], "failed_systems": {}, "egress_acknowledged": False,
        "systems": {
            "base-rubric": {"schema_adherence_pct": adherence, "record_exact_pct": record_exact,
                             "headline_field": "issue.priority", "headline_accuracy_pct": headline,
                             "p50_ms": p99 / 2, "p99_ms": p99, "mean_prompt_tokens": 43.0,
                             "cost_per_100k_usd": cost},
        },
    }


def test_slugify_model_is_stable_and_filesystem_safe():
    assert slugify_model("unsloth/Qwen2.5-3B-Instruct-bnb-4bit") == "stageA-unsloth-qwen2-5-3b-instruct-bnb-4bit"
    assert slugify_model("X") == slugify_model("X")  # deterministic for the same input


def test_default_stage_a_candidates_have_unique_ids():
    from ftplatform.candidates.generator import DEFAULT_STAGE_A_CANDIDATES
    ids = [c.candidate_id for c in DEFAULT_STAGE_A_CANDIDATES]
    assert len(ids) == len(set(ids))


def test_stage_a_refuses_to_run_without_a_corpus(monkeypatch, conn, tmp_path):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "nodata", "No Data Inc", "saas_support")
    ctx = CustomerContext(conn, "nodata")  # no eval.jsonl written for this one
    candidate = StageACandidate("c1", "some/model", "label")

    with pytest.raises(FileNotFoundError, match="baseline run"):
        runner.run_stage_a_candidate(ctx, candidate)


def test_stage_a_candidate_writes_its_own_manifest(monkeypatch, customer_ctx):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result())
    candidate = StageACandidate("stageA-x", "org/model-x", "Model X")

    result = runner.run_stage_a_candidate(customer_ctx, candidate)

    assert result["systems"]["base-rubric"]["schema_adherence_pct"] == 100.0
    manifest = customer_ctx.manifests_dir("stageA-x") / "evaluate.json"
    assert manifest.exists()


def test_two_candidates_for_the_same_customer_do_not_collide(monkeypatch, customer_ctx):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(cost=1.0))
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("cand-a", "model-a", "A"))
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(cost=9.0))
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("cand-b", "model-b", "B"))

    a_manifest = customer_ctx.manifests_dir("cand-a") / "evaluate.json"
    b_manifest = customer_ctx.manifests_dir("cand-b") / "evaluate.json"
    assert a_manifest != b_manifest
    import json
    a = json.loads(a_manifest.read_text(encoding="utf-8"))
    b = json.loads(b_manifest.read_text(encoding="utf-8"))
    assert a["metrics"]["systems"]["base-rubric"]["cost_per_100k_usd"] == 1.0
    assert b["metrics"]["systems"]["base-rubric"]["cost_per_100k_usd"] == 9.0


def test_leaderboard_ranks_candidates_read_back_from_disk(monkeypatch, customer_ctx):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(headline=50.0, record_exact=50.0))
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("weak", "model-weak", "Weak"))
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(headline=99.0, record_exact=99.0))
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("strong", "model-strong", "Strong"))

    ranked = leaderboard.build(customer_ctx, {"weak": "Weak", "strong": "Strong"})

    assert ranked[0]["candidate_id"] == "strong"
    board_dir = customer_ctx.benchmark_dir()
    assert (board_dir / "leaderboard.json").exists()
    md = (board_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert "Strong" in md and "Weak" in md
    assert "| GATED |" not in md  # both rows clear the default adherence floor


def test_leaderboard_raises_a_clean_error_for_a_candidate_never_run(customer_ctx):
    with pytest.raises(FileNotFoundError, match="never-run"):
        leaderboard.load_candidate_systems(customer_ctx, "never-run")


def test_leaderboards_for_two_customers_do_not_collide(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    a, b = CustomerContext(conn, "acme"), CustomerContext(conn, "globex")
    for ctx in (a, b):
        ctx.data_dir().mkdir(parents=True, exist_ok=True)
        (ctx.data_dir() / "eval.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(cost=1.0))
    runner.run_stage_a_candidate(a, StageACandidate("c1", "m", "M"))
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(cost=2.0))
    runner.run_stage_a_candidate(b, StageACandidate("c1", "m", "M"))  # same candidate_id, different customer

    leaderboard.build(a, {"c1": "M"})
    leaderboard.build(b, {"c1": "M"})

    assert a.benchmark_dir() != b.benchmark_dir()
    a_board = (a.benchmark_dir() / "leaderboard.json").read_text(encoding="utf-8")
    b_board = (b.benchmark_dir() / "leaderboard.json").read_text(encoding="utf-8")
    assert '"cost_per_100k_usd": 1.0' in a_board
    assert '"cost_per_100k_usd": 2.0' in b_board
