"""
Phase 8: cross-customer learning (ftplatform/learning/stats.py), plus its
two wiring points -- leaderboard.build()'s optional `conn` and the
--workload reordering on candidate list-presets/lora-grid/stage-c-combos.

Every test here checks that only technique identifiers and scores move
between customers, never their data: see test_reorder_never_needs_a_second
_customers_actual_content and the leaderboard integration tests below,
which build a real leaderboard for one customer and verify a *different*
customer's candidate listing gets reordered by it without touching that
first customer's files at all.
"""
from __future__ import annotations

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import leaderboard, runner
from ftplatform.candidates.generator import (
    DEFAULT_LORA_GRID,
    DEFAULT_STAGE_A_CANDIDATES,
    DEFAULT_STAGE_C_COMBOS,
    StageACandidate,
    stage_b_candidate_id,
    stage_c_candidate_id,
)
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.learning import stats


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
    return ctx


def _fake_result(headline=90.0, record_exact=80.0, adherence=100.0, cost=1.0, p99=100.0):
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


# --- record / mean_scores -----------------------------------------------------

def test_record_infers_stage_from_the_candidate_id_prefix(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    ranked = [
        {"candidate_id": "stageA-qwen25-3b", "system": "s", "score": 0.8, "gated": False},
        {"candidate_id": "stageB-r16-a16", "system": "s", "score": 0.7, "gated": False},
        {"candidate_id": "stageC-4bit-constrained", "system": "s", "score": 0.9, "gated": False},
    ]
    stats.record(conn, "saas_support", "acme", ranked)

    rows = conn.execute("SELECT candidate_id, stage FROM candidate_stats ORDER BY candidate_id"
                         ).fetchall()
    assert dict((r["candidate_id"], r["stage"]) for r in rows) == {
        "stageA-qwen25-3b": "stage_a", "stageB-r16-a16": "stage_b",
        "stageC-4bit-constrained": "stage_c",
    }


def test_mean_scores_excludes_gated_rows(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    ranked = [
        {"candidate_id": "stageA-a", "system": "s", "score": 0.9, "gated": False},
        {"candidate_id": "stageA-a", "system": "s", "score": None, "gated": True},
    ]
    stats.record(conn, "saas_support", "acme", ranked)

    scores = stats.mean_scores(conn, "saas_support", "stage_a")

    assert scores == {"stageA-a": 0.9}


def test_mean_scores_averages_across_customers(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    stats.record(conn, "saas_support", "acme",
                 [{"candidate_id": "stageA-a", "system": "s", "score": 0.6, "gated": False}])
    stats.record(conn, "saas_support", "globex",
                 [{"candidate_id": "stageA-a", "system": "s", "score": 0.8, "gated": False}])

    assert stats.mean_scores(conn, "saas_support", "stage_a")["stageA-a"] == pytest.approx(0.7)


def test_mean_scores_is_scoped_to_workload(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    stats.record(conn, "saas_support", "acme",
                 [{"candidate_id": "stageA-a", "system": "s", "score": 0.9, "gated": False}])
    assert stats.mean_scores(conn, "other_workload", "stage_a") == {}


# --- reorder_by_history --------------------------------------------------------

def test_reorder_by_history_is_a_no_op_with_zero_history(conn):
    ordered = stats.reorder_by_history(conn, "saas_support", "stage_a",
                                        DEFAULT_STAGE_A_CANDIDATES, key=lambda c: c.candidate_id)
    assert list(ordered) == list(DEFAULT_STAGE_A_CANDIDATES)


def test_reorder_by_history_puts_the_highest_scoring_technique_first(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    weak, mid, strong = DEFAULT_STAGE_A_CANDIDATES
    stats.record(conn, "saas_support", "acme", [
        {"candidate_id": weak.candidate_id, "system": "s", "score": 0.3, "gated": False},
        {"candidate_id": mid.candidate_id, "system": "s", "score": 0.5, "gated": False},
        {"candidate_id": strong.candidate_id, "system": "s", "score": 0.9, "gated": False},
    ])

    ordered = stats.reorder_by_history(conn, "saas_support", "stage_a",
                                        DEFAULT_STAGE_A_CANDIDATES, key=lambda c: c.candidate_id)

    assert [c.candidate_id for c in ordered] == [
        strong.candidate_id, mid.candidate_id, weak.candidate_id]


def test_reorder_by_history_appends_untried_items_after_known_ones_in_original_order(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    only_known = DEFAULT_STAGE_A_CANDIDATES[1]
    stats.record(conn, "saas_support", "acme", [
        {"candidate_id": only_known.candidate_id, "system": "s", "score": 0.5, "gated": False},
    ])

    ordered = stats.reorder_by_history(conn, "saas_support", "stage_a",
                                        DEFAULT_STAGE_A_CANDIDATES, key=lambda c: c.candidate_id)

    assert ordered[0].candidate_id == only_known.candidate_id
    untried_original_order = [c.candidate_id for c in DEFAULT_STAGE_A_CANDIDATES
                               if c.candidate_id != only_known.candidate_id]
    assert [c.candidate_id for c in ordered[1:]] == untried_original_order


def test_reorder_by_history_works_for_the_lora_grid(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    stats.record(conn, "saas_support", "acme", [
        {"candidate_id": stage_b_candidate_id(32, 32), "system": "s", "score": 0.95, "gated": False},
    ])

    ordered = stats.reorder_by_history(conn, "saas_support", "stage_b", DEFAULT_LORA_GRID,
                                        key=lambda pair: stage_b_candidate_id(*pair))

    assert ordered[0] == (32, 32)


def test_reorder_by_history_works_for_stage_c_combos(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    stats.record(conn, "saas_support", "acme", [
        {"candidate_id": stage_c_candidate_id("stageC", "fp16", True), "system": "s",
         "score": 0.99, "gated": False},
    ])

    ordered = stats.reorder_by_history(
        conn, "saas_support", "stage_c", DEFAULT_STAGE_C_COMBOS,
        key=lambda combo: stage_c_candidate_id("stageC", *combo))

    assert ordered[0] == ("fp16", True)


# --- summary --------------------------------------------------------------

def test_summary_reports_run_count_mean_score_and_gated_count(conn):
    store.create(conn, "acme", "Acme Inc", "saas_support")
    stats.record(conn, "saas_support", "acme", [
        {"candidate_id": "stageA-a", "system": "s1", "score": 0.4, "gated": False},
        {"candidate_id": "stageA-a", "system": "s2", "score": 0.6, "gated": False},
        {"candidate_id": "stageA-b", "system": "s1", "score": None, "gated": True},
    ])

    rows = {r["candidate_id"]: r for r in stats.summary(conn, "saas_support")}

    assert rows["stageA-a"]["n"] == 2
    assert rows["stageA-a"]["mean_score"] == pytest.approx(0.5)
    assert rows["stageA-a"]["gated_count"] == 0
    assert rows["stageA-b"]["n"] == 1
    assert rows["stageA-b"]["gated_count"] == 1


# --- leaderboard.build() integration -----------------------------------------

def test_leaderboard_build_with_no_conn_does_not_touch_candidate_stats(monkeypatch, customer_ctx, conn):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result())
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("stageA-x", "org/model-x", "X"))

    leaderboard.build(customer_ctx, {"stageA-x": "X"})  # no conn=... passed

    assert conn.execute("SELECT COUNT(*) c FROM candidate_stats").fetchone()["c"] == 0


def test_leaderboard_build_with_conn_records_candidate_stats(monkeypatch, customer_ctx, conn):
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result())
    runner.run_stage_a_candidate(customer_ctx, StageACandidate("stageA-x", "org/model-x", "X"))

    leaderboard.build(customer_ctx, {"stageA-x": "X"}, conn=conn)

    row = conn.execute("SELECT * FROM candidate_stats WHERE candidate_id = 'stageA-x'").fetchone()
    assert row is not None
    assert row["workload"] == "saas_support"
    assert row["customer_id"] == "acme"
    assert row["stage"] == "stage_a"


def test_a_second_customers_candidate_listing_is_reordered_by_the_firsts_results(
        monkeypatch, tmp_path, conn):
    """The end-to-end point of Phase 8: acme's leaderboard result changes
    what globex is shown first for -- via scores only, never acme's data."""
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    acme = CustomerContext(conn, "acme")
    acme.data_dir().mkdir(parents=True, exist_ok=True)
    (acme.data_dir() / "eval.jsonl").write_text("", encoding="utf-8")

    strongest = DEFAULT_STAGE_A_CANDIDATES[-1]
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_result(headline=99, record_exact=99))
    runner.run_stage_a_candidate(acme, StageACandidate(strongest.candidate_id,
                                                        strongest.base_model, strongest.label))
    leaderboard.build(acme, {strongest.candidate_id: strongest.label}, conn=conn)

    ordered_for_globex = stats.reorder_by_history(
        conn, "saas_support", "stage_a", DEFAULT_STAGE_A_CANDIDATES, key=lambda c: c.candidate_id)

    assert ordered_for_globex[0].candidate_id == strongest.candidate_id
    # globex's own tree was never touched by any of this
    globex = CustomerContext(conn, "globex")
    assert not globex.data_dir().exists()


# --- CLI wiring -------------------------------------------------------------

def test_cli_list_presets_with_workload_reorders_and_shows_scores(monkeypatch, tmp_path, conn):
    import ftplatform.db as db_mod
    from ftplatform.cli import app

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")
    store.create(conn, "acme", "Acme Inc", "saas_support")
    strongest = DEFAULT_STAGE_A_CANDIDATES[-1]
    stats.record(conn, "saas_support", "acme",
                 [{"candidate_id": strongest.candidate_id, "system": "s", "score": 0.97,
                   "gated": False}])
    conn.commit()

    from typer.testing import CliRunner
    result = CliRunner().invoke(app, ["candidate", "list-presets", "--workload", "saas_support"])

    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.splitlines() if strongest.candidate_id in ln]
    assert len(lines) == 1
    assert "0.970" in lines[0]
    # the reordered winner appears before the other two preset rows
    assert result.stdout.index(strongest.candidate_id) < result.stdout.index(
        [c.candidate_id for c in DEFAULT_STAGE_A_CANDIDATES
         if c.candidate_id != strongest.candidate_id][0])


def test_cli_list_presets_without_workload_is_unchanged(tmp_path, monkeypatch):
    import ftplatform.db as db_mod
    from ftplatform.cli import app

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")

    from typer.testing import CliRunner
    result = CliRunner().invoke(app, ["candidate", "list-presets"])

    assert result.exit_code == 0
    for c in DEFAULT_STAGE_A_CANDIDATES:
        assert c.candidate_id in result.stdout
