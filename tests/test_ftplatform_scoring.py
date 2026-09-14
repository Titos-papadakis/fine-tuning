"""
Phase 4's ranking formula: pure functions over plain numbers, so these run
with no GPU and no model, the same way `tests/test_evaluate_combine.py`
tests `ftspec.evaluation.benchmark`'s numbers without ever loading one.
"""
from __future__ import annotations

import pytest

from ftplatform.candidates.scoring import CandidateRow, rank


def _row(candidate_id, system, adherence=100.0, headline=90.0, record_exact=80.0,
         cost=1.0, p99=100.0):
    return CandidateRow(candidate_id=candidate_id, system=system,
                         schema_adherence_pct=adherence, headline_accuracy_pct=headline,
                         record_exact_pct=record_exact, cost_per_100k_usd=cost, p99_ms=p99)


def test_empty_input_ranks_to_nothing():
    assert rank([]) == []


def test_higher_quality_wins_when_cost_and_latency_are_equal():
    weak = _row("c1", "s1", headline=50.0, record_exact=50.0)
    strong = _row("c2", "s1", headline=95.0, record_exact=95.0)

    ranked = rank([weak, strong])

    assert [r["candidate_id"] for r in ranked] == ["c2", "c1"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_gated_candidates_sort_last_and_are_never_top():
    invalid = _row("c1", "s1", adherence=50.0, headline=99.0, record_exact=99.0)
    valid = _row("c2", "s1", adherence=99.0, headline=10.0, record_exact=10.0)

    ranked = rank([invalid, valid], min_adherence_pct=98.0)

    assert ranked[0]["candidate_id"] == "c2"
    assert ranked[0]["gated"] is False
    assert ranked[-1]["candidate_id"] == "c1"
    assert ranked[-1]["gated"] is True
    assert ranked[-1]["score"] is None
    assert "schema_adherence_pct" in ranked[-1]["gate_reason"]


def test_all_gated_returns_every_row_but_none_scored():
    rows = [_row("c1", "s1", adherence=10.0), _row("c2", "s1", adherence=20.0)]
    ranked = rank(rows, min_adherence_pct=98.0)
    assert len(ranked) == 2
    assert all(r["gated"] and r["score"] is None for r in ranked)


def test_cheaper_candidate_wins_when_quality_ties():
    expensive = _row("c1", "s1", cost=10.0)
    cheap = _row("c2", "s1", cost=1.0)

    ranked = rank([expensive, cheap])

    assert ranked[0]["candidate_id"] == "c2"
    assert ranked[0]["cost_score"] == 1.0
    assert ranked[1]["cost_score"] == 0.0


def test_faster_candidate_wins_when_quality_and_cost_tie():
    slow = _row("c1", "s1", p99=1000.0)
    fast = _row("c2", "s1", p99=10.0)

    ranked = rank([slow, fast])

    assert ranked[0]["candidate_id"] == "c2"
    assert ranked[0]["latency_score"] == 1.0


def test_identical_candidates_all_get_full_normalized_scores():
    rows = [_row("c1", "s1"), _row("c2", "s1")]
    ranked = rank(rows)
    assert all(r["cost_score"] == 1.0 and r["latency_score"] == 1.0 for r in ranked)


def test_weights_are_overridable_and_actually_change_the_order():
    # c1 is higher quality but much pricier; c2 is cheap but mediocre.
    quality_first = _row("c1", "s1", headline=99.0, record_exact=99.0, cost=100.0)
    cost_first = _row("c2", "s1", headline=60.0, record_exact=60.0, cost=1.0)

    by_quality = rank([quality_first, cost_first], weights={"quality": 1.0, "cost": 0.0, "latency": 0.0})
    by_cost = rank([quality_first, cost_first], weights={"quality": 0.0, "cost": 1.0, "latency": 0.0})

    assert by_quality[0]["candidate_id"] == "c1"
    assert by_cost[0]["candidate_id"] == "c2"


def test_row_headline_accuracy_none_becomes_zero_upstream_not_here():
    # scoring.py itself requires a float; leaderboard.py is responsible for
    # the `or 0.0` coercion when a profile has no headline field. Documented
    # here so a future refactor doesn't quietly drop that coercion.
    row = _row("c1", "s1", headline=0.0)  # record_exact defaults to 80.0
    assert row.quality == pytest.approx(0.6 * 0.0 + 0.4 * 0.8)
