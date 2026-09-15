"""
See ftplatform/learning/__init__.py for what this module does and does not
share across customers. Four functions: `record()` is called from
leaderboard.py every time a leaderboard is built; `mean_scores()` and
`reorder_by_history()` are the read side, used by cli.py to reorder the
default candidate lists; `summary()` is a human-facing report only, not
consumed by any automatic decision.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar

T = TypeVar("T")


def _stage_of(candidate_id: str) -> str:
    for prefix, stage in (("stageA-", "stage_a"), ("stageB-", "stage_b"), ("stageC-", "stage_c")):
        if candidate_id.startswith(prefix):
            return stage
    return "other"


def record(conn: sqlite3.Connection, workload: str, customer_id: str, ranked: list[dict]) -> None:
    """Persists every row of an already-computed leaderboard (the list
    candidates/scoring.py::rank() returns, via leaderboard.py::build()) into
    candidate_stats. Safe to call repeatedly: every leaderboard build for
    every customer appends, nothing is deduplicated or overwritten, so the
    full history stays available for whatever's aggregated over it later."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in ranked:
        conn.execute(
            "INSERT INTO candidate_stats (workload, stage, candidate_id, system, score, "
            "gated, customer_id, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (workload, _stage_of(row["candidate_id"]), row["candidate_id"], row["system"],
             row["score"], int(row["gated"]), customer_id, now))
    conn.commit()


def mean_scores(conn: sqlite3.Connection, workload: str, stage: str) -> dict[str, float]:
    """{candidate_id: mean score} across every customer that has run this
    workload+stage so far. Gated rows (score IS NULL) are excluded rather
    than counted as zero -- a technique that failed the adherence floor for
    one customer's data shouldn't be down-weighted by a NULL, it should
    simply not vote either way for or against trying it first elsewhere."""
    rows = conn.execute(
        "SELECT candidate_id, AVG(score) AS mean_score FROM candidate_stats "
        "WHERE workload = ? AND stage = ? AND gated = 0 AND score IS NOT NULL "
        "GROUP BY candidate_id", (workload, stage)).fetchall()
    return {r["candidate_id"]: r["mean_score"] for r in rows}


def reorder_by_history(conn: sqlite3.Connection, workload: str, stage: str,
                        items: list[T], key: Callable[[T], str]) -> list[T]:
    """Reorders `items` (any of generator.py's candidate lists/tuples) so
    the historically-highest-mean-score technique for this workload+stage
    comes first. `key(item)` extracts the candidate_id to look up. Items
    with no history keep their original relative order and are appended
    after every item that has one -- so a workload with zero history (a
    first customer) returns `items` completely unchanged, and an item this
    platform has simply never tried for this workload is never penalized
    for it, only left where generator.py already put it."""
    scores = mean_scores(conn, workload, stage)
    if not scores:
        return list(items)
    with_history = [it for it in items if key(it) in scores]
    without_history = [it for it in items if key(it) not in scores]
    with_history.sort(key=lambda it: scores[key(it)], reverse=True)
    return with_history + without_history


def summary(conn: sqlite3.Connection, workload: str) -> list[dict]:
    """One row per (stage, candidate_id) run so far for this workload: how
    many times, its mean score, and how often it was gated -- for
    `ftplatform learning summary`, a human-facing report only."""
    rows = conn.execute(
        "SELECT stage, candidate_id, COUNT(*) AS n, AVG(score) AS mean_score, "
        "SUM(gated) AS gated_count FROM candidate_stats WHERE workload = ? "
        "GROUP BY stage, candidate_id ORDER BY stage, mean_score DESC",
        (workload,)).fetchall()
    return [dict(r) for r in rows]
