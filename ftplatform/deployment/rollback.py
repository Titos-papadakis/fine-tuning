"""
Manual rollback: revert production/ to a previously-run candidate.

`deploy.maybe_deploy()`'s own gates are the *automatic* rollback -- a
candidate that fails any gate is simply never promoted, so production is
never pointed at it in the first place. This module is the escape hatch for
what the gates cannot catch: a candidate that passed every statistical check
here but turned out wrong in a way only real production traffic revealed.
It deliberately bypasses every gate -- an operator invoking this has already
decided, and re-litigating that decision here would defeat the point.
"""
from __future__ import annotations

from ftplatform.candidates import leaderboard as lb
from ftplatform.deployment.deploy import _promote


def rollback_to(ctx, candidate_id: str, system: str = "finetuned", conn=None) -> dict:
    """Point production/ at `candidate_id` again, no gates. `candidate_id`
    must already have an evaluate manifest (i.e. it was run at some point --
    this does not resurrect a candidate that never existed)."""
    rows = [r for r in lb.load_candidate_systems(ctx, candidate_id) if r.system == system]
    if not rows:
        raise ValueError(f"system {system!r} not found for candidate {candidate_id!r}")

    _promote(ctx, candidate_id, rows[0], conn=conn, is_rollback=True)
    return {"deployed": True, "candidate_id": candidate_id, "rollback": True}
