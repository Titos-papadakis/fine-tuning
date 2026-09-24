"""
Phase 7 -- the job worker.

A poll loop that pops the oldest pending job and executes it by dispatching
on `kind` to whichever ftplatform function already does that work --
nothing here is new orchestration logic, only routing. Run one job at a time
(`ftplatform job run-one`) or in a loop (`ftplatform job worker`), from the
developer's machine or from a Colab cell when the job needs a GPU.

A 'deploy' job for a customer's very first deployment is refused -- marked
failed with a clear reason, not silently skipped -- unless `ftplatform
approve <customer_id>` has already run. See jobs/approvals.py.
"""
from __future__ import annotations

import time

from ftplatform.customers.context import CustomerContext
from ftplatform.jobs import approvals, queue
from ftspec.run import get_logger

log = get_logger("ftplatform.jobs.worker")


def _dispatch(conn, ctx: CustomerContext, kind: str, payload: dict):
    if kind == "baseline":
        from ftplatform.candidates.runner import run_baseline
        return run_baseline(ctx, **payload)

    if kind == "stage_a":
        from ftplatform.candidates.generator import StageACandidate
        from ftplatform.candidates.runner import run_stage_a_candidate
        candidate = StageACandidate(**payload["candidate"])
        return run_stage_a_candidate(ctx, candidate, **payload.get("kwargs", {}))

    if kind == "stage_b":
        from ftplatform.candidates.generator import StageBCandidate
        from ftplatform.candidates.runner import run_stage_b_candidate
        candidate = StageBCandidate(**payload["candidate"])
        return run_stage_b_candidate(ctx, candidate, **payload.get("kwargs", {}))

    if kind == "stage_c":
        from ftplatform.candidates.generator import StageCCandidate
        from ftplatform.candidates.runner import run_stage_c_candidate
        candidate = StageCCandidate(**payload["candidate"])
        return run_stage_c_candidate(ctx, payload["winner_candidate_id"], candidate,
                                      **payload.get("kwargs", {}))

    if kind == "leaderboard":
        from ftplatform.candidates.leaderboard import build
        return {"ranked": build(ctx, payload["candidates"], conn=conn,
                                 **payload.get("kwargs", {}))}

    if kind == "deploy":
        from ftplatform.deployment.deploy import maybe_deploy
        from ftplatform.deployment.registry import current
        is_first_ever = current(conn, ctx.customer.id) is None
        if is_first_ever and not approvals.is_approved(conn, ctx.customer.id):
            raise PermissionError(
                f"customer {ctx.customer.id!r} has no prior deployment and has not been "
                f"approved -- run `ftplatform approve {ctx.customer.id}` first")
        return maybe_deploy(ctx, payload["candidate_id"], conn=conn, **payload.get("kwargs", {}))

    if kind == "retrain_cycle":
        from ftplatform.selfimprove.retrain_cycle import run_cycle
        return run_cycle(ctx, conn=conn, **payload)

    raise ValueError(f"unknown job kind: {kind!r}")


def run_one(conn) -> dict | None:
    """Pop and execute the oldest pending job. Returns the finished job
    dict, or None if the queue was empty."""
    job = queue.pop_next_pending(conn)
    if job is None:
        return None
    return run_claimed(conn, job)


def run_claimed(conn, job: dict) -> dict:
    """Execute an already-claimed ('running') job and record its outcome."""
    log.info("running job %s (%s) for customer %s", job["id"], job["kind"], job["customer_id"])
    try:
        ctx = CustomerContext(conn, job["customer_id"])
        result = _dispatch(conn, ctx, job["kind"], job["payload"])
        queue.mark_done(conn, job["id"], result if isinstance(result, dict) else {"result": result})
        log.info("job %s done", job["id"])
    except Exception as e:                                            # noqa: BLE001
        # One bad job (a missing customer, a candidate that failed to
        # train) must not take the whole poll loop down with it -- the
        # failure is recorded on the job, not raised out of the worker.
        queue.mark_failed(conn, job["id"], f"{type(e).__name__}: {e}")
        log.error("job %s failed: %s", job["id"], e)
    return queue.get(conn, job["id"])


def run_loop(conn, poll_interval_s: float = 5.0, max_jobs: int | None = None) -> int:
    """Run jobs until the queue is empty or `max_jobs` have run. Returns the
    count executed. Meant for a long-lived process or a Colab cell, not
    unattended overnight automation -- see the plan's note on free Colab T4
    session limits."""
    count = 0
    while max_jobs is None or count < max_jobs:
        job = run_one(conn)
        if job is None:
            break
        count += 1
        if max_jobs is None or count < max_jobs:
            time.sleep(poll_interval_s)
    return count
