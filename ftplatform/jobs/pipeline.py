"""
The whole onboarding chain for one customer, as one resumable state machine:

    baseline -> stage_a -> stage_b -> [stage_c] -> deploy -> done

Each stage enqueues ordinary jobs (the same kinds `ftplatform job enqueue`
takes); `advance()` checks whether the current stage's jobs have finished,
picks that stage's winner from their evaluate manifests, and enqueues the next
stage. All state lives in the `pipelines` table, so a driver killed mid-run
(a closed laptop, an expired Kaggle session) resumes exactly where it stopped.

`drive()` is the loop that makes it hands-off: advance, run whatever is
pending (locally, or on Kaggle one job at a time), repeat -- until the
pipeline is done, has failed, or reaches the one step a human must take:
approving a customer's first-ever deployment (see jobs/approvals.py). That
approval is deliberate, not a gap: nothing is served to a customer's traffic
without someone having looked at the numbers once.

Stage A/B tolerate individual job failures (a model that OOMs is simply not
a candidate) as long as at least one job in the stage succeeded.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone

from ftplatform.candidates import leaderboard as lb
from ftplatform.candidates import scoring
from ftplatform.candidates.generator import (
    DEFAULT_LORA_GRID,
    DEFAULT_STAGE_A_CANDIDATES,
    stage_b_candidates,
    stage_c_candidates,
)
from ftplatform.customers.context import CustomerContext
from ftplatform.deployment import deploy as deploy_mod
from ftplatform.deployment import registry
from ftplatform.jobs import approvals, queue
from ftspec.run import get_logger

log = get_logger("ftplatform.jobs.pipeline")

STAGES = ("baseline", "stage_a", "stage_b", "stage_c", "deploy", "done")
GPU_KINDS = {"baseline", "stage_a", "stage_b", "stage_c", "retrain_cycle"}
TERMINAL = {"done", "failed"}

DEFAULT_OPTIONS = {
    "n_train": 200, "n_val": 40, "n_eval": 150,
    "baseline_systems": "base-schema,base-rubric,base-constrained",
    "stage_a_top_k": None,        # None = every preset; an int keeps the historically best k
    "lora_grid_top_k": None,
    "with_stage_c": False,
    "max_steps": -1,
    "min_adherence_pct": scoring.DEFAULT_MIN_ADHERENCE_PCT,
    "epsilon": deploy_mod.DEFAULT_EPSILON,
    "gpu_cost_per_hour": 0.35,
}


class PipelineError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM pipelines WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["state"] = json.loads(d.pop("state_json"))
    return d


def _save(conn, customer_id: str, stage: str, status: str, state: dict) -> dict:
    conn.execute("UPDATE pipelines SET stage = ?, status = ?, state_json = ?, updated_at = ? "
                 "WHERE customer_id = ?",
                 (stage, status, json.dumps(state, ensure_ascii=False), _now(), customer_id))
    conn.commit()
    return get(conn, customer_id)


def start(conn: sqlite3.Connection, customer_id: str, restart: bool = False, **options) -> dict:
    """Create this customer's pipeline (or, with `restart`, replace a
    finished/failed one). Refuses to clobber one that is still in progress."""
    CustomerContext(conn, customer_id)  # raises UnknownCustomerError early
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"unknown pipeline option(s): {sorted(unknown)}")
    existing = get(conn, customer_id)
    if existing is not None:
        if existing["status"] not in TERMINAL and not restart:
            raise PipelineError(
                f"{customer_id!r} already has a pipeline at stage {existing['stage']!r} "
                f"({existing['status']}) -- use `pipeline drive` to continue it, or restart.")
        conn.execute("DELETE FROM pipelines WHERE customer_id = ?", (customer_id,))

    state = {"options": {**DEFAULT_OPTIONS, **options}, "jobs": {}, "winners": {}, "log": []}
    now = _now()
    conn.execute("INSERT INTO pipelines (customer_id, stage, status, state_json, created_at, "
                 "updated_at) VALUES (?, 'baseline', 'running', ?, ?, ?)",
                 (customer_id, json.dumps(state), now, now))
    conn.commit()
    return get(conn, customer_id)


# --- per-stage job construction ------------------------------------------------

def _top_k(conn, workload, stage, items, key, k):
    from ftplatform.learning import stats
    ordered = stats.reorder_by_history(conn, workload, stage, list(items), key=key)
    return ordered[:k] if k else ordered


def _enqueue_stage(conn, ctx: CustomerContext, stage: str, state: dict) -> list[str]:
    opts, cid, wl = state["options"], ctx.customer.id, ctx.customer.workload
    gpu = {"gpu_cost_per_hour": opts["gpu_cost_per_hour"]}

    if stage == "baseline":
        payload = {"n_train": opts["n_train"], "n_val": opts["n_val"], "n_eval": opts["n_eval"],
                   "systems": opts["baseline_systems"], **gpu}
        return [queue.enqueue(conn, cid, "baseline", payload)]

    if stage == "stage_a":
        presets = _top_k(conn, wl, "stage_a", DEFAULT_STAGE_A_CANDIDATES,
                         lambda c: c.candidate_id, opts["stage_a_top_k"])
        return [queue.enqueue(conn, cid, "stage_a", {"candidate": asdict(c), "kwargs": gpu})
                for c in presets]

    if stage == "stage_b":
        from ftplatform.candidates.generator import stage_b_candidate_id
        grid = _top_k(conn, wl, "stage_b", DEFAULT_LORA_GRID,
                      lambda p: stage_b_candidate_id(*p), opts["lora_grid_top_k"])
        base_model = state["winners"]["stage_a"]["base_model"]
        return [queue.enqueue(conn, cid, "stage_b",
                              {"candidate": asdict(c),
                               "kwargs": {"max_steps": opts["max_steps"], **gpu}})
                for c in stage_b_candidates(base_model, tuple(grid))]

    if stage == "stage_c":
        winner = state["winners"]["stage_b"]["candidate_id"]
        return [queue.enqueue(conn, cid, "stage_c",
                              {"candidate": asdict(c), "winner_candidate_id": winner,
                               "kwargs": gpu})
                for c in stage_c_candidates()]

    if stage == "deploy":
        final = state["winners"]["final"]
        return [queue.enqueue(conn, cid, "deploy", {
            "candidate_id": final["candidate_id"],
            "kwargs": {"new_system": final["system"],
                       "min_adherence_pct": opts["min_adherence_pct"],
                       "epsilon": opts["epsilon"]}})]

    raise ValueError(f"no jobs for stage {stage!r}")


# --- winner selection ------------------------------------------------------------

def _best(ctx, candidate_ids: list[str], min_adherence_pct: float,
          system: str | None = None) -> dict | None:
    rows = []
    for c in candidate_ids:
        try:
            rows.extend(r for r in lb.load_candidate_systems(ctx, c)
                        if system is None or r.system == system)
        except FileNotFoundError:
            continue
    ranked = scoring.rank(rows, min_adherence_pct=min_adherence_pct) if rows else []
    survivors = [r for r in ranked if not r["gated"]]
    return survivors[0] if survivors else None


def _finish_stage(conn, ctx, stage: str, state: dict, done_jobs: list[dict]) -> str | None:
    """Record the stage's outcome in `state`. Returns an error string if
    the pipeline cannot continue, else None."""
    opts = state["options"]
    ids = [j["payload"]["candidate"]["candidate_id"] for j in done_jobs
           if "candidate" in j["payload"]]

    if stage == "baseline":
        return None

    if stage == "stage_a":
        best = _best(ctx, ids, opts["min_adherence_pct"])
        if best is None:
            return "no Stage-A base model cleared the schema-adherence floor"
        base_model = deploy_mod.infer_base_model(ctx, best["candidate_id"])
        state["winners"]["stage_a"] = {"candidate_id": best["candidate_id"],
                                       "system": best["system"], "score": best["score"],
                                       "base_model": base_model}
        return None

    if stage == "stage_b":
        best = _best(ctx, ids, opts["min_adherence_pct"], system="finetuned")
        if best is None:
            return "no trained Stage-B candidate cleared the schema-adherence floor"
        state["winners"]["stage_b"] = {"candidate_id": best["candidate_id"],
                                       "system": best["system"], "score": best["score"]}
        state["winners"]["final"] = dict(state["winners"]["stage_b"])
        return None

    if stage == "stage_c":
        pool = [state["winners"]["stage_b"]["candidate_id"], *ids]
        best = _best(ctx, pool, opts["min_adherence_pct"], system="finetuned")
        if best is not None:
            state["winners"]["final"] = {"candidate_id": best["candidate_id"],
                                         "system": best["system"], "score": best["score"]}
        return None

    if stage == "deploy":
        result = done_jobs[0]["result"] or {}
        state["deploy_result"] = {"deployed": bool(result.get("deployed")),
                                  "reason": result.get("reason")}
        return None

    return None


def _next_stage(stage: str, state: dict) -> str:
    nxt = STAGES[STAGES.index(stage) + 1]
    if nxt == "stage_c" and not state["options"]["with_stage_c"]:
        return "deploy"
    return nxt


def _all_candidate_ids(state: dict, conn) -> dict[str, str]:
    ids = {}
    for stage in ("stage_a", "stage_b", "stage_c"):
        for job_id in state["jobs"].get(stage, []):
            job = queue.get(conn, job_id)
            if job and job["status"] == "done":
                c = job["payload"]["candidate"]["candidate_id"]
                ids[c] = c
    return ids


def advance(conn: sqlite3.Connection, customer_id: str) -> dict:
    """Move the pipeline forward as far as it can go without running
    anything. Returns the pipeline row (see get())."""
    p = get(conn, customer_id)
    if p is None:
        raise PipelineError(f"{customer_id!r} has no pipeline -- `pipeline start` it first.")
    ctx = CustomerContext(conn, customer_id)
    stage, state = p["stage"], p["state"]

    while True:
        if p["status"] in TERMINAL:
            return p

        if stage == "deploy" and not state["jobs"].get("deploy"):
            first_ever = registry.current(conn, customer_id) is None
            if first_ever and not approvals.is_approved(conn, customer_id):
                if p["status"] != "awaiting_approval":
                    state["log"].append(f"{_now()} waiting for `ftplatform approve {customer_id}`")
                return _save(conn, customer_id, stage, "awaiting_approval", state)

        job_ids = state["jobs"].get(stage)
        if job_ids is None:
            job_ids = state["jobs"][stage] = _enqueue_stage(conn, ctx, stage, state)
            state["log"].append(f"{_now()} {stage}: enqueued {len(job_ids)} job(s)")
            return _save(conn, customer_id, stage, "running", state)

        jobs = [queue.get(conn, j) for j in job_ids]
        if any(j["status"] in ("pending", "running") for j in jobs):
            return _save(conn, customer_id, stage, "running", state)

        done = [j for j in jobs if j["status"] == "done"]
        failed = [j for j in jobs if j["status"] == "failed"]
        for j in failed:
            state["log"].append(f"{_now()} {stage}: job {j['id']} failed: "
                                f"{(j['result'] or {}).get('error')}")
        if not done:
            state["error"] = f"every {stage} job failed"
            return _save(conn, customer_id, stage, "failed", state)

        error = _finish_stage(conn, ctx, stage, state, done)
        if error:
            state["error"] = error
            return _save(conn, customer_id, stage, "failed", state)
        state["log"].append(f"{_now()} {stage}: finished ({len(done)} ok, {len(failed)} failed)")

        stage = _next_stage(stage, state)
        if stage == "done":
            candidates = _all_candidate_ids(state, conn)
            if candidates:
                lb.build(ctx, candidates, conn=conn,
                         min_adherence_pct=state["options"]["min_adherence_pct"])
            return _save(conn, customer_id, "done", "done", state)
        p = _save(conn, customer_id, stage, "running", state)


# --- the driver ------------------------------------------------------------------

def _pending_jobs(conn, p: dict) -> list[dict]:
    ids = p["state"]["jobs"].get(p["stage"], [])
    jobs = [queue.get(conn, j) for j in ids]
    return [j for j in jobs if j and j["status"] in ("pending", "running")]


def run_job_locally(conn, job: dict) -> None:
    from ftplatform.jobs import worker
    claimed = queue.claim(conn, job["id"])
    if claimed is not None:
        worker.run_claimed(conn, claimed)


def kaggle_runner(owner: str, poll_interval_s: float = 60.0, required_hours: float | None = None,
                   sleep: Callable[[float], None] = time.sleep,
                   on_status: Callable[[str], None] | None = None):
    """A `run_job` for drive() that ships GPU jobs to Kaggle -- one at a time,
    since a second concurrent session would only split the same weekly
    quota -- and polls until each one's result is merged back. Non-GPU jobs
    (deploy) still run locally; they need no GPU and finish in seconds."""
    from ftplatform.remote.run_on_kaggle import (
        default_work_dir,
        kernel_id_for,
        poll_and_finish_job,
        start_job_on_kaggle,
    )

    def run_job(conn, job: dict) -> None:
        if job["kind"] not in GPU_KINDS:
            run_job_locally(conn, job)
            return
        work_dir = default_work_dir(job["id"])
        if job["status"] == "pending":
            start_job_on_kaggle(conn, job["customer_id"], job["id"], owner, work_dir,
                                required_hours=required_hours)
            if on_status:
                on_status(f"pushed {job['kind']} job {job['id']} to Kaggle")
        kernel_id = kernel_id_for(job["id"], owner)
        while poll_and_finish_job(conn, job["customer_id"], job["id"], kernel_id, work_dir) is None:
            sleep(poll_interval_s)
        if on_status:
            finished = queue.get(conn, job["id"])
            on_status(f"{job['kind']} job {job['id']} -> {finished['status']}")

    return run_job


def drive(conn: sqlite3.Connection, customer_id: str,
          run_job: Callable[[sqlite3.Connection, dict], None] = run_job_locally,
          on_status: Callable[[str], None] | None = None, max_steps: int = 1000) -> dict:
    """Advance and run jobs until the pipeline is done, failed, or waiting
    for approval. A 'running' job left over from a killed local driver is
    marked failed rather than silently re-run (it may have left partial
    state); a Kaggle-run one is resumed by polling, via `run_job`."""
    for _ in range(max_steps):
        p = advance(conn, customer_id)
        if on_status:
            on_status(f"stage={p['stage']} status={p['status']}")
        if p["status"] in TERMINAL or p["status"] == "awaiting_approval":
            return p
        pending = _pending_jobs(conn, p)
        if not pending:
            continue
        for job in pending:
            if job["status"] == "running" and run_job is run_job_locally:
                queue.mark_failed(conn, job["id"], "interrupted: its local driver stopped mid-run")
                continue
            run_job(conn, job)
    raise PipelineError(f"pipeline for {customer_id!r} did not settle in {max_steps} steps")
