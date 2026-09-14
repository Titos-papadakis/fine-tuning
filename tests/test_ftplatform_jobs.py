"""
Milestone 5: the job queue, the human-approval gate, and the worker that
ties them together. `worker._dispatch()` imports each real orchestrator
function lazily inside its own branch (matching every other lazy-import in
this codebase, so `ftplatform job --help` never needs GPU deps) -- so
these tests monkeypatch the *source* module's function (e.g.
`ftplatform.candidates.runner.run_baseline`), not an attribute on `worker`
itself, which never holds one.
"""
from __future__ import annotations

import time

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import runner as candidates_runner
from ftplatform.customers import store
from ftplatform.db import connect
from ftplatform.deployment import deploy as deploy_mod
from ftplatform.deployment import registry
from ftplatform.jobs import approvals, queue, worker


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture
def acme(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    return conn


# --- queue -------------------------------------------------------------------

def test_enqueue_get_round_trip(acme):
    job_id = queue.enqueue(acme, "acme", "baseline", {"n_eval": 10})
    job = queue.get(acme, job_id)
    assert job["kind"] == "baseline"
    assert job["status"] == "pending"
    assert job["payload"] == {"n_eval": 10}
    assert job["result"] is None


def test_pop_next_pending_is_fifo_and_marks_running(acme):
    id1 = queue.enqueue(acme, "acme", "baseline", {})
    id2 = queue.enqueue(acme, "acme", "stage_a", {})

    popped = queue.pop_next_pending(acme)

    assert popped["id"] == id1
    assert popped["status"] == "running"
    assert queue.get(acme, id2)["status"] == "pending"


def test_pop_next_pending_returns_none_when_empty(acme):
    assert queue.pop_next_pending(acme) is None


def test_mark_done_and_mark_failed(acme):
    id1 = queue.enqueue(acme, "acme", "baseline", {})
    queue.pop_next_pending(acme)
    queue.mark_done(acme, id1, {"n_eval": 10})
    assert queue.get(acme, id1)["status"] == "done"
    assert queue.get(acme, id1)["result"] == {"n_eval": 10}

    id2 = queue.enqueue(acme, "acme", "stage_a", {})
    queue.pop_next_pending(acme)
    queue.mark_failed(acme, id2, "boom")
    assert queue.get(acme, id2)["status"] == "failed"
    assert queue.get(acme, id2)["result"] == {"error": "boom"}


def test_list_jobs_filters_by_customer(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    queue.enqueue(conn, "acme", "baseline", {})
    queue.enqueue(conn, "globex", "baseline", {})

    assert len(queue.list_jobs(conn, "acme")) == 1
    assert len(queue.list_jobs(conn, "globex")) == 1
    assert len(queue.list_jobs(conn)) == 2


def test_requeue_stale_only_touches_jobs_past_the_threshold(acme):
    fresh = queue.enqueue(acme, "acme", "baseline", {})
    queue.pop_next_pending(acme)  # now "running", started_at = now

    stale_ids = queue.requeue_stale(acme, stale_after_s=3600)

    assert stale_ids == []
    assert queue.get(acme, fresh)["status"] == "running"


def test_requeue_stale_fails_a_job_that_has_been_running_too_long(acme):
    job_id = queue.enqueue(acme, "acme", "baseline", {})
    queue.pop_next_pending(acme)
    acme.execute("UPDATE jobs SET started_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (job_id,))
    acme.commit()

    stale_ids = queue.requeue_stale(acme, stale_after_s=3600)

    assert stale_ids == [job_id]
    job = queue.get(acme, job_id)
    assert job["status"] == "failed"
    assert "stale" in job["result"]["error"]


# --- approvals -----------------------------------------------------------------

def test_customer_starts_unapproved(acme):
    assert approvals.is_approved(acme, "acme") is False


def test_approve_is_idempotent(acme):
    approvals.approve(acme, "acme")
    approvals.approve(acme, "acme")  # must not raise
    assert approvals.is_approved(acme, "acme") is True


# --- worker: dispatch ------------------------------------------------------------

def test_run_one_on_an_empty_queue_returns_none(acme):
    assert worker.run_one(acme) is None


def test_run_one_dispatches_baseline_and_marks_done(monkeypatch, acme):
    monkeypatch.setattr(candidates_runner, "run_baseline",
                         lambda ctx, **kw: {"n_eval": kw.get("n_eval", 150), "systems": {}})
    queue.enqueue(acme, "acme", "baseline", {"n_eval": 5})

    job = worker.run_one(acme)

    assert job["status"] == "done"
    assert job["result"]["n_eval"] == 5


def test_run_one_marks_failed_not_raised_for_an_unknown_kind(acme):
    queue.enqueue(acme, "acme", "not-a-real-kind", {})
    job = worker.run_one(acme)
    assert job["status"] == "failed"
    assert "unknown job kind" in job["result"]["error"]


def test_run_one_marks_failed_for_an_unknown_customer(acme):
    queue.enqueue(acme, "does-not-exist", "baseline", {})
    job = worker.run_one(acme)
    assert job["status"] == "failed"


def test_run_one_a_bad_dispatch_does_not_crash_the_worker(monkeypatch, acme):
    def boom(ctx, **kw):
        raise RuntimeError("GPU OOM")
    monkeypatch.setattr(candidates_runner, "run_baseline", boom)
    queue.enqueue(acme, "acme", "baseline", {})

    job = worker.run_one(acme)  # must not raise

    assert job["status"] == "failed"
    assert "GPU OOM" in job["result"]["error"]


# --- worker: the approval gate ----------------------------------------------------

def test_deploy_job_is_refused_for_an_unapproved_first_time_customer(acme):
    queue.enqueue(acme, "acme", "deploy", {"candidate_id": "c1"})
    job = worker.run_one(acme)
    assert job["status"] == "failed"
    assert "ftplatform approve" in job["result"]["error"]


def test_deploy_job_runs_once_approved(monkeypatch, acme):
    monkeypatch.setattr(deploy_mod, "maybe_deploy",
                         lambda ctx, cid, conn=None, **kw: {"deployed": True, "candidate_id": cid})
    approvals.approve(acme, "acme")
    queue.enqueue(acme, "acme", "deploy", {"candidate_id": "c1"})

    job = worker.run_one(acme)

    assert job["status"] == "done"
    assert job["result"]["deployed"] is True


def test_deploy_job_does_not_need_reapproval_after_a_prior_deployment(monkeypatch, acme):
    registry.record(acme, "acme", "saas_support", "c0", {"systems": {}})  # a prior deployment exists
    monkeypatch.setattr(deploy_mod, "maybe_deploy",
                         lambda ctx, cid, conn=None, **kw: {"deployed": True, "candidate_id": cid})
    queue.enqueue(acme, "acme", "deploy", {"candidate_id": "c1"})  # never approved

    job = worker.run_one(acme)

    assert job["status"] == "done"


# --- worker: run_loop --------------------------------------------------------------

def test_run_loop_drains_the_queue_and_reports_the_count(monkeypatch, acme):
    monkeypatch.setattr(candidates_runner, "run_baseline", lambda ctx, **kw: {"ok": True})
    for _ in range(3):
        queue.enqueue(acme, "acme", "baseline", {})

    count = worker.run_loop(acme, poll_interval_s=0.0)

    assert count == 3
    assert queue.pop_next_pending(acme) is None
    assert all(j["status"] == "done" for j in queue.list_jobs(acme, "acme"))


def test_run_loop_respects_max_jobs(monkeypatch, acme):
    monkeypatch.setattr(candidates_runner, "run_baseline", lambda ctx, **kw: {"ok": True})
    for _ in range(3):
        queue.enqueue(acme, "acme", "baseline", {})

    started = time.monotonic()
    count = worker.run_loop(acme, poll_interval_s=0.0, max_jobs=2)
    elapsed = time.monotonic() - started

    assert count == 2
    assert elapsed < 2  # sanity: poll_interval_s=0.0 actually took effect
    remaining = [j for j in queue.list_jobs(acme, "acme") if j["status"] == "pending"]
    assert len(remaining) == 1
