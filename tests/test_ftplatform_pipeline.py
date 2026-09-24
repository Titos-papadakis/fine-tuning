"""
ftplatform.jobs.pipeline -- onboarding as one resumable chain. GPU work is
replaced by a fake `run_job` that writes the same evaluate manifests a real
candidate run leaves on disk; everything else (winner selection, approval
gate, the real deploy job, leaderboard) runs for real.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import runner
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.deployment import registry
from ftplatform.jobs import approvals, pipeline, queue

QWEN = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def _m(record_exact, headline=None, adherence=100.0, cost=1.0, p99=500.0):
    return {"schema_adherence_pct": adherence, "record_exact_pct": record_exact,
            "headline_accuracy_pct": headline if headline is not None else record_exact,
            "cost_per_100k_usd": cost, "p99_ms": p99, "p50_ms": p99 / 2,
            "mean_prompt_tokens": 50.0, "headline_field": "issue.category"}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    c = connect(tmp_path / "customers.db")
    store.create(c, "acme", "Acme Inc", "saas_support")
    yield c
    c.close()


def _write_eval(ctx, candidate_id, systems, params):
    d = ctx.manifests_dir(candidate_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "evaluate.json").write_text(json.dumps({"params": params, "metrics": {"systems": systems}}),
                                      encoding="utf-8")


class FakeGPU:
    """Stands in for the GPU stages. `scores` maps candidate_id -> record_exact
    for its main system; `fail` is a set of candidate_ids whose job fails."""

    def __init__(self, scores=None, fail=()):
        self.scores = scores or {}
        self.fail = set(fail)
        self.ran: list[str] = []

    def __call__(self, conn, job):
        if job["kind"] not in pipeline.GPU_KINDS:
            pipeline.run_job_locally(conn, job)
            return
        claimed = queue.claim(conn, job["id"])
        ctx = CustomerContext(conn, job["customer_id"])
        p = job["payload"]
        cid = p.get("candidate", {}).get("candidate_id", "baseline")
        self.ran.append(cid)
        if cid in self.fail:
            queue.mark_failed(conn, claimed["id"], "RuntimeError: CUDA out of memory")
            return
        if job["kind"] == "baseline":
            systems = {"base-schema": _m(30.0, cost=2.0, p99=900.0),
                       "base-constrained": _m(36.0, cost=2.0, p99=900.0)}
            _write_eval(ctx, "baseline", systems, {"systems": "x"})
            runner._write_baseline_snapshot(ctx, {"n_eval": 150, "systems": systems})
        elif job["kind"] == "stage_a":
            c = p["candidate"]
            _write_eval(ctx, cid, {"base-constrained": _m(self.scores.get(cid, 30.0))},
                        {"base_model": c["base_model"], "candidate_id": cid})
        elif job["kind"] == "stage_b":
            c = p["candidate"]
            _write_eval(ctx, cid, {"finetuned": _m(self.scores.get(cid, 80.0), cost=0.5, p99=300.0)},
                        {"candidate_id": cid})
            train_dir = ctx.manifests_dir(cid)
            (train_dir / "train.json").write_text(
                json.dumps({"params": {"base_model": c["base_model"]}}), encoding="utf-8")
            ctx.adapter_dir(cid).mkdir(parents=True, exist_ok=True)
            (ctx.adapter_dir(cid) / "adapter_model.safetensors").write_text(cid, encoding="utf-8")
        elif job["kind"] == "stage_c":
            _write_eval(ctx, cid, {"finetuned": _m(self.scores.get(cid, 70.0), cost=0.5, p99=300.0)},
                        {"candidate_id": cid, "winner_candidate_id": p["winner_candidate_id"]})
        queue.mark_done(conn, claimed["id"], {"ok": True})


def test_full_chain_stops_for_first_approval_then_deploys(conn):
    gpu = FakeGPU(scores={"stageA-qwen25-3b": 40.0, "stageB-r16-a16": 90.0})
    pipeline.start(conn, "acme")

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "awaiting_approval" and p["stage"] == "deploy"
    assert p["state"]["winners"]["stage_a"]["base_model"] == QWEN
    assert p["state"]["winners"]["stage_b"]["candidate_id"] == "stageB-r16-a16"
    assert registry.current(conn, "acme") is None   # nothing served before approval

    approvals.approve(conn, "acme")
    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "done"
    assert p["state"]["deploy_result"]["deployed"] is True
    assert registry.current(conn, "acme")["candidate_id"] == "stageB-r16-a16"
    ctx = CustomerContext(conn, "acme")
    assert (ctx.production_dir() / "lora_adapter").exists()
    assert (ctx.benchmark_dir() / "leaderboard.json").exists()
    # Stage B trained the Stage-A winner's base model, every grid point.
    stage_b_jobs = [queue.get(conn, j) for j in p["state"]["jobs"]["stage_b"]]
    assert {j["payload"]["candidate"]["base_model"] for j in stage_b_jobs} == {QWEN}
    assert len(stage_b_jobs) == 3


def test_an_already_deployed_customer_needs_no_new_approval(conn):
    gpu = FakeGPU(scores={"stageB-r16-a16": 90.0})
    registry.record(conn, "acme", "saas_support", "old", {"systems": {}})
    pipeline.start(conn, "acme")

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "done"


def test_one_failed_stage_a_model_is_tolerated(conn):
    gpu = FakeGPU(scores={"stageA-phi35-mini": 45.0}, fail={"stageA-qwen25-3b"})
    pipeline.start(conn, "acme")

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "awaiting_approval"
    assert p["state"]["winners"]["stage_a"]["candidate_id"] == "stageA-phi35-mini"
    assert any("failed" in line for line in p["state"]["log"])


def test_every_stage_a_job_failing_fails_the_pipeline(conn):
    gpu = FakeGPU(fail={"stageA-qwen25-3b", "stageA-phi35-mini", "stageA-llama31-8b"})
    pipeline.start(conn, "acme")

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "failed"
    assert "stage_a" in p["state"]["error"]
    assert "stageB-r8-a16" not in gpu.ran


def test_a_failed_baseline_fails_the_pipeline_before_any_candidate(conn):
    gpu = FakeGPU(fail={"baseline"})
    pipeline.start(conn, "acme")

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "failed" and p["stage"] == "baseline"
    assert gpu.ran == ["baseline"]


def test_advance_without_running_is_idempotent(conn):
    pipeline.start(conn, "acme")
    pipeline.advance(conn, "acme")
    pipeline.advance(conn, "acme")

    assert len(queue.list_jobs(conn, "acme")) == 1


def test_stage_a_top_k_limits_the_models_tried(conn):
    gpu = FakeGPU()
    pipeline.start(conn, "acme", stage_a_top_k=1, lora_grid_top_k=2)

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert len(p["state"]["jobs"]["stage_a"]) == 1
    assert len(p["state"]["jobs"]["stage_b"]) == 2


def test_stage_c_winner_is_deployable_with_the_stage_b_adapter(conn):
    gpu = FakeGPU(scores={"stageB-r16-a16": 85.0, "stageC-4bit-constrained": 95.0})
    approvals.approve(conn, "acme")
    pipeline.start(conn, "acme", with_stage_c=True)

    p = pipeline.drive(conn, "acme", run_job=gpu)

    assert p["status"] == "done"
    assert p["state"]["winners"]["final"]["candidate_id"] == "stageC-4bit-constrained"
    ctx = CustomerContext(conn, "acme")
    adapter = ctx.production_dir() / "lora_adapter" / "adapter_model.safetensors"
    assert adapter.read_text(encoding="utf-8") == "stageB-r16-a16"


def test_start_refuses_to_clobber_a_pipeline_in_progress(conn):
    pipeline.start(conn, "acme")
    with pytest.raises(pipeline.PipelineError):
        pipeline.start(conn, "acme")


def test_restart_replaces_a_finished_pipeline(conn):
    gpu = FakeGPU()
    approvals.approve(conn, "acme")
    pipeline.start(conn, "acme")
    assert pipeline.drive(conn, "acme", run_job=gpu)["status"] == "done"

    p = pipeline.start(conn, "acme", restart=False)  # finished -> allowed without restart
    assert p["stage"] == "baseline" and p["status"] == "running"


def test_unknown_option_is_rejected(conn):
    with pytest.raises(ValueError):
        pipeline.start(conn, "acme", n_trian=5)


def test_a_local_job_left_running_by_a_killed_driver_is_failed_not_rerun(conn):
    pipeline.start(conn, "acme")
    p = pipeline.advance(conn, "acme")
    queue.claim(conn, p["state"]["jobs"]["baseline"][0])  # a driver that died mid-job

    p = pipeline.drive(conn, "acme", run_job=pipeline.run_job_locally)

    assert p["status"] == "failed"
    job = queue.get(conn, p["state"]["jobs"]["baseline"][0])
    assert "interrupted" in job["result"]["error"]


def test_kaggle_runner_pushes_polls_and_runs_non_gpu_jobs_locally(conn, monkeypatch):
    from ftplatform.remote import run_on_kaggle

    pushed, polls, slept = [], [], []

    def fake_start(conn_, customer_id, job_id, owner, work_dir, required_hours=None):
        pushed.append(job_id)
        queue.claim(conn_, job_id)
        return {}

    def fake_poll(conn_, customer_id, job_id, kernel_id, work_dir):
        polls.append(job_id)
        if polls.count(job_id) < 2:
            return None
        queue.mark_done(conn_, job_id, {"ok": True})
        return queue.get(conn_, job_id)

    monkeypatch.setattr(run_on_kaggle, "start_job_on_kaggle", fake_start)
    monkeypatch.setattr(run_on_kaggle, "poll_and_finish_job", fake_poll)
    run_job = pipeline.kaggle_runner("me", poll_interval_s=7, sleep=slept.append)

    baseline_job = queue.get(conn, queue.enqueue(conn, "acme", "baseline", {}))
    run_job(conn, baseline_job)

    assert pushed == [baseline_job["id"]]
    assert polls == [baseline_job["id"]] * 2
    assert slept == [7]
    assert queue.get(conn, baseline_job["id"])["status"] == "done"

    # A job already running on Kaggle (driver restarted) is resumed, not re-pushed.
    resumed = queue.get(conn, queue.enqueue(conn, "acme", "stage_a", {}))
    queue.claim(conn, resumed["id"])
    run_job(conn, queue.get(conn, resumed["id"]))
    assert resumed["id"] not in pushed

    # Non-GPU kinds never go to Kaggle.
    deploy_job = queue.get(conn, queue.enqueue(conn, "acme", "deploy", {"candidate_id": "x"}))
    run_job(conn, deploy_job)
    assert deploy_job["id"] not in pushed
    assert queue.get(conn, deploy_job["id"])["status"] == "failed"  # ran locally, no approval
