"""
ftplatform.remote.run_on_kaggle: the two calls that actually drive a job
through Kaggle. Every `kaggle_ops` call is monkeypatched (never a real
subprocess), so these tests verify the orchestration -- job claiming, what
gets uploaded/pushed, how a finished/still-running/errored kernel is
handled -- not the `kaggle` CLI itself (see test_ftplatform_remote_kaggle_ops.py
for that boundary, and packet tests for the packet format).
"""
from __future__ import annotations

import sqlite3

import pytest

from ftplatform.customers import store
from ftplatform.db import connect
from ftplatform.jobs import queue
from ftplatform.remote import kaggle_ops, packet, run_on_kaggle


@pytest.fixture
def local(tmp_path):
    repo_root = tmp_path / "local_repo"
    repo_root.mkdir()
    conn = connect(repo_root / "customers.db")
    store.create(conn, "acme", "Acme Inc", "saas_support")
    yield conn, repo_root
    conn.close()


def test_default_work_dir_is_scoped_to_job_id(tmp_path):
    assert run_on_kaggle.default_work_dir("j1", repo_root=tmp_path) == tmp_path / ".kaggle_work" / "j1"


def test_kernel_id_for_is_deterministic_from_job_id_and_owner():
    assert run_on_kaggle.kernel_id_for("j1", "alice") == "alice/ftplatform-job-j1"


# --- start_job_on_kaggle -----------------------------------------------------

def test_start_job_on_kaggle_claims_the_job_and_returns_ids(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset",
                         lambda *a, **kw: "alice/ftplatform-job-" + job_id)
    pushed = []
    monkeypatch.setattr(kaggle_ops, "push_kernel", lambda kernel_dir, run=None: pushed.append(kernel_dir))

    result = run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice",
                                                repo_root / "work", repo_root=repo_root)

    assert result["kernel_id"] == f"alice/ftplatform-job-{job_id}"
    assert result["dataset_id"] == "alice/ftplatform-job-" + job_id
    assert queue.get(conn, job_id)["status"] == "running"
    assert len(pushed) == 1


def test_start_job_on_kaggle_copies_the_kernel_script_into_the_kernel_dir(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", lambda *a, **kw: "alice/ds")
    monkeypatch.setattr(kaggle_ops, "push_kernel", lambda *a, **kw: None)

    work_dir = repo_root / "work"
    run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice", work_dir, repo_root=repo_root)

    assert (work_dir / "kernel" / run_on_kaggle.KERNEL_CODE_FILE).exists()
    assert (work_dir / "kernel" / "kernel-metadata.json").exists()


def test_start_job_on_kaggle_writes_the_packet_before_uploading(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    seen_packet_dirs = []

    def fake_upload(packet_dir, owner, slug, title, run=None):
        seen_packet_dirs.append(packet_dir)
        assert (packet_dir / packet.PACKET_DB_NAME).exists()
        return f"{owner}/{slug}"

    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", fake_upload)
    monkeypatch.setattr(kaggle_ops, "push_kernel", lambda *a, **kw: None)

    run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice", repo_root / "work",
                                       repo_root=repo_root)

    assert len(seen_packet_dirs) == 1


def test_start_job_on_kaggle_skips_the_preflight_check_when_required_hours_is_none(monkeypatch, local):
    # Old default behavior, unchanged: no "kernels status" call at all.
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", lambda *a, **kw: "alice/ds")
    monkeypatch.setattr(kaggle_ops, "push_kernel", lambda *a, **kw: None)

    def must_not_be_called(*a, **kw):
        raise AssertionError("preflight_check should not run when required_hours is None")
    monkeypatch.setattr(kaggle_ops, "preflight_check", must_not_be_called)

    run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice", repo_root / "work",
                                       repo_root=repo_root)

    assert queue.get(conn, job_id)["status"] == "running"


def test_start_job_on_kaggle_runs_the_preflight_check_when_required_hours_is_given(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    seen = []
    monkeypatch.setattr(kaggle_ops, "preflight_check",
                         lambda kernel_id, hours, run=None: seen.append((kernel_id, hours)))
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", lambda *a, **kw: "alice/ds")
    monkeypatch.setattr(kaggle_ops, "push_kernel", lambda *a, **kw: None)

    run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice", repo_root / "work",
                                       repo_root=repo_root, required_hours=2.5)

    assert seen == [(f"alice/ftplatform-job-{job_id}", 2.5)]


def test_start_job_on_kaggle_does_not_claim_the_job_when_the_preflight_check_fails(monkeypatch, local):
    # A rejected push must leave the job exactly as pending as it was --
    # never marked 'running' for a push that never happened.
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})

    def fail(*a, **kw):
        raise kaggle_ops.ConcurrentSessionError("already running")
    monkeypatch.setattr(kaggle_ops, "preflight_check", fail)

    def must_not_be_called(*a, **kw):
        raise AssertionError("nothing should be uploaded/pushed after a failed preflight check")
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", must_not_be_called)
    monkeypatch.setattr(kaggle_ops, "push_kernel", must_not_be_called)

    with pytest.raises(kaggle_ops.ConcurrentSessionError):
        run_on_kaggle.start_job_on_kaggle(conn, "acme", job_id, "alice", repo_root / "work",
                                           repo_root=repo_root, required_hours=2.5)

    assert queue.get(conn, job_id)["status"] == "pending"


# --- poll_and_finish_job ------------------------------------------------------

def test_poll_and_finish_job_returns_none_while_still_running(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    monkeypatch.setattr(kaggle_ops, "kernel_status", lambda *a, **kw: "running")

    def must_not_be_called(*a, **kw):
        raise AssertionError("download_kernel_output should not run while still running")
    monkeypatch.setattr(kaggle_ops, "download_kernel_output", must_not_be_called)

    result = run_on_kaggle.poll_and_finish_job(conn, "acme", job_id, "alice/ftplatform-job-x",
                                                repo_root / "work", repo_root=repo_root)

    assert result is None


def test_poll_and_finish_job_merges_the_result_when_complete(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})

    # Build the result packet a real Kaggle run would have produced: export
    # -> apply on a "remote" root -> mark done there -> collect.
    packet_dir = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                           repo_root=repo_root)
    remote_root = repo_root / "remote"
    packet.apply_packet(packet_dir, repo_root=remote_root)
    remote_conn = sqlite3.connect(remote_root / "customers.db")
    remote_conn.execute("UPDATE jobs SET status = 'done', result_json = '{\"ok\": true}' "
                         "WHERE id = ?", (job_id,))
    remote_conn.commit()
    remote_conn.close()

    work_dir = repo_root / "work"
    monkeypatch.setattr(kaggle_ops, "kernel_status", lambda *a, **kw: "complete")

    def fake_download(kernel_id, out_dir, run=None):
        packet.collect_result_packet("acme", out_dir / "result_packet", repo_root=remote_root)
        return out_dir
    monkeypatch.setattr(kaggle_ops, "download_kernel_output", fake_download)

    result = run_on_kaggle.poll_and_finish_job(conn, "acme", job_id, "alice/ftplatform-job-x",
                                                work_dir, repo_root=repo_root)

    assert result["status"] == "done"
    assert result["result"] == {"ok": True}
    assert queue.get(conn, job_id)["status"] == "done"


def test_poll_and_finish_job_marks_failed_on_a_kaggle_error_status(monkeypatch, local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    monkeypatch.setattr(kaggle_ops, "kernel_status", lambda *a, **kw: "error")
    monkeypatch.setattr(kaggle_ops, "download_kernel_output",
                         lambda kernel_id, out_dir, run=None: out_dir)

    result = run_on_kaggle.poll_and_finish_job(conn, "acme", job_id, "alice/ftplatform-job-x",
                                                repo_root / "work", repo_root=repo_root)

    assert result["status"] == "failed"
    assert "error" in result["result"]["error"]
