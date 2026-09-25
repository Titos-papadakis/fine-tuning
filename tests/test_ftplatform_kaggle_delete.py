"""
Deleting a customer also deletes the private datasets/kernels their jobs
created on Kaggle -- those hold a copy of the customer's data.
"""
from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

import ftspec.config as config_mod
from ftplatform import audit
from ftplatform import db as db_mod
from ftplatform.cli import app
from ftplatform.customers import store
from ftplatform.db import connect
from ftplatform.remote import kaggle_ops, run_on_kaggle


class Run:
    """Fake subprocess.run: fails `delete <ref>` for refs in `fail` with the given stderr."""

    def __init__(self, fail=None):
        self.fail = fail or {}
        self.calls = []

    def __call__(self, cmd, capture_output=True, text=True, **kw):
        self.calls.append(cmd[3:])
        for ref, stderr in self.fail.items():
            if ref in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")
    c = connect()
    store.create(c, "acme", "Acme", "saas_support")
    store.create(c, "globex", "Globex", "saas_support")
    yield c
    c.close()


def _artifact(conn, customer, job):
    conn.execute("INSERT INTO kaggle_artifacts VALUES (?, ?, ?, ?, '2026-09-24')",
                 (customer, job, f"me/ftplatform-job-{job}", f"me/ftplatform-job-{job}"))
    conn.commit()


def test_delete_remote_distinguishes_deleted_gone_and_failed():
    assert kaggle_ops.delete_remote("datasets", "me/a", run=Run()) is True
    assert kaggle_ops.delete_remote("datasets", "me/a", run=Run({"me/a": "404 Not Found"})) is False
    with pytest.raises(kaggle_ops.KaggleCommandError):
        kaggle_ops.delete_remote("kernels", "me/a", run=Run({"me/a": "Connection reset"}))
    with pytest.raises(ValueError):
        kaggle_ops.delete_remote("models", "me/a", run=Run())


def test_delete_job_artifacts_removes_kernel_then_dataset_for_that_customer_only(conn):
    _artifact(conn, "acme", "j1")
    _artifact(conn, "globex", "j2")
    run = Run()

    r = kaggle_ops.delete_job_artifacts(conn, "acme", run=run)

    assert r == {"jobs": 1, "deleted": 2, "already_gone": 0}
    assert run.calls == [["kernels", "delete", "me/ftplatform-job-j1", "-y"],
                         ["datasets", "delete", "me/ftplatform-job-j1", "-y"]]
    left = conn.execute("SELECT customer_id FROM kaggle_artifacts").fetchall()
    assert [r[0] for r in left] == ["globex"]


def test_a_real_failure_keeps_the_row_for_retry(conn):
    _artifact(conn, "acme", "j1")
    with pytest.raises(kaggle_ops.KaggleCommandError):
        kaggle_ops.delete_job_artifacts(conn, "acme", run=Run({"me/ftplatform-job-j1": "timeout"}))
    assert conn.execute("SELECT COUNT(*) FROM kaggle_artifacts").fetchone()[0] == 1


def test_start_job_records_the_dataset_before_pushing_the_kernel(conn, monkeypatch, tmp_path):
    monkeypatch.setattr(run_on_kaggle.packet, "export_job_packet", lambda *a, **k: None)
    monkeypatch.setattr(kaggle_ops, "upload_packet_dataset", lambda *a, **k: "me/ftplatform-job-j9")

    def push_fails(*a, **k):
        raise kaggle_ops.KaggleCommandError("push rejected")

    monkeypatch.setattr(kaggle_ops, "push_kernel", push_fails)
    from ftplatform.jobs import queue
    job = queue.enqueue(conn, "acme", "baseline", {})

    with pytest.raises(kaggle_ops.KaggleCommandError):
        run_on_kaggle.start_job_on_kaggle(conn, "acme", job, "me", tmp_path / "w")

    row = conn.execute("SELECT dataset_id FROM kaggle_artifacts WHERE customer_id='acme'").fetchone()
    assert row[0] == "me/ftplatform-job-j9"


def test_cli_delete_removes_kaggle_artifacts_and_audits_it(conn, monkeypatch):
    _artifact(conn, "acme", "j1")
    seen = {}

    def fake(conn_, customer_id, run=None):
        seen["customer"] = customer_id
        conn_.execute("DELETE FROM kaggle_artifacts WHERE customer_id = ?", (customer_id,))
        return {"jobs": 1, "deleted": 2, "already_gone": 0}

    monkeypatch.setattr(kaggle_ops, "delete_job_artifacts", fake)
    result = CliRunner().invoke(app, ["customer", "delete", "acme", "--yes"])

    assert result.exit_code == 0, result.output
    assert seen["customer"] == "acme" and "2 Kaggle" in result.output
    assert store.get(conn, "acme") is None
    assert "kaggle.delete" in [e["action"] for e in audit.entries(conn, "acme")]


def test_cli_delete_stops_before_anything_local_when_kaggle_fails(conn, monkeypatch):
    _artifact(conn, "acme", "j1")

    def boom(*a, **k):
        raise kaggle_ops.KaggleCommandError("network down")

    monkeypatch.setattr(kaggle_ops, "delete_job_artifacts", boom)
    result = CliRunner().invoke(app, ["customer", "delete", "acme", "--yes"])

    assert result.exit_code == 1 and "nothing local was deleted" in result.output
    assert store.get(conn, "acme") is not None


def test_cli_delete_skip_kaggle_warns(conn, monkeypatch):
    _artifact(conn, "acme", "j1")
    monkeypatch.setattr(kaggle_ops, "delete_job_artifacts",
                        lambda *a, **k: pytest.fail("must not call Kaggle"))
    result = CliRunner().invoke(app, ["customer", "delete", "acme", "--yes", "--skip-kaggle"])

    assert result.exit_code == 0 and "NOT deleted" in result.output
    assert store.get(conn, "acme") is None


def test_cli_delete_checks_billing_before_touching_kaggle(conn, monkeypatch):
    from ftplatform.billing import stripe_billing as sb
    _artifact(conn, "acme", "j1")
    sb.link_customer(conn, "acme", "a@acme.test", "cus_1")
    sb.record_subscription(conn, "acme", "sub_1", "active")
    monkeypatch.setattr(kaggle_ops, "delete_job_artifacts",
                        lambda *a, **k: pytest.fail("must not call Kaggle"))

    result = CliRunner().invoke(app, ["customer", "delete", "acme", "--yes"])

    assert result.exit_code == 1 and store.get(conn, "acme") is not None
