"""
ftplatform.remote.packet: packaging one customer's job into a self-contained
directory and merging the result back. All filesystem/sqlite, no GPU and no
Kaggle -- every function here takes an explicit repo_root so tests never
depend on the real repo tree, matching the config_mod.REPO_ROOT-monkeypatch
pattern used everywhere else in this suite.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from ftplatform.customers import store
from ftplatform.db import connect
from ftplatform.jobs import queue
from ftplatform.remote import packet


@pytest.fixture
def local(tmp_path):
    repo_root = tmp_path / "local_repo"
    repo_root.mkdir()
    conn = connect(repo_root / "customers.db")
    store.create(conn, "acme", "Acme Inc", "saas_support")
    yield conn, repo_root
    conn.close()


def _write_customer_file(repo_root, customer_id, relative, content="hello"):
    p = repo_root / "customers" / customer_id / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --- export_job_packet -------------------------------------------------------

def test_export_job_packet_writes_customer_and_job_rows(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {"n_eval": 10})
    _write_customer_file(repo_root, "acme", "data/saas_support/train.jsonl", "{}\n")

    out = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                    repo_root=repo_root)

    pconn = sqlite3.connect(out / packet.PACKET_DB_NAME)
    pconn.row_factory = sqlite3.Row
    customer = pconn.execute("SELECT * FROM customers").fetchone()
    job = pconn.execute("SELECT * FROM jobs").fetchone()
    assert customer["id"] == "acme"
    assert job["id"] == job_id
    assert json.loads(job["payload_json"]) == {"n_eval": 10}


def test_export_job_packet_copies_the_customer_data_tree(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    _write_customer_file(repo_root, "acme", "data/saas_support/train.jsonl", "row1\nrow2\n")

    out = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                    repo_root=repo_root)

    copied = out / packet.PACKET_DATA_DIRNAME / "data" / "saas_support" / "train.jsonl"
    assert copied.read_text(encoding="utf-8") == "row1\nrow2\n"


def test_export_job_packet_never_carries_another_customers_rows(local):
    conn, repo_root = local
    store.create(conn, "globex", "Globex Corp", "saas_support")
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    queue.enqueue(conn, "globex", "baseline", {})
    _write_customer_file(repo_root, "globex", "data/saas_support/train.jsonl", "secret")

    out = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                    repo_root=repo_root)

    pconn = sqlite3.connect(out / packet.PACKET_DB_NAME)
    pconn.row_factory = sqlite3.Row
    assert [r["id"] for r in pconn.execute("SELECT id FROM customers")] == ["acme"]
    assert [r["id"] for r in pconn.execute("SELECT id FROM jobs")] == [job_id]
    assert not (out / packet.PACKET_DATA_DIRNAME / "globex").exists()


def test_export_job_packet_raises_for_unknown_customer(local):
    conn, repo_root = local
    with pytest.raises(ValueError, match="no such customer"):
        packet.export_job_packet(conn, "nope", "j1", repo_root / "packet", repo_root=repo_root)


def test_export_job_packet_raises_for_a_job_of_a_different_customer(local):
    conn, repo_root = local
    store.create(conn, "globex", "Globex Corp", "saas_support")
    job_id = queue.enqueue(conn, "globex", "baseline", {})
    with pytest.raises(ValueError, match="no such job"):
        packet.export_job_packet(conn, "acme", job_id, repo_root / "packet", repo_root=repo_root)


def test_export_job_packet_includes_prior_deployments_and_approval(local):
    conn, repo_root = local
    from ftplatform.deployment import registry
    from ftplatform.jobs import approvals
    registry.record(conn, "acme", "saas_support", "c0", {"systems": {}})
    approvals.approve(conn, "acme")
    job_id = queue.enqueue(conn, "acme", "deploy", {"candidate_id": "c1"})

    out = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                    repo_root=repo_root)

    pconn = sqlite3.connect(out / packet.PACKET_DB_NAME)
    pconn.row_factory = sqlite3.Row
    assert pconn.execute("SELECT COUNT(*) c FROM deployments").fetchone()["c"] == 1
    assert pconn.execute("SELECT COUNT(*) c FROM approvals").fetchone()["c"] == 1


# --- apply_packet / collect_result_packet (the remote side) ------------------

def test_apply_packet_recreates_db_and_tree_under_a_new_root(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    _write_customer_file(repo_root, "acme", "data/saas_support/train.jsonl", "row\n")
    packet_dir = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                           repo_root=repo_root)

    remote_root = repo_root / "remote"
    customer_id = packet.apply_packet(packet_dir, repo_root=remote_root)

    assert customer_id == "acme"
    assert (remote_root / "customers.db").exists()
    assert (remote_root / "customers" / "acme" / "data" / "saas_support" /
            "train.jsonl").read_text(encoding="utf-8") == "row\n"


def test_apply_packet_raises_if_packet_has_no_customer_row(tmp_path):
    packet_dir = tmp_path / "empty_packet"
    packet_dir.mkdir()
    sqlite3.connect(packet_dir / packet.PACKET_DB_NAME).close()  # no schema, no rows
    with pytest.raises(sqlite3.OperationalError):
        packet.apply_packet(packet_dir, repo_root=tmp_path / "remote")


def test_collect_result_packet_copies_db_and_tree_back_out(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    packet_dir = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                           repo_root=repo_root)
    remote_root = repo_root / "remote"
    packet.apply_packet(packet_dir, repo_root=remote_root)
    _write_customer_file(remote_root, "acme", "model/saas_support/lora_adapter/adapter.bin",
                          "weights")

    out = packet.collect_result_packet("acme", repo_root / "result", repo_root=remote_root)

    assert (out / packet.PACKET_DB_NAME).exists()
    assert (out / packet.PACKET_DATA_DIRNAME / "model" / "saas_support" / "lora_adapter" /
            "adapter.bin").read_text(encoding="utf-8") == "weights"


# --- import_job_result (back on the local side) -------------------------------

def _run_job_remotely_and_collect(conn, repo_root, customer_id, job_id, result: dict,
                                   new_file_relpath=None):
    """Test helper mirroring what the Kaggle kernel script does: export ->
    apply on a 'remote' root -> mark the job done there (+ optionally drop a
    new file, simulating a training artefact) -> collect a result packet."""
    packet_dir = packet.export_job_packet(conn, customer_id, job_id, repo_root / "packet",
                                           repo_root=repo_root)
    remote_root = repo_root / "remote"
    packet.apply_packet(packet_dir, repo_root=remote_root)

    remote_conn = sqlite3.connect(remote_root / "customers.db")
    remote_conn.execute(
        "UPDATE jobs SET status = 'done', result_json = ?, finished_at = ? WHERE id = ?",
        (json.dumps(result), "2026-01-01T00:00:00+00:00", job_id))
    remote_conn.commit()
    remote_conn.close()
    if new_file_relpath:
        _write_customer_file(remote_root, customer_id, new_file_relpath, "produced remotely")

    return packet.collect_result_packet(customer_id, repo_root / "result", repo_root=remote_root)


def test_import_job_result_updates_the_local_job_row(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    result_dir = _run_job_remotely_and_collect(conn, repo_root, "acme", job_id, {"ok": True})

    job = packet.import_job_result(conn, "acme", job_id, result_dir, repo_root=repo_root)

    assert job["status"] == "done"
    assert job["result"] == {"ok": True}
    assert queue.get(conn, job_id)["status"] == "done"


def test_import_job_result_copies_new_files_back_into_the_local_tree(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    result_dir = _run_job_remotely_and_collect(
        conn, repo_root, "acme", job_id, {"ok": True},
        new_file_relpath="model/saas_support/lora_adapter/adapter.bin")

    packet.import_job_result(conn, "acme", job_id, result_dir, repo_root=repo_root)

    produced = repo_root / "customers" / "acme" / "model" / "saas_support" / "lora_adapter" / \
        "adapter.bin"
    assert produced.read_text(encoding="utf-8") == "produced remotely"


def test_import_job_result_merges_new_deployment_rows(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "deploy", {"candidate_id": "c1"})
    packet_dir = packet.export_job_packet(conn, "acme", job_id, repo_root / "packet",
                                           repo_root=repo_root)
    remote_root = repo_root / "remote"
    packet.apply_packet(packet_dir, repo_root=remote_root)

    remote_conn = sqlite3.connect(remote_root / "customers.db")
    remote_conn.execute(
        "INSERT INTO deployments (customer_id, workload, candidate_id, deployed_at, "
        "is_rollback, metrics_json) VALUES ('acme', 'saas_support', 'c1', "
        "'2026-01-01T00:00:00+00:00', 0, '{}')")
    remote_conn.execute("UPDATE jobs SET status = 'done', result_json = '{}' WHERE id = ?",
                         (job_id,))
    remote_conn.commit()
    remote_conn.close()
    result_dir = packet.collect_result_packet("acme", repo_root / "result", repo_root=remote_root)

    packet.import_job_result(conn, "acme", job_id, result_dir, repo_root=repo_root)

    from ftplatform.deployment import registry
    assert registry.current(conn, "acme")["candidate_id"] == "c1"


def test_import_job_result_does_not_duplicate_a_deployment_already_known_locally(local):
    conn, repo_root = local
    from ftplatform.deployment import registry
    registry.record(conn, "acme", "saas_support", "c0", {"systems": {}})
    job_id = queue.enqueue(conn, "acme", "deploy", {"candidate_id": "c1"})
    result_dir = _run_job_remotely_and_collect(conn, repo_root, "acme", job_id, {"ok": True})
    # the remote packet also carries the pre-existing c0 deployment (exported
    # in export_job_packet) -- re-importing it must not create a duplicate row
    packet.import_job_result(conn, "acme", job_id, result_dir, repo_root=repo_root)

    rows = conn.execute("SELECT COUNT(*) c FROM deployments WHERE customer_id = 'acme'").fetchone()
    assert rows[0] == 1


def test_import_job_result_raises_if_job_missing_in_result_packet(local):
    conn, repo_root = local
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    other_job_id = queue.enqueue(conn, "acme", "stage_a", {})
    result_dir = _run_job_remotely_and_collect(conn, repo_root, "acme", job_id, {"ok": True})

    with pytest.raises(ValueError, match="no job"):
        packet.import_job_result(conn, "acme", other_job_id, result_dir, repo_root=repo_root)
