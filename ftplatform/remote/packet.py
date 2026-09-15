"""
Packages one customer's job into a self-contained "packet" -- a small
sqlite db plus a copy of that customer's `customers/<id>/` tree -- that can
run somewhere with no access to this machine's disk, and merges the result
back afterwards.

A packet is deliberately scoped to exactly one customer and one job: it
never carries another customer's rows or files, so a remote run cannot leak
cross-customer state even by accident. This is the same isolation guarantee
`CustomerContext` enforces locally (see customers/context.py), extended
across a machine boundary.

Four functions, two on each side of the boundary:
  export_job_packet()   local:  customers.db + customers/<id>/ -> packet/
  apply_packet()         remote: packet/ -> a fresh customers.db + customers/<id>/
  collect_result_packet() remote: customers.db + customers/<id>/ -> result/
  import_job_result()    local:  result/ -> merged into customers.db + customers/<id>/

`apply_packet`/`collect_result_packet` run on the remote side (inside the
Kaggle kernel script, see notebooks/kaggle_runner/kaggle_customer_job.py) but
live here rather than there because they're plain, independently testable
functions with no GPU dependency -- keeping them in the pip-installed
package means the Kaggle-side script that calls them stays a thin driver.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import ftspec.config as config_mod
from ftplatform.db import SCHEMA
from ftplatform.jobs import queue

PACKET_DB_NAME = "job.db"
PACKET_DATA_DIRNAME = "customer_data"


def _repo_root(repo_root: Path | None) -> Path:
    return repo_root if repo_root is not None else config_mod.REPO_ROOT


def _customer_tree(repo_root: Path, customer_id: str) -> Path:
    return Path(repo_root) / "customers" / customer_id


def export_job_packet(conn: sqlite3.Connection, customer_id: str, job_id: str,
                       out_dir: Path, repo_root: Path | None = None) -> Path:
    """Writes a packet for one pending job to `out_dir`. Returns `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    customer = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if customer is None:
        raise ValueError(f"no such customer: {customer_id!r}")
    job = conn.execute("SELECT * FROM jobs WHERE id = ? AND customer_id = ?",
                        (job_id, customer_id)).fetchone()
    if job is None:
        raise ValueError(f"no such job {job_id!r} for customer {customer_id!r}")

    packet_db_path = out_dir / PACKET_DB_NAME
    packet_db_path.unlink(missing_ok=True)
    packet = sqlite3.connect(packet_db_path)
    packet.executescript(SCHEMA)

    packet.execute(
        "INSERT INTO customers (id, name, workload, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (customer["id"], customer["name"], customer["workload"], customer["status"],
         customer["created_at"]))
    packet.execute(
        "INSERT INTO jobs (id, customer_id, kind, status, payload_json, result_json, "
        "created_at, started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job["id"], job["customer_id"], job["kind"], job["status"], job["payload_json"],
         job["result_json"], job["created_at"], job["started_at"], job["finished_at"]))
    for row in conn.execute("SELECT * FROM deployments WHERE customer_id = ?", (customer_id,)):
        packet.execute(
            "INSERT INTO deployments (customer_id, workload, candidate_id, deployed_at, "
            "is_rollback, metrics_json) VALUES (?, ?, ?, ?, ?, ?)",
            (row["customer_id"], row["workload"], row["candidate_id"], row["deployed_at"],
             row["is_rollback"], row["metrics_json"]))
    approval = conn.execute("SELECT * FROM approvals WHERE customer_id = ?",
                             (customer_id,)).fetchone()
    if approval is not None:
        packet.execute("INSERT INTO approvals (customer_id, approved_at) VALUES (?, ?)",
                        (approval["customer_id"], approval["approved_at"]))
    packet.commit()
    packet.close()

    src_tree = _customer_tree(_repo_root(repo_root), customer_id)
    dest_tree = out_dir / PACKET_DATA_DIRNAME
    if dest_tree.exists():
        shutil.rmtree(dest_tree)
    if src_tree.exists():
        shutil.copytree(src_tree, dest_tree)
    else:
        dest_tree.mkdir(parents=True)

    return out_dir


def apply_packet(packet_dir: Path, repo_root: Path | None = None) -> str:
    """Remote side: recreates a customers.db and customers/<id>/ tree under
    `repo_root` from a packet, so CustomerContext and the job worker behave
    exactly as they do locally. Returns the packet's customer id."""
    packet_dir = Path(packet_dir)
    packet_db_path = packet_dir / PACKET_DB_NAME
    packet = sqlite3.connect(packet_db_path)
    packet.row_factory = sqlite3.Row
    customer_row = packet.execute("SELECT * FROM customers").fetchone()
    packet.close()
    if customer_row is None:
        raise ValueError(f"packet at {packet_dir} carries no customer row")
    customer_id = customer_row["id"]

    root = _repo_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(packet_db_path, root / "customers.db")

    src_tree = packet_dir / PACKET_DATA_DIRNAME
    dest_tree = _customer_tree(root, customer_id)
    if dest_tree.exists():
        shutil.rmtree(dest_tree)
    dest_tree.parent.mkdir(parents=True, exist_ok=True)
    if src_tree.exists():
        shutil.copytree(src_tree, dest_tree)
    else:
        dest_tree.mkdir(parents=True)

    return customer_id


def collect_result_packet(customer_id: str, out_dir: Path, repo_root: Path | None = None) -> Path:
    """Remote side: the counterpart to apply_packet(), run after the job
    finishes -- copies the now-updated customers.db and this customer's tree
    out to `out_dir` so it can be downloaded and merged in with
    import_job_result(). Returns `out_dir`."""
    root = _repo_root(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "customers.db", out_dir / PACKET_DB_NAME)

    src_tree = _customer_tree(root, customer_id)
    dest_tree = out_dir / PACKET_DATA_DIRNAME
    if dest_tree.exists():
        shutil.rmtree(dest_tree)
    if src_tree.exists():
        shutil.copytree(src_tree, dest_tree)
    else:
        dest_tree.mkdir(parents=True)

    return out_dir


def import_job_result(conn: sqlite3.Connection, customer_id: str, job_id: str,
                       result_dir: Path, repo_root: Path | None = None) -> dict:
    """Local side: merges a completed packet (downloaded from wherever the
    job ran) back into `conn` and `customers/<id>/`. New deployment rows are
    matched on (customer_id, candidate_id, deployed_at) rather than the
    autoincrement id, which the remote db assigned independently. Returns the
    finished job as a dict (see jobs/queue.py's row shape)."""
    result_dir = Path(result_dir)
    remote = sqlite3.connect(result_dir / PACKET_DB_NAME)
    remote.row_factory = sqlite3.Row

    job = remote.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        raise ValueError(f"result packet at {result_dir} carries no job {job_id!r}")
    conn.execute(
        "UPDATE jobs SET status = ?, result_json = ?, started_at = ?, finished_at = ? "
        "WHERE id = ?",
        (job["status"], job["result_json"], job["started_at"], job["finished_at"], job_id))

    local_keys = {
        (row["customer_id"], row["candidate_id"], row["deployed_at"])
        for row in conn.execute(
            "SELECT customer_id, candidate_id, deployed_at FROM deployments "
            "WHERE customer_id = ?", (customer_id,))
    }
    for row in remote.execute("SELECT * FROM deployments WHERE customer_id = ?", (customer_id,)):
        key = (row["customer_id"], row["candidate_id"], row["deployed_at"])
        if key in local_keys:
            continue
        conn.execute(
            "INSERT INTO deployments (customer_id, workload, candidate_id, deployed_at, "
            "is_rollback, metrics_json) VALUES (?, ?, ?, ?, ?, ?)",
            (row["customer_id"], row["workload"], row["candidate_id"], row["deployed_at"],
             row["is_rollback"], row["metrics_json"]))
    conn.commit()
    remote.close()

    src_tree = result_dir / PACKET_DATA_DIRNAME
    if src_tree.exists():
        dest_tree = _customer_tree(_repo_root(repo_root), customer_id)
        dest_tree.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_tree, dest_tree, dirs_exist_ok=True)

    return queue.get(conn, job_id)
