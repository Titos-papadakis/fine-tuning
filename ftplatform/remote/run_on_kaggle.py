"""
Ties packet.py and kaggle_ops.py together into the two calls that actually
run a customer's job on Kaggle: `start_job_on_kaggle()` (export, upload,
push) and `poll_and_finish_job()` (check status; merge results back once
done). Split into two rather than one blocking call because a real job can
take up to ~90 minutes -- a poll loop belongs to whatever calls this
(CLI/worker/an interactive session), not to a function that would otherwise
have to sleep for the entire run inside one call.

A job is claimed (marked "running" locally) the moment it's pushed to
Kaggle, the same way `jobs/queue.py::pop_next_pending()` claims a job for a
local worker -- so a local `ftplatform job worker` running at the same time
won't also pick it up.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ftspec.config as config_mod
from ftplatform.remote import kaggle_ops, packet

KERNEL_SOURCE_DIR = (Path(__file__).resolve().parent.parent.parent
                      / "notebooks" / "kaggle_runner" / "customer_job")
KERNEL_CODE_FILE = "kaggle_customer_job.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_work_dir(job_id: str, repo_root: Path | None = None) -> Path:
    """Local staging directory for one job's packet/kernel/output -- fixed
    by convention on `job_id` alone so the CLI doesn't need to track it
    between `kaggle start` and `kaggle poll`."""
    root = repo_root if repo_root is not None else config_mod.REPO_ROOT
    return Path(root) / ".kaggle_work" / job_id


def kernel_id_for(job_id: str, owner: str) -> str:
    """Deterministic from job_id + owner, for the same reason as
    default_work_dir(): `kaggle poll` can recompute it without a sidecar
    file recording what `kaggle start` used."""
    return f"{owner}/ftplatform-job-{job_id}"


def _claim_job(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                 (_now(), job_id))
    conn.commit()


def start_job_on_kaggle(conn: sqlite3.Connection, customer_id: str, job_id: str,
                         owner: str, work_dir: Path, repo_root: Path | None = None,
                         required_hours: float | None = None,
                         run=subprocess.run) -> dict:
    """Exports the job as a packet, uploads it as a private Kaggle dataset,
    and pushes the kernel that will run it. Returns
    {"kernel_id", "dataset_id", "work_dir"}. Does not wait for the run to
    finish -- call poll_and_finish_job() afterwards, as many times as needed.

    `required_hours`, when given, runs kaggle_ops.preflight_check() before
    the job is claimed or anything is pushed -- so a quota/concurrency
    failure leaves the job exactly as pending as it was, not marked
    'running' for a push that never happened. None (the default) skips the
    check, unchanged from before this guard existed."""
    work_dir = Path(work_dir)
    packet_dir = work_dir / "packet"
    kernel_dir = work_dir / "kernel"
    kernel_id = kernel_id_for(job_id, owner)

    if required_hours is not None:
        kaggle_ops.preflight_check(kernel_id, required_hours, run=run)

    packet.export_job_packet(conn, customer_id, job_id, packet_dir, repo_root=repo_root)

    slug = f"ftplatform-job-{job_id}"
    dataset_id = kaggle_ops.upload_packet_dataset(
        packet_dir, owner, slug, title=f"ftplatform job {job_id} ({customer_id})", run=run)

    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(KERNEL_SOURCE_DIR / KERNEL_CODE_FILE, kernel_dir / KERNEL_CODE_FILE)
    kaggle_ops.write_kernel_metadata(
        kernel_dir, owner, slug, title=f"ftplatform job {job_id} ({customer_id})",
        code_file=KERNEL_CODE_FILE, dataset_id=dataset_id)
    kaggle_ops.push_kernel(kernel_dir, run=run)

    _claim_job(conn, job_id)
    return {"kernel_id": kernel_id, "dataset_id": dataset_id,
            "work_dir": str(work_dir)}


def poll_and_finish_job(conn: sqlite3.Connection, customer_id: str, job_id: str,
                         kernel_id: str, work_dir: Path, repo_root: Path | None = None,
                         run=subprocess.run) -> dict | None:
    """Checks the kernel's status once. Returns None if it's still running
    (call again later); otherwise downloads the result, merges it in via
    packet.import_job_result(), and returns the finished job. A Kaggle-side
    error or cancellation is recorded as a failed job with that status in
    the error message, not silently dropped."""
    status = kaggle_ops.kernel_status(kernel_id, run=run)
    if status not in kaggle_ops.DONE_STATUSES:
        return None

    work_dir = Path(work_dir)
    output_dir = work_dir / "output"
    kaggle_ops.download_kernel_output(kernel_id, output_dir, run=run)

    if status != "complete":
        conn.execute(
            "UPDATE jobs SET status = 'failed', "
            "result_json = json_object('error', ?), finished_at = ? WHERE id = ?",
            (f"kaggle kernel {kernel_id} ended with status {status!r}", _now(), job_id))
        conn.commit()
        from ftplatform.jobs import queue
        return queue.get(conn, job_id)

    result_packet_dir = output_dir / "result_packet"
    return packet.import_job_result(conn, customer_id, job_id, result_packet_dir,
                                     repo_root=repo_root)
