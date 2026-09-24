"""
Local-side wrapper around the `kaggle` CLI (v2.2.4, the current token-auth
rewrite -- not the older kaggle.json/`kaggle datasets create -u` API):
uploads a job packet as a private dataset, then pushes/polls/downloads the
kernel that runs it.

Honest caveat: `kernels status`'s plain-text output format was inspected via
--help only (no live Kaggle account was available while writing this), so
`_parse_kernel_status()` is a best-effort keyword match, not a verified
parser -- expect it may need a fix after the first real run. Everything else
here mirrors the CLI's own --help output exactly.

Every function takes `run=subprocess.run`, overridable in tests -- the same
dependency-injection style already used for `ftspec` calls elsewhere in this
package (see candidates/runner.py's module-level imports of `build`,
`validate_mod`, etc.), just for an external process instead of a function.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

POLL_INTERVAL_S = 30
DONE_STATUSES = {"complete", "error", "cancelled", "cancelling"}
# A kernel in either of these states already has a live GPU session. Pushing
# on top of one does NOT cancel it -- confirmed the hard way running this
# project's own saas_support baseline: a push meant to replace a stuck run
# instead started a second, concurrent one, and together they drained a
# week's GPU quota (25.93 of 30h) before either finished.
RUNNING_LIKE_STATUSES = {"running", "queued"}


class KaggleCommandError(RuntimeError):
    pass


class InsufficientQuotaError(KaggleCommandError):
    pass


class ConcurrentSessionError(KaggleCommandError):
    pass


def _kaggle(args: list[str], run=subprocess.run) -> subprocess.CompletedProcess:
    result = run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise KaggleCommandError(
            f"kaggle {' '.join(args)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}")
    return result


def write_dataset_metadata(packet_dir: Path, owner: str, slug: str, title: str) -> Path:
    """`kaggle datasets create/version` require a dataset-metadata.json
    (singular -- confirmed against a real account; `datasets-metadata.json`
    fails with "Metadata file not found") inside the upload folder naming
    the dataset -- written here rather than left to `kaggle datasets init`
    so the id is exactly the one we're about to poll for, not whatever init
    guesses from the folder name."""
    packet_dir = Path(packet_dir)
    meta_path = packet_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps({
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "unknown"}],
    }, indent=2), encoding="utf-8")
    return meta_path


def dataset_exists(owner: str, slug: str, run=subprocess.run) -> bool:
    result = run([sys.executable, "-m", "kaggle", "datasets", "status", f"{owner}/{slug}"],
                 capture_output=True, text=True)
    return result.returncode == 0


def upload_packet_dataset(packet_dir: Path, owner: str, slug: str, title: str,
                           run=subprocess.run) -> str:
    """Uploads `packet_dir` as a private dataset `owner/slug` -- created on
    the first call, versioned on every call after (a job packet's customer
    data can change between runs for the same customer). Returns the
    dataset id."""
    packet_dir = Path(packet_dir)
    write_dataset_metadata(packet_dir, owner, slug, title)
    if dataset_exists(owner, slug, run=run):
        _kaggle(["datasets", "version", "-p", str(packet_dir), "-m", "job update",
                 "--dir-mode", "zip", "-q"], run=run)
    else:
        _kaggle(["datasets", "create", "-p", str(packet_dir), "--dir-mode", "zip", "-q"], run=run)
    return f"{owner}/{slug}"


def write_kernel_metadata(kernel_dir: Path, owner: str, slug: str, title: str,
                           code_file: str, dataset_id: str) -> Path:
    """Points the kernel this job runs from at the just-uploaded dataset, so
    each job gets its own kernel-metadata.json rather than a hand-edited
    shared one (see kernel-metadata.json's KAGGLE_USERNAME placeholder for
    the one-off demo run, which this supersedes for per-customer jobs)."""
    kernel_dir = Path(kernel_dir)
    meta_path = kernel_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps({
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")
    return meta_path


def push_kernel(kernel_dir: Path, run=subprocess.run) -> None:
    _kaggle(["kernels", "push", "-p", str(kernel_dir)], run=run)


_QUOTA_ROW = re.compile(r"^(GPU|TPU)\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h\s+(\S+)", re.MULTILINE)


def get_gpu_quota(run=subprocess.run) -> dict:
    """Parses `kaggle quota`'s GPU row: {"used", "remaining", "total"} in
    hours (float) plus "refresh_at" (the ISO timestamp Kaggle prints, passed
    through as-is -- not parsed as a datetime since nothing here needs to do
    arithmetic on it, only show it in a message)."""
    result = _kaggle(["quota"], run=run)
    for resource, used, remaining, total, refresh_at in _QUOTA_ROW.findall(result.stdout):
        if resource == "GPU":
            return {"used": float(used), "remaining": float(remaining),
                     "total": float(total), "refresh_at": refresh_at}
    raise KaggleCommandError(f"could not find a GPU row in `kaggle quota` output:\n{result.stdout}")


def preflight_check(kernel_id: str, required_hours: float | None = None,
                     run=subprocess.run) -> None:
    """Raise before a push that would repeat either half of the same real
    failure: pushing a run that can't fit in what's left of the weekly GPU
    quota, or pushing a second session on top of one already running (see
    RUNNING_LIKE_STATUSES). The quota check is skipped when `required_hours`
    is None -- callers who don't know their run's expected length yet still
    get the concurrency check, which needs no estimate to be worth doing."""
    if required_hours is not None:
        quota = get_gpu_quota(run=run)
        if quota["remaining"] < required_hours:
            raise InsufficientQuotaError(
                f"only {quota['remaining']:.2f}h GPU quota remain (need ~{required_hours:.2f}h); "
                f"resets at {quota['refresh_at']}")

    status = kernel_status(kernel_id, run=run)
    if status in RUNNING_LIKE_STATUSES:
        raise ConcurrentSessionError(
            f"{kernel_id} already has status {status!r} -- pushing now would start a "
            f"second, concurrent session instead of replacing it, doubling GPU-hour "
            f"burn. Stop the existing session first.")


def safe_push_kernel(kernel_dir: Path, kernel_id: str, required_hours: float | None = None,
                      run=subprocess.run) -> None:
    """push_kernel(), guarded by preflight_check(). Raises instead of pushing
    when the check fails -- callers that need the caller-claims-the-job
    ordering of the old push_kernel() (see run_on_kaggle.start_job_on_kaggle)
    should call preflight_check() explicitly before claiming, then push_kernel()."""
    preflight_check(kernel_id, required_hours, run=run)
    push_kernel(kernel_dir, run=run)


def _parse_kernel_status(raw_stdout: str) -> str:
    """Best-effort: see this module's docstring. Falls back to "running" for
    anything unrecognized rather than raising, so a wording change on
    Kaggle's side degrades to "keep polling" instead of crashing the loop.

    Confirmed against a real account: Kaggle's actual wording is
    "KernelWorkerStatus.CANCEL_ACKNOWLEDGED", not "cancelled"/"cancelling" --
    neither matched, so this used to fall back to "running" and (with
    preflight_check() added later) wrongly refuse to push over a kernel that
    had, in fact, already stopped. The "cancel" catch-all below covers this
    and any other CANCEL_* wording the same way.
    """
    text = raw_stdout.lower()
    for status in ("error", "cancelled", "cancelling", "complete", "running", "queued"):
        if status in text:
            return status
    if "cancel" in text:
        return "cancelled"
    return "running"


def kernel_status(kernel_id: str, run=subprocess.run) -> str:
    """"not_found" for a kernel that has never been pushed -- confirmed
    against a real account: Kaggle reports this as a 'kernels.get' permission
    error, not a 404, presumably to avoid leaking whether a slug belongs to
    someone else's private kernel. Distinguished from a genuine failure by
    that specific wording; matters because preflight_check()'s concurrency
    guard must not block a job's first-ever push just because its
    brand-new, never-pushed kernel "fails" a status check."""
    try:
        result = _kaggle(["kernels", "status", kernel_id], run=run)
    except KaggleCommandError as e:
        if "kernels.get" in str(e) and "denied" in str(e).lower():
            return "not_found"
        raise
    return _parse_kernel_status(result.stdout)


# The kaggle CLI writes a kernel's log with a bare open() -- the locale
# encoding, cp1253 on a Greek Windows box -- and crashes with
# UnicodeEncodeError on any emoji in it (Unsloth prints one every run).
# PYTHONUTF8/PYTHONIOENCODING don't reach that call; defaulting open() itself
# to UTF-8 before the CLI starts does (confirmed against a real kernel).
_UTF8_KAGGLE = (
    "import builtins, runpy, sys\n"
    "_open = builtins.open\n"
    "def _utf8_open(f, mode='r', *a, **k):\n"
    "    if 'b' not in mode and not a and 'encoding' not in k:\n"
    "        k['encoding'] = 'utf-8'\n"
    "    return _open(f, mode, *a, **k)\n"
    "builtins.open = _utf8_open\n"
    "sys.argv = ['kaggle', *sys.argv[1:]]\n"
    "runpy.run_module('kaggle', run_name='__main__', alter_sys=True)\n"
)


def download_kernel_output(kernel_id: str, out_dir: Path, run=subprocess.run) -> Path:
    """Runs the CLI with open() defaulted to UTF-8 (see _UTF8_KAGGLE). Still
    tolerates that same crash if it somehow recurs once the files are on
    disk -- the log is the last thing written, so files present is success."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel_id, "-p", str(out_dir), "-q", "-o"]
    try:
        result = run([sys.executable, "-c", _UTF8_KAGGLE, *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise KaggleCommandError(f"kaggle {' '.join(args)} failed ({result.returncode}):\n"
                                     f"{result.stdout}\n{result.stderr}")
    except KaggleCommandError as e:
        downloaded = any(p.is_file() and not p.name.endswith(".log") for p in out_dir.rglob("*"))
        if "UnicodeEncodeError" in str(e) and downloaded:
            return out_dir
        raise
    return out_dir
