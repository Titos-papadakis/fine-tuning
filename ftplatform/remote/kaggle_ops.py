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
import subprocess
import sys
from pathlib import Path

POLL_INTERVAL_S = 30
DONE_STATUSES = {"complete", "error", "cancelled", "cancelling"}


class KaggleCommandError(RuntimeError):
    pass


def _kaggle(args: list[str], run=subprocess.run) -> subprocess.CompletedProcess:
    result = run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise KaggleCommandError(
            f"kaggle {' '.join(args)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}")
    return result


def write_dataset_metadata(packet_dir: Path, owner: str, slug: str, title: str) -> Path:
    """`kaggle datasets create/version` require a datasets-metadata.json
    inside the upload folder naming the dataset -- written here rather than
    left to `kaggle datasets init` so the id is exactly the one we're about
    to poll for, not whatever init guesses from the folder name."""
    packet_dir = Path(packet_dir)
    meta_path = packet_dir / "datasets-metadata.json"
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


def _parse_kernel_status(raw_stdout: str) -> str:
    """Best-effort: see this module's docstring. Falls back to "running" for
    anything unrecognized rather than raising, so a wording change on
    Kaggle's side degrades to "keep polling" instead of crashing the loop."""
    text = raw_stdout.lower()
    for status in ("error", "cancelled", "cancelling", "complete", "running", "queued"):
        if status in text:
            return status
    return "running"


def kernel_status(kernel_id: str, run=subprocess.run) -> str:
    result = _kaggle(["kernels", "status", kernel_id], run=run)
    return _parse_kernel_status(result.stdout)


def download_kernel_output(kernel_id: str, out_dir: Path, run=subprocess.run) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _kaggle(["kernels", "output", kernel_id, "-p", str(out_dir), "-q", "-o"], run=run)
    return out_dir
