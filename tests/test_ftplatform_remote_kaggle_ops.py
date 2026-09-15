"""
ftplatform.remote.kaggle_ops: the thin wrapper around the `kaggle` CLI.
Every test injects a fake `run` in place of subprocess.run, so nothing here
actually calls Kaggle -- these tests verify our own command construction and
result parsing, not the CLI's real behavior (which was only checked via
--help while writing this; see the module docstring).
"""
from __future__ import annotations

import json
import subprocess

import pytest

from ftplatform.remote import kaggle_ops


class FakeRun:
    """Records every invocation; dispatches a canned (returncode, stdout)
    per leading-subcommand prefix, defaulting to success with empty output."""

    def __init__(self, responses: dict[tuple, tuple] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, **kwargs):
        self.calls.append(cmd)
        for prefix, (returncode, stdout) in self.responses.items():
            if tuple(cmd[-len(prefix):]) == prefix or tuple(cmd[2:2 + len(prefix)]) == prefix:
                return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_write_dataset_metadata_writes_the_expected_id_and_title(tmp_path):
    kaggle_ops.write_dataset_metadata(tmp_path, "alice", "job-1", "job 1 title")

    meta = json.loads((tmp_path / "datasets-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "alice/job-1"
    assert meta["title"] == "job 1 title"


def test_dataset_exists_true_on_returncode_zero(tmp_path):
    fake = FakeRun({("datasets", "status", "alice/job-1"): (0, "ready")})
    assert kaggle_ops.dataset_exists("alice", "job-1", run=fake) is True


def test_dataset_exists_false_on_nonzero_returncode(tmp_path):
    fake = FakeRun({("datasets", "status", "alice/job-1"): (1, "")})
    assert kaggle_ops.dataset_exists("alice", "job-1", run=fake) is False


def test_upload_packet_dataset_creates_when_the_dataset_is_new(tmp_path):
    fake = FakeRun({("datasets", "status", "alice/job-1"): (1, "not found")})

    dataset_id = kaggle_ops.upload_packet_dataset(tmp_path, "alice", "job-1", "title", run=fake)

    assert dataset_id == "alice/job-1"
    create_calls = [c for c in fake.calls if "create" in c]
    version_calls = [c for c in fake.calls if "version" in c]
    assert len(create_calls) == 1
    assert len(version_calls) == 0


def test_upload_packet_dataset_versions_when_the_dataset_already_exists(tmp_path):
    fake = FakeRun({("datasets", "status", "alice/job-1"): (0, "ready")})

    kaggle_ops.upload_packet_dataset(tmp_path, "alice", "job-1", "title", run=fake)

    create_calls = [c for c in fake.calls if c[3:5] == ["datasets", "create"]]
    version_calls = [c for c in fake.calls if c[3:5] == ["datasets", "version"]]
    assert len(create_calls) == 0
    assert len(version_calls) == 1


def test_write_kernel_metadata_points_at_the_uploaded_dataset(tmp_path):
    kaggle_ops.write_kernel_metadata(tmp_path, "alice", "job-1", "title", "job.py",
                                      "alice/job-1-data")

    meta = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "alice/job-1"
    assert meta["code_file"] == "job.py"
    assert meta["dataset_sources"] == ["alice/job-1-data"]
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True


def test_push_kernel_invokes_kernels_push(tmp_path):
    fake = FakeRun()
    kaggle_ops.push_kernel(tmp_path, run=fake)
    assert any("push" in c and str(tmp_path) in c for c in fake.calls)


@pytest.mark.parametrize("stdout,expected", [
    ("Kernel has status \"complete\"", "complete"),
    ("Kernel has status \"running\"", "running"),
    ("Kernel has status \"queued\"", "queued"),
    ("Kernel has status \"error\"", "error"),
    ("Kernel has status \"cancelled\"", "cancelled"),
    ("something unrecognized entirely", "running"),
])
def test_parse_kernel_status(stdout, expected):
    assert kaggle_ops._parse_kernel_status(stdout) == expected


def test_kernel_status_uses_the_parser_on_real_stdout():
    fake = FakeRun({("kernels", "status", "alice/job-1"): (0, "status: complete")})
    assert kaggle_ops.kernel_status("alice/job-1", run=fake) == "complete"


def test_download_kernel_output_creates_the_output_dir(tmp_path):
    fake = FakeRun()
    out = kaggle_ops.download_kernel_output("alice/job-1", tmp_path / "out", run=fake)
    assert out.exists()
    assert any("output" in c for c in fake.calls)


def test_a_nonzero_exit_raises_kaggle_command_error(tmp_path):
    with pytest.raises(kaggle_ops.KaggleCommandError):
        kaggle_ops._kaggle(["kernels", "push", "-p", str(tmp_path)],
                            run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
