"""
Run manifests and logging.

Every stage that produces an artefact writes a manifest beside it recording
what produced it: git commit, dirty-tree flag, config fingerprint, package
versions, GPU, timestamps and the resulting metrics.

The reason is narrow and practical. Six weeks after a pilot, a customer asks
why the numbers in the report differ from the ones you just reproduced. Without
a manifest that question is unanswerable, and an unanswerable question about
your own benchmark is worse than a bad number.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # These are chatty at INFO and drown out our own progress lines.
    for noisy in ("httpx", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "trl", "peft", "unsloth", "vllm", "outlines", "pydantic"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def gpu_info() -> dict | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name,
                "vram_gb": round(props.total_memory / 1024 ** 3, 1),
                "count": torch.cuda.device_count()}
    except Exception:
        return None


@dataclass
class RunManifest:
    stage: str
    config_fingerprint: str | None = None
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    git_commit: str | None = None
    git_dirty: bool = False
    versions: dict = field(default_factory=dict)
    gpu: dict | None = None
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@contextmanager
def run_manifest(stage: str, out_path: Path, config_fingerprint: str | None = None,
                  params: dict | None = None):
    """Wrap a pipeline stage; always writes a manifest, success or failure."""
    manifest = RunManifest(
        stage=stage,
        config_fingerprint=config_fingerprint,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        versions=package_versions(),
        gpu=gpu_info(),
        params=params or {},
    )
    start = time.perf_counter()
    try:
        yield manifest
    finally:
        manifest.duration_s = round(time.perf_counter() - start, 2)
        manifest.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
        get_logger("ftspec.run").info("manifest written -> %s", out_path)
