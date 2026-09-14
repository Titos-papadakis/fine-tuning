"""
Phase 6 -- the human-reviewed correction workflow.

review_queue.py's heuristics only flag rows *worth a look*; they never
confirm a value was actually wrong. This module is where a human makes that
call, one row at a time: either the flagged output was fine after all
(`skip`), or it wasn't and the correct value is supplied (`record_correction`,
which is the only thing that actually produces new training data). Kept
deliberately low-throughput and file-based -- error_queue.jsonl should never
hold more than a handful of rows at once, so no database is warranted here.
"""
from __future__ import annotations

import json


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pending(ctx) -> list[dict]:
    """Every row still awaiting review, oldest first."""
    return _read_jsonl(ctx.memory_dir() / "error_queue.jsonl")


def _pop(ctx, request_id: str) -> tuple[dict, list[dict]]:
    rows = pending(ctx)
    matches = [r for r in rows if r["request_id"] == request_id]
    if not matches:
        raise ValueError(f"request_id {request_id!r} is not in the review queue")
    remaining = [r for r in rows if r["request_id"] != request_id]
    return matches[0], remaining


def skip(ctx, request_id: str) -> dict:
    """The flagged output was actually fine -- drop it, no correction
    produced. Returns the row that was skipped."""
    row, remaining = _pop(ctx, request_id)
    _write_jsonl(ctx.memory_dir() / "error_queue.jsonl", remaining)
    return row


def record_correction(ctx, request_id: str, corrected_output: dict) -> dict:
    """Confirm the flagged output was wrong and supply the right one.

    Appends to memory/corrections.jsonl in the `{"input", "output"}` shape
    `ftspec.data.build.load_external()` already accepts -- reused unmodified
    by `dataset_update.fold_corrections()`. Returns the row that was
    corrected.
    """
    row, remaining = _pop(ctx, request_id)
    if row.get("input") is None:
        raise ValueError(
            f"request_id {request_id!r} has no captured input text (a regulated "
            f"profile's traffic is never captured with text -- see monitoring/capture.py). "
            f"Cannot build a training example from it.")
    _write_jsonl(ctx.memory_dir() / "error_queue.jsonl", remaining)

    corrections_path = ctx.memory_dir() / "corrections.jsonl"
    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corrections_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"input": row["input"], "output": corrected_output},
                            ensure_ascii=False) + "\n")
    return row
