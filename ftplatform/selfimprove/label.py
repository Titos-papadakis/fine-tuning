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


# --- suggested corrections -----------------------------------------------------
# Typing a whole JSON record per flagged row is what makes review slow enough
# to be skipped. The model's own output is almost always right in most fields,
# so it is offered as the starting point: the reviewer either confirms it or
# overrides just the wrong fields, and the result is checked against the
# contract *now* -- not discovered invalid later, at selfimprove time.

class InvalidCorrectionError(ValueError):
    pass


def suggestion(row: dict) -> dict | None:
    """The model's output as an editable starting record, or None if it
    produced nothing parseable (or text wasn't captured)."""
    out = row.get("output")
    if isinstance(out, dict):
        return out
    if not isinstance(out, str):
        return None
    text = out.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```"))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_edits(record: dict, edits: dict[str, str]) -> dict:
    """`edits` is {"dotted.path": value}; a value that parses as JSON (a
    number, true/null, a list) is used as that, anything else as a string."""
    import copy
    out = copy.deepcopy(record)
    for path, raw in edits.items():
        keys = path.split(".")
        node = out
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = _parse_value(raw)
    return out


def parse_set_args(items: list[str]) -> dict[str, str]:
    edits = {}
    for item in items:
        if "=" not in item:
            raise InvalidCorrectionError(f"--set expects path=value, got {item!r}")
        path, value = item.split("=", 1)
        edits[path.strip()] = value
    return edits


def _row(ctx, request_id: str) -> dict:
    for r in pending(ctx):
        if r["request_id"] == request_id:
            return r
    raise ValueError(f"request_id {request_id!r} is not in the review queue")


def correct(ctx, request_id: str, edits: dict[str, str] | None = None,
            output: dict | None = None) -> dict:
    """Record a correction built from `output` (or, if None, the model's own
    suggestion) with `edits` applied on top. Raises InvalidCorrectionError
    -- leaving the row in the queue -- if the result breaks the contract.
    With neither `edits` nor `output`, this confirms the model's output was
    right, which still yields a real-traffic training example."""
    row = _row(ctx, request_id)
    base = output if output is not None else suggestion(row)
    if base is None:
        raise InvalidCorrectionError(
            f"{request_id!r} has no parseable model output to start from -- pass --output "
            f"with the full correct record.")
    corrected = apply_edits(base, edits or {})
    record, err = ctx.profile.contract.validate(json.dumps(corrected, ensure_ascii=False))
    if record is None:
        raise InvalidCorrectionError(f"correction does not match the schema: {err}")
    return record_correction(ctx, request_id, record)
