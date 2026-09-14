"""
Phase 6 -- a heuristic pre-filter over captured production traffic.

Fully automatic error detection without ground truth is not solvable in
general: none of these heuristics confirm the *value* a request extracted
was wrong, only that something about it was unusual. What this produces is a
short list of *candidates* for a human to look at (selfimprove/label.py),
not confirmed errors -- "flagged" and "wrong" are different claims, and nothing
here pretends otherwise.

For a regulated profile, `monitoring.capture` never wrote the row's input/
output text to disk in the first place (see its docstring), so a flagged row
from one of those has nothing for a human to correct against beyond "this
looked unusual" -- an honest limitation, not a bug.
"""
from __future__ import annotations

import json
import statistics

from ftspec.run import get_logger

log = get_logger("ftplatform.monitoring.review_queue")

DEFAULT_LATENCY_OUTLIER_STDEVS = 3.0


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def scan(ctx, latency_outlier_stdevs: float = DEFAULT_LATENCY_OUTLIER_STDEVS) -> list[dict]:
    """Read production_log.jsonl, flag rows worth a human look, append the
    newly-flagged ones (deduplicated against whatever is already pending) to
    error_queue.jsonl, and return just those new rows."""
    rows = _read_jsonl(ctx.memory_dir() / "production_log.jsonl")
    if not rows:
        return []

    valid_latencies = [r["latency_ms"] for r in rows if r.get("contract_valid")]
    mean = statistics.mean(valid_latencies) if valid_latencies else 0.0
    stdev = statistics.pstdev(valid_latencies) if len(valid_latencies) > 1 else 0.0
    threshold = mean + latency_outlier_stdevs * stdev

    flagged = []
    for r in rows:
        reasons = []
        if not r.get("contract_valid", True):
            reasons.append("schema_invalid")
        if stdev > 0 and r["latency_ms"] > threshold:
            reasons.append("latency_outlier")
        if reasons:
            flagged.append({**r, "reasons": reasons})

    if not flagged:
        return []

    existing_ids = {r["request_id"] for r in _read_jsonl(ctx.memory_dir() / "error_queue.jsonl")}
    new_rows = [r for r in flagged if r["request_id"] not in existing_ids]
    if new_rows:
        queue_path = ctx.memory_dir() / "error_queue.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_path, "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("customer %s: flagged %d new row(s) for review (%d already pending)",
                  ctx.customer.id, len(new_rows), len(existing_ids))
    return new_rows
