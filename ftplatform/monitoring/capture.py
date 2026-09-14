"""
Phase 6 -- capturing production traffic for later review.

Wires `ServerState.on_response` (the one additive hook `serve.py` gained for
this) to append one row per request to memory/production_log.jsonl.

There is no general-purpose text-redaction primitive in this codebase --
`ftspec.core.redaction` only *detects* PII/PHI patterns for auditing
generated output, it does not mask arbitrary free text, and inventing an
unverified masking pass here is worse than not capturing at all. So for a
profile with a real compliance regime (`allows_external_api=False`
verticals: fintech, healthcare), raw input/output text is never written to
disk here -- only structured metadata (validity, tokens, latency). For an
unregulated profile (`regime == "none"`, e.g. saas_support) there is no such
constraint, so the full text is captured, since it is the only thing
selfimprove/label.py can actually build a correction from.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ftspec.run import get_logger

log = get_logger("ftplatform.monitoring.capture")


def attach(ctx) -> None:
    """Point `ftspec.serving.serve.STATE.on_response` at this customer's
    production_log.jsonl. Call once, after `load_engine()` / before serving
    starts handling requests -- see `ftplatform serve`."""
    from ftspec.serving import serve as serve_mod

    log_dir = ctx.memory_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "production_log.jsonl"
    capture_text = ctx.profile.compliance.regime == "none"

    def on_response(request, text, prompt_tokens, completion_tokens, elapsed_ms):
        user_message = next((m.content for m in request.messages if m.role == "user"), "")
        record, _reason = ctx.profile.contract.validate(text)
        row = {
            "request_id": uuid.uuid4().hex[:16],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "contract_valid": record is not None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "input": user_message if capture_text else None,
            "output": (record if record is not None else text) if capture_text else None,
            "text_captured": capture_text,
        }
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("failed to write production log row for %s", ctx.customer.id)

    serve_mod.STATE.on_response = on_response
    log.info("production capture attached for customer %s -> %s (text_captured=%s)",
              ctx.customer.id, out_path, capture_text)
