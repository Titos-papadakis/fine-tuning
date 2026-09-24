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


def _write_row(ctx, request, text, prompt_tokens, completion_tokens, elapsed_ms) -> None:
    log_dir = ctx.memory_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "production_log.jsonl"
    capture_text = ctx.profile.compliance.regime == "none"

    user_message = next((m.content for m in request.messages if m.role == "user"), "")
    record, _reason = ctx.profile.contract.validate(text)
    if capture_text:
        # Validated first, redacted after: validity reflects what the model
        # actually returned, while what reaches disk never holds the
        # identifiers the customer's policy names.
        from ftplatform import privacy
        from ftspec.core.redaction import redact, redact_value
        rules = privacy.load(ctx)["redact"]
        user_message = redact(user_message, rules)
        text = redact(text, rules)
        record = redact_value(record, rules) if record is not None else None
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


def build_hook(ctx):
    """The raw on_response-shaped callable attach() installs -- exposed
    separately so a caller that also wants ftplatform.billing.usage's hook
    (or any other) can combine them with ftplatform.serving.hooks.compose()
    instead of the two stomping on ServerState.on_response in turn."""
    def on_response(request, text, prompt_tokens, completion_tokens, elapsed_ms,
                     customer_id=None, cache_hit=False):
        _write_row(ctx, request, text, prompt_tokens, completion_tokens, elapsed_ms)
    return on_response


def attach(ctx) -> None:
    """Point `ftspec.serving.serve.STATE.on_response` at this customer's
    production_log.jsonl. Call once, after `load_engine()` / before serving
    starts handling requests -- see `ftplatform serve`. For a single-
    customer deployment, so `ctx` is fixed for every request regardless of
    the `customer_id` on_response's 6th argument -- serve-shared's several
    customers on one process need attach_shared() below instead. If you
    also want usage metering on the same server, use build_hook() with
    ftplatform.serving.hooks.compose() instead of calling this."""
    from ftspec.serving import serve as serve_mod

    serve_mod.STATE.on_response = build_hook(ctx)
    log.info("production capture attached for customer %s -> %s (text_captured=%s)",
              ctx.customer.id, ctx.memory_dir() / "production_log.jsonl",
              ctx.profile.compliance.regime == "none")


def build_hook_shared(conn):
    """The raw on_response-shaped callable attach_shared() installs -- see
    build_hook()'s docstring for why this is exposed separately. Requires
    ftspec.serving.serve.STATE.api_key_resolver to already be set (auth
    on) -- without it there is no reliable per-request customer identity in
    a shared, multi-adapter server, and guessing would risk writing one
    customer's traffic into another's log, so this refuses to build a hook
    rather than silently doing that."""
    from ftplatform.customers.context import CustomerContext
    from ftspec.serving import serve as serve_mod

    if serve_mod.STATE.api_key_resolver is None:
        raise RuntimeError(
            "build_hook_shared() requires auth to be enabled (STATE.api_key_resolver set) -- "
            "without it there is no reliable per-request customer identity to log against.")

    ctx_cache: dict[str, object] = {}

    def on_response(request, text, prompt_tokens, completion_tokens, elapsed_ms,
                     customer_id=None, cache_hit=False):
        if customer_id is None:
            log.warning("production capture: no resolved customer_id for this request, "
                        "skipping rather than guessing where to log it")
            return
        ctx = ctx_cache.get(customer_id)
        if ctx is None:
            ctx = CustomerContext(conn, customer_id)
            ctx_cache[customer_id] = ctx
        _write_row(ctx, request, text, prompt_tokens, completion_tokens, elapsed_ms)

    return on_response


def attach_shared(conn) -> None:
    """Point `ftspec.serving.serve.STATE.on_response` at build_hook_shared()'s
    per-customer dispatch. See attach()'s docstring for combining this with
    usage metering instead of calling this directly."""
    from ftspec.serving import serve as serve_mod

    serve_mod.STATE.on_response = build_hook_shared(conn)
    log.info("shared production capture attached (per-customer, resolved from auth)")
