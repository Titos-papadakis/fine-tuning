"""
Composes several `ServerState.on_response` hooks (see ftspec.serving.serve)
into one callable -- there is only one hook slot, but
ftplatform.monitoring.capture (production log, for self-improve) and
ftplatform.billing.usage (per-customer metering) each want it
independently, and a real deployment usually wants both at once.
"""
from __future__ import annotations

from ftspec.run import get_logger

log = get_logger("ftplatform.serving.hooks")


def compose(*hook_fns):
    """Returns one on_response callable that calls every hook in
    `hook_fns`, in order. Each hook's failure is caught and logged
    individually here, so one broken hook can't prevent the others from
    running for the same request -- serve.py's own try/except around
    on_response only guards the *whole* call, so without this a failure in
    the first composed hook would silently skip every hook after it."""
    def on_response(request, text, prompt_tokens, completion_tokens, elapsed_ms,
                     customer_id=None, cache_hit=False):
        for hook_fn in hook_fns:
            try:
                hook_fn(request, text, prompt_tokens, completion_tokens, elapsed_ms,
                        customer_id, cache_hit)
            except Exception:                                          # noqa: BLE001
                log.exception("one composed on_response hook failed; continuing with the rest")

    return on_response
