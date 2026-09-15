"""ftplatform/serving/hooks.py::compose() -- pure composition logic, no serve.py involved."""
from __future__ import annotations

from ftplatform.serving.hooks import compose


def test_compose_calls_every_hook(monkeypatch):
    calls = []
    hook_a = lambda *a: calls.append(("a", a))   # noqa: E731
    hook_b = lambda *a: calls.append(("b", a))   # noqa: E731

    on_response = compose(hook_a, hook_b)
    on_response("req", "text", 1, 2, 3.0, "acme")

    assert [c[0] for c in calls] == ["a", "b"]
    assert calls[0][1] == ("req", "text", 1, 2, 3.0, "acme")


def test_compose_passes_customer_id_through(monkeypatch):
    seen = []
    on_response = compose(lambda *a: seen.append(a[-1]))
    on_response("req", "text", 1, 2, 3.0, "globex")
    assert seen == ["globex"]


def test_compose_a_failing_hook_does_not_stop_the_rest(monkeypatch):
    calls = []

    def broken(*a):
        raise RuntimeError("boom")

    on_response = compose(broken, lambda *a: calls.append("ran"))
    on_response("req", "text", 1, 2, 3.0, None)  # must not raise

    assert calls == ["ran"]


def test_compose_with_no_hooks_is_a_harmless_noop():
    on_response = compose()
    on_response("req", "text", 1, 2, 3.0, None)  # must not raise
