"""
`ServerState.on_response` (ftspec/serving/serve.py) -- the one additive hook
ftplatform's production-traffic capture leans on. Exercised over real HTTP
through the mock backend (the same fixture pattern test_edge_cases.py uses
for the streaming-accounting tests), on both the streaming and non-streaming
code paths, since they each call the hook from a different place and
event_stream()'s `text` variable has its own initialization edge case (see
serve.py's `text = ""` guard).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def mock_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ftspec.core.registry import load_profile
    from ftspec.serving import serve as S

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("sa_hook", root / "scripts" / "serve_adapter.py")
    sa = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sa
    spec.loader.exec_module(sa)

    saved = dict(S.STATE.__dict__)
    profile = load_profile("saas_support")
    sa.configure_state(profile, "mock", sa.build_parser().parse_args(["--backend", "mock"]))
    yield TestClient(S.create_app(False)), S
    S.STATE.__dict__.clear()
    S.STATE.__dict__.update(saved)


def test_default_on_response_is_none_and_nothing_breaks(mock_client):
    client, S = mock_client
    assert S.STATE.on_response is None
    response = client.post("/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200


def test_on_response_fires_for_a_non_streaming_request(mock_client):
    client, S = mock_client
    calls = []
    S.STATE.on_response = lambda *args: calls.append(args)

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert len(calls) == 1
    req, text, p_tok, c_tok, elapsed_ms, customer_id = calls[0]
    assert req.messages[0].content == "hi"
    assert isinstance(text, str) and text
    assert p_tok > 0 and c_tok > 0
    assert elapsed_ms >= 0
    assert customer_id is None  # no auth configured in this fixture


def test_on_response_fires_for_a_streaming_request_with_the_full_text(mock_client):
    client, S = mock_client
    calls = []
    S.STATE.on_response = lambda *args: calls.append(args)

    with client.stream("POST", "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        b"".join(r.iter_bytes())

    assert len(calls) == 1
    _req, text, p_tok, c_tok, elapsed_ms, customer_id = calls[0]
    assert isinstance(text, str) and text  # the full accumulated text, not just the last delta
    # Documented gap: streaming never computes token counts today.
    assert p_tok == 0 and c_tok == 0
    assert elapsed_ms >= 0
    assert customer_id is None  # no auth configured in this fixture


def test_on_response_fires_once_per_request_not_once_per_chunk(mock_client):
    client, S = mock_client
    calls = []
    S.STATE.on_response = lambda *args: calls.append(args)

    for _ in range(3):
        with client.stream("POST", "/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
            b"".join(r.iter_bytes())

    assert len(calls) == 3


def test_a_raising_on_response_does_not_break_the_response_itself(mock_client):
    """The hook is a side channel for ftplatform; a bug in it (a full disk,
    say) must never take an actual served response down with it."""
    client, S = mock_client

    def boom(*args):
        raise RuntimeError("disk full")

    S.STATE.on_response = boom
    response = client.post("/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200


def test_a_raising_on_response_does_not_break_a_stream_either(mock_client):
    client, S = mock_client
    S.STATE.on_response = lambda *args: (_ for _ in ()).throw(RuntimeError("disk full"))

    with client.stream("POST", "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        body = b"".join(r.iter_bytes())
    assert b"[DONE]" in body
