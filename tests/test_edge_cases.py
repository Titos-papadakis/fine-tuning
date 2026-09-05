"""
Regressions from an edge-case sweep.

Each test here corresponds to a defect that was found by probing rather than by
a failing test -- the code was self-consistent and wrong. They are grouped by
origin rather than by module because that is what they have in common: all four
failed silently, in the direction of a confident incorrect answer.
"""
from __future__ import annotations

import json

import pytest

from ftspec import prompt_audit as pa
from ftspec.core import redaction as RD

RULES = tuple(RD.PATTERNS)


# --- prompt-audit: a degenerate row must not zero the headline ---------------

PREAMBLE = "You are an extraction engine. " + ("SCHEMA FIELD BLAH " * 60)


def write_log(tmp_path, rows, name="log.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def good_rows(n=30):
    return [json.dumps({"messages": [
        {"role": "system", "content": PREAMBLE},
        {"role": "user", "content": f"transcript number {i} which varies"},
        {"role": "assistant", "content": "{}"},
    ]}) for i in range(n)]


def test_empty_messages_row_does_not_destroy_the_static_prefix(tmp_path):
    # The shared prefix is the prefix common to *every* sample, so one empty
    # string collapses it to nothing. Measured before the fix: 371 tokens and
    # 99.2% static became 0 and 0.0% from a single bad row in 31 -- the audit
    # reporting a clean bill of health on a workload that is almost entirely
    # static preamble.
    clean = pa.audit(pa.parse_samples(write_log(tmp_path, good_rows())))
    assert clean.static_tokens > 300
    assert clean.static_share > 0.9

    poisoned = pa.audit(pa.parse_samples(
        write_log(tmp_path, [*good_rows(), '{"messages": []}'], "poisoned.jsonl")))
    assert poisoned.static_tokens == clean.static_tokens
    assert poisoned.n_samples == clean.n_samples      # the row is dropped, not counted


@pytest.mark.parametrize("degenerate", [
    '{"messages": []}',
    '{"messages": [{"role": "system", "content": ""}]}',
    '{"prompt": ""}',
    '{"system": "", "user": ""}',
    '{"system": "   ", "user": "  "}',
])
def test_every_shape_of_empty_row_is_dropped(tmp_path, degenerate):
    result = pa.audit(pa.parse_samples(
        write_log(tmp_path, [*good_rows(5), degenerate])))
    assert result.n_samples == 5
    assert result.static_tokens > 300


def test_a_real_outlier_is_explained_rather_than_left_as_zero(tmp_path):
    # A call that genuinely has no shared preamble is not a parse error, so the
    # headline correctly falls to zero. What must not happen is that it falls
    # silently: "0% static" reads as a clean bill of health.
    rows = [*good_rows(30), json.dumps({"messages": [
        {"role": "system", "content": "a completely different instruction"},
        {"role": "user", "content": "unrelated call"}]})]
    result = pa.audit(pa.parse_samples(write_log(tmp_path, rows)))

    assert result.static_tokens == 0
    explained = [w for w in result.warnings if "do share" in w]
    assert explained, "the collapse was not explained to the reader"
    assert "30 of 31" in explained[0]
    assert "97%" in explained[0]


def test_dominant_group_is_not_reported_when_everything_agrees(tmp_path):
    result = pa.audit(pa.parse_samples(write_log(tmp_path, good_rows())))
    assert not [w for w in result.warnings if "do share" in w]


# --- serving: streamed requests are accounted like every other request -------

@pytest.fixture
def streaming_client():
    pytest.importorskip("fastapi")
    import importlib.util
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    from ftspec.core.registry import load_profile
    from ftspec.serving import serve as S

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("sa_edge", root / "scripts" / "serve_adapter.py")
    sa = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sa
    spec.loader.exec_module(sa)

    saved = dict(S.STATE.__dict__)
    profile = load_profile("healthcare_clinical")     # counters matter most here
    sa.configure_state(profile, "mock", sa.build_parser().parse_args(["--backend", "mock"]))
    S.STATE.served_requests = 0
    S.STATE.constrained_requests = 0
    S.STATE.latencies_ms.clear()
    yield TestClient(S.create_app(False)), S
    S.STATE.__dict__.clear()
    S.STATE.__dict__.update(saved)


def stream_once(client, **body):
    payload = {"messages": [{"role": "user", "content": "Patient: chest pain"}],
               "stream": True, **body}
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        b"".join(response.iter_bytes())


def test_streaming_requests_are_counted_as_constrained(streaming_client):
    # Before the fix /stats reported 5 of 5 streamed requests as unconstrained
    # although no client had opted out. On a regulated profile that is a false
    # compliance alarm raised by the endpoint an auditor would actually read.
    client, S = streaming_client
    for _ in range(5):
        stream_once(client)

    stats = client.get("/stats").json()
    assert stats["served_requests"] == 5
    assert stats["constrained_requests"] == 5
    assert stats["unconstrained_requests"] == 0


def test_streaming_requests_appear_in_the_latency_window(streaming_client):
    # Streaming was excluded from p50/p99 entirely -- the one workload where
    # latency is the reason the customer chose streaming.
    client, _ = streaming_client
    for _ in range(3):
        stream_once(client)

    latency = client.get("/health").json()["latency_ms"]
    assert latency["count"] == 3
    assert latency["p50"] > 0


def test_a_genuine_streamed_opt_out_is_still_recorded(streaming_client):
    client, _ = streaming_client
    stream_once(client)
    stream_once(client, ftspec_unconstrained=True)

    stats = client.get("/stats").json()
    assert stats["constrained_requests"] == 1
    assert stats["unconstrained_requests"] == 1


def test_streaming_and_non_streaming_agree_on_accounting(streaming_client):
    client, _ = streaming_client
    client.post("/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Patient: pain"}]})
    stream_once(client)

    stats = client.get("/stats").json()
    assert stats["served_requests"] == 2
    assert stats["constrained_requests"] == 2
    assert client.get("/health").json()["latency_ms"]["count"] == 2


# --- serving: sampling parameters are bounded like the API they mimic --------

@pytest.mark.parametrize("field,value,ok", [
    ("temperature", -5.0, False),
    ("temperature", 999.0, False),
    ("temperature", 0.0, True),
    ("temperature", 2.0, True),
    ("top_p", 0.0, False),
    ("top_p", 1.5, False),
    ("top_p", 1.0, True),
    ("max_tokens", 0, False),
    ("max_tokens", 4097, False),
    ("max_tokens", 4096, True),
])
def test_sampling_parameters_are_validated_at_the_edge(streaming_client, field, value, ok):
    # Unbounded, a negative temperature reached vLLM and returned a 500. A 422
    # naming the field is the difference between a bug report and a shrug.
    client, _ = streaming_client
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Patient: pain"}], field: value})
    assert (response.status_code == 200) is ok, response.text


# --- redaction: the formats humans actually write ----------------------------

@pytest.mark.parametrize("rule,text", [
    ("iban", "transfer to GB82 WEST 1234 5698 7654 32 please"),   # grouped in fours
    ("iban", "transfer to GB82WEST12345698765432"),               # machine form
    ("phone", "call 555.123.4567"),                               # dot separated
    ("phone", "call (555) 123-4567"),
    ("phone", "call 555-123-4567"),
    ("phone", "call +1 555 123 4567"),
    ("dob", "born 03/14/1985"),                                   # US
    ("dob", "born 14/03/1985"),                                   # EU
    ("dob", "born 1985/03/14"),
    ("dob", "born 1985-03-14"),                                   # ISO
    ("ssn", "SSN 123-45-6789"),
    ("pan", "card 4539 1488 0343 6467"),
])
def test_identifier_formats_are_detected(rule, text):
    hits = {f.rule for f in RD.scan(text, RULES)}
    assert rule in hits, f"{rule} missed in {text!r} (matched: {hits or 'nothing'})"


@pytest.mark.parametrize("text", [
    "order ORD-55210",
    "total was 412.50 USD",
    "card ending 4142",
    "BIN 411111",
    "internal case CASE-4471",
    "IN 2024 THE COMPANY REPORTED RECORD REVENUE GROWTH",   # all-caps prose vs IBAN
    "a 3/4 majority agreed",                                # ratio vs date
    "priced at 12.50/1.00",
    "running v1.2.3.4 in production",
    "4539148803436460",                                     # PAN-shaped, fails Luhn
    "4111111111111111",                                     # reserved test PAN
])
def test_broadened_patterns_do_not_fire_on_ordinary_text(text):
    # Widening a compliance scanner is only safe if it stays quiet on the things
    # these corpora are full of. A noisy scanner gets switched off.
    assert RD.scan(text, RULES) == []
