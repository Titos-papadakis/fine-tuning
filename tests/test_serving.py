"""
Serving-layer tests with a stubbed engine.

vLLM needs a GPU, so these tests replace the engine and tokenizer with stubs and
exercise everything around them: OpenAI wire compatibility, the system-prompt
injection that keeps requests on the model's training distribution, and the
schema guarantee being applied without the client asking for it.

That last property is the entire reason this server exists rather than plain
`vllm serve`, so it is asserted rather than assumed.
"""
from __future__ import annotations

import json
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ftspec.core.registry import load_profile  # noqa: E402
from ftspec.serving import serve as S  # noqa: E402

PROFILE = load_profile("saas_support")
PROMPT_SHORT = PROFILE.prompts.short

SAMPLE_OUTPUT = json.dumps({
    "ticket_summary": "Customer contacted support regarding a duplicate charge.",
    "customer": {"sentiment": "frustrated", "language": "en",
                  "tier": "enterprise", "churn_threat": True},
    "issue": {"category": "billing", "subcategory": "duplicate_charge",
               "priority": "urgent", "is_repeat_contact": True},
    "actions_taken": ["Reviewed billing history"],
    "resolution": {"status": "escalated", "requires_followup": True,
                    "followup_date": "2026-03-14"},
    "extracted_entities": {"order_id": "ORD-55210", "product_name": "NexaCRM",
                            "amount": 412.50},
})


class StubTokenizer:
    """Records the conversation it was asked to render, for prompt assertions."""

    def __init__(self):
        self.last_conversation = None

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        self.last_conversation = conversation
        return "\n".join(f"<{m['role']}>{m['content']}" for m in conversation)


class StubEngine:
    """Minimal stand-in for vLLM's AsyncLLMEngine.generate."""

    def __init__(self, text=SAMPLE_OUTPUT):
        self.text = text
        self.calls = []

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.calls.append({"prompt": prompt, "params": sampling_params,
                            "lora": lora_request})
        outputs = [types.SimpleNamespace(text=self.text, token_ids=list(range(12)))]
        yield types.SimpleNamespace(outputs=outputs, prompt_token_ids=list(range(30)))


@pytest.fixture
def client(monkeypatch):
    engine = StubEngine()
    tokenizer = StubTokenizer()

    S.STATE.engine = engine
    S.STATE.tokenizer = tokenizer
    S.STATE.profile = PROFILE
    S.STATE.schema = PROFILE.contract.json_schema()
    S.STATE.system_prompt = PROMPT_SHORT
    S.STATE.model_name = f"ftspec-{PROFILE.name}"
    S.STATE.lora_request = None
    S.STATE.served_requests = 0
    S.STATE.constrained_requests = 0

    # SamplingParams lives in vLLM, which is not installed in CI. Record the
    # arguments instead so the schema-enforcement assertions still hold.
    def fake_sampling_params(req, constrained):
        return {"constrained": constrained,
                "schema": S.STATE.schema if constrained else None,
                "max_tokens": req.max_tokens}

    monkeypatch.setattr(S, "build_sampling_params", fake_sampling_params)

    app = S.create_app(respect_client_system_prompt=False)
    with TestClient(app) as c:
        yield c, engine, tokenizer


def post(client, **overrides):
    body = {"model": f"ftspec-{PROFILE.name}",
             "messages": [{"role": "user", "content": "Customer: I was charged twice."}]}
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


# --- OpenAI wire compatibility ----------------------------------------------

def test_response_matches_openai_shape(client):
    c, _, _ = client
    r = post(c)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"])


def test_returned_content_satisfies_the_contract(client):
    c, _, _ = client
    content = post(c).json()["choices"][0]["message"]["content"]
    record, err = PROFILE.contract.validate(content)
    assert record is not None, err


def test_empty_messages_is_rejected(client):
    c, _, _ = client
    assert post(c, messages=[]).status_code == 400


def test_models_endpoint(client):
    c, _, _ = client
    body = c.get("/v1/models").json()
    assert body["data"][0]["id"] == f"ftspec-{PROFILE.name}"


def test_health_reports_schema_enforcement(client):
    c, _, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["schema_enforced"] is True


def test_schema_endpoint_serves_the_contract(client):
    c, _, _ = client
    assert c.get("/v1/schema").json() == PROFILE.contract.json_schema()


def test_profile_endpoint_reports_the_regulatory_envelope(client):
    c, _, _ = client
    body = c.get("/v1/profile").json()
    assert body["name"] == PROFILE.name
    assert body["regime"] == PROFILE.compliance.regime


# --- the guarantee -----------------------------------------------------------

def test_schema_is_enforced_without_the_client_asking(client):
    """The whole point: a client that knows nothing still gets constrained output."""
    c, engine, _ = client
    post(c)
    params = engine.calls[0]["params"]
    assert params["constrained"] is True
    assert params["schema"]["additionalProperties"] is False


def test_opting_out_is_explicit_and_counted(client):
    c, engine, _ = client
    post(c, ftspec_unconstrained=True)
    assert engine.calls[0]["params"]["constrained"] is False
    stats = c.get("/stats").json()
    assert stats["unconstrained_requests"] == 1
    assert stats["constrained_requests"] == 0


def test_trained_system_prompt_is_injected_when_absent(client):
    c, _, tokenizer = client
    post(c)
    assert tokenizer.last_conversation[0] == {"role": "system", "content": PROMPT_SHORT}


def test_client_system_prompt_is_replaced_by_default(client):
    """An arbitrary system prompt would move the model off its training distribution."""
    c, _, tokenizer = client
    post(c, messages=[{"role": "system", "content": "You are a pirate."},
                       {"role": "user", "content": "Customer: hello"}])
    assert tokenizer.last_conversation[0]["content"] == PROMPT_SHORT


def test_client_system_prompt_is_honoured_when_configured(monkeypatch):
    S.STATE.engine = StubEngine()
    S.STATE.tokenizer = StubTokenizer()
    S.STATE.profile = PROFILE
    S.STATE.system_prompt = PROMPT_SHORT
    monkeypatch.setattr(S, "build_sampling_params",
                         lambda req, constrained: {"constrained": constrained})
    app = S.create_app(respect_client_system_prompt=True)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "messages": [{"role": "system", "content": "You are a pirate."},
                          {"role": "user", "content": "Customer: hello"}]})
    assert S.STATE.tokenizer.last_conversation[0]["content"] == "You are a pirate."


# --- streaming ---------------------------------------------------------------

def test_streaming_emits_sse_chunks_and_terminates(client):
    c, _, _ = client
    with c.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Customer: hi"}],
            "stream": True}) as r:
        assert r.status_code == 200
        payload = "".join(r.iter_text())

    assert payload.startswith("data: ")
    assert payload.rstrip().endswith("data: [DONE]")

    chunks = [json.loads(line[6:]) for line in payload.splitlines()
              if line.startswith("data: ") and not line.endswith("[DONE]")]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    reassembled = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
    assert reassembled == SAMPLE_OUTPUT
