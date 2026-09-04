"""
Production inference server — OpenAI-compatible, schema-guaranteed.

Why not just `vllm serve`?
-------------------------
vLLM's own OpenAI server can do guided decoding, but only if the *client* asks
for it, by sending `extra_body={"guided_json": ...}` on every call. That makes
the contract a client-side convention: one integration that forgets it, one SDK
that strips unknown fields, one engineer copy-pasting a curl example, and you
are back to parsing hopeful JSON in production.

This server inverts that. The schema comes from `ftspec.schemas.SupportTicket`
and is applied to **every** request, server-side. A client cannot opt out,
cannot forget, and cannot drift from the contract, because the contract is not
something the client sends. Clients that know nothing about this service — any
OpenAI SDK, LangChain, a raw curl — get schema-valid output by default.

The endpoint stays wire-compatible with `/v1/chat/completions`, so it drops into
existing code by changing a base URL.

The schema comes from the loaded profile's contract, so the same server binary
serves any vertical — `--profile fintech_disputes` enforces the dispute record,
`--profile healthcare_clinical` the FHIR record — with no code change.

Run:
    ftspec serve --profile saas_support --model outputs/merged_model
    ftspec serve --profile fintech --model base --lora outputs/lora_adapter
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ftspec.core.profile import Profile
from ftspec.run import get_logger

log = get_logger("ftspec.serve")


# --- OpenAI wire types -------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ftspec-extractor"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = Field(default=512, gt=0, le=4096)
    stream: bool = False
    # Extension: allows a caller to opt OUT of the schema guarantee explicitly.
    # Off by default, and logged when used, so bypassing the contract is a
    # deliberate, visible act rather than an accident.
    ftspec_unconstrained: bool = False


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# --- Engine ------------------------------------------------------------------

@dataclass
class ServerState:
    engine: Any = None
    tokenizer: Any = None
    profile: Profile | None = None
    model_name: str = "ftspec-extractor"
    lora_request: Any = None
    respect_client_system_prompt: bool = False
    schema: dict = field(default_factory=dict)
    system_prompt: str = ""
    served_requests: int = 0
    constrained_requests: int = 0
    backend: str = "vllm"
    # Non-vLLM engines (a transformers pipeline, a mock) supply their own
    # sampling-parameter object. Set this and the vLLM import is never reached,
    # which is what lets one wire implementation serve every backend instead of
    # each backend growing its own copy of /v1/chat/completions.
    params_builder: Any = None
    started_at: float = field(default_factory=time.time)
    # Bounded on purpose: /health reports recent behaviour, and an unbounded
    # list on a long-lived server is a slow memory leak.
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=512))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100.0) * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 1)


STATE = ServerState()


def build_sampling_params(req: ChatCompletionRequest, constrained: bool):
    """SamplingParams with guided decoding, across vLLM's two API generations.

    vLLM renamed `guided_decoding`/`GuidedDecodingParams` to
    `structured_outputs`/`StructuredOutputsParams` in newer releases. Feature
    detection keeps this working on both rather than pinning the customer to
    whichever version we happened to develop against.
    """
    if STATE.params_builder is not None:
        return STATE.params_builder(req, constrained)

    from vllm import SamplingParams

    kwargs: dict = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
    }
    if not constrained:
        return SamplingParams(**kwargs)

    schema = STATE.schema
    try:  # newer vLLM
        from vllm.sampling_params import StructuredOutputsParams
        kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
        return SamplingParams(**kwargs)
    except ImportError:
        pass
    try:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
        return SamplingParams(**kwargs)
    except ImportError as e:
        raise RuntimeError(
            "This vLLM build exposes neither StructuredOutputsParams nor "
            "GuidedDecodingParams, so the schema guarantee cannot be enforced. "
            "Refusing to serve unconstrained output under a constrained contract."
        ) from e


def render_prompt(messages: list) -> str:
    """Apply the chat template, injecting the trained system prompt.

    The fine-tuned model was trained with exactly one system prompt. Honouring
    an arbitrary client-supplied one would silently move the model off its
    training distribution, so by default we replace it and say so in the logs.
    """
    conversation = [m.model_dump() for m in messages]
    has_system = conversation and conversation[0]["role"] == "system"
    trained_prompt = STATE.system_prompt

    if not STATE.respect_client_system_prompt:
        if has_system and conversation[0]["content"].strip() != trained_prompt:
            log.debug("replacing client system prompt with the trained one")
            conversation[0] = {"role": "system", "content": trained_prompt}
        elif not has_system:
            conversation.insert(0, {"role": "system", "content": trained_prompt})

    return STATE.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)


async def generate_once(req: ChatCompletionRequest) -> tuple[str, int, int]:
    prompt = render_prompt(req.messages)
    constrained = not req.ftspec_unconstrained
    params = build_sampling_params(req, constrained)
    request_id = f"ftspec-{uuid.uuid4().hex[:12]}"
    start = time.perf_counter()

    final = None
    async for output in STATE.engine.generate(prompt, params, request_id,
                                               lora_request=STATE.lora_request):
        final = output

    if final is None or not final.outputs:
        raise RuntimeError("engine produced no output")

    text = final.outputs[0].text
    prompt_tokens = len(final.prompt_token_ids or [])
    completion_tokens = len(final.outputs[0].token_ids or [])

    STATE.served_requests += 1
    STATE.latencies_ms.append((time.perf_counter() - start) * 1000)
    if constrained:
        STATE.constrained_requests += 1
    else:
        log.warning("request %s served UNCONSTRAINED at client request", request_id)

    return text, prompt_tokens, completion_tokens


def create_app(respect_client_system_prompt: bool = False):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    STATE.respect_client_system_prompt = respect_client_system_prompt
    app = FastAPI(title="ftspec support-ticket extractor",
                   description="OpenAI-compatible endpoint with a server-enforced JSON schema.",
                   version="1.0.0")

    @app.get("/health")
    async def health():
        """Liveness plus enough metadata to tell two deployments apart.

        A load balancer needs the status field. An operator staring at a wrong
        answer needs to know *which* profile and backend actually answered, and
        whether latency has moved — so both are here rather than in a separate
        metrics endpoint nobody has wired up yet.
        """
        profile = STATE.profile
        recent = list(STATE.latencies_ms)
        return {"status": "ok" if STATE.engine is not None else "loading",
                "model": STATE.model_name,
                "backend": STATE.backend,
                "profile": profile.name if profile else None,
                "regime": profile.compliance.regime if profile else None,
                "allows_external_api": profile.compliance.allows_external_api
                                        if profile else None,
                "schema_enforced": True,
                "schema_fields": sorted(STATE.schema.get("properties", {})),
                "lora": STATE.lora_request is not None,
                "uptime_s": round(time.time() - STATE.started_at, 1),
                "latency_ms": {"count": len(recent),
                                "p50": _percentile(recent, 50),
                                "p99": _percentile(recent, 99),
                                "last": round(recent[-1], 1) if recent else 0.0},
                "requests": {"served": STATE.served_requests,
                              "constrained": STATE.constrained_requests,
                              "unconstrained": STATE.served_requests
                                                - STATE.constrained_requests}}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list",
                "data": [{"id": STATE.model_name, "object": "model",
                           "created": int(time.time()), "owned_by": "ftspec"}]}

    @app.get("/v1/schema")
    async def schema():
        """The contract this endpoint enforces. Useful for client codegen."""
        return STATE.schema

    @app.get("/v1/profile")
    async def profile_info():
        """Which vertical is loaded, and under what regulatory envelope."""
        if STATE.profile is None:
            raise HTTPException(status_code=503, detail="no profile loaded")
        return STATE.profile.summary()

    @app.get("/stats")
    async def stats():
        return {"served_requests": STATE.served_requests,
                "constrained_requests": STATE.constrained_requests,
                "unconstrained_requests": STATE.served_requests - STATE.constrained_requests}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if STATE.engine is None:
            raise HTTPException(status_code=503, detail="engine still loading")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if not req.stream:
            try:
                text, p_tok, c_tok = await generate_once(req)
            except Exception as e:
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail=str(e)) from e
            return ChatCompletionResponse(
                id=completion_id, created=created, model=STATE.model_name,
                choices=[Choice(message=ChoiceMessage(content=text))],
                usage=Usage(prompt_tokens=p_tok, completion_tokens=c_tok,
                             total_tokens=p_tok + c_tok),
            )

        async def event_stream():
            first = {"id": completion_id, "object": "chat.completion.chunk",
                      "created": created, "model": STATE.model_name,
                      "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]}
            yield f"data: {json.dumps(first)}\n\n"

            prompt = render_prompt(req.messages)
            params = build_sampling_params(req, not req.ftspec_unconstrained)
            request_id = f"ftspec-{uuid.uuid4().hex[:12]}"
            emitted = 0
            try:
                async for output in STATE.engine.generate(prompt, params, request_id,
                                                           lora_request=STATE.lora_request):
                    text = output.outputs[0].text
                    if len(text) > emitted:
                        delta = text[emitted:]
                        emitted = len(text)
                        chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                  "created": created, "model": STATE.model_name,
                                  "choices": [{"index": 0, "delta": {"content": delta},
                                                "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                log.exception("streaming generation failed")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            done = {"id": completion_id, "object": "chat.completion.chunk",
                     "created": created, "model": STATE.model_name,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
            STATE.served_requests += 1

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def load_engine(model: str, profile: Profile, lora: str | None, max_model_len: int,
                 gpu_memory_utilization: float, max_lora_rank: int) -> None:
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    STATE.profile = profile
    STATE.schema = profile.contract.json_schema()
    STATE.system_prompt = profile.prompts.short
    STATE.model_name = f"ftspec-{profile.name}"

    log.info("loading tokenizer from %s", model)
    STATE.tokenizer = AutoTokenizer.from_pretrained(model)

    engine_args = AsyncEngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=lora is not None,
        max_lora_rank=max_lora_rank if lora else 16,
    )
    log.info("starting vLLM engine (lora=%s)", lora or "none")
    STATE.engine = AsyncLLMEngine.from_engine_args(engine_args)

    if lora:
        from vllm.lora.request import LoRARequest
        STATE.lora_request = LoRARequest("ftspec-adapter", 1, lora)
        log.info("serving LoRA adapter from %s", lora)

    log.info("profile '%s' (%s) — schema guarantee active on every request, "
              "%d top-level properties", profile.name, profile.compliance.regime,
              len(STATE.schema.get("properties", {})))
    if not profile.compliance.allows_external_api:
        log.info("this profile forbids external API egress; self-hosted serving is the "
                  "only admissible deployment for its data")


def serve(model: str, profile: Profile, lora: str | None = None,
           host: str = "0.0.0.0", port: int = 8000, max_model_len: int = 4096,
           gpu_memory_utilization: float = 0.90, max_lora_rank: int = 16,
           respect_client_system_prompt: bool = False) -> None:
    import uvicorn

    load_engine(model, profile, lora, max_model_len, gpu_memory_utilization, max_lora_rank)
    app = create_app(respect_client_system_prompt)

    log.info("listening on http://%s:%d/v1/chat/completions", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
