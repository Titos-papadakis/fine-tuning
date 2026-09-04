#!/usr/bin/env python
"""
Serve a trained adapter over the OpenAI wire format.

This is the day-to-day launcher. `ftspec serve` assumes vLLM and a GPU, which is
the right production answer and the wrong answer when you want to check an
integration on a laptop at 11pm. Same endpoint, three backends:

  vllm          Production. Paged attention, continuous batching, real
                throughput. Needs CUDA.
  transformers  One adapter, plain HuggingFace, grammar-constrained decoding via
                Outlines. Runs on a small GPU or (slowly) on CPU. Use it to
                verify a freshly trained adapter before standing up vLLM.
  mock          No weights at all. Emits schema-valid records from the profile's
                own generator. Exists so client integrations, health checks and
                deploy scripts can be tested with nothing downloaded.

What it does NOT do is reimplement /v1/chat/completions. The FastAPI app comes
from `ftspec.serving.serve`, so the wire format, the server-side schema
enforcement and the system-prompt injection are the same code paths production
runs. A mock that exercises a parallel implementation proves nothing about the
real one; that is the failure mode this file is shaped to avoid.

The schema is enforced server-side for every backend. A client sends a raw
transcript and gets a valid record back -- no `guided_json`, no custom headers,
no knowledge of the contract on the client at all.

    python scripts/serve_adapter.py --profile saas_support --backend mock

    python scripts/serve_adapter.py --profile saas_support \
        --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
        --adapter outputs/saas_support/lora_adapter --backend transformers

    python scripts/serve_adapter.py --profile saas_support \
        --model outputs/saas_support/merged_model --backend vllm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:          # runnable without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from ftspec.core.profile import Profile  # noqa: E402
from ftspec.core.registry import load_profile  # noqa: E402
from ftspec.run import get_logger, setup_logging  # noqa: E402
from ftspec.serving import serve as S  # noqa: E402

log = get_logger("ftspec.serve_adapter")


@dataclass
class SimpleParams:
    """What vLLM's SamplingParams carries, for backends that are not vLLM."""
    temperature: float
    top_p: float
    max_tokens: int
    constrained: bool


def simple_params_builder(req, constrained: bool) -> SimpleParams:
    return SimpleParams(temperature=req.temperature, top_p=req.top_p,
                        max_tokens=req.max_tokens, constrained=constrained)


def vllm_result(text: str, prompt_tokens: int, completion_tokens: int):
    """Shape a result the way vLLM does, so one response path handles all backends."""
    return types.SimpleNamespace(
        outputs=[types.SimpleNamespace(text=text, token_ids=[0] * completion_tokens)],
        prompt_token_ids=[0] * prompt_tokens,
    )


# --- mock --------------------------------------------------------------------

class MockTokenizer:
    """Chat template without a tokenizer download.

    Token counts from this are character estimates, not tokens. That is
    deliberate: the mock backend is labelled everywhere it surfaces, so nobody
    mistakes a plausible number for a measured one.
    """

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in conversation)
        return rendered + ("\n<|assistant|>\n" if add_generation_prompt else "")

    @staticmethod
    def rough_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class MockEngine:
    """Schema-valid output with no model behind it.

    Records come from the profile's own generator, so they satisfy the same
    contract the server enforces and change shape correctly when you switch
    profiles. They have nothing to do with the input transcript, which is the
    point: this backend tests plumbing, never accuracy.
    """

    def __init__(self, profile: Profile, seed: int = 0):
        self.profile = profile
        self.seed = seed
        self.calls = 0

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.calls += 1
        log.warning("MOCK backend: request %s answered from the synthetic generator, "
                    "not from a model", request_id)
        sample = self.profile.generate(1, Random(self.seed + self.calls))[0]
        text = json.dumps(sample.record, ensure_ascii=False)
        yield vllm_result(text,
                          MockTokenizer.rough_tokens(prompt),
                          MockTokenizer.rough_tokens(text))


# --- transformers ------------------------------------------------------------

class TransformersEngine:
    """A single adapter, served with grammar-constrained decoding.

    No batching and no paged attention -- one request at a time, and slow on
    CPU. It exists to answer "did the thing I just trained actually work"
    without provisioning vLLM, and it uses the same Outlines generator the
    benchmark measures, so validity here means the same thing it means there.
    """

    def __init__(self, model_path: str, adapter_path: str | None, profile: Profile,
                 device: str = "auto", load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ftspec.inference.constrained import build_grammar_generator

        tokenizer_src = adapter_path or model_path
        log.info("loading tokenizer from %s", tokenizer_src)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

        kwargs: dict = {} if device == "cpu" else {"device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        log.info("loading base model from %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

        if adapter_path:
            from peft import PeftModel
            log.info("attaching LoRA adapter from %s", adapter_path)
            # Left unmerged on purpose: merging allocates a full dequantised copy
            # of the base model, which is the allocation that fails on small
            # boxes. Inference is a little slower; the box stays alive.
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model
        self.generator = build_grammar_generator(model, self.tokenizer, profile.contract)
        log.info("grammar-constrained decoding active; invalid JSON is unreachable")

    def _count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        if not getattr(sampling_params, "constrained", True):
            raise RuntimeError(
                "This backend only serves grammar-constrained output. Refusing "
                "ftspec_unconstrained=true rather than silently dropping the guarantee.")
        # Generation is blocking and compute-bound; off the event loop it goes,
        # or /health stops answering while a request is in flight.
        text = await asyncio.to_thread(self.generator, prompt)
        yield vllm_result(text, self._count(prompt), self._count(text))


# --- backend selection -------------------------------------------------------

def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def choose_backend(requested: str, model: str | None) -> str:
    """Pick the strongest backend the machine can actually run."""
    if requested != "auto":
        return requested
    if model and module_available("vllm") and cuda_available():
        return "vllm"
    if model and module_available("transformers"):
        return "transformers"
    return "mock"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="serve_adapter.py",
        description="Serve a trained adapter on an OpenAI-compatible endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--profile", default="saas_support",
                    help="Vertical to serve; determines the enforced schema.")
    ap.add_argument("--schema", type=Path, default=None,
                    help="JSON Schema file, for --profile custom.")
    ap.add_argument("--model", default=None,
                    help="Merged model directory, or a base model to pair with --adapter.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter directory to serve on top of --model.")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "vllm", "transformers", "mock"])
    ap.add_argument("--host", default="127.0.0.1",
                    help="Loopback by default; this endpoint has no authentication.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="transformers backend: quantise the base model.")
    ap.add_argument("--device", default="auto", help="transformers backend device_map.")
    ap.add_argument("--seed", type=int, default=0, help="mock backend record seed.")
    ap.add_argument("--respect-client-system-prompt", action="store_true",
                    help="Honour client system prompts instead of injecting the trained one.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def configure_state(profile: Profile, backend: str, args) -> None:
    """Install a non-vLLM engine into the shared server state."""
    if backend == "mock":
        engine: object = MockEngine(profile, seed=args.seed)
        tokenizer: object = MockTokenizer()
        model_name = f"ftspec-{profile.name}-MOCK"
        log.warning("=" * 72)
        log.warning("MOCK BACKEND -- responses are synthetic and unrelated to the input.")
        log.warning("Schema-valid, so integrations can be tested. Never a quality signal.")
        log.warning("=" * 72)
    else:
        engine = TransformersEngine(args.model, args.adapter, profile,
                                    device=args.device, load_in_4bit=args.load_in_4bit)
        tokenizer = engine.tokenizer
        model_name = f"ftspec-{profile.name}"

    S.STATE.engine = engine
    S.STATE.tokenizer = tokenizer
    S.STATE.profile = profile
    S.STATE.schema = profile.contract.json_schema()
    S.STATE.system_prompt = profile.prompts.short
    S.STATE.model_name = model_name
    S.STATE.backend = backend
    S.STATE.params_builder = simple_params_builder
    S.STATE.lora_request = None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        profile = load_profile(args.profile, schema_path=args.schema)
    except Exception as e:
        log.error("%s", e)
        return 2

    backend = choose_backend(args.backend, args.model)
    if backend != "mock" and not args.model:
        log.error("--model is required for the %s backend "
                  "(or use --backend mock to serve without weights)", backend)
        return 2

    # vLLM has its own loader, engine args and LoRA plumbing already wired into
    # ftspec.serving.serve. Delegating keeps exactly one production path.
    if backend == "vllm":
        S.STATE.backend = "vllm"
        S.serve(model=args.model, profile=profile, lora=args.adapter,
                host=args.host, port=args.port, max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                respect_client_system_prompt=args.respect_client_system_prompt)
        return 0

    import uvicorn

    configure_state(profile, backend, args)
    app = S.create_app(args.respect_client_system_prompt)

    log.info("profile '%s' (%s), backend %s, %d schema fields enforced server-side",
             profile.name, profile.compliance.regime, backend,
             len(S.STATE.schema.get("properties", {})))
    log.info("POST http://%s:%d/v1/chat/completions", args.host, args.port)
    log.info("GET  http://%s:%d/health", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
