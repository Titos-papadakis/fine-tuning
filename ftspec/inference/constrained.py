"""
Structured generation.

Two strategies, both returning a contract-valid record, with very different
guarantees — and the difference is worth implementing both to show:

  GRAMMAR (Outlines)  The decoder is constrained at each step to tokens that can
                      still lead to a schema-valid document. Invalid output is
                      not unlikely, it is *unreachable*. One forward pass, no
                      retries, no tail latency, no failure mode to handle.

  RETRY (instructor-style)  Generate freely, validate, re-prompt with the error
                      on failure. Works against any endpoint, including ones you
                      do not control, but it is best-effort: extra calls, inflated
                      p99, and it can exhaust its budget and still return nothing.

Prefer GRAMMAR wherever you own the decoder. RETRY exists for third-party
endpoints, and as the honest comparison when someone claims retries are
"basically the same thing" — the p99 column in the benchmark shows they are not.

Outlines' public API changed between v0 and v1. Both are supported by feature
detection rather than a pinned version: a demo that fails to import on a
prospect's machine has already lost the room.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ftspec.core.contract import Contract
from ftspec.core.profile import Profile
from ftspec.run import get_logger

log = get_logger("ftspec.inference")


@dataclass
class Extraction:
    record: dict | None
    raw: str
    latency_ms: float
    attempts: int = 1
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.record is not None


class Extractor(Protocol):
    def extract(self, source_text: str) -> Extraction: ...


def build_grammar_generator(model, tokenizer, contract: Contract):
    """Bind an Outlines JSON generator for a contract, across v1 and v0 APIs.

    Shared with the benchmark's `base-constrained` system so the measured
    decoder is the one customers actually deploy, not a benchmark-only
    reimplementation of it.
    """
    try:
        import outlines
    except ImportError as e:
        raise RuntimeError(
            "Grammar-constrained decoding requires Outlines:  pip install 'ftspec[constrained]'\n"
            "Refusing to fall back to unconstrained generation, which would advertise a "
            "guarantee the process does not provide."
        ) from e

    # Outlines accepts a Pydantic model directly; for a customer-supplied raw
    # JSON Schema there is no model, so pass the schema document instead.
    target = contract.model if contract.model is not None else contract.schema_text()

    if hasattr(outlines, "from_transformers") and hasattr(outlines, "Generator"):
        ol_model = outlines.from_transformers(model, tokenizer)
        if contract.model is not None:
            generator = outlines.Generator(ol_model, contract.model)
        else:
            generator = outlines.Generator(ol_model, outlines.json_schema(target))
        log.info("Outlines v1 API (from_transformers + Generator)")
        return lambda prompt: str(generator(prompt))

    if hasattr(outlines, "generate") and hasattr(outlines.generate, "json"):
        from outlines.models import Transformers
        ol_model = Transformers(model, tokenizer)
        generator = outlines.generate.json(ol_model, target)
        log.info("Outlines v0 API (generate.json)")

        def call(prompt: str) -> str:
            result = generator(prompt)
            if hasattr(result, "model_dump_json"):
                return result.model_dump_json()
            if isinstance(result, dict):
                import json
                return json.dumps(result, ensure_ascii=False)
            return str(result)

        return call

    raise RuntimeError(
        f"Unrecognised Outlines API (version {getattr(outlines, '__version__', 'unknown')}). "
        "Expected `outlines.Generator` (v1) or `outlines.generate.json` (v0)."
    )


def _load_model(model_path: str, max_seq_length: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


class GrammarExtractor:
    """Grammar-constrained decoding. Adherence is structural, not hoped for."""

    def __init__(self, model, tokenizer, generator, profile: Profile):
        self._model = model
        self._tokenizer = tokenizer
        self._generator = generator
        self._profile = profile

    @classmethod
    def load(cls, model_path: str, profile: Profile, max_seq_length: int = 2048,
              load_in_4bit: bool = True) -> GrammarExtractor:
        model, tokenizer = _load_model(model_path, max_seq_length, load_in_4bit)
        generator = build_grammar_generator(model, tokenizer, profile.contract)
        return cls(model, tokenizer, generator, profile)

    def extract(self, source_text: str) -> Extraction:
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": self._profile.prompts.short},
             {"role": "user", "content": source_text}],
            tokenize=False, add_generation_prompt=True)

        start = time.perf_counter()
        raw = self._generator(prompt)
        elapsed = (time.perf_counter() - start) * 1000

        record, err = self._profile.contract.validate(raw)
        if record is None:
            # Reaching here means grammar and validator disagree, which is a bug
            # in this repo rather than a model failure. Say so plainly.
            log.error("grammar-constrained output failed contract validation (%s); "
                       "schema and grammar are out of sync", err)
        return Extraction(record=record, raw=raw, latency_ms=elapsed, error=err)


class RetryExtractor:
    """Unconstrained generation, validated, re-prompted on failure.

    The `instructor` pattern. Best-effort by construction: `max_attempts` is a
    budget, not a guarantee, and every retry is another full forward pass.
    """

    def __init__(self, model, tokenizer, profile: Profile,
                  max_attempts: int = 3, max_new_tokens: int = 512):
        self._model = model
        self._tokenizer = tokenizer
        self._profile = profile
        self._max_attempts = max_attempts
        self._max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, model_path: str, profile: Profile, max_seq_length: int = 2048,
              load_in_4bit: bool = True, max_attempts: int = 3) -> RetryExtractor:
        model, tokenizer = _load_model(model_path, max_seq_length, load_in_4bit)
        return cls(model, tokenizer, profile, max_attempts)

    def _generate(self, messages: list) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)

    def extract(self, source_text: str) -> Extraction:
        messages = [{"role": "system", "content": self._profile.prompts.short},
                     {"role": "user", "content": source_text}]
        start = time.perf_counter()
        raw, err = "", ""

        for attempt in range(1, self._max_attempts + 1):
            raw = self._generate(messages)
            record, err = self._profile.contract.validate(raw)
            if record is not None:
                return Extraction(record=record, raw=raw,
                                   latency_ms=(time.perf_counter() - start) * 1000,
                                   attempts=attempt)
            log.warning("attempt %d/%d failed validation: %s", attempt, self._max_attempts, err)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    f"That output was rejected: {err}. Return only a corrected JSON object "
                    f"matching the schema. No prose, no code fences."},
            ]

        return Extraction(record=None, raw=raw,
                           latency_ms=(time.perf_counter() - start) * 1000,
                           attempts=self._max_attempts, error=err)


def build_extractor(model_path: str, profile: Profile, strategy: str = "grammar",
                     **kwargs) -> Extractor:
    if strategy == "grammar":
        return GrammarExtractor.load(model_path, profile, **kwargs)
    if strategy == "retry":
        return RetryExtractor.load(model_path, profile, **kwargs)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'grammar' or 'retry'")
