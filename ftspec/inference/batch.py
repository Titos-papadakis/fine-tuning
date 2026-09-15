"""
Offline bulk extraction: process many documents through an already-trained
adapter using padded batched generation, for throughput -- not the same
thing `ftspec.evaluation.benchmark.run_local()` does, which deliberately
generates one document at a time to measure real per-call latency. A
batch's wall-clock time divided across N documents is not any single call's
latency, so this never feeds the benchmark/leaderboard path -- mixing the
two would corrupt exactly the p50/p99 numbers that path exists to measure.

Grammar-constrained decoding is NOT batched here. Outlines' generator (see
ftspec.inference.constrained) is built around one sequence at a time, and
batching it correctly (per-sequence finite-state-machine state within one
batch) is real additional work with real correctness risk that hasn't been
done. A batch run here gets the trained system prompt and the same
contract-validation every other path uses, just not constrained decoding's
schema *guarantee* -- the same tradeoff `ftspec evaluate --systems
base-schema` already documents for its own unconstrained baseline.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from ftspec.core.profile import Profile
from ftspec.run import get_logger

log = get_logger("ftspec.inference.batch")


def load_documents(path: Path) -> list[str]:
    """One document per line: a raw JSON string, `{"input": text}`, or the
    engine's own chat-format row (`{"messages": [system, user, ...]}`, the
    same shape train/eval.jsonl already use) -- so a batch job can point
    directly at an existing eval file without reshaping it first."""
    docs = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, str):
            docs.append(row)
        elif isinstance(row, dict) and "input" in row:
            docs.append(row["input"])
        elif isinstance(row, dict) and "messages" in row:
            docs.append(row["messages"][1]["content"])
        else:
            raise ValueError(f"{path}:{lineno}: no document text found in {row!r} -- expected "
                              f"a string, {{'input': ...}}, or {{'messages': [...]}}")
    return docs


def run_batch(model_path: str, lora: str | None, profile: Profile, input_path: Path,
              output_path: Path, batch_size: int = 8, max_new_tokens: int = 512,
              load_in_4bit: bool = True) -> dict:
    """Runs every document in `input_path` through the adapter, `batch_size`
    at a time, writing one `{"input", "output", "valid"}` per line to
    `output_path`. Returns a summary dict for the caller to print/log."""
    import torch
    from unsloth import FastLanguageModel

    system_prompt = profile.prompts.short

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=4096, load_in_4bit=load_in_4bit)
    if lora:
        model.load_adapter(lora)
    FastLanguageModel.for_inference(model)
    # Decoder-only batched generation needs left-padding, or shorter
    # sequences in the batch get right-padded and the model ends up
    # "continuing" from pad tokens instead of its own last real token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    docs = load_documents(input_path)
    log.info("loaded %d documents from %s", len(docs), input_path)

    results = []
    started = time.time()
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": d}],
            tokenize=False, add_generation_prompt=True) for d in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        for j, doc in enumerate(chunk):
            text = tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True)
            record, valid = profile.contract.validate(text)
            results.append({"input": doc, "output": record if valid else text, "valid": valid})
        log.info("  processed %d/%d", min(i + batch_size, len(docs)), len(docs))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_s = time.time() - started
    n_valid = sum(1 for r in results if r["valid"])
    docs_per_s = round(len(results) / elapsed_s, 2) if elapsed_s else 0.0
    log.info("batch done: %d/%d valid, %.1fs (%.2f docs/s)",
              n_valid, len(results), elapsed_s, docs_per_s)
    return {"n_processed": len(results), "n_valid": n_valid, "elapsed_s": round(elapsed_s, 1),
            "docs_per_s": docs_per_s, "output_path": str(output_path)}
