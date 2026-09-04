"""
Pre-training validation gate.

Runs before any GPU is allocated and fails loudly rather than letting a bad
dataset reach the trainer. It catches the failure modes that are expensive
precisely because they are silent:

  - **Truncation.** A sample whose prompt + completion exceeds `max_seq_length`
    is not rejected by the trainer -- it is truncated. The model is then trained
    on JSON that stops mid-object, and learns to emit unterminated JSON. The
    symptom appears hours later as a mysterious validity collapse in eval.
  - **Contract drift.** A record whose assistant turn no longer satisfies the
    Pydantic contract (a renamed enum, a field added to the schema but not the
    generator) trains the model to produce output your validator will reject.
  - **Malformed conversations.** Wrong role order, empty turns, or a missing
    assistant message produce training signal that is quietly meaningless.
  - **Duplicates and leakage.** Repeated samples inflate apparent performance;
    an eval transcript present in training invalidates the benchmark outright.

Exit code is non-zero on failure so it can gate CI or a training job.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ftspec.config import Config
from ftspec.core.profile import Profile
from ftspec.run import get_logger

log = get_logger("ftspec.validate")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRecord(BaseModel):
    """Structural contract for one training row."""
    model_config = ConfigDict(extra="allow")  # `meta` is permitted and ignored downstream

    messages: list[ChatMessage]


class SplitReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n_records: int = 0
    structural_errors: list[str] = []
    contract_errors: list[str] = []
    length_errors: list[str] = []
    duplicate_count: int = 0
    token_lengths: list[int] = []
    prompt_lengths: list[int] = []

    @property
    def ok(self) -> bool:
        return not (self.structural_errors or self.contract_errors or self.length_errors)

    def percentile(self, values: list[int], p: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def load_tokenizer(base_model: str, approx: bool):
    """Real tokenizer if available; otherwise an explicit approximation.

    The approximation is offered because a length gate that cannot run at all
    is worse than one that runs with a stated margin of error -- but it is
    labelled everywhere it is used, because token counts drive a hard limit.
    """
    if approx:
        log.warning("using approximate token counts (chars/3.6) -- install transformers for exact counts")
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        log.warning("could not load tokenizer for %s (%s); falling back to approximation",
                     base_model, type(e).__name__)
        return None


def count_tokens(tokenizer, messages: list, add_generation_prompt: bool) -> int:
    if tokenizer is None:
        text = "".join(m["content"] for m in messages)
        return int(len(text) / 3.6)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def validate_split(path: Path, tokenizer, cfg: Config, name: str,
                    profile: Profile) -> SplitReport:
    report = SplitReport(name=name)
    if not path.exists():
        report.structural_errors.append(f"missing file: {path}")
        return report

    seen: set = set()
    limits = cfg.preflight

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        report.n_records += 1

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            report.structural_errors.append(f"#{i}: line is not valid JSON: {e}")
            continue

        try:
            record = ChatRecord.model_validate(raw)
        except ValidationError as e:
            report.structural_errors.append(f"#{i}: {e.error_count()} structural error(s): "
                                             f"{e.errors()[0]['loc']} {e.errors()[0]['msg']}")
            continue

        roles = [m.role for m in record.messages]
        if roles != ["system", "user", "assistant"]:
            report.structural_errors.append(f"#{i}: unexpected role sequence {roles}")
            continue
        if any(not m.content.strip() for m in record.messages):
            report.structural_errors.append(f"#{i}: contains an empty message")
            continue

        # The assistant turn must satisfy the same contract enforced at eval
        # time and by the production server -- one validator, three stages.
        _, contract_error = profile.contract.validate(record.messages[2].content)
        if contract_error:
            report.contract_errors.append(f"#{i}: {contract_error}")

        transcript = record.messages[1].content
        if transcript in seen:
            report.duplicate_count += 1
        seen.add(transcript)

        as_dicts = [m.model_dump() for m in record.messages]
        prompt_tokens = count_tokens(tokenizer, as_dicts[:2], add_generation_prompt=True)
        total_tokens = count_tokens(tokenizer, as_dicts, add_generation_prompt=False)
        report.prompt_lengths.append(prompt_tokens)
        report.token_lengths.append(total_tokens)

        if total_tokens > cfg.model.max_seq_length:
            report.length_errors.append(
                f"#{i}: {total_tokens} tokens exceeds max_seq_length={cfg.model.max_seq_length} "
                "-- the completion would be silently truncated during training"
            )
        elif total_tokens > limits.max_total_tokens:
            report.length_errors.append(f"#{i}: {total_tokens} tokens exceeds preflight limit "
                                         f"{limits.max_total_tokens}")
        if prompt_tokens > limits.max_prompt_tokens:
            report.length_errors.append(f"#{i}: prompt {prompt_tokens} tokens exceeds limit "
                                         f"{limits.max_prompt_tokens}")

    return report


def render_report(reports: list, cfg: Config, approx: bool) -> str:
    lines = ["", "Dataset preflight" + (" (approximate token counts)" if approx else ""), ""]
    header = f"  {'split':<8} {'n':>5} {'tok mean':>9} {'p95':>6} {'max':>6} {'limit':>6} {'dupes':>6}  status"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in reports:
        if r.token_lengths:
            mean = statistics.mean(r.token_lengths)
            p95 = r.percentile(r.token_lengths, 0.95)
            mx = max(r.token_lengths)
        else:
            mean = p95 = mx = 0
        status = "OK" if r.ok else "FAIL"
        lines.append(f"  {r.name:<8} {r.n_records:>5} {mean:>9.0f} {p95:>6} {mx:>6} "
                      f"{cfg.model.max_seq_length:>6} {r.duplicate_count:>6}  {status}")
    lines.append("")

    for r in reports:
        for kind, errors in (("structural", r.structural_errors),
                              ("contract", r.contract_errors),
                              ("length", r.length_errors)):
            if errors:
                lines.append(f"  [{r.name}] {len(errors)} {kind} error(s):")
                for e in errors[:8]:
                    lines.append(f"    - {e}")
                if len(errors) > 8:
                    lines.append(f"    ... and {len(errors) - 8} more")
                lines.append("")
    return "\n".join(lines)


def run(cfg: Config, profile: Profile, data_dir: Path,
         approx: bool = False) -> tuple[bool, str, dict]:
    """Validate every split against the profile's contract.

    Returns (passed, printable_report, metrics).
    """
    tokenizer = load_tokenizer(cfg.model.base_model, approx)
    used_approx = tokenizer is None

    reports = []
    for name in ("train", "val", "eval"):
        reports.append(validate_split(data_dir / f"{name}.jsonl", tokenizer, cfg, name, profile))

    text = render_report(reports, cfg, used_approx)

    # Leakage: any eval transcript that also appears in training invalidates the benchmark.
    def transcripts(name: str) -> set:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            return set()
        out = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(json.loads(line)["messages"][1]["content"])
        return out

    train_texts = transcripts("train") | transcripts("val")
    overlap = train_texts & transcripts("eval")
    if overlap:
        text += f"\n  LEAKAGE: {len(overlap)} eval transcripts also appear in train/val\n"

    passed = all(r.ok for r in reports) and not overlap
    metrics = {
        "profile": profile.name,
        "splits": {r.name: {"n": r.n_records,
                             "max_tokens": max(r.token_lengths) if r.token_lengths else 0,
                             "duplicates": r.duplicate_count,
                             "ok": r.ok} for r in reports},
        "leakage": len(overlap),
        "approximate_tokens": used_approx,
        "passed": passed,
    }
    text += ("\n  PASSED — dataset is safe to train on.\n" if passed
              else "\n  FAILED — fix the errors above before training.\n")
    return passed, text, metrics
