"""
Prompt-tax audit: measures how much of a prompt bill is spent re-sending
content that never changes.

Runs entirely locally and never transmits anything. That is the point. Asking a
prospect to email fifty production prompts to a vendor is a request their legal
team will refuse and should refuse, so the audit is built as a tool they run
themselves against their own logs. The report it emits contains aggregate
numbers only — no prompt text unless explicitly asked for — so it is safe to
share back.

What it measures
----------------
The **static prefix**: the longest run of leading characters common to every
sampled prompt. In practice that is the system prompt, the schema, the policy
document and the few-shot examples — content that is identical on every call
and is re-tokenised, re-transmitted and re-charged every time.

That is the portion a specialized model moves into weights, and the portion no
prompt-engineering technique can remove.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ftspec.run import get_logger

log = get_logger("ftspec.prompt_audit")

# USD per 1M tokens, published rates.
PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-sonnet": (3.00, 15.00),
}

# Used when tiktoken is unavailable. Deliberately conservative: it understates
# token counts for structured text like JSON schemas, so the resulting waste
# figure is a floor rather than a flattering estimate.
CHARS_PER_TOKEN = 3.6


@dataclass
class Sample:
    system: str
    user: str
    output: str = ""

    @property
    def full_prompt(self) -> str:
        return self.system + "\n" + self.user


@dataclass
class AuditResult:
    n_samples: int
    static_tokens: int
    variable_tokens: int
    output_tokens: int
    exact_system_share: float
    exact_tokens: bool
    static_preview: str = ""
    warnings: list = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return self.static_tokens + self.variable_tokens

    @property
    def static_share(self) -> float:
        total = self.total_input_tokens
        return (self.static_tokens / total) if total else 0.0


def _load_tokenizer(model_hint: str):
    """tiktoken when present; a labelled approximation otherwise."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return lambda text: len(enc.encode(text)), True
    except ImportError:
        log.warning("tiktoken not installed — using a chars/%.1f approximation. "
                     "Install tiktoken for exact counts: pip install tiktoken", CHARS_PER_TOKEN)
        return lambda text: int(len(text) / CHARS_PER_TOKEN), False


def parse_samples(path: Path) -> list:
    """Read prompts from the shapes a real prompt log actually arrives in."""
    samples: list = []
    skipped = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        if isinstance(row, dict) and "messages" in row:
            system = "\n".join(m.get("content") or "" for m in row["messages"]
                                if m.get("role") == "system")
            user = "\n".join(m.get("content") or "" for m in row["messages"]
                              if m.get("role") == "user")
            output = "\n".join(m.get("content") or "" for m in row["messages"]
                                if m.get("role") == "assistant")
            samples.append(Sample(system, user, output))
        elif isinstance(row, dict) and ("system" in row or "user" in row):
            samples.append(Sample(row.get("system", ""), row.get("user", ""),
                                   row.get("output", "") or row.get("assistant", "")))
        elif isinstance(row, dict) and "prompt" in row:
            # No role split available; the whole prompt is treated as one blob
            # and the static prefix is still recoverable from it.
            samples.append(Sample("", row["prompt"], row.get("completion", "")))
        else:
            skipped += 1

    if skipped:
        log.warning("skipped %d line(s) that matched no known prompt-log shape", skipped)
    return samples


def longest_common_prefix(texts: list) -> str:
    """The run of leading characters shared by every sample.

    Character-level rather than token-level so it survives tokenizer
    differences, then converted to tokens for the cost arithmetic.
    """
    if not texts:
        return ""
    shortest = min(texts, key=len)
    for i, ch in enumerate(shortest):
        if any(t[i] != ch for t in texts):
            return shortest[:i]
    return shortest


def audit(samples: list, model: str = "gpt-4o") -> AuditResult:
    count_tokens, exact = _load_tokenizer(model)
    warnings: list = []

    if len(samples) < 5:
        warnings.append(
            f"Only {len(samples)} samples. The static prefix is whatever they happen to "
            "share, which at this size is unreliable. 30+ gives a trustworthy figure."
        )

    prompts = [s.full_prompt for s in samples]
    static = longest_common_prefix(prompts)

    # A prefix shorter than a sentence is coincidence, not a shared preamble.
    if len(static) < 40:
        warnings.append(
            "No meaningful shared prefix found. Either these calls use genuinely different "
            "prompts, or the log interleaves several distinct workloads — audit each "
            "workload separately for a real number."
        )

    static_tokens = count_tokens(static)
    variable_tokens = round(statistics.mean(
        max(count_tokens(p) - static_tokens, 0) for p in prompts))
    output_tokens = round(statistics.mean(count_tokens(s.output) for s in samples)) \
        if any(s.output for s in samples) else 0

    if output_tokens == 0:
        warnings.append("No assistant/output content in the log, so output cost is excluded. "
                         "The waste figure below covers input tokens only.")

    systems = [s.system for s in samples if s.system]
    exact_share = 0.0
    if systems:
        most_common = max(set(systems), key=systems.count)
        exact_share = systems.count(most_common) / len(systems)

    return AuditResult(
        n_samples=len(samples),
        static_tokens=static_tokens,
        variable_tokens=variable_tokens,
        output_tokens=output_tokens,
        exact_system_share=exact_share,
        exact_tokens=exact,
        static_preview=static[:400],
        warnings=warnings,
    )


def render(result: AuditResult, model: str, calls_per_month: int,
            include_sample: bool = False) -> str:
    in_price, out_price = PRICING.get(model, PRICING["gpt-4o"])

    static_monthly = result.static_tokens * calls_per_month / 1e6 * in_price
    variable_monthly = result.variable_tokens * calls_per_month / 1e6 * in_price
    output_monthly = result.output_tokens * calls_per_month / 1e6 * out_price
    total_monthly = static_monthly + variable_monthly + output_monthly

    basis = "exact (tiktoken)" if result.exact_tokens else \
        f"approximate (chars/{CHARS_PER_TOKEN})"

    L = [
        "# Prompt Tax Audit",
        "",
        f"**Samples analysed:** {result.n_samples}  ·  **Model:** `{model}`  ·  "
        f"**Assumed volume:** {calls_per_month:,} calls/month  ·  **Token counts:** {basis}",
        "",
        "This report was generated locally. No prompt content left the machine it ran on.",
        "",
        "## What every call is carrying",
        "",
        "| Component | Tokens/call | Share of input |",
        "|---|---:|---:|",
        f"| **Static prefix** (identical every call) | **{result.static_tokens:,}** | "
        f"**{100 * result.static_share:.0f}%** |",
        f"| Variable content (the actual request) | {result.variable_tokens:,} | "
        f"{100 * (1 - result.static_share):.0f}% |",
        f"| Output | {result.output_tokens:,} | — |",
        "",
    ]

    if result.exact_system_share:
        L += [f"{100 * result.exact_system_share:.0f}% of sampled calls send a byte-identical "
               "system prompt.", ""]

    L += [
        "## What the static prefix costs",
        "",
        "| | Monthly | Annual |",
        "|---|---:|---:|",
        f"| **Static prefix (re-sent, never changes)** | **${static_monthly:,.0f}** | "
        f"**${static_monthly * 12:,.0f}** |",
        f"| Variable input | ${variable_monthly:,.0f} | ${variable_monthly * 12:,.0f} |",
        f"| Output | ${output_monthly:,.0f} | ${output_monthly * 12:,.0f} |",
        f"| **Total** | ${total_monthly:,.0f} | ${total_monthly * 12:,.0f} |",
        "",
    ]

    if total_monthly > 0:
        share = 100 * static_monthly / total_monthly
        L += [
            f"**{share:.0f}% of this workload's spend is re-transmitting content that never "
            f"changes** — ${static_monthly * 12:,.0f} per year to re-send the same "
            f"{result.static_tokens:,} tokens.",
            "",
        ]

    L += [
        "## What that means",
        "",
        f"- **Cost.** A specialized model carries those {result.static_tokens:,} tokens in its "
        "weights instead of its prompt, so the static line above goes to roughly zero. The "
        "variable content and the output still cost what they cost.",
        f"- **Latency.** Prefill is proportional to prompt length. Removing "
        f"{result.static_tokens:,} tokens cuts time-to-first-token by roughly "
        f"{100 * result.static_share:.0f}% on the same hardware.",
        "- **Ceiling.** No prompt-engineering technique removes this. Caching helps on "
        "repeated identical prefixes where the provider supports it, but the tokens are still "
        "counted and the policy still cannot be changed without a redeploy.",
        "",
        "## Caveats, stated plainly",
        "",
        "- The static prefix is the longest run shared by **these** samples. A wider sample "
        "may find a shorter one.",
        "- Volume is the figure supplied on the command line, not a measurement.",
        "- Provider prompt caching, if enabled, reduces the static line by its own discount "
        "rate. It does not eliminate it, and it does not help latency as much as not sending "
        "the tokens at all.",
        "",
    ]

    if result.warnings:
        L += ["## Warnings", ""]
        L += [f"- {w}" for w in result.warnings]
        L += [""]

    if include_sample and result.static_preview:
        L += ["## Static prefix (first 400 characters)", "",
               "> Included because `--include-sample` was passed. Remove this section before "
               "sharing if the prefix contains anything sensitive.", "",
               "```", result.static_preview, "```", ""]

    L += [
        "---",
        "",
        "Generated by `ftspec prompt-audit` — "
        "https://github.com/Titos-papadakis/fine-tuning",
        "",
    ]
    return "\n".join(L)


def run(path: Path, model: str = "gpt-4o", calls_per_month: int = 100_000,
         include_sample: bool = False, out: Path | None = None) -> tuple[str, AuditResult]:
    samples = parse_samples(path)
    if not samples:
        raise ValueError(
            f"No usable prompts found in {path}. Expected JSONL with one of: "
            '{"messages": [...]}, {"system": ..., "user": ...}, or {"prompt": ...}'
        )
    result = audit(samples, model)
    report = render(result, model, calls_per_month, include_sample)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        log.info("report written -> %s", out)
    return report, result
