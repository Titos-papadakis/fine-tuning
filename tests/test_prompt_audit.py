"""
Prompt-tax audit tests.

This tool goes to prospects and its output becomes a number in a sales
conversation, so the arithmetic and — more importantly — the honesty
guarantees are tested: it must not leak prompt text by default, and it must
warn rather than invent a figure when the sample cannot support one.
"""
from __future__ import annotations

import json

import pytest

from ftspec import prompt_audit as pa

PREAMBLE = (
    "You are an extraction engine. Output ONLY JSON matching this schema:\n"
    + json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
    + "\nPolicy: escalate anything above threshold. Never invent fields."
)


def write(tmp_path, rows, name="log.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def messages_log(n=30, preamble=PREAMBLE):
    return [{"messages": [
        {"role": "system", "content": preamble},
        {"role": "user", "content": f"Request {i}: process order ORD-{1000 + i}."},
        {"role": "assistant", "content": json.dumps({"a": str(i)})},
    ]} for i in range(n)]


# --- parsing -----------------------------------------------------------------

def test_parses_openai_messages_shape(tmp_path):
    samples = pa.parse_samples(write(tmp_path, messages_log(5)))
    assert len(samples) == 5
    assert samples[0].system == PREAMBLE
    assert "ORD-1000" in samples[0].user
    assert samples[0].output


def test_parses_system_user_shape(tmp_path):
    rows = [{"system": PREAMBLE, "user": f"req {i}", "output": "{}"} for i in range(4)]
    samples = pa.parse_samples(write(tmp_path, rows))
    assert len(samples) == 4 and samples[0].system == PREAMBLE


def test_parses_flat_prompt_shape(tmp_path):
    rows = [{"prompt": PREAMBLE + f"\nreq {i}", "completion": "{}"} for i in range(4)]
    samples = pa.parse_samples(write(tmp_path, rows))
    assert len(samples) == 4
    assert PREAMBLE in samples[0].full_prompt


def test_unparseable_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text("not json\n" + json.dumps(messages_log(1)[0]) + "\n{}\n", encoding="utf-8")
    assert len(pa.parse_samples(path)) == 1


def test_empty_log_raises_with_a_usable_message(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable prompts"):
        pa.run(path)


# --- the measurement ---------------------------------------------------------

def test_longest_common_prefix():
    assert pa.longest_common_prefix(["abcdef", "abcxyz", "abc"]) == "abc"
    assert pa.longest_common_prefix(["abc", "abc"]) == "abc"
    assert pa.longest_common_prefix(["xyz", "abc"]) == ""
    assert pa.longest_common_prefix([]) == ""


def test_shared_preamble_is_detected_as_static(tmp_path):
    _, result = pa.run(write(tmp_path, messages_log(30)))
    assert result.static_tokens > 40
    assert result.static_share > 0.75
    assert result.exact_system_share == 1.0


def test_variable_content_is_excluded_from_static(tmp_path):
    _, result = pa.run(write(tmp_path, messages_log(30)))
    assert result.variable_tokens > 0
    assert result.variable_tokens < result.static_tokens


def test_distinct_prompts_yield_no_static_prefix_and_a_warning(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"{i} wholly unrelated text here"}]}
            for i in range(20)]
    _, result = pa.run(write(tmp_path, rows))
    assert result.static_tokens <= 2
    assert any("No meaningful shared prefix" in w for w in result.warnings)


def test_small_sample_is_flagged(tmp_path):
    _, result = pa.run(write(tmp_path, messages_log(3)))
    assert any("Only 3 samples" in w for w in result.warnings)


def test_missing_outputs_are_flagged_and_excluded(tmp_path):
    rows = [{"system": PREAMBLE, "user": f"req {i}"} for i in range(20)]
    _, result = pa.run(write(tmp_path, rows))
    assert result.output_tokens == 0
    assert any("output cost is excluded" in w for w in result.warnings)


# --- the report --------------------------------------------------------------

def test_report_scales_linearly_with_volume(tmp_path):
    path = write(tmp_path, messages_log(30))
    small, _ = pa.run(path, calls_per_month=10_000)
    large, _ = pa.run(path, calls_per_month=100_000)
    assert small != large
    assert "10,000 calls/month" in small and "100,000 calls/month" in large


def test_report_states_it_ran_locally(tmp_path):
    report, _ = pa.run(write(tmp_path, messages_log(20)))
    assert "No prompt content left the machine" in report


def test_report_withholds_prompt_text_by_default(tmp_path):
    """The report is meant to be shareable; leaking the prefix would defeat that."""
    report, _ = pa.run(write(tmp_path, messages_log(20)))
    assert "Static prefix (first 400" not in report
    assert "You are an extraction engine" not in report


def test_report_includes_prefix_only_when_asked(tmp_path):
    report, _ = pa.run(write(tmp_path, messages_log(20)), include_sample=True)
    assert "You are an extraction engine" in report
    assert "Remove this section before sharing" in report


def test_report_carries_its_caveats(tmp_path):
    report, _ = pa.run(write(tmp_path, messages_log(20)))
    assert "Caveats" in report
    assert "prompt caching" in report          # the obvious counter-argument
    assert "not a measurement" in report        # volume is an assumption


def test_cheaper_model_pricing_lowers_the_figure(tmp_path):
    path = write(tmp_path, messages_log(30))
    expensive, _ = pa.run(path, model="gpt-4o")
    cheap, _ = pa.run(path, model="gpt-4o-mini")
    assert expensive != cheap


def test_report_can_be_written_to_disk(tmp_path):
    out = tmp_path / "nested" / "audit.md"
    report, _ = pa.run(write(tmp_path, messages_log(20)), out=out)
    assert out.read_text(encoding="utf-8") == report
