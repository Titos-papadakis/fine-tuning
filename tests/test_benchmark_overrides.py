"""
`benchmark.run()`'s additive `load_in_4bit`/`constrained` overrides
(ftplatform's Stage C leans on these to re-evaluate an adapter under a
quantization or constrained-decoding toggle with no new backend code).

Verifies two things: the overrides actually reach the `SystemSpec` `run()`
passes to `run_local`, and -- the part that actually matters for backward
compatibility -- omitting them reproduces the exact defaults every existing
call site (`ftspec evaluate`, ftplatform's Stage A/B) already relies on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftspec.core.registry import load_profile
from ftspec.evaluation import benchmark as B

PROFILE = load_profile("saas_support")


@pytest.fixture
def eval_file(tmp_path):
    import shutil
    src = Path(__file__).resolve().parent.parent / "data" / "saas_support" / "eval.jsonl"
    dst = tmp_path / "eval.jsonl"
    shutil.copy(src, dst)
    return dst


def _capturing_run_local(captured):
    def run_local(spec, records, profile, max_new_tokens):
        captured.append(spec)
        result = B.RunResult(spec=spec)
        plan = PROFILE.scoring_plan()
        for r in records:
            gold = json.loads(r["messages"][2]["content"])
            result.raw_outputs.append(json.dumps(gold, ensure_ascii=False))
            result.valid_flags.append(1.0)
            result.latencies_ms.append(10.0)
            result.prompt_tokens.append(40)
            result.output_tokens.append(60)
            result.scores.append(B.M.score_record(gold, gold, plan))
        return result
    return run_local


def test_default_call_leaves_load_in_4bit_true_and_constrained_unset(monkeypatch, eval_file, tmp_path):
    captured = []
    monkeypatch.setattr(B, "run_local", _capturing_run_local(captured))

    B.run(PROFILE, eval_file, tmp_path, systems="base-rubric")

    assert captured[0].load_in_4bit is True
    assert captured[0].constrained is False  # base-rubric's catalogue default


def test_load_in_4bit_override_reaches_every_local_system(monkeypatch, eval_file, tmp_path):
    captured = []
    monkeypatch.setattr(B, "run_local", _capturing_run_local(captured))

    B.run(PROFILE, eval_file, tmp_path, systems="base-schema,base-rubric", load_in_4bit=False)

    assert all(spec.load_in_4bit is False for spec in captured)


def test_constrained_override_can_turn_it_on_for_finetuned(monkeypatch, eval_file, tmp_path):
    captured = []
    monkeypatch.setattr(B, "run_local", _capturing_run_local(captured))

    B.run(PROFILE, eval_file, tmp_path, systems="finetuned",
          finetuned_model="unused/path", constrained=True)

    assert captured[0].constrained is True


def test_constrained_override_can_turn_it_off_for_base_constrained(monkeypatch, eval_file, tmp_path):
    captured = []
    monkeypatch.setattr(B, "run_local", _capturing_run_local(captured))

    B.run(PROFILE, eval_file, tmp_path, systems="base-constrained", constrained=False)

    assert captured[0].constrained is False


def test_overrides_never_touch_hosted_systems(monkeypatch, eval_file, tmp_path):
    """load_in_4bit/constrained are inference-engine concepts; a hosted
    OpenAI system has no such knobs and must be refused/skipped exactly as
    before, not crash on an attribute that doesn't apply to it."""
    captured = []
    monkeypatch.setattr(B, "run_local", _capturing_run_local(captured))
    # No OPENAI_API_KEY / compliance ack in this test env -> gpt4o gets
    # skipped, not run; base-rubric still runs and should still show the
    # override was applied without error.
    result = B.run(PROFILE, eval_file, tmp_path, systems="base-rubric,gpt4o-rubric",
                    load_in_4bit=False)

    assert captured[0].load_in_4bit is False
    assert "gpt4o-rubric" not in result["systems"]
