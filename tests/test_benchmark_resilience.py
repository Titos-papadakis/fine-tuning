"""
One system's crash must not take the whole evaluate run down with it.

Confirmed on a real T4: `finetuned` completed with real results, then loading
the second model as part of the *same* `evaluate` invocation exhausted VRAM and
crashed before `base-constrained` produced anything. Because `run()` had no
per-system error boundary, the whole call raised, the systems that had already
succeeded were discarded, and no `benchmark_report.md` was written at all.

These tests stub `run_local`/`run_openai` (no GPU needed) to exercise the
orchestration logic in isolation: a failure partway through must still leave
the completed systems in the report, and must say plainly which system failed
and why, rather than being silent about it.
"""
from __future__ import annotations

import json

import pytest

from ftspec.core.registry import load_profile
from ftspec.evaluation import benchmark as B

PROFILE = load_profile("saas_support")
EVAL_FILE = None  # set in fixture


@pytest.fixture
def eval_file(tmp_path):
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "data" / "saas_support" / "eval.jsonl"
    dst = tmp_path / "eval.jsonl"
    shutil.copy(src, dst)
    return dst


def fake_result(spec, records, *_args, **_kwargs):
    """A minimal but well-formed RunResult, as if the system had actually run."""
    plan = PROFILE.scoring_plan()
    golds = [json.loads(r["messages"][2]["content"]) for r in records]
    result = B.RunResult(spec=spec)
    for gold in golds:
        result.raw_outputs.append(json.dumps(gold, ensure_ascii=False))
        result.valid_flags.append(1.0)
        result.latencies_ms.append(10.0)
        result.prompt_tokens.append(50)
        result.output_tokens.append(50)
        result.scores.append(B.M.score_record(gold, gold, plan))
    return result


def failing_result(*_args, **_kwargs):
    raise RuntimeError("CUDA out of memory: simulated for the test")


def test_a_mid_run_crash_keeps_the_systems_that_already_succeeded(monkeypatch, eval_file, tmp_path):
    calls = {"finetuned": fake_result, "base-constrained": failing_result,
             "base-rubric": fake_result}

    def dispatch(spec, records, profile, max_new_tokens):
        return calls[spec.name](spec, records, profile, max_new_tokens)

    monkeypatch.setattr(B, "run_local", dispatch)

    manifest = B.run(PROFILE, eval_file, tmp_path,
                      systems="finetuned,base-constrained,base-rubric")

    assert set(manifest["systems"]) == {"finetuned", "base-rubric"}
    assert set(manifest["failed_systems"]) == {"base-constrained"}
    assert "CUDA out of memory" in manifest["failed_systems"]["base-constrained"]

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "base-constrained" in report
    assert "Systems that failed to run" in report
    # The systems that succeeded still get their own results file.
    assert (tmp_path / "raw_finetuned.jsonl").exists()
    assert (tmp_path / "raw_base-rubric.jsonl").exists()
    assert not (tmp_path / "raw_base-constrained.jsonl").exists()


def test_a_clean_run_never_mentions_failures(monkeypatch, eval_file, tmp_path):
    monkeypatch.setattr(B, "run_local", fake_result)

    manifest = B.run(PROFILE, eval_file, tmp_path, systems="finetuned,base-rubric")

    assert manifest["failed_systems"] == {}
    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Systems that failed to run" not in report


def test_every_system_failing_raises_with_every_reason_named(monkeypatch, eval_file, tmp_path):
    monkeypatch.setattr(B, "run_local", failing_result)

    with pytest.raises(RuntimeError) as exc_info:
        B.run(PROFILE, eval_file, tmp_path, systems="finetuned,base-rubric")

    message = str(exc_info.value)
    assert "finetuned" in message
    assert "base-rubric" in message
    assert not (tmp_path / "benchmark_report.md").exists()


def test_report_render_omits_the_failed_section_when_failed_is_none(eval_file):
    # render_report's `failed` parameter is optional (older callers, and the
    # CLI's own self-test path, may not pass it); it must not require one.
    records = B.load_eval_set(eval_file, limit=5)
    result = fake_result(B.system_catalogue()["finetuned"], records)
    result.agg = B.M.aggregate(result.scores, PROFILE.scoring_plan())
    report = B.render_report({"finetuned": result}, PROFILE, 0.35, len(records), refused=[])
    assert "Systems that failed to run" not in report
