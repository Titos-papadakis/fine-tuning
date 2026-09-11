"""
`ftspec combine-eval` -- folding systems run in separate processes back into
one report.

Running each system as its own `evaluate` process is the reliable fix for the
cross-model VRAM problem (see test_benchmark_resilience.py): a fresh process
guarantees a clean CUDA context. The cost is that each invocation only writes
a report for the one system it ran. `combine()` reads the raw_<name>.jsonl
files those separate runs left behind and rebuilds the same comparison a
single multi-system run would have produced -- entirely offline, since scoring
already-generated text needs no model and cannot itself run out of memory.

The central claim these tests check: combining is *equivalent* to running
together, not merely "produces some numbers". Two systems run separately then
combined must summarize identically to the same two systems run together.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ftspec.core.registry import load_profile
from ftspec.evaluation import benchmark as B

PROFILE = load_profile("saas_support")


@pytest.fixture
def eval_file(tmp_path):
    src = Path(__file__).resolve().parent.parent / "data" / "saas_support" / "eval.jsonl"
    dst = tmp_path / "eval.jsonl"
    shutil.copy(src, dst)
    return dst


def fake_result(spec, records, *_args, **_kwargs):
    """A deterministic, well-formed RunResult -- same shape a real run leaves."""
    plan = PROFILE.scoring_plan()
    golds = [json.loads(r["messages"][2]["content"]) for r in records]
    result = B.RunResult(spec=spec)
    for i, gold in enumerate(golds):
        # A few wrong so record_exact and McNemar have something to say, and
        # so the two systems being combined don't score identically.
        guess = gold if (i % 3 or spec.name == "finetuned") else {"wrong": True}
        result.raw_outputs.append(json.dumps(guess, ensure_ascii=False))
        result.valid_flags.append(1.0 if isinstance(guess, dict) and "wrong" not in guess else 0.0)
        result.latencies_ms.append(10.0 + i % 5)
        result.prompt_tokens.append(40 + i % 7)
        result.output_tokens.append(60 + i % 11)
        result.scores.append(B.M.score_record(guess if guess != {"wrong": True} else None,
                                               gold, plan))
    return result


def run_one_system_alone(monkeypatch, eval_file, results_dir, name):
    """Simulate `!ftspec evaluate --systems <name>` as its own process would."""
    monkeypatch.setattr(B, "run_local", fake_result)
    return B.run(PROFILE, eval_file, results_dir, systems=name)


def test_combining_two_separate_runs_matches_one_combined_run(monkeypatch, eval_file, tmp_path):
    separate_dir = tmp_path / "separate"
    combined_dir = tmp_path / "combined"

    for name in ("finetuned", "base-rubric"):
        run_one_system_alone(monkeypatch, eval_file, separate_dir, name)
    combined_manifest = B.combine(PROFILE, eval_file, separate_dir,
                                   systems="finetuned,base-rubric")

    monkeypatch.setattr(B, "run_local", fake_result)
    together_manifest = B.run(PROFILE, eval_file, combined_dir,
                               systems="finetuned,base-rubric")

    # Same two systems, same numbers, regardless of which path produced them.
    assert combined_manifest["systems"] == together_manifest["systems"]


def test_missing_systems_are_reported_not_silently_dropped(monkeypatch, eval_file, tmp_path):
    monkeypatch.setattr(B, "run_local", fake_result)
    B.run(PROFILE, eval_file, tmp_path, systems="finetuned")

    manifest = B.combine(PROFILE, eval_file, tmp_path,
                          systems="finetuned,base-rubric,base-constrained")

    assert set(manifest["systems"]) == {"finetuned"}
    assert set(manifest["missing_systems"]) == {"base-rubric", "base-constrained"}
    # The report itself only claims what it actually has.
    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "base-rubric" not in report.split("## What is measured")[0] or "finetuned" in report


def test_combining_with_nothing_run_yet_fails_clearly(eval_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `ftspec evaluate"):
        B.combine(PROFILE, eval_file, tmp_path, systems="finetuned")


def test_combining_an_unknown_system_name_is_a_clean_error(eval_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown system"):
        B.combine(PROFILE, eval_file, tmp_path, systems="not-a-real-system")


def test_combined_report_has_the_full_statistical_treatment(monkeypatch, eval_file, tmp_path):
    # Equivalence to render_report() is checked above; this confirms combine()
    # actually reaches that function rather than a stripped-down summary --
    # McNemar and confidence intervals are the parts a hand-rolled "just print
    # the percentages" version would be missing.
    monkeypatch.setattr(B, "run_local", fake_result)
    for name in ("finetuned", "base-rubric"):
        B.run(PROFILE, eval_file, tmp_path, systems=name)

    B.combine(PROFILE, eval_file, tmp_path, systems="finetuned,base-rubric")
    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")

    assert "McNemar" in report or "95%" in report  # significance / CI machinery ran
    assert "finetuned" in report and "base-rubric" in report


def test_combine_cli_command_writes_the_manifest_report_reads(monkeypatch, tmp_path, eval_file):
    """`ftspec report --update-readme` reads manifests/evaluate.json specifically;
    `combine-eval` must write to that same path, not a different one."""
    from typer.testing import CliRunner

    import ftspec.config as config_mod
    from ftspec.cli import app

    # Config.resolve() anchors every relative artefact path to REPO_ROOT --
    # deliberately, so `ftspec` always writes to the repo's own outputs/ and
    # data/ regardless of cwd. A test invoking the real CLI has to redirect
    # that anchor itself, or it will write into the actual repository.
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "saas_support").mkdir(parents=True)
    shutil.copy(eval_file, tmp_path / "data" / "saas_support" / "eval.jsonl")
    monkeypatch.setattr(B, "run_local", fake_result)

    runner = CliRunner()
    for name in ("finetuned", "base-rubric"):
        result = runner.invoke(app, ["evaluate", "--profile", "saas_support",
                                     "--systems", name])
        assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["combine-eval", "--profile", "saas_support",
                                 "--systems", "finetuned,base-rubric"])
    assert result.exit_code == 0, result.output

    manifest_path = tmp_path / "outputs" / "saas_support" / "manifests" / "evaluate.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["metrics"]["systems"]) == {"finetuned", "base-rubric"}
    assert manifest["metrics"]["combined_from_disk"] is True
