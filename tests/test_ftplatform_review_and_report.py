"""
Suggested corrections in review (label.correct/suggestion/apply_edits), the
monthly customer report, Kaggle output download tolerance, and CLI wiring for
the commands added alongside them.
"""
from __future__ import annotations

import json
import subprocess

import pytest
from typer.testing import CliRunner

import ftspec.config as config_mod
from ftplatform import db as db_mod
from ftplatform.billing import usage
from ftplatform.cli import app
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.deployment import registry
from ftplatform.remote import kaggle_ops
from ftplatform.reporting import monthly
from ftplatform.selfimprove import label

VALID = {
    "ticket_summary": "Customer reports billing issue.",
    "customer": {"sentiment": "neutral", "language": "en", "tier": "pro", "churn_threat": False},
    "issue": {"category": "billing", "subcategory": "overcharge", "priority": "medium",
              "is_repeat_contact": False},
    "actions_taken": ["refund issued"],
    "resolution": {"status": "resolved", "requires_followup": False, "followup_date": None},
    "extracted_entities": {"order_id": "ORD-12345", "product_name": None, "amount": 19.99},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")
    conn = connect()
    store.create(conn, "acme", "Acme <Inc>", "saas_support")
    ctx = CustomerContext(conn, "acme")
    yield conn, ctx
    conn.close()


def _queue(ctx, rows):
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(ctx.memory_dir() / "error_queue.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(rid, output):
    return {"request_id": rid, "latency_ms": 900.0, "contract_valid": True,
            "reasons": ["latency_outlier"], "input": f"ticket {rid}", "output": output}


def _corrections(ctx):
    path = ctx.memory_dir() / "corrections.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- suggested corrections -------------------------------------------------------

def test_confirming_the_suggestion_records_the_models_own_output(env):
    _, ctx = env
    _queue(ctx, [_row("r1", VALID)])

    label.correct(ctx, "r1")

    assert _corrections(ctx) == [{"input": "ticket r1", "output": VALID}]
    assert label.pending(ctx) == []


def test_set_overrides_only_the_named_field(env):
    _, ctx = env
    _queue(ctx, [_row("r1", VALID)])

    label.correct(ctx, "r1", edits=label.parse_set_args(
        ["issue.subcategory=duplicate_charge", "extracted_entities.amount=42.5"]))

    out = _corrections(ctx)[0]["output"]
    assert out["issue"]["subcategory"] == "duplicate_charge"
    assert out["extracted_entities"]["amount"] == 42.5
    assert out["customer"] == VALID["customer"]


def test_a_correction_that_breaks_the_schema_is_refused_and_stays_queued(env):
    _, ctx = env
    _queue(ctx, [_row("r1", VALID)])

    with pytest.raises(label.InvalidCorrectionError):
        label.correct(ctx, "r1", edits={"issue.priority": "whenever"})

    assert [r["request_id"] for r in label.pending(ctx)] == ["r1"]
    assert not (ctx.memory_dir() / "corrections.jsonl").exists()


def test_suggestion_parses_fenced_json_text_and_gives_up_on_garbage():
    assert label.suggestion({"output": "```json\n" + json.dumps(VALID) + "\n```"}) == VALID
    assert label.suggestion({"output": "sorry, I can't"}) is None
    assert label.suggestion({"output": None}) is None


def test_no_parseable_output_needs_a_full_record(env):
    _, ctx = env
    _queue(ctx, [_row("r1", "garbage")])

    with pytest.raises(label.InvalidCorrectionError):
        label.correct(ctx, "r1")
    label.correct(ctx, "r1", output=VALID)
    assert _corrections(ctx)[0]["output"] == VALID


def test_parse_set_args_rejects_a_missing_equals():
    with pytest.raises(label.InvalidCorrectionError):
        label.parse_set_args(["issue.priority"])


def test_cli_review_correct_with_set(env):
    _, ctx = env
    _queue(ctx, [_row("r1", VALID)])

    result = CliRunner().invoke(app, ["review", "correct", "acme", "--request-id", "r1",
                                      "--set", "issue.priority=high"])

    assert result.exit_code == 0, result.output
    assert _corrections(ctx)[0]["output"]["issue"]["priority"] == "high"


def test_cli_review_pending_shows_the_suggestion(env):
    _, ctx = env
    _queue(ctx, [_row("r1", VALID)])

    result = CliRunner().invoke(app, ["review", "pending", "acme"])

    assert "overcharge" in result.output and "--set" in result.output


# --- monthly report ----------------------------------------------------------------

def _setup_month(conn, ctx, period="2026-09", with_gpt4o=True):
    for _ in range(3):
        usage.record(conn, "acme", prompt_tokens=100, completion_tokens=200, period=period)
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(ctx.memory_dir() / "production_log.jsonl", "w", encoding="utf-8") as f:
        for i, valid in enumerate([True, True, True, False]):
            f.write(json.dumps({"timestamp": f"{period}-0{i + 1}T10:00:00+00:00",
                                "contract_valid": valid, "latency_ms": 100.0 * (i + 1)}) + "\n")
        f.write(json.dumps({"timestamp": "2026-08-30T10:00:00+00:00", "contract_valid": False,
                            "latency_ms": 9999.0}) + "\n")
    (ctx.memory_dir() / "deployed.json").write_text(json.dumps({
        "kind": "deployed", "candidate_id": "stageB-r16-a16", "captured_at": f"{period}-05T00:00:00",
        "systems": {"finetuned": {"schema_adherence_pct": 100.0, "record_exact_pct": 88.0,
                                  "headline_accuracy_pct": 97.3, "cost_per_100k_usd": 1.0,
                                  "p99_ms": 400.0}}}), encoding="utf-8")
    systems = {"base-constrained": {"record_exact_pct": 36.7, "mean_prompt_tokens": 900.0}}
    if with_gpt4o:
        systems["gpt4o-rubric"] = {"record_exact_pct": 60.0, "mean_prompt_tokens": 1500.0}
    d = ctx.manifests_dir("baseline")
    d.mkdir(parents=True, exist_ok=True)
    (d / "evaluate.json").write_text(json.dumps({"metrics": {"systems": systems}}), encoding="utf-8")
    registry.record(conn, "acme", "saas_support", "stageB-r16-a16", {"systems": {}})
    conn.execute("UPDATE deployments SET deployed_at = ?", (f"{period}-05T00:00:00",))
    conn.commit()
    (ctx.memory_dir() / f"corrections.{period.replace('-', '')}10T120000.jsonl").write_text(
        '{"input": "a"}\n{"input": "b"}\n', encoding="utf-8")
    (ctx.memory_dir() / "corrections.20260801T120000.jsonl").write_text('{"input": "old"}\n',
                                                                        encoding="utf-8")


def test_monthly_report_numbers(env):
    conn, ctx = env
    _setup_month(conn, ctx)

    r = monthly.build(conn, ctx, period="2026-09", monthly_fee_usd=0.001)

    assert r["usage"]["requests"] == 3
    assert r["live"]["captured"] == 4 and r["live"]["valid_pct"] == 75.0
    assert r["model"]["record_exact_pct"] == 88.0
    assert r["best_prompted_at_onboarding"] == {"system": "gpt4o-rubric", "record_exact_pct": 60.0}
    assert r["corrections_folded"] == 2
    assert [p["candidate_id"] for p in r["promotions"]] == ["stageB-r16-a16"]
    # measured gpt4o-rubric prompt size: 1500 in @ $2.50/M + 200 out @ $10/M per call
    assert r["cost"]["per_call_usd"] == pytest.approx(0.00575, abs=1e-5)
    assert r["cost"]["equivalent_usd"] == pytest.approx(0.02, abs=0.01)
    assert "measured" in r["cost"]["basis"]
    assert r["cost"]["savings_usd"] is not None


def test_monthly_report_never_shows_negative_savings_and_labels_estimates(env):
    conn, ctx = env
    _setup_month(conn, ctx, with_gpt4o=False)

    r = monthly.build(conn, ctx, period="2026-09", monthly_fee_usd=3000.0)

    assert r["cost"]["savings_usd"] is None
    assert r["cost"]["basis"].startswith("estimated")


def test_monthly_report_with_no_traffic_is_still_valid(env):
    conn, ctx = env
    r = monthly.build(conn, ctx, period="2026-09")
    assert r["usage"]["requests"] == 0 and r["cost"] is None and r["model"] is None
    assert "<html" in monthly.render_html(r)


def test_monthly_report_html_escapes_customer_text(env):
    conn, ctx = env
    _setup_month(conn, ctx)

    path, _ = monthly.write(conn, ctx, period="2026-09")

    page = path.read_text(encoding="utf-8")
    assert "Acme &lt;Inc&gt;" in page and "Acme <Inc>" not in page
    assert path.with_suffix(".json").exists()


def test_cli_report_monthly(env):
    conn, ctx = env
    _setup_month(conn, ctx)

    result = CliRunner().invoke(app, ["report", "monthly", "acme", "--period", "2026-09"])

    assert result.exit_code == 0, result.output
    assert (ctx.customer_root() / "reports" / "monthly-2026-09.html").exists()


# --- CLI: import / delete / pipeline ------------------------------------------------

def test_cli_customer_import(env, tmp_path):
    _, ctx = env
    src = tmp_path / "tickets.jsonl"
    src.write_text(json.dumps({"input": "hello", "output": VALID}) + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["customer", "import", "acme", str(src)])

    assert result.exit_code == 0, result.output
    assert "labeled by customer:  1" in result.output
    assert ctx.imported_corpus_path().exists()


def test_cli_customer_delete_needs_yes(env):
    conn, ctx = env
    ctx.memory_dir().mkdir(parents=True)

    dry = CliRunner().invoke(app, ["customer", "delete", "acme"])
    assert dry.exit_code == 0 and "--yes" in dry.output
    assert store.get(conn, "acme") is not None

    real = CliRunner().invoke(app, ["customer", "delete", "acme", "--yes"])
    assert real.exit_code == 0, real.output
    assert store.get(conn, "acme") is None
    assert not ctx.customer_root().exists()


def test_cli_pipeline_start_and_status(env):
    result = CliRunner().invoke(app, ["pipeline", "start", "acme", "--stage-a-top-k", "1"])
    assert result.exit_code == 0, result.output
    assert "stage=baseline" in result.output

    status = CliRunner().invoke(app, ["pipeline", "status", "acme"])
    assert status.exit_code == 0 and "stage=baseline" in status.output

    again = CliRunner().invoke(app, ["pipeline", "start", "acme"])
    assert again.exit_code == 1


# --- kaggle_ops: output download on a non-UTF-8 console -----------------------------

def _failing_output_run(tmp_path, stderr, write_file=True):
    def run(cmd, capture_output=True, text=True, **kw):
        if write_file:
            out = tmp_path / "out" / "result_packet"
            out.mkdir(parents=True, exist_ok=True)
            (out / "job.db").write_text("x", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)
    return run


def test_download_tolerates_the_log_write_unicode_crash_once_files_are_down(tmp_path):
    run = _failing_output_run(tmp_path, "UnicodeEncodeError: 'charmap' codec can't encode '\\U0001f9a5'")
    out = kaggle_ops.download_kernel_output("me/k", tmp_path / "out", run=run)
    assert (out / "result_packet" / "job.db").exists()


def test_download_still_raises_on_unicode_crash_with_nothing_downloaded(tmp_path):
    run = _failing_output_run(tmp_path, "UnicodeEncodeError: boom", write_file=False)
    with pytest.raises(kaggle_ops.KaggleCommandError):
        kaggle_ops.download_kernel_output("me/k", tmp_path / "out", run=run)


def test_download_still_raises_on_any_other_failure(tmp_path):
    run = _failing_output_run(tmp_path, "403 Forbidden")
    with pytest.raises(kaggle_ops.KaggleCommandError):
        kaggle_ops.download_kernel_output("me/k", tmp_path / "out", run=run)
