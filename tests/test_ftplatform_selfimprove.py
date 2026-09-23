"""
Milestone 4, self-improvement: folding human-reviewed corrections into
training, and the retrain-and-maybe-redeploy cycle built on top of it.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.selfimprove import dataset_update, label, retrain_cycle

VALID_RECORD = {
    "ticket_summary": "Customer reports billing issue.",
    "customer": {"sentiment": "neutral", "language": "en", "tier": "pro", "churn_threat": False},
    "issue": {"category": "billing", "subcategory": "overcharge", "priority": "medium",
              "is_repeat_contact": False},
    "actions_taken": ["refund issued"],
    "resolution": {"status": "resolved", "requires_followup": False, "followup_date": None},
    "extracted_entities": {"order_id": "ORD-12345", "product_name": None, "amount": 19.99},
}


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture
def customer_ctx(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    return CustomerContext(conn, "acme")


# --- label -----------------------------------------------------------------------

def _queue_row(request_id, has_text=True):
    return {"request_id": request_id, "latency_ms": 50.0, "contract_valid": False,
            "reasons": ["schema_invalid"],
            "input": f"transcript-{request_id}" if has_text else None,
            "output": "not json" if has_text else None}


def test_pending_is_empty_with_no_queue_file(customer_ctx):
    assert label.pending(customer_ctx) == []


def test_skip_removes_the_row_and_writes_no_correction(customer_ctx):
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(customer_ctx.memory_dir() / "error_queue.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(_queue_row("r1")) + "\n")

    row = label.skip(customer_ctx, "r1")

    assert row["request_id"] == "r1"
    assert label.pending(customer_ctx) == []
    assert not (customer_ctx.memory_dir() / "corrections.jsonl").exists()


def test_record_correction_moves_row_from_queue_to_corrections(customer_ctx):
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(customer_ctx.memory_dir() / "error_queue.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(_queue_row("r1")) + "\n")
        f.write(json.dumps(_queue_row("r2")) + "\n")

    label.record_correction(customer_ctx, "r1", VALID_RECORD)

    assert [r["request_id"] for r in label.pending(customer_ctx)] == ["r2"]
    corrections = (customer_ctx.memory_dir() / "corrections.jsonl").read_text(encoding="utf-8")
    saved = json.loads(corrections.strip())
    assert saved["input"] == "transcript-r1"
    assert saved["output"] == VALID_RECORD


def test_record_correction_refuses_a_row_with_no_captured_text(customer_ctx):
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(customer_ctx.memory_dir() / "error_queue.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(_queue_row("r1", has_text=False)) + "\n")

    with pytest.raises(ValueError, match="no captured input"):
        label.record_correction(customer_ctx, "r1", VALID_RECORD)


def test_unknown_request_id_is_a_clean_error(customer_ctx):
    with pytest.raises(ValueError, match="not in the review queue"):
        label.skip(customer_ctx, "never-queued")


# --- dataset_update ----------------------------------------------------------------

def test_fold_corrections_with_no_file_is_a_no_op(customer_ctx):
    result = dataset_update.fold_corrections(customer_ctx)
    assert result == {"added": 0, "rejected": 0, "rejected_reasons": []}


def test_fold_corrections_appends_a_valid_correction_to_train_jsonl(customer_ctx):
    customer_ctx.data_dir().mkdir(parents=True, exist_ok=True)
    (customer_ctx.data_dir() / "train.jsonl").write_text('{"existing": true}\n', encoding="utf-8")
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(customer_ctx.memory_dir() / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "a transcript", "output": VALID_RECORD}) + "\n")

    result = dataset_update.fold_corrections(customer_ctx)

    assert result == {"added": 1, "rejected": 0, "rejected_reasons": []}
    lines = (customer_ctx.data_dir() / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # the pre-existing row plus the folded correction
    new_row = json.loads(lines[-1])
    assert new_row["messages"][1]["content"] == "a transcript"
    assert json.loads(new_row["messages"][2]["content"]) == VALID_RECORD


def test_fold_corrections_rejects_a_record_that_fails_the_contract(customer_ctx):
    customer_ctx.data_dir().mkdir(parents=True, exist_ok=True)
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    bad_record = {"ticket_summary": "missing everything else"}
    with open(customer_ctx.memory_dir() / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "a transcript", "output": bad_record}) + "\n")

    result = dataset_update.fold_corrections(customer_ctx)

    assert result["added"] == 0
    assert result["rejected"] == 1
    assert not (customer_ctx.data_dir() / "train.jsonl").exists()


def test_fold_corrections_archives_the_file_afterward(customer_ctx):
    customer_ctx.data_dir().mkdir(parents=True, exist_ok=True)
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    corrections_path = customer_ctx.memory_dir() / "corrections.jsonl"
    with open(corrections_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "x", "output": VALID_RECORD}) + "\n")

    dataset_update.fold_corrections(customer_ctx)

    assert not corrections_path.exists()
    archived = list(customer_ctx.memory_dir().glob("corrections.*.jsonl"))
    assert len(archived) == 1


def test_fold_corrections_is_isolated_per_customer(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    store.create(conn, "globex", "Globex Corp", "saas_support")
    a, b = CustomerContext(conn, "acme"), CustomerContext(conn, "globex")

    a.data_dir().mkdir(parents=True, exist_ok=True)
    a.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(a.memory_dir() / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "acme's", "output": VALID_RECORD}) + "\n")

    dataset_update.fold_corrections(a)

    assert not (b.data_dir() / "train.jsonl").exists()
    assert (a.data_dir() / "train.jsonl").exists()


# --- retrain_cycle -----------------------------------------------------------------

def test_run_cycle_skips_training_when_nothing_to_fold(customer_ctx, conn):
    result = retrain_cycle.run_cycle(customer_ctx, "org/model", 16, 16, conn=conn)

    assert result["candidate_id"] is None
    assert result["deployed"] is False
    assert result["folded"]["added"] == 0


def test_run_cycle_trains_and_deploys_when_gates_pass(monkeypatch, customer_ctx, conn):
    customer_ctx.data_dir().mkdir(parents=True, exist_ok=True)
    customer_ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    with open(customer_ctx.memory_dir() / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "x", "output": VALID_RECORD}) + "\n")
    with open(customer_ctx.memory_dir() / "deployed.json", "w", encoding="utf-8") as f:
        json.dump({"candidate_id": "baseline", "systems": {
            "base-rubric": {"schema_adherence_pct": 100.0, "headline_accuracy_pct": 50.0,
                              "record_exact_pct": 50.0, "cost_per_100k_usd": 5.0, "p99_ms": 200.0}}}, f)

    trained_ids = []

    def fake_run_stage_b_candidate(ctx, candidate, **kwargs):
        trained_ids.append(candidate.candidate_id)
        manifests = ctx.manifests_dir(candidate.candidate_id)
        manifests.mkdir(parents=True, exist_ok=True)
        manifest = {"params": {"base_model": candidate.base_model},
                    "metrics": {"systems": {"finetuned": {
                        "schema_adherence_pct": 100.0, "headline_accuracy_pct": 99.0,
                        "record_exact_pct": 99.0, "cost_per_100k_usd": 1.0, "p99_ms": 20.0}}}}
        (manifests / "evaluate.json").write_text(json.dumps(manifest), encoding="utf-8")
        reports = ctx.reports_dir(candidate.candidate_id)
        reports.mkdir(parents=True, exist_ok=True)
        candidate_dir = ctx.candidate_dir(candidate.candidate_id)
        (candidate_dir / "lora_adapter").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "lora_adapter" / "x.bin").write_text("fake", encoding="utf-8")

    monkeypatch.setattr(retrain_cycle, "run_stage_b_candidate", fake_run_stage_b_candidate)

    result = retrain_cycle.run_cycle(customer_ctx, "org/model", 16, 16, conn=conn)

    assert result["folded"]["added"] == 1
    assert result["candidate_id"] in trained_ids
    assert result["deployed"] is True


# --- CLI wiring: rejected corrections must be visible, not silently dropped ---
# Regression coverage for a real bug: fold_corrections() always returned
# `rejected`/`rejected_reasons`, but the CLI only ever printed `added` and
# the generic "no corrections to fold" message -- identical output whether
# nothing was submitted or everything submitted failed validation and was
# archived unfolded. A human reviewer's work could vanish with no visible
# signal.

def _invoke_selfimprove_run(customer_id="acme"):
    from typer.testing import CliRunner

    from ftplatform.cli import app
    return CliRunner().invoke(app, ["selfimprove", "run", customer_id,
                                      "--base-model", "org/model",
                                      "--lora-r", "16", "--lora-alpha", "16"])


def test_cli_reports_when_every_correction_is_rejected(monkeypatch, tmp_path, conn):
    import ftplatform.db as db_mod

    # REPO_ROOT (and DB_PATH, for the CLI's own connect()) must be patched
    # before anything resolves a path against them -- CustomerContext.
    # memory_dir() reads the module-level REPO_ROOT at call time, so
    # constructing it (or writing through it) before patching would resolve
    # against the *real* repo instead of tmp_path.
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")

    store.create(conn, "acme", "Acme Inc", "saas_support")
    ctx = CustomerContext(conn, "acme")
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    bad_record = {"ticket_summary": "missing everything else"}
    with open(ctx.memory_dir() / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "a transcript", "output": bad_record}) + "\n")
    conn.commit()

    result = _invoke_selfimprove_run()

    assert result.exit_code == 0
    combined = result.stdout + result.stderr
    # The old, ambiguous message must not be the *only* thing shown here --
    # that's exactly what made a fully-rejected batch indistinguishable from
    # nothing having been submitted at all.
    assert "1 correction(s) all FAILED" in combined


def test_cli_genuinely_nothing_to_fold_still_says_so_plainly(monkeypatch, tmp_path, conn):
    import ftplatform.db as db_mod

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")

    store.create(conn, "acme", "Acme Inc", "saas_support")
    conn.commit()

    result = _invoke_selfimprove_run()

    assert result.exit_code == 0
    combined = result.stdout + result.stderr
    assert "no corrections to fold" in combined
    assert "FAILED validation" not in combined
