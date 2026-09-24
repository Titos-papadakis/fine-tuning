"""
Custom-schema workloads, the audit log, per-customer redaction/retention, and
a tenant-isolation tripwire that runs every customer-scoped path for one
customer and proves another customer's files and rows were never touched.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import ftspec.config as config_mod
from ftplatform import audit, privacy
from ftplatform import db as db_mod
from ftplatform.billing import usage
from ftplatform.cli import app
from ftplatform.customers import importer, store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.jobs import approvals, pipeline
from ftplatform.monitoring import capture
from ftplatform.remote import packet
from ftplatform.reporting import monthly
from ftspec.core.redaction import redact

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["refund", "upgrade", "bug"]},
        "urgent": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["intent", "urgent", "summary"],
    "additionalProperties": False,
}
REC = {"intent": "refund", "urgent": True, "summary": "wants money back"}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")
    monkeypatch.setenv("FTPLATFORM_ACTOR", "tester")
    return tmp_path


@pytest.fixture
def schema_file(tmp_path):
    p = tmp_path / "schema_src.json"
    p.write_text(json.dumps(SCHEMA), encoding="utf-8")
    return p


def _add_custom(root, schema_file, cid="initech", *extra):
    return CliRunner().invoke(app, ["customer", "add", cid, "--name", "Initech",
                                    "--workload", "custom", "--schema", str(schema_file),
                                    "--headline-field", "intent", *extra])


# --- custom workloads ------------------------------------------------------------

def test_custom_customer_gets_their_own_schema_profile(root, schema_file):
    result = _add_custom(root, schema_file, "initech", "--free-text-field", "summary")
    assert result.exit_code == 0, result.output

    conn = connect()
    ctx = CustomerContext(conn, "initech")
    assert ctx.profile.contract.validate(json.dumps(REC))[0] == REC
    assert ctx.profile.contract.validate(json.dumps({**REC, "intent": "nope"}))[0] is None
    plan = ctx.profile.scoring_plan()
    assert plan.headline().path == "intent"
    assert "summary" not in plan.record_match_paths
    assert (root / "customers" / "initech" / "workload" / "schema.json").exists()
    conn.close()


def test_custom_add_rejects_a_bad_schema_or_field_before_creating_anything(root, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _add_custom(root, bad).exit_code == 1

    good = tmp_path / "good.json"
    good.write_text(json.dumps(SCHEMA), encoding="utf-8")
    result = CliRunner().invoke(app, ["customer", "add", "initech", "--name", "I",
                                      "--workload", "custom", "--schema", str(good),
                                      "--headline-field", "no.such.field"])
    assert result.exit_code == 1 and "not in the schema" in result.output

    conn = connect()
    assert store.get(conn, "initech") is None
    conn.close()


def test_schema_only_with_custom_and_custom_needs_schema(root, schema_file):
    r1 = CliRunner().invoke(app, ["customer", "add", "a", "--name", "A",
                                  "--workload", "saas_support", "--schema", str(schema_file)])
    r2 = CliRunner().invoke(app, ["customer", "add", "b", "--name", "B", "--workload", "custom"])
    assert r1.exit_code == 2 and r2.exit_code == 2


def test_regulated_custom_workload_forbids_egress_and_text_capture(root, schema_file):
    assert _add_custom(root, schema_file, "clinic", "--regime", "HIPAA").exit_code == 0
    conn = connect()
    ctx = CustomerContext(conn, "clinic")
    assert ctx.profile.compliance.allows_external_api is False
    with pytest.raises(PermissionError):
        importer.openai_labeler(ctx.profile, client=object())
    conn.close()


def test_custom_pipeline_requires_imported_data_then_imports_against_the_custom_schema(
        root, schema_file, tmp_path):
    _add_custom(root, schema_file)
    conn = connect()
    with pytest.raises(pipeline.PipelineError, match="customer import"):
        pipeline.start(conn, "initech")

    src = tmp_path / "t.jsonl"
    src.write_text(json.dumps({"input": "please refund me", "output": REC}) + "\n"
                   + json.dumps({"input": "x", "output": {"intent": "refund"}}) + "\n",
                   encoding="utf-8")
    r = importer.import_corpus(CustomerContext(conn, "initech"), src)
    assert (r["labeled"], r["rejected"]) == (1, 1)
    assert pipeline.start(conn, "initech")["stage"] == "baseline"
    conn.close()


def test_serve_shared_refuses_custom(root):
    result = CliRunner().invoke(app, ["serve-shared", "custom"])
    assert result.exit_code == 2


def test_custom_workload_survives_a_kaggle_packet_round_trip(root, schema_file, tmp_path):
    _add_custom(root, schema_file)
    conn = connect()
    job = pipeline.queue.enqueue(conn, "initech", "baseline", {})
    packet.export_job_packet(conn, "initech", job, tmp_path / "pkt")

    remote = tmp_path / "remote"
    packet.apply_packet(tmp_path / "pkt", repo_root=remote)
    remote_conn = sqlite3.connect(remote / "customers.db")
    remote_conn.row_factory = sqlite3.Row
    import ftspec.config as cfg
    old = cfg.REPO_ROOT
    cfg.REPO_ROOT = remote
    try:
        ctx = CustomerContext(remote_conn, "initech")
        assert ctx.profile.scoring_plan().headline().path == "intent"
    finally:
        cfg.REPO_ROOT = old
        remote_conn.close()
    conn.close()


# --- audit log ------------------------------------------------------------------------

def test_audit_records_actions_without_customer_text_and_survives_delete(root, tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["customer", "add", "acme", "--name", "Acme", "--workload", "saas_support"])
    runner.invoke(app, ["api-key", "create", "acme", "--label", "prod"])
    runner.invoke(app, ["approve", "acme"])
    runner.invoke(app, ["privacy", "set", "acme", "--retention-days", "30"])
    runner.invoke(app, ["customer", "delete", "acme", "--yes"])

    conn = connect()
    rows = audit.entries(conn, "acme")
    assert [r["action"] for r in rows] == ["customer.add", "apikey.create", "deploy.approve",
                                           "privacy.set", "customer.delete"]
    assert all(r["actor"] == "tester" for r in rows)
    assert rows[-1]["detail"]["rows"]["customers"] == 1
    assert "sk-" not in json.dumps([r["detail"] for r in rows])
    conn.close()

    listed = runner.invoke(app, ["audit", "list", "acme"])
    assert listed.exit_code == 0 and "customer.delete" in listed.output


# --- redaction / retention -----------------------------------------------------------

def test_redact_masks_identifiers_but_not_order_ids_or_amounts():
    text = ("mail me at jane.doe@acme.io or call (555) 123-4567, card 4539 1488 0343 6467, "
            "order ORD-49225 for 399.98 on 2026-05-21")
    out = redact(text, privacy.DEFAULT_REDACT)
    assert "jane.doe" not in out and "123-4567" not in out and "4539" not in out
    assert "ORD-49225" in out and "399.98" in out and "2026-05-21" in out


def _capture(ctx, text_in, text_out):
    request = SimpleNamespace(messages=[SimpleNamespace(role="user", content=text_in)])
    capture._write_row(ctx, request, text_out, 10, 20, 5.0)
    rows = (ctx.memory_dir() / "production_log.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(rows[-1])


def test_capture_writes_redacted_text_and_keeps_validity(root):
    conn = connect()
    store.create(conn, "acme", "Acme", "saas_support")
    ctx = CustomerContext(conn, "acme")

    row = _capture(ctx, "I am bob@example.com, refund please", "not json")

    assert "bob@example.com" not in row["input"] and "[REDACTED:email]" in row["input"]
    assert row["contract_valid"] is False

    privacy.save(ctx, redact=())
    assert "bob@example.com" in _capture(ctx, "I am bob@example.com", "x")["input"]
    conn.close()


def test_retention_drops_old_and_undated_rows_only(root):
    conn = connect()
    store.create(conn, "acme", "Acme", "saas_support")
    ctx = CustomerContext(conn, "acme")
    privacy.save(ctx, retention_days=30)
    now = datetime(2026, 9, 24, tzinfo=timezone.utc)
    ctx.memory_dir().mkdir(parents=True)
    rows = [{"timestamp": (now - timedelta(days=d)).isoformat()} for d in (1, 29, 31, 400)]
    rows.append({"no": "timestamp"})
    (ctx.memory_dir() / "production_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    removed = privacy.enforce_retention(ctx, now=now)

    assert removed == {"production_log.jsonl": 3}
    left = (ctx.memory_dir() / "production_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(left) == 2
    conn.close()


def test_privacy_set_rejects_unknown_rules(root):
    conn = connect()
    store.create(conn, "acme", "Acme", "saas_support")
    with pytest.raises(ValueError):
        privacy.save(CustomerContext(conn, "acme"), redact=("telepathy",))
    conn.close()


# --- tenant isolation tripwire -------------------------------------------------------

def _snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def test_every_customer_scoped_path_leaves_other_customers_untouched(root, tmp_path):
    from test_ftplatform_pipeline import FakeGPU

    conn = connect()
    store.create(conn, "acme", "Acme", "saas_support")
    store.create(conn, "globex", "Globex", "saas_support")
    b = CustomerContext(conn, "globex")
    b.memory_dir().mkdir(parents=True)
    (b.memory_dir() / "production_log.jsonl").write_text(
        json.dumps({"timestamp": "2020-01-01T00:00:00+00:00", "input": "GLOBEX SECRET"}) + "\n",
        encoding="utf-8")
    b.imports_dir().mkdir(parents=True)
    (b.imports_dir() / "corpus.jsonl").write_text('{"input": "GLOBEX SECRET"}\n', encoding="utf-8")
    usage.record(conn, "globex", 5, 5)
    b_files = _snapshot(b.customer_root())
    b_rows = {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE customer_id='globex'").fetchone()[0]  # noqa: S608
              for t in ("usage_counters", "jobs", "deployments", "pipelines")}

    # Every customer-scoped operation, for acme only.
    a = CustomerContext(conn, "acme")
    src = tmp_path / "acme.jsonl"
    src.write_text(json.dumps({"input": "acme ticket", "output": json.loads(
        (config_mod.Path(__file__).resolve().parent.parent / "data" / "saas_support" / "eval.jsonl")
        .read_text(encoding="utf-8").splitlines()[0])["messages"][2]["content"]}) + "\n",
        encoding="utf-8")
    importer.import_corpus(a, src)
    approvals.approve(conn, "acme")
    pipeline.start(conn, "acme", stage_a_top_k=1, lora_grid_top_k=1)
    assert pipeline.drive(conn, "acme", run_job=FakeGPU())["status"] == "done"
    _capture(a, "acme user", "x")
    privacy.enforce_retention(a)
    monthly.write(conn, a, period=usage.current_period())
    job = pipeline.queue.enqueue(conn, "acme", "baseline", {})
    packet.export_job_packet(conn, "acme", job, tmp_path / "pkt")

    assert _snapshot(b.customer_root()) == b_files
    assert {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE customer_id='globex'").fetchone()[0]  # noqa: S608
            for t in b_rows} == b_rows
    written = {p.relative_to(root).parts[:2] for p in root.rglob("*")
               if p.is_file() and "customers" == p.relative_to(root).parts[0]}
    assert written <= {("customers", "acme"), ("customers", "globex")}

    pkt_db = sqlite3.connect(tmp_path / "pkt" / packet.PACKET_DB_NAME)
    assert pkt_db.execute("SELECT id FROM customers").fetchall() == [("acme",)]
    assert pkt_db.execute("SELECT COUNT(*) FROM jobs WHERE customer_id != 'acme'").fetchone()[0] == 0
    pkt_db.close()
    pkt_text = "".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in (tmp_path / "pkt").rglob("*") if p.is_file() and p.suffix != ".db")
    assert "GLOBEX SECRET" not in pkt_text

    report = json.loads((a.customer_root() / "reports" /
                         f"monthly-{usage.current_period()}.json").read_text(encoding="utf-8"))
    assert report["usage"]["requests"] == 0  # globex's 1 request is not acme's
    conn.close()
