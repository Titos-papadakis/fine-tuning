"""
ftplatform.customers.importer + `customer delete` -- getting a real customer's
data in, and a customer (with everything they own) back out.
"""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

import ftspec.config as config_mod
from ftplatform.billing import stripe_billing as sb
from ftplatform.candidates import runner
from ftplatform.customers import importer, store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftplatform.jobs import queue

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
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    c = connect(tmp_path / "customers.db")
    store.create(c, "acme", "Acme Inc", "saas_support")
    store.create(c, "globex", "Globex", "saas_support")
    yield c
    c.close()


@pytest.fixture
def ctx(conn):
    return CustomerContext(conn, "acme")


def _csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label_json"])
        w.writeheader()
        for text, label in rows:
            w.writerow({"text": text, "label_json": json.dumps(label) if label else ""})
    return path


def test_csv_import_sorts_rows_into_accepted_rejected_and_to_label(ctx, tmp_path):
    bad = {**VALID, "issue": {**VALID["issue"], "priority": "whenever"}}
    path = _csv(tmp_path / "t.csv", [("ticket one", VALID), ("ticket two", bad),
                                    ("ticket three", None), ("Ticket   ONE", VALID), ("", VALID)])

    r = importer.import_corpus(ctx, path)

    assert (r["read"], r["labeled"], r["rejected"], r["to_label"]) == (5, 1, 1, 1)
    assert (r["duplicate"], r["empty"]) == (1, 1)
    corpus = [json.loads(line) for line in ctx.imported_corpus_path().read_text(encoding="utf-8").splitlines()]
    assert corpus == [{"input": "ticket one", "output": VALID, "meta": {"label_source": "customer"}}]
    rejected = (ctx.imports_dir() / "rejected.jsonl").read_text(encoding="utf-8")
    assert "label invalid" in rejected and "ticket two" in rejected
    assert r["enough_to_train"] is False


def test_jsonl_import_accepts_chat_format_and_flat_shape(ctx, tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([
        json.dumps({"input": "flat", "output": VALID}),
        json.dumps({"messages": [{"role": "system", "content": "s"},
                                 {"role": "user", "content": "chat"},
                                 {"role": "assistant", "content": json.dumps(VALID)}]}),
    ]) + "\n", encoding="utf-8")

    r = importer.import_corpus(ctx, path)

    assert r["labeled"] == 2


def test_auto_labeler_fills_unlabeled_rows_and_writes_a_review_sample(ctx, tmp_path):
    path = _csv(tmp_path / "t.csv", [("a", None), ("b", None), ("c", None)])
    outputs = {"a": json.dumps(VALID), "b": "not json at all", "c": json.dumps(VALID)}

    r = importer.import_corpus(ctx, path, labeler=lambda t: outputs[t], labeler_name="fake")

    assert (r["auto_labeled"], r["rejected"], r["to_label"]) == (2, 1, 0)
    sample = (ctx.imports_dir() / "review_sample.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sample) == 2
    assert json.loads(sample[0])["meta"]["label_source"] == "auto:fake"


def test_auto_label_budget_caps_spend(ctx, tmp_path):
    path = _csv(tmp_path / "t.csv", [(f"t{i}", None) for i in range(5)])
    calls = []

    def labeler(text):
        calls.append(text)
        return json.dumps(VALID)

    r = importer.import_corpus(ctx, path, labeler=labeler, max_auto_label=2)

    assert len(calls) == 2
    assert (r["auto_labeled"], r["to_label"]) == (2, 3)


def test_a_labeler_exception_rejects_that_row_only(ctx, tmp_path):
    path = _csv(tmp_path / "t.csv", [("ok", None), ("boom", None)])

    def labeler(text):
        if text == "boom":
            raise TimeoutError("api down")
        return json.dumps(VALID)

    r = importer.import_corpus(ctx, path, labeler=labeler)

    assert (r["auto_labeled"], r["rejected"]) == (1, 1)


def test_reimport_appends_without_duplicates_and_prunes_to_label(ctx, tmp_path):
    importer.import_corpus(ctx, _csv(tmp_path / "a.csv", [("one", VALID), ("two", None)]))
    r = importer.import_corpus(ctx, _csv(tmp_path / "b.csv", [("one", VALID), ("two", VALID)]))

    assert r["duplicate"] == 1 and r["labeled"] == 1
    assert r["corpus_total"] == 2
    assert r["to_label"] == 0


def test_unsupported_file_type_is_refused(ctx, tmp_path):
    p = tmp_path / "t.xlsx"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        importer.import_corpus(ctx, p)


def test_openai_labeler_refuses_a_regulated_profile():
    profile = SimpleNamespace(name="fintech_disputes",
                              compliance=SimpleNamespace(allows_external_api=False, regime="PCI-DSS"))
    with pytest.raises(PermissionError):
        importer.openai_labeler(profile, client=object())


def test_openai_labeler_uses_the_schema_rubric_prompt(ctx):
    seen = {}

    class Completions:
        def create(self, **kw):
            seen.update(kw)
            msg = SimpleNamespace(content=json.dumps(VALID))
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    label = importer.openai_labeler(ctx.profile, model="gpt-4o-mini", client=client)

    assert json.loads(label("hello")) == VALID
    assert seen["model"] == "gpt-4o-mini" and seen["temperature"] == 0
    assert seen["messages"][0]["content"] == ctx.profile.prompts.variants()["schema+rubric"]


def test_run_baseline_uses_the_imported_corpus(ctx, tmp_path, monkeypatch):
    importer.import_corpus(ctx, _csv(tmp_path / "a.csv", [("one", VALID)]))
    seen = {}

    def fake_build(profile, out_dir, n_train, n_val, n_eval, seed, eval_seed, from_jsonl):
        seen["from_jsonl"] = from_jsonl
        return {}

    monkeypatch.setattr(runner.build, "run", fake_build)
    monkeypatch.setattr(runner.validate_mod, "run", lambda *a, **k: (True, "", {}))
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: {"n_eval": 1, "systems": {}})

    runner.run_baseline(ctx)

    assert seen["from_jsonl"] == ctx.imported_corpus_path()


# --- customer delete -------------------------------------------------------------

def test_delete_removes_every_row_and_file_but_leaves_other_customers(conn, tmp_path):
    ctx = CustomerContext(conn, "acme")
    ctx.memory_dir().mkdir(parents=True)
    (ctx.memory_dir() / "production_log.jsonl").write_text("{}\n", encoding="utf-8")
    other = CustomerContext(conn, "globex")
    other.memory_dir().mkdir(parents=True)
    job_id = queue.enqueue(conn, "acme", "baseline", {})
    (tmp_path / ".kaggle_work" / job_id).mkdir(parents=True)
    queue.enqueue(conn, "globex", "baseline", {})

    result = store.delete(conn, "acme")

    assert result["rows"]["customers"] == 1 and result["rows"]["jobs"] == 1
    assert store.get(conn, "acme") is None
    assert not (tmp_path / "customers" / "acme").exists()
    assert not (tmp_path / ".kaggle_work" / job_id).exists()
    assert store.get(conn, "globex") is not None
    assert other.memory_dir().exists()
    assert len(queue.list_jobs(conn, "globex")) == 1


def test_delete_refuses_a_customer_still_being_billed(conn):
    sb.link_customer(conn, "acme", "a@acme.test", "cus_1")
    sb.record_subscription(conn, "acme", "sub_1", "active")

    with pytest.raises(store.CustomerHasLiveSubscriptionError):
        store.delete(conn, "acme")
    assert store.get(conn, "acme") is not None

    store.delete(conn, "acme", force=True)
    assert store.get(conn, "acme") is None


def test_delete_allows_a_canceled_subscription(conn):
    sb.link_customer(conn, "acme", "a@acme.test", "cus_1")
    sb.record_subscription(conn, "acme", "sub_1", "canceled")

    store.delete(conn, "acme")

    assert store.get(conn, "acme") is None


def test_delete_unknown_customer_raises(conn):
    with pytest.raises(ValueError):
        store.delete(conn, "nobody")
