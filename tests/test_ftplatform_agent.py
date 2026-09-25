"""
The agent workload: step schema, tool permissions and dry-run, the respond
loop and its stop conditions, knowledge search, the HTTP API, retention,
the ABCD converter -- plus the conversation-grouped split it relies on.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import ftspec.config as config_mod
from ftplatform import audit, privacy
from ftplatform import db as db_mod
from ftplatform.agent import abcd, executor, knowledge, server
from ftplatform.agent import session as session_mod
from ftplatform.agent import tools as tools_mod
from ftplatform.auth import keys
from ftplatform.cli import app
from ftplatform.customers import importer, store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect
from ftspec.core.profile import Sample
from ftspec.core.registry import load_profile
from ftspec.data.build import load_external, split_by_group

CATALOG = {
    "mode": "live",
    "policy": "Pull up the account before any refund.",
    "tools": [
        {"name": "pull-up-account", "args": "customer_name", "permission": "auto",
         "handler": {"type": "mock", "template": "Account has been pulled up for {0}."}},
        {"name": "offer-refund", "args": "amount", "permission": "approval",
         "handler": {"type": "mock", "template": "A refund of ${0} has been made."}},
        {"name": "delete-account", "permission": "forbidden", "handler": {"type": "mock"}},
        {"name": "crm-update", "permission": "auto",
         "handler": {"type": "http", "url": "https://crm.example/hook", "secret_env": "CRM_TOKEN"}},
        {"name": "search-kb", "permission": "auto", "handler": {"type": "knowledge_search"}},
    ],
}


def step(next_step, tool=None, args=(), reply=None, intent="refund"):
    return json.dumps({"intent": intent, "next_step": next_step, "tool": tool,
                       "args": list(args), "reply": reply})


def scripted(*outputs):
    """A model that returns these steps in order and records what it was shown."""
    seen: list[str] = []
    it = iter(outputs)

    def model(transcript):
        seen.append(transcript)
        return next(it)
    model.seen = seen
    return model


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "customers.db")
    monkeypatch.setenv("FTPLATFORM_ACTOR", "tester")
    return tmp_path


@pytest.fixture
def agent(root):
    """A custom-workload customer whose schema is the step schema of CATALOG."""
    schema = root / "agent_schema.json"
    schema.write_text(json.dumps(tools_mod.step_schema(CATALOG, intents=["refund", "other"])),
                      encoding="utf-8")
    result = CliRunner().invoke(app, ["customer", "add", "acme", "--name", "Acme",
                                      "--workload", "custom", "--schema", str(schema)])
    assert result.exit_code == 0, result.output
    conn = connect()
    ctx = CustomerContext(conn, "acme")
    tools_mod.save(ctx, CATALOG)
    yield conn, ctx
    conn.close()


# --- grouped split -------------------------------------------------------------

def _samples(groups):
    return [Sample(source_text=f"t{i}", record={}, meta={"group": g} if g else {})
            for i, g in enumerate(groups)]


def test_split_by_group_never_puts_a_group_on_both_sides():
    samples = _samples(["a", "a", "b", "b", "b", "c", "d", "d", "e"])
    ev, val, train = split_by_group(samples, n_eval=3, n_val=1)
    groups = [{s.meta["group"] for s in part} for part in (ev, val, train)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(ev) + len(val) + len(train) == len(samples)


def test_split_without_groups_is_the_old_row_split():
    samples = _samples([None] * 10)
    ev, val, train = split_by_group(samples, n_eval=2, n_val=3)
    assert ev == samples[:2] and val == samples[2:5] and train == samples[5:]


def test_group_reaches_meta_from_flat_rows_and_from_import(agent, tmp_path):
    conn, ctx = agent
    rec = json.loads(step("wait"))
    flat = tmp_path / "flat.jsonl"
    flat.write_text(json.dumps({"input": "Customer: hi", "output": rec, "group": 7}) + "\n",
                    encoding="utf-8")
    assert load_external(flat, ctx.profile)[0].meta["group"] == "7"

    csv_path = tmp_path / "tickets.csv"
    csv_path.write_text("text,output,conversation_id\n"
                        f"Customer: hello,\"{json.dumps(rec).replace(chr(34), chr(34) * 2)}\",c-1\n",
                        encoding="utf-8")
    importer.import_corpus(ctx, csv_path)
    row = json.loads(ctx.imported_corpus_path().read_text(encoding="utf-8").splitlines()[0])
    assert row["meta"]["group"] == "c-1" and row["meta"]["label_source"] == "customer"


# --- schema and catalog -----------------------------------------------------------

def test_schema_settings_travel_in_the_file_and_never_reach_the_prompt(root):
    path = root / "s.json"
    path.write_text(json.dumps(tools_mod.step_schema(CATALOG)), encoding="utf-8")
    profile = load_profile("custom", schema_path=path)
    assert profile.headline_field == "tool"
    assert profile.free_text_fields == ("reply",)
    assert profile.prompts.short == tools_mod.STEP_PROMPT
    assert "x-ftspec" not in profile.prompts.schema
    assert "Pull up the account before any refund." in profile.prompts.schema
    assert "reply" not in profile.scoring_plan().record_match_paths


def test_customer_add_picks_up_declared_settings(agent):
    _, ctx = agent
    assert ctx.profile.headline_field == "tool"
    assert ctx.profile.free_text_fields == ("reply",)


def test_step_contract_and_consistency(agent):
    _, ctx = agent
    ok, _ = ctx.profile.contract.validate(step("action", "pull-up-account", ["joe"]))
    assert ok is not None and tools_mod.consistency_error(ok) == ""
    assert ctx.profile.contract.validate(step("action", "launch-missiles"))[0] is None
    no_tool, _ = ctx.profile.contract.validate(step("action"))
    assert "no tool" in tools_mod.consistency_error(no_tool)
    empty, _ = ctx.profile.contract.validate(step("reply", reply="  "))
    assert "empty" in tools_mod.consistency_error(empty)


@pytest.mark.parametrize("bad, message", [
    ({"tools": []}, "non-empty"),
    ({"mode": "yolo", "tools": [{"name": "x"}]}, "mode"),
    ({"tools": [{"name": "x"}, {"name": "x"}]}, "twice"),
    ({"tools": [{"name": "x", "permission": "sometimes"}]}, "permission"),
    ({"tools": [{"name": "x", "handler": {"type": "http"}}]}, "url"),
])
def test_catalog_validation(bad, message):
    with pytest.raises(tools_mod.CatalogError, match=message):
        tools_mod.validate(bad)


def test_unknown_tools_default_to_approval():
    assert tools_mod.validate({"tools": [{"name": "x"}]})["tools"][0]["permission"] == "approval"


# --- executor ----------------------------------------------------------------------

def test_auto_tool_runs_and_is_recorded_without_args_in_audit(agent):
    conn, ctx = agent
    row = executor.execute(conn, ctx, tools_mod.load(ctx), "s1", "pull-up-account", ["Joe Bloggs"])
    assert row["status"] == "executed"
    assert row["result"] == "Account has been pulled up for Joe Bloggs."
    assert executor.get(conn, "acme", row["id"])["args"] == ["Joe Bloggs"]
    entry = [e for e in audit.entries(conn, "acme") if e["action"] == "agent.action"][-1]
    assert entry["detail"]["tool"] == "pull-up-account"
    assert "Joe Bloggs" not in json.dumps(entry)


def test_dry_run_records_but_does_not_call_the_customer(agent):
    conn, ctx = agent
    catalog = {**tools_mod.load(ctx), "mode": "dry_run"}
    calls = []
    row = executor.execute(conn, ctx, catalog, "s1", "crm-update", ["tag:vip"],
                           http_post=lambda *a: calls.append(a) or {})
    assert row["status"] == "dry_run" and not calls
    mock = executor.execute(conn, ctx, catalog, "s1", "pull-up-account", ["Joe"])
    assert mock["status"] == "dry_run" and mock["result"].startswith("Account has been pulled up for Joe.")


def test_forbidden_and_unknown_tools_are_refused(agent):
    conn, ctx = agent
    catalog = tools_mod.load(ctx)
    assert executor.execute(conn, ctx, catalog, "s1", "delete-account", [])["status"] == "forbidden"
    assert executor.execute(conn, ctx, catalog, "s1", "nope", [])["status"] == "unknown_tool"


def test_approval_waits_then_runs_on_approve(agent):
    conn, ctx = agent
    catalog = tools_mod.load(ctx)
    row = executor.execute(conn, ctx, catalog, "s1", "offer-refund", ["40"])
    assert row["status"] == executor.PENDING
    assert [a["id"] for a in executor.list_actions(conn, "acme", status=executor.PENDING)] == [row["id"]]
    done = executor.decide(conn, ctx, catalog, row["id"], approve=True)
    assert done["status"] == "executed" and done["result"] == "A refund of $40 has been made."
    assert done["decided_by"] == "tester"
    with pytest.raises(executor.NotPendingError):
        executor.decide(conn, ctx, catalog, row["id"], approve=True)


def test_reject_never_runs_the_tool(agent):
    conn, ctx = agent
    catalog = tools_mod.load(ctx)
    row = executor.execute(conn, ctx, catalog, "s1", "offer-refund", ["40"])
    assert executor.decide(conn, ctx, catalog, row["id"], approve=False)["status"] == "rejected"


def test_approval_cannot_cross_customers(agent):
    conn, ctx = agent
    catalog = tools_mod.load(ctx)
    row = executor.execute(conn, ctx, catalog, "s1", "offer-refund", ["40"])
    store.create(conn, "globex", "Globex", "saas_support")
    other = CustomerContext(conn, "globex")
    with pytest.raises(executor.NotPendingError):
        executor.decide(conn, other, catalog, row["id"], approve=True)


def test_http_handler_sends_bearer_secret_and_returns_result(agent, monkeypatch):
    conn, ctx = agent
    sent = {}

    def fake_post(url, payload, headers):
        sent.update(url=url, payload=payload, headers=headers)
        return {"result": "tag added"}
    monkeypatch.setenv("CRM_TOKEN", "s3cret")
    row = executor.execute(conn, ctx, tools_mod.load(ctx), "s1", "crm-update", ["vip"],
                           http_post=fake_post)
    assert row["status"] == "executed" and row["result"] == "tag added"
    assert sent["headers"]["Authorization"] == "Bearer s3cret"
    assert sent["payload"] == {"tool": "crm-update", "args": ["vip"], "session_id": "s1",
                               "customer_id": "acme"}


def test_handler_failure_is_recorded_not_raised(agent, monkeypatch):
    conn, ctx = agent
    monkeypatch.delenv("CRM_TOKEN", raising=False)
    row = executor.execute(conn, ctx, tools_mod.load(ctx), "s1", "crm-update", ["vip"],
                           http_post=lambda *a: {"result": "never"})
    assert row["status"] == "failed" and "RuntimeError" in row["result"]


def test_template_tolerates_missing_args():
    assert executor.render_template("{0} and {1}", ["a"]) == "a and"


# --- knowledge search (RAG) ---------------------------------------------------------

def test_knowledge_search_finds_the_right_document(agent, tmp_path):
    conn, ctx = agent
    (tmp_path / "returns.md").write_text(
        "Returns policy\n\nItems can be returned within 90 days with a receipt.\n\n"
        "Gold members get free return shipping.", encoding="utf-8")
    (tmp_path / "shipping.txt").write_text("Standard shipping takes 5 business days.",
                                           encoding="utf-8")
    assert knowledge.add(ctx, [tmp_path / "returns.md", tmp_path / "shipping.txt"])
    hits = knowledge.search(ctx, "how many days to return an item")
    assert hits[0]["source"].startswith("returns.md") and "90 days" in hits[0]["text"]
    assert knowledge.search(ctx, "zebra") == []
    # read-only, so it runs even while the catalog is in dry_run
    catalog = {**tools_mod.load(ctx), "mode": "dry_run"}
    row = executor.execute(conn, ctx, catalog, "s1", "search-kb", ["shipping days"])
    assert row["status"] == "executed" and "5 business days" in row["result"]


def test_knowledge_rejects_unsupported_files(agent, tmp_path):
    _, ctx = agent
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(ValueError, match="unsupported"):
        knowledge.add(ctx, [tmp_path / "x.exe"])


def test_chunks_split_long_text():
    parts = knowledge.chunks("\n\n".join(["word " * 100] * 5), size=800)
    assert len(parts) > 1 and all(len(p) <= 1600 for p in parts)


# --- the respond loop --------------------------------------------------------------

def test_respond_runs_action_then_replies_then_waits(agent):
    conn, ctx = agent
    s = session_mod.new_session(ctx)
    model = scripted(step("action", "pull-up-account", ["joe bloggs"]),
                     step("reply", reply="Thanks Joe, I have your account."),
                     step("wait"))
    out = session_mod.respond(conn, ctx, s, "Hi, I'm Joe Bloggs", model, tools_mod.load(ctx))
    assert out["stopped"] == "wait" and out["intent"] == "refund"
    assert out["replies"] == ["Thanks Joe, I have your account."]
    assert [a["tool"] for a in out["actions"]] == ["pull-up-account"]
    # the model saw the tool's result before replying -- the same line format as training
    assert model.seen[1].splitlines()[-1] == (
        "Action: pull-up-account(joe bloggs) -> Account has been pulled up for joe bloggs.")
    stored = session_mod.load(ctx, s["id"])
    assert [t["role"] for t in stored["turns"]] == ["customer", "action", "agent"]


def test_respond_stops_on_a_repeated_action(agent):
    conn, ctx = agent
    s = session_mod.new_session(ctx)
    same = step("action", "pull-up-account", ["joe"])
    out = session_mod.respond(conn, ctx, s, "hi", scripted(same, same, same), tools_mod.load(ctx))
    assert out["stopped"] == "repeated_action" and len(out["actions"]) == 1


def test_respond_stops_at_max_steps(agent):
    conn, ctx = agent
    s = session_mod.new_session(ctx)
    replies = [step("reply", reply=f"r{i}") for i in range(10)]
    out = session_mod.respond(conn, ctx, s, "hi", scripted(*replies), tools_mod.load(ctx),
                              max_steps=3)
    assert out["stopped"] == "max_steps" and out["replies"] == ["r0", "r1", "r2"]


@pytest.mark.parametrize("bad", ["not json", step("action")])
def test_invalid_step_falls_back_to_a_safe_reply(agent, bad):
    conn, ctx = agent
    s = session_mod.new_session(ctx)
    out = session_mod.respond(conn, ctx, s, "hi", scripted(bad), tools_mod.load(ctx))
    assert out["stopped"] == "invalid_step" and out["replies"] == [session_mod.FALLBACK_REPLY]
    assert not out["actions"]


def test_pending_approval_is_reported_and_decision_reaches_the_transcript(agent):
    conn, ctx = agent
    s = session_mod.new_session(ctx)
    model = scripted(step("action", "offer-refund", ["40"]),
                     step("reply", reply="I've asked for the refund to be approved."),
                     step("wait"))
    catalog = tools_mod.load(ctx)
    out = session_mod.respond(conn, ctx, s, "refund please", model, catalog)
    assert len(out["pending_approvals"]) == 1
    row = executor.decide(conn, ctx, catalog, out["pending_approvals"][0], approve=True)
    session_mod.apply_decision(ctx, row)
    action_turn = session_mod.load(ctx, s["id"])["turns"][1]
    assert action_turn["status"] == "executed" and "refund of $40" in action_turn["result"]


def test_session_ids_cannot_escape_the_sessions_dir(agent):
    _, ctx = agent
    with pytest.raises(FileNotFoundError):
        session_mod.load(ctx, "../../customers")


# --- HTTP API ------------------------------------------------------------------------

def test_agent_api_end_to_end_with_auth(agent):
    from fastapi.testclient import TestClient

    conn, ctx = agent
    key = keys.create_key(conn, "acme", "test")
    model = scripted(step("action", "offer-refund", ["40"]), step("wait"))
    shared = connect(shared=True)       # as `ftplatform agent serve` opens it
    client = TestClient(server.create_app(shared, ctx, model,
                                          api_key_resolver=keys.build_resolver(shared, {"acme"})))
    assert client.post("/v1/agent/sessions").status_code == 401
    h = {"Authorization": f"Bearer {key}"}
    sid = client.post("/v1/agent/sessions", headers=h).json()["session_id"]
    out = client.post(f"/v1/agent/sessions/{sid}/messages", json={"text": "refund"}, headers=h).json()
    assert "pending_approvals" in out, out
    aid = out["pending_approvals"][0]
    pending = client.get("/v1/agent/actions", params={"status": executor.PENDING}, headers=h).json()
    assert [a["id"] for a in pending] == [aid]
    assert client.post(f"/v1/agent/actions/{aid}/reject", headers=h).json()["status"] == "rejected"
    assert client.post(f"/v1/agent/actions/{aid}/approve", headers=h).status_code == 409
    assert client.get(f"/v1/agent/sessions/{sid}", headers=h).json()["turns"][1]["status"] == "rejected"
    assert client.get("/v1/agent/sessions/nope", headers=h).status_code == 404


# --- retention, deletion ----------------------------------------------------------------

def test_retention_drops_old_agent_sessions(agent):
    _, ctx = agent
    old, new = session_mod.new_session(ctx), session_mod.new_session(ctx)
    p = session_mod.sessions_dir(ctx) / f"{old['id']}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["updated_at"] = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    p.write_text(json.dumps(data), encoding="utf-8")
    assert privacy.enforce_retention(ctx)["agent_sessions"] == 1
    assert not p.exists() and (session_mod.sessions_dir(ctx) / f"{new['id']}.json").exists()


def test_customer_delete_removes_agent_actions(agent, root):
    conn, ctx = agent
    executor.execute(conn, ctx, tools_mod.load(ctx), "s1", "pull-up-account", ["joe"])
    store.delete(conn, "acme", repo_root=root)
    assert conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0] == 0


# --- CLI ------------------------------------------------------------------------------

def test_cli_tools_set_and_actions(agent):
    conn, ctx = agent
    r = CliRunner().invoke(app, ["agent", "tools", "set", "acme", "offer-refund",
                                 "--permission", "forbidden"])
    assert r.exit_code == 0, r.output
    assert tools_mod.tool(tools_mod.load(ctx), "offer-refund")["permission"] == "forbidden"
    r = CliRunner().invoke(app, ["agent", "tools", "set", "acme", "--mode", "nonsense"])
    assert r.exit_code == 1
    executor.execute(conn, ctx, tools_mod.load(ctx), "s1", "pull-up-account", ["joe"])
    r = CliRunner().invoke(app, ["agent", "actions", "acme"])
    assert "pull-up-account(joe)" in r.output


# --- serve keeps its db connection open -------------------------------------------------

def test_a_shared_connection_works_from_a_request_thread(root):
    """A server holds one connection and answers requests on other threads
    (TestClient, a threadpool); the default connection refuses that."""
    conn = connect(shared=True)
    store.create(conn, "acme", "Acme", "saas_support")
    key = keys.create_key(conn, "acme")
    resolver = keys.build_resolver(conn, {"acme"})
    got = []
    t = threading.Thread(target=lambda: got.append(resolver(key)))
    t.start()
    t.join()
    assert got == ["acme"]
    conn.close()

    plain = connect()
    errors = []

    def use():
        try:
            plain.execute("SELECT 1")
        except sqlite3.ProgrammingError as e:
            errors.append(e)
    t = threading.Thread(target=use)
    t.start()
    t.join()
    assert errors, "the default connection should keep sqlite's same-thread check"
    plain.close()


def test_serve_command_does_not_close_the_db_before_serving(agent, monkeypatch, root):
    conn, ctx = agent
    key = keys.create_key(conn, "acme")
    prod = ctx.production_dir()
    (prod / "lora_adapter").mkdir(parents=True)
    (prod / "candidate_meta.json").write_text(json.dumps({"base_model": "base"}), encoding="utf-8")
    seen = {}

    def fake_serve(**kw):
        seen["customer"] = kw["api_key_resolver"](key)
    import ftspec.serving.serve as serve_mod
    monkeypatch.setattr(serve_mod, "serve", fake_serve)
    # the command installs hooks on the global server state; restore it after
    monkeypatch.setattr(serve_mod.STATE, "on_response", serve_mod.STATE.on_response)
    r = CliRunner().invoke(app, ["serve", "acme", "--no-capture"])
    assert r.exit_code == 0, r.output
    assert seen["customer"] == "acme"


# --- ABCD converter ----------------------------------------------------------------------

def _abcd_fixture(d):
    def conv(cid, name, subflow):
        original = [["agent", "Hi, how can I help?"], ["customer", f"I want a refund, I'm {name}"],
                    ["agent", "Let me pull up your account."], ["action", f"Account has been pulled up for {name}."],
                    ["agent", "Done. Anything else?"], ["customer", "No thanks"]]
        targets = [[subflow, "retrieve_utterance", None, [], 0], [subflow, None, None, [], 0],
                   [subflow, "retrieve_utterance", None, [], 0],
                   [subflow, "take_action", "pull-up-account", [name.lower()], 0],
                   [subflow, "retrieve_utterance", None, [], 0], [subflow, None, None, [], 0]]
        return {"convo_id": cid, "scenario": {"subflow": subflow},
                "original": original,
                "delexed": [{"speaker": o[0], "targets": t} for o, t in zip(original, targets, strict=True)]}
    data = {"train": [conv(1, "Joe Bloggs", "refund_initiate"), conv(2, "Ann Lee", "refund_status")],
            "dev": [conv(3, "Max Po", "refund_initiate")], "test": []}
    with gzip.open(d / "abcd_v1.1.json.gz", "wt", encoding="utf-8") as f:
        json.dump(data, f)
    (d / "ontology.json").write_text(json.dumps(
        {"actions": {"interaction": {"pull-up-account": ["customer_name", "account_id"]}}}),
        encoding="utf-8")
    (d / "kb.json").write_text(json.dumps({"refund_initiate": ["pull-up-account"]}), encoding="utf-8")


def test_abcd_build_produces_valid_grouped_rows_and_a_catalog(tmp_path):
    _abcd_fixture(tmp_path)
    r = abcd.build(tmp_path, tmp_path / "out")
    assert r["rows"] == 3 and r["tools"] == 1 and r["intents"] == 2
    catalog = json.loads((tmp_path / "out" / "tools.json").read_text(encoding="utf-8"))
    assert catalog["tools"][0]["handler"]["template"] == "Account has been pulled up for {0}."
    assert "refund_initiate: pull-up-account" in catalog["policy"]
    profile = load_profile("custom", schema_path=tmp_path / "out" / "schema.json")
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["group"] for row in rows} == {"1", "2", "3"}
    assert len({row["input"] for row in rows}) == 3
    for row in rows:
        assert profile.contract.validate(json.dumps(row["output"]))[0] is not None
        assert row["input"].startswith("Agent: Hi, how can I help?\nCustomer:")


def test_abcd_decision_points_skip_before_the_customer_speaks():
    turns = [{"role": "agent", "text": "hi"}, {"role": "agent", "text": "how can I help"},
             {"role": "customer", "text": "refund"}, {"role": "action", "tool": "x", "args": []},
             {"role": "customer", "text": "ok"}]
    assert abcd.decision_points(turns) == {"action": [3], "reply": [], "wait": [4]}


# --- process packs ------------------------------------------------------------------

def test_shipped_packs_are_valid_catalogs_with_disjoint_intents():
    from ftplatform.agent import packs

    shipped = packs.available()
    assert {p["name"] for p in shipped} == set(packs.DEFINITIONS)
    seen: set = set()
    for p in shipped:
        tools_mod.validate({"tools": p["tools"]})
        assert p["intents"] and not (seen & set(p["intents"]))
        seen |= set(p["intents"])
        assert "ABCD" in p["source"]


def test_combine_merges_packs_and_points_tools_at_the_customer():
    from ftplatform.agent import packs

    catalog, intents = packs.combine(["returns_refunds", "product_questions"],
                                     endpoint="https://acme.example/agent/", secret_env="ACME_TOKEN")
    names = [t["name"] for t in catalog["tools"]]
    assert len(names) == len(set(names)) and catalog["mode"] == "dry_run"
    refund = tools_mod.tool(catalog, "offer-refund")
    assert refund["handler"] == {"type": "http", "url": "https://acme.example/agent/offer-refund",
                                 "secret_env": "ACME_TOKEN"}
    assert refund["permission"] == "approval"
    assert tools_mod.tool(catalog, "knowledge-search")["handler"]["type"] == "knowledge_search"
    assert "refund_initiate" in intents
    with pytest.raises(KeyError):
        packs.combine(["nope"])


def test_breakdown_scores_each_pack():
    from ftplatform.agent import packs

    refund = json.loads(step("action", "offer-refund", ["40"], intent="refund_initiate"))
    wait = json.loads(step("wait", intent="recover_password"))
    pairs = [(refund, refund), (refund, {**refund, "args": ["41"]}), (wait, wait), (wait, None)]
    b = packs.breakdown(pairs)
    assert b["returns_refunds"]["action_pct"] == 50.0 and b["returns_refunds"]["next_step_pct"] == 100.0
    assert b["account_access"]["next_step_pct"] == 50.0 and b["account_access"]["action_pct"] is None


def test_packs_export_writes_what_customer_add_needs(root):
    out = root / "acme_agent"
    r = CliRunner().invoke(app, ["agent", "packs", "export", "orders_shipping", "-o", str(out)])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["customer", "add", "acme", "--name", "Acme", "--workload", "custom",
                                 "--schema", str(out / "step_schema.json")])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["agent", "tools", "install", "acme", str(out / "tools.json")])
    assert r.exit_code == 0, r.output


def test_packs_score_pairs_eval_rows_with_raw_outputs(tmp_path):
    from ftplatform.agent import packs

    schema = tmp_path / "s.json"
    catalog = {"tools": [{"name": "offer-refund", "permission": "auto"}]}
    schema.write_text(json.dumps(tools_mod.step_schema(catalog, intents=["refund_initiate"])),
                      encoding="utf-8")
    gold = step("action", "offer-refund", ["40"], intent="refund_initiate")
    (tmp_path / "eval.jsonl").write_text(
        "\n".join(json.dumps({"messages": [{}, {}, {"content": gold}]}) for _ in range(2)),
        encoding="utf-8")
    (tmp_path / "raw.jsonl").write_text(
        json.dumps({"output": gold}) + "\n" + json.dumps({"output": "garbage"}), encoding="utf-8")
    r = CliRunner().invoke(app, ["agent", "packs", "score", str(schema), str(tmp_path / "eval.jsonl"),
                                 str(tmp_path / "raw.jsonl")])
    assert r.exit_code == 0, r.output
    assert "returns_refunds" in r.output and "50.0%" in r.output
    contract = load_profile("custom", schema_path=schema).contract
    assert packs.score_files(tmp_path / "eval.jsonl", tmp_path / "raw.jsonl",
                             contract)["returns_refunds"]["action_ok"] == 1
