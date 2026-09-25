"""
The only path from a model's decision to a customer's systems.

Order of checks for every tool call the model makes:

    unknown tool          -> refused (the schema makes this rare, not impossible
                             once a catalog changes under a deployed model)
    permission forbidden  -> refused, recorded
    permission approval   -> recorded as pending; a human approves or rejects
    catalog mode dry_run  -> recorded, not executed (read-only knowledge
                             search still runs: it touches nothing)
    otherwise             -> executed through the tool's handler

Every call becomes a row in agent_actions (the customer's own record of what
their agent did, arguments included) and an audit_log entry (tool, status and
ids only -- arguments can carry personal data, the audit log never does).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from ftplatform import audit
from ftplatform.agent import knowledge
from ftplatform.agent import tools as tools_mod

HTTP_TIMEOUT_S = 15
PENDING = "pending_approval"
READ_ONLY_HANDLERS = {"knowledge_search"}
KNOWLEDGE_RESULT_CHARS = 900


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_template(template: str, args: list) -> str:
    """'{0}' -> args[0]; a placeholder with no matching argument becomes ''."""
    return re.sub(r"\{(\d+)\}", lambda m: args[int(m.group(1))] if int(m.group(1)) < len(args) else "",
                  template).strip()


def _http_post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:        # noqa: S310
        body = resp.read().decode("utf-8") or "{}"
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"result": body}


def run_handler(ctx, tool: dict, args: list, session_id: str,
                http_post: Callable[[str, dict, dict], dict] | None = None) -> str:
    handler = tool["handler"]
    kind = handler["type"]
    if kind == "mock":
        template = handler.get("template")
        return render_template(template, args) if template else f"{tool['name']} done."
    if kind == "knowledge_search":
        hits = knowledge.search(ctx, " ".join(args))
        if not hits:
            return "No matching document found."
        text = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        return text[:KNOWLEDGE_RESULT_CHARS]
    # http: the customer's own endpoint (or a connector) does the work.
    headers = {}
    if handler.get("secret_env"):
        secret = os.environ.get(handler["secret_env"])
        if not secret:
            raise RuntimeError(f"environment variable {handler['secret_env']} is not set")
        headers["Authorization"] = f"Bearer {secret}"
    body = (http_post or _http_post)(handler["url"], {
        "tool": tool["name"], "args": args, "session_id": session_id,
        "customer_id": ctx.customer.id}, headers)
    return str(body.get("result", json.dumps(body, ensure_ascii=False)))


def _dry_run_result(tool: dict, args: list) -> str:
    template = tool["handler"].get("template") if tool["handler"]["type"] == "mock" else None
    note = f"[dry run: {tool['name']} recorded, not executed]"
    return f"{render_template(template, args)} {note}" if template else note


def _insert(conn, ctx, session_id, tool_name, args, status, result) -> dict:
    row = {"id": uuid.uuid4().hex[:12], "customer_id": ctx.customer.id, "session_id": session_id,
           "tool": tool_name, "args": list(args), "status": status, "result": result,
           "created_at": _now()}
    conn.execute(
        "INSERT INTO agent_actions (id, customer_id, session_id, tool, args_json, status, result, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (row["id"], row["customer_id"], session_id, tool_name,
         json.dumps(row["args"], ensure_ascii=False), status, result, row["created_at"]))
    conn.commit()
    audit.record(conn, "agent.action", ctx.customer.id,
                 {"action_id": row["id"], "tool": tool_name, "status": status,
                  "session_id": session_id})
    return row


def execute(conn: sqlite3.Connection, ctx, catalog: dict, session_id: str, tool_name: str,
            args: list, http_post: Callable | None = None) -> dict:
    """Apply the checks above to one call and record it. Returns the row,
    whose `result` is what the model is shown as the tool's output."""
    args = [str(a) for a in (args or [])]
    tool = tools_mod.tool(catalog, tool_name)
    if tool is None:
        return _insert(conn, ctx, session_id, tool_name, args, "unknown_tool",
                       f"No tool named {tool_name!r} is available.")
    if tool["permission"] == "forbidden":
        return _insert(conn, ctx, session_id, tool_name, args, "forbidden",
                       "This action is not permitted for the assistant.")
    if tool["permission"] == "approval":
        return _insert(conn, ctx, session_id, tool_name, args, PENDING,
                       "Waiting for approval from a staff member.")
    if catalog["mode"] == "dry_run" and tool["handler"]["type"] not in READ_ONLY_HANDLERS:
        return _insert(conn, ctx, session_id, tool_name, args, "dry_run",
                       _dry_run_result(tool, args))
    try:
        result = run_handler(ctx, tool, args, session_id, http_post)
        status = "executed"
    except Exception as e:                                               # noqa: BLE001
        result, status = f"The action failed: {type(e).__name__}.", "failed"
    return _insert(conn, ctx, session_id, tool_name, args, status, result)


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["args"] = json.loads(d.pop("args_json"))
    return d


def list_actions(conn: sqlite3.Connection, customer_id: str, status: str | None = None,
                 limit: int = 50) -> list[dict]:
    sql, params = "SELECT * FROM agent_actions WHERE customer_id = ?", [customer_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    params.append(limit)
    return [_row(r) for r in conn.execute(sql, params)]


def get(conn: sqlite3.Connection, customer_id: str, action_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM agent_actions WHERE id = ? AND customer_id = ?",
                     (action_id, customer_id)).fetchone()
    return _row(r) if r else None


class NotPendingError(ValueError):
    pass


def decide(conn: sqlite3.Connection, ctx, catalog: dict, action_id: str, approve: bool,
           http_post: Callable | None = None) -> dict:
    """A human's approve/reject of a pending call. Approving runs it (or, in
    dry_run mode, records it as a dry run) -- the same handler path as auto."""
    row = get(conn, ctx.customer.id, action_id)
    if row is None or row["status"] != PENDING:
        raise NotPendingError(f"no pending action {action_id!r} for {ctx.customer.id!r}")
    tool = tools_mod.tool(catalog, row["tool"])
    if not approve:
        status, result = "rejected", "A staff member declined this action."
    elif tool is None:
        status, result = "unknown_tool", f"No tool named {row['tool']!r} is available any more."
    elif catalog["mode"] == "dry_run" and tool["handler"]["type"] not in READ_ONLY_HANDLERS:
        status, result = "dry_run", _dry_run_result(tool, row["args"])
    else:
        try:
            result = run_handler(ctx, tool, row["args"], row["session_id"], http_post)
            status = "executed"
        except Exception as e:                                           # noqa: BLE001
            status, result = "failed", f"The action failed: {type(e).__name__}."
    conn.execute("UPDATE agent_actions SET status = ?, result = ?, decided_at = ?, decided_by = ? "
                 "WHERE id = ?", (status, result, _now(), audit.actor(), action_id))
    conn.commit()
    audit.record(conn, "agent.approve" if approve else "agent.reject", ctx.customer.id,
                 {"action_id": action_id, "tool": row["tool"], "status": status})
    return get(conn, ctx.customer.id, action_id)
