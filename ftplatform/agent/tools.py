"""
A customer's tool catalog, and the step schema the model is trained to emit.

customers/<id>/agent/tools.json:

    {"mode": "dry_run" | "live",
     "policy": "free text: the business rules, shown to prompted baselines",
     "tools": [{"name": "offer-refund",
                "description": "Refund an order.",
                "args": "amount",
                "permission": "auto" | "approval" | "forbidden",
                "handler": {"type": "mock", "template": "Refund of {0} issued."}
                         | {"type": "http", "url": "https://...", "secret_env": "ACME_TOOLS_TOKEN"}
                         | {"type": "knowledge_search"}}]}

A new customer starts in dry_run: the agent decides and records every
action, nothing reaches their systems until they have watched it and
switched to live. Write-back into a CRM (set fields, add tags, route to a
queue, add an internal note) is the same thing as any other tool here: an
http handler pointed at the customer's endpoint or a connector.
"""
from __future__ import annotations

import json
from pathlib import Path

PERMISSIONS = ("auto", "approval", "forbidden")
MODES = ("dry_run", "live")
HANDLER_TYPES = ("mock", "http", "knowledge_search")
NEXT_STEPS = ("reply", "action", "wait")
STEP_PROMPT = "You are the support agent. Decide the next step for this conversation as JSON."


class CatalogError(ValueError):
    pass


def validate(catalog: dict) -> dict:
    """Normalises and checks a catalog; raises CatalogError naming the problem."""
    mode = catalog.get("mode", "dry_run")
    if mode not in MODES:
        raise CatalogError(f"mode must be one of {MODES}, got {mode!r}")
    tools = catalog.get("tools")
    if not isinstance(tools, list) or not tools:
        raise CatalogError("catalog needs a non-empty 'tools' list")
    seen: set = set()
    out = []
    for i, t in enumerate(tools):
        name = t.get("name")
        if not name or not isinstance(name, str):
            raise CatalogError(f"tools[{i}] has no name")
        if name in seen:
            raise CatalogError(f"tool {name!r} is listed twice")
        seen.add(name)
        permission = t.get("permission", "approval")
        if permission not in PERMISSIONS:
            raise CatalogError(f"tool {name!r}: permission must be one of {PERMISSIONS}")
        handler = t.get("handler") or {"type": "mock"}
        if handler.get("type") not in HANDLER_TYPES:
            raise CatalogError(f"tool {name!r}: handler type must be one of {HANDLER_TYPES}")
        if handler["type"] == "http" and not handler.get("url"):
            raise CatalogError(f"tool {name!r}: http handler needs a url")
        out.append({"name": name, "description": t.get("description", ""),
                    "args": t.get("args", ""), "permission": permission, "handler": handler})
    return {"mode": mode, "policy": catalog.get("policy", ""), "tools": out}


def catalog_path(ctx) -> Path:
    return ctx.customer_root() / "agent" / "tools.json"


def load(ctx) -> dict:
    p = catalog_path(ctx)
    if not p.exists():
        raise FileNotFoundError(f"{ctx.customer.id!r} has no tool catalog -- "
                                f"run `ftplatform agent tools install {ctx.customer.id} tools.json`")
    return validate(json.loads(p.read_text(encoding="utf-8")))


def save(ctx, catalog: dict) -> dict:
    catalog = validate(catalog)
    p = catalog_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return catalog


def tool(catalog: dict, name: str) -> dict | None:
    return next((t for t in catalog["tools"] if t["name"] == name), None)


def step_schema(catalog: dict, intents: list | None = None) -> dict:
    """The JSON Schema of one agent step, for `customer add --workload custom`.

    Tool descriptions and the policy go into the schema's descriptions, so a
    prompted baseline is shown the same catalog and rules the fine-tuned
    model learned -- the comparison stays fair."""
    catalog = validate(catalog)
    tool_lines = "; ".join(
        f"{t['name']}({t['args']})" + (f": {t['description']}" if t["description"] else "")
        for t in catalog["tools"])
    props: dict = {}
    required = []
    if intents:
        props["intent"] = {"type": "string", "enum": sorted(set(intents)),
                           "description": "What the customer needs, once it is clear."}
        required.append("intent")
    props["next_step"] = {
        "type": "string", "enum": list(NEXT_STEPS),
        "description": "reply: write to the customer. action: call one tool. "
                       "wait: stop until the customer writes again."}
    props["tool"] = {
        "anyOf": [{"type": "string", "enum": [t["name"] for t in catalog["tools"]]},
                  {"type": "null"}],
        "description": "The tool to call when next_step is action, else null. Tools: " + tool_lines
                       + (f". Policy: {catalog['policy']}" if catalog["policy"] else "")}
    props["args"] = {"type": "array", "items": {"type": "string"},
                     "description": "The tool's arguments in order, copied from the conversation; "
                                    "[] unless next_step is action."}
    props["reply"] = {"anyOf": [{"type": "string"}, {"type": "null"}],
                      "description": "The message to the customer when next_step is reply, else null."}
    required += ["next_step", "tool", "args", "reply"]
    return {"title": "AgentStep", "type": "object", "additionalProperties": False,
            "properties": props, "required": required,
            # Read by ftspec.core.registry.read_custom_schema, never shown to a model.
            "x-ftspec": {"headline_field": "tool", "free_text_fields": ["reply"],
                         "short_prompt": STEP_PROMPT}}


def consistency_error(step: dict) -> str:
    """A schema-valid step can still contradict itself (an action with no tool).
    Returns why, or '' when the step is usable."""
    ns = step.get("next_step")
    if ns == "action" and not step.get("tool"):
        return "next_step is action but no tool was named"
    if ns == "reply" and not (step.get("reply") or "").strip():
        return "next_step is reply but the reply is empty"
    return ""
