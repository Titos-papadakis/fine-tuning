"""
One conversation with a customer's agent, and the loop that drives it.

The model sees the conversation as plain lines -- exactly what it was
trained on (render_transcript is shared with the training-data converter,
so the two can never drift apart):

    Customer: I need to return an item
    Agent: Sure, may I have your name?
    Customer: Crystal Minh
    Action: pull-up-account(crystal minh) -> Account has been pulled up for Crystal Minh.

respond() adds the customer's message, then asks the model for one step at a
time: a reply is sent, an action goes through executor.execute and its
result is appended for the model to read, "wait" hands the turn back. The
loop stops after MAX_STEPS, on an invalid or self-contradictory step, or
when the model repeats the action it just took -- a model must never be
able to loop against a customer's systems.
"""
from __future__ import annotations

import json
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ftplatform.agent import executor
from ftplatform.agent import tools as tools_mod

MAX_STEPS = 8
FALLBACK_REPLY = "Let me check this with a colleague and get back to you."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_turn(turn: dict) -> str:
    role = turn["role"]
    if role == "action":
        return f"Action: {turn['tool']}({' | '.join(turn.get('args') or [])}) -> {turn.get('result', '')}"
    return f"{'Customer' if role == 'customer' else 'Agent'}: {turn['text']}"


def render_transcript(turns: list[dict]) -> str:
    return "\n".join(render_turn(t) for t in turns)


# --- storage: customers/<id>/agent/sessions/<session_id>.json ---------------

def sessions_dir(ctx) -> Path:
    return ctx.customer_root() / "agent" / "sessions"


def new_session(ctx) -> dict:
    s = {"id": uuid.uuid4().hex[:16], "created_at": _now(), "updated_at": _now(), "turns": []}
    save(ctx, s)
    return s


def load(ctx, session_id: str) -> dict:
    if not session_id.isalnum():
        raise FileNotFoundError(f"no session {session_id!r}")
    p = sessions_dir(ctx) / f"{session_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"no session {session_id!r}")
    return json.loads(p.read_text(encoding="utf-8"))


def save(ctx, session: dict) -> None:
    session["updated_at"] = _now()
    d = sessions_dir(ctx)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session['id']}.tmp"
    tmp.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / f"{session['id']}.json")


# --- the model --------------------------------------------------------------

def http_model(url: str, api_key: str | None = None, model: str = "default",
               system_prompt: str | None = None) -> Callable[[str], str]:
    """The customer's deployed model behind `ftplatform serve` (or any
    OpenAI-compatible endpoint that enforces the step schema)."""
    endpoint = url.rstrip("/") + "/v1/chat/completions"

    def call(transcript: str) -> str:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": transcript}]
        req = urllib.request.Request(
            endpoint, method="POST",
            data=json.dumps({"model": model, "messages": messages, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
        with urllib.request.urlopen(req, timeout=120) as resp:            # noqa: S310
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    return call


# --- the loop ---------------------------------------------------------------

def respond(conn, ctx, session: dict, customer_text: str, model_fn: Callable[[str], str],
            catalog: dict, max_steps: int = MAX_STEPS, http_post: Callable | None = None) -> dict:
    """Add the customer's message and run the agent until it waits.

    Returns {"replies", "actions", "intent", "stopped", "pending_approvals"}
    where stopped is "wait", "max_steps", "invalid_step" or "repeated_action"."""
    if customer_text.strip():
        session["turns"].append({"role": "customer", "text": customer_text.strip(), "at": _now()})
    contract = ctx.profile.contract
    replies: list[str] = []
    actions: list[dict] = []
    intent = None
    stopped = "max_steps"
    last_call = None

    for _ in range(max_steps):
        raw = model_fn(render_transcript(session["turns"]))
        step, err = contract.validate(raw)
        if step is None or tools_mod.consistency_error(step):
            stopped = "invalid_step"
            replies.append(FALLBACK_REPLY)
            session["turns"].append({"role": "agent", "text": FALLBACK_REPLY, "at": _now(),
                                     "flag": err or tools_mod.consistency_error(step)})
            break
        intent = step.get("intent", intent)
        if step["next_step"] == "wait":
            stopped = "wait"
            break
        if step["next_step"] == "reply":
            replies.append(step["reply"])
            session["turns"].append({"role": "agent", "text": step["reply"], "at": _now()})
            continue
        call = (step["tool"], tuple(step["args"]))
        if call == last_call:
            stopped = "repeated_action"
            break
        last_call = call
        row = executor.execute(conn, ctx, catalog, session["id"], step["tool"], step["args"],
                               http_post=http_post)
        actions.append(row)
        session["turns"].append({"role": "action", "tool": row["tool"], "args": row["args"],
                                 "result": row["result"], "action_id": row["id"],
                                 "status": row["status"], "at": _now()})
        # A pending call's result tells the model it is waiting on a human; it
        # may still reply ("one moment"), but the call itself goes nowhere
        # until someone runs executor.decide.
    save(ctx, session)
    return {"session_id": session["id"], "replies": replies, "actions": actions,
            "intent": intent, "stopped": stopped,
            "pending_approvals": [a["id"] for a in actions if a["status"] == executor.PENDING]}


def apply_decision(ctx, action: dict) -> None:
    """After a human approves/rejects, show the model the real outcome in
    place of 'waiting for approval', so its next step is based on it."""
    try:
        session = load(ctx, action["session_id"])
    except FileNotFoundError:
        return
    for turn in session["turns"]:
        if turn.get("action_id") == action["id"]:
            turn["result"] = action["result"]
            turn["status"] = action["status"]
    save(ctx, session)
