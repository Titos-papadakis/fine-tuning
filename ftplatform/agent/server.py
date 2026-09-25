"""
HTTP front for one customer's agent: chat sessions and action approvals.

Sits in front of the customer's model server (`ftplatform serve`, which
enforces the step schema) and talks to it over its OpenAI-compatible API,
so the model server is unchanged and each can be restarted on its own.

    POST /v1/agent/sessions                          -> {"session_id"}
    POST /v1/agent/sessions/{id}/messages {"text"}   -> replies, actions, pending approvals
    GET  /v1/agent/sessions/{id}                     -> the conversation so far
    GET  /v1/agent/actions?status=pending_approval   -> what is waiting on a human
    POST /v1/agent/actions/{id}/approve | /reject

Every route requires one of this customer's API keys when a resolver is
given (the default from the CLI), the same keys as the model server.
"""
from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from ftplatform.agent import executor
from ftplatform.agent import session as session_mod
from ftplatform.agent import tools as tools_mod


class Message(BaseModel):
    """Module level on purpose: with postponed annotations FastAPI resolves
    the body type by name, and a class local to create_app() is invisible to
    it -- the body is then silently read as a query parameter."""
    text: str


def create_app(conn, ctx, model_fn: Callable[[str], str],
               api_key_resolver: Callable[[str], str | None] | None = None,
               http_post: Callable | None = None):
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title=f"agent · {ctx.customer.id}")

    def authorize(authorization: str | None) -> None:
        if api_key_resolver is None:
            return
        key = (authorization or "").removeprefix("Bearer ").strip()
        if not key or api_key_resolver(key) != ctx.customer.id:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def load_session(session_id: str) -> dict:
        try:
            return session_mod.load(ctx, session_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/health")
    async def health():
        return {"status": "ok", "customer": ctx.customer.id}

    @app.post("/v1/agent/sessions")
    async def create_session(authorization: str | None = Header(None)):
        authorize(authorization)
        return {"session_id": session_mod.new_session(ctx)["id"]}

    @app.get("/v1/agent/sessions/{session_id}")
    async def get_session(session_id: str, authorization: str | None = Header(None)):
        authorize(authorization)
        return load_session(session_id)

    @app.post("/v1/agent/sessions/{session_id}/messages")
    async def post_message(session_id: str, msg: Message,
                           authorization: str | None = Header(None)):
        authorize(authorization)
        s = load_session(session_id)
        catalog = tools_mod.load(ctx)       # re-read: a permission change applies at once
        return session_mod.respond(conn, ctx, s, msg.text, model_fn, catalog, http_post=http_post)

    @app.get("/v1/agent/actions")
    async def list_actions(status: str | None = None, limit: int = 50,
                           authorization: str | None = Header(None)):
        authorize(authorization)
        return executor.list_actions(conn, ctx.customer.id, status=status, limit=limit)

    async def _decide(action_id: str, approve: bool, authorization: str | None):
        authorize(authorization)
        try:
            row = executor.decide(conn, ctx, tools_mod.load(ctx), action_id, approve,
                                  http_post=http_post)
        except executor.NotPendingError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        session_mod.apply_decision(ctx, row)
        return row

    @app.post("/v1/agent/actions/{action_id}/approve")
    async def approve(action_id: str, authorization: str | None = Header(None)):
        return await _decide(action_id, True, authorization)

    @app.post("/v1/agent/actions/{action_id}/reject")
    async def reject(action_id: str, authorization: str | None = Header(None)):
        return await _decide(action_id, False, authorization)

    return app
