"""
Retrieval over a customer's own documents -- the agent's knowledge_search tool.

Fine-tuning teaches the model how this customer works (its rules, its
format); facts that change -- prices, FAQ answers, return terms -- belong in
documents looked up at the moment they are needed. That lookup is one more
tool the model can call, so retrieval sits inside the same permission and
audit path as every other action.

Plain BM25 over paragraph chunks, standard library only: no embedding API
(which would send the customer's documents off-box and cost money per
call) and no vector database. For a support knowledge base of a few
hundred pages that is fast enough to rebuild on every search.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

SUPPORTED = (".md", ".txt", ".json")
CHUNK_CHARS = 800
_WORD = re.compile(r"[\w']+", re.UNICODE)


def knowledge_dir(ctx) -> Path:
    return ctx.customer_root() / "agent" / "knowledge"


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return raw
    # A JSON export (a help-centre dump, ABCD's guidelines) -> its string leaves.
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(json.loads(raw))
    return "\n\n".join(out)


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraphs, merged up to `size` characters so a chunk keeps its context."""
    out, cur = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > size * 2:           # one enormous paragraph
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return out


def add(ctx, paths: list[Path]) -> list[str]:
    """Copy documents into the customer's knowledge dir. Returns the names added."""
    d = knowledge_dir(ctx)
    d.mkdir(parents=True, exist_ok=True)
    added = []
    for p in map(Path, paths):
        if p.suffix.lower() not in SUPPORTED:
            raise ValueError(f"{p.name}: unsupported type {p.suffix!r}; use {SUPPORTED}")
        shutil.copy(p, d / p.name)
        added.append(p.name)
    return added


def _corpus(ctx) -> list[dict]:
    d = knowledge_dir(ctx)
    if not d.exists():
        return []
    docs = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in SUPPORTED:
            for i, c in enumerate(chunks(_text_of(p))):
                docs.append({"source": f"{p.name}#{i}", "text": c, "tokens": _tokens(c)})
    return docs


def search(ctx, query: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """Top-k chunks by BM25; [] when nothing shares a word with the query."""
    docs = _corpus(ctx)
    q = set(_tokens(query))
    if not docs or not q:
        return []
    n = len(docs)
    avg = sum(len(d["tokens"]) for d in docs) / n
    df = Counter(t for d in docs for t in set(d["tokens"]) if t in q)
    scored = []
    for d in docs:
        tf = Counter(t for t in d["tokens"] if t in q)
        score = 0.0
        for term, f in tf.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d["tokens"]) / avg))
        if score > 0:
            scored.append({"source": d["source"], "text": d["text"], "score": round(score, 3)})
    scored.sort(key=lambda r: -r["score"])
    return scored[:k]
