"""
ABCD (Action-Based Conversations Dataset, ASAPP Research, MIT licence) as an
agent workload: https://github.com/asappresearch/abcd

About 10k customer-service conversations in which the agent follows a
written company policy and takes actions (pull up an account, validate a
purchase, offer a refund) with the values the customer gave. That is the
closest public stand-in for "a company's support agents and their rules".

Each conversation contributes ONE decision point -- the conversation up to
some turn, and what the agent did next (reply / action / wait) -- tagged
with its conversation id as the group, so ftspec.data.build.split_by_group
never puts one conversation in both train and eval. The step type is drawn
40/40/20 (action/reply/wait) so actions, the part that matters, are well
represented in both training and the benchmark.

Outputs, all under out_dir:
    corpus.jsonl   rows for `ftplatform customer import`
    tools.json     the tool catalog (mock handlers whose result text is the
                   most common ABCD wording for that action)
    schema.json    tools.step_schema(...) for `customer add --workload custom`
    knowledge/     ABCD's agent guidelines, for the knowledge_search tool
"""
from __future__ import annotations

import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ftplatform.agent import tools as tools_mod
from ftplatform.agent.session import render_transcript

STEP_WEIGHTS = {"action": 0.4, "reply": 0.4, "wait": 0.2}
# Actions that move money or change what a customer is sold: a human signs off.
APPROVAL_TOOLS = {"offer-refund", "make-purchase", "update-order"}
CATEGORY_TEXT = {"kb_query": "Look up or verify", "interaction": "Change or record",
                 "faq_policy": "Answer from the FAQ / policy"}


def _turns(conv: dict) -> list[dict]:
    """Aligns ABCD's original text with its delexed action targets."""
    out = []
    for (speaker, text), dx in zip(conv["original"], conv["delexed"], strict=True):
        if speaker == "action":
            out.append({"role": "action", "tool": dx["targets"][2],
                        "args": [str(v) for v in dx["targets"][3]], "result": text})
        else:
            out.append({"role": speaker, "text": text})
    return out


def _template(text: str, args: list) -> str:
    t = text
    for i, a in enumerate(args):
        if a:
            t = re.sub(re.escape(a), f"{{{i}}}", t, flags=re.IGNORECASE)
    return t


def build_catalog(convs: list[dict], ontology: dict, kb: dict) -> dict:
    templates: dict = defaultdict(Counter)
    for conv in convs:
        for t in _turns(conv):
            if t["role"] == "action":
                templates[t["tool"]][_template(t["result"], t["args"])] += 1
    category_of = {a: cat for cat, acts in ontology["actions"].items() for a in acts}
    catalog_tools = []
    for name in sorted(templates):
        slots = ontology["actions"].get(category_of.get(name, ""), {}).get(name, [])
        catalog_tools.append({
            "name": name,
            "description": CATEGORY_TEXT.get(category_of.get(name), "Action"),
            "args": " / ".join(slots),
            "permission": "approval" if name in APPROVAL_TOOLS else "auto",
            "handler": {"type": "mock", "template": templates[name].most_common(1)[0][0]},
        })
    policy = "; ".join(f"{flow}: {' > '.join(seq)}" for flow, seq in sorted(kb.items()))
    return tools_mod.validate({"mode": "dry_run", "tools": catalog_tools,
                               "policy": "Required actions per intent, in order -- " + policy})


def decision_points(turns: list[dict]) -> dict[str, list[int]]:
    """Positions p where turns[:p] is the history and turns[p] is the target."""
    points: dict = {"action": [], "reply": [], "wait": []}
    seen_customer = False
    for p in range(1, len(turns)):
        seen_customer = seen_customer or turns[p - 1]["role"] == "customer"
        if not seen_customer:
            continue
        role = turns[p]["role"]
        if role == "action":
            points["action"].append(p)
        elif role == "agent":
            points["reply"].append(p)
        elif turns[p - 1]["role"] in ("agent", "action"):
            points["wait"].append(p)
    return points


def target(turn: dict, intent: str) -> dict:
    if turn["role"] == "action":
        return {"intent": intent, "next_step": "action", "tool": turn["tool"],
                "args": turn["args"], "reply": None}
    if turn["role"] == "agent":
        return {"intent": intent, "next_step": "reply", "tool": None, "args": [],
                "reply": turn["text"]}
    return {"intent": intent, "next_step": "wait", "tool": None, "args": [], "reply": None}


def build(src_dir: Path, out_dir: Path, max_rows: int | None = None, seed: int = 0) -> dict:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    data = json.load(gzip.open(src_dir / "abcd_v1.1.json.gz", "rt", encoding="utf-8"))
    ontology = json.loads((src_dir / "ontology.json").read_text(encoding="utf-8"))
    kb = json.loads((src_dir / "kb.json").read_text(encoding="utf-8"))
    convs = [c for split in ("train", "dev", "test") for c in data[split]]

    catalog = build_catalog(convs, ontology, kb)
    intents = sorted({c["scenario"]["subflow"] for c in convs})
    schema = tools_mod.step_schema(catalog, intents=intents)

    rng = random.Random(seed)
    rng.shuffle(convs)
    rows, kinds, used = [], Counter(), set()
    for conv in convs:
        turns = _turns(conv)
        points = {k: v for k, v in decision_points(turns).items() if v}
        if not points:
            continue
        # Two conversations can open identically ("Customer: hi"); an input
        # already taken would be a train/eval duplicate, so draw again.
        for _ in range(5):
            kinds_here = list(points)
            kind = rng.choices(kinds_here, weights=[STEP_WEIGHTS[k] for k in kinds_here])[0]
            p = rng.choice(points[kind])
            text = render_transcript(turns[:p])
            if " ".join(text.split()).lower() not in used:
                break
        else:
            continue
        used.add(" ".join(text.split()).lower())
        rows.append({"input": text,
                     "output": target(turns[p], conv["scenario"]["subflow"]),
                     "group": str(conv["convo_id"])})
        kinds[kind] += 1
        if max_rows and len(rows) >= max_rows:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "tools.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    if (src_dir / "guidelines.json").exists():
        (out_dir / "knowledge").mkdir(exist_ok=True)
        (out_dir / "knowledge" / "abcd_guidelines.json").write_bytes(
            (src_dir / "guidelines.json").read_bytes())
    return {"rows": len(rows), "steps": dict(kinds), "tools": len(catalog["tools"]),
            "intents": len(intents), "out_dir": str(out_dir)}
