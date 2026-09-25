"""
Ready-made process packs: the support processes almost every company runs.

A pack is a starting tool catalog + policy + the intents it covers, for one
process (returns and refunds, order and shipping status, ...). A new customer
picks packs, points the tools at their own endpoints, and fine-tunes on their
own history -- the pack saves the blank page; their history supplies their
exceptions.

The shipped packs (ftplatform/agent/packs/*.json) are built from ABCD
(ASAPP Research, MIT licence): each pack's tools are the actions ABCD's
written policy requires for the pack's intents, plus the handful every
process uses (look the customer up, verify them, escalate, search the FAQ).
Because the demo model is trained on the same data, every pack has a
measured accuracy -- breakdown() splits one benchmark run by pack.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ftplatform.agent import tools as tools_mod

PACKS_DIR = Path(__file__).parent / "packs"
SOURCE = "ABCD, ASAPP Research (MIT licence) -- https://github.com/asappresearch/abcd"

# name -> (title, what it handles, ABCD flows it covers)
DEFINITIONS = {
    "returns_refunds": (
        "Returns & refunds",
        "Returns for size, colour or damage; refund requests, status and changes; price "
        "disputes, wrong charges and promo codes.",
        ("product_defect", "purchase_dispute")),
    "orders_shipping": (
        "Orders & shipping",
        "Where is my order, delivery dates, missing packages, shipping cost; creating, "
        "cancelling, upgrading and changing orders.",
        ("order_issue", "shipping_issue")),
    "account_access": (
        "Account & access",
        "Password and username recovery, two-factor reset; changing name, address, phone "
        "or payment method.",
        ("account_access", "manage_account")),
    "billing_subscriptions": (
        "Billing & subscriptions",
        "Amount and date due, paying a bill, extensions, disputed subscription charges.",
        ("subscription_inquiry",)),
    "product_questions": (
        "Product & policy questions",
        "Answers from the company's own FAQ and policies, and website troubleshooting. "
        "Uses knowledge search over the customer's documents.",
        ("single_item_query", "storewide_query", "troubleshoot_site")),
}
COMMON_TOOLS = ("pull-up-account", "verify-identity", "notify-team", "search-faq")
# ABCD's policy lists no required actions for FAQ-style intents; these are
# the ones its agents actually use to answer them.
EXTRA_TOOLS = {"product_questions": ("select-faq", "search-policy", "search-pricing",
                                     "search-timing", "search-membership")}


def build(convs: list[dict], catalog: dict, kb: dict) -> dict[str, dict]:
    """Packs from ABCD conversations, the full ABCD catalog and its policy."""
    intents_by_flow: dict = defaultdict(set)
    for c in convs:
        intents_by_flow[c["scenario"]["flow"]].add(c["scenario"]["subflow"])
    packs = {}
    for name, (title, description, flows) in DEFINITIONS.items():
        intents = sorted(set().union(*(intents_by_flow[f] for f in flows)))
        needed = set(COMMON_TOOLS).union(EXTRA_TOOLS.get(name, ()),
                                         *(kb.get(i, ()) for i in intents))
        pack_tools = [t for t in catalog["tools"] if t["name"] in needed]
        if name == "product_questions":
            pack_tools.append({"name": "knowledge-search", "description": "Search the company's "
                               "own documents (FAQ, policies) and read the answer.",
                               "args": "query", "permission": "auto",
                               "handler": {"type": "knowledge_search"}})
        policy = "; ".join(f"{i}: {' > '.join(kb[i])}" for i in intents if i in kb)
        packs[name] = {"name": name, "title": title, "description": description,
                       "source": SOURCE, "intents": intents,
                       "policy": f"Required actions per intent, in order -- {policy}" if policy else "",
                       "tools": pack_tools}
    return packs


def write(packs: dict[str, dict], out_dir: Path = PACKS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, pack in packs.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def available(packs_dir: Path = PACKS_DIR) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(packs_dir.glob("*.json"))]


def load(name: str, packs_dir: Path = PACKS_DIR) -> dict:
    p = packs_dir / f"{name}.json"
    if not p.exists():
        names = [x["name"] for x in available(packs_dir)]
        raise KeyError(f"no pack {name!r}; available: {names}")
    return json.loads(p.read_text(encoding="utf-8"))


def combine(names: list[str], endpoint: str | None = None, secret_env: str | None = None,
            packs_dir: Path = PACKS_DIR) -> tuple[dict, list[str]]:
    """(catalog, intents) for a customer taking these packs, in dry_run.

    With `endpoint`, every non-search tool becomes an http tool at
    <endpoint>/<tool-name> -- the customer implements those routes (or a
    connector does); without it the tools stay mock, for a demo."""
    if not names:
        raise ValueError("name at least one pack")
    tools_by_name: dict = {}
    intents: list = []
    policies: list = []
    for name in names:
        pack = load(name, packs_dir)
        for t in pack["tools"]:
            tools_by_name.setdefault(t["name"], json.loads(json.dumps(t)))
        intents += [i for i in pack["intents"] if i not in intents]
        if pack["policy"]:
            policies.append(pack["policy"])
    if endpoint:
        for t in tools_by_name.values():
            if t["handler"]["type"] == "mock":
                t["handler"] = {"type": "http", "url": f"{endpoint.rstrip('/')}/{t['name']}",
                                **({"secret_env": secret_env} if secret_env else {})}
    catalog = tools_mod.validate({"mode": "dry_run", "tools": list(tools_by_name.values()),
                                  "policy": " | ".join(policies)})
    return catalog, intents


def pack_of_intent(packs_dir: Path = PACKS_DIR) -> dict[str, str]:
    return {i: p["name"] for p in available(packs_dir) for i in p["intents"]}


def breakdown(pairs: list[tuple[dict, dict | None]], packs_dir: Path = PACKS_DIR) -> dict:
    """Per-pack accuracy from (gold step, predicted step or None) pairs.

    next_step: reply/action/wait chosen right. action: on steps where the
    right move was a tool call, the right tool with the right arguments."""
    owner = pack_of_intent(packs_dir)
    acc: dict = defaultdict(lambda: {"steps": 0, "next_step_ok": 0, "actions": 0, "action_ok": 0})
    for gold, pred in pairs:
        row = acc[owner.get(gold.get("intent"), "other")]
        row["steps"] += 1
        pred = pred or {}
        row["next_step_ok"] += pred.get("next_step") == gold["next_step"]
        if gold["next_step"] == "action":
            row["actions"] += 1
            row["action_ok"] += (pred.get("tool") == gold["tool"]
                                 and list(pred.get("args") or []) == list(gold["args"]))
    out = {}
    for name, r in sorted(acc.items()):
        out[name] = {**r,
                     "next_step_pct": round(100 * r["next_step_ok"] / r["steps"], 1),
                     "action_pct": round(100 * r["action_ok"] / r["actions"], 1) if r["actions"] else None}
    return out


def score_files(eval_path: Path, raw_path: Path, contract) -> dict:
    """breakdown() over a benchmark run: eval.jsonl (gold, chat format) paired
    in order with raw_<system>.jsonl (one {"output": text} per eval row). An
    output that fails the contract counts as wrong everywhere."""
    gold = [json.loads(json.loads(line)["messages"][2]["content"])
            for line in Path(eval_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = [json.loads(line)["output"]
           for line in Path(raw_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [(g, contract.validate(r)[0]) for g, r in zip(gold, raw, strict=False)]
    return breakdown(pairs)
