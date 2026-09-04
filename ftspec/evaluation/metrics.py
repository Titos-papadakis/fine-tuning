"""
Semantic scoring — driven by a profile's scoring plan, not a hardcoded field list.

Why this file exists
--------------------
"% valid JSON" is a solved problem. Grammar-constrained decoding and provider
structured-output modes both guarantee 100% syntactic validity with zero
training. Any benchmark whose headline number is schema adherence is measuring
something available for free.

What decides whether specialization is worth building is whether the extracted
VALUES are right. So every field is scored:

  - exact match on enums, booleans and identifiers
  - numeric match with a cent-level tolerance
  - set F1 on list fields (order-insensitive, partial credit)
  - token F1 on declared free text, where exact match would be a paraphrase test
  - record-level exact match across all non-free-text fields

with 95% bootstrap confidence intervals and McNemar's exact test for paired
comparisons. On a 150-document eval set a 4-point gap between two systems is
frequently noise, and a report that presents it as a win will not survive
contact with the buyer's data scientist.

A prediction that violates the contract scores 0.0 on every field. Adherence is
folded into accuracy rather than reported as a separate vanity metric, because a
record that cannot be ingested has no usable fields.
"""
from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ftspec.core.fields import FREE_TEXT_KIND, ScoringPlan

MISSING = object()

RECORD_EXACT = "_record_exact"
RUBRIC_INPUTS = "_rubric_inputs"


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def _tokens(text: Any) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _f1(pred_items: list, gold_items: list) -> float:
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    pred_set, gold_set = set(pred_items), set(gold_items)
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _canonical(value: Any) -> Any:
    """Compare list-valued fields without regard to element order."""
    if isinstance(value, list):
        return sorted(json_key(v) for v in value)
    return value


def json_key(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def score_field(kind: str, pred: Any, gold: Any) -> float:
    if pred is MISSING:
        return 0.0

    if kind == "exact":
        return 1.0 if _canonical(pred) == _canonical(gold) else 0.0

    if kind == "numeric":
        if pred is None and gold is None:
            return 1.0
        if pred is None or gold is None:
            return 0.0
        try:
            return 1.0 if abs(float(pred) - float(gold)) <= 0.01 else 0.0
        except (TypeError, ValueError):
            return 0.0

    if kind == "set_f1":
        if not isinstance(pred, list) or not isinstance(gold, list):
            return 0.0
        # Elements may be dicts (FHIR conditions) or scalars; compare canonically.
        return _f1([json_key(x) for x in pred], [json_key(x) for x in gold])

    if kind == FREE_TEXT_KIND:
        if pred is None or isinstance(pred, (dict, list)):
            return 0.0
        return _f1(_tokens(pred), _tokens(gold))

    raise ValueError(f"unknown scoring kind: {kind}")


def score_record(pred: dict | None, gold: dict, plan: ScoringPlan) -> dict:
    """Score one prediction. `pred=None` means contract violation -> all zeros."""
    scores: dict = {}
    for spec in plan.fields:
        scores[spec.path] = 0.0 if pred is None else score_field(
            spec.kind, get_path(pred, spec.path), get_path(gold, spec.path))

    match_paths = plan.record_match_paths
    scores[RECORD_EXACT] = 1.0 if match_paths and all(
        scores[p] == 1.0 for p in match_paths) else 0.0

    rubric_paths = plan.rubric_input_paths
    scores[RUBRIC_INPUTS] = statistics.mean(
        scores[p] for p in rubric_paths) if rubric_paths else 0.0
    return scores


# --- Statistics --------------------------------------------------------------

def bootstrap_ci(values: list, iters: int = 2000, alpha: float = 0.05,
                  seed: int = 12345) -> tuple:
    """Percentile bootstrap CI for the mean. Pure stdlib, deterministic."""
    if not values:
        return (0.0, 0.0)
    if len(set(values)) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(iters)]
    means.sort()
    return (means[int((alpha / 2) * iters)],
            means[min(iters - 1, int((1 - alpha / 2) * iters))])


def mcnemar_exact(a_correct: list, b_correct: list) -> tuple:
    """Two-sided exact McNemar test on paired binary outcomes.

    Returns (n_a_only, n_b_only, p_value). Run this before claiming system A
    beats system B: at n=150, differences under roughly 6 points are usually noise.
    """
    b = sum(1 for a, bb in zip(a_correct, b_correct, strict=True) if a == 1.0 and bb == 0.0)
    c = sum(1 for a, bb in zip(a_correct, b_correct, strict=True) if a == 0.0 and bb == 1.0)
    n = b + c
    if n == 0:
        return (b, c, 1.0)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return (b, c, min(1.0, 2 * tail))


def percentile(values: list, p: float) -> float:
    """Nearest-rank percentile. p50 and p99 are what an SLA is written against."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


# --- Aggregation -------------------------------------------------------------

@dataclass
class FieldStat:
    path: str
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class Aggregate:
    n: int
    plan: ScoringPlan | None = None
    fields: dict = field(default_factory=dict)
    record_exact: FieldStat | None = None
    rubric_inputs: FieldStat | None = None
    per_record: list = field(default_factory=list)

    def field_mean(self, path: str) -> float:
        stat = self.fields.get(path)
        return stat.mean if stat else 0.0

    def headline_mean(self) -> float:
        spec = self.plan.headline() if self.plan else None
        return self.field_mean(spec.path) if spec else 0.0


def aggregate(per_record: list, plan: ScoringPlan) -> Aggregate:
    agg = Aggregate(n=len(per_record), plan=plan, per_record=per_record)
    if not per_record:
        return agg
    for spec in plan.fields:
        values = [r[spec.path] for r in per_record]
        lo, hi = bootstrap_ci(values)
        agg.fields[spec.path] = FieldStat(spec.path, statistics.mean(values), lo, hi)
    for key, attr in ((RECORD_EXACT, "record_exact"), (RUBRIC_INPUTS, "rubric_inputs")):
        values = [r[key] for r in per_record]
        lo, hi = bootstrap_ci(values)
        setattr(agg, attr, FieldStat(key, statistics.mean(values), lo, hi))
    return agg


def confusion(preds: list, golds: list, labels: list) -> dict:
    """Confusion counts for a categorical field: {(gold, pred): count}.

    Showing *how* a system misjudges the headline field is far more persuasive
    than a single accuracy number: systematic under-triage of regulated or
    high-value cases is a business risk, not a rounding error.
    """
    table = {(g, p): 0 for g in labels for p in labels + ["<invalid>"]}
    for pred, gold in zip(preds, golds, strict=True):
        key = (gold, pred if pred in labels else "<invalid>")
        table[key] = table.get(key, 0) + 1
    return table


def format_confusion(table: dict, labels: list) -> str:
    cols = labels + ["<invalid>"]
    rows = ["| gold \\ pred | " + " | ".join(cols) + " |",
            "|---" * (len(cols) + 1) + "|"]
    for gold in labels:
        rows.append(f"| **{gold}** | " + " | ".join(
            str(table.get((gold, p), 0)) for p in cols) + " |")
    return "\n".join(rows)
