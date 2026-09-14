"""
Phase 4 -- automatic candidate ranking.

Turns the uniform per-system metrics `ftspec.evaluation.benchmark` already
produces (`_summarize_systems()`'s dict shape, reused verbatim) into one
ranked list across an arbitrary number of candidates. This is deliberately
not built on `benchmark.render_report()`, which is hardcoded prose around
the fixed 3-baseline story -- these are pure functions over plain numbers,
independent of how many candidates exist or what produced them, so they can
be tested with no GPU and no model.
"""
from __future__ import annotations

from dataclasses import dataclass

# An output that can't be parsed has no usable fields regardless of anything
# else, so this gate runs before quality/cost/latency are even considered.
DEFAULT_MIN_ADHERENCE_PCT = 98.0

# Quality-first: a cheap wrong answer is worse than an accurate, pricier one.
# Cost is still weighted meaningfully because lowering it is the product's
# actual value proposition. Overridable per call, not hardcoded elsewhere.
DEFAULT_WEIGHTS = {"quality": 0.60, "cost": 0.25, "latency": 0.15}


@dataclass(frozen=True)
class CandidateRow:
    """One scored system, from one candidate's benchmark run."""

    candidate_id: str
    system: str
    schema_adherence_pct: float
    headline_accuracy_pct: float
    record_exact_pct: float
    cost_per_100k_usd: float
    p99_ms: float
    label: str = ""

    @property
    def quality(self) -> float:
        # Headline field weighted higher: it's the business-critical one,
        # matching profile.scoring_plan().headline()'s special status in
        # ftspec's own benchmark report.
        return 0.6 * self.headline_accuracy_pct / 100 + 0.4 * self.record_exact_pct / 100


def _normalize(values: list[float]) -> list[float]:
    """1.0 = cheapest/fastest in the set, 0.0 = most expensive/slowest.
    Everyone tied -> everyone gets 1.0 (nothing left to penalize)."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [1 - (v - lo) / (hi - lo) for v in values]


def rank(rows: list[CandidateRow], *, min_adherence_pct: float = DEFAULT_MIN_ADHERENCE_PCT,
          weights: dict | None = None) -> list[dict]:
    """Score and rank candidates, best first.

    Rows failing the adherence floor are still returned -- visible, not
    silently dropped -- but carry `gated: True`, `score: None`, and sort
    after every surviving candidate; nothing gated is ever the top pick.
    """
    if not rows:
        return []
    w = weights or DEFAULT_WEIGHTS
    survivors = [r for r in rows if r.schema_adherence_pct >= min_adherence_pct]
    gated = [r for r in rows if r.schema_adherence_pct < min_adherence_pct]

    scored = []
    if survivors:
        cost_scores = _normalize([r.cost_per_100k_usd for r in survivors])
        latency_scores = _normalize([r.p99_ms for r in survivors])
        for r, cost_score, latency_score in zip(survivors, cost_scores, latency_scores, strict=True):
            quality = r.quality
            score = w["quality"] * quality + w["cost"] * cost_score + w["latency"] * latency_score
            scored.append({**_row_dict(r), "quality": round(quality, 4),
                            "cost_score": round(cost_score, 4),
                            "latency_score": round(latency_score, 4),
                            "score": round(score, 4), "gated": False, "gate_reason": None})
    scored.sort(key=lambda d: d["score"], reverse=True)

    gated_out = [{**_row_dict(r), "quality": round(r.quality, 4), "cost_score": None,
                  "latency_score": None, "score": None, "gated": True,
                  "gate_reason": f"schema_adherence_pct {r.schema_adherence_pct:.1f} < "
                                 f"{min_adherence_pct:.1f}"} for r in gated]

    return scored + gated_out


def _row_dict(r: CandidateRow) -> dict:
    return {"candidate_id": r.candidate_id, "system": r.system, "label": r.label,
            "schema_adherence_pct": r.schema_adherence_pct,
            "headline_accuracy_pct": r.headline_accuracy_pct,
            "record_exact_pct": r.record_exact_pct,
            "cost_per_100k_usd": r.cost_per_100k_usd, "p99_ms": r.p99_ms}
