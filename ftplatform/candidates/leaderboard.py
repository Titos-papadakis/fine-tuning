"""
Reads each candidate's already-written evaluate manifest back off disk,
flattens their systems into `CandidateRow`s, ranks them, and writes the
result both machine- and human-readable under customers/<id>/benchmark/.

No model loads here -- this is pure post-processing of numbers `evaluate`
already produced, the same offline-from-disk pattern `combine()` established
in `ftspec.evaluation.benchmark` for folding separately-run systems together.
"""
from __future__ import annotations

import json

from ftplatform.candidates.scoring import CandidateRow, rank
from ftspec.run import get_logger

log = get_logger("ftplatform.candidates.leaderboard")


def load_candidate_systems(ctx, candidate_id: str, label: str = "") -> list[CandidateRow]:
    """Read one candidate's evaluate manifest and flatten its systems into rows."""
    manifest_path = ctx.manifests_dir(candidate_id) / "evaluate.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no evaluate manifest for candidate {candidate_id!r} at {manifest_path}. "
            f"Run that candidate first.")
    metrics = json.loads(manifest_path.read_text(encoding="utf-8")).get("metrics", {})
    systems = metrics.get("systems", {})
    return [
        CandidateRow(candidate_id=candidate_id, system=name, label=label or candidate_id,
                      schema_adherence_pct=m["schema_adherence_pct"],
                      headline_accuracy_pct=m["headline_accuracy_pct"] or 0.0,
                      record_exact_pct=m["record_exact_pct"],
                      cost_per_100k_usd=m["cost_per_100k_usd"], p99_ms=m["p99_ms"])
        for name, m in systems.items()
    ]


def build(ctx, candidates: dict[str, str], conn=None, **rank_kwargs) -> list[dict]:
    """`candidates`: {candidate_id: label}. Ranks every system from every
    candidate together and writes benchmark/leaderboard.{json,md}.

    `conn`, when given, also records this leaderboard into the cross-customer
    `candidate_stats` table (see ftplatform/learning/) -- optional and
    additive, default None preserves the exact prior behavior for any caller
    that doesn't pass one.
    """
    rows: list[CandidateRow] = []
    for candidate_id, label in candidates.items():
        rows.extend(load_candidate_systems(ctx, candidate_id, label))

    ranked = rank(rows, **rank_kwargs)

    out_dir = ctx.benchmark_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "leaderboard.md").write_text(render_markdown(ranked), encoding="utf-8")
    log.info("leaderboard written -> %s", out_dir / "leaderboard.md")

    if conn is not None:
        from ftplatform.learning import stats
        stats.record(conn, ctx.customer.workload, ctx.customer.id, ranked)

    return ranked


def render_markdown(ranked: list[dict]) -> str:
    lines = [
        "# Candidate leaderboard",
        "",
        "Ranked by composite score (quality-weighted; see `scoring.py` for the "
        "formula). Rows marked GATED failed the schema-adherence floor and are "
        "never recommended -- shown so nothing is silently hidden.",
        "",
        "| Rank | Candidate | System | Adherence | Record exact | Headline | "
        "Cost/100k | p99 | Score |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        score = f"{r['score']:.3f}" if r["score"] is not None else "GATED"
        lines.append(
            f"| {i} | {r['label']} | {r['system']} | {r['schema_adherence_pct']:.1f}% | "
            f"{r['record_exact_pct']:.1f}% | {r['headline_accuracy_pct']:.1f}% | "
            f"${r['cost_per_100k_usd']:.2f} | {r['p99_ms']:.0f}ms | {score} |")
    lines.append("")
    return "\n".join(lines)
