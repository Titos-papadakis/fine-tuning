"""
Phase 6 -- deploy-if-better.

Promotes a candidate to production/ only if it clears three checks against
whatever is currently deployed (memory/deployed.json, initially the Phase-2
baseline): it doesn't fall below the schema-adherence floor, it isn't a
*statistically significant* regression on record-level accuracy (McNemar,
paired on the same eval set -- reused verbatim from ftspec.evaluation.metrics),
and its composite score (ftplatform.candidates.scoring, the same formula the
leaderboard ranks with) doesn't fall more than a small epsilon short.

Failing any gate simply never promotes -- that *is* the rollback; nothing is
ever pointed at an untested candidate. `rollback.py` is the separate,
deliberately gate-free escape hatch for a candidate that passed every check
here but turned out wrong once real traffic exercised it.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from ftplatform.candidates import leaderboard as lb
from ftplatform.candidates.scoring import DEFAULT_MIN_ADHERENCE_PCT, CandidateRow, rank
from ftplatform.deployment import registry
from ftspec.evaluation import metrics as M
from ftspec.evaluation.metrics import RECORD_EXACT
from ftspec.run import get_logger

log = get_logger("ftplatform.deployment.deploy")

DEFAULT_EPSILON = 0.02


def _record_exact_scores(reports_dir, system_name: str) -> list[float]:
    path = reports_dir / f"raw_{system_name}.jsonl"
    if not path.exists():
        return []
    scores = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line)["scores"][RECORD_EXACT])
    return scores


def _current_rows(deployed: dict) -> list[CandidateRow]:
    return [
        CandidateRow(candidate_id=deployed["candidate_id"], system=name, label="current",
                      schema_adherence_pct=m["schema_adherence_pct"],
                      headline_accuracy_pct=m["headline_accuracy_pct"] or 0.0,
                      record_exact_pct=m["record_exact_pct"],
                      cost_per_100k_usd=m["cost_per_100k_usd"], p99_ms=m["p99_ms"])
        for name, m in deployed["systems"].items()
    ]


def evaluate_gates(ctx, new_candidate_id: str, new_system: str = "finetuned",
                    min_adherence_pct: float = DEFAULT_MIN_ADHERENCE_PCT,
                    epsilon: float = DEFAULT_EPSILON) -> dict:
    """Check whether `new_candidate_id` clears every currently-deployed
    system, without writing anything. Returns `{"passed": bool, "reason":
    str | None, "new": CandidateRow, "current": [CandidateRow, ...]}`.
    """
    deployed_path = ctx.memory_dir() / "deployed.json"
    if not deployed_path.exists():
        raise FileNotFoundError(
            f"no baseline recorded for customer {ctx.customer.id!r}. "
            f"Run `ftplatform baseline run {ctx.customer.id}` first.")
    deployed = json.loads(deployed_path.read_text(encoding="utf-8"))
    current_rows = _current_rows(deployed)

    new_rows = [r for r in lb.load_candidate_systems(ctx, new_candidate_id, label=new_candidate_id)
                if r.system == new_system]
    if not new_rows:
        raise ValueError(f"system {new_system!r} not found for candidate {new_candidate_id!r}")
    new_row = new_rows[0]

    if new_row.schema_adherence_pct < min_adherence_pct:
        return {"passed": False, "new": new_row, "current": current_rows,
                "reason": f"schema_adherence_pct {new_row.schema_adherence_pct:.1f} < "
                          f"{min_adherence_pct:.1f}"}

    # McNemar against the strongest currently-deployed system by raw
    # record-exact accuracy -- reject only if it shows that system
    # *significantly* beats the new one, on the same paired eval set.
    strongest = max(current_rows, key=lambda r: r.record_exact_pct)
    new_scores = _record_exact_scores(ctx.reports_dir(new_candidate_id), new_system)
    current_scores = _record_exact_scores(ctx.reports_dir(strongest.candidate_id), strongest.system)
    if new_scores and current_scores and len(new_scores) == len(current_scores):
        b, c, p = M.mcnemar_exact(new_scores, current_scores)
        if p < 0.05 and c > b:
            return {"passed": False, "new": new_row, "current": current_rows,
                     "reason": f"McNemar p={p:.4f}: current beats new significantly "
                               f"({b} new-only vs {c} current-only wins)"}

    # Composite score, ranked head-to-head against every currently-deployed
    # system together -- reuses the exact leaderboard formula, not a
    # separate one, so "wins the leaderboard" and "clears deployment" never
    # quietly disagree.
    ranked = rank([new_row, *current_rows], min_adherence_pct=min_adherence_pct)
    new_ranked = next(r for r in ranked if r["candidate_id"] == new_row.candidate_id
                       and r["system"] == new_row.system)
    others = [r for r in ranked if r is not new_ranked and r["score"] is not None]
    if new_ranked["score"] is not None and others:
        best_current_score = max(r["score"] for r in others)
        if new_ranked["score"] < best_current_score - epsilon:
            return {"passed": False, "new": new_row, "current": current_rows,
                     "reason": f"composite score {new_ranked['score']:.4f} falls more than "
                               f"{epsilon} short of the best current system's "
                               f"{best_current_score:.4f}"}

    return {"passed": True, "reason": None, "new": new_row, "current": current_rows}


def infer_base_model(ctx, candidate_id: str) -> str | None:
    """Looks up which base model a candidate was built from, from whatever
    manifest that candidate already has -- train.json for a Stage-B
    candidate, evaluate.json's own base_model for Stage A, or (Stage C,
    which trains nothing) the Stage-B winner it re-evaluated."""
    manifests = ctx.manifests_dir(candidate_id)
    for name in ("train.json", "evaluate.json"):
        path = manifests / name
        if not path.exists():
            continue
        params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
        if params.get("base_model"):
            return params["base_model"]
        winner = params.get("winner_candidate_id")
        if winner and winner != candidate_id:
            return infer_base_model(ctx, winner)
    return None


def _winner_of(ctx, candidate_id: str) -> str | None:
    path = ctx.manifests_dir(candidate_id) / "evaluate.json"
    if not path.exists():
        return None
    winner = json.loads(path.read_text(encoding="utf-8")).get("params", {}).get("winner_candidate_id")
    return winner if winner and winner != candidate_id else None


def _promote(ctx, candidate_id: str, row: CandidateRow, conn=None, is_rollback: bool = False) -> None:
    src = ctx.candidate_dir(candidate_id)
    dst = ctx.production_dir()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # A Stage-C candidate trains nothing -- it re-evaluates a Stage-B winner's
    # adapter -- so its own dir has no weights. Without this, promoting one
    # left production/ with nothing `serve` could load.
    winner = _winner_of(ctx, candidate_id)
    if winner and not (dst / "lora_adapter").exists() and ctx.adapter_dir(winner).exists():
        shutil.copytree(ctx.adapter_dir(winner), dst / "lora_adapter")

    base_model = infer_base_model(ctx, candidate_id)
    if base_model:
        (dst / "candidate_meta.json").write_text(
            json.dumps({"base_model": base_model, "candidate_id": candidate_id},
                       ensure_ascii=False), encoding="utf-8")

    snapshot = {
        "kind": "deployed", "candidate_id": candidate_id,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "systems": {row.system: {
            "schema_adherence_pct": row.schema_adherence_pct,
            "headline_accuracy_pct": row.headline_accuracy_pct,
            "record_exact_pct": row.record_exact_pct,
            "cost_per_100k_usd": row.cost_per_100k_usd, "p99_ms": row.p99_ms,
        }},
    }
    ctx.memory_dir().mkdir(parents=True, exist_ok=True)
    (ctx.memory_dir() / "deployed.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    if conn is not None:
        registry.record(conn, ctx.customer.id, ctx.profile.name, candidate_id, snapshot,
                         is_rollback=is_rollback)
        from ftplatform import audit
        audit.record(conn, "deploy.rollback" if is_rollback else "deploy.promote",
                     ctx.customer.id, {"candidate_id": candidate_id, "system": row.system})


def maybe_deploy(ctx, new_candidate_id: str, new_system: str = "finetuned",
                  min_adherence_pct: float = DEFAULT_MIN_ADHERENCE_PCT,
                  epsilon: float = DEFAULT_EPSILON, conn=None) -> dict:
    """The whole deploy-if-better decision. Returns `{"deployed": bool,
    "reason": str | None}` (plus `"candidate_id"` when deployed)."""
    gates = evaluate_gates(ctx, new_candidate_id, new_system, min_adherence_pct, epsilon)
    if not gates["passed"]:
        log.info("candidate %s NOT promoted for customer %s: %s",
                  new_candidate_id, ctx.customer.id, gates["reason"])
        return {"deployed": False, "reason": gates["reason"]}

    _promote(ctx, new_candidate_id, gates["new"], conn=conn)
    log.info("candidate %s promoted to production for customer %s",
              new_candidate_id, ctx.customer.id)
    return {"deployed": True, "reason": None, "candidate_id": new_candidate_id}
