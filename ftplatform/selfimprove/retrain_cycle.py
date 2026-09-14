"""
Phase 6 -- the retrain-and-maybe-redeploy cycle.

`dataset_update.fold_corrections()` grows train.jsonl -> one new Stage-B
candidate is trained on it, repeating the *winning* (base_model, r, alpha)
point rather than re-running the whole Stage-B sweep (the search already
happened once; this is retraining, not re-searching) ->
`deploy.maybe_deploy()` applies the exact same three gates any other
candidate faces. A retrain that doesn't clear them simply never promotes --
production stays on whatever was already there.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ftplatform.candidates.generator import StageBCandidate
from ftplatform.candidates.runner import run_stage_b_candidate
from ftplatform.deployment.deploy import maybe_deploy
from ftplatform.selfimprove.dataset_update import fold_corrections
from ftspec.run import get_logger

log = get_logger("ftplatform.selfimprove.retrain_cycle")


def run_cycle(ctx, base_model: str, lora_r: int, lora_alpha: int, conn=None,
              min_adherence_pct: float = 98.0, epsilon: float = 0.02,
              max_steps: int = -1, gpu_cost_per_hour: float = 0.35) -> dict:
    """One full self-improvement cycle for this customer.

    Returns `{"folded": {...from fold_corrections...}, "candidate_id": str
    | None, "deployed": bool, "reason": str | None}`. `candidate_id` is
    `None` and nothing trains if there were no corrections to fold --
    spending GPU time retraining on an unchanged corpus would just
    reproduce the candidate that's already deployed.
    """
    folded = fold_corrections(ctx)
    if folded["added"] == 0:
        log.info("customer %s: no corrections to fold, skipping retrain", ctx.customer.id)
        return {"folded": folded, "candidate_id": None, "deployed": False,
                "reason": "no corrections to fold"}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    candidate_id = f"retrain-{stamp}"
    candidate = StageBCandidate(candidate_id, base_model, lora_r, lora_alpha)

    run_stage_b_candidate(ctx, candidate, max_steps=max_steps, gpu_cost_per_hour=gpu_cost_per_hour)
    result = maybe_deploy(ctx, candidate_id, min_adherence_pct=min_adherence_pct,
                           epsilon=epsilon, conn=conn)
    return {"folded": folded, "candidate_id": candidate_id, **result}
