"""
Phase 2 -- baseline capture.

For a brand-new customer, the "current model" a specialization has to beat
is almost always exactly what `ftspec.evaluation.benchmark` already knows
how to run: a prompted base model, with and without grammar-constrained
decoding. No training, no new backend code -- `run_baseline()` just scopes
`ftspec`'s existing `prepare` / `validate` / `evaluate` stages through a
`CustomerContext` and records the result as `memory/deployed.json`, the
baseline that Phase 6's deploy-if-better logic later compares new candidates
against.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ftplatform.customers.context import CustomerContext
from ftspec.data import build
from ftspec.data import validate as validate_mod
from ftspec.evaluation import benchmark
from ftspec.run import get_logger, run_manifest

log = get_logger("ftplatform.candidates.runner")

# Local-only: no OpenAI key needed, no external cost. A customer that wants
# the gpt4o-* comparison can pass a systems string naming it explicitly --
# it is never on by default, matching the platform's zero-paid-infra stance.
DEFAULT_BASELINE_SYSTEMS = "base-schema,base-rubric,base-constrained"

BASELINE_CANDIDATE_ID = "baseline"


def run_baseline(
    ctx: CustomerContext,
    *,
    n_train: int = 200,
    n_val: int = 40,
    n_eval: int = 150,
    seed: int = 42,
    eval_seed: int = 1337,
    from_jsonl: Path | None = None,
    systems: str = DEFAULT_BASELINE_SYSTEMS,
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    gpu_cost_per_hour: float = 0.35,
) -> dict:
    """Build the customer's corpus, gate it, then benchmark prompted-baseline
    systems against the held-out set. Writes memory/deployed.json.

    Returns the same dict `ftspec.evaluation.benchmark.run()` returns, so a
    caller gets identical fields to what `ftspec evaluate` would report.
    """
    data_dir = ctx.data_dir()
    manifests_dir = ctx.manifests_dir(BASELINE_CANDIDATE_ID)
    reports_dir = ctx.reports_dir(BASELINE_CANDIDATE_ID)
    fingerprint = ctx.config.fingerprint()

    with run_manifest("prepare", manifests_dir / "prepare.json", fingerprint,
                       params={"customer": ctx.customer.id, "n_train": n_train, "n_val": n_val,
                                "n_eval": n_eval, "seed": seed, "eval_seed": eval_seed}) as m:
        m.metrics = build.run(ctx.profile, data_dir, n_train, n_val, n_eval,
                               seed, eval_seed, from_jsonl)

    with run_manifest("validate", manifests_dir / "validate.json", fingerprint) as m:
        passed, report, metrics = validate_mod.run(ctx.config, ctx.profile, data_dir)
        m.metrics = metrics
    if not passed:
        # Baseline systems here are all prompted, not trained -- unlike
        # `ftspec train`, no GPU-hour is at risk, so this is a loud warning
        # rather than a hard gate. The report is still on disk for review.
        log.warning("corpus failed validation for customer %s; baseline numbers "
                     "may reflect truncated or leaking samples -- see %s",
                     ctx.customer.id, manifests_dir / "validate.json")

    eval_file = data_dir / "eval.jsonl"
    with run_manifest("evaluate", manifests_dir / "evaluate.json", fingerprint,
                       params={"customer": ctx.customer.id, "systems": systems}) as m:
        result = benchmark.run(
            profile=ctx.profile, eval_file=eval_file, results_dir=reports_dir,
            systems=systems, base_model=base_model, finetuned_model=None,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        m.metrics = result

    _write_baseline_snapshot(ctx, result)
    return result


def run_stage_a_candidate(ctx: CustomerContext, candidate, gpu_cost_per_hour: float = 0.35) -> dict:
    """Evaluate one Stage-A candidate (a base model swap, no training) against
    the customer's existing held-out set.

    `candidate` is a `ftplatform.candidates.generator.StageACandidate`.
    Assumes the corpus already exists (`run_baseline()` or `ftspec prepare`
    already built it) -- Stage A only evaluates, it never rebuilds data.
    """
    eval_file = ctx.data_dir() / "eval.jsonl"
    if not eval_file.exists():
        raise FileNotFoundError(
            f"no eval set for customer {ctx.customer.id!r} at {eval_file}. "
            f"Run `ftplatform baseline run {ctx.customer.id}` first.")

    manifests_dir = ctx.manifests_dir(candidate.candidate_id)
    reports_dir = ctx.reports_dir(candidate.candidate_id)
    fingerprint = ctx.config.fingerprint()

    with run_manifest("evaluate", manifests_dir / "evaluate.json", fingerprint,
                       params={"customer": ctx.customer.id, "candidate_id": candidate.candidate_id,
                                "base_model": candidate.base_model,
                                "systems": candidate.systems}) as m:
        result = benchmark.run(
            profile=ctx.profile, eval_file=eval_file, results_dir=reports_dir,
            systems=candidate.systems, base_model=candidate.base_model, finetuned_model=None,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        m.metrics = result
    return result


def _write_baseline_snapshot(ctx: CustomerContext, result: dict) -> None:
    memory_dir = ctx.memory_dir()
    memory_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "kind": "baseline",
        "candidate_id": BASELINE_CANDIDATE_ID,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_eval": result["n_eval"],
        "systems": result["systems"],
    }
    out = memory_dir / "deployed.json"
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("baseline recorded -> %s", out)
