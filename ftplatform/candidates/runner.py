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
from ftspec.training import train as train_mod

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


def run_stage_b_candidate(ctx: CustomerContext, candidate, max_steps: int = -1,
                            resume: bool = False, gpu_cost_per_hour: float = 0.35) -> dict:
    """Fine-tune one Stage-B LoRA candidate, then immediately evaluate it.

    `candidate` is a `ftplatform.candidates.generator.StageBCandidate`: one
    (base_model, lora_r, lora_alpha) point in the grid built for the Stage-A
    winner. Trains via `ftspec.training.train.run`, unmodified except for
    its additive `data_dir` override -- the corpus stays shared across every
    candidate while each candidate's adapter/checkpoints land in their own
    directory (see train.py's docstring on why that split was needed).
    """
    data_dir = ctx.data_dir()
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"no training corpus for customer {ctx.customer.id!r} at {data_dir}. "
            f"Run `ftplatform baseline run {ctx.customer.id}` first.")

    cfg = ctx.config.model_copy(deep=True)
    cfg.model.base_model = candidate.base_model
    cfg.lora.r = candidate.lora_r
    cfg.lora.lora_alpha = candidate.lora_alpha

    key = ctx.candidate_key(candidate.candidate_id)
    manifests_dir = ctx.manifests_dir(candidate.candidate_id)
    fingerprint = cfg.fingerprint()

    with run_manifest("train", manifests_dir / "train.json", fingerprint,
                       params={"customer": ctx.customer.id, "candidate_id": candidate.candidate_id,
                                "base_model": candidate.base_model, "lora_r": candidate.lora_r,
                                "lora_alpha": candidate.lora_alpha, "max_steps": max_steps,
                                "resume": resume}) as m:
        m.metrics = train_mod.run(cfg, key, max_steps=max_steps, resume=resume, data_dir=data_dir)

    with run_manifest("evaluate", manifests_dir / "evaluate.json", fingerprint,
                       params={"customer": ctx.customer.id, "candidate_id": candidate.candidate_id,
                                "systems": "finetuned"}) as m:
        result = benchmark.run(
            profile=ctx.profile, eval_file=data_dir / "eval.jsonl",
            results_dir=ctx.reports_dir(candidate.candidate_id),
            systems="finetuned", base_model=candidate.base_model,
            finetuned_model=str(ctx.adapter_dir(candidate.candidate_id)),
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        m.metrics = result
    return result


def run_stage_c_candidate(ctx: CustomerContext, winner_candidate_id: str, candidate,
                            gpu_cost_per_hour: float = 0.35, max_new_tokens: int = 512) -> dict:
    """Re-evaluate an already-trained Stage-B winner's adapter under one
    quantization/constrained-decoding toggle. Purely inference-time -- no
    training, reusing `benchmark.run()`'s additive `load_in_4bit`/
    `constrained` overrides, so it costs one evaluate pass, not a retrain.

    `candidate` is a `ftplatform.candidates.generator.StageCCandidate`.
    """
    adapter_dir = ctx.adapter_dir(winner_candidate_id)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"no trained adapter for candidate {winner_candidate_id!r} at {adapter_dir}. "
            f"Run `ftplatform candidate stage-b` for it first.")

    eval_file = ctx.data_dir() / "eval.jsonl"
    manifests_dir = ctx.manifests_dir(candidate.candidate_id)
    fingerprint = ctx.config.fingerprint()

    with run_manifest("evaluate", manifests_dir / "evaluate.json", fingerprint,
                       params={"customer": ctx.customer.id, "candidate_id": candidate.candidate_id,
                                "winner_candidate_id": winner_candidate_id,
                                "quantization": candidate.quantization,
                                "constrained": candidate.constrained}) as m:
        result = benchmark.run(
            profile=ctx.profile, eval_file=eval_file,
            results_dir=ctx.reports_dir(candidate.candidate_id),
            systems="finetuned", finetuned_model=str(adapter_dir),
            load_in_4bit=candidate.load_in_4bit, constrained=candidate.constrained,
            gpu_cost_per_hour=gpu_cost_per_hour, max_new_tokens=max_new_tokens,
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
