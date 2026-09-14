"""
`ftplatform` — customer-scoped orchestration around the `ftspec` engine.

    ftplatform customer add <id> --name "Acme" --workload saas_support
    ftplatform customer list

Every command resolves a customer id to an isolated Config/Profile pair via
CustomerContext (ftplatform/customers/context.py) before touching `ftspec` --
no command here is allowed to use an unscoped path.
"""
from __future__ import annotations

import typer

from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext, UnknownCustomerError
from ftplatform.db import connect
from ftspec.core.registry import available
from ftspec.run import get_logger, setup_logging

app = typer.Typer(
    name="ftplatform",
    help="Multi-customer orchestration around the ftspec engine.",
    add_completion=False,
    no_args_is_help=True,
)
customer_app = typer.Typer(help="Register and list customers.", no_args_is_help=True)
app.add_typer(customer_app, name="customer")
baseline_app = typer.Typer(help="Measure a customer's prompted-baseline systems.",
                            no_args_is_help=True)
app.add_typer(baseline_app, name="baseline")
candidate_app = typer.Typer(help="Generate and run optimization candidates.",
                             no_args_is_help=True)
app.add_typer(candidate_app, name="candidate")
leaderboard_app = typer.Typer(help="Rank candidates that have already been run.",
                               no_args_is_help=True)
app.add_typer(leaderboard_app, name="leaderboard")
deploy_app = typer.Typer(help="Promote a candidate to production, with gates.",
                          no_args_is_help=True)
app.add_typer(deploy_app, name="deploy")
rollback_app = typer.Typer(help="Manually revert production to a prior candidate.",
                            no_args_is_help=True)
app.add_typer(rollback_app, name="rollback")

log = get_logger("ftplatform.cli")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging.")):
    setup_logging(verbose)


@customer_app.command("add")
def customer_add(
    customer_id: str = typer.Argument(..., help="Short, URL-safe id, e.g. 'acme'."),
    name: str = typer.Option(..., help="Display name."),
    workload: str = typer.Option(..., help="Vertical to run for this customer, e.g. saas_support."),
):
    """Register a new customer. Its data and models are isolated from every other one."""
    if workload not in available():
        typer.secho(f"Unknown workload {workload!r}. Choices: {', '.join(available())}",
                     fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    conn = connect()
    try:
        customer = store.create(conn, customer_id, name, workload)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    typer.secho(f"customer {customer.id!r} added (workload={customer.workload})",
                fg=typer.colors.GREEN)


@customer_app.command("list")
def customer_list():
    """List every registered customer."""
    conn = connect()
    try:
        customers = store.list_all(conn)
    finally:
        conn.close()

    if not customers:
        typer.echo("no customers yet -- `ftplatform customer add <id> --name ... --workload ...`")
        return

    typer.echo(f"\n  {'id':<16} {'name':<24} {'workload':<18} {'status':<8} created")
    typer.echo("  " + "-" * 84)
    for c in customers:
        typer.echo(f"  {c.id:<16} {c.name:<24} {c.workload:<18} {c.status:<8} {c.created_at}")
    typer.echo("")


@baseline_app.command("run")
def baseline_run(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    systems: str = typer.Option(
        "base-schema,base-rubric,base-constrained",
        help="Comma-separated systems. Defaults to local-only baselines "
             "(no OpenAI key, no external cost)."),
    n_train: int = typer.Option(200),
    n_val: int = typer.Option(40),
    n_eval: int = typer.Option(150),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Build this customer's corpus and benchmark prompted-baseline systems.

    Requires a GPU (the baseline systems still run local models) via the
    same `train`/`serve` extras `ftspec evaluate` needs. Writes
    memory/deployed.json: the number a later fine-tuned candidate must beat.
    """
    from ftplatform.candidates import runner

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    try:
        result = runner.run_baseline(ctx, systems=systems, n_train=n_train,
                                       n_val=n_val, n_eval=n_eval,
                                       gpu_cost_per_hour=gpu_cost_per_hour)
    except (RuntimeError, ValueError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"\nbaseline recorded for {customer_id!r} "
                f"({len(result['systems'])} system(s), n_eval={result['n_eval']})")
    for name, m in result["systems"].items():
        typer.echo(f"  {name:<18} adherence={m['schema_adherence_pct']:.1f}%  "
                    f"record_exact={m['record_exact_pct']:.1f}%  "
                    f"cost/100k=${m['cost_per_100k_usd']:.2f}")


@candidate_app.command("list-presets")
def candidate_list_presets():
    """List the built-in Stage-A candidate base models."""
    from ftplatform.candidates.generator import DEFAULT_STAGE_A_CANDIDATES

    typer.echo(f"\n  {'candidate_id':<20} {'label':<16} base_model")
    typer.echo("  " + "-" * 80)
    for c in DEFAULT_STAGE_A_CANDIDATES:
        typer.echo(f"  {c.candidate_id:<20} {c.label:<16} {c.base_model}")
    typer.echo("")


@candidate_app.command("stage-a")
def candidate_stage_a(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    base_model: str = typer.Option(..., help="Base model to evaluate, e.g. a preset's base_model."),
    candidate_id: str | None = typer.Option(
        None, help="Defaults to a slug derived from --base-model."),
    systems: str = typer.Option("base-schema,base-rubric,base-constrained"),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Evaluate one no-training candidate (a base model swap) for this customer.

    Run once per candidate model, in its own process -- the same reason the
    Colab notebook runs each system in its own cell. `ftplatform candidate
    list-presets` shows a starting set of 2-3 model sizes; any base model
    ftspec can load also works via --base-model.
    """
    from ftplatform.candidates import runner
    from ftplatform.candidates.generator import StageACandidate, slugify_model

    cid = candidate_id or slugify_model(base_model)
    candidate = StageACandidate(cid, base_model, label=cid, systems=systems)

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    try:
        result = runner.run_stage_a_candidate(ctx, candidate, gpu_cost_per_hour=gpu_cost_per_hour)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"\ncandidate {cid!r} recorded for {customer_id!r} ({base_model})")
    for name, m in result["systems"].items():
        typer.echo(f"  {name:<18} adherence={m['schema_adherence_pct']:.1f}%  "
                    f"record_exact={m['record_exact_pct']:.1f}%  "
                    f"cost/100k=${m['cost_per_100k_usd']:.2f}")


@candidate_app.command("stage-b")
def candidate_stage_b(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    base_model: str = typer.Option(
        ..., help="Base model to train -- normally the Stage-A leaderboard's winner."),
    lora_r: int = typer.Option(..., help="LoRA rank."),
    lora_alpha: int = typer.Option(..., help="LoRA alpha."),
    candidate_id: str | None = typer.Option(
        None, help="Defaults to 'stageB-r<r>-a<alpha>'."),
    max_steps: int = typer.Option(-1, help="Override epochs with a fixed step count."),
    resume: bool = typer.Option(False, help="Resume from the latest checkpoint."),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Train and evaluate one Stage-B LoRA candidate for this customer.

    Run once per (r, alpha) point in the grid, in its own process, after a
    corpus already exists (`ftplatform baseline run`). `ftplatform candidate
    lora-grid` shows the default 3-point grid this trains for a fixed base
    model.
    """
    from ftplatform.candidates import runner
    from ftplatform.candidates.generator import StageBCandidate

    cid = candidate_id or f"stageB-r{lora_r}-a{lora_alpha}"
    candidate = StageBCandidate(cid, base_model, lora_r, lora_alpha)

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    try:
        result = runner.run_stage_b_candidate(ctx, candidate, max_steps=max_steps,
                                                resume=resume, gpu_cost_per_hour=gpu_cost_per_hour)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"\ncandidate {cid!r} trained and recorded for {customer_id!r} "
                f"(r={lora_r}, alpha={lora_alpha})")
    for name, m in result["systems"].items():
        typer.echo(f"  {name:<18} adherence={m['schema_adherence_pct']:.1f}%  "
                    f"record_exact={m['record_exact_pct']:.1f}%  "
                    f"cost/100k=${m['cost_per_100k_usd']:.2f}")


@candidate_app.command("lora-grid")
def candidate_lora_grid():
    """Show the default Stage-B LoRA (r, alpha) grid."""
    from ftplatform.candidates.generator import DEFAULT_LORA_GRID

    typer.echo(f"\n  {'r':>4} {'alpha':>6}")
    for r, alpha in DEFAULT_LORA_GRID:
        typer.echo(f"  {r:>4} {alpha:>6}")
    typer.echo("")


@candidate_app.command("stage-c")
def candidate_stage_c(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    winner: str = typer.Option(..., help="candidate_id of an already-trained Stage-B adapter."),
    quantization: str = typer.Option(..., help="'4bit' or 'fp16'."),
    constrained: bool = typer.Option(..., help="Grammar-constrained decoding on/off."),
    candidate_id: str | None = typer.Option(
        None, help="Defaults to 'stageC-<quantization>-<un/constrained>'."),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Re-evaluate a trained adapter under one quantization/constrained toggle.

    No training -- one evaluate pass against the adapter --winner already
    produced. `ftplatform candidate stage-c-combos` lists the default
    3-combo sweep (the 4th, 4bit+unconstrained, is what Stage B already
    measured).
    """
    from ftplatform.candidates import runner
    from ftplatform.candidates.generator import StageCCandidate

    if quantization not in ("4bit", "fp16"):
        typer.secho("--quantization must be '4bit' or 'fp16'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    cid = candidate_id or f"stageC-{quantization}-{'constrained' if constrained else 'unconstrained'}"
    candidate = StageCCandidate(cid, quantization, constrained)

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    try:
        result = runner.run_stage_c_candidate(ctx, winner, candidate,
                                                 gpu_cost_per_hour=gpu_cost_per_hour)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"\ncandidate {cid!r} recorded for {customer_id!r} "
                f"(quantization={quantization}, constrained={constrained})")
    for name, m in result["systems"].items():
        typer.echo(f"  {name:<18} adherence={m['schema_adherence_pct']:.1f}%  "
                    f"record_exact={m['record_exact_pct']:.1f}%  "
                    f"cost/100k=${m['cost_per_100k_usd']:.2f}")


@candidate_app.command("stage-c-combos")
def candidate_stage_c_combos():
    """Show the default Stage-C quantization/constrained-decoding combos."""
    from ftplatform.candidates.generator import DEFAULT_STAGE_C_COMBOS

    typer.echo(f"\n  {'quantization':<14} constrained")
    for q, c in DEFAULT_STAGE_C_COMBOS:
        typer.echo(f"  {q:<14} {c}")
    typer.echo("")


@leaderboard_app.command("build")
def leaderboard_build(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    candidates: str = typer.Option(
        ..., help="Comma-separated candidate ids already run, e.g. "
                   "'stageA-qwen25-3b,stageA-phi35-mini,stageA-llama31-8b'."),
    min_adherence: float = typer.Option(98.0, help="Schema-adherence floor to survive ranking."),
):
    """Rank every system from the given already-run candidates together.

    Reads each candidate's evaluate manifest back off disk -- no model
    loads, so this cannot itself run out of memory. Writes
    benchmark/leaderboard.{json,md}.
    """
    from ftplatform.candidates import leaderboard

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    ids = {c.strip(): c.strip() for c in candidates.split(",") if c.strip()}
    try:
        ranked = leaderboard.build(ctx, ids, min_adherence_pct=min_adherence)
    except FileNotFoundError as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"\ntop candidate: {ranked[0]['label']} / {ranked[0]['system']} "
                f"(score={ranked[0]['score']})" if ranked and not ranked[0]["gated"]
                else "\nno candidate survived the adherence floor")
    typer.echo(leaderboard.render_markdown(ranked))


@deploy_app.command("check")
def deploy_check(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    candidate_id: str = typer.Option(..., help="An already-run candidate to check."),
    system: str = typer.Option("finetuned"),
    min_adherence: float = typer.Option(98.0),
    epsilon: float = typer.Option(0.02),
):
    """Show whether a candidate would pass deploy-if-better, without deploying it."""
    from ftplatform.deployment import deploy

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    try:
        gates = deploy.evaluate_gates(ctx, candidate_id, system, min_adherence, epsilon)
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    if gates["passed"]:
        typer.secho(f"\n{candidate_id!r} WOULD be promoted.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{candidate_id!r} would NOT be promoted: {gates['reason']}",
                     fg=typer.colors.YELLOW)


@deploy_app.command("run")
def deploy_run(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    candidate_id: str = typer.Option(..., help="An already-run candidate to promote."),
    system: str = typer.Option("finetuned"),
    min_adherence: float = typer.Option(98.0),
    epsilon: float = typer.Option(0.02),
):
    """Promote a candidate to production/ if it clears every gate.

    Failing any gate is not an error -- it means production stays exactly
    as it was, which is the point. Use `deploy check` first to see the
    decision without acting on it.
    """
    from ftplatform.deployment import deploy

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
        result = deploy.maybe_deploy(ctx, candidate_id, system, min_adherence, epsilon, conn=conn)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    if result["deployed"]:
        typer.secho(f"\n{candidate_id!r} promoted to production for {customer_id!r}.",
                     fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{candidate_id!r} NOT promoted: {result['reason']}",
                     fg=typer.colors.YELLOW)


@deploy_app.command("history")
def deploy_history(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """List every promotion and rollback for this customer, oldest first."""
    from ftplatform.deployment import registry

    conn = connect()
    try:
        rows = registry.history(conn, customer_id)
    finally:
        conn.close()

    if not rows:
        typer.echo("no deployments yet")
        return
    typer.echo(f"\n  {'deployed_at':<22} {'candidate_id':<20} rollback")
    typer.echo("  " + "-" * 60)
    for r in rows:
        typer.echo(f"  {r['deployed_at']:<22} {r['candidate_id']:<20} {bool(r['is_rollback'])}")
    typer.echo("")


@rollback_app.command("run")
def rollback_run(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    candidate_id: str = typer.Option(..., help="An already-run candidate to revert to."),
    system: str = typer.Option("finetuned"),
):
    """Revert production/ to a previously-run candidate. Bypasses every
    deploy-if-better gate on purpose -- see rollback.py's docstring."""
    from ftplatform.deployment import rollback

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
        result = rollback.rollback_to(ctx, candidate_id, system, conn=conn)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    except (FileNotFoundError, ValueError) as e:
        typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    typer.secho(f"\nproduction reverted to {result['candidate_id']!r} for {customer_id!r}.",
                 fg=typer.colors.GREEN)


@app.command("serve")
def serve_customer(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    max_model_len: int = typer.Option(4096),
    gpu_memory_utilization: float = typer.Option(0.90),
):
    """Serve this customer's currently-deployed production model.

    Prefers the merged 16-bit model if the deployed candidate's training
    merge succeeded; falls back to base_model + LoRA adapter otherwise.
    Nothing to serve until `deploy run` or `rollback run` has pointed
    production/ at a candidate.
    """
    import json as json_mod

    from ftspec.serving.serve import serve as serve_fn

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    production = ctx.production_dir()
    merged = production / "merged_model"
    adapter = production / "lora_adapter"
    meta_path = production / "candidate_meta.json"

    if merged.exists():
        model, lora = str(merged), None
    elif adapter.exists() and meta_path.exists():
        base_model = json_mod.loads(meta_path.read_text(encoding="utf-8"))["base_model"]
        model, lora = base_model, str(adapter)
    else:
        typer.secho(f"\nnothing deployed for {customer_id!r} yet -- run "
                     f"`ftplatform deploy run {customer_id} --candidate-id ...` first.\n",
                     fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    serve_fn(model=model, profile=ctx.profile, lora=lora, host=host, port=port,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)


if __name__ == "__main__":
    app()
