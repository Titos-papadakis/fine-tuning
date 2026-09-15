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
review_app = typer.Typer(help="Human review of flagged production traffic.",
                          no_args_is_help=True)
app.add_typer(review_app, name="review")
selfimprove_app = typer.Typer(help="Fold reviewed corrections back into training.",
                               no_args_is_help=True)
app.add_typer(selfimprove_app, name="selfimprove")
job_app = typer.Typer(help="The automation job queue.", no_args_is_help=True)
app.add_typer(job_app, name="job")
kaggle_app = typer.Typer(help="Run a queued job on Kaggle's free GPU instead of locally.",
                          no_args_is_help=True)
app.add_typer(kaggle_app, name="kaggle")
learning_app = typer.Typer(
    help="Cross-customer technique stats (scores/configs only, never customer data).",
    no_args_is_help=True)
app.add_typer(learning_app, name="learning")

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
def candidate_list_presets(
    workload: str | None = typer.Option(
        None, help="Reorder by historical mean score for this workload, if any prior "
                   "customer has run Stage A for it (see `ftplatform learning summary`). "
                   "Every preset is still shown -- this only changes the order."),
):
    """List the built-in Stage-A candidate base models."""
    from ftplatform.candidates.generator import DEFAULT_STAGE_A_CANDIDATES

    candidates = DEFAULT_STAGE_A_CANDIDATES
    scores: dict[str, float] = {}
    if workload:
        from ftplatform.learning import stats
        conn = connect()
        try:
            candidates = stats.reorder_by_history(conn, workload, "stage_a",
                                                   candidates, key=lambda c: c.candidate_id)
            scores = stats.mean_scores(conn, workload, "stage_a")
        finally:
            conn.close()

    typer.echo(f"\n  {'candidate_id':<20} {'label':<16} {'hist. score':>11}  base_model")
    typer.echo("  " + "-" * 92)
    for c in candidates:
        score = f"{scores[c.candidate_id]:.3f}" if c.candidate_id in scores else "-"
        typer.echo(f"  {c.candidate_id:<20} {c.label:<16} {score:>11}  {c.base_model}")
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
def candidate_lora_grid(
    workload: str | None = typer.Option(
        None, help="Reorder by historical mean score for this workload, if any prior "
                   "customer has run Stage B for it. Every grid point is still shown."),
):
    """Show the default Stage-B LoRA (r, alpha) grid."""
    from ftplatform.candidates.generator import DEFAULT_LORA_GRID, stage_b_candidate_id

    grid = DEFAULT_LORA_GRID
    scores: dict[str, float] = {}
    if workload:
        from ftplatform.learning import stats
        conn = connect()
        try:
            grid = stats.reorder_by_history(conn, workload, "stage_b", grid,
                                             key=lambda pair: stage_b_candidate_id(*pair))
            scores = stats.mean_scores(conn, workload, "stage_b")
        finally:
            conn.close()

    typer.echo(f"\n  {'r':>4} {'alpha':>6}  {'hist. score':>11}")
    for r, alpha in grid:
        score = scores.get(stage_b_candidate_id(r, alpha))
        typer.echo(f"  {r:>4} {alpha:>6}  {f'{score:.3f}' if score is not None else '-':>11}")
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
def candidate_stage_c_combos(
    workload: str | None = typer.Option(
        None, help="Reorder by historical mean score for this workload, if any prior "
                   "customer has run Stage C for it. Every combo is still shown."),
):
    """Show the default Stage-C quantization/constrained-decoding combos."""
    from ftplatform.candidates.generator import DEFAULT_STAGE_C_COMBOS, stage_c_candidate_id

    combos = DEFAULT_STAGE_C_COMBOS
    scores: dict[str, float] = {}
    if workload:
        from ftplatform.learning import stats
        conn = connect()
        try:
            combos = stats.reorder_by_history(
                conn, workload, "stage_c", combos,
                key=lambda combo: stage_c_candidate_id("stageC", *combo))
            scores = stats.mean_scores(conn, workload, "stage_c")
        finally:
            conn.close()

    typer.echo(f"\n  {'quantization':<14} {'constrained':<12} {'hist. score':>11}")
    for q, c in combos:
        score = scores.get(stage_c_candidate_id("stageC", q, c))
        typer.echo(f"  {q:<14} {str(c):<12} {f'{score:.3f}' if score is not None else '-':>11}")
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
        try:
            ctx = CustomerContext(conn, customer_id)
        except UnknownCustomerError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e

        ids = {c.strip(): c.strip() for c in candidates.split(",") if c.strip()}
        try:
            ranked = leaderboard.build(ctx, ids, conn=conn, min_adherence_pct=min_adherence)
        except FileNotFoundError as e:
            typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
    finally:
        conn.close()

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
    capture: bool = typer.Option(
        True, help="Capture request/response pairs for the self-improve review queue."),
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

    if capture:
        from ftplatform.monitoring import capture as capture_mod
        capture_mod.attach(ctx)

    serve_fn(model=model, profile=ctx.profile, lora=lora, host=host, port=port,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)


@review_app.command("scan")
def review_scan(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """Scan captured production traffic and queue anything worth a look."""
    from ftplatform.monitoring import review_queue
    from ftplatform.selfimprove.label import pending as list_pending

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    new_rows = review_queue.scan(ctx)
    typer.echo(f"\n{len(new_rows)} new row(s) flagged for review "
                f"({len(list_pending(ctx))} pending total)")


@review_app.command("pending")
def review_pending(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """List rows currently awaiting human review."""
    from ftplatform.selfimprove.label import pending

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    rows = pending(ctx)
    if not rows:
        typer.echo("nothing pending")
        return
    for r in rows:
        typer.echo(f"\n[{r['request_id']}] reasons={r['reasons']} "
                    f"latency_ms={r['latency_ms']:.0f} valid={r['contract_valid']}")
        if r.get("input"):
            typer.echo(f"  input:  {r['input'][:200]}")
            typer.echo(f"  output: {r['output']}")
        else:
            typer.echo("  (text not captured for this profile's compliance regime)")


@review_app.command("skip")
def review_skip(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    request_id: str = typer.Option(..., help="From `ftplatform review pending`."),
):
    """Confirm a flagged row was actually fine -- drop it, no correction produced."""
    from ftplatform.selfimprove import label

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
        row = label.skip(ctx, request_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    typer.secho(f"\nskipped {row['request_id']!r} -- no correction recorded.",
                 fg=typer.colors.GREEN)


@review_app.command("correct")
def review_correct(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    request_id: str = typer.Option(..., help="From `ftplatform review pending`."),
    output: str = typer.Option(..., help="The correct output, as a JSON object."),
):
    """Confirm a flagged row was wrong and supply the correct output.

    Appends to memory/corrections.jsonl; `ftplatform selfimprove run` folds
    it into training.
    """
    import json as json_mod

    from ftplatform.selfimprove import label

    try:
        corrected = json_mod.loads(output)
    except json_mod.JSONDecodeError as e:
        typer.secho(f"--output must be valid JSON: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
        row = label.record_correction(ctx, request_id, corrected)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    typer.secho(f"\ncorrection recorded for {row['request_id']!r}.", fg=typer.colors.GREEN)


@selfimprove_app.command("run")
def selfimprove_run(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    base_model: str = typer.Option(..., help="The currently-winning base model to retrain."),
    lora_r: int = typer.Option(..., help="The currently-winning LoRA rank."),
    lora_alpha: int = typer.Option(..., help="The currently-winning LoRA alpha."),
    min_adherence: float = typer.Option(98.0),
    epsilon: float = typer.Option(0.02),
    max_steps: int = typer.Option(-1),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Fold reviewed corrections into training, retrain, and deploy-if-better.

    Does nothing (no GPU time spent) if `ftplatform review correct` has not
    produced any corrections since the last cycle.
    """
    from ftplatform.selfimprove import retrain_cycle

    conn = connect()
    try:
        ctx = CustomerContext(conn, customer_id)
        result = retrain_cycle.run_cycle(ctx, base_model, lora_r, lora_alpha, conn=conn,
                                           min_adherence_pct=min_adherence, epsilon=epsilon,
                                           max_steps=max_steps, gpu_cost_per_hour=gpu_cost_per_hour)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    if result["candidate_id"] is None:
        typer.echo(f"\n{result['reason']}")
    elif result["deployed"]:
        typer.secho(f"\nretrained candidate {result['candidate_id']!r} "
                     f"({result['folded']['added']} correction(s) folded) -- PROMOTED.",
                     fg=typer.colors.GREEN)
    else:
        typer.secho(f"\nretrained candidate {result['candidate_id']!r} "
                     f"({result['folded']['added']} correction(s) folded) -- "
                     f"NOT promoted: {result['reason']}", fg=typer.colors.YELLOW)


@app.command("approve")
def approve_customer(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """Approve a customer for their first automated deployment.

    Required exactly once, before the job worker will run a 'deploy' job
    for a customer with no prior deployment (see jobs/approvals.py).
    Deploying by hand (`ftplatform deploy run`) never needed this -- running
    that command yourself already is the human decision; this is only for
    the automated `job` pipeline.
    """
    from ftplatform.jobs import approvals

    conn = connect()
    try:
        approvals.approve(conn, customer_id)
    finally:
        conn.close()
    typer.secho(f"\n{customer_id!r} approved for automated deployment.", fg=typer.colors.GREEN)


@job_app.command("enqueue")
def job_enqueue(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    kind: str = typer.Option(
        ..., help="baseline|stage_a|stage_b|stage_c|leaderboard|deploy|retrain_cycle"),
    payload: str = typer.Option("{}", help="JSON kwargs for the job -- shape depends on --kind; "
                                  "see jobs/worker.py::_dispatch for each kind's expected keys."),
):
    """Queue one unit of work for this customer. Nothing runs until a
    worker (`job run-one` / `job worker`) picks it up."""
    import json as json_mod

    from ftplatform.jobs import queue

    try:
        payload_dict = json_mod.loads(payload)
    except json_mod.JSONDecodeError as e:
        typer.secho(f"--payload must be valid JSON: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    conn = connect()
    try:
        job_id = queue.enqueue(conn, customer_id, kind, payload_dict)
    finally:
        conn.close()
    typer.secho(f"\nqueued job {job_id!r} ({kind}) for {customer_id!r}.", fg=typer.colors.GREEN)


@job_app.command("list")
def job_list(customer_id: str | None = typer.Argument(None, help="Filter to one customer.")):
    """List jobs, newest last."""
    from ftplatform.jobs import queue

    conn = connect()
    try:
        jobs = queue.list_jobs(conn, customer_id)
    finally:
        conn.close()

    if not jobs:
        typer.echo("no jobs")
        return
    typer.echo(f"\n  {'id':<18} {'customer':<12} {'kind':<16} {'status':<8} created")
    typer.echo("  " + "-" * 80)
    for j in jobs:
        typer.echo(f"  {j['id']:<18} {j['customer_id']:<12} {j['kind']:<16} "
                    f"{j['status']:<8} {j['created_at']}")
    typer.echo("")


@job_app.command("run-one")
def job_run_one():
    """Pop and execute the oldest pending job, then exit."""
    from ftplatform.jobs import worker

    conn = connect()
    try:
        job = worker.run_one(conn)
    finally:
        conn.close()

    if job is None:
        typer.echo("queue is empty")
        return
    color = typer.colors.GREEN if job["status"] == "done" else typer.colors.RED
    typer.secho(f"\njob {job['id']!r} ({job['kind']}) -> {job['status']}", fg=color)
    if job["status"] == "failed":
        typer.echo(f"  {job['result']['error']}")


@job_app.command("worker")
def job_worker(
    poll_interval_s: float = typer.Option(5.0),
    max_jobs: int | None = typer.Option(None, help="Stop after this many jobs (default: run until empty)."),
):
    """Run jobs until the queue is empty. Meant for a long-lived process or
    a Colab cell -- not for unattended overnight automation (see the
    module's docstring on free Colab T4 session limits)."""
    from ftplatform.jobs import worker

    conn = connect()
    try:
        count = worker.run_loop(conn, poll_interval_s=poll_interval_s, max_jobs=max_jobs)
    finally:
        conn.close()
    typer.echo(f"\nran {count} job(s); queue empty.")


@job_app.command("requeue-stale")
def job_requeue_stale(
    stale_after_s: int = typer.Option(3600, help="How long 'running' before assuming the worker died."),
):
    """Mark 'running' jobs whose worker likely died as failed, for manual re-queue."""
    from ftplatform.jobs import queue

    conn = connect()
    try:
        stale_ids = queue.requeue_stale(conn, stale_after_s)
    finally:
        conn.close()
    typer.echo(f"\n{len(stale_ids)} stale job(s) marked failed: {stale_ids}")


@kaggle_app.command("start")
def kaggle_start(
    job_id: str = typer.Argument(..., help="A job id from `job enqueue`/`job list`."),
    customer_id: str = typer.Argument(..., help="Customer id the job belongs to."),
    owner: str = typer.Option(..., help="Your Kaggle username."),
):
    """Export a pending job, upload it as a private Kaggle dataset, and push
    the kernel that runs it. Does not wait -- run `kaggle poll` afterwards
    (repeatedly, until it reports something other than 'still running')."""
    from ftplatform.remote.run_on_kaggle import default_work_dir, start_job_on_kaggle

    conn = connect()
    try:
        result = start_job_on_kaggle(conn, customer_id, job_id, owner,
                                      default_work_dir(job_id))
    finally:
        conn.close()
    typer.secho(f"\npushed kernel {result['kernel_id']!r} for job {job_id!r}.",
                fg=typer.colors.GREEN)
    typer.echo(f"  dataset: {result['dataset_id']}")
    typer.echo(f"  check progress with: ftplatform kaggle poll {job_id} {customer_id} "
               f"--owner {owner}")


@kaggle_app.command("poll")
def kaggle_poll(
    job_id: str = typer.Argument(..., help="The job id passed to `kaggle start`."),
    customer_id: str = typer.Argument(..., help="Customer id the job belongs to."),
    owner: str = typer.Option(..., help="Your Kaggle username."),
):
    """Check a Kaggle run's status once. If it's finished, download and
    merge the result back into customers.db and customers/<id>/; otherwise
    report that it's still running so you can call this again later."""
    from ftplatform.remote.run_on_kaggle import default_work_dir, kernel_id_for, poll_and_finish_job

    conn = connect()
    try:
        job = poll_and_finish_job(conn, customer_id, job_id,
                                   kernel_id_for(job_id, owner), default_work_dir(job_id))
    finally:
        conn.close()

    if job is None:
        typer.echo("still running -- check again later")
        return
    color = typer.colors.GREEN if job["status"] == "done" else typer.colors.RED
    typer.secho(f"\njob {job['id']!r} ({job['kind']}) -> {job['status']}", fg=color)
    if job["status"] == "failed":
        typer.echo(f"  {job['result']['error']}")


@learning_app.command("summary")
def learning_summary(
    workload: str = typer.Argument(..., help="e.g. 'saas_support'."),
):
    """How each technique has scored so far across every customer on this
    workload. Never shows customer data -- only candidate_id (a technique
    descriptor, e.g. 'stageB-r16-a16') and the same composite score the
    leaderboard computes. See `ftplatform/learning/` for what is and isn't
    shared across customers."""
    from ftplatform.learning import stats

    conn = connect()
    try:
        rows = stats.summary(conn, workload)
    finally:
        conn.close()

    if not rows:
        typer.echo(f"no leaderboard has been built yet for workload {workload!r}")
        return
    typer.echo(f"\n  {'stage':<10} {'candidate_id':<20} {'runs':>5} {'mean score':>11} {'gated':>6}")
    typer.echo("  " + "-" * 60)
    for r in rows:
        mean_score = f"{r['mean_score']:.3f}" if r["mean_score"] is not None else "-"
        typer.echo(f"  {r['stage']:<10} {r['candidate_id']:<20} {r['n']:>5} "
                    f"{mean_score:>11} {r['gated_count']:>6}")
    typer.echo("")


@app.command("serve-shared")
def serve_shared(
    workload: str = typer.Argument(..., help="e.g. 'saas_support'."),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    max_model_len: int = typer.Option(4096),
    gpu_memory_utilization: float = typer.Option(0.90),
    max_lora_rank: int = typer.Option(16),
):
    """Serve every customer on `workload` with a production deployment from
    ONE shared vLLM engine -- one base model's weights loaded once, each
    customer's LoRA adapter hot-swapped in per request (see
    ftplatform/serving/shared_server.py) -- instead of one process per
    customer.

    Customers whose production adapter was trained from a different base
    model than the majority are excluded and named in the output; a request
    picks its adapter by passing the customer id as the OpenAI `model`
    field.
    """
    from ftplatform.serving.shared_server import discover_production_adapters, group_by_base_model
    from ftspec.core.registry import load_profile
    from ftspec.serving.serve import serve as ftspec_serve

    conn = connect()
    try:
        adapters = discover_production_adapters(conn, workload)
    finally:
        conn.close()

    if not adapters:
        typer.secho(f"no customer on workload {workload!r} has a production deployment yet",
                     fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    groups = group_by_base_model(adapters)
    if not groups:
        typer.secho("no customer's base model could be determined -- nothing to serve",
                     fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    base_model, lora = max(groups.items(), key=lambda kv: len(kv[1]))
    excluded = set(adapters) - set(lora)
    typer.echo(f"\nserving {len(lora)} customer(s) on base model {base_model!r}: "
                f"{sorted(lora)}")
    if excluded:
        typer.secho(f"excluded (different base model, run again on a separate instance "
                     f"for these): {sorted(excluded)}", fg=typer.colors.YELLOW)

    profile = load_profile(workload)
    ftspec_serve(base_model, profile, lora=lora, host=host, port=port,
                 max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                 max_lora_rank=max_lora_rank)


if __name__ == "__main__":
    app()
