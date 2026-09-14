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


if __name__ == "__main__":
    app()
