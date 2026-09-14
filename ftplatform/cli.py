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


if __name__ == "__main__":
    app()
