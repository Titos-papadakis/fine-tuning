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


if __name__ == "__main__":
    app()
