"""
`ftplatform` — customer-scoped orchestration around the `ftspec` engine.

    ftplatform customer add <id> --name "Acme" --workload saas_support
    ftplatform customer list

Every command resolves a customer id to an isolated Config/Profile pair via
CustomerContext (ftplatform/customers/context.py) before touching `ftspec` --
no command here is allowed to use an unscoped path.
"""
from __future__ import annotations

from pathlib import Path

import typer

from ftplatform import audit
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
apikey_app = typer.Typer(help="API keys for the serving layer.", no_args_is_help=True)
app.add_typer(apikey_app, name="api-key")
usage_app = typer.Typer(help="Per-customer usage/billing counters.", no_args_is_help=True)
app.add_typer(usage_app, name="usage")
db_app = typer.Typer(help="The platform database itself (customers.db).", no_args_is_help=True)
app.add_typer(db_app, name="db")
pipeline_app = typer.Typer(help="The whole onboarding chain for a customer, hands-off.",
                            no_args_is_help=True)
app.add_typer(pipeline_app, name="pipeline")
report_app = typer.Typer(help="Customer-facing reports.", no_args_is_help=True)
app.add_typer(report_app, name="report")
privacy_app = typer.Typer(help="Per-customer redaction and retention of captured traffic.",
                          no_args_is_help=True)
app.add_typer(privacy_app, name="privacy")
audit_app = typer.Typer(help="Who changed what, when -- the append-only audit log.",
                        no_args_is_help=True)
app.add_typer(audit_app, name="audit")
agent_app = typer.Typer(help="The customer's model as a support agent: chat plus tool calls.",
                        no_args_is_help=True)
app.add_typer(agent_app, name="agent")
agent_tools_app = typer.Typer(help="The agent's tool catalog and permissions.", no_args_is_help=True)
agent_app.add_typer(agent_tools_app, name="tools")
agent_kb_app = typer.Typer(help="Documents the agent can search (knowledge_search tool).",
                           no_args_is_help=True)
agent_app.add_typer(agent_kb_app, name="knowledge")
agent_packs_app = typer.Typer(help="Ready-made process packs (returns, orders, account, ...).",
                              no_args_is_help=True)
agent_app.add_typer(agent_packs_app, name="packs")


def _ctx_or_exit(conn, customer_id: str) -> CustomerContext:
    try:
        return CustomerContext(conn, customer_id)
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

log = get_logger("ftplatform.cli")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging.")):
    setup_logging(verbose)


@customer_app.command("add")
def customer_add(
    customer_id: str = typer.Argument(..., help="Short, URL-safe id, e.g. 'acme'."),
    name: str = typer.Option(..., help="Display name."),
    workload: str = typer.Option(
        ..., help="Vertical to run for this customer, e.g. saas_support -- or 'custom' "
                  "with --schema for the customer's own JSON Schema."),
    schema: Path | None = typer.Option(None, exists=True, dir_okay=False,
                                       help="JSON Schema of the record to extract (custom only)."),
    headline_field: str | None = typer.Option(
        None, help="Dotted path of the field that matters most, e.g. 'issue.priority' "
                   "(custom only; reported as the headline accuracy)."),
    free_text_field: list[str] = typer.Option(
        [], help="Field scored by token overlap instead of exact match, e.g. a summary "
                 "(custom only, repeatable)."),
    regime: str = typer.Option("none", help="none | GDPR | HIPAA | PCI-DSS (custom only)."),
    allow_external_api: bool | None = typer.Option(
        None, "--allow-external-api/--no-external-api",
        help="May this customer's text be sent to hosted APIs (auto-labeling, gpt4o "
             "baselines)? Default: no for HIPAA/PCI-DSS, yes otherwise."),
):
    """Register a new customer. Its data and models are isolated from every other one."""
    from ftplatform.customers import custom_workload

    is_custom = workload == custom_workload.CUSTOM
    if not is_custom and workload not in available():
        typer.secho(f"Unknown workload {workload!r}. Choices: "
                    f"{', '.join([*available(), 'custom'])}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if is_custom != (schema is not None):
        typer.secho("--schema is required with --workload custom, and only valid with it",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    settings = {"headline_field": headline_field, "free_text_fields": tuple(free_text_field),
                "regime": regime, "allows_external_api": allow_external_api}
    if is_custom:
        try:
            custom_workload.build_profile(schema, **settings)  # fail before creating anything
        except (ValueError, OSError) as e:
            typer.secho(f"invalid schema: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e

    conn = connect()
    try:
        customer = store.create(conn, customer_id, name, workload)
        if is_custom:
            import ftspec.config as config_mod
            custom_workload.install(config_mod.REPO_ROOT / "customers" / customer.id,
                                    schema, **settings)
        audit.record(conn, "customer.add", customer.id,
                     {"workload": workload, "regime": regime if is_custom else None})
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    typer.secho(f"customer {customer.id!r} added (workload={customer.workload})",
                fg=typer.colors.GREEN)
    if is_custom:
        typer.echo(f"  custom workloads train only on the customer's own data -- next: "
                   f"ftplatform customer import {customer.id} <tickets.csv>")


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


@customer_app.command("import")
def customer_import(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    path: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="The customer's tickets: .csv (a text column, optional "
                                     "output/label_json column) or .jsonl (input/output)."),
    label_with: str | None = typer.Option(
        None, help="Auto-label rows that have no label, e.g. 'gpt-4o-mini' (needs "
                   "OPENAI_API_KEY; costs roughly $0.0003 per ticket). Off by default: "
                   "unlabeled rows go to imports/to_label.jsonl instead."),
    max_auto_label: int | None = typer.Option(None, help="Cap on auto-labeled rows (cost guard)."),
):
    """Bring a customer's own tickets in as training data.

    Every row is validated against the workload's schema; bad labels are
    rejected with a reason (imports/rejected.jsonl), never coerced. The
    result (imports/corpus.jsonl) is used automatically by `baseline run`
    and `pipeline start` instead of synthetic data."""
    from ftplatform.customers import importer

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
    finally:
        conn.close()

    labeler = None
    if label_with:
        try:
            labeler = importer.openai_labeler(ctx.profile, model=label_with)
        except PermissionError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
    try:
        r = importer.import_corpus(ctx, path, labeler=labeler, labeler_name=label_with or "",
                                   max_auto_label=max_auto_label)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    conn = connect()
    try:
        audit.record(conn, "data.import", customer_id,
                     {k: r[k] for k in ("read", "labeled", "auto_labeled", "rejected",
                                         "to_label", "corpus_total")}
                     | {"labeler": label_with})
    finally:
        conn.close()

    typer.echo(f"\nread {r['read']} row(s) from {path}")
    typer.echo(f"  labeled by customer:  {r['labeled']}")
    typer.echo(f"  auto-labeled:         {r['auto_labeled']}")
    typer.echo(f"  rejected:             {r['rejected']}  (see imports/rejected.jsonl)")
    typer.echo(f"  still need a label:   {r['to_label']}  (imports/to_label.jsonl)")
    typer.echo(f"  skipped (dup/empty):  {r['duplicate'] + r['empty']}")
    color = typer.colors.GREEN if r["enough_to_train"] else typer.colors.YELLOW
    typer.secho(f"\ncorpus now {r['corpus_total']} example(s) -> {r['corpus_path']}", fg=color)
    if not r["enough_to_train"]:
        typer.secho(f"  under {importer.MIN_RECOMMENDED_ROWS}: fine-tuning on this little is "
                    f"unlikely to beat prompting -- label more first.", fg=typer.colors.YELLOW)
    if r["auto_labeled"]:
        typer.echo("  spot-check imports/review_sample.jsonl before training on auto-labels.")


@customer_app.command("delete")
def customer_delete(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    yes: bool = typer.Option(False, "--yes", help="Actually delete. Without it, only shows "
                                                 "what would be removed."),
    force: bool = typer.Option(False, help="Delete even with a live Stripe subscription."),
    skip_kaggle: bool = typer.Option(
        False, help="Don't delete the customer's private datasets/kernels on Kaggle (do it by "
                    "hand). Without this, a Kaggle failure stops the deletion before anything "
                    "local is removed, so it can be retried."),
):
    """Remove a customer completely: their private Kaggle datasets/kernels,
    database rows, customers/<id>/ (data, models, logs) and local staging
    dirs. Irreversible -- take a `db backup` first if in doubt."""
    from ftplatform.jobs import queue
    from ftplatform.remote import kaggle_ops

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        n_remote = conn.execute("SELECT COUNT(*) FROM kaggle_artifacts WHERE customer_id = ?",
                                (customer_id,)).fetchone()[0]
        if not yes:
            n_jobs = len(queue.list_jobs(conn, customer_id))
            typer.echo(f"\nwould delete {customer_id!r} ({ctx.customer.name}): {n_jobs} job(s), "
                       f"{n_remote} Kaggle job artefact set(s), every db row, and "
                       f"{ctx.customer_root()}")
            typer.echo("re-run with --yes to do it.")
            return
        if store.billable(conn, customer_id) and not force:
            typer.secho(f"{customer_id!r} still has a billable Stripe subscription -- cancel it in "
                        f"Stripe first, or pass --force.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        remote = None
        if n_remote and not skip_kaggle:
            try:
                remote = kaggle_ops.delete_job_artifacts(conn, customer_id)
            except kaggle_ops.KaggleCommandError as e:
                typer.secho(f"Kaggle deletion failed -- nothing local was deleted, retry "
                            f"later:\n{e}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1) from e
        result = store.delete(conn, customer_id, force=force)
        if remote:
            audit.record(conn, "kaggle.delete", customer_id, remote)
    finally:
        conn.close()

    rows = sum(result["rows"].values())
    typer.secho(f"\ndeleted {customer_id!r}: {rows} db row(s), {len(result['paths'])} "
                f"director(ies)" + (f", {remote['deleted']} Kaggle dataset(s)/kernel(s)"
                                    if remote else "") + ".", fg=typer.colors.GREEN)
    if n_remote and skip_kaggle:
        typer.secho("  Kaggle artefacts were NOT deleted (--skip-kaggle) -- remove the "
                    "ftplatform-job-* datasets/notebooks by hand.", fg=typer.colors.YELLOW)


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
    cache_size: int = typer.Option(0, help="Exact-match response cache size (0 disables it)."),
    require_auth: bool = typer.Option(
        True, "--require-auth/--no-require-auth",
        help="Require a valid API key on every request (see `ftplatform api-key create`). "
             "On by default for anything customer-facing -- turn off only for local dev."),
    rate_limit_per_min: int = typer.Option(0, help="Per-minute request cap (0 disables it)."),
):
    """Serve this customer's currently-deployed production model.

    Prefers the merged 16-bit model if the deployed candidate's training
    merge succeeded; falls back to base_model + LoRA adapter otherwise.
    Nothing to serve until `deploy run` or `rollback run` has pointed
    production/ at a candidate.
    """
    import json as json_mod

    from ftplatform.auth import keys as keys_mod
    from ftplatform.billing import usage as usage_mod
    from ftplatform.serving.hooks import compose
    from ftspec.serving.serve import serve as serve_fn

    conn = connect(shared=True)            # held while serving, used per request
    try:
        try:
            ctx = CustomerContext(conn, customer_id)
        except UnknownCustomerError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e

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

        api_key_resolver = None
        if require_auth:
            if not keys_mod.list_keys(conn, customer_id):
                typer.secho(
                    f"\n{customer_id!r} has no API key yet -- run "
                    f"`ftplatform api-key create {customer_id}` first, or pass "
                    f"--no-require-auth for local dev.\n", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            api_key_resolver = keys_mod.build_resolver(conn, customer_ids={customer_id})

        hooks = [usage_mod.build_hook(conn)]
        if capture:
            from ftplatform.monitoring import capture as capture_mod
            hooks.append(capture_mod.build_hook(ctx))
        import ftspec.serving.serve as serve_mod
        serve_mod.STATE.on_response = compose(*hooks)
    except BaseException:
        conn.close()
        raise
    # conn stays open while serving: the API-key resolver and the usage hook
    # query it on every request. (Closing it here made every authenticated
    # request fail with "Cannot operate on a closed database".)
    try:
        serve_fn(model=model, profile=ctx.profile, lora=lora, host=host, port=port,
                 max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                 cache_size=cache_size, api_key_resolver=api_key_resolver,
                 rate_limit_per_min=rate_limit_per_min)
    finally:
        conn.close()


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

    import json as json_mod

    from ftplatform.selfimprove.label import suggestion

    rows = pending(ctx)
    if not rows:
        typer.echo("nothing pending")
        return
    for r in rows:
        typer.echo(f"\n[{r['request_id']}] reasons={r['reasons']} "
                    f"latency_ms={r['latency_ms']:.0f} valid={r['contract_valid']}")
        if r.get("input"):
            typer.echo(f"  input:  {r['input'][:200]}")
            s = suggestion(r)
            if s is not None:
                typer.echo(f"  model:  {json_mod.dumps(s, ensure_ascii=False)}")
                typer.echo(f"  -> right?  ftplatform review correct {customer_id} "
                           f"--request-id {r['request_id']}")
                typer.echo("  -> fix one field:  ... --set path.to.field=value")
            else:
                typer.echo(f"  output: {r['output']}")
                typer.echo("  -> no parseable output; supply the full record with --output '{...}'")
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
    output: str | None = typer.Option(
        None, help="The full correct output as a JSON object. Omit to start from the "
                   "model's own output instead."),
    set_: list[str] = typer.Option(
        [], "--set", help="Override one field of the starting record: path.to.field=value "
                          "(repeatable). Values that parse as JSON are used as such."),
):
    """Record the correct output for a flagged row.

    Starts from the model's own output (or --output), applies any --set
    overrides, and checks the result against the schema before recording
    it -- a row with neither --output nor --set confirms the model was
    right, which is still a real-traffic training example. Appends to
    memory/corrections.jsonl; `ftplatform selfimprove run` folds it in.
    """
    import json as json_mod

    from ftplatform.selfimprove import label

    corrected = None
    if output is not None:
        try:
            corrected = json_mod.loads(output)
        except json_mod.JSONDecodeError as e:
            typer.secho(f"--output must be valid JSON: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        edits = label.parse_set_args(set_)
        row = label.correct(ctx, request_id, edits=edits, output=corrected)
        audit.record(conn, "review.correct", customer_id,
                     {"request_id": request_id, "fields_set": sorted(edits),
                      "full_output_supplied": corrected is not None})
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()

    what = "confirmed as correct" if output is None and not set_ else "correction recorded"
    typer.secho(f"\n{what} for {row['request_id']!r}.", fg=typer.colors.GREEN)


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
        audit.record(conn, "selfimprove.run", customer_id,
                     {"added": result["folded"]["added"], "rejected": result["folded"]["rejected"],
                      "candidate_id": result["candidate_id"], "deployed": result.get("deployed")})
    except UnknownCustomerError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    finally:
        conn.close()

    folded = result["folded"]
    if result["candidate_id"] is None:
        if folded["rejected"]:
            # Distinct from "nothing submitted" -- every pending correction
            # failed contract validation and was archived unfolded. Silently
            # printing the same "no corrections to fold" message here would
            # hide that a human's review work was just discarded.
            typer.secho(f"\n{folded['rejected']} correction(s) all FAILED validation and were "
                        f"archived without training -- nothing was silently lost, but nothing "
                        f"was folded either:", fg=typer.colors.RED, err=True)
            for reason in folded["rejected_reasons"]:
                typer.echo(f"  - {reason}")
        else:
            typer.echo(f"\n{result['reason']}")
    elif result["deployed"]:
        typer.secho(f"\nretrained candidate {result['candidate_id']!r} "
                     f"({folded['added']} correction(s) folded) -- PROMOTED.",
                     fg=typer.colors.GREEN)
    else:
        typer.secho(f"\nretrained candidate {result['candidate_id']!r} "
                     f"({folded['added']} correction(s) folded) -- "
                     f"NOT promoted: {result['reason']}", fg=typer.colors.YELLOW)
    if folded["rejected"] and result["candidate_id"] is not None:
        # Some corrections were folded, but not all -- worth surfacing even
        # on a path that otherwise looks successful.
        typer.secho(f"\n({folded['rejected']} other correction(s) failed validation and were "
                    f"archived unfolded)", fg=typer.colors.YELLOW)
        for reason in folded["rejected_reasons"]:
            typer.echo(f"  - {reason}")


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
        audit.record(conn, "deploy.approve", customer_id)
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
    required_hours: float = typer.Option(
        None, help="Refuse to push if remaining weekly GPU quota is below this, or if "
                    "this job's kernel already has a session running/queued."),
):
    """Export a pending job, upload it as a private Kaggle dataset, and push
    the kernel that runs it. Does not wait -- run `kaggle poll` afterwards
    (repeatedly, until it reports something other than 'still running')."""
    from ftplatform.remote import kaggle_ops
    from ftplatform.remote.run_on_kaggle import default_work_dir, start_job_on_kaggle

    conn = connect()
    try:
        result = start_job_on_kaggle(conn, customer_id, job_id, owner,
                                      default_work_dir(job_id), required_hours=required_hours)
    except (kaggle_ops.InsufficientQuotaError, kaggle_ops.ConcurrentSessionError) as e:
        typer.secho(f"refusing to push: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()
    typer.secho(f"\npushed kernel {result['kernel_id']!r} for job {job_id!r}.",
                fg=typer.colors.GREEN)
    typer.echo(f"  dataset: {result['dataset_id']}")
    typer.echo(f"  check progress with: ftplatform kaggle poll {job_id} {customer_id} "
               f"--owner {owner}")


@kaggle_app.command("quota")
def kaggle_quota():
    """Show the weekly Kaggle GPU/TPU quota (used/remaining/total/refresh)."""
    from ftplatform.remote import kaggle_ops

    quota = kaggle_ops.get_gpu_quota()
    typer.echo(f"GPU: {quota['used']:.2f}h used, {quota['remaining']:.2f}h remaining "
               f"of {quota['total']:.2f}h -- resets {quota['refresh_at']}")


@kaggle_app.command("push")
def kaggle_push(
    kernel_dir: Path = typer.Argument(..., help="Directory with kernel-metadata.json + code, "
                                                  "e.g. notebooks/kaggle_runner."),
    kernel_id: str = typer.Option(..., help="owner/slug -- must match kernel-metadata.json's id."),
    required_hours: float = typer.Option(
        None, help="Refuse to push if remaining weekly GPU quota is below this."),
):
    """Push any kernel directory through the same quota/concurrency guard as
    `kaggle start` -- for ad-hoc kernels (like notebooks/kaggle_runner's
    baseline script) that don't go through the job-packet pipeline. Always
    refuses if the kernel already has a session running/queued, since a push
    on top of one does not cancel it -- it starts a second, concurrent
    session and doubles GPU-hour burn instead."""
    from ftplatform.remote import kaggle_ops

    try:
        kaggle_ops.safe_push_kernel(kernel_dir, kernel_id, required_hours)
    except (kaggle_ops.InsufficientQuotaError, kaggle_ops.ConcurrentSessionError) as e:
        typer.secho(f"refusing to push: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.secho(f"pushed {kernel_id!r} from {kernel_dir}", fg=typer.colors.GREEN)


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
    cache_size: int = typer.Option(0, help="Exact-match response cache size (0 disables it)."),
    capture: bool = typer.Option(
        True, help="Capture request/response pairs for the self-improve review queue "
                    "(requires auth -- see --require-auth)."),
    require_auth: bool = typer.Option(
        True, "--require-auth/--no-require-auth",
        help="Require a valid API key on every request, resolved to the customer whose "
             "adapter is then used -- without this, ANY caller can select ANY served "
             "customer's adapter just by naming it, so --no-require-auth is a real "
             "cross-customer exposure, not a convenience toggle. Off only for local dev."),
    rate_limit_per_min: int = typer.Option(0, help="Per-customer request cap (0 disables it)."),
):
    """Serve every customer on `workload` with a production deployment from
    ONE shared vLLM engine -- one base model's weights loaded once, each
    customer's LoRA adapter hot-swapped in per request (see
    ftplatform/serving/shared_server.py) -- instead of one process per
    customer.

    Customers whose production adapter was trained from a different base
    model than the majority are excluded and named in the output. With auth
    on (the default), the adapter is chosen by the caller's API key, not by
    the `model` field it sends.
    """
    from ftplatform.auth import keys as keys_mod
    from ftplatform.billing import usage as usage_mod
    from ftplatform.serving.hooks import compose
    from ftplatform.serving.shared_server import discover_production_adapters, group_by_base_model
    from ftspec.core.registry import load_profile
    from ftspec.serving.serve import serve as ftspec_serve

    if workload == "custom":
        typer.secho("custom-workload customers each have their own schema, so they cannot share "
                    "one server's profile -- serve each with `ftplatform serve <customer_id>`.",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    conn = connect(shared=True)            # held while serving, used per request
    try:
        adapters = discover_production_adapters(conn, workload)
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

        api_key_resolver = None
        if require_auth:
            api_key_resolver = keys_mod.build_resolver(conn, customer_ids=set(lora))
            keyless = [c for c in lora if not keys_mod.list_keys(conn, c)]
            if keyless:
                typer.secho(f"note: {sorted(keyless)} have no API key yet and cannot be "
                             f"reached until one exists (`ftplatform api-key create ...`)",
                             fg=typer.colors.YELLOW)
        else:
            typer.secho("\n--no-require-auth on a shared server: any caller can select any "
                         "served customer's adapter by name. Do not use this against real "
                         "customer traffic.\n", fg=typer.colors.RED, err=True)

        import ftspec.serving.serve as serve_mod
        serve_mod.STATE.api_key_resolver = api_key_resolver  # so build_hook_shared() sees it

        hooks = [usage_mod.build_hook(conn)]
        if capture and require_auth:
            from ftplatform.monitoring import capture as capture_mod
            hooks.append(capture_mod.build_hook_shared(conn))
        elif capture:
            typer.secho("capture disabled: it requires auth to know which customer a "
                         "request belongs to (see --require-auth)", fg=typer.colors.YELLOW)
        serve_mod.STATE.on_response = compose(*hooks)
    except BaseException:
        conn.close()
        raise
    # conn stays open while serving: the API-key resolver and the usage hook
    # query it on every request. (Closing it here made every authenticated
    # request fail with "Cannot operate on a closed database".)
    try:
        profile = load_profile(workload)
        ftspec_serve(base_model, profile, lora=lora, host=host, port=port,
                     max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                     max_lora_rank=max_lora_rank, cache_size=cache_size,
                     api_key_resolver=api_key_resolver, rate_limit_per_min=rate_limit_per_min)
    finally:
        conn.close()


@apikey_app.command("create")
def apikey_create(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    label: str = typer.Option("", help="A note to tell this key apart from others later."),
):
    """Create a new API key for a customer. Shown once, here -- only its
    hash is ever stored, so losing it means creating a new one, not
    recovering this one."""
    from ftplatform.auth import keys as keys_mod

    conn = connect()
    try:
        try:
            CustomerContext(conn, customer_id)
        except UnknownCustomerError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e
        key = keys_mod.create_key(conn, customer_id, label=label)
        audit.record(conn, "apikey.create", customer_id,
                     {"label": label, "key_hash_prefix": keys_mod._hash(key)[:16]})
    finally:
        conn.close()

    typer.secho(f"\n{key}\n", fg=typer.colors.GREEN)
    typer.echo("This is shown once -- store it now. It will not be shown again.")


@apikey_app.command("list")
def apikey_list(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """List a customer's keys -- metadata only, never the plaintext."""
    from ftplatform.auth import keys as keys_mod

    conn = connect()
    try:
        rows = keys_mod.list_keys(conn, customer_id)
    finally:
        conn.close()

    if not rows:
        typer.echo(f"no API keys for {customer_id!r}")
        return
    typer.echo(f"\n  {'key_hash (prefix)':<20} {'label':<16} {'created_at':<22} status")
    typer.echo("  " + "-" * 80)
    for r in rows:
        status = "revoked" if r["revoked_at"] else "active"
        typer.echo(f"  {r['key_hash'][:16]:<20} {r['label']:<16} {r['created_at']:<22} {status}")
    typer.echo("")


@apikey_app.command("revoke")
def apikey_revoke(
    key_hash_prefix: str = typer.Argument(
        ..., help="A prefix of the key_hash shown by `api-key list` -- long enough to be "
                   "unambiguous, e.g. the first 12-16 characters."),
):
    """Revoke a key immediately. A revoked key never resolves again, but
    the row stays (audit trail) -- it is not deleted."""
    from ftplatform.auth import keys as keys_mod

    conn = connect()
    try:
        n = keys_mod.revoke_key(conn, key_hash_prefix)
        if n:
            audit.record(conn, "apikey.revoke", None,
                         {"key_hash_prefix": key_hash_prefix, "revoked": n})
    finally:
        conn.close()

    if n == 0:
        typer.secho(f"no active key matched prefix {key_hash_prefix!r}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho(f"\n{n} key(s) revoked.", fg=typer.colors.GREEN)


@usage_app.command("report")
def usage_report(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    period: str | None = typer.Option(None, help="'YYYY-MM'. Defaults to the current month."),
):
    """This customer's usage for one billing period."""
    from ftplatform.billing import usage as usage_mod

    conn = connect()
    try:
        r = usage_mod.report(conn, customer_id, period=period)
    finally:
        conn.close()

    typer.echo(f"\n{customer_id} / {r['period']}")
    typer.echo(f"  requests:           {r['requests']}")
    typer.echo(f"  prompt tokens:      {r['prompt_tokens']}")
    typer.echo(f"  completion tokens:  {r['completion_tokens']}")
    typer.echo(f"  cache hits:         {r['cache_hits']}\n")


@usage_app.command("report-all")
def usage_report_all(
    period: str | None = typer.Option(None, help="'YYYY-MM'. Defaults to the current month."),
):
    """Every customer's usage for one billing period -- the input to
    actually invoicing people."""
    from ftplatform.billing import usage as usage_mod

    conn = connect()
    try:
        rows = usage_mod.report_all(conn, period=period)
    finally:
        conn.close()

    if not rows:
        typer.echo("no usage recorded for this period")
        return
    typer.echo(f"\n  {'customer':<16} {'requests':>9} {'prompt tok':>11} "
                f"{'completion tok':>15} {'cache hits':>11}")
    typer.echo("  " + "-" * 70)
    for r in rows:
        typer.echo(f"  {r['customer_id']:<16} {r['requests']:>9} {r['prompt_tokens']:>11} "
                    f"{r['completion_tokens']:>15} {r['cache_hits']:>11}")
    typer.echo("")


# --- Stripe: whether a customer is an actual paying subscriber ---------------
# Separate from usage report/report-all above (what they consumed) -- this is
# whether money is actually changing hands. Requires `pip install ftspec[billing]`
# (the `stripe` package) for stripe-link/stripe-subscribe/stripe-status, which
# talk to a real Stripe account; nothing else in ftplatform needs it.

@usage_app.command("stripe-link")
def usage_stripe_link(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    email: str = typer.Option(..., help="Billing contact email for the Stripe customer."),
    name: str = typer.Option(..., help="Display name for the Stripe customer."),
):
    """Create a Stripe customer for CUSTOMER_ID and record the link --
    the step before `stripe-subscribe`. Safe to re-run; updates in place."""
    from ftplatform.billing import stripe_billing as sb

    conn = connect()
    try:
        stripe_customer_id = sb.create_stripe_customer(customer_id, email, name)
        sb.link_customer(conn, customer_id, email, stripe_customer_id)
        status = sb.billing_status(conn, customer_id)
    finally:
        conn.close()
    typer.secho(f"\n{customer_id} linked to Stripe customer {status['stripe_customer_id']!r}.",
                fg=typer.colors.GREEN)


@usage_app.command("stripe-subscribe")
def usage_stripe_subscribe(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    price: str = typer.Option(..., help="Stripe Price id for the flat monthly fee "
                                          "(created once, by hand, in the Stripe dashboard)."),
):
    """Subscribe an already-`stripe-link`ed customer to PRICE. This is the
    call that actually starts charging them."""
    from ftplatform.billing import stripe_billing as sb

    conn = connect()
    try:
        account = sb.billing_status(conn, customer_id)
        if account["stripe_customer_id"] is None:
            typer.secho(f"{customer_id} has no Stripe customer yet -- run "
                        f"`ftplatform usage stripe-link {customer_id} --email ... --name ...` "
                        f"first.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        stripe_subscription_id = sb.create_subscription(account["stripe_customer_id"], price)
        status = sb.fetch_subscription_status(stripe_subscription_id)
        sb.record_subscription(conn, customer_id, stripe_subscription_id, status)
        result = sb.billing_status(conn, customer_id)
        audit.record(conn, "billing.subscribe", customer_id,
                     {"price": price, "status": result["status"]})
    finally:
        conn.close()
    color = typer.colors.GREEN if result["status"] == "active" else typer.colors.YELLOW
    typer.secho(f"\n{customer_id} subscribed -> status {result['status']!r}.", fg=color)


@usage_app.command("stripe-status")
def usage_stripe_status(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    refresh: bool = typer.Option(False, help="Re-check Stripe for the latest status "
                                              "instead of showing the last-known one."),
):
    """This customer's billing status: unlinked, linked, or a live Stripe
    subscription status (active, past_due, canceled, ...)."""
    from ftplatform.billing import stripe_billing as sb

    conn = connect()
    try:
        status = (sb.refresh_subscription_status(conn, customer_id) if refresh
                   else sb.billing_status(conn, customer_id))
    finally:
        conn.close()
    typer.echo(f"\n{customer_id}: {status['status']}")
    if status["stripe_customer_id"]:
        typer.echo(f"  stripe customer:     {status['stripe_customer_id']}")
    if status["stripe_subscription_id"]:
        typer.echo(f"  stripe subscription: {status['stripe_subscription_id']}")
    typer.echo("")


@db_app.command("backup")
def db_backup(
    out: Path = typer.Option(
        None, help="Destination file. Defaults to backups/customers-<timestamp>.db "
                     "next to the live database."),
):
    """Consistent snapshot of customers.db -- every customer's metadata,
    deployments, API keys (hashed), usage counters and billing status.
    Never includes raw customer text, corpora, or model weights -- those
    live under customers/<id>/ on disk, not in this database (see
    ftplatform/db.py's module docstring)."""
    from datetime import datetime, timezone

    from ftplatform.db import DB_PATH
    from ftplatform.db import backup as db_backup_fn

    dest = out or (DB_PATH.parent / "backups" /
                    f"customers-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.db")
    conn = connect()
    try:
        written = db_backup_fn(conn, dest)
        audit.record(conn, "db.backup", None, {"path": str(written)})
    finally:
        conn.close()
    typer.secho(f"\nbacked up -> {written}", fg=typer.colors.GREEN)


# --- pipeline: onboarding as one command ------------------------------------------

def _print_pipeline(p: dict) -> None:
    color = {"done": typer.colors.GREEN, "failed": typer.colors.RED,
             "awaiting_approval": typer.colors.YELLOW}.get(p["status"])
    typer.secho(f"\n{p['customer_id']}: stage={p['stage']} status={p['status']}", fg=color)
    for stage, w in p["state"].get("winners", {}).items():
        typer.echo(f"  {stage:<8} winner: {w['candidate_id']} / {w['system']} (score {w['score']})")
    if p["state"].get("deploy_result"):
        d = p["state"]["deploy_result"]
        typer.echo(f"  deploy: {'PROMOTED' if d['deployed'] else 'not promoted -- ' + str(d['reason'])}")
    if p["state"].get("error"):
        typer.secho(f"  error: {p['state']['error']}", fg=typer.colors.RED)
    if p["status"] == "awaiting_approval":
        typer.echo(f"  review benchmark/leaderboard numbers, then: ftplatform approve "
                   f"{p['customer_id']} && ftplatform pipeline drive {p['customer_id']}")


@pipeline_app.command("start")
def pipeline_start(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    n_train: int = typer.Option(200, help="Synthetic corpus size (ignored with an imported corpus)."),
    n_val: int = typer.Option(40),
    n_eval: int = typer.Option(150),
    stage_a_top_k: int | None = typer.Option(
        None, help="Try only the k historically-best base models (all by default). Saves GPU "
                   "hours once earlier customers on this workload have run."),
    lora_grid_top_k: int | None = typer.Option(None, help="Same, for the LoRA grid."),
    with_stage_c: bool = typer.Option(False, help="Also sweep quantization/constrained decoding."),
    max_steps: int = typer.Option(-1, help="Cap Stage-B training steps (-1: full epochs)."),
    restart: bool = typer.Option(False, help="Replace a finished/failed pipeline."),
):
    """Set up the whole chain: baseline -> Stage A -> Stage B -> [Stage C] ->
    deploy. Nothing runs until `pipeline drive`."""
    from ftplatform.jobs import pipeline

    conn = connect()
    try:
        _ctx_or_exit(conn, customer_id)
        p = pipeline.start(conn, customer_id, restart=restart, n_train=n_train, n_val=n_val,
                           n_eval=n_eval, stage_a_top_k=stage_a_top_k,
                           lora_grid_top_k=lora_grid_top_k, with_stage_c=with_stage_c,
                           max_steps=max_steps)
        audit.record(conn, "pipeline.start", customer_id, {"restart": restart})
    except pipeline.PipelineError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()
    _print_pipeline(p)
    typer.echo(f"  next: ftplatform pipeline drive {customer_id} [--kaggle-owner <user>]")


@pipeline_app.command("drive")
def pipeline_drive(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    kaggle_owner: str | None = typer.Option(
        None, help="Run GPU jobs on Kaggle as this user (one at a time, polled until done) "
                   "instead of on this machine."),
    poll_interval_s: float = typer.Option(120.0, help="Seconds between Kaggle status checks."),
    required_hours: float | None = typer.Option(
        None, help="Refuse a Kaggle push if less weekly GPU quota than this remains."),
):
    """Run the pipeline until it's done, fails, or needs your approval.
    Safe to stop and re-run: it resumes where it left off."""
    from ftplatform.jobs import pipeline

    def say(msg: str) -> None:
        typer.echo(f"  {msg}")

    run_job = (pipeline.kaggle_runner(kaggle_owner, poll_interval_s=poll_interval_s,
                                      required_hours=required_hours, on_status=say)
               if kaggle_owner else pipeline.run_job_locally)
    conn = connect()
    try:
        _ctx_or_exit(conn, customer_id)
        p = pipeline.drive(conn, customer_id, run_job=run_job, on_status=say)
    except pipeline.PipelineError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    finally:
        conn.close()
    _print_pipeline(p)


@pipeline_app.command("status")
def pipeline_status(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """Where this customer's pipeline is, and its recent log."""
    from ftplatform.jobs import pipeline

    conn = connect()
    try:
        p = pipeline.get(conn, customer_id)
    finally:
        conn.close()
    if p is None:
        typer.echo(f"no pipeline for {customer_id!r} -- `ftplatform pipeline start {customer_id}`")
        return
    _print_pipeline(p)
    for line in p["state"]["log"][-10:]:
        typer.echo(f"    {line}")


# --- reports ------------------------------------------------------------------------

@report_app.command("monthly")
def report_monthly(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    period: str | None = typer.Option(None, help="'YYYY-MM'. Defaults to the current month."),
    monthly_fee: float | None = typer.Option(
        None, help="The customer's plan fee in USD, to show net savings (only when positive)."),
    out: Path | None = typer.Option(None, help="Output .html path (default: "
                                               "customers/<id>/reports/monthly-<period>.html)."),
):
    """The page to send a customer each month: volume handled, live quality,
    equivalent GPT-4o cost, and how their model improved."""
    from ftplatform.reporting import monthly

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        path, r = monthly.write(conn, ctx, period=period, monthly_fee_usd=monthly_fee, out_path=out)
    finally:
        conn.close()
    typer.secho(f"\nreport -> {path}", fg=typer.colors.GREEN)
    typer.echo(f"  {r['usage']['requests']} request(s) in {r['period']}"
               + (f", GPT-4o equivalent ${r['cost']['equivalent_usd']:,.2f}" if r["cost"] else ""))


# --- privacy / audit ----------------------------------------------------------------

@privacy_app.command("show")
def privacy_show(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """This customer's redaction rules and retention window."""
    from ftplatform import privacy

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
    finally:
        conn.close()
    p = privacy.load(ctx)
    captured = ctx.profile.compliance.regime == "none"
    typer.echo(f"\n{customer_id}: regime={ctx.profile.compliance.regime}, "
               f"text capture {'ON' if captured else 'OFF (regulated -- metadata only)'}")
    typer.echo(f"  redacted before writing: {', '.join(p['redact']) or 'nothing'}")
    typer.echo(f"  captured traffic kept:   {p['retention_days']} day(s)\n")


@privacy_app.command("set")
def privacy_set(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    retention_days: int | None = typer.Option(None, help="Keep captured traffic this long."),
    redact: str | None = typer.Option(
        None, help="Comma-separated rules to redact, e.g. 'email,phone,pan' ('' for none)."),
):
    """Change what is redacted from captured traffic and how long it is kept."""
    from ftplatform import privacy

    rules = None if redact is None else tuple(r.strip() for r in redact.split(",") if r.strip())
    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        try:
            p = privacy.save(ctx, redact=rules, retention_days=retention_days)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        audit.record(conn, "privacy.set", customer_id,
                     {"redact": list(p["redact"]), "retention_days": p["retention_days"]})
    finally:
        conn.close()
    typer.secho(f"\n{customer_id}: redact={', '.join(p['redact']) or 'nothing'}, "
                f"retention={p['retention_days']}d", fg=typer.colors.GREEN)


@privacy_app.command("enforce")
def privacy_enforce(
    customer_id: str | None = typer.Argument(None, help="One customer (default: all)."),
):
    """Delete captured traffic older than each customer's retention window.
    Meant to run daily (cron / Task Scheduler)."""
    from ftplatform import privacy

    conn = connect()
    try:
        ids = [customer_id] if customer_id else [c.id for c in store.list_all(conn)]
        for cid in ids:
            removed = privacy.enforce_retention(_ctx_or_exit(conn, cid))
            total = sum(removed.values())
            if total:
                audit.record(conn, "privacy.retention", cid, removed)
            typer.echo(f"  {cid}: removed {total} captured row(s) past retention")
    finally:
        conn.close()


@audit_app.command("list")
def audit_list(
    customer_id: str | None = typer.Argument(None, help="Filter to one customer."),
    limit: int = typer.Option(50, help="Most recent N entries."),
):
    """The audit trail, oldest first -- ids and counts only, never customer text."""
    import json as json_mod

    conn = connect()
    try:
        rows = audit.entries(conn, customer_id, limit=limit)
    finally:
        conn.close()
    if not rows:
        typer.echo("no audit entries")
        return
    for r in rows:
        typer.echo(f"  {r['at']}  {r['actor']:<12} {r['action']:<20} {r['customer_id'] or '-':<14} "
                   f"{json_mod.dumps(r['detail'], ensure_ascii=False)}")


@kaggle_app.command("watch")
def kaggle_watch(
    job_id: str = typer.Argument(..., help="The job id passed to `kaggle start`."),
    customer_id: str = typer.Argument(..., help="Customer id the job belongs to."),
    owner: str = typer.Option(..., help="Your Kaggle username."),
    poll_interval_s: float = typer.Option(120.0),
):
    """`kaggle poll`, repeated until the run finishes and its result is merged."""
    import time

    from ftplatform.remote.run_on_kaggle import default_work_dir, kernel_id_for, poll_and_finish_job

    conn = connect()
    try:
        while (job := poll_and_finish_job(conn, customer_id, job_id, kernel_id_for(job_id, owner),
                                          default_work_dir(job_id))) is None:
            typer.echo(f"  still running -- next check in {poll_interval_s:.0f}s")
            time.sleep(poll_interval_s)
    finally:
        conn.close()
    color = typer.colors.GREEN if job["status"] == "done" else typer.colors.RED
    typer.secho(f"\njob {job['id']!r} ({job['kind']}) -> {job['status']}", fg=color)
    if job["status"] == "failed":
        typer.echo(f"  {job['result']['error']}")


# --- agent -------------------------------------------------------------------

@agent_app.command("schema")
def agent_schema(
    tools_json: Path = typer.Argument(..., exists=True,
                                      help="A tool catalog (see ftplatform/agent/tools.py)."),
    out: Path = typer.Option(Path("agent_schema.json"), "--out", "-o"),
    intents: str | None = typer.Option(None, help="Comma-separated intents to classify, if any."),
):
    """Write the step schema for a catalog, for `customer add --workload custom --schema`."""
    import json as json_mod

    from ftplatform.agent import tools as tools_mod

    catalog = json_mod.loads(tools_json.read_text(encoding="utf-8"))
    try:
        schema = tools_mod.step_schema(
            catalog, intents=[i.strip() for i in intents.split(",")] if intents else None)
    except tools_mod.CatalogError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    out.write_text(json_mod.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.secho(f"wrote {out} ({len(catalog['tools'])} tools)", fg=typer.colors.GREEN)


@agent_app.command("build-abcd")
def agent_build_abcd(
    src_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                   help="Folder with ABCD's abcd_v1.1.json.gz, kb.json, ontology.json."),
    out_dir: Path = typer.Argument(...),
    max_rows: int | None = typer.Option(None, help="Stop after this many conversations."),
):
    """Turn the public ABCD dataset into an agent corpus, catalog and schema (a demo workload)."""
    from ftplatform.agent import abcd

    r = abcd.build(src_dir, out_dir, max_rows=max_rows)
    typer.secho(f"\n{r['rows']} steps {r['steps']}, {r['tools']} tools, {r['intents']} intents "
                f"-> {r['out_dir']}\n", fg=typer.colors.GREEN)


@agent_tools_app.command("install")
def agent_tools_install(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    tools_json: Path = typer.Argument(..., exists=True),
):
    """Install (or replace) this customer's tool catalog."""
    import json as json_mod

    from ftplatform.agent import tools as tools_mod

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        try:
            catalog = tools_mod.save(ctx, json_mod.loads(tools_json.read_text(encoding="utf-8")))
        except tools_mod.CatalogError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        audit.record(conn, "agent.tools.install", customer_id,
                     {"tools": len(catalog["tools"]), "mode": catalog["mode"]})
    finally:
        conn.close()
    typer.secho(f"{customer_id}: {len(catalog['tools'])} tools installed, mode={catalog['mode']}",
                fg=typer.colors.GREEN)


@agent_tools_app.command("show")
def agent_tools_show(customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'.")):
    """The catalog: every tool, its permission and handler."""
    from ftplatform.agent import tools as tools_mod

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
    finally:
        conn.close()
    catalog = tools_mod.load(ctx)
    typer.echo(f"\n{customer_id}: mode={catalog['mode']}\n")
    for t in catalog["tools"]:
        typer.echo(f"  {t['name']:<24} {t['permission']:<10} {t['handler']['type']:<17} {t['args']}")
    typer.echo("")


@agent_tools_app.command("set")
def agent_tools_set(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    tool_name: str | None = typer.Argument(None, help="Tool to change (omit with --mode)."),
    permission: str | None = typer.Option(None, help="auto | approval | forbidden"),
    mode: str | None = typer.Option(None, help="dry_run | live -- for the whole catalog."),
):
    """Change one tool's permission, or switch the catalog between dry_run and live."""
    from ftplatform.agent import tools as tools_mod

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        catalog = tools_mod.load(ctx)
        if mode is not None:
            catalog["mode"] = mode
        if permission is not None:
            t = tools_mod.tool(catalog, tool_name or "")
            if t is None:
                typer.secho(f"no tool {tool_name!r}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            t["permission"] = permission
        try:
            catalog = tools_mod.save(ctx, catalog)
        except tools_mod.CatalogError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        audit.record(conn, "agent.tools.set", customer_id,
                     {"tool": tool_name, "permission": permission, "mode": mode})
    finally:
        conn.close()
    typer.secho(f"{customer_id}: saved (mode={catalog['mode']})", fg=typer.colors.GREEN)


@agent_kb_app.command("add")
def agent_knowledge_add(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    paths: list[Path] = typer.Argument(..., exists=True, help=".md, .txt or .json documents."),
):
    """Add documents the agent's knowledge_search tool can look things up in."""
    from ftplatform.agent import knowledge

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        try:
            added = knowledge.add(ctx, paths)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        audit.record(conn, "agent.knowledge.add", customer_id, {"documents": len(added)})
    finally:
        conn.close()
    typer.secho(f"{customer_id}: added {', '.join(added)}", fg=typer.colors.GREEN)


@agent_kb_app.command("search")
def agent_knowledge_search(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    query: str = typer.Argument(...),
    k: int = typer.Option(3),
):
    """What knowledge_search would return for a query."""
    from ftplatform.agent import knowledge

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
    finally:
        conn.close()
    hits = knowledge.search(ctx, query, k=k)
    if not hits:
        typer.echo("no match")
    for h in hits:
        typer.echo(f"\n[{h['source']}  score {h['score']}]\n{h['text'][:500]}")


@agent_app.command("chat")
def agent_chat(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    model_url: str = typer.Option("http://localhost:8000", help="The customer's model server."),
    api_key: str | None = typer.Option(None, envvar="FTPLATFORM_API_KEY"),
):
    """Talk to the agent in the terminal, as a customer would. Ctrl-C to stop."""
    from ftplatform.agent import session as session_mod
    from ftplatform.agent import tools as tools_mod

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        catalog = tools_mod.load(ctx)
        model_fn = session_mod.http_model(model_url, api_key)
        s = session_mod.new_session(ctx)
        typer.echo(f"\nsession {s['id']} · mode={catalog['mode']} · Ctrl-C to stop\n")
        while True:
            try:
                text = typer.prompt("you")
            except (KeyboardInterrupt, EOFError, typer.Abort):
                break
            out = session_mod.respond(conn, ctx, s, text, model_fn, catalog)
            for a in out["actions"]:
                typer.secho(f"  [{a['status']}] {a['tool']}({', '.join(a['args'])}) -> {a['result']}",
                            fg=typer.colors.YELLOW)
            for r in out["replies"]:
                typer.secho(f"agent: {r}", fg=typer.colors.CYAN)
            if out["stopped"] not in ("wait", "max_steps"):
                typer.secho(f"  (stopped: {out['stopped']})", fg=typer.colors.RED)
    finally:
        conn.close()


@agent_app.command("serve")
def agent_serve(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    model_url: str = typer.Option("http://localhost:8000", help="The customer's model server."),
    model_api_key: str | None = typer.Option(None, envvar="FTPLATFORM_MODEL_API_KEY"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8100),
    require_auth: bool = typer.Option(True, "--require-auth/--no-require-auth"),
):
    """The agent's HTTP API (sessions, messages, approvals) in front of the model server."""
    import uvicorn

    from ftplatform.agent import server
    from ftplatform.agent import session as session_mod
    from ftplatform.agent import tools as tools_mod
    from ftplatform.auth import keys as keys_mod

    conn = connect(shared=True)            # held while serving, used per request
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        tools_mod.load(ctx)                                  # fail now, not on the first message
        resolver = None
        if require_auth:
            if not keys_mod.list_keys(conn, customer_id):
                typer.secho(f"{customer_id!r} has no API key yet -- `ftplatform api-key create "
                            f"{customer_id}`, or --no-require-auth for local dev.",
                            fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            resolver = keys_mod.build_resolver(conn, customer_ids={customer_id})
        app_ = server.create_app(conn, ctx, session_mod.http_model(model_url, model_api_key),
                                 api_key_resolver=resolver)
        uvicorn.run(app_, host=host, port=port)
    finally:
        conn.close()


@agent_app.command("actions")
def agent_actions(
    customer_id: str = typer.Argument(..., help="Customer id, e.g. 'acme'."),
    status: str | None = typer.Option(None, help="e.g. pending_approval, executed, dry_run"),
    limit: int = typer.Option(30),
):
    """What the agent did, or asked to do."""
    from ftplatform.agent import executor

    conn = connect()
    try:
        _ctx_or_exit(conn, customer_id)
        rows = executor.list_actions(conn, customer_id, status=status, limit=limit)
    finally:
        conn.close()
    if not rows:
        typer.echo("no actions")
    for a in rows:
        typer.echo(f"  {a['id']}  {a['created_at']}  {a['status']:<17} "
                   f"{a['tool']}({', '.join(a['args'])})")


def _agent_decide(customer_id: str, action_id: str, approve: bool) -> None:
    from ftplatform.agent import executor
    from ftplatform.agent import session as session_mod
    from ftplatform.agent import tools as tools_mod

    conn = connect()
    try:
        ctx = _ctx_or_exit(conn, customer_id)
        try:
            row = executor.decide(conn, ctx, tools_mod.load(ctx), action_id, approve)
        except executor.NotPendingError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        session_mod.apply_decision(ctx, row)
    finally:
        conn.close()
    typer.secho(f"{action_id}: {row['status']} -> {row['result']}", fg=typer.colors.GREEN)


@agent_app.command("approve")
def agent_approve(customer_id: str = typer.Argument(...), action_id: str = typer.Argument(...)):
    """Approve a pending action; it runs now (or is recorded, in dry_run)."""
    _agent_decide(customer_id, action_id, True)


@agent_app.command("reject")
def agent_reject(customer_id: str = typer.Argument(...), action_id: str = typer.Argument(...)):
    """Reject a pending action."""
    _agent_decide(customer_id, action_id, False)


@agent_packs_app.command("list")
def agent_packs_list():
    """The ready-made process packs."""
    from ftplatform.agent import packs

    for p in packs.available():
        typer.echo(f"\n  {p['name']:<22} {p['title']}")
        typer.echo(f"  {'':<22} {p['description']}")
        typer.echo(f"  {'':<22} {len(p['intents'])} intents, {len(p['tools'])} tools")
    typer.echo("")


@agent_packs_app.command("export")
def agent_packs_export(
    names: list[str] = typer.Argument(..., help="One or more pack names."),
    out_dir: Path = typer.Option(Path("."), "--out-dir", "-o"),
    endpoint: str | None = typer.Option(
        None, help="Base URL of the customer's tool endpoints; each tool posts to <endpoint>/<tool>."),
    secret_env: str | None = typer.Option(None, help="Env var holding their endpoint's bearer token."),
):
    """Write tools.json + step_schema.json for these packs -- the two files a new agent customer needs."""
    import json as json_mod

    from ftplatform.agent import packs
    from ftplatform.agent import tools as tools_mod

    try:
        catalog, intents = packs.combine(names, endpoint=endpoint, secret_env=secret_env)
    except (KeyError, ValueError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tools.json").write_text(json_mod.dumps(catalog, indent=2), encoding="utf-8")
    (out_dir / "step_schema.json").write_text(
        json_mod.dumps(tools_mod.step_schema(catalog, intents=intents), indent=2), encoding="utf-8")
    typer.secho(f"\n{len(catalog['tools'])} tools, {len(intents)} intents -> {out_dir}/tools.json, "
                f"{out_dir}/step_schema.json (mode dry_run)\n", fg=typer.colors.GREEN)
    typer.echo(f"  ftplatform customer add <id> --name ... --workload custom "
               f"--schema {out_dir / 'step_schema.json'}")
    typer.echo(f"  ftplatform agent tools install <id> {out_dir / 'tools.json'}\n")


@agent_packs_app.command("score")
def agent_packs_score(
    schema: Path = typer.Argument(..., exists=True, help="The step schema the run used."),
    eval_jsonl: Path = typer.Argument(..., exists=True),
    raw_jsonl: Path = typer.Argument(..., exists=True, help="raw_<system>.jsonl from the run's reports."),
):
    """Per-pack accuracy of one benchmarked system."""
    from ftplatform.agent import packs
    from ftspec.core.registry import read_custom_schema

    contract, _ = read_custom_schema(schema)
    rows = packs.score_files(eval_jsonl, raw_jsonl, contract)
    typer.echo(f"\n  {'pack':<22} {'steps':>5} {'next step':>10} {'actions':>8} {'right action':>13}")
    for name, r in rows.items():
        act = f"{r['action_pct']}%" if r["action_pct"] is not None else "-"
        typer.echo(f"  {name:<22} {r['steps']:>5} {r['next_step_pct']:>9}% {r['actions']:>8} {act:>13}")
    typer.echo("")


if __name__ == "__main__":
    app()
