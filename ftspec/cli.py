"""
`ftspec` — the pipeline entry point.

    ftspec profiles                    list available verticals
    ftspec prepare  --profile X        build the corpus
    ftspec audit    --profile X        groundedness + compliance gate
    ftspec validate --profile X        contract + token-length gate (pre-GPU)
    ftspec train    --profile X        QLoRA fine-tune
    ftspec evaluate --profile X        benchmark matrix
    ftspec tco                         break-even analysis
    ftspec report   --profile X        splice measured results into the README
    ftspec prompt-audit <log.jsonl>    measure the prompt tax on your own logs
    ftspec infer    --profile X        one document, constrained decoding
    ftspec serve    --profile X        OpenAI-compatible endpoint

Every command takes `--profile`, which selects a shipped vertical
(`saas_support`, `fintech_disputes`, `healthcare_clinical`) or, with
`--profile custom --schema mine.json`, a customer's own contract.

Stages that produce artefacts write a run manifest to `outputs/<profile>/manifests/`
recording git commit, config fingerprint, package versions, GPU and metrics.

Gating is deliberate: `train` refuses to start unless `validate` passes, because
the failure it prevents — silently truncated training samples — is invisible
until eval and expensive to rediscover.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from ftspec.config import REPO_ROOT, Config, load_config
from ftspec.core.registry import available, load_profile
from ftspec.run import get_logger, run_manifest, setup_logging

app = typer.Typer(
    name="ftspec",
    help="Enterprise SLM specialization engine: modular domain adapters for "
          "regulated and high-volume extraction workflows.",
    add_completion=False,
    no_args_is_help=True,
)
log = get_logger("ftspec.cli")

CONFIG_DIR = REPO_ROOT / "configs"

ProfileOpt = typer.Option(None, "--profile", "-p",
                           help="Vertical to use, or 'custom' with --schema.")
SchemaOpt = typer.Option(None, "--schema",
                          help="JSON Schema file; only valid with --profile custom.")
ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config file.")


def _load_config(config: Path | None, profile_name: str | None) -> Config:
    path = config
    if path is None:
        # A profile-specific config is used automatically when one exists.
        candidate = CONFIG_DIR / f"{profile_name or 'default'}.yaml"
        path = candidate if candidate.exists() else (
            CONFIG_DIR / "default.yaml" if (CONFIG_DIR / "default.yaml").exists() else None)
    try:
        cfg = load_config(path)
    except Exception as e:
        typer.secho(f"Invalid configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    if profile_name:
        cfg.profile = profile_name
    log.info("config %s (fingerprint %s)", path or "<defaults>", cfg.fingerprint())
    return cfg


def _resolve(config: Path | None, profile_name: str | None,
              schema: Path | None) -> tuple:
    cfg = _load_config(config, profile_name)
    try:
        profile = load_profile(cfg.profile, schema_path=schema)
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    return cfg, profile, cfg.data_dir(profile.name), cfg.outputs_dir(profile.name)


def _manifest_path(outputs: Path, stage: str) -> Path:
    return outputs / "manifests" / f"{stage}.json"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging.")):
    setup_logging(verbose)


# --- profiles ----------------------------------------------------------------

@app.command()
def profiles():
    """List available verticals and their regulatory envelopes."""
    typer.echo("")
    typer.echo(f"  {'profile':<22} {'regime':<10} {'ext. API':<10} {'fields':>6}  title")
    typer.echo("  " + "-" * 88)
    for name in available():
        try:
            p = load_profile(name)
        except Exception as e:
            typer.echo(f"  {name:<22} <failed to load: {e}>")
            continue
        s = p.summary()
        egress = "allowed" if s["allows_external_api"] else "PROHIBITED"
        typer.echo(f"  {s['name']:<22} {s['regime']:<10} {egress:<10} "
                    f"{s['scored_fields']:>6}  {s['title']}")
    typer.echo(f"  {'custom':<22} {'none':<10} {'allowed':<10} {'—':>6}  "
                "Your own JSON Schema (--schema mine.json)")
    typer.echo("")


# --- prepare -----------------------------------------------------------------

@app.command()
def prepare(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    n_train: int = typer.Option(200, help="Training documents."),
    n_val: int = typer.Option(40, help="Validation documents."),
    n_eval: int = typer.Option(150, help="Held-out benchmark documents."),
    seed: int = typer.Option(42),
    eval_seed: int = typer.Option(1337, help="Separate stream for the held-out set."),
    from_jsonl: Path | None = typer.Option(
        None, help="Use your own corpus instead of generating one."),
):
    """Build the corpus (train / val / eval) for a profile."""
    from ftspec.data import build

    cfg, prof, data_dir, outputs = _resolve(config, profile, schema)
    with run_manifest("prepare", _manifest_path(outputs, "prepare"), cfg.fingerprint(),
                       params={"profile": prof.name, "n_train": n_train, "n_val": n_val,
                                "n_eval": n_eval, "seed": seed, "eval_seed": eval_seed,
                                "from_jsonl": str(from_jsonl) if from_jsonl else None}) as manifest:
        manifest.metrics = build.run(prof, data_dir, n_train, n_val, n_eval,
                                      seed, eval_seed, from_jsonl)


# --- audit -------------------------------------------------------------------

@app.command()
def audit(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
):
    """Verify every label is derivable from its source, and check compliance rules."""
    from ftspec.data import audit as audit_mod

    cfg, prof, data_dir, outputs = _resolve(config, profile, schema)
    with run_manifest("audit", _manifest_path(outputs, "audit"), cfg.fingerprint()) as manifest:
        passed, report, metrics = audit_mod.run(prof, data_dir)
        manifest.metrics = metrics
    typer.echo(report)
    if not passed:
        raise typer.Exit(code=1)


# --- validate ----------------------------------------------------------------

@app.command()
def validate(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    approx: bool = typer.Option(False, help="Estimate token counts instead of loading a tokenizer."),
):
    """Pre-training gate: contract, token lengths, duplicates, leakage."""
    from ftspec.data import validate as validate_mod

    cfg, prof, data_dir, outputs = _resolve(config, profile, schema)
    with run_manifest("validate", _manifest_path(outputs, "validate"),
                       cfg.fingerprint()) as manifest:
        passed, report, metrics = validate_mod.run(cfg, prof, data_dir, approx)
        manifest.metrics = metrics
    typer.echo(report)
    if not passed:
        raise typer.Exit(code=1)


# --- train -------------------------------------------------------------------

@app.command()
def train(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    max_steps: int = typer.Option(-1, help="Override epochs with a fixed step count."),
    resume: bool = typer.Option(False, help="Resume from the latest checkpoint."),
    skip_validation: bool = typer.Option(
        False, help="Train without the preflight gate. Recorded in the run manifest."),
):
    """Fine-tune the base model with QLoRA for this profile."""
    from ftspec.data import validate as validate_mod
    from ftspec.training import train as train_mod

    cfg, prof, data_dir, outputs = _resolve(config, profile, schema)

    if skip_validation:
        log.warning("preflight validation SKIPPED by request — truncated samples "
                     "will not be caught")
    else:
        passed, report, _ = validate_mod.run(cfg, prof, data_dir)
        if not passed:
            typer.echo(report)
            typer.secho("Refusing to train on a corpus that failed validation. "
                         "Fix it, or re-run with --skip-validation.",
                         fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        log.info("preflight validation passed")

    with run_manifest("train", _manifest_path(outputs, "train"), cfg.fingerprint(),
                       params={"profile": prof.name, "max_steps": max_steps, "resume": resume,
                                "skip_validation": skip_validation,
                                "base_model": cfg.model.base_model}) as manifest:
        manifest.metrics = train_mod.run(cfg, prof.name, max_steps=max_steps, resume=resume)


# --- evaluate ----------------------------------------------------------------

@app.command(name="evaluate")
def evaluate(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    systems: str = typer.Option("finetuned", help="Comma-separated system names."),
    base_model: str = typer.Option("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
    finetuned_model: str | None = typer.Option(None, help="Adapter or merged model path."),
    limit: int | None = typer.Option(None, help="Evaluate only the first N documents."),
    gpu_cost_per_hour: float = typer.Option(0.35),
    max_new_tokens: int = typer.Option(512),
    acknowledge_egress: bool = typer.Option(
        False, "--acknowledge-egress",
        help="Permit hosted-API baselines for a profile that forbids external egress."),
    self_test: bool = typer.Option(False, help="Validate the metrics pipeline; runs no model."),
):
    """Benchmark matrix: adherence, accuracy, p50/p99 latency, cost per 100k calls."""
    from ftspec.evaluation import benchmark

    cfg, prof, data_dir, outputs = _resolve(config, profile, schema)
    eval_file = data_dir / "eval.jsonl"
    reports_dir = cfg.reports_dir(prof.name)

    if self_test:
        records = benchmark.load_eval_set(eval_file, limit)
        with run_manifest("selftest", _manifest_path(outputs, "selftest"),
                           cfg.fingerprint()) as manifest:
            manifest.metrics = benchmark.self_test(records, prof)
        return

    with run_manifest("evaluate", _manifest_path(outputs, "evaluate"), cfg.fingerprint(),
                       params={"profile": prof.name, "systems": systems, "limit": limit,
                                "acknowledge_egress": acknowledge_egress}) as manifest:
        try:
            manifest.metrics = benchmark.run(
                profile=prof, eval_file=eval_file, results_dir=reports_dir,
                systems=systems, base_model=base_model,
                finetuned_model=finetuned_model or str(cfg.adapter_dir(prof.name)),
                limit=limit, gpu_cost_per_hour=gpu_cost_per_hour,
                max_new_tokens=max_new_tokens, acknowledge_egress=acknowledge_egress,
            )
        except (RuntimeError, ValueError) as e:
            # A compliance refusal or a missing credential is an expected
            # operator-facing outcome, not a crash. The manifest still records
            # what was attempted and why it stopped.
            manifest.metrics = {"error": str(e), "profile": prof.name}
            typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e


# --- combine-eval --------------------------------------------------------------

@app.command(name="combine-eval")
def combine_eval(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    systems: str = typer.Option(
        "finetuned,base-constrained,base-rubric",
        help="Comma-separated system names to fold together."),
    gpu_cost_per_hour: float = typer.Option(0.35),
):
    """Combine systems that were each run with a separate `evaluate` call.

    Running every system in one `evaluate` process is fine when it fits: a
    fine-tuned adapter and two base-model baselines loaded back to back can
    exceed a small GPU's VRAM even though each one fits alone, because a
    prior model's memory is not always fully released before the next loads.
    Running each system as its own `evaluate` invocation sidesteps that
    entirely -- a fresh process gets a clean CUDA context -- at the cost of
    each run only knowing about the one system it ran.

    This reads every already-written raw_<name>.jsonl back off disk and
    produces the same combined report `evaluate` would have, including the
    full statistical treatment: no model loads, so it cannot itself run out
    of memory.
    """
    from ftspec.evaluation import benchmark

    cfg, prof, _, outputs = _resolve(config, profile, schema)
    eval_file = cfg.data_dir(prof.name) / "eval.jsonl"
    reports_dir = cfg.reports_dir(prof.name)

    with run_manifest("evaluate", _manifest_path(outputs, "evaluate"), cfg.fingerprint(),
                       params={"profile": prof.name, "systems": systems,
                                "combined_from_disk": True}) as manifest:
        try:
            manifest.metrics = benchmark.combine(
                profile=prof, eval_file=eval_file, results_dir=reports_dir,
                systems=systems, gpu_cost_per_hour=gpu_cost_per_hour)
        except (FileNotFoundError, ValueError) as e:
            manifest.metrics = {"error": str(e), "profile": prof.name}
            typer.secho(f"\n{e}\n", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e


# --- tco ---------------------------------------------------------------------

@app.command()
def tco(
    gpu_name: str = typer.Option("A10G"),
    gpu_hourly: float = typer.Option(0.75),
    throughput_rps: float = typer.Option(8.0, help="Sustained req/s/GPU. Measure this."),
    peak_rps: float = typer.Option(5.0),
    training_cost: float = typer.Option(15.0),
    amortise_months: int = typer.Option(12),
    prompt_tokens_api: int = typer.Option(
        1437, help="Prompt tokens a hosted baseline pays per call (schema + policy + document)."),
    output_tokens: int = typer.Option(162, help="Mean output tokens per call."),
):
    """Break-even analysis: at what volume does self-hosting actually win?"""
    from ftspec.evaluation import tco as tco_mod

    with run_manifest("tco", REPO_ROOT / "outputs" / "manifests" / "tco.json") as manifest:
        report, metrics = tco_mod.run(
            gpu_name, gpu_hourly, throughput_rps, peak_rps,
            training_cost, amortise_months, prompt_tokens_api, output_tokens)
        manifest.metrics = metrics
    typer.echo(report)


# --- prompt-audit ------------------------------------------------------------

@app.command(name="prompt-audit")
def prompt_audit(
    file: Path = typer.Argument(..., help="JSONL of your prompts. Never leaves this machine."),
    model: str = typer.Option("gpt-4o", help="Model whose pricing to apply."),
    calls_per_month: int = typer.Option(100_000, help="Your call volume."),
    include_sample: bool = typer.Option(
        False, help="Include the first 400 chars of the static prefix in the report."),
    out: Path | None = typer.Option(None, help="Write the markdown report to a file."),
):
    """Measure how much of your prompt bill re-sends content that never changes.

    Runs entirely locally. Accepts {"messages": [...]}, {"system", "user"} or
    {"prompt"} shapes, so most prompt logs work without reshaping.
    """
    from ftspec import prompt_audit as audit_mod

    try:
        report, _ = audit_mod.run(file, model, calls_per_month, include_sample, out)
    except (ValueError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(report)


# --- report ------------------------------------------------------------------

@app.command()
def report(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    readme: Path = typer.Option(REPO_ROOT / "README.md", help="File to splice results into."),
    gpu_cost_per_hour: float = typer.Option(0.35),
    latency_file: Path | None = typer.Option(
        None, help="latency.json from the Colab notebook; enables the TTFT table."),
    update_readme: bool = typer.Option(
        False, "--update-readme", help="Write the block into README between the markers."),
):
    """Turn the last evaluate run into the README's measured-results block."""
    from ftspec import reporting

    cfg, prof, _, _ = _resolve(config, profile, schema)
    try:
        block, changed = reporting.run(cfg, prof.name, readme, gpu_cost_per_hour,
                                        latency_file, write=update_readme)
    except (FileNotFoundError, ValueError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(block)
    if update_readme:
        verb = "updated" if changed else "no change to"
        typer.secho(f"\n{verb} {readme}",
                     fg=typer.colors.GREEN if changed else None)
    else:
        typer.echo("\n(preview only — pass --update-readme to write it in)")


# --- infer -------------------------------------------------------------------

@app.command()
def infer(
    model: str = typer.Option(..., help="LoRA adapter or merged model directory."),
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    text: str | None = typer.Option(None, help="Raw source document."),
    file: Path | None = typer.Option(None, help="Read the source document from a file."),
    strategy: str = typer.Option("grammar", help="'grammar' (guaranteed) or 'retry' (best-effort)."),
):
    """Extract one document locally, with constrained decoding."""
    from ftspec.inference.constrained import build_extractor
    from ftspec.inference.demo import example_document, prompt_overhead

    cfg, prof, data_dir, _ = _resolve(config, profile, schema)

    if file:
        source = file.read_text(encoding="utf-8")
    elif text:
        source = text
    else:
        source = example_document(prof, data_dir)
        log.info("no --text/--file given; using an example from the eval split")

    extractor = build_extractor(model, prof, strategy=strategy,
                                 max_seq_length=cfg.model.max_seq_length,
                                 load_in_4bit=cfg.model.load_in_4bit)
    result = extractor.extract(source)

    typer.echo("\n--- INPUT ---")
    typer.echo(source)
    typer.echo("\n--- OUTPUT ---")
    if result.ok:
        typer.echo(json.dumps(result.record, ensure_ascii=False, indent=2))
    else:
        typer.secho(f"FAILED after {result.attempts} attempt(s): {result.error}",
                     fg=typer.colors.RED)
        typer.echo(result.raw)

    typer.echo(f"\nprofile={prof.name}  strategy={strategy}  valid={result.ok}  "
                f"attempts={result.attempts}  latency={result.latency_ms:.0f}ms")
    typer.echo(prompt_overhead(extractor, prof))
    if not result.ok:
        raise typer.Exit(code=1)


# --- serve -------------------------------------------------------------------

@app.command()
def serve(
    model: str = typer.Option(..., help="Merged model directory, or a base model with --lora."),
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    lora: str | None = typer.Option(None, help="LoRA adapter to serve on top of --model."),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    max_model_len: int = typer.Option(4096),
    gpu_memory_utilization: float = typer.Option(0.90),
    max_lora_rank: int = typer.Option(16),
    respect_client_system_prompt: bool = typer.Option(
        False, help="Honour client system prompts instead of injecting the trained one."),
):
    """Serve an OpenAI-compatible endpoint with the profile's schema enforced server-side."""
    from ftspec.serving.serve import serve as serve_fn

    _, prof, _, _ = _resolve(config, profile, schema)
    serve_fn(model=model, profile=prof, lora=lora, host=host, port=port,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              max_lora_rank=max_lora_rank,
              respect_client_system_prompt=respect_client_system_prompt)


# --- schema ------------------------------------------------------------------

@app.command(name="schema")
def schema_cmd(
    profile: str | None = ProfileOpt,
    schema: Path | None = SchemaOpt,
    config: Path | None = ConfigOpt,
    out: Path | None = typer.Option(None, help="Write the JSON Schema to a file."),
    show_fields: bool = typer.Option(False, help="Show the derived scoring plan instead."),
):
    """Print the profile's contract, or the metrics derived from it."""
    _, prof, _, _ = _resolve(config, profile, schema)

    if show_fields:
        plan = prof.scoring_plan()
        typer.echo(f"\n  {prof.name}: {len(plan.fields)} scored fields\n")
        typer.echo(f"  {'field':<44} {'kind':<10} role")
        typer.echo("  " + "-" * 74)
        for spec in plan.fields:
            role = "HEADLINE" if spec.headline else ("policy input" if spec.rubric_input else "")
            typer.echo(f"  {spec.path:<44} {spec.kind:<10} {role}")
        typer.echo("")
        return

    text = json.dumps(prof.contract.json_schema(), indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
