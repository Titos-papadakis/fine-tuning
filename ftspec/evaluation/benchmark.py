"""
Enterprise benchmarking matrix.

The comparison set is deliberately hostile to our own thesis. A result that only
beats "GPT-4o with no schema in the prompt" proves nothing, because nobody
deploys that. These are the options a competent team actually weighs:

  base-schema          8B base, schema in prompt                    (no training)
  base-rubric          8B base, schema + business policy            (no training)
  base-constrained     8B base + grammar-constrained decoding       (100% valid, free)
  gpt4o-schema         GPT-4o, schema in prompt, structured outputs
  gpt4o-rubric         GPT-4o, schema + policy                      (strongest baseline)
  gpt4o-mini-rubric    GPT-4o-mini, schema + policy                 (cheapest credible baseline)
  finetuned            the specialized 8B, ~20-token prompt

Reported per system: schema adherence, record and per-field accuracy with
confidence intervals, p50/p99 latency, and cost per 100k calls.

Compliance gate
---------------
For a profile declaring `allows_external_api=False` (PCI-DSS, HIPAA), the hosted
baselines are refused unless the operator passes `--acknowledge-egress`, and
the refusal is recorded in the report. For those verticals the frontier-model
columns are not a cost comparison that happens to lose — they are a deployment
option that is not lawfully available, which is a stronger argument than any
number in the table.

If the fine-tuned model does not win, this script says so. A benchmark you
cannot lose is a benchmark nobody believes.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from ftspec.core.profile import Profile
from ftspec.evaluation import metrics as M
from ftspec.run import get_logger

log = get_logger("ftspec.benchmark")

# Published API pricing, USD per 1M tokens. Printed in every report so a reader
# can check them against current rate cards rather than trusting the script.
API_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

CALLS_PER_BLOCK = 100_000


@dataclass
class SystemSpec:
    name: str
    backend: str            # "local" | "openai"
    model: str
    prompt_variant: str     # "short" | "schema" | "schema+rubric"
    constrained: bool = False

    @property
    def is_hosted(self) -> bool:
        return self.backend == "openai"


def system_catalogue() -> dict:
    return {
        "base-schema": SystemSpec("base-schema", "local", "", "schema"),
        "base-rubric": SystemSpec("base-rubric", "local", "", "schema+rubric"),
        "base-constrained": SystemSpec("base-constrained", "local", "", "schema+rubric",
                                        constrained=True),
        "finetuned": SystemSpec("finetuned", "local", "", "short"),
        "gpt4o-schema": SystemSpec("gpt4o-schema", "openai", "gpt-4o", "schema"),
        "gpt4o-rubric": SystemSpec("gpt4o-rubric", "openai", "gpt-4o", "schema+rubric"),
        "gpt4o-mini-rubric": SystemSpec("gpt4o-mini-rubric", "openai", "gpt-4o-mini",
                                         "schema+rubric"),
    }


@dataclass
class RunResult:
    spec: SystemSpec
    raw_outputs: list = field(default_factory=list)
    valid_flags: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)
    prompt_tokens: list = field(default_factory=list)
    output_tokens: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    gold_headline: list = field(default_factory=list)
    agg: M.Aggregate | None = None

    @property
    def adherence_pct(self) -> float:
        return 100.0 * statistics.mean(self.valid_flags) if self.valid_flags else 0.0

    @property
    def p50(self) -> float:
        return M.percentile(self.latencies_ms, 0.50)

    @property
    def p99(self) -> float:
        return M.percentile(self.latencies_ms, 0.99)

    @property
    def mean_prompt_tokens(self) -> float:
        return statistics.mean(self.prompt_tokens) if self.prompt_tokens else 0.0

    @property
    def mean_output_tokens(self) -> float:
        return statistics.mean(self.output_tokens) if self.output_tokens else 0.0


def load_eval_set(path: Path, limit: int | None = None) -> list:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def parse_prediction(raw: str, profile: Profile) -> tuple:
    """(record_or_None, satisfies_contract) under the profile's own contract.

    Adherence here means exactly what it means in production, because it is the
    same validator the server enforces.
    """
    record, _ = profile.contract.validate(raw)
    if record is not None:
        return record, True

    # Not contract-valid but possibly parseable; kept only for the confusion
    # matrix. Scoring call sites pass None, because a record that violates the
    # contract cannot be ingested and none of its fields are usable.
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines()
                          if not ln.strip().startswith("```")).strip()
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        return None, False


# --- Backends ----------------------------------------------------------------

def run_local(spec: SystemSpec, records: list, profile: Profile,
               max_new_tokens: int) -> RunResult:
    import torch
    from unsloth import FastLanguageModel

    result = RunResult(spec=spec)
    plan = profile.scoring_plan()
    system_prompt = profile.prompts.variants()[spec.prompt_variant]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.model, max_seq_length=4096, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    generator = None
    if spec.constrained:
        from ftspec.inference.constrained import build_grammar_generator
        generator = build_grammar_generator(model, tokenizer, profile.contract)

    for row in records:
        source = row["messages"][1]["content"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": source}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        if generator is not None:
            text = generator(prompt)
            n_out = len(tokenizer(text)["input_ids"])
        else:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            n_out = len(generated)
        elapsed = (time.perf_counter() - start) * 1000

        pred, valid = parse_prediction(text, profile)
        gold = json.loads(row["messages"][2]["content"])
        result.raw_outputs.append(text)
        result.latencies_ms.append(elapsed)
        result.prompt_tokens.append(inputs["input_ids"].shape[1])
        result.output_tokens.append(n_out)
        result.valid_flags.append(1.0 if valid else 0.0)
        result.scores.append(M.score_record(pred if valid else None, gold, plan))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_openai(spec: SystemSpec, records: list, profile: Profile) -> RunResult:
    from openai import OpenAI

    result = RunResult(spec=spec)
    plan = profile.scoring_plan()
    client = OpenAI()
    system_prompt = profile.prompts.variants()[spec.prompt_variant]

    for row in records:
        source = row["messages"][1]["content"]
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=spec.model,
            messages=[{"role": "system", "content": system_prompt},
                       {"role": "user", "content": source}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        elapsed = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""

        pred, valid = parse_prediction(text, profile)
        gold = json.loads(row["messages"][2]["content"])
        result.raw_outputs.append(text)
        result.latencies_ms.append(elapsed)
        result.prompt_tokens.append(response.usage.prompt_tokens)
        result.output_tokens.append(response.usage.completion_tokens)
        result.valid_flags.append(1.0 if valid else 0.0)
        result.scores.append(M.score_record(pred if valid else None, gold, plan))

    return result


# --- Cost --------------------------------------------------------------------

def cost_per_100k(result: RunResult, gpu_cost_per_hour: float) -> float:
    if result.spec.is_hosted:
        pricing = API_PRICING[result.spec.model]
        per_call = (result.mean_prompt_tokens / 1e6 * pricing["input"] +
                     result.mean_output_tokens / 1e6 * pricing["output"])
        return per_call * CALLS_PER_BLOCK

    # Self-hosted, measured at batch size 1: a deliberately conservative upper
    # bound. Production serving with continuous batching is typically an order
    # of magnitude cheaper; `ftspec tco` models that with explicit assumptions.
    mean_latency = statistics.mean(result.latencies_ms) if result.latencies_ms else 0.0
    if mean_latency <= 0:
        return 0.0
    calls_per_hour = 3_600_000 / mean_latency
    return (gpu_cost_per_hour / calls_per_hour) * CALLS_PER_BLOCK


# --- Reporting ---------------------------------------------------------------

def _ci(stat: M.FieldStat) -> str:
    return f"{100*stat.mean:.1f}% [{100*stat.ci_low:.0f}–{100*stat.ci_high:.0f}]"


def render_report(results: dict, profile: Profile, gpu_cost_per_hour: float,
                   n_eval: int, refused: list) -> str:
    order = [n for n in system_catalogue() if n in results]
    plan = profile.scoring_plan()
    headline = plan.headline()
    compliance = profile.compliance
    L: list = []

    L.append(f"# Benchmark Matrix — {profile.title}")
    L.append("")
    L.append(f"Profile `{profile.name}` · regulatory regime **{compliance.regime}** · "
              f"held-out set **n = {n_eval}**, generated from a separate seed and "
              "de-duplicated against training data.")
    L.append("")

    if refused:
        L.append("## Compliance gate")
        L.append("")
        L.append(f"The following hosted baselines were **not run**: "
                  f"{', '.join('`' + r + '`' for r in refused)}.")
        L.append("")
        L.append(f"> {compliance.residency_note}")
        L.append("")
        L.append("For this vertical the comparison is not 'specialized model versus frontier "
                  "API at lower cost'. The hosted options are not lawfully available for this "
                  "data, so a self-hosted specialized model is the only admissible design. "
                  "Cost is a secondary argument here, not the primary one.")
        L.append("")

    L.append("## What is measured")
    L.append("")
    L.append("Schema adherence is reported but is **not** the headline. Grammar-constrained "
              "decoding and provider structured-output modes both guarantee 100% adherence "
              "with no training, so it is available for free. What decides whether "
              "specialization pays is whether the extracted values are correct — above all "
              f"`{headline.path if headline else 'the derived field'}`, which is fixed by an "
              "internal policy a general model cannot know unless you pay to put it in every "
              "prompt.")
    L.append("")

    L.append("## Matrix")
    L.append("")
    L.append("| System | Prompt tok | Schema adherence | Record exact | "
              f"`{headline.path if headline else 'headline'}` | p50 | p99 | Cost / 100k |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in order:
        r = results[name]
        head = _ci(r.agg.fields[headline.path]) if headline else "—"
        L.append(f"| `{name}` | {r.mean_prompt_tokens:.0f} | {r.adherence_pct:.1f}% | "
                  f"{_ci(r.agg.record_exact)} | {head} | {r.p50:.0f} ms | {r.p99:.0f} ms | "
                  f"${cost_per_100k(r, gpu_cost_per_hour):,.2f} |")
    L.append("")
    L.append("Bracketed ranges are 95% bootstrap confidence intervals. Overlapping intervals "
              "mean this sample size does not establish a difference.")
    L.append("")

    if "base-constrained" in results:
        bc = results["base-constrained"]
        L.append("## Why schema adherence is not the result")
        L.append("")
        L.append(f"`base-constrained` is the untrained base model with grammar-constrained "
                  f"decoding: **{bc.adherence_pct:.1f}% adherence**, free, no fine-tuning. It "
                  f"nonetheless gets only **{100*bc.agg.record_exact.mean:.1f}%** of records "
                  f"fully correct")
        if headline:
            L.append(f"and **{100*bc.agg.fields[headline.path].mean:.1f}%** of "
                      f"`{headline.path}` values right.")
        L.append("")
        L.append("Syntactic validity and semantic correctness are independent problems. A "
                  "vendor quoting the first as their headline is selling you something you "
                  "already have.")
        L.append("")

    L.append("## The prompt tax")
    L.append("")
    L.append("A prompted model carries the schema and the business policy in **every request, "
              "forever**. A specialized model carries them in its weights.")
    L.append("")
    L.append("| System | Prompt tokens / call | Extra vs specialized | Extra tokens / 100k calls |")
    L.append("|---|---:|---:|---:|")
    ft_tokens = results["finetuned"].mean_prompt_tokens if "finetuned" in results else 0.0
    for name in order:
        r = results[name]
        delta = r.mean_prompt_tokens - ft_tokens
        L.append(f"| `{name}` | {r.mean_prompt_tokens:.0f} | "
                  f"{'—' if delta <= 0 else f'+{delta:.0f}'} | "
                  f"{'—' if delta <= 0 else f'{delta * CALLS_PER_BLOCK:,.0f}'} |")
    L.append("")

    L.append("## Per-field accuracy")
    L.append("")
    L.append("| Field | " + " | ".join(f"`{n}`" for n in order) + " |")
    L.append("|---" * (len(order) + 1) + "|")
    for spec in plan.fields:
        marker = " ⭐" if spec.headline else ""
        cells = [f"{100*results[n].agg.fields[spec.path].mean:.1f}%" for n in order]
        L.append(f"| `{spec.path}`{marker} | " + " | ".join(cells) + " |")
    if plan.rubric_input_paths:
        L.append("| **policy inputs (mean)** | " + " | ".join(
            f"{100*results[n].agg.rubric_inputs.mean:.1f}%" for n in order) + " |")
    L.append("| **record exact** | " + " | ".join(
        f"{100*results[n].agg.record_exact.mean:.1f}%" for n in order) + " |")
    L.append("")

    if "finetuned" in results and len(order) > 1:
        L.append("## Statistical significance (McNemar exact, paired)")
        L.append("")
        L.append("Whether the specialized model's record-level accuracy genuinely differs from "
                  "each baseline on the same inputs, rather than differing by chance.")
        L.append("")
        L.append("| Comparison | specialized only | baseline only | p | verdict |")
        L.append("|---|---:|---:|---:|---|")
        ft = [s[M.RECORD_EXACT] for s in results["finetuned"].scores]
        for name in order:
            if name == "finetuned":
                continue
            other = [s[M.RECORD_EXACT] for s in results[name].scores]
            b, c, p = M.mcnemar_exact(ft, other)
            verdict = "significant (p<0.05)" if p < 0.05 else "not established"
            L.append(f"| finetuned vs `{name}` | {b} | {c} | {p:.4f} | {verdict} |")
        L.append("")

    baseline = next((n for n in ("gpt4o-rubric", "gpt4o-schema", "base-rubric", "base-constrained")
                      if n in results), None)
    if baseline and headline:
        r = results[baseline]
        labels = sorted({str(g) for g in r.gold_headline})
        preds = []
        for raw in r.raw_outputs:
            try:
                obj = json.loads(raw.strip().strip("`"))
                value = M.get_path(obj, headline.path)
                preds.append(str(value) if value is not M.MISSING else "<invalid>")
            except Exception:
                preds.append("<invalid>")
        if len(preds) == len(r.gold_headline):
            L.append(f"## How `{baseline}` misjudges `{headline.path}`")
            L.append("")
            L.append("Rows are the correct value per the internal policy; columns are the "
                      "prediction. Systematic off-diagonal mass is the business cost of a model "
                      "that does not know your rules.")
            L.append("")
            L.append(M.format_confusion(
                M.confusion(preds, [str(g) for g in r.gold_headline], labels), labels))
            L.append("")

    L.append("## Cost model and assumptions")
    L.append("")
    L.append(f"- Self-hosted: (GPU ${gpu_cost_per_hour:.2f}/hour) ÷ (calls/hour at the "
              "**measured batch-size-1 latency**) × 100,000. A conservative upper bound; "
              "continuous batching in production is materially cheaper (`ftspec tco`).")
    for model_name, pricing in API_PRICING.items():
        L.append(f"- `{model_name}`: ${pricing['input']}/1M input, "
                  f"${pricing['output']}/1M output tokens.")
    L.append("- Hosted costs use **measured** token counts from the provider's usage response.")
    L.append("- Latency is end-to-end per call, single concurrency, same host and eval set "
              "for every local system.")
    L.append("")
    return "\n".join(L)


# --- Orchestration -----------------------------------------------------------

def run(profile: Profile, eval_file: Path, results_dir: Path,
         systems: str = "finetuned",
         base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
         finetuned_model: str | None = None,
         limit: int | None = None,
         gpu_cost_per_hour: float = 0.35,
         max_new_tokens: int = 512,
         acknowledge_egress: bool = False) -> dict:
    """Run the requested systems and write the comparison matrix."""
    catalogue = system_catalogue()
    records = load_eval_set(eval_file, limit)
    log.info("loaded %d eval documents from %s", len(records), eval_file)

    requested = [s.strip() for s in systems.split(",") if s.strip()]
    unknown = [s for s in requested if s not in catalogue]
    if unknown:
        raise ValueError(f"Unknown system(s): {unknown}. Available: {list(catalogue)}")

    headline = profile.scoring_plan().headline()
    results: dict = {}
    refused: list = []

    for name in requested:
        spec = catalogue[name]
        spec.model = finetuned_model if name == "finetuned" else (
            base_model if spec.backend == "local" else spec.model)

        if spec.is_hosted:
            # Compliance gate: regulated data does not leave the boundary on a
            # default flag value.
            if not profile.compliance.allows_external_api and not acknowledge_egress:
                log.error("refusing to run %s\n%s", name, profile.compliance.egress_refusal())
                refused.append(name)
                continue
            if not os.environ.get("OPENAI_API_KEY"):
                log.warning("skipping %s: OPENAI_API_KEY not set", name)
                continue
            if not profile.compliance.allows_external_api:
                log.warning("running %s under --acknowledge-egress for a %s profile; "
                             "this is recorded in the run manifest",
                             name, profile.compliance.regime)

        log.info("running %s (prompt=%s, model=%s)", name, spec.prompt_variant, spec.model)
        result = run_openai(spec, records, profile) if spec.is_hosted \
            else run_local(spec, records, profile, max_new_tokens)

        if headline:
            result.gold_headline = [
                M.get_path(json.loads(r["messages"][2]["content"]), headline.path)
                for r in records]
        result.agg = M.aggregate(result.scores, profile.scoring_plan())
        results[name] = result
        log.info("  %s: adherence=%.1f%% record_exact=%.1f%% p50=%.0fms p99=%.0fms",
                  name, result.adherence_pct, 100 * result.agg.record_exact.mean,
                  result.p50, result.p99)

    if not results:
        if refused:
            raise RuntimeError(
                "Every requested system was refused by the compliance gate.\n"
                + profile.compliance.egress_refusal())
        raise RuntimeError("No systems ran. Check --systems and credentials.")

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        with open(results_dir / f"raw_{name}.jsonl", "w", encoding="utf-8") as f:
            for raw, score, latency in zip(r.raw_outputs, r.scores, r.latencies_ms, strict=True):
                f.write(json.dumps({"output": raw, "scores": score, "latency_ms": latency},
                                    ensure_ascii=False) + "\n")

    report = render_report(results, profile, gpu_cost_per_hour, len(records), refused)
    out = results_dir / "benchmark_report.md"
    out.write_text(report + "\n", encoding="utf-8")
    log.info("matrix written -> %s", out)

    return {
        "profile": profile.name,
        "regime": profile.compliance.regime,
        "n_eval": len(records),
        "report_path": str(out),
        "refused_for_compliance": refused,
        "egress_acknowledged": acknowledge_egress,
        "systems": {
            name: {
                "schema_adherence_pct": round(r.adherence_pct, 2),
                "record_exact_pct": round(100 * r.agg.record_exact.mean, 2),
                "headline_field": headline.path if headline else None,
                "headline_accuracy_pct": round(100 * r.agg.field_mean(headline.path), 2)
                if headline else None,
                "p50_ms": round(r.p50, 1),
                "p99_ms": round(r.p99, 1),
                "mean_prompt_tokens": round(r.mean_prompt_tokens, 1),
                "cost_per_100k_usd": round(cost_per_100k(r, gpu_cost_per_hour), 2),
            } for name, r in results.items()
        },
    }


# --- Self-test ---------------------------------------------------------------

def self_test(records: list, profile: Profile) -> dict:
    """Verify the scoring pipeline without a GPU or an API key.

    Not a benchmark: it scores gold against itself (must be 100%), against a
    deliberately corrupted copy, and against unparseable output (must be 0%).
    """
    plan = profile.scoring_plan()
    golds = [json.loads(r["messages"][2]["content"]) for r in records]
    headline = plan.headline()

    perfect = M.aggregate([M.score_record(g, g, plan) for g in golds], plan)
    assert perfect.record_exact.mean == 1.0, "gold-vs-gold must score 1.0"

    corrupted_scores = []
    for gold in golds:
        bad = json.loads(json.dumps(gold))
        if headline:
            cur = bad
            parts = headline.path.split(".")
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = "__wrong__"
        corrupted_scores.append(M.score_record(bad, gold, plan))
    corrupted = M.aggregate(corrupted_scores, plan)

    invalid = M.aggregate([M.score_record(None, g, plan) for g in golds], plan)
    assert invalid.record_exact.mean == 0.0, "contract violation must score 0"

    b, c, p = M.mcnemar_exact([s[M.RECORD_EXACT] for s in
                                [M.score_record(g, g, plan) for g in golds]],
                               [s[M.RECORD_EXACT] for s in corrupted_scores])

    print(f"Scoring self-test — profile '{profile.name}' ({len(plan.fields)} scored fields)")
    print(f"  gold vs gold   : record_exact = {100*perfect.record_exact.mean:.1f}%  (expect 100.0%)")
    print(f"  corrupted      : record_exact = {100*corrupted.record_exact.mean:.1f}%"
           + (f", {headline.path} = {100*corrupted.fields[headline.path].mean:.1f}%"
               if headline else ""))
    print(f"  contract break : record_exact = {100*invalid.record_exact.mean:.1f}%  (expect 0.0%)")
    print(f"  McNemar perfect vs corrupted: b={b} c={c} p={p:.2e}")
    print(f"  bootstrap CI on gold: [{100*perfect.record_exact.ci_low:.1f}, "
           f"{100*perfect.record_exact.ci_high:.1f}]")
    print("OK — metrics pipeline behaves correctly.")

    return {
        "profile": profile.name,
        "n": len(golds),
        "scored_fields": len(plan.fields),
        "gold_vs_gold_record_exact": round(100 * perfect.record_exact.mean, 2),
        "corrupted_record_exact": round(100 * corrupted.record_exact.mean, 2),
        "contract_break_record_exact": round(100 * invalid.record_exact.mean, 2),
        "mcnemar_p": p,
        "passed": True,
    }
