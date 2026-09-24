"""
The monthly page a customer actually reads: what their model did for them
this month, in their terms -- volume handled, live quality, what the same
traffic would have cost on GPT-4o, and how the model improved.

Built only from data the platform already records: usage_counters (volume,
tokens), memory/production_log.jsonl (live schema validity, latency),
memory/deployed.json + the deployments table (current model, promotions),
the baseline manifest (what prompting alone scored at onboarding) and the
archived corrections files (human fixes folded into training).

Every number is labeled with where it comes from. The GPT-4o comparison
uses the gpt4o-rubric baseline's *measured* prompt size when onboarding ran
it; otherwise an estimate from the profile's own prompts, and says so.
Savings are shown only when positive -- a report that spins a loss into a
win would be found out on the first invoice.
"""
from __future__ import annotations

import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ftplatform.billing import usage as usage_mod
from ftplatform.deployment import registry
from ftspec.evaluation.benchmark import API_PRICING

COMPARE_MODEL = "gpt-4o"
CHARS_PER_TOKEN = 4


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def _baseline_systems(ctx) -> dict:
    path = ctx.manifests_dir("baseline") / "evaluate.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {}).get("systems", {})


def _gpt4o_prompt_tokens(ctx, baseline: dict, our_mean_prompt: float) -> tuple[float, str]:
    measured = baseline.get("gpt4o-rubric", {}).get("mean_prompt_tokens")
    if measured:
        return float(measured), "measured on your eval set at onboarding (gpt4o-rubric)"
    v = ctx.profile.prompts.variants()
    extra = max(0, len(v["schema+rubric"]) - len(v["short"])) / CHARS_PER_TOKEN
    return our_mean_prompt + extra, ("estimated: your model's actual prompt size plus the "
                                     "longer schema+rubric instructions GPT-4o would need "
                                     f"(~{CHARS_PER_TOKEN} characters per token)")


def _corrections_folded(ctx, period: str) -> int:
    stamp_prefix = period.replace("-", "")
    total = 0
    for p in ctx.memory_dir().glob("corrections.*.jsonl"):
        if p.name.split(".")[1].startswith(stamp_prefix):
            total += len(_read_jsonl(p))
    return total


def build(conn, ctx, period: str | None = None, monthly_fee_usd: float | None = None) -> dict:
    period = period or usage_mod.current_period()
    usage = usage_mod.report(conn, ctx.customer.id, period=period)

    live = [r for r in _read_jsonl(ctx.memory_dir() / "production_log.jsonl")
            if str(r.get("timestamp", "")).startswith(period)]
    latencies = [r["latency_ms"] for r in live if r.get("latency_ms") is not None]
    valid_pct = (100.0 * sum(1 for r in live if r.get("contract_valid")) / len(live)
                 if live else None)

    deployed_path = ctx.memory_dir() / "deployed.json"
    deployed = json.loads(deployed_path.read_text(encoding="utf-8")) if deployed_path.exists() else None
    model = None
    if deployed and deployed.get("kind") == "deployed":
        m = next(iter(deployed["systems"].values()))
        model = {"candidate_id": deployed["candidate_id"], "since": deployed["captured_at"],
                 "record_exact_pct": m["record_exact_pct"],
                 "headline_accuracy_pct": m.get("headline_accuracy_pct"),
                 "schema_adherence_pct": m["schema_adherence_pct"]}

    baseline = _baseline_systems(ctx)
    best_prompted = None
    if baseline:
        name, m = max(baseline.items(), key=lambda kv: kv[1]["record_exact_pct"])
        best_prompted = {"system": name, "record_exact_pct": m["record_exact_pct"]}

    promotions = [r for r in registry.history(conn, ctx.customer.id)
                  if str(r["deployed_at"]).startswith(period)]

    requests = usage["requests"]
    cost = None
    if requests:
        our_prompt = usage["prompt_tokens"] / requests
        our_completion = usage["completion_tokens"] / requests
        gpt_prompt, basis = _gpt4o_prompt_tokens(ctx, baseline, our_prompt)
        price = API_PRICING[COMPARE_MODEL]
        per_call = gpt_prompt / 1e6 * price["input"] + our_completion / 1e6 * price["output"]
        equivalent = per_call * requests
        cost = {"equivalent_usd": round(equivalent, 2), "per_call_usd": round(per_call, 5),
                "basis": basis, "model": COMPARE_MODEL,
                "monthly_fee_usd": monthly_fee_usd,
                "savings_usd": (round(equivalent - monthly_fee_usd, 2)
                                if monthly_fee_usd is not None and equivalent > monthly_fee_usd
                                else None)}

    return {
        "customer": {"id": ctx.customer.id, "name": ctx.customer.name,
                     "workload": ctx.customer.workload},
        "period": period,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage": {"requests": requests, "cache_hits": usage["cache_hits"],
                  "prompt_tokens": usage["prompt_tokens"],
                  "completion_tokens": usage["completion_tokens"]},
        "live": {"captured": len(live), "valid_pct": round(valid_pct, 2) if valid_pct is not None else None,
                 "p50_ms": _pct(latencies, 0.50), "p99_ms": _pct(latencies, 0.99),
                 "mean_ms": round(statistics.mean(latencies), 1) if latencies else None},
        "model": model,
        "best_prompted_at_onboarding": best_prompted,
        "promotions": [{"candidate_id": r["candidate_id"], "deployed_at": r["deployed_at"],
                        "is_rollback": bool(r["is_rollback"])} for r in promotions],
        "corrections_folded": _corrections_folded(ctx, period),
        "pending_review": len(_read_jsonl(ctx.memory_dir() / "error_queue.jsonl")),
        "cost": cost,
    }


def _fmt(v, suffix="", digits=1) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{suffix}"
    return f"{v:,}{suffix}"


def render_html(r: dict) -> str:
    e = html.escape
    c = r["customer"]
    cards = [("Tickets processed", _fmt(r["usage"]["requests"]), "requests served this month")]
    if r["model"]:
        cards.append(("Record accuracy", _fmt(r["model"]["record_exact_pct"], "%"),
                      "every field correct, measured on your held-out set"))
    if r["live"]["valid_pct"] is not None:
        cards.append(("Valid output", _fmt(r["live"]["valid_pct"], "%"),
                      f"of {r['live']['captured']:,} live responses matched your schema"))
    if r["cost"]:
        cards.append((f"Same traffic on {e(r['cost']['model'])}", f"${r['cost']['equivalent_usd']:,.2f}",
                      "what the equivalent API calls would have cost"))

    card_html = "".join(
        f'<div class="card"><div class="label">{e(t)}</div><div class="value">{v}</div>'
        f'<div class="note">{e(n)}</div></div>' for t, v, n in cards)

    rows = []
    if r["model"] and r["best_prompted_at_onboarding"]:
        b = r["best_prompted_at_onboarding"]
        rows.append(("Accuracy vs. prompting alone",
                     f"{_fmt(r['model']['record_exact_pct'], '%')} vs {_fmt(b['record_exact_pct'], '%')} "
                     f"({e(b['system'])}, measured at onboarding)"))
    if r["model"]:
        rows.append(("Model in production", f"{e(r['model']['candidate_id'])} since {e(r['model']['since'][:10])}"))
    if r["live"]["p50_ms"] is not None:
        rows.append(("Latency (live)", f"median {_fmt(r['live']['p50_ms'], ' ms', 0)}, "
                                        f"p99 {_fmt(r['live']['p99_ms'], ' ms', 0)}"))
    if r["usage"]["cache_hits"]:
        rows.append(("Answered from cache", _fmt(r["usage"]["cache_hits"])))
    rows.append(("Human corrections folded into training", _fmt(r["corrections_folded"])))
    rows.append(("Model updates this month",
                 ", ".join(f"{e(p['candidate_id'])} ({e(p['deployed_at'][:10])}"
                           f"{', rollback' if p['is_rollback'] else ''})" for p in r["promotions"])
                 or "none"))
    if r["pending_review"]:
        rows.append(("Flagged for review", _fmt(r["pending_review"])))
    if r["cost"]:
        cost = r["cost"]
        rows.append((f"{e(cost['model'])} cost per call", f"${cost['per_call_usd']:.5f} "
                                                           f"<span class=\"muted\">({e(cost['basis'])})</span>"))
        if cost["savings_usd"] is not None:
            rows.append(("Saved vs. your plan fee", f"${cost['savings_usd']:,.2f}"))

    table = "".join(f"<tr><th>{t}</th><td>{v}</td></tr>" for t, v in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(c['name'])} &middot; {e(r['period'])}</title>
<style>
:root {{ --bg:#f7f7f5; --fg:#1b1b1b; --muted:#6b6b6b; --card:#fff; --line:#e3e3df; --accent:#1f6f5c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#ececec; --muted:#9a9a9a;
  --card:#1e1e1e; --line:#2e2e2e; --accent:#5fc3a6; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:32px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin:0 0 24px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.label {{ color:var(--muted); font-size:13px; }} .value {{ font-size:28px; font-weight:650;
  color:var(--accent); margin:4px 0; }} .note {{ color:var(--muted); font-size:12px; }}
table {{ width:100%; border-collapse:collapse; margin-top:24px; background:var(--card);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
th, td {{ text-align:left; padding:10px 14px; border-top:1px solid var(--line); vertical-align:top; }}
tr:first-child th, tr:first-child td {{ border-top:0; }} th {{ width:40%; font-weight:500; color:var(--muted); }}
.muted {{ color:var(--muted); font-size:12px; }} footer {{ color:var(--muted); font-size:12px; margin-top:24px; }}
</style></head><body><main>
<h1>{e(c['name'])} &mdash; monthly report</h1>
<p class="sub">{e(r['period'])} &middot; {e(c['workload'])}</p>
<div class="cards">{card_html}</div>
<table>{table}</table>
<footer>Generated {e(r['generated_at'])}. Accuracy figures come from your held-out evaluation
set; live figures from this month's production traffic.</footer>
</main></body></html>
"""


def write(conn, ctx, period: str | None = None, monthly_fee_usd: float | None = None,
          out_path: Path | None = None) -> tuple[Path, dict]:
    report = build(conn, ctx, period=period, monthly_fee_usd=monthly_fee_usd)
    out = Path(out_path) if out_path else (ctx.customer_root() / "reports"
                                           / f"monthly-{report['period']}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return out, report
