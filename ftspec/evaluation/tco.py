"""
Total cost of ownership and break-even analysis.

"70% cheaper" is a claim that collapses the moment a CFO asks "at what volume?"
Self-hosting is a fixed cost: the GPU bills 24/7 whether or not you send it
traffic. An API is a pure variable cost. So there is always a crossover volume
below which the API is genuinely the cheaper choice, and saying so is what makes
the analysis credible when it says self-hosting wins above that line.

This module computes that crossover explicitly, including:
  - one-time fine-tuning cost, amortised over a stated period
  - GPU count driven by peak throughput, not average
  - continuous-batching throughput as a NAMED assumption, not a hidden fudge

Run:
    python evaluation/tco.py
    python evaluation/tco.py --peak-rps 12 --gpu-hourly 0.75 --gpu-name L4
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ftspec.config import REPO_ROOT
from ftspec.run import get_logger

log = get_logger("ftspec.tco")
RESULTS_DIR = REPO_ROOT / "outputs" / "reports"

HOURS_PER_MONTH = 730


@dataclass
class ApiOption:
    name: str
    price_in_per_1m: float
    price_out_per_1m: float
    prompt_tokens: int
    output_tokens: int

    def cost_per_request(self) -> float:
        return (self.prompt_tokens / 1e6 * self.price_in_per_1m +
                self.output_tokens / 1e6 * self.price_out_per_1m)

    def monthly_cost(self, volume: int) -> float:
        return self.cost_per_request() * volume


@dataclass
class SelfHostOption:
    name: str
    gpu_hourly: float
    throughput_rps: float          # sustained requests/sec/GPU with continuous batching
    training_cost_once: float
    amortise_months: int

    def gpus_needed(self, peak_rps: float) -> int:
        return max(1, math.ceil(peak_rps / self.throughput_rps))

    def monthly_cost(self, volume: int, peak_rps: float) -> float:
        gpus = self.gpus_needed(peak_rps)
        compute = gpus * self.gpu_hourly * HOURS_PER_MONTH
        return compute + self.training_cost_once / self.amortise_months


def break_even_volume(api: ApiOption, host: SelfHostOption, peak_rps: float) -> float:
    """Monthly request volume at which self-hosting becomes cheaper."""
    fixed = host.monthly_cost(0, peak_rps)
    per_req = api.cost_per_request()
    if per_req <= 0:
        return float("inf")
    return fixed / per_req


def render(apis: list, host: SelfHostOption, volumes: list, peak_rps: float) -> str:
    L = []
    L.append("# Total Cost of Ownership — Self-Hosted Specialized 8B vs Proprietary APIs")
    L.append("")
    L.append("## Assumptions (change these before quoting any number)")
    L.append("")
    L.append(f"- Self-hosted GPU: **{host.name}** at **${host.gpu_hourly:.2f}/hour**, billed "
              f"{HOURS_PER_MONTH} hours/month (always-on).")
    L.append(f"- Sustained throughput: **{host.throughput_rps:.1f} requests/sec/GPU** with "
              "continuous batching (vLLM/TGI). This is the single most load-bearing assumption "
              "here — measure it on your own hardware before relying on it.")
    L.append(f"- Peak load: **{peak_rps:.1f} requests/sec**, requiring "
              f"**{host.gpus_needed(peak_rps)} GPU(s)**. Capacity is sized on peak, not average.")
    L.append(f"- One-time fine-tuning cost: **${host.training_cost_once:.2f}**, amortised over "
              f"{host.amortise_months} months.")
    for a in apis:
        L.append(f"- `{a.name}`: {a.prompt_tokens} prompt + {a.output_tokens} output tokens/request "
                  f"at ${a.price_in_per_1m}/${a.price_out_per_1m} per 1M — "
                  f"**${a.cost_per_request()*1000:.2f} per 1,000 requests**.")
    L.append("")
    L.append("The prompt-token counts above are the real difference: an API call must carry the "
              "JSON schema and the triage rubric every single time, while the fine-tuned model "
              "carries them in its weights.")
    L.append("")

    L.append("## Monthly cost by volume")
    L.append("")
    header = "| Monthly requests | " + " | ".join(f"`{a.name}`" for a in apis) + \
             f" | self-hosted ({host.name}) | cheapest |"
    L.append(header)
    L.append("|---:" * (len(apis) + 2) + "|---|")
    for v in volumes:
        api_costs = [a.monthly_cost(v) for a in apis]
        host_cost = host.monthly_cost(v, peak_rps)
        options = list(zip([a.name for a in apis], api_costs, strict=True)) + [(host.name, host_cost)]
        winner = min(options, key=lambda t: t[1])[0]
        cells = " | ".join(f"${c:,.0f}" for c in api_costs)
        L.append(f"| {v:,} | {cells} | ${host_cost:,.0f} | **{winner}** |")
    L.append("")

    L.append("## Break-even volume")
    L.append("")
    L.append("| API | Break-even (requests/month) | Interpretation |")
    L.append("|---|---:|---|")
    for a in apis:
        be = break_even_volume(a, host, peak_rps)
        interp = ("self-hosting wins above this volume" if be < float("inf")
                   else "n/a")
        L.append(f"| `{a.name}` | {be:,.0f} | {interp} |")
    L.append("")
    L.append("**Below the break-even line, the honest recommendation is to keep using the API.** "
              "Specialization pays off on sustained, high-volume, narrow workloads — which is "
              "exactly the profile worth checking before committing to a project.")
    L.append("")

    L.append("## Beyond cost")
    L.append("")
    L.append("Cost is rarely the only driver, and often not the deciding one:")
    L.append("")
    L.append("- **Data residency**: transcripts never leave your infrastructure.")
    L.append("- **Latency floor**: no network round trip to a third-party region.")
    L.append("- **Behavioural stability**: the weights are yours; nobody deprecates or silently "
              "updates your model mid-quarter.")
    L.append("- **Policy encoding**: your triage rubric lives in the model, not in a prompt that "
              "every integration has to remember to include correctly.")
    L.append("")
    return "\n".join(L)


def run(gpu_name: str = "A10G", gpu_hourly: float = 0.75, throughput_rps: float = 8.0,
        peak_rps: float = 5.0, training_cost: float = 15.0, amortise_months: int = 12,
        prompt_tokens_api: int = 820, output_tokens: int = 190) -> tuple[str, dict]:
    """Compute the break-even analysis. Returns (report_markdown, metrics)."""
    apis = [
        ApiOption("gpt-4o", 2.50, 10.00, prompt_tokens_api, output_tokens),
        ApiOption("gpt-4o-mini", 0.15, 0.60, prompt_tokens_api, output_tokens),
    ]
    host = SelfHostOption(gpu_name, gpu_hourly, throughput_rps, training_cost, amortise_months)

    volumes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    report = render(apis, host, volumes, peak_rps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tco_report.md"
    out.write_text(report + "\n", encoding="utf-8")
    log.info("report written -> %s", out)

    metrics = {
        "gpus_needed": host.gpus_needed(peak_rps),
        "monthly_self_hosted_usd": round(host.monthly_cost(0, peak_rps), 2),
        "break_even_requests_per_month": {
            a.name: round(break_even_volume(a, host, peak_rps)) for a in apis
        },
        "cost_per_1k_usd": {a.name: round(a.cost_per_request() * 1000, 4) for a in apis},
        "report_path": str(out),
    }
    return report, metrics
