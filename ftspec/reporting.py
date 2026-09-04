"""
Turns a completed run into the README's measured-results block.

The repository publishes *projected* economics until someone runs the pipeline
on real hardware. This module replaces those projections with what a run
actually produced, in place, between the BENCHMARK markers.

Keeping the block format here rather than in the notebook means the Colab run
and a local `ftspec report` emit byte-identical output, so a number in the
README is traceable to exactly one code path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ftspec.config import Config
from ftspec.run import get_logger

log = get_logger("ftspec.report")

BEGIN = "<!-- BENCHMARK:BEGIN -->"
END = "<!-- BENCHMARK:END -->"

# Published pricing, USD per 1M tokens. Printed in the block so a reader can
# check it against a current rate card instead of trusting the script.
GPT4O_IN, GPT4O_OUT = 2.50, 10.00
GPT4O_MINI_IN, GPT4O_MINI_OUT = 0.15, 0.60


@dataclass
class LatencyPair:
    """Measured TTFT and decode rate for the two prompt regimes."""
    specialized: dict
    prompted_baseline: dict

    @classmethod
    def load(cls, path: Path) -> LatencyPair | None:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "specialized" not in raw or "prompted_baseline" not in raw:
            log.warning("%s lacks the expected keys; skipping the latency table", path)
            return None
        return cls(specialized=raw["specialized"], prompted_baseline=raw["prompted_baseline"])


def _fmt_usd(value: float) -> str:
    return f"${value:,.5f}" if value < 0.01 else f"${value:,.2f}"


def build_block(profile_name: str, manifest: dict, latency: LatencyPair | None,
                 gpu_hourly: float, gpu_name: str = "") -> str:
    metrics = manifest.get("metrics", {})
    systems = metrics.get("systems", {})
    if not systems:
        raise ValueError(
            "The evaluate manifest has no system results. Run `ftspec evaluate` first "
            "(the run may have been refused by the compliance gate)."
        )

    gpu = gpu_name or (manifest.get("gpu") or {}).get("name") or "an unspecified GPU"
    n_eval = metrics.get("n_eval", "?")
    commit = (manifest.get("git_commit") or "")[:8]
    dirty = " (working tree dirty)" if manifest.get("git_dirty") else ""

    lines = [
        "### Measured results",
        "",
        f"Produced by `ftspec evaluate` on **{gpu}**, profile `{profile_name}`, "
        f"n = {n_eval} held-out documents"
        + (f", commit `{commit}`{dirty}" if commit else "") + ".",
        "Reproduce with the Colab badge above, or `ftspec evaluate --profile "
        f"{profile_name}`.",
        "",
    ]

    finetuned = systems.get("finetuned", {})
    out_tokens = finetuned.get("mean_output_tokens") or 162

    if latency is not None:
        spec, base = latency.specialized, latency.prompted_baseline
        ratio = base["prompt_tokens"] / max(spec["prompt_tokens"], 1)
        lines += [
            "| Metric | Specialized 8B | Prompted baseline |",
            "|---|---:|---:|",
            f"| System + document prompt | {spec['prompt_tokens']:,} tok | "
            f"{base['prompt_tokens']:,} tok |",
            f"| TTFT p50 | **{spec['ttft_p50_ms']:,} ms** | {base['ttft_p50_ms']:,} ms |",
            f"| TTFT max | {spec.get('ttft_max_ms', 0):,} ms | "
            f"{base.get('ttft_max_ms', 0):,} ms |",
            f"| Decode throughput | {spec['decode_tok_s']} tok/s | "
            f"{base['decode_tok_s']} tok/s |",
            "",
            f"The specialized model does **{ratio:.1f}× less prefill** on every call, which is "
            "why its TTFT is lower on identical hardware.",
            "",
        ]

    lines += [
        "| System | Schema adherence | Record exact | p50 | p99 | Cost / 1k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in systems.items():
        lines.append(
            f"| `{name}` | {s['schema_adherence_pct']}% | {s['record_exact_pct']}% | "
            f"{s['p50_ms']:.0f} ms | {s['p99_ms']:.0f} ms | "
            f"${s['cost_per_100k_usd'] / 100:.3f} |"
        )
    lines.append("")

    # Cost comparison, from measured throughput where available.
    if latency is not None:
        spec = latency.specialized
        base = latency.prompted_baseline
        sec_per_req = spec["ttft_p50_ms"] / 1000 + out_tokens / max(spec["decode_tok_s"], 1e-6)
        self_hosted = gpu_hourly / (3600 / sec_per_req)
        gpt4o = base["prompt_tokens"] / 1e6 * GPT4O_IN + out_tokens / 1e6 * GPT4O_OUT
        mini = base["prompt_tokens"] / 1e6 * GPT4O_MINI_IN + out_tokens / 1e6 * GPT4O_MINI_OUT
        saving = 100 * (gpt4o - self_hosted) / gpt4o

        lines += [
            f"**Cost per request** — self-hosted {_fmt_usd(self_hosted)} vs GPT-4o "
            f"{_fmt_usd(gpt4o)} (**{saving:.0f}% reduction**) at ${gpu_hourly:.2f}/hour and the "
            "measured single-stream throughput above. Continuous batching lowers the "
            "self-hosted figure substantially further.",
            "",
        ]
        if mini < self_hosted:
            # Stated plainly rather than omitted: a buyer who finds this themselves
            # stops believing the rest of the table.
            lines += [
                f"> GPT-4o-mini lands at {_fmt_usd(mini)}/request here, **below** the "
                f"unbatched self-hosted figure. At single-stream concurrency and low volume, "
                "mini is the cheaper option and this table says so. Self-hosting wins on cost "
                "once you batch — and wins outright wherever the data cannot leave your "
                "boundary at all.",
                "",
            ]

    lines += [
        f"Pricing used: GPT-4o ${GPT4O_IN}/1M input, ${GPT4O_OUT}/1M output; "
        f"GPT-4o-mini ${GPT4O_MINI_IN}/1M input, ${GPT4O_MINI_OUT}/1M output. "
        "Self-hosted cost derives from measured latency, not a vendor estimate.",
        "",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> bool:
    """Replace the marked region. Returns True if the file changed."""
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ValueError(
            f"{readme} is missing the {BEGIN} / {END} markers, so there is nowhere "
            "to put the results without guessing at the layout."
        )
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    updated = pattern.sub(f"{BEGIN}\n{block.strip()}\n{END}", text)

    # Drop the now-superseded projection notice, if it is still present.
    updated = updated.replace(
        "## Projected economics\n\n> **These are modeled, not measured.**",
        "## Economics\n\n> **Superseded by measured results below.** The projection that "
        "shipped with this repository is kept for comparison.\n>\n> Originally modeled:")

    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True


def run(cfg: Config, profile_name: str, readme: Path, gpu_hourly: float = 0.35,
         latency_file: Path | None = None, write: bool = True) -> tuple[str, bool]:
    """Build the block and optionally splice it into the README."""
    manifest_path = cfg.manifests_dir(profile_name) / "evaluate.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No evaluate manifest at {manifest_path}. Run "
            f"`ftspec evaluate --profile {profile_name}` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    latency_path = latency_file or (cfg.reports_dir(profile_name) / "latency.json")
    latency = LatencyPair.load(latency_path)
    if latency is None:
        log.info("no latency measurements at %s; omitting the TTFT table", latency_path)

    block = build_block(profile_name, manifest, latency, gpu_hourly)
    changed = update_readme(readme, block) if write else False
    return block, changed
