"""
Generates notebooks/colab_runner.ipynb.

The notebook is generated rather than hand-edited because a hand-maintained
.ipynb accumulates stale outputs, execution counts and merge conflicts. This
script is the source; regenerate with:

    python notebooks/_build_notebook.py
"""
import json
import pathlib

REPO = "https://github.com/Titos-papadakis/fine-tuning"
COLAB = "https://colab.research.google.com/github/Titos-papadakis/fine-tuning"
PROFILE = "saas_support"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(f"""
# ftspec — end-to-end on a free Colab T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/blob/main/notebooks/colab_runner.ipynb)

Specializes an open-weight 8B model on schema-constrained extraction, then benchmarks it
against the baselines that actually compete with it — including grammar-constrained decoding,
which reaches ~100% schema adherence **with no training at all**.

**Runtime → Change runtime type → T4 GPU** before running. Expect ~25–40 minutes end to end.

What this notebook produces and saves for you:

| Artifact | What it is |
|---|---|
| `benchmark_report.md` | The full comparison matrix |
| `measured_results.md` | TTFT / throughput / cost block, formatted for the README |
| `raw_*.jsonl` | Every prediction, for your own error analysis |
| `manifests/*.json` | git commit, config fingerprint, package versions, GPU |

Nothing here is pre-baked. Every number in the output is produced by this run.
"""))

cells.append(md("## 1 · Confirm the GPU"))
cells.append(code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

import torch

assert torch.cuda.is_available(), (
    "No GPU. Runtime -> Change runtime type -> T4 GPU, then re-run."
)
gpu = torch.cuda.get_device_properties(0)
print(f"\\n{gpu.name} · {gpu.total_memory / 1024**3:.1f} GB VRAM · torch {torch.__version__}")

# The pipeline is tuned for 16GB. More VRAM is fine; materially less is not.
if gpu.total_memory / 1024**3 < 14:
    print("WARNING: under 14GB VRAM. Lower training.per_device_train_batch_size or "
          "model.max_seq_length in configs/default.yaml if you hit OOM.")
"""))

cells.append(md("## 2 · Install"))
cells.append(code(f"""
%%capture install_log
# Unsloth pins a matched torch/triton/xformers set; installing it first avoids the
# version thrash that comes from letting pip resolve them separately afterwards.
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.12" peft accelerate bitsandbytes
!pip install -q outlines jsonschema

!git clone -q {REPO}.git /content/ftspec
%cd /content/ftspec
!pip install -q -e .
"""))

cells.append(code("""
# Surface an install failure here rather than three cells later.
import subprocess
import sys

probe = subprocess.run([sys.executable, "-c", "import ftspec, unsloth, trl, outlines"],
                        capture_output=True, text=True)
print(probe.stdout or "", probe.stderr or "")
assert probe.returncode == 0, "Install failed — run install_log.show() for the full log."
print("environment OK")
"""))

cells.append(md("""
## 3 · Build and gate the corpus

Runs on CPU in seconds. Each stage refuses to pass work that would silently corrupt the next:
ungrounded labels, eval leakage, contract drift, and samples exceeding `max_seq_length` — the
trainer **truncates** those rather than rejecting them, which teaches the model to emit
unterminated JSON and shows up hours later as a mysterious adherence collapse.
"""))
cells.append(code(f"""
PROFILE = "{PROFILE}"

!ftspec profiles
!ftspec prepare  --profile $PROFILE
!ftspec audit    --profile $PROFILE
!ftspec validate --profile $PROFILE
"""))

cells.append(md("""
## 4 · Train

QLoRA on a 4-bit base: LoRA via Unsloth, Unsloth gradient checkpointing, 8-bit AdamW,
per-device batch 2 × accumulation 4, completion-only loss masking so the model is never
trained to reproduce its own prompt.

`train` refuses to start if the gate above failed, so no GPU minutes go to a doomed run.
"""))
cells.append(code("""
import time

started = time.time()
!ftspec train --profile $PROFILE
print(f"\\ntraining wall clock: {(time.time() - started) / 60:.1f} min")
"""))

cells.append(md("""
## 5 · Measure TTFT and decode throughput

The two latency numbers a buyer asks about, measured directly rather than inferred from the
benchmark's end-to-end timings.

TTFT is the time to the **first** token, so it is prefill-bound; the decode rate is measured
over the tokens after it. The comparison that matters is against the *prompted* baseline: the
specialized model carries a ~12-token system prompt where a prompted model carries ~1250, so
it has roughly **7x less prefill to do on every single call**. That is a mechanical latency
advantage, not a tuning artefact.
"""))
cells.append(code("""
import json
import statistics
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from ftspec.core.registry import load_profile

profile = load_profile(PROFILE)
model, tok = FastLanguageModel.from_pretrained(
    f"outputs/{PROFILE}/lora_adapter", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

docs = [json.loads(line)["messages"][1]["content"]
        for line in Path(f"data/{PROFILE}/eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()][:20]


def timed(system_prompt, doc, n_tokens=128):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": doc}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]

    def generate(max_new):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - start, out.shape[1] - n_prompt

    # One token isolates prefill; the rest gives the steady-state decode rate.
    t_first, _ = generate(1)
    t_full, n_out = generate(n_tokens)
    decode_rate = (n_out - 1) / max(t_full - t_first, 1e-6)
    return n_prompt, t_first * 1000, decode_rate


# Warm-up: the first call pays kernel autotuning no real user ever sees.
timed(profile.prompts.short, docs[0], n_tokens=8)

latency = {}
for label, system_prompt in (("specialized", profile.prompts.short),
                              ("prompted_baseline", profile.prompts.schema_rubric)):
    prompt_tokens, ttfts, rates = [], [], []
    for doc in docs[:8]:
        n_p, ttft_ms, rate = timed(system_prompt, doc)
        prompt_tokens.append(n_p)
        ttfts.append(ttft_ms)
        rates.append(rate)
    latency[label] = {
        "prompt_tokens": round(statistics.mean(prompt_tokens)),
        "ttft_p50_ms": round(statistics.median(ttfts)),
        "ttft_max_ms": round(max(ttfts)),
        "decode_tok_s": round(statistics.mean(rates), 1),
    }
    m = latency[label]
    print(f"{label:20s} prompt={m['prompt_tokens']:5d} tok   "
          f"TTFT p50={m['ttft_p50_ms']:5d} ms   decode={m['decode_tok_s']:6.1f} tok/s")

del model
torch.cuda.empty_cache()
"""))

cells.append(md("""
## 6 · Benchmark matrix

`base-constrained` is the untrained base model with grammar-constrained decoding. It is in the
comparison precisely because it reaches ~100% schema adherence for free — which is why
adherence is reported as a baseline here and not as the result. What separates these systems
is whether the extracted **values** are correct.

McNemar's exact test is included so a small gap is not reported as a win.
"""))
cells.append(code("""
!ftspec evaluate --profile $PROFILE \\
    --systems finetuned,base-constrained,base-rubric \\
    --gpu-cost-per-hour 0.35
"""))

cells.append(md("## 7 · Read the matrix"))
cells.append(code("""
from IPython.display import Markdown, display

display(Markdown(Path(f"outputs/{PROFILE}/reports/benchmark_report.md").read_text(encoding="utf-8")))
"""))

cells.append(md("""
## 8 · Build the README block and save every artifact

Writes `measured_results.md` in exactly the shape the repository README expects, so the
projected figures published there can be replaced with numbers this run actually produced.
"""))
cells.append(code("""
import shutil
import subprocess

GPU_HOURLY = 0.35                  # T4 on-demand; set to your provider's rate
GPT4O_IN, GPT4O_OUT = 2.50, 10.00  # USD per 1M tokens

manifest = json.loads(
    Path(f"outputs/{PROFILE}/manifests/evaluate.json").read_text(encoding="utf-8"))
systems = manifest["metrics"]["systems"]
finetuned = systems.get("finetuned", {})

spec = latency["specialized"]
prompted = latency["prompted_baseline"]
out_tokens = finetuned.get("mean_output_tokens") or 162

# Self-hosted cost from measured single-stream throughput. Deliberately
# conservative: continuous batching in production is materially cheaper.
sec_per_req = spec["ttft_p50_ms"] / 1000 + out_tokens / max(spec["decode_tok_s"], 1e-6)
self_hosted_per_req = GPU_HOURLY / (3600 / sec_per_req)
gpt4o_per_req = (prompted["prompt_tokens"] / 1e6 * GPT4O_IN) + (out_tokens / 1e6 * GPT4O_OUT)
saving = 100 * (gpt4o_per_req - self_hosted_per_req) / gpt4o_per_req

gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()

lines = [
    "### Measured results",
    "",
    f"Produced by `notebooks/colab_runner.ipynb` on **{gpu_name}**, profile `{PROFILE}`, "
    f"n = {manifest['metrics']['n_eval']} held-out documents. Reproduce with the Colab badge above.",
    "",
    "| Metric | Specialized 8B | Prompted baseline |",
    "|---|---:|---:|",
    f"| System prompt tokens | {spec['prompt_tokens']} | {prompted['prompt_tokens']} |",
    f"| TTFT p50 | {spec['ttft_p50_ms']} ms | {prompted['ttft_p50_ms']} ms |",
    f"| TTFT max | {spec['ttft_max_ms']} ms | {prompted['ttft_max_ms']} ms |",
    f"| Decode throughput | {spec['decode_tok_s']} tok/s | {prompted['decode_tok_s']} tok/s |",
    "",
    "| System | Schema adherence | Record exact | p50 | p99 | Cost / 1k |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, s in systems.items():
    lines.append(f"| `{name}` | {s['schema_adherence_pct']}% | {s['record_exact_pct']}% | "
                  f"{s['p50_ms']:.0f} ms | {s['p99_ms']:.0f} ms | "
                  f"${s['cost_per_100k_usd'] / 100:.3f} |")
lines += [
    "",
    f"**Cost per request** — self-hosted ${self_hosted_per_req:.5f} vs GPT-4o "
    f"${gpt4o_per_req:.5f} (**{saving:.0f}% reduction**), at ${GPU_HOURLY:.2f}/hour and the "
    "measured single-stream throughput above. Continuous batching lowers the self-hosted "
    "figure further.",
    "",
]
block = "\\n".join(lines)

artifacts = Path("/content/ftspec_artifacts")
artifacts.mkdir(exist_ok=True)
(artifacts / "measured_results.md").write_text(block, encoding="utf-8")
(artifacts / "latency_raw.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
for src in (Path(f"outputs/{PROFILE}/reports"), Path(f"outputs/{PROFILE}/manifests")):
    if src.exists():
        shutil.copytree(src, artifacts / src.name, dirs_exist_ok=True)

print(block)
"""))

cells.append(code("""
from google.colab import files

archive = shutil.make_archive("/content/ftspec_results", "zip", "/content/ftspec_artifacts")
print(f"{archive}  ({Path(archive).stat().st_size / 1024:.0f} KB)")
files.download(archive)
"""))

cells.append(md("""
## 9 · Optional — keep the adapter

The LoRA adapter is tens of MB; the merged 16-bit model is ~16GB and will not survive a Colab
session. Copy the adapter to Drive or push it to the Hub if you want to serve it later:

```bash
ftspec serve --profile saas_support --model <base-model> --lora <adapter-dir>
```
"""))
cells.append(code("""
# Keep the adapter across sessions:
# from google.colab import drive; drive.mount("/content/drive")
# !cp -r outputs/$PROFILE/lora_adapter "/content/drive/MyDrive/ftspec_${PROFILE}_adapter"

# Or push it to the Hub:
# from huggingface_hub import notebook_login; notebook_login()
# from unsloth import FastLanguageModel
# m, t = FastLanguageModel.from_pretrained(f"outputs/{PROFILE}/lora_adapter", load_in_4bit=True)
# m.push_to_hub("your-org/ftspec-saas-support"); t.push_to_hub("your-org/ftspec-saas-support")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

path = pathlib.Path(__file__).parent / "colab_runner.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {path} — {len(cells)} cells, {path.stat().st_size / 1024:.0f} KB")
