"""
Headless Kaggle equivalent of notebooks/colab_runner.ipynb -- now building
the stronger generic baseline (not a per-customer run): a bigger synthetic
corpus than the 200/40/150 defaults, plus grammar-constrained decoding
evaluated on the finetuned adapter itself (not just the untrained base
model). This is the first two steps of the original 4-point accuracy plan
(scale the data, turn on constrained decoding); real per-field error
analysis is already free from `render_report()`'s existing "Per-field
accuracy" table -- no new tooling needed, just reading its output after this
runs. A base-model swap (8B -> something larger) is deliberately left out of
this run: it's the riskiest, most VRAM-constrained lever, and only worth
spending T4 time on once we've seen whether the cheaper two moves are
already enough.

Runs the ftspec saas_support pipeline end to end (prepare -> audit ->
validate -> train -> evaluate x4 -> combine-eval -> report) as a plain
script instead of notebook cells, so it can execute unattended on Kaggle's
free GPU quota via `kaggle kernels push` instead of a human clicking through
Colab cells.

Kaggle kernels write to /kaggle/working, which becomes the kernel's
downloadable output -- everything worth keeping is copied there at the end
so `kaggle kernels output` can pull it back.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/Titos-papadakis/fine-tuning.git"
CLONE_DIR = Path("/kaggle/working/ftspec")
PROFILE = "saas_support"
GPU_COST_PER_HOUR = "0.35"
# 7.5x the 200-example default. n_eval is left at the original 150 on
# purpose -- that's what the 36.7% Record Exact Match baseline being chased
# here was measured on, so this run's eval stays directly comparable to it
# rather than moving both the training data and the measuring stick at once.
N_TRAIN = 1500
N_VAL = 150
N_EVAL = 150
EVAL_SYSTEMS = ("finetuned", "finetuned-constrained", "base-constrained", "base-rubric")


def run(cmd, cwd=None, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def step(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)


step("0 . Confirm the GPU")
run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    check=False)

step("1 . Clone the repo")
if CLONE_DIR.exists():
    shutil.rmtree(CLONE_DIR)
run(["git", "clone", "-q", REPO, str(CLONE_DIR)])

step("2 . Install")
# Same install order as colab_runner.ipynb, and for the same reason: unsloth
# pins a matched torch/triton/xformers set, so it goes in before anything
# else gets a chance to pull a conflicting version.
run([sys.executable, "-m", "pip", "install", "-q",
     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "trl<0.12", "peft", "accelerate", "bitsandbytes"])
run([sys.executable, "-m", "pip", "install", "-q", "outlines", "jsonschema"])
# transformers' Trainer only accepts the legacy tokenizer= kwarg (which
# unsloth's compiled trainer still passes internally) up to version 5 --
# confirmed as a hard TypeError on real T4 hardware past that.
run([sys.executable, "-m", "pip", "install", "-q", "transformers<5"])
run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=CLONE_DIR)

probe = subprocess.run([sys.executable, "-c", "import ftspec, unsloth, trl, outlines"],
                        cwd=CLONE_DIR, capture_output=True, text=True)
print(probe.stdout, probe.stderr)
if probe.returncode != 0:
    raise SystemExit("environment install failed -- see the pip/import output above")
print("environment OK")

step("3 . Build and gate the corpus")
run(["ftspec", "profiles"], cwd=CLONE_DIR)
run(["ftspec", "prepare", "--profile", PROFILE,
     "--n-train", str(N_TRAIN), "--n-val", str(N_VAL), "--n-eval", str(N_EVAL)], cwd=CLONE_DIR)
run(["ftspec", "audit", "--profile", PROFILE], cwd=CLONE_DIR)
run(["ftspec", "validate", "--profile", PROFILE], cwd=CLONE_DIR)

step("4 . Train")
started = time.time()
run(["ftspec", "train", "--profile", PROFILE], cwd=CLONE_DIR)
print(f"training wall clock: {(time.time() - started) / 60:.1f} min")

step("5 . Benchmark matrix -- one process per system, on purpose")
# Loading a second 8B model right after the first in the same process does
# not reliably get the first model's VRAM back on a free T4; a fresh process
# per system sidesteps it entirely (same reasoning as colab_runner.ipynb).
# check=False: if one system OOMs, combine-eval below still reports on
# whichever systems did succeed instead of losing the whole run.
# finetuned-constrained re-evaluates the same trained adapter with
# grammar-constrained decoding on -- a separate catalogue entry from
# "finetuned" (see ftspec/evaluation/benchmark.py) so both show up as their
# own columns/McNemar comparison in one combined report.
for systems in EVAL_SYSTEMS:
    run(["ftspec", "evaluate", "--profile", PROFILE, "--systems", systems,
         "--gpu-cost-per-hour", GPU_COST_PER_HOUR], cwd=CLONE_DIR, check=False)

step("6 . Combine the per-system runs into one matrix")
run(["ftspec", "combine-eval", "--profile", PROFILE,
     "--systems", ",".join(EVAL_SYSTEMS),
     "--gpu-cost-per-hour", GPU_COST_PER_HOUR], cwd=CLONE_DIR)

step("7 . Report + splice README")
run(["ftspec", "report", "--profile", PROFILE, "--gpu-cost-per-hour", GPU_COST_PER_HOUR,
     "--update-readme"], cwd=CLONE_DIR)

step("8 . Collect artifacts into /kaggle/working")
out = Path("/kaggle/working/ftspec_artifacts")
out.mkdir(exist_ok=True)
for name in ("reports", "manifests"):
    src = CLONE_DIR / "outputs" / PROFILE / name
    if src.exists():
        shutil.copytree(src, out / name, dirs_exist_ok=True)
readme_path = CLONE_DIR / "README.md"
if readme_path.exists():
    shutil.copy(readme_path, out / "README_with_results.md")
adapter = CLONE_DIR / "outputs" / PROFILE / "lora_adapter"
if adapter.exists():
    shutil.copytree(adapter, out / "lora_adapter", dirs_exist_ok=True)

readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
if "<!-- BENCHMARK:BEGIN -->" in readme:
    block = readme.split("<!-- BENCHMARK:BEGIN -->")[1].split("<!-- BENCHMARK:END -->")[0]
    (out / "measured_results.md").write_text(block.strip() + "\n", encoding="utf-8")

print("\nDONE -- artifacts in", out)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(out))
