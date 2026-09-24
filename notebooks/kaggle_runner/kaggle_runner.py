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
# 50x the 200-example default -- deliberately large: this baseline is meant
# to be built once and reused by every future customer's Stage B (see
# ftplatform/learning/), so investing real GPU time in it now is worth more
# than a fast, mediocre first pass. The generator's signal space (continuous
# random amounts/dates, 5-digit order ids, shuffled/typo'd phrasing -- see
# profiles/saas_support/generate.py) comfortably supports far more than
# 10k unique, contract-valid documents; verified locally before spending any
# GPU time on it. n_eval is left at the original 150 on purpose -- that's
# what the 36.7% Record Exact Match baseline being chased here was measured
# on, so this run's eval stays directly comparable to it rather than moving
# both the training data and the measuring stick at once.
N_TRAIN = 10000
N_VAL = 500
N_EVAL = 150
# finetuned-constrained scored identically to finetuned on the last run (same
# 87.3%, McNemar p=1.0) and base-rubric identically to base-constrained on
# record exact (0.0%) -- ~0.5h of GPU each for no new information.
EVAL_SYSTEMS = ("finetuned", "base-constrained")
# Kaggle GPU sessions cap out around 12h. At the default num_train_epochs=3
# this run's own eval_loss already crashed from 1.78 -> 0.10 inside the first
# 5% of epoch 1 -- 3 full passes over 10k examples buys little beyond that,
# while costing ~11h of training alone. 2 epochs trades a bit of that
# diminishing-returns tail for real headroom under the session cap.
# Measured on that 2-epoch run: 526.9 min of training alone. 1 epoch (~4.4h)
# leaves real margin under both the session cap and a partly-spent weekly quota.
N_EPOCHS = 1
EFFECTIVE_BATCH_SIZE = 8  # per_device_train_batch_size(2) x gradient_accumulation_steps(4), see config.py
MAX_STEPS = -(-N_TRAIN // EFFECTIVE_BATCH_SIZE) * N_EPOCHS  # ceil-div, matches train.py's own step math


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
# --config no_merge.yaml: skip the default 16-bit merged-model save. It costs
# real GPU minutes to dequantize+write (~16GB for an 8B model) and this run
# doesn't use it -- evaluate below reads the LoRA adapter directly, and
# production serving is vLLM's LoRA loading, not a merged checkpoint. Left
# on, a prior run's output download dragged in that ~16GB for nothing.
started = time.time()
run(["ftspec", "train", "--profile", PROFILE, "--max-steps", str(MAX_STEPS),
     "--config", "notebooks/kaggle_runner/no_merge.yaml"], cwd=CLONE_DIR)
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
#
# Baseline systems (base-*) regenerate identical text whenever the eval set is
# unchanged, which it is across runs (fixed eval_seed) unless the generator
# itself changed -- --cache-dir skips those model loads (~1-2h each on a T4).
# A previous run's cache is picked up from any attached input that carries an
# eval_cache/ folder (e.g. this kernel's earlier output re-uploaded as a
# dataset); the cache key covers the exact eval inputs, so a stale one is
# simply ignored, never wrongly reused.
CACHE_DIR = CLONE_DIR / "outputs" / "eval_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
for prior in Path("/kaggle/input").glob("**/eval_cache"):
    if prior.is_dir():
        shutil.copytree(prior, CACHE_DIR, dirs_exist_ok=True)
        print(f"seeded eval cache from {prior}")
for systems in EVAL_SYSTEMS:
    run(["ftspec", "evaluate", "--profile", PROFILE, "--systems", systems,
         "--gpu-cost-per-hour", GPU_COST_PER_HOUR, "--cache-dir", str(CACHE_DIR)],
        cwd=CLONE_DIR, check=False)

step("6 . Combine the per-system runs into one matrix")
run(["ftspec", "combine-eval", "--profile", PROFILE,
     "--systems", ",".join(EVAL_SYSTEMS),
     "--gpu-cost-per-hour", GPU_COST_PER_HOUR], cwd=CLONE_DIR)

report_path = CLONE_DIR / "outputs" / PROFILE / "reports" / "benchmark_report.md"
if not report_path.exists():
    # Surface this loudly in the log rather than silently continuing --
    # a prior run on this exact pipeline finished "complete" with real
    # eval numbers in evaluate.json but no benchmark_report.md ever reached
    # the downloaded output, cause unconfirmed. Whatever the cause, the
    # rest of this script (adapter collection) still has value even if the
    # report itself is missing, so this warns rather than raises.
    print(f"\n!!! WARNING: {report_path} does not exist right after combine-eval -- "
          f"the report step may not have produced it. Continuing anyway so the "
          f"adapter/manifests below still get collected.", flush=True)
else:
    print(f"\nconfirmed: {report_path} exists ({report_path.stat().st_size} bytes)")

step("7 . Collect artifacts into /kaggle/working -- BEFORE the readme-splice step below, "
     "so a failure there can't lose what combine-eval already produced")
out = Path("/kaggle/working/ftspec_artifacts")
out.mkdir(exist_ok=True)
for name in ("reports", "manifests"):
    src = CLONE_DIR / "outputs" / PROFILE / name
    if src.exists():
        shutil.copytree(src, out / name, dirs_exist_ok=True)
adapter = CLONE_DIR / "outputs" / PROFILE / "lora_adapter"
if adapter.exists():
    shutil.copytree(adapter, out / "lora_adapter", dirs_exist_ok=True)
if CACHE_DIR.exists():
    shutil.copytree(CACHE_DIR, out / "eval_cache", dirs_exist_ok=True)
print("collected so far:")
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(out))

step("8 . Report + splice README (best-effort -- see step 7's note)")
# check=False: this touches README.md and reads back the just-written report;
# if it fails for any reason, the artifacts step 7 already collected are not
# lost, unlike when this ran before artifact collection.
run(["ftspec", "report", "--profile", PROFILE, "--gpu-cost-per-hour", GPU_COST_PER_HOUR,
     "--update-readme"], cwd=CLONE_DIR, check=False)

readme_path = CLONE_DIR / "README.md"
if readme_path.exists():
    shutil.copy(readme_path, out / "README_with_results.md")
    readme = readme_path.read_text(encoding="utf-8")
    if "<!-- BENCHMARK:BEGIN -->" in readme:
        block = readme.split("<!-- BENCHMARK:BEGIN -->")[1].split("<!-- BENCHMARK:END -->")[0]
        (out / "measured_results.md").write_text(block.strip() + "\n", encoding="utf-8")

step("9 . Strip bulk that isn't worth downloading, before Kaggle snapshots /kaggle/working")
# .git (a full clone, not needed once the code is installed), intermediate
# training checkpoints (full optimizer state per checkpoint -- only the
# final lora_adapter/ matters once training succeeded), and merged_model/
# (the ~16GB dequantized 16-bit merge -- step 4 now skips saving it via
# --config no_merge.yaml, so this is a safety net for a run without that
# flag, not the primary fix) all inflate this run's output for no benefit,
# and are leading suspects for why reports/ didn't make it into the
# original sanity-check run's download.
shutil.rmtree(CLONE_DIR / ".git", ignore_errors=True)
shutil.rmtree(CLONE_DIR / "outputs" / PROFILE / "checkpoints", ignore_errors=True)
shutil.rmtree(CLONE_DIR / "outputs" / PROFILE / "merged_model", ignore_errors=True)

print("\nDONE -- artifacts in", out)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(out))
