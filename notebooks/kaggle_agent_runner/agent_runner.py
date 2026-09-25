"""
Kaggle run for the agent workload: the ABCD support conversations turned into
"what does the agent do next" steps (reply / call a tool / wait), trained and
benchmarked with the same ftspec pipeline as notebooks/kaggle_runner/.

The only differences from that runner:
  * the corpus is built from ABCD (downloaded here, MIT licence) by
    ftplatform.agent.abcd instead of generated synthetically;
  * the profile is `custom` with the step schema that build writes -- its
    "x-ftspec" key makes `tool` the headline field and `reply` free text;
  * the split is by conversation (meta.group), so no conversation is in both
    train and eval.

The prompted baseline (base-constrained) is shown the full tool catalog and
the per-intent action policy in its schema prompt -- the same information the
fine-tuned model learned from data -- so the comparison is fair.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/Titos-papadakis/fine-tuning.git"
CLONE_DIR = Path("/kaggle/working/ftspec")
ABCD_SRC = Path("/kaggle/working/abcd_src")
ABCD_OUT = Path("/kaggle/working/abcd_agent")
ABCD_BASE = "https://github.com/asappresearch/abcd/raw/master/data/"
ABCD_FILES = ("abcd_v1.1.json.gz", "kb.json", "ontology.json", "guidelines.json")
SCHEMA = ABCD_OUT / "schema.json"
PROFILE_ARGS = ["--profile", "custom", "--schema", str(SCHEMA)]
OUT_KEY = "custom"                 # ftspec writes a custom profile's outputs under outputs/custom
GPU_COST_PER_HOUR = "0.35"
# One decision point per conversation. 8,500 conversations -> 250 eval, 150
# val, ~8,100 train: sized so training plus two eval systems (the prompted
# one carries a ~2.7k-token schema+policy prompt) fit well inside one ~12h
# Kaggle session, on a 1-epoch budget that measured 271 min for 10k examples.
MAX_ROWS = 8500
N_VAL = 150
N_EVAL = 250
EVAL_SYSTEMS = ("finetuned", "base-constrained")
N_EPOCHS = 1
EFFECTIVE_BATCH_SIZE = 8  # per_device_train_batch_size(2) x gradient_accumulation_steps(4)


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
run([sys.executable, "-m", "pip", "install", "-q",
     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "trl<0.12", "peft", "accelerate", "bitsandbytes"])
run([sys.executable, "-m", "pip", "install", "-q", "outlines", "jsonschema"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers<5"])
run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=CLONE_DIR)
probe = subprocess.run([sys.executable, "-c", "import ftspec, ftplatform, unsloth, trl, outlines"],
                        cwd=CLONE_DIR, capture_output=True, text=True)
print(probe.stdout, probe.stderr)
if probe.returncode != 0:
    raise SystemExit("environment install failed -- see the pip/import output above")

step("3 . Build the ABCD agent corpus and gate it")
ABCD_SRC.mkdir(parents=True, exist_ok=True)
for name in ABCD_FILES:
    run(["curl", "-sSfL", "-o", str(ABCD_SRC / name), ABCD_BASE + name])
run([sys.executable, "-c",
     "from ftplatform.agent import abcd; "
     f"print(abcd.build({str(ABCD_SRC)!r}, {str(ABCD_OUT)!r}, max_rows={MAX_ROWS}))"],
    cwd=CLONE_DIR)
run(["ftspec", "prepare", *PROFILE_ARGS, "--from-jsonl", str(ABCD_OUT / "corpus.jsonl"),
     "--n-val", str(N_VAL), "--n-eval", str(N_EVAL)], cwd=CLONE_DIR)
run(["ftspec", "audit", *PROFILE_ARGS], cwd=CLONE_DIR, check=False)
run(["ftspec", "validate", *PROFILE_ARGS], cwd=CLONE_DIR)
# data/custom/, exactly: the repo also ships data/<vertical>/train.jsonl for
# the built-in profiles, and counting one of those would size training wrong.
DATA_DIR = CLONE_DIR / "data" / OUT_KEY
n_train = sum(1 for _ in open(DATA_DIR / "train.jsonl", encoding="utf-8"))
if n_train < 5000:
    raise SystemExit(f"only {n_train} train rows in {DATA_DIR} -- expected ~8,100")
max_steps = -(-n_train // EFFECTIVE_BATCH_SIZE) * N_EPOCHS
print(f"train rows {n_train} ({DATA_DIR}) -> max_steps {max_steps}")

step("4 . Train")
started = time.time()
run(["ftspec", "train", *PROFILE_ARGS, "--max-steps", str(max_steps),
     "--config", "notebooks/kaggle_runner/no_merge.yaml"], cwd=CLONE_DIR)
print(f"training wall clock: {(time.time() - started) / 60:.1f} min")

step("5 . Benchmark -- one process per system")
CACHE_DIR = CLONE_DIR / "outputs" / "eval_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
for prior in Path("/kaggle/input").glob("**/eval_cache"):
    if prior.is_dir():
        shutil.copytree(prior, CACHE_DIR, dirs_exist_ok=True)
for systems in EVAL_SYSTEMS:
    started = time.time()
    run(["ftspec", "evaluate", *PROFILE_ARGS, "--systems", systems,
         "--gpu-cost-per-hour", GPU_COST_PER_HOUR, "--cache-dir", str(CACHE_DIR)],
        cwd=CLONE_DIR, check=False)
    print(f"{systems} eval wall clock: {(time.time() - started) / 60:.1f} min")

step("6 . Combine")
run(["ftspec", "combine-eval", *PROFILE_ARGS, "--systems", ",".join(EVAL_SYSTEMS),
     "--gpu-cost-per-hour", GPU_COST_PER_HOUR], cwd=CLONE_DIR, check=False)

step("7 . Collect artifacts")
out = Path("/kaggle/working/ftspec_artifacts")
out.mkdir(exist_ok=True)
for name in ("reports", "manifests", "lora_adapter"):
    src = CLONE_DIR / "outputs" / OUT_KEY / name
    if src.exists():
        shutil.copytree(src, out / name, dirs_exist_ok=True)
if CACHE_DIR.exists():
    shutil.copytree(CACHE_DIR, out / "eval_cache", dirs_exist_ok=True)
if (DATA_DIR / "eval.jsonl").exists():
    shutil.copy(DATA_DIR / "eval.jsonl", out / "eval.jsonl")
for name in ("tools.json", "schema.json"):
    shutil.copy(ABCD_OUT / name, out / name)

step("8 . Strip bulk")
shutil.rmtree(CLONE_DIR / ".git", ignore_errors=True)
shutil.rmtree(CLONE_DIR / "outputs" / OUT_KEY / "checkpoints", ignore_errors=True)
shutil.rmtree(CLONE_DIR / "outputs" / OUT_KEY / "merged_model", ignore_errors=True)
shutil.rmtree(ABCD_SRC, ignore_errors=True)

print("\nDONE -- artifacts in", out)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(out))
