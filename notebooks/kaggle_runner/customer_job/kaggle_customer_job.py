"""
Runs inside a Kaggle kernel to execute exactly one customer's job.

Pushed by ftplatform.remote.run_on_kaggle.start_job_on_kaggle(), which
mounts a private, per-job dataset (built by ftplatform.remote.packet) as
this kernel's `dataset_sources` entry -- that dataset carries the one
customer's data and the one pending job, never anything else. Code (ftspec
+ ftplatform) comes from the public GitHub repo via a plain git clone;
customer data never touches git, only this private per-job dataset.

Mirrors notebooks/kaggle_runner/kaggle_runner.py's install steps exactly --
see that file for why each pinned version is pinned. The difference is what
runs after install: that script drives the fixed `saas_support` demo
profile directly via `ftspec` CLI calls; this one drives whatever job the
mounted packet names, via `ftplatform.jobs.worker.run_one()`, so it works
for any customer and any job kind (baseline / stage_a / stage_b / stage_c /
deploy / retrain_cycle) without the kernel code itself needing to know which.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Titos-papadakis/fine-tuning.git"
CLONE_DIR = Path("/kaggle/working/ftspec")
RESULT_DIR = Path("/kaggle/working/result_packet")


def run(cmd, cwd=None, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def step(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)


step("0 . Locate the mounted job packet")
# Found by its job.db, not by assuming a mount depth: Kaggle moved dataset
# mounts from /kaggle/input/<slug>/ to /kaggle/input/datasets/<owner>/<slug>/,
# and a real run died picking /kaggle/input/datasets itself as the packet.
db_files = sorted(Path("/kaggle/input").rglob("job.db"))
if len(db_files) != 1:
    raise SystemExit(f"expected exactly one job.db under /kaggle/input, found "
                     f"{[str(p) for p in db_files]}")
packet_dir = db_files[0].parent
# `datasets create --dir-mode zip` uploads customer_data/ as a zip; if Kaggle
# left it zipped, unpack it into a writable copy (inputs are read-only).
zipped = packet_dir / "customer_data.zip"
if zipped.exists() and not (packet_dir / "customer_data").exists():
    staging = Path("/tmp/job_packet")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy(db_files[0], staging / "job.db")
    shutil.unpack_archive(str(zipped), str(staging / "customer_data"))
    nested = staging / "customer_data" / "customer_data"
    if nested.is_dir():
        shutil.move(str(nested), str(staging / "unnested"))
        shutil.rmtree(staging / "customer_data")
        (staging / "unnested").rename(staging / "customer_data")
    packet_dir = staging
print("packet:", packet_dir)
for p in sorted(packet_dir.iterdir()):
    print(" ", p.name)

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

probe = subprocess.run(
    [sys.executable, "-c", "import ftspec, ftplatform, unsloth, trl, outlines"],
    cwd=CLONE_DIR, capture_output=True, text=True)
print(probe.stdout, probe.stderr)
if probe.returncode != 0:
    raise SystemExit("environment install failed -- see the pip/import output above")
print("environment OK")

# Imported only after `pip install -e .` above -- ftplatform isn't installed
# yet when this script starts.
sys.path.insert(0, str(CLONE_DIR))
from ftplatform import db  # noqa: E402
from ftplatform.jobs import worker  # noqa: E402
from ftplatform.remote import packet  # noqa: E402

step("3 . Apply the packet -- recreate customers.db and customers/<id>/ here")
customer_id = packet.apply_packet(packet_dir, repo_root=CLONE_DIR)
print("customer:", customer_id)

step("4 . Run the one pending job")
conn = db.connect(CLONE_DIR / "customers.db")
job = worker.run_one(conn)
if job is None:
    raise SystemExit("packet carried no pending job -- nothing to run")
print("job finished:", job["id"], job["status"])
if job["status"] == "failed":
    print("job error:", job["result"])
conn.close()

step("5 . Collect the result packet for download")
packet.collect_result_packet(customer_id, RESULT_DIR, repo_root=CLONE_DIR)
print("DONE -- result packet at", RESULT_DIR)
for p in sorted(RESULT_DIR.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(RESULT_DIR))
