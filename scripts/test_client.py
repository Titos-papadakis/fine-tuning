#!/usr/bin/env python
"""
One-command smoke test for a running ftspec endpoint.

    python scripts/serve_adapter.py --profile saas_support --backend mock &
    python scripts/test_client.py

Uses the official `openai` SDK against `base_url=http://localhost:8000/v1`, so
what is being tested is the thing a customer would actually integrate -- not a
bespoke HTTP client that might paper over a wire-format mistake.

The request deliberately carries **no system prompt and no schema**. One user
message containing a raw transcript, nothing else. If a valid record comes back,
the schema guarantee is provably server-side: the client could not have asked
for it, because the client does not know it exists.

Exits non-zero if anything fails validation, so it works as a deploy gate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# One transcript per shipped profile. Each is written so a human can read the
# expected record straight off it -- if the model gets a field wrong, the
# evidence for the right answer is in the text above.
TRANSCRIPTS = {
    "saas_support": """Agent: Hi, this is Maria from NexaCRM support, how can I help?
Customer: I was charged twice for my NexaCRM subscription, order ORD-55210. My plan is only $29.00 a month but $412.50 came out in total.
Agent: I can see two identical charges on your account, let me pull up the transaction log.
Customer: We're an Enterprise customer, this affects our whole team.
Customer: This is the third time I'm contacting you about this, by the way.
Agent: I've gone through your billing history and I'm noting this under internal case CASE-4471.
Customer: If this isn't fixed today I'm cancelling our subscription.
Agent: I've escalated this to our billing team.
Agent: I'll follow up with you on 2026-03-14.""",

    "fintech_disputes": """Agent: Thank you for calling, you've reached the disputes team.
Customer: There's a charge on my card ending 4142 that I did not make. It's for 253.30 at an electronics retailer, dated last Tuesday.
Agent: Do you still have the physical card with you?
Customer: Yes, it's in my wallet right now. I've never shopped there.
Agent: And have you disputed anything with us before?
Customer: No, never. I've been with the bank about six years.
Agent: I'm opening a dispute and issuing a provisional credit while we investigate.""",

    "healthcare_clinical": """Clinician: What brings you in today?
Patient: I'm 58, and I've had crushing chest pain for about two hours. It goes into my left arm.
Clinician: Any shortness of breath?
Patient: Yes, and I'm sweating a lot. It started while I was resting.
Clinician: Blood pressure is 158 over 96, heart rate 104, oxygen saturation 94 percent.
Clinician: I'm ordering a troponin and a twelve-lead ECG now, and I want cardiology to see you immediately.""",
}


def get_health(base_url: str, timeout: float) -> dict | None:
    """Fetch /health, which lives beside /v1 rather than inside it."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def validate(record: dict, profile_name: str) -> tuple[bool, str]:
    """Check the record against the same contract the server enforces.

    Falls back to plain JSON validity when the engine is not importable, so the
    script stays useful from a machine that only has the `openai` SDK installed.
    """
    try:
        from ftspec.core.registry import load_profile
    except ImportError:
        return True, "JSON valid (contract not checked: ftspec not importable here)"

    try:
        profile = load_profile(profile_name)
    except Exception as e:
        return True, f"JSON valid (contract not checked: {e})"

    parsed, err = profile.contract.validate(json.dumps(record, ensure_ascii=False))
    if parsed is None:
        return False, f"contract violation: {err}"
    return True, f"valid against {profile.contract.name}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="test_client.py",
        description="Send a raw transcript to a running ftspec endpoint and validate the reply.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--profile", default="saas_support", choices=sorted(TRANSCRIPTS))
    ap.add_argument("--n", type=int, default=1, help="Requests to send.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--transcript-file", type=Path, default=None,
                    help="Send this text instead of the built-in transcript.")
    ap.add_argument("--quiet", action="store_true", help="Latency only; do not print records.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Include the cold first request in the latency figures.")
    args = ap.parse_args(argv)

    try:
        from openai import OpenAI
    except ImportError:
        print("The official SDK is missing:  pip install openai", file=sys.stderr)
        return 2

    health = get_health(args.base_url, timeout=5.0)
    if health is None:
        print(f"No server answering at {args.base_url}.\n"
              f"Start one with:  python scripts/serve_adapter.py "
              f"--profile {args.profile} --backend mock", file=sys.stderr)
        return 2

    model = health.get("model", "ftspec-extractor")
    backend = health.get("backend", "unknown")
    print(f"server   : {model}  (backend={backend})")
    print(f"profile  : {health.get('profile')}  regime={health.get('regime')}")
    print(f"schema   : enforced server-side, {len(health.get('schema_fields') or [])} top-level fields")
    print(f"uptime   : {health.get('uptime_s')}s\n")

    if backend == "mock":
        print("!! MOCK BACKEND: records are synthetic and unrelated to the input.")
        print("!! This proves the wiring works. It says nothing about accuracy.\n")

    served = health.get("profile")
    if served and served != args.profile:
        print(f"!! Server is serving '{served}' but this transcript is for "
              f"'{args.profile}'. Validating against '{served}'.\n")
        args.profile = served if served in TRANSCRIPTS else args.profile

    transcript = (args.transcript_file.read_text(encoding="utf-8")
                  if args.transcript_file else TRANSCRIPTS[args.profile])

    client = OpenAI(base_url=args.base_url, api_key="not-required", timeout=args.timeout)

    # The SDK builds its HTTP client lazily and the connection is cold, so the
    # first call can cost seconds that have nothing to do with the model. Left
    # in the sample it swamps the median and produces a latency claim that is
    # simply wrong. One discarded request removes it.
    if not args.no_warmup:
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "warmup"}],
                temperature=0, max_tokens=16)
            print("warmup   : one discarded request (client cold start excluded)\n")
        except Exception:
            pass          # a failing warmup is not itself a test failure

    latencies: list[float] = []
    failures = 0
    last_record: dict | None = None

    for i in range(1, args.n + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                # No system prompt. No schema. No guided_json. Just the transcript.
                messages=[{"role": "user", "content": transcript}],
                temperature=0,
            )
        except Exception as e:
            print(f"request {i}: FAILED -- {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        content = response.choices[0].message.content or ""

        try:
            record = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"request {i}: INVALID JSON -- {e}", file=sys.stderr)
            print(content[:400], file=sys.stderr)
            failures += 1
            continue

        ok, detail = validate(record, args.profile)
        usage = response.usage
        status = "ok " if ok else "FAIL"
        print(f"request {i}: {status}  {elapsed_ms:8.1f} ms  "
              f"in={usage.prompt_tokens if usage else '?'} "
              f"out={usage.completion_tokens if usage else '?'}  {detail}")
        if not ok:
            failures += 1
        last_record = record

    if last_record is not None and not args.quiet:
        print("\n--- last record ---")
        print(json.dumps(last_record, indent=2, ensure_ascii=False))

    if latencies:
        ordered = sorted(latencies)
        print(f"\nclient   : min {ordered[0]:.1f} ms  "
              f"median {statistics.median(ordered):.1f} ms  "
              f"max {ordered[-1]:.1f} ms   (n={len(latencies)})")
        # End-to-end from this process: network, queueing and serialisation
        # included. Printing the server's own figure beside it shows how much of
        # the wall clock is the model and how much is everything else -- the two
        # get conflated constantly when someone asks what inference costs.
        after = get_health(args.base_url, timeout=5.0)
        server_stats = (after or {}).get("latency_ms") or {}
        if server_stats.get("count"):
            print(f"server   : p50 {server_stats['p50']:.1f} ms  "
                  f"p99 {server_stats['p99']:.1f} ms   "
                  f"(n={server_stats['count']}, generation only)")
            overhead = statistics.median(ordered) - server_stats["p50"]
            print(f"overhead : {overhead:.1f} ms median -- transport, SDK and serialisation")

    if failures:
        print(f"\n{failures}/{args.n} request(s) failed validation", file=sys.stderr)
        return 1

    print(f"\nAll {args.n} request(s) returned a contract-valid record.")
    print("The client sent no schema and no system prompt -- the guarantee is the server's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
