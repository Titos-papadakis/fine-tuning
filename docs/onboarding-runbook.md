# Bringing on a paying customer — operator runbook

Every step below is a real `ftplatform` command that exists today. This is
the actual path from "someone agreed to pay $3k/month" to "their endpoint is
live and billed" — not a plan, a checklist for the operator (you) to follow
by hand for each new customer.

**What this deliberately does not cover:** provisioning the always-on GPU
host itself. Every step here runs on the same free Kaggle/Colab GPU quota
used throughout development — that's enough to build and validate a
customer's model, but not to serve it 24/7. The one paid step (a small
GPU instance to run `ftplatform serve` continuously) is left for the moment
there's an actual signed customer, per the standing "no cost before a
customer" rule. Everything up to and including that step is free and can
be rehearsed today.

## 0. Prerequisites (once, not per customer)

- `pip install -e ".[train,serve,constrained,billing]"` — the full stack.
- A Stripe account, with one flat-fee recurring Price created by hand in the
  dashboard (this codebase never creates Prices, only subscribes customers
  to one you already made — see `ftplatform/billing/stripe_billing.py`).
- `customers.db` exists the moment any `ftplatform` command runs (auto-
  created by `ftplatform.db.connect()`).

## 1. Register the customer

```
ftplatform customer add acme --name "Acme Inc" --workload saas_support
```

Isolated from every other customer from this point on — see
`ftplatform/customers/context.py`. Nothing shared but GPU capacity if you
later move them onto `serve-shared`.

## 2. Measure their baseline (what prompting alone costs/gets them)

```
ftplatform baseline run acme --n-train 10000 --n-val 500 --n-eval 150
```

Runs on a free Kaggle/Colab GPU. Writes `memory/deployed.json` — the number
a fine-tuned candidate must beat. This is also the raw material for a
customer-specific version of the pitch artifact (see the published pitch
deck) — swap in *their* numbers before sending it to them.

## 3. Search for the best technique

```
ftplatform candidate list-presets --workload saas_support
ftplatform candidate stage-a acme            # no training -- prompt/model-size sweep
ftplatform candidate stage-b acme            # LoRA grid on the Stage-A winner
ftplatform candidate stage-c acme            # constrained decoding on/off, etc.
ftplatform leaderboard build acme
```

Each stage is its own GPU-time investment — Stage A is free (no training),
Stage B/C cost real (still free-quota) GPU hours. Read the leaderboard
before committing to Stage B on a customer with a small enough budget that
Stage A alone might already be good enough.

## 4. Check, then deploy

```
ftplatform deploy check acme --candidate-id <id-from-leaderboard>
ftplatform deploy run acme --candidate-id <id>
```

`check` shows the decision without acting on it — three gates (schema
adherence floor, McNemar significance vs. current production, composite
score not regressing). Failing any gate is not an error: production stays
exactly as it was. `deploy run` only writes to `production/` if every gate
passes.

## 5. Issue an API key

```
ftplatform api-key create acme --label "acme prod"
```

Shown once. Store it in whatever secret manager the eventual paid host
uses — it never touches this repo or `customers.db` in plaintext (only its
sha256 hash is stored, see `ftplatform/auth/keys.py`).

## 6. Link and subscribe them to billing

```
ftplatform usage stripe-link acme --email billing@acme.example --name "Acme Inc"
ftplatform usage stripe-subscribe acme --price price_XXXXXXXXXXXX
ftplatform usage stripe-status acme
```

This is the actual "start charging them" moment. `stripe-status --refresh`
re-checks Stripe live (a payment failure, a cancellation) rather than
trusting a stale local record.

## 7. Serve it

Locally / on free quota, to validate end-to-end before the real host exists:

```
ftplatform serve acme --port 8000
```

Auth is on by default (`--require-auth`, refuses to start without an
API key already issued in step 5) and usage metering + production-traffic
capture attach automatically. **On the eventual paid host**, this is the
same command — nothing about it changes, only where it runs. Multiple
customers sharing one GPU use `ftplatform serve-shared <workload>` instead
(discovers every active customer's production adapter for that workload
and serves the largest base-model-compatible group from one engine).

## 8. Ongoing: the self-improving loop

```
ftplatform review scan acme         # heuristic pre-filter over captured production traffic
ftplatform review pending acme      # what's actually waiting on a human
ftplatform review correct <id> ...  # a human's corrected label
ftplatform selfimprove run acme     # fold corrections back into the corpus
```

Then repeat step 3-4 (candidate search + deploy check/run) on the updated
corpus. `deploy run`'s gates are what prevent a bad retrain from ever
reaching production — a regression simply never promotes.

## 9. Back up the platform database

```
ftplatform db backup
```

A consistent snapshot (safe even with `serve` writing to the same db
concurrently — see `ftplatform/db.py::backup()`) of every customer's
metadata, deployments, hashed API keys, usage counters, and billing status.
Never includes raw customer text, corpora, or model weights (those live
under `customers/<id>/` on disk). Run this on a schedule once there's
anything worth losing — cron or a scheduled task, not built here yet.

## What's still manual / out of scope here

- **The paid host itself.** A `Dockerfile` exists at the repo root
  (`docker run --gpus all ...`) so this is a deploy command away, not a
  from-scratch build, once you're ready to pay for it.
- **DNS / TLS / a real domain** in front of the serve port.
- **Real legal terms (DPA, SLA)** — the pricing/inclusions list in the
  pitch deck is a sales pitch, not a contract. Get real terms reviewed
  before a customer signs anything.
- **Automatic backup scheduling** — `db backup` exists; wiring it to cron
  does not, yet.
