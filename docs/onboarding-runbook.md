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

## Fast path (the whole thing in four commands)

```
ftplatform customer add acme --name "Acme Inc" --workload saas_support
ftplatform customer import acme tickets.csv [--label-with gpt-4o-mini]
ftplatform pipeline start acme
ftplatform pipeline drive acme --kaggle-owner <your-kaggle-user>
```

`customer import` validates every row against the schema and writes
`imports/corpus.jsonl`, which the baseline uses instead of synthetic data
(unlabeled rows go to `imports/to_label.jsonl` unless `--label-with` is
given; auto-labels get a spot-check sample in `imports/review_sample.jsonl`).
`pipeline drive` then runs baseline -> Stage A -> Stage B -> deploy on
Kaggle one job at a time, polling each until its result is merged back, and
stops at exactly one point for a first-time customer: it prints the winning
candidate and waits for `ftplatform approve acme`, after which the same
`pipeline drive` command finishes the deploy. It is safe to Ctrl-C and
re-run at any point; `pipeline status acme` shows where it is.

Baseline generations are cached per customer, so Stage A never re-runs the
base model the baseline already measured. After that, steps 5-9 below
(API key, billing, serve, review loop, backups) are unchanged -- in review,
`ftplatform review correct acme --request-id <id> --set issue.subcategory=...`
now fixes just the wrong field on top of the model's own output. Each month:

```
ftplatform report monthly acme --monthly-fee 3000
```

writes `customers/acme/reports/monthly-<YYYY-MM>.html` to send them.
`ftplatform customer delete acme --yes` removes a customer entirely (it
refuses while their Stripe subscription is still billable).

**A customer with their own record format** (not one of the shipped
workloads) needs no code, only their JSON Schema:

```
ftplatform customer add initech --name "Initech" --workload custom \
    --schema initech_schema.json --headline-field issue.priority \
    [--free-text-field summary] [--regime GDPR|HIPAA|PCI-DSS]
ftplatform customer import initech tickets.csv
```

then `pipeline start` / `pipeline drive` exactly as above. A custom
workload trains only on imported data, and HIPAA/PCI-DSS regimes block
hosted-API labeling and text capture automatically.

**Data handling, once per customer and then daily:**

```
ftplatform privacy set acme --retention-days 90 --redact email,phone,pan,ssn,iban
ftplatform privacy enforce            # schedule daily (cron / Task Scheduler)
ftplatform audit list acme            # the trail a security reviewer asks for
```

`docs/security-and-data-handling.md` is the document to send a customer's
security/procurement team. It lists Kaggle as a subprocessor for training.
`customer delete` removes the customer's `ftplatform-job-*` datasets and
notebooks on Kaggle as well (jobs pushed before this was recorded, such as
the original `pipeline-test` run, still need deleting by hand).

The numbered sections below are the same flow step by step, for when you
want to run or re-run one stage by hand.

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
