# Enterprise SLM Specialization Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Titos-papadakis/fine-tuning/blob/main/notebooks/colab_runner.ipynb)
[![CI](https://github.com/Titos-papadakis/fine-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/Titos-papadakis/fine-tuning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Modular, privacy-preserving 8B model adapters for regulated and high-volume workflows.**

A domain-agnostic engine for turning an open-weight small language model into a specialist
that converts unstructured documents into strictly-typed records — with the schema guaranteed
at the decoder, the business policy compiled into the weights, and the data never leaving
your boundary.

Ships with three production-shaped verticals (SaaS support, card disputes, clinical notes)
and takes your own schema as a fourth.

```bash
ftspec prepare  --profile fintech_disputes
ftspec validate --profile fintech_disputes
ftspec train    --profile fintech_disputes
ftspec evaluate --profile fintech_disputes --systems finetuned,base-constrained
ftspec serve    --profile fintech_disputes --model outputs/fintech_disputes/merged_model
```

---

## The problem this actually solves

Most "structured output" projects are sold on a premise that stopped being true a while ago:

> *"Frontier models return broken JSON, so pay us to fix it."*

They don't, and you don't need to. `response_format: json_schema`, vLLM `guided_json`,
Outlines and llama.cpp GBNF all give **100% schema-valid output for free, with zero training**.
Any vendor whose headline metric is "% valid JSON" is charging you for a solved problem.

The real costs of running extraction on a hosted frontier model are three, and none of them
are fixed by better prompting:

| Cost | Why prompting cannot fix it |
|---|---|
| **The prompt tax** | Your schema and your business policy must be re-sent on **every single call, forever**. Measured on this repo's fintech profile: ~1,670 tokens per call — 167M tokens per 100k calls, paid indefinitely. |
| **Policy blindness** | `risk_band`, `priority`, `acuity` are set by *your* internal rubric. A general model cannot infer your thresholds. It can be told them, at the cost above, or it guesses — and guesses wrong systematically, not randomly. |
| **Egress** | For PCI-DSS or HIPAA data, sending records to a third-party API is not an expensive option. It is frequently not a lawful one. |

## What specialization actually buys

Three things, stated precisely — and one thing it does not:

1. **Prompt compression.** The schema and the rubric live in the weights. Measured on the
   `saas_support` corpus, the system prompt collapses from **1,253 tokens to 12** — a 7.3×
   cut in total prefill. Permanent, and it compounds with volume.
2. **Encoded judgement.** 200 examples teach a policy you cannot fit, or would rather not
   ship, in every prompt. This is the part no prompting technique replicates.
3. **Sovereignty.** The model runs inside your boundary. For regulated verticals this is not
   a cost argument, it is the *only* admissible architecture.

**It does not buy you JSON validity.** That is free, this repo proves it is free, and the
benchmark reports it as a baseline column rather than a result.

---

## Architecture

A domain-agnostic core, plus swappable vertical profiles. Adding a domain is a
schema-and-data exercise, not an engineering one.

```
┌──────────────────────── CORE ENGINE (domain-agnostic) ────────────────────────┐
│                                                                               │
│   Contract          Pydantic model  ─or─  raw JSON Schema                     │
│      │              one validator for training, eval and production           │
│      ├──────────────┬──────────────┬──────────────┬─────────────────┐         │
│      ▼              ▼              ▼              ▼                 ▼         │
│  prompt text   preflight gate   Outlines      vLLM guided      auto-derived   │
│                (pydantic +      grammar       decoding         scoring plan   │
│                 token length)                                                 │
│                                                                               │
│   prepare → audit → validate → train (QLoRA/Unsloth) → evaluate → serve       │
└───────────────────────────────────────────────────────────────────────────────┘
                                     ▲
                 ┌───────────────────┼───────────────────┬──────────────────┐
                 │                   │                   │                  │
         saas_support         fintech_disputes    healthcare_clinical     custom
         CRM records          Reg E / chargeback  ICD-10 / FHIR        your schema
         —                    PCI-DSS             HIPAA                 —
```

**One contract, four consumers.** The profile's schema is the single source of truth for
prompt text, the pre-training gate, the Outlines grammar, and the vLLM server. In a
hand-maintained setup these are four copies that drift: the prompt promises one shape, the
validator enforces another, the server guarantees a third. Here, changing a field changes all
four or the tests fail.

**Metrics are derived, not written.** The engine walks the schema and assigns each leaf a
comparison rule from its declared type — enums exact, numbers with a cent tolerance, arrays by
set F1, declared free text by token F1. Supply nothing but a JSON Schema and you still get
per-field accuracy, record-level exact match and bootstrap confidence intervals.

---

## Shipped profiles

| Profile | Task | Regime | External API | Policy field |
|---|---|---|---|---|
| `saas_support` | Support transcripts → CRM records | none | allowed | `issue.priority` |
| `fintech_disputes` | Cardholder intake → Reg E / chargeback record | **PCI-DSS** | **prohibited** | `assessment.risk_band` |
| `healthcare_clinical` | Clinical notes → ICD-10 / FHIR entities | **HIPAA** | **prohibited** | `encounter.acuity` |
| `custom` | Your JSON Schema, your corpus | declare your own | — | you choose |

```
$ ftspec profiles

  profile                regime     ext. API   fields  title
  ----------------------------------------------------------------------------
  fintech_disputes       PCI-DSS    PROHIBITED     22  Fintech Disputes …
  healthcare_clinical    HIPAA      PROHIBITED     12  Clinical Notes …
  saas_support           none       allowed        16  SaaS Support …
```

### Compliance is enforced, not annotated

`allows_external_api=False` is load-bearing code. Ask the benchmark to run a hosted baseline
against a PCI-DSS or HIPAA profile and it refuses:

```
$ ftspec evaluate --profile fintech_disputes --systems gpt4o-rubric

Profile is governed by PCI-DSS and declares allows_external_api=False. Sending source
records to a hosted API would move regulated data outside the compliance boundary.
Cardholder dispute intakes are CHD-adjacent. Under PCI-DSS v4.0 they must remain inside
the assessed cardholder data environment; a hosted inference API is a third-party service
provider requiring its own attestation.
Re-run with --acknowledge-egress only if your DPA and controls genuinely permit it; the
acknowledgement is recorded in the run manifest.

exit 1
```

The override exists, requires an explicit flag, and is written into the run manifest.

Schema design carries the same weight. The fintech contract has **no field capable of holding
a PAN** — `card_last4` is four digits, `card_bin` is six. A field that cannot exist cannot
leak. The clinical contract follows Safe Harbor by omission: no name, MRN, address or date of
birth exists, and `age_years` is capped at 89. Synthetic corpora use only reserved test BINs,
and an automated scanner asserts it, because this data is committed to a public repository.

---

## The gates

Each stage refuses to pass work that would silently corrupt the next one.

| Gate | Catches | Why it matters |
|---|---|---|
| `ftspec audit` | A label the source document never states | Unpredictable by construction. It caps the achievable score at an arbitrary ceiling and makes the benchmark unfalsifiable. |
| `ftspec audit` | Eval documents present in training | Invalidates every number downstream. |
| `ftspec audit` | Identifiers in a regulated corpus | A "synthetic" card number that passes Luhn against a live BIN is a liability in a public repo. |
| `ftspec validate` | Sample exceeding `max_seq_length` | The trainer does not reject it — it **truncates**. You train the model to emit unterminated JSON and find out hours later. |
| `ftspec validate` | Contract drift, bad role order, duplicates | Training signal that is quietly meaningless. |
| `ftspec train` | A corpus that failed `validate` | Refuses before loading torch. No GPU is allocated to a doomed run. |
| config load | An unknown YAML key | `learning_rte: 2e-4` costs a GPU-hour and trains at the default rate with nothing in the logs. |

Every stage writes a run manifest (`outputs/<profile>/manifests/`) with git commit,
dirty-tree flag, config fingerprint, package versions, GPU and metrics — so "why do these
numbers differ from the ones I just reproduced" is an answerable question.

---

## Benchmark matrix

The comparison set is deliberately hostile to this repo's own thesis. Beating "GPT-4o with no
schema in the prompt" would prove nothing, because nobody deploys that.

| System | What it is |
|---|---|
| `base-schema` | 8B base, schema in prompt, no training |
| `base-rubric` | 8B base, schema + policy in prompt, no training |
| `base-constrained` | 8B base + grammar-constrained decoding — **100% valid, free** |
| `gpt4o-schema` | GPT-4o, structured outputs |
| `gpt4o-rubric` | GPT-4o, schema + policy — the strongest prompted baseline |
| `gpt4o-mini-rubric` | GPT-4o-mini — the cheapest credible baseline |
| `finetuned` | The specialized 8B, ~20-token prompt |

Reported per system: **schema adherence**, **record-level and per-field accuracy** with 95%
bootstrap confidence intervals, **p50 / p99 latency**, and **cost per 100k calls**, plus
McNemar's exact test for paired significance and a confusion matrix over the policy field.

The `base-constrained` row exists to make the honest point in public: it reaches ~100%
adherence with no training whatsoever, and its *accuracy* is what tells you whether
specialization was worth it.

---

## Projected economics

> **These are modeled, not measured.** Token counts are real — measured from the committed
> `saas_support` eval corpus. Throughput and TTFT are engineering estimates for an 8B model in
> 4-bit on a T4, and they are the numbers you should be most sceptical of, because they move
> with batch size more than anything else. The arithmetic is printed below so you can check it
> rather than trust it. **[Run the notebook](https://colab.research.google.com/github/Titos-papadakis/fine-tuning/blob/main/notebooks/colab_runner.ipynb)
> and this section gets replaced with numbers from your own hardware.**

Measured prompt sizes (`saas_support`, mean over the 150-document held-out split):

| | System prompt | + document | Output |
|---|---:|---:|---:|
| Specialized 8B | **12 tok** | 196 tok | 162 tok |
| Prompted baseline (schema + policy) | **1,253 tok** | 1,437 tok | 162 tok |

That is a **7.3× prefill reduction on every call**, and it is the mechanism behind both the
latency and the cost figures below.

### Cost per 1,000 requests

| System | Input | Output | Cost / 1k | vs GPT-4o |
|---|---:|---:|---:|---:|
| GPT-4o | 1,437 tok @ $2.50/1M | 162 tok @ $10/1M | **$5.21** | — |
| GPT-4o-mini | 1,437 tok @ $0.15/1M | 162 tok @ $0.60/1M | **$0.31** | −94% |
| Self-hosted T4, single stream | — | ~18 tok/s | **~$0.90** | **−83%** |
| Self-hosted T4, batched (~8 concurrent) | — | ~110 tok/s aggregate | **~$0.17** | **−97%** |

<details>
<summary>The arithmetic</summary>

```
GPT-4o        (1437/1e6 × $2.50) + (162/1e6 × $10.00)  = $0.00521/req  → $5.21/1k
GPT-4o-mini   (1437/1e6 × $0.15) + (162/1e6 × $0.60)   = $0.00031/req  → $0.31/1k

Self-hosted   sec/req = TTFT + output_tokens / decode_rate
              single:  0.25s + 162/18   =  9.25 s/req → 389 req/hr
              batched: 0.25s + 162/110  =  1.72 s/req → 2,093 req/hr
              cost/req = $0.35/hr ÷ req/hr
              single:  $0.00090/req → $0.90/1k
              batched: $0.00017/req → $0.17/1k
```
T4 on-demand assumed at **$0.35/hour**. Substitute your provider's rate.
</details>

### Cost & latency breakdown

**60% of the GPT-4o bill is prompt tax.** Of the $5.21 per 1,000 requests, $3.10 is
re-transmitting a schema and a triage policy that never change. You pay it on every call,
forever, and no prompting technique removes it — only moving those tokens into weights does.

**Latency is prefill-bound, so the short prompt wins twice.** At ~196 prompt tokens the
specialized model has roughly 7× less prefill than a prompted baseline at ~1,437. Projected
TTFT on a T4 is **~200–300 ms** against **~1.1–2.0 s** for the prompted baseline on the same
hardware. Sub-200 ms is realistic on an L4 or A10G, not reliably on a T4 — the notebook
measures it rather than assuming it.

**Where the honest limit is.** GPT-4o-mini at $0.31/1k **undercuts an unbatched self-hosted
T4**. If your only argument is unit cost and your volume is low, mini wins and you should use
it. Self-hosting wins on cost when you batch (~$0.17/1k, a 45× gap to GPT-4o-mini's
per-request rate at scale), and `ftspec tco` computes the break-even volume for your numbers.

**For regulated verticals the cost argument is secondary.** Cardholder and clinical data
cannot be sent to a hosted API without an attestation or a BAA. There, the comparison table
has empty columns by law, and a self-hosted specialized model is the only admissible design —
which is why the benchmark refuses to fill them in.

### Measured results

<!-- BENCHMARK:BEGIN -->
*Not yet populated.* Run
[the Colab notebook](https://colab.research.google.com/github/Titos-papadakis/fine-tuning/blob/main/notebooks/colab_runner.ipynb);
it writes a `measured_results.md` formatted to drop straight into this block, alongside the
full matrix, every raw prediction, and a run manifest recording the git commit, config
fingerprint, package versions and GPU.

Pre-baked numbers in a vendor README are marketing. These are reproducible or they are
nothing — and McNemar's exact test is wired in so a 4-point gap is not reported as a win.
<!-- BENCHMARK:END -->

---

## Quickstart

**No GPU?** [Open the notebook in Colab](https://colab.research.google.com/github/Titos-papadakis/fine-tuning/blob/main/notebooks/colab_runner.ipynb)
— it clones, installs, trains and benchmarks end to end on a free T4 in ~30 minutes, then
hands you a zip with the matrix, every raw prediction and a README-ready results block.

```bash
pip install -e ".[train,constrained]"      # add ",serve" for vLLM, ",openai" for baselines

ftspec profiles                            # what's available
ftspec prepare  --profile saas_support     # 200 train / 40 val / 150 held-out
ftspec audit    --profile saas_support     # groundedness + leakage + compliance
ftspec validate --profile saas_support     # contract + token length, before any GPU
ftspec train    --profile saas_support     # QLoRA on a single 16GB T4
ftspec evaluate --profile saas_support --systems finetuned,base-constrained,base-rubric
ftspec serve    --profile saas_support --model outputs/saas_support/merged_model
```

Training fits free-tier Colab: 4-bit NF4 base, LoRA via Unsloth, Unsloth gradient
checkpointing, 8-bit AdamW, per-device batch 2 × accumulation 4, and completion-only loss
masking so the model is never trained to reproduce its own prompt.

### Measure your own prompt tax

Nothing here is specific to this repo's corpora. Point it at your own prompt log and it
reports how much of your bill re-sends content that never changes:

```bash
pip install tiktoken
ftspec prompt-audit your_prompts.jsonl --model gpt-4o --calls-per-month 250000
```

Runs entirely locally and transmits nothing; the report contains aggregate numbers only, with
no prompt text unless you pass `--include-sample`. Accepts OpenAI `messages` logs,
`{"system", "user"}` pairs, or flat `{"prompt"}` rows.

### Bring your own schema

```bash
ftspec schema  --profile custom --schema mine.json --show-fields   # derived metrics
ftspec prepare --profile custom --schema mine.json --from-jsonl mydata.jsonl
ftspec train   --profile custom --schema mine.json
```

Generation is deliberately unavailable for `custom`: the engine has no idea what a plausible
document in an unseen domain looks like, and inventing one would train your model on fiction.
Bring your corpus; the engine brings everything else. For a proprietary vertical with its own
rubric and generator, register a `Profile` subclass under the `ftspec.profiles` entry-point
group and keep it in a private package — nothing about your domain has to be contributed
upstream.

---

## Production serving

`ftspec serve` is OpenAI wire-compatible, so an existing integration switches by changing a
base URL. The difference from plain `vllm serve` is where the guarantee lives.

vLLM's own server does guided decoding **only if the client asks**, by sending
`extra_body={"guided_json": ...}` on every call. That makes your contract a client-side
convention: one integration that forgets, one SDK that strips unknown fields, one engineer
copying a curl example, and you are parsing hopeful JSON in production again.

This server inverts it. The schema comes from the loaded profile and is applied to **every**
request, server-side. A client cannot opt out, cannot forget, and cannot drift — because the
contract is not something the client sends.

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")
response = client.chat.completions.create(
    model="ftspec-saas_support",
    messages=[{"role": "user", "content": transcript}],
)
# schema-valid by construction; the client never mentioned a schema
```

Endpoints: `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/v1/schema`
(the enforced contract, for client codegen), `/v1/profile`, `/health`, `/stats`.

---

## Repository layout

```
ftspec/                        core engine — no domain knowledge
  core/{contract,fields,profile,registry,redaction}.py
  data/{build,audit,validate}.py
  training/train.py            QLoRA via Unsloth
  evaluation/{metrics,benchmark,tco}.py
  inference/constrained.py     Outlines grammar (v0 and v1 APIs)
  serving/serve.py             OpenAI-compatible vLLM, schema enforced server-side
  reporting.py                 splices measured results into this README
  prompt_audit.py              measures the prompt tax on any prompt log, locally
  cli.py                       prepare · audit · validate · train · evaluate · report · serve

profiles/                      plug-and-play verticals
  saas_support/ fintech_disputes/ healthcare_clinical/
    {schema,rubric,scenarios,generate,profile}.py

notebooks/
  colab_runner.ipynb           free-T4 end-to-end run; generated by _build_notebook.py
sales/                         B2B kit: one-pager, audit playbook, objections, outbound
configs/                       typed YAML, validated with extra="forbid"
data/<profile>/                generated corpora (committed; CI asserts reproducibility)
outputs/<profile>/             checkpoints, reports, manifests (gitignored)
examples/client.py             drop-in OpenAI SDK client
tests/                         91 tests, CPU-only
```

---

## Verification status

Honest accounting of what has been executed versus what needs hardware:

| | Status |
|---|---|
| Core engine, contracts, metric derivation, registry, redaction | ✅ 91 tests passing, CPU-only |
| All three profiles: generation, balance, groundedness, compliance | ✅ verified end-to-end |
| Pipeline gates (`prepare`/`audit`/`validate`), incl. negative tests | ✅ each defect class injected and caught |
| Serving layer: OpenAI shape, SSE streaming, schema enforcement | ✅ tested against a stubbed engine |
| Compliance egress gate | ✅ verified to refuse, and to record the override |
| Colab notebook: builds, all code cells parse, nbformat valid | ✅ structurally verified |
| `ftspec train` on a real GPU | ⏳ requires CUDA hardware |
| Benchmark matrix numbers | ⏳ produced by the notebook or `ftspec evaluate` |
| Projected economics table | ⚠️ modeled — token counts measured, throughput estimated |

The entire CPU-side pipeline runs in CI on every commit, including a check that the committed
corpora still match what the committed seed regenerates — so the benchmark stays reproducible
from source rather than from a stale artefact.

---

## License

MIT.
