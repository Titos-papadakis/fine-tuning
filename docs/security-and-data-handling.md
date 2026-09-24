# Security & Data Handling

This document describes how customer data is stored, isolated, processed and
deleted. Every statement below reflects what the software does today; where
a control is not yet in place, it says so under
[Not yet in place](#not-yet-in-place).

## 1. What data we hold

| Data | Where | Kept until |
|---|---|---|
| Tickets you provide for training (imported corpus) and the labels on them | Your customer directory | You are deleted as a customer |
| Train / validation / evaluation splits built from them | Your customer directory | You are deleted as a customer |
| Your trained model adapters and evaluation reports | Your customer directory | You are deleted as a customer |
| Captured production traffic (requests/responses), if enabled | Your customer directory | Your retention window (default **90 days**) |
| Corrections you approve during review | Your customer directory, then folded into training data | You are deleted as a customer |
| Account metadata: name, workload, job status, deployments, API key **hashes**, monthly usage counters, billing status | Platform database | You are deleted as a customer |
| Audit log (see §6) | Platform database | Retained, including after deletion |

The platform database stores metadata and pointers only. It never holds
ticket text, labels, model weights or plaintext API keys.

## 2. Tenant isolation

Each customer has exactly one directory subtree (`customers/<id>/`) and their
own rows in the platform database. Every code path reaches a customer's data
through a single component that resolves paths from the customer id alone,
so no operation can address another customer's directory.

- **Remote training jobs** are packaged per customer: a job packet contains
  that customer's rows and files only.
- **Shared serving** (several customers on one GPU) selects the customer's
  model from their authenticated API key, not from anything the caller sends.
  Authentication is on by default.
- **Production capture** on a shared server is written only against the
  customer the API key resolves to. A request with no resolved identity is
  not logged at all, rather than guessed. A dedicated server serves, and
  logs for, exactly one customer.
- An automated test runs every customer-scoped operation (import, the full
  training/deployment pipeline, capture, retention, reporting, job
  packaging) for one customer. It then verifies that another customer's
  files are byte-for-byte unchanged, that their database rows are unchanged,
  and that none of their text appears in the job packet.

This is application-level isolation, not OS- or hardware-level. Customers
that need a dedicated environment should see
[Not yet in place](#not-yet-in-place).

## 3. What is shared across customers

Only which *technique* worked: for example, "LoRA rank 16 on base model X
scored 0.91 on the support-ticket workload". This is used to try the most
promising configurations first for the next customer. It never includes
ticket text, labels, schemas or model weights. One customer's trained model
is never used as a starting point for another customer.

## 4. Processing locations and subprocessors

| Subprocessor | Used for | What it receives | When |
|---|---|---|---|
| **Kaggle (Google)** | GPU compute for training and evaluation | Your training/evaluation data and job packet, uploaded as a **private** dataset under our account; your trained adapter | Every training/evaluation job run on Kaggle |
| **OpenAI** | Optional: auto-labeling unlabeled tickets; optional GPT-4o comparison benchmark | The ticket text being labeled or benchmarked | Only when explicitly enabled per job. **Refused automatically** for workloads declared HIPAA or PCI-DSS |
| **Stripe** | Subscription billing | Billing contact email, company name | When billing is set up |
| Hosting provider (to be confirmed at contract) | Serving your production endpoint | Production requests and responses | While your endpoint is live |

For customers who cannot allow training data on Kaggle, training can run on
dedicated compute instead. This must be agreed before onboarding.

## 5. Production traffic: redaction and retention

- **Redaction before storage.** Card numbers (Luhn-validated), SSNs, email
  addresses, phone numbers and IBANs are replaced with `[REDACTED:<type>]`
  *before* anything is written to disk. The rule set is configurable per
  customer. Order ids, amounts and dates are not affected.
- **Regulated workloads** (declared HIPAA or PCI-DSS) never have text
  captured at all. Only metadata is kept (validity, latency, token counts).
- **Retention.** Captured traffic older than the customer's window (default
  90 days) is deleted by the daily retention job. Rows whose age cannot be
  established are deleted as well.

## 6. Access control and audit

- **API keys** are shown once, at creation, and only their SHA-256 hash is
  stored. Keys can be revoked immediately, and a revoked key never
  authenticates again. Per-customer rate limits are available.
- **Deployment gate.** A new model reaches production only if it clears
  three checks: a schema-validity floor, a paired statistical test showing
  it is not worse than the current model, and a score check. A customer's
  first deployment also requires explicit human approval.
- **Audit log.** An append-only log records who did what and when: customer
  creation/deletion, data imports, approvals, deployments and rollbacks, API
  key creation/revocation, billing changes, privacy-policy changes,
  retention runs, reviewed corrections and retraining. It records
  identifiers and counts only, never ticket text or keys. It is kept after a
  customer is deleted, so the deletion itself remains provable.

## 7. Deletion

On request, a single operation removes the customer's database rows (except
the audit log), their entire customer directory (data, models, logs,
reports) and local job staging files. It is refused while a Stripe
subscription is still billable, so that deleting the records cannot hide
live charges.

**Manual step today:** the private Kaggle datasets and job notebooks created
for the customer's training jobs (§4) are removed by hand as part of the same
deletion request. They are not yet deleted automatically.

## 8. Backups

The platform database can be snapshotted consistently while running.
Customer directories are backed up separately by the hosting environment.

## Not yet in place

Stated plainly, so nothing here is assumed:

- No SOC 2 / ISO 27001 certification and no third-party penetration test yet.
- No SSO/SAML for customer access. Access is by API key.
- Encryption at rest is provided by the host's disk encryption, not by the
  application. TLS is terminated by the hosting environment in front of the
  service.
- Isolation is application-level (§2), not per-customer infrastructure.
  Dedicated infrastructure is available by agreement.
- Kaggle-side artefacts are deleted manually (§7).
