"""
Drop-in client example.

The point of this file is how boring it is. The server is wire-compatible with
`/v1/chat/completions`, so an existing integration switches to a self-hosted,
schema-guaranteed model by changing a base URL — no new SDK, no vendor client,
no `guided_json` plumbing in application code.

Note what is absent: nothing here mentions the JSON schema. The client does not
send it, does not know it, and cannot forget it. The guarantee is the server's.

    ftspec serve --model outputs/merged_model
    python examples/client.py
"""
from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("FTSPEC_BASE_URL", "http://localhost:8000/v1")

TRANSCRIPT = """Agent: Hi, this is Maria from NexaCRM support, how can I help?
Customer: I was charged twice for my NexaCRM subscription, order ORD-55210. My plan is only $29.00 a month but $412.50 came out in total.
Agent: I can see two identical charges on your account, let me pull up the transaction log.
Customer: We're an Enterprise customer, this affects our whole team.
Customer: This is the third time I'm contacting you about this, by the way.
Agent: I'm noting this under internal case CASE-4471.
Customer: If this isn't fixed today I'm cancelling our subscription.
Agent: I've escalated this to our billing team.
Agent: I'll follow up with you on 2026-03-14.
Customer: I want this escalated, properly."""


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key="not-required")

    response = client.chat.completions.create(
        model=os.environ.get("FTSPEC_MODEL", "ftspec-saas_support"),
        messages=[{"role": "user", "content": TRANSCRIPT}],
        temperature=0,
    )
    content = response.choices[0].message.content
    print(json.dumps(json.loads(content), indent=2, ensure_ascii=False))

    usage = response.usage
    print(f"\nprompt_tokens={usage.prompt_tokens}  "
           f"completion_tokens={usage.completion_tokens}")

    # Optional: the same contract the server enforces, available client-side for
    # anyone who wants typed access rather than a dict.
    try:
        from ftspec.schemas import SupportTicket
        ticket = SupportTicket.model_validate_json(content, strict=True)
        print(f"\ntyped access -> priority={ticket.issue.priority!r} "
               f"tier={ticket.customer.tier!r} amount={ticket.extracted_entities.amount}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
