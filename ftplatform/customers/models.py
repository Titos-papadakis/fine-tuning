"""Typed records for the platform's customer store."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

# A customer id doubles as a filesystem directory name (customers/<id>/...)
# and a Config path fragment, so it's restricted to a short, safe character
# set -- this validates a system boundary (customer-supplied input becoming
# a path), not a hypothetical.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    workload: str
    status: str = "active"
    created_at: str = ""

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(
                "customer id must be 1-32 lowercase letters/digits/_/-, "
                "starting with a letter or digit")
        return v
