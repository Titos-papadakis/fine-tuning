"""
Compliance scanners.

Used in two places, both enforcement rather than reporting:

  1. Corpus gate. Regulated profiles must not ship synthetic data that contains
     anything resembling a real identifier. A "synthetic" card number that
     happens to pass a Luhn check against a live BIN range is a liability
     sitting in a public git repository.
  2. Output gate. A model must not echo a raw identifier into a field that the
     schema says holds a masked value. `last4` means four digits, and a model
     that helpfully returns all sixteen has produced a PCI-DSS incident, not a
     more informative record.

The detectors are deliberately conservative: they over-flag rather than
under-flag, because the cost of a missed identifier is regulatory and the cost
of a false positive is a developer reading one line of output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Reserved test ranges. Real-world card BINs never begin 4111 11 / 5555 55 etc.,
# so synthetic corpora that use only these can never collide with a live PAN.
SAFE_TEST_PANS = (
    "4111111111111111", "4012888888881881",
    "5555555555554444", "5105105105105100",
    "378282246310005", "371449635398431",
)

PATTERNS = {
    # 13-19 digits, optionally separated, is PAN-shaped regardless of formatting.
    "pan": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    # No leading \b: there is no word boundary between a space and "(", so a
    # \b-anchored pattern silently misses the very common "(555) 123-4567" form.
    # Digit lookaround instead, which anchors correctly in every format.
    "phone": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)[ -]?|\d{3}[ -])\d{3}[ -]?\d{4}(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "nhs_mrn": re.compile(r"\b(?:MRN|NHS)[ :#-]*\d{6,10}\b", re.IGNORECASE),
    "dob": re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
}


@dataclass(frozen=True)
class Finding:
    rule: str
    value: str
    context: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.value!r} in {self.context}"


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def scan(text: str, rules: tuple, context: str = "") -> list:
    """Return findings for the named rules. Unknown rule names are ignored."""
    findings: list = []
    for rule in rules:
        pattern = PATTERNS.get(rule)
        if pattern is None:
            continue
        for match in pattern.finditer(text or ""):
            value = match.group(0)

            if rule == "pan":
                digits = re.sub(r"\D", "", value)
                # Only Luhn-valid numbers are plausible PANs; anything else is an
                # order id or a long reference and flagging it would be noise.
                if not (13 <= len(digits) <= 19 and luhn_ok(digits)):
                    continue
                if digits in SAFE_TEST_PANS:
                    continue

            findings.append(Finding(rule=rule, value=value, context=context))
    return findings


def scan_record(record: dict, rules: tuple, prefix: str = "") -> list:
    """Recursively scan every string in a structured record."""
    findings: list = []
    if isinstance(record, dict):
        for key, value in record.items():
            findings += scan_record(value, rules, f"{prefix}.{key}" if prefix else key)
    elif isinstance(record, list):
        for i, value in enumerate(record):
            findings += scan_record(value, rules, f"{prefix}[{i}]")
    elif isinstance(record, str):
        findings += scan(record, rules, context=prefix or "<root>")
    return findings


def assert_masked(value: str, expected_len: int, context: str) -> list:
    """A masked field must contain exactly the digits it promises, and no more."""
    if value is None:
        return []
    digits = re.sub(r"\D", "", str(value))
    if len(digits) > expected_len:
        return [Finding(rule="unmasked_identifier",
                         value=str(value),
                         context=f"{context} (expected {expected_len} digits, got {len(digits)})")]
    return []
