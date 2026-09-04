"""
Encounter acuity protocol.

`acuity` plays the same role here that `priority` and `risk_band` play in the
other verticals: a field determined by a local protocol rather than by general
medical knowledge. Triage thresholds are institution-specific — one trust's
"urgent" is another's "emergent" — so a general-purpose model has no way to
infer them and must be told, in every single prompt, or taught once in weights.

The protocol below is a simplified composite of early-warning-score logic. It is
illustrative scaffolding for a benchmark, not clinical guidance, and the profile
says so in its compliance notes.
"""
from __future__ import annotations

from dataclasses import dataclass

ACUITY_RUBRIC_TEXT = """Assign encounter.acuity using this internal triage protocol, evaluated top-down (first match wins):

1. CRITICAL if any of:
   - oxygen saturation below 90%
   - systolic blood pressure below 90 mmHg
   - Glasgow Coma Scale below 13
   - respiratory rate above 30 per minute
2. EMERGENT if any of:
   - two or more observations are outside their reference range
   - heart rate above 120 or below 45 per minute
   - temperature at or above 39.5 C
   - a primary condition with severe presentation in an emergency encounter
3. URGENT if any of:
   - exactly one observation is outside its reference range
   - the encounter class is emergency
   - a primary condition is recorded as active and unconfirmed
4. ROUTINE otherwise."""


@dataclass
class VitalSigns:
    """Observations the protocol reads. None means not measured."""
    spo2: float | None = None
    systolic_bp: float | None = None
    gcs: float | None = None
    respiratory_rate: float | None = None
    heart_rate: float | None = None
    temperature_c: float | None = None


@dataclass
class TriageSignals:
    vitals: VitalSigns
    abnormal_count: int
    encounter_class: str
    primary_severity: str | None
    primary_unconfirmed: bool


def derive_acuity(s: TriageSignals) -> str:
    """Deterministic execution of ACUITY_RUBRIC_TEXT."""
    v = s.vitals

    # 1. CRITICAL
    if v.spo2 is not None and v.spo2 < 90:
        return "critical"
    if v.systolic_bp is not None and v.systolic_bp < 90:
        return "critical"
    if v.gcs is not None and v.gcs < 13:
        return "critical"
    if v.respiratory_rate is not None and v.respiratory_rate > 30:
        return "critical"

    # 2. EMERGENT
    if s.abnormal_count >= 2:
        return "emergent"
    if v.heart_rate is not None and (v.heart_rate > 120 or v.heart_rate < 45):
        return "emergent"
    if v.temperature_c is not None and v.temperature_c >= 39.5:
        return "emergent"
    if s.primary_severity == "severe" and s.encounter_class == "emergency":
        return "emergent"

    # 3. URGENT
    if s.abnormal_count == 1:
        return "urgent"
    if s.encounter_class == "emergency":
        return "urgent"
    if s.primary_unconfirmed:
        return "urgent"

    # 4. ROUTINE
    return "routine"
