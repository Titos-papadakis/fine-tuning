"""
Clinical contract: free-text encounter note -> ICD-10 / FHIR-aligned record.

HIPAA shapes this schema the same way PCI-DSS shapes the fintech one: by
omission. There is no field for a patient name, address, MRN, or date of birth,
because the Safe Harbor de-identification standard enumerates exactly those as
identifiers that must be stripped. `age_years` is capped at 89 for the same
reason — ages above that are themselves an identifier under Safe Harbor.

Field names follow FHIR resource vocabulary (Condition, MedicationStatement,
Observation, Encounter) so extracted records map onto a FHIR bundle without a
translation layer.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STRICT = ConfigDict(extra="forbid")

EncounterClass = Literal["ambulatory", "emergency", "inpatient", "virtual", "home_health"]
ClinicalStatus = Literal["active", "recurrence", "relapse", "inactive", "remission", "resolved"]
Verification = Literal["unconfirmed", "provisional", "differential", "confirmed", "refuted"]
Severity = Literal["mild", "moderate", "severe"]
Acuity = Literal["routine", "urgent", "emergent", "critical"]
Disposition = Literal[
    "discharged_home", "admitted", "referred_specialist",
    "transferred", "observation", "left_without_being_seen",
]


class Condition(BaseModel):
    """FHIR Condition."""
    model_config = STRICT

    icd10_code: str = Field(min_length=3, max_length=8,
                             description="ICD-10-CM code, e.g. 'E11.9'.")
    display: str = Field(min_length=1, description="Human-readable condition name.")
    clinical_status: ClinicalStatus
    verification_status: Verification
    is_primary: bool = Field(description="True for the primary encounter diagnosis.")


class MedicationStatement(BaseModel):
    """FHIR MedicationStatement."""
    model_config = STRICT

    name: str = Field(min_length=1)
    dose: str | None = Field(default=None, description="Dose as stated, e.g. '500 mg'.")
    frequency: str | None = Field(default=None, description="Frequency as stated, e.g. 'BD'.")
    status: Literal["active", "stopped", "on_hold", "not_taken"]


class Observation(BaseModel):
    """FHIR Observation — a vital sign or measured value."""
    model_config = STRICT

    code: str = Field(min_length=1, description="Observation name, e.g. 'systolic_blood_pressure'.")
    value: float
    unit: str = Field(min_length=1)
    abnormal: bool = Field(description="True if outside the stated reference range.")


class Patient(BaseModel):
    """De-identified subject. Safe Harbor: no name, MRN, address or date of birth."""
    model_config = STRICT

    age_years: int | None = Field(
        default=None, ge=0, le=89,
        description="Age in years. Safe Harbor caps this at 89; use null if older or unstated.")
    sex: Literal["male", "female", "other", "unknown"]
    is_pregnant: bool | None = None


class Encounter(BaseModel):
    """FHIR Encounter."""
    model_config = STRICT

    encounter_class: EncounterClass
    acuity: Acuity = Field(description="Assigned per the internal triage protocol.")
    disposition: Disposition
    follow_up_days: int | None = Field(
        default=None, ge=0, description="Days until planned follow-up, or null.")


class ClinicalRecord(BaseModel):
    """A structured clinical encounter record."""
    model_config = STRICT

    chief_complaint: str = Field(min_length=1, description="Presenting complaint in one phrase.")
    patient: Patient
    encounter: Encounter
    conditions: list[Condition]
    medications: list[MedicationStatement]
    observations: list[Observation]
    allergies: list[str]
