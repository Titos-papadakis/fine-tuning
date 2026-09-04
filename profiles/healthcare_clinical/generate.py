"""
Synthetic clinical encounter notes.

Signal-first as everywhere else: sample the presentation and vitals, execute the
triage protocol to get `acuity`, then render a note in the clipped register
clinicians actually write in — abbreviations, sentence fragments, inconsistent
capitalisation — rather than clean prose that would make extraction artificially
easy.

HIPAA note: these notes are wholly invented and contain no identifiers by
construction. There is no name, MRN, address or date of birth to strip, because
the generator never produces one. Ages are capped at 89 per Safe Harbor. The
audit in profile.py asserts this rather than assuming it.
"""
from __future__ import annotations

import hashlib
import random

from ftspec.core.profile import Sample
from profiles.healthcare_clinical.rubric import (
    TriageSignals,
    VitalSigns,
    derive_acuity,
)

# (icd10, display, presenting complaint, typical severity pool)
PRESENTATIONS = [
    ("J45.909", "Unspecified asthma, uncomplicated", "wheeze and shortness of breath",
     ["mild", "moderate", "severe"]),
    ("E11.9", "Type 2 diabetes mellitus without complications", "poor glycaemic control",
     ["mild", "moderate"]),
    ("I10", "Essential (primary) hypertension", "elevated blood pressure on home readings",
     ["mild", "moderate"]),
    ("J18.9", "Pneumonia, unspecified organism", "productive cough and fever",
     ["moderate", "severe"]),
    ("N39.0", "Urinary tract infection, site not specified", "dysuria and frequency",
     ["mild", "moderate"]),
    ("R07.9", "Chest pain, unspecified", "central chest discomfort",
     ["moderate", "severe"]),
    ("K21.9", "Gastro-oesophageal reflux disease without oesophagitis", "epigastric burning",
     ["mild", "moderate"]),
    ("M54.5", "Low back pain", "mechanical lower back pain",
     ["mild", "moderate"]),
    ("G43.909", "Migraine, unspecified, not intractable", "unilateral headache with photophobia",
     ["moderate", "severe"]),
    ("A09", "Infectious gastroenteritis and colitis, unspecified", "vomiting and loose stools",
     ["mild", "moderate"]),
    ("I48.91", "Unspecified atrial fibrillation", "palpitations and irregular pulse",
     ["moderate", "severe"]),
    ("F41.1", "Generalized anxiety disorder", "persistent worry and poor sleep",
     ["mild", "moderate"]),
]

COMORBIDITIES = [
    ("E78.5", "Hyperlipidaemia, unspecified"),
    ("I10", "Essential (primary) hypertension"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
    ("N18.3", "Chronic kidney disease, stage 3"),
]

MEDICATIONS = [
    ("Salbutamol", "100 mcg", "PRN"), ("Metformin", "500 mg", "BD"),
    ("Amlodipine", "5 mg", "OD"), ("Atorvastatin", "20 mg", "ON"),
    ("Amoxicillin", "500 mg", "TDS"), ("Omeprazole", "20 mg", "OD"),
    ("Ramipril", "2.5 mg", "OD"), ("Sertraline", "50 mg", "OD"),
    ("Paracetamol", "1 g", "QDS"), ("Apixaban", "5 mg", "BD"),
]

ALLERGIES = ["penicillin", "sulfonamides", "latex", "peanuts", "ibuprofen", "codeine"]

# (code, unit, normal range, plausible abnormal range)
VITAL_SPECS = {
    "oxygen_saturation": ("%", (94, 99), (85, 93)),
    "systolic_blood_pressure": ("mmHg", (105, 135), (80, 100)),
    "heart_rate": ("bpm", (60, 95), (105, 145)),
    "respiratory_rate": ("breaths/min", (12, 18), (24, 36)),
    "temperature": ("C", (36.2, 37.3), (38.0, 40.2)),
}

VITAL_TO_FIELD = {
    "oxygen_saturation": "spo2",
    "systolic_blood_pressure": "systolic_bp",
    "heart_rate": "heart_rate",
    "respiratory_rate": "respiratory_rate",
    "temperature": "temperature_c",
}

NOTE_OPENERS = [
    "Pt presents with {complaint}.",
    "{age}{sex} attending with {complaint}.",
    "Seen today re: {complaint}.",
]

# Decoy numbers: figures a careless extractor might record as observations.
DECOY_LINES = [
    "Advised to return if sx worsen over next 48 hrs.",
    "Waiting time in dept was 35 mins.",
    "Pt reports similar episode approx 3 yrs ago.",
    "Weight stable, down 2 kg since last review.",
]


def sample_vitals(rng: random.Random, n_abnormal: int) -> list:
    """Pick a vitals panel with exactly `n_abnormal` values out of range."""
    chosen = rng.sample(list(VITAL_SPECS), k=rng.randint(3, 5))
    abnormal_idx = set(rng.sample(range(len(chosen)), k=min(n_abnormal, len(chosen))))

    panel = []
    for i, code in enumerate(chosen):
        unit, normal, abnormal = VITAL_SPECS[code]
        low, high = abnormal if i in abnormal_idx else normal
        value = round(rng.uniform(low, high), 1)
        panel.append({"code": code, "value": value, "unit": unit,
                       "abnormal": i in abnormal_idx})
    return panel


def make_sample(rng: random.Random) -> Sample:
    icd10, display, complaint, severity_pool = rng.choice(PRESENTATIONS)
    severity = rng.choice(severity_pool)
    encounter_class = rng.choice(
        ["ambulatory", "emergency", "inpatient", "virtual", "home_health"])

    # Skew abnormal counts so all four acuity bands are represented.
    n_abnormal = rng.choices([0, 1, 2, 3], weights=[0.42, 0.28, 0.20, 0.10])[0]
    observations = sample_vitals(rng, n_abnormal)

    vitals = VitalSigns()
    for obs in observations:
        field = VITAL_TO_FIELD.get(obs["code"])
        if field:
            setattr(vitals, field, obs["value"])

    primary_unconfirmed = rng.random() < 0.30
    acuity = derive_acuity(TriageSignals(
        vitals=vitals,
        abnormal_count=sum(1 for o in observations if o["abnormal"]),
        encounter_class=encounter_class,
        primary_severity=severity,
        primary_unconfirmed=primary_unconfirmed,
    ))

    if acuity == "critical":
        disposition = rng.choice(["admitted", "transferred"])
    elif acuity == "emergent":
        disposition = rng.choice(["admitted", "observation", "referred_specialist"])
    else:
        disposition = rng.choice(["discharged_home", "referred_specialist", "observation"])

    follow_up = rng.choice([None, 7, 14, 28, 42]) if disposition != "admitted" else None

    conditions = [{
        "icd10_code": icd10, "display": display,
        "clinical_status": "active",
        "verification_status": "provisional" if primary_unconfirmed else "confirmed",
        "is_primary": True,
    }]
    for code, name in rng.sample(COMORBIDITIES, k=rng.randint(0, 2)):
        if code == icd10:
            continue
        conditions.append({
            "icd10_code": code, "display": name,
            "clinical_status": rng.choice(["active", "inactive"]),
            "verification_status": "confirmed", "is_primary": False,
        })

    meds = []
    for name, dose, freq in rng.sample(MEDICATIONS, k=rng.randint(0, 3)):
        meds.append({"name": name, "dose": dose, "frequency": freq,
                      "status": rng.choices(["active", "stopped", "on_hold", "not_taken"],
                                             weights=[0.7, 0.15, 0.1, 0.05])[0]})

    allergies = rng.sample(ALLERGIES, k=rng.randint(0, 2))
    age = rng.choice([None] + list(range(18, 90)))
    sex = rng.choice(["male", "female", "other", "unknown"])
    is_pregnant = rng.choice([None, False, True]) if sex == "female" else None

    # --- render the note ---
    age_str = f"{age}yo " if age is not None else ""
    sex_str = {"male": "M", "female": "F", "other": "person", "unknown": "pt"}[sex]
    lines = [
        f"ENCOUNTER: {encounter_class.replace('_', ' ')}",
        rng.choice(NOTE_OPENERS).format(complaint=complaint, age=age_str, sex=sex_str),
        f"Impression: {display} ({icd10}), {severity}"
        + (", provisional pending ix" if primary_unconfirmed else ", confirmed") + ".",
    ]

    other_conditions = [c for c in conditions if not c["is_primary"]]
    if other_conditions:
        lines.append("PMH: " + "; ".join(
            f"{c['display']} ({c['icd10_code']}), {c['clinical_status']}"
            for c in other_conditions))

    lines.append("Obs: " + ", ".join(
        f"{o['code'].replace('_', ' ')} {o['value']} {o['unit']}"
        + (" (abnormal)" if o["abnormal"] else "") for o in observations))

    if is_pregnant is not None:
        lines.append(f"Pregnancy status: {'positive' if is_pregnant else 'negative'}.")
    if meds:
        lines.append("Meds: " + "; ".join(
            f"{m['name']} {m['dose']} {m['frequency']} [{m['status']}]" for m in meds))
    lines.append("Allergies: " + (", ".join(allergies) if allergies else "NKDA"))

    if rng.random() < 0.45:
        lines.append(rng.choice(DECOY_LINES))

    lines.append(f"Triage acuity: {acuity}.")
    lines.append(f"Plan: {disposition.replace('_', ' ')}"
                  + (f", review in {follow_up} days." if follow_up else "."))

    record = {
        "chief_complaint": complaint,
        "patient": {"age_years": age, "sex": sex, "is_pregnant": is_pregnant},
        "encounter": {
            "encounter_class": encounter_class,
            "acuity": acuity,
            "disposition": disposition,
            "follow_up_days": follow_up,
        },
        "conditions": conditions,
        "medications": meds,
        "observations": observations,
        "allergies": allergies,
    }

    return Sample(
        source_text="\n".join(lines),
        record=record,
        meta={"acuity": acuity, "abnormal_count": sum(1 for o in observations if o["abnormal"]),
               "encounter_class": encounter_class, "severity": severity},
    )


def generate(n: int, rng: random.Random) -> list:
    out, seen, attempts = [], set(), 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        sample = make_sample(rng)
        key = hashlib.sha256(sample.source_text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    if len(out) < n:
        raise RuntimeError(f"could only generate {len(out)}/{n} unique samples")
    return out
