"""
Phase 3, Stage A -- the no-training candidate search space.

Stage A never trains anything: it only varies which base model
`ftspec.evaluation.benchmark` loads, reusing its existing `base-schema` /
`base-rubric` / `base-constrained` systems as-is (three fixed points on the
prompt-variant x constrained-decoding grid; there is no fourth combination in
the catalogue today). A full sweep over every base model this platform might
ever support is not realistic on one free T4 -- this list is deliberately
small (2-3 entries) and meant to be edited per customer, not treated as
exhaustive.

Each candidate is evaluated by its own `run_stage_a_candidate()` call --
one CLI invocation, one OS process -- the same reason the current Colab
notebook runs each `evaluate` system as its own cell: a crash loading one
candidate model must never threaten the ones already measured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StageACandidate:
    candidate_id: str
    base_model: str
    label: str
    systems: str = "base-schema,base-rubric,base-constrained"


DEFAULT_STAGE_A_CANDIDATES: tuple[StageACandidate, ...] = (
    StageACandidate("stageA-qwen25-3b", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "Qwen2.5 3B"),
    StageACandidate("stageA-phi35-mini", "unsloth/Phi-3.5-mini-instruct-bnb-4bit", "Phi-3.5 mini"),
    StageACandidate("stageA-llama31-8b", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "Llama 3.1 8B"),
)


def slugify_model(base_model: str) -> str:
    """A stable, filesystem-safe candidate id derived from a model name, for
    ad-hoc base models not in DEFAULT_STAGE_A_CANDIDATES."""
    slug = re.sub(r"[^a-z0-9]+", "-", base_model.lower()).strip("-")
    return f"stageA-{slug}"
