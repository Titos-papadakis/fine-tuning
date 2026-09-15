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


# --- Stage B: LoRA sweep, training required -----------------------------
#
# For the Stage-A winner's base model only -- a full grid over (base model x
# LoRA config) is not realistic on one free T4, so this narrows to one base
# model before spending any GPU-hour on training. Each entry is a full QLoRA
# fine-tune (ftspec.training.train.run, unmodified), run as its own process
# for the same VRAM-isolation reason Stage A candidates are.

DEFAULT_LORA_GRID: tuple[tuple[int, int], ...] = ((8, 16), (16, 16), (32, 32))


@dataclass(frozen=True)
class StageBCandidate:
    candidate_id: str
    base_model: str
    lora_r: int
    lora_alpha: int


def stage_b_candidate_id(r: int, alpha: int) -> str:
    """Split out from stage_b_candidates() so ftplatform/learning/stats.py
    can compute the same id from a raw (r, alpha) pair -- e.g. from
    DEFAULT_LORA_GRID -- without duplicating the format string."""
    return f"stageB-r{r}-a{alpha}"


def stage_b_candidates(base_model: str,
                         grid: tuple[tuple[int, int], ...] = DEFAULT_LORA_GRID
                         ) -> list[StageBCandidate]:
    """One candidate per (r, alpha) pair in `grid`, all training the same
    `base_model` -- normally the Stage-A leaderboard's winning model."""
    return [StageBCandidate(stage_b_candidate_id(r, alpha), base_model, r, alpha)
            for r, alpha in grid]


# --- Stage C: inference-time toggles, no training -------------------------
#
# For the Stage-B winner's already-trained adapter only: does quantization
# (4bit vs fp16) or grammar-constrained decoding change the cost/quality
# tradeoff enough to matter? Purely inference-time -- each combo is one
# `ftspec.evaluation.benchmark.run()` call with its load_in_4bit/constrained
# overrides, no new backend, no GPU-hour spent on training.

DEFAULT_STAGE_C_COMBOS: tuple[tuple[str, bool], ...] = (
    ("4bit", True), ("fp16", False), ("fp16", True),
)  # ("4bit", False) is what Stage B already measured -- not repeated here.


@dataclass(frozen=True)
class StageCCandidate:
    candidate_id: str
    quantization: str  # "4bit" | "fp16"
    constrained: bool

    @property
    def load_in_4bit(self) -> bool:
        return self.quantization == "4bit"


def stage_c_candidate_id(prefix: str, quantization: str, constrained: bool) -> str:
    """Split out from stage_c_candidates() for the same reason as
    stage_b_candidate_id() above."""
    return f"{prefix}-{quantization}-{'constrained' if constrained else 'unconstrained'}"


def stage_c_candidates(prefix: str = "stageC",
                         combos: tuple[tuple[str, bool], ...] = DEFAULT_STAGE_C_COMBOS
                         ) -> list[StageCCandidate]:
    return [StageCCandidate(stage_c_candidate_id(prefix, q, c), q, c) for q, c in combos]
