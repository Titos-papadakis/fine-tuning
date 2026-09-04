"""
Typed configuration.

The config file is validated as strictly as the model output is. A silent typo
in a YAML key -- `learning_rte: 2e-4` -- costs a GPU-hour and produces a model
trained with the default learning rate, with nothing in the logs to say so.
`extra="forbid"` turns that into an error before the first token is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    model_config = STRICT

    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = Field(default=2048, gt=0)
    load_in_4bit: bool = True
    dtype: str | None = None


class LoraConfig(BaseModel):
    model_config = STRICT

    r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42


class DataConfig(BaseModel):
    model_config = STRICT

    # Corpora live under data/<profile>/, so switching verticals never silently
    # trains on the previous one's data.
    root: str = "data"

    def dir_for(self, profile_name: str) -> str:
        return f"{self.root}/{profile_name}"


class TrainingConfig(BaseModel):
    model_config = STRICT

    num_train_epochs: float = Field(default=3, gt=0)
    per_device_train_batch_size: int = Field(default=2, gt=0)
    per_device_eval_batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    gradient_checkpointing: bool = True
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.01, ge=0)
    optim: str = "adamw_8bit"
    logging_steps: int = 5
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 20
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int = 20
    save_total_limit: int = 2
    seed: int = 42
    packing: bool = False
    report_to: str = "none"
    train_on_responses_only: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class MergeConfig(BaseModel):
    """Artefact names only.

    Full paths are composed by `Config` from the active profile, so training
    and evaluation cannot disagree about where the adapter lives. They used to:
    training wrote outputs/<profile>/lora_adapter while evaluation defaulted to
    outputs/lora_adapter, and the mismatch only surfaced after a GPU had already
    been paid for.
    """
    model_config = STRICT

    adapter_name: str = "lora_adapter"
    merged_name: str = "merged_model"
    checkpoints_name: str = "checkpoints"
    save_merged_16bit: bool = True


class PreflightConfig(BaseModel):
    """Limits enforced by `ftspec validate` before any GPU time is spent."""
    model_config = STRICT

    max_prompt_tokens: int = Field(default=1536, gt=0,
                                    description="Reject samples whose rendered prompt exceeds this.")
    max_total_tokens: int = Field(default=2048, gt=0,
                                   description="Must not exceed model.max_seq_length.")
    max_truncated_fraction: float = Field(default=0.0, ge=0, le=1,
                                           description="Fraction of samples allowed to exceed the limit.")


class Config(BaseModel):
    model_config = STRICT

    # Which vertical this config targets. Overridable with --profile.
    profile: str = "saas_support"
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)

    def fingerprint(self) -> str:
        """Stable hash of the full config, recorded in every run manifest."""
        blob = json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path

    # --- artefact layout -----------------------------------------------------
    # Every stage derives its paths from here. Adding a path anywhere else is
    # how train and evaluate drifted apart the first time.

    def data_dir(self, profile_name: str) -> Path:
        return self.resolve(self.data.dir_for(profile_name))

    def outputs_dir(self, profile_name: str) -> Path:
        return self.resolve(f"outputs/{profile_name}")

    def adapter_dir(self, profile_name: str) -> Path:
        return self.outputs_dir(profile_name) / self.merge.adapter_name

    def merged_dir(self, profile_name: str) -> Path:
        return self.outputs_dir(profile_name) / self.merge.merged_name

    def checkpoints_dir(self, profile_name: str) -> Path:
        return self.outputs_dir(profile_name) / self.merge.checkpoints_name

    def reports_dir(self, profile_name: str) -> Path:
        return self.outputs_dir(profile_name) / "reports"

    def manifests_dir(self, profile_name: str) -> Path:
        return self.outputs_dir(profile_name) / "manifests"


def load_config(path: Path | None = None) -> Config:
    """Load and validate a YAML config. Unknown keys are errors, not warnings."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
