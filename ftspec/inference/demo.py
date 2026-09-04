"""Demo helpers for `ftspec infer` — kept out of the inference path itself."""
from __future__ import annotations

import json
from pathlib import Path

from ftspec.core.profile import Profile


def example_document(profile: Profile, data_dir: Path) -> str:
    """Pick a document from the held-out split.

    Drawn from eval rather than train so an ad-hoc demo is never accidentally
    showing the model a document it memorised.
    """
    eval_file = data_dir / "eval.jsonl"
    if eval_file.exists():
        for line in eval_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line)["messages"][1]["content"]
    raise FileNotFoundError(
        f"No eval corpus at {eval_file}. Run `ftspec prepare --profile {profile.name}` "
        "first, or pass --text / --file."
    )


def prompt_overhead(extractor, profile: Profile) -> str:
    """The prompt tax, measured in the tokenizer the model actually uses."""
    tokenizer = getattr(extractor, "_tokenizer", None)
    if tokenizer is None:
        return ""
    prompts = profile.prompts
    ours = len(tokenizer(prompts.short)["input_ids"])
    theirs = len(tokenizer(prompts.schema_rubric)["input_ids"])
    if theirs <= ours:
        return ""
    return (f"prompt overhead: {ours} system tokens here vs {theirs} for a prompted "
             f"baseline (+{theirs - ours} on every call, forever)")
