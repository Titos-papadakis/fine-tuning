"""
The TRL keyword shim.

`ftspec.training.train` runs on hardware CI does not have, so the parts of it
that can fail *before* a single step executes are tested here in isolation. A
TypeError from a renamed keyword costs a Colab session; this is cheap insurance
against the one failure mode that is fully knowable without a GPU.
"""
from __future__ import annotations

from ftspec.training.train import (
    SFT_CONFIG_ALIASES,
    TRAINER_ALIASES,
    adapt_kwargs,
    detect_family,
)

# --- stand-ins for the TRL versions we have to survive ------------------------

def old_sft_config(output_dir, max_seq_length=None, dataset_text_field=None):
    """TRL ~0.9: the spelling this module was written against."""


def new_sft_config(output_dir, max_length=None, dataset_text_field=None):
    """TRL ~0.20+: max_seq_length removed."""


def deprecating_sft_config(output_dir, max_seq_length=None, max_length=None,
                            dataset_text_field=None):
    """The window where both are accepted."""


def old_trainer(model, args=None, tokenizer=None):
    ...


def new_trainer(model, args=None, processing_class=None):
    ...


def permissive_trainer(model, **kwargs):
    ...


BASE = {"output_dir": "out", "max_seq_length": 2048, "dataset_text_field": "text"}


def test_old_name_kept_when_the_version_still_has_it():
    assert adapt_kwargs(old_sft_config, BASE, SFT_CONFIG_ALIASES) == BASE


def test_renamed_when_only_the_new_spelling_exists():
    got = adapt_kwargs(new_sft_config, BASE, SFT_CONFIG_ALIASES)
    assert got == {"output_dir": "out", "max_length": 2048, "dataset_text_field": "text"}
    assert "max_seq_length" not in got


def test_deprecation_window_prefers_the_original_spelling():
    # Both are accepted. Passing the one this code was written against is the
    # conservative choice: it cannot be a different argument that happens to
    # share a name.
    got = adapt_kwargs(deprecating_sft_config, BASE, SFT_CONFIG_ALIASES)
    assert got["max_seq_length"] == 2048
    assert "max_length" not in got


def test_unknown_keyword_is_dropped_rather_than_raising():
    got = adapt_kwargs(old_sft_config, {**BASE, "invented_in_2027": True},
                        SFT_CONFIG_ALIASES)
    assert "invented_in_2027" not in got
    assert got["output_dir"] == "out"          # the rest survives


def test_trainer_tokenizer_becomes_processing_class():
    kwargs = {"model": object(), "args": object(), "tokenizer": "TOK"}
    old = adapt_kwargs(old_trainer, kwargs, TRAINER_ALIASES)
    new = adapt_kwargs(new_trainer, kwargs, TRAINER_ALIASES)
    assert old["tokenizer"] == "TOK"
    assert new["processing_class"] == "TOK"
    assert "tokenizer" not in new


def test_var_keywords_pass_through_untouched():
    # A signature with **kwargs tells us nothing about what is accepted, so
    # filtering against it would silently drop valid arguments.
    kwargs = {"model": 1, "tokenizer": "TOK", "anything": 2}
    assert adapt_kwargs(permissive_trainer, kwargs, TRAINER_ALIASES) == kwargs


def test_uninspectable_target_passes_through():
    # Some C-implemented callables have no retrievable signature. Filtering
    # against nothing would drop every argument, so the shim stands aside.
    assert adapt_kwargs(object(), BASE, SFT_CONFIG_ALIASES) == BASE


def test_every_call_site_is_covered_by_the_shim():
    # Guards against a future edit adding a raw SFTConfig(...) / SFTTrainer(...)
    # call that bypasses the adapter.
    import inspect

    from ftspec.training import train as mod

    source = inspect.getsource(mod)
    for symbol in ("SFTConfig(", "SFTTrainer("):
        for line in source.splitlines():
            if symbol in line and "def " not in line:
                assert "adapt_kwargs" in line, f"unadapted {symbol} call: {line.strip()}"


def test_chat_template_family_detection():
    assert detect_family("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit") == "llama"
    assert detect_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"
    assert detect_family("mistralai/Mistral-7B-v0.3") == ""     # falls back safely
