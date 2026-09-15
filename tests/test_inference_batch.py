"""
ftspec/inference/batch.py: only load_documents() is tested here -- run_batch()
needs unsloth/torch (real GPU-adjacent deps not present in CI), consistent
with how ftspec.evaluation.benchmark.run_local() is never unit-tested
directly either, only the orchestration around it.
"""
from __future__ import annotations

import json

import pytest

from ftspec.inference.batch import load_documents


def _write(path, lines):
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def test_loads_a_raw_string_per_line(tmp_path):
    path = _write(tmp_path / "docs.jsonl", ["hello", "world"])
    assert load_documents(path) == ["hello", "world"]


def test_loads_the_input_key_shape(tmp_path):
    path = _write(tmp_path / "docs.jsonl", [{"input": "doc one"}, {"input": "doc two"}])
    assert load_documents(path) == ["doc one", "doc two"]


def test_loads_the_engine_chat_format_shape(tmp_path):
    """Same shape train/eval.jsonl already use, so a batch job can point
    directly at an existing eval file."""
    row = {"messages": [{"role": "system", "content": "sys"},
                          {"role": "user", "content": "the actual document"},
                          {"role": "assistant", "content": "{}"}]}
    path = _write(tmp_path / "docs.jsonl", [row])
    assert load_documents(path) == ["the actual document"]


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('"a"\n\n   \n"b"\n', encoding="utf-8")
    assert load_documents(path) == ["a", "b"]


def test_a_row_with_no_recognizable_shape_raises_a_clear_error(tmp_path):
    path = _write(tmp_path / "docs.jsonl", [{"unexpected": "shape"}])
    with pytest.raises(ValueError, match="no document text found"):
        load_documents(path)


def test_mixed_shapes_across_lines_are_all_supported(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps("raw string") + "\n" +
        json.dumps({"input": "input-shaped"}) + "\n" +
        json.dumps({"messages": [{"role": "system", "content": "s"},
                                    {"role": "user", "content": "chat-shaped"}]}) + "\n",
        encoding="utf-8")
    assert load_documents(path) == ["raw string", "input-shaped", "chat-shaped"]
