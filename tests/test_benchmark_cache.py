"""
benchmark.run(cache_dir=...) -- reusing a baseline system's generations on an
unchanged eval set instead of reloading the model.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ftspec.core.registry import load_profile
from ftspec.evaluation import benchmark as B

PROFILE = load_profile("saas_support")


@pytest.fixture
def eval_file(tmp_path):
    src = Path(__file__).resolve().parent.parent / "data" / "saas_support" / "eval.jsonl"
    rows = src.read_text(encoding="utf-8").splitlines()[:12]
    dst = tmp_path / "eval.jsonl"
    dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dst


class CountingLocal:
    """Replaces run_local: echoes gold for every record, counting calls per system."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, spec, records, profile, max_new_tokens):
        self.calls.append(spec.name)
        plan = profile.scoring_plan()
        result = B.RunResult(spec=spec)
        for i, r in enumerate(records):
            gold = json.loads(r["messages"][2]["content"])
            result.raw_outputs.append(json.dumps(gold, ensure_ascii=False))
            result.valid_flags.append(1.0)
            result.latencies_ms.append(100.0 + i)
            result.prompt_tokens.append(40)
            result.output_tokens.append(60)
            result.scores.append(B.M.score_record(gold, gold, plan))
        return result


def _run(eval_file, tmp_path, systems, cache_dir):
    return B.run(PROFILE, eval_file, tmp_path / "reports", systems=systems,
                 finetuned_model="adapter", cache_dir=cache_dir)


def test_second_run_reuses_baseline_generations(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"

    first = _run(eval_file, tmp_path, "base-schema,base-constrained", cache)
    second = _run(eval_file, tmp_path, "base-schema,base-constrained", cache)

    assert fake.calls == ["base-schema", "base-constrained"]
    assert second["cached_systems"] == ["base-schema", "base-constrained"]
    assert second["systems"] == first["systems"]


def test_finetuned_is_never_cached(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"

    _run(eval_file, tmp_path, "finetuned", cache)
    second = _run(eval_file, tmp_path, "finetuned", cache)

    assert fake.calls == ["finetuned", "finetuned"]
    assert second["cached_systems"] == []


def test_changed_eval_inputs_miss_the_cache(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"
    _run(eval_file, tmp_path, "base-schema", cache)

    rows = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines()]
    rows[0]["messages"][1]["content"] += " (edited)"
    eval_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _run(eval_file, tmp_path, "base-schema", cache)

    assert fake.calls == ["base-schema", "base-schema"]


def test_a_different_base_model_misses_the_cache(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"
    _run(eval_file, tmp_path, "base-schema", cache)

    B.run(PROFILE, eval_file, tmp_path / "reports", systems="base-schema",
          base_model="some/other-model", cache_dir=cache)

    assert fake.calls == ["base-schema", "base-schema"]


def test_cached_generations_are_rescored_against_current_gold(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"
    first = _run(eval_file, tmp_path, "base-schema", cache)
    assert first["systems"]["base-schema"]["record_exact_pct"] == 100.0

    # Same inputs (same cache key), but one gold label changed: the cached
    # text must now score as wrong on that record, not reuse the old score.
    rows = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines()]
    gold = json.loads(rows[0]["messages"][2]["content"])
    gold["issue"]["priority"] = "urgent" if gold["issue"]["priority"] != "urgent" else "low"
    rows[0]["messages"][2]["content"] = json.dumps(gold)
    eval_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    second = _run(eval_file, tmp_path, "base-schema", cache)

    assert second["cached_systems"] == ["base-schema"]
    assert second["systems"]["base-schema"]["record_exact_pct"] < 100.0


def test_no_cache_dir_writes_nothing(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)

    _run(eval_file, tmp_path, "base-schema", None)
    _run(eval_file, tmp_path, "base-schema", None)

    assert fake.calls == ["base-schema", "base-schema"]
    assert not (tmp_path / "cache").exists()


def test_a_truncated_cache_file_is_ignored(monkeypatch, eval_file, tmp_path):
    fake = CountingLocal()
    monkeypatch.setattr(B, "run_local", fake)
    cache = tmp_path / "cache"
    _run(eval_file, tmp_path, "base-schema", cache)
    (entry,) = cache.glob("*.jsonl")
    entry.write_text(entry.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    _run(eval_file, tmp_path, "base-schema", cache)

    assert fake.calls == ["base-schema", "base-schema"]
    shutil.rmtree(cache)
