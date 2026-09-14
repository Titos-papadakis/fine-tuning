"""
Milestone 3, Stage B: the LoRA training sweep.

`run_stage_b_candidate()` is tested by monkeypatching
`ftspec.training.train.run` and `ftspec.evaluation.benchmark.run`, the same
technique used for Stage A -- no GPU/model needed to verify the
orchestration, isolation, and (this milestone's actual discovery) that the
customer's shared `Config` is never mutated by a candidate's LoRA sweep.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.candidates import runner
from ftplatform.candidates.generator import DEFAULT_LORA_GRID, StageBCandidate, stage_b_candidates
from ftplatform.customers import store
from ftplatform.customers.context import CustomerContext
from ftplatform.db import connect


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture
def customer_ctx(monkeypatch, tmp_path, conn):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "acme", "Acme Inc", "saas_support")
    ctx = CustomerContext(conn, "acme")
    ctx.data_dir().mkdir(parents=True, exist_ok=True)
    (ctx.data_dir() / "train.jsonl").write_text("", encoding="utf-8")
    (ctx.data_dir() / "eval.jsonl").write_text("", encoding="utf-8")
    return ctx


def _fake_train_result(adapter_dir):
    return {"profile": "unused", "train_runtime_s": 1.0, "train_loss": 0.1,
            "trainable_params": 100, "trainable_pct": 0.01, "peak_vram_gb": 4.0,
            "adapter_dir": str(adapter_dir), "merged_dir": None, "merge_error": None}


def _fake_eval_result(cost=1.0):
    return {
        "profile": "saas_support", "regime": "none", "n_eval": 150, "report_path": "unused",
        "refused_for_compliance": [], "failed_systems": {}, "egress_acknowledged": False,
        "systems": {"finetuned": {"schema_adherence_pct": 100.0, "record_exact_pct": 90.0,
                                    "headline_field": "issue.priority", "headline_accuracy_pct": 90.0,
                                    "p50_ms": 10.0, "p99_ms": 12.0, "mean_prompt_tokens": 43.0,
                                    "cost_per_100k_usd": cost}},
    }


def _stub(monkeypatch, cost=1.0):
    captured = {}

    def fake_train_run(cfg, key, **kwargs):
        captured["cfg"] = cfg
        captured["key"] = key
        captured["data_dir"] = kwargs.get("data_dir")
        return _fake_train_result(cfg.adapter_dir(key))

    monkeypatch.setattr(runner.train_mod, "run", fake_train_run)
    monkeypatch.setattr(runner.benchmark, "run", lambda **k: _fake_eval_result(cost))
    return captured


def test_stage_b_grid_has_unique_ids_and_matches_the_default_grid():
    candidates = stage_b_candidates("some/model")
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids)) == len(DEFAULT_LORA_GRID)
    assert {(c.lora_r, c.lora_alpha) for c in candidates} == set(DEFAULT_LORA_GRID)


def test_stage_b_refuses_to_run_without_a_training_corpus(monkeypatch, conn, tmp_path):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    store.create(conn, "nodata", "No Data Inc", "saas_support")
    ctx = CustomerContext(conn, "nodata")
    candidate = StageBCandidate("stageB-r8-a16", "some/model", 8, 16)

    with pytest.raises(FileNotFoundError, match="baseline run"):
        runner.run_stage_b_candidate(ctx, candidate)


def test_stage_b_trains_with_the_candidates_lora_params_via_a_copy(monkeypatch, customer_ctx):
    captured = _stub(monkeypatch)
    original_base_model = customer_ctx.config.model.base_model
    candidate = StageBCandidate("stageB-r8-a16", "org/model-x", lora_r=8, lora_alpha=16)

    runner.run_stage_b_candidate(customer_ctx, candidate)

    cfg = captured["cfg"]
    assert cfg.model.base_model == "org/model-x"
    assert cfg.lora.r == 8
    assert cfg.lora.lora_alpha == 16
    # The customer's own shared Config must be untouched by this candidate's sweep.
    assert customer_ctx.config.model.base_model == original_base_model
    assert customer_ctx.config.lora.r != 8  # default LoraConfig.r is 16, not this candidate's 8


def test_stage_b_passes_the_shared_customer_data_dir_not_a_candidate_scoped_one(monkeypatch, customer_ctx):
    captured = _stub(monkeypatch)
    candidate = StageBCandidate("stageB-r8-a16", "org/model-x", 8, 16)

    runner.run_stage_b_candidate(customer_ctx, candidate)

    assert captured["data_dir"] == customer_ctx.data_dir()
    assert "candidates" not in str(captured["data_dir"])


def test_stage_b_writes_train_and_evaluate_manifests(monkeypatch, customer_ctx):
    _stub(monkeypatch)
    candidate = StageBCandidate("stageB-r8-a16", "org/model-x", 8, 16)

    result = runner.run_stage_b_candidate(customer_ctx, candidate)

    assert result["systems"]["finetuned"]["record_exact_pct"] == 90.0
    manifests = customer_ctx.manifests_dir("stageB-r8-a16")
    assert (manifests / "train.json").exists()
    assert (manifests / "evaluate.json").exists()


def test_two_stage_b_candidates_do_not_collide(monkeypatch, customer_ctx):
    _stub(monkeypatch, cost=1.0)
    runner.run_stage_b_candidate(customer_ctx, StageBCandidate("stageB-r8-a16", "m", 8, 16))
    _stub(monkeypatch, cost=9.0)
    runner.run_stage_b_candidate(customer_ctx, StageBCandidate("stageB-r32-a32", "m", 32, 32))

    a = json.loads((customer_ctx.manifests_dir("stageB-r8-a16") / "evaluate.json")
                    .read_text(encoding="utf-8"))
    b = json.loads((customer_ctx.manifests_dir("stageB-r32-a32") / "evaluate.json")
                    .read_text(encoding="utf-8"))
    assert a["metrics"]["systems"]["finetuned"]["cost_per_100k_usd"] == 1.0
    assert b["metrics"]["systems"]["finetuned"]["cost_per_100k_usd"] == 9.0
    assert customer_ctx.adapter_dir("stageB-r8-a16") != customer_ctx.adapter_dir("stageB-r32-a32")


def test_stage_b_result_feeds_the_leaderboard_the_same_as_stage_a(monkeypatch, customer_ctx):
    """Stage B's evaluate manifest must have the exact shape leaderboard.py
    already reads for Stage A -- one more reason not to special-case it."""
    from ftplatform.candidates import leaderboard

    _stub(monkeypatch, cost=1.0)
    runner.run_stage_b_candidate(customer_ctx, StageBCandidate("stageB-r8-a16", "m", 8, 16))

    rows = leaderboard.load_candidate_systems(customer_ctx, "stageB-r8-a16", label="r8/a16")
    assert len(rows) == 1
    assert rows[0].system == "finetuned"
    assert rows[0].cost_per_100k_usd == 1.0
