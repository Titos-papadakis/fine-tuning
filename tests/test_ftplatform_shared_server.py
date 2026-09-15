"""
ftplatform/serving/shared_server.py: discovering which customers have a
production adapter ready to be served from one shared multi-LoRA vLLM
engine, and grouping them by base model (an engine can only serve adapters
that share one). No GPU/vLLM here -- these are pure filesystem/db lookups;
tests/test_serving.py covers the actual multi-adapter request routing with
a stubbed engine.
"""
from __future__ import annotations

import json

import pytest

import ftspec.config as config_mod
from ftplatform.customers import store
from ftplatform.customers.context import PRODUCTION, CustomerContext
from ftplatform.db import connect
from ftplatform.deployment import registry
from ftplatform.serving import shared_server


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "customers.db")
    yield c
    c.close()


@pytest.fixture
def repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    return tmp_path


def _deploy(conn, repo_root, customer_id, workload, candidate_id, base_model):
    """Writes exactly what deploy.py::_promote() would have left behind:
    the original candidate's manifest (with base_model in its params, the
    same shape infer_base_model() reads), a production/ adapter directory,
    and a registry row."""
    store.get(conn, customer_id) or store.create(conn, customer_id, customer_id, workload)
    ctx = CustomerContext(conn, customer_id)

    manifests = ctx.manifests_dir(candidate_id)
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "evaluate.json").write_text(
        json.dumps({"params": {"base_model": base_model}, "metrics": {"systems": {}}}),
        encoding="utf-8")

    prod_adapter = ctx.adapter_dir(PRODUCTION)
    prod_adapter.mkdir(parents=True, exist_ok=True)
    (prod_adapter / "adapter_model.bin").write_text("fake", encoding="utf-8")

    registry.record(conn, customer_id, workload, candidate_id, {"systems": {}})
    return ctx


def test_discover_finds_a_customer_with_a_production_deployment(conn, repo_root):
    _deploy(conn, repo_root, "acme", "saas_support", "stageB-r16-a16", "org/model-a")

    found = shared_server.discover_production_adapters(conn, "saas_support")

    assert set(found) == {"acme"}
    assert found["acme"]["base_model"] == "org/model-a"
    assert found["acme"]["adapter_path"].endswith("lora_adapter") or \
        "lora_adapter" in found["acme"]["adapter_path"]


def test_discover_skips_a_customer_with_no_deployment_yet(conn, repo_root):
    store.create(conn, "onboarding", "Onboarding Inc", "saas_support")

    found = shared_server.discover_production_adapters(conn, "saas_support")

    assert found == {}


def test_discover_skips_customers_on_a_different_workload(conn, repo_root):
    _deploy(conn, repo_root, "acme", "saas_support", "c1", "org/model-a")
    _deploy(conn, repo_root, "other", "fintech_disputes", "c1", "org/model-a")

    found = shared_server.discover_production_adapters(conn, "saas_support")

    assert set(found) == {"acme"}


def test_discover_skips_an_inactive_customer(conn, repo_root):
    _deploy(conn, repo_root, "acme", "saas_support", "c1", "org/model-a")
    conn.execute("UPDATE customers SET status = 'inactive' WHERE id = 'acme'")
    conn.commit()

    found = shared_server.discover_production_adapters(conn, "saas_support")

    assert found == {}


def test_discover_two_customers_are_both_found_with_their_own_paths(conn, repo_root):
    _deploy(conn, repo_root, "acme", "saas_support", "c1", "org/model-a")
    _deploy(conn, repo_root, "globex", "saas_support", "c1", "org/model-a")

    found = shared_server.discover_production_adapters(conn, "saas_support")

    assert set(found) == {"acme", "globex"}
    assert found["acme"]["adapter_path"] != found["globex"]["adapter_path"]


# --- group_by_base_model -----------------------------------------------------

def test_group_by_base_model_groups_matching_customers_together():
    adapters = {
        "acme": {"adapter_path": "/a", "base_model": "org/model-a"},
        "globex": {"adapter_path": "/g", "base_model": "org/model-a"},
    }
    groups = shared_server.group_by_base_model(adapters)
    assert groups == {"org/model-a": {"acme": "/a", "globex": "/g"}}


def test_group_by_base_model_splits_customers_on_different_base_models():
    adapters = {
        "acme": {"adapter_path": "/a", "base_model": "org/model-a"},
        "globex": {"adapter_path": "/g", "base_model": "org/model-b"},
    }
    groups = shared_server.group_by_base_model(adapters)
    assert set(groups) == {"org/model-a", "org/model-b"}
    assert groups["org/model-a"] == {"acme": "/a"}
    assert groups["org/model-b"] == {"globex": "/g"}


def test_group_by_base_model_drops_a_customer_with_no_determinable_base_model():
    adapters = {
        "acme": {"adapter_path": "/a", "base_model": "org/model-a"},
        "mystery": {"adapter_path": "/m", "base_model": None},
    }
    groups = shared_server.group_by_base_model(adapters)
    assert "mystery" not in {c for g in groups.values() for c in g}
    assert set(groups) == {"org/model-a"}
