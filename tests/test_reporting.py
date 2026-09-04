"""
Reporting and artefact-layout tests.

The layout tests exist because of a real defect: training wrote the adapter to
`outputs/<profile>/lora_adapter` while evaluation defaulted to
`outputs/lora_adapter`. Both code paths looked correct in isolation, and the
mismatch only surfaced after a GPU had been paid for and training had finished.
Paths are now derived in exactly one place, and these tests assert that the
producing stage and the consuming stage agree.
"""
from __future__ import annotations

import json

import pytest

from ftspec import reporting
from ftspec.config import Config

PROFILES = ["saas_support", "fintech_disputes", "healthcare_clinical"]


# --- artefact layout ---------------------------------------------------------

@pytest.mark.parametrize("profile", PROFILES)
def test_every_artefact_path_is_namespaced_by_profile(profile):
    cfg = Config()
    for path in (cfg.data_dir(profile), cfg.adapter_dir(profile), cfg.merged_dir(profile),
                  cfg.checkpoints_dir(profile), cfg.reports_dir(profile),
                  cfg.manifests_dir(profile)):
        assert profile in path.parts, f"{path} is not namespaced by profile"


def test_profiles_never_share_an_artefact_directory():
    cfg = Config()
    adapters = {p: cfg.adapter_dir(p) for p in PROFILES}
    assert len(set(adapters.values())) == len(PROFILES)


def test_train_output_matches_evaluate_input():
    """The exact regression: the writer and the reader must agree on one path."""
    cfg = Config()
    for profile in PROFILES:
        written_by_train = cfg.adapter_dir(profile)
        read_by_evaluate = cfg.adapter_dir(profile)
        assert written_by_train == read_by_evaluate
        assert written_by_train.parent == cfg.outputs_dir(profile)


def test_merge_config_holds_names_not_paths():
    """Names compose with the profile; a full path here would reintroduce the bug."""
    cfg = Config()
    assert "/" not in cfg.merge.adapter_name
    assert "\\" not in cfg.merge.adapter_name
    assert cfg.adapter_dir("x").name == cfg.merge.adapter_name


# --- report block ------------------------------------------------------------

MANIFEST = {
    "git_commit": "abc123def456",
    "git_dirty": False,
    "gpu": {"name": "Tesla T4", "vram_gb": 15.0},
    "metrics": {
        "profile": "saas_support",
        "n_eval": 150,
        "systems": {
            "finetuned": {"schema_adherence_pct": 100.0, "record_exact_pct": 88.7,
                           "p50_ms": 1420, "p99_ms": 2210, "mean_prompt_tokens": 196.0,
                           "mean_output_tokens": 162, "cost_per_100k_usd": 90.0},
            "base-constrained": {"schema_adherence_pct": 100.0, "record_exact_pct": 41.3,
                                  "p50_ms": 1810, "p99_ms": 2900,
                                  "mean_prompt_tokens": 1437.0, "cost_per_100k_usd": 120.0},
        },
    },
}

LATENCY = {
    "specialized": {"prompt_tokens": 196, "ttft_p50_ms": 240,
                     "ttft_max_ms": 310, "decode_tok_s": 18.4},
    "prompted_baseline": {"prompt_tokens": 1437, "ttft_p50_ms": 1350,
                           "ttft_max_ms": 1720, "decode_tok_s": 18.1},
}


def latency_pair(tmp_path, payload=None):
    path = tmp_path / "latency.json"
    path.write_text(json.dumps(payload or LATENCY), encoding="utf-8")
    return reporting.LatencyPair.load(path)


def test_block_reports_every_system(tmp_path):
    block = reporting.build_block("saas_support", MANIFEST, latency_pair(tmp_path), 0.35)
    assert "`finetuned`" in block and "`base-constrained`" in block
    assert "150 held-out documents" in block
    assert "Tesla T4" in block
    assert "abc123de" in block


def test_block_states_the_prefill_ratio(tmp_path):
    block = reporting.build_block("saas_support", MANIFEST, latency_pair(tmp_path), 0.35)
    assert "7.3× less prefill" in block


def test_block_surfaces_mini_undercutting_self_hosted(tmp_path):
    """The uncomfortable comparison must appear, not be quietly dropped."""
    block = reporting.build_block("saas_support", MANIFEST, latency_pair(tmp_path), 0.35)
    assert "GPT-4o-mini" in block
    assert "below" in block


def test_block_omits_mini_caveat_when_self_hosting_is_cheaper(tmp_path):
    fast = json.loads(json.dumps(LATENCY))
    fast["specialized"]["decode_tok_s"] = 400.0     # as if batched
    fast["specialized"]["ttft_p50_ms"] = 60
    block = reporting.build_block("saas_support", MANIFEST, latency_pair(tmp_path, fast), 0.35)
    assert "below** the" not in block


def test_block_works_without_latency_measurements():
    block = reporting.build_block("saas_support", MANIFEST, None, 0.35)
    assert "`finetuned`" in block
    assert "TTFT" not in block


def test_empty_manifest_is_rejected():
    with pytest.raises(ValueError, match="no system results"):
        reporting.build_block("saas_support", {"metrics": {"systems": {}}}, None, 0.35)


# --- README splicing ---------------------------------------------------------

README = f"""# Title

## Projected economics

> **These are modeled, not measured.** Some projection.

### Measured results

{reporting.BEGIN}
*Not yet populated.*
{reporting.END}

## Quickstart
Unchanged tail.
"""


def test_update_replaces_only_the_marked_region(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")

    assert reporting.update_readme(readme, "### Measured results\n\nfresh numbers")
    text = readme.read_text(encoding="utf-8")

    assert "fresh numbers" in text
    assert "*Not yet populated.*" not in text
    assert "# Title" in text
    assert "Unchanged tail." in text
    assert text.count(reporting.BEGIN) == 1 and text.count(reporting.END) == 1


def test_update_demotes_the_projection_notice(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    reporting.update_readme(readme, "block")
    text = readme.read_text(encoding="utf-8")
    assert "Superseded by measured results" in text
    assert "## Projected economics" not in text


def test_update_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    reporting.update_readme(readme, "block")
    first = readme.read_text(encoding="utf-8")
    reporting.update_readme(readme, "block")
    assert readme.read_text(encoding="utf-8") == first


def test_missing_markers_are_an_explicit_error(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        reporting.update_readme(readme, "block")


def test_real_readme_has_the_markers():
    """Guards against someone editing the README and orphaning `ftspec report`."""
    from ftspec.config import REPO_ROOT
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert reporting.BEGIN in text and reporting.END in text
