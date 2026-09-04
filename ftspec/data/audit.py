"""
Groundedness and compliance audit — generic shell, profile-supplied rules.

Two checks the engine runs for every profile:

  * **Groundedness.** Delegated to `profile.audit_sample`. A label the source
    document never mentions cannot be predicted by any model; shipping one caps
    the achievable score at an arbitrary ceiling and makes the benchmark
    unfalsifiable. Regulated profiles also assert their masking rules here.
  * **Leakage.** No eval document may appear in train or val. This is checked
    independently of the builder that is supposed to guarantee it, because a
    benchmark's central claim should not rest on a single unverified code path.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ftspec.core.profile import Profile, Sample
from ftspec.run import get_logger

log = get_logger("ftspec.audit")


def load(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_sample(row: dict) -> Sample:
    return Sample(
        source_text=row["messages"][1]["content"],
        record=json.loads(row["messages"][2]["content"]),
        meta=row.get("meta", {}),
    )


def _headline_counts(rows: list, profile: Profile) -> dict:
    path = profile.headline_field
    if not path:
        return {}
    counts: Counter = Counter()
    for row in rows:
        cur = json.loads(row["messages"][2]["content"])
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        counts[str(cur)] += 1
    return dict(sorted(counts.items()))


def run(profile: Profile, data_dir: Path) -> tuple:
    """Audit every split. Returns (passed, printable_report, metrics)."""
    splits = {name: load(data_dir / f"{name}.jsonl") for name in ("train", "val", "eval")}
    if not any(splits.values()):
        return False, f"\n  No data found in {data_dir}. Run `ftspec prepare` first.\n", {}

    failures: list = []
    for split, rows in splits.items():
        for i, row in enumerate(rows):
            for problem in profile.audit_sample(_as_sample(row)):
                failures.append(f"{split}#{i}: {problem}")

    train_texts = {r["messages"][1]["content"] for r in splits["train"] + splits["val"]}
    eval_texts = {r["messages"][1]["content"] for r in splits["eval"]}
    overlap = train_texts & eval_texts
    if overlap:
        failures.append(f"LEAKAGE: {len(overlap)} documents appear in both train/val and eval")

    compliance = profile.compliance
    lines = [
        "",
        f"Groundedness & compliance audit — profile '{profile.name}'",
        f"  regime: {compliance.regime}   external API: "
        f"{'allowed' if compliance.allows_external_api else 'PROHIBITED'}",
        "",
    ]

    metrics: dict = {"profile": profile.name, "regime": compliance.regime,
                      "splits": {}, "leakage": len(overlap)}
    for split, rows in splits.items():
        counts = _headline_counts(rows, profile)
        majority = (100 * max(counts.values()) / len(rows)) if counts and rows else 0.0
        lines.append(f"  {split:<6} n={len(rows):<4} {profile.headline_field or 'labels'}={counts} "
                      f"majority-class baseline={majority:.1f}%")
        metrics["splits"][split] = {"n": len(rows), "counts": counts,
                                     "majority_class_baseline": round(majority, 1)}
    lines.append(f"  train/val vs eval document overlap: {len(overlap)}")
    lines.append("")

    if compliance.forbidden_patterns:
        lines.append(f"  compliance scanners active: {', '.join(compliance.forbidden_patterns)}")
        lines.append("")

    metrics["violations"] = len(failures)
    metrics["passed"] = not failures

    if failures:
        lines.append(f"  FAILED — {len(failures)} violations:")
        for f in failures[:25]:
            lines.append(f"    - {f}")
        if len(failures) > 25:
            lines.append(f"    ... and {len(failures) - 25} more")
        lines.append("")
        return False, "\n".join(lines), metrics

    lines.append("  PASSED — every label is derivable from its source document, "
                  "no eval leakage, no compliance violations.")
    lines.append("")
    return True, "\n".join(lines), metrics
