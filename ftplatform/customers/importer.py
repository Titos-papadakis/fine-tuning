"""
Bring a customer's own tickets in as training data.

A real customer hands over historical tickets, usually as a CSV export, often
without labels in our schema. This turns that into the flat
{"input", "output"} corpus `ftspec.data.build.load_external()` already
accepts, written to imports/corpus.jsonl, which `run_baseline()` then picks up
automatically instead of synthetic data.

Every row is validated against the workload's own contract before it can
reach training -- a label that doesn't fit the schema is rejected with its
reason, never silently coerced. Unlabeled rows are either auto-labeled by a
pluggable `labeler` (a callable text -> raw JSON string, e.g.
openai_labeler()) or set aside in imports/to_label.jsonl. Auto-labels are
marked in each row's meta, and a random sample is written to
imports/review_sample.jsonl so a human can spot-check the labeler before
spending GPU time training on its output.
"""
from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from pathlib import Path

MIN_RECOMMENDED_ROWS = 200
REVIEW_SAMPLE_SIZE = 20
TEXT_KEYS = ("input", "text", "ticket", "body", "message")
OUTPUT_KEYS = ("output", "label_json", "label")


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _pick(row: dict, keys: tuple) -> str | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def read_raw_rows(path: Path) -> list[dict]:
    """[{"text": str, "output": dict | str | None}] from a .jsonl or .csv file.

    Accepts the engine's own chat format, a flat input/output shape, or any
    obvious text column name (see TEXT_KEYS) -- a customer's export should not
    need rewriting before it can be imported."""
    path = Path(path)
    raw: list[dict] = []
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            raw = [dict(r) for r in csv.DictReader(f)]
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}#{i}: not valid JSON: {e}") from e
    else:
        raise ValueError(f"unsupported file type {path.suffix!r} -- use .csv or .jsonl")

    rows = []
    for r in raw:
        if "messages" in r:
            rows.append({"text": r["messages"][1]["content"], "output": r["messages"][2]["content"]})
            continue
        rows.append({"text": _pick(r, TEXT_KEYS), "output": _pick(r, OUTPUT_KEYS)})
    return rows


def _validate_output(contract, output) -> tuple[dict | None, str]:
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    return contract.validate(text)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_corpus(ctx, path: Path, labeler: Callable[[str], str] | None = None,
                   labeler_name: str = "labeler", max_auto_label: int | None = None,
                   seed: int = 0) -> dict:
    """Validate, dedupe and (optionally) auto-label `path` into
    imports/corpus.jsonl for this customer. Re-importing appends; a text
    already in the corpus is skipped, not duplicated. Returns counts."""
    contract = ctx.profile.contract
    corpus_path = ctx.imported_corpus_path()
    imports = ctx.imports_dir()
    seen = {_norm(r["input"]) for r in _read_jsonl(corpus_path)}
    already_to_label = {_norm(r["input"]) for r in _read_jsonl(imports / "to_label.jsonl")}

    accepted, rejected, to_label, auto_labeled = [], [], [], []
    counts = {"read": 0, "empty": 0, "duplicate": 0, "labeled": 0, "auto_labeled": 0,
              "rejected": 0, "to_label": 0}
    auto_budget = max_auto_label

    for i, row in enumerate(read_raw_rows(path)):
        counts["read"] += 1
        text = (row["text"] or "").strip()
        if not text:
            counts["empty"] += 1
            continue
        key = _norm(text)
        if key in seen:
            counts["duplicate"] += 1
            continue

        if row["output"] is not None:
            record, err = _validate_output(contract, row["output"])
            if record is None:
                rejected.append({"row": i, "input": text, "reason": f"label invalid: {err}"})
                continue
            accepted.append({"input": text, "output": record, "meta": {"label_source": "customer"}})
            counts["labeled"] += 1
            seen.add(key)
            continue

        if labeler is None or (auto_budget is not None and auto_budget <= 0):
            if key not in already_to_label:
                to_label.append({"row": i, "input": text})
                already_to_label.add(key)
            seen.add(key)
            continue

        if auto_budget is not None:
            auto_budget -= 1
        try:
            raw = labeler(text)
        except Exception as e:                                        # noqa: BLE001
            rejected.append({"row": i, "input": text,
                             "reason": f"auto-label failed: {type(e).__name__}: {e}"})
            continue
        record, err = contract.validate(raw)
        if record is None:
            rejected.append({"row": i, "input": text, "reason": f"auto-label invalid: {err}"})
            continue
        item = {"input": text, "output": record, "meta": {"label_source": f"auto:{labeler_name}"}}
        accepted.append(item)
        auto_labeled.append(item)
        counts["auto_labeled"] += 1
        seen.add(key)

    counts["rejected"] = len(rejected)
    if accepted:
        _write_jsonl(corpus_path, accepted, mode="a")
    _write_jsonl(imports / "rejected.jsonl", rejected)
    labeled_now = {_norm(r["input"]) for r in accepted}
    still_to_label = [r for r in _read_jsonl(imports / "to_label.jsonl")
                      if _norm(r["input"]) not in labeled_now] + to_label
    if still_to_label or (imports / "to_label.jsonl").exists():
        _write_jsonl(imports / "to_label.jsonl", still_to_label)
    if auto_labeled:
        sample = random.Random(seed).sample(auto_labeled, min(REVIEW_SAMPLE_SIZE, len(auto_labeled)))
        _write_jsonl(imports / "review_sample.jsonl", sample)

    counts["to_label"] = len(still_to_label)
    counts["corpus_total"] = len(_read_jsonl(corpus_path))
    counts["enough_to_train"] = counts["corpus_total"] >= MIN_RECOMMENDED_ROWS
    counts["corpus_path"] = str(corpus_path)
    return counts


def openai_labeler(profile, model: str = "gpt-4o-mini", client=None) -> Callable[[str], str]:
    """Labels with a hosted model using the profile's own schema+rubric
    prompt -- the same prompt the gpt4o-* benchmark baselines use. Refuses
    for a profile whose compliance regime forbids sending text off-box,
    since customer tickets are exactly the text that rule protects."""
    if not profile.compliance.allows_external_api:
        raise PermissionError(
            f"profile {profile.name!r} ({profile.compliance.regime}) forbids external APIs -- "
            f"its data cannot be sent to {model} for labeling. Label it in-house instead.")
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    system_prompt = profile.prompts.variants()["schema+rubric"]

    def label(text: str) -> str:
        response = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": text}])
        return response.choices[0].message.content or ""

    return label
