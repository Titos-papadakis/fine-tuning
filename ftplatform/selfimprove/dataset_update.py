"""
Phase 6 -- folding human-reviewed production corrections into the training
corpus.

Appends validated corrections straight to train.jsonl rather than rebuilding
the whole corpus through `ftspec.data.build.run(..., from_jsonl=...)`, which
reshuffles and re-splits train/val/eval from scratch. That's the right
behavior for a first `ftspec prepare`, but wrong here: it would silently
regenerate eval.jsonl, and `deploy.py`'s McNemar gate depends on every
candidate across every retrain cycle being scored against the exact same
held-out set. val/eval stay untouched; only train.jsonl grows.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from ftspec.data.build import load_external, to_chat_record
from ftspec.run import get_logger

log = get_logger("ftplatform.selfimprove.dataset_update")


def fold_corrections(ctx, corrections_path=None) -> dict:
    """Validate every pending correction against the profile's own contract
    (the same validator production serving uses -- a correction that fails
    it would poison training exactly like a bad synthetic sample would) and
    append the ones that pass to train.jsonl. Archives the corrections file
    afterward so a re-run doesn't fold the same rows in twice.

    Returns {"added": int, "rejected": int, "rejected_reasons": [str, ...]}.
    """
    path = corrections_path or (ctx.memory_dir() / "corrections.jsonl")
    if not path.exists() or path.stat().st_size == 0:
        return {"added": 0, "rejected": 0, "rejected_reasons": []}

    samples = load_external(path, ctx.profile)
    accepted, rejected_reasons = [], []
    for sample in samples:
        record_text = json.dumps(sample.record, ensure_ascii=False)
        parsed, reason = ctx.profile.contract.validate(record_text)
        if parsed is None:
            rejected_reasons.append(reason)
            continue
        accepted.append(sample)

    if accepted:
        train_path = ctx.data_dir() / "train.jsonl"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(train_path, "a", encoding="utf-8") as f:
            for sample in accepted:
                f.write(json.dumps(to_chat_record(sample, ctx.profile), ensure_ascii=False) + "\n")
        log.info("customer %s: folded %d correction(s) into %s (%d rejected)",
                  ctx.customer.id, len(accepted), train_path, len(rejected_reasons))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archived = path.with_name(f"corrections.{stamp}.jsonl")
    shutil.move(str(path), str(archived))

    return {"added": len(accepted), "rejected": len(rejected_reasons),
            "rejected_reasons": rejected_reasons}
