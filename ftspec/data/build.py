"""
Corpus construction — generic across profiles.

The engine owns splitting, de-duplication, leakage prevention, chat formatting
and contract verification. The profile owns only what a document in its domain
looks like. That split is what makes a new vertical a data exercise rather than
an engineering one.

Two invariants are enforced here rather than trusted:

  * every generated label satisfies the profile's own contract, checked with the
    same strict validator the production server uses. Generating data our own
    validator would reject poisons training and the benchmark simultaneously.
  * the eval split shares no source document with train or val. It is generated
    from a separate seed *and* de-duplicated globally, because a different seed
    alone does not guarantee different samples.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from ftspec.core.profile import Profile, Sample
from ftspec.run import get_logger

log = get_logger("ftspec.build")


def to_chat_record(sample: Sample, profile: Profile) -> dict:
    """Chat format for TRL.

    The stored system prompt is the SHORT one, because that is what the model is
    trained against. The benchmark swaps in longer variants for the prompted
    baselines at eval time.
    """
    return {
        "messages": [
            {"role": "system", "content": profile.prompts.short},
            {"role": "user", "content": sample.source_text},
            {"role": "assistant", "content": json.dumps(sample.record, ensure_ascii=False)},
        ],
        "meta": sample.meta,
    }


def _fingerprint(sample: Sample) -> str:
    return hashlib.sha256(sample.source_text.encode("utf-8")).hexdigest()


def _generate_unique(profile: Profile, n: int, rng: random.Random, seen: set) -> list:
    """Draw `n` samples the corpus has not already used."""
    out: list = []
    for _ in range(12):  # a few rounds of over-generation to absorb collisions
        if len(out) >= n:
            break
        for sample in profile.generate(n - len(out), rng):
            key = _fingerprint(sample)
            if key in seen:
                continue
            seen.add(key)
            out.append(sample)
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(
            f"profile '{profile.name}' produced only {len(out)}/{n} unique samples; "
            "its generator needs more entropy for a corpus this size"
        )
    return out


def _distribution(samples: list, profile: Profile) -> dict:
    """Label balance for the headline field, plus the majority-class baseline.

    A degenerate distribution makes accuracy meaningless: if 85% of records are
    'routine', a model that always says 'routine' scores 85% and has learned
    nothing. The baseline is printed so nobody has to work that out themselves.
    """
    path = profile.headline_field
    if not path:
        return {}

    def get(record: dict):
        cur = record
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    counts = Counter(get(s.record) for s in samples)
    majority = max(counts.values()) / len(samples) if samples else 0.0
    return {
        "field": path,
        "counts": dict(sorted((str(k), v) for k, v in counts.items())),
        "majority_class_baseline": round(100 * majority, 1),
    }


def write_split(path: Path, samples: list, profile: Profile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(to_chat_record(sample, profile), ensure_ascii=False) + "\n")
    log.info("wrote %4d samples -> %s", len(samples), path)


def load_external(path: Path, profile: Profile) -> list:
    """Adopt a customer's own corpus.

    Accepts either the engine's chat format or a flat {"input", "output"} shape,
    so bringing existing data does not require rewriting it first.
    """
    samples: list = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            source = row["messages"][1]["content"]
            record = json.loads(row["messages"][2]["content"])
        elif "input" in row and "output" in row:
            source = row["input"]
            record = row["output"] if isinstance(row["output"], dict) else json.loads(row["output"])
        else:
            raise ValueError(
                f"{path}#{i}: expected either a 'messages' array or 'input'/'output' keys"
            )
        samples.append(Sample(source_text=source, record=record, meta=row.get("meta", {})))
    log.info("loaded %d samples from %s", len(samples), path)
    return samples


def run(profile: Profile, out_dir: Path, n_train: int = 200, n_val: int = 40,
         n_eval: int = 150, seed: int = 42, eval_seed: int = 1337,
         from_jsonl: Path | None = None) -> dict:
    """Build train/val/eval for a profile. Returns metrics for the run manifest."""
    seen: set = set()

    if from_jsonl is not None:
        samples = load_external(Path(from_jsonl), profile)
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_eval = min(n_eval, max(1, len(samples) // 5))
        n_val = min(n_val, max(1, len(samples) // 10))
        evalset = samples[:n_eval]
        val = samples[n_eval:n_eval + n_val]
        train = samples[n_eval + n_val:]
    else:
        if not profile.supports_generation():
            raise ValueError(
                f"Profile '{profile.name}' cannot generate synthetic data. "
                "Supply your own corpus with --from-jsonl."
            )
        rng = random.Random(seed)
        train = _generate_unique(profile, n_train, rng, seen)
        val = _generate_unique(profile, n_val, rng, seen)
        # Separate seed AND global de-duplication.
        evalset = _generate_unique(profile, n_eval, random.Random(eval_seed), seen)

    contract = profile.contract
    for split_name, samples in (("train", train), ("val", val), ("eval", evalset)):
        for i, sample in enumerate(samples):
            record, err = contract.validate(json.dumps(sample.record, ensure_ascii=False))
            if record is None:
                raise ValueError(f"{split_name}#{i} violates the profile contract: {err}")

    write_split(out_dir / "train.jsonl", train, profile)
    write_split(out_dir / "val.jsonl", val, profile)
    write_split(out_dir / "eval.jsonl", evalset, profile)

    metrics = {
        "profile": profile.name,
        "n_train": len(train), "n_val": len(val), "n_eval": len(evalset),
        "seed": seed, "eval_seed": eval_seed,
        "source": str(from_jsonl) if from_jsonl else "synthetic",
        "train_distribution": _distribution(train, profile),
        "eval_distribution": _distribution(evalset, profile),
    }

    dist = metrics["train_distribution"]
    if dist:
        log.info("train %s: %s (majority-class baseline %.1f%%)",
                  dist["field"], dist["counts"], dist["majority_class_baseline"])
        if dist["majority_class_baseline"] > 70:
            log.warning("one class covers %.1f%% of the corpus; accuracy on %s will be "
                         "hard to interpret", dist["majority_class_baseline"], dist["field"])
    log.info("%d documents written, all unique and contract-valid",
              len(train) + len(val) + len(evalset))
    return metrics
