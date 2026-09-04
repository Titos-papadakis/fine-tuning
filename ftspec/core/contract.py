"""
The output contract — one abstraction over two ways of declaring a schema.

Built-in profiles declare their output as a Pydantic model: typed, IDE-friendly,
and strict by construction. A customer bringing their own schema
(`--profile custom --schema mine.json`) has a raw JSON Schema document and no
Python classes at all.

Both must drive the identical downstream machinery — validation, prompt text,
Outlines grammars, vLLM `guided_json`, and metric derivation — so both are
wrapped here and the rest of the engine only ever sees a `Contract`.

Validation is strict in both modes. A model emitting `"amount": "412.50"` where
the contract says number has violated the contract even though a lenient parser
would coerce it, because the consumer downstream will not coerce it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


def _strip_code_fence(text: str) -> str:
    """Tolerate a markdown fence and nothing else.

    Trailing prose, single quotes and trailing commas are genuine contract
    violations: a downstream JSON parser rejects them, so the benchmark must
    too. Repairing them here would flatter the model and mislead the buyer.
    """
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines()
                          if not ln.strip().startswith("```")).strip()
    return text


@dataclass(frozen=True)
class Contract:
    """Validation and schema export for a profile's output record."""

    name: str
    _model: type[BaseModel] | None = None
    _schema: dict | None = None

    # --- construction --------------------------------------------------------

    @classmethod
    def from_model(cls, model: type[BaseModel], name: str = "") -> Contract:
        return cls(name=name or model.__name__, _model=model)

    @classmethod
    def from_json_schema(cls, schema: dict, name: str = "CustomRecord") -> Contract:
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("a contract schema must be a JSON Schema object with type: object")
        if "additionalProperties" not in schema:
            # Without this an extra invented key would pass validation, and the
            # guarantee we advertise would be weaker than the one we enforce.
            schema = {**schema, "additionalProperties": False}
        return cls(name=name, _schema=schema)

    @classmethod
    def from_file(cls, path: Path, name: str = "") -> Contract:
        schema = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_schema(schema, name=name or Path(path).stem)

    # --- use -----------------------------------------------------------------

    @property
    def model(self) -> type[BaseModel] | None:
        """The Pydantic model, when there is one. Outlines prefers it."""
        return self._model

    def json_schema(self) -> dict:
        """The schema handed to prompts, Outlines and vLLM `guided_json`."""
        if self._model is not None:
            return self._model.model_json_schema()
        return dict(self._schema or {})

    def schema_text(self) -> str:
        return json.dumps(self.json_schema(), ensure_ascii=False)

    def required_fields(self) -> list[str]:
        return list(self.json_schema().get("required", []))

    def validate(self, raw_text: str) -> tuple[dict | None, str]:
        """Parse and strictly validate raw model output.

        Returns (record_dict, "") on success or (None, reason) on failure, in
        both backends, so callers never branch on which one is in play.
        """
        text = _strip_code_fence(raw_text)

        if self._model is not None:
            try:
                obj = self._model.model_validate_json(text, strict=True)
            except Exception as e:
                return None, _first_error(e)
            return obj.model_dump(), ""

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"JSONDecodeError: {e}"

        try:
            from jsonschema import Draft202012Validator
        except ImportError as e:
            raise RuntimeError(
                "Custom JSON Schema profiles need jsonschema:  pip install 'ftspec[custom]'"
            ) from e

        validator = Draft202012Validator(self._schema)
        errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.path) or "<root>"
            return None, f"SchemaError: {loc}: {first.message}"
        return payload, ""


def _first_error(exc: Exception) -> str:
    """Condense a Pydantic ValidationError to one actionable line."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ())) or "<root>"
            return f"{type(exc).__name__}: {loc}: {first.get('msg', '')}"
        except Exception:
            pass
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {(lines[1] if len(lines) > 1 else lines[0]).strip()}"
