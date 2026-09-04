"""
Automatic metric derivation.

A domain-agnostic engine cannot ship a hardcoded list of fields to score. So the
scoring plan is derived by walking the contract's JSON Schema and assigning each
leaf a comparison rule from its declared type:

    enum / string / boolean / date   ->  exact
    number / integer                 ->  numeric   (cent-level tolerance)
    array of scalars                 ->  set_f1    (order-insensitive, partial credit)
    declared free text               ->  token_f1  (exact match would be a paraphrase test)

The practical consequence: a customer who supplies nothing but a JSON Schema
gets a complete, sensible evaluation — per-field accuracy, record-level exact
match, confidence intervals — without writing a line of metric code.

Profiles override where domain knowledge beats inference: which field is the
headline result, which fields are free text, and which feed a business rule and
should therefore be reported as a group.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["exact", "numeric", "set_f1", "token_f1"]

# Free text is excluded from record-level exact match everywhere.
FREE_TEXT_KIND: Kind = "token_f1"


@dataclass(frozen=True)
class FieldSpec:
    path: str                       # dotted path, e.g. "issue.priority"
    kind: Kind
    headline: bool = False          # led with in reports; the field that sells the result
    rubric_input: bool = False      # feeds a business rule; reported as a group
    label: str = ""                 # display name, defaults to the path

    def display(self) -> str:
        return self.label or self.path


@dataclass
class ScoringPlan:
    fields: list = field(default_factory=list)

    @property
    def paths(self) -> list:
        return [f.path for f in self.fields]

    @property
    def record_match_paths(self) -> list:
        """Everything except free text: a verbatim summary is not the contract."""
        return [f.path for f in self.fields if f.kind != FREE_TEXT_KIND]

    @property
    def rubric_input_paths(self) -> list:
        return [f.path for f in self.fields if f.rubric_input]

    def headline(self) -> FieldSpec | None:
        for f in self.fields:
            if f.headline:
                return f
        return None

    def kind_of(self, path: str) -> Kind:
        for f in self.fields:
            if f.path == path:
                return f.kind
        return "exact"


def _resolve(node: dict, defs: dict) -> dict:
    """Follow a local $ref, as emitted by Pydantic for nested models."""
    ref = node.get("$ref")
    if not ref:
        return node
    key = ref.rsplit("/", 1)[-1]
    return defs.get(key, {})


def _unwrap_optional(node: dict, defs: dict) -> dict:
    """Collapse Optional[...] / anyOf[X, null] to the meaningful branch."""
    for key in ("anyOf", "oneOf"):
        options = node.get(key)
        if not options:
            continue
        concrete = [o for o in options if _resolve(o, defs).get("type") != "null"]
        if len(concrete) == 1:
            merged = _resolve(concrete[0], defs)
            return {**{k: v for k, v in node.items() if k not in ("anyOf", "oneOf")}, **merged}
    return node


def _kind_for(node: dict, defs: dict) -> Kind:
    node = _unwrap_optional(node, defs)
    declared = node.get("type")

    if declared == "array":
        return "set_f1"
    if declared in ("number", "integer"):
        return "numeric"
    return "exact"


def derive_scoring_plan(
    schema: dict,
    free_text: tuple = (),
    headline: str | None = None,
    rubric_inputs: tuple = (),
    labels: dict | None = None,
    max_depth: int = 4,
) -> ScoringPlan:
    """Build a scoring plan from a JSON Schema.

    `free_text`, `headline` and `rubric_inputs` are dotted paths supplied by the
    profile; everything else is inferred.
    """
    defs = schema.get("$defs", {}) or schema.get("definitions", {}) or {}
    labels = labels or {}
    specs: list = []

    def walk(node: dict, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        node = _resolve(_unwrap_optional(node, defs), defs)
        properties = node.get("properties")

        if not properties:
            return

        for name, child_raw in properties.items():
            path = f"{prefix}.{name}" if prefix else name
            child = _resolve(_unwrap_optional(child_raw, defs), defs)

            # Nested object: recurse rather than comparing whole subtrees, so a
            # report can say *which* nested field a model got wrong.
            if child.get("type") == "object" and child.get("properties"):
                walk(child, path, depth + 1)
                continue

            kind: Kind = FREE_TEXT_KIND if path in free_text else _kind_for(child_raw, defs)
            specs.append(FieldSpec(
                path=path,
                kind=kind,
                headline=(path == headline),
                rubric_input=(path in rubric_inputs),
                label=labels.get(path, ""),
            ))

    walk(schema, "", 0)

    # Order the report so the field that carries the argument comes first.
    specs.sort(key=lambda s: (not s.headline, not s.rubric_input, s.path))
    return ScoringPlan(fields=specs)
