"""silica.bench.spec_metrics — speculative-decoding metrics schema.

P5.9 step 2(f) per D-021 (v1.7.14). Predeclares the
``ScenarioResult.metadata`` fields every Track C speculative variant
will emit, before any C.x lands. The point: the C.4 DFlash spike's
gate threshold (≥1.8× silica-integrated speedup vs C.1) must be
evaluated on the **same axes** the C.1 / C.2 / C.3 / C.5 / C.6
variants report. If each variant invents its own metadata shape,
cross-variant comparison degrades into ad-hoc field-mapping in
the bench report layer.

This module owns the schema and a single validator. Concrete
oracle / runner code under Track C populates a ``dict`` matching the
schema and calls :func:`validate_speculative_metrics` to fail loud
on missing / mistyped fields. The validator is **non-coercive**:
it returns a list of structured reason strings rather than raising,
mirroring the existing oracle pattern (``ok, reason, metadata``)
so callers can decide whether to fail their bench row or just log.

Schema (one-line per field; full semantics in
:data:`SPECULATIVE_METRIC_FIELD_DOCS`):

  - ``accept_rate``: ``float`` in ``[0.0, 1.0]``.
  - ``verify_cost_ms``: ``float`` (ms per target forward).
  - ``draft_cost_ms``: ``float`` (ms per draft forward; ``0.0`` for
    same-model self-spec like C.6 QuantSpec).
  - ``tokens_per_target_forward``: ``float`` (mean accepted tokens
    per target verification — the headline speedup observable).
  - ``rollback_count``: ``int`` (rollback events triggered by
    rejected drafts during the measurement window).
  - ``tree_node_visits``: ``int`` (tree-variant drafters only;
    ``0`` for trajectory drafters like C.1 / C.4).
  - ``quality_parity_status``: ``QualityParityStatus`` enum value.

P5.9 step 2(f) **lands the schema only**. No oracle is updated to
emit these fields here — that wiring belongs to C.1 (the spec
foundation, D-021 step 5) and propagates to C.2 / C.3 / C.4 / C.5 /
C.6 from there. The schema being live in code before any C.x
implementation forces variant authors to either match or
explicitly extend it (with a corresponding update to this
module's docs and the validator).
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class QualityParityStatus(str, Enum):
    """Greedy-decode parity result for a speculative variant.

    - ``PARITY`` — spec on / spec off produced byte-equivalent
      token sequences under fixed seed (C.1 acceptance criterion;
      every variant should hit this on temperature=0).
    - ``DIVERGED`` — spec on != spec off. A real correctness
      regression; phase-exit must record the divergence and
      retire the variant.
    - ``NOT_TESTED`` — parity gate not run for this row. Allowed
      during exploratory measurement but never at phase-exit
      attestation.
    """

    PARITY = "parity"
    DIVERGED = "diverged"
    NOT_TESTED = "not_tested"


# Canonical schema. The set is frozen so a runtime mutation (e.g. an
# oracle silently dropping a field) cannot smuggle past the validator.
SPECULATIVE_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "accept_rate",
        "verify_cost_ms",
        "draft_cost_ms",
        "tokens_per_target_forward",
        "rollback_count",
        "tree_node_visits",
        "quality_parity_status",
    }
)


# Per-field documentation. Used by the bench report renderer when
# the schema is surfaced into a JSONL row's column header; also
# anchors the test that pins the schema's composition against
# silent drift.
SPECULATIVE_METRIC_FIELD_DOCS: dict[str, str] = {
    "accept_rate": (
        "Mean fraction of draft tokens accepted by the target "
        "verifier across the measurement window. Bounded in "
        "[0.0, 1.0]; <50% on dense 27B chat is a common signal "
        "that the variant has under-trained drafts."
    ),
    "verify_cost_ms": (
        "Mean wall-clock milliseconds per target-side verification "
        "forward across the measurement window. Includes the "
        "target's own prefill / decode cost plus speculative-"
        "specific overhead (mask construction, rollback bookkeeping)."
    ),
    "draft_cost_ms": (
        "Mean wall-clock milliseconds per drafter forward across "
        "the measurement window. ``0.0`` for same-model self-spec "
        "(C.6 QuantSpec) where draft and target share the same "
        "forward; non-zero for separate-model variants (C.1 "
        "draft-target, C.2 ReDrafter, C.3 MTP head, C.4 DFlash, "
        "C.5 DDTree)."
    ),
    "tokens_per_target_forward": (
        "Mean number of accepted tokens emitted per target-side "
        "verification forward. The headline speedup observable: "
        "a variant with `tokens_per_target_forward` = 4 amortizes "
        "one target weight read across 4 emitted tokens, lifting "
        "the bandwidth-bound ceiling proportionally."
    ),
    "rollback_count": (
        "Total rollback events triggered by rejected draft tokens "
        "during the measurement window. Each rollback pairs a "
        "``KVManager.rollback`` call with the adapter's "
        "``rollback_state``; a non-zero count on a "
        "``RecurrentStateAdapter`` row also exercises the P5.9 "
        "step 2(c) Qwen3.5 recurrent rollback path."
    ),
    "tree_node_visits": (
        "Total nodes visited across the draft tree's per-target-"
        "forward verification, summed over the measurement "
        "window. ``0`` for trajectory drafters (C.1 draft-target, "
        "C.4 DFlash). For C.5 DDTree this is the headline tree-"
        "shape observable; the ratio "
        "``tree_node_visits / num_target_forwards`` is the average "
        "tree budget actually consumed."
    ),
    "quality_parity_status": (
        "Greedy spec-on / spec-off parity result. See "
        ":class:`QualityParityStatus`. Phase-exit attestation "
        "requires PARITY; DIVERGED retires the variant; "
        "NOT_TESTED is allowed only during exploratory measurement."
    ),
}


def validate_speculative_metrics(
    metadata: dict[str, Any],
) -> list[str]:
    """Return a list of structured violation reasons.

    Empty list means ``metadata`` carries every speculative field
    in :data:`SPECULATIVE_METRIC_FIELDS` with a value of the right
    type. Each non-empty entry is a short, greppable reason
    suitable for the oracle's ``reason`` channel — e.g.
    ``"spec_metrics_missing:accept_rate"`` or
    ``"spec_metrics_type_error:accept_rate:expected_float_got_str"``.

    The validator is **non-coercive**: it does not mutate
    ``metadata`` or convert types. Callers (oracle / runner) decide
    whether a violation is fatal for their row.

    Range checks: ``accept_rate`` is bounded in ``[0.0, 1.0]``;
    millisecond / count fields must be non-negative; the
    ``quality_parity_status`` field accepts either a
    :class:`QualityParityStatus` enum value or one of its
    string aliases.
    """
    violations: list[str] = []
    for field in sorted(SPECULATIVE_METRIC_FIELDS):
        if field not in metadata:
            violations.append(f"spec_metrics_missing:{field}")
            continue
        value = metadata[field]
        violations.extend(_validate_field(field, value))
    return violations


def _validate_field(field: str, value: Any) -> list[str]:
    """Per-field type + range check. Centralised so a future field
    addition only adds one elif branch here."""
    if field == "accept_rate":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [
                f"spec_metrics_type_error:{field}:"
                f"expected_float_got_{type(value).__name__}"
            ]
        if not (0.0 <= float(value) <= 1.0):
            return [
                f"spec_metrics_range_error:{field}:"
                f"expected_in_[0,1]_got_{value}"
            ]
        return []
    if field in ("verify_cost_ms", "draft_cost_ms", "tokens_per_target_forward"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [
                f"spec_metrics_type_error:{field}:"
                f"expected_float_got_{type(value).__name__}"
            ]
        if float(value) < 0.0:
            return [
                f"spec_metrics_range_error:{field}:"
                f"expected_non_negative_got_{value}"
            ]
        return []
    if field in ("rollback_count", "tree_node_visits"):
        if not isinstance(value, int) or isinstance(value, bool):
            return [
                f"spec_metrics_type_error:{field}:"
                f"expected_int_got_{type(value).__name__}"
            ]
        if value < 0:
            return [
                f"spec_metrics_range_error:{field}:"
                f"expected_non_negative_got_{value}"
            ]
        return []
    if field == "quality_parity_status":
        if isinstance(value, QualityParityStatus):
            return []
        if isinstance(value, str):
            try:
                QualityParityStatus(value)
                return []
            except ValueError:
                allowed = ", ".join(sorted(s.value for s in QualityParityStatus))
                return [
                    f"spec_metrics_value_error:{field}:"
                    f"expected_one_of_[{allowed}]_got_{value!r}"
                ]
        return [
            f"spec_metrics_type_error:{field}:"
            f"expected_QualityParityStatus_or_str_got_"
            f"{type(value).__name__}"
        ]
    # Unknown field reached this helper — should be impossible
    # because the iteration above is keyed on
    # SPECULATIVE_METRIC_FIELDS. Defensive return for type-checker.
    return [f"spec_metrics_unknown_field:{field}"]  # pragma: no cover


__all__ = [
    "QualityParityStatus",
    "SPECULATIVE_METRIC_FIELDS",
    "SPECULATIVE_METRIC_FIELD_DOCS",
    "validate_speculative_metrics",
]
