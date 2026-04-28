"""P5.9 step 2(f) — speculative metrics schema validation.

Predeclares the shared metadata shape every Track C speculative
variant will emit so the C.4 DFlash spike's gate threshold (≥1.8×
silica-integrated speedup vs C.1) can be evaluated on identical
axes across C.1 / C.2 / C.3 / C.4 / C.5 / C.6.

Tests in this file are pure-unit on the schema + validator; no
oracle / runner / engine wiring depends on the schema yet (that
lands with C.1 in D-021 step 5). Pinning the schema in code
**before** any C.x implementation forces variant authors to either
match the contract or extend it explicitly (with a corresponding
update to ``SPECULATIVE_METRIC_FIELD_DOCS`` and these tests).
"""

from __future__ import annotations

import pytest

from silica.bench.spec_metrics import (
    SPECULATIVE_METRIC_FIELD_DOCS,
    SPECULATIVE_METRIC_FIELDS,
    QualityParityStatus,
    validate_speculative_metrics,
)

# --- schema composition ---


def test_schema_field_set_pinned() -> None:
    """The set of speculative metric fields is pinned to the v1.7.14
    D-021 step 2(f) baseline. Adding a field is a deliberate plan
    decision — update this test in the same PR.
    """
    expected = frozenset(
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
    assert SPECULATIVE_METRIC_FIELDS == expected, (
        f"SPECULATIVE_METRIC_FIELDS drifted from the v1.7.14 baseline. "
        f"Expected {sorted(expected)}, got "
        f"{sorted(SPECULATIVE_METRIC_FIELDS)}. Adding / removing a "
        f"field needs a PLAN.md Decisions Log entry naming the C.x "
        f"variant that drove the change."
    )


def test_every_field_has_docs() -> None:
    """Every schema field must have a per-field doc-string in
    SPECULATIVE_METRIC_FIELD_DOCS so the bench report's column
    header / JSONL row layer can render meaningful labels."""
    missing = SPECULATIVE_METRIC_FIELDS - set(
        SPECULATIVE_METRIC_FIELD_DOCS
    )
    assert not missing, (
        f"fields missing from SPECULATIVE_METRIC_FIELD_DOCS: "
        f"{sorted(missing)}"
    )
    extras = set(SPECULATIVE_METRIC_FIELD_DOCS) - SPECULATIVE_METRIC_FIELDS
    assert not extras, (
        f"SPECULATIVE_METRIC_FIELD_DOCS has stale entries no longer "
        f"in the schema: {sorted(extras)}"
    )


def test_quality_parity_status_alphabet_pinned() -> None:
    """Three-state alphabet — PARITY / DIVERGED / NOT_TESTED. Adding
    a fourth state is also a plan decision."""
    assert {s.value for s in QualityParityStatus} == {
        "parity",
        "diverged",
        "not_tested",
    }


# --- validator: passing shapes ---


def _valid_metadata(**overrides) -> dict:  # type: ignore[no-untyped-def]
    """Reference well-formed metadata. Tests override individual
    fields to exercise per-field validation."""
    base = {
        "accept_rate": 0.65,
        "verify_cost_ms": 12.4,
        "draft_cost_ms": 1.8,
        "tokens_per_target_forward": 2.6,
        "rollback_count": 3,
        "tree_node_visits": 0,
        "quality_parity_status": QualityParityStatus.PARITY,
    }
    base.update(overrides)
    return base


def test_validator_accepts_well_formed_metadata() -> None:
    assert validate_speculative_metrics(_valid_metadata()) == []


def test_validator_accepts_string_alias_for_quality_parity_status() -> None:
    """Oracles emitting JSONL-friendly strings (not enum members) are
    accepted as long as the string matches a QualityParityStatus
    value. Both forms round-trip through json.dumps cleanly."""
    for alias in ("parity", "diverged", "not_tested"):
        meta = _valid_metadata(quality_parity_status=alias)
        assert validate_speculative_metrics(meta) == [], alias


def test_validator_accepts_zero_draft_cost_self_spec_path() -> None:
    """C.6 QuantSpec-like self-spec uses one forward for both draft
    and target; ``draft_cost_ms = 0.0`` is the canonical signal."""
    meta = _valid_metadata(draft_cost_ms=0.0)
    assert validate_speculative_metrics(meta) == []


def test_validator_accepts_zero_tree_node_visits_for_trajectory_drafters() -> None:
    """C.1 draft-target / C.4 DFlash are trajectory drafters; tree-
    visit count stays at 0."""
    meta = _valid_metadata(tree_node_visits=0)
    assert validate_speculative_metrics(meta) == []


# --- validator: failing shapes ---


@pytest.mark.parametrize(
    "field",
    sorted(SPECULATIVE_METRIC_FIELDS),
)
def test_validator_reports_missing_field(field: str) -> None:
    meta = _valid_metadata()
    del meta[field]
    violations = validate_speculative_metrics(meta)
    assert any(
        v == f"spec_metrics_missing:{field}" for v in violations
    ), violations


def test_validator_reports_type_error_on_string_accept_rate() -> None:
    meta = _valid_metadata(accept_rate="0.65")
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith("spec_metrics_type_error:accept_rate:")
        for v in violations
    ), violations


def test_validator_reports_range_error_on_out_of_band_accept_rate() -> None:
    meta = _valid_metadata(accept_rate=1.5)
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith("spec_metrics_range_error:accept_rate:")
        for v in violations
    ), violations
    meta = _valid_metadata(accept_rate=-0.1)
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith("spec_metrics_range_error:accept_rate:")
        for v in violations
    ), violations


def test_validator_reports_negative_costs() -> None:
    for field in ("verify_cost_ms", "draft_cost_ms", "tokens_per_target_forward"):
        meta = _valid_metadata(**{field: -1.0})
        violations = validate_speculative_metrics(meta)
        assert any(
            v.startswith(f"spec_metrics_range_error:{field}:")
            for v in violations
        ), (field, violations)


def test_validator_reports_negative_counts() -> None:
    for field in ("rollback_count", "tree_node_visits"):
        meta = _valid_metadata(**{field: -1})
        violations = validate_speculative_metrics(meta)
        assert any(
            v.startswith(f"spec_metrics_range_error:{field}:")
            for v in violations
        ), (field, violations)


def test_validator_rejects_bool_for_int_count() -> None:
    """Python's ``bool`` is an ``int`` subclass; a stray ``True`` /
    ``False`` slipping into a count field would silently pass an
    isinstance(value, int) guard. The validator's explicit
    ``isinstance(value, bool)`` check rejects it."""
    meta = _valid_metadata(rollback_count=True)
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith("spec_metrics_type_error:rollback_count:")
        for v in violations
    ), violations


def test_validator_rejects_bool_for_float_metric() -> None:
    meta = _valid_metadata(accept_rate=False)
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith("spec_metrics_type_error:accept_rate:")
        for v in violations
    ), violations


def test_validator_rejects_unknown_quality_parity_string() -> None:
    meta = _valid_metadata(quality_parity_status="acceptable")
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith(
            "spec_metrics_value_error:quality_parity_status:"
        )
        for v in violations
    ), violations


def test_validator_rejects_non_str_non_enum_quality_parity() -> None:
    meta = _valid_metadata(quality_parity_status=42)
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith(
            "spec_metrics_type_error:quality_parity_status:"
        )
        for v in violations
    ), violations


def test_validator_int_accepted_where_float_expected() -> None:
    """Integer literals are valid floats per the schema (mypy /
    runtime treat ``int <: float``); the validator must not reject
    ``accept_rate=1`` or ``verify_cost_ms=10``."""
    meta = _valid_metadata(
        accept_rate=1,
        verify_cost_ms=10,
        draft_cost_ms=0,
        tokens_per_target_forward=2,
    )
    assert validate_speculative_metrics(meta) == []


# --- regression guard: schema is decoupled from existing oracles ---


# --- P5.9.1 hardening: non-finite float values ---


@pytest.mark.parametrize(
    "field",
    ["accept_rate", "verify_cost_ms", "draft_cost_ms", "tokens_per_target_forward"],
)
def test_validator_rejects_nan_for_float_metric(field: str) -> None:
    """P5.9.1: ``float('nan')`` slips past ``< 0.0`` / ``<=`` /
    ``>=`` comparisons (any nan comparison is False), so a Track C
    timer that explodes to nan would silently pass the range band
    pre-P5.9.1. The validator now rejects nan via
    ``math.isfinite`` before the range comparison."""
    meta = _valid_metadata(**{field: float("nan")})
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith(f"spec_metrics_value_error:{field}:")
        and "expected_finite" in v
        for v in violations
    ), (field, violations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("verify_cost_ms", float("inf")),
        ("draft_cost_ms", float("inf")),
        ("tokens_per_target_forward", float("inf")),
    ],
)
def test_validator_rejects_inf_for_non_negative_float(
    field: str, value: float
) -> None:
    """``float('inf') >= 0.0`` is True, so positive infinity passes
    the non-negative range check pre-P5.9.1. The finite check now
    catches it. Negative infinity already fails the non-negative
    check, so the inf failure mode here is the positive-inf one."""
    meta = _valid_metadata(**{field: value})
    violations = validate_speculative_metrics(meta)
    assert any(
        v.startswith(f"spec_metrics_value_error:{field}:")
        and "expected_finite" in v
        for v in violations
    ), (field, violations)


def test_validator_rejects_inf_for_accept_rate() -> None:
    """``accept_rate = +inf`` fails the finite check (P5.9.1) before
    the range check would have caught it; ``-inf`` fails the finite
    check too. Either way the reason names ``expected_finite``,
    not ``expected_in_[0,1]``, so the failure surface is honest
    about the nature of the error."""
    for value in (float("inf"), float("-inf")):
        meta = _valid_metadata(accept_rate=value)
        violations = validate_speculative_metrics(meta)
        assert any(
            v.startswith("spec_metrics_value_error:accept_rate:")
            and "expected_finite" in v
            for v in violations
        ), (value, violations)


def test_validator_rejects_negative_inf_for_non_negative_float() -> None:
    """Negative infinity fails the finite check (P5.9.1). Pre-P5.9.1
    it would have hit the range check; post-P5.9.1 the finite check
    runs first and is the more informative reason."""
    for field in (
        "verify_cost_ms",
        "draft_cost_ms",
        "tokens_per_target_forward",
    ):
        meta = _valid_metadata(**{field: float("-inf")})
        violations = validate_speculative_metrics(meta)
        assert any(
            v.startswith(f"spec_metrics_value_error:{field}:")
            and "expected_finite" in v
            for v in violations
        ), (field, violations)


# --- regression guard: schema is decoupled from existing oracles ---


def test_existing_oracle_metadata_does_not_satisfy_spec_schema() -> None:
    """Negative-control: validate that the WARM_DECODE oracle's
    metadata shape is **not** silently mistaken for spec metadata.
    The two schemas are deliberately disjoint at v1.7.14 — Track C
    will mix them when C.x emits both warm-decode AND spec metrics
    on the same row, but until then a WARM_DECODE row's metadata
    must produce missing-field reasons under the spec validator.
    """
    warm_decode_metadata = {
        "decode_tok_s_warm_aggregate": 154.0,
        "decode_tok_s_warm_per_row_mean": 153.0,
        "warmup_min_steps": 32,
        "warmup_rolling_window": 16,
        "warmup_rel_std_threshold": 0.05,
        "measurement_steps_min": 64,
        "rows": [],
    }
    violations = validate_speculative_metrics(warm_decode_metadata)
    assert violations, (
        "WARM_DECODE metadata accidentally satisfies the spec "
        "schema — schemas have collided"
    )
    # All seven fields should be missing.
    missing_count = sum(
        1
        for v in violations
        if v.startswith("spec_metrics_missing:")
    )
    assert missing_count == len(SPECULATIVE_METRIC_FIELDS), (
        f"expected {len(SPECULATIVE_METRIC_FIELDS)} missing-field "
        f"violations, got {missing_count} (violations={violations})"
    )
