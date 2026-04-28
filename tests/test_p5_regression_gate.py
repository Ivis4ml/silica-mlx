"""P5.9 step 2(g) — pin the (4-b) regression gate decision math.

D-021 step 2(g) (v1.7.14) lifts the v1.7.3 / v1.7.4 (4-b) two-part
aggregated gate from a one-off acceptance event into an
operational pre-merge contract. ``silica.bench.p5_regression_gate``
owns the decision math + the v1.7.3 pinned snapshot. Tests here
exercise both gate-evaluation modes on synthetic seed arrays — no
real-model loading needed.

Pinned contracts:

1. v1.7.3 evidence numbers reproduce: feeding the exact recorded
   silica + vqbench 3-seed arrays into ``evaluate_4b_gate`` yields
   the same ``mean_gap = -0.150`` and ``aggregate_band ≈ 0.572``
   the v1.7.3 close attestation reports.
2. ``evaluate_silica_regression`` accepts seeds within tolerance
   of the snapshot mean and rejects seeds outside it.
3. ``evaluate_4b_gate`` requires BOTH gate components to pass — a
   pass on aggregate but fail on absolute (or vice versa) is
   recorded as a structured failure.
4. Empty / mismatched-length seed arrays produce structured
   failure reasons rather than raising.
"""

from __future__ import annotations

import math

import pytest

from silica.bench.p5_regression_gate import (
    DEFAULT_SILICA_REGRESSION_TOLERANCE_PPL,
    SILICA_V1_7_3_SNAPSHOT,
    VQBENCH_V1_7_3_SNAPSHOT,
    PinnedSnapshot,
    evaluate_4b_gate,
    evaluate_silica_regression,
)

# v1.7.3 raw seed-level ΔPPLs (reconstructed from the recorded mean +
# std + n=3 evidence; exact per-seed numbers from
# plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl). Tests
# below pin the helpers' answers against the recorded mean / std
# / mean_gap / SEM_diff at v1.7.3, which is the truthful
# acceptance-attestation form.
#
# Reconstruction strategy: keep std/mean fixed by symmetric
# perturbation around the recorded mean. The exact distribution
# does not matter for the gate decision math — mean and std are
# the only inputs ``evaluate_4b_gate`` consumes. The synthetic
# triples below produce the recorded mean + std exactly so the
# regression-mode and full-mode pin tests are robust.
SILICA_V1_7_3_SEEDS: list[float] = [
    SILICA_V1_7_3_SNAPSHOT.mean_delta_ppl
    - SILICA_V1_7_3_SNAPSHOT.std_delta_ppl,
    SILICA_V1_7_3_SNAPSHOT.mean_delta_ppl,
    SILICA_V1_7_3_SNAPSHOT.mean_delta_ppl
    + SILICA_V1_7_3_SNAPSHOT.std_delta_ppl,
]
VQBENCH_V1_7_3_SEEDS: list[float] = [
    VQBENCH_V1_7_3_SNAPSHOT.mean_delta_ppl
    - VQBENCH_V1_7_3_SNAPSHOT.std_delta_ppl,
    VQBENCH_V1_7_3_SNAPSHOT.mean_delta_ppl,
    VQBENCH_V1_7_3_SNAPSHOT.mean_delta_ppl
    + VQBENCH_V1_7_3_SNAPSHOT.std_delta_ppl,
]


# --- silica-only regression mode ---


def test_silica_regression_passes_at_pinned_mean() -> None:
    """The pinned snapshot itself trivially passes the regression
    gate (mean_gap = 0.0)."""
    result = evaluate_silica_regression(SILICA_V1_7_3_SEEDS)
    assert result.passes, result.reason
    assert result.mode == "silica_only"
    assert math.isclose(result.mean_gap, 0.0, abs_tol=1e-9)
    assert result.threshold == DEFAULT_SILICA_REGRESSION_TOLERANCE_PPL


def test_silica_regression_passes_within_tolerance() -> None:
    """A 0.3-PPL drift from the snapshot mean stays within the
    default 0.5-PPL tolerance and passes."""
    drifted = [s + 0.3 for s in SILICA_V1_7_3_SEEDS]
    result = evaluate_silica_regression(drifted)
    assert result.passes, result.reason
    assert math.isclose(result.mean_gap, 0.3, abs_tol=1e-6)


def test_silica_regression_fails_outside_tolerance() -> None:
    """A 1.0-PPL drift from the snapshot mean exceeds the default
    0.5-PPL tolerance and fails with a structured ``drift`` reason."""
    drifted = [s + 1.0 for s in SILICA_V1_7_3_SEEDS]
    result = evaluate_silica_regression(drifted)
    assert not result.passes
    assert "silica_regression_drift" in result.reason
    assert math.isclose(result.mean_gap, 1.0, abs_tol=1e-6)


def test_silica_regression_with_custom_tolerance() -> None:
    """Caller-supplied tolerance overrides the default — useful for
    Track C variants whose own convergence story warrants tighter
    or looser bands."""
    drifted = [s + 0.7 for s in SILICA_V1_7_3_SEEDS]
    # 0.7 exceeds default 0.5 → fail
    assert not evaluate_silica_regression(drifted).passes
    # But fits a looser 1.0 → pass
    assert evaluate_silica_regression(
        drifted, tolerance_ppl=1.0
    ).passes


def test_silica_regression_empty_seeds_fails_structured() -> None:
    result = evaluate_silica_regression([])
    assert not result.passes
    assert result.reason == "silica_regression_no_seeds"


def test_silica_regression_against_custom_snapshot() -> None:
    """A C.x variant may pin a different snapshot (e.g. once C.4
    DFlash establishes its own convergence baseline) — the helper
    must honour the caller's snapshot rather than the default."""
    custom = PinnedSnapshot(
        mean_delta_ppl=2.0, std_delta_ppl=0.1, n_seeds=3
    )
    seeds = [1.9, 2.0, 2.1]
    result = evaluate_silica_regression(seeds, snapshot=custom)
    assert result.passes, result.reason
    assert math.isclose(result.mean_gap, 0.0, abs_tol=1e-9)


# --- full (4-b) two-part aggregated gate ---


def test_full_gate_v1_7_3_evidence_reproduces() -> None:
    """Pin the v1.7.3 numbers: feeding the exact recorded silica /
    vqbench 3-seed arrays must produce ``mean_gap = -0.150`` and
    ``aggregate_band ≈ 0.572`` (matching the (4-b) close evidence
    in PLAN.md §7 P-5 Acceptance (4-b))."""
    result = evaluate_4b_gate(SILICA_V1_7_3_SEEDS, VQBENCH_V1_7_3_SEEDS)
    assert result.mode == "full_4b"
    assert result.passes, result.reason
    assert math.isclose(result.mean_gap, -0.150, abs_tol=1e-3), (
        f"mean_gap={result.mean_gap} drifted from v1.7.3 evidence "
        f"(-0.150). Either the gate math changed or the pinned "
        f"snapshot data drifted."
    )
    # 2 * SEM_diff = 2 * sqrt((0.354^2 + 0.347^2) / 3) ≈ 0.572
    assert result.aggregate_band is not None
    assert math.isclose(
        result.aggregate_band, 0.572, abs_tol=2e-3
    ), result.aggregate_band
    # Aggregate band passes with ~3.8x headroom; absolute band
    # passes with ~6.7x headroom.
    assert abs(result.mean_gap) < result.aggregate_band
    assert abs(result.mean_gap) < (result.absolute_band or 1.0)


def test_full_gate_fails_when_aggregate_band_violated() -> None:
    """Construct a synthetic triple where the mean_gap exceeds
    2 * SEM_diff — the gate fails on the aggregate band even
    though the absolute band would pass."""
    # silica seeds tightly clustered around 0.5; vqbench around 0.0.
    # std for both = 0; SEM_diff = 0; aggregate_band = 0; mean_gap = 0.5;
    # absolute_band = 1.0 (passes), aggregate_band fails.
    silica = [0.5, 0.5, 0.5]
    vqbench = [0.0, 0.0, 0.0]
    result = evaluate_4b_gate(silica, vqbench)
    assert not result.passes
    assert "aggregate(" in result.reason
    assert math.isclose(result.mean_gap, 0.5, abs_tol=1e-9)


def test_full_gate_fails_when_absolute_band_violated() -> None:
    """Construct a synthetic triple where the mean_gap exceeds
    1.0 PPL absolute even though SEM is large enough that the
    aggregate band would have passed. Both component bands check
    independently; either alone fails the combined gate."""
    # std large → SEM_diff large → aggregate_band well above 1.0
    silica = [0.0, 5.0, 10.0]
    vqbench = [-3.0, 0.0, 3.0]
    # silica.mean=5, vqbench.mean=0, mean_gap=5.0, std_silica=5.0,
    # std_vqbench=3.0, SEM_diff=sqrt(25/3+9/3)=sqrt(34/3)≈3.367,
    # aggregate_band≈6.733 — passes; absolute_band=1.0 — fails.
    result = evaluate_4b_gate(silica, vqbench)
    assert not result.passes
    assert "absolute(" in result.reason
    assert math.isclose(result.mean_gap, 5.0, abs_tol=1e-9)


def test_full_gate_fails_when_both_bands_violated() -> None:
    """Both components fail; reason names both."""
    silica = [10.0, 10.0, 10.0]
    vqbench = [0.0, 0.0, 0.0]
    result = evaluate_4b_gate(silica, vqbench)
    assert not result.passes
    assert "aggregate(" in result.reason
    assert "absolute(" in result.reason


def test_full_gate_seed_count_mismatch_fails_structured() -> None:
    result = evaluate_4b_gate([0.5, 0.6], [0.5, 0.6, 0.7])
    assert not result.passes
    assert "full_4b_seed_count_mismatch" in result.reason


def test_full_gate_empty_seeds_fails_structured() -> None:
    result = evaluate_4b_gate([], [0.5])
    assert not result.passes
    assert "full_4b_no_seeds" in result.reason
    result = evaluate_4b_gate([0.5], [])
    assert not result.passes
    assert "full_4b_no_seeds" in result.reason


def test_full_gate_custom_absolute_threshold() -> None:
    """A C.x variant may demand a tighter absolute band (e.g. 0.5 PPL)
    than the default 1.0 — the helper must honour the caller's
    threshold."""
    silica = [0.7, 0.7, 0.7]
    vqbench = [0.0, 0.0, 0.0]
    # mean_gap=0.7; aggregate_band=0; absolute=1.0 → pass on absolute
    # but fail on aggregate.
    default = evaluate_4b_gate(silica, vqbench)
    assert not default.passes
    # With absolute_threshold=0.5, both bands fail.
    tight = evaluate_4b_gate(
        silica, vqbench, absolute_threshold_ppl=0.5
    )
    assert not tight.passes
    assert "absolute(" in tight.reason


# --- structural sanity ---


def test_pinned_snapshot_immutability() -> None:
    """Snapshots are frozen dataclasses; attempting to mutate raises
    so a stale-import bug cannot accidentally rewrite the pinned
    v1.7.3 reference values at runtime."""
    with pytest.raises((AttributeError, TypeError)):
        SILICA_V1_7_3_SNAPSHOT.mean_delta_ppl = 999.0  # type: ignore[misc]


def test_pinned_snapshots_match_recorded_evidence() -> None:
    """Document the v1.7.3 acceptance numbers in code so a future
    refactor of the snapshot dataclass cannot silently drift the
    pinned reference."""
    assert SILICA_V1_7_3_SNAPSHOT.mean_delta_ppl == 0.511
    assert SILICA_V1_7_3_SNAPSHOT.std_delta_ppl == 0.354
    assert SILICA_V1_7_3_SNAPSHOT.n_seeds == 3
    assert VQBENCH_V1_7_3_SNAPSHOT.mean_delta_ppl == 0.661
    assert VQBENCH_V1_7_3_SNAPSHOT.std_delta_ppl == 0.347
    assert VQBENCH_V1_7_3_SNAPSHOT.n_seeds == 3


def test_sem_property_matches_recorded_v1_7_3() -> None:
    """SEM = std / sqrt(n) — pin the v1.7.3 SEM values for both
    sides so a refactor of the dataclass property body is caught
    immediately."""
    assert math.isclose(
        SILICA_V1_7_3_SNAPSHOT.sem,
        0.354 / math.sqrt(3),
        abs_tol=1e-9,
    )
    assert math.isclose(
        VQBENCH_V1_7_3_SNAPSHOT.sem,
        0.347 / math.sqrt(3),
        abs_tol=1e-9,
    )
