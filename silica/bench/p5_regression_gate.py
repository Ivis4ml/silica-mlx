"""silica.bench.p5_regression_gate — P-5 (4-b) quality-regression gate.

P5.9 step 2(g) per D-021 (v1.7.14). Lifts the v1.7.3 / v1.7.4
(4-b) two-part aggregated gate from a one-off acceptance event into
an operational pre-merge contract: every Track A / B / C PR must
prove the gate still passes before it lands. The gate's pinned
reference values + decision math live here so the bench harness
and any future C.x oracle reach the same answer; the canonical
bench command lives in
``plans/P5_REGRESSION_GATE.md`` (operator's how-to).

Two evaluation modes:

1. **silica-only regression** — runs the anchor scenario
   ``qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned``
   across the canonical 3 seeds and asserts silica's mean ΔPPL
   stays within tolerance of the v1.7.3 :data:`SILICA_V1_7_3_SNAPSHOT`.
   Cheap (~10 s on a dev box with the Qwen3-0.6B HF cache pulled);
   suitable as a pre-merge bench invocation gating every Track
   A / B / C PR.
2. **full (4-b) aggregated gate** — runs both silica's path and
   vqbench's reproduce-script subprocess (via the existing
   ``--vqbench-xcheck`` plumbing). Requires a separate vqbench
   venv per D-009 (silica's runtime carries no torch /
   transformers / datasets). Slower; the canonical phase-exit
   attestation form documented in ``plans/P5_REGRESSION_GATE.md``.

Both modes use the same :func:`evaluate_4b_gate` helper for the
two-part aggregated check (mode 2) and the same
:func:`evaluate_silica_regression` helper for the cheap one-sided
check (mode 1). Unit tests in
``tests/test_p5_regression_gate.py`` exercise both helpers on
synthetic seed arrays — no real-model run is needed to verify
the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedSnapshot:
    """A 3-seed (mean, std, n) snapshot pinned at a known-good revision.

    Used as the silica-side reference value the cheap pre-merge
    regression check (:func:`evaluate_silica_regression`) compares
    against. The vqbench-side numbers in the v1.7.3 evidence are
    captured in :data:`VQBENCH_V1_7_3_SNAPSHOT` for completeness;
    they enter only the full (4-b) aggregated gate.
    """

    mean_delta_ppl: float
    std_delta_ppl: float
    n_seeds: int

    @property
    def sem(self) -> float:
        """Standard error of the mean: std / sqrt(n)."""
        if self.n_seeds <= 0:
            return float("inf")
        return self.std_delta_ppl / math.sqrt(self.n_seeds)


# v1.7.3 close evidence (2026-04-24, commit ed57be1; raw
# plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl). Anchor:
# qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned, seeds
# {42, 43, 44}, BlockTurboQuantMSE B=64 4-bit K+V on Qwen3-0.6B
# WikiText-2 chunked NLL.
SILICA_V1_7_3_SNAPSHOT: PinnedSnapshot = PinnedSnapshot(
    mean_delta_ppl=0.511,
    std_delta_ppl=0.354,
    n_seeds=3,
)
VQBENCH_V1_7_3_SNAPSHOT: PinnedSnapshot = PinnedSnapshot(
    mean_delta_ppl=0.661,
    std_delta_ppl=0.347,
    n_seeds=3,
)


# Default tolerance for the cheap silica-only regression mode. Picked
# at twice the v1.7.3 silica SEM (≈ 0.41 PPL) so legitimate seed-to-
# seed variance does not fire the gate, but a structural regression
# (wrong rotation, codec shape mismatch, missed projection-output
# capture) shifts the mean by enough to trip it. A tighter
# tolerance is appropriate when the C.x track under test has its
# own convergence story (see D-021 step 2(g) per-track adjustment
# guidance).
DEFAULT_SILICA_REGRESSION_TOLERANCE_PPL: float = 0.5


@dataclass(frozen=True)
class GateResult:
    """Outcome of either gate-evaluation mode.

    ``passes`` is the decision flag the caller acts on.
    ``mean_gap`` is the silica-vs-vqbench mean ΔPPL difference
    (full gate) or silica-vs-snapshot mean ΔPPL difference
    (silica-only mode); ``threshold`` is the tolerance applied;
    ``aggregate_band`` and ``absolute_band`` carry the two-part
    gate's component decisions in the full-mode result and stay
    ``None`` in silica-only mode.

    ``reason`` is a structured greppable string for the caller's
    log / oracle ``reason`` channel.
    """

    passes: bool
    mode: str  # "silica_only" or "full_4b"
    mean_gap: float
    threshold: float
    reason: str
    aggregate_band: float | None = None  # 2 * SEM_diff (full mode)
    absolute_band: float | None = None  # 1.0 PPL (full mode)


def _sample_std(values: list[float]) -> float:
    """Bessel-corrected sample standard deviation (n-1).

    Mirrors the v1.7.3 evidence calculation; numpy's default ddof=0
    would diverge from the recorded `std=0.354` on the same seeds.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def evaluate_silica_regression(
    silica_seed_delta_ppls: list[float],
    snapshot: PinnedSnapshot = SILICA_V1_7_3_SNAPSHOT,
    tolerance_ppl: float = DEFAULT_SILICA_REGRESSION_TOLERANCE_PPL,
    expected_n_seeds: int | None = None,
) -> GateResult:
    """Cheap one-sided regression check.

    Compares the mean of ``silica_seed_delta_ppls`` against the
    pinned snapshot's mean. Passes when
    ``|mean - snapshot.mean| <= tolerance_ppl``. Uses no
    vqbench data; suitable for the pre-merge bench gate that
    every Track A / B / C PR runs.

    P5.9.1 hardening: ``expected_n_seeds`` defaults to
    ``snapshot.n_seeds`` (3 under the v1.7.3 anchor) and the helper
    rejects seed arrays of any other length with a structured
    ``silica_regression_seed_count_mismatch`` reason. Without this
    a 1-seed run would silently pass the gate and defeat the
    statistical assumption the snapshot's std encodes. Callers who
    legitimately want a single-seed exploratory check pass
    ``expected_n_seeds=1`` explicitly.
    """
    n_required = (
        snapshot.n_seeds if expected_n_seeds is None else expected_n_seeds
    )
    if not silica_seed_delta_ppls:
        return GateResult(
            passes=False,
            mode="silica_only",
            mean_gap=float("nan"),
            threshold=tolerance_ppl,
            reason="silica_regression_no_seeds",
        )
    if len(silica_seed_delta_ppls) != n_required:
        return GateResult(
            passes=False,
            mode="silica_only",
            mean_gap=float("nan"),
            threshold=tolerance_ppl,
            reason=(
                f"silica_regression_seed_count_mismatch:"
                f"expected_{n_required}_got_{len(silica_seed_delta_ppls)}"
            ),
        )
    silica_mean = sum(silica_seed_delta_ppls) / len(
        silica_seed_delta_ppls
    )
    mean_gap = silica_mean - snapshot.mean_delta_ppl
    passes = abs(mean_gap) <= tolerance_ppl
    if passes:
        reason = (
            f"silica_regression_ok:mean={silica_mean:.4f}_vs_"
            f"snapshot={snapshot.mean_delta_ppl:.4f}_gap="
            f"{mean_gap:+.4f}_tol={tolerance_ppl}"
        )
    else:
        reason = (
            f"silica_regression_drift:mean={silica_mean:.4f}_vs_"
            f"snapshot={snapshot.mean_delta_ppl:.4f}_gap="
            f"{mean_gap:+.4f}_exceeds_tol={tolerance_ppl}"
        )
    return GateResult(
        passes=passes,
        mode="silica_only",
        mean_gap=mean_gap,
        threshold=tolerance_ppl,
        reason=reason,
    )


def evaluate_4b_gate(
    silica_seed_delta_ppls: list[float],
    vqbench_seed_delta_ppls: list[float],
    absolute_threshold_ppl: float = 1.0,
    expected_n_seeds: int = 3,
) -> GateResult:
    """Full (4-b) two-part aggregated gate.

    Both parts must hold simultaneously:

      ``|mean_gap| <= 2 * SEM_diff``    (aggregate band)
      ``|mean_gap| < absolute_threshold_ppl``  (absolute band, default 1.0)

    where
      ``mean_gap = mean(silica) - mean(vqbench)``
      ``SEM_diff = sqrt(std_silica^2 / n + std_vqbench^2 / n)``

    Bessel-corrected sample std (n-1). v1.7.3 closed at
    ``mean_gap = -0.150`` PPL, ``2 * SEM_diff = 0.572`` — both
    gates passed with comfortable headroom (~3.8x on the
    aggregate band, ~6.7x on the absolute band).

    P5.9.1 hardening: ``expected_n_seeds`` defaults to ``3`` (the
    v1.7.3 canonical seed-count {42, 43, 44}) and the helper
    rejects either array if its length differs. Without this a
    1-seed run would yield ``SEM_diff = 0`` (Bessel-corrected std
    on n=1 is 0 by convention) and mechanically collapse the
    aggregate band to 0, defeating the statistical contract the
    gate encodes. Callers who legitimately want a different
    seed budget pass ``expected_n_seeds`` explicitly.
    """
    if (
        not silica_seed_delta_ppls
        or not vqbench_seed_delta_ppls
    ):
        return GateResult(
            passes=False,
            mode="full_4b",
            mean_gap=float("nan"),
            threshold=absolute_threshold_ppl,
            reason="full_4b_no_seeds",
        )
    if len(silica_seed_delta_ppls) != len(vqbench_seed_delta_ppls):
        return GateResult(
            passes=False,
            mode="full_4b",
            mean_gap=float("nan"),
            threshold=absolute_threshold_ppl,
            reason=(
                f"full_4b_seed_count_mismatch:"
                f"silica={len(silica_seed_delta_ppls)}_"
                f"vqbench={len(vqbench_seed_delta_ppls)}"
            ),
        )
    if len(silica_seed_delta_ppls) != expected_n_seeds:
        return GateResult(
            passes=False,
            mode="full_4b",
            mean_gap=float("nan"),
            threshold=absolute_threshold_ppl,
            reason=(
                f"full_4b_seed_count_mismatch:"
                f"expected_{expected_n_seeds}_got_"
                f"{len(silica_seed_delta_ppls)}"
            ),
        )
    n = len(silica_seed_delta_ppls)
    silica_mean = sum(silica_seed_delta_ppls) / n
    vqbench_mean = sum(vqbench_seed_delta_ppls) / n
    mean_gap = silica_mean - vqbench_mean
    silica_std = _sample_std(silica_seed_delta_ppls)
    vqbench_std = _sample_std(vqbench_seed_delta_ppls)
    sem_diff = math.sqrt(
        silica_std**2 / n + vqbench_std**2 / n
    )
    aggregate_band = 2.0 * sem_diff
    aggregate_pass = abs(mean_gap) <= aggregate_band
    absolute_pass = abs(mean_gap) < absolute_threshold_ppl
    passes = aggregate_pass and absolute_pass
    if passes:
        reason = (
            f"full_4b_ok:mean_gap={mean_gap:+.4f}_aggregate_band="
            f"{aggregate_band:.4f}_absolute_band="
            f"{absolute_threshold_ppl}"
        )
    else:
        failed = []
        if not aggregate_pass:
            failed.append(
                f"aggregate({abs(mean_gap):.4f}>{aggregate_band:.4f})"
            )
        if not absolute_pass:
            failed.append(
                f"absolute({abs(mean_gap):.4f}>={absolute_threshold_ppl})"
            )
        reason = (
            "full_4b_fail:" + ",".join(failed) + f":mean_gap="
            f"{mean_gap:+.4f}"
        )
    return GateResult(
        passes=passes,
        mode="full_4b",
        mean_gap=mean_gap,
        threshold=absolute_threshold_ppl,
        reason=reason,
        aggregate_band=aggregate_band,
        absolute_band=absolute_threshold_ppl,
    )


__all__ = [
    "PinnedSnapshot",
    "SILICA_V1_7_3_SNAPSHOT",
    "VQBENCH_V1_7_3_SNAPSHOT",
    "DEFAULT_SILICA_REGRESSION_TOLERANCE_PPL",
    "GateResult",
    "evaluate_silica_regression",
    "evaluate_4b_gate",
]
