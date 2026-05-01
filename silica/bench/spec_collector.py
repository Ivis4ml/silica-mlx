"""silica.bench.spec_collector — SpecMetricCollector for D-021 step 5 (g).

Lightweight accumulator the speculative engine emits to during a
single ``Engine.generate`` invocation. The bench runner constructs
one collector per scenario, threads it into the engine, and after
the run calls :meth:`SpecMetricCollector.materialize` to obtain a
dict matching :data:`silica.bench.spec_metrics.SPECULATIVE_METRIC_FIELDS`,
ready for ``ScenarioResult.metadata`` and the schema validator.

The collector is **passive**: it records what the engine tells it
and computes derived means at materialize time. It does not own
threading, locking, or any side-effect (no logging, no metrics
registry coupling) — that keeps the engine's hot path identical
modulo the optional ``spec_collector.record_*`` calls.

Scope at (g):

  - Single-request ``Engine.generate`` emission. The
    ``ContinuousBatcher`` multi-request path is **not** wired here;
    that lands in (h) along with the batched bench scenarios.
  - ``quality_parity_status`` is set externally by the harness
    (e.g. the bench runner's parity-check oracle). Default is
    :attr:`QualityParityStatus.NOT_TESTED`; the harness flips it
    to ``PARITY`` / ``DIVERGED`` after running the comparison.

The seven schema fields and their derivation:

  - ``accept_rate`` — ``accepted / proposed`` (drafts the verifier
    accepted out of drafts proposed). ``0.0`` when no drafts proposed.
  - ``verify_cost_ms`` — mean ms per target verify forward.
    ``0.0`` when no verify forward ran.
  - ``draft_cost_ms`` — mean ms per draft propose call.
    ``0.0`` when the drafter is same-model self-spec (caller passes
    explicit ``0.0`` via :meth:`set_self_spec`) or no drafts ran.
  - ``tokens_per_target_forward`` —
    ``(yielded_drafts + bonus_tokens) / target_forward_count``.
    Each verify forward produces logits that may emit up to
    ``yielded_count`` accepted drafts **plus** one bonus token
    (sampled from the rejected-position logits, or from the
    last-position logits on full accept). The bonus is emitted
    when neither ``max_tokens`` nor a stop-token cut the cycle
    short before the bonus path runs; ``record_bonus`` accounts
    for it. Counting only ``yielded_drafts`` would systematically
    understate spec efficacy on low-accept-rate workloads (full
    reject would read ``0.0`` even though the verify forward
    yielded one bonus token). ``0.0`` when no verify forward ran.
  - ``rollback_count`` — number of cycles that fired a target-side
    KV rollback (``un_committed > 0``).
  - ``tree_node_visits`` — always ``0`` for trajectory drafters
    (linear k-token verify); tree variants (C.5) override.
  - ``quality_parity_status`` — see :meth:`record_parity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from silica.bench.spec_metrics import QualityParityStatus


@dataclass
class SpecMetricCollector:
    """Per-scenario speculative-decoding metric accumulator.

    Construct one per ``Engine.generate`` invocation. Pass to the
    engine via the ``spec_collector`` keyword. After the generation
    completes, call :meth:`materialize` to lift the seven schema
    fields into a dict for ``ScenarioResult.metadata``.
    """

    # Counts.
    proposed_drafts: int = 0
    accepted_drafts: int = 0
    yielded_drafts: int = 0
    bonus_tokens: int = 0
    target_forward_count: int = 0
    draft_forward_count: int = 0
    rollback_count: int = 0
    tree_node_visits: int = 0

    # Cumulative timings, in milliseconds.
    target_forward_ms_total: float = 0.0
    draft_forward_ms_total: float = 0.0

    # Same-model self-spec flag. When true, ``draft_cost_ms`` is
    # forced to ``0.0`` regardless of accumulated draft timing —
    # the drafter shares forward with the target so attributing a
    # separate cost is meaningless.
    self_spec: bool = False

    # Parity status — externally set by the bench harness.
    parity_status: QualityParityStatus = QualityParityStatus.NOT_TESTED

    # --- engine-side recorders (called from Engine.generate spec branch) ---

    def record_propose(
        self, *, draft_count: int, elapsed_ms: float
    ) -> None:
        """Record one ``DraftEngine.propose`` call.

        ``draft_count`` is the number of drafts the drafter actually
        returned (may be less than ``γ``). ``elapsed_ms`` is the
        wall-clock cost; callers running same-model self-spec should
        either pass ``0.0`` or set :attr:`self_spec` so
        :meth:`materialize` zeros ``draft_cost_ms``.
        """
        self.proposed_drafts += int(draft_count)
        self.draft_forward_count += 1
        self.draft_forward_ms_total += float(elapsed_ms)

    def record_verify(
        self, *, accepted_len: int, yielded_count: int, elapsed_ms: float
    ) -> None:
        """Record one target-side verify forward.

        ``accepted_len`` is the verifier's accept count;
        ``yielded_count`` is what the engine actually emitted (may
        be lower if ``max_tokens`` or a stop token cut the yield
        short). ``accept_rate`` is built from
        ``accepted_drafts / proposed_drafts``;
        ``tokens_per_target_forward`` is built from
        ``(yielded_drafts + bonus_tokens) / target_forward_count``,
        with ``bonus_tokens`` accumulated separately by
        :meth:`record_bonus`.
        """
        self.accepted_drafts += int(accepted_len)
        self.yielded_drafts += int(yielded_count)
        self.target_forward_count += 1
        self.target_forward_ms_total += float(elapsed_ms)

    def record_rollback(self) -> None:
        """Record one cycle that fired a target-side KV rollback."""
        self.rollback_count += 1

    def record_bonus(self) -> None:
        """Record one bonus token emitted from the most recent verify
        forward's logits. Suppressed when ``max_tokens`` or a stop token
        cut the cycle short before the bonus path runs. Contributes to
        ``tokens_per_target_forward`` alongside ``yielded_drafts``."""
        self.bonus_tokens += 1

    # --- harness-side setter (called by the bench runner / parity oracle) ---

    def record_parity(self, status: QualityParityStatus) -> None:
        """Set the quality-parity status from an external comparison."""
        self.parity_status = status

    def set_self_spec(self, value: bool = True) -> None:
        """Mark the run as same-model self-spec; ``draft_cost_ms``
        will materialize as ``0.0``."""
        self.self_spec = value

    # --- output ------------------------------------------------------------

    def materialize(self) -> dict[str, Any]:
        """Return the seven-field dict matching the spec-metrics schema.

        Means are computed at call time (not maintained as running
        averages) so the dict reflects the final state of the run.
        Callers can call this multiple times — the collector is
        otherwise read-only after the engine returns.
        """
        accept_rate = (
            self.accepted_drafts / self.proposed_drafts
            if self.proposed_drafts > 0
            else 0.0
        )
        verify_cost_ms = (
            self.target_forward_ms_total / self.target_forward_count
            if self.target_forward_count > 0
            else 0.0
        )
        if self.self_spec or self.draft_forward_count == 0:
            draft_cost_ms = 0.0
        else:
            draft_cost_ms = (
                self.draft_forward_ms_total / self.draft_forward_count
            )
        tokens_per_target_forward = (
            (self.yielded_drafts + self.bonus_tokens)
            / self.target_forward_count
            if self.target_forward_count > 0
            else 0.0
        )
        return {
            "accept_rate": accept_rate,
            "verify_cost_ms": verify_cost_ms,
            "draft_cost_ms": draft_cost_ms,
            "tokens_per_target_forward": tokens_per_target_forward,
            "rollback_count": int(self.rollback_count),
            "tree_node_visits": int(self.tree_node_visits),
            "quality_parity_status": self.parity_status,
        }


__all__ = ["SpecMetricCollector"]
