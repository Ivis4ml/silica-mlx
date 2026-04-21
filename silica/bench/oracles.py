"""silica.bench.oracles — P-4.1 oracle implementations.

Each oracle decides pass / fail for one scenario run. The runner in
:mod:`silica.bench.runner` invokes the oracle registered for
``Scenario.oracle`` after the workload executes, passing the emitted
token ids plus a context dict the runner assembles from adapter /
engine state. Oracles return ``(ok, reason, metadata)`` so failures
carry a structured reason into ``ScenarioResult.reason`` without
growing a parallel error-code enum.

Only :class:`OracleKind.SMOKE` is implemented in P-4.1 because the
first migrated scenario (Qwen3-0.6B smoke) is the only one that
needs an oracle at all in this phase. The other kinds registered
here raise ``NotImplementedError`` with a phase pointer so the
runner can fail fast rather than silently treat them as always-pass
if the catalog (or a user-authored scenario) points at one before
its implementation lands.

Phase ownership:

  * ``B1_PARITY_VS_SINGLE`` — lands with the D3.1 migration in P-4.2
    (the B=1 half of ``tests/test_p3_gemma4_batched_parity.py``).
  * ``BGT1_DIRECT_BATCHED_REFERENCE`` — lands with the D3.1
    migration in P-4.2 (the B>1 vs direct mlx-lm reference half).
  * ``TEACHER_FORCED_ARGMAX`` — lands in P-4.3 as the stronger
    next-token-argmax oracle.

The smoke oracle checks the minimal invariants any "did the run
crash?" test would check:

  1. At least one token was emitted (empty generation = upstream
     problem, not "passed").
  2. Every emitted id is an int in ``[0, vocab_size)`` (guards
     against the engine returning raw tensors or placeholder
     sentinels if the adapter regresses).

The context dict the runner passes is deliberately a free-form
``Any`` rather than a typed dataclass — oracle-specific requirements
(vocab_size here, reference tokens later, logits tensors in P-4.3)
change per kind and carrying all of them in one typed object would
bake premature structure into the oracle seam.
"""

from __future__ import annotations

from typing import Any

from silica.bench.scenario import OracleFn, OracleKind, Scenario


def smoke_oracle(
    scenario: Scenario, token_ids: list[int], context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """SMOKE oracle: did the workload emit at least one valid token?

    ``context`` must be a mapping containing ``"vocab_size"``. The
    runner assembles it from ``adapter.config.vocab_size`` before
    invoking the oracle.

    Metadata on success carries ``total_tokens`` and the max id
    observed so the JSONL row is self-describing — a reviewer can
    tell at a glance whether the "4 tokens, all < 150000" claim was
    actually met without re-running the scenario.
    """
    if not isinstance(context, dict) or "vocab_size" not in context:
        return (
            False,
            "smoke_oracle_missing_context_vocab_size",
            {},
        )
    vocab_size = int(context["vocab_size"])
    if not token_ids:
        return (False, "smoke_no_tokens_emitted", {"vocab_size": vocab_size})

    max_id = 0
    for i, tok in enumerate(token_ids):
        if not isinstance(tok, int):
            return (
                False,
                f"smoke_token_{i}_not_int:{type(tok).__name__}",
                {"vocab_size": vocab_size},
            )
        if tok < 0 or tok >= vocab_size:
            return (
                False,
                f"smoke_token_{i}_out_of_vocab:{tok}",
                {"vocab_size": vocab_size},
            )
        if tok > max_id:
            max_id = tok
    return (
        True,
        None,
        {
            "total_tokens": len(token_ids),
            "max_token_id": max_id,
            "vocab_size": vocab_size,
        },
    )


def _not_implemented(phase: str) -> OracleFn:
    """Build a stub oracle that names the phase where it lands."""

    def _stub(
        scenario: Scenario, token_ids: list[int], context: Any
    ) -> tuple[bool, str | None, dict[str, Any]]:
        raise NotImplementedError(
            f"Oracle {scenario.oracle.value!r} for scenario "
            f"{scenario.id!r} lands in {phase}."
        )

    return _stub


ORACLES: dict[OracleKind, OracleFn] = {
    OracleKind.SMOKE: smoke_oracle,
    OracleKind.B1_PARITY_VS_SINGLE: _not_implemented("P-4.2"),
    OracleKind.BGT1_DIRECT_BATCHED_REFERENCE: _not_implemented("P-4.2"),
    OracleKind.TEACHER_FORCED_ARGMAX: _not_implemented("P-4.3"),
}


__all__ = ["smoke_oracle", "ORACLES"]
