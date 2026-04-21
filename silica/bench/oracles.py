"""silica.bench.oracles — oracle implementations.

Each oracle decides pass / fail for one scenario run. The runner in
:mod:`silica.bench.runner` invokes the oracle registered for
``Scenario.oracle`` after the workload executes, passing the emitted
token ids plus a context dict the runner assembles from adapter /
engine state. Oracles return ``(ok, reason, metadata)`` so failures
carry a structured reason into ``ScenarioResult.reason`` without
growing a parallel error-code enum.

Implemented kinds:

  * :class:`OracleKind.SMOKE` (P-4.1) — at least one token, every
    id in ``[0, vocab_size)``.
  * :class:`OracleKind.B1_PARITY_VS_SINGLE` (P-4.2b) — B=1 batched
    token stream must equal a single-request reference stream
    token-by-token. Runner drives both executions with the same
    ``SamplingParams`` and passes reference tokens via
    ``context['reference_tokens']``; oracle is a pure equality
    check with structured mismatch reporting.

Not yet implemented (registered as stubs that raise):

  * ``BGT1_DIRECT_BATCHED_REFERENCE`` — lands in P-4.2c (B>1 vs
    direct mlx-lm batched reference).
  * ``TEACHER_FORCED_ARGMAX`` — lands in P-4.3.

The smoke oracle checks the minimal invariants any "did the run
crash?" test would check:

  1. At least one token was emitted (empty generation = upstream
     problem, not "passed").
  2. Every emitted id is an int in ``[0, vocab_size)`` (guards
     against the engine returning raw tensors or placeholder
     sentinels if the adapter regresses).

The context dict the runner passes is deliberately a free-form
``Any`` rather than a typed dataclass — oracle-specific requirements
(vocab_size, reference tokens, logits tensors in P-4.3) change per
kind and carrying all of them in one typed object would bake
premature structure into the oracle seam.
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


def b1_parity_oracle(
    scenario: Scenario, token_ids: list[int], context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """B=1 parity oracle: batched token stream must equal the
    single-request reference stream element-by-element.

    ``context`` must be a mapping containing ``"reference_tokens"``
    (the single-request output) and ``"vocab_size"`` (for
    defence-in-depth bounds reporting in metadata). The runner is
    responsible for driving both executions with byte-identical
    :class:`SamplingParams`; divergence at this oracle means the
    scheduler's B=1 handling drifted from the engine's
    single-request path, regardless of kernel stability.

    Success metadata carries ``reference_len``, ``batch_len``, and
    ``first_mismatch_index=-1``. Failure metadata keeps the same
    three keys so a reviewer can see the exact divergence position
    plus both stream lengths in the JSONL row without re-running.

    The ``first_mismatch_index`` convention:
      * ``-1`` — streams agree.
      * ``0..min(len(a), len(b))-1`` — first position where tokens
        differ.
      * ``min(len(a), len(b))`` — one stream is a strict prefix of
        the other (lengths differ, no mismatch before the shorter
        stream ends).
    """
    if not isinstance(context, dict):
        return (False, "b1_parity_missing_context", {})
    if "reference_tokens" not in context:
        return (
            False,
            "b1_parity_missing_context_reference_tokens",
            {},
        )

    reference = list(context["reference_tokens"])
    ref_len = len(reference)
    batch_len = len(token_ids)
    common = min(ref_len, batch_len)

    for i in range(common):
        if reference[i] != token_ids[i]:
            return (
                False,
                f"b1_parity_first_mismatch_index:{i}",
                {
                    "reference_len": ref_len,
                    "batch_len": batch_len,
                    "first_mismatch_index": i,
                    "reference_token_at_mismatch": reference[i],
                    "batch_token_at_mismatch": token_ids[i],
                },
            )

    if ref_len != batch_len:
        return (
            False,
            f"b1_parity_length_mismatch:ref={ref_len}_batch={batch_len}",
            {
                "reference_len": ref_len,
                "batch_len": batch_len,
                "first_mismatch_index": common,
            },
        )

    return (
        True,
        None,
        {
            "reference_len": ref_len,
            "batch_len": batch_len,
            "first_mismatch_index": -1,
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
    OracleKind.B1_PARITY_VS_SINGLE: b1_parity_oracle,
    OracleKind.BGT1_DIRECT_BATCHED_REFERENCE: _not_implemented("P-4.2c"),
    OracleKind.TEACHER_FORCED_ARGMAX: _not_implemented("P-4.3"),
}


__all__ = ["smoke_oracle", "b1_parity_oracle", "ORACLES"]
