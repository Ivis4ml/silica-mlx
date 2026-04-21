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
  * :class:`OracleKind.BGT1_DIRECT_BATCHED_REFERENCE` (P-4.2c) —
    B>1 Silica batched per-row output must equal a direct mlx-lm
    batched reference driven with the same
    ``adapter.make_batch_cache(left_padding)`` list. Runner drives
    both paths with ``stop_token_ids=()`` so the direct reference
    (which runs unconditionally for ``max_tokens``) and Silica's
    stream stay length-aligned regardless of EOS.

Not yet implemented (registered as stubs that raise):

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


def bgt1_direct_batched_reference_oracle(
    scenario: Scenario,
    batch_tokens: dict[int, list[int]],
    context: Any,
) -> tuple[bool, str | None, dict[str, Any]]:
    """B>1 parity: Silica batched per-row output equals a direct
    mlx-lm batched reference run with the same cache list.

    ``batch_tokens`` is the Silica-side output keyed by row index.
    ``context`` must be a mapping containing ``"reference_tokens"``
    (same shape, ``dict[int, list[int]]``) and ``"vocab_size"``.

    The claim the runner stands behind is *scheduler glue
    equivalence to a direct batched execution path* — not
    batched-vs-solo parity. See ``tests/test_p3_gemma4_batched_parity.py``
    §test_bgt1_matches_direct_mlx_lm_batched_reference for the
    rationale: 4-bit greedy can drift between batched and solo
    runs, but the direct reference isolates scheduler correctness
    from upstream kernel noise.

    Mismatch reporting is per-row: metadata carries ``rows`` (a
    list of per-row summaries) and, on failure, the first row index
    that diverged plus the first mismatch position within that row.
    Successful runs populate ``first_mismatch`` as ``None``.
    """
    if not isinstance(context, dict):
        return (False, "bgt1_parity_missing_context", {})
    if "reference_tokens" not in context:
        return (
            False,
            "bgt1_parity_missing_context_reference_tokens",
            {},
        )
    if not isinstance(batch_tokens, dict):
        return (
            False,
            f"bgt1_parity_batch_tokens_not_dict:"
            f"{type(batch_tokens).__name__}",
            {},
        )

    reference = context["reference_tokens"]
    if not isinstance(reference, dict):
        return (
            False,
            f"bgt1_parity_reference_not_dict:{type(reference).__name__}",
            {},
        )

    batch_rows = set(batch_tokens)
    ref_rows = set(reference)
    if batch_rows != ref_rows:
        return (
            False,
            "bgt1_parity_row_set_mismatch:"
            f"batch={sorted(batch_rows)}_ref={sorted(ref_rows)}",
            {
                "batch_row_set": sorted(batch_rows),
                "reference_row_set": sorted(ref_rows),
            },
        )

    rows_metadata: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None

    for row in sorted(batch_rows):
        b = batch_tokens[row]
        r = reference[row]
        common = min(len(b), len(r))
        mismatch_index: int | None = None
        for i in range(common):
            if b[i] != r[i]:
                mismatch_index = i
                break
        if mismatch_index is None and len(b) != len(r):
            # One is a strict prefix of the other; mark the common
            # length as the "mismatch" position, same convention as
            # b1_parity_oracle.
            mismatch_index = common

        row_entry: dict[str, Any] = {
            "row": row,
            "batch_len": len(b),
            "reference_len": len(r),
            "first_mismatch_index": (
                -1 if mismatch_index is None else mismatch_index
            ),
        }
        if mismatch_index is not None and first_failure is None:
            row_entry["batch_token_at_mismatch"] = (
                b[mismatch_index] if mismatch_index < len(b) else None
            )
            row_entry["reference_token_at_mismatch"] = (
                r[mismatch_index] if mismatch_index < len(r) else None
            )
            first_failure = row_entry
        rows_metadata.append(row_entry)

    if first_failure is not None:
        return (
            False,
            f"bgt1_parity_row_{first_failure['row']}_mismatch_index:"
            f"{first_failure['first_mismatch_index']}",
            {
                "rows": rows_metadata,
                "first_failure": first_failure,
            },
        )

    return (
        True,
        None,
        {
            "rows": rows_metadata,
            "first_failure": None,
        },
    )


def _not_implemented(phase: str) -> OracleFn:
    """Build a stub oracle that names the phase where it lands."""

    def _stub(
        scenario: Scenario, token_ids: Any, context: Any
    ) -> tuple[bool, str | None, dict[str, Any]]:
        raise NotImplementedError(
            f"Oracle {scenario.oracle.value!r} for scenario "
            f"{scenario.id!r} lands in {phase}."
        )

    return _stub


ORACLES: dict[OracleKind, OracleFn] = {
    OracleKind.SMOKE: smoke_oracle,
    OracleKind.B1_PARITY_VS_SINGLE: b1_parity_oracle,
    OracleKind.BGT1_DIRECT_BATCHED_REFERENCE: bgt1_direct_batched_reference_oracle,
    OracleKind.TEACHER_FORCED_ARGMAX: _not_implemented("P-4.3"),
}


__all__ = [
    "smoke_oracle",
    "b1_parity_oracle",
    "bgt1_direct_batched_reference_oracle",
    "ORACLES",
]
