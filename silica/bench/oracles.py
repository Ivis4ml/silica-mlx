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

  * :class:`OracleKind.TEACHER_FORCED_ARGMAX` (P-4.3) — for a
    fixed prompt + target continuation, the Silica adapter's
    position-by-position next-token argmax (driven by
    ``adapter.prefill`` + ``decode_step`` with teacher-forced
    targets) must agree with a direct mlx-lm single-forward
    reference at >= ``min_agreement_rate`` of positions. PLAN §P-3
    exit criterion; the oracle is pure (runner supplies both
    streams) with structured agreement metadata.

All registered oracle kinds are now implemented.

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

import math
from typing import Any

from silica.bench.scenario import OracleFn, OracleKind, Scenario


def smoke_oracle(
    scenario: Scenario, token_ids: Any, context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """SMOKE oracle: did the workload emit at least one valid token?

    ``token_ids`` may be either:

      * ``list[int]`` — single-request output (B=1 SMOKE path).
      * ``dict[int, list[int]]`` — per-row output (B>1 SMOKE via
        ``Engine.generate_batch``). Every row is validated
        independently; an empty row fails the oracle with a
        ``smoke_row_X_no_tokens_emitted`` reason so concurrent /
        shared-prefix scenarios cannot silently mask a scheduler
        bug that drops a request.

    ``context`` must be a mapping containing ``"vocab_size"``. The
    runner assembles it from ``adapter.config.vocab_size`` before
    invoking the oracle.

    Metadata on success carries ``total_tokens`` and the max id
    observed so the JSONL row is self-describing — a reviewer can
    tell at a glance whether the "4 tokens, all < 150000" claim was
    actually met without re-running the scenario. The batched path
    additionally populates a ``rows`` list with the per-row token
    count keyed by ``row``.
    """
    if not isinstance(context, dict) or "vocab_size" not in context:
        return (
            False,
            "smoke_oracle_missing_context_vocab_size",
            {},
        )
    vocab_size = int(context["vocab_size"])
    if isinstance(token_ids, dict):
        # Runner instruments per-row first-token wall-clock
        # offsets; surface them through per-row metadata when
        # available so the JSONL row's ``rows[].first_token_ms_offset``
        # carries TTFT-under-concurrency signal. ``Engine.generate_batch``
        # does not populate ``MetricsRegistry`` so this is the only
        # place batched TTFT actually shows up in the JSONL.
        first_token_ms = (
            context.get("first_token_ms_per_row")
            if isinstance(context, dict)
            else None
        )
        return _smoke_batched(
            token_ids, vocab_size, first_token_ms_per_row=first_token_ms
        )
    return _smoke_single(token_ids, vocab_size)


def _smoke_single(
    token_ids: list[int], vocab_size: int
) -> tuple[bool, str | None, dict[str, Any]]:
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


def _smoke_batched(
    rows: dict[int, list[int]],
    vocab_size: int,
    *,
    first_token_ms_per_row: dict[int, float] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    total = 0
    max_id = 0
    rows_metadata: list[dict[str, Any]] = []
    for row_idx in sorted(rows):
        row_tokens = rows[row_idx]
        if not row_tokens:
            return (
                False,
                f"smoke_row_{row_idx}_no_tokens_emitted",
                {"vocab_size": vocab_size, "row": row_idx},
            )
        for i, tok in enumerate(row_tokens):
            if not isinstance(tok, int):
                return (
                    False,
                    f"smoke_row_{row_idx}_token_{i}_not_int:"
                    f"{type(tok).__name__}",
                    {"vocab_size": vocab_size, "row": row_idx, "index": i},
                )
            if tok < 0 or tok >= vocab_size:
                return (
                    False,
                    f"smoke_row_{row_idx}_token_{i}_out_of_vocab:{tok}",
                    {"vocab_size": vocab_size, "row": row_idx, "index": i},
                )
            if tok > max_id:
                max_id = tok
        total += len(row_tokens)
        row_entry: dict[str, Any] = {
            "row": row_idx,
            "token_count": len(row_tokens),
        }
        if first_token_ms_per_row is not None and row_idx in first_token_ms_per_row:
            row_entry["first_token_ms_offset"] = round(
                first_token_ms_per_row[row_idx], 3
            )
        rows_metadata.append(row_entry)
    return (
        True,
        None,
        {
            "total_tokens": total,
            "max_token_id": max_id,
            "vocab_size": vocab_size,
            "rows": rows_metadata,
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


def teacher_forced_argmax_oracle(
    scenario: Scenario,
    silica_predictions: Any,
    context: Any,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Teacher-forced argmax parity: silica adapter path's positional
    next-token argmax must agree with a direct mlx-lm reference.

    ``silica_predictions`` is a ``list[int]`` of length N with the
    argmax at each teacher-forced position produced by
    ``adapter.prefill`` + ``decode_step`` with the target tokens
    fed deterministically.

    ``context`` must contain:

      * ``reference_predictions`` — ``list[int]`` of the same
        length, coming from a single direct mlx-lm forward
        (``model(prompt + target[:-1], cache=fresh)``) with its
        positional logits sliced to N predictions.
      * ``target_tokens`` — the teacher-forcing target ids (used
        only for metadata reporting, not for the pass/fail
        decision — the oracle measures silica-vs-reference, not
        silica-vs-expected-continuation).
      * ``min_agreement_rate`` — float in [0, 1]; oracle passes
        when the fraction of positions where silica == reference
        is >= this threshold. PLAN §P-3 set this at 0.98 to
        absorb fp16 + quantization numerical jitter.
      * ``vocab_size`` — for defence-in-depth vocabulary bounds
        reporting.

    Metadata always includes the agreement rate, per-stream
    lengths, the first mismatch position (``-1`` when all match),
    the configured threshold, and a ``target_match_rate`` that
    separately reports how often the silica path also agreed with
    the teacher-forced target (informative, non-normative).
    """
    if not isinstance(context, dict):
        return (False, "teacher_forced_argmax_missing_context", {})
    for key in ("reference_predictions", "min_agreement_rate", "target_tokens"):
        if key not in context:
            return (
                False,
                f"teacher_forced_argmax_missing_context_{key}",
                {},
            )
    if not isinstance(silica_predictions, list):
        return (
            False,
            "teacher_forced_argmax_silica_predictions_not_list:"
            f"{type(silica_predictions).__name__}",
            {},
        )

    reference = list(context["reference_predictions"])
    target = list(context["target_tokens"])
    threshold = float(context["min_agreement_rate"])

    silica_len = len(silica_predictions)
    ref_len = len(reference)
    if silica_len != ref_len:
        return (
            False,
            f"teacher_forced_argmax_length_mismatch:"
            f"silica={silica_len}_reference={ref_len}",
            {
                "silica_len": silica_len,
                "reference_len": ref_len,
                "min_agreement_rate": threshold,
            },
        )
    if silica_len == 0:
        return (
            False,
            "teacher_forced_argmax_empty_predictions",
            {"min_agreement_rate": threshold},
        )

    matches = 0
    first_mismatch = -1
    for i, (s, r) in enumerate(zip(silica_predictions, reference, strict=True)):
        if s == r:
            matches += 1
        elif first_mismatch == -1:
            first_mismatch = i
    agreement_rate = matches / silica_len

    target_matches = 0
    if len(target) == silica_len:
        for s, t in zip(silica_predictions, target, strict=True):
            if s == t:
                target_matches += 1
        target_match_rate = target_matches / silica_len
    else:
        # Mismatched target length is not a failure — the target is
        # informational. Report absent so the reader sees why.
        target_match_rate = None

    metadata: dict[str, Any] = {
        "length": silica_len,
        "matches": matches,
        "agreement_rate": agreement_rate,
        "first_mismatch_index": first_mismatch,
        "min_agreement_rate": threshold,
        "target_match_rate": target_match_rate,
    }
    if agreement_rate < threshold:
        return (
            False,
            f"teacher_forced_argmax_agreement_below_threshold:"
            f"{agreement_rate:.3f}<{threshold:.3f}",
            metadata,
        )
    return (True, None, metadata)


def decode_tok_s_with_prefix_hit_oracle(
    scenario: Scenario, collected: Any, context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """P-5-A.3b oracle: reports row 1's decode tok/s on the prefix-hit
    admission path and guards the measurement against trivially-passing
    scenarios (zero prefix hits, truncated rows, invalid tokens).

    Workload contract (validated by the runner's
    ``_run_prefix_hit_decode``): exactly 2 identical prompts,
    ``max_batch_size=1``, ``prefix_cache=True``. Row 0 miss-path
    prefills + decodes; row 1 enters the waiting queue and is
    admitted mid-run through ``_admit_single_hit_row``; its decode
    loop exercises the codec on admission (seeded-prefix K/V) and
    on every step append.

    ``collected`` is the ``(tokens, token_ts_ms)`` pair from the
    runner's collector. ``context`` must be a mapping with
    ``vocab_size`` + ``prefix_cache_hits`` (the radix cache's
    ``hits`` counter at collection end).

    Metric definition:

    - ``row1_decode_tok_s = (N - 1) / ((t_last - t_first) / 1000.0)``
      where ``N = len(token_ts_ms[1])`` and timestamps are in ms
      from the start of ``generate_batch``. Dividing by ``N - 1``
      (not ``N``) and using ``t_last - t_first`` (not ``t_last``)
      excludes row 1's first-token latency, which is dominated by
      the seeded-admission overhead (prefix decode × num_layers) —
      not the steady-state decode tok/s the gate is comparing.
    - ``row1_first_token_ms = token_ts_ms[1][0]`` is surfaced
      separately in metadata so the seeded-admission overhead can
      be inspected side-by-side with the steady-state throughput.

    Gates:

    - Every row must emit ≥ 1 valid token (token_id is int in vocab
      range).
    - Row 1 must emit ≥ 2 tokens — the inter-token interval formula
      needs at least two samples; 1-token rows would divide by zero
      and silently report infinity.
    - ``prefix_cache_hits >= 1`` — if the hit counter is zero, the
      workload somehow ran the miss path on row 1 (scheduler
      regression or a scenario-authoring error); the reported
      decode tok/s would reflect a different path than the
      measurement claims.

    The 0.85× ratio gate itself is not part of this oracle — it is
    a comparison between two scenarios (BlockTQ vs fp16 baseline),
    driven by the A.3c acceptance test. This oracle's job is to
    report an honest row-1 decode tok/s.
    """
    if not isinstance(context, dict):
        return (False, "decode_tok_s_missing_context", {})
    if "vocab_size" not in context or "prefix_cache_hits" not in context:
        return (
            False,
            "decode_tok_s_context_missing_required_keys",
            {"keys_present": sorted(context)},
        )
    vocab_size = int(context["vocab_size"])
    prefix_cache_hits = int(context["prefix_cache_hits"])

    if (
        not isinstance(collected, tuple)
        or len(collected) != 2
        or not isinstance(collected[0], dict)
        or not isinstance(collected[1], dict)
    ):
        return (False, "decode_tok_s_collected_shape_mismatch", {})
    tokens, token_ts_ms = collected

    # Shape + vocab validation per row.
    for row_idx in sorted(tokens):
        row_tokens = tokens[row_idx]
        if not row_tokens:
            return (
                False,
                f"decode_tok_s_row_{row_idx}_no_tokens_emitted",
                {},
            )
        for i, tok in enumerate(row_tokens):
            if not isinstance(tok, int):
                return (
                    False,
                    f"decode_tok_s_row_{row_idx}_token_{i}_not_int:"
                    f"{type(tok).__name__}",
                    {},
                )
            if tok < 0 or tok >= vocab_size:
                return (
                    False,
                    f"decode_tok_s_row_{row_idx}_token_{i}_out_of_vocab:"
                    f"{tok}",
                    {},
                )

    if 1 not in tokens or 1 not in token_ts_ms:
        return (False, "decode_tok_s_row_1_missing", {})
    row1_ts = token_ts_ms[1]
    if len(row1_ts) < 2:
        return (
            False,
            "decode_tok_s_row_1_needs_at_least_2_tokens_for_interval",
            {"row1_tokens": len(row1_ts)},
        )

    if prefix_cache_hits < 1:
        return (
            False,
            "decode_tok_s_prefix_cache_never_hit",
            {"prefix_cache_hits": prefix_cache_hits},
        )

    # Steady-state inter-token throughput excluding first-token.
    interval_s = (row1_ts[-1] - row1_ts[0]) / 1000.0
    if interval_s <= 0.0:
        return (
            False,
            "decode_tok_s_row_1_nonpositive_interval",
            {"interval_s": interval_s},
        )
    row1_decode_tok_s = (len(row1_ts) - 1) / interval_s
    row1_first_token_ms = row1_ts[0]

    # Row 0's steady-state — miss path, for side-by-side comparison.
    row0_ts = token_ts_ms.get(0, [])
    if len(row0_ts) >= 2:
        row0_interval_s = (row0_ts[-1] - row0_ts[0]) / 1000.0
        row0_decode_tok_s: float | None = (
            (len(row0_ts) - 1) / row0_interval_s
            if row0_interval_s > 0.0
            else None
        )
        row0_first_token_ms: float | None = row0_ts[0]
    else:
        row0_decode_tok_s = None
        row0_first_token_ms = None

    metadata: dict[str, Any] = {
        "row1_decode_tok_s": row1_decode_tok_s,
        "row1_first_token_ms": row1_first_token_ms,
        "row0_decode_tok_s": row0_decode_tok_s,
        "row0_first_token_ms": row0_first_token_ms,
        "row0_tokens": len(tokens.get(0, [])),
        "row1_tokens": len(tokens[1]),
        "prefix_cache_hits": prefix_cache_hits,
    }
    return (True, None, metadata)


def ppl_oracle(
    scenario: Scenario, collected: Any, context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """P-5-C.2 oracle: structural validation of the PPL result
    produced by the runner's ``_run_ppl`` driver.

    The driver computes ``(nll_sum, n_tokens)`` via
    :func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll`
    (fp16 baseline when ``workload.kv_codec is None``) or
    :func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec`
    (codec-backed when ``workload.kv_codec`` names a registry entry),
    then wraps the perplexity via :func:`perplexity_from_nll` into
    ``collected = {"nll_sum", "n_tokens", "ppl"}``.

    Validation here is **structural only**: field presence, types,
    finite ``ppl``, non-negative ``n_tokens``, and an optional floor
    from ``scenario.oracle_config["min_scored_tokens"]`` (defaults to
    1 — a zero-scored-token run is almost always a caller-side
    configuration bug).

    The ΔPPL magnitude gate — "codec ΔPPL against fp16 baseline is
    below ε_ppl" — is C.6's vqbench cross-check responsibility, not
    this oracle's. When ``context["ppl_fp16"]`` is supplied the
    oracle surfaces ``delta_ppl`` / ``delta_ppl_pct`` in metadata
    for downstream consumers (bench report, C.6 acceptance); it does
    not gate on their values here.

    Args:
        scenario: the bench scenario carrying ``oracle_config``
            (reads ``min_scored_tokens``).
        collected: the per-row result dict the runner produced; must
            contain ``nll_sum`` (float), ``n_tokens`` (int), and
            ``ppl`` (float).
        context: runner-supplied context dict. Reads the optional
            ``ppl_fp16`` for ΔPPL computation; also copies all keys
            into metadata.

    Returns:
        ``(ok, reason, metadata)``. Metadata surfaces the full
        collected numbers + the context's chunk_size / max_tokens /
        wikitext_path / kv_codec plus the computed delta_ppl when
        available.
    """
    if not isinstance(collected, dict):
        return (False, "ppl_collected_shape_mismatch", {})
    for field in ("nll_sum", "n_tokens", "ppl"):
        if field not in collected:
            return (
                False,
                f"ppl_collected_missing_field:{field}",
                {"fields_present": sorted(collected)},
            )

    try:
        nll_sum = float(collected["nll_sum"])
        n_tokens = int(collected["n_tokens"])
        ppl = float(collected["ppl"])
    except (TypeError, ValueError) as exc:
        return (
            False,
            f"ppl_collected_field_not_castable:{exc!s}",
            {},
        )

    if n_tokens < 0:
        return (
            False,
            "ppl_n_tokens_negative",
            {"n_tokens": n_tokens},
        )

    min_scored_tokens = int(
        scenario.oracle_config.get("min_scored_tokens", 1)
    )
    if n_tokens < min_scored_tokens:
        return (
            False,
            "ppl_n_tokens_below_min_scored_tokens",
            {
                "n_tokens": n_tokens,
                "min_scored_tokens": min_scored_tokens,
            },
        )

    if not math.isfinite(ppl) or ppl < 0.0:
        return (
            False,
            "ppl_not_finite_or_negative",
            {"ppl": ppl},
        )

    metadata: dict[str, Any] = {
        "nll_sum": nll_sum,
        "n_tokens": n_tokens,
        "ppl": ppl,
    }
    # Forward context fields the runner supplied (chunk_size,
    # max_tokens, wikitext_path, kv_codec, ppl_fp16) so the bench
    # report can show them alongside the PPL number without the
    # oracle having to know every possible key.
    if isinstance(context, dict):
        for key, value in context.items():
            metadata.setdefault(key, value)

    # ΔPPL computation when a baseline is supplied in ``context``.
    # Step 3a does NOT propagate the fp16 baseline between rows at
    # runtime — ``_run_ppl`` never populates ``ppl_fp16``, so every
    # in-tree PPL row currently records ``delta_ppl = None``. The
    # plumbing here is future-facing: a downstream bench-report
    # pass (or the C.6 vqbench cross-check) can call this oracle
    # with a populated ``ppl_fp16`` context when comparing codec
    # rows to their baseline, and delta / delta_pct appear in
    # metadata. Keeping the computation here rather than in the
    # report layer means ``ppl`` and ``delta_ppl`` stay produced
    # by one function, so they cannot drift.
    ppl_fp16 = None
    if isinstance(context, dict):
        raw = context.get("ppl_fp16")
        if raw is not None:
            try:
                ppl_fp16 = float(raw)
            except (TypeError, ValueError):
                ppl_fp16 = None

    if ppl_fp16 is not None and math.isfinite(ppl_fp16) and ppl_fp16 > 0.0:
        metadata["delta_ppl"] = ppl - ppl_fp16
        metadata["delta_ppl_pct"] = (ppl - ppl_fp16) / ppl_fp16 * 100.0
    else:
        metadata.setdefault("delta_ppl", None)
        metadata.setdefault("delta_ppl_pct", None)

    return (True, None, metadata)


def storage_oracle(
    scenario: Scenario, collected: Any, context: Any
) -> tuple[bool, str | None, dict[str, Any]]:
    """P-5-C.3 oracle: structural validation of the prefix-cache
    store residency snapshot the runner's ``_run_storage`` driver
    produces after a shared-prefix 2-prompt workload has populated
    the store.

    The runner collects:

    - ``resident_bytes`` — ``store.resident_bytes()``, the sum of
      per-side ``CodedPayload.resident_bytes`` across every detached
      block. Under ``kv_codec="fp16"`` (IdentityCodec) this equals
      ``num_layers × num_blocks × block_size × (2 × n_kv_heads ×
      head_dim × dtype.size)``; under a non-identity codec it
      reflects compressed residency, so the fp16 row is the
      directly-comparable baseline.
    - ``resident_bytes_per_block`` — ``store.resident_bytes_per_block()``
      or ``None`` (the latter under pass-through stores with no
      codec, not produced by the bench rows since all four
      ``qwen3-0.6b-compression-*`` rows pin an explicit codec).
    - ``live_blocks`` — ``len(store.live_block_ids())``.
    - ``prefix_cache_hits`` — ``prefix_cache.hits``, confirming the
      hit path actually fired during the 2-prompt workload.

    Validation is **pure structural**: field presence, correct
    types, ``resident_bytes >= 0``, ``live_blocks >= 1`` (empty-store
    shape guard — a workload that "ran but did not register any
    detached prefix" is noise, not a measurement; the runner
    already surfaces the zero-block case as a ``RuntimeError`` so
    reaching the oracle with ``live_blocks == 0`` means a silent
    regression elsewhere). ``resident_bytes_per_block`` must be
    ``None`` OR a positive int.

    Compression-ratio gate (e.g. ``block_tq_b64_b4`` resident_bytes
    < ``fp16`` resident_bytes) is NOT enforced here — cross-row
    numeric comparisons live in the bench report / C.6 vqbench
    cross-check layers. The oracle's job is to guarantee the
    per-row numbers are honest and reportable.
    """
    if not isinstance(collected, dict):
        return (False, "storage_collected_shape_mismatch", {})
    for field in (
        "resident_bytes",
        "resident_bytes_per_block",
        "live_blocks",
        "prefix_cache_hits",
    ):
        if field not in collected:
            return (
                False,
                f"storage_collected_missing_field:{field}",
                {"fields_present": sorted(collected)},
            )

    try:
        resident_bytes = int(collected["resident_bytes"])
        live_blocks = int(collected["live_blocks"])
        prefix_cache_hits = int(collected["prefix_cache_hits"])
    except (TypeError, ValueError) as exc:
        return (
            False,
            f"storage_collected_field_not_castable:{exc!s}",
            {},
        )

    raw_bpb = collected["resident_bytes_per_block"]
    if raw_bpb is None:
        resident_bytes_per_block: int | None = None
    else:
        try:
            resident_bytes_per_block = int(raw_bpb)
        except (TypeError, ValueError) as exc:
            return (
                False,
                f"storage_collected_field_not_castable:{exc!s}",
                {},
            )
        if resident_bytes_per_block <= 0:
            return (
                False,
                "storage_resident_bytes_per_block_not_positive",
                {"resident_bytes_per_block": resident_bytes_per_block},
            )

    if resident_bytes < 0:
        return (
            False,
            "storage_resident_bytes_negative",
            {"resident_bytes": resident_bytes},
        )

    if live_blocks < 1:
        return (
            False,
            "storage_live_blocks_below_one",
            {"live_blocks": live_blocks},
        )

    if prefix_cache_hits < 0:
        return (
            False,
            "storage_prefix_cache_hits_negative",
            {"prefix_cache_hits": prefix_cache_hits},
        )

    metadata: dict[str, Any] = {
        "resident_bytes": resident_bytes,
        "resident_bytes_per_block": resident_bytes_per_block,
        "live_blocks": live_blocks,
        "prefix_cache_hits": prefix_cache_hits,
    }
    if isinstance(context, dict):
        for key, value in context.items():
            metadata.setdefault(key, value)
    return (True, None, metadata)


ORACLES: dict[OracleKind, OracleFn] = {
    OracleKind.SMOKE: smoke_oracle,
    OracleKind.B1_PARITY_VS_SINGLE: b1_parity_oracle,
    OracleKind.BGT1_DIRECT_BATCHED_REFERENCE: bgt1_direct_batched_reference_oracle,
    OracleKind.TEACHER_FORCED_ARGMAX: teacher_forced_argmax_oracle,
    OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT: decode_tok_s_with_prefix_hit_oracle,
    OracleKind.PPL: ppl_oracle,
    OracleKind.STORAGE: storage_oracle,
}


__all__ = [
    "smoke_oracle",
    "b1_parity_oracle",
    "bgt1_direct_batched_reference_oracle",
    "teacher_forced_argmax_oracle",
    "decode_tok_s_with_prefix_hit_oracle",
    "ppl_oracle",
    "storage_oracle",
    "ORACLES",
]
