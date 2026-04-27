"""Tests for P-4.5-B.1 admission-reorder under Q-010 fairness fix.

Split into five sections, matching `plans/P4_5_CHUNKED_PREFILL_OPENING.md` §8:

  1. `_sort_admissions_by_length` stability + req_index preservation.
  2. `_initial_cohort_cap` preconditions + worked reverse examples from
     the opening doc §6.1.
  3. `generate_batch` end-to-end wiring — default threshold vs opt-out,
     req_index preserved through event stream, heterogeneous batches
     actually route long prompts through the mid-run-admit path.
  4. Dual-gated on-device direct mlx-lm sub-cohort reference test
     (`test_reordered_cohort_matches_mlx_lm_direct_batched_reference`)
     — PLAN §7 P-4.5 Acceptance (c) numerical reference: Silica's
     per-row tokens on the Q-010 catalog workload must match a
     direct mlx-lm batched reference run over each sub-cohort
     scoped by Option (C). Inverse-permutation index mapping from
     sub-cohort-local index back to original req_index.
  5. Dual-gated on-device Q-010 acceptance harness
     (`test_q010_ratio_below_threshold_on_five_runs`) — gated on the
     0.6B HF cache to match every other P-4 bench-backed test. The
     ratio assertion lives here so the acceptance is part of the
     regression lock, not only a manual bench command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica.core.sampling import SamplingParams
from silica.engine import (
    Engine,
    _initial_cohort_cap,
    _sort_admissions_by_length,
)
from silica.kvcache.manager import NullKVManager
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)

# --- Scripted fake engine fixtures (mirrors tests/test_engine_generate_batch.py)


class _PerPromptTokenizer:
    """Tokenizer whose encode returns a prompt-specific length.

    Accepts a mapping ``{prompt_text: token_ids}``. Unknown prompts
    encode to ``[]`` so the admission filter drops them. Every
    prompt-specific return is a fresh list so the caller can mutate
    it without affecting the tokenizer's state.
    """

    vocab_size = 32

    def __init__(self, lengths_by_prompt: dict[str, list[int]]) -> None:
        self._map = {k: list(v) for k, v in lengths_by_prompt.items()}

    def encode(self, text: str) -> list[int]:
        if text in self._map:
            return list(self._map[text])
        return []

    def decode(self, ids: Any) -> str:
        return ""


class _Model:
    """Fake model that emits a constant next-token for every call."""

    VOCAB = 32

    def __init__(self, script: Sequence[int]) -> None:
        self.script: list[int] = list(script)
        self.calls: list[tuple[int, int]] = []

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        self.calls.append((B, T))
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            v = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        target = self.script.pop(0) if self.script else 0
        one_hot = mx.zeros((self.VOCAB,), dtype=mx.float32)
        one_hot[target] = 1.0
        return mx.broadcast_to(
            one_hot.reshape(1, 1, self.VOCAB),
            (B, T, self.VOCAB),
        )


class _Adapter:
    def __init__(
        self,
        lengths_by_prompt: dict[str, list[int]],
        script: Sequence[int] = (),
        n_layers: int = 1,
    ) -> None:
        self.config = ModelConfig(
            model_name="scripted",
            num_layers=n_layers,
            hidden_size=16,
            vocab_size=_Model.VOCAB,
        )
        self._model = _Model(script=script)
        self._tok = _PerPromptTokenizer(lengths_by_prompt)
        self._pattern = AttentionPattern(
            per_layer=tuple(AttentionKind.GLOBAL for _ in range(n_layers))
        )

    def build(self, weight_provider: Any) -> Any:
        return self._model

    def kv_layout(self) -> KVLayout:
        return KVLayout(
            num_layers=self.config.num_layers,
            n_kv_heads=1,
            head_dim=4,
            dtype=mx.float16,
        )

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def capabilities(self) -> ModelCapabilities:
        return capabilities_from_attention_pattern(self._pattern)

    def tokenizer(self) -> _PerPromptTokenizer:
        return self._tok

    def prefill(
        self, tokens: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:
        raise NotImplementedError  # pragma: no cover

    def decode_step(
        self, token: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:
        raise NotImplementedError  # pragma: no cover


def _make_engine(
    lengths_by_prompt: dict[str, list[int]],
    script: Sequence[int] = (1,),
) -> tuple[Engine, _Adapter]:
    adapter = _Adapter(lengths_by_prompt=lengths_by_prompt, script=script)
    return Engine(adapter, NullKVManager()), adapter


# --- Section 1. _sort_admissions_by_length ----------------------------------


def test_sort_preserves_req_index_on_tuples() -> None:
    """Sort reorders the list but keeps (req_index, ids) tuples intact."""
    admissions = [(0, [1, 2, 3, 4]), (1, [1]), (2, [1, 2])]
    out = _sort_admissions_by_length(admissions)
    assert [ri for ri, _ in out] == [1, 2, 0]
    # Tuples are intact — no silent reindexing.
    for ri, ids in out:
        assert admissions[ri] == (ri, ids)


def test_sort_is_stable_on_tied_lengths() -> None:
    """Tied lengths preserve the user-supplied order (stable)."""
    admissions = [(0, [1]), (1, [1]), (2, [1]), (3, [1])]
    out = _sort_admissions_by_length(admissions)
    # Python's sort is stable; all lengths are 1, so req_index 0..3.
    assert [ri for ri, _ in out] == [0, 1, 2, 3]


def test_sort_empty_input_returns_empty() -> None:
    assert _sort_admissions_by_length([]) == []


def test_sort_single_input_returns_single() -> None:
    assert _sort_admissions_by_length([(7, [1, 2])]) == [(7, [1, 2])]


def test_sort_returns_new_list_not_input_alias() -> None:
    """`sorted()` returns a fresh list; callers can rely on that."""
    admissions = [(0, [1, 2]), (1, [1])]
    out = _sort_admissions_by_length(admissions)
    out.append((999, [999]))
    assert admissions == [(0, [1, 2]), (1, [1])]  # unchanged


def test_inverse_permutation_round_trip() -> None:
    """Direct-batched reference helpers return tokens by sub-cohort-local
    index; Silica's event stream emits original req_index. The helper's
    sort permutation must be invertible so tests comparing tokens by
    position can map back.
    """
    original = [(0, [1]), (1, [1, 2, 3]), (2, [1, 2]), (3, [1, 2, 3, 4])]
    sorted_ = _sort_admissions_by_length(original)
    # Build: sub_cohort_local_index → original_req_index
    local_to_original = [ri for ri, _ in sorted_]
    # Inverse: original_req_index → sub_cohort_local_index
    original_to_local = {ri: i for i, ri in enumerate(local_to_original)}
    # Round-trip
    for i, (ri, _) in enumerate(sorted_):
        assert original_to_local[ri] == i
    # Every original req_index reachable.
    assert set(original_to_local) == {0, 1, 2, 3}


# --- Section 2. _initial_cohort_cap -----------------------------------------


def test_cap_rejects_effective_batch_size_lt_1() -> None:
    with pytest.raises(ValueError, match="effective_batch_size must be >= 1"):
        _initial_cohort_cap([(0, [1])], 0, 2.0)


def test_cap_rejects_threshold_le_1() -> None:
    with pytest.raises(ValueError, match="length_spread_threshold must be > 1.0"):
        _initial_cohort_cap([(0, [1]), (1, [2])], 2, 1.0)
    with pytest.raises(ValueError, match="length_spread_threshold must be > 1.0"):
        _initial_cohort_cap([(0, [1]), (1, [2])], 2, 0.5)


def test_cap_rejects_nan_threshold() -> None:
    """NaN comparisons always yield False, which would silently
    disable the split (every ``len > NaN*min_len`` check returns
    False) — equivalent to ``float('inf')`` but without the caller
    asking for it. Reject NaN explicitly so a mistyped threshold
    surfaces loudly.
    """
    import math as _math

    with pytest.raises(ValueError, match="must not be NaN"):
        _initial_cohort_cap([(0, [1]), (1, [2])], 2, _math.nan)
    with pytest.raises(ValueError, match="must not be NaN"):
        _initial_cohort_cap([(0, [1]), (1, [2])], 2, float("nan"))


def test_cap_accepts_inf_as_opt_out_sentinel() -> None:
    """float('inf') disables the split — the homogeneous fast path
    returns the whole admission set, bounded by effective_batch_size.
    """
    admissions = [(0, [1]), (1, [1, 2, 3, 4, 5])]  # ratio 5.0
    assert _initial_cohort_cap(admissions, 2, float("inf")) == 2


def test_cap_homogeneous_fast_path_single() -> None:
    """Single admission → min(effective_batch_size, 1)."""
    assert _initial_cohort_cap([(0, [1, 2, 3])], 4, 2.0) == 1
    assert _initial_cohort_cap([(0, [1, 2, 3])], 1, 2.0) == 1


def test_cap_homogeneous_fast_path_empty() -> None:
    assert _initial_cohort_cap([], 4, 2.0) == 0


def test_cap_homogeneous_equal_lengths() -> None:
    """Equal lengths → ratio 1.0 ≤ threshold → no split."""
    admissions = [(0, [1, 2, 3]), (1, [1, 2, 3]), (2, [1, 2, 3])]
    assert _initial_cohort_cap(admissions, 4, 2.0) == 3
    # Bounded by effective_batch_size.
    assert _initial_cohort_cap(admissions, 2, 2.0) == 2


def test_cap_homogeneous_boundary_ratio_equals_threshold() -> None:
    """Ratio exactly equals threshold → no split (uses `<=`)."""
    admissions = [(0, [1]), (1, [1, 2])]  # ratio 2.0
    assert _initial_cohort_cap(admissions, 2, 2.0) == 2


# Worked reverse examples from opening doc §6.1.


def test_cap_opening_doc_example_1_clamps_at_max_batch_size() -> None:
    """lens=[1,1,1,3], threshold=2.0, max_batch_size=2 → cap=2.

    Without the `min(effective_batch_size, ...)` clamp the naive
    first_exceeding_index return would be 3, violating max_batch_size.
    """
    admissions = [(0, [1]), (1, [2]), (2, [3]), (3, [1, 2, 3])]
    assert _initial_cohort_cap(admissions, 2, 2.0) == 2


def test_cap_opening_doc_example_2_target_q010_shape() -> None:
    """lens=[1,1,1,3], threshold=2.0, max_batch_size=4 → cap=3.

    The Q-010 shape: three short rows go pre-step, the long row queues.
    """
    admissions = [(0, [1]), (1, [2]), (2, [3]), (3, [1, 2, 3])]
    assert _initial_cohort_cap(admissions, 4, 2.0) == 3


def test_cap_opening_doc_example_3_homogeneous_no_split() -> None:
    """lens=[300,300,300,300], threshold=2.0, max_batch_size=4 → 4."""
    admissions = [(i, list(range(300))) for i in range(4)]
    assert _initial_cohort_cap(admissions, 4, 2.0) == 4


def test_cap_opening_doc_example_4_single_prompt() -> None:
    """lens=[1], threshold=2.0, max_batch_size=4 → 1.

    Single-prompt call unchanged.
    """
    assert _initial_cohort_cap([(0, [1])], 4, 2.0) == 1


def test_cap_enforces_floor_of_one() -> None:
    """max(1, ...) floor guarantees cap >= 1 even if first_exceeding
    computes to 0 under a degenerate input (defensive guard)."""
    # Every len exceeds the threshold against the first. In practice
    # this is unreachable because element 0 compared to itself never
    # exceeds, but the floor is what guarantees it.
    admissions = [(0, [1]), (1, [1, 2, 3])]  # ratio 3.0
    # first_exceeding_index = 1 → cap = min(ebs, 1).
    assert _initial_cohort_cap(admissions, 8, 2.0) == 1


def test_cap_parametrized_by_threshold() -> None:
    """Same admission list, varying threshold crosses split boundaries.

    Lengths: 1, 2, 4 tokens (ratio max/min = 4.0). The helper looks
    at the prompt-token-list length, not its contents.
    """
    admissions = [(0, [1]), (1, [1, 2]), (2, [1, 2, 3, 4])]
    # threshold=1.5 → max/min=4.0 > 1.5 split. First exceeding: index
    # where len > 1 * 1.5 = 1.5. index 1 has len 2 > 1.5. cap=1.
    assert _initial_cohort_cap(admissions, 4, 1.5) == 1
    # threshold=2.0 → ratio 4.0 > 2.0 split. First exceeding: index
    # where len > 1 * 2.0 = 2.0. index 1 has len 2 NOT > 2.0; index
    # 2 has len 4 > 2.0. cap=2.
    assert _initial_cohort_cap(admissions, 4, 2.0) == 2
    # threshold=4.0 → ratio 4.0 <= 4.0 → homogeneous fast path → 3.
    assert _initial_cohort_cap(admissions, 4, 4.0) == 3


# --- Section 3. generate_batch end-to-end wiring ----------------------------


def test_generate_batch_preserves_req_index_under_split() -> None:
    """Heterogeneous lengths trigger the split; req_index emitted on
    each event still equals the user-supplied index (not sub-cohort
    local index).
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={
            "long": [1, 2, 3, 4, 5, 6, 7, 8],  # 8 tokens
            "a": [1],
            "b": [1],
            "c": [1],
        },
        script=[7],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(
        engine.generate_batch(
            ["long", "a", "b", "c"],
            params,
            max_batch_size=4,
        )
    )
    seen_req_indices = {e.req_index for e in events if e.kind == "token"}
    assert seen_req_indices == {0, 1, 2, 3}


def test_generate_batch_inf_threshold_disables_split() -> None:
    """Opt-out via float('inf') routes all prompts through the initial
    cohort, identical pre-step to the old behaviour.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={"x": [1], "yyyyyy": [1, 2, 3, 4, 5, 6]},
        script=[9, 9, 9],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    list(
        engine.generate_batch(
            ["x", "yyyyyy"],
            params,
            max_batch_size=2,
            length_spread_threshold=float("inf"),
        )
    )
    # Single prefill forward covers both (B=2). The fake _Model.calls
    # records (B, T) per forward. Only one forward at B=2 here.
    assert adapter._model.calls[0] == (2, 6)


def test_generate_batch_inf_threshold_preserves_original_admission_order() -> None:
    """``length_spread_threshold=float('inf')`` is documented as the
    opt-out for strict-parity tests that expect the pre-P-4.5
    admission order (original user-supplied order). This test pins
    that semantic: the event stream's per-row first ``token`` event
    appears in the same order as the user's ``prompts`` list, even
    when the batch is heterogeneous enough that the default
    threshold would otherwise reorder.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={
            "long5": [1, 2, 3, 4, 5],
            "short1": [1],
        },
        script=[9, 9],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(
        engine.generate_batch(
            ["long5", "short1"],  # user's order: long first, short second
            params,
            max_batch_size=2,
            length_spread_threshold=float("inf"),
        )
    )
    # Event emission order follows row order in _rows, which under
    # opt-out is the user's original admission order. First token
    # event should be req_index=0 (long5), not req_index=1 (short1).
    first_tokens = [e for e in events if e.kind == "token"]
    assert [e.req_index for e in first_tokens] == [0, 1], (
        "opt-out (length_spread_threshold=inf) must preserve the "
        "user's original admission order; observed event stream "
        f"req_index order: {[e.req_index for e in first_tokens]}"
    )


def test_generate_batch_ratio_at_threshold_preserves_original_admission_order() -> None:
    """When the batch's length-spread ratio is exactly at (or below)
    the threshold, the split path does not fire — neither cohort
    shape nor admission order should change vs pre-P-4.5 behaviour.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={
            "len2": [1, 2],
            "len1": [1],  # ratio exactly 2.0 with threshold=2.0 → fast path
        },
        script=[9, 9],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(
        engine.generate_batch(
            ["len2", "len1"],  # user's order: longer first
            params,
            max_batch_size=2,
            length_spread_threshold=2.0,
        )
    )
    first_tokens = [e for e in events if e.kind == "token"]
    # No split triggered (ratio <= threshold), so the user's original
    # admission order is preserved in the event stream.
    assert [e.req_index for e in first_tokens] == [0, 1], (
        "ratio exactly at threshold must NOT reorder; observed "
        f"req_index order: {[e.req_index for e in first_tokens]}"
    )
    # And the forward call is a single B=2 prefill at T=2, not a
    # split into B=1 + B=1.
    assert adapter._model.calls == [(2, 2)]


def test_generate_batch_default_threshold_splits_heterogeneous() -> None:
    """Default threshold=2.0 on heterogeneous lengths routes the long
    prompt through the mid-run admit path — observable via the
    sequence of (B, T) forward calls the fake _Model records.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={"x": [1], "yyyyyy": [1, 2, 3, 4, 5, 6]},
        script=[9, 9, 9],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(
        engine.generate_batch(
            ["x", "yyyyyy"],
            params,
            max_batch_size=2,
        )
    )
    # First forward: short cohort prefill at B=1, T=1.
    assert adapter._model.calls[0] == (1, 1)
    # Second forward: mid-run admit prefill for the long prompt at B=1,
    # T=6. This is the signature of ``_admit_miss_cohort`` running on
    # the queued long row.
    assert adapter._model.calls[1] == (1, 6)
    # Every row reaches done; no aborts.
    done_rows = {e.req_index for e in events if e.kind == "done"}
    aborts = [e for e in events if e.kind == "aborted"]
    assert done_rows == {0, 1}
    assert aborts == []


def test_generate_batch_homogeneous_lengths_do_not_split() -> None:
    """Homogeneous-length batch produces one B=N prefill under default
    threshold — no regression for the common case where every prompt
    is roughly the same length.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]},
        script=[1],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    list(
        engine.generate_batch(
            ["x", "y", "z"],
            params,
            max_batch_size=3,
        )
    )
    # Single prefill forward at B=3, T=3.
    assert adapter._model.calls[0] == (3, 3)


def test_generate_batch_raises_on_invalid_threshold() -> None:
    """Precondition bubbles up through generate_batch (validated inside
    _initial_cohort_cap, not silently swallowed)."""
    engine, _ = _make_engine(
        lengths_by_prompt={"a": [1], "bb": [1, 2]},
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    with pytest.raises(ValueError, match="length_spread_threshold must be > 1.0"):
        # threshold=1.0 is invalid.
        list(engine.generate_batch(["a", "bb"], params, length_spread_threshold=1.0))


def test_generate_batch_single_prompt_is_unchanged() -> None:
    """Single-prompt generate_batch is unchanged by P-4.5-B.1 — the
    single admission hits the homogeneous fast path regardless of
    threshold.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={"only": [1, 2, 3]},
        script=[2],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(engine.generate_batch(["only"], params))
    token_events = [e for e in events if e.kind == "token"]
    assert len(token_events) == 1
    assert token_events[0].req_index == 0
    # Single B=1 prefill forward.
    assert adapter._model.calls[0] == (1, 3)


def test_generate_batch_queued_short_gets_batched_with_long_not_fixed() -> None:
    """Pin the known limitation (opening doc §5.2): Option (C) only
    fixes first-token fairness within the *initial* cohort. When
    ``max_batch_size < short_count + 1``, shorts that overflow the
    initial cohort land in the waiting queue alongside the long
    prompt, and ``_admit_miss_cohort`` batches the entire queue in
    one forward — so the queued short is again dragged to the long
    prompt's ``T_max``.

    Setup: four prompts ``[short_a=1 tok, short_b=1 tok, short_c=1 tok,
    long=6 tok]`` with ``max_batch_size=2``. Sort ASC keeps the order
    (shorts already precede the long in admission order); cap is
    ``min(2, first_exceeding_index=3) = 2``, so the initial cohort
    is ``[short_a, short_b]`` and the queue holds
    ``[short_c, long]``. The drain loop then runs a single mid-run
    admission forward at ``B=2, T=6`` — observable via the fake
    model's ``calls`` log. The assertion on ``(B=2, T=6)`` pins the
    queued-cohort behaviour; if a future change routes queued
    shorts separately from the long prompt, the observed shape
    changes and this test fails loudly so the documented scope
    boundary is re-examined, not silently widened.
    """
    engine, adapter = _make_engine(
        lengths_by_prompt={
            "short_a": [1],
            "short_b": [1],
            "short_c": [1],
            "long6": [1, 2, 3, 4, 5, 6],
        },
        script=[4, 4, 4],
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    events = list(
        engine.generate_batch(
            ["short_a", "short_b", "short_c", "long6"],
            params,
            max_batch_size=2,
        )
    )
    # First forward: initial cohort {short_a, short_b} prefill at B=2, T=1.
    assert adapter._model.calls[0] == (2, 1)
    # Second forward: mid-run admit for the queued {short_c, long6} pair
    # at B=2, T=6 — short_c's row has to tread the long's T_max. This
    # is the documented limitation (opening doc §5.2 "queued-cohort
    # fairness NOT provided").
    assert adapter._model.calls[1] == (2, 6)
    aborts = [e for e in events if e.kind == "aborted"]
    assert aborts == []


# --- Shared on-device fixtures (used by Sections 4 and 5) -------------------


_REPO = "Qwen/Qwen3-0.6B"


def _hf_cache_has_repo(repo: str) -> bool:
    """Mirror ``silica.bench.scenario.hf_cache_path_for_repo`` sentinel
    without depending on the bench-harness import path.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = (
        Path(hf_home)
        / "hub"
        / f"models--{repo.replace('/', '--')}"
    )
    return cache_dir.exists()


_SKIP_Q010 = not _hf_cache_has_repo(_REPO)
_SKIP_Q010_REASON = (
    f"Qwen3-0.6B ({_REPO}) not present in the local HF cache. Run any "
    f"cache-only P-4 bench scenario once to populate."
)

# P-5-A.1a follow-up: ``test_q010_ratio_below_threshold_on_five_runs``
# is a wall-clock timing measurement, noise-prone on machine load. It
# is out of the default pytest sweep via the ``q010_timing`` marker;
# ``conftest.py::pytest_collection_modifyitems`` skips marked tests
# unless the user opts in via ``SILICA_Q010_TIMING=1`` or invokes
# ``pytest -m q010_timing`` explicitly. See `pyproject.toml`
# ``[tool.pytest.ini_options].markers`` for the marker registration
# and `conftest.py` for the gating implementation.


def _runtime_admission_partition(
    lengths: list[int],
    effective_batch_size: int,
    threshold: float,
) -> tuple[list[int], list[int]]:
    """Replicate ``generate_batch``'s admission partition end-to-end so
    on-device tests can compare against a reference for the *same*
    cohorts the runtime actually forms.

    Given prompt-token ``lengths`` (indexed by original ``req_index``)
    plus ``effective_batch_size`` (the ``max_batch_size`` the scenario
    passes to ``generate_batch``) and ``threshold`` (the
    ``length_spread_threshold`` kwarg), returns
    ``(pre_step_orig_indices, remainder_orig_indices)`` — indices into
    the user-supplied prompts list.

    Mirrors ``generate_batch`` step by step:

      1. Build admission tuples ``(req_index, ids)``.
      2. Sort by length ASC; call ``_initial_cohort_cap`` to derive
         the cap under the current threshold (homogeneous fast path
         vs split path, subject to the ``min(effective_batch_size,
         first_exceeding)`` clamp).
      3. Switch to sorted order only if the split path would actually
         fire (``max/min > threshold``); otherwise preserve the
         original admission order.
      4. ``pre_step = ordered[:cap]``, ``remainder = ordered[cap:]``.

    Used by: §4 direct-batched sub-cohort reference test (so the
    reference matches the runtime's actual sub-cohort shapes) and §5
    Q-010 short-row filter (so the ratio denominator is correct even
    when a future catalog editor changes ``max_batch_size`` to a
    value smaller than ``short_count + 1``).
    """
    admissions = list(enumerate([[0] * n for n in lengths]))
    sorted_adm = _sort_admissions_by_length(admissions)
    cap = _initial_cohort_cap(sorted_adm, effective_batch_size, threshold)
    needs_reorder = (
        len(admissions) > 1
        and min(lengths, default=0) > 0
        and max(lengths) / min(lengths) > threshold
    )
    ordered = sorted_adm if needs_reorder else admissions
    pre = [ri for ri, _ in ordered[:cap]]
    rem = [ri for ri, _ in ordered[cap:]]
    return pre, rem


# --- Section 4. Direct mlx-lm sub-cohort numerical reference ----------------


@pytest.mark.skipif(_SKIP_Q010, reason=_SKIP_Q010_REASON)
def test_reordered_cohort_matches_mlx_lm_direct_batched_reference() -> None:
    """PLAN §7 P-4.5 Acceptance (c) and opening doc §8 item 4.

    Pins the three-layer correctness criterion's numerical reference:
    under Option (C), Silica's per-row tokens on the Q-010 catalog
    workload (``qwen3-0.6b-ttft-under-concurrency``) must match a
    direct mlx-lm batched forward run over each sub-cohort. This is
    the P-3-D3.1 Gemma4 precedent applied to Qwen3-0.6B: "Silica
    batched == mlx-lm batched" rather than "Silica batched == Silica
    single-request" — the latter is blocked by fp16 batched SDPA
    drift across batch compositions (P-2 Qwen3-0.6B; §7 P-4.5
    Amendment log 2026-04-21).

    Procedure:
      1. Read the live catalog scenario (NOT hard-coded prompts so
         the test stays robust if a future editor re-authors the
         workload).
      2. Tokenize via the real adapter tokenizer; derive the runtime
         admission partition via ``_runtime_admission_partition`` so
         ``pre_step`` and ``remainder`` match the sub-cohorts
         ``generate_batch`` actually forms (including the
         ``min(effective_batch_size, first_exceeding)`` clamp — if a
         future editor tightens ``max_batch_size`` below
         ``short_count + 1``, some shorts end up in remainder, and
         the reference must reflect that).
      3. Assert both sub-cohorts are non-empty — otherwise the test
         isn't exercising Option (C) at all.
      4. Silica path: run ``engine.generate_batch(prompts, params)``
         with default threshold=2.0; capture per-row tokens by
         original ``req_index``.
      5. Reference path: call
         ``silica.bench.runner._direct_mlx_lm_batched_reference`` on
         the ``pre_step`` prompts and again on the ``remainder``
         prompts. Each call returns tokens keyed by
         sub-cohort-local index (0..k-1).
      6. Map sub-cohort local indices back to original req_index via
         the ``pre_step`` / ``remainder`` index lists and assert
         byte-for-byte equality.

    ``stop_token_ids=()`` on both sides so neither side early-stops
    on EOS — matches the reference's "does not honor EOS" contract,
    per ``_direct_mlx_lm_batched_reference`` docstring.
    """
    from silica.bench.runner import _direct_mlx_lm_batched_reference
    from silica.bench.scenarios import get_scenario
    from silica.models.qwen3 import Qwen3Adapter

    scenario = get_scenario("qwen3-0.6b-ttft-under-concurrency")
    prompts = list(scenario.workload.prompts)

    adapter, kv = Qwen3Adapter.from_hf_repo(_REPO)
    tokenizer = adapter.tokenizer()
    lengths = [len(tokenizer.encode(p)) for p in prompts]

    threshold = 2.0
    ebs = scenario.workload.max_batch_size
    pre_step_orig, remainder_orig = _runtime_admission_partition(
        lengths, ebs, threshold
    )
    assert pre_step_orig, (
        "Expected a non-empty initial cohort; catalog scenario may "
        "have been re-authored with a degenerate shape. This test "
        "requires the Q-010 shape (one long + several short "
        f"prompts). lengths={lengths}, effective_batch_size={ebs}"
    )
    assert remainder_orig, (
        "Expected a non-empty waiting-queue remainder; the Q-010 "
        "shape demands at least one row ends up mid-run admitted so "
        "the direct-batched reference can compare the two "
        f"sub-cohorts. lengths={lengths}, effective_batch_size={ebs}"
    )

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        stop_token_ids=(),
    )

    # Silica path.
    engine = Engine(adapter, kv)
    silica_by_req: dict[int, list[int]] = {i: [] for i in range(len(prompts))}
    aborts: list[tuple[int, str | None]] = []
    done_rows: set[int] = set()
    for event in engine.generate_batch(
        prompts, params, max_batch_size=scenario.workload.max_batch_size
    ):
        if event.kind == "token" and event.token_id is not None:
            silica_by_req[event.req_index].append(event.token_id)
        elif event.kind == "done":
            done_rows.add(event.req_index)
        elif event.kind == "aborted":
            aborts.append((event.req_index, event.finish_reason))

    assert aborts == [], f"Silica aborted rows: {aborts}"
    assert done_rows == set(range(len(prompts))), (
        f"Silica did not finish all rows: done={sorted(done_rows)}"
    )

    # Reference: pre-step sub-cohort. _direct_mlx_lm_batched_reference
    # builds its own cache via adapter.make_batch_cache(left_padding);
    # sharing the adapter's already-loaded model is safe because the
    # cache is fresh per call.
    pre_step_prompts = [prompts[k] for k in pre_step_orig]
    pre_step_ref = _direct_mlx_lm_batched_reference(
        adapter, pre_step_prompts, params
    )
    # Inverse permutation: sub-cohort local index i → original
    # req_index pre_step_orig[i]. Same round-trip pinned in
    # test_inverse_permutation_round_trip (§1).
    for local_idx, original_req in enumerate(pre_step_orig):
        assert silica_by_req[original_req] == pre_step_ref[local_idx], (
            "Silica pre-step sub-cohort token stream diverged from direct "
            "mlx-lm batched reference:\n"
            f"  original req_index: {original_req}\n"
            f"  prompt:             {prompts[original_req]!r}\n"
            f"  silica tokens:      {silica_by_req[original_req]}\n"
            f"  reference tokens:   {pre_step_ref[local_idx]}"
        )

    # Reference: remainder (mid-run admit) sub-cohort.
    remainder_prompts = [prompts[k] for k in remainder_orig]
    remainder_ref = _direct_mlx_lm_batched_reference(
        adapter, remainder_prompts, params
    )
    for local_idx, original_req in enumerate(remainder_orig):
        assert silica_by_req[original_req] == remainder_ref[local_idx], (
            "Silica remainder (mid-run) sub-cohort token stream diverged "
            "from direct mlx-lm batched reference:\n"
            f"  original req_index: {original_req}\n"
            f"  prompt:             {prompts[original_req]!r}\n"
            f"  silica tokens:      {silica_by_req[original_req]}\n"
            f"  reference tokens:   {remainder_ref[local_idx]}"
        )


# --- Section 5. Dual-gated on-device Q-010 acceptance harness ---------------


@pytest.mark.q010_timing
@pytest.mark.skipif(_SKIP_Q010, reason=_SKIP_Q010_REASON)
def test_q010_ratio_below_threshold_on_five_runs() -> None:
    """PLAN §7 P-4.5 Acceptance (a) and opening doc §8 item 1.

    Drive the same `(qwen3-0.6b-smoke, qwen3-0.6b-ttft-under-concurrency)`
    bench pair `scripts/bench.py` emits; compute
    ``ratio = max(offsets_short) / smoke_ttft_ms`` per run, over five
    consecutive runs; apply a **statistical gate**: the
    second-highest-of-five is < 3.5×, AND the worst run is < 5.0×
    (Q-010's original promotion trigger). Both clauses must hold; each
    catches a different regression shape.

    The 3.5× half is the load-bearing signal — PLAN's exit-criterion
    threshold after the 2026-04-21 amendment (see §7 P-4.5 Amendment
    log; original lean was < 3× but empirical measurement showed
    option (C)'s B=3 short-cohort prefill has an intrinsic ~2-3×
    overhead vs B=1 isolated smoke, so < 3× would require sub-B=3
    short cohorts and lose batching entirely). Taking the
    second-highest rather than the max absorbs one wall-clock outlier
    while still asserting the bulk distribution is below the bar.

    The 5.0× max-cap half is a safety net for the rare case where the
    bulk signal is clean (second-highest < 3.5×) but one outlier is
    severely inflated by e.g. Metal-compile contention on an otherwise
    warmed run. 5.0× is Q-010's promotion trigger; any run above it
    indicates either a true scheduler regression or a measurement
    environment that should not be asserted against.

    ``offsets_short`` filters rows whose prompt-token length is below
    ``max_prompt_len / length_spread_threshold`` — the same rule the
    admission reorder uses to decide what queues. That means the
    filter adapts automatically if the catalog's
    ``qwen3-0.6b-ttft-under-concurrency`` is ever re-authored with
    the long prompt at a non-zero req_index, per the opening doc's
    §8 warning about hard-coded indices.

    Dual-gated on the HF cache: the test skips when Qwen3-0.6B is not
    cached locally, matching every other P-4 bench-backed test. Each
    run spawns a fresh ``python -m scripts.bench`` subprocess so
    MLX's in-process caches cannot skew the wall-clock measurement
    across runs.

    Warmup + single-subprocess measurement: every fresh subprocess
    pays a metal-kernel-compile tax on its first forward of a given
    ``(B, T)`` shape. The Q-010 scenario exercises multiple shapes
    the smoke scenario does not touch — under the admission-reorder
    fix it becomes ``(B=3, T=1)`` short-cohort prefill +
    ``(B=1, T=301)`` long-cohort prefill — so across-subprocess
    warmup would leave the second subprocess still paying its own
    cold compile. Instead, we spawn **one** subprocess that runs
    ``(smoke, ttft)`` pairs ``n_warmup + n_runs`` times; the metal
    kernel cache is process-local and remains warm across
    ``BenchRunner`` iterations even though each iteration loads a
    fresh adapter (weights come from a warm HF cache). We then
    discard the first ``n_warmup`` pair(s) and assert on the
    remaining ``n_runs`` pairs. This matches the PLAN §7 P-4.5
    Acceptance (a) wording ("over five consecutive runs on the same
    machine") read as "five consecutive measured runs after warmup"
    — the production chat REPL / HTTP server never hits user-visible
    TTFT on its first forward either, so the acceptance target is
    steady-state.
    """
    import json
    from tempfile import TemporaryDirectory

    repo_root = Path(__file__).resolve().parent.parent
    # Threshold 3.5× reflects the measured post-fix steady state on
    # Qwen3-0.6B: short-row first-token offset is dominated by B=3
    # short-cohort prefill (≈ 50 ms) vs B=1 isolated smoke (≈ 17 ms),
    # giving a ratio ≈ 2.5-3.2× with noise. Pre-fix ratios were
    # 4.1-4.8× on the same scenario pair (PLAN §10 Q-010 Resolution).
    # 3.5× is: (i) clearly below Q-010's 5× promotion trigger; (ii)
    # above the measured post-fix p95 with headroom so the test does
    # not flake on single-sample noise; (iii) tight enough that if the
    # scheduler regresses back into cohort-level prefill the ratio
    # would jump to ~5× and fail loudly.
    threshold = 3.5
    n_runs = 5
    n_warmup = 1

    with TemporaryDirectory() as td:
        out_path = Path(td) / "all_pairs.jsonl"
        scenario_args: list[str] = []
        for _ in range(n_warmup + n_runs):
            scenario_args += [
                "--scenario",
                "qwen3-0.6b-smoke",
                "--scenario",
                "qwen3-0.6b-ttft-under-concurrency",
            ]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.bench",
                *scenario_args,
                "--out",
                str(out_path),
            ],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        rows = [
            json.loads(line)
            for line in out_path.read_text().splitlines()
            if line.strip()
        ]
        # BenchRunner writes one JSONL row per --scenario in the
        # order they appeared on the CLI, so the 2N rows are
        # interleaved [smoke, ttft, smoke, ttft, ...]. Pair them up.
        assert len(rows) == 2 * (n_warmup + n_runs), (
            f"expected {2 * (n_warmup + n_runs)} JSONL rows, "
            f"got {len(rows)}; bench did not produce per-scenario rows"
        )
        pairs: list[tuple[dict, dict]] = [
            (rows[i], rows[i + 1]) for i in range(0, len(rows), 2)
        ]
        measured_pairs = pairs[n_warmup:]

        # Resolve the short-row filter once per test, directly from
        # the catalog scenario. Earlier iterations tried to read
        # prompts from the bench metadata, but the SMOKE oracle's
        # per-row metadata only carries ``{row, token_count,
        # first_token_ms_offset}`` — no prompt text — so the adaptive
        # filter silently fell back to "row != 0" (hard-coded). Using
        # ``get_scenario(...).workload.prompts`` + the runtime
        # partition helper is both adaptive AND authoritative: it
        # always matches the shape the runtime actually admits
        # (including the ``min(effective_batch_size,
        # first_exceeding)`` clamp), and it auto-adapts to any future
        # catalog re-author without further test changes.
        from mlx_lm import load as _load  # local import

        from silica.bench.scenarios import get_scenario as _get_scenario

        scenario = _get_scenario("qwen3-0.6b-ttft-under-concurrency")
        prompts = list(scenario.workload.prompts)
        ebs = scenario.workload.max_batch_size
        _, tokenizer = _load(_REPO)
        lens = [len(tokenizer.encode(p)) for p in prompts]
        short_rows, long_rows = _runtime_admission_partition(
            lens, ebs, 2.0
        )

        assert short_rows, (
            "no short rows identified for ratio calculation "
            f"(prompts={prompts}, lengths={lens}, ebs={ebs}); the "
            "catalog scenario may have been re-authored with "
            "homogeneous lengths so Option (C) would not split"
        )
        assert long_rows, (
            "no long rows identified; the Q-010 shape requires at "
            f"least one long prompt (prompts={prompts}, "
            f"lengths={lens}, ebs={ebs})"
        )

        results: list[float] = []
        for smoke, ttft_row in measured_pairs:
            assert smoke["scenario_id"] == "qwen3-0.6b-smoke"
            assert ttft_row["scenario_id"] == "qwen3-0.6b-ttft-under-concurrency"
            smoke_ttft_ms = smoke["ttft_ms"]
            assert smoke_ttft_ms is not None, (
                "smoke scenario did not report ttft_ms"
            )
            per_row_offsets = {
                r["row"]: r["first_token_ms_offset"]
                for r in ttft_row["metadata"]["rows"]
            }
            short_offsets = [
                per_row_offsets[r] for r in short_rows if r in per_row_offsets
            ]
            assert short_offsets, "short-row first_token_ms_offset missing"
            ratio = max(short_offsets) / smoke_ttft_ms
            results.append(ratio)

    # Statistical gate (P-5-A.1a follow-up). Previous "every run must
    # clear the bar" rule tripped on single wall-clock outliers from
    # machine load even when the scheduler was correct. The gate below
    # absorbs one outlier while still catching real regressions:
    #
    # (i) ``sorted(results)[-2] < threshold`` — the second-highest of
    #     five is below 3.5×. This is the load-bearing check that the
    #     bulk distribution is clean. A regression back to pre-fix
    #     cohort-level prefill was 4.1-4.8×, which would fail here
    #     because four or five runs would be above 3.5× and the
    #     second-highest would be well above it.
    # (ii) ``max(results) < outlier_cap`` — worst run is below 5.0×
    #     (Q-010's original promotion trigger). A clean bulk
    #     (second-highest < 3.5×) with one run >= 5× suggests a
    #     compromised measurement environment, not a scheduler
    #     regression; this cap surfaces that case rather than treating
    #     it as a pass.
    #
    # Both clauses must pass. Either clause failing alone is the useful
    # regression signal.
    outlier_cap = 5.0
    sorted_results = sorted(results)
    second_highest = sorted_results[-2]
    worst = sorted_results[-1]
    assert second_highest < threshold, (
        f"Q-010 p80 ratio {second_highest:.2f}× (second-highest of "
        f"{n_runs}) exceeds the {threshold}× P-4.5 acceptance threshold. "
        f"All runs: {[f'{x:.2f}' for x in results]}"
    )
    assert worst < outlier_cap, (
        f"Q-010 worst-run ratio {worst:.2f}× exceeds the "
        f"{outlier_cap}× outlier cap (Q-010's 5× promotion trigger). "
        f"A regression to pre-fix cohort-level prefill would land in "
        f"the 4.1-4.8× range, so a run above {outlier_cap}× indicates "
        f"a real scheduler regression, not wall-clock noise. All runs: "
        f"{[f'{x:.2f}' for x in results]}"
    )
