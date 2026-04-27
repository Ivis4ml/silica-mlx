"""P-3-C5.1 — adapter-owned snapshot / restore unit tests.

Covers the four C5.1 review points:

1. ``cache_list, row_idx`` API shape — snapshot reads from the
   batcher-supplied cache list (NOT ``KVManager.cache_list(req_id)``).
2. Lazy-slot handling — snapshot taken before any forward writes
   into a slot stores ``None``; restore leaves the slot ``None``
   so the next forward lazy-allocates rather than seeing fp32
   zeros.
3. R-C5-2 aliasing — snapshot tensors must not change when the
   live cache is mutated by subsequent forwards. Implementation
   uses ``mx.array(...)`` + ``mx.eval(...)`` to detach.
4. Restore preserves other rows — splicing into a B>1 cache at
   ``row_idx=k`` must not change rows ``!= k``.

All tests gate on the Qwen3.5-0.8B HF cache (single skip gate
per ``plans/P3_C5_OPENING.md`` §2.6 / §3.1).

Tests do NOT exercise:

- ``ContinuousBatcher`` integration (C5.2).
- ``RadixPrefixCache`` cooperation (C5.3).
- The ``has_recurrent_state + prefix_cache`` guard (still in
  place; removed at C5.4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica.models.qwen3_5 import Qwen3_5Adapter

_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache_has_repo(repo: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    return cache_dir.exists()


_SKIP = not _hf_cache_has_repo(_REPO)
_SKIP_REASON = (
    f"Qwen3.5-0.8B ({_REPO}) not present in the local HF cache. "
    f"Run qwen3.5-0.8b-b1-parity once to populate, or export HF_HOME."
)


# Two short prompts, deliberately different lengths to exercise the
# left-padding case for B=2 splice tests.
_PROMPT_A = "The recurrent state accumulates across every processed token."
_PROMPT_B = "Snapshot before forward; restore after mutation."


@pytest.fixture(scope="module")
def adapter_and_model() -> tuple[Qwen3_5Adapter, Any]:
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    from silica.models.factory import adapter_for_repo

    adapter, _kv = adapter_for_repo(_REPO)
    # Bench-style ``adapter_for_repo`` returns a generic ``ModelAdapter``;
    # narrow to ``Qwen3_5Adapter`` here so the test can read the
    # private ``_model`` attribute the snapshot probe needs.
    assert isinstance(adapter, Qwen3_5Adapter)
    model = adapter._model
    return adapter, model


def _build_b1_batched_cache(adapter: Qwen3_5Adapter) -> list[Any]:
    """Build a B=1 batched cache via ``adapter.make_batch_cache``."""
    return adapter.make_batch_cache([0])


def _build_b2_batched_cache(
    adapter: Qwen3_5Adapter, padding: list[int]
) -> list[Any]:
    return adapter.make_batch_cache(padding)


def _forward_b1(
    model: Any, tokens_1d: list[int], cache_list: list[Any]
) -> mx.array:
    """Run one ``forward_batched`` pass at B=1 over ``tokens_1d``."""
    from silica.mlx.runner import forward_batched

    tokens_2d = mx.array([tokens_1d], dtype=mx.int32)
    logits = forward_batched(model, tokens_2d, cache_list)
    mx.eval(logits)
    return logits


def _forward_b2(
    model: Any,
    tokens_a: list[int],
    tokens_b: list[int],
    cache_list: list[Any],
) -> mx.array:
    """Run one ``forward_batched`` pass at B=2 with left-padding."""
    from silica.mlx.runner import forward_batched

    max_len = max(len(tokens_a), len(tokens_b))
    pad_a = max_len - len(tokens_a)
    pad_b = max_len - len(tokens_b)
    tokens_2d = mx.array(
        [
            [0] * pad_a + tokens_a,
            [0] * pad_b + tokens_b,
        ],
        dtype=mx.int32,
    )
    logits = forward_batched(model, tokens_2d, cache_list)
    mx.eval(logits)
    return logits


def _deltanet_layer_indices(model: Any) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.layers)
        if getattr(layer, "is_linear", False)
    ]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSnapshotRoundTripBitExact:
    """The canonical bit-exactness test: snapshot at a known token
    boundary, mutate the live cache by feeding more tokens, restore
    the snapshot, feed the same continuation tokens — final logits
    must match the no-snapshot oracle."""

    def test_snapshot_then_mutate_then_restore_recovers_state(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        prefix = tokenizer.encode(_PROMPT_A)
        suffix = tokenizer.encode(_PROMPT_B)

        # Path 1: oracle — fresh cache, prefix then suffix in two forwards.
        oracle_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, oracle_cache)
        oracle_logits = _forward_b1(model, suffix, oracle_cache)

        # Path 2: subject — prefix forward, snapshot, mutate by feeding
        # different tokens, restore, feed suffix.
        subject_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, subject_cache)
        snapshot = adapter.snapshot_recurrent_state(subject_cache, row_idx=0)
        # Mutate the live cache by feeding the suffix once — this
        # advances state past where the snapshot was taken.
        _forward_b1(model, suffix, subject_cache)
        # Restore — recurrent slots return to end-of-prefix state.
        adapter.restore_recurrent_state(subject_cache, 0, snapshot)

        # CRITICAL: re-snapshot the live cache AFTER restore so we
        # actually verify what restore wrote back. Comparing the
        # original ``snapshot`` object to the reference would only
        # confirm the two snapshot operations agreed; a no-op restore
        # would silently pass that check. Reading the live cache
        # post-restore is what locks the canonical restore gate.
        restored_snapshot = adapter.snapshot_recurrent_state(
            subject_cache, row_idx=0
        )

        # GLOBAL K/V is NOT restored (out of C5.1 scope), so we cannot
        # finish the round-trip via another forward. Instead we
        # compare the post-restore live cache against a freshly-
        # captured end-of-prefix snapshot from a parallel oracle cache
        # — that confirms restore wrote the right bytes back.
        ref_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, ref_cache)
        ref_snapshot = adapter.snapshot_recurrent_state(ref_cache, row_idx=0)

        # The post-restore re-snapshot of subject_cache must byte-match
        # the parallel oracle cache's snapshot at every DeltaNet layer.
        assert len(restored_snapshot.entries) == len(ref_snapshot.entries)
        for restored_e, ref_e in zip(
            restored_snapshot.entries, ref_snapshot.entries
        ):
            assert restored_e.layer_idx == ref_e.layer_idx
            # Both came from caches that ran the prefix forward, so
            # both fields are populated. Narrow for mypy.
            assert restored_e.conv_state is not None
            assert ref_e.conv_state is not None
            assert restored_e.recurrent_state is not None
            assert ref_e.recurrent_state is not None
            assert mx.array_equal(restored_e.conv_state, ref_e.conv_state)
            assert mx.array_equal(
                restored_e.recurrent_state, ref_e.recurrent_state
            )

        # Sanity: oracle path's logits are fully real (used to confirm
        # the test setup actually exercised the model). ``forward_batched``
        # returns the last-token slice, so shape is ``(B, V)``.
        assert oracle_logits.ndim == 2
        assert oracle_logits.shape[0] == 1


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestLazySlotSemantics:
    """Snapshot taken before any forward populates the recurrent
    slots: every entry's slots are ``None``; restore is a no-op."""

    def test_snapshot_before_forward_has_none_slots(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        cache = _build_b1_batched_cache(adapter)
        # No forward; ArraysCache slots are still None.
        snapshot = adapter.snapshot_recurrent_state(cache, row_idx=0)

        delta_indices = _deltanet_layer_indices(model)
        assert len(snapshot.entries) == len(delta_indices)
        for entry, idx in zip(snapshot.entries, delta_indices):
            assert entry.layer_idx == idx
            assert entry.conv_state is None
            assert entry.recurrent_state is None
        assert snapshot.nbytes == 0

    def test_restore_of_none_snapshot_into_none_cache_stays_none(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        snap_cache = _build_b1_batched_cache(adapter)
        snapshot = adapter.snapshot_recurrent_state(snap_cache, row_idx=0)

        target_cache = _build_b1_batched_cache(adapter)
        # Slots start as None; restore must not write fp32 zeros.
        adapter.restore_recurrent_state(target_cache, 0, snapshot)

        for layer_idx in _deltanet_layer_indices(model):
            inner = target_cache[layer_idx].cache
            assert inner[0] is None, (
                f"layer {layer_idx} slot 0 should remain None after "
                f"None-restore; got {type(inner[0]).__name__}"
            )
            assert inner[1] is None, (
                f"layer {layer_idx} slot 1 should remain None after "
                f"None-restore; got {type(inner[1]).__name__}"
            )

    def test_restore_of_none_snapshot_wipes_populated_b1_cache(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        """Value-object semantics: if the snapshot captured an
        unallocated state, restoring it onto a populated B=1 cache
        wipes the slots back to None so the next forward lazy-
        allocates fresh state. Anything else would silently retain
        stale recurrent state past the restore point."""
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        prefix = tokenizer.encode(_PROMPT_A)

        # Snapshot from a fresh (no-forward-yet) cache — entries are None.
        empty_cache = _build_b1_batched_cache(adapter)
        empty_snapshot = adapter.snapshot_recurrent_state(
            empty_cache, row_idx=0
        )

        # Build a populated B=1 cache by running a forward.
        target_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, target_cache)
        # Confirm slots are populated before restore.
        for layer_idx in _deltanet_layer_indices(model):
            inner = target_cache[layer_idx].cache
            assert inner[0] is not None
            assert inner[1] is not None

        adapter.restore_recurrent_state(target_cache, 0, empty_snapshot)

        # After restore, every DeltaNet slot should be wiped back to
        # None — the snapshot's "unallocated" capture is reproduced.
        for layer_idx in _deltanet_layer_indices(model):
            inner = target_cache[layer_idx].cache
            assert inner[0] is None, (
                f"layer {layer_idx} slot 0 should be wiped to None "
                f"after restoring an empty snapshot; got "
                f"{type(inner[0]).__name__}"
            )
            assert inner[1] is None, (
                f"layer {layer_idx} slot 1 should be wiped to None "
                f"after restoring an empty snapshot; got "
                f"{type(inner[1]).__name__}"
            )

    def test_restore_of_none_snapshot_into_populated_b2_cache_raises(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        """The "lazy / unallocated" state is per-slot, not per-row.
        Trying to restore an empty snapshot for one row of a
        populated B=2 cache is a contract violation — the implementation
        raises rather than silently wiping all rows or only one."""
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        tokens_a = tokenizer.encode(_PROMPT_A)
        tokens_b = tokenizer.encode(_PROMPT_B)

        empty_cache = _build_b1_batched_cache(adapter)
        empty_snapshot = adapter.snapshot_recurrent_state(
            empty_cache, row_idx=0
        )

        max_len = max(len(tokens_a), len(tokens_b))
        padding = [max_len - len(tokens_a), max_len - len(tokens_b)]
        populated_cache = _build_b2_batched_cache(adapter, padding)
        _forward_b2(model, tokens_a, tokens_b, populated_cache)

        with pytest.raises(ValueError, match="multi-row"):
            adapter.restore_recurrent_state(populated_cache, 0, empty_snapshot)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestAliasingR_C5_2:
    """R-C5-2: snapshot tensors must not be observable through the
    live cache after subsequent in-place rebinds at
    ``mlx_lm/models/qwen3_5.py:164/166/197``."""

    def test_snapshot_unchanged_after_live_cache_mutation(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        prefix = tokenizer.encode(_PROMPT_A)
        suffix = tokenizer.encode(_PROMPT_B)

        cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, cache)
        snapshot = adapter.snapshot_recurrent_state(cache, row_idx=0)

        # Capture the snapshot's element-wise values into a parallel
        # detached structure for later comparison.
        captured = []
        for e in snapshot.entries:
            captured.append({
                "layer_idx": e.layer_idx,
                "conv_state": (
                    mx.array(e.conv_state) if e.conv_state is not None else None
                ),
                "recurrent_state": (
                    mx.array(e.recurrent_state)
                    if e.recurrent_state is not None
                    else None
                ),
            })
        for c in captured:
            for k in ("conv_state", "recurrent_state"):
                if c[k] is not None:
                    mx.eval(c[k])

        # Mutate the live cache via a second forward — this rebinds
        # ``cache.cache[0]`` and ``cache.cache[1]`` on every DeltaNet
        # layer at qwen3_5.py:164/166/197.
        _forward_b1(model, suffix, cache)

        # Snapshot tensors must equal the values captured before
        # mutation. If aliasing leaked, the snapshot's tensors would
        # have advanced along with the live cache.
        for entry, ref in zip(snapshot.entries, captured):
            assert entry.layer_idx == ref["layer_idx"]
            if entry.conv_state is not None:
                ref_conv = ref["conv_state"]
                assert ref_conv is not None
                assert mx.array_equal(entry.conv_state, ref_conv), (
                    f"layer {entry.layer_idx} conv_state changed after "
                    f"live-cache mutation — R-C5-2 aliasing leak."
                )
            if entry.recurrent_state is not None:
                ref_recur = ref["recurrent_state"]
                assert ref_recur is not None
                assert mx.array_equal(entry.recurrent_state, ref_recur), (
                    f"layer {entry.layer_idx} recurrent_state changed "
                    f"after live-cache mutation — R-C5-2 aliasing leak."
                )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestRowIsolation:
    """Restore at ``row_idx=k`` in a B>1 cache must not change rows
    ``!= k``."""

    def test_restore_row_0_preserves_row_1(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        tokens_a = tokenizer.encode(_PROMPT_A)
        tokens_b = tokenizer.encode(_PROMPT_B)

        # Build a B=2 cache that has been advanced by one forward pass.
        max_len = max(len(tokens_a), len(tokens_b))
        padding = [max_len - len(tokens_a), max_len - len(tokens_b)]
        cache = _build_b2_batched_cache(adapter, padding)
        _forward_b2(model, tokens_a, tokens_b, cache)

        # Snapshot row 0 so we can restore it later.
        snap_row0 = adapter.snapshot_recurrent_state(cache, row_idx=0)

        # Capture row 1's slots BEFORE the restore so we can prove they
        # stay byte-identical across the restore call.
        row1_before = adapter.snapshot_recurrent_state(cache, row_idx=1)

        # Mutate row 0 in some way that would normally also touch the
        # tensors. The simplest mutation is a second forward that
        # rebinds all slots to fresh tensors at qwen3_5.py:164/166/197.
        # After this forward, cache slots are different from their
        # previous values for BOTH rows.
        _forward_b2(model, tokens_a, tokens_b, cache)

        # Capture row 1 again — it has now diverged from row1_before.
        row1_after_mutation = adapter.snapshot_recurrent_state(
            cache, row_idx=1
        )
        # Sanity: at least one entry's recurrent_state must have
        # changed; otherwise the test setup didn't exercise mutation.
        any_changed = False
        for old_e, new_e in zip(row1_before.entries, row1_after_mutation.entries):
            if old_e.recurrent_state is None or new_e.recurrent_state is None:
                continue
            if not mx.array_equal(old_e.recurrent_state, new_e.recurrent_state):
                any_changed = True
                break
        assert any_changed, (
            "test setup did not actually mutate row 1's recurrent state; "
            "the row-isolation assertion below would be vacuous."
        )

        # Now restore row 0 from the snapshot. Row 1 must NOT regress
        # to its earlier state — restore is row-isolated.
        adapter.restore_recurrent_state(cache, 0, snap_row0)

        row1_after_restore = adapter.snapshot_recurrent_state(cache, row_idx=1)
        for ref_e, post_e in zip(
            row1_after_mutation.entries, row1_after_restore.entries
        ):
            assert ref_e.layer_idx == post_e.layer_idx
            # The reference and post snapshots come from the same
            # cache; the lazy-vs-populated state agrees per layer, so
            # both fields are jointly None or jointly non-None.
            if ref_e.conv_state is not None:
                assert post_e.conv_state is not None
                assert mx.array_equal(ref_e.conv_state, post_e.conv_state), (
                    f"layer {ref_e.layer_idx} row 1 conv_state changed "
                    f"after row-0 restore — splice contaminated row 1."
                )
            else:
                assert post_e.conv_state is None
            if ref_e.recurrent_state is not None:
                assert post_e.recurrent_state is not None
                assert mx.array_equal(
                    ref_e.recurrent_state, post_e.recurrent_state
                ), (
                    f"layer {ref_e.layer_idx} row 1 recurrent_state "
                    f"changed after row-0 restore — splice contaminated "
                    f"row 1."
                )
            else:
                assert post_e.recurrent_state is None

    def test_restore_row_1_preserves_row_0(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        """Mirror of the row-0 test — restoring at ``row_idx=1`` must
        not touch ``row_idx=0``. Catches off-by-one splice errors."""
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        tokens_a = tokenizer.encode(_PROMPT_A)
        tokens_b = tokenizer.encode(_PROMPT_B)

        max_len = max(len(tokens_a), len(tokens_b))
        padding = [max_len - len(tokens_a), max_len - len(tokens_b)]
        cache = _build_b2_batched_cache(adapter, padding)
        _forward_b2(model, tokens_a, tokens_b, cache)

        snap_row1 = adapter.snapshot_recurrent_state(cache, row_idx=1)

        # Mutate (advance both rows).
        _forward_b2(model, tokens_a, tokens_b, cache)

        row0_before_restore = adapter.snapshot_recurrent_state(
            cache, row_idx=0
        )

        # Restore row 1 — row 0 must stay at its current state.
        adapter.restore_recurrent_state(cache, 1, snap_row1)

        row0_after_restore = adapter.snapshot_recurrent_state(cache, row_idx=0)
        for ref_e, post_e in zip(
            row0_before_restore.entries, row0_after_restore.entries
        ):
            assert ref_e.layer_idx == post_e.layer_idx
            if ref_e.conv_state is not None:
                assert post_e.conv_state is not None
                assert mx.array_equal(ref_e.conv_state, post_e.conv_state), (
                    f"row-1 restore contaminated row 0 conv_state at "
                    f"layer {ref_e.layer_idx}."
                )
            if ref_e.recurrent_state is not None:
                assert post_e.recurrent_state is not None
                assert mx.array_equal(
                    ref_e.recurrent_state, post_e.recurrent_state
                ), (
                    f"row-1 restore contaminated row 0 recurrent_state "
                    f"at layer {ref_e.layer_idx}."
                )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestRowIdxValidation:
    """``row_idx`` is validated up front for every contract case;
    out-of-range values raise rather than silently picking the wrong
    row or returning the snapshot tensor verbatim."""

    def test_snapshot_negative_row_idx_raises(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, _ = adapter_and_model
        cache = _build_b1_batched_cache(adapter)
        with pytest.raises(ValueError, match="non-negative"):
            adapter.snapshot_recurrent_state(cache, row_idx=-1)

    def test_restore_negative_row_idx_raises(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, _ = adapter_and_model
        snap_cache = _build_b1_batched_cache(adapter)
        snapshot = adapter.snapshot_recurrent_state(snap_cache, row_idx=0)
        target = _build_b1_batched_cache(adapter)
        with pytest.raises(ValueError, match="non-negative"):
            adapter.restore_recurrent_state(target, -1, snapshot)

    def test_snapshot_row_idx_out_of_range_raises(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        prefix = tokenizer.encode(_PROMPT_A)
        cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, cache)
        # Live B=1; row_idx=1 must raise.
        with pytest.raises(IndexError, match="out of range"):
            adapter.snapshot_recurrent_state(cache, row_idx=1)

    def test_restore_row_idx_out_of_range_raises(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        """Restore with snap-not-None and live populated must reject
        out-of-range ``row_idx`` rather than splicing nothing."""
        adapter, model = adapter_and_model
        tokenizer = adapter.tokenizer()
        prefix = tokenizer.encode(_PROMPT_A)

        snap_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, snap_cache)
        snapshot = adapter.snapshot_recurrent_state(snap_cache, row_idx=0)

        target_cache = _build_b1_batched_cache(adapter)
        _forward_b1(model, prefix, target_cache)
        # Target is B=1; row_idx=1 must raise.
        with pytest.raises(IndexError, match="out of range"):
            adapter.restore_recurrent_state(target_cache, 1, snapshot)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestProtocolMixin:
    """Sanity check on the runtime-checkable Protocol — callers can
    dispatch on ``isinstance(adapter, RecurrentStateAdapter)``."""

    def test_qwen3_5_is_recurrent_state_adapter(self, adapter_and_model: tuple[Qwen3_5Adapter, Any]) -> None:
        from silica.models.recurrent import RecurrentStateAdapter

        adapter, _ = adapter_and_model
        assert isinstance(adapter, RecurrentStateAdapter)

    def test_qwen3_adapter_is_not_recurrent_state_adapter(self) -> None:
        """The plain Qwen3 adapter does not have recurrent state and
        must not satisfy the mixin. Uses a synthetic adapter so the
        test does not need a Qwen3-0.6B HF cache hit."""
        from silica.models.recurrent import RecurrentStateAdapter

        class _StubAdapter:
            def tokenizer(self) -> None:
                return None

        assert not isinstance(_StubAdapter(), RecurrentStateAdapter)
