"""mlx-lm ``BatchKVCache.prepare(right_padding=...) + finalize()`` pin.

D-021 step 5 sub-unit (c) slice-1 deliverable 0. The speculative
verify path leans on this primitive for per-row variable rollback
under Option C.1 padding (`plans/P6_SPEC_FOUNDATION_C_ORIENTATION.md`
§4 [F-3]). The primitive is external (mlx-lm) and load-bearing —
this file pins its observable contract before any silica patch
consumes it, so a future mlx-lm semantic drift fails here, not deep
in the spec-active batcher path.

Pinned behaviours:

  - **Per-row right-trim signature.** ``prepare(right_padding=[0, 3])``
    followed by ``finalize()`` on a B=2 cache filled with 6 K/V
    positions uniformly produces ``offset == [6, 3]``,
    ``left_padding == [0, 3]``, ``_idx == 6``. Row 0's K/V
    contents are untouched; row 1's first three positions become
    mask (rolled-in zeros from the un-written buffer tail) and
    its last three carry what were originally row 1's first three
    K/V values — the dynamic-roll-to-the-right signature.
  - **All-zero short-circuit.** ``prepare(right_padding=[0, 0])``
    leaves ``_right_padding`` at ``None`` because the
    ``max(right_padding) > 0`` guard short-circuits; the
    subsequent ``finalize()`` is then a no-op. State invariants
    after the roundtrip equal the pre-call state byte-for-byte.
  - **Post-trim ``update_and_fetch`` advances per-row offset
    correctly.** After a non-trivial trim, feeding one new K/V
    position via ``update_and_fetch`` advances ``offset`` per
    row uniformly by 1 — so the trimmed row at offset 3 reaches
    offset 4 and the kept row at offset 6 reaches offset 7. This
    is the contract the spec-active decode loop relies on for
    "verify forward → trim rejected → next-step decode" sequencing.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache


def _filled_two_row_cache(
    *, n_positions: int, head_dim: int = 4
) -> tuple[BatchKVCache, mx.array, mx.array]:
    """Build a B=2 BatchKVCache with ``n_positions`` distinct K/V tokens.

    K row 0 carries values 0..(n_positions * head_dim - 1) reshaped to
    (1, 1, n_positions, head_dim); row 1 starts from
    ``n_positions * head_dim`` so the two rows are easy to tell
    apart in assertions. V is K + 100 to prove K and V follow the
    same per-row roll arithmetic.
    """
    B = 2
    H = 1
    T = n_positions
    D = head_dim
    keys_seed = mx.arange(B * H * T * D, dtype=mx.float32).reshape(B, H, T, D)
    vals_seed = (
        mx.arange(B * H * T * D, dtype=mx.float32) + 100.0
    ).reshape(B, H, T, D)
    cache = BatchKVCache(left_padding=[0, 0])
    cache.update_and_fetch(keys_seed, vals_seed)
    return cache, keys_seed, vals_seed


def _as_list(arr: mx.array) -> list[int]:
    """Narrow an ``mx.array.tolist()`` result for mypy.

    mlx stubs declare ``tolist()`` as returning a union covering scalar
    and nested-list shapes; the per-row ledger fields used here are
    always 1-D arrays, so the result is a flat ``list[int]`` after the
    ``isinstance`` narrowing.
    """
    raw = arr.tolist()
    assert isinstance(raw, list)
    out: list[int] = []
    for x in raw:
        assert isinstance(x, int | float)
        out.append(int(x))
    return out


def test_prepare_right_padding_per_row_dynamic_roll_signature() -> None:
    """`prepare(right_padding=[0, 3]) + finalize()` rolls row 1 by 3.

    Pins the exact post-roll contents the spec helper relies on:
    row 0 untouched, row 1's last three positions carry what were
    originally its first three (cyclic-shift-to-the-right by 3),
    and offset / left_padding / _idx match the per-row trim ledger.
    """
    cache, keys_seed, vals_seed = _filled_two_row_cache(n_positions=6)

    # Pre-call sanity: every position has the expected seed values.
    assert _as_list(cache.offset) == [6, 6]
    assert _as_list(cache.left_padding) == [0, 0]
    assert cache._idx == 6
    assert cache.keys is not None
    assert cache.values is not None

    cache.prepare(right_padding=[0, 3])
    cache.finalize()

    # Per-row offset / left_padding bookkeeping matches the F-3 doc.
    assert _as_list(cache.offset) == [6, 3]
    assert _as_list(cache.left_padding) == [0, 3]
    # _idx is the buffer high-water mark; the roll moves data without
    # changing the write head, so it stays at 6.
    assert cache._idx == 6
    assert cache.keys is not None
    assert cache.values is not None

    # Row 0 (right_padding=0) keys/values unchanged at axis-2 [0, 6).
    assert mx.array_equal(
        cache.keys[0, :, : cache._idx, :], keys_seed[0:1, :, :, :][0]
    ).item()
    assert mx.array_equal(
        cache.values[0, :, : cache._idx, :], vals_seed[0:1, :, :, :][0]
    ).item()

    # Row 1 dynamic-roll signature: positions [3, 6) carry what were
    # originally positions [0, 3); positions [0, 3) carry buffer-tail
    # zeros (the cache buffer was rounded up to the step=256 capacity
    # when first allocated, never written past position 6).
    assert mx.array_equal(
        cache.keys[1, :, 3:6, :], keys_seed[1, :, 0:3, :]
    ).item()
    assert mx.array_equal(
        cache.values[1, :, 3:6, :], vals_seed[1, :, 0:3, :]
    ).item()
    assert mx.all(cache.keys[1, :, 0:3, :] == 0).item()
    assert mx.all(cache.values[1, :, 0:3, :] == 0).item()


def test_prepare_right_padding_all_zero_short_circuits() -> None:
    """`prepare(right_padding=[0, 0])` leaves the cache untouched.

    The mlx-lm guard ``if right_padding is not None and max(...) > 0``
    short-circuits when every row's trim is zero; ``_right_padding``
    stays None so the subsequent ``finalize`` does not touch K/V.
    Pinning this matters: the spec helper unconditionally calls
    prepare+finalize each cycle, and the all-zero short-circuit is
    what guarantees full-accept cycles are byte-identical to a
    no-rollback cycle.
    """
    cache, keys_seed, vals_seed = _filled_two_row_cache(n_positions=6)
    pre_offset = _as_list(cache.offset)
    pre_left_padding = _as_list(cache.left_padding)
    pre_idx = cache._idx

    cache.prepare(right_padding=[0, 0])
    # ``_right_padding`` is the documented no-op signal.
    assert cache._right_padding is None

    cache.finalize()

    # Offsets, padding, and physical contents survive the roundtrip
    # exactly.
    assert _as_list(cache.offset) == pre_offset
    assert _as_list(cache.left_padding) == pre_left_padding
    assert cache._idx == pre_idx
    assert cache.keys is not None
    assert cache.values is not None
    assert mx.array_equal(
        cache.keys[:, :, : cache._idx, :], keys_seed
    ).item()
    assert mx.array_equal(
        cache.values[:, :, : cache._idx, :], vals_seed
    ).item()


def test_update_and_fetch_after_trim_advances_per_row_offset() -> None:
    """One-token append after a per-row trim advances every offset by 1.

    The spec-active decode loop's next step feeds one new token per
    row through ``update_and_fetch``; this pin proves the per-row
    offset advances uniformly so the trimmed row's logical end (3)
    reaches 4 and the kept row's (6) reaches 7. Without this
    contract, "verify forward → trim → next-step decode" would
    desync the per-row offset ledger.
    """
    cache, _, _ = _filled_two_row_cache(n_positions=6)
    cache.prepare(right_padding=[0, 3])
    cache.finalize()
    assert _as_list(cache.offset) == [6, 3]

    # Append one new K/V position to both rows (B=2, H=1, T=1, D=4).
    new_k = mx.full((2, 1, 1, 4), 7.0, dtype=mx.float32)
    new_v = mx.full((2, 1, 1, 4), 17.0, dtype=mx.float32)
    cache.update_and_fetch(new_k, new_v)

    # Per-row offset advanced by 1 uniformly: row 0 6→7, row 1 3→4.
    assert _as_list(cache.offset) == [7, 4]
    # _idx advances by the appended token count (uniform).
    assert cache._idx == 7
    # left_padding is unchanged by update_and_fetch — only finalize
    # mutates it.
    assert _as_list(cache.left_padding) == [0, 3]
    assert cache.keys is not None
    assert cache.values is not None
    # The new K row at the appended position carries the seed values
    # for both rows (row 0 directly, row 1 via the same axis-2 write
    # site since update_and_fetch advances `_idx` uniformly).
    assert mx.all(cache.keys[:, :, 6:7, :] == 7.0).item()
    assert mx.all(cache.values[:, :, 6:7, :] == 17.0).item()


@pytest.mark.parametrize("trim_n", [0, 1, 3, 5, 6])
def test_per_row_trim_amount_sweep(trim_n: int) -> None:
    """Sweep the right-padding amount on row 1 with row 0 untouched.

    Pins that the offset arithmetic is linear across the full range
    [0, n_positions]: row 1's offset becomes ``6 - trim_n`` and
    left_padding becomes ``trim_n``. This sanity-pins the formula
    the spec helper uses to compute per-row right_padding from
    yielded_count.
    """
    cache, _, _ = _filled_two_row_cache(n_positions=6)
    cache.prepare(right_padding=[0, trim_n])
    cache.finalize()

    assert _as_list(cache.offset) == [6, 6 - trim_n]
    assert _as_list(cache.left_padding) == [0, trim_n]
    assert cache._idx == 6
