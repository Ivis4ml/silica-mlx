"""Gate-0.75 probe — seeded BatchKVCache + detached-slice lifetime.

Unit 16c.2 plans to install a radix-prefix-cache admission path that
**seeds** a per-row BatchKVCache from detached K/V slices (instead of
re-running prefill over a shared prefix), then extends that seeded cache
into the main batched cache. Three physical assumptions underpin this
design; none of them were exercised by Gate-0.5. This probe converts
them into pass/fail gates **before** any 16c.2 code change.

  A.1 — `cache.state = (K, V, offset, left_padding)` setter seeds an
        empty BatchKVCache equivalently to calling `update_and_fetch`
        on an empty cache with the same (K, V). "Equivalently" means
        bit-identical state tuple after seeding: keys, values, offset,
        left_padding, _idx. If the state setter leaves any field in an
        inconsistent state relative to update_and_fetch, 16c.2's
        `_seed_batch_cache_state` helper must go through a different
        path (explicit update_and_fetch replay).

  A.2 — `main_cache.extend(row_cache)` where `row_cache._idx > 0` (i.e.
        the row cache was seeded rather than accumulated via
        update_and_fetch) produces the same extended state as the
        reference path where both main and row were accumulated
        naturally. This is Open-Q Q1 from plans/P2_UNIT_16C_2_PREP.md
        §9 — a hard gate. If extend requires `other._idx == 0`,
        option-B prefix reuse collapses to "seed + replay prefill",
        which defeats the whole compute-saving point.

  B   — `mx.contiguous(cache.keys[row, :, a:b, :])` with `mx.eval`
        forcing materialisation produces a slice whose contents survive
        a subsequent `cache.filter([other_rows])` call that would have
        dropped or shifted the source memory. 16c.2's reclaim path
        slices detached K/V out of a row immediately BEFORE the filter
        that drops that row; the slice must remain correct afterward.

Exit code: 0 on PASS (all three gates), 1 on FAIL.

Run: ``python scripts/probe_16c2_seed_and_detach.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

N_KV_HEADS = 2
HEAD_DIM = 8


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _synthetic_kv(
    B: int, T: int, *, offset_value: float = 0.0
) -> tuple[mx.array, mx.array]:
    """Identity-encoded K/V so equality checks are elementwise-literal.

    keys[b, h, t, d] = offset_value + b*1000 + t*10 + d*0.1 + h
    values = keys + 0.5
    """
    b = mx.arange(B, dtype=mx.float32).reshape(B, 1, 1, 1) * 1000.0
    h = mx.arange(N_KV_HEADS, dtype=mx.float32).reshape(1, N_KV_HEADS, 1, 1)
    t = mx.arange(T, dtype=mx.float32).reshape(1, 1, T, 1) * 10.0
    d = mx.arange(HEAD_DIM, dtype=mx.float32).reshape(1, 1, 1, HEAD_DIM) * 0.1
    keys = (b + t + d + h + offset_value).astype(mx.float16)
    values = (keys + mx.array(0.5, dtype=mx.float16)).astype(mx.float16)
    # Broadcast keys to full (B, H, T, D) shape (they already are, since
    # b/h/t/d broadcast-combine to that shape).
    return keys, values


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    if tuple(a.shape) != tuple(b.shape):
        return False
    eq = a == b
    return bool(mx.all(eq).item())  # type: ignore[arg-type]


def _cache_tensors(cache: BatchKVCache, label: str) -> tuple[mx.array, mx.array]:
    keys = cache.keys
    values = cache.values
    if keys is None or values is None:
        raise AssertionError(f"{label}: cache tensors are not initialised")
    return keys, values


# --- A.1: state setter vs update_and_fetch on an empty cache ---


def _probe_a1_state_setter_equivalence() -> tuple[bool, str]:
    """Two empty caches, one seeded via state setter, one via update_and_fetch.

    After the same (K, V, offset, left_padding) target, their full state
    tuples must be bit-identical.
    """
    B = 2
    T = 6  # deliberately not a multiple of BatchKVCache.step (256), so
    # update_and_fetch preallocates a padded tail and the probe must compare
    # only the live [:_idx] region.
    left_padding = [1, 0]
    k, v = _synthetic_kv(B, T)

    # Reference path: fresh cache, update_and_fetch.
    ref = BatchKVCache(left_padding=left_padding)
    ref.update_and_fetch(k, v)
    mx.eval(ref.keys, ref.values)
    ref_keys, ref_values = _cache_tensors(ref, "A.1 ref")

    # Seeded path: fresh cache, state setter with the SAME tuple.
    seeded = BatchKVCache(left_padding=left_padding)
    # Derive the offset/left_padding values that update_and_fetch would
    # have produced. BatchKVCache.__init__ sets offset = [-l for l in lp],
    # then update_and_fetch adds keys.shape[2]. So the target offset is
    # [T - l for l in left_padding].
    target_offset = mx.array([T - lp for lp in left_padding])
    target_left_padding = mx.array(left_padding)
    # The state property (getter) returns k/v truncated to _idx; we seed
    # with the same truncation so the setter doesn't see padded tail.
    seeded_keys = ref_keys[..., :T, :]  # (B, H, T, D) — no step-padding tail
    seeded_values = ref_values[..., :T, :]
    seeded.state = (seeded_keys, seeded_values, target_offset, target_left_padding)
    mx.eval(seeded.keys, seeded.values)
    seeded_live_keys, seeded_live_values = _cache_tensors(seeded, "A.1 seeded")

    # Compare every surface field.
    if seeded._idx != T:
        return False, f"A.1 _idx: seeded={seeded._idx} expected={T}"
    # After the state setter, seeded.keys has shape (B, H, T, D) — the
    # update_and_fetch path pads to step=256. Compare only the live
    # [:T] region and assert _idx matches, which is the load-bearing
    # equivalence. Tail padding beyond _idx is unobservable to callers.
    if not _arrays_equal(seeded_live_keys[..., :T, :], ref_keys[..., :T, :]):
        return False, "A.1 keys region [:T] differs from reference"
    if not _arrays_equal(seeded_live_values[..., :T, :], ref_values[..., :T, :]):
        return False, "A.1 values region [:T] differs from reference"
    if not _arrays_equal(seeded.offset, ref.offset):
        return False, (
            f"A.1 offset: seeded={seeded.offset.tolist()} "
            f"ref={ref.offset.tolist()}"
        )
    if not _arrays_equal(seeded.left_padding, ref.left_padding):
        return False, (
            f"A.1 left_padding: seeded={seeded.left_padding.tolist()} "
            f"ref={ref.left_padding.tolist()}"
        )

    # Behavioural check: a subsequent update_and_fetch on the seeded
    # cache must advance state the same way as on the reference.
    k2, v2 = _synthetic_kv(B, 1, offset_value=50_000.0)
    seeded.update_and_fetch(k2, v2)
    ref.update_and_fetch(k2, v2)
    mx.eval(seeded.keys, seeded.values, ref.keys, ref.values)
    seeded_post_keys, _ = _cache_tensors(seeded, "A.1 seeded post-uaf")
    ref_post_keys, _ = _cache_tensors(ref, "A.1 ref post-uaf")
    if seeded._idx != ref._idx:
        return False, (
            f"A.1 post-uaf _idx drift: seeded={seeded._idx} ref={ref._idx}"
        )
    if not _arrays_equal(
        seeded_post_keys[..., : seeded._idx, :], ref_post_keys[..., : ref._idx, :]
    ):
        return False, "A.1 post-uaf keys region differs — seeding corrupted state"

    return True, "A.1 PASS: state setter equivalent to update_and_fetch"


# --- A.2: seeded extend with row_cache._idx > 0 ---


def _probe_a2_seeded_extend() -> tuple[bool, str]:
    """Extend a main cache with a row cache that was SEEDED (not uaf'd).

    Reference: both main and row accumulated via update_and_fetch, then
    extended. Candidate: row seeded via state setter at the same target,
    then extended. Final state must match row-by-row.
    """
    # Main cache: B_main=2 rows, _idx=5.
    main_lp = [0, 1]
    k_main, v_main = _synthetic_kv(2, 5)
    main_ref = BatchKVCache(left_padding=main_lp)
    main_ref.update_and_fetch(k_main, v_main)
    main_cand = BatchKVCache(left_padding=main_lp)
    main_cand.update_and_fetch(k_main, v_main)

    # Row cache: B_row=1 row, _idx=4 (shorter prefix — typical prefix
    # hit case where the new row starts with less context than main).
    row_T = 4
    row_lp = [0]
    k_row, v_row = _synthetic_kv(1, row_T, offset_value=9_000.0)

    # Reference row: update_and_fetch path.
    row_ref = BatchKVCache(left_padding=row_lp)
    row_ref.update_and_fetch(k_row, v_row)
    mx.eval(row_ref.keys, row_ref.values)
    row_ref_keys, row_ref_values = _cache_tensors(row_ref, "A.2 row_ref")

    # Candidate row: state setter seeding.
    row_cand = BatchKVCache(left_padding=row_lp)
    row_cand_offset = mx.array([row_T - row_lp[0]])
    row_cand_lp = mx.array(row_lp)
    row_cand.state = (
        row_ref_keys[..., :row_T, :],
        row_ref_values[..., :row_T, :],
        row_cand_offset,
        row_cand_lp,
    )
    mx.eval(row_cand.keys, row_cand.values)

    # Sanity: seeded row must have _idx == row_T (A.1 territory, but
    # explicitly re-check here since it's load-bearing for extend).
    if row_cand._idx != row_T:
        return False, f"A.2 precondition: seeded row _idx={row_cand._idx} != {row_T}"

    # Extend both.
    main_ref.extend(row_ref)
    main_cand.extend(row_cand)
    mx.eval(main_ref.keys, main_ref.values, main_cand.keys, main_cand.values)
    main_ref_keys, main_ref_values = _cache_tensors(main_ref, "A.2 main_ref")
    main_cand_keys, main_cand_values = _cache_tensors(main_cand, "A.2 main_cand")

    # Full-state equality.
    if main_ref._idx != main_cand._idx:
        return False, (
            f"A.2 _idx drift after extend: "
            f"ref={main_ref._idx} cand={main_cand._idx}"
        )
    if tuple(main_ref_keys.shape) != tuple(main_cand_keys.shape):
        return False, (
            f"A.2 keys shape drift: "
            f"ref={main_ref_keys.shape} cand={main_cand_keys.shape}"
        )
    if not _arrays_equal(main_ref_keys, main_cand_keys):
        return False, "A.2 extended keys differ between ref and seeded-row path"
    if not _arrays_equal(main_ref_values, main_cand_values):
        return False, "A.2 extended values differ between ref and seeded-row path"
    if not _arrays_equal(main_ref.offset, main_cand.offset):
        return False, (
            f"A.2 offset drift: ref={main_ref.offset.tolist()} "
            f"cand={main_cand.offset.tolist()}"
        )
    if not _arrays_equal(main_ref.left_padding, main_cand.left_padding):
        return False, (
            f"A.2 left_padding drift: "
            f"ref={main_ref.left_padding.tolist()} "
            f"cand={main_cand.left_padding.tolist()}"
        )

    # Behavioural check: one more decode step on both; still identical.
    k2, v2 = _synthetic_kv(
        main_ref_keys.shape[0], 1, offset_value=70_000.0
    )
    main_ref.update_and_fetch(k2, v2)
    main_cand.update_and_fetch(k2, v2)
    mx.eval(main_ref.keys, main_ref.values, main_cand.keys, main_cand.values)
    main_ref_post_keys, _ = _cache_tensors(main_ref, "A.2 main_ref post-decode")
    main_cand_post_keys, _ = _cache_tensors(main_cand, "A.2 main_cand post-decode")
    if not _arrays_equal(main_ref_post_keys, main_cand_post_keys):
        return False, "A.2 post-extend decode step diverges"

    return True, "A.2 PASS: seeded extend equivalent to uaf-accumulated extend"


# --- B: extracted slice survives a filter that drops the source row ---


def _probe_b_extract_slice_lifetime() -> tuple[bool, str]:
    """Slice K/V out of a row via mx.contiguous + mx.eval, then filter that
    row out of the source cache. The slice must still be readable and
    bit-identical to its pre-filter snapshot.

    This is the reclaim-path invariant: the batcher extracts detached K/V
    for the terminating row BEFORE calling filter, and trusts that the
    extracted tensors are independent of the source's subsequent
    mutation.
    """
    B = 3
    T = 5
    left_padding = [0, 2, 1]
    cache = BatchKVCache(left_padding=left_padding)
    k, v = _synthetic_kv(B, T)
    cache.update_and_fetch(k, v)
    # Decode step so _idx > T and there's a tail to prove independence.
    k1, v1 = _synthetic_kv(B, 1, offset_value=30_000.0)
    cache.update_and_fetch(k1, v1)
    mx.eval(cache.keys, cache.values)
    cache_keys, cache_values = _cache_tensors(cache, "B source")

    # Pretend row 1 is terminating. Slice out its K/V (full effective
    # region [lp:_idx]). mx.contiguous forces a copy; mx.eval forces
    # materialisation so the slice is backed by independent memory.
    target_row = 1
    lp = int(cache.left_padding[target_row].item())
    idx = cache._idx
    k_slice = mx.contiguous(cache_keys[target_row : target_row + 1, :, lp:idx, :])
    v_slice = mx.contiguous(cache_values[target_row : target_row + 1, :, lp:idx, :])
    mx.eval(k_slice, v_slice)

    # Snapshot the slice values BEFORE filter, for later comparison.
    k_snapshot = mx.array(k_slice)  # defensive copy; cheap
    v_snapshot = mx.array(v_slice)
    mx.eval(k_snapshot, v_snapshot)

    # Filter out row 1 (drop it). filter() may shift-left the source by
    # min(left_padding[kept]); that internal mutation must not reach
    # our detached slice.
    kept = [0, 2]
    cache.filter(kept)
    mx.eval(cache.keys, cache.values)
    filtered_keys, _ = _cache_tensors(cache, "B filtered")

    # The slice variables should still be valid.
    if tuple(k_slice.shape) != tuple(k_snapshot.shape):
        return False, (
            f"B k_slice shape changed after filter: "
            f"{k_slice.shape} vs {k_snapshot.shape}"
        )
    if not _arrays_equal(k_slice, k_snapshot):
        return False, "B k_slice content changed after filter — not detached!"
    if not _arrays_equal(v_slice, v_snapshot):
        return False, "B v_slice content changed after filter — not detached!"

    # One more forcing-mutation: append a decode step to the filtered
    # cache. Still must not touch our slice.
    B_after = filtered_keys.shape[0]
    k2, v2 = _synthetic_kv(B_after, 1, offset_value=99_000.0)
    cache.update_and_fetch(k2, v2)
    mx.eval(cache.keys, cache.values)
    _cache_tensors(cache, "B post-uaf")
    if not _arrays_equal(k_slice, k_snapshot):
        return False, "B k_slice changed after post-filter update_and_fetch"
    if not _arrays_equal(v_slice, v_snapshot):
        return False, "B v_slice changed after post-filter update_and_fetch"

    return True, "B PASS: extracted slice independent of source filter / uaf"


def main() -> int:
    all_pass = True
    for probe in (
        _probe_a1_state_setter_equivalence,
        _probe_a2_seeded_extend,
        _probe_b_extract_slice_lifetime,
    ):
        ok, msg = probe()
        print(msg)
        if not ok:
            all_pass = False
    if all_pass:
        print("\nALL PASS — Gate-0.75 green, proceed to 16c.2 step 2.")
        return 0
    print("\nONE OR MORE FAILED — pivot decision required before step 2.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
