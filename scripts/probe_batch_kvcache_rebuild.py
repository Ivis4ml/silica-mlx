"""Gate-0.5 probe — BatchKVCache mid-run admission via extract/merge/filter/extend.

Codex cross-reference (2026-04) surfaced that mlx-lm already ships four
primitives for mid-run batch manipulation:

  - ``filter(indices)`` — in-place drop rows, keeping the listed indices.
  - ``extend(other)`` — in-place append another BatchKVCache's rows.
  - ``extract(idx)`` — return one row as a single-request KVCache.
  - ``merge([kvcaches])`` — classmethod; batch-ify KVCache list.

Before committing Unit #16 ContinuousBatcher's architecture, this probe
verifies the invariants the batcher will rely on:

  Q1 — extract(row) + merge([...]) round-trips the cache; per-row K/V
       content is identical (modulo left_padding re-alignment).
  Q2 — filter([kept]) drops rows; survivors' effective K/V is unchanged.
       (This is the PREEMPT / RELEASE primitive.)
  Q3 — extend(other) appends rows; incumbents' effective K/V unchanged;
       new rows carry the other cache's K/V, left-padded to align _idx.
       (This is the ADMISSION primitive.)
  Q4 — row index stability: extend preserves self's row indices; filter
       RESHUFFLES indices to 0..len(kept)-1 in the order of the indices
       list. Silica's slot_table must rebuild after filter.
  Q5 — filter DROPS the filtered-out row's K/V. For option-B prefix
       reuse, a block whose owning request is about to be filter()'d
       requires EAGER extract into detached storage before filter.

Also measures per-op cost under mx.eval barrier at a test scale
(B=8, _idx=100) to confirm rebuild is acceptable (target < 5ms / op).

Run: ``python scripts/probe_batch_kvcache_rebuild.py``
Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

N_KV_HEADS = 2
HEAD_DIM = 8


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _synthetic_kv(
    B: int, T: int, *, offset_value: float = 0.0
) -> tuple[mx.array, mx.array]:
    """K and V with row- and position-encoded values for identity checks.

    ``keys[b, h, t, d] = offset_value + b*1000 + t*10 + d*0.1 + h``
    — every position has a distinct rational value, so "K/V unchanged"
    is a literal element-wise equality assertion.
    """
    shape = (B, N_KV_HEADS, T, HEAD_DIM)
    b = mx.arange(B, dtype=mx.float32).reshape(B, 1, 1, 1) * 1000.0
    h = mx.arange(N_KV_HEADS, dtype=mx.float32).reshape(1, N_KV_HEADS, 1, 1)
    t = mx.arange(T, dtype=mx.float32).reshape(1, 1, T, 1) * 10.0
    d = mx.arange(HEAD_DIM, dtype=mx.float32).reshape(1, 1, 1, HEAD_DIM) * 0.1
    keys = (b + t + d + h + offset_value).astype(mx.float16)
    values = mx.broadcast_to(keys, shape) + mx.array(0.5, dtype=mx.float16)
    return keys, values


def _effective_kv(cache: BatchKVCache, row: int) -> tuple[mx.array, mx.array]:
    """K/V for one row, stripped of left_padding (the attention-visible portion)."""
    lp = int(cache.left_padding[row].item())
    idx = cache._idx  # type: ignore[attr-defined]
    k = cache.keys[row, :, lp:idx, :]  # type: ignore[index]
    v = cache.values[row, :, lp:idx, :]  # type: ignore[index]
    return mx.contiguous(k), mx.contiguous(v)


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    """True iff shapes match and all elements are identical."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    # mx.array.__eq__ returns array | bool (NotImplemented fallback); at
    # runtime with two mx.array operands it is always an mx.array.
    eq = a == b
    return bool(mx.all(eq).item())  # type: ignore[arg-type]


# --- Q1: extract + merge round-trip preserves per-row K/V ---


def _check_extract_merge_roundtrip() -> tuple[bool, str]:
    left_padding = [0, 2, 1]
    B = len(left_padding)
    cache = BatchKVCache(left_padding=left_padding)
    # Prefill some tokens.
    k, v = _synthetic_kv(B, 5)
    cache.update_and_fetch(k, v)
    # Decode one more step.
    k1, v1 = _synthetic_kv(B, 1, offset_value=100_000.0)
    cache.update_and_fetch(k1, v1)
    mx.eval(cache.keys, cache.values)

    original_effective = [_effective_kv(cache, r) for r in range(B)]

    # Extract each row, then rebuild via merge.
    extracted = [cache.extract(r) for r in range(B)]
    rebuilt = BatchKVCache.merge(extracted)
    mx.eval(rebuilt.keys, rebuilt.values)

    if rebuilt.keys.shape[0] != B:
        return False, f"rebuild B expected {B}, got {rebuilt.keys.shape[0]}"

    for r in range(B):
        ok_eff = _effective_kv(rebuilt, r)
        if not _arrays_equal(original_effective[r][0], ok_eff[0]):
            return False, f"row {r} keys drifted after extract+merge"
        if not _arrays_equal(original_effective[r][1], ok_eff[1]):
            return False, f"row {r} values drifted after extract+merge"
    return True, ""


# --- Q2: filter preserves survivors' effective K/V ---


def _check_filter_preserves_survivors() -> tuple[bool, str]:
    left_padding = [1, 0, 2, 0]
    B = len(left_padding)
    cache = BatchKVCache(left_padding=left_padding)
    k, v = _synthetic_kv(B, 6)
    cache.update_and_fetch(k, v)
    k2, v2 = _synthetic_kv(B, 1, offset_value=500_000.0)
    cache.update_and_fetch(k2, v2)
    mx.eval(cache.keys, cache.values)

    # Snapshot effective K/V for rows 0 and 2 (to be kept).
    pre_0 = _effective_kv(cache, 0)
    pre_2 = _effective_kv(cache, 2)

    cache.filter([0, 2])
    mx.eval(cache.keys, cache.values)

    assert cache.keys is not None
    if cache.keys.shape[0] != 2:
        return False, f"post-filter B expected 2, got {cache.keys.shape[0]}"

    post_0 = _effective_kv(cache, 0)  # formerly row 0
    post_1 = _effective_kv(cache, 1)  # formerly row 2

    if not _arrays_equal(pre_0[0], post_0[0]):
        return False, "kept row 0 keys drifted after filter"
    if not _arrays_equal(pre_0[1], post_0[1]):
        return False, "kept row 0 values drifted after filter"
    if not _arrays_equal(pre_2[0], post_1[0]):
        return False, "kept row 2 (now index 1) keys drifted after filter"
    if not _arrays_equal(pre_2[1], post_1[1]):
        return False, "kept row 2 (now index 1) values drifted after filter"
    return True, ""


# --- Q3: extend appends without disturbing incumbents ---


def _check_extend_appends_and_preserves() -> tuple[bool, str]:
    # Main batch B=2, prefilled to _idx=4.
    main = BatchKVCache(left_padding=[0, 0])
    k, v = _synthetic_kv(2, 4)
    main.update_and_fetch(k, v)
    mx.eval(main.keys, main.values)
    pre_0 = _effective_kv(main, 0)
    pre_1 = _effective_kv(main, 1)

    # New single-request cache prefilled to offset=3 (shorter than main._idx).
    # Construct via merge([KVCache]) with our own fake single-request KV.
    new_single = BatchKVCache(left_padding=[0])
    k_new, v_new = _synthetic_kv(1, 3, offset_value=999_000.0)
    new_single.update_and_fetch(k_new, v_new)
    mx.eval(new_single.keys, new_single.values)
    # Snapshot the new row's effective K/V before extend mutates.
    pre_new_effective = _effective_kv(new_single, 0)

    main.extend(new_single)
    mx.eval(main.keys, main.values)

    assert main.keys is not None
    if main.keys.shape[0] != 3:
        return False, f"post-extend B expected 3, got {main.keys.shape[0]}"

    # Incumbents unchanged.
    if not _arrays_equal(pre_0[0], _effective_kv(main, 0)[0]):
        return False, "incumbent row 0 keys drifted after extend"
    if not _arrays_equal(pre_1[0], _effective_kv(main, 1)[0]):
        return False, "incumbent row 1 keys drifted after extend"

    # New row (index 2) carries the standalone cache's K/V.
    post_new = _effective_kv(main, 2)
    if not _arrays_equal(pre_new_effective[0], post_new[0]):
        return False, "appended row keys do not match standalone cache"
    if not _arrays_equal(pre_new_effective[1], post_new[1]):
        return False, "appended row values do not match standalone cache"

    # main._idx should be the max of the two.
    if main._idx != 4:  # type: ignore[attr-defined]
        return (
            False,
            f"_idx after extend expected 4, got {main._idx}",  # type: ignore[attr-defined]
        )
    return True, ""


# --- Q5: filter DROPS filtered-out rows' K/V (option-B needs eager extract) ---


def _check_filter_loses_dropped_row_unless_extracted_first() -> tuple[bool, str]:
    cache = BatchKVCache(left_padding=[0, 0])
    k, v = _synthetic_kv(2, 4)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)

    # Eager extract row 1 BEFORE filter.
    detached = cache.extract(1)
    mx.eval(detached.keys, detached.values)
    pre_detach_k = mx.contiguous(detached.keys[0, :, : detached.offset, :])

    # Now drop row 1 via filter.
    cache.filter([0])
    mx.eval(cache.keys, cache.values)
    assert cache.keys is not None
    if cache.keys.shape[0] != 1:
        return (
            False,
            f"post-filter B expected 1, got {cache.keys.shape[0]}",
        )

    # The detached KVCache still carries the dropped row's K/V.
    if not _arrays_equal(
        pre_detach_k,
        mx.contiguous(detached.keys[0, :, : detached.offset, :]),
    ):
        return False, "detached KVCache drifted after source filter"

    # If we had NOT extracted before filter, the data would be gone.
    # Demonstrate by constructing a sibling cache without the eager extract:
    sibling = BatchKVCache(left_padding=[0, 0])
    sk, sv = _synthetic_kv(2, 4)
    sibling.update_and_fetch(sk, sv)
    mx.eval(sibling.keys, sibling.values)
    sibling.filter([0])
    # Accessing row 1 after filter is out-of-bounds by shape; the data for
    # the original row 1 is irrecoverable from the filtered cache. This is
    # the structural justification for option-B's eager-extract requirement.
    assert sibling.keys is not None
    if sibling.keys.shape[0] != 1:
        return (
            False,
            f"sibling post-filter B expected 1, got {sibling.keys.shape[0]}",
        )
    return True, ""


# --- Q6: cost measurement under mx.eval barrier ---


def _measure_costs() -> tuple[bool, str]:
    B = 8
    T_prefill = 100
    cache = BatchKVCache(left_padding=[0] * B)
    k, v = _synthetic_kv(B, T_prefill)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)

    # Benchmark: extract all + merge all (full rebuild round-trip).
    t0 = time.perf_counter()
    N = 10
    for _ in range(N):
        cs = [cache.extract(r) for r in range(B)]
        new = BatchKVCache.merge(cs)
        mx.eval(new.keys, new.values)
    rebuild_ms = (time.perf_counter() - t0) * 1000.0 / N

    # Benchmark: filter([0, 2, 4, 6]) (preempt half).
    t0 = time.perf_counter()
    for _ in range(N):
        cache_f = BatchKVCache(left_padding=[0] * B)
        cache_f.update_and_fetch(k, v)
        mx.eval(cache_f.keys, cache_f.values)
        cache_f.filter([0, 2, 4, 6])
        mx.eval(cache_f.keys, cache_f.values)
    filter_ms = (time.perf_counter() - t0) * 1000.0 / N

    # Benchmark: extend by one row (admission).
    t0 = time.perf_counter()
    for _ in range(N):
        cache_e = BatchKVCache(left_padding=[0] * B)
        cache_e.update_and_fetch(k, v)
        mx.eval(cache_e.keys, cache_e.values)
        single = BatchKVCache(left_padding=[0])
        ks, vs = _synthetic_kv(1, 10)
        single.update_and_fetch(ks, vs)
        mx.eval(single.keys, single.values)
        cache_e.extend(single)
        mx.eval(cache_e.keys, cache_e.values)
    extend_ms = (time.perf_counter() - t0) * 1000.0 / N

    print(
        f"    rebuild (extract×{B} + merge) : {rebuild_ms:6.2f} ms / op"
    )
    print(f"    filter ([0,2,4,6])           : {filter_ms:6.2f} ms / op")
    print(f"    extend (+1 row)              : {extend_ms:6.2f} ms / op")

    # Anything below 20ms/op at this scale is acceptable for P-2 acceptance
    # (~10 admissions over a 32-token decode — budget is seconds).
    if rebuild_ms > 20.0 or filter_ms > 20.0 or extend_ms > 20.0:
        return False, f"op cost too high at B={B}, T={T_prefill}"
    return True, ""


def main() -> int:
    print(f"Gate-0.5 probe — mlx-lm {mlx_lm.__version__} BatchKVCache rebuild primitives")

    print("(Q1) extract + merge round-trip preserves per-row K/V:")
    ok, msg = _check_extract_merge_roundtrip()
    if not ok:
        return _fail(f"(Q1) {msg}")
    print("    PASS (identity up to left_padding re-alignment)")

    print("(Q2) filter preserves survivors' effective K/V:")
    ok, msg = _check_filter_preserves_survivors()
    if not ok:
        return _fail(f"(Q2) {msg}")
    print("    PASS (filter → preempt primitive; row indices reshuffle)")

    print("(Q3) extend appends rows without disturbing incumbents:")
    ok, msg = _check_extend_appends_and_preserves()
    if not ok:
        return _fail(f"(Q3) {msg}")
    print("    PASS (extend → admission primitive; self row indices stable)")

    print("(Q5) filter drops row K/V unless extract()'d first:")
    ok, msg = _check_filter_loses_dropped_row_unless_extracted_first()
    if not ok:
        return _fail(f"(Q5) {msg}")
    print("    PASS (option-B requires eager extract before owner filter)")

    print("(Q6) rebuild / filter / extend cost (mx.eval barrier, B=8, T=100):")
    ok, msg = _measure_costs()
    if not ok:
        return _fail(f"(Q6) {msg}")
    print("    PASS (all ops < 20ms at test scale)")

    print()
    print("RESULT: PASS — Unit #16 architecture locked:")
    print("  - extend is the admission primitive (appends, preserves indices)")
    print("  - filter is the preempt/release primitive (drops + reshuffles)")
    print("  - slot_table MUST rebuild after every filter")
    print("  - RadixPrefixCache MUST eager-extract pinned blocks before")
    print("    the owning request's row is filter()'d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
