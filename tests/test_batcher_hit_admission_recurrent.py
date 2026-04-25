"""P-3-C5.3.3a — Phase-B classifier + hit-admission ancestor seeding /
restore / suffix slice + two test-only flags.

Synthetic-adapter coverage. The slice-regime byte-exact-vs-oracle gate
on Qwen3.5-0.8B lands separately at C5.3.3b.

Acceptance shape per ``docs/P3_C5_3_DESIGN.md`` §4.4 + the C5.3.3
recon findings:

- Two test-only ``ContinuousBatcher.__init__`` flags:
  ``_allow_recurrent_prefix_cache_for_c5_3_testing`` bypasses the
  hybrid + prefix_cache ctor guard; without it the ctor still raises.
  ``_force_recurrent_slice_prefill_for_c5_3_oracle: int | None``
  flips ``_slice_prefill_active()`` for hybrid + prefix_cache=None
  using its int value as the slice block_size.
- Phase-B classifier consults ``peek_with_node(prompt[:usable])`` —
  i.e., the deepest USABLE node, NOT the raw deepest leaf — and
  routes to miss when that node lacks a recurrent snapshot. The
  ``peek_with_node`` walk is side-effect-free; no ``retain_hit``
  fires before the routing decision (design §3.5).
- ``_admit_single_hit_row`` performs three additive insertions when
  the adapter is recurrent:
  - **A**: walk ``deepest_usable``'s parent chain, populate
    ``row.recurrent_snapshots_per_block[i]`` for i ∈ [0, K) from
    each ancestor's ``recurrent_snapshot`` (legacy ``None``
    ancestors leave the key absent), and set
    ``row.absolute_consumed_tokens = K * block_size``.
  - **B**: ``adapter.restore_recurrent_state(row_cache, 0, snap)``
    with ``snap = deepest_usable.recurrent_snapshot``.
  - **C**: suffix prefill routes through the slice helper when
    ``_slice_prefill_active()`` is True — captures at absolute
    indices ``K, K+1, ...``.
- Restore strictly precedes the first suffix slice forward (call
  order verified via spy log).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import pytest

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import AttentionKind
from silica.models.recurrent import (
    RecurrentSnapshot,
    _RecurrentLayerEntry,
)
from silica.scheduler.batcher import ContinuousBatcher

BLOCK_SIZE = 4
N_KV_HEADS = 1  # _ScriptedModel.N_KV
HEAD_DIM = 4  # _ScriptedModel.HEAD_DIM


_ScriptStep = int | Sequence[int]


@dataclass
class _Event:
    kind: str  # "snapshot" | "restore"
    row_idx: int
    marker: int


@dataclass
class _SpyLog:
    """Unified-order log of snapshot/restore calls.

    Each entry preserves the order calls happened so a test can
    assert "restore preceded the first snapshot" by checking
    the events list directly.
    """

    events: list[_Event] = field(default_factory=list)
    next_marker: int = 0


def _make_recurrent_hybrid_spy_adapter(
    log: _SpyLog,
    *,
    n_layers: int = 1,
    script: Sequence[_ScriptStep] = (),
) -> Any:
    """Synthetic adapter: HYBRID_DELTANET pattern (so
    ``capabilities().has_recurrent_state == True``) + RecurrentState
    Adapter mixin. Lets tests exercise the C5.3.3 ctor-guard bypass
    flag and the slice-regime predicate end-to-end.
    """
    from silica.models.adapter import AttentionPattern
    from tests.test_batcher import _ScriptedAdapter

    pattern = AttentionPattern(
        per_layer=tuple(
            AttentionKind.HYBRID_DELTANET for _ in range(n_layers)
        )
    )

    class _RecurrentHybridSpyAdapter(_ScriptedAdapter):
        def snapshot_recurrent_state(
            self, cache_list: list[Any], row_idx: int
        ) -> RecurrentSnapshot:
            marker = log.next_marker
            log.next_marker += 1
            log.events.append(
                _Event(kind="snapshot", row_idx=row_idx, marker=marker)
            )
            return _make_marker_snapshot(marker, n_layers=n_layers)

        def restore_recurrent_state(
            self,
            cache_list: list[Any],
            row_idx: int,
            snapshot: RecurrentSnapshot,
        ) -> None:
            log.events.append(
                _Event(
                    kind="restore",
                    row_idx=row_idx,
                    marker=_read_marker(snapshot),
                )
            )

    return _RecurrentHybridSpyAdapter(
        n_layers=n_layers,
        script=script,
        attention_pattern=pattern,
    )


def _make_marker_snapshot(
    marker: int, *, n_layers: int = 1
) -> RecurrentSnapshot:
    return RecurrentSnapshot(
        entries=tuple(
            _RecurrentLayerEntry(
                layer_idx=layer,
                conv_state=mx.array(
                    [[float(marker)]], dtype=mx.float32
                ),
                recurrent_state=None,
            )
            for layer in range(n_layers)
        ),
        nbytes=0,
    )


def _read_marker(snapshot: RecurrentSnapshot | None) -> int:
    assert snapshot is not None
    conv = snapshot.entries[0].conv_state
    assert conv is not None
    return int(float(conv[0, 0]))


def _per_layer_kv(seed: float, *, n_layers: int = 1) -> list[
    tuple[mx.array, mx.array]
]:
    shape = (1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    return [
        (
            mx.full(shape, seed + layer, dtype=mx.float16),
            mx.full(shape, seed + layer + 0.5, dtype=mx.float16),
        )
        for layer in range(n_layers)
    ]


def _detached_blocks(
    n_blocks: int, *, n_layers: int = 1
) -> list[list[tuple[mx.array, mx.array]]]:
    return [
        _per_layer_kv(seed=b * 100.0, n_layers=n_layers)
        for b in range(n_blocks)
    ]


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _prefix_cache(block_size: int = BLOCK_SIZE) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def _prep_cohort(batcher: ContinuousBatcher) -> None:
    """One empty ``step()`` flips ``_cohort_prepared=True`` so a
    subsequent ``add_request`` routes to the waiting queue instead of
    direct ``_rows`` append. Required for any test exercising the
    Phase-B classifier / ``_admit_single_hit_row`` path — pre-step
    admissions go through ``_prefill_phase``, not the mid-run admit
    pipeline."""
    assert batcher.step() == []  # 0 rows, 0 work


# --- two test-only flags ---


class TestCtorGuardBypassFlag:
    def test_ctor_raises_without_flag(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        with pytest.raises(NotImplementedError, match="has_recurrent_state"):
            ContinuousBatcher(adapter, prefix_cache=_prefix_cache())

    def test_ctor_succeeds_with_flag(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        # Flag opens the hybrid + prefix_cache combination for tests.
        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=_prefix_cache(),
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        assert batcher._slice_prefill_active() is True

    def test_flag_does_not_affect_non_hybrid_adapter_ctor(self) -> None:
        # Non-recurrent adapter constructs without the flag (no guard
        # to bypass); the flag is a no-op there.
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1)
        ContinuousBatcher(
            adapter,
            prefix_cache=_prefix_cache(),
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )


class TestOracleSliceFlag:
    def test_default_off_for_hybrid_no_prefix_cache(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        # Hybrid + prefix_cache=None defaults to off — production
        # behaviour stays contiguous.
        batcher = ContinuousBatcher(adapter, prefix_cache=None)
        assert batcher._slice_prefill_active() is False

    def test_int_value_activates_slice_regime(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=None,
            _force_recurrent_slice_prefill_for_c5_3_oracle=BLOCK_SIZE,
        )
        assert batcher._slice_prefill_active() is True
        assert batcher._effective_slice_block_size() == BLOCK_SIZE

    def test_flag_inert_on_non_recurrent_adapter(self) -> None:
        # Predicate's first clause (RecurrentStateAdapter) gates the
        # flag — non-recurrent adapters stay False even with the flag.
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1)
        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=None,
            _force_recurrent_slice_prefill_for_c5_3_oracle=BLOCK_SIZE,
        )
        assert batcher._slice_prefill_active() is False

    def test_zero_block_size_rejected(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        with pytest.raises(ValueError, match="must be > 0"):
            ContinuousBatcher(
                adapter,
                prefix_cache=None,
                _force_recurrent_slice_prefill_for_c5_3_oracle=0,
            )

    def test_negative_block_size_rejected(self) -> None:
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(log)
        with pytest.raises(ValueError, match="must be > 0"):
            ContinuousBatcher(
                adapter,
                prefix_cache=None,
                _force_recurrent_slice_prefill_for_c5_3_oracle=-4,
            )

    def test_oracle_path_produces_block_size_aligned_captures(
        self,
    ) -> None:
        # Drive a 2-block prefill in oracle mode; captures must fire
        # at the flag's block_size, not some other value.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 1)
        )
        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=None,
            _force_recurrent_slice_prefill_for_c5_3_oracle=BLOCK_SIZE,
        )
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == 2 * BLOCK_SIZE
        assert set(row.recurrent_snapshots_per_block.keys()) == {0, 1}


# --- Phase-B classifier ---


def _seed_tree_with_snapshots(
    pc: RadixPrefixCache,
    prompt: list[int],
    snapshots: list[RecurrentSnapshot | None],
) -> None:
    """Helper: insert ``len(snapshots)`` blocks of K/V + per-block
    snapshots into the radix tree.
    """
    n_blocks = len(snapshots)
    pc.insert_detached(
        prompt[: n_blocks * BLOCK_SIZE],
        _detached_blocks(n_blocks),
        recurrent_snapshots=snapshots,
    )


class TestPhaseBClampCorrectness:
    """The P2 #1 design fix: Phase-B classifier must check the deepest
    USABLE node (the one ``_admit_single_hit_row`` will restore from),
    not the raw deepest leaf.
    """

    def test_fully_cached_8token_prompt_routes_to_hit_via_4depth_node(
        self,
    ) -> None:
        # block_size=4, 2-block prefix in tree with snapshots at both
        # depths. Submit a prompt EXACTLY equal to the inserted prefix
        # (8 tokens). raw.num_hit_tokens=8 but
        # max_aligned=((8-1)//4)*4=4 → usable=4. Classifier walks
        # peek_with_node(prompt[:4]) → returns the 4-depth node with
        # marker 0 (its snapshot is non-None). Routes to hit.
        # Suffix = 4 tokens = 1 slice forward.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0,)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        snap_b0 = _make_marker_snapshot(900)
        snap_b1 = _make_marker_snapshot(901)
        _seed_tree_with_snapshots(pc, prompt, [snap_b0, snap_b1])

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()
        assert batcher.prefix_hits == 1
        # Restore consumed the 4-depth node's snapshot (marker 900),
        # NOT the 8-depth leaf's (marker 901).
        restore_events = [e for e in log.events if e.kind == "restore"]
        assert len(restore_events) == 1
        assert restore_events[0].marker == 900

    def test_fully_cached_8token_with_snapshotless_4depth_routes_to_miss(
        self,
    ) -> None:
        # Same fully-cached 8-token shape, but the 4-depth node has
        # NO snapshot. Phase-B classifier must route to miss because
        # the node it would restore from has no snapshot — even
        # though the 8-depth leaf DOES have one (the leaf is the
        # raw deepest, not deepest-USABLE). Miss path runs B=1
        # slice helper over the full 8-token prompt → 2 forwards.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 0)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        snap_b1 = _make_marker_snapshot(901)
        _seed_tree_with_snapshots(pc, prompt, [None, snap_b1])

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()
        # Hit counter unchanged: route-to-miss.
        assert batcher.prefix_hits == 0
        # No restore happened (miss path doesn't restore).
        restore_events = [e for e in log.events if e.kind == "restore"]
        assert restore_events == []


class TestPhaseBFallbackToMiss:
    def test_snapshotless_usable_node_routes_to_miss(self) -> None:
        # 12-token prompt + 12-token prefix in tree. block_size=4 →
        # max_aligned=((12-1)//4)*4=8, raw.num_hit_tokens=12 →
        # usable=8. peek_with_node(prompt[:8]) returns the 8-depth
        # node (block index 1). Insert with [snap_a, None, snap_c]:
        # the deepest-usable (8-depth, index 1) is None → miss.
        # Miss path slice-prefills full 12-token prompt → 3 forwards.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 0, 0)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))
        snap_b0 = _make_marker_snapshot(800)
        snap_b2 = _make_marker_snapshot(802)
        _seed_tree_with_snapshots(pc, prompt, [snap_b0, None, snap_b2])

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()
        # Routed to miss; no hit recorded; no restore call.
        assert batcher.prefix_hits == 0
        restore_events = [e for e in log.events if e.kind == "restore"]
        assert restore_events == []


# --- _admit_single_hit_row insertions A / B / C ---


class TestAncestorSeeding:
    def test_two_ancestors_seed_dict_and_counter(self) -> None:
        # K=2 cached, prompt = 3 * block_size → suffix = block_size.
        # max_aligned=((12-1)//4)*4=8, raw.num_hit_tokens=12 →
        # usable=8. Hit path admits with K=2 cached blocks.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0,)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))
        snap_b0 = _make_marker_snapshot(700)
        snap_b1 = _make_marker_snapshot(701)
        # Insert just the 2-block prefix the request will hit on.
        # peek_with_node(prompt[:8]) returns the 8-depth (block 1)
        # node with marker 701; ancestor walk recovers block 0
        # (marker 700).
        pc.insert_detached(
            prompt[: 2 * BLOCK_SIZE],
            _detached_blocks(2),
            recurrent_snapshots=[snap_b0, snap_b1],
        )

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        row = batcher._rows[0]
        # Counter seeded to K * block_size, then suffix slice
        # advances it by one full block (block_size tokens).
        assert row.absolute_consumed_tokens == 3 * BLOCK_SIZE
        # Dict has the two seeded ancestors at indices 0 and 1, plus
        # the suffix-captured snapshot at index 2.
        assert set(row.recurrent_snapshots_per_block.keys()) == {
            0, 1, 2
        }
        # Seeded markers preserved (referenced — same objects).
        assert (
            row.recurrent_snapshots_per_block[0] is snap_b0
        )
        assert (
            row.recurrent_snapshots_per_block[1] is snap_b1
        )
        # Suffix capture's marker is whatever the spy assigned next.
        assert (
            row.recurrent_snapshots_per_block[2]
            is not snap_b0
        )
        assert (
            row.recurrent_snapshots_per_block[2]
            is not snap_b1
        )

    def test_seeding_skips_none_ancestors(self) -> None:
        # Insert with [None, snap_b]: ancestor at depth 1 (block 0)
        # has no snapshot, deepest-usable at depth 2 (block 1) has
        # snap_b. Phase-B routes to hit (deepest-usable has
        # snapshot). Ancestor-seeding walk skips the None entry —
        # dict key 0 absent.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0,)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))
        snap_b1 = _make_marker_snapshot(601)
        pc.insert_detached(
            prompt[: 2 * BLOCK_SIZE],
            _detached_blocks(2),
            recurrent_snapshots=[None, snap_b1],
        )

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        row = batcher._rows[0]
        # Index 0 absent (None ancestor); index 1 seeded; index 2
        # captured during suffix slice.
        assert 0 not in row.recurrent_snapshots_per_block
        assert row.recurrent_snapshots_per_block[1] is snap_b1
        assert 2 in row.recurrent_snapshots_per_block


class TestRestoreBeforeSuffixSlice:
    def test_restore_event_strictly_precedes_first_suffix_snapshot(
        self,
    ) -> None:
        # Drive a hit admission with K=1 cached + suffix of
        # block_size tokens (one suffix slice forward → one
        # snapshot capture). Verify the unified-order log records:
        # [restore, snapshot] in that order — restore happened
        # before the first suffix slice forward.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0,)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        snap_b0 = _make_marker_snapshot(500)
        pc.insert_detached(
            prompt[:BLOCK_SIZE],
            _detached_blocks(1),
            recurrent_snapshots=[snap_b0],
        )

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        # Order check: first event is restore, second is snapshot.
        # No other events should fire on this small case.
        kinds = [e.kind for e in log.events]
        assert kinds == ["restore", "snapshot"], (
            f"expected [restore, snapshot] event order, got {kinds!r}"
        )
        # Restore consumed the K=1 ancestor's marker.
        restore_events = [e for e in log.events if e.kind == "restore"]
        assert restore_events[0].marker == 500


def _walk_chain(
    pc: RadixPrefixCache, tokens: list[int]
) -> list[Any]:
    """Returns the chain of nodes from root's first child down to
    the deepest matched node, in insertion order (block 0 →
    block N-1)."""
    _, deepest = pc.peek_with_node(tokens)
    chain: list[Any] = []
    cursor = deepest
    while cursor is not None and cursor.parent is not None:
        chain.append(cursor)
        cursor = cursor.parent
    chain.reverse()
    return chain


# --- extract-side end-to-end (covers C5.3.2 acceptance bullets that
#     were deferred to C5.3.3 because they depend on hit admission /
#     fallback-to-miss landing first; design §4.3 acceptance #2/#3) ---


class TestExtractAfterFallbackBackfillsLegacy:
    def test_fallback_to_miss_extract_backfills_snapshotless_node(
        self,
    ) -> None:
        # Insert prefix with [None, snap_b1]: 4-depth (block 0)
        # has no snapshot, 8-depth (block 1) has snap_b1. Submit
        # the matching 8-token prompt:
        #   - Phase-B clamp: usable=4 → peek_with_node(prompt[:4])
        #     returns block 0 with recurrent_snapshot=None →
        #     route-to-miss.
        #   - Miss path: B=1 slice-regime → captures fresh markers
        #     M_a, M_b at row.dict[0], row.dict[1].
        #   - Reclaim's _extract_and_insert_prefix walks both
        #     blocks and forwards [M_a, M_b] to insert_detached.
        #   - Block 0: existing.recurrent_snapshot is None →
        #     BACKFILL with M_a (per §3.5.1).
        #   - Block 1: existing snap_b1, new M_b — both non-None →
        #     keep existing.
        #
        # Net effect: the legacy snapshotless node at block 0 is
        # self-healed by the fallback row's slice-prefill captures.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 0)  # 2 miss-path slice forwards
        )
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        snap_b1 = _make_marker_snapshot(901)
        _seed_tree_with_snapshots(pc, prompt, [None, snap_b1])

        # Confirm starting tree state.
        chain_before = _walk_chain(pc, prompt)
        assert chain_before[0].recurrent_snapshot is None
        assert chain_before[1].recurrent_snapshot is snap_b1

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # admit via miss + slice-prefill captures
        batcher.step()  # reclaim → extract → backfill

        assert batcher.prefix_hits == 0  # confirmed miss path
        chain_after = _walk_chain(pc, prompt)
        # Block 0's snapshot was backfilled. The captured marker is
        # the spy's first snapshot call (marker 0, since spy
        # next_marker started at 0).
        assert chain_after[0].recurrent_snapshot is not None
        assert _read_marker(chain_after[0].recurrent_snapshot) == 0
        # Block 1's snapshot is unchanged (existing wins under
        # both-non-None duplicate-keep semantics).
        assert chain_after[1].recurrent_snapshot is snap_b1


class TestHitAdmissionExtractAttachesNewSuffixNodes:
    def test_hit_with_new_suffix_blocks_extract_attaches_new_nodes(
        self,
    ) -> None:
        # Insert K=1 cached block with snap_b0. Submit a 12-token
        # prompt:
        #   - Phase-B: raw.num_hit_tokens=4 (only 1 block in tree),
        #     max_aligned=((12-1)//4)*4=8, usable=min(4,8)=4 →
        #     1-block hit.
        #   - Hit admission: ancestor seed dict[0]=snap_b0,
        #     counter=4, restore snap_b0, suffix slice 8 tokens →
        #     2 forwards → captures markers M_x, M_y at indices 1
        #     and 2.
        #   - Reclaim: extract walks 3 blocks, forwards
        #     [snap_b0, M_x, M_y] to insert_detached.
        #   - Block 0: duplicate-prefix branch — keep snap_b0.
        #   - Block 1: NEW node, attaches M_x.
        #   - Block 2: NEW node, attaches M_y.
        #
        # Net effect: the row's suffix slice captures end up as
        # fresh radix nodes; existing block 0 untouched.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 0)  # 2 suffix slice forwards
        )
        pc = _prefix_cache()
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))
        snap_b0 = _make_marker_snapshot(700)
        # Insert just block 0; blocks 1 and 2 are absent before
        # this row's extract runs.
        pc.insert_detached(
            prompt[:BLOCK_SIZE],
            _detached_blocks(1),
            recurrent_snapshots=[snap_b0],
        )

        # Sanity: only block 0 in the tree before the request runs.
        chain_before = _walk_chain(pc, prompt)
        assert len(chain_before) == 1
        assert chain_before[0].recurrent_snapshot is snap_b0

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # admit via hit + suffix slice captures
        batcher.step()  # reclaim → extract → 2 new nodes attached

        assert batcher.prefix_hits == 1  # confirmed hit path
        chain_after = _walk_chain(pc, prompt)
        assert len(chain_after) == 3
        # Block 0: existing kept.
        assert chain_after[0].recurrent_snapshot is snap_b0
        # Blocks 1 and 2: NEW nodes with the captured markers.
        # The spy's marker counter started at 0; the first
        # suffix slice captured marker 0, the second captured
        # marker 1.
        assert chain_after[1].recurrent_snapshot is not None
        assert _read_marker(chain_after[1].recurrent_snapshot) == 0
        assert chain_after[2].recurrent_snapshot is not None
        assert _read_marker(chain_after[2].recurrent_snapshot) == 1


class TestSuffixSliceCapturesAtAbsoluteIndex:
    def test_suffix_capture_block_idx_offset_by_K(self) -> None:
        # K=2 cached, suffix of 2 * block_size tokens (2 full slice
        # forwards → 2 captures). Captures must land at absolute
        # indices K=2 and K+1=3, NOT 0 and 1.
        log = _SpyLog()
        adapter = _make_recurrent_hybrid_spy_adapter(
            log, script=(0, 0)
        )
        pc = _prefix_cache()
        prompt = list(range(1, 4 * BLOCK_SIZE + 1))  # 16 tokens
        snap_b0 = _make_marker_snapshot(400)
        snap_b1 = _make_marker_snapshot(401)
        pc.insert_detached(
            prompt[: 2 * BLOCK_SIZE],
            _detached_blocks(2),
            recurrent_snapshots=[snap_b0, snap_b1],
        )

        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
            _allow_recurrent_prefix_cache_for_c5_3_testing=True,
        )
        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        row = batcher._rows[0]
        # Counter ends at K*block_size + 2*block_size = 4*block_size.
        assert row.absolute_consumed_tokens == 4 * BLOCK_SIZE
        # Suffix captures at absolute indices 2 and 3.
        assert 2 in row.recurrent_snapshots_per_block
        assert 3 in row.recurrent_snapshots_per_block
        # Seeded ancestors at 0 and 1 (the K=2 cached blocks).
        assert row.recurrent_snapshots_per_block[0] is snap_b0
        assert row.recurrent_snapshots_per_block[1] is snap_b1
