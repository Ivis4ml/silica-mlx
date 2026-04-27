"""P-3-C5.3.4 — vertical slice + smoke convergence (synthetic).

Closes the C5.3.0-.3 chain on the synthetic-adapter side. Two
acceptance gates from ``plans/P3_C5_3_DESIGN.md`` §4.5 are pinned
here; the third (real-model end-to-end smoke) lives in
``tests/test_p3_hybrid_batched_smoke.py`` as a sibling to the
existing pre-cache hybrid smoke.

  1. **Backfill convergence** (§3.5.1): a snapshotless prefix
     pre-loaded into the radix tree routes the first request to
     miss via the Phase-B classifier; that request's reclaim
     extract path backfills the existing nodes' recurrent_snapshot
     via insert_detached's duplicate-prefix branch (C5.3.0); a
     second request with the same prompt then routes to hit. This
     is the self-healing claim made in §3.5.1 — legacy
     snapshotless nodes get filled in as soon as a row that
     re-prefills the same prefix is extracted.

  2. **Production-path-no-prefix-cache stays contiguous** (§4.5
     acceptance bullet 3): a hybrid adapter constructed with
     ``prefix_cache=None`` must NOT activate the slice-prefill
     regime. The predicate ``_slice_prefill_active()`` returns
     False, no snapshot captures fire, and the row's
     ``recurrent_snapshots_per_block`` stays empty. Pre-C5.4 this
     gate also defended against two test-only flags leaking into
     production; both flags were removed at C5.4 so the gate now
     just verifies the production predicate's ``prefix_cache``
     clause.

Both tests reuse the het.1 ``_ScriptedHybridAdapter`` /
``_SpyLog`` scaffolding so the contract surface is the same one
het.1/het.2 admit-side tests pin.
"""

from __future__ import annotations

import mlx.core as mx

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
)
from silica.models.recurrent import (
    RecurrentSnapshot,
    _RecurrentLayerEntry,
)
from silica.scheduler.batcher import ContinuousBatcher
from tests.test_batcher_extract_hybrid_heterogeneity import (
    BLOCK_SIZE,
    HEAD_DIM,
    N_KV,
    _ScriptedHybridAdapter,
    _SpyLog,
)


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _prefix_cache() -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )


def _per_attn_position_kv(
    markers: list[float],
) -> list[tuple[mx.array, mx.array]]:
    """One ``(K, V)`` tuple per attention position. Mirrors the
    helper in tests/test_batcher_admit_hybrid_heterogeneity.py;
    inlined here so this file doesn't take a cross-file dependency
    on the het.2 admit-side test module."""
    shape = (1, N_KV, BLOCK_SIZE, HEAD_DIM)
    return [
        (
            mx.full(shape, m, dtype=mx.float16),
            mx.full(shape, m + 0.5, dtype=mx.float16),
        )
        for m in markers
    ]


# --- backfill convergence (synthetic) ---


class TestBackfillConvergence:
    def test_snapshotless_prefix_route_to_miss_then_backfilled_to_hit(
        self,
    ) -> None:
        # Pattern: 1 GLOBAL + 1 HYBRID_DELTANET. 1 attention position
        # → ``_per_attn_position_kv`` produces 1 (K, V) per block.
        # The exact layer split doesn't matter for the convergence
        # gate; the relevant invariants are (a) Phase-B routes to
        # miss when the deepest USABLE node lacks a snapshot, (b)
        # the missed request's reclaim extract backfills, (c)
        # the next same-prompt request then routes to hit.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()

        # Pre-seed the tree with K=2 snapshotless blocks. prompt has
        # 9 tokens so usable = ((9-1)//4)*4 = 8 = 2 * BLOCK_SIZE; the
        # Phase-B classifier walks peek_with_node(prompt[:8]) and
        # finds the depth-8 node, whose recurrent_snapshot is None
        # → route-to-miss.
        prompt = list(range(1, 2 * BLOCK_SIZE + 2))  # 9 tokens
        detached_pre = [
            _per_attn_position_kv([100.0]),
            _per_attn_position_kv([200.0]),
        ]
        pc.insert_detached(
            prompt[: 2 * BLOCK_SIZE],
            detached_pre,
            recurrent_snapshots=[None, None],
        )

        # Sanity: tree carries the snapshotless prefix.
        _, deepest_pre = pc.peek_with_node(prompt[: 2 * BLOCK_SIZE])
        assert deepest_pre is not None
        assert deepest_pre.recurrent_snapshot is None, (
            "test setup error: pre-seeded deepest node must have "
            "recurrent_snapshot=None to drive the route-to-miss path"
        )

        # The synthetic adapter's slice helper does ceil(9/4) = 3
        # chunks for request 1's miss-path prefill (each chunk is
        # one forward → one element of script consumed). Request 2's
        # hit-path suffix is 1 chunk (1 token). Total: 4 forwards.
        adapter = _ScriptedHybridAdapter(
            pattern, log, script=(0, 0, 0, 0)
        )
        batcher = ContinuousBatcher(
            adapter,
            prefix_cache=pc,
        )

        # Flip _cohort_prepared so add_request lands in the waiting
        # queue and the next step routes through the Phase-B
        # classifier rather than the initial-cohort prefill path.
        assert batcher.step() == []

        # Request 1: routes to miss because deepest USABLE has None.
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()
        assert batcher.prefix_hits == 0, (
            f"request 1 must route-to-miss (deepest USABLE is "
            f"snapshotless); prefix_hits={batcher.prefix_hits}"
        )

        # Drive the row through reclaim. With max_tokens=1 the row
        # is DONE after the admit-step's sample; the next step's
        # reclaim phase fires the extract that backfills the tree.
        batcher.step()  # reclaim → extract → backfill

        # Tree is now backfilled. peek_with_node returns the same
        # node identity as before, but with a non-None snapshot.
        _, deepest_post = pc.peek_with_node(prompt[: 2 * BLOCK_SIZE])
        assert deepest_post is not None
        assert deepest_post.recurrent_snapshot is not None, (
            "extract on request 1's reclaim must backfill the "
            "deepest USABLE node's recurrent_snapshot via "
            "insert_detached's duplicate-prefix branch"
        )

        # Request 2: same prompt → now routes to hit.
        batcher.add_request(1, prompt, _params(max_tokens=1))
        batcher.step()
        assert batcher.prefix_hits == 1, (
            f"request 2 must hit after backfill convergence; "
            f"prefix_hits={batcher.prefix_hits}"
        )

    def test_backfill_only_overrides_none_not_existing_snapshot(
        self,
    ) -> None:
        # Companion check: insert_detached's duplicate-prefix branch
        # must NOT overwrite an existing non-None snapshot. The
        # backfill rule (C5.3.0 §3.5.1) is one-directional —
        # None gets filled, populated stays put. Without this,
        # repeated extracts would clobber earlier (potentially
        # different-trajectory) snapshots.
        pc = _prefix_cache()
        prompt = list(range(1, BLOCK_SIZE + 1))
        first_snap = RecurrentSnapshot(
            entries=(
                _RecurrentLayerEntry(
                    layer_idx=0,
                    conv_state=mx.array([[1.0]], dtype=mx.float32),
                    recurrent_state=None,
                ),
            ),
            nbytes=0,
        )
        second_snap = RecurrentSnapshot(
            entries=(
                _RecurrentLayerEntry(
                    layer_idx=0,
                    conv_state=mx.array([[2.0]], dtype=mx.float32),
                    recurrent_state=None,
                ),
            ),
            nbytes=0,
        )
        pc.insert_detached(
            prompt,
            [_per_attn_position_kv([10.0])],
            recurrent_snapshots=[first_snap],
        )
        pc.insert_detached(
            prompt,
            [_per_attn_position_kv([10.0])],
            recurrent_snapshots=[second_snap],
        )
        _, node = pc.peek_with_node(prompt)
        assert node is not None
        assert node.recurrent_snapshot is not None
        # Conv state retained from the FIRST insert.
        conv = node.recurrent_snapshot.entries[0].conv_state
        assert conv is not None
        marker = float(conv[0, 0])
        assert marker == 1.0, (
            f"existing non-None snapshot must be retained on "
            f"duplicate insert; got marker {marker}"
        )


# --- production no-flag stays contiguous ---


class TestProductionPathStaysContiguous:
    def test_no_prefix_cache_disables_slice_regime(self) -> None:
        # Hybrid adapter, prefix_cache=None. The slice predicate is
        # ``RecurrentStateAdapter + prefix_cache is not None`` (post-
        # C5.4); with prefix_cache=None the predicate is False even
        # though the adapter is a RecurrentStateAdapter.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        adapter = _ScriptedHybridAdapter(pattern, log, script=(0,))
        batcher = ContinuousBatcher(adapter, prefix_cache=None)
        assert batcher._slice_prefill_active() is False, (
            "production path (prefix_cache=None, no flag) must "
            "leave the slice-prefill regime inactive"
        )

    def test_production_prefill_does_not_capture_snapshots(self) -> None:
        # Drive a real prefill on the production path and confirm
        # (a) no snapshot calls were issued by the slice helper
        # (because the slice helper itself never runs), (b) the
        # row's recurrent_snapshots_per_block dict stays empty.
        # The synthetic spy's snapshot_recurrent_state increments a
        # counter; len(log.snapshot_calls) == 0 asserts the slice
        # helper never invoked it.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        adapter = _ScriptedHybridAdapter(pattern, log, script=(0,))
        batcher = ContinuousBatcher(adapter, prefix_cache=None)

        prompt = list(range(1, 2 * BLOCK_SIZE + 1))  # 8 tokens
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # contiguous prefill (1 forward over 8 tokens)

        # No snapshot calls — the slice helper's
        # ``snapshot_recurrent_state`` invocation site is the only
        # caller in the synthetic flow.
        assert log.snapshot_calls == [], (
            f"production path must not invoke snapshot capture; "
            f"got {len(log.snapshot_calls)} call(s)"
        )

        # Row's per-block snapshot dict stayed empty.
        assert batcher._rows
        row = batcher._rows[0]
        assert row.recurrent_snapshots_per_block == {}, (
            f"production path must leave row.recurrent_snapshots"
            f"_per_block empty; got {row.recurrent_snapshots_per_block}"
        )
        # ``absolute_consumed_tokens`` stays at the C5.3.1 default
        # (0) because the contiguous prefill path doesn't bump it.
        assert row.absolute_consumed_tokens == 0
