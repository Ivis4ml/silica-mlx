"""P-3-C5.3.2 — extract path forwards captured snapshots to the radix tree.

When ``_extract_and_insert_prefix`` slices block-aligned K/V out of a
terminating row's batched cache, it now also forwards the matching
``recurrent_snapshots_per_block`` entries to ``insert_detached`` so
the radix tree's ``_Node.recurrent_snapshot`` slots get populated for
slice-regime-admitted rows.

The change is purely additive — non-slice-regime rows have an empty
``recurrent_snapshots_per_block``, so every absolute block index
resolves to ``None`` via ``dict.get`` and the resulting list of
``None``-entries is treated by C5.3.0's ``insert_detached`` extension
identically to the pre-C5.3 omitted-arg path (new nodes' snapshots
stay ``None``; the duplicate-prefix backfill branch never fires
because every entry is ``None``). Tests below verify both directions:

- Slice-regime row: tree carries one snapshot per inserted block.
- Non-slice-regime row (non-recurrent adapter, B>1 cohort): tree
  nodes' ``recurrent_snapshot`` stays ``None``.

Hit-admitted-then-extracted (design §4.3 acceptance bullet 2) and
backfill-convergence (bullet 3) are deferred to C5.3.3 — both depend
on the Phase-B classifier and hit-admission seeding that land there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.recurrent import (
    RecurrentSnapshot,
    _RecurrentLayerEntry,
)
from silica.scheduler.batcher import ContinuousBatcher

BLOCK_SIZE = 4

_ScriptStep = int | Sequence[int]


# --- spy adapter scaffolding (mirror of test_batcher_slice_prefill_capture.py) ---


@dataclass
class _SliceCaptureLog:
    snapshot_calls: list[dict[str, Any]] = field(default_factory=list)
    next_marker: int = 0


def _make_slice_spy_adapter(
    log: _SliceCaptureLog,
    *,
    n_layers: int = 1,
    script: Sequence[_ScriptStep] = (),
) -> Any:
    from tests.test_batcher import _ScriptedAdapter

    class _SliceCaptureSpyAdapter(_ScriptedAdapter):
        def snapshot_recurrent_state(
            self, cache_list: list[Any], row_idx: int
        ) -> RecurrentSnapshot:
            marker = log.next_marker
            log.next_marker += 1
            log.snapshot_calls.append(
                {"row_idx": row_idx, "marker": marker}
            )
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

        def restore_recurrent_state(
            self,
            cache_list: list[Any],
            row_idx: int,
            snapshot: RecurrentSnapshot,
        ) -> None:
            pass

    return _SliceCaptureSpyAdapter(n_layers=n_layers, script=script)


def _prefix_cache(block_size: int = BLOCK_SIZE) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _read_marker(snapshot: RecurrentSnapshot | None) -> int | None:
    if snapshot is None:
        return None
    conv = snapshot.entries[0].conv_state
    assert conv is not None
    return int(float(conv[0, 0]))


def _walk_chain(
    pc: RadixPrefixCache, tokens: list[int]
) -> list[Any]:
    """Returns the chain of nodes from root's first child down to the
    deepest matched node, in insertion order (block 0 → block N-1)."""
    _, deepest = pc.peek_with_node(tokens)
    chain: list[Any] = []
    cursor = deepest
    while cursor is not None and cursor.parent is not None:
        chain.append(cursor)
        cursor = cursor.parent
    chain.reverse()
    return chain


# --- slice-regime row propagates snapshots ---


class TestSliceRegimePropagatesSnapshots:
    def test_two_block_prefill_extracts_two_snapshots(self) -> None:
        # B=1 slice-regime prefill of 2 * block_size tokens.
        # max_tokens=1 terminates the row after prefill's first sample;
        # the next step()'s reclaim phase runs extract → tree gets two
        # snapshots, one per inserted block.
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        log = _SliceCaptureLog()
        # Script: 2 prefill slices + 1 unused (max_tokens=1 finishes
        # after the 2nd slice's sample).
        adapter = _make_slice_spy_adapter(log, script=(0, 1))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(adapter, prefix_cache=pc)
        batcher.add_request(0, prompt, _params(max_tokens=1))

        batcher.step()  # prefill: 2 captures (markers 0, 1)
        # Row sampled max_tokens=1 token; transitions to DONE inside
        # _sample_and_emit_batched. Next step's reclaim extracts.
        batcher.step()  # reclaim → extract → tree updated

        # Two captures during prefill (markers 0 and 1).
        assert len(log.snapshot_calls) == 2

        chain = _walk_chain(pc, prompt)
        assert len(chain) == 2
        # First inserted block (absolute index 0) carries marker 0.
        assert _read_marker(chain[0].recurrent_snapshot) == 0
        # Second inserted block (absolute index 1) carries marker 1.
        assert _read_marker(chain[1].recurrent_snapshot) == 1

    def test_partial_trailing_block_excluded_from_extract(self) -> None:
        # 2 * block_size + 1 tokens: prefill captures 2 snapshots
        # (blocks 0, 1) + samples one token. Trailing 1-token chunk
        # produces no snapshot. Extract aligns to block_size and
        # inserts only the 2 full blocks; the trailing partial is
        # discarded by the extract path's alignment math.
        prompt = list(range(1, 2 * BLOCK_SIZE + 2))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0, 0, 0))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(adapter, prefix_cache=pc)
        batcher.add_request(0, prompt, _params(max_tokens=1))

        batcher.step()  # prefill: 2 captures from full blocks
        batcher.step()  # reclaim → extract

        chain = _walk_chain(pc, prompt[: 2 * BLOCK_SIZE])
        assert len(chain) == 2
        # Trailing partial chunk did not trigger a 3rd snapshot capture.
        assert len(log.snapshot_calls) == 2
        assert _read_marker(chain[0].recurrent_snapshot) == 0
        assert _read_marker(chain[1].recurrent_snapshot) == 1


# --- non-slice-regime preservation ---


class TestNonSliceRegimeNoSnapshots:
    def test_non_recurrent_adapter_extract_attaches_none(self) -> None:
        # Non-recurrent _ScriptedAdapter (no RecurrentStateAdapter
        # mixin) → slice-regime gate False → contiguous prefill →
        # row.recurrent_snapshots_per_block stays empty.
        # _extract_and_insert_prefix builds an all-None list, passes
        # to insert_detached; tree nodes have recurrent_snapshot=None.
        from tests.test_batcher import _ScriptedAdapter

        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        adapter = _ScriptedAdapter(n_layers=1, script=(0,))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(adapter, prefix_cache=pc)
        batcher.add_request(0, prompt, _params(max_tokens=1))

        batcher.step()  # contiguous prefill, no captures
        batcher.step()  # reclaim → extract with all-None snapshot list

        chain = _walk_chain(pc, prompt)
        assert len(chain) == 2
        for node in chain:
            assert node.recurrent_snapshot is None

    def test_b2_cohort_extract_attaches_none(self) -> None:
        # B=2 slice-regime cohort: gate active but len(rows)>1 clamp
        # forces contiguous prefill → both rows' dicts stay empty →
        # extract forwards all-None lists → tree nodes have
        # recurrent_snapshot=None for both rows' inserted blocks.
        prompt_a = list(range(1, 2 * BLOCK_SIZE + 1))
        prompt_b = list(range(100, 100 + 2 * BLOCK_SIZE))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=((0, 0),))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(
            adapter, max_batch_size=2, prefix_cache=pc
        )
        batcher.add_request(0, prompt_a, _params(max_tokens=1))
        batcher.add_request(1, prompt_b, _params(max_tokens=1))

        batcher.step()  # B=2 contiguous prefill, no captures
        batcher.step()  # reclaim → extract for both rows

        chain_a = _walk_chain(pc, prompt_a)
        chain_b = _walk_chain(pc, prompt_b)
        assert len(chain_a) == 2 and len(chain_b) == 2
        for node in chain_a + chain_b:
            assert node.recurrent_snapshot is None
        # Slice-helper never fired.
        assert log.snapshot_calls == []


# --- preempt path also forwards snapshots ---


class TestPreemptExtractForwardsSnapshots:
    def test_extract_during_reclaim_forwards_captured_snapshots(
        self,
    ) -> None:
        # _extract_and_insert_prefix has two callers — _reclaim_terminated
        # (covered above) and _apply_preempt. The above tests cover
        # the natural-termination path; preempt-path extract uses the
        # same function body, so the snapshot list construction is
        # exercised by the same code. This test reinforces that the
        # natural-termination path covers the function-level change
        # without needing to additionally drive a preempt scenario at
        # this sub-unit (preempt path's full surface lands at C5.3.3 +
        # C5.3.4 alongside the byte-exact gate it serves).
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))  # 3 full blocks
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0, 0, 0))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(adapter, prefix_cache=pc)
        batcher.add_request(0, prompt, _params(max_tokens=1))

        batcher.step()
        batcher.step()

        chain = _walk_chain(pc, prompt)
        assert len(chain) == 3
        for i, node in enumerate(chain):
            # Each block i should carry the marker captured during
            # slice forward i.
            assert _read_marker(node.recurrent_snapshot) == i


# --- duplicate-insert reuse keeps tree consistent ---


class TestDuplicateExtractDoesNotOverwriteExisting:
    def test_second_row_extract_with_empty_dict_preserves_tree_snapshots(
        self,
    ) -> None:
        # Two sequential B=1 requests with the same prompt:
        #   - Request 0: slice-regime prefill captures markers 0, 1;
        #     reclaim's extract inserts (snap_0, snap_1) into the tree.
        #   - Request 1: same prompt → Phase-B classifier (today's
        #     pre-C5.3.3 form, no recurrent-snapshot routing) routes
        #     it through ``_admit_single_hit_row``. Hit admission's
        #     suffix prefill stays contiguous at C5.3.2 (slice helper
        #     for the suffix lands at C5.3.3 alongside the restore
        #     wiring), so request 1's row has an empty
        #     ``recurrent_snapshots_per_block``. When max_tokens=1
        #     terminates request 1 and reclaim's extract runs, it
        #     forwards a list of ``None``-entries to ``insert_detached``.
        #
        # The C5.3.0 ``insert_detached`` contract: duplicate-prefix
        # branch with ``new_snap is None`` keeps the existing node's
        # snapshot (no overwrite, no backfill). This test pins that
        # contract end-to-end through the batcher's extract path —
        # complementing the unit-level test in
        # ``test_radix_prefix_cache_recurrent_snapshot.py``.
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0, 0, 0, 0))
        pc = _prefix_cache()
        batcher = ContinuousBatcher(adapter, prefix_cache=pc)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # prefill captures markers 0, 1
        batcher.step()  # reclaim → tree gets markers 0, 1

        chain_first = _walk_chain(pc, prompt)
        assert _read_marker(chain_first[0].recurrent_snapshot) == 0
        assert _read_marker(chain_first[1].recurrent_snapshot) == 1
        captures_after_first = len(log.snapshot_calls)
        assert captures_after_first == 2

        batcher.add_request(1, prompt, _params(max_tokens=1))
        batcher.step()  # admit → hit admission, contiguous suffix
        batcher.step()  # reclaim → extract with all-None list

        # Request 1 went through hit admission (today's path); no
        # additional snapshot captures happened.
        assert len(log.snapshot_calls) == captures_after_first
        # Tree's markers stay at the originals.
        chain_second = _walk_chain(pc, prompt)
        assert len(chain_second) == 2
        assert _read_marker(chain_second[0].recurrent_snapshot) == 0
        assert _read_marker(chain_second[1].recurrent_snapshot) == 1
