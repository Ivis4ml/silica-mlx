"""P-3-C5.5 — B>=1 slice-prefill cohort capture (α MVP).

Synthetic-adapter coverage of the batched slice-prefill helper
``_slice_prefill_with_capture_batched``. The pre-C5.5 batcher
clamped slice-prefill to ``len(rows) == 1``; B>1 cohorts under
slice regime fell back to contiguous prefill with zero captures.
C5.5 lifts that clamp under the α MVP scope (per
``plans/P3_C5_3_DESIGN.md`` §5.2 addendum):

- **Pad-aligned rows** (``pad_i ≡ 0 (mod block_size)``) get
  captures at every full block boundary during prefill, matching
  B=1 coverage.
- **Pad-misaligned rows** may receive zero prefill captures when
  the row's absolute counter never lands on a block boundary at
  any chunk-end. The decode-era capture path
  (``_decode_phase``) picks up subsequent boundaries past
  ``L_i``; the prefill-side partial-coverage limitation is the
  α MVP scope tradeoff.

Three acceptance tests pin the post-C5.5 behaviour:

1. **Aligned B=2 cohort**: same-length prompts produce captures
   for both rows at all full-block boundaries.
2. **Misaligned B=2 cohort**: only pad-aligned row(s) are
   captured during prefill; pad-misaligned rows have an empty
   ``recurrent_snapshots_per_block`` post-prefill.
3. **Misaligned + decode follow-up** (advisor sharpening 1):
   driving decode steps past the misaligned row's prompt end
   eventually crosses a block boundary in absolute coordinates;
   the decode-era capture path populates the dict at that
   boundary. Confirms the C5.5 predicate change
   (post-prefill ``absolute_consumed_tokens > 0`` for misaligned
   rows) does not regress decode-era capture.
"""

from __future__ import annotations

from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.scheduler.batcher import ContinuousBatcher
from tests.test_batcher_slice_prefill_capture import (
    BLOCK_SIZE,
    _make_slice_spy_adapter,
    _params,
    _read_marker,
    _SliceCaptureLog,
)


def _prefix_cache() -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )


def _make_batcher_b_n(adapter: object, max_batch_size: int) -> ContinuousBatcher:
    return ContinuousBatcher(
        adapter,  # type: ignore[arg-type]
        max_batch_size=max_batch_size,
        prefix_cache=_prefix_cache(),
    )


# --- aligned B=2 cohort: both rows captured ---


class TestAlignedBatchedCohortCaptures:
    def test_b2_same_length_captures_both_rows_all_blocks(self) -> None:
        # Both prompts have length 2 * BLOCK_SIZE (= 8). max_L = 8.
        # pad_a = pad_b = 0; both rows are aligned by construction.
        # After 2 chunked forwards the slice helper captures block 0
        # and block 1 for each row. With the spy's marker-stamping
        # snapshot, the captures fire in row-major order per chunk
        # (row 0 then row 1, then advance to next chunk).
        prompt_a = list(range(1, 2 * BLOCK_SIZE + 1))
        prompt_b = list(range(100, 100 + 2 * BLOCK_SIZE))
        log = _SliceCaptureLog()
        # 2 chunked forwards under slice regime — script provides
        # per-row targets at each forward (tuple of 2 ints).
        adapter = _make_slice_spy_adapter(
            log, script=((0, 0), (0, 0))
        )
        batcher = _make_batcher_b_n(adapter, max_batch_size=2)
        batcher.add_request(0, prompt_a, _params(max_tokens=1))
        batcher.add_request(1, prompt_b, _params(max_tokens=1))
        batcher.step()  # batched slice prefill

        rows = batcher._rows
        assert len(rows) == 2
        for row in rows:
            assert row.absolute_consumed_tokens == 2 * BLOCK_SIZE, (
                f"row {row.req_index} absolute_consumed_tokens="
                f"{row.absolute_consumed_tokens}; expected 2 * "
                f"BLOCK_SIZE = {2 * BLOCK_SIZE}"
            )
            assert set(row.recurrent_snapshots_per_block.keys()) == {0, 1}, (
                f"row {row.req_index} captures="
                f"{set(row.recurrent_snapshots_per_block.keys())}; "
                f"expected blocks {{0, 1}}"
            )
        # 2 chunks * 2 rows = 4 snapshot calls.
        assert len(log.snapshot_calls) == 4
        # Per-chunk row-major order: chunk 0 captures (row=0, m=0),
        # (row=1, m=1); chunk 1 captures (row=0, m=2), (row=1, m=3).
        # Verify markers landed at the expected (row, block) cells.
        assert _read_marker(rows[0].recurrent_snapshots_per_block[0]) == 0
        assert _read_marker(rows[1].recurrent_snapshots_per_block[0]) == 1
        assert _read_marker(rows[0].recurrent_snapshots_per_block[1]) == 2
        assert _read_marker(rows[1].recurrent_snapshots_per_block[1]) == 3


# --- misaligned B=2 cohort: longest captured, shorter not ---


class TestMisalignedBatchedCohortPartialCoverage:
    def test_b2_mixed_length_only_aligned_row_captured(self) -> None:
        # Long prompt L_a = 8 (pad_a = 0; aligned). Short prompt
        # L_b = 5 (pad_b = 3; pad_b mod BLOCK_SIZE != 0).
        # max_L = 8 → 2 chunks of BLOCK_SIZE = 4.
        # Row a absolute progression: 0 → 4 → 8. Both block
        # boundaries hit. Row a captured at blocks 0 and 1.
        # Row b absolute progression: 0 → max(0, 4-3) = 1 → 1 + 4 = 5.
        # Neither chunk-end is a multiple of BLOCK_SIZE = 4. No
        # captures during prefill for row b.
        prompt_a = list(range(1, 2 * BLOCK_SIZE + 1))  # length 8
        prompt_b = list(range(100, 100 + BLOCK_SIZE + 1))  # length 5
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(
            log, script=((0, 0), (0, 0))
        )
        batcher = _make_batcher_b_n(adapter, max_batch_size=2)
        batcher.add_request(0, prompt_a, _params(max_tokens=1))
        batcher.add_request(1, prompt_b, _params(max_tokens=1))
        batcher.step()

        row_a, row_b = batcher._rows
        # Row a: full coverage.
        assert row_a.absolute_consumed_tokens == 2 * BLOCK_SIZE
        assert set(row_a.recurrent_snapshots_per_block.keys()) == {0, 1}
        # Row b: zero prefill captures, counter = L_b = 5.
        assert row_b.absolute_consumed_tokens == 5
        assert row_b.recurrent_snapshots_per_block == {}
        # Snapshot calls: 2 chunks * 1 aligned row = 2.
        assert len(log.snapshot_calls) == 2
        # All snapshot calls were for row 0; row 1 was skipped by
        # the predicate (absolute % BLOCK_SIZE != 0).
        for call in log.snapshot_calls:
            assert call["row_idx"] == 0, (
                f"snapshot call captured row_idx={call['row_idx']}; "
                f"only the aligned row 0 should fire under α MVP"
            )


# --- decode-era follow-up on misaligned row ---


class TestMisalignedRowDecodeEraCapture:
    def test_decode_steps_past_l_i_capture_at_next_block_boundary(
        self,
    ) -> None:
        # Setup mirrors the misaligned test (L_a=8, L_b=5,
        # BLOCK_SIZE=4) but drives decode steps past the prompt
        # end. Row b's post-prefill absolute is 5; with 3 decode
        # steps absolute reaches 8 → block 1 captured by the
        # decode-era path. The advisor sharpening: confirms C5.5's
        # predicate change (B>1 rows have non-zero counter
        # post-prefill) does not regress the decode-era capture
        # gate at silica/scheduler/batcher.py:_decode_phase.
        prompt_a = list(range(1, 2 * BLOCK_SIZE + 1))  # 8
        prompt_b = list(range(100, 100 + BLOCK_SIZE + 1))  # 5
        log = _SliceCaptureLog()
        # 2 prefill chunks + 3 decode steps = 5 forwards. Each
        # forward with B=2 takes a per-row tuple.
        adapter = _make_slice_spy_adapter(
            log,
            script=(
                (0, 0),  # prefill chunk 0
                (0, 0),  # prefill chunk 1
                (0, 0),  # decode 1
                (0, 0),  # decode 2
                (0, 0),  # decode 3
            ),
        )
        batcher = _make_batcher_b_n(adapter, max_batch_size=2)
        # max_tokens=4 → after prefill's first sample, 3 more
        # decode steps fire before max_tokens is reached.
        batcher.add_request(0, prompt_a, _params(max_tokens=4))
        batcher.add_request(1, prompt_b, _params(max_tokens=4))
        batcher.step()  # batched slice prefill
        for _ in range(3):
            batcher.step()  # decode steps

        row_a, row_b = batcher._rows
        # Row b post-decode absolute = 5 + 3 = 8 → block 1
        # captured by decode-era capture loop (predicate
        # absolute % BLOCK_SIZE == 0 fires at step 3).
        assert row_b.absolute_consumed_tokens == 5 + 3
        assert 1 in row_b.recurrent_snapshots_per_block, (
            f"row b decode-era capture should populate block 1 at "
            f"absolute=8; got dict keys "
            f"{set(row_b.recurrent_snapshots_per_block.keys())}"
        )
        # Row a post-decode absolute = 8 + 3 = 11; no new block
        # boundary crossed (next would be at 12). Captures stay
        # at {0, 1} from prefill.
        assert row_a.absolute_consumed_tokens == 2 * BLOCK_SIZE + 3
        assert set(row_a.recurrent_snapshots_per_block.keys()) == {0, 1}
