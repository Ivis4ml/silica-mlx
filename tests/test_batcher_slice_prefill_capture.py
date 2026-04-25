"""P-3-C5.3.1 — slice-prefill in-flight capture tests.

Synthetic-adapter coverage for the slice-prefill regime gate and the
in-flight per-block snapshot capture path. No real model load — every
test runs against ``_ScriptedAdapter`` extended with the
``RecurrentStateAdapter`` mixin (test-only configuration where the
adapter's ``capabilities().has_recurrent_state`` returns ``False``,
bypassing the ctor guard, while ``isinstance(...,
RecurrentStateAdapter)`` returns ``True`` so the slice-regime
predicate fires).

Acceptance shape per ``docs/P3_C5_3_DESIGN.md`` §4.2:

- B=1 prefill of ``N * block_size`` tokens captures one snapshot at
  every full-block boundary; ``recurrent_snapshots_per_block.keys()``
  equals ``{0, 1, ..., N-1}``.
- A trailing chunk shorter than ``block_size`` does NOT produce a
  snapshot (mid-block boundary).
- B>1 cohort prefill stays contiguous; no snapshots are captured.
- Decode-step capture rolls the counter forward by ``+1`` per step
  and captures whenever ``absolute_consumed_tokens % block_size ==
  0``.
- The ``absolute_consumed_tokens > 0`` precondition prevents
  contiguous-admitted rows (B>1 miss cohort) from ever capturing
  decode-era snapshots — keeps cross-regime contamination out of the
  radix tree (finding C).
- Non-recurrent adapters and ``prefix_cache=None`` configurations
  preserve today's contiguous prefill behaviour bit-for-bit.

Hit-admission seeding (``recurrent_snapshots_per_block`` populated
from K ancestor nodes, ``absolute_consumed_tokens = K * block_size``)
is NOT exercised here — it lands at C5.3.3 alongside the
``lookup_with_node`` / restore wiring it serves (per design §4.4).
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

# Mirror of _ScriptedModel's per-step script type so per-row steps
# (e.g. (0, 0)) type-check alongside scalar broadcasts (e.g. 0). The
# runtime helper in test_batcher.py uses `int | Sequence[int]`; we
# keep the alias narrow to the one we feed.
_ScriptStep = int | Sequence[int]


# --- spy adapter scaffolding ---


@dataclass
class _SliceCaptureLog:
    """Records every ``snapshot_recurrent_state`` call from the
    in-flight capture path so tests can assert call counts, row_idx
    sequencing, and per-call markers without inspecting cache
    internals."""

    snapshot_calls: list[dict[str, Any]] = field(default_factory=list)
    next_marker: int = 0


def _make_slice_spy_adapter(
    log: _SliceCaptureLog,
    *,
    n_layers: int = 2,
    script: Sequence[_ScriptStep] = (),
) -> Any:
    """Synthetic adapter implementing ``RecurrentStateAdapter``.

    Reuses ``_ScriptedAdapter`` from ``test_batcher.py`` for the basic
    ModelAdapter surface (capabilities, kv_layout, build, etc.). The
    GLOBAL-only attention pattern means
    ``capabilities().has_recurrent_state == False``, so
    ``ContinuousBatcher.__init__`` accepts it alongside a
    ``RadixPrefixCache``. The ``RecurrentStateAdapter`` mixin makes
    ``isinstance(adapter, RecurrentStateAdapter)`` return ``True`` so
    the slice-regime predicate fires.
    """
    from tests.test_batcher import _ScriptedAdapter

    class _SliceCaptureSpyAdapter(_ScriptedAdapter):
        def snapshot_recurrent_state(
            self, cache_list: list[Any], row_idx: int
        ) -> RecurrentSnapshot:
            marker = log.next_marker
            log.next_marker += 1
            log.snapshot_calls.append(
                {
                    "row_idx": row_idx,
                    "marker": marker,
                    "cache_id": id(cache_list),
                }
            )
            # Stamp the marker into the conv_state's first scalar so
            # tests can read it back from
            # ``row.recurrent_snapshots_per_block[i]``.
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
            # Required by the Protocol; not exercised at C5.3.1.
            pass

    return _SliceCaptureSpyAdapter(n_layers=n_layers, script=script)


def _make_batcher(
    adapter: Any,
    *,
    prefix_cache: RadixPrefixCache | None,
    max_batch_size: int = 1,
) -> ContinuousBatcher:
    return ContinuousBatcher(
        adapter,
        max_batch_size=max_batch_size,
        prefix_cache=prefix_cache,
    )


def _prefix_cache(block_size: int = BLOCK_SIZE) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _read_marker(snapshot: RecurrentSnapshot) -> int:
    """Read the marker stamped into a captured snapshot."""
    conv = snapshot.entries[0].conv_state
    assert conv is not None
    return int(float(conv[0, 0]))


# --- gate predicate ---


class TestSlicePrefillActivePredicate:
    def test_recurrent_adapter_with_prefix_cache_active(self) -> None:
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log)
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        assert batcher._slice_prefill_active() is True

    def test_recurrent_adapter_without_prefix_cache_inactive(self) -> None:
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log)
        batcher = _make_batcher(adapter, prefix_cache=None)
        assert batcher._slice_prefill_active() is False

    def test_non_recurrent_adapter_with_prefix_cache_inactive(self) -> None:
        # _ScriptedAdapter alone (no RecurrentStateAdapter mixin) — the
        # isinstance() arm of the predicate returns False, so the
        # slice-regime path is dormant.
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1)
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        assert batcher._slice_prefill_active() is False

    def test_neither_inactive(self) -> None:
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1)
        batcher = _make_batcher(adapter, prefix_cache=None)
        assert batcher._slice_prefill_active() is False


# --- B=1 prefill capture ---


class TestB1PrefillCapture:
    def test_aligned_prefill_captures_one_per_block(self) -> None:
        # Prefill exactly N * block_size tokens. Slice helper makes
        # N forward calls; every call is a full block, so N snapshots.
        N = 3
        prompt = list(range(1, N * BLOCK_SIZE + 1))  # 12 tokens
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(
            log, script=tuple(range(N))  # one script step per slice
        )
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == N * BLOCK_SIZE
        assert set(row.recurrent_snapshots_per_block.keys()) == {0, 1, 2}
        # Per-snapshot markers monotonic in capture order.
        for i in range(N):
            assert _read_marker(
                row.recurrent_snapshots_per_block[i]
            ) == i
        # All captures targeted row_idx=0 (B=1).
        assert all(
            call["row_idx"] == 0 for call in log.snapshot_calls
        )
        assert len(log.snapshot_calls) == N

    def test_non_aligned_prefill_skips_trailing_partial_block(self) -> None:
        # 2 * block_size + 1 = 9 tokens → 3 slice forwards: 2 full
        # blocks + 1 trailing 1-token partial. Only the two full
        # forwards capture. Picking T_extra < block_size keeps the
        # trailing chunk genuinely partial (any T_extra >= block_size
        # would roll into another full block).
        T_extra = 1
        prompt = list(range(1, 2 * BLOCK_SIZE + T_extra + 1))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(
            log, script=tuple(range(3))
        )
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == 2 * BLOCK_SIZE + T_extra
        # Only blocks 0 and 1 captured; trailing 1-token chunk did not
        # cross a block boundary.
        assert set(row.recurrent_snapshots_per_block.keys()) == {0, 1}
        assert len(log.snapshot_calls) == 2

    def test_short_prefill_below_block_size_captures_nothing(self) -> None:
        # 3-token prompt, block_size=4 → single slice forward of 3
        # tokens; no full-block boundary crossed; no capture.
        prompt = [1, 2, 3]
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0,))
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == 3
        assert row.recurrent_snapshots_per_block == {}
        assert log.snapshot_calls == []


# --- B=1 clamp: B>1 cohorts skip slicing ---


class TestB1OnlyClamp:
    def test_b2_cohort_stays_contiguous_and_skips_capture(self) -> None:
        # Two requests admit pre-step → B=2 cohort. Slice gate
        # requires len(rows)==1; B=2 falls back to contiguous prefill,
        # NO snapshots.
        prompt_a = list(range(1, BLOCK_SIZE * 2 + 1))  # 8 tokens
        prompt_b = list(range(100, 100 + BLOCK_SIZE * 2))  # 8 tokens
        log = _SliceCaptureLog()
        # Per-row script step for the single contiguous prefill
        # forward: each row gets a target.
        adapter = _make_slice_spy_adapter(log, script=((0, 0),))
        batcher = _make_batcher(
            adapter, prefix_cache=_prefix_cache(), max_batch_size=2
        )
        batcher.add_request(0, prompt_a, _params())
        batcher.add_request(1, prompt_b, _params())
        batcher.step()

        for row in batcher._rows:
            assert row.absolute_consumed_tokens == 0
            assert row.recurrent_snapshots_per_block == {}
        assert log.snapshot_calls == []


# --- decode-step capture ---


class TestDecodeStepCapture:
    def test_decode_step_rollover_after_non_aligned_prefill(self) -> None:
        # Prefill lands at 2*block_size + 5 = 13 (counter ends at 13).
        # Each decode step adds 1; on the (block_size - 5) = -1th step
        # ... wait, 5 mod block_size = 1, so after block_size - 1 = 3
        # decode steps the counter reaches 16, hits boundary, captures
        # block 3. Adjust: T_extra = 1 → counter ends 9, 7 decode
        # steps to reach 16 → captures at block 3.
        # Easier: T_extra = 1, decode 7 steps to reach 16. Block 3
        # captured.
        T_extra = 1
        prompt = list(range(1, 2 * BLOCK_SIZE + T_extra + 1))
        n_decode = BLOCK_SIZE * 4 - len(prompt)  # decode to 16
        log = _SliceCaptureLog()
        # Prefill: 3 slice forwards (2 full blocks + 1 partial) → 3
        # script steps. Decode: n_decode steps → n_decode script
        # steps. max_tokens needs to be n_decode + 1 (the prefill's
        # initial sample counts).
        adapter = _make_slice_spy_adapter(
            log, script=tuple(range(3 + n_decode))
        )
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(
            0, prompt, _params(max_tokens=n_decode + 1)
        )
        batcher.step()  # prefill: 2 captures (blocks 0, 1).
        for _ in range(n_decode):
            batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == 4 * BLOCK_SIZE  # 16
        # Capture set: blocks 0 and 1 from prefill; block 3 from
        # decode-step rollover (counter went 9..16; hit 12, 16). So
        # blocks 2 and 3 captured during decode.
        assert set(row.recurrent_snapshots_per_block.keys()) == {
            0, 1, 2, 3
        }
        # First two captures came from prefill; the rest from decode.
        assert len(log.snapshot_calls) == 4
        # All captures targeted this row's row_idx (0; sole row).
        assert all(c["row_idx"] == 0 for c in log.snapshot_calls)

    def test_terminal_row_is_not_captured(self) -> None:
        # Prompt of 4 tokens + max_tokens=1 → row terminates after the
        # prefill's first sample. _decode_phase is never reached, so
        # no decode-era capture even if the counter would land on a
        # boundary.
        prompt = list(range(1, BLOCK_SIZE + 1))  # 4 tokens
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0,))
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        # max_tokens=1 → finishes after sampling the first token.
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        row = batcher._rows[0]
        # Prefill captured block 0 (4 tokens consumed).
        assert set(row.recurrent_snapshots_per_block.keys()) == {0}
        # Counter at 4 (prefill end); no decode incremented it.
        assert row.absolute_consumed_tokens == BLOCK_SIZE


# --- decode-step capture gating: > 0 clause ---


class TestDecodeCaptureGate:
    def test_zero_counter_row_skips_decode_capture(self) -> None:
        # B=2 cohort admits via contiguous prefill (no slicing per the
        # B=1-only clamp). Both rows have counter=0. After a decode
        # step the gate's > 0 check skips them — no decode-era
        # snapshots reach the radix tree.
        prompt_a = list(range(1, 5))  # 4 tokens
        prompt_b = list(range(100, 104))  # 4 tokens
        log = _SliceCaptureLog()
        # Prefill (B=2 contiguous): per-row targets.
        # Decode steps follow; with max_tokens=2 each row decodes once.
        adapter = _make_slice_spy_adapter(
            log,
            script=((0, 0), (0, 0)),  # prefill, decode-step 0
        )
        batcher = _make_batcher(
            adapter, prefix_cache=_prefix_cache(), max_batch_size=2
        )
        batcher.add_request(0, prompt_a, _params(max_tokens=2))
        batcher.add_request(1, prompt_b, _params(max_tokens=2))
        batcher.step()  # prefill: contiguous (B=2), no capture
        batcher.step()  # decode: gate's > 0 clause skips

        for row in batcher._rows:
            assert row.absolute_consumed_tokens == 0
            assert row.recurrent_snapshots_per_block == {}
        assert log.snapshot_calls == []


# --- non-recurrent adapter / prefix_cache=None preservation ---


class TestNonSliceRegimeUnchanged:
    def test_non_recurrent_adapter_with_prefix_cache_no_capture(
        self,
    ) -> None:
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1, script=(0,))
        # Non-recurrent adapter: no RecurrentStateAdapter mixin →
        # gate returns False. Prompt that WOULD cross a block
        # boundary if sliced is irrelevant — slice path never runs.
        prompt = list(range(1, BLOCK_SIZE * 2 + 1))
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        batcher.step()

        row = batcher._rows[0]
        assert row.absolute_consumed_tokens == 0
        assert row.recurrent_snapshots_per_block == {}

    def test_recurrent_adapter_without_prefix_cache_no_capture(
        self,
    ) -> None:
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0,))
        prompt = list(range(1, BLOCK_SIZE * 2 + 1))
        batcher = _make_batcher(adapter, prefix_cache=None)
        batcher.add_request(0, prompt, _params())
        batcher.step()

        row = batcher._rows[0]
        # Counter stays 0 — slice helper never fired.
        assert row.absolute_consumed_tokens == 0
        assert row.recurrent_snapshots_per_block == {}
        assert log.snapshot_calls == []


# --- forward call accounting ---


class TestForwardCallShape:
    def test_slice_helper_makes_n_forwards_for_n_blocks(self) -> None:
        # 3 full blocks → 3 slice forwards. Verifies the slice loop
        # actually makes per-block forwards rather than bunching the
        # work into one contiguous call.
        prompt = list(range(1, 3 * BLOCK_SIZE + 1))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0, 0, 0))
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        # Snapshot forward_calls before stepping to isolate this
        # phase's contribution.
        before = adapter._model.forward_calls
        batcher.step()
        after = adapter._model.forward_calls
        assert after - before == 3

    def test_partial_chunk_is_part_of_forward_count(self) -> None:
        # 2 * block_size + 1 → 3 forwards (2 full + 1 partial of 1
        # token). 2 captures (the two full-block ones).
        prompt = list(range(1, 2 * BLOCK_SIZE + 2))
        log = _SliceCaptureLog()
        adapter = _make_slice_spy_adapter(log, script=(0, 0, 0))
        batcher = _make_batcher(adapter, prefix_cache=_prefix_cache())
        batcher.add_request(0, prompt, _params())
        before = adapter._model.forward_calls
        batcher.step()
        after = adapter._model.forward_calls
        assert after - before == 3
        assert len(log.snapshot_calls) == 2
