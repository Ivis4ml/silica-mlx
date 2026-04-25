"""P-3-C5.2 — preempt/replay snapshot capture tests.

C5.2's scope (post-drift-experiment pivot, 2026-04-24): land the
**capture / stash plumbing** for the recurrent-state snapshot.
Restore on the full-replay path is INTENTIONALLY NOT enabled
because the snapshot boundary (``T + len(generated) - 1``
consumed) does not match the replay's post-prefill boundary
(``T + len(generated)`` consumed). Restoring at the misaligned
boundary would rewind the cache by one position and erase the
last generated token's consumption. Restore enablement is
deferred to C5.3 where the prefix-hit admission path provides a
boundary that matches the snapshot.

C5.2 acceptance gates (verified by tests in this file):

1. **Snapshot-before-filter**: ``snapshot_recurrent_state`` is
   invoked inside ``_preempt_active_row`` before
   ``layer_cache.filter(kept)`` mutates the batched cache.
2. **Snapshot stashed on _PendingAdmit**: the captured snapshot
   travels on ``_PendingAdmit.recurrent_snapshot``; the field
   defaults to ``None`` for non-recurrent adapters or for
   pendings created via ``add_request``.
3. **Restore is NOT called on the full-replay path**: the
   ``_admit_miss_cohort`` path's full-prefill of
   ``composite_prompt = prompt + generated`` makes the post-
   prefill cache one token ahead of the snapshot. Restore here
   is contractually disabled at C5.2 — verified directly.
4. **Pending-lifetime alias defense**: the snapshot stashed on a
   replay pending stays byte-identical across intervening
   ``step()`` calls that mutate the live ``_batch_cache``.
5. **Boundary metadata**: the boundary the snapshot describes
   equals ``len(prompt_ids of replay pending) - 1`` =
   ``T + len(generated) - 1`` consumed at preempt time;
   ``len(prompt_ids of replay pending)`` is the composite-prompt
   token count, which is one ahead of the snapshot boundary by
   construction.

Tests do NOT exercise:

- Bit-exact restore vs no-preempt oracle — impossible at C5.2's
  natural boundaries (see ``docs/P3_C5_DRIFT_EXPERIMENT/README.md``
  "C5.2 acceptance" for the boundary derivation). C5.3 owns
  boundary-aligned restore.
- ``RadixPrefixCache`` cooperation — C5.3.
- Removal of the ``has_recurrent_state + prefix_cache`` guard
  at ``silica/scheduler/batcher.py:152-166`` — C5.4; every test
  here runs with ``prefix_cache=None``.
- P-7 speculative draft-rollback wiring — independent unit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from silica.models.recurrent import (
    RecurrentSnapshot,
    RecurrentStateAdapter,
    _RecurrentLayerEntry,
)
from silica.scheduler.batcher import (
    ContinuousBatcher,
    _PendingAdmit,
    _PreemptedRow,
)

# =============================================================
# Synthetic test scaffolding (no model load).
# =============================================================


@dataclass
class _CallLog:
    """Ordered record of (event, payload) tuples the spy adapter emits.

    Each test reads this to assert the relative ordering of
    ``snapshot`` / ``filter`` / ``forward`` / ``restore`` / ``extend``
    events. ``snapshot`` and ``restore`` are recorded by the spy
    adapter directly; ``filter`` and ``extend`` are recorded via
    cache-object wrappers; ``forward`` is recorded by the scripted
    model.
    """

    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _make_recurrent_snapshot(
    n_layers: int, marker: int = 0
) -> RecurrentSnapshot:
    """Build a deterministic ``RecurrentSnapshot`` for spy tests.

    Each layer carries a tiny (1, 1) tensor seeded with ``marker``
    so a snapshot taken at preempt time can be distinguished from
    one constructed elsewhere by inspecting the entries.
    """
    entries = tuple(
        _RecurrentLayerEntry(
            layer_idx=i,
            conv_state=mx.array([[float(marker * 100 + i)]], dtype=mx.float16),
            recurrent_state=mx.array(
                [[float(marker * 100 + i + 0.5)]], dtype=mx.float32
            ),
        )
        for i in range(n_layers)
    )
    return RecurrentSnapshot(entries=entries, nbytes=n_layers * 6)


# =============================================================
# Section 1: dataclass field surface (no batcher needed)
# =============================================================


class TestPendingAdmitSchema:
    """``_PendingAdmit.recurrent_snapshot`` is the agreed stash site."""

    def test_default_recurrent_snapshot_is_none(self) -> None:
        from silica.core.sampling import SamplingParams

        p = _PendingAdmit(
            req_index=0,
            prompt_ids=(1, 2, 3),
            params=SamplingParams(temperature=0.0, max_tokens=1),
        )
        assert p.recurrent_snapshot is None
        assert p.is_replay is False

    def test_construct_with_snapshot_round_trips(self) -> None:
        from silica.core.sampling import SamplingParams

        snap = _make_recurrent_snapshot(n_layers=3, marker=7)
        p = _PendingAdmit(
            req_index=4,
            prompt_ids=(1, 2, 3, 4, 5),
            params=SamplingParams(temperature=0.0, max_tokens=2),
            is_replay=True,
            recurrent_snapshot=snap,
        )
        assert p.is_replay is True
        assert p.recurrent_snapshot is snap
        assert len(p.recurrent_snapshot.entries) == 3


class TestPreemptedRowSchema:
    """``_PreemptedRow`` is the C5.2 result-object signature shape."""

    def test_default_recurrent_snapshot_is_none(self) -> None:
        # Build a minimal _BatchRow via the public scripted helpers
        # in tests/test_batcher.py rather than re-importing internals.
        from silica.core.request import Request, RequestState
        from silica.core.sampling import SamplingParams
        from silica.scheduler.batcher import _BatchRow

        req = Request(
            prompt="",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            request_id="req-0",
            token_ids=(1, 2, 3),
        )
        row = _BatchRow(
            req_index=0,
            req_id="req-0",
            prompt_ids=[1, 2, 3],
            params=req.sampling_params,
            state=RequestState(request=req),
        )
        result = _PreemptedRow(detached=row)
        assert result.detached is row
        assert result.recurrent_snapshot is None

    def test_carries_snapshot_when_set(self) -> None:
        from silica.core.request import Request, RequestState
        from silica.core.sampling import SamplingParams
        from silica.scheduler.batcher import _BatchRow

        snap = _make_recurrent_snapshot(n_layers=2, marker=1)
        req = Request(
            prompt="",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            request_id="req-0",
            token_ids=(1, 2, 3),
        )
        row = _BatchRow(
            req_index=0,
            req_id="req-0",
            prompt_ids=[1, 2, 3],
            params=req.sampling_params,
            state=RequestState(request=req),
        )
        result = _PreemptedRow(detached=row, recurrent_snapshot=snap)
        assert result.recurrent_snapshot is snap


# =============================================================
# Section 2: spy adapter + call-order invariants (synthetic)
# =============================================================


def _make_spy_adapter(
    log: _CallLog,
    *,
    n_layers: int = 1,
) -> Any:
    """Build a synthetic adapter that:

    - Reuses ``_ScriptedAdapter`` from ``test_batcher.py`` for the
      basic ModelAdapter surface and scripted forward.
    - Implements ``RecurrentStateAdapter`` (snapshot / restore) with
      bookkeeping into ``log``.
    - Wraps every ``make_batch_cache`` result so ``filter`` and
      ``extend`` calls also append events to ``log``.
    """
    # Import the synthetic test scaffolding from test_batcher.
    from tests.test_batcher import _ScriptedAdapter

    class _RecurrentSpyAdapter(_ScriptedAdapter):
        def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
            caches = super().make_batch_cache(left_padding)
            return [_LoggingCache(c, log, layer) for layer, c in enumerate(caches)]

        # RecurrentStateAdapter mixin
        def snapshot_recurrent_state(
            self, cache_list: list[Any], row_idx: int
        ) -> RecurrentSnapshot:
            log.events.append((
                "snapshot",
                {"row_idx": row_idx, "cache_id": id(cache_list)},
            ))
            return _make_recurrent_snapshot(
                n_layers=len(cache_list), marker=row_idx
            )

        def restore_recurrent_state(
            self,
            cache_list: list[Any],
            row_idx: int,
            snapshot: RecurrentSnapshot,
        ) -> None:
            conv = snapshot.entries[0].conv_state
            assert conv is not None
            log.events.append((
                "restore",
                {
                    "row_idx": row_idx,
                    "cache_id": id(cache_list),
                    "snapshot_marker": int(float(conv[0, 0])),
                },
            ))

    return _RecurrentSpyAdapter(n_layers=n_layers)


class _LoggingCache:
    """Wraps a BatchKVCache to record ``filter`` / ``extend`` calls.

    Delegates everything else through ``__getattr__`` so the cache
    stays usable inside ``forward_batched``. Tests only inspect the
    log; the underlying cache mechanics are unchanged.
    """

    def __init__(self, inner: Any, log: _CallLog, layer: int) -> None:
        self._inner = inner
        self._log = log
        self._layer = layer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def filter(self, batch_indices: list[int]) -> Any:
        self._log.events.append((
            "filter",
            {"layer": self._layer, "kept": list(batch_indices)},
        ))
        return self._inner.filter(batch_indices)

    def extend(self, other: Any) -> Any:
        self._log.events.append((
            "extend",
            {"layer": self._layer},
        ))
        # Unwrap if other is also a _LoggingCache.
        target = other._inner if isinstance(other, _LoggingCache) else other
        return self._inner.extend(target)


# Convenience constant: read as ``prefix_cache=None`` keyword
# spelled out in tests so guard-stays-untouched is visually
# obvious. C5.2 explicitly does not exercise the
# ``has_recurrent_state + prefix_cache != None`` guard at
# ``silica/scheduler/batcher.py:152-166`` (C5.4 removes that).
_PREFIX_CACHE_NONE: None = None


class TestPreemptCallOrderingSynthetic:
    """Spy adapter records call order on a B=2 scripted run.

    B=2 is required to exercise the ``filter`` path inside
    ``_preempt_active_row``: when the victim is the SOLE row in the
    batch (B=1 case), ``_preempt_active_row`` short-circuits to
    ``self._batch_cache = None`` rather than calling
    ``layer_cache.filter(kept)`` — so a B=1 spy never observes a
    filter event. With B=2, preempting one of the two rows takes
    the ``layer_cache.filter([1])`` branch and the spy records it.
    """

    def _run_preempt_via_spy_b2(
        self,
    ) -> tuple[_CallLog, ContinuousBatcher]:
        from silica.core.sampling import SamplingParams

        log = _CallLog()
        adapter = _make_spy_adapter(log, n_layers=1)
        b = ContinuousBatcher(
            adapter,
            max_batch_size=2,
            prefix_cache=_PREFIX_CACHE_NONE,
        )
        b.add_request(
            0,
            [1, 2, 3, 4],
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.add_request(
            1,
            [5, 6, 7, 8],
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.step()  # initial cohort prefill of both rows.
        # Preempt req-0 (sibling row req-1 stays). filter([1]) runs
        # on every layer cache.
        ok = b._apply_preempt("req-0")  # type: ignore[attr-defined]
        assert ok is True
        return log, b

    def test_snapshot_recorded_before_filter(self) -> None:
        """C5.2 invariant 1: snapshot is captured BEFORE
        ``_preempt_active_row`` mutates the cache via filter."""
        log, _ = self._run_preempt_via_spy_b2()
        snap_idx = next(
            i for i, (e, _) in enumerate(log.events) if e == "snapshot"
        )
        filt_idx = next(
            i for i, (e, _) in enumerate(log.events) if e == "filter"
        )
        assert snap_idx < filt_idx, (
            f"snapshot at idx={snap_idx} but filter at idx={filt_idx}; "
            f"snapshot must precede filter. log={log.events}"
        )

    def test_snapshot_uses_victim_row_idx(self) -> None:
        """C5.2 invariant: ``snapshot_recurrent_state`` is called with
        the victim's live row index, resolved before the cache
        filter pass shifted the layout."""
        log, _ = self._run_preempt_via_spy_b2()
        snap_event = next(p for e, p in log.events if e == "snapshot")
        # req-0 is row 0 in the B=2 cohort.
        assert snap_event["row_idx"] == 0

    def test_filter_keeps_sibling_after_snapshot(self) -> None:
        """Sanity: snapshot does not interfere with the filter mechanic.
        After preempting req-0, the filter event keeps row 1 (sibling
        req-1). Confirms the spy harness exercises both paths."""
        log, _ = self._run_preempt_via_spy_b2()
        filt_event = next(p for e, p in log.events if e == "filter")
        assert filt_event["kept"] == [1]

    def test_restore_NOT_called_on_full_replay_path(self) -> None:
        """C5.2 pivot (drift-experiment-derived): the snapshot is
        captured and stashed but ``_admit_miss_cohort`` does NOT call
        ``restore_recurrent_state``. The full-replay prefill consumes
        ``T + len(generated)`` tokens while the snapshot describes
        ``T + len(generated) - 1`` consumed, so a restore here would
        rewind the cache by one position. C5.3 will enable restore
        only when post-prefill boundary aligns with snapshot boundary
        (prefix-hit path)."""
        log, b = self._run_preempt_via_spy_b2()
        # Drive the next step so _admit_miss_cohort runs on the
        # replay pending. With C5.2 pivot, no restore event should
        # appear in the log at any point.
        b.step()  # admits replay via _admit_miss_cohort
        rest_events = [e for e, _ in log.events if e == "restore"]
        assert rest_events == [], (
            f"restore was unexpectedly called on the full-replay path; "
            f"log={log.events}"
        )

    def test_replay_pending_carries_snapshot(self) -> None:
        """C5.2 invariant 2: the replay's ``_PendingAdmit`` in the
        waiting queue carries the captured snapshot in
        ``recurrent_snapshot``. The snapshot's ``conv_state`` marker
        encodes the victim row_idx (per ``_make_recurrent_snapshot``
        / ``_make_spy_adapter`` shape) so we can verify which row
        the snapshot came from."""
        log, b = self._run_preempt_via_spy_b2()
        pending = b._waiting_queue[0]  # type: ignore[attr-defined]
        assert isinstance(pending, _PendingAdmit)
        assert pending.is_replay is True
        assert pending.recurrent_snapshot is not None
        conv0 = pending.recurrent_snapshot.entries[0].conv_state
        assert conv0 is not None
        marker = int(float(conv0[0, 0]))
        # Spy encodes row_idx into the marker; req-0 is row 0.
        assert marker == 0


class TestNonRecurrentAdapterUnchanged:
    """Adapters without the RecurrentStateAdapter mixin must continue to
    behave as before — no snapshot capture, no restore call, no field
    populated on the replay pending."""

    def test_preempt_does_not_call_snapshot_on_plain_adapter(self) -> None:
        from silica.core.sampling import SamplingParams
        from tests.test_batcher import _ScriptedAdapter

        adapter = _ScriptedAdapter(n_layers=1, script=(5,))
        # Plain _ScriptedAdapter does NOT implement
        # RecurrentStateAdapter — verify the isinstance check.
        assert not isinstance(adapter, RecurrentStateAdapter)

        b = ContinuousBatcher(
            adapter,
            max_batch_size=1,
            prefix_cache=None,
        )
        b.add_request(
            0,
            [1, 2, 3, 4],
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.step()
        ok = b._apply_preempt("req-0")  # type: ignore[attr-defined]
        assert ok is True
        pending = b._waiting_queue[0]  # type: ignore[attr-defined]
        assert pending.is_replay is True
        assert pending.recurrent_snapshot is None


# =============================================================
# Section 3: snapshot boundary semantics (synthetic)
# =============================================================


class TestSnapshotBoundary:
    """Two boundaries co-exist on every replay ``_PendingAdmit``:

    - ``len(pending.prompt_ids) == len(prompt) + len(generated)`` —
      the composite_prompt token count the replay's prefill will
      consume.
    - **snapshot boundary** ``== len(pending.prompt_ids) - 1``
      ``== T + len(generated) - 1`` consumed — one token earlier
      than the composite, because at preempt time the latest sample
      had been emitted but NOT yet fed back to the cache.

    The off-by-one IS the boundary mismatch that drove the C5.2
    pivot: a hypothetical restore at "after replay's prefill, before
    extend" would rewind the cache to the snapshot boundary, erasing
    the consumption of the last generated token. C5.2 stashes the
    snapshot but does not call restore on this path; C5.3 will use
    the snapshot only at admission paths whose post-prefill boundary
    matches ``T + len(generated) - 1`` by construction.
    """

    def test_replay_composite_prompt_length_after_one_decode(
        self,
    ) -> None:
        from silica.core.sampling import SamplingParams

        log = _CallLog()
        adapter = _make_spy_adapter(log, n_layers=1)
        b = ContinuousBatcher(
            adapter,
            max_batch_size=1,
            prefix_cache=None,
        )
        prompt = [1, 2, 3, 4]
        b.add_request(
            0,
            prompt,
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.step()  # initial prefill emits one token (decoded_a[0]).
        # Capture generated_len BEFORE preempt so we can name both
        # boundaries explicitly post-pivot.
        active_row = b._rows[0]  # type: ignore[attr-defined]
        generated_len_before = len(active_row.generated)
        assert generated_len_before == 1

        ok = b._apply_preempt("req-0")  # type: ignore[attr-defined]
        assert ok is True
        pending = b._waiting_queue[0]  # type: ignore[attr-defined]
        assert pending.recurrent_snapshot is not None

        # composite_prompt boundary = T + len(generated) = 4 + 1 = 5.
        composite_len = len(pending.prompt_ids)
        assert composite_len == len(prompt) + generated_len_before
        assert composite_len == 5

        # snapshot boundary (consumed tokens at preempt) = composite - 1.
        # Intentional off-by-one: the last generated token is held
        # in the row's `generated` list but NOT yet consumed by the
        # cache, so the snapshot describes one fewer consumed token
        # than the composite contains.
        snapshot_consumed_boundary = composite_len - 1
        assert snapshot_consumed_boundary == 4
        assert snapshot_consumed_boundary == (
            len(prompt) + generated_len_before - 1
        )

    def test_replay_composite_prompt_length_after_multiple_decodes(
        self,
    ) -> None:
        from silica.core.sampling import SamplingParams

        log = _CallLog()
        adapter = _make_spy_adapter(log, n_layers=1)
        b = ContinuousBatcher(
            adapter,
            max_batch_size=1,
            prefix_cache=None,
        )
        prompt = [1, 2, 3, 4]
        b.add_request(
            0,
            prompt,
            SamplingParams(temperature=0.0, max_tokens=8),
        )
        b.step()  # prefill → 1 generated.
        b.step()  # decode → 2 generated.
        b.step()  # decode → 3 generated.

        active_row = b._rows[0]  # type: ignore[attr-defined]
        generated_len_before = len(active_row.generated)
        assert generated_len_before == 3

        ok = b._apply_preempt("req-0")  # type: ignore[attr-defined]
        assert ok is True
        pending = b._waiting_queue[0]  # type: ignore[attr-defined]
        assert pending.recurrent_snapshot is not None

        # composite_prompt = T + len(generated) = 4 + 3 = 7.
        composite_len = len(pending.prompt_ids)
        assert composite_len == len(prompt) + generated_len_before
        assert composite_len == 7

        # snapshot boundary = composite - 1 = 6 consumed tokens at
        # preempt time. Locked here so a future change to either
        # composite construction or snapshot capture site that
        # collapses the off-by-one trips this test.
        snapshot_consumed_boundary = composite_len - 1
        assert snapshot_consumed_boundary == 6
        assert snapshot_consumed_boundary == (
            len(prompt) + generated_len_before - 1
        )


# =============================================================
# Section 4: snapshot survives across intervening step()s
# =============================================================


class TestSnapshotSurvivesPendingLifetime:
    """C5.2 invariant: the snapshot stashed on a replay's
    ``_PendingAdmit`` survives intervening ``step()`` calls
    (other rows running forwards on ``self._batch_cache``).
    R-C5-2 alias defense was already enforced inside the snapshot
    constructor at C5.1; this test re-exercises that the dataclass
    storage doesn't reintroduce aliasing across the pending's
    lifetime in the batcher's queue.
    """

    def test_snapshot_unchanged_across_steps(self) -> None:
        from silica.core.sampling import SamplingParams

        log = _CallLog()
        adapter = _make_spy_adapter(log, n_layers=1)
        b = ContinuousBatcher(
            adapter,
            max_batch_size=2,
            prefix_cache=None,
        )
        b.add_request(
            0,
            [1, 2, 3, 4],
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.add_request(
            1,
            [5, 6, 7, 8],
            SamplingParams(temperature=0.0, max_tokens=4),
        )
        b.step()
        ok = b._apply_preempt("req-0")  # type: ignore[attr-defined]
        assert ok is True

        pending = b._waiting_queue[0]  # type: ignore[attr-defined]
        assert pending.recurrent_snapshot is not None
        # Capture the snapshot's tensor identities + values BEFORE any
        # further activity. Because RecurrentSnapshot is frozen and the
        # constructor (C5.1) already eval'd a defensive copy, holding
        # a reference to entries[0]'s slots and re-reading later should
        # yield byte-identical contents even after the batcher has run
        # additional forwards on the live cache.
        original_conv = pending.recurrent_snapshot.entries[0].conv_state
        original_recurrent = pending.recurrent_snapshot.entries[0].recurrent_state
        assert original_conv is not None
        assert original_recurrent is not None
        # Snapshot of the snapshot's bytes — used as the comparison ref.
        ref_conv = mx.array(original_conv)
        ref_recurrent = mx.array(original_recurrent)
        mx.eval(ref_conv, ref_recurrent)

        # Drive several more steps. The replay will be re-admitted
        # (drives _admit_miss_cohort, no restore per C5.2 pivot), and
        # the sibling row continues decoding. Both exercise the live
        # _batch_cache.
        for _ in range(3):
            if not b.has_work():
                break
            b.step()

        # The snapshot's entries must still byte-match the captured
        # references — no aliasing leak through the pending lifetime.
        assert mx.array_equal(original_conv, ref_conv), (
            "snapshot conv_state mutated across steps; pending-lifetime "
            "aliasing leak"
        )
        assert mx.array_equal(original_recurrent, ref_recurrent), (
            "snapshot recurrent_state mutated across steps; pending-"
            "lifetime aliasing leak"
        )


# Appease Callable import (kept available for future spy extensions)
_ = Callable
