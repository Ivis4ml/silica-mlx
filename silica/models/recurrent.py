"""P-3-C5.1 — adapter-owned recurrent state snapshot / restore.

Per ``plans/P3_C5_OPENING.md`` §5: snapshot / restore are
**adapter-owned** hooks; the batcher owns ``self._batch_cache``
and the adapter interprets which slots are recurrent. Per
``plans/P3_C5_OPENING.md`` §5.3: lean is Option A, a
``runtime_checkable`` Protocol mixin so ``ContinuousBatcher``
can ``isinstance``-dispatch without knowing the concrete adapter
class.

C5.1 lands the API surface plus its sole implementation today —
``Qwen3_5Adapter`` (`silica/models/qwen3_5.py`). Adapters without
recurrent state (``Qwen3Adapter``, ``Gemma4Adapter``) do not
implement the mixin; ``isinstance(adapter, RecurrentStateAdapter)``
returns ``False`` and callers fall back to the
"lose-state-on-preempt" semantics that hold today.

Storage / interpretation split (per ``plans/P3_C5_OPENING.md``
§5.1): the snapshot / restore methods take the **batcher-supplied**
``cache_list`` plus a ``row_idx``; they do NOT consult any
adapter-local request-keyed store. The adapter walks DeltaNet
layers (``layer.is_linear == True``), reads the per-row slot
tensors out of each layer's ``ArraysCache``, materialises a
detached copy (R-C5-2: ``mx.eval`` + ``mx.array`` so subsequent
in-place writes at ``mlx_lm/models/qwen3_5.py:164/166/197``
cannot mutate the snapshot's contents), and packs the result
into a frozen ``RecurrentSnapshot`` value object. ``restore`` is
the inverse — splice the snapshot's per-row tensors back into
the live cache without touching other rows.

Out of C5.1 scope (will land in later sub-units):

- ``ContinuousBatcher`` integration (preempt / replay) — C5.2.
- ``RadixPrefixCache`` cooperation (in-flight per-block snapshot
  capture) — C5.3.
- The ``has_recurrent_state=True + prefix_cache`` guard at
  ``silica/scheduler/batcher.py:152-166`` — removed at C5.4.
- P-7 speculative draft-rollback wiring — independent unit; this
  surface is shape-compatible per opening §5.4.

Inventory pointers:

- DeltaNet ``ArraysCache(size=2)`` slot layout per Qwen3.5 target
  is documented in ``plans/P3_C5_STATE_INVENTORY/README.md`` §2/§3.
- Silica today never calls ``ArraysCache.prepare(lengths=...)``
  (verified by ``rg -n '\\.prepare\\(|prepare\\(lengths' silica/``
  → zero matches; see inventory README §5), so the
  ``cache.lengths is None`` branch is the only one C5.1 must
  handle. The ``mlx_lm/models/qwen3_5.py:164``
  ``mx.take_along_axis`` branch is dead in the current call
  graph; future adoption of ``prepare(lengths=...)`` would flip
  that branch on but leaves the snapshot shape unchanged (both
  branches write to ``cache[0]`` with the same per-row layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx


@dataclass(frozen=True)
class _RecurrentLayerEntry:
    """One DeltaNet layer's frozen state pair (conv + recurrent).

    Both ``conv_state`` and ``recurrent_state`` are sliced to
    leading ``(1, ...)`` (single row) and detached from any
    live-cache aliasing via ``mx.array(...)`` + ``mx.eval(...)``
    at ``RecurrentSnapshot`` construction time. Either field may
    be ``None`` when the corresponding slot was lazy-allocated
    but unwritten at snapshot time (no forward has run yet on
    that layer for the snapshotted row).

    Frozen at the dataclass level; the mlx arrays themselves are
    immutable by the standard MLX semantics, so the snapshot is
    safe against subsequent in-place rebinds at
    ``mlx_lm/models/qwen3_5.py:164/166/197``.

    ``layer_idx`` is the absolute index into the model's layer
    list (e.g. ``[0, 1, 2, 4, 5, 6, 8, ...]`` on Qwen3.5-0.8B —
    DeltaNet layers only, GLOBAL layers omitted).
    """

    layer_idx: int
    conv_state: mx.array | None
    recurrent_state: mx.array | None


@dataclass(frozen=True)
class RecurrentSnapshot:
    """Opaque, read-only snapshot of one batch row's recurrent
    state across all DeltaNet layers. External consumers
    (``ContinuousBatcher`` at C5.2, prefix cache at C5.3) handle
    whole snapshots; they do not inspect individual entries.

    Memory bytes are reported via ``nbytes`` for the C5.3
    per-block-boundary memory budget (``plans/P3_C5_STATE_INVENTORY/README.md``
    §6 expects ~19.5 MB on Qwen3.5-0.8B, ~51.5 MB on 4B,
    ~64.4 MB on 35B-A3B at B=1).
    """

    entries: tuple[_RecurrentLayerEntry, ...]
    nbytes: int


@runtime_checkable
class RecurrentStateAdapter(Protocol):
    """Mixin implemented by adapters that carry recurrent state.

    Absent on ``Qwen3Adapter`` / ``Gemma4Adapter`` (no recurrent
    state); present on ``Qwen3_5Adapter`` (and, when it lands,
    ``Qwen3_5MoEAdapter``). The ``@runtime_checkable`` decorator
    lets call sites dispatch via ``isinstance(adapter,
    RecurrentStateAdapter)`` without knowing the concrete
    adapter class.
    """

    def snapshot_recurrent_state(
        self, cache_list: list[Any], row_idx: int
    ) -> RecurrentSnapshot:
        """Return a frozen snapshot of recurrent state for batch
        row ``row_idx`` across all DeltaNet layers in
        ``cache_list``. Implementation walks the per-layer cache
        objects, slices the recurrent slots to the target row,
        materialises detached copies (R-C5-2), and packs into a
        ``RecurrentSnapshot``."""
        ...

    def restore_recurrent_state(
        self,
        cache_list: list[Any],
        row_idx: int,
        snapshot: RecurrentSnapshot,
    ) -> None:
        """Splice ``snapshot``'s contents back into ``cache_list``
        at batch row ``row_idx``. Other rows in the same batched
        cache are NOT modified."""
        ...


__all__ = [
    "RecurrentSnapshot",
    "RecurrentStateAdapter",
    "_RecurrentLayerEntry",
]
