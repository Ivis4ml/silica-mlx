"""silica.kvcache.store — PrefixBlockStore Protocol + synthetic impl (Unit 16c.2).

RadixPrefixCache was born coupled to ``PagedKVCache``: its ``insert`` /
``lookup`` / ``evict_until`` all call ``kv.incref`` / ``kv.decref``.
Under P-2 Option B, the batcher does not use paged attention — it uses
mlx-lm's ``BatchKVCache``. The prefix cache therefore needs a backend
that **looks like refcount + block storage** to RadixPrefixCache but
internally can be either (a) a PagedKVCache (future paged-attention
kernel track) or (b) a synthetic id-allocator + K/V dict that rides
on top of BatchKVCache via detached slices.

This module pins the Protocol (``PrefixBlockStore``) and ships the
synthetic implementation that 16c.2's ContinuousBatcher will install.

**Two refcount dimensions — source vs hit.**

``retain_source`` pins a block as a copy source inside the radix tree.
It survives the owning request's ``free`` and persists until
``evict_until`` removes the corresponding node. ``retain_hit`` pins a
block because some live request is currently consuming its K/V (a
lookup → release pair). The final source release is safe only when
hits == 0; a ``release_source`` that would drop source to 0 while
hits > 0 indicates a scheduler bug and fails loudly.

Keeping the two dimensions separate in the store (rather than summing
into one refcount) is load-bearing: the eviction decision needs to
ask "are any LIVE requests still reading this?" and "are any
FUTURE-admission sources still owning this?" as independent questions.
``PagedKVCache.incref/decref`` collapses both — 16c.2 unifies the
store interface but not the refcount view.

**Detached K/V storage (synthetic backend only).**

``register_detached`` associates per-layer ``(K, V)`` slices with a
block id; ``fetch_detached`` returns them for later copy into a
freshly-seeded ``BatchKVCache`` at admission time. Lifetime rule:
detached storage must be released **before the final source release**,
so a ``release_source`` transition to zero fails loudly if detached K/V
is still registered. This enforces the L-3 ⊆ L-1 invariant from
docs/P2_UNIT_16C_2_PREP.md §2 and prevents silent leaks on eviction.

The Paged backend stubs ``register_detached`` / ``fetch_detached`` /
``release_detached`` with ``NotImplementedError`` — the paged-attention
kernel will define its own detached-K/V model when the trigger-gated
track starts. Step 2 ships only the synthetic backend; the Paged
backend lands with step 3 (RadixPrefixCache refactor).

**KVCodec hook (P-4.5-C.1, docs/P4_5_C_KVCODEC_OPENING.md §3.2 / §6).**

``SyntheticPrefixBlockStore`` accepts an optional ``codec: KVCodec``
kwarg. When supplied, ``register_detached`` calls
``codec.encode_block(k, v)`` per layer and stores the resulting
``CodedBlock`` objects; ``fetch_detached`` calls
``codec.decode_block(cb)`` per layer to return shape-preserving fp16
``(K, V)`` tuples the downstream ``build_seeded_batch_kv`` expects.
When ``codec`` is ``None`` (the pre-C.1 default), the store runs a
pass-through path that wraps the raw tensors in ``CodedBlock`` with
``resident_bytes = k.nbytes + v.nbytes`` — semantically identical to
``IdentityCodec`` but without requiring ``n_kv_heads`` / ``head_dim``
construction arguments. Either path preserves the external return
shape contract; the two paths produce byte-identical outputs on the
hit path under ``IdentityCodec`` because the codec returns its input
tensors by reference.

The ``PagedPrefixBlockStore`` deliberately does not accept a codec —
Option (A) / (C) codec integration on the active-KV path is excluded
in v0.1 by D-003 (no compressed-domain attention) and Q-009 / R-7
(no MLX variable-length SDPA).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mlx.core as mx

from silica.kvcache.codec import CodedBlock, KVCodec

if TYPE_CHECKING:
    from silica.kvcache.paged import PagedKVCache


@runtime_checkable
class PrefixBlockStore(Protocol):
    """Refcount + block-storage backend for RadixPrefixCache (16c.2).

    See module docstring for the two refcount dimensions
    (source / hit) and the detached-K/V lifetime rule.
    """

    block_size: int

    def allocate_id(self) -> int:
        """Return a fresh block id not currently in use by this store."""
        ...

    def retain_source(self, block_id: int) -> None:
        """Add a source retention to ``block_id``.

        Called when a radix node is created (``insert_detached``).
        First retain_source on a block id registers the block with the
        store; subsequent retain_source calls bump the source refcount
        (duplicate-prefix inserts).
        """
        ...

    def release_source(self, block_id: int) -> None:
        """Drop a source retention from ``block_id``.

        Called when a radix node is evicted. The final source release
        fails loudly if any live hits still reference the block
        (L-2 ⊆ L-1) or if detached K/V is still registered
        (L-3 ⊆ L-1). When source refs reach 0 the block is no longer
        live in this store.
        """
        ...

    def retain_hit(self, block_id: int) -> None:
        """Add a live-hit retention to ``block_id``.

        Called from ``RadixPrefixCache.lookup``. Requires the block
        to already have a source retention (L-2 ⊆ L-1); raises
        ``KeyError`` otherwise.
        """
        ...

    def release_hit(self, block_id: int) -> None:
        """Drop a live-hit retention from ``block_id``.

        Called from ``RadixPrefixCache.release``. Does NOT touch
        source refs or detached storage.
        """
        ...

    def hit_refs(self, block_id: int) -> int:
        """Current live-hit refcount for ``block_id`` (0 if none).

        Load-bearing for ``RadixPrefixCache.evict_until``: the tree
        walks candidate nodes and skips any whose block still has a
        live hit. Without this query evict_until would have to probe
        the store via ``release_source`` and catch ``RuntimeError``,
        which turns an O(1) check into O(1) + exception machinery.
        """
        ...

    def has_detached(self, block_id: int) -> bool:
        """Whether detached K/V is currently registered for ``block_id``.

        Load-bearing for ``RadixPrefixCache.evict_until``: the tree
        must call ``release_detached`` only when there is something
        to release. Paged-backend implementations return ``False``
        unconditionally; Synthetic checks its ``_detached`` dict.
        """
        ...

    def register_detached(
        self,
        block_id: int,
        per_layer_kv: Sequence[tuple[mx.array, mx.array]],
    ) -> None:
        """Associate detached per-layer K/V slices with ``block_id``.

        Must be called after ``retain_source`` on the same id. Each
        entry in ``per_layer_kv`` is one ``(K, V)`` pair for one
        transformer layer, shaped ``(1, n_kv_heads, block_size,
        head_dim)``. Fails loudly on missing source or duplicate
        registration.
        """
        ...

    def fetch_detached(
        self, block_id: int
    ) -> Sequence[tuple[mx.array, mx.array]]:
        """Return the per-layer detached K/V for ``block_id``.

        Raises ``KeyError`` if no detached K/V is registered.
        """
        ...

    def release_detached(self, block_id: int) -> None:
        """Drop the detached K/V for ``block_id``. Must be called
        before the paired ``release_source`` during eviction.
        """
        ...


class SyntheticPrefixBlockStore:
    """In-memory ``PrefixBlockStore`` for the BatchKVCache admission path.

    Owns its own id counter, two refcount dicts (source / hit), and a
    detached-K/V dict keyed by block id. Does NOT interact with
    ``PagedKVCache`` — the synthetic name reflects that block ids here
    are internal handles, not rows in a paged physical store.
    """

    def __init__(
        self,
        *,
        block_size: int,
        codec: KVCodec | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if codec is not None and codec.block_size != block_size:
            raise ValueError(
                f"codec.block_size ({codec.block_size}) must match "
                f"store.block_size ({block_size})"
            )
        self.block_size = block_size
        self._codec = codec
        self._next_id: int = 0
        self._source_refs: dict[int, int] = {}
        self._hit_refs: dict[int, int] = {}
        self._detached: dict[int, tuple[CodedBlock, ...]] = {}

    def _encode(self, k: mx.array, v: mx.array) -> CodedBlock:
        if self._codec is None:
            return CodedBlock(k=k, v=v, resident_bytes=k.nbytes + v.nbytes)
        return self._codec.encode_block(k, v)

    def _decode(self, block: CodedBlock) -> tuple[mx.array, mx.array]:
        if self._codec is None:
            return block.k, block.v
        return self._codec.decode_block(block)

    def allocate_id(self) -> int:
        """Return a monotonically-increasing fresh block id.

        Ids are not reused after ``release_source`` drops a block's
        source refcount to zero. Monotonic ids simplify reasoning about
        "is this the same block as before" in the presence of
        insert/evict churn — a reused id would be ambiguous.
        """
        out = self._next_id
        self._next_id += 1
        return out

    def retain_source(self, block_id: int) -> None:
        self._source_refs[block_id] = self._source_refs.get(block_id, 0) + 1

    def release_source(self, block_id: int) -> None:
        if block_id not in self._source_refs:
            raise KeyError(f"block {block_id}: no outstanding source ref")
        new_count = self._source_refs[block_id] - 1
        if new_count > 0:
            # Still has another source holder — L-2 / L-3 guards do not
            # fire here because the block remains alive. A hit ref
            # or detached entry is still fully reachable through the
            # surviving source.
            self._source_refs[block_id] = new_count
            return
        # About to fully release — the block's lifetime is ending, so
        # any outstanding hit ref (L-2) or detached K/V (L-3) would
        # strand. Fail loudly instead of silently leaking.
        hit = self._hit_refs.get(block_id, 0)
        if hit > 0:
            raise RuntimeError(
                f"block {block_id}: cannot release_source while "
                f"hit_refs={hit} (would strand live prefix-cache hits)"
            )
        if block_id in self._detached:
            raise RuntimeError(
                f"block {block_id}: cannot release_source while "
                f"detached K/V is still registered (call release_detached first)"
            )
        del self._source_refs[block_id]

    def retain_hit(self, block_id: int) -> None:
        if block_id not in self._source_refs:
            raise KeyError(
                f"block {block_id}: no source ref; retain_hit requires "
                f"the block to be live in the radix tree (L-2 ⊆ L-1)"
            )
        self._hit_refs[block_id] = self._hit_refs.get(block_id, 0) + 1

    def release_hit(self, block_id: int) -> None:
        count = self._hit_refs.get(block_id, 0)
        if count <= 0:
            raise KeyError(
                f"block {block_id}: no outstanding hit ref "
                f"(mismatched release — caller bug)"
            )
        if count == 1:
            del self._hit_refs[block_id]
        else:
            self._hit_refs[block_id] = count - 1

    def register_detached(
        self,
        block_id: int,
        per_layer_kv: Sequence[tuple[mx.array, mx.array]],
    ) -> None:
        if block_id not in self._source_refs:
            raise KeyError(
                f"block {block_id}: register_detached requires a live "
                f"source ref (call retain_source first)"
            )
        if block_id in self._detached:
            raise ValueError(
                f"block {block_id}: detached K/V already registered "
                f"(duplicate register — caller bug)"
            )
        self._detached[block_id] = tuple(
            self._encode(k, v) for k, v in per_layer_kv
        )

    def fetch_detached(
        self, block_id: int
    ) -> Sequence[tuple[mx.array, mx.array]]:
        if block_id not in self._detached:
            raise KeyError(
                f"block {block_id}: no detached K/V registered"
            )
        return tuple(self._decode(cb) for cb in self._detached[block_id])

    def release_detached(self, block_id: int) -> None:
        if block_id not in self._detached:
            raise KeyError(
                f"block {block_id}: no detached K/V to release"
            )
        del self._detached[block_id]

    def hit_refs(self, block_id: int) -> int:
        """Current hit refcount for ``block_id`` (0 if none)."""
        return self._hit_refs.get(block_id, 0)

    def has_detached(self, block_id: int) -> bool:
        """Whether detached K/V is currently registered for ``block_id``."""
        return block_id in self._detached

    # --- debug / inspection helpers (not part of the Protocol) ---

    def source_refs(self, block_id: int) -> int:
        """Current source refcount for ``block_id`` (0 if none)."""
        return self._source_refs.get(block_id, 0)

    def live_block_ids(self) -> frozenset[int]:
        """Snapshot of all block ids with nonzero source refs."""
        return frozenset(self._source_refs.keys())

    def resident_bytes(self) -> int:
        """Sum of ``CodedBlock.resident_bytes`` across all detached blocks.

        Parallel observable per P-4.5-C.0 opening doc §6.2 / §8.3 —
        the total resident prefix-cache K/V bytes held by this store.
        Under the default pass-through path (no codec) or under
        ``IdentityCodec``, this equals ``len(live_block_ids()) ×
        num_layers × block_size × (2 × n_kv_heads × head_dim ×
        dtype.size)``. Under a future non-identity codec (BlockTQ /
        RaBitQ, P-5 proper) it reflects the compressed residency.

        Intentionally **not** on the ``PrefixBlockStore`` Protocol —
        the paged backend does not track per-layer physical K/V
        residency this way, and adding it to the Protocol would force
        ``PagedPrefixBlockStore`` to choose between a wrong number and
        a ``NotImplementedError`` at the Protocol boundary.
        """
        return sum(
            sum(cb.resident_bytes for cb in coded_tuple)
            for coded_tuple in self._detached.values()
        )


class PagedPrefixBlockStore:
    """``PrefixBlockStore`` that adapts a ``PagedKVCache`` for RadixPrefixCache.

    Block ids are **allocated by the caller** (via
    ``PagedKVCache.reserve_for_prefill`` / ``append_slot``), not by this
    store. ``allocate_id`` raises — the Paged path uses
    ``RadixPrefixCache.insert(tokens, block_ids)`` where the caller
    supplies ids; ``insert_detached`` is a Synthetic-only admission
    path.

    Detached K/V is not modelled in the Paged backend; the
    paged-attention kernel track will define its own detached-K/V
    semantics once that P-2 Opening trigger fires. Until then
    ``register_detached`` / ``fetch_detached`` / ``release_detached``
    raise ``NotImplementedError``; ``has_detached`` returns ``False``
    unconditionally so ``evict_until`` can loop without special-casing.

    Under the hood this store maintains its own source / hit counter
    dicts and forwards refcount changes to ``PagedKVCache.incref /
    decref`` — the aggregate kv refcount ("any holder at all?") still
    drives the physical free-pool, while the split source / hit view
    preserves the L-1 / L-2 invariants that ``evict_until`` depends
    on. This mirrors what RadixPrefixCache's own ``_live_hits`` dict
    did before step 3.
    """

    def __init__(self, kv: "PagedKVCache") -> None:
        self._kv = kv
        self.block_size = kv.block_size
        self._source_refs: dict[int, int] = {}
        self._hit_refs: dict[int, int] = {}

    def allocate_id(self) -> int:
        raise NotImplementedError(
            "PagedPrefixBlockStore does not allocate block ids; the "
            "caller supplies them from PagedKVCache.reserve_for_prefill "
            "or append_slot. Use RadixPrefixCache.insert(tokens, "
            "block_ids) for the Paged admission path — insert_detached "
            "is Synthetic-only."
        )

    def retain_source(self, block_id: int) -> None:
        self._kv.incref(block_id)
        self._source_refs[block_id] = self._source_refs.get(block_id, 0) + 1

    def release_source(self, block_id: int) -> None:
        if block_id not in self._source_refs:
            raise KeyError(f"block {block_id}: no outstanding source ref")
        new_count = self._source_refs[block_id] - 1
        if new_count > 0:
            self._kv.decref(block_id)
            self._source_refs[block_id] = new_count
            return
        # Transition to zero — L-2 guard (L-3 N/A; no detached in Paged).
        hit = self._hit_refs.get(block_id, 0)
        if hit > 0:
            raise RuntimeError(
                f"block {block_id}: cannot release_source while "
                f"hit_refs={hit} (would strand live prefix-cache hits)"
            )
        self._kv.decref(block_id)
        del self._source_refs[block_id]

    def retain_hit(self, block_id: int) -> None:
        if block_id not in self._source_refs:
            raise KeyError(
                f"block {block_id}: no source ref; retain_hit requires "
                f"the block to be live in the radix tree (L-2 ⊆ L-1)"
            )
        self._kv.incref(block_id)
        self._hit_refs[block_id] = self._hit_refs.get(block_id, 0) + 1

    def release_hit(self, block_id: int) -> None:
        count = self._hit_refs.get(block_id, 0)
        if count <= 0:
            raise KeyError(
                f"block {block_id}: no outstanding hit ref "
                f"(mismatched release — caller bug)"
            )
        self._kv.decref(block_id)
        if count == 1:
            del self._hit_refs[block_id]
        else:
            self._hit_refs[block_id] = count - 1

    def hit_refs(self, block_id: int) -> int:
        return self._hit_refs.get(block_id, 0)

    def has_detached(self, block_id: int) -> bool:
        return False

    def register_detached(
        self,
        block_id: int,
        per_layer_kv: Sequence[tuple[mx.array, mx.array]],
    ) -> None:
        raise NotImplementedError(
            "PagedPrefixBlockStore does not store detached K/V; the "
            "paged-attention kernel track owns that model (see "
            "docs/P2_OPENING.md — trigger-gated future track)."
        )

    def fetch_detached(
        self, block_id: int
    ) -> Sequence[tuple[mx.array, mx.array]]:
        raise NotImplementedError(
            "PagedPrefixBlockStore does not store detached K/V."
        )

    def release_detached(self, block_id: int) -> None:
        raise NotImplementedError(
            "PagedPrefixBlockStore does not store detached K/V."
        )

    # --- debug / inspection helpers (not part of the Protocol) ---

    def source_refs(self, block_id: int) -> int:
        return self._source_refs.get(block_id, 0)


__all__ = [
    "PagedPrefixBlockStore",
    "PrefixBlockStore",
    "SyntheticPrefixBlockStore",
]
