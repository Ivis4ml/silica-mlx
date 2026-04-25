"""silica.kvcache.prefix — RadixPrefixCache for P-2 shared-prefix cache.

Block-granular trie keyed by blocks of ``block_size`` tokens. Each tree
node holds one block id; edges are always exactly ``block_size`` tokens
long. Inspired by mini-sglang's prefix tree — no cross-reference to
runtime mini-sglang code (D-009: no non-MLX runtime imports).

**Physical semantics — option B (copy-on-admit), per docs/P2_OPENING.md v2.1:**

A prefix hit means: on admission, the batcher copies the source blocks'
K/V into the new request's fresh batch row, skipping that prefix's
forward compute. The source blocks are **not** shared across live
requests at the physical level; they are retained as copy sources that
survive their originating request via refcount pinning.

**Backend seam — PrefixBlockStore Protocol (Unit 16c.2 step 3):**

Prior to 16c.2, this class depended concretely on ``PagedKVCache`` and
called ``kv.incref`` / ``kv.decref`` with the aggregate refcount
collapsing "I am a radix source" and "I am a live hit" into one number.
Step 3 inverts the dependency: RadixPrefixCache now holds a
``PrefixBlockStore`` (see ``silica.kvcache.store``) with four explicit
retain/release methods (``retain_source`` / ``release_source`` /
``retain_hit`` / ``release_hit``). Two implementations:

- ``PagedPrefixBlockStore`` — wraps ``PagedKVCache``, preserves today's
  refcount / evict behaviour bit-for-bit; caller supplies block ids via
  ``insert(tokens, block_ids)``.
- ``SyntheticPrefixBlockStore`` — allocates its own ids and stores
  detached per-layer K/V slices; enables ``insert_detached`` for
  16c.2's BatchKVCache admission path.

**Two radix operations for lookup (Unit 16c.2 step 3):**

- ``peek(tokens)`` — side-effect-free walk. No ``retain_hit``, no
  LRU touch, no ``self.hits`` increment. Use during admission planning
  before you have decided to actually copy K/V.
- ``lookup(tokens)`` — retained walk. Retains each hit block via
  ``store.retain_hit`` and advances LRU + ``self.hits``. Must be paired
  with a later ``release(block_ids)``.

Keeping the walk atomic inside the class (not split across
``peek``-then-external-``retain_hit``) guarantees every future change
to "what happens on a hit" travels through one code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from silica.kvcache.manager import PrefixHit
from silica.kvcache.store import PrefixBlockStore

if TYPE_CHECKING:
    from silica.models.recurrent import RecurrentSnapshot

_CHUNK = tuple[int, ...]


class _Node:
    """Single-block node in the radix tree.

    Root has empty ``tokens`` / ``block_id = -1`` and no parent. Every
    non-root node represents exactly ``block_size`` tokens and one block id.

    ``recurrent_snapshot`` (P-3-C5.3.0): optional per-tree-path recurrent
    state snapshot, captured at the boundary marked by this node's tokens.
    Default ``None``; backfilled by ``insert_detached``'s duplicate-prefix
    branch when an insertion brings a fresh snapshot for a previously
    snapshotless node. Eviction is automatic — when ``_evict_node`` drops
    the node, Python GC releases the snapshot's mlx-array tensors.
    """

    __slots__ = (
        "parent",
        "tokens",
        "block_id",
        "children",
        "access_tick",
        "recurrent_snapshot",
    )

    def __init__(
        self,
        parent: "_Node | None",
        tokens: _CHUNK,
        block_id: int,
        access_tick: int = 0,
        recurrent_snapshot: "RecurrentSnapshot | None" = None,
    ) -> None:
        self.parent = parent
        self.tokens = tokens
        self.block_id = block_id
        # Children keyed by the chunk tuple of the child's tokens.
        self.children: dict[_CHUNK, _Node] = {}
        self.access_tick = access_tick
        self.recurrent_snapshot: "RecurrentSnapshot | None" = recurrent_snapshot


class RadixPrefixCache:
    """Block-granular radix trie for P-2 shared-prefix cache."""

    def __init__(
        self,
        *,
        block_size: int,
        store: PrefixBlockStore,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if store.block_size != block_size:
            raise ValueError(
                f"block_size mismatch: pc={block_size} vs store={store.block_size}"
            )
        self._block_size = block_size
        self._store = store
        self._root = _Node(parent=None, tokens=(), block_id=-1)
        self._tick = 0
        self.hits: int = 0

    @property
    def store(self) -> PrefixBlockStore:
        """Read-only access to the underlying ``PrefixBlockStore``.

        Consumers outside this module should use ``prefix_cache.store``
        rather than poking ``prefix_cache._store``. Lands in P-5-A.0.4 so
        the P-5-A.2 ``MemoryBudgeter`` can read ``store.resident_bytes()``
        without reaching into a private attribute.

        Note: ``resident_bytes()`` is deliberately **not** on the
        ``PrefixBlockStore`` Protocol — see the rationale in
        ``silica/kvcache/store.py::SyntheticPrefixBlockStore.resident_bytes``.
        Consumers that need to read it must use a structural capability
        check (``SupportsResidentBytes`` Protocol or a ``hasattr`` guard),
        not assume every store exposes it.
        """
        return self._store

    # --- public surface ---

    @property
    def block_size(self) -> int:
        """Token count per radix block."""
        return self._block_size

    def peek(self, tokens: Sequence[int]) -> PrefixHit:
        """Walk the tree, return the longest block-aligned match
        **without any side effects**.

        No ``retain_hit``, no ``self.hits`` increment, no LRU touch.
        Used by the admission planner to size the prefix hit before
        committing to the copy-on-admit path.
        """
        node = self._root
        hit_blocks: list[int] = []
        idx = 0
        n = len(tokens)
        while idx + self._block_size <= n:
            chunk = tuple(tokens[idx : idx + self._block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            hit_blocks.append(child.block_id)
            node = child
            idx += self._block_size
        return PrefixHit(block_ids=tuple(hit_blocks), num_hit_tokens=idx)

    def lookup(self, tokens: Sequence[int]) -> PrefixHit:
        """Walk the tree and **retain** each hit block in the store.

        Caller is responsible for a later ``release(block_ids)`` once
        the hit K/V has been copied (or the admission has aborted).
        Non-block-aligned trailing tokens do not contribute.
        """
        node = self._root
        hit_blocks: list[int] = []
        idx = 0
        n = len(tokens)
        while idx + self._block_size <= n:
            chunk = tuple(tokens[idx : idx + self._block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            hit_blocks.append(child.block_id)
            self._store.retain_hit(child.block_id)
            self._touch(child)
            node = child
            idx += self._block_size
        if hit_blocks:
            self.hits += 1
        return PrefixHit(block_ids=tuple(hit_blocks), num_hit_tokens=idx)

    def peek_with_node(
        self, tokens: Sequence[int]
    ) -> tuple[PrefixHit, _Node | None]:
        """Side-effect-free walk that also returns the deepest matched node.

        Same contract as ``peek``: no ``retain_hit``, no ``self.hits``
        increment, no LRU touch. Adds a second tuple element — the
        deepest ``_Node`` reached, or ``None`` when zero blocks matched.

        P-3-C5.3.0. Used by the Phase-B classifier (C5.3.3) on a
        usable-aligned prompt slice to inspect the deepest USABLE node's
        ``recurrent_snapshot`` before committing to the hit path.
        """
        node = self._root
        deepest: _Node | None = None
        hit_blocks: list[int] = []
        idx = 0
        n = len(tokens)
        while idx + self._block_size <= n:
            chunk = tuple(tokens[idx : idx + self._block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            hit_blocks.append(child.block_id)
            deepest = child
            node = child
            idx += self._block_size
        return (
            PrefixHit(block_ids=tuple(hit_blocks), num_hit_tokens=idx),
            deepest,
        )

    def lookup_with_node(
        self, tokens: Sequence[int]
    ) -> tuple[PrefixHit, _Node | None]:
        """Retained walk that also returns the deepest matched node.

        Same contract as ``lookup``: each hit block is ``retain_hit``'d,
        each touched node advances the LRU tick, ``self.hits`` increments
        on a non-empty hit. Returns the deepest ``_Node`` reached
        alongside the ``PrefixHit`` (``None`` when zero blocks matched).

        P-3-C5.3.0. Replaces ``lookup`` at C5.3.3's
        ``_admit_single_hit_row`` call site so the atomic phase keeps a
        reference to the deepest USABLE node — the one whose
        ``recurrent_snapshot`` will be restored before suffix prefill.
        """
        node = self._root
        deepest: _Node | None = None
        hit_blocks: list[int] = []
        idx = 0
        n = len(tokens)
        while idx + self._block_size <= n:
            chunk = tuple(tokens[idx : idx + self._block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            hit_blocks.append(child.block_id)
            self._store.retain_hit(child.block_id)
            self._touch(child)
            deepest = child
            node = child
            idx += self._block_size
        if hit_blocks:
            self.hits += 1
        return (
            PrefixHit(block_ids=tuple(hit_blocks), num_hit_tokens=idx),
            deepest,
        )

    def insert(
        self, tokens: Sequence[int], block_ids: Sequence[int]
    ) -> None:
        """Add caller-allocated blocks to the tree at block-aligned prefix points.

        Used by the Paged backend path where block ids come from
        ``PagedKVCache.reserve_for_prefill`` / ``append_slot``.
        Partial trailing blocks are discarded. Idempotent on
        already-covered prefixes: a duplicate insert reuses the existing
        block id; the caller's corresponding block stays with its
        owning request.

        For each newly-added node the block is ``store.retain_source``'d.
        """
        n_blocks_want = len(block_ids)
        n_blocks_feasible = min(n_blocks_want, len(tokens) // self._block_size)
        if n_blocks_feasible == 0:
            return

        node = self._root
        for i in range(n_blocks_feasible):
            start = i * self._block_size
            chunk = tuple(tokens[start : start + self._block_size])
            existing = node.children.get(chunk)
            if existing is not None:
                self._touch(existing)
                node = existing
                continue
            new_node = _Node(
                parent=node,
                tokens=chunk,
                block_id=block_ids[i],
            )
            self._store.retain_source(block_ids[i])
            self._touch(new_node)
            node.children[chunk] = new_node
            node = new_node

    def insert_detached(
        self,
        tokens: Sequence[int],
        detached_blocks: Sequence[Sequence[tuple[mx.array, mx.array]]],
        recurrent_snapshots: "Sequence[RecurrentSnapshot | None] | None" = None,
    ) -> tuple[int, ...]:
        """Add tokens as aligned-block nodes with attached K/V slices.

        ``detached_blocks`` is indexed ``[block_idx][layer_idx]`` →
        ``(K, V)``. Precondition: at least one entry per aligned block
        in ``tokens`` and every inner sequence has length ``num_layers``.
        Extra trailing blocks beyond ``len(tokens) // block_size`` are
        silently discarded (token layer is the authority).

        For each **new** node: ``store.allocate_id`` → ``retain_source``
        → ``register_detached(block_id, detached_blocks[i])``. For
        **duplicate-prefix** nodes (the tree already has a matching
        child): the existing node is touched; the caller's corresponding
        entry in ``detached_blocks`` is NOT registered and becomes
        GC-eligible when the caller's outer list goes out of scope.

        ``recurrent_snapshots`` (P-3-C5.3.0): optional per-block recurrent
        state snapshots. ``None`` (default) preserves pre-C5.3 behaviour
        bit-for-bit — every newly-created node has
        ``recurrent_snapshot=None`` and the duplicate-prefix branch leaves
        existing nodes untouched. When provided, length must cover every
        aligned block; ``recurrent_snapshots[i]`` may itself be ``None``
        for blocks where no snapshot was captured this insertion.

        With ``recurrent_snapshots`` provided:

        - **New node**: ``recurrent_snapshots[i]`` is attached to the
          newly created node (may itself be ``None``).
        - **Duplicate-prefix branch, existing.recurrent_snapshot is None
          and new is non-None**: backfill the existing node with the
          fresh snapshot (per design §3.5.1 — self-healing for legacy
          snapshotless nodes).
        - **Duplicate-prefix branch, both non-None or new is None**:
          keep the existing node's snapshot. The caller's
          ``recurrent_snapshots[i]`` becomes GC-eligible. No equality
          check today — ``RecurrentSnapshot`` lacks the boundary
          metadata to make one mechanical (design §3.5.1).

        Returns a tuple of the block ids **actually newly inserted**
        (empty tuple for a fully duplicate insert).
        """
        n_aligned_blocks = len(tokens) // self._block_size
        if n_aligned_blocks == 0:
            return ()
        if len(detached_blocks) < n_aligned_blocks:
            raise ValueError(
                "detached_blocks must cover every aligned token block: "
                f"got {len(detached_blocks)}, need {n_aligned_blocks}"
            )
        if (
            recurrent_snapshots is not None
            and len(recurrent_snapshots) < n_aligned_blocks
        ):
            raise ValueError(
                "recurrent_snapshots must cover every aligned token block "
                f"when provided: got {len(recurrent_snapshots)}, need "
                f"{n_aligned_blocks}"
            )

        node = self._root
        new_ids: list[int] = []
        for i in range(n_aligned_blocks):
            start = i * self._block_size
            chunk = tuple(tokens[start : start + self._block_size])
            new_snap = (
                recurrent_snapshots[i]
                if recurrent_snapshots is not None
                else None
            )
            existing = node.children.get(chunk)
            if existing is not None:
                if (
                    new_snap is not None
                    and existing.recurrent_snapshot is None
                ):
                    existing.recurrent_snapshot = new_snap
                self._touch(existing)
                node = existing
                continue
            new_id = self._store.allocate_id()
            self._store.retain_source(new_id)
            try:
                self._store.register_detached(new_id, detached_blocks[i])
            except Exception:
                self._store.release_source(new_id)
                raise
            new_node = _Node(
                parent=node,
                tokens=chunk,
                block_id=new_id,
                recurrent_snapshot=new_snap,
            )
            self._touch(new_node)
            node.children[chunk] = new_node
            new_ids.append(new_id)
            node = new_node
        return tuple(new_ids)

    def release(self, block_ids: Sequence[int]) -> None:
        """Release a set of hit blocks (paired with a prior ``lookup``).

        Delegates to ``store.release_hit`` — does NOT touch source refs
        or detached storage.

        Raises ``KeyError`` if any block has no outstanding hit — a
        mismatched release indicates a caller bug (D-011 loud-fail).
        """
        for b in block_ids:
            self._store.release_hit(b)

    def fetch_detached_blocks(
        self, block_ids: Sequence[int]
    ) -> list[Sequence[tuple[mx.array, mx.array]]]:
        """Per-block detached K/V for the given hit block ids.

        Shape matches what ``silica.scheduler.seed_kv.build_seeded_batch_kv``
        expects: indexed ``[block_idx][layer_idx]`` → ``(K, V)``.

        Used by 16c.2's admission path after a retained ``lookup`` —
        the batcher passes the result straight into the seeded-cache
        builder. Raises ``KeyError`` (via the underlying store) if any
        id has no detached K/V registered, which would indicate either
        a Paged-backend misuse or a retain/release ordering bug in the
        caller.

        Keeping this as a public one-liner on ``RadixPrefixCache``
        preserves the step-3 abstraction boundary: the batcher never
        needs to reach into ``self._prefix_cache._store``.
        """
        return [self._store.fetch_detached(b) for b in block_ids]

    def evict_until(self, n_blocks: int) -> int:
        """Evict LRU leaf nodes with zero live hits until ``n_blocks`` freed.

        Returns the actual number of blocks freed (may be < ``n_blocks``
        if the tree runs out of evictable nodes).

        Per-node eviction order: ``release_detached`` (if any) THEN
        ``release_source``. The store's transition-to-zero guards
        enforce this order — reversing would raise in the synthetic
        backend. See ``docs/P2_UNIT_16C_2_PREP.md`` §2 (L-3 ⊆ L-1).
        """
        freed = 0
        while freed < n_blocks:
            victim = self._find_oldest_evictable()
            if victim is None:
                break
            self._evict_node(victim)
            freed += 1
        return freed

    # --- debug / inspection (not part of public API) ---

    def node_count(self) -> int:
        """Total non-root nodes in the tree (debug + test helper)."""
        return sum(1 for _ in self._walk_non_root())

    def live_hits(self, block_id: int) -> int:
        """Current live-hit count for ``block_id`` (0 if none).

        Delegates to ``store.hit_refs``; kept as a method on the cache
        so existing callers / tests written against the pre-step-3
        interface continue to work.
        """
        return self._store.hit_refs(block_id)

    # --- internals ---

    def _touch(self, node: _Node) -> None:
        self._tick += 1
        node.access_tick = self._tick

    def _find_oldest_evictable(self) -> _Node | None:
        best: _Node | None = None
        best_tick = 0
        for node in self._walk_non_root():
            if node.children:
                continue  # non-leaf; evicting would orphan the subtree
            if self._store.hit_refs(node.block_id) > 0:
                continue  # still held by a live hit
            if best is None or node.access_tick < best_tick:
                best = node
                best_tick = node.access_tick
        return best

    def _evict_node(self, node: _Node) -> None:
        # Precondition: node is a non-root leaf with no live hits.
        parent = node.parent
        assert parent is not None
        assert not node.children
        del parent.children[node.tokens]
        if self._store.has_detached(node.block_id):
            self._store.release_detached(node.block_id)
        self._store.release_source(node.block_id)

    def _walk_non_root(self) -> list[_Node]:
        out: list[_Node] = []
        stack = list(self._root.children.values())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children.values())
        return out
