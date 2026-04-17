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

import mlx.core as mx

from silica.kvcache.manager import PrefixHit
from silica.kvcache.store import PrefixBlockStore

_CHUNK = tuple[int, ...]


class _Node:
    """Single-block node in the radix tree.

    Root has empty ``tokens`` / ``block_id = -1`` and no parent. Every
    non-root node represents exactly ``block_size`` tokens and one block id.
    """

    __slots__ = ("parent", "tokens", "block_id", "children", "access_tick")

    def __init__(
        self,
        parent: "_Node | None",
        tokens: _CHUNK,
        block_id: int,
        access_tick: int = 0,
    ) -> None:
        self.parent = parent
        self.tokens = tokens
        self.block_id = block_id
        # Children keyed by the chunk tuple of the child's tokens.
        self.children: dict[_CHUNK, _Node] = {}
        self.access_tick = access_tick


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

    # --- public surface ---

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

        node = self._root
        new_ids: list[int] = []
        for i in range(n_aligned_blocks):
            start = i * self._block_size
            chunk = tuple(tokens[start : start + self._block_size])
            existing = node.children.get(chunk)
            if existing is not None:
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
