"""silica.kvcache.prefix — RadixPrefixCache for P-2 shared-prefix cache (Unit #14).

Block-granular trie keyed by blocks of ``block_size`` tokens. Each tree node
holds one block id; edges are always exactly ``block_size`` tokens long.
Inspired by mini-sglang's prefix tree — no cross-reference to runtime
mini-sglang code (D-009: no non-MLX runtime imports).

**Physical semantics — option B (copy-on-admit), per docs/P2_OPENING.md v2.1:**

A prefix hit means: on admission, the batcher copies the source blocks' K/V
into the new request's fresh batch row, skipping that prefix's forward
compute. The source blocks are **not** shared across live requests at the
physical level; they are retained as copy sources that survive their
originating request via refcount pinning.

**Refcount protocol (load-bearing for option B):**

- ``insert(tokens, block_ids)``: for each newly-added node, call
  ``kv.incref(block_id)``. This pins the block as a prefix source. The
  owning request's subsequent ``kv.free`` decrements its own ref but
  the prefix-cache ref keeps the block alive.
- ``lookup(tokens) -> PrefixHit``: for each hit block, call
  ``kv.incref(block_id)`` and increment a **local** ``_live_hits``
  counter (independent of ``kv._refcount``). The live-hits counter
  drives eviction decisions — a node cannot be evicted while any live
  request still holds its block as a hit.
- ``release(block_ids)``: paired with lookup; decrements ``_live_hits``
  and ``kv.decref(block_id)``. Called once the live request has copied
  the K/V into its own row (or finished).
- ``evict_until(n)``: walks LRU for leaf nodes with ``_live_hits == 0``;
  on each eviction, removes the node from the tree and ``kv.decref``s
  its block (releasing the prefix-source hold). If no other holder
  remains, the block returns to the free pool via PagedKVCache.

**Why two refcount views (``kv._refcount`` vs PC's ``_live_hits``):**

``kv._refcount`` sums every retention source (owning req + PC's prefix role
+ live hits) without distinguishing them. The eviction decision needs a
PC-specific view: "is this block still holding someone's copy-source hit?"
That information is not recoverable from the aggregate. Unifying them
would be a premature optimisation that makes eviction undecidable.
"""

from __future__ import annotations

from collections.abc import Sequence

from silica.kvcache.manager import PrefixHit
from silica.kvcache.paged import PagedKVCache

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
        kv: PagedKVCache,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if kv.block_size != block_size:
            raise ValueError(
                f"block_size mismatch: pc={block_size} vs kv={kv.block_size}"
            )
        self._block_size = block_size
        self._kv = kv
        self._root = _Node(parent=None, tokens=(), block_id=-1)
        self._tick = 0
        # Per-block live-hit count (independent of kv.refcount — see module doc).
        self._live_hits: dict[int, int] = {}
        self.hits: int = 0

    # --- public surface ---

    def lookup(self, tokens: Sequence[int]) -> PrefixHit:
        """Walk the tree, return the longest block-aligned prefix match.

        For each hit block: incref in ``kv`` and increment local
        ``_live_hits``. Caller is responsible for the paired ``release``
        once the live request has consumed (copied) the hit data.

        Non-block-aligned trailing tokens do not contribute — a partial
        trailing block is not retainable under this scheme.
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
            self._kv.incref(child.block_id)
            self._live_hits[child.block_id] = (
                self._live_hits.get(child.block_id, 0) + 1
            )
            self._touch(child)
            node = child
            idx += self._block_size
        if hit_blocks:
            self.hits += 1
        return PrefixHit(block_ids=tuple(hit_blocks), num_hit_tokens=idx)

    def insert(
        self, tokens: Sequence[int], block_ids: Sequence[int]
    ) -> None:
        """Add the caller's blocks to the tree at block-aligned prefix points.

        Partial trailing blocks (tokens not aligned to ``block_size``) are
        discarded — only whole blocks can be copy sources under option B.

        Idempotent on already-covered prefixes: if the tree already contains
        a node for a given (parent, chunk) pair, the new block id is **not**
        added to the tree. The caller's block stays with its owning
        request; the tree continues to use the existing block as source.

        For each newly-added node the block is ``kv.incref``'d, pinning it
        against the owning request's eventual ``free``.
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
            self._touch(new_node)
            node.children[chunk] = new_node
            self._kv.incref(block_ids[i])
            node = new_node

    def release(self, block_ids: Sequence[int]) -> None:
        """Release a set of hit blocks (paired with a prior ``lookup``).

        Decrements ``_live_hits`` and ``kv.decref``s each block.

        Raises ``KeyError`` if any block is not currently held as a live
        hit — mismatched release indicates a caller bug (D-011 loud-fail).
        """
        for b in block_ids:
            count = self._live_hits.get(b)
            if not count:
                raise KeyError(
                    f"block {b} has no outstanding live-hit in prefix cache"
                )
            if count == 1:
                del self._live_hits[b]
            else:
                self._live_hits[b] = count - 1
            self._kv.decref(b)

    def evict_until(self, n_blocks: int) -> int:
        """Evict LRU leaf nodes with zero live hits until ``n_blocks`` freed.

        Walks candidate nodes in ascending access order, evicting each
        leaf whose block has no outstanding live hit. Non-leaf nodes and
        nodes whose block is still held by a live request are skipped.

        Returns the actual number of blocks freed (may be < ``n_blocks`` if
        the tree runs out of evictable nodes — the caller must be prepared
        for this partial-return case).
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
        """Current live-hit count for ``block_id`` (0 if none)."""
        return self._live_hits.get(block_id, 0)

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
            if self._live_hits.get(node.block_id, 0) > 0:
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
        self._kv.decref(node.block_id)

    def _walk_non_root(self) -> list[_Node]:
        out: list[_Node] = []
        stack = list(self._root.children.values())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children.values())
        return out
