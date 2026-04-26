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

**VectorCodec hook — side-level K/V dispatch (P-5-A.0.4).**

``SyntheticPrefixBlockStore`` accepts three codec kwargs:

- ``codec``: shorthand that sets both sides to the same ``VectorCodec``
  instance (``k_codec = v_codec = codec``).
- ``k_codec`` / ``v_codec``: explicit side-level codecs; used when K and
  V require different configurations (vqbench's split-codec pattern).

Resolution rules (first match wins; any other combination raises
``ValueError`` at construction):

- All three ``None`` → pass-through store; ``_detached`` holds raw
  ``(K, V)`` ``mx.array`` references with no payload wrapping. Byte-
  for-byte identical to the pre-P-5 ``codec=None`` default.
- ``codec=X``, both side kwargs ``None`` → ``self._k_codec = self._v_codec = X``.
- ``k_codec=K``, ``v_codec=V`` (both non-``None``), ``codec=None`` →
  split dispatch: K side runs through ``K.encode_tensor`` /
  ``K.decode_tensor``, V side through ``V``'s.
- ``codec=X`` combined with any non-``None`` side kwarg → raise.
- Exactly one of ``k_codec`` / ``v_codec`` ``None`` → raise (both or
  neither for the split form; asymmetric configs pass an explicit
  ``IdentityCodec`` on the identity side).

When either codec is supplied, ``register_detached`` calls
``k_codec.encode_tensor(k)`` and ``v_codec.encode_tensor(v)``
independently per layer and stores the pair in a private
``_DetachedLayer`` dataclass; ``fetch_detached`` calls
``k_codec.decode_tensor`` / ``v_codec.decode_tensor`` to return
shape-preserving fp16 ``(K, V)`` tuples the downstream
``build_seeded_batch_kv`` expects. Either path preserves the external
return shape contract; pass-through and ``IdentityCodec`` produce
byte-identical outputs on the hit path because the codec returns its
input tensor by reference (``RawFp16Payload.t is original``).

Block-size precondition applies uniformly: after resolution, every
effective codec must satisfy ``codec.block_size == store.block_size``.
The shorthand form asserts once; the split form asserts both sides.

The ``PagedPrefixBlockStore`` deliberately does not accept a codec —
Option (A) / (C) codec integration on the active-KV path is excluded
in v0.1 by D-003 (no compressed-domain attention) and Q-009 / R-7
(no MLX variable-length SDPA).

**Pre-norm contract — P-5-F (3b) projection-output capture.**

``SyntheticPrefixBlockStore`` accepts a ``pre_norm: bool = False``
flag at construction. The flag is a contract tag declaring the
semantic shape of the K side of every ``register_detached`` payload
the store will accept; encode / decode behaviour is unchanged.

- ``pre_norm=False`` (default, legacy): K is post-RoPE — the K
  tensor mlx-lm's attention forward writes into the cache after
  ``k_norm`` and ``rope`` have been applied. Reclaim slices K
  directly out of the live cache; admit seeds K directly into a
  freshly-built ``BatchKVCache``. Codec noise enters at chunk-
  boundary RoPE phase, paying an additional cost relative to the
  pre-RoPE injection vqbench's ``_QuantizedProj`` uses (see
  ``docs/P5_D2_INVESTIGATION/README.md`` §Root cause).

- ``pre_norm=True`` (P-5-F default after F.3): K is pre-k_norm —
  the output of ``attn.k_proj(x)`` captured via the
  ``PreNormCaptureAdapter`` Protocol installed at adapter
  construction (``silica/models/pre_norm_capture.py``). The
  ``ContinuousBatcher`` arms / disarms the capture buffer per
  prefill chunk forward; reclaim reads K_pre out of the buffer
  instead of slicing the live cache; admit calls
  ``adapter.apply_k_norm_then_rope`` per block per attention-layer
  position to reconstruct post-RoPE K before seeding the cache.
  V is unchanged across the contract — V never normalises or
  rotates and can flow through either path identically.

The constructor gate at ``ContinuousBatcher.__init__`` raises when
``store.pre_norm=True`` is paired with an adapter that does not
implement ``PreNormCaptureAdapter``; bench oracle paths drive the
adapter Protocol directly and do not consume the store flag, but
the gate keeps the contract honest at the production-deployment
boundary.

The three codec_quality_path arms retained as bench-only opt-ins
(``docs/P5_F_OPENING.md`` §6.9 reading order):

1. ``prefix_store_post_rope`` — the cost of NOT shipping P-5-F.
   K sliced from the live cache after RoPE; chunk-boundary cost
   on top of codec reconstruction error.
2. ``prefix_store_pre_rope`` — F.0b's inverse-RoPE round-trip
   (post-k_norm pre-RoPE space). Useful when isolating codec
   sensitivity to ``k_norm`` from the chunk-boundary RoPE
   mismatch.
3. ``vqbench_aligned`` — D.2a's chunk-grained re-encoding via
   the projection-patch wrapper. Cross-implementation parity
   anchor against vqbench's harness; structurally distinct from
   (3b) (chunk-grained re-encode vs persistent block-grained
   store).

Production rows route through ``prefix_store_pre_norm`` after
F.3 (default in ``_WIKITEXT_PPL_ORACLE_CONFIG``); the three
arms above remain reachable via explicit ``codec_quality_path``
overrides on individual scenarios.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mlx.core as mx

from silica.kvcache.codec import CodedPayload, VectorCodec

if TYPE_CHECKING:
    from silica.kvcache.paged import PagedKVCache


@dataclass(frozen=True, slots=True)
class _DetachedLayer:
    """One layer's K/V payloads (or raw tensors) for one detached block.

    Internal to ``SyntheticPrefixBlockStore`` — not exported. The pass-
    through path (all three codec kwargs ``None`` at store construction)
    holds raw ``mx.array`` references. The codec path holds concrete
    ``CodedPayload`` subclasses produced by the effective side codec.
    """

    k: CodedPayload | mx.array
    v: CodedPayload | mx.array


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
        k_codec: VectorCodec | None = None,
        v_codec: VectorCodec | None = None,
        codec: VectorCodec | None = None,
        pre_norm: bool = False,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        # Rule 1: codec= shorthand cannot combine with split kwargs.
        if codec is not None and (k_codec is not None or v_codec is not None):
            raise ValueError(
                "codec= shorthand cannot be combined with k_codec / v_codec "
                "kwargs; pass codec=X for symmetric K+V, or k_codec=K, "
                "v_codec=V for split"
            )

        # Rule 2: split form is both-or-neither (when codec is None).
        if codec is None and (k_codec is None) != (v_codec is None):
            raise ValueError(
                "k_codec and v_codec must be both provided or both None; "
                "mixed None / non-None is not supported — pass an explicit "
                "IdentityCodec on the identity side for asymmetric configs"
            )

        # Resolve effective side codecs.
        if codec is not None:
            effective_k: VectorCodec | None = codec
            effective_v: VectorCodec | None = codec
        else:
            effective_k = k_codec
            effective_v = v_codec

        # Rule 3: block_size precondition applies to every effective codec.
        if effective_k is not None and effective_k.block_size != block_size:
            raise ValueError(
                f"k_codec.block_size ({effective_k.block_size}) must match "
                f"store.block_size ({block_size})"
            )
        if effective_v is not None and effective_v.block_size != block_size:
            raise ValueError(
                f"v_codec.block_size ({effective_v.block_size}) must match "
                f"store.block_size ({block_size})"
            )

        self.block_size = block_size
        self._k_codec = effective_k
        self._v_codec = effective_v
        # P-5-F F.2a: ``pre_norm`` is a contract tag declaring how the
        # K side of every ``register_detached`` payload was produced.
        # ``False`` (default, legacy) — K is post-RoPE, the cache-
        # extracted shape ``ContinuousBatcher`` slices today. ``True``
        # — K is pre-k_norm, the output of ``attn.k_proj(x)`` captured
        # via ``PreNormCaptureAdapter.install_pre_norm_capture``;
        # consumers fetching K must apply ``adapter.apply_k_norm_then_rope``
        # before seeding the live cache. The store does not encode /
        # decode the K differently — the codec sees the same per-head
        # tensor shape either way — but the flag lets a consumer that
        # reads ``store.pre_norm`` branch on which reconstruction step
        # is required, and ``RadixPrefixCache`` validates the flag
        # against the active adapter's Protocol surface at construction
        # time (an adapter without ``PreNormCaptureAdapter`` cannot
        # operate against a ``pre_norm=True`` store).
        self.pre_norm: bool = pre_norm
        self._next_id: int = 0
        self._source_refs: dict[int, int] = {}
        self._hit_refs: dict[int, int] = {}
        self._detached: dict[int, tuple[_DetachedLayer, ...]] = {}
        # Number of layers per detached block, learned from the first
        # ``register_detached`` call. Subsequent calls assert match.
        # ``resident_bytes_per_block`` uses this to scale one codec
        # round trip's bytes up to the actual eviction unit (one radix
        # block_id covers all layers) — without it the budgeter
        # under-counts bytes freed per eviction by a factor of
        # num_layers and over-rejects.
        self._num_layers: int | None = None

    def _encode_layer(self, k: mx.array, v: mx.array) -> _DetachedLayer:
        """Encode one layer's K and V tensors per the resolved side codecs.

        Pass-through sides (``None`` codec) store the raw tensor by
        reference; codec sides run ``encode_tensor`` and store the
        returned ``CodedPayload``.
        """
        k_payload: CodedPayload | mx.array = (
            k if self._k_codec is None else self._k_codec.encode_tensor(k)
        )
        v_payload: CodedPayload | mx.array = (
            v if self._v_codec is None else self._v_codec.encode_tensor(v)
        )
        return _DetachedLayer(k=k_payload, v=v_payload)

    def _decode_layer(self, layer: _DetachedLayer) -> tuple[mx.array, mx.array]:
        """Reconstruct fp16 K and V tensors for one layer."""
        if isinstance(layer.k, mx.array):
            k_out = layer.k
        else:
            assert self._k_codec is not None, (
                "pass-through K payload stored but k_codec is not None — "
                "invariant violation"
            )
            k_out = self._k_codec.decode_tensor(layer.k)
        if isinstance(layer.v, mx.array):
            v_out = layer.v
        else:
            assert self._v_codec is not None, (
                "pass-through V payload stored but v_codec is not None — "
                "invariant violation"
            )
            v_out = self._v_codec.decode_tensor(layer.v)
        return k_out, v_out

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
        # Track num_layers for resident_bytes_per_block accounting.
        # Homogeneous-shape models (Qwen3-family, P-5-A scope) pass the
        # same num_layers on every call; heterogeneous support is a
        # P-5-A follow-up (opening §2.6). Asserting consistency here
        # surfaces drift early rather than silently returning wrong
        # eviction-bytes numbers to the budgeter.
        n_layers = len(per_layer_kv)
        if self._num_layers is None:
            self._num_layers = n_layers
        elif self._num_layers != n_layers:
            raise ValueError(
                f"block {block_id}: register_detached got {n_layers} "
                f"layers but store was initialized to "
                f"{self._num_layers}; heterogeneous-shape support is "
                f"P-5-A follow-up scope (opening §2.6)"
            )
        self._detached[block_id] = tuple(
            self._encode_layer(k, v) for k, v in per_layer_kv
        )

    def fetch_detached(
        self, block_id: int
    ) -> Sequence[tuple[mx.array, mx.array]]:
        if block_id not in self._detached:
            raise KeyError(
                f"block {block_id}: no detached K/V registered"
            )
        return tuple(self._decode_layer(layer) for layer in self._detached[block_id])

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

    def resident_bytes_per_block(self) -> int | None:
        """Bytes freed by evicting one radix ``block_id`` under the
        currently-bound codecs.

        One radix ``block_id`` covers **all layers** of one token-
        block: ``register_detached(bid, per_layer_kv)`` stores
        ``len(per_layer_kv) == num_layers`` ``_DetachedLayer`` entries
        for that id, and ``RadixPrefixCache.evict_until(n)`` drops all
        of them together on eviction. This method must therefore
        return ``num_layers × (k_codec.resident_bytes(1) +
        v_codec.resident_bytes(1))`` — not just the per-layer figure.
        Returning the single-layer value would cause
        ``MemoryBudgeter.admit`` to under-count bytes freed per
        eviction by a factor of ``num_layers`` and reject requests
        that should fit via single-block eviction (regression-locked
        by ``tests/test_memory_budgeter.py::test_mode_c_admit_evicts_one_block_when_shortfall_between_single_and_all_layer_bytes``).

        ``num_layers`` is learned from the first ``register_detached``
        call; before that (no blocks yet registered) the method
        returns ``None`` — the budgeter's ``_evict_bytes_per_block``
        then falls back to the fp16 baseline, which is harmless
        because no evictable blocks exist at that point anyway.

        Returns ``None`` on the pass-through path (both ``k_codec``
        and ``v_codec`` ``None``). Pass-through stores raw
        ``mx.array`` references whose per-block size the store does
        not track symbolically; callers fall back to their own
        ``bytes_per_token × block_size`` formula (the
        ``MemoryBudgeter`` fp16 baseline).

        Consumed by ``silica.scheduler.budget.MemoryBudgeter`` under
        ``account_prefix_residency=True``, via a structural capability
        check (``hasattr(store, 'resident_bytes_per_block')``). Paged
        backend does not implement this; the budgeter's capability
        check treats absence as "fall back to fp16 baseline".
        """
        if self._k_codec is None or self._v_codec is None:
            return None
        if self._num_layers is None:
            return None
        per_layer = (
            self._k_codec.resident_bytes(1) + self._v_codec.resident_bytes(1)
        )
        return self._num_layers * per_layer

    def resident_bytes(self) -> int:
        """Sum of per-side ``CodedPayload.resident_bytes`` across all detached blocks.

        Parallel observable per P-4.5-C.0 opening doc §6.2 / §8.3 —
        the total resident prefix-cache K/V bytes held by this store.
        Each ``_DetachedLayer`` contributes ``k.resident_bytes +
        v.resident_bytes`` when the sides are codec payloads, or
        ``k.nbytes + v.nbytes`` when they are raw mx.array (pass-through
        path). Under the default pass-through or under ``IdentityCodec``,
        the total equals ``len(live_block_ids()) × num_layers ×
        block_size × (2 × n_kv_heads × head_dim × dtype.size)``. Under a
        future non-identity codec (BlockTQ / RaBitQ, P-5 proper) it
        reflects the compressed residency.

        Intentionally **not** on the ``PrefixBlockStore`` Protocol —
        the paged backend does not track per-layer physical K/V
        residency this way, and adding it to the Protocol would force
        ``PagedPrefixBlockStore`` to choose between a wrong number and
        a ``NotImplementedError`` at the Protocol boundary. Consumers
        that need to read this from any ``PrefixBlockStore`` (e.g. the
        P-5-A.2 ``MemoryBudgeter``) must use a structural capability
        check (``SupportsResidentBytes`` Protocol or a ``hasattr``
        guard) rather than assuming every store exposes it.
        """
        total = 0
        for layers in self._detached.values():
            for layer in layers:
                if isinstance(layer.k, mx.array):
                    total += int(layer.k.nbytes)
                else:
                    total += layer.k.resident_bytes
                if isinstance(layer.v, mx.array):
                    total += int(layer.v.nbytes)
                else:
                    total += layer.v.resident_bytes
        return total


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
