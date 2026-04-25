"""P-3-C5.3.3-het.1 — hybrid heterogeneous extract path.

Today's ``_extract_and_insert_prefix`` reads offset / left_padding
from the wrong layer when the batch cache is heterogeneous (Qwen3.5
DeltaNet+attention interleave: layer 0 is ``ArraysCache``, no
``.offset``). C5.3.3a's synthetic spy adapter unintentionally hid
this because ``_ScriptedAdapter.make_batch_cache`` returns all
``BatchKVCache`` regardless of the declared attention pattern. This
file introduces a *truly* heterogeneous synthetic adapter that
returns a mix of ``ArraysCache`` and ``BatchKVCache`` from
``make_batch_cache`` AND has a model whose ``__call__`` actually
writes K/V to each ``BatchKVCache`` layer.

Acceptance gates:

- ``_token_kv_layer_indices`` returns only ``BatchKVCache`` layer
  positions for hybrid; loud-fails on pure-recurrent stacks.
- ``_extract_and_insert_prefix`` reads offset / left_padding from
  the first attention layer (NOT layer 0), and produces
  ``detached_blocks[b][i]`` indexed by **attention-layer position**,
  where position ``i`` corresponds to source transformer layer
  ``attn_layer_indices[i]``. The user-flagged off-by-one regression
  class (mixing transformer-layer index with attention-layer
  position) is verified by per-layer marker stamps.
- Pure-attention regression: a homogeneous all-``BatchKVCache``
  cache still works exactly as today (every transformer layer is
  an attention layer; ``attn_layer_indices == range(num_layers)``).

Het.2 lands the seed-assembly side (``_admit_single_hit_row``
heterogeneous row_cache build); the C5.3.3b real-model byte-exact
gate test (currently held untracked) becomes runnable after both
lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
)
from silica.models.capabilities import capabilities_from_attention_pattern
from silica.models.recurrent import (
    RecurrentSnapshot,
    _RecurrentLayerEntry,
)
from silica.scheduler.batcher import ContinuousBatcher

BLOCK_SIZE = 4
N_KV = 1
HEAD_DIM = 4
VOCAB = 32

_ScriptStep = int | Sequence[int]


# --- hybrid spy scaffolding ---


class _MockHybridLayer:
    """Stand-in for a real transformer layer.

    The Qwen3.5 adapter introspects ``layer.is_linear`` to distinguish
    DeltaNet from attention layers; the synthetic spy needs the same
    attribute to feed any cross-cutting helper that walks
    ``model.layers``.
    """

    def __init__(self, is_linear: bool) -> None:
        self.is_linear = is_linear


class _ScriptedHybridModel:
    """Model whose ``__call__`` writes per-layer-marker K/V to every
    ``BatchKVCache`` layer in the supplied cache list.

    K is filled with ``layer_idx + 1`` (so layer 0 → 1.0, layer 2 →
    3.0, etc.); V is filled with ``layer_idx + 1.5``. ArraysCache
    layers are skipped — the slice helper's ``snapshot_recurrent_state``
    call captures their state via the spy adapter's marker-stamped
    fake (no real recurrent K/V to read here).
    """

    def __init__(
        self,
        attention_pattern: AttentionPattern,
        script: Sequence[_ScriptStep] = (),
    ) -> None:
        self._pattern = attention_pattern
        self.layers = [
            _MockHybridLayer(
                is_linear=(kind == AttentionKind.HYBRID_DELTANET)
            )
            for kind in attention_pattern.per_layer
        ]
        self.script: list[_ScriptStep] = list(script)
        self.forward_calls = 0

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        self.forward_calls += 1
        B, T = tokens.shape
        if cache is not None:
            for layer_idx, layer_cache in enumerate(cache):
                if not isinstance(layer_cache, BatchKVCache):
                    continue
                marker = float(layer_idx + 1)
                k = mx.full(
                    (B, N_KV, T, HEAD_DIM), marker, dtype=mx.float16
                )
                v = mx.full(
                    (B, N_KV, T, HEAD_DIM), marker + 0.5, dtype=mx.float16
                )
                layer_cache.update_and_fetch(k, v)
        target = self.script.pop(0) if self.script else 0
        return self._logits_for_target(target, B=B, T=T)

    def _logits_for_target(
        self, target: _ScriptStep, *, B: int, T: int
    ) -> mx.array:
        if isinstance(target, int):
            one_hot = mx.zeros((VOCAB,), dtype=mx.float32)
            one_hot[target] = 1.0
            return mx.broadcast_to(
                one_hot.reshape(1, 1, VOCAB), (B, T, VOCAB)
            )
        targets = list(target)
        rows: list[mx.array] = []
        for tok in targets:
            row_hot = mx.zeros((VOCAB,), dtype=mx.float32)
            row_hot[int(tok)] = 1.0
            rows.append(
                mx.broadcast_to(
                    row_hot.reshape(1, VOCAB), (T, VOCAB)
                )
            )
        return mx.stack(rows, axis=0)


class _ScriptedTokenizer:
    pad_token_id = 0


@dataclass
class _SpyLog:
    snapshot_calls: list[dict[str, Any]] = field(default_factory=list)
    next_marker: int = 0


class _ScriptedHybridAdapter:
    """Hybrid synthetic adapter — produces a heterogeneous batch cache
    + implements ``RecurrentStateAdapter`` via marker-stamped fakes.
    """

    def __init__(
        self,
        attention_pattern: AttentionPattern,
        log: _SpyLog,
        script: Sequence[_ScriptStep] = (),
    ) -> None:
        self._pattern = attention_pattern
        self._log = log
        n_layers = len(attention_pattern.per_layer)
        self._n_layers = n_layers
        self._model = _ScriptedHybridModel(attention_pattern, script)
        self._tokenizer = _ScriptedTokenizer()
        self.config = ModelConfig(
            model_name="scripted-hybrid",
            num_layers=n_layers,
            hidden_size=16,
            vocab_size=VOCAB,
        )
        self._kv_layout = KVLayout(
            num_layers=n_layers,
            n_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            dtype=mx.float16,
        )

    # ModelAdapter Protocol surface
    def build(self, weight_provider: Any) -> Any:
        del weight_provider  # spy uses pre-loaded self._model
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def capabilities(self) -> Any:
        return capabilities_from_attention_pattern(self._pattern)

    def tokenizer(self) -> Any:
        return self._tokenizer

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        """Truly heterogeneous: ArraysCache for HYBRID_DELTANET layers,
        BatchKVCache for GLOBAL layers."""
        caches: list[Any] = []
        for kind in self._pattern.per_layer:
            if kind == AttentionKind.HYBRID_DELTANET:
                caches.append(
                    ArraysCache(size=2, left_padding=left_padding)
                )
            else:
                caches.append(BatchKVCache(left_padding=left_padding))
        return caches

    # RecurrentStateAdapter mixin
    def snapshot_recurrent_state(
        self, cache_list: list[Any], row_idx: int
    ) -> RecurrentSnapshot:
        del cache_list  # spy stamps a marker, doesn't read live state
        marker = self._log.next_marker
        self._log.next_marker += 1
        self._log.snapshot_calls.append(
            {"row_idx": row_idx, "marker": marker}
        )
        # One marker-stamped layer entry per HYBRID_DELTANET layer.
        n_recurrent = sum(
            1
            for kind in self._pattern.per_layer
            if kind == AttentionKind.HYBRID_DELTANET
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
                for layer in range(n_recurrent)
            ),
            nbytes=0,
        )

    def restore_recurrent_state(
        self,
        cache_list: list[Any],
        row_idx: int,
        snapshot: RecurrentSnapshot,
    ) -> None:
        del cache_list, row_idx, snapshot  # spy ignores restore inputs

    # ModelAdapter Protocol stubs — never reached (the batcher uses
    # ``self._model`` directly via ``forward_batched``), but mypy
    # requires them for the structural type match.
    def prefill(
        self, tokens: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        del tokens, kv_handle
        raise NotImplementedError

    def decode_step(
        self, token: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        del token, kv_handle
        raise NotImplementedError


# --- helpers ---


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _prefix_cache(block_size: int = BLOCK_SIZE) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def _make_hybrid_batcher(
    pattern: AttentionPattern,
    log: _SpyLog,
    *,
    script: Sequence[_ScriptStep] = (),
) -> ContinuousBatcher:
    adapter = _ScriptedHybridAdapter(pattern, log, script)
    return ContinuousBatcher(
        adapter,
        prefix_cache=_prefix_cache(),
        _allow_recurrent_prefix_cache_for_c5_3_testing=True,
    )


def _read_k_marker(k: mx.array) -> float:
    """Read the marker from a stored K tensor (shape (1, N_KV,
    block_size, HEAD_DIM)). Helper-of-record for the off-by-one
    acceptance — every cell of K was written to the same marker by
    the hybrid spy model, so reading any cell returns it."""
    return float(k[0, 0, 0, 0])


# --- _token_kv_layer_indices ---


class TestTokenKVLayerIndices:
    def test_all_attention_returns_full_range(self) -> None:
        # All-GLOBAL pattern: every layer is BatchKVCache.
        pattern = AttentionPattern(
            per_layer=tuple([AttentionKind.GLOBAL] * 3)
        )
        log = _SpyLog()
        # Pure-attention adapter with no recurrent mixin still goes
        # through this batcher path; use the hybrid adapter scaffold
        # but with all-GLOBAL pattern. Same _token_kv_layer_indices
        # behavior.
        batcher = _make_hybrid_batcher(pattern, log, script=(0,))
        batcher.add_request(0, [1, 2, 3, 4], _params(max_tokens=1))
        batcher.step()  # populate _batch_cache
        assert batcher._token_kv_layer_indices() == [0, 1, 2]

    def test_hybrid_returns_only_attention_positions(self) -> None:
        # 4 layers: DeltaNet, GLOBAL, DeltaNet, GLOBAL.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
            )
        )
        log = _SpyLog()
        batcher = _make_hybrid_batcher(pattern, log, script=(0,))
        batcher.add_request(0, [1, 2, 3, 4], _params(max_tokens=1))
        batcher.step()  # populate _batch_cache
        # Only positions 1 and 3 are attention.
        assert batcher._token_kv_layer_indices() == [1, 3]

    def test_pure_recurrent_loud_fails(self) -> None:
        # All-DeltaNet pattern: zero attention layers. Construct the
        # batcher and step once to populate _batch_cache; then call
        # the helper directly (an actual extract path can't reach
        # here through normal flow because _extract_and_insert_prefix
        # would have been the first caller anyway).
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        batcher = _make_hybrid_batcher(pattern, log, script=(0,))
        batcher.add_request(0, [1, 2, 3, 4], _params(max_tokens=1))
        batcher.step()  # populate _batch_cache (all ArraysCache)
        with pytest.raises(
            RuntimeError, match="no token-K/V layers found"
        ):
            batcher._token_kv_layer_indices()


# --- end-to-end extract path on hybrid cache ---


class TestHybridExtractPathStoresAttentionLayersOnly:
    def test_extract_stores_attn_layers_at_correct_source_indices(
        self,
    ) -> None:
        # 4 layers: GLOBAL, DeltaNet, GLOBAL, DeltaNet (attention at
        # source indices 0 and 2). After prefill of 2 * block_size
        # tokens, reclaim's extract slices K/V from the two attention
        # layers and inserts 2 blocks. The spy model writes a
        # per-layer marker into K (layer_idx + 1.0), so
        # detached_blocks[b][0].K should carry marker 1.0 (transformer
        # layer 0) and detached_blocks[b][1].K should carry marker
        # 3.0 (transformer layer 2). This pins the
        # attention-layer-position → source-transformer-layer-index
        # mapping that the user flagged as the off-by-one bug class.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        # Script: 2 prefill slices.
        batcher = _make_hybrid_batcher(pattern, log, script=(0, 0))
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))  # 8 tokens
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # prefill (slice-regime, B=1, hybrid)
        batcher.step()  # reclaim → extract

        pc = batcher._prefix_cache
        assert pc is not None
        # Two blocks should be in the tree.
        assert pc.node_count() == 2
        # Walk the tree and recover the stored detached K/V.
        hit = pc.peek(prompt)
        assert hit.num_hit_tokens == 2 * BLOCK_SIZE
        assert len(hit.block_ids) == 2

        fetched = pc.fetch_detached_blocks(list(hit.block_ids))
        # Each block must carry exactly num_attention_layers = 2 entries.
        for block_idx, per_layer in enumerate(fetched):
            assert len(per_layer) == 2, (
                f"block {block_idx}: expected 2 attention-layer K/V "
                f"entries, got {len(per_layer)}"
            )
            # Position 0 → transformer layer 0 (GLOBAL, marker 1.0).
            k0_marker = _read_k_marker(per_layer[0][0])
            assert k0_marker == 1.0, (
                f"block {block_idx} attn-position 0: expected K "
                f"marker 1.0 (transformer layer 0), got {k0_marker}"
            )
            # Position 1 → transformer layer 2 (GLOBAL, marker 3.0).
            k1_marker = _read_k_marker(per_layer[1][0])
            assert k1_marker == 3.0, (
                f"block {block_idx} attn-position 1: expected K "
                f"marker 3.0 (transformer layer 2), got {k1_marker}"
            )

    def test_pure_attention_extract_unchanged(self) -> None:
        # Regression: all-GLOBAL pattern stores K/V for every layer
        # (attn_layer_indices == range(num_layers)). Markers go 1.0,
        # 2.0, 3.0 for layers 0, 1, 2.
        pattern = AttentionPattern(
            per_layer=tuple([AttentionKind.GLOBAL] * 3)
        )
        log = _SpyLog()
        batcher = _make_hybrid_batcher(pattern, log, script=(0, 0))
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()
        batcher.step()

        pc = batcher._prefix_cache
        assert pc is not None
        hit = pc.peek(prompt)
        fetched = pc.fetch_detached_blocks(list(hit.block_ids))
        for block_idx, per_layer in enumerate(fetched):
            assert len(per_layer) == 3, (
                f"block {block_idx}: pure-attention should store all "
                f"3 layers; got {len(per_layer)}"
            )
            for pos in range(3):
                marker = _read_k_marker(per_layer[pos][0])
                expected = float(pos + 1)
                assert marker == expected, (
                    f"block {block_idx} layer {pos}: expected K "
                    f"marker {expected}, got {marker}"
                )


# --- offset/left_padding read source ---


class TestExtractReadsAuthoritativeFromAttentionLayer:
    def test_offset_and_left_padding_read_from_first_attn_layer(
        self,
    ) -> None:
        # 3 layers: DeltaNet, DeltaNet, GLOBAL. Layer 0 and 1 are
        # ArraysCache (no .offset). The pre-het code would crash on
        # _batch_cache[0].offset; the het.1 fix walks to layer 2 (the
        # first attention layer) for the offset read.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
            )
        )
        log = _SpyLog()
        batcher = _make_hybrid_batcher(pattern, log, script=(0,))
        prompt = list(range(1, BLOCK_SIZE + 1))  # exactly 1 block
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()  # prefill (1 slice forward, captures 1 block)
        # No exception thrown by extract on layer 0 (ArraysCache).
        batcher.step()  # reclaim → extract via attention-layer offset

        pc = batcher._prefix_cache
        assert pc is not None
        assert pc.node_count() == 1
        hit = pc.peek(prompt)
        fetched = pc.fetch_detached_blocks(list(hit.block_ids))
        assert len(fetched) == 1
        # One attention layer at source-transformer-index 2 → marker 3.0.
        assert len(fetched[0]) == 1
        assert _read_k_marker(fetched[0][0][0]) == 3.0
