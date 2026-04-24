"""P-5-D.2a unit tests for
:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_vqbench_aligned`.

Invariants pinned here:

1. **Identity-codec equivalence.** Under a codec_factory that produces
   an identity round trip, the vqbench-aligned path must agree with
   the plain C.1 oracle on the same token stream to fp tolerance —
   the only difference between the two paths is that the former
   installs projection wrappers, so an identity wrapper must not
   alter the numerical output.
2. **Projection wrapper fires on every forward.** A counting codec
   records ``encode_tensor`` + ``decode_tensor`` calls; the expected
   count equals ``num_layers × num_chunks × (2 if wrap_v else 1)``
   per encode/decode side — one pass per K/V projection per forward.
3. **Restoration contract — normal exit.** After a successful call,
   every ``attn.k_proj`` / ``attn.v_proj`` is restored to its
   original identity (``is`` check, not just functional parity).
4. **Restoration contract — exception during forward.** If the
   delegated ``teacher_forced_chunked_nll`` raises, projections are
   still restored (``try/finally``) — leaving them wrapped across
   oracle calls would poison every subsequent forward on the same
   adapter.
5. **wrap_v=False skips the V-side wrapper.** The K-only variant
   installs the wrapper on ``k_proj`` only, ``v_proj`` is untouched.

Uses a tiny fake model with the mlx-lm attention attribute layout
(``.layers[i].self_attn.{k_proj, v_proj}``) so the wrapper
install/restore can be exercised without loading real weights.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from silica.bench.ppl_oracle import (
    _install_vqbench_wrappers,
    _restore_vqbench_wrappers,
    _WrappedProj,
    teacher_forced_chunked_nll,
    teacher_forced_chunked_nll_vqbench_aligned,
)
from silica.kvcache.codec import IdentityCodec, VectorCodec

# =============================================================================
# Shapes
# =============================================================================


_VOCAB = 8
_N_KV_HEADS = 2
_HEAD_DIM = 8
_N_LAYERS = 3


# =============================================================================
# Fake model / adapter
# =============================================================================


class _FakeLinear:
    """Stand-in for ``mlx.nn.Linear`` holding a fixed weight and bias.

    Does not inherit from ``mlx.nn.Module`` on purpose — the wrapper
    replaces ``attn.k_proj`` / ``attn.v_proj`` with a
    :class:`_WrappedProj`, which is also not a Module. Keeping both
    sides off the Module lifecycle avoids parameter-registration
    side effects in this test (the real ``mlx.nn.Linear`` is a
    Module, but that does not matter for this oracle's hot path —
    only ``__call__`` is observed).

    Linear maps ``(B, L, in_features) -> (B, L, out_features)``. For
    the fake model ``in_features = VOCAB`` (pretend-input-embedding
    width) and ``out_features = N_KV_HEADS * HEAD_DIM``.
    """

    def __init__(self, out_features: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._W = mx.array(
            rng.standard_normal((out_features, _VOCAB)).astype(np.float32)
            * 0.1,
            dtype=mx.float16,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, in_features). Output: (B, L, out_features).
        return mx.matmul(x.astype(mx.float16), self._W.T)


class _FakeAttention:
    """Minimal stand-in for a Qwen3 attention module.

    Only ``k_proj`` / ``v_proj`` are needed for the wrapper test — the
    full forward is not exercised here (the D.2a oracle calls
    :func:`teacher_forced_chunked_nll` which drives the model's top-
    level ``__call__``; a fake model's top-level call is free to
    ignore ``k_proj`` / ``v_proj`` entirely).
    """

    def __init__(self, layer_idx: int) -> None:
        out_features = _N_KV_HEADS * _HEAD_DIM
        self.k_proj = _FakeLinear(out_features, seed=layer_idx * 2 + 1)
        self.v_proj = _FakeLinear(out_features, seed=layer_idx * 2 + 2)


class _FakeLayer:
    def __init__(self, layer_idx: int) -> None:
        self.self_attn = _FakeAttention(layer_idx)


class _FakeCache:
    """Offset-tracking stand-in identical to the C.1 test's cache."""

    def __init__(self) -> None:
        self.offset = 0

    def advance(self, n: int) -> None:
        self.offset += n


class _FakeModel:
    """Fake mlx-lm-shaped model.

    - ``.layers`` is a list of :class:`_FakeLayer` so the wrapper
      install path can replace ``layers[i].self_attn.k_proj`` etc.
    - ``__call__(tokens, cache_list)`` produces deterministic
      position-dependent logits. The fake's top-level call does *not*
      invoke ``k_proj`` / ``v_proj`` — a richer fake could chain
      them through a faux attention, but for the install/restore
      tests the wrapper's side effects (call counting, identity
      check) are observable without a full forward path.
    """

    def __init__(self) -> None:
        self.layers = [_FakeLayer(i) for i in range(_N_LAYERS)]

    def __call__(
        self, tokens: mx.array, cache: list[Any]
    ) -> mx.array:
        B, T = tokens.shape
        cache_obj = cache[0]
        offset = cache_obj.offset

        # For the IdentityCodec equivalence test we need a call path
        # where wrapping `k_proj` / `v_proj` has an observable effect
        # on the output. Drive each layer's k_proj + v_proj on the
        # input embedding so the wrapper's identity round trip can be
        # observed in the final logits; a richer fake would chain
        # their outputs through attention + cache updates, but for
        # this test's purpose (lossless round-trip agreement) the
        # k_proj/v_proj output only needs to be *used* somewhere
        # downstream so the identity-factory assertion gets a
        # non-trivial comparison.
        tokens_as_float = tokens.astype(mx.float32) / float(_VOCAB)
        one_hot = mx.zeros((B, T, _VOCAB), dtype=mx.float16)
        one_hot = one_hot + mx.take(
            mx.eye(_VOCAB, dtype=mx.float16), tokens.astype(mx.int32), axis=0
        )  # (B, T, V)

        kv_sum = mx.zeros((B, T), dtype=mx.float32)
        for layer in self.layers:
            k_out = layer.self_attn.k_proj(one_hot)  # (B, T, nkv*hd)
            v_out = layer.self_attn.v_proj(one_hot)
            # Collapse over feature axis so the scalar sum depends on
            # every projected coordinate — makes a lossy codec swing
            # the logits away from the unwrapped reference. Cast to
            # fp32 before summing to avoid fp16 saturation from the
            # codec's decode_tensor output.
            kv_sum = kv_sum + mx.sum(k_out.astype(mx.float32), axis=-1)
            kv_sum = kv_sum + mx.sum(v_out.astype(mx.float32), axis=-1)

        positions = mx.arange(T, dtype=mx.float32) + float(offset)
        v_axis = mx.arange(_VOCAB, dtype=mx.float32)
        pos_scaled = (positions + 1.0) / 32.0
        v_scaled = (v_axis + 1.0) / float(_VOCAB)
        grid = pos_scaled[:, None] * v_scaled[None, :]
        per_pos_logits = mx.cos(grid * math.pi) * 3.0
        logits = mx.broadcast_to(
            per_pos_logits[None, :, :], (B, T, _VOCAB)
        )
        # Couple kv_sum into the logits so the projection path
        # contributes to the NLL; the magnitude is tiny so the
        # signal is observable but the per-token CE is still well-
        # defined. Broadcast (B, T) -> (B, T, V).
        logits = logits + (kv_sum[:, :, None] * 1e-5)
        _ = tokens_as_float  # silence linter on the unused alias

        cache_obj.advance(T)
        return logits


class _FakePPLAdapter:
    """Adapter exposing ``_model`` + ``make_batch_cache([0])`` +
    ``kv_layout()`` for the vqbench-aligned oracle."""

    def __init__(self) -> None:
        self._model = _FakeModel()

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        assert left_padding == [0], (
            f"fake adapter supports [0] left_padding only; got "
            f"{left_padding}"
        )
        return [_FakeCache()]

    def kv_layout(self) -> Any:
        class _Layout:
            n_kv_heads = _N_KV_HEADS
            head_dim = _HEAD_DIM
            num_layers = _N_LAYERS
            dtype = mx.float16

        return _Layout()


# =============================================================================
# Codec factories
# =============================================================================


def _identity_codec_factory(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
    seed: int = 42,
) -> VectorCodec:
    # ``seed`` is ignored — IdentityCodec has no PRNG dependency.
    del seed
    return IdentityCodec(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )


class _CountingCodec:
    """Wraps :class:`IdentityCodec` and counts encode/decode calls.

    Mirrors the C.2 counting-codec pattern but operates at the
    per-call granularity instead of per-block.
    """

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self._inner = IdentityCodec(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.block_size = block_size
        self.dtype = dtype
        self.encode_calls = 0
        self.decode_calls = 0

    def encode_tensor(self, x: mx.array) -> Any:
        self.encode_calls += 1
        return self._inner.encode_tensor(x)

    def decode_tensor(self, payload: Any) -> mx.array:
        self.decode_calls += 1
        return self._inner.decode_tensor(payload)

    def logical_bytes(self, num_tokens: int) -> int:
        return self._inner.logical_bytes(num_tokens)

    def resident_bytes(self, num_blocks: int) -> int:
        return self._inner.resident_bytes(num_blocks)


def _make_counting_factory() -> tuple[Any, list[_CountingCodec]]:
    """Return (factory, captured_codecs). The factory appends every
    produced codec to the list so the test can inspect per-projection
    call counts after the run."""

    captured: list[_CountingCodec] = []

    def factory(
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
        seed: int = 42,
    ) -> _CountingCodec:
        del seed
        codec = _CountingCodec(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        captured.append(codec)
        return codec

    return factory, captured


# =============================================================================
# Fixtures
# =============================================================================


def _make_tokens(seq_len: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, _VOCAB, size=(1, seq_len), dtype=np.int32)
    return mx.array(ids)


# =============================================================================
# Tests
# =============================================================================


class TestIdentityEquivalence:
    """The vqbench-aligned path with an identity factory must reproduce
    the plain C.1 oracle's NLL to fp tolerance."""

    @pytest.mark.parametrize("chunk_size", [16, 32, 64])
    def test_matches_c1_baseline(self, chunk_size: int) -> None:
        adapter_unwrapped = _FakePPLAdapter()
        adapter_wrapped = _FakePPLAdapter()
        # Build the fake model's projections with the same RNG seed
        # on both adapters so the reference and the wrapped run see
        # identical weights.
        tokens = _make_tokens(seq_len=128, seed=1)

        nll_ref, n_ref = teacher_forced_chunked_nll(
            adapter_unwrapped, tokens, chunk_size=chunk_size
        )
        nll_wrapped, n_wrapped = teacher_forced_chunked_nll_vqbench_aligned(
            adapter_wrapped,
            tokens,
            chunk_size=chunk_size,
            codec_factory=_identity_codec_factory,
            seed=42,
            wrap_v=True,
        )

        assert n_ref == n_wrapped
        # The two paths run through different codec instances that
        # both short-circuit to the input tensor, but the reshape /
        # transpose round trip inside ``_WrappedProj`` goes through
        # fp16 and can drop a few bits on the low end. A tight
        # relative tolerance catches any substantive difference.
        assert math.isclose(nll_ref, nll_wrapped, rel_tol=1e-4, abs_tol=1e-3), (
            f"identity wrapper should not perturb NLL: "
            f"C.1 baseline={nll_ref}, vqbench-aligned={nll_wrapped}"
        )


class TestCodecHotPath:
    """The wrapped projections must call ``encode_tensor`` /
    ``decode_tensor`` on every forward (= every chunk)."""

    def test_encode_decode_called_per_layer_per_chunk(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(seq_len=64, seed=2)
        chunk_size = 32
        expected_chunks = 64 // chunk_size  # 2

        factory, captured = _make_counting_factory()
        teacher_forced_chunked_nll_vqbench_aligned(
            adapter,
            tokens,
            chunk_size=chunk_size,
            codec_factory=factory,
            seed=42,
            wrap_v=True,
        )

        # Two projections per layer (k + v) × N_LAYERS. Each wrapper
        # builds one codec per observed L; a clean-multiple run
        # observes exactly one L, so each wrapper yields exactly one
        # codec.
        expected_codecs = _N_LAYERS * 2
        assert len(captured) == expected_codecs, (
            f"expected {expected_codecs} codec instances (one per "
            f"projection across {_N_LAYERS} layers × 2 sides), got "
            f"{len(captured)}"
        )
        # Every codec was called ``expected_chunks`` times (one
        # forward call per chunk, one quantize + one dequantize pass
        # per forward).
        for codec in captured:
            assert codec.encode_calls == expected_chunks, (
                f"encode_calls mismatch: expected {expected_chunks}, "
                f"got {codec.encode_calls}"
            )
            assert codec.decode_calls == expected_chunks, (
                f"decode_calls mismatch: expected {expected_chunks}, "
                f"got {codec.decode_calls}"
            )


class TestRestorationContract:
    """Projections must be restored after normal exit AND after an
    exception inside the delegated C.1 call."""

    def test_restored_on_normal_exit(self) -> None:
        adapter = _FakePPLAdapter()
        originals = [
            (layer.self_attn.k_proj, layer.self_attn.v_proj)
            for layer in adapter._model.layers
        ]
        tokens = _make_tokens(seq_len=32, seed=3)

        teacher_forced_chunked_nll_vqbench_aligned(
            adapter,
            tokens,
            chunk_size=16,
            codec_factory=_identity_codec_factory,
            seed=42,
            wrap_v=True,
        )

        for i, layer in enumerate(adapter._model.layers):
            assert layer.self_attn.k_proj is originals[i][0], (
                f"layer {i} k_proj not restored after normal exit"
            )
            assert layer.self_attn.v_proj is originals[i][1], (
                f"layer {i} v_proj not restored after normal exit"
            )

    def test_restored_on_exception(self) -> None:
        adapter = _FakePPLAdapter()
        originals = [
            (layer.self_attn.k_proj, layer.self_attn.v_proj)
            for layer in adapter._model.layers
        ]

        class _ExplodingCodec:
            """Raises on the first encode — simulates a codec that
            fails mid-forward."""

            block_size = 16
            dtype = mx.float16

            def encode_tensor(self, x: mx.array) -> Any:
                raise RuntimeError("synthetic codec failure")

            def decode_tensor(self, payload: Any) -> mx.array:
                raise RuntimeError("unreachable")

            def logical_bytes(self, num_tokens: int) -> int:
                return 0

            def resident_bytes(self, num_blocks: int) -> int:
                return 0

        def exploding_factory(**_: Any) -> _ExplodingCodec:
            return _ExplodingCodec()

        tokens = _make_tokens(seq_len=32, seed=4)

        with pytest.raises(RuntimeError, match="synthetic codec failure"):
            teacher_forced_chunked_nll_vqbench_aligned(
                adapter,
                tokens,
                chunk_size=16,
                codec_factory=exploding_factory,
                seed=42,
                wrap_v=True,
            )

        for i, layer in enumerate(adapter._model.layers):
            assert layer.self_attn.k_proj is originals[i][0], (
                f"layer {i} k_proj not restored after exception"
            )
            assert layer.self_attn.v_proj is originals[i][1], (
                f"layer {i} v_proj not restored after exception"
            )


class TestWrapVFlag:
    """``wrap_v=False`` must leave ``v_proj`` untouched (K-only
    ablation). Not used by the current block_tq row — reserved for
    future K-only scenarios (RaBitQ1Bit)."""

    def test_wrap_v_false_leaves_v_proj_unwrapped(self) -> None:
        adapter = _FakePPLAdapter()
        originals = [
            (layer.self_attn.k_proj, layer.self_attn.v_proj)
            for layer in adapter._model.layers
        ]

        restorations = _install_vqbench_wrappers(
            adapter._model,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            factory=_identity_codec_factory,
            seed=42,
            dtype=mx.float16,
            wrap_v=False,
        )
        try:
            for i, layer in enumerate(adapter._model.layers):
                assert isinstance(layer.self_attn.k_proj, _WrappedProj), (
                    f"layer {i} k_proj should be wrapped"
                )
                assert layer.self_attn.v_proj is originals[i][1], (
                    f"layer {i} v_proj should NOT be wrapped under "
                    f"wrap_v=False"
                )
        finally:
            _restore_vqbench_wrappers(restorations)

        # All projections must be back to their originals after
        # restoration too.
        for i, layer in enumerate(adapter._model.layers):
            assert layer.self_attn.k_proj is originals[i][0]
            assert layer.self_attn.v_proj is originals[i][1]
