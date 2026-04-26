"""silica.vq.rabitq.rabitq_ext — ExtRaBitQ VectorCodec (P-5-B.2a).

MLX-native port of vqbench's ExtRaBitQ (Gao et al., arXiv 2409.09913, §3).
Side-level VectorCodec[ExtRaBitQPayload]; one instance covers one side (K
or V) of per-layer prefix-cache blocks. Unlike :class:`RaBitQ1Bit`,
ExtRaBitQ is symmetric (K and V both supported) — the estimator-native
attention path is the same on either side, and multi-bit RaBitQ's
reconstruction MSE is competitive on V.

Algorithm (batch, matching vqbench quantize_batch verbatim modulo
MLX-native primitives):

1. Centroid subtraction: centered = x - c  (c = 0 in P-5-B per §5.3)
2. Per-vector norm: norm_o = ||centered||_2
3. Normalize: o_bar = centered / max(norm_o, 1e-30)
4. Rotate: Y = o_bar @ rotation.T
5. Scale to codebook range:
     sigma = 1 / sqrt(d)
     quant_scale = (2^B - 1) / (3 · sigma)  = (2^B - 1) · sqrt(d) / 3
     scaled = Y · quant_scale
6. Round to nearest odd integer using half-to-even (banker's) rounding
   — matches vqbench's np.round:
     rounded = mx.round((scaled - 1) / 2) · 2 + 1
     rounded = clip(rounded, -(2^B - 1), +(2^B - 1))
7. Integer codebook indices ∈ [0, 2^B - 1]:
     indices = (rounded + (2^B - 1)) / 2
   Pack via pack_sub_byte(indices, num_bits=B).
8. ip_coeff = <x_bar, Y> where x_bar = codebook[indices] / ||codebook[indices]||
   (≥ 0 always, since codebook[indices] shares signs with sign(Y) after
   round-to-odd).
9. Store per-vector fp16 ``scale = 1 / quant_scale`` (the dequantization
   factor). Constant across vectors in v0.1; stored per-vector for
   schema parity with future data-driven variants.

Decode:
1. Unpack indices → codebook[indices] = y_raw (integer-valued fp32).
2. Dequantize: y_hat = y_raw · scale  (stored per-vector scale_inv).
3. Re-normalize: y_hat /= ||y_hat||  (only if ||y_hat|| > 1e-30).
   This restores the unit-norm invariant vqbench's unbiased estimator
   assumes; zero-norm vectors fall through to the centroid directly.
4. Inverse rotate: x_hat = y_hat @ rotation.
5. Scale by norm_o, add centroid, cast to codec.dtype, reshape to
   (1, n_kv_heads, block_size, head_dim).

Zero-vector semantics match vqbench's batch path:
- norm_o = 0, ip_coeff = 0, scale = 1 / quant_scale (the constant).
- Y = 0 → scaled = 0; round((0 - 1) / 2) · 2 + 1 under half-to-even
  = round(-0.5) · 2 + 1 = 0 · 2 + 1 = 1 → rounded = +1 for all coords.
  Indices = (1 + (2^B - 1)) / 2 = 2^(B-1).
- Decode: y_raw = codebook[2^(B-1)] = +1 for all coords. y_hat has
  ||y_hat|| > 0, but norm_o = 0 scales it back to zero.

D-009: hot path is mx.* only; NumPy quarantined to silica.vq._calibration.
D-012: ExtRaBitQPayload.resident_bytes equals sum of array .nbytes across
the four mx.array fields (packed_indices uint8 + norm_o, ip_coeff, scale
all fp16).
D-003: decode output dtype is codec.dtype (fp16 or bf16).
"""

from __future__ import annotations

import math

import mlx.core as mx

from silica.kvcache.codec import ExtRaBitQPayload
from silica.vq._calibration import haar_rotation
from silica.vq.core.packing import pack_sub_byte, unpack_sub_byte


class ExtRaBitQ:
    """ExtRaBitQ multi-bit quantization codec.

    Side-level ``VectorCodec[ExtRaBitQPayload]``. One instance quantizes
    one side (K or V) of detached prefix-cache blocks.

    Paper: Gao et al., arXiv 2409.09913.

    Args:
        block_size: KV-cache block size (token axis). Stored to satisfy
            the VectorCodec interface.
        n_kv_heads: KV heads per layer on the target model.
        head_dim: per-head vector dimension ``d``. Must be a multiple
            of 8 so :func:`pack_sub_byte` can pack the sub-byte indices
            regardless of ``num_bits`` — the pack-helper requires
            ``d % 8 == 0`` on the raw last axis, not on ``d × B``.
        num_bits: bits per coordinate, ``B ∈ {2, 3, 4}``. ``B = 1``
            lives in :class:`RaBitQ1Bit` and is rejected here; ``B ≥ 5``
            exceeds the :mod:`silica.vq.core.packing` codec catalog cap
            (uint8 payload limit).
        seed: Haar-rotation PRNG seed. Matches BlockTQ / RaBitQ1Bit
            conventions so the same rotation matrix is shared. When
            ``per_head_rotation=True`` this becomes the base seed; the
            per-head seed is ``seed * 1000 + head_idx``.
        per_head_rotation: opt-in flag (default ``False``). When
            ``True`` the codec draws ``n_kv_heads`` independent Haar
            rotations and applies one per head, mirroring vqbench's
            ``actual_seed = run_seed * 1000 + head_idx`` convention
            (`vqbench/scripts/variance_qwen35_4b.py:63`). Default OFF
            preserves the shared-rotation baseline.
        dtype: output dtype for :meth:`decode_tensor`; must be
            ``mx.float16`` or ``mx.bfloat16`` (D-003).
    """

    _ALLOWED_DTYPES: tuple[mx.Dtype, ...] = (mx.float16, mx.bfloat16)
    _ALLOWED_NUM_BITS: frozenset[int] = frozenset({2, 3, 4})

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        num_bits: int,
        seed: int = 42,
        per_head_rotation: bool = False,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if num_bits not in self._ALLOWED_NUM_BITS:
            raise ValueError(
                f"ExtRaBitQ supports num_bits in {{2, 3, 4}}; got "
                f"num_bits={num_bits}. For 1-bit RaBitQ use RaBitQ1Bit; "
                f"higher bit widths exceed the uint8-payload cap of "
                f"silica.vq.core.packing."
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0; got {block_size}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0; got {n_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a multiple of 8 for "
                f"sub-byte packing (see silica.vq.core.packing)"
            )
        if dtype not in self._ALLOWED_DTYPES:
            raise ValueError(
                f"dtype must be one of {self._ALLOWED_DTYPES} (D-003 — "
                f"decode_tensor output is consumed by fp16/bf16 SDPA); "
                f"got {dtype}"
            )

        self.block_size = block_size
        self.dtype = dtype

        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._num_bits = num_bits
        self._seed = seed
        self._per_head_rotation = per_head_rotation

        # Explicit zero centroid (P-5-B §5.3). encode subtracts, decode
        # adds, fit is a no-op. Lets the algorithm flow read naturally
        # and matches RaBitQ1Bit's convention.
        self._centroid = mx.zeros((head_dim,), dtype=mx.float32)

        # Haar rotation calibrated once, shared with BlockTQ / RaBitQ1Bit
        # at the same (head_dim, seed). Per-head mode draws
        # ``n_kv_heads`` independent rotations seeded ``seed * 1000 +
        # head_idx`` to match vqbench's per-head convention
        # (vqbench/scripts/variance_qwen35_4b.py:63).
        if per_head_rotation:
            per_head = [
                haar_rotation(head_dim, seed * 1000 + h)
                for h in range(n_kv_heads)
            ]
            self._rotation = mx.stack(
                [mx.array(R, dtype=mx.float32) for R in per_head],
                axis=0,
            )  # (n_kv_heads, d, d)
        else:
            rotation_np = haar_rotation(head_dim, seed)
            self._rotation = mx.array(rotation_np, dtype=mx.float32)  # (d, d)

        # Odd-integer codebook: [-(2^B - 1), -(2^B - 3), ..., -1, +1, ..., +(2^B - 1)]
        # indexed by [0, 2^B - 1]. Paper arXiv 2409.09913 Eq. 3.
        half_range = (1 << num_bits) - 1
        self._codebook_size = 1 << num_bits  # 2^B
        self._half_range = half_range
        codebook = 2 * mx.arange(self._codebook_size, dtype=mx.int32) - half_range
        self._codebook = codebook.astype(mx.float32)  # (2^B,)

        # Quantization scale factor: (2^B - 1) / (3 · sigma) where
        # sigma = 1 / sqrt(d). Constant per codec instance.
        self._quant_scale: float = float(half_range) * math.sqrt(head_dim) / 3.0
        self._scale_inv: float = 1.0 / self._quant_scale
        # Pre-computed uniform fp16 scale tensor constructor uses
        # mx.full at encode time; the constant itself stays Python
        # float so codec-level arithmetic doesn't drag an mx.array
        # through capture-by-reference.

    # -------------------------------------------------------------------------
    # VectorCodec surface
    # -------------------------------------------------------------------------

    def fit(self, X: mx.array) -> None:  # noqa: N803
        """No-op. Centroid = 0 per opening §5.3 for P-5-B.

        The ``X`` argument is accepted for interface parity with vqbench's
        ``VectorQuantizer.fit(X)`` signature and future data-driven
        centroid variants, but is not consumed in v0.1.
        """
        del X

    def encode_tensor(self, x: mx.array) -> ExtRaBitQPayload:
        """Quantize one side of one block into an :class:`ExtRaBitQPayload`.

        Input shape must be exactly
        ``(1, n_kv_heads, block_size, head_dim)``.

        Raises:
            ValueError: if input shape or dtype does not match the
                codec's configured parameters.
        """
        expected_shape = (1, self._n_kv_heads, self.block_size, self._head_dim)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"ExtRaBitQ.encode_tensor: input shape {tuple(x.shape)} "
                f"does not match codec's configured {expected_shape}."
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"ExtRaBitQ.encode_tensor: input dtype {x.dtype} does "
                f"not match codec dtype {self.dtype}."
            )

        d = self._head_dim
        flat = x.reshape(-1, d).astype(mx.float32)
        n_vectors = flat.shape[0]

        # Centroid subtract + per-vector norm + normalize + rotate.
        # Per-head mode reshapes (n_vectors, d) → (n_kv_heads, B, d) and
        # uses batched matmul against the (n_kv_heads, d, d) rotation;
        # the input row order from ``x.reshape(-1, head_dim)`` is head-
        # major so this preserves per-head grouping.
        centered = flat - self._centroid
        norm_o = mx.linalg.norm(centered, axis=1)              # (n_vectors,)
        safe_norm = mx.maximum(norm_o, mx.array(1e-30, dtype=mx.float32))
        o_bar = centered / safe_norm[:, None]                  # (n_vectors, d)
        if self._per_head_rotation:
            o_bar_ph = o_bar.reshape(
                self._n_kv_heads, self.block_size, d
            )
            Y_ph = mx.matmul(o_bar_ph, self._rotation.swapaxes(-1, -2))
            Y = Y_ph.reshape(-1, d)
        else:
            Y = o_bar @ self._rotation.T                       # (n_vectors, d)

        # Scale to codebook range.
        scaled = Y * self._quant_scale                         # (n_vectors, d)

        # Round to nearest odd integer, half-to-even (matches np.round
        # and mx.round semantics; pinned in tests).
        rounded = mx.round((scaled - 1.0) / 2.0) * 2.0 + 1.0
        half_range = float(self._half_range)
        rounded = mx.clip(rounded, -half_range, half_range)

        # Convert to indices ∈ [0, 2^B - 1].
        indices_fp = (rounded + half_range) / 2.0
        indices_u8 = indices_fp.astype(mx.uint8)               # (n_vectors, d)
        packed = pack_sub_byte(indices_u8, num_bits=self._num_bits)

        # ip_coeff per vector: <x_bar, Y> with x_bar = codebook[i] / ||codebook[i]||.
        x_bar_raw = self._codebook[indices_u8.astype(mx.int32)]  # (n_vectors, d)
        x_bar_norm = mx.linalg.norm(x_bar_raw, axis=1, keepdims=True)
        safe_bar_norm = mx.maximum(x_bar_norm, mx.array(1e-30, dtype=mx.float32))
        x_bar = x_bar_raw / safe_bar_norm                      # (n_vectors, d)
        ip_coeff = mx.sum(x_bar * Y, axis=1)                   # (n_vectors,)

        # Per-vector fp16 dequantization scale (constant in v0.1).
        scale_fp16 = mx.full(
            (n_vectors,),
            vals=self._scale_inv,
            dtype=mx.float16,
        )

        norm_o_fp16 = norm_o.astype(mx.float16)
        ip_coeff_fp16 = ip_coeff.astype(mx.float16)

        total_bytes = (
            int(packed.nbytes)
            + int(norm_o_fp16.nbytes)
            + int(ip_coeff_fp16.nbytes)
            + int(scale_fp16.nbytes)
        )
        return ExtRaBitQPayload(
            resident_bytes=total_bytes,
            packed_indices=packed,
            norm_o=norm_o_fp16,
            ip_coeff=ip_coeff_fp16,
            scale=scale_fp16,
        )

    def decode_tensor(self, payload: ExtRaBitQPayload) -> mx.array:
        """Reconstruct ``(1, n_kv_heads, block_size, head_dim)`` tensor.

        Validates payload shapes before decoding; malformed payloads
        (wrong codec configuration, trimmed arrays) raise rather than
        silently reshaping to an incorrect semantic layout.
        """
        d = self._head_dim
        expected_n_vectors = self._n_kv_heads * self.block_size
        packed_width = self._num_bits * d // 8
        expected_packed_shape = (expected_n_vectors, packed_width)
        expected_meta_shape = (expected_n_vectors,)

        packed_shape = tuple(payload.packed_indices.shape)
        if packed_shape != expected_packed_shape:
            raise ValueError(
                f"ExtRaBitQ.decode_tensor: packed_indices shape "
                f"{packed_shape} does not match expected "
                f"{expected_packed_shape} (n_kv_heads={self._n_kv_heads}, "
                f"block_size={self.block_size}, head_dim={d}, "
                f"num_bits={self._num_bits})."
            )
        for field_name, field_value in (
            ("norm_o", payload.norm_o),
            ("ip_coeff", payload.ip_coeff),
            ("scale", payload.scale),
        ):
            actual = tuple(field_value.shape)
            if actual != expected_meta_shape:
                raise ValueError(
                    f"ExtRaBitQ.decode_tensor: {field_name} shape "
                    f"{actual} does not match expected "
                    f"{expected_meta_shape}."
                )

        # Unpack indices, look up integer codebook values, dequantize.
        indices = unpack_sub_byte(
            payload.packed_indices, num_bits=self._num_bits, d=d
        )
        y_raw = self._codebook[indices.astype(mx.int32)]       # (n_vectors, d)
        scale_fp32 = payload.scale.astype(mx.float32)[:, None]  # (n_vectors, 1)
        y_hat = y_raw * scale_fp32                             # (n_vectors, d)

        # Re-normalize to unit norm (unbiased-estimator invariant).
        y_norm = mx.linalg.norm(y_hat, axis=1, keepdims=True)
        safe_y_norm = mx.maximum(y_norm, mx.array(1e-30, dtype=mx.float32))
        y_hat = y_hat / safe_y_norm                            # (n_vectors, d)

        # Inverse rotate + scale by norm_o + add centroid. Per-head
        # mode mirrors the encode reshape.
        if self._per_head_rotation:
            y_hat_ph = y_hat.reshape(
                self._n_kv_heads, self.block_size, d
            )
            x_hat_ph = mx.matmul(y_hat_ph, self._rotation)
            x_hat = x_hat_ph.reshape(-1, d)
        else:
            x_hat = y_hat @ self._rotation                     # (n_vectors, d)
        norm_o_fp32 = payload.norm_o.astype(mx.float32)[:, None]  # (n_vectors, 1)
        x_hat = x_hat * norm_o_fp32 + self._centroid

        return x_hat.astype(self.dtype).reshape(
            1, self._n_kv_heads, self.block_size, d
        )

    def logical_bytes(self, num_tokens: int) -> int:
        """fp16-equivalent bytes for ``num_tokens`` tokens of one side."""
        return num_tokens * self._n_kv_heads * self._head_dim * self.dtype.size

    def resident_bytes(self, num_blocks: int) -> int:
        """Physical bytes for ``num_blocks`` blocks of one side.

        Per vector: ``num_bits · head_dim / 8`` bytes (packed indices) +
        2 bytes fp16 norm_o + 2 bytes fp16 ip_coeff + 2 bytes fp16
        scale = ``num_bits · head_dim / 8 + 6`` bytes. Offset is not
        stored (opening §5.4 amendment).
        """
        n_vectors_per_block = self._n_kv_heads * self.block_size
        packed_bytes = self._num_bits * self._head_dim // 8
        meta_bytes = 6  # norm_o + ip_coeff + scale, all fp16
        return num_blocks * n_vectors_per_block * (packed_bytes + meta_bytes)
