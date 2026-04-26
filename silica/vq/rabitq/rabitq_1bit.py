"""silica.vq.rabitq.rabitq_1bit — RaBitQ 1-bit VectorCodec (P-5-B.1a).

MLX-native port of vqbench's RaBitQ1Bit (arXiv 2405.12497, Algorithm 1).
Side-level VectorCodec[RaBitQPayload]; one instance covers one side (K or V)
of per-layer prefix-cache blocks.

Algorithm (batch, matching vqbench quantize_batch verbatim):

1. Centroid subtraction: O = x - c  (c = 0 in P-5-B per opening §5.3)
2. Per-vector norm: norm_o = ||O||_2
3. Normalize: O_bar = O / max(norm_o, 1e-30)
4. Rotate: Y = O_bar @ rotation.T  (Haar-random P, same convention as BlockTQ)
5. Sign with tie-break: signs = where(Y == 0, +1, sign(Y)) → {-1, +1}
6. Encode: bits = (signs + 1) / 2 ∈ {0, 1}; pack via pack_sub_byte(num_bits=1)
7. ip_coeff = sum(signs * Y, axis=-1) / sqrt(d)  per-vector (≥ 0 always)
8. Store: packed_indices (uint8), norm_o (fp16), ip_coeff (fp16)

Decode:
1. Unpack → bits {0, 1} → signs_fp32 = 2·bits − 1 ∈ {-1, +1}
2. y_hat = signs_fp32 / sqrt(d)
3. x_hat = norm_o · (y_hat @ rotation)  (inverse rotation: Y-space → O-space)
4. Cast to codec.dtype, reshape to (1, n_kv_heads, block_size, head_dim)

Zero-vector semantics match vqbench quantize_batch: ip_coeff = 0, norm_o = 0,
all-ones packed bits. Differs from vqbench's single-vector quantize(), which
returns ip_coeff = RABITQ_EXPECTED_IP_COEFF ≈ 0.7979 for the zero case.

D-009: hot path is mx.* only; NumPy quarantined to silica.vq._calibration.
D-012: RaBitQPayload.resident_bytes equals sum of array .nbytes.
D-003: decode output dtype is codec.dtype (fp16 or bf16).
"""

from __future__ import annotations

import math

import mlx.core as mx

from silica.kvcache.codec import RaBitQPayload
from silica.vq._calibration import haar_rotation
from silica.vq.core.packing import pack_sub_byte, unpack_sub_byte


class RaBitQ1Bit:
    """RaBitQ 1-bit quantization codec (side-level VectorCodec[RaBitQPayload]).

    Paper: Gao & Long, arXiv 2405.12497

    Args:
        block_size: KV-cache block size (token axis). Stored to satisfy the
            VectorCodec interface; not used in the quantization arithmetic.
        n_kv_heads: KV heads per layer on the target model.
        head_dim: per-head vector dimension ``d``. Must be a multiple of 8
            (required by pack_sub_byte for 1-bit packing). P-5-B codec
            catalog entries use head_dim in {64, 128}; both satisfy this.
        num_bits: must be 1. Any other value raises ValueError with message
            "1-bit-only". For multi-bit extended RaBitQ, use ExtRaBitQ.
        seed: Haar-rotation PRNG seed. Matches the convention of
            BlockTurboQuantMSE so the same rotation matrix is shared when
            both codecs are instantiated with the same (head_dim, seed).
            When ``per_head_rotation=True`` this becomes the base seed;
            the per-head seed is ``seed * 1000 + head_idx``.
        per_head_rotation: opt-in flag (default ``False``). When
            ``True`` the codec draws ``n_kv_heads`` independent Haar
            rotations and applies one per head, mirroring vqbench's
            ``actual_seed = run_seed * 1000 + head_idx`` convention
            (`vqbench/scripts/variance_qwen35_4b.py:63`). Default OFF
            preserves the shared-rotation baseline used by the rest of
            the silica codec catalog.
        dtype: output dtype for decode_tensor. Must be mx.float16 or
            mx.bfloat16 (D-003). Default is mx.float16.
    """

    _ALLOWED_DTYPES: tuple[mx.Dtype, ...] = (mx.float16, mx.bfloat16)

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        num_bits: int = 1,
        seed: int = 42,
        per_head_rotation: bool = False,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if num_bits != 1:
            raise ValueError(
                f"RaBitQ1Bit is a 1-bit-only codec; got num_bits={num_bits}. "
                "For multi-bit extended RaBitQ, use ExtRaBitQ."
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0; got {block_size}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0; got {n_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a multiple of 8 for 1-bit "
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
        self._seed = seed
        self._per_head_rotation = per_head_rotation

        # Centroid lives on the codec instance per opening §5.3. P-5-B pins
        # it at zero; fit() keeps it unchanged. Stored explicitly so encode
        # subtract + decode add are visible in the algorithm flow even when
        # the value is a no-op; ExtRaBitQ will reuse the same field.
        self._centroid = mx.zeros((head_dim,), dtype=mx.float32)

        # Offline calibration: Haar rotation float32 mx.array constant.
        # Default mode draws (d, d); per-head mode draws (n_kv_heads, d, d)
        # with seeds ``seed * 1000 + head_idx`` to match vqbench's
        # per-head convention (vqbench/scripts/variance_qwen35_4b.py:63).
        # NumPy is used only inside haar_rotation (D-009 boundary); the
        # result is immediately uploaded to mx.array and never re-consulted
        # on the hot path.
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

    def encode_tensor(self, x: mx.array) -> RaBitQPayload:
        """Quantize one side of one block into a RaBitQPayload.

        Input shape must be exactly ``(1, n_kv_heads, block_size, head_dim)``.

        Raises:
            ValueError: if input shape or dtype does not match the codec's
                configured parameters.
        """
        expected_shape = (1, self._n_kv_heads, self.block_size, self._head_dim)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"RaBitQ1Bit.encode_tensor: input shape {tuple(x.shape)} does "
                f"not match codec's configured {expected_shape}. VectorCodec "
                f"operates on one side (K or V) of one detached block."
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"RaBitQ1Bit.encode_tensor: input dtype {x.dtype} does not "
                f"match codec dtype {self.dtype}."
            )

        # (n_vectors, head_dim) fp32.
        flat = x.reshape(-1, self._head_dim).astype(mx.float32)
        d = self._head_dim

        # Centroid subtraction (broadcast (d,) over (n, d)); zero in P-5-B.
        centered = flat - self._centroid

        # Per-vector L2 norm; safe_norm guards zero-vector division.
        norm_o = mx.linalg.norm(centered, axis=1)              # (n_vectors,)
        safe_norm = mx.maximum(norm_o, mx.array(1e-30, dtype=mx.float32))

        # Normalize then rotate: Y = o_bar @ P^T. Per-head mode reshapes
        # (n_vectors=n_kv_heads*B, d) → (n_kv_heads, B, d) and uses
        # batched matmul against the (n_kv_heads, d, d) rotation; the
        # input row order from ``x.reshape(-1, head_dim)`` is head-major
        # so this reshape preserves the per-head grouping.
        o_bar = centered / safe_norm[:, None]                  # (n_vectors, d)
        if self._per_head_rotation:
            o_bar_ph = o_bar.reshape(
                self._n_kv_heads, self.block_size, self._head_dim
            )
            Y_ph = mx.matmul(o_bar_ph, self._rotation.swapaxes(-1, -2))
            Y = Y_ph.reshape(-1, self._head_dim)
        else:
            Y = o_bar @ self._rotation.T                       # (n_vectors, d)

        # Sign with tie-break 0 → +1 (matches vqbench signs[signs==0] = 1).
        signs_fp32 = mx.where(
            Y == 0.0,
            mx.array(1.0, dtype=mx.float32),
            mx.sign(Y),
        )                                                  # (n_vectors, d)

        # ip_coeff = sum(signs/sqrt(d) * Y) = sum(signs*Y)/sqrt(d).
        # Always >= 0 since sign(y_i)*y_i = |y_i|.
        ip_coeff = mx.sum(signs_fp32 * Y, axis=1) / math.sqrt(d)  # (n_vectors,)

        # Convert signs {-1, +1} → bits {0, 1} → pack 8 bits per uint8 byte.
        # sign = -1 → (-1+1)/2 = 0; sign = +1 → (+1+1)/2 = 1.
        bits = ((signs_fp32 + 1.0) * 0.5).astype(mx.uint8)        # (n_vectors, d)
        packed = pack_sub_byte(bits, num_bits=1)                    # (n_vectors, d//8)

        norm_o_fp16 = norm_o.astype(mx.float16)
        ip_coeff_fp16 = ip_coeff.astype(mx.float16)

        total_bytes = (
            int(packed.nbytes) + int(norm_o_fp16.nbytes) + int(ip_coeff_fp16.nbytes)
        )
        return RaBitQPayload(
            resident_bytes=total_bytes,
            packed_indices=packed,
            norm_o=norm_o_fp16,
            ip_coeff=ip_coeff_fp16,
        )

    def decode_tensor(self, payload: RaBitQPayload) -> mx.array:
        """Reconstruct fp16 ``(1, n_kv_heads, block_size, head_dim)`` tensor.

        D-003: output dtype is self.dtype (fp16 or bf16).
        ip_coeff is stored in the payload but not consumed in v0.1 decode.

        Payload shape contract: ``packed_indices`` must be
        ``(n_kv_heads * block_size, head_dim // 8)``; ``norm_o`` and
        ``ip_coeff`` must be ``(n_kv_heads * block_size,)``. A malformed
        payload (e.g. round-tripped through a different codec
        configuration) is rejected rather than silently reshaping to the
        wrong semantic layout.
        """
        d = self._head_dim
        expected_n_vectors = self._n_kv_heads * self.block_size
        expected_packed_shape = (expected_n_vectors, d // 8)
        expected_meta_shape = (expected_n_vectors,)

        packed_shape = tuple(payload.packed_indices.shape)
        if packed_shape != expected_packed_shape:
            raise ValueError(
                f"RaBitQ1Bit.decode_tensor: packed_indices shape "
                f"{packed_shape} does not match expected "
                f"{expected_packed_shape} (n_kv_heads={self._n_kv_heads}, "
                f"block_size={self.block_size}, head_dim={d})."
            )
        norm_shape = tuple(payload.norm_o.shape)
        if norm_shape != expected_meta_shape:
            raise ValueError(
                f"RaBitQ1Bit.decode_tensor: norm_o shape {norm_shape} "
                f"does not match expected {expected_meta_shape}."
            )
        ip_shape = tuple(payload.ip_coeff.shape)
        if ip_shape != expected_meta_shape:
            raise ValueError(
                f"RaBitQ1Bit.decode_tensor: ip_coeff shape {ip_shape} "
                f"does not match expected {expected_meta_shape}."
            )

        # Unpack → {0, 1} uint8 → float signs {-1, +1}.
        bits = unpack_sub_byte(payload.packed_indices, num_bits=1, d=d)
        signs_fp32 = 2.0 * bits.astype(mx.float32) - 1.0         # (n_vectors, d)

        # y_hat is the approximated rotated-normalized vector in Y-space.
        y_hat = signs_fp32 / math.sqrt(d)                        # (n_vectors, d)

        # Inverse rotation: Y-space → O-space (y_hat @ P recovers O_bar
        # approx). Per-head mode mirrors the encode reshape.
        if self._per_head_rotation:
            y_hat_ph = y_hat.reshape(
                self._n_kv_heads, self.block_size, self._head_dim
            )
            x_hat_ph = mx.matmul(y_hat_ph, self._rotation)
            x_hat = x_hat_ph.reshape(-1, self._head_dim)
        else:
            x_hat = y_hat @ self._rotation                       # (n_vectors, d)

        # Scale by stored per-vector norm, add centroid (no-op for zero centroid).
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

        Per vector: ``head_dim // 8`` bytes (packed sign bits, 1 bit each) +
        2 bytes (fp16 norm_o) + 2 bytes (fp16 ip_coeff) = ``head_dim // 8 + 4``.
        """
        n_vectors_per_block = self._n_kv_heads * self.block_size
        bytes_per_vector = self._head_dim // 8 + 4  # packed + norm_o + ip_coeff
        return num_blocks * n_vectors_per_block * bytes_per_vector
