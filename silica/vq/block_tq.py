"""silica.vq.block_tq — BlockTurboQuantMSE VectorCodec.

P-5-A.1 — MLX-native port of vqbench's production BlockTurboQuantMSE
algorithm (``vqbench/vqbench/methods/turboquant/block_mse.py``). Runs on
the P-4.5-C / P-5-A.0 side-level ``VectorCodec[BlockTQPayload]`` seam:
one instance represents one side (K or V) of the per-layer K/V pair.

Algorithm (per-vector, matching vqbench verbatim except for MLX-native
primitives):

1. Rotate: ``y = rotation @ x`` using a Haar-random orthogonal matrix
   shared across all codec families at the same ``(head_dim, seed)``
   (``silica.vq._calibration.haar_rotation``).
2. Block split: reshape ``y`` from ``(head_dim,)`` to
   ``(num_blocks, vq_block_size)``.
3. Per-block norm extraction: ``scales = ||y_blocks||_2`` along axis 1;
   normalize each block to unit norm.
4. Scalar quantize each coordinate against a block-local Lloyd-Max
   codebook on ``N(0, 1/vq_block_size)``
   (``silica.vq._calibration.lloyd_max_codebook``). MLX lacks
   ``mx.searchsorted``, so this is expressed as an argmin over
   ``|value - centroid|`` — equivalent to boundary lookup at
   Lloyd-Max-optimal midpoints.
5. Pack sub-byte indices via ``silica.vq.core.packing.pack_sub_byte``.

Dequantize is the inverse: unpack indices → lookup centroids → optional
per-block norm correction (matches vqbench's production default) →
rescale by stored per-block fp16 scales → inverse rotate → reshape to
the original input's ``(1, n_kv_heads, kv_block_size, head_dim)`` shape.

Scalar ``TurboQuantMSE`` is the ``vq_block_size = head_dim`` special
case: ``num_blocks = 1``, one fp16 scale per vector, block-local
codebook reduces to the scalar N(0, 1/head_dim) Lloyd-Max from the
TurboQuant paper. ``silica.vq.turboquant.TurboQuantMSE`` aliases this.

D-009: runtime hot path uses ``mx.*`` only. NumPy lives exclusively in
``silica.vq._calibration``; calibration arrays are uploaded to
``mx.array`` at codec ``__init__`` and never re-consulted on encode /
decode. D-012: ``BlockTQPayload.resident_bytes`` equals the sum of
``.nbytes`` across ``packed_indices`` (uint8) + ``scales`` (fp16);
enforced at payload construction.
"""

from __future__ import annotations

import math

import mlx.core as mx

from silica.kvcache.codec import BlockTQPayload
from silica.vq._calibration import haar_rotation, lloyd_max_codebook
from silica.vq.core.packing import pack_sub_byte, unpack_sub_byte


class BlockTurboQuantMSE:
    """Block-wise TurboQuant with per-vq-block fp16 scales.

    Side-level ``VectorCodec[BlockTQPayload]``. One instance quantizes
    one side (K or V) of detached prefix-cache blocks.

    Args:
        block_size: kvcache block size (token axis, shared with the
            containing ``SyntheticPrefixBlockStore``). Codec-agnostic —
            kept here to satisfy the ``VectorCodec`` interface and so
            the store's block-size precondition matches.
        n_kv_heads: KV heads per layer on the target model; needed for
            the ``logical_bytes`` / ``resident_bytes`` per-side
            accounting.
        head_dim: per-head vector dimension ``d``. Must be divisible
            by ``vq_block_size``.
        vq_block_size: per-vq-block size ``B``. ``B = head_dim``
            recovers scalar TurboQuantMSE (one block covers the whole
            vector); ``B ∈ {32, 64}`` is the production range per
            vqbench REPORT §3.
        num_bits: bits per coordinate for the block-local codebook;
            must be in ``{1, 2, 3, 4}`` (matches the P-5 codec catalog
            cap; see ``silica.vq.core.packing``).
        seed: Haar-rotation PRNG seed. Shared across codec families at
            the same ``(head_dim, seed)`` for fairness (the same
            rotation lands under ``TurboQuantMSE``, ``RaBitQ1Bit``, and
            ``ExtRaBitQ`` with matching args).
        norm_correction: if ``True`` (default), renormalize each
            dequantized block to unit norm before rescaling. Matches
            the ``turboquant_plus`` production behaviour and improves
            recon error.
        dtype: output dtype for ``decode_tensor``. D-003 restricts
            this to ``{mx.float16, mx.bfloat16}`` — attention kernels
            (mlx-lm SDPA) operate on fp16 / bf16; fp32 is rejected.
            ``mx.float16`` is the default; ``mx.bfloat16`` is allowed
            for Gemma-style models that carry bf16 K/V.
    """

    # Output dtypes permitted on the D-003 path. fp32 is deliberately
    # rejected — attention kernels (mlx-lm SDPA) operate on fp16 /
    # bfloat16; letting a codec return fp32 would silently promote
    # downstream K/V, doubling active-KV residency and breaking the
    # budgeter's fp16 baseline arithmetic. bfloat16 is left in the set
    # so Gemma-style models (which use bf16 K/V) can install BlockTQ
    # without a dtype widening.
    _ALLOWED_DTYPES: tuple[mx.Dtype, ...] = (mx.float16, mx.bfloat16)

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        vq_block_size: int,
        num_bits: int,
        seed: int = 42,
        norm_correction: bool = True,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0; got {block_size}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0; got {n_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if vq_block_size <= 0 or head_dim % vq_block_size != 0:
            raise ValueError(
                f"vq_block_size={vq_block_size} must be a positive divisor "
                f"of head_dim={head_dim}"
            )
        if num_bits not in (1, 2, 3, 4):
            raise ValueError(
                f"num_bits must be in {{1, 2, 3, 4}}; got {num_bits}"
            )
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a multiple of 8 for sub-byte "
                f"packing (see silica.vq.core.packing)"
            )
        if dtype not in self._ALLOWED_DTYPES:
            raise ValueError(
                f"dtype must be one of {self._ALLOWED_DTYPES} (D-003 — "
                f"decode_tensor output is consumed by fp16 / bf16 SDPA); "
                f"got {dtype}"
            )

        self.block_size = block_size
        self.dtype = dtype

        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._vq_block_size = vq_block_size
        self._num_vq_blocks = head_dim // vq_block_size
        self._num_bits = num_bits
        self._codebook_size = 1 << num_bits  # 2 ** num_bits
        self._norm_correction = norm_correction
        self._seed = seed

        # -- Offline calibration (NumPy + stdlib math — scipy not used;
        #    Lloyd-Max uses math.erf / math.exp. All NumPy is quarantined
        #    to silica.vq._calibration). Upload the outputs as frozen
        #    fp32 mx.array constants held on the codec instance. Runtime
        #    hot path reads only these mx.array fields.
        rotation_np = haar_rotation(head_dim, seed)  # (d, d), float64, read-only
        sigma = 1.0 / math.sqrt(vq_block_size)
        centroids_np, _boundaries_np = lloyd_max_codebook(num_bits, sigma)

        # fp32 for numerical stability on matmul + argmin; fp16 would
        # introduce avoidable quantization noise before Lloyd-Max's
        # boundary decisions. Memory cost is a (d x d) fp32 matrix —
        # for d=256 that is 256 KB; negligible next to the KV cache.
        self._rotation = mx.array(rotation_np, dtype=mx.float32)
        self._centroids = mx.array(centroids_np, dtype=mx.float32)

    # --- VectorCodec surface ---------------------------------------------

    def encode_tensor(self, x: mx.array) -> BlockTQPayload:
        """Quantize ``x`` (one side of one block) into a ``BlockTQPayload``.

        Input shape must be exactly
        ``(1, n_kv_heads, kv_block_size, head_dim)`` — the shape
        ``ContinuousBatcher._extract_and_insert_prefix`` produces and
        ``SyntheticPrefixBlockStore.register_detached`` passes through.
        Leading axes are flattened to ``n_vectors = n_kv_heads ×
        kv_block_size`` so the last-axis quantization is purely
        vector-in / vector-out.

        Raises:
            ValueError: if ``x.shape`` does not match the codec's
                configured ``(1, n_kv_heads, block_size, head_dim)`` or
                if ``x.dtype`` does not match ``self.dtype``. Wrong-
                shape inputs with a coincidentally matching element
                count would silently decode into a different semantic
                layout, corrupting K/V; the strict check catches the
                miswire at encode time.
        """
        expected_shape = (1, self._n_kv_heads, self.block_size, self._head_dim)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"BlockTurboQuantMSE.encode_tensor: input shape "
                f"{tuple(x.shape)} does not match codec's configured "
                f"{expected_shape}. VectorCodec operates on one side "
                f"(K or V) of one detached block; reshape at the caller "
                f"if semantic layout differs."
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"BlockTurboQuantMSE.encode_tensor: input dtype "
                f"{x.dtype} does not match codec dtype {self.dtype}. "
                f"A silent upcast from bf16 to fp16 (or vice versa) "
                f"would leak rounding into the Haar rotation and shift "
                f"the decoded output relative to vqbench reference."
            )

        flat = x.reshape(-1, self._head_dim).astype(mx.float32)
        n_vectors = flat.shape[0]

        # 1. Rotate: y[i] = rotation @ x[i]  ==  flat @ rotation.T
        y = flat @ self._rotation.T  # (n_vectors, head_dim)

        # 2. Block split: (n_vectors, num_blocks, vq_block_size)
        y_blocks = y.reshape(n_vectors, self._num_vq_blocks, self._vq_block_size)

        # 3. Per-block norm extraction + unit-norm rescale.
        scales = mx.linalg.norm(y_blocks, axis=2)  # (n_vectors, num_blocks)
        safe = mx.maximum(scales, mx.array(1e-30, dtype=mx.float32))
        y_normed = y_blocks / safe[..., None]  # (n_vectors, num_blocks, vq_block_size)

        # 4. Scalar quantize via argmin over centroids. MLX lacks
        #    searchsorted; broadcast-and-argmin is equivalent for
        #    Lloyd-Max-optimal codebooks (boundaries are midpoints).
        #    diff: (n_vectors, num_blocks, vq_block_size, codebook_size)
        diff = y_normed[..., None] - self._centroids
        indices_u32 = mx.argmin(diff * diff, axis=-1)  # (n_vectors, num_blocks, vq_block_size)
        indices_u8 = indices_u32.astype(mx.uint8).reshape(n_vectors, self._head_dim)

        # 5. Pack.
        packed = pack_sub_byte(indices_u8, num_bits=self._num_bits)
        scales_fp16 = scales.astype(mx.float16)

        total = int(packed.nbytes) + int(scales_fp16.nbytes)
        return BlockTQPayload(
            resident_bytes=total,
            packed_indices=packed,
            scales=scales_fp16,
        )

    def decode_tensor(self, payload: BlockTQPayload) -> mx.array:
        """Reconstruct fp16 ``(1, n_kv_heads, kv_block_size, head_dim)``
        tensor from a ``BlockTQPayload``. D-003: output is fp16.

        Payload shape contract: ``packed_indices`` must be
        ``(n_kv_heads * block_size, num_bits * head_dim // 8)`` and
        ``scales`` must be ``(n_kv_heads * block_size, num_vq_blocks)``.
        A payload whose shapes disagree with this codec's configuration
        (e.g. round-tripped through a different `(head_dim, num_bits,
        vq_block_size)`) is rejected rather than silently reshaping
        into an incorrect semantic layout. Mirrors the defensive guards
        in ``RaBitQ1Bit.decode_tensor`` / ``ExtRaBitQ.decode_tensor``.
        """
        d = self._head_dim
        expected_n_vectors = self._n_kv_heads * self.block_size
        packed_width = self._num_bits * d // 8
        expected_packed_shape = (expected_n_vectors, packed_width)
        expected_scales_shape = (expected_n_vectors, self._num_vq_blocks)

        packed_shape = tuple(payload.packed_indices.shape)
        if packed_shape != expected_packed_shape:
            raise ValueError(
                f"BlockTurboQuantMSE.decode_tensor: packed_indices shape "
                f"{packed_shape} does not match expected "
                f"{expected_packed_shape} (n_kv_heads={self._n_kv_heads}, "
                f"block_size={self.block_size}, head_dim={d}, "
                f"num_bits={self._num_bits})."
            )
        scales_shape = tuple(payload.scales.shape)
        if scales_shape != expected_scales_shape:
            raise ValueError(
                f"BlockTurboQuantMSE.decode_tensor: scales shape "
                f"{scales_shape} does not match expected "
                f"{expected_scales_shape} (n_kv_heads={self._n_kv_heads}, "
                f"block_size={self.block_size}, "
                f"num_vq_blocks={self._num_vq_blocks})."
            )

        packed = payload.packed_indices
        scales = payload.scales

        n_vectors = packed.shape[0]

        # 1. Unpack indices → (n_vectors, head_dim) uint8.
        indices_u8 = unpack_sub_byte(
            packed, num_bits=self._num_bits, d=self._head_dim
        )
        # Reshape to (n_vectors, num_blocks, vq_block_size).
        indices_grid = indices_u8.reshape(
            n_vectors, self._num_vq_blocks, self._vq_block_size
        )

        # 2. Centroid lookup.
        y_blocks = self._centroids[indices_grid]  # (n_vectors, num_blocks, B)

        # 3. Optional per-block norm correction.
        if self._norm_correction:
            block_norms = mx.linalg.norm(y_blocks, axis=2, keepdims=True)
            block_norms = mx.maximum(
                block_norms, mx.array(1e-30, dtype=mx.float32)
            )
            y_blocks = y_blocks / block_norms

        # 4. Rescale by stored per-block fp16 scales (upcast to fp32 for
        #    the multiply so small scales do not underflow).
        scales_fp32 = scales.astype(mx.float32)  # (n_vectors, num_blocks)
        y_blocks = y_blocks * scales_fp32[..., None]

        # 5. Flatten block axis → (n_vectors, head_dim), inverse rotate,
        #    reshape to original shape, cast to output dtype.
        y = y_blocks.reshape(n_vectors, self._head_dim)
        x_hat = y @ self._rotation  # inverse of ``flat @ rotation.T``
        out = x_hat.astype(self.dtype).reshape(
            1, self._n_kv_heads, self.block_size, self._head_dim
        )
        return out

    def logical_bytes(self, num_tokens: int) -> int:
        """fp16-baseline-equivalent bytes for ``num_tokens`` tokens of
        one side (K or V alone). Matches the baseline ``IdentityCodec``
        arithmetic so the budgeter's compression ratio is honest."""
        return num_tokens * self._n_kv_heads * self._head_dim * self.dtype.size

    def resident_bytes(self, num_blocks: int) -> int:
        """Physical bytes held for ``num_blocks`` blocks of one side.

        Per block: ``n_kv_heads × kv_block_size`` vectors, each storing
        ``ceil(head_dim × num_bits / 8)`` packed uint8 bytes (``==
        num_bits × head_dim // 8`` when ``head_dim % 8 == 0``, which
        is enforced at construction) plus ``num_vq_blocks × 2`` bytes
        of fp16 scales.
        """
        n_vectors_per_block = self._n_kv_heads * self.block_size
        packed_bytes_per_vector = self._num_bits * self._head_dim // 8
        scale_bytes_per_vector = self._num_vq_blocks * 2  # fp16 per vq-block
        bytes_per_block = n_vectors_per_block * (
            packed_bytes_per_vector + scale_bytes_per_vector
        )
        return num_blocks * bytes_per_block
