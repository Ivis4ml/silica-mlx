"""silica.vq.core.packing — sub-byte bit-plane packing for MLX-native codecs.

Shared across ``silica.vq.turboquant.TurboQuantMSE``,
``silica.vq.block_tq.BlockTurboQuantMSE``, ``silica.vq.rabitq.RaBitQ1Bit``,
``silica.vq.rabitq.ExtRaBitQ`` (P-5-A.1 + P-5-B). One implementation, pure
MLX — no NumPy on the runtime hot path (D-009).

Layout: bit-plane. For ``b`` bits per value and ``d`` values per row
(``d % 8 == 0``):

1. For each bit position ``k`` in ``[0, b)``: extract the ``k``-th bit of
   every value and pack the resulting ``d`` bits into ``d // 8`` uint8
   bytes, MSB first within each byte (bit of value 0 lands in bit 7 of
   byte 0).
2. Concatenate the ``b`` planes along the last axis.

Output last-axis size = ``b × (d // 8) = (d × b) // 8``, equal to the
``ceil(d × b / 8)`` arithmetic in the payload schemas
(``BlockTQPayload`` / ``RaBitQPayload``) when ``d`` is a multiple of 8.

Constraints:

- ``num_bits`` in ``{1, 2, 3, 4}`` — the P-5 codec catalog caps at 4 bits
  per coordinate (opening §5 / §4.2). Extending beyond 4 would require a
  uint16 payload refactor; not triggered in v0.1.
- ``indices.dtype == mx.uint8``; caller is responsible for ensuring values
  are in ``[0, 2 ** num_bits)``. Pack inspects only the low ``num_bits``
  bits via ``mx.bitwise_and``.
- ``d = indices.shape[-1]`` must be a multiple of 8. ``head_dim`` in the
  P-5 codec catalog is 64 or 256; both satisfy this.

D-003 boundary: packing / unpacking stays uint8 throughout; the fp16
tensor these bytes encode is reconstructed by the caller (codec
``decode_tensor``), which is what mlx-lm SDPA consumes.
"""

from __future__ import annotations

import mlx.core as mx

# Shift amounts for MSB-first bit packing inside one byte: bit of value 0
# occupies bit 7 of the byte, bit of value 7 occupies bit 0. Module-level
# constant — MLX's lazy evaluation keeps this a single array allocation
# reused across every pack / unpack call.
_SHIFTS_MSB_FIRST = mx.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=mx.uint8)


def pack_sub_byte(indices: mx.array, num_bits: int) -> mx.array:
    """Bit-plane pack sub-byte indices into ``uint8`` bytes.

    Args:
        indices: ``uint8`` array, shape ``(..., d)``. Each element is
            interpreted modulo ``2 ** num_bits`` (only the low
            ``num_bits`` bits are used). ``d`` must be a multiple of 8.
        num_bits: bits per value. Must be in ``{1, 2, 3, 4}``.

    Returns:
        ``uint8`` array, shape ``(..., num_bits * d // 8)``. Each leading-
        axis configuration is packed independently; last axis carries
        ``num_bits`` concatenated bit-planes of ``d // 8`` bytes each.

    Raises:
        ValueError: if ``num_bits`` is outside ``{1, 2, 3, 4}`` or ``d``
            is not a multiple of 8.
        TypeError: if ``indices.dtype`` is not ``mx.uint8``.
    """
    if num_bits not in (1, 2, 3, 4):
        raise ValueError(f"num_bits must be in {{1, 2, 3, 4}}; got {num_bits}")
    if indices.dtype != mx.uint8:
        raise TypeError(f"indices must be uint8; got {indices.dtype}")

    shape = tuple(indices.shape)
    if len(shape) < 1:
        raise ValueError("indices must have at least one axis")
    d = shape[-1]
    if d % 8 != 0:
        raise ValueError(
            f"last-axis size {d} must be a multiple of 8 for bit-plane packing"
        )

    leading = shape[:-1]
    one_u8 = mx.array(1, dtype=mx.uint8)

    planes = []
    for k in range(num_bits):
        shift_u8 = mx.array(k, dtype=mx.uint8)
        bit_plane = mx.bitwise_and(mx.right_shift(indices, shift_u8), one_u8)
        # (..., d) -> (..., d // 8, 8); each 8-group becomes one byte.
        grouped = bit_plane.reshape(*leading, d // 8, 8)
        # Place bit j at position (7 - j) within the byte.
        shifted = mx.left_shift(grouped, _SHIFTS_MSB_FIRST)
        # Bits are non-overlapping across the 8-axis, so sum == bitwise OR.
        # mx.sum promotes uint8 -> uint32; cast back to uint8 (values <= 255).
        packed_plane = mx.sum(shifted, axis=-1).astype(mx.uint8)
        planes.append(packed_plane)

    return mx.concatenate(planes, axis=-1)


def unpack_sub_byte(packed: mx.array, num_bits: int, d: int) -> mx.array:
    """Inverse of :func:`pack_sub_byte`. Returns ``uint8`` indices in
    ``[0, 2 ** num_bits)``.

    Args:
        packed: ``uint8`` array, shape ``(..., num_bits * d // 8)``, as
            produced by :func:`pack_sub_byte`.
        num_bits: bits per value; must match the value used to pack.
        d: the original last-axis size before packing; must be a multiple
            of 8.

    Returns:
        ``uint8`` array, shape ``(..., d)``, values in ``[0, 2 ** num_bits)``.

    Raises:
        ValueError: if ``num_bits`` is outside ``{1, 2, 3, 4}``, ``d`` is
            not a multiple of 8, or ``packed``'s last-axis size does not
            match ``num_bits * d // 8``.
        TypeError: if ``packed.dtype`` is not ``mx.uint8``.
    """
    if num_bits not in (1, 2, 3, 4):
        raise ValueError(f"num_bits must be in {{1, 2, 3, 4}}; got {num_bits}")
    if packed.dtype != mx.uint8:
        raise TypeError(f"packed must be uint8; got {packed.dtype}")
    if d % 8 != 0:
        raise ValueError(f"d={d} must be a multiple of 8 for bit-plane unpacking")

    shape = tuple(packed.shape)
    if len(shape) < 1:
        raise ValueError("packed must have at least one axis")
    total_bytes = shape[-1]
    expected = num_bits * d // 8
    if total_bytes != expected:
        raise ValueError(
            f"packed last-axis size {total_bytes} must equal num_bits * d // 8 = {expected}"
        )

    leading = shape[:-1]
    bytes_per_plane = d // 8
    one_u8 = mx.array(1, dtype=mx.uint8)

    result = mx.zeros((*leading, d), dtype=mx.uint8)
    for k in range(num_bits):
        lo = k * bytes_per_plane
        hi = lo + bytes_per_plane
        plane_packed = packed[..., lo:hi]  # (..., d // 8)
        # (..., d // 8) -> (..., d // 8, 1) for broadcast over the 8 shifts.
        expanded = plane_packed.reshape(*leading, bytes_per_plane, 1)
        bits_per_byte = mx.bitwise_and(
            mx.right_shift(expanded, _SHIFTS_MSB_FIRST), one_u8
        )  # (..., d // 8, 8)
        plane_bits = bits_per_byte.reshape(*leading, d)  # (..., d)
        # Shift each bit back to position k in the value, then OR into result.
        shift_u8 = mx.array(k, dtype=mx.uint8)
        result = mx.bitwise_or(result, mx.left_shift(plane_bits, shift_u8))

    return result
