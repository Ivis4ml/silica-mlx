"""silica.vq.core — codec-agnostic helpers shared by every VQ method.

This package holds building blocks that are not specific to any single
codec family (``silica.vq.turboquant``, ``silica.vq.block_tq``,
``silica.vq.rabitq``). Everything under ``silica.vq.core`` is MLX-native
per D-009 — no NumPy on the runtime hot path.
"""

from silica.vq.core.packing import pack_sub_byte, unpack_sub_byte

__all__ = [
    "pack_sub_byte",
    "unpack_sub_byte",
]
