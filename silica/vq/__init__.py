"""silica.vq — MLX-native VQ codec platform (P-5).

Side-level ``VectorCodec[P]`` implementations. Runtime hot path is
MLX-native (``mx.*``); offline calibration helpers (Haar rotation,
Lloyd-Max codebook) are quarantined to ``silica.vq._calibration``.
"""

from silica.vq.block_tq import BlockTurboQuantMSE
from silica.vq.turboquant import TurboQuantMSE

__all__ = [
    "BlockTurboQuantMSE",
    "TurboQuantMSE",
]
