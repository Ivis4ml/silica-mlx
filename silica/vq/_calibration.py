"""silica.vq._calibration — NumPy quarantine for offline codec calibration.

D-009 boundary: this module is the **only** place under ``silica.vq.*``
where NumPy is permitted on the silica runtime stack. Consumers are codec
``__init__`` bodies (``BlockTurboQuantMSE``, ``TurboQuantMSE``,
``RaBitQ1Bit``, ``ExtRaBitQ`` — P-5-A.1 + P-5-B) that run once at
construction time, convert the returned NumPy arrays into fp16 ``mx.array``
constants stored on the codec instance, and then never reach back into
this module on the hot path.

Runtime enforcement: every other module under ``silica.vq.*``
(``silica.vq.core.packing``, ``silica.vq.block_tq``, ``silica.vq.turboquant``,
``silica.vq.rabitq``) must pass a "no NumPy on the hot path" grep — the
leading underscore on ``_calibration`` and its "calibration" suffix are
the documented exception. See ``plans/P5_OPENING.md`` §4.6.

Contents:

- ``haar_rotation(d, seed)`` — Haar-random orthogonal ``(d, d)`` matrix
  via QR of a Gaussian matrix, sign-fixed per Stewart (1980). The caller
  matmuls fp16 ``mx.array`` tensors against this matrix after uploading
  it as an ``mx.array`` constant on the codec instance. Algorithm matches
  ``vqbench/vqbench/core/rotation.py::haar_rotation`` verbatim so codec-
  level recon-error cross-checks (acceptance (a)) are apples-to-apples.

- ``lloyd_max_codebook(num_bits, sigma)`` — Lloyd-Max optimal scalar
  quantizer for ``N(0, sigma^2)``. Returns ``(centroids, boundaries)``
  as NumPy ``float64`` arrays; caller uploads to fp16 ``mx.array``.
  Algorithm matches ``vqbench/vqbench/methods/turboquant/codebook.py::
  _gaussian_lloyd_max`` verbatim (same iteration schedule, same
  convergence tolerance). The standard-normal CDF / PDF needed inside
  the iteration come from ``math.erf`` and ``math.exp`` rather than
  ``scipy.stats.norm``, so silica does not pick up a scipy dependency
  for one scalar CDF call.

- ``CENTROIDS_SCALED_REF`` — dimension-independent paper reference
  values from TurboQuant (arXiv 2504.19874) Table 1 for ``num_bits in
  {1, 2}``, scaled by ``sqrt(d)``. Used by unit tests to regression-lock
  Lloyd-Max convergence.

No on-disk serialization format in v0.1 (per P-5 opening §4.6): every
``Engine`` re-calibrates on construction. Module-level caches here avoid
the O(d^3) QR and ~200-iteration Lloyd-Max repeating inside a single
process. Codec ``__init__`` still owns the numpy-to-mx.array upload.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

# =============================================================================
# Haar rotation
# =============================================================================

_rotation_cache: dict[tuple[int, int], np.ndarray] = {}


def haar_rotation(d: int, seed: int) -> np.ndarray:
    """Return a Haar-distributed orthogonal matrix in ``R^{d x d}``.

    Args:
        d: dimensionality (must be positive).
        seed: PRNG seed.

    Returns:
        ``np.ndarray`` of shape ``(d, d)``, dtype ``float64``, marked
        read-only (``flags.writeable = False``). Orthogonal:
        ``Q @ Q.T == I`` up to float rounding; ``det(Q) in {-1, +1}``.
        Callers must not mutate the return value — construct an
        ``mx.array`` copy instead.

    Algorithm: Stewart (1980), "The Efficient Generation of Random
    Orthogonal Matrices". Draw ``G ~ N(0, 1)^{d x d}``, QR-decompose
    to ``Q, R``, then correct the sign ambiguity by
    ``Q <- Q @ diag(sign(diag(R)))``. Cached by ``(d, seed)`` — the
    same key returns the same ``np.ndarray`` object.

    Raises:
        ValueError: if ``d < 1``.
    """
    if d < 1:
        raise ValueError(f"d must be positive; got {d}")

    key = (d, seed)
    if key in _rotation_cache:
        return _rotation_cache[key]

    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    Q.flags.writeable = False
    _rotation_cache[key] = Q
    return Q


# =============================================================================
# Lloyd-Max codebook
# =============================================================================

_SQRT_2: Final[float] = math.sqrt(2.0)
_SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)


def _std_normal_cdf(x: float) -> float:
    """Scalar standard-normal CDF, matches ``scipy.stats.norm.cdf``."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _std_normal_pdf(x: float) -> float:
    """Scalar standard-normal PDF, matches ``scipy.stats.norm.pdf``."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


_codebook_cache: dict[
    tuple[int, float, int, float], tuple[np.ndarray, np.ndarray]
] = {}


def lloyd_max_codebook(
    num_bits: int,
    sigma: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max optimal scalar quantizer for ``N(0, sigma^2)``.

    Args:
        num_bits: bits per coordinate; produces ``2 ** num_bits``
            centroids.
        sigma: standard deviation of the target Gaussian.
        max_iter: maximum Lloyd iterations (default 200).
        tol: convergence tolerance on L^inf centroid change (default
            1e-12).

    Returns:
        ``(centroids, boundaries)``:

        - ``centroids``: ``np.ndarray`` shape ``(2 ** num_bits,)``,
          dtype ``float64``, sorted ascending, read-only.
        - ``boundaries``: ``np.ndarray`` shape ``(2 ** num_bits - 1,)``,
          dtype ``float64``, midpoints between adjacent centroids,
          read-only.

    Usage: for ``BlockTurboQuantMSE`` at ``vq_block_size = B, num_bits = b``,
    call ``lloyd_max_codebook(b, sigma=1.0 / math.sqrt(B))``. Scalar
    ``TurboQuantMSE`` is the ``B = head_dim`` case of the same call.

    Cached by ``(num_bits, sigma, max_iter, tol)`` so repeated
    construction reuses the result; first call converges in
    O(max_iter) Gaussian CDF / PDF evaluations (each call is scalar
    stdlib ``math.erf`` / ``math.exp`` — no scipy dependency).

    Raises:
        ValueError: if ``num_bits < 1``, ``sigma`` is not a finite positive
            float, ``max_iter < 1``, or ``tol`` is not a finite positive
            float. Validated before any cache lookup so malformed inputs
            never land in ``_codebook_cache``.
    """
    if num_bits < 1:
        raise ValueError(f"num_bits must be >= 1; got {num_bits}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and positive; got {sigma}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive; got {tol}")

    key = (num_bits, float(sigma), max_iter, tol)
    if key in _codebook_cache:
        return _codebook_cache[key]

    k = 2 ** num_bits
    centroids = np.linspace(-3.0 * sigma, 3.0 * sigma, k)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        new_centroids = np.empty(k)
        edges = np.concatenate(
            ([np.float64(-np.inf)], boundaries, [np.float64(np.inf)])
        )

        for i in range(k):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            lo_s = lo / sigma
            hi_s = hi / sigma
            prob = _std_normal_cdf(hi_s) - _std_normal_cdf(lo_s)
            if prob < 1e-30:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = (
                    sigma * (_std_normal_pdf(lo_s) - _std_normal_pdf(hi_s)) / prob
                )

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    centroids.flags.writeable = False
    boundaries.flags.writeable = False
    _codebook_cache[key] = (centroids, boundaries)
    return centroids, boundaries


# =============================================================================
# Paper reference values
# =============================================================================

def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


CENTROIDS_SCALED_REF: Final[dict[int, np.ndarray]] = {
    # TurboQuant (arXiv 2504.19874), Table 1. Values are centroids * sqrt(d)
    # — the dimension-independent reference for rotated coordinate
    # distribution N(0, 1/d). num_bits in {1, 2} are the only rows the paper
    # pins; higher num_bits values are checked against convergence /
    # symmetry / sort-order invariants in unit tests. Arrays are frozen
    # (writeable=False) so callers cannot mutate the paper references in
    # place; the dict itself is not frozen but is not expected to be
    # written to at runtime.
    1: _frozen(np.array([-0.7979, 0.7979])),
    2: _frozen(np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
}
