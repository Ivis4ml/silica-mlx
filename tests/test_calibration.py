"""Tests for silica.vq._calibration — Haar rotation + Lloyd-Max codebook.

Covers the NumPy-quarantined offline helpers that codec ``__init__``
bodies consume once at construction time (see ``docs/P5_OPENING.md``
§4.6 + §5.2):

- ``haar_rotation``: orthogonality, determinism under ``(d, seed)``,
  cache identity, write-protection, shape + dtype, seed separation.
- ``lloyd_max_codebook``: convergence to paper reference values for
  ``num_bits in {1, 2}`` (TurboQuant Table 1), shape + dtype,
  centroid ordering, boundary = midpoint invariant, symmetry about
  zero (the Gaussian is symmetric), cache identity, error handling.

D-009 boundary: this module is explicitly the NumPy quarantine under
``silica.vq.*``; a separate test asserts that other ``silica.vq.*``
modules do not import NumPy on the hot path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from silica.vq._calibration import (
    CENTROIDS_SCALED_REF,
    haar_rotation,
    lloyd_max_codebook,
)

# =============================================================================
# haar_rotation
# =============================================================================


@pytest.mark.parametrize("d", [8, 16, 64, 128, 256])
def test_haar_rotation_is_orthogonal(d: int) -> None:
    Q = haar_rotation(d, seed=42)
    # Q @ Q.T ≈ I to machine precision
    err = np.max(np.abs(Q @ Q.T - np.eye(d)))
    assert err < 1e-10, f"d={d}: orthogonality error {err:.2e} exceeds 1e-10"


@pytest.mark.parametrize("d", [8, 64, 256])
def test_haar_rotation_shape_and_dtype(d: int) -> None:
    Q = haar_rotation(d, seed=42)
    assert Q.shape == (d, d)
    assert Q.dtype == np.float64


def test_haar_rotation_deterministic_under_same_seed() -> None:
    """Same (d, seed) produces the same matrix."""
    Q1 = haar_rotation(64, seed=42)
    # Call from a fresh context — bypass the cache to verify algorithmic
    # determinism, not just memoization.
    from silica.vq import _calibration as cal

    cal._rotation_cache.pop((64, 42), None)
    Q2 = haar_rotation(64, seed=42)
    assert np.array_equal(Q1, Q2)


def test_haar_rotation_cache_returns_same_object() -> None:
    """Repeated calls with matching (d, seed) return the cached ndarray."""
    Q1 = haar_rotation(128, seed=7)
    Q2 = haar_rotation(128, seed=7)
    assert Q1 is Q2


def test_haar_rotation_different_seeds_differ() -> None:
    Q1 = haar_rotation(64, seed=42)
    Q2 = haar_rotation(64, seed=43)
    assert not np.array_equal(Q1, Q2)


def test_haar_rotation_is_write_protected() -> None:
    """Callers must copy into an mx.array rather than mutating in place."""
    Q = haar_rotation(32, seed=42)
    assert not Q.flags.writeable
    with pytest.raises(ValueError):
        Q[0, 0] = 0.0


def test_haar_rotation_rejects_nonpositive_d() -> None:
    with pytest.raises(ValueError, match="d must be positive"):
        haar_rotation(0, seed=42)
    with pytest.raises(ValueError, match="d must be positive"):
        haar_rotation(-1, seed=42)


# =============================================================================
# lloyd_max_codebook
# =============================================================================


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
def test_lloyd_max_shape_and_dtype(num_bits: int) -> None:
    k = 2 ** num_bits
    centroids, boundaries = lloyd_max_codebook(num_bits, sigma=1.0 / math.sqrt(64))
    assert centroids.shape == (k,)
    assert boundaries.shape == (k - 1,)
    assert centroids.dtype == np.float64
    assert boundaries.dtype == np.float64


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
def test_lloyd_max_centroids_sorted_ascending(num_bits: int) -> None:
    centroids, _ = lloyd_max_codebook(num_bits, sigma=1.0 / math.sqrt(64))
    assert bool(np.all(centroids[1:] > centroids[:-1]))


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
def test_lloyd_max_boundaries_are_midpoints(num_bits: int) -> None:
    """Boundaries must be midpoints between adjacent centroids."""
    centroids, boundaries = lloyd_max_codebook(num_bits, sigma=1.0 / math.sqrt(64))
    expected = (centroids[:-1] + centroids[1:]) / 2.0
    assert np.allclose(boundaries, expected, atol=1e-15)


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
def test_lloyd_max_centroids_symmetric_about_zero(num_bits: int) -> None:
    """``N(0, sigma^2)`` is symmetric → optimal centroids come in
    sign-flipped pairs."""
    centroids, _ = lloyd_max_codebook(num_bits, sigma=1.0 / math.sqrt(64))
    assert np.allclose(centroids, -centroids[::-1], atol=1e-10)


@pytest.mark.parametrize("num_bits", [1, 2])
def test_lloyd_max_matches_paper_reference(num_bits: int) -> None:
    """TurboQuant (arXiv 2504.19874) Table 1 pins centroids × sqrt(d) for
    ``num_bits in {1, 2}`` on the rotated coordinate distribution. With
    ``sigma = 1 / sqrt(d)`` (the BlockTQ ``B = d`` special case), our
    centroids scale to the paper values within ``rtol=0.01``."""
    d = 128
    centroids, _ = lloyd_max_codebook(num_bits, sigma=1.0 / math.sqrt(d))
    scaled = centroids * math.sqrt(d)
    ref = CENTROIDS_SCALED_REF[num_bits]
    assert np.allclose(scaled, ref, rtol=0.01), (
        f"num_bits={num_bits}: centroids * sqrt(d) = {scaled} "
        f"does not match paper ref {ref} within rtol=0.01"
    )


def test_lloyd_max_scales_with_sigma() -> None:
    """Centroids for ``N(0, sigma^2)`` scale linearly with ``sigma``. Compare
    ``sigma_1 = 1/sqrt(64)`` to ``sigma_2 = 2/sqrt(64)``: centroids double."""
    sigma_1 = 1.0 / math.sqrt(64)
    sigma_2 = 2.0 / math.sqrt(64)
    c1, _ = lloyd_max_codebook(4, sigma=sigma_1)
    c2, _ = lloyd_max_codebook(4, sigma=sigma_2)
    assert np.allclose(c2, 2.0 * c1, rtol=1e-6)


def test_lloyd_max_cache_returns_same_tuple() -> None:
    """Repeated calls with matching args return the cached arrays."""
    sigma = 1.0 / math.sqrt(64)
    c1, b1 = lloyd_max_codebook(4, sigma=sigma)
    c2, b2 = lloyd_max_codebook(4, sigma=sigma)
    assert c1 is c2
    assert b1 is b2


def test_lloyd_max_outputs_are_write_protected() -> None:
    centroids, boundaries = lloyd_max_codebook(4, sigma=1.0 / math.sqrt(64))
    assert not centroids.flags.writeable
    assert not boundaries.flags.writeable
    with pytest.raises(ValueError):
        centroids[0] = 0.0
    with pytest.raises(ValueError):
        boundaries[0] = 0.0


def test_lloyd_max_rejects_nonpositive_num_bits() -> None:
    with pytest.raises(ValueError, match="num_bits must be >= 1"):
        lloyd_max_codebook(0, sigma=0.1)
    with pytest.raises(ValueError, match="num_bits must be >= 1"):
        lloyd_max_codebook(-1, sigma=0.1)


def test_lloyd_max_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        lloyd_max_codebook(4, sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        lloyd_max_codebook(4, sigma=-0.5)


def test_lloyd_max_rejects_nonfinite_sigma() -> None:
    """``sigma=nan`` and ``sigma=inf`` would produce all-NaN cached
    codebooks if not rejected up front."""
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        lloyd_max_codebook(4, sigma=float("nan"))
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        lloyd_max_codebook(4, sigma=float("inf"))


def test_lloyd_max_rejects_nonpositive_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        lloyd_max_codebook(4, sigma=0.1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        lloyd_max_codebook(4, sigma=0.1, max_iter=-5)


def test_lloyd_max_rejects_invalid_tol() -> None:
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        lloyd_max_codebook(4, sigma=0.1, tol=0.0)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        lloyd_max_codebook(4, sigma=0.1, tol=-1e-6)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        lloyd_max_codebook(4, sigma=0.1, tol=float("nan"))
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        lloyd_max_codebook(4, sigma=0.1, tol=float("inf"))


def test_lloyd_max_malformed_inputs_do_not_pollute_cache() -> None:
    """Validation runs before the cache lookup, so a rejected input never
    lands in ``_codebook_cache``. Regression-lock the ordering."""
    from silica.vq import _calibration as cal

    cache_before = dict(cal._codebook_cache)
    for bad in [float("nan"), float("inf"), 0.0, -0.1]:
        with pytest.raises(ValueError):
            lloyd_max_codebook(4, sigma=bad)
    assert cal._codebook_cache == cache_before


# =============================================================================
# Paper reference constants are write-protected
# =============================================================================


def test_paper_reference_arrays_are_write_protected() -> None:
    """``CENTROIDS_SCALED_REF`` holds paper constants used for regression
    locking; callers must not be able to mutate the values in place."""
    for num_bits in (1, 2):
        arr = CENTROIDS_SCALED_REF[num_bits]
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 999.0


# =============================================================================
# D-009 quarantine boundary
# =============================================================================


def test_calibration_module_is_the_numpy_quarantine_exception() -> None:
    """``silica.vq._calibration`` is the documented exception that may
    import NumPy; every other module under ``silica.vq.*`` must not touch
    NumPy on the runtime hot path. This test greps every module source
    — including package ``__init__.py`` files — and asserts the boundary.
    """
    import pathlib
    import pkgutil

    import silica.vq as vq

    cal_name = "silica.vq._calibration"
    vq_root = pathlib.Path(vq.__file__).parent

    scanned: list[str] = []
    for pkg_info in pkgutil.walk_packages([str(vq_root)], prefix="silica.vq."):
        if pkg_info.name == cal_name:
            continue  # the allowed quarantine module
        rel = pkg_info.name.removeprefix("silica.vq.").replace(".", "/")
        if pkg_info.ispkg:
            # Subpackage (e.g. silica.vq.core): check the __init__.py, since
            # walk_packages reports the package name but not the init file.
            mod_path = vq_root / rel / "__init__.py"
        else:
            mod_path = vq_root / (rel + ".py")
        if not mod_path.exists():
            continue
        src = mod_path.read_text()
        scanned.append(str(mod_path.relative_to(vq_root)))
        assert "import numpy" not in src, (
            f"{pkg_info.name} ({mod_path}): unexpected ``import numpy`` on "
            f"the runtime hot path. Calibration-time NumPy must live in "
            f"silica.vq._calibration."
        )
        assert "from numpy" not in src, (
            f"{pkg_info.name} ({mod_path}): unexpected ``from numpy`` on "
            f"the runtime hot path."
        )

    # Smoke-check that the walk actually finds something — if pkgutil
    # starts skipping silica.vq.core entirely, this test would silently
    # pass even on real violations.
    assert scanned, "walk_packages found no silica.vq.* modules to scan"
