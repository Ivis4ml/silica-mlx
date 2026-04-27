"""P-5-A.1c — algorithmic-parity gate: MLX BlockTQ vs NumPy reference.

Closes P-5 §7(a-algo), the synthetic-Gaussian half of opening §7(a).
The real-activation half — (a-real) — closed separately at v1.7.5
in ``tests/test_block_tq_real_activation_xcheck.py``, which reuses
this file's ``_numpy_block_tq_round_trip`` reference and runs it on
pre-RoPE K / V activations extracted from a Qwen3.5-0.8B prefill
pass (the v1.5.1 "vqbench venv subprocess" design was superseded
by the inline-NumPy idiom — see ``plans/P5_A_REAL_OPENING.md`` §2.3).

What this test does:

- Builds a NumPy reference implementation of BlockTurboQuantMSE's encode
  / decode pipeline (``_numpy_block_tq_round_trip`` below). The
  reference is a verbatim transcription of
  ``vqbench/vqbench/methods/turboquant/block_mse.py``'s algorithm, but
  all arithmetic is reimplemented from scratch with ``numpy.*`` rather
  than imported from vqbench (vqbench's transitive ``scipy.stats``
  dependency is outside silica's venv).
- Shares ``silica.vq._calibration.haar_rotation`` and
  ``lloyd_max_codebook`` with the silica MLX codec — this is the
  **only** cross-talk between reference and codec, which is safe
  because those helpers are verbatim NumPy ports of vqbench's own
  calibration (paper-reference-pinned at
  ``tests/test_calibration.py::test_lloyd_max_matches_paper_reference``).
  Hot-path arithmetic (matmul, block split, scale extract, quantize,
  centroid lookup, norm correction, inverse rotate) is independent.
- Drives both paths with the same fp16 Gaussian input tensor,
  deterministically seeded, across ``(vq_block_size, num_bits) ∈
  {32, 64} × {3, 4}``.
- Compares per-block relative Frobenius error between silica MLX
  output and NumPy reference output. Under fp32 MLX matmul vs fp64
  NumPy matmul + fp16 scale round-trip on both sides, observed error
  is 2.0-2.1e-4 on BlockTQ paths and 1.2e-3 on the scalar
  ``num_vq_blocks=1`` path (measured 2026-04-22). Tolerance 5e-3
  gives ~4× headroom over scalar and ~25× over BlockTQ, which catches
  reshape / rotation-direction / argmin-vs-searchsorted tie-break
  drift while absorbing seed-to-seed variation. A tighter 1e-3
  tolerance regression-locks the production-recommended config
  (B=64 b=4) specifically.

What this test does NOT do:

- Run the full vqbench NumPy codec in-process. That would need scipy.
- Load a real model. Real-activation Frobenius xcheck lives in
  ``tests/test_block_tq_real_activation_xcheck.py`` (v1.7.5 close of
  §7(a-real)); it reuses this file's ``_numpy_block_tq_round_trip``
  reference on real Qwen3.5-0.8B K / V activations.
- Compare PPL. That's Acceptance (4-b) — closed at v1.7.3 via the
  D.2a vqbench-aligned oracle (mean-over-seeds gate on
  ``qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned``).
  The (b-static) static Qwen3.5-4B PPL vs ``vqbench/REPORT.md``
  baseline remains post-P-5 backlog (blocked on P-3-C5 recurrent-
  state snapshot or the monkey-patch route; see PLAN §7 P-5 Notes).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from silica.vq import BlockTurboQuantMSE
from silica.vq._calibration import haar_rotation, lloyd_max_codebook


def _numpy_block_tq_round_trip(
    x: np.ndarray,
    *,
    rotation: np.ndarray,
    centroids: np.ndarray,
    boundaries: np.ndarray,
    vq_block_size: int,
    norm_correction: bool = True,
) -> np.ndarray:
    """NumPy reference implementation of BlockTurboQuantMSE encode+decode.

    Mirrors ``vqbench/vqbench/methods/turboquant/block_mse.py``'s
    batch path verbatim, but written against NumPy only (no vqbench
    import needed) and taking calibration outputs as explicit inputs so
    the reference shares nothing with silica's codec beyond the
    calibration helpers themselves.

    Args:
        x: ``(n_vectors, head_dim)`` float64 input.
        rotation: ``(head_dim, head_dim)`` float64 Haar matrix.
        centroids: ``(2 ** num_bits,)`` float64, sorted ascending.
        boundaries: ``(2 ** num_bits - 1,)`` float64, midpoints between
            adjacent centroids. Used for ``np.searchsorted`` quantize.
        vq_block_size: per-vq-block size ``B``.
        norm_correction: if ``True`` (default, matches silica + vqbench
            production), renormalize each dequantized block to unit
            norm before rescaling.

    Returns:
        ``(n_vectors, head_dim)`` float64 reconstruction.
    """
    n_vectors, head_dim = x.shape
    num_vq_blocks = head_dim // vq_block_size

    # 1. Haar rotate — y[i] = rotation @ x[i]  ==  x @ rotation.T (batch).
    y = x @ rotation.T  # (n_vectors, head_dim)

    # 2. Block split.
    y_blocks = y.reshape(n_vectors, num_vq_blocks, vq_block_size)

    # 3. Per-block norm extract + unit-normalize.
    scales = np.linalg.norm(y_blocks, axis=2)  # (n_vectors, num_vq_blocks)
    safe = np.maximum(scales, 1e-30)
    y_normed = y_blocks / safe[:, :, None]

    # Silica stores scales as fp16 on the payload. Match that here so
    # the reference sees the same fp16-quantized scale the silica
    # dequantize path will use.
    scales_fp16 = scales.astype(np.float16).astype(np.float64)

    # 4. Quantize: searchsorted against Lloyd-Max boundaries
    #    (midpoints between adjacent centroids).
    indices = np.searchsorted(boundaries, y_normed)  # int in [0, 2**num_bits)

    # --- Dequantize ---

    # 5. Centroid lookup.
    y_recon_blocks = centroids[indices]  # (n_vectors, num_vq_blocks, B)

    # 6. Optional per-block norm correction.
    if norm_correction:
        block_norms = np.linalg.norm(y_recon_blocks, axis=2, keepdims=True)
        block_norms = np.maximum(block_norms, 1e-30)
        y_recon_blocks = y_recon_blocks / block_norms

    # 7. Rescale by stored per-block fp16 scales.
    y_recon_blocks = y_recon_blocks * scales_fp16[:, :, None]

    # 8. Flatten block axis and inverse rotate.
    y_recon = y_recon_blocks.reshape(n_vectors, head_dim)
    x_recon: np.ndarray = y_recon @ rotation  # (rotation @ y).T ↔ y @ rotation

    return x_recon


def _fixed_gaussian_input(
    *, n_kv_heads: int, block_size: int, head_dim: int, seed: int
) -> tuple[mx.array, np.ndarray]:
    """Produce a reproducible fp16 Gaussian tensor for both paths.

    Seeds NumPy, draws float32 Gaussian at the flat shape the codecs
    see after reshape, round-trips through fp16 to simulate real K/V
    arriving in fp16, then returns:
    - ``x_mx``: ``(1, n_kv_heads, block_size, head_dim)`` ``mx.float16``
    - ``x_ref``: ``(n_vectors, head_dim)`` ``float64`` (upcast from the
      same fp16 data so both paths process numerically-identical input
      up to fp16 truncation that happened once at the source).
    """
    np.random.seed(seed)
    n_vectors = n_kv_heads * block_size
    x_fp32 = np.random.randn(n_vectors, head_dim).astype(np.float32)
    x_fp16 = x_fp32.astype(np.float16)

    x_mx = mx.array(
        x_fp16.reshape(1, n_kv_heads, block_size, head_dim)
    ).astype(mx.float16)
    x_ref = x_fp16.astype(np.float64)  # upcast fp16 → float64 for reference

    return x_mx, x_ref


def _per_block_frobenius(
    silica_out: np.ndarray, numpy_out: np.ndarray, *, block_size: int
) -> np.ndarray:
    """Per-kv-block relative Frobenius error over the block axis.

    Inputs are ``(n_vectors, head_dim)`` where ``n_vectors = n_kv_heads
    × block_size``. Reshape to ``(block_count, n_kv_heads, block_size,
    head_dim)`` — here ``block_count == 1`` because we test one block
    at a time, but the formula is written to cover multi-block cases.

    Returns ``(block_count,)`` float64 array of relative Frobenius errors.
    """
    n_vectors, head_dim = silica_out.shape
    assert n_vectors % block_size == 0
    n_kv_heads = n_vectors // block_size
    silica_block = silica_out.reshape(1, n_kv_heads, block_size, head_dim)
    numpy_block = numpy_out.reshape(1, n_kv_heads, block_size, head_dim)
    diff = silica_block - numpy_block
    # Frobenius over the non-leading axes of each kv-block.
    diff_fro = np.sqrt((diff * diff).sum(axis=(1, 2, 3)))
    orig_fro = np.sqrt((numpy_block * numpy_block).sum(axis=(1, 2, 3)))
    result: np.ndarray = diff_fro / np.maximum(orig_fro, 1e-30)
    return result


# Parameter grid — covers all BlockTQ registry configs except the
# head_dim=64 edge case (handled by the smaller shape set below).
_BLOCK_TQ_PARAMS = [
    # (vq_block_size, num_bits)
    (32, 3),
    (32, 4),
    (64, 3),
    (64, 4),
]

# Observed per-block relative Frobenius error under fp32 MLX matmul vs
# fp64 NumPy matmul, with fp16 scale storage on both sides, on the
# canonical ``(1, 4, 16, 128)`` Gaussian input at seed 42:
#
#   vq_block_size=32, num_bits=3: 2.1e-4
#   vq_block_size=32, num_bits=4: 2.0e-4
#   vq_block_size=64, num_bits=3: 2.1e-4
#   vq_block_size=64, num_bits=4: 2.1e-4
#   scalar (B=d=64), num_bits=4: 1.2e-3   (num_vq_blocks=1 path)
#
# _TOLERANCE = 5e-3 leaves ~4× headroom over the scalar path (the
# worst observed) and ~25× over BlockTQ paths. Tight enough to catch
# reshape / rotation-direction / tie-break drift while absorbing
# seed-to-seed variation. Loosening to 1e-2 would mask a regression
# where fp16-scale precision deteriorates by 2×; tightening to 1e-3
# flakes on the scalar path's larger fp32/fp64 matmul gap at
# num_vq_blocks=1.
_TOLERANCE = 5e-3

# Tighter regression lock for the production-recommended config. 1e-3
# is ~5× headroom over the observed 2.1e-4; catches smaller drift
# specifically on the path the opening doc pins as production.
_PRODUCTION_TOLERANCE = 1e-3


@pytest.mark.parametrize("vq_block_size,num_bits", _BLOCK_TQ_PARAMS)
def test_blocktq_silica_matches_numpy_reference(
    vq_block_size: int, num_bits: int
) -> None:
    """Silica MLX-native BlockTurboQuantMSE output matches a NumPy
    reference using the same calibration (Haar rotation + Lloyd-Max
    codebook), within fp32 vs fp64 arithmetic tolerance.

    Catches: wrong rotation direction (``rotation`` vs ``rotation.T``),
    wrong block-axis reshape (num_vq_blocks vs vq_block_size swap),
    wrong scale storage dtype (fp32 vs fp16), wrong centroid lookup
    (argmin vs searchsorted mismatched beyond ties), incorrect
    norm-correction flag application.
    """
    block_size = 16
    n_kv_heads = 4
    head_dim = 128
    seed = 42

    x_mx, x_ref = _fixed_gaussian_input(
        n_kv_heads=n_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        seed=seed,
    )

    # Silica MLX encode + decode.
    codec = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=vq_block_size,
        num_bits=num_bits,
        seed=seed,
    )
    silica_out_mx = codec.decode_tensor(codec.encode_tensor(x_mx))
    silica_out = (
        np.array(silica_out_mx.astype(mx.float32))
        .reshape(n_kv_heads * block_size, head_dim)
        .astype(np.float64)
    )

    # NumPy reference using shared calibration only; hot-path arithmetic
    # is independent.
    rotation = haar_rotation(head_dim, seed).copy()
    sigma = 1.0 / math.sqrt(vq_block_size)
    centroids_frozen, boundaries_frozen = lloyd_max_codebook(num_bits, sigma)
    # Copy before passing to the reference — lloyd_max_codebook returns
    # write-protected arrays, which is fine for reading but avoids any
    # accidental mutation in the reference function.
    centroids = centroids_frozen.copy()
    boundaries = boundaries_frozen.copy()

    numpy_out = _numpy_block_tq_round_trip(
        x_ref,
        rotation=rotation,
        centroids=centroids,
        boundaries=boundaries,
        vq_block_size=vq_block_size,
    )

    # Primary metric — per-block relative Frobenius error.
    per_block_err = _per_block_frobenius(
        silica_out, numpy_out, block_size=block_size
    )
    max_err = float(per_block_err.max())
    assert max_err < _TOLERANCE, (
        f"BlockTQ parity failed: per-block relative Frobenius error "
        f"{max_err:.4e} exceeds tolerance {_TOLERANCE:.4e} at "
        f"vq_block_size={vq_block_size}, num_bits={num_bits}. "
        f"Per-block errors: {per_block_err.tolist()}"
    )


def test_blocktq_production_recommendation_parity_tighter() -> None:
    """The production-recommended config (B=64 b=4 per opening §5.2 +
    vqbench REPORT §3.1) should land well inside the tolerance envelope.
    Tighter assertion here regression-locks the specific production row."""
    block_size = 16
    n_kv_heads = 4
    head_dim = 128
    vq_block_size = 64
    num_bits = 4
    seed = 42

    x_mx, x_ref = _fixed_gaussian_input(
        n_kv_heads=n_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        seed=seed,
    )

    codec = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=vq_block_size,
        num_bits=num_bits,
        seed=seed,
    )
    silica_out_mx = codec.decode_tensor(codec.encode_tensor(x_mx))
    silica_out = (
        np.array(silica_out_mx.astype(mx.float32))
        .reshape(n_kv_heads * block_size, head_dim)
        .astype(np.float64)
    )

    rotation = haar_rotation(head_dim, seed).copy()
    sigma = 1.0 / math.sqrt(vq_block_size)
    centroids_frozen, boundaries_frozen = lloyd_max_codebook(num_bits, sigma)
    numpy_out = _numpy_block_tq_round_trip(
        x_ref,
        rotation=rotation,
        centroids=centroids_frozen.copy(),
        boundaries=boundaries_frozen.copy(),
        vq_block_size=vq_block_size,
    )

    per_block_err = _per_block_frobenius(
        silica_out, numpy_out, block_size=block_size
    )
    assert float(per_block_err.max()) < _PRODUCTION_TOLERANCE, (
        f"BlockTQ B=64 b=4 production parity: max per-block rel "
        f"Frobenius {per_block_err.max():.4e} exceeds production "
        f"tolerance {_PRODUCTION_TOLERANCE:.4e}. Per-block errors: "
        f"{per_block_err.tolist()}"
    )


def test_parity_holds_across_independent_seeds() -> None:
    """Parity should hold across multiple input seeds at the
    production config. Three seeds' max per-block err should all stay
    below the 5e-3 tolerance — regression-locks that tie-break / fp32
    precision doesn't create seed-dependent flakiness."""
    block_size = 16
    n_kv_heads = 4
    head_dim = 128
    vq_block_size = 64
    num_bits = 4

    for input_seed in (7, 42, 99):
        x_mx, x_ref = _fixed_gaussian_input(
            n_kv_heads=n_kv_heads,
            block_size=block_size,
            head_dim=head_dim,
            seed=input_seed,
        )
        # Codec seed pinned at 42 so the Haar rotation is shared across
        # input seeds — isolates the input-distribution variation from
        # the calibration variation.
        codec = BlockTurboQuantMSE(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            vq_block_size=vq_block_size,
            num_bits=num_bits,
            seed=42,
        )
        silica_out_mx = codec.decode_tensor(codec.encode_tensor(x_mx))
        silica_out = (
            np.array(silica_out_mx.astype(mx.float32))
            .reshape(n_kv_heads * block_size, head_dim)
            .astype(np.float64)
        )

        rotation = haar_rotation(head_dim, 42).copy()
        sigma = 1.0 / math.sqrt(vq_block_size)
        centroids_frozen, boundaries_frozen = lloyd_max_codebook(num_bits, sigma)
        numpy_out = _numpy_block_tq_round_trip(
            x_ref,
            rotation=rotation,
            centroids=centroids_frozen.copy(),
            boundaries=boundaries_frozen.copy(),
            vq_block_size=vq_block_size,
        )

        per_block_err = _per_block_frobenius(
            silica_out, numpy_out, block_size=block_size
        )
        assert float(per_block_err.max()) < _TOLERANCE, (
            f"BlockTQ parity failed at input_seed={input_seed}: max "
            f"per-block rel Frobenius {per_block_err.max():.4e} > "
            f"{_TOLERANCE}. Per-block errors: {per_block_err.tolist()}"
        )


def test_scalar_tq_parity_b_equals_d() -> None:
    """Scalar TurboQuantMSE is ``BlockTurboQuantMSE(vq_block_size =
    head_dim)``. At this boundary the NumPy reference's
    ``num_vq_blocks=1`` path should still parity-match silica."""
    block_size = 16
    n_kv_heads = 4
    head_dim = 64
    num_bits = 4
    seed = 42

    x_mx, x_ref = _fixed_gaussian_input(
        n_kv_heads=n_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        seed=seed,
    )

    # Scalar TQ = BlockTQ with vq_block_size = head_dim.
    codec = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=head_dim,
        num_bits=num_bits,
        seed=seed,
    )
    silica_out_mx = codec.decode_tensor(codec.encode_tensor(x_mx))
    silica_out = (
        np.array(silica_out_mx.astype(mx.float32))
        .reshape(n_kv_heads * block_size, head_dim)
        .astype(np.float64)
    )

    rotation = haar_rotation(head_dim, seed).copy()
    sigma = 1.0 / math.sqrt(head_dim)
    centroids_frozen, boundaries_frozen = lloyd_max_codebook(num_bits, sigma)
    numpy_out = _numpy_block_tq_round_trip(
        x_ref,
        rotation=rotation,
        centroids=centroids_frozen.copy(),
        boundaries=boundaries_frozen.copy(),
        vq_block_size=head_dim,
    )

    per_block_err = _per_block_frobenius(
        silica_out, numpy_out, block_size=block_size
    )
    assert float(per_block_err.max()) < _TOLERANCE, (
        f"Scalar TQ parity failed: {per_block_err.max():.4e} > "
        f"{_TOLERANCE}"
    )
