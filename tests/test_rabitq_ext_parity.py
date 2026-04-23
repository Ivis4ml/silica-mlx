"""P-5-B.2c — algorithmic-parity gate: MLX ExtRaBitQ vs NumPy reference.

Companion parity file to test_rabitq_1bit_parity.py. The companion proves
the 1-bit hypercube path agrees with vqbench; this file proves the
multi-bit integer-grid path agrees across num_bits in {2, 3, 4}.

What this test does:

- Transcribes the relevant batch paths from
  ``vqbench/vqbench/methods/rabitq/rabitq_ext.py::quantize_batch`` and
  ``::dequantize``, plus the Haar rotation from
  ``vqbench/vqbench/core/rotation.py::haar_rotation``, as small NumPy
  helpers inline below. **No vqbench import.** Matches the pattern of
  test_rabitq_1bit_parity.py / test_block_tq_vqbench_xcheck.py; see
  feedback_vqbench_reference_pattern.md for the convention.

- The Haar helper is intentionally duplicated (5 lines) in each parity
  file rather than lifted into a shared ``tests/_vqbench_reference.py``
  module. Duplication keeps each parity file self-contained (one can
  read any one file without jumping) and preserves the cross-check
  value of "two independent reproductions of the algorithm".
  Docstring convention: "verbatim transcription of
  vqbench.core.rotation.haar_rotation" so grep / audit sweeps catch
  every copy.

- Both sides consume the same fp16-rounded input tensor (fp32 Gaussian
  -> cast to fp16 once -> fed to silica as mx.array(fp16), fed to
  reference as astype(fp64)). Sign / rounding parity under matched
  input.

- Matches silica's payload storage precision in the reference's decode:
  norm_o round-tripped through fp16 (``norm_o_storage_dtype=np.float16``),
  so Test 5's tolerance measures decode algorithmic drift, not fp16
  storage precision drift. Identical convention to B.1c for
  consistency.

What this test does NOT do:

- Import vqbench. See feedback_vqbench_reference_pattern.md.
- Re-test formulas against themselves (that is test_rabitq_ext.py).
- Gate on cross-B monotone MSE (that is test_rabitq_ext.py).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from silica.kvcache.codec import ExtRaBitQPayload
from silica.vq import ExtRaBitQ
from silica.vq.core.packing import unpack_sub_byte

_BLOCK_SIZE = 16
_N_KV_HEADS = 4
_HEAD_DIM = 64
_SEED = 42
_N_VECTORS = _N_KV_HEADS * _BLOCK_SIZE  # 64


# =============================================================================
# NumPy reference transcription (no vqbench import)
# =============================================================================


def _reference_haar_rotation(d: int, seed: int) -> np.ndarray:
    """Verbatim transcription of ``vqbench.core.rotation.haar_rotation``.

    Deliberately duplicated from ``test_rabitq_1bit_parity.py`` rather
    than shared via a helper module; see this file's module docstring
    for the rationale. Same 5-line Stewart (1980) QR + sign-fix recipe
    over the same numpy RNG; same ``(d, seed)`` → bit-identical matrix
    across any reproduction.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[np.newaxis, :]


def _reference_ext_rabitq_codebook(num_bits: int) -> np.ndarray:
    """Verbatim transcription of
    ``vqbench.methods.rabitq.rabitq_ext.ext_rabitq_codebook``.

    Odd-integer grid per paper arXiv 2409.09913 Eq. 3:
    ``{-(2^B - 1), -(2^B - 3), ..., -1, +1, ..., +(2^B - 1)}`` indexed
    by ``[0, 2^B - 1]``.
    """
    levels = np.arange(2 ** num_bits, dtype=np.int64)
    out: np.ndarray = (2 * levels - (2 ** num_bits - 1)).astype(np.float64)
    return out


def _reference_ext_rabitq_quantize_batch(
    x: np.ndarray,
    *,
    rotation: np.ndarray,
    num_bits: int,
    centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim transcription of
    ``vqbench.methods.rabitq.rabitq_ext.ExtRaBitQ.quantize_batch``.

    Returns ``(indices, norm_o, ip_coeff, scale_inv)`` with everything
    cast to fp64 at the boundary for test-time comparison. Internal
    arithmetic is fp32 to match silica's MLX hot-path precision —
    without this, the fp64 vs fp32 rounding gap at the codebook
    boundaries (``scaled = ±2k``) would cause O(~0.1-1%) of indices to
    flip, confounding the algorithm-correctness test with an
    unrelated precision difference. vqbench itself runs at fp64; we
    mirror silica's fp32 here deliberately, since the gate is "silica
    MLX correctly implements the algorithm", not "silica matches
    vqbench's numerical behaviour byte-for-byte across precisions".

    Rounding: ``np.round`` is half-to-even (banker's) — pinned in
    silica's TestMLXRoundSemantics class to match this.

    Centroid defaults to zeros per opening §5.3 (P-5-B scope); matches
    silica's ``_centroid`` field.
    """
    d = x.shape[-1]
    if centroid is None:
        centroid = np.zeros(d, dtype=np.float32)

    # Match silica's fp32 hot-path precision: every intermediate the
    # rounding decision depends on (Y, scaled, rounded) is fp32. The
    # rotation arrives in fp64 from _reference_haar_rotation; cast
    # once here so the matmul produces an fp32 Y (rather than fp32 @
    # fp64 promoting).
    rotation_f32 = rotation.astype(np.float32)
    flat = x.reshape(-1, d).astype(np.float32)
    n_vectors = flat.shape[0]
    centered = flat - centroid.astype(np.float32)
    norm_o = np.linalg.norm(centered, axis=1)              # fp32
    safe_norm = np.maximum(norm_o, np.float32(1e-30))
    o_bar = centered / safe_norm[:, None]                  # fp32
    Y = o_bar @ rotation_f32.T                              # fp32

    half_range = (2 ** num_bits) - 1
    # quant_scale / scale_inv are Python floats (fp64 internally);
    # NumPy multiplication of a fp32 array by a Python float preserves
    # the array's fp32 dtype, matching MLX's mx.array * Python-float.
    quant_scale = float(half_range) * math.sqrt(d) / 3.0
    scale_inv = 1.0 / quant_scale

    scaled = Y * quant_scale                                # fp32
    rounded = np.round((scaled - 1.0) / 2.0) * 2.0 + 1.0   # fp32, half-to-even
    rounded = np.clip(rounded, -float(half_range), float(half_range))
    indices_fp = (rounded + half_range) / 2.0
    indices = indices_fp.astype(np.uint8)                   # (n_vectors, d)

    codebook = _reference_ext_rabitq_codebook(num_bits).astype(np.float32)
    x_bar_raw = codebook[indices.astype(np.int64)]          # fp32
    x_bar_norm = np.linalg.norm(x_bar_raw, axis=1, keepdims=True)
    safe_bar_norm = np.maximum(x_bar_norm, np.float32(1e-30))
    x_bar = x_bar_raw / safe_bar_norm                       # fp32
    ip_coeff = np.sum(x_bar * Y, axis=1)                    # fp32

    scale_per_vec = np.full((n_vectors,), scale_inv, dtype=np.float64)

    # Lift scalars back to fp64 at the return boundary so the test-
    # side rtol/atol arithmetic runs in fp64 (standard for assert_allclose).
    return (
        indices,
        norm_o.astype(np.float64),
        ip_coeff.astype(np.float64),
        scale_per_vec,
    )


def _reference_ext_rabitq_decode(
    indices: np.ndarray,
    norm_o: np.ndarray,
    scale: np.ndarray,
    *,
    rotation: np.ndarray,
    num_bits: int,
    centroid: np.ndarray | None = None,
    norm_o_storage_dtype: np.dtype | type = np.float16,
) -> np.ndarray:
    """Batch-form transcription of
    ``vqbench.methods.rabitq.rabitq_ext.ExtRaBitQ.dequantize``.

    ``norm_o_storage_dtype`` round-trips norm_o through the payload's
    storage dtype before scaling so the comparison isolates decode
    algorithm drift from fp16 storage precision drift (silica's payload
    stores fp16 norm_o and fp16 scale; reference would otherwise be
    unfairly higher-precision). Setting it to ``np.float64`` recovers
    the "true" reference decode.

    ``scale`` is expected to already be the dequantization factor (the
    inverse of ``quant_scale``); this mirrors silica's payload
    semantics where ``payload.scale`` is already ``1 / quant_scale``.
    """
    d = indices.shape[-1]
    if centroid is None:
        centroid = np.zeros(d, dtype=np.float32)

    # Match silica's fp32 decode precision. fp16 → fp32 round-trip for
    # stored scalars mirrors silica's ``payload.norm_o.astype(fp32)`` /
    # ``payload.scale.astype(fp32)`` calls in decode_tensor.
    rotation_f32 = rotation.astype(np.float32)
    norm_o_stored = norm_o.astype(norm_o_storage_dtype).astype(np.float32)
    scale_stored = scale.astype(norm_o_storage_dtype).astype(np.float32)

    codebook = _reference_ext_rabitq_codebook(num_bits).astype(np.float32)
    y_raw = codebook[indices.astype(np.int64)]              # fp32
    y_hat = y_raw * scale_stored[:, None]                   # fp32

    # Re-normalize to unit norm (unbiased-estimator invariant). Matches
    # silica's decode.
    y_norm = np.linalg.norm(y_hat, axis=1, keepdims=True)
    safe_y_norm = np.maximum(y_norm, np.float32(1e-30))
    y_hat = y_hat / safe_y_norm

    # Inverse rotate (row-vector convention: y_hat @ rotation).
    x_hat = y_hat @ rotation_f32                             # fp32
    out: np.ndarray = (
        x_hat * norm_o_stored[:, None] + centroid.astype(np.float32)
    )
    return out.astype(np.float64)


# =============================================================================
# Shared input fixture
# =============================================================================


def _make_shared_fp16_input(
    *,
    n_kv_heads: int = _N_KV_HEADS,
    block_size: int = _BLOCK_SIZE,
    head_dim: int = _HEAD_DIM,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic fp16-rounded input, flattened to
    ``(n_vectors, head_dim)``. Both silica and the reference consume
    this same array; fp32→fp16 rounding happens exactly once."""
    rng = np.random.default_rng(seed)
    x_fp32 = rng.standard_normal(
        (n_kv_heads * block_size, head_dim)
    ).astype(np.float32)
    return x_fp32.astype(np.float16)


def _silica_encode(
    codec: ExtRaBitQ, x_fp16: np.ndarray
) -> tuple[np.ndarray, ExtRaBitQPayload]:
    """Encode the shared fp16 input through silica and return
    ``(unpacked_indices, payload)``."""
    reshaped = x_fp16.reshape(1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM)
    x_mx = mx.array(reshaped).astype(mx.float16)
    payload = codec.encode_tensor(x_mx)
    unpacked = unpack_sub_byte(
        payload.packed_indices, num_bits=codec._num_bits, d=_HEAD_DIM
    )
    return np.asarray(unpacked), payload


def _make_codec(num_bits: int) -> ExtRaBitQ:
    return ExtRaBitQ(
        block_size=_BLOCK_SIZE,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_bits=num_bits,
        seed=_SEED,
    )


# =============================================================================
# Test 1 — indices near-bit-for-bit (off-by-one at fp32 boundary fuzz)
# =============================================================================

# Bound on fraction of coordinates where silica's MLX fp32 matmul and the
# reference's NumPy fp32 matmul disagree by one codebook step. This fuzz
# happens only when the rotated coordinate ``scaled`` lands within one
# ULP-level distance of a rounding boundary (``scaled = ±2k``); MLX's
# Metal accumulation order differs subtly from NumPy's CPU-BLAS
# accumulation at matching fp32 precision. Empirically the fraction is
# ~0.3% at d=64 on random-normal input; we bound loosely at 1.5% so
# the test is robust across seed / MLX version / Metal driver while
# still catching real algorithm drift (wrong rotation or codebook would
# flip 30-90% of indices, not 0.3%). Every mismatch that does occur
# must be bounded to a single codebook step — a wilder disagreement
# means the algorithm itself drifted.
_BOUNDARY_FUZZ_MAX_FRACTION = 0.015
_BOUNDARY_FUZZ_MAX_STEP_DELTA = 1


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_indices_mostly_match_reference(num_bits: int) -> None:
    """Under the same fp16-valued input and the same Haar rotation,
    silica's unpacked indices must agree with the reference on all but
    a small fraction of coordinates (boundary fuzz), and every
    disagreement must be a single codebook step. Wilder mismatches
    or higher mismatch rates signal algorithm drift (wrong rotation,
    wrong codebook, tie-break flipped)."""
    codec = _make_codec(num_bits)
    x_fp16 = _make_shared_fp16_input(seed=0)

    silica_indices, _ = _silica_encode(codec, x_fp16)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_indices, _, _, _ = _reference_ext_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )

    diff = np.abs(
        silica_indices.astype(np.int32) - ref_indices.astype(np.int32)
    )
    n_total = int(diff.size)
    n_mismatch = int(np.sum(diff > 0))
    max_delta = int(np.max(diff))
    mismatch_frac = n_mismatch / n_total

    # Every disagreement must be a single codebook step.
    assert max_delta <= _BOUNDARY_FUZZ_MAX_STEP_DELTA, (
        f"B={num_bits}: max index delta {max_delta} exceeds boundary "
        f"fuzz limit {_BOUNDARY_FUZZ_MAX_STEP_DELTA}. A multi-step "
        f"disagreement is not a precision issue — the algorithm drifted."
    )
    # Mismatch rate must stay below the boundary-fuzz bound.
    assert mismatch_frac < _BOUNDARY_FUZZ_MAX_FRACTION, (
        f"B={num_bits}: {n_mismatch}/{n_total} = {mismatch_frac:.4%} "
        f"indices disagree (bound {_BOUNDARY_FUZZ_MAX_FRACTION:.2%}). "
        f"Rate this high suggests the algorithm drifted rather than "
        f"boundary-precision fuzz."
    )


# =============================================================================
# Test 2 — norm_o within fp16 tolerance
# =============================================================================


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_norm_o_matches_reference_within_fp16(num_bits: int) -> None:
    """silica stores norm_o as fp16; reference computes fp64. Tolerance
    ``rtol=5e-3, atol=5e-4`` matches B.1c — fp16 precision on values
    clustered around ``sqrt(d) ≈ 8`` for random-normal inputs."""
    codec = _make_codec(num_bits)
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode(codec, x_fp16)
    silica_norm = np.asarray(payload.norm_o).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    _, ref_norm, _, _ = _reference_ext_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )

    np.testing.assert_allclose(silica_norm, ref_norm, rtol=5e-3, atol=5e-4)


# =============================================================================
# Test 3 — ip_coeff within fp16 tolerance
# =============================================================================


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_ip_coeff_matches_reference_within_fp16(num_bits: int) -> None:
    """silica fp16 vs reference fp64 ip_coeff. Tolerance absorbs both
    fp16 storage precision and the per-vector boundary-fuzz
    contribution: each off-by-one index flip perturbs ip_coeff by up
    to ``(2 / ||codebook[indices]||) · |Y[i]|`` which is O(1e-2) for
    B=2 on the tightest-step codebook, tighter for higher B. ``rtol
    =2e-2`` gives headroom for the worst case while still catching
    structural algorithm drift (wrong codebook, missed normalization
    in x_bar)."""
    codec = _make_codec(num_bits)
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode(codec, x_fp16)
    silica_ip = np.asarray(payload.ip_coeff).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    _, _, ref_ip, _ = _reference_ext_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )

    np.testing.assert_allclose(silica_ip, ref_ip, rtol=2e-2, atol=5e-4)


# =============================================================================
# Test 4 — scale within fp16 tolerance
# =============================================================================


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_scale_matches_reference_within_fp16(num_bits: int) -> None:
    """Per-vector scale is a constant in v0.1: ``3 / ((2^B - 1) · √d)``.
    silica stores it as fp16 per vector; reference computes the same
    constant in fp64 and broadcasts to per-vector. Comparison catches
    the case where silica's ``_scale_inv`` formula drifts from the
    paper / vqbench convention by a factor or sign."""
    codec = _make_codec(num_bits)
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode(codec, x_fp16)
    silica_scale = np.asarray(payload.scale).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    _, _, _, ref_scale = _reference_ext_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )

    np.testing.assert_allclose(silica_scale, ref_scale, rtol=5e-3, atol=5e-4)


# =============================================================================
# Test 5 — decoded vector within fp16 tolerance
# =============================================================================


@pytest.mark.parametrize("num_bits", [2, 3, 4])
def test_decoded_vector_matches_reference_within_fp16(num_bits: int) -> None:
    """Decode algorithm parity under matched payload-storage precision.

    Reference decode applies the same fp16 round-trip on both
    ``norm_o`` and ``scale`` that silica's payload stores. Comparison
    uses **relative Frobenius error** rather than per-element tolerance
    because the 0.02-0.3% boundary-fuzz index flips (see
    test_indices_mostly_match_reference) each contribute up to one
    codebook-step × scale_inv × norm_o to the affected coord, which
    exceeds any reasonable per-element ``rtol``. The Frobenius norm
    averages these contributions over all coords.

    Observed values (d=64, seed=0) under the correct algorithm:
      B=2 -> rel_frob ≈ 0.026
      B=3 -> rel_frob ≈ 0.014
      B=4 -> rel_frob ≈ 0.020

    Threshold 0.08 is ~3× larger than the observed worst case, giving
    seed / MLX-version / Metal-driver robustness. Structural algorithm
    drift (transposed rotation, missed re-normalization, flipped
    codebook signs, wrong scale factor) produces rel_frob of O(0.2-1.0),
    well above the threshold.
    """
    codec = _make_codec(num_bits)
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode(codec, x_fp16)
    silica_decoded = np.asarray(codec.decode_tensor(payload)).astype(np.float64)
    silica_decoded_flat = silica_decoded.reshape(_N_VECTORS, _HEAD_DIM)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_indices, ref_norm, _, ref_scale = _reference_ext_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )
    ref_decoded = _reference_ext_rabitq_decode(
        ref_indices,
        ref_norm,
        ref_scale,
        rotation=rotation_ref,
        num_bits=num_bits,
        norm_o_storage_dtype=np.float16,
    )

    err = silica_decoded_flat - ref_decoded
    ref_frob = float(np.linalg.norm(ref_decoded))
    assert ref_frob > 0.0, (
        f"B={num_bits}: reference decode has zero Frobenius norm; "
        f"input is degenerate."
    )
    rel_frob = float(np.linalg.norm(err)) / ref_frob
    assert rel_frob < 0.08, (
        f"B={num_bits}: relative Frobenius error {rel_frob:.4f} "
        f"exceeds 0.08 headroom. Expected O(0.01-0.03) from boundary "
        f"fuzz; this suggests a structural decode-algorithm drift."
    )


# =============================================================================
# Test 6 — zero-vector batch parity
# =============================================================================


@pytest.mark.parametrize(
    "num_bits,expected_index",
    [(2, 2), (3, 4), (4, 8)],
)
def test_zero_vector_batch_parity(num_bits: int, expected_index: int) -> None:
    """Zero input on the batch path: both silica and the reference must
    produce ``norm_o = 0``, ``ip_coeff = 0``, and all-``2^(B-1)`` indices
    (the +1 codebook entry, which is where half-to-even rounding of
    ``(0 - 1)/2 = -0.5`` lands). ``scale`` is the same non-zero
    constant ``3 / ((2^B - 1) · √d)`` as for any other vector. Pins
    the zero-vector batch-path semantics against the independent
    reference."""
    codec = _make_codec(num_bits)
    x_zero_fp16 = np.zeros((_N_VECTORS, _HEAD_DIM), dtype=np.float16)

    silica_indices, payload = _silica_encode(codec, x_zero_fp16)
    silica_norm = np.asarray(payload.norm_o).astype(np.float64)
    silica_ip = np.asarray(payload.ip_coeff).astype(np.float64)
    silica_scale = np.asarray(payload.scale).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_indices, ref_norm, ref_ip, ref_scale = _reference_ext_rabitq_quantize_batch(
        x_zero_fp16.astype(np.float64),
        rotation=rotation_ref,
        num_bits=num_bits,
    )

    # Reference self-consistency.
    expected_indices = np.full_like(ref_indices, expected_index)
    np.testing.assert_array_equal(ref_indices, expected_indices)
    np.testing.assert_array_equal(ref_norm, np.zeros(_N_VECTORS, dtype=np.float64))
    np.testing.assert_array_equal(ref_ip, np.zeros(_N_VECTORS, dtype=np.float64))
    expected_scale_inv = 3.0 / (
        (2 ** num_bits - 1) * math.sqrt(_HEAD_DIM)
    )
    np.testing.assert_allclose(
        ref_scale,
        np.full(_N_VECTORS, expected_scale_inv),
        rtol=0.0,
        atol=0.0,
    )

    # silica agrees with reference.
    np.testing.assert_array_equal(silica_indices, ref_indices)
    np.testing.assert_array_equal(silica_norm, ref_norm)
    np.testing.assert_array_equal(silica_ip, ref_ip)
    # Scale parity under fp16 round-trip (silica stores fp16, reference
    # computes fp64 — same constant though, so rtol tracks fp16
    # precision on the 0.188 / 0.054 / 0.025 values seen here).
    np.testing.assert_allclose(
        silica_scale, ref_scale, rtol=5e-3, atol=5e-4
    )
