"""P-5-B.1c — algorithmic-parity gate: MLX RaBitQ1Bit vs NumPy reference.

Second-pair unit with test_rabitq_1bit.py. test_rabitq_1bit.py proves the
codec is internally consistent (formula closes on itself); this file
proves the silica MLX port agrees with an independent NumPy reference.

What this test does:

- Transcribes the relevant batch paths from
  ``vqbench/vqbench/methods/rabitq/rabitq_1bit.py::quantize_batch`` and
  ``::dequantize``, plus the Haar rotation from
  ``vqbench/vqbench/core/rotation.py::haar_rotation``, as small NumPy
  helpers inline below. **No vqbench import.** vqbench/ is source
  material we read while writing silica's port; it is not a runtime or
  test dependency (D-009 boundary extended to tests per feedback).
  Mirrors the pattern established in
  ``tests/test_block_tq_vqbench_xcheck.py``.

- Drives both silica's MLX codec and the inline NumPy reference from
  the **same fp16-rounded input tensor**. Input is first generated as
  fp32 Gaussian, immediately cast to fp16, and then:
    * silica gets ``mx.array(x_fp16).astype(mx.float16)``
    * reference gets ``x_fp16.astype(np.float64)``
  This means sign-level parity is truly an algorithm comparison — any
  fp32→fp16 rounding happens before either codec sees the tensor, so
  it cannot cause a sign flip between the two paths.

- Matches silica's payload storage precision in the reference's decode
  path: ``norm_o`` is round-tripped through fp16 before being used in
  the reference decode, so Test 5's tolerance measures decode
  algorithmic drift, not fp16 storage precision drift.

What this test does NOT do:

- Import vqbench. See feedback_vqbench_reference_pattern.md.
- Re-test formulas against themselves (that is test_rabitq_1bit.py).
- Test ExtRaBitQ (B.2c).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from silica.kvcache.codec import RaBitQPayload
from silica.vq import RaBitQ1Bit
from silica.vq._calibration import haar_rotation
from silica.vq.core.packing import unpack_sub_byte

# Standard parameters; matched against test_rabitq_1bit.py for consistency.
_BLOCK_SIZE = 16
_N_KV_HEADS = 4
_HEAD_DIM = 64
_SEED = 42
_N_VECTORS = _N_KV_HEADS * _BLOCK_SIZE  # 64


# =============================================================================
# NumPy reference transcription (no vqbench import)
# =============================================================================


def _reference_haar_rotation(d: int, seed: int) -> np.ndarray:
    """Verbatim transcription of ``vqbench/vqbench/core/rotation.py::haar_rotation``.

    Stewart (1980) algorithm: QR of a ``N(0, 1)^{d×d}`` Gaussian matrix
    followed by ``Q ← Q @ diag(sign(diag(R)))`` to fix the sign
    ambiguity. Same ``(d, seed)`` → bit-identical matrix across any
    implementation that follows this recipe over the same numpy RNG.

    Intentionally does not cache (silica's version does); parity is
    between the two returned arrays, not object identity.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[np.newaxis, :]


def _reference_rabitq_quantize_batch(
    x: np.ndarray,
    *,
    rotation: np.ndarray,
    centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim transcription of
    ``vqbench/vqbench/methods/rabitq/rabitq_1bit.py::quantize_batch``.

    Returns ``(signs, norm_o, ip_coeff)``:

    - ``signs``: ``(n_vectors, d)`` int8, values in ``{-1, +1}``.
      Tie-break sends ``np.sign(Y) == 0`` to ``+1`` (batch-path
      semantics; matches silica exactly).
    - ``norm_o``: ``(n_vectors,)`` fp64 L2 norm of the centered vector.
    - ``ip_coeff``: ``(n_vectors,)`` fp64, ``sum(signs * Y) / sqrt(d)``.
      Always non-negative since ``signs * Y = |Y|`` elementwise.

    Centroid defaults to zeros per opening §5.3 (P-5-B scope); matches
    silica's ``_centroid`` field and vqbench's ``_get_centroid``
    fallback when ``_centroid is None`` (which is the case when vqbench
    ``fit()`` is never called).

    Reference for the full arithmetic:
    ``vqbench/vqbench/methods/rabitq/rabitq_1bit.py`` lines 120-143
    (commit at read time).
    """
    d = x.shape[-1]
    if centroid is None:
        centroid = np.zeros(d, dtype=np.float64)

    flat = x.reshape(-1, d).astype(np.float64)
    centered = flat - centroid
    norm_o = np.linalg.norm(centered, axis=1)                  # (n_vectors,)
    safe_norm = np.maximum(norm_o, 1e-30)
    o_bar = centered / safe_norm[:, None]                      # (n_vectors, d)
    Y = o_bar @ rotation.T                                      # (n_vectors, d)

    signs = np.sign(Y).astype(np.int8)
    signs[signs == 0] = 1                                       # tie-break 0 → +1

    signs_fp = signs.astype(np.float64)
    ip_coeff = np.sum(signs_fp * Y, axis=1) / np.sqrt(d)

    return signs, norm_o, ip_coeff


def _reference_rabitq_decode(
    signs: np.ndarray,
    norm_o: np.ndarray,
    *,
    rotation: np.ndarray,
    centroid: np.ndarray | None = None,
    norm_o_storage_dtype: np.dtype | type = np.float16,
) -> np.ndarray:
    """Batch-form transcription of
    ``vqbench/vqbench/methods/rabitq/rabitq_1bit.py::dequantize``.

    vqbench's single-vector ``dequantize`` uses column-vector convention
    ``x_hat = rotation.T @ y_hat``; this is identical to the row-vector
    ``y_hat @ rotation`` form used here since ``rotation`` is
    orthogonal.

    ``norm_o_storage_dtype`` round-trips the norm through the dtype
    silica's payload actually stores (fp16 by default). Setting it to
    ``np.float64`` recovers the "true" reference decode; we use fp16 by
    default so Test 5 compares decode algorithm drift, not fp16 storage
    precision drift (silica's payload stores fp16 norm_o and the
    reference would otherwise be unfairly higher precision).
    """
    d = signs.shape[-1]
    if centroid is None:
        centroid = np.zeros(d, dtype=np.float64)

    # Round-trip the norm through the payload-storage dtype so the
    # reference operates on the same quantized scalar the silica decode
    # actually consumes at decode time.
    norm_o_stored = norm_o.astype(norm_o_storage_dtype).astype(np.float64)

    y_hat = signs.astype(np.float64) / np.sqrt(d)               # (n_vectors, d)
    x_hat = y_hat @ rotation                                     # inverse rotate
    out: np.ndarray = x_hat * norm_o_stored[:, None] + centroid
    return out


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
    """Produce a deterministic fp16-rounded input array, flattened to
    ``(n_vectors, head_dim)``. Both silica and the reference consume
    this same array; the fp32→fp16 rounding happens exactly once,
    before either path sees it."""
    rng = np.random.default_rng(seed)
    x_fp32 = rng.standard_normal(
        (n_kv_heads * block_size, head_dim)
    ).astype(np.float32)
    return x_fp32.astype(np.float16)


def _silica_encode_signs_and_payload(
    codec: RaBitQ1Bit, x_fp16: np.ndarray
) -> tuple[np.ndarray, RaBitQPayload]:
    """Encode the shared fp16 input through silica and return
    ``(signs, payload)`` where ``signs`` are in ``{-1, +1}`` int8."""
    reshaped = x_fp16.reshape(1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM)
    x_mx = mx.array(reshaped).astype(mx.float16)
    payload = codec.encode_tensor(x_mx)
    bits = unpack_sub_byte(payload.packed_indices, num_bits=1, d=_HEAD_DIM)
    signs = 2 * np.asarray(bits).astype(np.int8) - 1
    return signs, payload


def _make_codec() -> RaBitQ1Bit:
    return RaBitQ1Bit(
        block_size=_BLOCK_SIZE,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        seed=_SEED,
    )


# =============================================================================
# Test 1 — Haar rotation equality (gate on everything below)
# =============================================================================


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("seed", [0, 42, 1337])
def test_haar_rotation_matches_reference(d: int, seed: int) -> None:
    """silica.vq._calibration.haar_rotation must be bit-identical to the
    inline reference for every ``(d, seed)`` — both implement Stewart
    (1980) QR + sign-fix over the same numpy RNG, so any divergence
    means one of them diverged from the algorithm.

    This test gates every other test in the file: if Haar rotations
    disagree, the downstream sign / norm / ip_coeff parity tests would
    be chasing a rotation mismatch rather than a codec-algorithm bug.
    """
    silica = np.asarray(haar_rotation(d, seed))
    reference = _reference_haar_rotation(d, seed)
    np.testing.assert_array_equal(silica, reference)


# =============================================================================
# Test 2 — signs bit-for-bit under fp16-rounded input
# =============================================================================


def test_signs_match_reference_bit_for_bit() -> None:
    """Under the same fp16-valued input and the same Haar rotation,
    silica's packed sign bits (unpacked to ``{-1, +1}``) must equal the
    reference's signs exactly. No tolerance — the sign function is
    discrete, and the only source of sign disagreement would be a
    genuine algorithm drift (e.g., wrong rotation direction, wrong
    tie-break)."""
    codec = _make_codec()
    x_fp16 = _make_shared_fp16_input(seed=0)

    silica_signs, _ = _silica_encode_signs_and_payload(codec, x_fp16)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_signs, _, _ = _reference_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
    )

    np.testing.assert_array_equal(silica_signs, ref_signs)


# =============================================================================
# Test 3 — norm_o within fp16 tolerance
# =============================================================================


def test_norm_o_matches_reference_within_fp16() -> None:
    """silica stores norm_o as fp16; reference computes fp64. Tolerance
    ``rtol=5e-3, atol=5e-4`` covers fp16 precision on values clustered
    around ``sqrt(d) ≈ 8`` for random-normal inputs in R^64."""
    codec = _make_codec()
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode_signs_and_payload(codec, x_fp16)
    silica_norm = np.asarray(payload.norm_o).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    _, ref_norm, _ = _reference_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
    )

    np.testing.assert_allclose(silica_norm, ref_norm, rtol=5e-3, atol=5e-4)


# =============================================================================
# Test 4 — ip_coeff within fp16 tolerance
# =============================================================================


def test_ip_coeff_matches_reference_within_fp16() -> None:
    """silica stores ip_coeff as fp16; reference computes fp64. Tolerance
    matches the norm_o test — ip_coeff values cluster around
    ``sqrt(2/pi) ≈ 0.798``, small enough that the ``atol=5e-4`` floor
    catches absolute drift even if ``rtol=5e-3`` loosens near zero."""
    codec = _make_codec()
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode_signs_and_payload(codec, x_fp16)
    silica_ip = np.asarray(payload.ip_coeff).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    _, _, ref_ip = _reference_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
    )

    np.testing.assert_allclose(silica_ip, ref_ip, rtol=5e-3, atol=5e-4)


# =============================================================================
# Test 5 — decoded vector within fp16 tolerance
# =============================================================================


def test_decoded_vector_matches_reference_within_fp16() -> None:
    """Decode algorithm parity under matched payload-storage precision.

    Reference decode applies the same fp16 round-trip on ``norm_o``
    that silica's payload stores (``norm_o_storage_dtype=np.float16``),
    so this test isolates decode algorithm drift — ``rtol=1e-2,
    atol=1e-3`` absorbs the final fp16 cast silica applies before
    returning the tensor, and nothing more.
    """
    codec = _make_codec()
    x_fp16 = _make_shared_fp16_input(seed=0)

    _, payload = _silica_encode_signs_and_payload(codec, x_fp16)
    silica_decoded = np.asarray(codec.decode_tensor(payload)).astype(np.float64)
    silica_decoded_flat = silica_decoded.reshape(_N_VECTORS, _HEAD_DIM)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_signs, ref_norm, _ = _reference_rabitq_quantize_batch(
        x_fp16.astype(np.float64),
        rotation=rotation_ref,
    )
    ref_decoded = _reference_rabitq_decode(
        ref_signs,
        ref_norm,
        rotation=rotation_ref,
        norm_o_storage_dtype=np.float16,
    )

    np.testing.assert_allclose(silica_decoded_flat, ref_decoded, rtol=1e-2, atol=1e-3)


# =============================================================================
# Test 6 — zero-vector batch parity
# =============================================================================


def test_zero_vector_batch_parity() -> None:
    """Zero input on the batch path: both silica and the reference must
    produce ``ip_coeff = 0``, ``norm_o = 0``, and all-``+1`` signs (the
    tie-break direction). Pins the batch-path semantics that
    test_rabitq_1bit.py already locks on silica's side, against the
    independent reference."""
    codec = _make_codec()
    x_zero_fp16 = np.zeros((_N_VECTORS, _HEAD_DIM), dtype=np.float16)

    silica_signs, payload = _silica_encode_signs_and_payload(codec, x_zero_fp16)
    silica_norm = np.asarray(payload.norm_o).astype(np.float64)
    silica_ip = np.asarray(payload.ip_coeff).astype(np.float64)

    rotation_ref = _reference_haar_rotation(_HEAD_DIM, _SEED)
    ref_signs, ref_norm, ref_ip = _reference_rabitq_quantize_batch(
        x_zero_fp16.astype(np.float64),
        rotation=rotation_ref,
    )

    # Reference self-consistency: zero-vector batch-path semantics.
    np.testing.assert_array_equal(ref_signs, np.ones_like(ref_signs))
    np.testing.assert_array_equal(ref_norm, np.zeros(_N_VECTORS, dtype=np.float64))
    np.testing.assert_array_equal(ref_ip, np.zeros(_N_VECTORS, dtype=np.float64))

    # silica agrees with reference.
    np.testing.assert_array_equal(silica_signs, ref_signs)
    np.testing.assert_array_equal(silica_norm, ref_norm)
    np.testing.assert_array_equal(silica_ip, ref_ip)
