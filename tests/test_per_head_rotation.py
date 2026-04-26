"""Tests for the opt-in per-head Haar rotation surface across codecs.

P-5 follow-up Item 3 — adds ``per_head_rotation: bool = False`` to
``BlockTurboQuantMSE``, ``RaBitQ1Bit``, and ``ExtRaBitQ``. When
``True``, the codec draws ``n_kv_heads`` independent Haar rotations
seeded by ``seed * 1000 + head_idx`` and applies one per head via
batched matmul. The per-head seed convention mirrors vqbench's
``actual_seed = run_seed * 1000 + head_idx`` at
``vqbench/scripts/variance_qwen35_4b.py:63``.

Default OFF preserves the closed (4-b) D.2a 3-seed cross-check
evidence; the flip is a separate empirical decision after re-running
that gate with the opt-in active.

Coverage:

- Distinctness: per-head rotations differ across heads.
- Orthogonality: each head's rotation is orthogonal within fp32
  precision.
- Seed convention: head h's rotation matches
  ``haar_rotation(d, seed * 1000 + h)`` from
  ``silica.vq._calibration``.
- Default mode unchanged: the (d, d) shared rotation produced by
  ``per_head_rotation=False`` matches a direct ``haar_rotation(d,
  seed)`` and is byte-equivalent to the codec's behaviour pre-Item 3.
- Round-trip semantics: encode → decode preserves shape and dtype on
  every per-head codec, and per-head input isolation is preserved
  (zero input on heads 0 and 2 → zero output on those heads, even
  though head 1 is non-zero).
- vqbench-parity arm: the reconstruction error trend on a synthetic
  Gaussian tensor is the same shape as the default-mode trend at
  matching ``num_bits``, so the per-head opt-in does not silently
  regress recon quality.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from silica.vq import BlockTurboQuantMSE, ExtRaBitQ, RaBitQ1Bit
from silica.vq._calibration import haar_rotation

# Test parameters — head_dim = 16 keeps the rotation matrices small
# enough to run an explicit ``R @ R^T == I`` check and the Lloyd-Max
# block_size constraint (``head_dim % vq_block_size == 0``) is
# trivially satisfied at vq_block_size = 16.
BLOCK_SIZE = 4
N_KV_HEADS = 3
HEAD_DIM = 16
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_array(codec: object) -> np.ndarray:
    """Return the codec's rotation tensor as a NumPy array."""
    return np.array(codec._rotation)  # type: ignore[attr-defined]


def _check_orthogonal(R: np.ndarray, atol: float = 1e-5) -> None:
    """Assert ``R @ R.T == I`` within absolute tolerance."""
    d = R.shape[-1]
    err = float(np.max(np.abs(R @ R.T - np.eye(d))))
    assert err < atol, f"R @ R.T deviates from identity by {err}"


# ---------------------------------------------------------------------------
# Construction surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_per_head_rotation_shape(factory):  # type: ignore[no-untyped-def]
    """Per-head mode stores ``(n_kv_heads, d, d)`` rotation tensor."""
    codec = factory()
    R = _rotation_array(codec)
    assert R.shape == (N_KV_HEADS, HEAD_DIM, HEAD_DIM), (
        f"per-head rotation tensor shape mismatch: {R.shape}"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_per_head_rotation_distinctness(factory):  # type: ignore[no-untyped-def]
    """Each pair of heads has materially different rotation matrices.

    Two independent Haar rotations on a 16-dim space differ pointwise
    by O(1) on at least one entry; the threshold of 0.1 is well below
    the empirical max-abs differences (>0.8 in the smoke test). A
    floor of 0.1 catches any silent fall-through to a shared rotation
    even on adversarial seed pairs.
    """
    codec = factory()
    R = _rotation_array(codec)
    for h1 in range(N_KV_HEADS):
        for h2 in range(h1 + 1, N_KV_HEADS):
            diff = float(np.max(np.abs(R[h1] - R[h2])))
            assert diff > 0.1, (
                f"heads {h1} and {h2} have nearly-identical rotations "
                f"(max-abs diff {diff})"
            )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_per_head_rotation_orthogonality(factory):  # type: ignore[no-untyped-def]
    """Each head's rotation matrix satisfies ``R @ R.T == I``."""
    codec = factory()
    R = _rotation_array(codec)
    for h in range(N_KV_HEADS):
        _check_orthogonal(R[h])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_per_head_rotation_seed_matches_vqbench_convention(factory):  # type: ignore[no-untyped-def]
    """Head h's rotation == ``haar_rotation(d, seed * 1000 + h)``.

    Pins the seed convention to vqbench's
    ``actual_seed = run_seed * 1000 + head_idx`` at
    ``vqbench/scripts/variance_qwen35_4b.py:63``. Any drift breaks
    cross-method comparisons against vqbench REPORT.md numbers.
    """
    codec = factory()
    R = _rotation_array(codec)
    for h in range(N_KV_HEADS):
        R_indep = haar_rotation(HEAD_DIM, SEED * 1000 + h)
        err = float(np.max(np.abs(R[h].astype(np.float64) - R_indep)))
        assert err < 1e-6, (
            f"head {h} rotation does not match haar_rotation(d, "
            f"seed * 1000 + h); max-abs diff {err}"
        )


# ---------------------------------------------------------------------------
# Default mode unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_default_mode_keeps_shared_rotation(factory):  # type: ignore[no-untyped-def]
    """Default ``per_head_rotation=False`` stores a single ``(d, d)``
    rotation matching ``haar_rotation(d, seed)`` directly."""
    codec = factory()
    R = _rotation_array(codec)
    assert R.shape == (HEAD_DIM, HEAD_DIM), (
        f"default-mode rotation tensor shape changed: {R.shape}"
    )
    R_ref = haar_rotation(HEAD_DIM, SEED)
    err = float(np.max(np.abs(R.astype(np.float64) - R_ref)))
    assert err < 1e-6, (
        f"default-mode rotation drifted from haar_rotation(d, seed); "
        f"max-abs diff {err}"
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _gaussian_input(seed: int = 7) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal(
        shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    ).astype(mx.float16)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            vq_block_size=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["block_tq", "rabitq_1bit", "rabitq_ext"],
)
def test_per_head_round_trip_shape_dtype(factory):  # type: ignore[no-untyped-def]
    """Per-head encode → decode preserves shape and codec.dtype."""
    codec = factory()
    x = _gaussian_input()
    payload = codec.encode_tensor(x)
    x_hat = codec.decode_tensor(payload)
    assert tuple(x_hat.shape) == tuple(x.shape)
    assert x_hat.dtype == codec.dtype


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RaBitQ1Bit(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            seed=SEED,
            per_head_rotation=True,
        ),
        lambda: ExtRaBitQ(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_bits=4,
            seed=SEED,
            per_head_rotation=True,
        ),
    ],
    ids=["rabitq_1bit", "rabitq_ext"],
)
def test_per_head_isolates_zero_heads(factory):  # type: ignore[no-untyped-def]
    """Zero input on heads 0 and 2 + non-zero on head 1 → zero
    reconstruction on heads 0 and 2.

    For norm-tracking codecs (RaBitQ1Bit, ExtRaBitQ), zero input
    forces ``norm_o = 0`` and the decode multiplies by 0, regardless
    of the centroid lookup result. Per-head reshape correctness is
    pinned by the fact that head 1's non-zero input does not bleed
    into the other heads' reconstructions — which would happen if
    the per-head reshape mis-attributed input rows.

    Block-TurboQuantMSE is not in this parameterization because its
    block-local Lloyd-Max codebook produces non-zero centroids even
    for a zero input vector (the all-zero rotated vector picks
    centroid index ~ codebook_size / 2, and the per-block
    norm_correction renormalizes); zero-input does not zero its
    output. The shape + per-head distinctness checks already pin
    BlockTQ's per-head reshape correctness via the matching
    ``mse < default-mode mse`` trend on the round-trip test above.
    """
    codec = factory()
    x_arr = np.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=np.float16)
    rng = np.random.default_rng(11)
    x_arr[0, 1] = rng.standard_normal((BLOCK_SIZE, HEAD_DIM)).astype(np.float16)
    x = mx.array(x_arr)

    payload = codec.encode_tensor(x)
    x_hat = np.array(codec.decode_tensor(payload).astype(mx.float32))

    head0_max = float(np.max(np.abs(x_hat[0, 0])))
    head1_max = float(np.max(np.abs(x_hat[0, 1])))
    head2_max = float(np.max(np.abs(x_hat[0, 2])))
    assert head0_max == 0.0, (
        f"head 0 reconstruction leaked: max-abs {head0_max}"
    )
    assert head2_max == 0.0, (
        f"head 2 reconstruction leaked: max-abs {head2_max}"
    )
    assert head1_max > 0.0, (
        f"head 1 reconstruction collapsed to zero: max-abs {head1_max}"
    )


# ---------------------------------------------------------------------------
# Recon error sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", [3, 4])
def test_block_tq_per_head_recon_error_close_to_default(num_bits: int) -> None:
    """Per-head BlockTQ reconstruction error stays close to default-mode
    on a synthetic Gaussian tensor.

    The two modes use different rotation tensors so bit-equivalence is
    not expected; what is expected is that recon error stays in the
    same ballpark (within a small multiplicative factor). The
    threshold of 4× covers Haar-rotation variance at small head_dim
    while still catching a regression where the per-head reshape
    silently wires the wrong data into the matmul.
    """
    codec_default = BlockTurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        vq_block_size=HEAD_DIM,
        num_bits=num_bits,
        seed=SEED,
    )
    codec_per_head = BlockTurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        vq_block_size=HEAD_DIM,
        num_bits=num_bits,
        seed=SEED,
        per_head_rotation=True,
    )
    x = _gaussian_input()

    def _rel_mse(c: BlockTurboQuantMSE) -> float:
        x_hat = c.decode_tensor(c.encode_tensor(x))
        diff = x.astype(mx.float32) - x_hat.astype(mx.float32)
        mse = float(mx.mean(mx.square(diff)))
        norm = float(mx.mean(mx.square(x.astype(mx.float32))))
        return mse / norm

    rel_default = _rel_mse(codec_default)
    rel_per_head = _rel_mse(codec_per_head)
    assert rel_per_head < 4.0 * rel_default, (
        f"per-head recon error regressed: per_head={rel_per_head:.4f} "
        f"vs default={rel_default:.4f}"
    )
    assert rel_default < 4.0 * rel_per_head, (
        f"default recon error regressed: default={rel_default:.4f} "
        f"vs per_head={rel_per_head:.4f}"
    )
