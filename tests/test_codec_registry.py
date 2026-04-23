"""Tests for silica.bench.codec_registry — CodecSpec schema + registry entries.

Covers the P-5-A.1b deliverable from ``docs/P5_OPENING.md`` §6.1 +
§8 P-5-A.1:

- Registry schema: every entry's ``id`` matches its dict key; ids are
  unique; the expected P-5-A.1 set is present (``fp16`` + scalar TQ +
  Block TQ entries).
- Exactly one ``production_recommended`` entry (``block_tq_b64_b4``
  per vqbench REPORT §3.1).
- Factory smoke: every registered factory returns a working
  ``VectorCodec`` and round-trips a canonical input tensor without
  shape / dtype drift.
- ``bits_per_value`` accounting: stored nominal matches the exact
  formula for BlockTQ (``num_bits + 16 / vq_block_size``);
  ``effective_bits_per_value(head_dim)`` correctly adds the fp16
  per-vector-scale overhead for scalar TQ; fp16 returns 16.
- Lookup / error behaviour: unknown id raises ``KeyError`` with
  available-id list; ``list_codec_ids`` is sorted and stable.
- ``CodecSpec`` is frozen (no post-construction mutation).
- Compression ratio for the production recommendation lands in the
  vqbench-REPORT range (16 / 4.25 ≈ 3.76×).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import mlx.core as mx
import pytest

from silica.bench.codec_registry import (
    CODEC_REGISTRY,
    CodecSpec,
    get_codec_spec,
    list_codec_ids,
)
from silica.kvcache.codec import VectorCodec
from silica.vq import BlockTurboQuantMSE, ExtRaBitQ, RaBitQ1Bit

# Shared shape defaults for factory-smoke tests.
BLOCK_SIZE = 16
N_KV_HEADS = 4
HEAD_DIM = 64


def _canonical_input() -> mx.array:
    """One side of one detached block at the test default shape."""
    mx.random.seed(0)
    return mx.random.normal(
        shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
    ).astype(mx.float16)


# ---------------------------------------------------------------------------
# Registry schema
# ---------------------------------------------------------------------------


def test_all_registered_ids_match_spec_ids() -> None:
    for key, spec in CODEC_REGISTRY.items():
        assert spec.id == key, (
            f"dict key {key!r} disagrees with spec.id {spec.id!r}"
        )


def test_codec_ids_are_unique() -> None:
    ids = [s.id for s in CODEC_REGISTRY.values()]
    assert len(ids) == len(set(ids))


def test_expected_codec_ids_present() -> None:
    expected = {
        "fp16",
        "tq_mse_b3",
        "tq_mse_b4",
        "block_tq_b32_b3",
        "block_tq_b32_b4",
        "block_tq_b64_b3",
        "block_tq_b64_b4",
        "rabitq_b1",
        "ext_rabitq_b2",
        "ext_rabitq_b3",
        "ext_rabitq_b4",
    }
    assert set(CODEC_REGISTRY.keys()) == expected


def test_exactly_one_production_recommended() -> None:
    """vqbench REPORT §3.1 pins Block B=64 4-bit K+V as the single
    production recommendation; all other entries are bench-row only."""
    prod = [s for s in CODEC_REGISTRY.values() if s.production_recommended]
    assert len(prod) == 1
    assert prod[0].id == "block_tq_b64_b4"


def test_fp16_baseline_is_uncompressed() -> None:
    spec = get_codec_spec("fp16")
    assert spec.family == "fp16"
    assert spec.bits_per_value == 16.0
    assert not spec.payload_packed
    assert not spec.requires_fit
    assert not spec.production_recommended


def test_families_partition_by_prefix() -> None:
    """id prefix ↔ family mapping. Order matters: ``ext_rabitq_*`` must
    match before ``rabitq_*`` since ``rabitq`` is a proper prefix of
    ``ext_rabitq_*`` would NOT be — but the tests use ``startswith`` so
    the more-specific prefix checks first."""
    for spec in CODEC_REGISTRY.values():
        if spec.id == "fp16":
            assert spec.family == "fp16"
        elif spec.id.startswith("tq_mse"):
            assert spec.family == "tq_mse"
        elif spec.id.startswith("block_tq"):
            assert spec.family == "block_tq"
        elif spec.id.startswith("ext_rabitq"):
            assert spec.family == "ext_rabitq"
        elif spec.id.startswith("rabitq"):
            assert spec.family == "rabitq"
        else:
            pytest.fail(f"unclassified id {spec.id!r}")


def test_symmetric_entries_support_both_sides() -> None:
    """All P-5-A.1 entries are symmetric (both K and V). P-5-B lands
    ``rabitq_b1`` as the first K-only codec; asymmetric specs are
    exempt here and must independently pass the
    ``_maybe_build_prefix_cache`` symmetry guard (covered in
    test_bench_workload_kv_codec.py) so the ``kv_codec=`` shorthand
    refuses to install them on both sides."""
    asymmetric_ids = {"rabitq_b1"}
    for spec in CODEC_REGISTRY.values():
        if spec.id in asymmetric_ids:
            continue
        assert spec.k_supported, spec.id
        assert spec.v_supported, spec.id


def test_rabitq_b1_is_k_only() -> None:
    """``rabitq_b1`` must declare ``v_supported=False``. The estimator-
    native attention path the ``ip_coeff`` field feeds lives on K; a
    V-side RaBitQ1Bit would waste the ip_coeff storage without ever
    consuming it. The symmetry guard in ``_maybe_build_prefix_cache``
    enforces this at installation time; the spec here declares the
    intent."""
    spec = get_codec_spec("rabitq_b1")
    assert spec.k_supported
    assert not spec.v_supported


def test_no_entry_requires_fit() -> None:
    """Analytical calibration across the whole P-5-A.1 + P-5-B catalog:
    BlockTQ + scalar TQ use Haar rotation + Lloyd-Max (Gaussian-based);
    RaBitQ1Bit uses Haar rotation + zero centroid (P-5-B §5.3). No
    entry needs a real-activation fit pass."""
    for spec in CODEC_REGISTRY.values():
        assert not spec.requires_fit, spec.id


# ---------------------------------------------------------------------------
# Factory smoke — every registered factory round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec_id", list_codec_ids())
def test_factory_returns_vector_codec(codec_id: str) -> None:
    spec = get_codec_spec(codec_id)
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    assert isinstance(codec, VectorCodec)


@pytest.mark.parametrize("codec_id", list_codec_ids())
def test_factory_round_trips(codec_id: str) -> None:
    spec = get_codec_spec(codec_id)
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    x = _canonical_input()
    payload = codec.encode_tensor(x)
    out = codec.decode_tensor(payload)
    assert out.shape == x.shape
    assert out.dtype == mx.float16


@pytest.mark.parametrize("codec_id", list_codec_ids())
def test_factory_honours_dtype_override(codec_id: str) -> None:
    """Every factory accepts ``dtype=bf16`` and produces a codec whose
    ``decode_tensor`` returns bf16. Gemma-style K/V layouts use bf16."""
    spec = get_codec_spec(codec_id)
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=mx.bfloat16,
    )
    x = mx.random.normal(
        shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
    ).astype(mx.bfloat16)
    out = codec.decode_tensor(codec.encode_tensor(x))
    assert out.dtype == mx.bfloat16


def test_scalar_tq_factory_yields_block_tq_with_b_equals_d() -> None:
    """Scalar TurboQuantMSE is aliased to BlockTurboQuantMSE with
    vq_block_size = head_dim."""
    spec = get_codec_spec("tq_mse_b4")
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    assert isinstance(codec, BlockTurboQuantMSE)
    assert codec._vq_block_size == HEAD_DIM
    assert codec._num_vq_blocks == 1


def test_rabitq_b1_factory_yields_rabitq_1bit() -> None:
    """``rabitq_b1`` factory must produce a ``RaBitQ1Bit`` instance —
    not a BlockTQ alias or a TurboQuant variant. RaBitQ1Bit's own
    constructor enforces ``num_bits=1`` via the "1-bit-only" ValueError,
    so the factory cannot silently emit a multi-bit codec; isinstance
    alone pins the family."""
    spec = get_codec_spec("rabitq_b1")
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    assert isinstance(codec, RaBitQ1Bit)


@pytest.mark.parametrize(
    "codec_id,expected_num_bits",
    [
        ("ext_rabitq_b2", 2),
        ("ext_rabitq_b3", 3),
        ("ext_rabitq_b4", 4),
    ],
)
def test_ext_rabitq_factory_yields_ext_rabitq(
    codec_id: str, expected_num_bits: int
) -> None:
    """Each ``ext_rabitq_b{B}`` factory must produce an ``ExtRaBitQ``
    instance with the correct ``num_bits``. ExtRaBitQ's constructor
    enforces ``num_bits in {2, 3, 4}`` so a factory that threaded the
    wrong bit count would raise at construction; this test also pins
    the bit count to the id suffix so a future edit cannot silently
    re-wire ``ext_rabitq_b3`` to instantiate a B=2 codec."""
    spec = get_codec_spec(codec_id)
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    assert isinstance(codec, ExtRaBitQ)
    assert codec._num_bits == expected_num_bits


def test_ext_rabitq_family_symmetric() -> None:
    """All ``ext_rabitq_b{2,3,4}`` entries must declare both
    ``k_supported`` and ``v_supported`` True. Unlike ``rabitq_b1``
    (K-only), multi-bit ExtRaBitQ has reconstruction MSE competitive
    with BlockTQ on V; the symmetric kv_codec= shorthand is valid."""
    for codec_id in ("ext_rabitq_b2", "ext_rabitq_b3", "ext_rabitq_b4"):
        spec = get_codec_spec(codec_id)
        assert spec.family == "ext_rabitq", spec.id
        assert spec.k_supported, spec.id
        assert spec.v_supported, spec.id


# ---------------------------------------------------------------------------
# bits_per_value arithmetic
# ---------------------------------------------------------------------------


def test_fp16_bits_per_value_is_16() -> None:
    spec = get_codec_spec("fp16")
    assert spec.bits_per_value == 16.0
    for head_dim in (64, 128, 256):
        assert spec.effective_bits_per_value(head_dim=head_dim) == 16.0


@pytest.mark.parametrize(
    "codec_id,expected",
    [
        ("block_tq_b32_b3", 3.0 + 16.0 / 32),  # 3.5
        ("block_tq_b32_b4", 4.0 + 16.0 / 32),  # 4.5
        ("block_tq_b64_b3", 3.0 + 16.0 / 64),  # 3.25
        ("block_tq_b64_b4", 4.0 + 16.0 / 64),  # 4.25
    ],
)
def test_block_tq_bits_per_value_matches_formula(
    codec_id: str, expected: float
) -> None:
    spec = get_codec_spec(codec_id)
    assert spec.bits_per_value == expected
    # Effective bits/value is head-dim-independent for BlockTQ.
    for head_dim in (64, 128, 256):
        assert spec.effective_bits_per_value(head_dim=head_dim) == expected


@pytest.mark.parametrize(
    "codec_id,num_bits",
    [("tq_mse_b3", 3), ("tq_mse_b4", 4)],
)
def test_scalar_tq_nominal_vs_effective_bits(
    codec_id: str, num_bits: int
) -> None:
    """Scalar TQ stores the nominal num_bits as bits_per_value; the
    effective value adds the ``16 / head_dim`` per-vector scale
    overhead."""
    spec = get_codec_spec(codec_id)
    assert spec.bits_per_value == float(num_bits)
    # head_dim=128: effective = num_bits + 0.125
    assert spec.effective_bits_per_value(head_dim=128) == pytest.approx(
        num_bits + 16.0 / 128
    )
    # head_dim=64 (larger overhead)
    assert spec.effective_bits_per_value(head_dim=64) == pytest.approx(
        num_bits + 16.0 / 64
    )


def test_rabitq_b1_nominal_bits_per_value_is_one() -> None:
    """``rabitq_b1`` stores 1 nominal bit per coordinate. The fp16
    ``norm_o`` + fp16 ``ip_coeff`` metadata pair is amortized into
    :meth:`effective_bits_per_value` and is not counted here."""
    spec = get_codec_spec("rabitq_b1")
    assert spec.bits_per_value == 1.0


@pytest.mark.parametrize(
    "head_dim,expected",
    [
        (64, 1.0 + 32.0 / 64),   # 1.5
        (128, 1.0 + 32.0 / 128),  # 1.25
        (256, 1.0 + 32.0 / 256),  # 1.125
    ],
)
def test_rabitq_b1_effective_bits_per_value_matches_formula(
    head_dim: int, expected: float
) -> None:
    """Effective = nominal + 32/head_dim for the RaBitQ family. 32
    comes from two per-vector fp16 scalars (``norm_o`` + ``ip_coeff``)
    amortized over ``head_dim`` coordinates."""
    spec = get_codec_spec("rabitq_b1")
    assert spec.effective_bits_per_value(head_dim=head_dim) == pytest.approx(
        expected
    )


def test_rabitq_b1_effective_matches_silica_resident_bytes() -> None:
    """Cross-check: the registry's effective_bits_per_value must match
    the byte accounting on the codec itself (``RaBitQ1Bit.resident_bytes``
    and ``.logical_bytes``). If the two drift, either the registry
    lies or the codec does; this test catches either."""
    spec = get_codec_spec("rabitq_b1")
    for head_dim in (64, 128, 256):
        codec = spec.factory(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=head_dim,
        )
        # 1 block's physical bytes / fp16-equivalent baseline bytes,
        # scaled up to per-coordinate bits (× 8 bits/byte, scaled by
        # fp16 overhead ratio).
        num_tokens = BLOCK_SIZE
        resident = codec.resident_bytes(num_blocks=1)
        logical_fp16 = codec.logical_bytes(num_tokens=num_tokens)
        effective = spec.effective_bits_per_value(head_dim=head_dim)
        # fp16 baseline is 16 bits/coord; compression ratio = 16 / effective.
        assert resident / logical_fp16 == pytest.approx(
            effective / 16.0, rel=1e-6
        )


@pytest.mark.parametrize(
    "codec_id,num_bits",
    [
        ("ext_rabitq_b2", 2),
        ("ext_rabitq_b3", 3),
        ("ext_rabitq_b4", 4),
    ],
)
def test_ext_rabitq_nominal_bits_per_value_equals_num_bits(
    codec_id: str, num_bits: int
) -> None:
    """Each ``ext_rabitq_b{B}`` stores nominal ``B`` bits per coord.
    The fp16 metadata overhead (norm_o + ip_coeff + scale) amortizes
    into :meth:`effective_bits_per_value` and is not counted here."""
    spec = get_codec_spec(codec_id)
    assert spec.bits_per_value == float(num_bits)


@pytest.mark.parametrize(
    "codec_id,num_bits,head_dim,expected",
    [
        # effective = num_bits + 48/head_dim (three fp16 scalars per vector).
        ("ext_rabitq_b2", 2, 64, 2.0 + 48.0 / 64),   # 2.75
        ("ext_rabitq_b2", 2, 128, 2.0 + 48.0 / 128),  # 2.375
        ("ext_rabitq_b2", 2, 256, 2.0 + 48.0 / 256),  # 2.1875
        ("ext_rabitq_b3", 3, 64, 3.0 + 48.0 / 64),    # 3.75
        ("ext_rabitq_b3", 3, 128, 3.0 + 48.0 / 128),  # 3.375
        ("ext_rabitq_b3", 3, 256, 3.0 + 48.0 / 256),  # 3.1875
        ("ext_rabitq_b4", 4, 64, 4.0 + 48.0 / 64),    # 4.75
        ("ext_rabitq_b4", 4, 128, 4.0 + 48.0 / 128),  # 4.375
        ("ext_rabitq_b4", 4, 256, 4.0 + 48.0 / 256),  # 4.1875
    ],
)
def test_ext_rabitq_effective_bits_per_value_matches_formula(
    codec_id: str, num_bits: int, head_dim: int, expected: float
) -> None:
    """Effective = nominal + 48/head_dim for the ext_rabitq family.
    48 comes from three per-vector fp16 scalars (``norm_o`` +
    ``ip_coeff`` + ``scale``) amortized over ``head_dim`` coordinates —
    one more fp16 than the ``rabitq`` family's 32/head_dim."""
    spec = get_codec_spec(codec_id)
    assert spec.effective_bits_per_value(head_dim=head_dim) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "codec_id", ["ext_rabitq_b2", "ext_rabitq_b3", "ext_rabitq_b4"]
)
def test_ext_rabitq_effective_matches_silica_resident_bytes(
    codec_id: str,
) -> None:
    """Cross-check: the registry's effective_bits_per_value must match
    ``ExtRaBitQ.resident_bytes / .logical_bytes`` at every supported
    head_dim. If the two drift (registry says 48/head_dim but the
    codec stores only 32 or 64 fp16-bits of metadata, or vice versa),
    this test catches it."""
    spec = get_codec_spec(codec_id)
    for head_dim in (64, 128, 256):
        codec = spec.factory(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=head_dim,
        )
        num_tokens = BLOCK_SIZE
        resident = codec.resident_bytes(num_blocks=1)
        logical_fp16 = codec.logical_bytes(num_tokens=num_tokens)
        effective = spec.effective_bits_per_value(head_dim=head_dim)
        assert resident / logical_fp16 == pytest.approx(
            effective / 16.0, rel=1e-6
        )


def test_effective_bits_per_value_rejects_unknown_family() -> None:
    bogus = CodecSpec(
        id="bogus",
        family="not_a_real_family",
        bits_per_value=4.0,
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        production_recommended=False,
        factory=lambda **_k: pytest.fail("factory should not be called"),
    )
    with pytest.raises(ValueError, match="unknown codec family"):
        bogus.effective_bits_per_value(head_dim=64)


@pytest.mark.parametrize("bad_head_dim", [0, -1, -128])
def test_effective_bits_per_value_rejects_nonpositive_head_dim(
    bad_head_dim: int,
) -> None:
    """head_dim appears in the denominator for tq_mse, rabitq, and
    ext_rabitq; 0 would ZeroDivisionError and a negative value would
    produce a meaningless negative bits/value. The guard fires before
    any family branch so even fp16 / block_tq (head-dim-independent
    families) reject a malformed head_dim consistently."""
    for codec_id in (
        "fp16",
        "tq_mse_b4",
        "block_tq_b64_b4",
        "rabitq_b1",
        "ext_rabitq_b4",
    ):
        spec = get_codec_spec(codec_id)
        with pytest.raises(ValueError, match="head_dim must be a positive integer"):
            spec.effective_bits_per_value(head_dim=bad_head_dim)


# ---------------------------------------------------------------------------
# Production-recommendation compression ratio
# ---------------------------------------------------------------------------


def test_block_tq_b64_b4_compression_ratio_matches_report_headline() -> None:
    """vqbench REPORT §3.1 headline: Block B=64 4-bit K+V delivers
    3.76× total-KV compression. Since both K and V use the same codec
    at this config, per-side ratio equals total-KV ratio.

    Our stored bits_per_value = 4.25 → 16 / 4.25 = 3.7647... ≈ 3.76×.
    """
    spec = get_codec_spec("block_tq_b64_b4")
    ratio = 16.0 / spec.bits_per_value
    assert ratio == pytest.approx(16.0 / 4.25)
    assert 3.7 < ratio < 3.8


# ---------------------------------------------------------------------------
# Lookup / error behaviour
# ---------------------------------------------------------------------------


def test_get_codec_spec_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown codec id"):
        get_codec_spec("does_not_exist")


def test_get_codec_spec_error_lists_available_ids() -> None:
    try:
        get_codec_spec("does_not_exist")
    except KeyError as e:
        msg = str(e)
        # Error should include the set of valid ids so the caller sees
        # alternatives inline rather than hunting through docs.
        for cid in list_codec_ids():
            assert cid in msg


def test_list_codec_ids_is_sorted() -> None:
    ids = list_codec_ids()
    assert ids == sorted(ids)


def test_list_codec_ids_matches_registry() -> None:
    assert set(list_codec_ids()) == set(CODEC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# CodecSpec frozen-dataclass invariant
# ---------------------------------------------------------------------------


def test_codec_spec_is_frozen() -> None:
    spec = get_codec_spec("fp16")
    with pytest.raises(FrozenInstanceError):
        spec.id = "something_else"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.production_recommended = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Determinism — repeated factory calls produce equivalent codecs
# ---------------------------------------------------------------------------


def test_repeated_factory_calls_produce_equivalent_codecs() -> None:
    """Same spec + same args → same Haar rotation (cached in
    silica.vq._calibration) → bit-identical round-trip on the same
    input. Regression-locks that the registry does not accidentally
    randomize across factory invocations."""
    spec = get_codec_spec("block_tq_b64_b4")
    c1 = spec.factory(
        block_size=BLOCK_SIZE, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM
    )
    c2 = spec.factory(
        block_size=BLOCK_SIZE, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM
    )
    x = _canonical_input()
    out1 = c1.decode_tensor(c1.encode_tensor(x))
    out2 = c2.decode_tensor(c2.encode_tensor(x))
    assert bool(mx.array_equal(out1, out2).item())
