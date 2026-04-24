"""Unit tests for :func:`silica.bench.runner._run_ppl`.

Covers the runner's PPL workload driver against the
BatchKVCache-compatible fake adapter + the vendored WikiText fixture
(``tests/fixtures/wikitext2_tiny.txt``). Does not touch real model
weights; the integration test that loads Qwen3-0.6B lives in the
follow-up step-3b commit, cache-gated the same way the decode-speed
tests are.

Two invariants pinned:

1. **fp16 baseline path reachable.** ``workload.kv_codec=None`` and
   ``prefix_cache=False`` routes through
   :func:`teacher_forced_chunked_nll`; ``_run_ppl`` returns a finite
   PPL and the expected ``n_tokens == seq_len - 1``.

2. **Codec path reachable under identity.** ``workload.kv_codec=
   "tq_mse_b4"`` builds a fresh ``RadixPrefixCache`` via
   ``_maybe_build_prefix_cache`` and routes through
   :func:`teacher_forced_chunked_nll_with_codec`. The returned
   ``n_tokens`` matches the baseline; the NLL is close (identity
   tq_mse at 4 bits is not lossless, so equality is not expected —
   but the shape contract is).

Failure modes tested: missing ``wikitext_path``, invalid
``chunk_size``, non-existent fixture file, below-min-tokens floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache

from silica.bench.runner import _check_gates, _run_ppl
from silica.bench.scenario import OracleKind, Scenario, Workload
from silica.models.adapter import KVLayout, ModelAdapter

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wikitext2_tiny.txt"


# =============================================================================
# Fake adapter — same shape as test_ppl_oracle_codec's
# _BatchKVCacheCompatAdapter plus .tokenizer / .kv_layout() for the
# runner helper contract.
# =============================================================================

_VOCAB = 8
_N_KV_HEADS = 2
_HEAD_DIM = 8
_N_LAYERS = 1
# _maybe_build_prefix_cache hardcodes block_size=16; keep chunk_size a
# multiple of 16 for the codec-backed path.
_BLOCK_SIZE = 16


class _BatchKVCacheCompatModel:
    """Cache-aware fake model (same shape as the C.2 test's copy).

    Emits deterministic K/V driven by absolute cache position so
    update_and_fetch populates the cache the way a real mlx-lm
    attention module would, and logits depend on absolute position so
    any cache-lifetime bug in the runner path would surface as a CE
    mismatch.
    """

    VOCAB = _VOCAB
    N_KV_HEADS = _N_KV_HEADS
    HEAD_DIM = _HEAD_DIM

    def __call__(
        self, tokens: mx.array, cache: list[BatchKVCache]
    ) -> mx.array:
        B, T = tokens.shape
        cache_obj = cache[0]
        pre_offset = int(cache_obj.offset[0].item())

        positions = mx.arange(T, dtype=mx.float16) + float(pre_offset)
        k = mx.broadcast_to(
            positions[None, None, :, None],
            (B, self.N_KV_HEADS, T, self.HEAD_DIM),
        ).astype(mx.float16)
        v = (k + mx.array(0.125, dtype=mx.float16)).astype(mx.float16)
        cache_obj.update_and_fetch(k, v)

        import math

        positions_fp32 = (
            mx.arange(T, dtype=mx.float32) + float(pre_offset)
        )
        v_axis = mx.arange(self.VOCAB, dtype=mx.float32)
        pos_scaled = (positions_fp32 + 1.0) / 32.0
        v_scaled = (v_axis + 1.0) / float(self.VOCAB)
        grid = pos_scaled[:, None] * v_scaled[None, :]
        per_pos_logits = mx.cos(grid * math.pi) * 3.0
        logits = mx.broadcast_to(
            per_pos_logits[None, :, :], (B, T, self.VOCAB)
        )
        return logits


class _ByteTokenizer:
    """Tokenizer fake: one int per UTF-8 byte, capped at the fake
    model's vocab size via modulo.

    The fake model declares ``VOCAB = 8``; passing raw byte values
    0-255 as cross-entropy targets against an 8-wide logits axis is
    undefined under ``mx.take_along_axis`` and manifests as huge /
    overflow NLL numbers that vary across test-order permutations.
    The modulo keeps targets in range so CE is numerically stable."""

    def encode(self, text: str) -> list[int]:
        return [b % _VOCAB for b in text.encode("utf-8")]


class _FakePPLAdapter:
    """Adapter fake exposing the four surfaces ``_run_ppl`` consumes:
    ``_model``, ``tokenizer``, ``kv_layout()``, and
    ``make_batch_cache(left_padding)``.
    """

    def __init__(self) -> None:
        self._model = _BatchKVCacheCompatModel()
        self._tokenizer = _ByteTokenizer()
        self._kv_layout = KVLayout(
            num_layers=_N_LAYERS,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=mx.float16,
        )

    def tokenizer(self) -> Any:
        """Method (not property) to match the real
        ``ModelAdapter.tokenizer`` Protocol shape — `_run_ppl` calls
        ``adapter.tokenizer()``."""
        return self._tokenizer

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def make_batch_cache(
        self, left_padding: list[int]
    ) -> list[BatchKVCache]:
        assert left_padding == [0]
        return [
            BatchKVCache(left_padding=list(left_padding))
            for _ in range(_N_LAYERS)
        ]


# =============================================================================
# Scenario builders
# =============================================================================


def _make_scenario(
    *,
    kv_codec: str | None = None,
    wikitext_path: str | None = None,
    chunk_size: int = _BLOCK_SIZE,
    max_tokens: int = 64,
    min_scored_tokens: int = 1,
) -> Scenario:
    wl = Workload(
        name=f"wikitext-ppl-{kv_codec or 'fp16'}",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=kv_codec is not None,
        temperature=0.0,
        top_p=1.0,
        kv_codec=kv_codec,
    )
    oracle_cfg: dict[str, Any] = {
        "wikitext_path": wikitext_path if wikitext_path is not None
        else str(_FIXTURE_PATH),
        "chunk_size": chunk_size,
        "max_tokens": max_tokens,
        "min_scored_tokens": min_scored_tokens,
    }
    return Scenario(
        id=f"test-wikitext-ppl-{kv_codec or 'fp16'}",
        repo="dummy/repo",
        workload=wl,
        oracle=OracleKind.PPL,
        oracle_config=oracle_cfg,
    )


def _call_run_ppl(
    scenario: Scenario, adapter: _FakePPLAdapter
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Wrap ``_run_ppl`` behind a ``cast(ModelAdapter, ...)`` so the
    fake adapter (which implements only the subset of the
    ``ModelAdapter`` Protocol that ``_run_ppl`` actually calls)
    satisfies mypy at the test-call boundary."""
    return _run_ppl(scenario, cast(ModelAdapter, adapter))


# =============================================================================
# Tests
# =============================================================================


class TestFp16BaselinePath:
    """No codec, no prefix cache — routes through
    ``teacher_forced_chunked_nll`` (C.1 oracle)."""

    def test_returns_finite_ppl_and_expected_token_count(self) -> None:
        scenario = _make_scenario(
            kv_codec=None, max_tokens=64, chunk_size=_BLOCK_SIZE
        )
        adapter = _FakePPLAdapter()
        collected, context = _call_run_ppl(scenario, adapter)

        # Tokenized length = min(fixture_bytes, 64) = 64.
        # Oracle scores seq_len - 1 = 63 tokens.
        assert collected["n_tokens"] == 63
        assert collected["nll_sum"] > 0.0
        import math

        assert math.isfinite(collected["ppl"])
        assert collected["ppl"] > 0.0
        assert context["kv_codec"] is None
        assert context["chunk_size"] == _BLOCK_SIZE
        assert context["max_tokens"] == 64
        assert context["wikitext_path"] == str(_FIXTURE_PATH)


class TestCodecPath:
    """``kv_codec`` set — routes through
    ``teacher_forced_chunked_nll_with_codec``. ``_maybe_build_prefix_cache``
    builds a fresh ``RadixPrefixCache`` wrapping a ``SyntheticPrefixBlockStore``
    whose codecs come from the ``tq_mse_b4`` registry entry."""

    def test_tq_mse_b4_returns_finite_ppl_and_same_token_count(
        self,
    ) -> None:
        scenario = _make_scenario(
            kv_codec="tq_mse_b4",
            max_tokens=64,
            chunk_size=_BLOCK_SIZE,
        )
        adapter = _FakePPLAdapter()
        collected, context = _call_run_ppl(scenario, adapter)

        import math

        assert collected["n_tokens"] == 63
        assert math.isfinite(collected["ppl"])
        assert collected["ppl"] > 0.0
        assert context["kv_codec"] == "tq_mse_b4"


class TestInputValidation:
    def test_missing_wikitext_path_raises_runtime_error(self) -> None:
        adapter = _FakePPLAdapter()
        wl = Workload(
            name="wikitext-ppl-no-path",
            prompts=(),
            max_tokens=0,
            max_batch_size=1,
            prefix_cache=False,
            temperature=0.0,
            top_p=1.0,
            kv_codec=None,
        )
        scenario = Scenario(
            id="test-no-path",
            repo="dummy/repo",
            workload=wl,
            oracle=OracleKind.PPL,
            oracle_config={"chunk_size": 16},  # no wikitext_path
        )
        with pytest.raises(RuntimeError, match="wikitext_path"):
            _call_run_ppl(scenario, adapter)

    def test_non_existent_wikitext_path_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "not_here.txt"
        scenario = _make_scenario(wikitext_path=str(missing))
        adapter = _FakePPLAdapter()
        with pytest.raises(FileNotFoundError):
            _call_run_ppl(scenario, adapter)

    def test_below_min_tokens_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """The loader's ``min_tokens`` floor equals ``chunk_size``;
        a tokenized length shorter than that raises."""
        # Small fixture with fewer bytes than chunk_size.
        small = tmp_path / "small.txt"
        small.write_text("hi", encoding="utf-8")
        scenario = _make_scenario(
            wikitext_path=str(small), chunk_size=_BLOCK_SIZE
        )
        adapter = _FakePPLAdapter()
        with pytest.raises(ValueError, match="below min_tokens"):
            _call_run_ppl(scenario, adapter)

    def test_zero_chunk_size_rejected(self) -> None:
        adapter = _FakePPLAdapter()
        scenario = _make_scenario(chunk_size=0)
        with pytest.raises(RuntimeError, match="chunk_size"):
            _call_run_ppl(scenario, adapter)


class TestCheckGatesWikitextPath:
    """``_check_gates`` must skip ``OracleKind.PPL`` rows when the
    ``wikitext_path`` is missing from disk, before the engine
    factory runs. Loading ~600 MB of Qwen3-0.6B weights only to
    fail on a missing tokenizer input is wasted work; the gate
    catches the common "forgot to populate the wikitext cache"
    case up front."""

    def test_skips_when_wikitext_path_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stand up a faked HF hub cache so the model-cache gate passes.
        monkeypatch.setattr(
            Path, "home", classmethod(lambda cls: tmp_path)
        )
        (tmp_path / ".cache" / "huggingface" / "hub"
         / "models--dummy--repo").mkdir(parents=True)

        scenario = _make_scenario(
            wikitext_path=str(tmp_path / "absent.txt")
        )
        scenario_with_real_repo = Scenario(
            id=scenario.id,
            repo="dummy/repo",
            workload=scenario.workload,
            oracle=scenario.oracle,
            oracle_config=scenario.oracle_config,
        )
        reason = _check_gates(scenario_with_real_repo)
        assert reason is not None
        assert reason.startswith("wikitext_cache_missing:")

    def test_skips_when_oracle_config_has_no_wikitext_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Path, "home", classmethod(lambda cls: tmp_path)
        )
        (tmp_path / ".cache" / "huggingface" / "hub"
         / "models--dummy--repo").mkdir(parents=True)

        wl = Workload(
            name="wikitext-ppl-no-path",
            prompts=(),
            max_tokens=0,
            max_batch_size=1,
            prefix_cache=False,
            temperature=0.0,
            top_p=1.0,
            kv_codec=None,
        )
        scenario = Scenario(
            id="test-no-path",
            repo="dummy/repo",
            workload=wl,
            oracle=OracleKind.PPL,
            oracle_config={},
        )
        reason = _check_gates(scenario)
        assert reason == "ppl_wikitext_path_missing_in_oracle_config"

    def test_passes_when_wikitext_path_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Path, "home", classmethod(lambda cls: tmp_path)
        )
        (tmp_path / ".cache" / "huggingface" / "hub"
         / "models--dummy--repo").mkdir(parents=True)

        scenario = _make_scenario(wikitext_path=str(_FIXTURE_PATH))
        scenario_with_real_repo = Scenario(
            id=scenario.id,
            repo="dummy/repo",
            workload=scenario.workload,
            oracle=scenario.oracle,
            oracle_config=scenario.oracle_config,
        )
        reason = _check_gates(scenario_with_real_repo)
        assert reason is None


class TestContextForwarding:
    def test_context_carries_all_runner_populated_keys(self) -> None:
        scenario = _make_scenario(
            kv_codec="tq_mse_b4", chunk_size=16, max_tokens=32
        )
        adapter = _FakePPLAdapter()
        _, context = _call_run_ppl(scenario, adapter)

        expected_keys = {
            "chunk_size",
            "max_tokens",
            "wikitext_path",
            "kv_codec",
            # P-5-D.2a: codec_quality_path always lands in metadata,
            # even on the fp16 / default path — bench report readers
            # should never have to branch on "is this key present?"
            "codec_quality_path",
        }
        assert expected_keys.issubset(context.keys())
        assert context["chunk_size"] == 16
        assert context["max_tokens"] == 32
        assert context["kv_codec"] == "tq_mse_b4"


class TestCodecQualityPathDispatcher:
    """P-5-D.2a — ``codec_quality_path`` selects between the C.2
    prefix-cache-store path and the D.2a projection-patch path.

    The fp16 fake model in this file does not expose
    ``.layers[i].self_attn.k_proj`` (the projection-patch surface),
    so the ``vqbench_aligned`` branch is exercised via a monkey-
    patched recording spy rather than a real forward through the
    fake. The spy asserts that dispatch lands on the correct oracle
    function and forwards the expected codec_factory + seed /
    wrap_v kwargs.
    """

    def test_default_quality_path_is_prefix_store_post_rope(
        self,
    ) -> None:
        """No ``codec_quality_path`` key in oracle_config → default
        ``"prefix_store_post_rope"`` (legacy C.2 behaviour)."""
        scenario = _make_scenario(kv_codec=None, chunk_size=_BLOCK_SIZE)
        adapter = _FakePPLAdapter()
        _, context = _call_run_ppl(scenario, adapter)
        assert context["codec_quality_path"] == "prefix_store_post_rope"

    def test_explicit_prefix_store_post_rope(self) -> None:
        scenario = _make_scenario(kv_codec="tq_mse_b4", chunk_size=_BLOCK_SIZE)
        # Mutate oracle_config in place — the scenario builder above
        # does not surface this key yet.
        scenario.oracle_config["codec_quality_path"] = "prefix_store_post_rope"
        adapter = _FakePPLAdapter()
        _, context = _call_run_ppl(scenario, adapter)
        assert context["codec_quality_path"] == "prefix_store_post_rope"

    def test_unknown_quality_path_raises(self) -> None:
        scenario = _make_scenario(kv_codec=None, chunk_size=_BLOCK_SIZE)
        scenario.oracle_config["codec_quality_path"] = "not_a_real_path"
        adapter = _FakePPLAdapter()
        with pytest.raises(
            RuntimeError,
            match="codec_quality_path.*not a known path",
        ):
            _call_run_ppl(scenario, adapter)

    def test_vqbench_aligned_routes_to_projection_patch_oracle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``codec_quality_path="vqbench_aligned"`` calls
        :func:`teacher_forced_chunked_nll_vqbench_aligned` (recorded
        via spy) instead of the prefix-cache-store oracle."""
        from silica.bench import ppl_oracle as ppl_oracle_module

        recorded: dict[str, Any] = {}

        def spy(
            adapter: Any,
            token_ids: mx.array,
            *,
            chunk_size: int,
            codec_factory: Any,
            seed: int,
            wrap_v: bool,
        ) -> tuple[float, int]:
            recorded["called"] = True
            recorded["chunk_size"] = chunk_size
            recorded["seed"] = seed
            recorded["wrap_v"] = wrap_v
            recorded["codec_factory"] = codec_factory
            # Return a fixed NLL to keep the oracle contract; values
            # are irrelevant for this dispatch assertion.
            return 1.0, 5

        # Patch both the module attribute (for external callers) and
        # the inline import inside _run_ppl. The inline import is what
        # _run_ppl actually calls, so that's the load-bearing patch.
        monkeypatch.setattr(
            ppl_oracle_module,
            "teacher_forced_chunked_nll_vqbench_aligned",
            spy,
        )

        scenario = _make_scenario(
            kv_codec="block_tq_b64_b4", chunk_size=_BLOCK_SIZE
        )
        scenario.oracle_config["codec_quality_path"] = "vqbench_aligned"
        adapter = _FakePPLAdapter()

        collected, context = _call_run_ppl(scenario, adapter)

        assert recorded.get("called") is True, (
            "vqbench-aligned oracle was not invoked"
        )
        assert recorded["chunk_size"] == _BLOCK_SIZE
        assert recorded["seed"] == 42  # _run_ppl default seed
        assert recorded["wrap_v"] is True
        # The recorded factory must be the block_tq_b64_b4 factory
        # from the registry. Identity check — not a smoke call —
        # because block_tq_b64_b4's ``vq_block_size=64`` does not
        # divide this fake's ``head_dim=8``, so the factory would
        # raise on a layout-shaped invocation here. The identity
        # assertion is the load-bearing part: dispatch routed the
        # correct registered factory through to the oracle.
        from silica.bench.codec_registry import get_codec_spec

        expected_factory = get_codec_spec("block_tq_b64_b4").factory
        assert recorded["codec_factory"] is expected_factory

        # context carries the path label for metadata.
        assert context["codec_quality_path"] == "vqbench_aligned"
        # collected wrapped the spy's (nll_sum, n_tokens) → ppl.
        assert collected["nll_sum"] == 1.0
        assert collected["n_tokens"] == 5
        import math

        assert math.isclose(
            collected["ppl"], math.exp(1.0 / 5.0), rel_tol=1e-9
        )

    def test_vqbench_aligned_asymmetric_codec_rejected(self) -> None:
        """K-only codecs (``rabitq_b1``) cannot install symmetrically
        on the projection-patch path — the wrap_v=True default would
        force a K-only codec onto V."""
        scenario = _make_scenario(
            kv_codec="rabitq_b1", chunk_size=_BLOCK_SIZE
        )
        scenario.oracle_config["codec_quality_path"] = "vqbench_aligned"
        adapter = _FakePPLAdapter()
        with pytest.raises(RuntimeError, match="symmetric codec"):
            _call_run_ppl(scenario, adapter)
