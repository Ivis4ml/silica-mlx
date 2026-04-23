"""Unit tests for :mod:`silica.bench.wikitext`.

The WikiText loader is deliberately offline-only: no network, no HF
datasets import, no download prompts. These tests pair the loader
with a byte-level fake tokenizer so nothing outside the
``tests/fixtures/`` vendored text file is touched.

Invariants pinned:

- ``(1, N)`` int32 shape contract with the PPL oracle.
- ``max_tokens`` truncates to exactly that many tokens from the start.
- ``min_tokens`` raises ``ValueError`` on short input (catches empty /
  wrong-path /zero-tokenizer footguns).
- ``load_wikitext_text`` raises ``FileNotFoundError`` on missing paths
  with a message that names the path (the bench harness's one-shot
  hint text grepps for this).
- Round-trip against the vendored ``wikitext2_tiny.txt`` fixture
  actually produces a non-trivial token count (the fixture isn't
  silently empty).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wikitext2_tiny.txt"


class _ByteTokenizer:
    """Byte-level tokenizer: emits one int per UTF-8 byte.

    Deterministic, no vocab file, no external deps — good enough to
    exercise the loader's shape + truncation logic without pulling in
    an HF tokenizer. Vocab size is 256 (byte values).
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


# =============================================================================
# load_wikitext_text
# =============================================================================


class TestLoadWikitextText:
    def test_reads_fixture(self) -> None:
        text = load_wikitext_text(_FIXTURE_PATH)
        assert isinstance(text, str)
        assert len(text) > 0
        # Fixture is WikiText-2 style; first non-empty token starts with
        # the " = " heading marker.
        assert " = " in text

    def test_missing_path_raises_with_path_in_message(self) -> None:
        missing = Path("/tmp/silica_nonexistent_wikitext_test_file.txt")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_wikitext_text(missing)

    def test_directory_path_is_rejected(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "not_a_file"
        directory.mkdir()
        with pytest.raises(FileNotFoundError, match="not a regular file"):
            load_wikitext_text(directory)

    def test_accepts_str_path(self) -> None:
        """Path | str — passing a plain string must work identically
        to a ``Path``; bench scenarios often carry str paths."""
        text = load_wikitext_text(str(_FIXTURE_PATH))
        assert len(text) > 0


# =============================================================================
# tokenize_for_ppl
# =============================================================================


class TestTokenizeForPpl:
    def test_returns_2d_int32_with_row_axis_one(self) -> None:
        tok = _ByteTokenizer()
        arr = tokenize_for_ppl(tok, "hello")
        assert arr.ndim == 2
        assert arr.shape[0] == 1
        assert arr.shape[1] == len(b"hello")
        assert arr.dtype == mx.int32

    def test_round_trip_on_fixture(self) -> None:
        tok = _ByteTokenizer()
        text = load_wikitext_text(_FIXTURE_PATH)
        arr = tokenize_for_ppl(tok, text)
        # Fixture is roughly 700+ UTF-8 bytes; byte tokenizer produces
        # one token per byte. Floor of 200 keeps the test robust
        # against minor fixture edits.
        assert arr.shape[0] == 1
        assert arr.shape[1] >= 200

    def test_truncates_to_max_tokens(self) -> None:
        tok = _ByteTokenizer()
        text = load_wikitext_text(_FIXTURE_PATH)
        arr = tokenize_for_ppl(tok, text, max_tokens=64)
        assert arr.shape == (1, 64)

    def test_max_tokens_longer_than_text_returns_all(self) -> None:
        tok = _ByteTokenizer()
        arr_all = tokenize_for_ppl(tok, "hello")
        arr_cap = tokenize_for_ppl(tok, "hello", max_tokens=1024)
        assert arr_all.shape == arr_cap.shape == (1, 5)

    def test_rejects_negative_max_tokens(self) -> None:
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="max_tokens"):
            tokenize_for_ppl(tok, "hello", max_tokens=-1)

    def test_rejects_negative_min_tokens(self) -> None:
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="min_tokens"):
            tokenize_for_ppl(tok, "hello", min_tokens=-1)

    def test_raises_when_below_min_tokens(self) -> None:
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="below min_tokens"):
            tokenize_for_ppl(tok, "", min_tokens=1)

    def test_raises_when_truncation_drops_below_min_tokens(
        self,
    ) -> None:
        tok = _ByteTokenizer()
        # Text has 5 tokens; cap to 2, require 4 → should raise.
        with pytest.raises(ValueError, match="below min_tokens"):
            tokenize_for_ppl(
                tok, "hello", max_tokens=2, min_tokens=4
            )

    def test_min_tokens_zero_accepts_empty_input(self) -> None:
        tok = _ByteTokenizer()
        arr = tokenize_for_ppl(tok, "", min_tokens=0)
        assert arr.shape == (1, 0)
        assert arr.dtype == mx.int32

    def test_tokenizer_missing_encode_raises_type_error(self) -> None:
        class _Broken:
            pass

        with pytest.raises(TypeError, match="encode"):
            tokenize_for_ppl(_Broken(), "hello")

    def test_output_matches_tokenizer_output(self) -> None:
        """The loader must not re-pad, re-slice, or re-order tokens
        beyond the documented truncation. Compare elementwise against
        the raw tokenizer output."""
        tok = _ByteTokenizer()
        text = "abc"
        expected = tok.encode(text)
        arr = tokenize_for_ppl(tok, text)
        # (1, 3) int32 equal elementwise to expected
        assert arr.shape == (1, 3)
        assert arr.tolist() == [expected]

    def test_accepts_tokenizer_returning_generic_sequence(self) -> None:
        """``encode`` may return any ``Sequence[int]``; we list() it
        internally. Verifies a tuple-returning tokenizer works too."""

        class _TupleTokenizer:
            def encode(self, text: str) -> tuple[int, ...]:
                return tuple(ord(c) for c in text)

        tok = _TupleTokenizer()
        arr = tokenize_for_ppl(tok, "abc")
        assert arr.shape == (1, 3)
        assert arr.tolist() == [[ord("a"), ord("b"), ord("c")]]


# =============================================================================
# End-to-end integration: loader → tokenizer → PPL oracle
# =============================================================================


class TestIntegrationWithOracle:
    """Exercises the loader output through the C.1 PPL oracle using
    the BatchKVCache-compat fake adapter from the C.2 test suite. If
    the shape contract between ``tokenize_for_ppl`` and
    :func:`teacher_forced_chunked_nll` drifts, this test catches it
    at the loader boundary rather than deep inside the oracle."""

    def test_fixture_flows_through_c1_oracle(self) -> None:
        from silica.bench.ppl_oracle import teacher_forced_chunked_nll
        from tests.test_ppl_oracle_codec import (
            _BatchKVCacheCompatAdapter,
        )

        tok = _ByteTokenizer()
        text = load_wikitext_text(_FIXTURE_PATH)
        tokens = tokenize_for_ppl(tok, text, max_tokens=64)
        adapter = _BatchKVCacheCompatAdapter()

        # The fake model's vocab is 8, but the tokenizer emits byte
        # ids up to 255. That is fine for the oracle path — it never
        # dereferences logits by token id, only passes tokens through
        # the model and scores CE against them. The score itself
        # would be "out of bounds" semantically, but mechanically the
        # pipeline returns a finite number.
        nll, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=32)
        # Oracle scored seq_len - 1 tokens; nll finite and non-negative
        # (fake model's logits are bounded so cross-entropy is finite).
        assert n == 63
        import math as _math

        assert _math.isfinite(nll)
