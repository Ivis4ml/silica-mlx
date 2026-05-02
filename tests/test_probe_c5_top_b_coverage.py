"""D-021 step 8 (β.2) — top-b coverage probe pure-function tests.

Pins the load-bearing pure functions in
``scripts/probe_c5_top_b_coverage.py`` so the probe's rank /
coverage / tokenizer-alignment logic is correct before any real
model load. Loads the script as a module via importlib so the
top-level ``import`` does not need to pull in mlx (the probe's
mlx-touching code lives inside ``main()`` and is unreachable
from this test file).

Every test uses synthetic Python ``list`` inputs — no mlx, no
mlx-lm, no model load. Skipped paths in ``main()`` (gate
checks, real-model forwards, JSONL emission) are exercised by
the β.1/β.2 real-model run, not here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def _load_probe_module() -> Any:
    here = Path(__file__).resolve().parents[1]
    path = here / "scripts" / "probe_c5_top_b_coverage.py"
    spec = importlib.util.spec_from_file_location("_probe_c5_cov", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe_module()


class _FakeTokenizer:
    """Minimal stand-in for an mlx-lm TokenizerWrapper."""

    def __init__(
        self,
        *,
        vocab_size: int = 100,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        pad_token_id: int | None = 2,
        encode_map: dict[str, list[int]] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._encode_map = encode_map or {}

    def encode(self, text: str) -> list[int]:
        if text in self._encode_map:
            return list(self._encode_map[text])
        return [ord(c) % self.vocab_size for c in text]


def test_compute_ranks_target_at_top() -> None:
    """rank=0 when target argmax is the drafter's #1 candidate."""
    target = [5, 7, 9]
    drafter_top = [
        [5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 11, 12, 13, 14, 15, 16],
        [7, 4, 3, 2, 1, 0, 9, 8, 5, 6, 11, 12, 13, 14, 15, 16],
        [9, 4, 3, 2, 1, 0, 7, 8, 5, 6, 11, 12, 13, 14, 15, 16],
    ]
    ranks = probe.compute_ranks(target, drafter_top, max_b=16)
    assert ranks == [0, 0, 0]


def test_compute_ranks_target_at_known_positions() -> None:
    """rank reflects the exact index in drafter_top_ids."""
    target = [3, 8, 15]
    drafter_top = [
        [9, 3, 1, 0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
    ]
    ranks = probe.compute_ranks(target, drafter_top, max_b=16)
    assert ranks == [1, 9, 15]


def test_compute_ranks_target_outside_topk() -> None:
    """rank == max_b when target_argmax is not in drafter top-K."""
    target = [42]
    drafter_top = [list(range(16))]
    ranks = probe.compute_ranks(target, drafter_top, max_b=16)
    assert ranks == [16]


def test_compute_ranks_length_mismatch_raises() -> None:
    target = [1, 2]
    drafter_top = [[0] * 16]
    with pytest.raises(ValueError, match="length mismatch"):
        probe.compute_ranks(target, drafter_top, max_b=16)


def test_compute_ranks_short_drafter_row_raises() -> None:
    """If a drafter row has fewer than max_b ids, fail loud."""
    target = [0]
    drafter_top = [[0, 1, 2]]
    with pytest.raises(ValueError, match="need >= 16"):
        probe.compute_ranks(target, drafter_top, max_b=16)


def test_coverage_at_basic() -> None:
    """coverage@b = (1/N) * |{i : ranks[i] < b}|."""
    ranks = [0, 0, 1, 3, 5, 9, 32]
    cov = probe.coverage_at(ranks, (1, 4, 8, 16, 32))
    assert cov == pytest.approx(
        {1: 2 / 7, 4: 4 / 7, 8: 5 / 7, 16: 6 / 7, 32: 6 / 7}
    )


def test_coverage_at_empty() -> None:
    """Empty rank list -> coverage 0 at every b (no NaN)."""
    cov = probe.coverage_at([], (1, 4, 8, 16))
    assert cov == {1: 0.0, 4: 0.0, 8: 0.0, 16: 0.0}


def test_coverage_at_all_above_b() -> None:
    """When every rank exceeds the largest b, coverage is 0."""
    ranks = [16] * 5
    cov = probe.coverage_at(ranks, (1, 4, 8))
    assert cov == {1: 0.0, 4: 0.0, 8: 0.0}


def test_coverage_at_all_below_b() -> None:
    """When every rank is 0, coverage is 1 at every b >= 1."""
    ranks = [0] * 5
    cov = probe.coverage_at(ranks, (1, 4, 8, 16))
    assert cov == {1: 1.0, 4: 1.0, 8: 1.0, 16: 1.0}


def test_coverage_at_rejects_zero_or_negative_b() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        probe.coverage_at([0, 1], (0,))
    with pytest.raises(ValueError, match="must be positive"):
        probe.coverage_at([0, 1], (-1,))


def test_rank_histogram_basic() -> None:
    """Right-open bucketed counts; tail bin catches anything past
    the last bucket."""
    ranks = [0, 0, 3, 7, 7, 15, 99, 250]
    hist = probe.rank_histogram(ranks, (1, 4, 8, 16, 32, 100))
    assert hist == {
        "<1": 2,
        "<4": 1,
        "<8": 2,
        "<16": 1,
        "<32": 0,
        "<100": 1,
        ">=100": 1,
    }


def test_rank_histogram_empty() -> None:
    hist = probe.rank_histogram([], (1, 4, 8))
    assert hist == {"<1": 0, "<4": 0, "<8": 0, ">=8": 0}


def test_rank_histogram_rejects_unsorted_buckets() -> None:
    with pytest.raises(ValueError, match="ascending"):
        probe.rank_histogram([0, 1], (4, 1, 8))


def test_rank_histogram_rejects_empty_buckets() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        probe.rank_histogram([0, 1], ())


def test_tokenizer_attestation_aligned() -> None:
    """Happy path: identical encode + identical specials -> attestation
    dict with vocab_size, sample hashes, config hash."""
    encode_map = {
        "alpha": [3, 7, 11],
        "beta": [13, 17, 19, 23],
    }
    target = _FakeTokenizer(vocab_size=100, encode_map=encode_map)
    drafter = _FakeTokenizer(vocab_size=100, encode_map=encode_map)

    attestation = probe.tokenizer_attestation(
        target, drafter, ["alpha", "beta"]
    )
    assert attestation["vocab_size"] == 100
    assert attestation["sample_count"] == 2
    assert len(attestation["sample_hashes"]) == 2
    assert attestation["sample_hashes"][0]["len_ids"] == 3
    assert attestation["sample_hashes"][1]["len_ids"] == 4
    assert attestation["special_tokens"] == {
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }
    assert isinstance(attestation["config_hash"], str)
    assert len(attestation["config_hash"]) == 16


def test_tokenizer_attestation_vocab_mismatch_raises() -> None:
    target = _FakeTokenizer(vocab_size=100)
    drafter = _FakeTokenizer(vocab_size=200)
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        probe.tokenizer_attestation(target, drafter, ["sample"])


def test_tokenizer_attestation_encoding_mismatch_raises() -> None:
    """Same vocab_size but different encode output -> fail loud."""
    target = _FakeTokenizer(
        vocab_size=100, encode_map={"sample": [1, 2, 3]}
    )
    drafter = _FakeTokenizer(
        vocab_size=100, encode_map={"sample": [1, 2, 4]}
    )
    with pytest.raises(ValueError, match="encoding divergence"):
        probe.tokenizer_attestation(target, drafter, ["sample"])


def test_tokenizer_attestation_special_token_mismatch_raises() -> None:
    target = _FakeTokenizer(vocab_size=100, eos_token_id=1)
    drafter = _FakeTokenizer(vocab_size=100, eos_token_id=2)
    with pytest.raises(ValueError, match="eos_token_id mismatch"):
        probe.tokenizer_attestation(target, drafter, [])


def test_tokenizer_attestation_missing_vocab_size_raises() -> None:
    """If either tokenizer lacks vocab_size, refuse to proceed —
    silent rank comparison would be undefined."""

    class _NoVocab:
        bos_token_id = 0
        eos_token_id = 1

        def encode(self, text: str) -> list[int]:
            return []

    target = _FakeTokenizer(vocab_size=100)
    drafter = _NoVocab()
    with pytest.raises(ValueError, match="vocab_size unavailable"):
        probe.tokenizer_attestation(target, drafter, [])


def test_tokenizer_attestation_handles_partial_specials() -> None:
    """When only one tokenizer exposes a special-token id, the
    attestation records the available value rather than raising."""

    class _PartialSpecials:
        vocab_size = 100
        bos_token_id = 0
        eos_token_id = None
        pad_token_id = None

        def encode(self, text: str) -> list[int]:
            return [ord(c) % 100 for c in text]

    target = _FakeTokenizer(
        vocab_size=100, eos_token_id=5, pad_token_id=None
    )
    drafter = _PartialSpecials()
    attestation = probe.tokenizer_attestation(target, drafter, [])
    assert attestation["special_tokens"]["bos_token_id"] == 0
    assert attestation["special_tokens"]["eos_token_id"] == 5
    assert attestation["special_tokens"]["pad_token_id"] is None


def test_summarize_run_schema_complete() -> None:
    """JSONL row carries every field the (β.2) REPORT consumer reads
    and round-trips through json.dumps cleanly."""
    row = probe.summarize_run(
        ranks=[0, 1, 16],
        coverage={1: 1 / 3, 4: 2 / 3, 8: 2 / 3, 16: 2 / 3, 32: 3 / 3},
        histogram={"<1": 1, "<4": 1, "<8": 0, "<16": 0, "<32": 1, ">=32": 0},
        target_repo="mlx-community/Qwen3.5-27B-4bit",
        drafter_repo="Qwen/Qwen3.5-0.8B",
        target_gate_env="SILICA_REAL_QWEN3_5_27B",
        drafter_gate_env="SILICA_REAL_QWEN3_5_0_8B_DRAFT",
        corpus_path="/tmp/wikitext.txt",
        n_positions=3,
        n_corpus_tokens=4,
        max_b=32,
        seed=0,
        tokenizer_attestation_data={
            "vocab_size": 152064,
            "sample_count": 1,
            "sample_hashes": [],
            "special_tokens": {
                "bos_token_id": None,
                "eos_token_id": 151645,
                "pad_token_id": 151643,
            },
            "config_hash": "deadbeefdeadbeef",
        },
        elapsed_s=42.5,
    )

    expected_keys = {
        "probe_id",
        "target_repo",
        "drafter_repo",
        "target_gate_env",
        "drafter_gate_env",
        "corpus_path",
        "n_positions_scored",
        "n_corpus_tokens_loaded",
        "max_b_measured",
        "b_values",
        "coverage_at",
        "rank_histogram",
        "tokenizer_attestation",
        "seed",
        "elapsed_s",
        "n_ranks_recorded",
    }
    assert set(row.keys()) == expected_keys
    assert row["probe_id"] == "c5_top_b_coverage"
    assert row["b_values"] == [1, 4, 8, 16, 32]
    assert row["coverage_at"]["1"] == pytest.approx(1 / 3)
    assert row["coverage_at"]["32"] == pytest.approx(1.0)

    encoded = json.dumps(row)
    round_trip = json.loads(encoded)
    assert round_trip == row


def test_slice_sample_texts_basic() -> None:
    """``_slice_sample_texts`` carves N non-empty slices of the
    requested length when the corpus is long enough."""
    text = "x" * 5000
    samples = probe._slice_sample_texts(text, n=3, chars=1024)
    assert len(samples) == 3
    assert all(len(s) == 1024 for s in samples)


def test_slice_sample_texts_short_corpus() -> None:
    """When the corpus is shorter than ``chars``, return the whole
    corpus as a single sample (no padding, no exception)."""
    text = "short"
    samples = probe._slice_sample_texts(text, n=3, chars=1024)
    assert samples == ["short"]


def test_slice_sample_texts_zero_request() -> None:
    samples = probe._slice_sample_texts("anything", n=0, chars=1024)
    assert samples == []
    samples = probe._slice_sample_texts("anything", n=3, chars=0)
    assert samples == []


def test_build_parser_smoke() -> None:
    """Defaults parse without --out missing-required error."""
    parser = probe.build_parser()
    ns = parser.parse_args(["--out", "/tmp/whatever.jsonl"])
    assert ns.target == probe.DEFAULT_TARGET_REPO
    assert ns.drafter == probe.DEFAULT_DRAFTER_REPO
    assert ns.target_gate_env == probe.DEFAULT_GATE_TARGET
    assert ns.drafter_gate_env == probe.DEFAULT_GATE_DRAFTER
    assert ns.max_tokens == probe.DEFAULT_MAX_TOKENS
    assert ns.seed == 0


def test_build_parser_required_out() -> None:
    parser = probe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
