"""D-021 step 8 sub-unit (β.2) — top-b coverage probe for C.5.

Stand-alone read-only probe that measures
``coverage@b = Pr(target_argmax in drafter's top-b candidates)``
for ``b in {1, 4, 8, 16, 32}`` against a fixed prompt corpus.

Bypasses the spec engine entirely: two parallel teacher-forced
forwards (one through the cached target, one through the cached
drafter) over the *same token-id prefix*, then per-position rank
of the target's argmax in the drafter's logit-sorted token ids.

The metric is the load-bearing decision input for the C.5
tree-shape spike (see ``plans/P6_C5_DDTREE_OPENING.md`` §6 β.2).
Coverage is what tree-shape can extract; linear ``accept_rate``
(= ``coverage@1``) is only one point on the curve.

Tokenizer-alignment guard runs first and fails loud on any
divergence between target and drafter tokenizers (vocab size,
encoding output on each sample text, key special-token ids).
A silent vocab mismatch would make rank comparisons meaningless,
so the probe refuses to proceed without the attestation.

Pure functions (rank computation, coverage aggregation,
tokenizer attestation, JSONL row composition) live at module
top so ``tests/test_probe_c5_top_b_coverage.py`` can exercise
them with mock data, no model load required. The mlx-lm /
silica imports live inside ``main()`` so importing this module
does not pull in MLX.

Usage:

    SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \\
        uv run python -m scripts.probe_c5_top_b_coverage \\
            --target mlx-community/Qwen3.5-27B-4bit \\
            --drafter Qwen/Qwen3.5-0.8B \\
            --max-tokens 512 \\
            --out plans/P6_C5_DDTREE/coverage_probe.jsonl

The two gate envs mirror ``qwen3.5-27b-warm-decode-spec-on``'s
quad-gating so a developer who has not consciously authorised
either checkpoint cannot trigger a load by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_TARGET_REPO = "mlx-community/Qwen3.5-27B-4bit"
DEFAULT_DRAFTER_REPO = "Qwen/Qwen3.5-0.8B"
DEFAULT_GATE_TARGET = "SILICA_REAL_QWEN3_5_27B"
DEFAULT_GATE_DRAFTER = "SILICA_REAL_QWEN3_5_0_8B_DRAFT"
DEFAULT_WIKITEXT_PATH = (
    Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
)
DEFAULT_MAX_TOKENS = 512
DEFAULT_N_SAMPLE_TEXTS = 3
DEFAULT_SAMPLE_TEXT_CHARS = 1024

B_VALUES: tuple[int, ...] = (1, 4, 8, 16, 32)
HISTOGRAM_BUCKETS: tuple[int, ...] = (1, 4, 8, 16, 32, 100, 1000)


def compute_ranks(
    target_argmax: list[int],
    drafter_top_ids: list[list[int]],
    *,
    max_b: int,
) -> list[int]:
    """Per-position rank of ``target_argmax[i]`` in
    ``drafter_top_ids[i]`` (drafter's logit-descending top-K ids).

    Length contract: ``len(target_argmax) == len(drafter_top_ids)``.
    Each row of ``drafter_top_ids`` must have at least ``max_b``
    entries (the probe slices ``argsort`` output to ``max_b`` columns
    before passing in, so this is enforced upstream).

    Returns a list of integer ranks. If ``target_argmax[i]`` is not
    present in ``drafter_top_ids[i]``, the rank is reported as
    ``max_b`` (sentinel meaning "outside the measured top-K");
    callers reading ``coverage@b`` for ``b <= max_b`` see this
    correctly as "not in the b-prefix."
    """
    if len(target_argmax) != len(drafter_top_ids):
        raise ValueError(
            f"length mismatch: len(target_argmax)={len(target_argmax)} "
            f"vs len(drafter_top_ids)={len(drafter_top_ids)}"
        )
    ranks: list[int] = []
    for i, t_id in enumerate(target_argmax):
        row = drafter_top_ids[i]
        if len(row) < max_b:
            raise ValueError(
                f"row {i} has {len(row)} drafter top-ids; need >= {max_b}"
            )
        try:
            ranks.append(row[:max_b].index(t_id))
        except ValueError:
            ranks.append(max_b)
    return ranks


def coverage_at(
    ranks: list[int], b_values: tuple[int, ...] | list[int]
) -> dict[int, float]:
    """``coverage@b = (1/N) * |{i : ranks[i] < b}|`` for each ``b``."""
    n = len(ranks)
    if n == 0:
        return {int(b): 0.0 for b in b_values}
    out: dict[int, float] = {}
    for b in b_values:
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        hits = sum(1 for r in ranks if r < b)
        out[int(b)] = hits / n
    return out


def rank_histogram(
    ranks: list[int], buckets: tuple[int, ...] | list[int]
) -> dict[str, int]:
    """Right-open bucketed counts: ``"<b" -> count of ranks < b``,
    plus a final ``">=max_bucket"`` bin.

    Buckets must be strictly ascending. Counts are cumulative-by-
    bucket-edge: each rank contributes to exactly one bin (the
    smallest bucket it falls below, or the tail bin if it exceeds
    the largest bucket).
    """
    if not buckets:
        raise ValueError("buckets must be non-empty")
    sorted_buckets = sorted(buckets)
    if sorted_buckets != list(buckets):
        raise ValueError(f"buckets must be strictly ascending: {buckets}")
    if any(b <= 0 for b in buckets):
        raise ValueError(f"buckets must be positive: {buckets}")
    out: dict[str, int] = {f"<{b}": 0 for b in buckets}
    out[f">={buckets[-1]}"] = 0
    for r in ranks:
        placed = False
        for b in buckets:
            if r < b:
                out[f"<{b}"] += 1
                placed = True
                break
        if not placed:
            out[f">={buckets[-1]}"] += 1
    return out


def _hash_ids(ids: list[int]) -> str:
    payload = b",".join(str(i).encode() for i in ids)
    return hashlib.sha256(payload).hexdigest()[:16]


def _hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_int_attr(obj: Any, name: str) -> int | None:
    val = getattr(obj, name, None)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def tokenizer_attestation(
    target_tokenizer: Any,
    drafter_tokenizer: Any,
    sample_texts: list[str],
) -> dict[str, Any]:
    """Assert target and drafter tokenizers agree on sample-text
    encoding and key special-token ids; return attestation dict.

    Raises ``ValueError`` (loud, not silent) on:
    - ``vocab_size`` mismatch,
    - any sample text encoding to different id sequences across the
      two tokenizers,
    - any key special-token id (``bos_token_id``, ``eos_token_id``,
      ``pad_token_id``) mismatch when both tokenizers expose it.

    Returns a dict with ``vocab_size``, per-sample id-hashes, and a
    config-hash derived from the special-token ids — load-bearing
    provenance that gets emitted into the JSONL output so a future
    re-read can verify the same alignment is still in force.
    """
    target_vocab = _safe_int_attr(target_tokenizer, "vocab_size")
    drafter_vocab = _safe_int_attr(drafter_tokenizer, "vocab_size")
    if target_vocab is None or drafter_vocab is None:
        raise ValueError(
            "tokenizer.vocab_size unavailable on at least one tokenizer; "
            "probe cannot guard rank comparison without it"
        )
    if target_vocab != drafter_vocab:
        raise ValueError(
            f"tokenizer vocab_size mismatch: "
            f"target={target_vocab} vs drafter={drafter_vocab}; "
            f"rank comparisons would be over disjoint id spaces"
        )

    sample_hashes: list[dict[str, Any]] = []
    for idx, text in enumerate(sample_texts):
        t_ids = list(target_tokenizer.encode(text))
        d_ids = list(drafter_tokenizer.encode(text))
        if t_ids != d_ids:
            raise ValueError(
                f"tokenizer encoding divergence on sample {idx} "
                f"(len(text)={len(text)}): target produced "
                f"{len(t_ids)} ids, drafter produced {len(d_ids)} ids; "
                f"first divergent prefix lengths: "
                f"target_head={t_ids[:8]!r}, drafter_head={d_ids[:8]!r}"
            )
        sample_hashes.append(
            {
                "len_chars": len(text),
                "len_ids": len(t_ids),
                "ids_hash": _hash_ids(t_ids),
            }
        )

    special_tokens: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        t_val = _safe_int_attr(target_tokenizer, name)
        d_val = _safe_int_attr(drafter_tokenizer, name)
        if t_val is not None and d_val is not None and t_val != d_val:
            raise ValueError(
                f"tokenizer {name} mismatch: target={t_val} vs "
                f"drafter={d_val}; same id space requires same "
                f"special-token mapping"
            )
        special_tokens[name] = t_val if t_val is not None else d_val

    config_hash = _hash_dict(
        {
            "vocab_size": target_vocab,
            "special_tokens": special_tokens,
        }
    )

    return {
        "vocab_size": int(target_vocab),
        "sample_count": len(sample_texts),
        "sample_hashes": sample_hashes,
        "special_tokens": special_tokens,
        "config_hash": config_hash,
    }


def summarize_run(
    *,
    ranks: list[int],
    coverage: dict[int, float],
    histogram: dict[str, int],
    target_repo: str,
    drafter_repo: str,
    target_gate_env: str,
    drafter_gate_env: str,
    corpus_path: str,
    n_positions: int,
    n_corpus_tokens: int,
    max_b: int,
    seed: int,
    tokenizer_attestation_data: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    """Compose the JSONL output row.

    Schema is intentionally flat so downstream tooling
    (``plans/P6_C5_DDTREE/REPORT.md`` β.2 section, future
    coverage-comparison scripts) can read it without bespoke
    deserialisation.
    """
    return {
        "probe_id": "c5_top_b_coverage",
        "target_repo": target_repo,
        "drafter_repo": drafter_repo,
        "target_gate_env": target_gate_env,
        "drafter_gate_env": drafter_gate_env,
        "corpus_path": corpus_path,
        "n_positions_scored": int(n_positions),
        "n_corpus_tokens_loaded": int(n_corpus_tokens),
        "max_b_measured": int(max_b),
        "b_values": list(B_VALUES),
        "coverage_at": {str(k): float(v) for k, v in coverage.items()},
        "rank_histogram": histogram,
        "tokenizer_attestation": tokenizer_attestation_data,
        "seed": int(seed),
        "elapsed_s": float(elapsed_s),
        "n_ranks_recorded": len(ranks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_c5_top_b_coverage",
        description=(
            "D-021 step 8 (β.2) — read-only top-b coverage probe "
            "for the C.5 tree-shape decision spike."
        ),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_REPO,
        help=f"Target HF repo (default: {DEFAULT_TARGET_REPO}).",
    )
    parser.add_argument(
        "--drafter",
        default=DEFAULT_DRAFTER_REPO,
        help=f"Drafter HF repo (default: {DEFAULT_DRAFTER_REPO}).",
    )
    parser.add_argument(
        "--target-gate-env",
        default=DEFAULT_GATE_TARGET,
        help=(
            f"Env var the user must set to '1' to authorise the "
            f"target load (default: {DEFAULT_GATE_TARGET})."
        ),
    )
    parser.add_argument(
        "--drafter-gate-env",
        default=DEFAULT_GATE_DRAFTER,
        help=(
            f"Env var the user must set to '1' to authorise the "
            f"drafter load (default: {DEFAULT_GATE_DRAFTER})."
        ),
    )
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_WIKITEXT_PATH),
        help=(
            "Path to a UTF-8 plain-text corpus file (default: the "
            "WikiText-2 cache at "
            f"{DEFAULT_WIKITEXT_PATH}; populate via "
            "scripts/prepare_wikitext2_cache.py if missing)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            f"Maximum corpus tokens to load and forward through "
            f"both models (default: {DEFAULT_MAX_TOKENS}). The probe "
            f"scores N = max_tokens - 1 positions."
        ),
    )
    parser.add_argument(
        "--n-sample-texts",
        type=int,
        default=DEFAULT_N_SAMPLE_TEXTS,
        help=(
            f"How many sample text slices to use for the tokenizer "
            f"alignment attestation (default: {DEFAULT_N_SAMPLE_TEXTS})."
        ),
    )
    parser.add_argument(
        "--sample-text-chars",
        type=int,
        default=DEFAULT_SAMPLE_TEXT_CHARS,
        help=(
            f"Length in characters of each tokenizer-alignment sample "
            f"slice (default: {DEFAULT_SAMPLE_TEXT_CHARS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reproducibility seed (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSONL output path. The probe writes a single row.",
    )
    return parser


def _gate_check(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        print(
            f"error: {env_name} is not set to '1'. The probe loads a "
            "real checkpoint; set the gate explicitly to authorise.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _slice_sample_texts(
    text: str, n: int, chars: int
) -> list[str]:
    if n <= 0 or chars <= 0:
        return []
    samples: list[str] = []
    if len(text) >= chars:
        for i in range(n):
            start = (i * chars) % max(len(text) - chars, 1)
            samples.append(text[start : start + chars])
    else:
        samples.append(text)
    return samples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    _gate_check(args.target_gate_env)
    _gate_check(args.drafter_gate_env)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load as _mlx_lm_load

    from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl
    from silica.mlx.runner import forward_full

    print(
        f"# β.2 top-b coverage probe — "
        f"target={args.target!r} drafter={args.drafter!r}"
    )
    print(
        f"# corpus={args.corpus!r} max_tokens={args.max_tokens} "
        f"seed={args.seed}"
    )

    text = load_wikitext_text(args.corpus)
    sample_texts = _slice_sample_texts(
        text, args.n_sample_texts, args.sample_text_chars
    )

    t0 = time.perf_counter()
    target_model, target_tokenizer = _mlx_lm_load(args.target)  # type: ignore[misc]
    target_load_s = time.perf_counter() - t0
    print(f"# target loaded in {target_load_s:.2f} s")

    t1 = time.perf_counter()
    drafter_model, drafter_tokenizer = _mlx_lm_load(args.drafter)  # type: ignore[misc]
    drafter_load_s = time.perf_counter() - t1
    print(f"# drafter loaded in {drafter_load_s:.2f} s")

    attestation = tokenizer_attestation(
        target_tokenizer, drafter_tokenizer, sample_texts
    )
    print(
        f"# tokenizer alignment OK: vocab={attestation['vocab_size']} "
        f"config_hash={attestation['config_hash']}"
    )

    tokens_2d = tokenize_for_ppl(
        target_tokenizer, text, max_tokens=args.max_tokens
    )
    tokens_1d = tokens_2d.reshape(-1)
    n_corpus_tokens = int(tokens_1d.shape[0])
    print(f"# corpus tokens: {n_corpus_tokens}")

    target_cache = make_prompt_cache(target_model)
    drafter_cache = make_prompt_cache(drafter_model)

    t2 = time.perf_counter()
    target_logits = forward_full(target_model, tokens_1d, target_cache)
    mx.eval(target_logits)
    target_fwd_s = time.perf_counter() - t2
    print(f"# target forward_full: {target_fwd_s:.2f} s")

    t3 = time.perf_counter()
    drafter_logits = forward_full(drafter_model, tokens_1d, drafter_cache)
    mx.eval(drafter_logits)
    drafter_fwd_s = time.perf_counter() - t3
    print(f"# drafter forward_full: {drafter_fwd_s:.2f} s")

    target_argmax = mx.argmax(target_logits, axis=-1)
    max_b = int(B_VALUES[-1])
    drafter_topk = mx.argpartition(-drafter_logits, kth=max_b, axis=-1)[
        :, :max_b
    ]
    drafter_topk_logits = mx.take_along_axis(
        drafter_logits, drafter_topk, axis=-1
    )
    sort_order = mx.argsort(-drafter_topk_logits, axis=-1)
    drafter_top_ids = mx.take_along_axis(drafter_topk, sort_order, axis=-1)
    mx.eval(target_argmax, drafter_top_ids)

    target_argmax_raw: Any = target_argmax.tolist()
    drafter_top_raw: Any = drafter_top_ids.tolist()
    if not isinstance(target_argmax_raw, list):
        raise RuntimeError(
            "expected mx.argmax(...).tolist() to return a list; "
            f"got {type(target_argmax_raw).__name__}"
        )
    if not isinstance(drafter_top_raw, list):
        raise RuntimeError(
            "expected mx.take_along_axis(...).tolist() to return a list; "
            f"got {type(drafter_top_raw).__name__}"
        )
    target_argmax_py: list[int] = [int(x) for x in target_argmax_raw]
    drafter_top_py: list[list[int]] = [
        [int(x) for x in row] for row in drafter_top_raw
    ]

    target_argmax_for_rank = target_argmax_py[:-1]
    drafter_top_for_rank = drafter_top_py[1:]

    ranks = compute_ranks(
        target_argmax_for_rank, drafter_top_for_rank, max_b=max_b
    )
    coverage = coverage_at(ranks, B_VALUES)
    histogram = rank_histogram(ranks, HISTOGRAM_BUCKETS)

    elapsed_s = time.perf_counter() - t0

    row = summarize_run(
        ranks=ranks,
        coverage=coverage,
        histogram=histogram,
        target_repo=args.target,
        drafter_repo=args.drafter,
        target_gate_env=args.target_gate_env,
        drafter_gate_env=args.drafter_gate_env,
        corpus_path=str(args.corpus),
        n_positions=len(ranks),
        n_corpus_tokens=n_corpus_tokens,
        max_b=max_b,
        seed=args.seed,
        tokenizer_attestation_data=attestation,
        elapsed_s=elapsed_s,
    )

    args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")

    print()
    print("# coverage@b:")
    for b in B_VALUES:
        print(f"  coverage@{b:<2d} = {coverage[int(b)]:.4f}")
    print()
    print("# rank histogram (right-open buckets):")
    for k, v in histogram.items():
        print(f"  {k:>8s}: {v}")
    print()
    print(f"# wrote 1 row to {args.out}")
    print(f"# total elapsed: {elapsed_s:.2f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
