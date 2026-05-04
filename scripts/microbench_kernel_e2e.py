"""End-to-end warm-decode microbench with kernel shadow install.

Runs a warm-decode loop at B=4 on cached Qwen3.5-27B-4bit, with kernel
shadow-install controlled by env flags. Compare aggregate tok/s under
each configuration.

Usage (each row):
    SILICA_REAL_QWEN3_5_27B=1 \
        uv run python scripts/microbench_kernel_e2e.py \
            --b 4 --warmup 3 --iters 5 \
            --out plans/P6_AUTORESEARCH/kernel_e2e.jsonl

Each invocation reads SILICA_USE_FUSED_GATED_OUTPUT / SILICA_USE_FUSED_SILU_MUL
env flags to determine which kernels are active. The script emits one
JSONL row per measurement iter plus a summary row.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import cache as mlx_cache


def _short_prompt_tokens(tokenizer, target_len):  # noqa: ANN001
    text = (
        "Memory bandwidth has emerged as the dominant constraint in "
        "single-stream autoregressive decoding for large language models. "
        "Each decode step must read the entire active parameter set from "
        "unified memory before any computation can begin."
    )
    ids = tokenizer.encode(text)
    if len(ids) < target_len:
        ids = ids + [ids[-1]] * (target_len - len(ids))
    return mx.array(ids[:target_len], dtype=mx.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument(
        "--decode-pattern",
        choices=["per_step_eval", "lazy_chain", "lazy_argmax_chain"],
        default="per_step_eval",
        help="per_step_eval = current production pattern (mx.eval after every step); "
             "lazy_chain = constant-zeros input, single mx.eval at end (synthetic); "
             "lazy_argmax_chain = next input = argmax(prev logits), single mx.eval (production-applicable)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If > 0 with lazy_argmax_chain: sync every chunk_size steps. 0 = single sync at end.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plans/P6_AUTORESEARCH/kernel_e2e.jsonl"),
    )
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit("SILICA_REAL_QWEN3_5_27B=1 required.")

    from silica.kernels import shadow_install
    from silica.mlx.runner import forward_batched, forward_batched_full
    from silica.models.factory import adapter_for_repo

    # Snapshot env state for the report
    env_state = {
        "SILICA_USE_FUSED_GATED_OUTPUT": os.environ.get("SILICA_USE_FUSED_GATED_OUTPUT", "0"),
        "SILICA_USE_FUSED_SILU_MUL": os.environ.get("SILICA_USE_FUSED_SILU_MUL", "0"),
        "SILICA_USE_FUSED_QK_NORM": os.environ.get("SILICA_USE_FUSED_QK_NORM", "0"),
    }

    print(f"[kernel_e2e] env: {env_state}")
    print(f"[kernel_e2e] loading {args.repo}...")
    adapter, _kv = adapter_for_repo(args.repo)
    model = adapter._model  # noqa: SLF001
    tokens_1d = _short_prompt_tokens(adapter.tokenizer(), args.prompt_len)

    # Install shadow kernels (no-op if all env flags are 0)
    installed = shadow_install.install(model)
    print(f"[kernel_e2e] kernels installed: {installed}")

    rows = []
    ts = datetime.now(timezone.utc).isoformat()

    # Pre-allocate a single zeros input (reused across steps).
    single_input = mx.zeros((args.b, 1), dtype=mx.int32)
    mx.eval(single_input)

    def run_decode_loop(cache_list: list) -> None:
        """One decode loop in the requested pattern. Returns nothing; mlx must be synced after."""
        if args.decode_pattern == "per_step_eval":
            for _ in range(args.decode_tokens):
                logits = forward_batched(model, single_input, cache_list)
                mx.eval(logits)
        elif args.decode_pattern == "lazy_chain":
            # Constant input across steps; lazy graph; single sync at end.
            last = None
            for _ in range(args.decode_tokens):
                logits = forward_batched(model, single_input, cache_list)
                last = logits
            mx.eval(last)
        elif args.decode_pattern == "lazy_argmax_chain":
            # Next input = argmax(prev logits), keeping dependency chain lazy.
            # Optionally sync every `chunk_size` steps for stop-token-style use.
            prev_token = single_input
            chunk = args.chunk_size if args.chunk_size > 0 else args.decode_tokens
            steps_done = 0
            while steps_done < args.decode_tokens:
                this_chunk = min(chunk, args.decode_tokens - steps_done)
                for _ in range(this_chunk):
                    logits = forward_batched(model, prev_token, cache_list)
                    prev_token = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
                mx.eval(prev_token)
                steps_done += this_chunk

    # Warmup
    for _ in range(args.warmup):
        cache_list = mlx_cache.make_prompt_cache(model)
        tokens_b = mx.tile(tokens_1d[None], (args.b, 1))
        _ = forward_batched_full(model, tokens_b, cache_list)
        mx.eval(_)
        run_decode_loop(cache_list)

    # Measurement: prefill + decode N tokens, measure wall time of decode loop
    for it in range(args.iters):
        cache_list = mlx_cache.make_prompt_cache(model)
        tokens_b = mx.tile(tokens_1d[None], (args.b, 1))
        _ = forward_batched_full(model, tokens_b, cache_list)
        mx.eval(_)

        t_decode_start = time.perf_counter_ns()
        run_decode_loop(cache_list)
        t_decode_total = (time.perf_counter_ns() - t_decode_start) / 1e9  # seconds

        # Aggregate tok/s = (b * decode_tokens) / wall_seconds
        n_tokens = args.b * args.decode_tokens
        agg_tok_s = n_tokens / t_decode_total
        per_step_ms = (t_decode_total / args.decode_tokens) * 1000
        rows.append({
            "kind": "iter",
            "iter": it,
            "decode_wall_s": t_decode_total,
            "decode_tokens": args.decode_tokens,
            "b": args.b,
            "agg_tok_s": agg_tok_s,
            "per_step_ms": per_step_ms,
            "ts": ts,
        })
        print(f"[kernel_e2e] iter {it}: {t_decode_total:.2f}s for {n_tokens} tok = "
              f"{agg_tok_s:.2f} tok/s ({per_step_ms:.2f} ms/step)")

    agg_samples = [r["agg_tok_s"] for r in rows]
    summary = {
        "kind": "summary",
        "ts": ts,
        "repo": args.repo,
        "b": args.b,
        "warmup_iters": args.warmup,
        "measurement_iters": args.iters,
        "decode_tokens_per_iter": args.decode_tokens,
        "decode_pattern": args.decode_pattern,
        "chunk_size": args.chunk_size,
        "env": env_state,
        "kernels_installed": installed,
        "agg_tok_s_median": statistics.median(agg_samples),
        "agg_tok_s_mean": statistics.mean(agg_samples),
        "agg_tok_s_min": min(agg_samples),
        "agg_tok_s_max": max(agg_samples),
        "agg_tok_s_stdev": statistics.stdev(agg_samples) if len(agg_samples) > 1 else 0.0,
    }
    rows.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[kernel_e2e] wrote {len(rows)} rows to {args.out}")
    print(f"[kernel_e2e] aggregate tok/s median = "
          f"{summary['agg_tok_s_median']:.2f} ± {summary['agg_tok_s_stdev']:.2f}")
    print(f"[kernel_e2e] vs P-6.0.5 baseline 42.17 tok/s: "
          f"delta = {summary['agg_tok_s_median'] - 42.17:+.2f} tok/s "
          f"({(summary['agg_tok_s_median'] / 42.17 - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
