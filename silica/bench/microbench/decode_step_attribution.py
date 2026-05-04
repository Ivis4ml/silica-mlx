"""Per-step decode time attribution microbench for hybrid Qwen3.5 adapters.

Decomposes a single warm-decode step into per-layer cost (full-attention
vs linear-attention / DeltaNet) using ``mx.eval`` barriers around each
layer's ``__call__``. Goal: identify which layer kind owns the largest
share of per-step wall time, so a custom MLX kernel candidate can be
opened against measured evidence rather than paper claims.

Why per-layer barriers, not layer-skip-subtraction:
    Skipping DeltaNet layers in a hybrid model breaks the recurrent-state
    pipeline subsequent layers depend on. Per-layer barriers preserve
    forward semantics; cost attribution is direct, not by subtraction.

Methodology (mirrors ``target_verify`` pattern):
    1. Load adapter via ``adapter_for_repo`` (cache-hit on the production
       target).
    2. Replace each ``model.layers[i]`` with a callable wrapper that
       (a) calls the original layer, (b) ``mx.eval``-s the layer output,
       (c) records ``(layer_idx, is_linear, ns)``.
    3. Per measurement iter, rebuild a fresh ``cache_list`` via
       ``mlx_cache.make_prompt_cache(model)`` (same pattern as Unit 7),
       prefill at the requested ``B`` (untimed), then time exactly one
       decode step.
    4. Aggregate per-layer-kind median wall time. Emit a JSONL row per
       measurement plus a summary row.
    5. Restore ``model.layers`` to original references on exit.

Output schema (JSONL, one row per record):
    {"kind": "per_layer", "iter": int, "layer_idx": int, "is_linear": bool,
     "wall_ms": float, "b": int, "prompt_len": int, "ts": "..."}
    {"kind": "step_total", "iter": int, "wall_ms": float, ...}
    {"kind": "instrumented_overhead", "iter": int, "wall_ms": float, ...}
    {"kind": "summary", ...}

Usage:
    SILICA_REAL_QWEN3_5_27B=1 \\
        uv run python -m silica.bench.microbench.decode_step_attribution \\
            --repo mlx-community/Qwen3.5-27B-4bit --b 4 --warmup 3 --iters 20 \\
            --out plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as mlx_cache

TIMING_LOG: list[tuple[int, bool, int]] = []  # (layer_idx, is_linear, ns)


def _make_timed_layer(original: Any, layer_idx: int) -> Callable[..., Any]:
    """Build a callable wrapper that times ``original`` and forces eval."""
    is_linear = bool(getattr(original, "is_linear", False))

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter_ns()
        out = original(*args, **kwargs)
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, tuple):
            for a in out:
                if isinstance(a, mx.array):
                    mx.eval(a)
        elapsed = time.perf_counter_ns() - t0
        TIMING_LOG.append((layer_idx, is_linear, elapsed))
        return out

    wrapper.is_linear = is_linear  # type: ignore[attr-defined]
    return wrapper


def _install_timing(model: Any) -> list[Any]:
    """Replace each ``model.layers[i]`` with a timed wrapper, return originals."""
    originals = list(model.layers)
    for i, layer in enumerate(originals):
        model.layers[i] = _make_timed_layer(layer, i)
    return originals


def _restore_timing(model: Any, originals: list[Any]) -> None:
    """Restore ``model.layers`` to the original references."""
    for i, layer in enumerate(originals):
        model.layers[i] = layer


def _short_prompt_tokens(tokenizer: Any, target_len: int) -> mx.array:
    """Encode a short prompt and right-pad/truncate to ``target_len``."""
    text = (
        "Memory bandwidth has emerged as the dominant constraint in "
        "single-stream autoregressive decoding for large language "
        "models on consumer hardware platforms. Each decode step must "
        "read the entire active parameter set from unified memory "
        "before any computation can begin. On Apple Silicon the "
        "available bandwidth caps the achievable tokens per second "
        "well below what arithmetic throughput would otherwise allow. "
        "As model parameter counts continue to grow, this bottleneck "
        "becomes more pronounced. Hardware vendors respond with wider "
        "memory interfaces and dedicated neural acceleration units."
    )
    ids = tokenizer.encode(text)
    if len(ids) < target_len:
        ids = ids + [ids[-1]] * (target_len - len(ids))
    ids = ids[:target_len]
    return mx.array(ids, dtype=mx.int32)


def run_attribution(
    repo: str,
    b: int,
    warmup_iters: int,
    measurement_iters: int,
    prompt_len: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run per-step decode attribution; emit JSONL; return summary dict."""
    from silica.models.factory import adapter_for_repo
    from silica.mlx.runner import forward_batched, forward_batched_full

    print(f"[attribution] loading adapter for {repo} (cache-warm)...")
    adapter, _kv = adapter_for_repo(repo)
    model = adapter._model  # noqa: SLF001 — bench-side introspection.

    n_layers = len(model.layers)
    n_linear = sum(1 for layer in model.layers if getattr(layer, "is_linear", False))
    n_full = n_layers - n_linear
    print(f"[attribution] num_layers={n_layers} (linear={n_linear}, full={n_full})")

    tokens_1d = _short_prompt_tokens(adapter.tokenizer(), prompt_len)
    print(f"[attribution] prompt tokens: {tokens_1d.shape[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    originals = _install_timing(model)
    try:
        # Warmup phase: prefill + decode on a fresh cache, discard timing.
        for warmup_idx in range(warmup_iters):
            cache_list = mlx_cache.make_prompt_cache(model)
            tokens_b = mx.tile(tokens_1d[None], (b, 1))
            _ = forward_batched_full(model, tokens_b, cache_list)
            mx.eval(_)
            single = mx.zeros((b, 1), dtype=mx.int32)
            _ = forward_batched(model, single, cache_list)
            mx.eval(_)
            TIMING_LOG.clear()
        print(f"[attribution] warmup done ({warmup_iters} iters)")

        # Measurement phase.
        for it in range(measurement_iters):
            cache_list = mlx_cache.make_prompt_cache(model)
            tokens_b = mx.tile(tokens_1d[None], (b, 1))

            # Prefill (untimed; prime the cache).
            _ = forward_batched_full(model, tokens_b, cache_list)
            mx.eval(_)
            TIMING_LOG.clear()

            # Time exactly one decode step.
            single = mx.zeros((b, 1), dtype=mx.int32)
            t_step_start = time.perf_counter_ns()
            logits = forward_batched(model, single, cache_list)
            mx.eval(logits)
            t_step_total_ns = time.perf_counter_ns() - t_step_start

            iter_layer_times = list(TIMING_LOG)
            TIMING_LOG.clear()

            for (layer_idx, is_linear, ns) in iter_layer_times:
                rows.append(
                    {
                        "kind": "per_layer",
                        "iter": it,
                        "layer_idx": layer_idx,
                        "is_linear": is_linear,
                        "wall_ms": ns / 1e6,
                        "b": b,
                        "prompt_len": prompt_len,
                        "ts": ts,
                    }
                )
            rows.append(
                {
                    "kind": "step_total",
                    "iter": it,
                    "wall_ms": t_step_total_ns / 1e6,
                    "b": b,
                    "prompt_len": prompt_len,
                    "ts": ts,
                }
            )
            sum_layer_ms = sum(ns for (_, _, ns) in iter_layer_times) / 1e6
            overhead_ms = t_step_total_ns / 1e6 - sum_layer_ms
            rows.append(
                {
                    "kind": "instrumented_overhead",
                    "iter": it,
                    "wall_ms": overhead_ms,
                    "sum_layer_ms": sum_layer_ms,
                    "b": b,
                    "prompt_len": prompt_len,
                    "ts": ts,
                }
            )

    finally:
        _restore_timing(model, originals)
        TIMING_LOG.clear()

    # Aggregate
    step_totals = [r["wall_ms"] for r in rows if r["kind"] == "step_total"]
    overheads = [r["wall_ms"] for r in rows if r["kind"] == "instrumented_overhead"]
    linear_per_iter: dict[int, list[float]] = {}
    full_per_iter: dict[int, list[float]] = {}
    for r in rows:
        if r["kind"] == "per_layer":
            target = linear_per_iter if r["is_linear"] else full_per_iter
            target.setdefault(r["iter"], []).append(r["wall_ms"])
    linear_iter_totals = [sum(v) for v in linear_per_iter.values()]
    full_iter_totals = [sum(v) for v in full_per_iter.values()]

    def _stat(xs: list[float]) -> dict[str, float]:
        return {
            "n": len(xs),
            "median_ms": statistics.median(xs) if xs else 0.0,
            "mean_ms": statistics.mean(xs) if xs else 0.0,
            "min_ms": min(xs) if xs else 0.0,
            "max_ms": max(xs) if xs else 0.0,
        }

    summary: dict[str, Any] = {
        "kind": "summary",
        "ts": ts,
        "repo": repo,
        "b": b,
        "n_layers": n_layers,
        "n_linear": n_linear,
        "n_full": n_full,
        "warmup_iters": warmup_iters,
        "measurement_iters": measurement_iters,
        "prompt_len": prompt_len,
        "step_total_ms": _stat(step_totals),
        "linear_layers_total_ms": _stat(linear_iter_totals),
        "full_layers_total_ms": _stat(full_iter_totals),
        "instrumented_overhead_ms": _stat(overheads),
    }
    if step_totals:
        med_step = statistics.median(step_totals)
        summary["linear_pct_of_step"] = (
            statistics.median(linear_iter_totals) / med_step
            if linear_iter_totals
            else 0.0
        )
        summary["full_pct_of_step"] = (
            statistics.median(full_iter_totals) / med_step
            if full_iter_totals
            else 0.0
        )
        summary["overhead_pct_of_step"] = (
            statistics.median(overheads) / med_step if overheads else 0.0
        )
    summary["mean_per_linear_layer_ms"] = (
        statistics.median(linear_iter_totals) / n_linear
        if n_linear and linear_iter_totals
        else 0.0
    )
    summary["mean_per_full_layer_ms"] = (
        statistics.median(full_iter_totals) / n_full
        if n_full and full_iter_totals
        else 0.0
    )
    rows.append(summary)

    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[attribution] wrote {len(rows)} rows to {out_path}")
    print(f"[attribution] step_total median = "
          f"{summary['step_total_ms']['median_ms']:.2f} ms")
    print(f"[attribution] linear (DeltaNet) layers total = "
          f"{summary['linear_layers_total_ms']['median_ms']:.2f} ms "
          f"({summary['linear_pct_of_step']:.1%} of step)")
    print(f"[attribution] full attention layers total = "
          f"{summary['full_layers_total_ms']['median_ms']:.2f} ms "
          f"({summary['full_pct_of_step']:.1%} of step)")
    print(f"[attribution] mean per-linear-layer = "
          f"{summary['mean_per_linear_layer_ms']:.3f} ms")
    print(f"[attribution] mean per-full-layer = "
          f"{summary['mean_per_full_layer_ms']:.3f} ms")
    print(f"[attribution] instrumented overhead = "
          f"{summary['instrumented_overhead_ms']['median_ms']:.2f} ms "
          f"({summary['overhead_pct_of_step']:.1%} of step)")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plans/P6_AUTORESEARCH/decode_step_attribution.jsonl"),
    )
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit(
            "SILICA_REAL_QWEN3_5_27B=1 required to load the production "
            "target. Set the env var if you have authorised the real-model run."
        )

    run_attribution(
        repo=args.repo,
        b=args.b,
        warmup_iters=args.warmup,
        measurement_iters=args.iters,
        prompt_len=args.prompt_len,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
