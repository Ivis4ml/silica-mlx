"""Layer-internal decomposition microbench for hybrid Qwen3.5 adapters.

Whereas ``decode_step_attribution`` instruments at the layer-block level,
this microbench instruments at the layer's child-module level:
``input_layernorm``, ``self_attn`` (or ``linear_attn``),
``post_attention_layernorm``, ``mlp`` separately. Goal: identify which
intra-layer component (norm vs attention/recurrence vs MLP) owns the
largest share of per-layer wall time, so kernel work targets the dominant
sub-component.

Methodology mirrors ``decode_step_attribution``: ``mx.eval`` barriers
after each child-module call; per-layer-kind summary; reused-adapter
shape; restore on exit.

Usage:
    SILICA_REAL_QWEN3_5_27B=1 \\
        uv run python -m silica.bench.microbench.layer_internal_attribution \\
            --b 4 --warmup 3 --iters 10 \\
            --out plans/P6_AUTORESEARCH/layer_internal_attribution_b4.jsonl
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

# (layer_idx, is_linear, child_name, ns)
TIMING_LOG: list[tuple[int, bool, str, int]] = []

CHILD_NAMES = (
    "input_layernorm",
    "self_attn",
    "linear_attn",
    "post_attention_layernorm",
    "mlp",
)


def _make_timed_child(
    original: Any, layer_idx: int, is_linear: bool, child_name: str
) -> Callable[..., Any]:
    """Wrap ``original`` to time its call, force eval, append to TIMING_LOG."""
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
        TIMING_LOG.append((layer_idx, is_linear, child_name, elapsed))
        return out

    return wrapper


def _install_child_timing(model: Any) -> list[tuple[int, str, Any]]:
    """Patch each layer's child modules with timed wrappers; return originals.

    Returns a list of ``(layer_idx, child_name, original)`` tuples for restore.
    """
    originals: list[tuple[int, str, Any]] = []
    for i, layer in enumerate(model.layers):
        is_linear = bool(getattr(layer, "is_linear", False))
        for child_name in CHILD_NAMES:
            if not hasattr(layer, child_name):
                continue
            original = getattr(layer, child_name)
            originals.append((i, child_name, original))
            wrapped = _make_timed_child(original, i, is_linear, child_name)
            setattr(layer, child_name, wrapped)
    return originals


def _restore_child_timing(
    model: Any, originals: list[tuple[int, str, Any]]
) -> None:
    """Restore each layer's child modules to their pre-instrumentation state."""
    for layer_idx, child_name, original in originals:
        setattr(model.layers[layer_idx], child_name, original)


def _short_prompt_tokens(tokenizer: Any, target_len: int) -> mx.array:
    text = (
        "Memory bandwidth has emerged as the dominant constraint in "
        "single-stream autoregressive decoding for large language "
        "models on consumer hardware platforms. Each decode step must "
        "read the entire active parameter set from unified memory "
        "before any computation can begin. On Apple Silicon the "
        "available bandwidth caps the achievable tokens per second."
    )
    ids = tokenizer.encode(text)
    if len(ids) < target_len:
        ids = ids + [ids[-1]] * (target_len - len(ids))
    ids = ids[:target_len]
    return mx.array(ids, dtype=mx.int32)


def run(
    repo: str,
    b: int,
    warmup_iters: int,
    measurement_iters: int,
    prompt_len: int,
    out_path: Path,
) -> dict[str, Any]:
    from silica.models.factory import adapter_for_repo
    from silica.mlx.runner import forward_batched, forward_batched_full

    print(f"[layer_internal] loading {repo}...")
    adapter, _kv = adapter_for_repo(repo)
    model = adapter._model  # noqa: SLF001

    n_layers = len(model.layers)
    n_linear = sum(1 for layer in model.layers if getattr(layer, "is_linear", False))
    n_full = n_layers - n_linear
    print(f"[layer_internal] num_layers={n_layers} (linear={n_linear}, full={n_full})")

    tokens_1d = _short_prompt_tokens(adapter.tokenizer(), prompt_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    originals = _install_child_timing(model)
    try:
        # Warmup
        for _ in range(warmup_iters):
            cache_list = mlx_cache.make_prompt_cache(model)
            tokens_b = mx.tile(tokens_1d[None], (b, 1))
            _ = forward_batched_full(model, tokens_b, cache_list)
            mx.eval(_)
            single = mx.zeros((b, 1), dtype=mx.int32)
            _ = forward_batched(model, single, cache_list)
            mx.eval(_)
            TIMING_LOG.clear()
        print(f"[layer_internal] warmup done ({warmup_iters} iters)")

        # Measurement
        for it in range(measurement_iters):
            cache_list = mlx_cache.make_prompt_cache(model)
            tokens_b = mx.tile(tokens_1d[None], (b, 1))
            _ = forward_batched_full(model, tokens_b, cache_list)
            mx.eval(_)
            TIMING_LOG.clear()

            single = mx.zeros((b, 1), dtype=mx.int32)
            t_step_start = time.perf_counter_ns()
            logits = forward_batched(model, single, cache_list)
            mx.eval(logits)
            t_step_total_ns = time.perf_counter_ns() - t_step_start

            iter_times = list(TIMING_LOG)
            TIMING_LOG.clear()

            for (layer_idx, is_linear, child_name, ns) in iter_times:
                rows.append({
                    "kind": "child",
                    "iter": it,
                    "layer_idx": layer_idx,
                    "is_linear": is_linear,
                    "child": child_name,
                    "wall_ms": ns / 1e6,
                    "b": b,
                    "ts": ts,
                })
            rows.append({
                "kind": "step_total",
                "iter": it,
                "wall_ms": t_step_total_ns / 1e6,
                "b": b,
                "ts": ts,
            })

    finally:
        _restore_child_timing(model, originals)
        TIMING_LOG.clear()

    # Aggregate per (is_linear, child) across all iters and layers.
    agg: dict[tuple[bool, str], list[float]] = {}
    per_iter_step: list[float] = []
    for r in rows:
        if r["kind"] == "step_total":
            per_iter_step.append(r["wall_ms"])
        elif r["kind"] == "child":
            agg.setdefault((r["is_linear"], r["child"]), []).append(r["wall_ms"])

    # Aggregate per-iter totals by (kind, child) so we can compute median over iters
    iter_totals: dict[int, dict[tuple[bool, str], float]] = {}
    for r in rows:
        if r["kind"] == "child":
            it = r["iter"]
            iter_totals.setdefault(it, {})
            iter_totals[it][(r["is_linear"], r["child"])] = (
                iter_totals[it].get((r["is_linear"], r["child"]), 0.0) + r["wall_ms"]
            )

    # Per-(kind, child) median across iters
    components_summary: dict[str, dict[str, Any]] = {}
    median_step_ms = statistics.median(per_iter_step) if per_iter_step else 0.0
    for (is_linear, child) in agg.keys():
        across_iters = [iter_totals[it].get((is_linear, child), 0.0) for it in iter_totals]
        med = statistics.median(across_iters) if across_iters else 0.0
        kind_label = "linear" if is_linear else "full"
        key = f"{kind_label}.{child}"
        n_layers_of_kind = n_linear if is_linear else n_full
        components_summary[key] = {
            "median_total_ms": med,
            "median_per_layer_ms": (med / n_layers_of_kind) if n_layers_of_kind else 0.0,
            "pct_of_step": (med / median_step_ms) if median_step_ms else 0.0,
            "n_layers_of_kind": n_layers_of_kind,
        }

    summary = {
        "kind": "summary",
        "ts": ts,
        "repo": repo,
        "b": b,
        "n_layers": n_layers,
        "n_linear": n_linear,
        "n_full": n_full,
        "warmup_iters": warmup_iters,
        "measurement_iters": measurement_iters,
        "step_total_ms": {
            "median": median_step_ms,
            "min": min(per_iter_step) if per_iter_step else 0.0,
            "max": max(per_iter_step) if per_iter_step else 0.0,
            "n": len(per_iter_step),
        },
        "components": components_summary,
    }
    rows.append(summary)

    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[layer_internal] wrote {len(rows)} rows to {out_path}")
    print(f"[layer_internal] step_total median = {median_step_ms:.2f} ms")
    print("[layer_internal] component breakdown:")
    for key in sorted(components_summary.keys()):
        v = components_summary[key]
        print(f"  {key:42s}  total={v['median_total_ms']:6.2f} ms "
              f"({v['pct_of_step']:5.1%})  per-layer={v['median_per_layer_ms']:.3f} ms "
              f"(over {v['n_layers_of_kind']} layers)")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plans/P6_AUTORESEARCH/layer_internal_attribution.jsonl"),
    )
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit("SILICA_REAL_QWEN3_5_27B=1 required.")

    run(
        repo=args.repo,
        b=args.b,
        warmup_iters=args.warmup,
        measurement_iters=args.iters,
        prompt_len=args.prompt_len,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
