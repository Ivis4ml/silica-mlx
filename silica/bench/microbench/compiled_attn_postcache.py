"""β.1 microbench: mx.compile on Qwen3.5 attention post-cache half.

Reproduces cycle-16's 1.08× claim ("attention forward without cache
mutation: 1.08×" per ``plans/P6_AUTORESEARCH_NOTES.md``) on the sonnet
toolchain (``mlx==0.31.1``) at production attention shapes.

Why this microbench. Cycle 16 measured a synthetic post-cache callable;
that result has not been integration-tested. β.1's job is to confirm
the win exists on the current stack at production shapes, characterise
``mx.compile`` shape-recompilation cost (variable T_kv at decode), and
gate β.2 integration.

What is the post-cache half. From
``silica/kernels/shadow_install.py:80-136`` (v10 path) and the
upstream ``mlx_lm.models.qwen3_next.Qwen3NextAttention.__call__``:

    queries, keys, values = ... cache.update_and_fetch(keys, values) ...
    # post-cache region begins:
    output = scaled_dot_product_attention(queries, keys, values, ...)
    output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
    gated  = output * mx.sigmoid(gate_flat)
    return self.o_proj(gated)

That post-cache region is what we time. Cache mutation (rope-with-offset
+ ``cache.update_and_fetch``) is Python-side and is excluded from the
traced region by design — it depends on mutable cache state that
``mx.compile`` cannot trace cleanly.

Three timing arms per (T_kv) shape:

    1. Uncompiled — direct Python call.
    2. ``mx.compile(fn, shapeless=True)`` — variable T_kv supported in a
       single compiled trace; this is the realistic decode path.
    3. ``mx.compile(fn)`` (default ``shapeless=False``) — fixed-shape
       compile; faster ceiling but recompiles when T_kv changes (so per
       decode step in production it would recompile each call, dominating
       gain). Reported as the upper bound only.

Decision rule for the single-session, pre-variance screen:

    PASS_SHAPELESS    — best shapeless speedup ≥ 1.05× across T_kv values.
                        The combined report must still apply the
                        cross-session σ_ratio gate before β.2 opens.
    PASS_FIXED_ONLY   — only fixed-shape clears 1.05×; shapeless does
                        not. β.2 must add a shape-bucketing layer or
                        β closes with a design negative. Surface this
                        case to the user explicitly.
    FAIL              — neither arm clears 1.05× on any T_kv. β closes
                        with a measurement-anchored negative; γ next.

Output schema (JSONL, one row per record):

    {"kind": "shape_result", "t_kv": int, "b": int, "num_q_heads": int,
     "num_kv_heads": int, "head_dim": int, "dtype": str,
     "uncompiled_median_ms": float, "shapeless_compiled_median_ms": float,
     "fixed_compiled_median_ms": float, "speedup_shapeless": float,
     "speedup_fixed": float, "iters": int, "warmup": int, "ts": str}
    {"kind": "summary", ...}

Usage:

    SILICA_REAL_QWEN3_5_27B=1 \\
        uv run python -m silica.bench.microbench.compiled_attn_postcache \\
            --repo mlx-community/Qwen3.5-27B-4bit --b 4 \\
            --t-kv-list 128,1024,4096 --warmup 3 --iters 20 \\
            --out plans/P6_SMALL_B/BETA/microbench/compiled_attn_postcache_b4.jsonl
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


def find_full_attention_layer(model: Any) -> tuple[int, Any]:
    """Return ``(layer_idx, layer)`` for the first non-DeltaNet layer.

    Hybrid Qwen3.5 layers carry ``is_linear=True`` for DeltaNet blocks;
    full-attention blocks are ``is_linear=False`` and carry ``self_attn``
    instead of ``linear_attn``.
    """
    for i, layer in enumerate(model.layers):
        if not getattr(layer, "is_linear", False):
            return i, layer
    raise RuntimeError("no full-attention layer found in model.layers")


def post_cache_call(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    gate_flat: mx.array,
    scale: float,
    o_proj: Callable[[mx.array], mx.array],
) -> mx.array:
    """Post-cache half of ``Qwen3NextAttention.__call__``.

    Inputs: queries (B, H_q, T_q=1, D), keys (B, H_kv, T_kv, D),
    values (B, H_kv, T_kv, D), gate_flat (B, T_q=1, H_q*D).

    Output: o_proj(gated) of shape (B, T_q=1, hidden_dim).

    Calls ``mx.fast.scaled_dot_product_attention`` directly. The mlx-lm
    wrapper (``mlx_lm.models.base.scaled_dot_product_attention``) only
    branches to a quantized-cache path when ``cache.bits`` exists; for
    the production Qwen3.5-27B-4bit + standard cache configuration that
    branch is dead, so calling the fast op directly keeps the
    microbench representative without paying for wrapper Python
    overhead inside the timed region.
    """
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=None
    )
    b = queries.shape[0]
    output = output.transpose(0, 2, 1, 3).reshape(b, 1, -1)
    gated = output * mx.sigmoid(gate_flat)
    return o_proj(gated)


def _synth_inputs(
    b: int,
    t_kv: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    seed: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build representative-shape inputs at fixed seed."""
    keypair = mx.random.split(mx.random.key(seed), 4)
    q = mx.random.normal(
        (b, num_q_heads, 1, head_dim), key=keypair[0], dtype=mx.float32
    ).astype(dtype)
    k = mx.random.normal(
        (b, num_kv_heads, t_kv, head_dim), key=keypair[1], dtype=mx.float32
    ).astype(dtype)
    v = mx.random.normal(
        (b, num_kv_heads, t_kv, head_dim), key=keypair[2], dtype=mx.float32
    ).astype(dtype)
    g = mx.random.normal(
        (b, 1, num_q_heads * head_dim), key=keypair[3], dtype=mx.float32
    ).astype(dtype)
    mx.eval(q, k, v, g)
    return q, k, v, g


def _time_arm(
    fn: Callable[..., mx.array],
    inputs: tuple[mx.array, ...],
    warmup: int,
    iters: int,
) -> list[float]:
    """Wall-time ``fn(*inputs)`` over ``iters`` with ``warmup`` priming."""
    for _ in range(warmup):
        out = fn(*inputs)
        mx.eval(out)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = fn(*inputs)
        mx.eval(out)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return times


def run_microbench(
    repo: str,
    b: int,
    t_kv_values: list[int],
    warmup_iters: int,
    measurement_iters: int,
    dtype_str: str,
    out_path: Path,
) -> dict[str, Any]:
    """Run the three-arm post-cache microbench; emit JSONL; return summary."""
    from silica.models.factory import adapter_for_repo

    print(f"[β.1] loading adapter for {repo} (cache-warm)...")
    adapter, _ = adapter_for_repo(repo)
    model = adapter._model  # type: ignore[attr-defined]  # noqa: SLF001 — bench-side introspection.

    layer_idx, layer = find_full_attention_layer(model)
    self_attn = layer.self_attn
    print(f"[β.1] full-attention layer at model.layers[{layer_idx}]")

    num_q_heads = int(self_attn.num_attention_heads)
    num_kv_heads = int(self_attn.num_key_value_heads)
    head_dim = int(self_attn.head_dim)
    scale = float(self_attn.scale)
    o_proj = self_attn.o_proj
    print(
        f"[β.1] shape config: b={b} "
        f"num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} "
        f"head_dim={head_dim} scale={scale:.6f}"
    )

    dtype = mx.bfloat16 if dtype_str == "bfloat16" else mx.float16

    # Closures over scale / o_proj. Both compiled arms trace the same
    # callable; the difference is only the shapeless flag.
    def fn_uncompiled(q: mx.array, k: mx.array, v: mx.array, g: mx.array) -> mx.array:
        return post_cache_call(q, k, v, g, scale, o_proj)

    fn_shapeless = mx.compile(fn_uncompiled, shapeless=True)
    fn_fixed = mx.compile(fn_uncompiled)

    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for t_kv in t_kv_values:
        print(f"[β.1] T_kv={t_kv}: timing all three arms...")
        inputs = _synth_inputs(
            b, t_kv, num_q_heads, num_kv_heads, head_dim, dtype, seed=t_kv
        )

        # Arm 1: uncompiled.
        u_times = _time_arm(fn_uncompiled, inputs, warmup_iters, measurement_iters)
        # Arm 2: shapeless compile.
        sl_times = _time_arm(fn_shapeless, inputs, warmup_iters, measurement_iters)
        # Arm 3: fixed-shape compile.
        fx_times = _time_arm(fn_fixed, inputs, warmup_iters, measurement_iters)

        u_med = statistics.median(u_times)
        sl_med = statistics.median(sl_times)
        fx_med = statistics.median(fx_times)
        sp_sl = u_med / sl_med if sl_med > 0 else float("nan")
        sp_fx = u_med / fx_med if fx_med > 0 else float("nan")

        row = {
            "kind": "shape_result",
            "t_kv": t_kv,
            "b": b,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": dtype_str,
            "uncompiled_median_ms": u_med,
            "shapeless_compiled_median_ms": sl_med,
            "fixed_compiled_median_ms": fx_med,
            "uncompiled_min_ms": min(u_times),
            "shapeless_compiled_min_ms": min(sl_times),
            "fixed_compiled_min_ms": min(fx_times),
            "speedup_shapeless": sp_sl,
            "speedup_fixed": sp_fx,
            "iters": measurement_iters,
            "warmup": warmup_iters,
            "ts": ts,
        }
        rows.append(row)
        print(
            f"  uncompiled        = {u_med:7.3f} ms"
            f"  (min {min(u_times):7.3f})"
        )
        print(
            f"  shapeless compile = {sl_med:7.3f} ms  ({sp_sl:5.3f}×)"
            f"  (min {min(sl_times):7.3f})"
        )
        print(
            f"  fixed-shape       = {fx_med:7.3f} ms  ({sp_fx:5.3f}×)"
            f"  (min {min(fx_times):7.3f})"
        )

    # Best-of-shapes verdict for this single session. Opening §4.1's
    # line-level gate also requires a combined cross-session σ_ratio check.
    best_sl = max(r["speedup_shapeless"] for r in rows)
    best_fx = max(r["speedup_fixed"] for r in rows)
    pass_sl = best_sl >= 1.05
    pass_fx = best_fx >= 1.05

    if pass_sl:
        verdict = "PASS_SHAPELESS"
    elif pass_fx:
        verdict = "PASS_FIXED_ONLY"
    else:
        verdict = "FAIL"

    summary = {
        "kind": "summary",
        "repo": repo,
        "b": b,
        "t_kv_values": t_kv_values,
        "dtype": dtype_str,
        "iters": measurement_iters,
        "warmup": warmup_iters,
        "best_speedup_shapeless": best_sl,
        "best_speedup_fixed": best_fx,
        "pass_shapeless_1_05x": pass_sl,
        "pass_fixed_1_05x": pass_fx,
        "verdict": verdict,
        "ts": ts,
    }
    rows.append(summary)

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print()
    print(f"[β.1] best shapeless speedup: {best_sl:.3f}×")
    print(f"[β.1] best fixed-shape speedup: {best_fx:.3f}×")
    print(f"[β.1] single-session verdict before σ gate: {verdict}")
    print(f"[β.1] wrote {len(rows)} rows to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument(
        "--t-kv-list",
        default="128,1024,4096",
        help="comma-separated T_kv values to sweep (default 128,1024,4096)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16"]
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit(
            "SILICA_REAL_QWEN3_5_27B=1 required to load the production target."
        )

    t_kv_values = [int(x.strip()) for x in args.t_kv_list.split(",") if x.strip()]
    if not t_kv_values:
        raise SystemExit("--t-kv-list produced an empty list")

    run_microbench(
        repo=args.repo,
        b=args.b,
        t_kv_values=t_kv_values,
        warmup_iters=args.warmup,
        measurement_iters=args.iters,
        dtype_str=args.dtype,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
