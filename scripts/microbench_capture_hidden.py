"""D-021 step 6 (αβ.1) microbench — c_capture_hidden(k) on Qwen3.5-0.8B.

Measures the additive cost of capturing target hidden states at the
drafter-consumed layer ids during a verify forward, relative to the
existing ``decode_step_multi(k=16)`` baseline. Closes OQ-5 in
``plans/P6_C4_DFLASH_OPENING.md`` §5.5.

Usage:

    python -m scripts.microbench_capture_hidden \\
        --repo Qwen/Qwen3.5-0.8B --k 16 --warmup 3 --iters 20

The 0.8B fixture's c_capture_hidden(k) is a *lower bound* on the 27B
target's cost — bandwidth utilisation differs and the layer-output
hidden state is wider. The 0.8B number is used to gate sub-unit (β)
landing: if the relative cost on 0.8B already exceeds 25%, we either
revisit the capture-set choice (capturing fewer layers) or open a
re-evaluation of the §1 prediction band before (β) commits any
drafter-wrapper code.

Not gated as a bench scenario — this is a one-shot microbench whose
output lands in ``plans/P6_C4_DFLASH/REPORT.md`` as the (αβ.1)
precondition row.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.models.factory import adapter_for_repo
from silica.models.hidden_capture import HiddenCaptureAdapter

DEFAULT_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache(repo: str) -> Path:
    flat = repo.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{flat}"


def _bench_one(
    *,
    adapter: Any,
    kv: SimpleKVCache,
    k: int,
    capture_layer_ids: frozenset[int] | None,
    warmup: int,
    iters: int,
) -> float:
    """Return median per-call wall-clock seconds.

    The adapter is loaded once by the caller and reused; each iteration
    runs ``decode_step_multi`` (or the capture variant) against a
    freshly-reserved request id so the verify forward starts from an
    empty KV cache. This isolates ``c_capture_hidden(k)`` from
    model-load cost — important on heavy MoE fixtures where a 20 GB
    fresh-load dominates the wall-clock and drowns the signal we care
    about.
    """
    samples: list[float] = []
    tokens = mx.array([101] * k, dtype=mx.int32)

    def _release(req_id: str) -> None:
        # Adapter-side per-request snapshot (Qwen3.5 hybrid only).
        if hasattr(adapter, "free_state"):
            adapter.free_state(req_id)
        # SimpleKVCache itself is single-request; free its claim so
        # the next req_id can ``reserve_for_prefill``.
        if hasattr(kv, "free"):
            kv.free(req_id)

    for i in range(warmup):
        req_id = f"capture-bench-warmup-{i}"
        handle = KVHandle(req_id=req_id)
        kv.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
        if capture_layer_ids is None:
            adapter.decode_step_multi(tokens, handle)
        else:
            adapter.decode_step_multi_with_capture(
                tokens, handle, capture_layer_ids
            )
        _release(req_id)

    for i in range(iters):
        req_id = f"capture-bench-iter-{i}"
        handle = KVHandle(req_id=req_id)
        kv.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
        # mx.synchronize is not exposed; mx.eval forces materialisation.
        # Use a tight wall-clock window.
        t0 = time.perf_counter()
        if capture_layer_ids is None:
            logits, _ = adapter.decode_step_multi(tokens, handle)
            mx.eval(logits)
        else:
            logits, captured, _ = adapter.decode_step_multi_with_capture(
                tokens, handle, capture_layer_ids
            )
            # Materialise captured arrays to include them in the wall-clock.
            mx.eval(logits, *captured.values())
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        _release(req_id)

    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--capture-layer-ids",
        type=str,
        default=None,
        help=(
            "comma-separated layer ids to capture. Defaults to a "
            "spread of three layers (0, num_layers // 2, num_layers)."
        ),
    )
    args = parser.parse_args()

    cache = _hf_cache(args.repo)
    if not cache.exists():
        print(
            f"error: HF cache for {args.repo} missing at {cache}. "
            f"Run `huggingface-cli download {args.repo}` first.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("SILICA_SKIP_MODEL_TESTS"):
        print(
            "skipping: SILICA_SKIP_MODEL_TESTS set",
            file=sys.stderr,
        )
        return 0

    # Load once for the layer-count + hidden-size header AND reuse for
    # both bench runs. Dispatch via adapter_for_repo so dense and MoE
    # Qwen3.5 fixtures both go through the registered adapter class.
    probe_adapter, probe_kv = adapter_for_repo(args.repo)
    if not isinstance(probe_adapter, HiddenCaptureAdapter):
        print(
            f"error: adapter for {args.repo} ({type(probe_adapter).__name__}) "
            "does not implement HiddenCaptureAdapter. (αβ.1) covers Qwen3.5 "
            "dense; (αβ.2) extends to Qwen3.5-MoE.",
            file=sys.stderr,
        )
        return 2
    num_layers = probe_adapter.config.num_layers  # type: ignore[attr-defined]
    hidden_size = probe_adapter.config.hidden_size  # type: ignore[attr-defined]
    if args.capture_layer_ids is None:
        capture_set = frozenset({0, num_layers // 2, num_layers})
    else:
        capture_set = frozenset(
            int(x) for x in args.capture_layer_ids.split(",")
        )

    print(
        f"# c_capture_hidden microbench — repo={args.repo}, k={args.k}, "
        f"layers={num_layers}, hidden={hidden_size}"
    )
    print(
        f"# capture layer ids = {sorted(capture_set)} "
        f"(|capture| = {len(capture_set)})"
    )
    print(f"# warmup={args.warmup}, iters={args.iters} (median reported)")
    print("# adapter loaded once and reused; iters use fresh per-req KV.")

    # Reuse the probe-loaded adapter for both runs so the load cost
    # does not contribute to the per-iteration wall-clock.
    baseline_s = _bench_one(
        adapter=probe_adapter,
        kv=probe_kv,
        k=args.k,
        capture_layer_ids=None,
        warmup=args.warmup,
        iters=args.iters,
    )
    capture_s = _bench_one(
        adapter=probe_adapter,
        kv=probe_kv,
        k=args.k,
        capture_layer_ids=capture_set,
        warmup=args.warmup,
        iters=args.iters,
    )

    delta_s = capture_s - baseline_s
    ratio = capture_s / baseline_s if baseline_s > 0 else float("nan")
    rel_pct = 100.0 * (ratio - 1.0)

    print()
    print(f"baseline  decode_step_multi(k={args.k})            "
          f"= {baseline_s * 1000:8.3f} ms")
    print(f"capture   decode_step_multi_with_capture(k={args.k}) "
          f"= {capture_s * 1000:8.3f} ms")
    print(f"delta     c_capture_hidden(k={args.k})              "
          f"= {delta_s * 1000:8.3f} ms")
    print(f"ratio     capture / baseline                         "
          f"= {ratio:.4f} ({rel_pct:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
