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
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.models.qwen3_5 import Qwen3_5Adapter

DEFAULT_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache(repo: str) -> Path:
    flat = repo.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{flat}"


def _bench_one(
    *,
    adapter_factory: Callable[[], tuple[Qwen3_5Adapter, SimpleKVCache]],
    k: int,
    capture_layer_ids: frozenset[int] | None,
    warmup: int,
    iters: int,
) -> float:
    """Return median per-call wall-clock seconds.

    Each iteration loads a fresh adapter so the verify forward starts
    from an empty KV cache (matching the (αβ) microbench precondition:
    cycle-1 cost, no warm cache to amortise the embed_tokens lookup
    against). The factory closure encapsulates the load.
    """
    samples: list[float] = []
    handle = KVHandle(req_id="capture-bench")
    tokens = mx.array([101] * k, dtype=mx.int32)

    for _ in range(warmup):
        adapter, kv = adapter_factory()
        kv.reserve_for_prefill(handle.req_id, [])  # type: ignore[arg-type]
        if capture_layer_ids is None:
            adapter.decode_step_multi(tokens, handle)
        else:
            adapter.decode_step_multi_with_capture(
                tokens, handle, capture_layer_ids
            )

    for _ in range(iters):
        adapter, kv = adapter_factory()
        kv.reserve_for_prefill(handle.req_id, [])  # type: ignore[arg-type]
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

    # Probe-load once for the layer-count + hidden-size header.
    adapter, _ = Qwen3_5Adapter.from_hf_repo(args.repo)
    num_layers = adapter.config.num_layers
    hidden_size = adapter.config.hidden_size
    if args.capture_layer_ids is None:
        capture_set = frozenset({0, num_layers // 2, num_layers})
    else:
        capture_set = frozenset(
            int(x) for x in args.capture_layer_ids.split(",")
        )

    def factory() -> tuple[Qwen3_5Adapter, SimpleKVCache]:
        return Qwen3_5Adapter.from_hf_repo(args.repo)

    print(
        f"# c_capture_hidden microbench — repo={args.repo}, k={args.k}, "
        f"layers={num_layers}, hidden={hidden_size}"
    )
    print(
        f"# capture layer ids = {sorted(capture_set)} "
        f"(|capture| = {len(capture_set)})"
    )
    print(f"# warmup={args.warmup}, iters={args.iters} (median reported)")

    baseline_s = _bench_one(
        adapter_factory=factory,
        k=args.k,
        capture_layer_ids=None,
        warmup=args.warmup,
        iters=args.iters,
    )
    capture_s = _bench_one(
        adapter_factory=factory,
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
