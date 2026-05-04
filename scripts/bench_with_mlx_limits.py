"""Run the standard bench CLI after setting MLX memory-limit knobs.

This is a thin wrapper around ``scripts/bench.py`` for P-6 autoresearch
probes such as the 40 GB cliff investigation. MLX memory/cache limits must
be set inside the same Python process that loads the model; exporting an
environment variable or invoking ``scripts/bench.py`` as a subprocess would
miss that timing.

Examples:
    uv run python scripts/bench_with_mlx_limits.py \
        --cache-limit-gb 1 \
        --scenario qwen3.5-27b-warm-decode-b66 \
        --out plans/P6_AUTORESEARCH/cliff_b66_cache1.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench as bench_cli  # noqa: E402


def _bytes_from_gb(value: float | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("limit values must be non-negative")
    return int(value * 1_000_000_000)


def _setter(name: str):
    direct = getattr(mx, name, None)
    if direct is not None:
        return direct
    metal = getattr(mx, "metal", None)
    if metal is not None:
        return getattr(metal, name, None)
    return None


def _apply_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    fn = _setter(name)
    if fn is None:
        raise RuntimeError(f"MLX does not expose {name}")
    previous = fn(value)
    print(
        f"[mlx-limit] {name}={value} bytes "
        f"(previous {previous} bytes)",
        file=sys.stderr,
    )


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--cache-limit-gb",
        type=float,
        default=None,
        help="set mx.set_cache_limit / mx.metal.set_cache_limit before loading",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="set mx.set_memory_limit / mx.metal.set_memory_limit before loading",
    )
    parser.add_argument(
        "--wired-limit-gb",
        type=float,
        default=None,
        help="set mx.set_wired_limit / mx.metal.set_wired_limit before loading",
    )
    ns, forwarded = parser.parse_known_args(argv)
    return ns, forwarded


def main(argv: list[str] | None = None) -> int:
    ns, forwarded = _parse_args(list(sys.argv[1:] if argv is None else argv))
    _apply_limit("set_cache_limit", _bytes_from_gb(ns.cache_limit_gb))
    _apply_limit("set_memory_limit", _bytes_from_gb(ns.memory_limit_gb))
    _apply_limit("set_wired_limit", _bytes_from_gb(ns.wired_limit_gb))
    return bench_cli.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
