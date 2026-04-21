"""scripts/vqbench_baseline.py — P-4.4 CLI for subprocess PPL collection.

Invocation:

    # Default script (reproduce_qwen35_4b_headline.py) with default args
    python scripts/vqbench_baseline.py \
        --python-executable /path/to/vqbench/venv/bin/python

    # Custom vqbench reproduce-script + forwarded args, JSONL output
    python scripts/vqbench_baseline.py \
        --script vqbench/scripts/reproduce_qwen35_4b_headline.py \
        --python-executable /path/to/vqbench/venv/bin/python \
        --out vqbench-baseline.jsonl \
        -- \
        --bits 4 --method TurboQuantMSE

The ``--`` separator (argparse convention) delimits flags meant
for this CLI from the trailing args forwarded to the vqbench
script. Everything after ``--`` lands in ``script_args``.

Exit codes:
  * 0 — the subprocess ran and the headline row parsed cleanly.
  * 1 — status="failed" (subprocess non-zero / timeout / parse).
  * 2 — argument parsing rejected the invocation.

The silica-mlx venv does NOT depend on torch / transformers /
datasets (D-009 forbids it in hot paths), so a default
``--python-executable`` would run the vqbench script under a
Python that crashes on ``import transformers``. Users point
``--python-executable`` at a dedicated vqbench venv's Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

# Ensure the repo root is importable when run as `python scripts/vqbench_baseline.py`
# from any cwd — mirrors the pattern used by every other scripts/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.bench.vqbench_baseline import (  # noqa: E402
    VqbenchBaselineResult,
    default_reproduce_script_path,
    run_vqbench_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silica-vqbench-baseline",
        description=(
            "Run a vqbench reproduce-script in a subprocess and "
            "collect its PPL headline row. Serves PLAN §P-4's "
            "vqbench_baseline deliverable — the reference column "
            "for P-5 numeric cross-checks."
        ),
    )
    p.add_argument(
        "--script",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "vqbench reproduce-script to run. Default: the "
            "checked-in reproduce_qwen35_4b_headline.py."
        ),
    )
    p.add_argument(
        "--python-executable",
        dest="python_executable",
        default=None,
        metavar="PATH",
        help=(
            "Python interpreter to invoke the script with. "
            "Required for real runs because the silica venv does "
            "NOT ship torch / transformers; point this at a "
            "dedicated vqbench venv's Python. Defaults to the "
            "current interpreter (useful only for parse-failure "
            "dry runs)."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help=(
            "Subprocess timeout (default 600 s; the reference "
            "reproduce_qwen35_4b_headline.py takes ~60 s on M5 Pro)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="append one JSONL row of VqbenchBaselineResult to PATH",
    )
    p.add_argument(
        "script_args",
        nargs="*",
        default=[],
        help=(
            "positional args forwarded to the vqbench script; "
            "separate from CLI flags with `--` as usual"
        ),
    )
    return p


def _print_summary(result: VqbenchBaselineResult) -> None:
    print(f"status              = {result.status}")
    if result.reason:
        print(f"reason              = {result.reason}")
    if result.script:
        print(f"script              = {result.script}")
    if result.python_executable:
        print(f"python_executable   = {result.python_executable}")
    if result.script_args:
        print(f"script_args         = {' '.join(result.script_args)}")
    if result.model is not None:
        print(f"model               = {result.model}")
        print(f"method              = {result.method}")
        print(f"bits                = {result.bits}")
        assert result.ppl_fp16 is not None
        print(f"ppl_fp16            = {result.ppl_fp16:.4f}")
        assert result.ppl_quant is not None
        print(f"ppl_quant           = {result.ppl_quant:.4f}")
        assert result.delta_ppl is not None
        assert result.delta_pct is not None
        print(f"delta_ppl           = {result.delta_ppl:+.4f}")
        print(f"delta_pct           = {result.delta_pct:+.2f}%")
    if result.wall_s is not None:
        print(f"wall_s              = {result.wall_s:.1f}")
    if result.returncode is not None:
        print(f"returncode          = {result.returncode}")


def _write_jsonl(result: VqbenchBaselineResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = args.script if args.script is not None else default_reproduce_script_path()

    result = run_vqbench_baseline(
        script=script,
        script_args=args.script_args,
        python_executable=args.python_executable,
        timeout_s=args.timeout,
    )

    _print_summary(result)
    if args.out is not None:
        _write_jsonl(result, args.out)

    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
