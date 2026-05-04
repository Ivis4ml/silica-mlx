"""Render the autoresearch progress chart from the TSV ledger.

Karpathy-autoresearch style: x = experiment number, y = primary metric,
discarded points small gray, kept improvements green, running best as a
green step line. Reference horizontal lines anchor the chip envelope.

Reads `plans/P6_AUTORESEARCH_LOG.tsv` and writes
`plans/P6_AUTORESEARCH_PROGRESS_<METRIC>.png`. Pure read-only on the
ledger; no other side effects.

Usage:
    uv run python scripts/render_autoresearch_chart.py
    uv run python scripts/render_autoresearch_chart.py --metric decode_tok_s
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

LEDGER = Path(__file__).resolve().parent.parent / "plans" / "P6_AUTORESEARCH_LOG.tsv"
OUTDIR = LEDGER.parent

STATUS_COLORS = {
    "keep": "#2ca02c",
    "discard": "#9e9e9e",
    "diagnostic": "#1f77b4",
    "crash": "#d62728",
}

REFERENCE_LINES_DECODE = [
    (16.05, "B=1 baseline (16.05 tok/s)", "#bbbbbb", ":"),
    (20.29, "B=1 weights ceiling (20.29 tok/s)", "#888888", ":"),
    (42.17, "B=4 running best (42.17 tok/s)", "#2ca02c", "-"),
    (60.00, "(1b) milestone (60 tok/s)", "#ff7f0e", "--"),
    (81.16, "B=4 weights ceiling (81.16 tok/s)", "#444444", ":"),
]


def load_ledger(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def filter_metric(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    return [r for r in rows if r["metric_name"] == metric]


def render(rows: list[dict[str, str]], metric: str, outpath: Path) -> None:
    if not rows:
        raise SystemExit(f"no rows match metric_name={metric!r}")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Per-experiment markers
    running_best = float("-inf")
    running_best_xs: list[int] = []
    running_best_ys: list[float] = []
    keep_xs: list[int] = []
    keep_ys: list[float] = []
    for i, row in enumerate(rows, start=1):
        val = float(row["metric_value"])
        status = row["status"].strip()
        color = STATUS_COLORS.get(status, "#888888")
        marker = "o" if status in ("keep", "diagnostic") else "x"
        size = 90 if status == "keep" else 55
        ax.scatter([i], [val], c=color, marker=marker, s=size, zorder=3,
                   edgecolors="black", linewidths=0.5)
        # Annotate experiment id
        ax.annotate(row["experiment_id"], (i, val), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7,
                    rotation=30, color=color)
        # Track running best on keep-only
        if status == "keep" and val > running_best:
            running_best = val
            keep_xs.append(i)
            keep_ys.append(val)
        # Reset running best once we have at least one keep
        if keep_xs:
            running_best_xs.append(i)
            running_best_ys.append(running_best)

    # Running-best step line, if any keeps exist
    if running_best_xs:
        ax.step(running_best_xs, running_best_ys, where="post",
                color="#2ca02c", linewidth=2, label="running best", zorder=2)

    # Reference horizontal lines
    if metric == "decode_tok_s":
        for y, label, color, linestyle in REFERENCE_LINES_DECODE:
            ax.axhline(y, color=color, linestyle=linestyle, linewidth=1, alpha=0.7)
            ax.text(len(rows) + 0.3, y, label, fontsize=7, va="center",
                    color=color)

    # Status legend (shown only for present statuses)
    present_statuses = sorted({r["status"].strip() for r in rows})
    legend_handles = []
    for status in present_statuses:
        color = STATUS_COLORS.get(status, "#888888")
        marker = "o" if status in ("keep", "diagnostic") else "x"
        legend_handles.append(plt.Line2D([0], [0], marker=marker, color="w",
                                         markerfacecolor=color, markersize=8,
                                         markeredgecolor="black",
                                         markeredgewidth=0.5, label=status))
    if running_best_xs:
        legend_handles.append(plt.Line2D([0], [0], color="#2ca02c",
                                         linewidth=2, label="running best"))
    ax.legend(handles=legend_handles, loc="upper left")

    n_keep = sum(1 for r in rows if r["status"].strip() == "keep")
    rb = max((float(r["metric_value"]) for r in rows
              if r["status"].strip() == "keep"), default=None)
    rb_str = f"{rb:.2f}" if rb is not None else "n/a (no keeps yet)"
    ax.set_title(
        f"Silica-MLX autoresearch — {len(rows)} experiments on {metric} / "
        f"{n_keep} kept / running best = {rb_str}",
        fontsize=11,
    )
    ax.set_xlabel("experiment #")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0.5, len(rows) + 4.5)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"wrote {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="decode_tok_s",
                        help="metric_name column to filter on")
    parser.add_argument("--ledger", default=str(LEDGER), type=Path,
                        help="path to TSV ledger")
    args = parser.parse_args()

    rows = load_ledger(args.ledger)
    filtered = filter_metric(rows, args.metric)
    out = OUTDIR / f"P6_AUTORESEARCH_PROGRESS_{args.metric.upper()}.png"
    render(filtered, args.metric, out)


if __name__ == "__main__":
    main()
