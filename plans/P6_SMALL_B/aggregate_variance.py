"""Aggregate P-6 small-B sub-unit α measurements across sessions.

Reads every ``warm_decode_b{4,8,12}.jsonl`` and attribution-microbench
JSONL under ``plans/P6_SMALL_B/<session>/``, computes combined mean /
sample-std / n per scenario across sessions, gate-checks the α
variance protocol (combined σ ≤ 1.5 tok/s aggregate), and emits a
Markdown ``REPORT.md`` summary plus a JSON-line summary on stdout.

Usage:
    uv run python plans/P6_SMALL_B/aggregate_variance.py \\
        --root plans/P6_SMALL_B \\
        --out plans/P6_SMALL_B/REPORT.md

Deliberately stdlib-only (no numpy / pandas) so the tool runs in the
same environment as the bench commands without extra dependency
wrangling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ALPHA_SIGMA_GATE = 1.5  # tok/s on aggregate throughput, per opening §4 α

WARM_DECODE_FILES = {
    "qwen3.5-27b-warm-decode-b4": "warm_decode_b4.jsonl",
    "qwen3.5-27b-warm-decode-b8": "warm_decode_b8.jsonl",
    "qwen3.5-27b-warm-decode-b12": "warm_decode_b12.jsonl",
}

ATTRIBUTION_FILES = {
    "decode_step_attr_b4": "decode_step_attr_b4.jsonl",
    "decode_step_attr_b8": "decode_step_attr_b8.jsonl",
    "layer_internal_attr_b4": "layer_internal_attr_b4.jsonl",
}


@dataclass
class WarmDecodeRow:
    session: str
    seed: int
    aggregate_tok_s: float
    per_row_tok_s: float
    peak_memory_mb: float | None


@dataclass
class WarmDecodeStats:
    scenario_id: str
    rows: list[WarmDecodeRow] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def _sample_std(self, values: list[float]) -> float:
        if len(values) < 2:
            return float("nan")
        m = self._mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    @property
    def aggregate_mean(self) -> float:
        return self._mean([r.aggregate_tok_s for r in self.rows])

    @property
    def aggregate_std(self) -> float:
        return self._sample_std([r.aggregate_tok_s for r in self.rows])

    @property
    def per_row_mean(self) -> float:
        return self._mean([r.per_row_tok_s for r in self.rows])

    @property
    def per_row_std(self) -> float:
        return self._sample_std([r.per_row_tok_s for r in self.rows])

    @property
    def peak_memory_max_mb(self) -> float:
        peaks = [r.peak_memory_mb for r in self.rows if r.peak_memory_mb is not None]
        return max(peaks) if peaks else float("nan")

    @property
    def sessions(self) -> list[str]:
        return sorted({r.session for r in self.rows})

    @property
    def passes_alpha_gate(self) -> bool:
        return (
            self.n >= 6  # ≥3 reps × ≥2 sessions
            and len(self.sessions) >= 2
            and not math.isnan(self.aggregate_std)
            and self.aggregate_std <= ALPHA_SIGMA_GATE
        )


def _load_warm_decode(session_dir: Path, scenario_id: str, jsonl_name: str) -> list[WarmDecodeRow]:
    path = session_dir / jsonl_name
    if not path.exists():
        return []
    rows: list[WarmDecodeRow] = []
    for line_idx, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"malformed JSON in {path}:{line_idx + 1}: {exc}")
        if obj.get("scenario_id") != scenario_id:
            continue
        if obj.get("status") != "ok":
            continue
        meta = obj.get("metadata") or {}
        agg = meta.get("decode_tok_s_warm_aggregate")
        per_row = meta.get("decode_tok_s_warm_per_row_mean")
        if agg is None or per_row is None:
            continue
        rows.append(
            WarmDecodeRow(
                session=session_dir.name,
                seed=int(meta.get("seed", 0)),
                aggregate_tok_s=float(agg),
                per_row_tok_s=float(per_row),
                peak_memory_mb=obj.get("peak_memory_mb"),
            )
        )
    return rows


def _load_attribution_summaries(session_dir: Path, jsonl_name: str) -> list[dict]:
    """Return summary-kind rows from an attribution microbench JSONL."""

    path = session_dir / jsonl_name
    if not path.exists():
        return []
    summaries: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("kind") == "summary":
            obj["_session"] = session_dir.name
            summaries.append(obj)
    return summaries


def _format_table_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _fmt_float(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def collect(root: Path) -> tuple[dict[str, WarmDecodeStats], dict[str, list[dict]]]:
    sessions = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs")
    if not sessions:
        raise SystemExit(
            f"no session subdirectories found under {root}. "
            "expected plans/P6_SMALL_B/<timestamp>/<scenario>.jsonl per RUNBOOK.md."
        )

    warm: dict[str, WarmDecodeStats] = {
        sid: WarmDecodeStats(scenario_id=sid) for sid in WARM_DECODE_FILES
    }
    for session_dir in sessions:
        for sid, fname in WARM_DECODE_FILES.items():
            warm[sid].rows.extend(_load_warm_decode(session_dir, sid, fname))

    attribution: dict[str, list[dict]] = {key: [] for key in ATTRIBUTION_FILES}
    for session_dir in sessions:
        for key, fname in ATTRIBUTION_FILES.items():
            attribution[key].extend(_load_attribution_summaries(session_dir, fname))

    return warm, attribution


def render_report(warm: dict[str, WarmDecodeStats], attribution: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("# P-6 small-B sub-unit α — combined-session report")
    lines.append("")
    lines.append(
        "Auto-generated by `plans/P6_SMALL_B/aggregate_variance.py`. "
        "Do not hand-edit; regenerate after each new session."
    )
    lines.append("")
    lines.append("## Warm-decode baseline (combined across sessions)")
    lines.append("")
    lines.append(
        _format_table_row(
            "Scenario", "Sessions", "n",
            "Aggregate mean", "Aggregate σ",
            "Per-row mean", "Per-row σ",
            "Peak mem (MB)", "α gate (σ ≤ 1.5)",
        )
    )
    lines.append(
        _format_table_row(
            "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
        )
    )
    any_warm = False
    for sid in WARM_DECODE_FILES:
        stats = warm[sid]
        if stats.n == 0:
            lines.append(
                _format_table_row(
                    f"`{sid}`", "0", "0", "—", "—", "—", "—", "—", "no data",
                )
            )
            continue
        any_warm = True
        passes = stats.passes_alpha_gate
        lines.append(
            _format_table_row(
                f"`{sid}`",
                str(len(stats.sessions)),
                str(stats.n),
                _fmt_float(stats.aggregate_mean),
                _fmt_float(stats.aggregate_std),
                _fmt_float(stats.per_row_mean),
                _fmt_float(stats.per_row_std),
                _fmt_float(stats.peak_memory_max_mb, digits=1),
                "PASS" if passes else "FAIL",
            )
        )
    lines.append("")

    if not any_warm:
        lines.append("> _No warm-decode rows captured yet. Run RUNBOOK.md._")
        lines.append("")

    # α gate verdict block
    lines.append("## α gate verdict")
    lines.append("")
    verdicts: list[str] = []
    for sid, stats in warm.items():
        if stats.n == 0:
            verdicts.append(f"- `{sid}`: **no data** — run RUNBOOK.md")
        elif stats.passes_alpha_gate:
            verdicts.append(
                f"- `{sid}`: **PASS** "
                f"(σ = {_fmt_float(stats.aggregate_std)} tok/s ≤ 1.5; "
                f"{len(stats.sessions)} sessions, n={stats.n})"
            )
        else:
            reason: list[str] = []
            if stats.n < 6:
                reason.append(f"n={stats.n} < 6")
            if len(stats.sessions) < 2:
                reason.append(f"sessions={len(stats.sessions)} < 2")
            if (
                not math.isnan(stats.aggregate_std)
                and stats.aggregate_std > ALPHA_SIGMA_GATE
            ):
                reason.append(
                    f"σ = {_fmt_float(stats.aggregate_std)} tok/s > {ALPHA_SIGMA_GATE}"
                )
            verdicts.append(
                f"- `{sid}`: **FAIL** ({', '.join(reason) or 'unknown reason'})"
            )
    lines.extend(verdicts)
    lines.append("")

    # Attribution microbench summaries
    lines.append("## Attribution-microbench summaries (per session)")
    lines.append("")
    any_attribution = False
    for key, summaries in attribution.items():
        if not summaries:
            continue
        any_attribution = True
        lines.append(f"### `{key}`")
        lines.append("")
        keys = sorted({k for s in summaries for k in s.keys() if not k.startswith("_")})
        keys = [k for k in keys if k != "kind"]
        head = _format_table_row("session", *keys)
        sep = _format_table_row("---", *(["---:"] * len(keys)))
        lines.append(head)
        lines.append(sep)
        for s in summaries:
            row_cells = [s.get("_session", "—")]
            for k in keys:
                v = s.get(k)
                if isinstance(v, float):
                    row_cells.append(_fmt_float(v, digits=3))
                elif v is None:
                    row_cells.append("—")
                else:
                    row_cells.append(str(v))
            lines.append(_format_table_row(*row_cells))
        lines.append("")

    if not any_attribution:
        lines.append("> _No attribution-microbench summaries captured yet._")
        lines.append("")

    lines.append("## How α opens β / γ / δ")
    lines.append("")
    lines.append(
        "Per `plans/P6_SMALL_B_OPENING.md` §4, a single attribution session "
        "is enough to read the bucket distribution. Sub-units open as:"
    )
    lines.append("")
    lines.append("- β (attention `mx.compile` + cache reroute) — full-attn ≥ 15% of step time.")
    lines.append("- γ (`mx.compile` on `Qwen3NextMLP`) — MLP-attributable share ≥ 5%.")
    lines.append("- δ (`mx.eval` cadence / per-layer loop sync) — overhead bucket ≥ 3%.")
    lines.append(
        "- **Close the line** if DeltaNet ≥ 95% (no reachable lever; cycle-31 "
        "confirmed mlx `gated_delta` at HBM-bandwidth limit)."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="plans/P6_SMALL_B directory containing per-session subdirs")
    parser.add_argument("--out", required=True, type=Path,
                        help="path to write the Markdown REPORT.md")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root not a directory: {args.root}")

    warm, attribution = collect(args.root)
    report = render_report(warm, attribution)
    args.out.write_text(report)

    summary = {
        "scenarios": {
            sid: {
                "n": stats.n,
                "sessions": stats.sessions,
                "aggregate_mean_tok_s": stats.aggregate_mean if stats.n else None,
                "aggregate_std_tok_s": stats.aggregate_std if stats.n >= 2 else None,
                "per_row_mean_tok_s": stats.per_row_mean if stats.n else None,
                "per_row_std_tok_s": stats.per_row_std if stats.n >= 2 else None,
                "passes_alpha_gate": stats.passes_alpha_gate,
            }
            for sid, stats in warm.items()
        },
        "report_path": str(args.out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
