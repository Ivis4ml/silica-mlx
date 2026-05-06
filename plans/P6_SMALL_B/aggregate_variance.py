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

    # Decode-step attribution (B=4) — per-session step-share table
    lines.append("## Decode-step attribution (B=4) — per-session bucket distribution")
    lines.append("")
    decode_step_summaries = attribution.get("decode_step_attr_b4", [])
    if decode_step_summaries:
        lines.append(
            _format_table_row(
                "session", "step_total median (ms)",
                "DeltaNet (linear) %", "Full-attn %", "Overhead %",
                "DeltaNet (ms)", "Full-attn (ms)",
            )
        )
        lines.append(
            _format_table_row(
                "---", "---:", "---:", "---:", "---:", "---:", "---:",
            )
        )
        for s in decode_step_summaries:
            step = s.get("step_total_ms") or {}
            linear = s.get("linear_layers_total_ms") or {}
            full = s.get("full_layers_total_ms") or {}
            lines.append(
                _format_table_row(
                    s.get("_session", "—"),
                    _fmt_float(step.get("median_ms", float("nan")), digits=2),
                    _fmt_float((s.get("linear_pct_of_step") or 0) * 100, digits=1),
                    _fmt_float((s.get("full_pct_of_step") or 0) * 100, digits=1),
                    _fmt_float((s.get("overhead_pct_of_step") or 0) * 100, digits=1),
                    _fmt_float(linear.get("median_ms", float("nan")), digits=2),
                    _fmt_float(full.get("median_ms", float("nan")), digits=2),
                )
            )
        lines.append("")
    else:
        lines.append("> _No decode-step attribution summaries captured yet._")
        lines.append("")

    # Layer-internal attribution (B=4) — per-component median + pct
    lines.append("## Layer-internal attribution (B=4) — per-component step-share")
    lines.append("")
    layer_internal_summaries = attribution.get("layer_internal_attr_b4", [])
    if layer_internal_summaries:
        # Discover all components across sessions
        all_components: set[str] = set()
        for s in layer_internal_summaries:
            comps = s.get("components") or {}
            all_components.update(comps.keys())
        ordered_components = sorted(all_components)

        sessions_in_order = [s.get("_session", "—") for s in layer_internal_summaries]
        head_cells = ["component"] + [f"{ses} %" for ses in sessions_in_order] + ["mean %"]
        sep_cells = ["---"] + (["---:"] * (len(sessions_in_order) + 1))
        lines.append(_format_table_row(*head_cells))
        lines.append(_format_table_row(*sep_cells))

        for comp in ordered_components:
            row = [f"`{comp}`"]
            pct_values: list[float] = []
            for s in layer_internal_summaries:
                comp_dict = (s.get("components") or {}).get(comp) or {}
                pct = comp_dict.get("pct_of_step")
                if pct is None:
                    row.append("—")
                else:
                    pct_pct = pct * 100
                    pct_values.append(pct_pct)
                    row.append(_fmt_float(pct_pct, digits=2))
            mean_pct = sum(pct_values) / len(pct_values) if pct_values else float("nan")
            row.append(_fmt_float(mean_pct, digits=2))
            lines.append(_format_table_row(*row))
        lines.append("")
    else:
        lines.append("> _No layer-internal attribution summaries captured yet._")
        lines.append("")

    # Sub-unit β / γ / δ gate verdict (driven by attribution data)
    lines.append("## Sub-unit β / γ / δ gate verdict")
    lines.append("")
    lines.append(
        "Per `plans/P6_SMALL_B_OPENING.md` §4 open-conditions. Bucket "
        "percentages are mean across sessions where multiple are present."
    )
    lines.append("")

    if decode_step_summaries:
        full_pcts = [(s.get("full_pct_of_step") or 0) * 100 for s in decode_step_summaries]
        linear_pcts = [(s.get("linear_pct_of_step") or 0) * 100 for s in decode_step_summaries]
        overhead_pcts = [(s.get("overhead_pct_of_step") or 0) * 100 for s in decode_step_summaries]

        full_mean = sum(full_pcts) / len(full_pcts)
        linear_mean = sum(linear_pcts) / len(linear_pcts)
        overhead_mean = sum(overhead_pcts) / len(overhead_pcts)

        beta_open = full_mean >= 15.0
        delta_open = overhead_mean >= 3.0
        line_close = linear_mean >= 95.0

        # γ: MLP attribution from layer-internal summaries (sum of linear.mlp + full.mlp)
        mlp_pcts: list[float] = []
        for s in layer_internal_summaries:
            comps = s.get("components") or {}
            lin_mlp = (comps.get("linear.mlp") or {}).get("pct_of_step", 0) or 0
            full_mlp = (comps.get("full.mlp") or {}).get("pct_of_step", 0) or 0
            mlp_pcts.append((lin_mlp + full_mlp) * 100)
        mlp_mean = sum(mlp_pcts) / len(mlp_pcts) if mlp_pcts else float("nan")
        gamma_open = (not math.isnan(mlp_mean)) and mlp_mean >= 5.0

        verdict_status = "OPEN" if not line_close else "CLOSE"

        def _gate(name: str, value: float, threshold: float, op: str) -> str:
            return f"{value:.1f}% {op} {threshold:.0f}%"

        lines.append(
            f"- **β (attention `mx.compile` + cache reroute)**: "
            f"full-attn = {_gate('full', full_mean, 15.0, '≥' if beta_open else '<')} "
            f"→ {'**OPEN**' if beta_open else 'closed'}"
        )
        lines.append(
            f"- **γ (`mx.compile` on `Qwen3NextMLP`)**: "
            f"MLP attribution = {_gate('mlp', mlp_mean, 5.0, '≥' if gamma_open else '<')} "
            f"→ {'**OPEN**' if gamma_open else 'closed'}"
        )
        lines.append(
            f"- **δ (`mx.eval` cadence / per-layer loop sync)**: "
            f"overhead = {_gate('overhead', overhead_mean, 3.0, '≥' if delta_open else '<')} "
            f"→ {'**OPEN**' if delta_open else 'closed'}"
        )
        lines.append(
            f"- **Line-close check**: DeltaNet = "
            f"{linear_mean:.1f}% {'≥' if line_close else '<'} 95% "
            f"→ {'**CLOSE the small-B line** (no reachable lever)' if line_close else 'continue'}"
        )
        lines.append("")
        lines.append(f"**Overall α verdict: {verdict_status}**")
        lines.append("")
        if not line_close:
            opens = [name for name, ok in [("β", beta_open), ("γ", gamma_open), ("δ", delta_open)] if ok]
            if opens:
                lines.append(
                    f"Sub-units {' / '.join(opens)} unlocked. "
                    f"ε remains waitlist (mlx 0.32+ async-copy)."
                )
            else:
                lines.append(
                    "No sub-unit threshold met. Investigate before opening any of β / γ / δ."
                )
            lines.append("")
    else:
        lines.append(
            "> _No attribution data captured. Sub-unit decisions blocked until "
            "decode_step_attr_b4 and layer_internal_attr_b4 are run._"
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
