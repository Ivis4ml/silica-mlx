#!/usr/bin/env python3
"""D-023 B=1 sweep aggregator.

Reads one or more session JSONL files produced by `run_b1_sweep.py`
and emits a markdown report (`REPORT.md` next to the input) plus a
machine-readable summary on stdout.

Three load-bearing questions per the locked spike doc:
1. Does each prompt's **decision row** clear B=1 per-row >= 1.3x?
   (Decision row = block_size with highest measured `generation_tps`.)
2. Are accept_rates on natural-language prompts noticeably lower than
   on code/template prompts?
3. Is there a throughput regression or accept_rate cliff at the
   block_size=9 stress row?

Variance treatment: combined sample sigma across reps (and across
sessions when multiple files are passed). Per cycle-27 protocol the
absolute sigma <= 1.5 tok/s gate applies; the B=1 noise-floor caveat
in spike doc 6.6 also notes the [1.2x, 1.4x] grey-band sigma_ratio
supplement.

Stdlib-only, runs from any python (project or isolated venv).
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(jsonl_path):
    meta = None
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


def combined_stats(values):
    """Return (mean, sample_sigma, n)."""
    n = len(values)
    if n == 0:
        return (None, None, 0)
    mean = sum(values) / n
    if n == 1:
        return (mean, None, 1)
    sigma = statistics.stdev(values)
    return (mean, sigma, n)


def aggregate(jsonl_paths):
    """Aggregate measurements grouped by (prompt_key, mode, block_size)."""
    metas = []
    by_group = defaultdict(list)
    for p in jsonl_paths:
        meta, rows = load_rows(p)
        if meta is not None:
            metas.append(meta)
        for r in rows:
            key = (
                r["prompt_key"],
                r["mode"],
                r.get("draft_block_size"),
            )
            by_group[key].append(r)
    return metas, by_group


def fmt_pct(x):
    return f"{x * 100:.2f}%" if x is not None else "—"


def fmt_num(x, prec=2):
    return f"{x:.{prec}f}" if x is not None else "—"


def fmt_pm(mean, sigma, prec=2):
    if mean is None:
        return "—"
    if sigma is None:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {sigma:.{prec}f}"


def render_report(metas, by_group, block_sizes):
    out = []
    out.append("# D-023 B=1 sweep — aggregated report\n\n")

    out.append("## Toolchain (runtime stack divergence)\n\n")
    if metas:
        m = metas[0]
        tc = m.get("toolchain", {})
        out.append("| Item | Value |\n| --- | --- |\n")
        for k in [
            "venv", "mlx", "mlx-lm", "mlx-metal",
            "mlx-vlm", "transformers",
        ]:
            if k in tc:
                out.append(f"| `{k}` | `{tc[k]}` |\n")
        sp = tc.get("silica_project_pin", {})
        if sp:
            sp_str = " / ".join(f"{k}=={v}" for k, v in sp.items())
            out.append(f"| silica project pin | `{sp_str}` (untouched) |\n")
        if "divergence_statement" in tc:
            out.append(f"\n> {tc['divergence_statement']}\n\n")

    sessions = ", ".join(m.get("timestamp", "?") for m in metas)
    out.append(f"\n**Sessions aggregated:** {sessions}\n\n")

    # Per-prompt off-spec baseline
    prompts = sorted({k[0] for k in by_group.keys()})

    out.append("## Per-prompt off-spec baseline\n\n")
    out.append("| prompt | mean gen_tps | sigma | n | mean prompt_tok | mean gen_tok |\n")
    out.append("| --- | --- | --- | --- | --- | --- |\n")
    off_means = {}
    for p in prompts:
        rows = by_group.get((p, "off", None), [])
        gen_tps = [r["generation_tps"] for r in rows]
        m, s, n = combined_stats(gen_tps)
        off_means[p] = m
        prompt_toks = [r["prompt_tokens"] for r in rows]
        gen_toks = [r["generation_tokens"] for r in rows]
        m_pt = sum(prompt_toks) / len(prompt_toks) if prompt_toks else None
        m_gt = sum(gen_toks) / len(gen_toks) if gen_toks else None
        out.append(
            f"| `{p}` | {fmt_num(m)} | {fmt_num(s)} | {n} | "
            f"{fmt_num(m_pt, 1)} | {fmt_num(m_gt, 1)} |\n"
        )
    out.append("\n")

    # Per-(prompt, block_size) on-spec rows
    out.append("## Per-(prompt, block_size) on-spec measurements\n\n")
    out.append(
        "| prompt | block | k_cand | gen_tps mean ± sigma | n | "
        "speedup vs off | accept_rate mean | mean rounds | mean gen_tok |\n"
    )
    out.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    speedup_table = {}
    accept_table = {}
    for p in prompts:
        for b in block_sizes:
            rows = by_group.get((p, "on", b), [])
            gen_tps = [r["generation_tps"] for r in rows]
            m, s, n = combined_stats(gen_tps)
            accepts = [r["accept_rate"] for r in rows
                       if r.get("accept_rate") is not None]
            ar_mean = (sum(accepts) / len(accepts)) if accepts else None
            rounds = [r["rounds"] for r in rows
                      if r.get("rounds") is not None]
            r_mean = (sum(rounds) / len(rounds)) if rounds else None
            gen_toks = [r["generation_tokens"] for r in rows]
            gt_mean = (sum(gen_toks) / len(gen_toks)) if gen_toks else None
            off_m = off_means.get(p)
            speedup = (m / off_m) if (m is not None and off_m) else None
            speedup_table[(p, b)] = speedup
            accept_table[(p, b)] = ar_mean
            kc = b - 1 if b is not None else None
            out.append(
                f"| `{p}` | {b} | {kc} | {fmt_pm(m, s)} | {n} | "
                f"{fmt_num(speedup, 3) + 'x' if speedup is not None else '—'} | "
                f"{fmt_pct(ar_mean)} | {fmt_num(r_mean, 1)} | "
                f"{fmt_num(gt_mean, 1)} |\n"
            )
    out.append("\n")

    # Decision rows (highest mean gen_tps per prompt)
    out.append("## Decision row per prompt (highest gen_tps)\n\n")
    out.append(
        "| prompt | decision block | k_cand | gen_tps | speedup vs off | "
        "accept_rate | gate (>=1.3x B=1)? |\n"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for p in prompts:
        best_b = None
        best_tps = -1.0
        for b in block_sizes:
            rows = by_group.get((p, "on", b), [])
            tps_vals = [r["generation_tps"] for r in rows]
            if not tps_vals:
                continue
            m = sum(tps_vals) / len(tps_vals)
            if m > best_tps:
                best_tps = m
                best_b = b
        speedup = speedup_table.get((p, best_b))
        ar = accept_table.get((p, best_b))
        gate = "PASS" if (speedup is not None and speedup >= 1.3) else "FAIL"
        out.append(
            f"| `{p}` | {best_b} | {best_b - 1 if best_b else '—'} | "
            f"{fmt_num(best_tps)} | "
            f"{fmt_num(speedup, 3) + 'x' if speedup is not None else '—'} | "
            f"{fmt_pct(ar)} | **{gate}** |\n"
        )
    out.append("\n")

    # Block-size 9 stress check
    out.append("## block_size=9 stress check\n\n")
    out.append(
        "Comparison: best on-spec (decision row) vs block_size=9 to detect "
        "any throughput regression or accept-rate cliff at long drafts.\n\n"
    )
    out.append(
        "| prompt | best_block | best_tps | block=9 tps | block=9 speedup vs off | "
        "block=9 accept_rate | regression vs decision row? |\n"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for p in prompts:
        best_b = None
        best_tps = -1.0
        for b in block_sizes:
            rows = by_group.get((p, "on", b), [])
            tps_vals = [r["generation_tps"] for r in rows]
            if not tps_vals:
                continue
            m = sum(tps_vals) / len(tps_vals)
            if m > best_tps:
                best_tps = m
                best_b = b
        rows9 = by_group.get((p, "on", 9), [])
        tps9 = [r["generation_tps"] for r in rows9]
        m9 = (sum(tps9) / len(tps9)) if tps9 else None
        sp9 = speedup_table.get((p, 9))
        ar9 = accept_table.get((p, 9))
        regression = (m9 < best_tps) if (m9 is not None) else None
        regression_str = (
            f"YES (-{(1 - m9/best_tps) * 100:.1f}%)"
            if regression is True else "NO"
        ) if regression is not None else "—"
        out.append(
            f"| `{p}` | {best_b} | {fmt_num(best_tps)} | {fmt_num(m9)} | "
            f"{fmt_num(sp9, 3) + 'x' if sp9 is not None else '—'} | "
            f"{fmt_pct(ar9)} | {regression_str} |\n"
        )
    out.append("\n")

    # Accept-rate by prompt category
    out.append("## Accept-rate by prompt type\n\n")
    out.append(
        "Heuristic categorization: `factorial`, `bst` are code/template; "
        "`creative_scene`, `factual_explain` are natural-language. Lower "
        "accept-rate on natural prompts would suggest the high template "
        "rate is template-specific rather than fundamental to the pairing.\n\n"
    )
    code_prompts = {"factorial", "bst"}
    natural_prompts = {"creative_scene", "factual_explain"}
    out.append("| block | code mean | natural mean | gap |\n")
    out.append("| --- | --- | --- | --- |\n")
    for b in block_sizes:
        code_rates = [accept_table.get((p, b)) for p in code_prompts
                      if accept_table.get((p, b)) is not None]
        nat_rates = [accept_table.get((p, b)) for p in natural_prompts
                     if accept_table.get((p, b)) is not None]
        cm = (sum(code_rates) / len(code_rates)) if code_rates else None
        nm = (sum(nat_rates) / len(nat_rates)) if nat_rates else None
        gap = (cm - nm) if (cm is not None and nm is not None) else None
        out.append(
            f"| {b} | {fmt_pct(cm)} | {fmt_pct(nm)} | "
            f"{fmt_pct(gap) if gap is not None else '—'} |\n"
        )
    out.append("\n")

    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", help="One or more b1_sweep.jsonl files")
    parser.add_argument("--out", type=str, default=None,
                        help="Output REPORT.md path (default: alongside the first input)")
    parser.add_argument("--block-sizes", type=str, default="2,3,6,9")
    args = parser.parse_args()

    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    paths = [Path(p) for p in args.jsonl]
    metas, by_group = aggregate(paths)
    report = render_report(metas, by_group, block_sizes)

    out_path = (Path(args.out) if args.out
                else paths[0].parent / "REPORT.md")
    out_path.write_text(report)
    print(f"Report: {out_path}\n")
    # Echo a compact json summary on stdout for piping into a ledger
    summary = {
        "sessions": [m.get("timestamp") for m in metas],
        "n_prompts": len({k[0] for k in by_group.keys()}),
        "n_measurements": sum(len(v) for v in by_group.values()),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
