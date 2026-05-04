"""Render the comprehensive 11-cycle autoresearch summary chart.

Multi-panel figure showing the journey from cycle 1 (42.17 tok/s baseline)
through cycle 10 (193.9 tok/s breakthrough at B=48) and cycle 11 (FA-decode
kernel-level beat over mlx; E2E flat):

    Panel A: Running-best trajectory across 11 cycles + milestones
    Panel B: B-sweep curve (aggregate tok/s vs B) with 36 GB envelope
    Panel C: FA-decode kernel ablation across T_kv (silica vs mlx, cycle 11)
    Panel D: QMM kernel tuning bar chart (cycle 7-9 foundation)

Usage:
    uv run --extra bench python scripts/render_summary_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUTDIR = Path(__file__).resolve().parent.parent / "plans"


# Cycle-by-cycle running best (decode_tok_s on warm-decode-* row family)
CYCLES = list(range(1, 24))
RUNNING_BEST = [42.17] * 9 + [193.9, 193.3, 192.5, 200.8, 206.2, 206.2, 206.2, 206.2, 206.2, 206.2, 206.2, 206.2, 206.2, 206.2]
CYCLE_LABELS = [
    "C1\norient",
    "C2\nsimple\nkernels",
    "C3\nQMM naive",
    "C4\nlazy chain",
    "C5\nbatcher",
    "C6\nsimdgroup",
    "C7\nv3-v6",
    "C8\nv7-v11",
    "C9\nv12-v13",
    "C10\nB=48",
    "C11\nFA-decode",
    "C12\nbf16 probe",
    "C13\nB=52 bf16",
    "C14 ⭐\nstack",
    "C15\nconfirm",
    "C16\nconfirm",
    "C17\nconfirm",
    "C18\nconfirm",
    "C19\ncov@64",
    "C20\nverify",
    "C21\ndrafter",
    "C22\ndesign",
    "C23\nspec end",
]

# B-sweep through Engine.generate_batch (warm-decode oracle results).
# Cycle 10 fp32 state up through B=48; cycle 13 bf16 state from B=52 onward.
# Cycle 14 adds B=53/66/68 probes and bumps B=52/64 to the v10+bf16 stack tunes.
B_SWEEP = [4, 8, 12, 16, 24, 32, 40, 44, 48, 52, 56, 60, 64, 66, 72]
B_SWEEP_AGG = [42.17, 43.0, 63.1, 81.1, 112.8, 150.6, 171.6, 183.3, 193.9,
               206.2, 212.2, 219.1, 232.2, 166.8, 173.2]
B_SWEEP_PEAK_GB = [17.1, 18.7, 20.2, 21.7, 24.8, 27.9, 30.8, 32.4, 33.95,
                   35.52, 36.90, 38.45, 40.01, 40.79, 43.36]
B_SWEEP_BF16 = [False] * 9 + [True] * 6

# QMM kernel iterations
KERNELS = [
    ("naive", 0.66, "discard", "#e57373"),
    ("v2", 0.97, "discard", "#ef9a9a"),
    ("v3 hoist s/b", 0.89, "diag", "#90caf9"),
    ("v4 +4N/sg", 0.83, "diag", "#64b5f6"),
    ("v5 group dq", 1.21, "discard", "#e57373"),
    ("v6 uint4-view", 0.83, "diag", "#64b5f6"),
    ("v7 2sg/tg", 0.89, "discard", "#ef9a9a"),
    ("v8 8N/sg", 1.24, "discard", "#e57373"),
    ("v9 ⭐", 0.59, "best", "#1976d2"),
    ("v10 2KT/barr", 0.61, "discard", "#90caf9"),
    ("v11 sg-barr", 0.65, "discard", "#90caf9"),
    ("v12 uint32$", 0.73, "discard", "#ef9a9a"),
    ("v13 uint4 vec", 0.78, "discard", "#ef9a9a"),
]
MLX_REF_MS = 0.45

# Cumulative counts per cycle
NEW_KERNELS_PER_CYCLE = [0, 3, 1, 0, 0, 1, 4, 5, 2, 0, 8, 0]
NEW_SCENARIOS_PER_CYCLE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0]
TESTS_PER_CYCLE = [2640] * 5 + [2647, 2673, 2673, 2673, 2681, 2770, 2770]

# Cycle 11 FA-decode kernel data — silica vs mlx at production B=48 (5-run median).
FA_T_KV = [128, 256, 512, 1024]
FA_SILICA_P50 = [0.28, 0.44, 0.71, 1.07]   # ms
FA_MLX_P50 = [0.51, 0.70, 0.89, 1.35]

# Cycles 10-14 KEEP ladder — the running-best progression with each compositional add.
E2E_LABELS = ["C10 baseline\nB=48 fp32\n(n=3)",
              "C13 B=52\nbf16 only\n(n=3)",
              "C14 ⭐ B=52\nv10+bf16 stack\n(n=3) ENVELOPE",
              "C13 B=64\nbf16 only\n(n=3)",
              "C14 ⭐⭐ B=64\nv10+bf16 stack\n(n=3) HARDWARE"]
E2E_MEANS = [193.9, 200.8, 206.2, 229.8, 232.2]
E2E_ERRORS = [0.6, 1.5, 0.5, 2.0, 0.3]


def render() -> None:
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.22, height_ratios=[1.2, 1, 1])

    # ============================================================
    # Panel A: Running-best trajectory
    # ============================================================
    ax_a = fig.add_subplot(gs[0, :])
    xs = list(range(len(CYCLES)))
    ax_a.plot(xs, RUNNING_BEST, "o-", color="#2ca02c", linewidth=3,
              markersize=12, zorder=5, label="running best")
    ax_a.fill_between(xs, RUNNING_BEST, alpha=0.15, color="#2ca02c", zorder=1)

    # Reference horizontal lines
    ax_a.axhline(60, color="#ff7f0e", linestyle="--", linewidth=2, alpha=0.7,
                 label="(1b) milestone (60 tok/s)")
    ax_a.axhline(67, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.7,
                 label="not-the-limit projection (67 tok/s)")
    ax_a.axhline(81.16, color="#444", linestyle=":", linewidth=1.2, alpha=0.7,
                 label="B=4 weights ceiling (81.16 tok/s)")

    # Annotate values
    for i, (x, v, lbl) in enumerate(zip(xs, RUNNING_BEST, CYCLE_LABELS)):
        if i < 9:
            ax_a.annotate(f"{v:.2f}", (x, v), xytext=(0, -22),
                          textcoords="offset points", ha="center",
                          fontsize=9, color="#2ca02c", fontweight="bold")
        elif i == 9:
            ax_a.annotate(f"{v:.2f}", (x, v), xytext=(0, 16),
                          textcoords="offset points", ha="center",
                          fontsize=12, color="#2ca02c", fontweight="bold")
        elif i in (10, 11):
            ax_a.annotate(f"{v:.2f}", (x, v), xytext=(0, 16),
                          textcoords="offset points", ha="center",
                          fontsize=9, color="#888", fontweight="bold")
        else:
            ax_a.annotate(f"{v:.2f}", (x, v), xytext=(0, 16),
                          textcoords="offset points", ha="center",
                          fontsize=14, color="#2ca02c", fontweight="bold")

    # Breakthrough arrow
    arrow = FancyArrowPatch((8, 42.17), (9, 193.9),
                            arrowstyle="->,head_width=0.4,head_length=0.6",
                            color="#d4a017", linewidth=3, mutation_scale=15)
    ax_a.add_patch(arrow)
    ax_a.text(8.3, 110, "BREAKTHROUGH\n4.60× baseline\nAR.md axis-shift",
              fontsize=10, color="#d4a017", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#d4a017"))

    # C11/C12 marker: kernel/state wins didn't move E2E at fixed B=48
    ax_a.annotate("C11/C12: kernel wins\nat fixed B=48",
                  xy=(11, 192.5), xytext=(9.5, 75),
                  fontsize=8, color="#666", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#666", lw=1),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa", edgecolor="#666"))
    # C13 second-breakthrough annotation
    ax_a.annotate("C13 SECOND BREAKTHROUGH\nbf16 state's peak save (3.5 GB)\nunlocks B-axis past cycle-10 cap\n→ B=52 = 200.8 strict-envelope KEEP\n→ B=64 = 229.8 hardware ceiling",
                  xy=(12, 200.8), xytext=(11.0, 130),
                  fontsize=10, color="#d4a017", ha="center", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#d4a017"))
    # Add the demonstrated ceiling line
    ax_a.axhline(229.8, color="#1976d2", linestyle="--", linewidth=1.5, alpha=0.5)
    ax_a.text(0.05, 232, "C13 demonstrated ceiling B=64 = 229.8 (within 48 GB hardware)",
              fontsize=9, color="#1976d2", fontweight="bold")

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(CYCLE_LABELS, fontsize=9)
    ax_a.set_ylabel("decode_tok_s (B chosen to maximise aggregate)", fontsize=11)
    ax_a.set_title("Silica-MLX P-6 Autoresearch: 23-cycle running-best trajectory on dense Qwen3.5-27B-4bit\n"
                   "C10 axis-shift (42.17 → 193.9); C13 envelope-extension (200.8); C14 v10+bf16 stack (206.2 envelope, 232.2 hardware); C15-18 kernel local optimum; C19-23 spec-decode research thread closes",
                   fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=10)
    ax_a.grid(True, alpha=0.3, linestyle="--")
    ax_a.set_ylim(0, 250)

    # ============================================================
    # Panel B: B-sweep curve (cycle 10 fp32 + cycle 13 bf16-state)
    # ============================================================
    ax_b = fig.add_subplot(gs[1, 0])
    bx = B_SWEEP

    # Split into fp32 vs bf16 phase markers
    fp32_x = [b for b, is_bf16 in zip(bx, B_SWEEP_BF16) if not is_bf16]
    fp32_y = [v for v, is_bf16 in zip(B_SWEEP_AGG, B_SWEEP_BF16) if not is_bf16]
    bf16_x = [b for b, is_bf16 in zip(bx, B_SWEEP_BF16) if is_bf16]
    bf16_y = [v for v, is_bf16 in zip(B_SWEEP_AGG, B_SWEEP_BF16) if is_bf16]

    ax_b.plot(fp32_x, fp32_y, "o-", color="#1976d2", linewidth=2.5,
              markersize=10, label="C10 fp32 state (B=4..48)")
    ax_b.plot(bf16_x, bf16_y, "o-", color="#2ca02c", linewidth=2.5,
              markersize=10, label="C13 bf16 state (B=52..72)")
    for x, v in zip(bx, B_SWEEP_AGG):
        ax_b.annotate(f"{v:.1f}", (x, v), xytext=(0, 10),
                      textcoords="offset points", ha="center",
                      fontsize=8)

    # Mark the C14 stack KEEPs
    ax_b.scatter([52], [206.2], s=200, marker="*", color="#2ca02c",
                 edgecolor="black", zorder=5, label="C14 envelope KEEP (v10+bf16)")
    ax_b.scatter([64], [232.2], s=200, marker="*", color="#d4a017",
                 edgecolor="black", zorder=5, label="C14 hardware ceiling (v10+bf16)")
    ax_b.annotate("CLIFF at 40 GB peak\nB=64 → B=66\n229.8 → 166.8 (-26%)",
                  xy=(66, 166.8), xytext=(58, 120), fontsize=8, color="#888",
                  arrowprops=dict(arrowstyle="->", color="#888"),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa", edgecolor="#888"))

    # Reference lines
    ax_b.axhline(60, color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.6)
    ax_b.text(72, 62, "(1b) milestone 60", fontsize=8, color="#ff7f0e", ha="right")

    # Memory envelope on second axis
    ax_b2 = ax_b.twinx()
    bar_colors = ["#d62728" if not is_bf16 else "#2ca02c" for is_bf16 in B_SWEEP_BF16]
    ax_b2.bar(bx, B_SWEEP_PEAK_GB, alpha=0.20, color=bar_colors,
              width=2.5, zorder=1)
    ax_b2.axhline(36, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.6)
    ax_b2.text(4, 36.5, "36 GB AR.md envelope", fontsize=8, color="#d62728")
    ax_b2.axhline(48, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.4)
    ax_b2.text(4, 48.5, "48 GB hardware ceiling", fontsize=8, color="#d62728")
    ax_b2.set_ylabel("peak memory (GB)", fontsize=10, color="#d62728")
    ax_b2.set_ylim(0, 52)
    ax_b2.tick_params(axis="y", labelcolor="#d62728")

    ax_b.set_xlabel("B (max_batch_size)")
    ax_b.set_ylabel("aggregate decode_tok_s", fontsize=10, color="#1976d2")
    ax_b.tick_params(axis="y", labelcolor="#1976d2")
    ax_b.set_title("Panel B: B-sweep — C10 fp32 (B≤48) + C13/C14 bf16+v10 stack (B=52..64)\n"
                   "Aggregate climbs 42.17 → 232.2 (5.51×); sharp cliff past 40 GB peak",
                   fontsize=10, fontweight="bold")
    ax_b.grid(True, alpha=0.3, linestyle="--")
    ax_b.set_xticks(bx)
    ax_b.legend(loc="upper left", fontsize=8)
    ax_b.set_ylim(0, 260)

    # ============================================================
    # Panel C: cycle 11 FA-decode kernel ablation (silica vs mlx across T_kv)
    # ============================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(FA_T_KV, FA_SILICA_P50, "o-", color="#1976d2", linewidth=2.5,
              markersize=10, label="silica v8/v10 (cycle 11)")
    ax_c.plot(FA_T_KV, FA_MLX_P50, "s-", color="#2ca02c", linewidth=2.5,
              markersize=10, label="mlx mx.fast.scaled_dot_product_attention")
    for tk, sp, mp in zip(FA_T_KV, FA_SILICA_P50, FA_MLX_P50):
        r = sp / mp
        ax_c.annotate(f"{sp:.2f}", (tk, sp), xytext=(0, -15),
                      textcoords="offset points", ha="center",
                      fontsize=8, color="#1976d2")
        ax_c.annotate(f"{mp:.2f}", (tk, mp), xytext=(0, 8),
                      textcoords="offset points", ha="center",
                      fontsize=8, color="#2ca02c")
        ax_c.annotate(f"{r:.2f}×", (tk, (sp + mp) / 2),
                      xytext=(15, 0), textcoords="offset points",
                      fontsize=8, color="#666", fontweight="bold")
    ax_c.set_xlabel("T_kv")
    ax_c.set_ylabel("p50 latency at B=48 (ms, lower better)", fontsize=10)
    ax_c.set_title("Panel C: cycle 11 FA-decode kernel — silica beats mlx 1.25-1.81×\n"
                   "(kernel-level victory, no E2E delta because attn is 22% of warm-decode-b48 step)",
                   fontsize=10, fontweight="bold")
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(FA_T_KV)
    ax_c.set_xticklabels([str(t) for t in FA_T_KV])
    ax_c.legend(loc="upper left", fontsize=9)
    ax_c.grid(True, alpha=0.3, linestyle="--")

    # ============================================================
    # Panel D: C10 → C13 → C14 KEEP ladder on both envelopes
    # ============================================================
    ax_d = fig.add_subplot(gs[2, :])
    xs = list(range(len(E2E_LABELS)))
    colors = ["#1976d2", "#9467bd", "#2ca02c", "#bcbd22", "#d4a017"]
    bars = ax_d.bar(xs, E2E_MEANS, yerr=E2E_ERRORS, color=colors,
                    edgecolor="black", linewidth=0.5, capsize=8, zorder=3)
    for x, v, e in zip(xs, E2E_MEANS, E2E_ERRORS):
        ax_d.annotate(f"{v:.1f} ± {e:.1f}", (x, v), xytext=(0, 12),
                      textcoords="offset points", ha="center",
                      fontsize=10, fontweight="bold")
    ax_d.axhline(193.9, color="#1976d2", linestyle="--", linewidth=1.5, alpha=0.4,
                 label="C10 baseline = 193.9")
    ax_d.axhline(206.2, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="C14 envelope KEEP = 206.2")
    ax_d.axhline(232.2, color="#d4a017", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="C14 hardware ceiling = 232.2")
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(E2E_LABELS, fontsize=10)
    ax_d.set_ylabel("decode_tok_s on warm-decode-bN", fontsize=11)
    ax_d.set_title("Panel D: KEEP ladder — C10 baseline → C13 bf16-only → C14 v10+bf16 stack\n"
                   "C14 envelope: 206.2 (4.89× C1, 3.4σ over C13). C14 ceiling: 232.2 (5.51× C1).",
                   fontsize=11, fontweight="bold")
    ax_d.set_ylim(180, 245)
    ax_d.legend(loc="upper left", fontsize=9)
    ax_d.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.suptitle(
        "Silica-MLX P-6 Autoresearch — Final Summary (cycles 1-23, 2026-05-02 → 2026-05-04)\n"
        "Mission: push dense Qwen3.5-27B-4bit on M5 Pro 48 GB toward the hardware limit.\n"
        "Result: 42.17 → 206.2 envelope KEEP (4.89×) / 232.2 hardware ceiling (5.51×); C19-23 spec-decode arm: cov@64=40.5% feasible but B×k verify-cost wall blocks net throughput; 21 kernels; 2779 tests pass.",
        fontsize=12, fontweight="bold", y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUTDIR / "P6_AUTORESEARCH_SUMMARY.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
