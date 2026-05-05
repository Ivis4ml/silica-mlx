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
CYCLES = list(range(1, 36))
# CYCLE 27 CORRECTION: cycle-14's claimed 206.2 was attribution error (v10
# wasn't firing due to dtype defect). Honest running-best is bf16-only at
# B=52 ≈ 204.5 ± ~1.5 tok/s, attributed to C10 axis-shift × C12 bf16 peak save.
# Cycles 32-35 are post-23-cycle continuation (dense unchanged; MoE
# secondary track unlocked).
RUNNING_BEST = [42.17] * 9 + [
    193.9,                              # C10
    193.3, 192.5,                       # C11, C12
    200.8,                              # C13 (bf16 peak save unlocks B=52)
    204.5,                              # C14 (corrected from 206.2)
    204.5, 204.5, 204.5, 204.5,         # C15-18 confirmation
    204.5, 204.5, 204.5, 204.5, 204.5,  # C19-23 spec-decode arm
    204.5,                              # C24 dep pin
    204.5, 204.5,                       # C25, C26
    204.5,                              # C27 correction landed
    204.5,                              # C28 ceiling 231.9 re-measured
    204.5, 204.5, 204.5,                # C29 cliff / C30 88% / C31 parity
    204.5, 204.5,                       # C32 chart re-render / C33 variance
    204.5, 204.5,                       # C34 MoE / C35 MoE B=128 (dense unchanged)
]
CYCLE_LABELS = [
    "C1\norient", "C2\nsimple\nkernels", "C3\nQMM naive", "C4\nlazy chain",
    "C5\nbatcher", "C6\nsimdgroup", "C7\nv3-v6", "C8\nv7-v11",
    "C9\nv12-v13", "C10 ⭐\nB=48", "C11\nFA-decode", "C12\nbf16 probe",
    "C13\nB=52 bf16", "C14\n(retracted)", "C15\nconfirm", "C16\nconfirm",
    "C17\nconfirm", "C18\nconfirm", "C19\ncov@64", "C20\nverify",
    "C21\ndrafter", "C22\ndesign", "C23\nspec end", "C24\ndep pin",
    "C25\nreverify", "C26\nbf16 FA fix", "C27 ⭐\ncorrection",
    "C28\nceiling 232", "C29\ncliff arch", "C30\ndeltanet 88%",
    "C31\ndeltanet parity", "C32\ncharts", "C33\nvariance",
    "C34 MoE\nportability", "C35 MoE\nB=128 ⭐⭐",
]

# MoE 35B-A3B secondary-track running-best (cycles 1, 34, 35 are the milestones)
MOE_CYCLES = list(range(1, 36))
# Baseline 188.5 from cycle 1; first MoE secondary KEEP at cycle 34 (464.1
# at B=64); B=128 hardware-ceiling unlock at cycle 35 (791.8).
MOE_RUNNING_BEST = (
    [188.5] * 33                       # C1-C33 stay at MoE B=4 baseline
    + [464.1]                          # C34 portability test KEEP at B=64
    + [791.8]                          # C35 B=128 expert-amortisation unlock
)

# B-sweep through Engine.generate_batch (warm-decode oracle results).
# Cycle 10 fp32 state up through B=48; cycle 13 bf16 state from B=52 onward.
# CYCLE 27/28 CORRECTION: B=52 / B=64 numbers revised — v10 was not firing
# due to dtype defect; running-best is bf16-state alone, not v10+bf16 stack.
B_SWEEP = [4, 8, 12, 16, 24, 32, 40, 44, 48, 52, 56, 60, 64, 66, 72]
B_SWEEP_AGG = [42.17, 43.0, 63.1, 81.1, 112.8, 150.6, 171.6, 183.3, 193.9,
               204.5, 212.2, 219.1, 231.9, 166.8, 173.2]
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

# MoE 35B-A3B B-sweep at bf16 DeltaNet state (cycle 34 + cycle 35 measurements)
MOE_B_SWEEP = [4, 8, 16, 32, 48, 64, 72, 80, 96, 128]
MOE_B_SWEEP_AGG = [188.5, 242.0, 306.3, 384.8, 434.3, 464.1, 444.5, 447.9, 467.1, 791.8]
MOE_B_SWEEP_PEAK_GB = [20.0, 21.7, 23.4, 26.8, 30.3, 33.8, 35.5, 37.3, 40.8, 47.96]

# Cycles 10-14 KEEP ladder — the running-best progression with each compositional add.
# Cycle 27/28 corrected: v10 was not firing due to dtype bug; bf16-only
# at B=52/64 is the honest running-best.
E2E_LABELS = ["C10 baseline\nB=48 fp32\n(n=3)",
              "C13 B=52\nbf16 only\n(n=3)",
              "C27 ⭐ B=52\nbf16 only\n(n=8) ENVELOPE",
              "C28 B=64 v10\n(n=3) regression",
              "C28 ⭐⭐ B=64\nbf16 only (n=3)\nHARDWARE"]
E2E_MEANS = [193.9, 200.8, 204.5, 230.2, 231.9]
E2E_ERRORS = [0.6, 1.5, 1.5, 1.6, 0.3]


def render() -> None:
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.22, height_ratios=[1.2, 1, 1])

    # ============================================================
    # Panel A: Running-best trajectory
    # ============================================================
    ax_a = fig.add_subplot(gs[0, :])
    xs = list(range(len(CYCLES)))
    ax_a.plot(xs, RUNNING_BEST, "o-", color="#2ca02c", linewidth=3,
              markersize=12, zorder=5, label="dense 27B running best (primary)")
    ax_a.fill_between(xs, RUNNING_BEST, alpha=0.15, color="#2ca02c", zorder=1)

    # MoE 35B-A3B secondary-track running-best (cycles 1, 34, 35 mark the steps)
    ax_a.plot(xs, MOE_RUNNING_BEST, "s-", color="#d4a017", linewidth=2.5,
              markersize=8, zorder=4, alpha=0.85,
              label="MoE 35B-A3B running best (secondary)")
    # Annotate the MoE step values
    ax_a.annotate(f"{188.5:.1f}\nC1 MoE B=4", (0, 188.5), xytext=(8, 8),
                  textcoords="offset points", ha="left",
                  fontsize=8, color="#d4a017")
    ax_a.annotate(f"{464.1:.1f}\nC34 MoE B=64\n(secondary KEEP)",
                  (33, 464.1), xytext=(-110, -36),
                  textcoords="offset points", ha="left",
                  fontsize=8, color="#d4a017", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1))
    ax_a.annotate(f"{791.8:.1f}\nC35 MoE B=128\n(48 GB hardware ceiling)",
                  (34, 791.8), xytext=(-180, 6),
                  textcoords="offset points", ha="left",
                  fontsize=9, color="#d4a017", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="#d4a017"))

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
    ax_a.annotate("C13 SECOND BREAKTHROUGH\nbf16 state's peak save (3.5 GB)\nunlocks B-axis past C10 cap\n→ B=52 = 200.8 envelope (corrected)\n→ B=64 = 231.9 ± 0.3 ceiling",
                  xy=(12, 200.8), xytext=(11.0, 130),
                  fontsize=9, color="#d4a017", ha="center", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#d4a017"))
    # C27 correction annotation
    ax_a.annotate("C27 ⭐ CORRECTION\nv10 dtype bug found by Codex\n— v10 was not firing in C12-C26\nrunning-best: 204.5 ± ~1.5\n(C14 phantom 206.2 retracted)",
                  xy=(26, 204.5), xytext=(22, 60),
                  fontsize=9, color="#9467bd", ha="center", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e5f5", edgecolor="#9467bd"))
    # Updated demonstrated ceiling line (corrected)
    ax_a.axhline(231.9, color="#1976d2", linestyle="--", linewidth=1.5, alpha=0.5)
    ax_a.text(0.05, 234, "C28 demonstrated ceiling B=64 bf16-only = 231.9 ± 0.3 (corrected from C14's phantom 232.2)",
              fontsize=9, color="#1976d2", fontweight="bold")

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(CYCLE_LABELS, fontsize=8)
    ax_a.set_ylabel("decode_tok_s (B chosen to maximise aggregate)", fontsize=11)
    ax_a.set_title("Silica-MLX P-6 Autoresearch: 35-cycle running-best trajectory\n"
                   "Dense 27B (primary): 42.17 → 204.5 envelope / 231.9 hardware (5.50×). "
                   "MoE 35B-A3B (secondary): 188.5 → 464.1 envelope / 791.8 hardware (4.20×; cycles 34-35).",
                   fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=10)
    ax_a.grid(True, alpha=0.3, linestyle="--")
    ax_a.set_ylim(0, 850)

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
    ax_b.scatter([52], [204.5], s=200, marker="*", color="#2ca02c",
                 edgecolor="black", zorder=5, label="C27 envelope KEEP (bf16-only, corrected)")
    ax_b.scatter([64], [231.9], s=200, marker="*", color="#d4a017",
                 edgecolor="black", zorder=5, label="C28 hardware ceiling (bf16-only, corrected)")
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
    ax_b.set_title("Panel B: B-sweep — C10 fp32 (B≤48) + C13 bf16-state (B=52..64) corrected\n"
                   "Aggregate climbs 42.17 → 231.9 (5.50×); cliff past 40 GB peak (architectural per C29)",
                   fontsize=10, fontweight="bold")
    ax_b.grid(True, alpha=0.3, linestyle="--")
    ax_b.set_xticks(bx)
    ax_b.legend(loc="upper left", fontsize=8)
    ax_b.set_ylim(0, 260)

    # ============================================================
    # Panel C: MoE 35B-A3B B-sweep at bf16 DeltaNet state (cycles 34-35)
    # ============================================================
    ax_c = fig.add_subplot(gs[1, 1])
    moe_x = MOE_B_SWEEP
    ax_c.plot(moe_x, MOE_B_SWEEP_AGG, "s-", color="#d4a017", linewidth=2.5,
              markersize=10, label="MoE bf16 state (cycles 34-35)")
    for x, v in zip(moe_x, MOE_B_SWEEP_AGG):
        ax_c.annotate(f"{v:.1f}", (x, v), xytext=(0, 10),
                      textcoords="offset points", ha="center",
                      fontsize=8)
    # Mark the cycle 34 + cycle 35 KEEPs
    ax_c.scatter([64], [464.1], s=200, marker="*", color="#2ca02c",
                 edgecolor="black", zorder=5,
                 label="C34 secondary KEEP (within 36 GB)")
    ax_c.scatter([128], [791.8], s=240, marker="*", color="#d4a017",
                 edgecolor="black", zorder=5,
                 label="C35 secondary KEEP (48 GB hardware)")
    ax_c.annotate("expert routing\namortisation\nthreshold near B=128\n(8/256 experts active;\n~4 activations/expert)",
                  xy=(128, 791.8), xytext=(80, 650), fontsize=8, color="#888",
                  arrowprops=dict(arrowstyle="->", color="#888"),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa", edgecolor="#888"))

    # Memory envelope twin axis
    ax_c2 = ax_c.twinx()
    ax_c2.bar(moe_x, MOE_B_SWEEP_PEAK_GB, alpha=0.18, color="#d62728",
              width=4.0, zorder=1)
    ax_c2.axhline(36, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.5)
    ax_c2.text(4, 36.7, "36 GB envelope", fontsize=7, color="#d62728")
    ax_c2.axhline(48, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.4)
    ax_c2.text(4, 48.7, "48 GB hardware", fontsize=7, color="#d62728")
    ax_c2.set_ylabel("peak memory (GB)", fontsize=9, color="#d62728")
    ax_c2.set_ylim(0, 55)
    ax_c2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)

    ax_c.set_xlabel("B (max_batch_size)")
    ax_c.set_ylabel("MoE aggregate decode_tok_s", fontsize=10, color="#d4a017")
    ax_c.tick_params(axis="y", labelcolor="#d4a017")
    ax_c.set_title("Panel C: MoE 35B-A3B-4bit B-sweep — secondary track (cycles 34-35)\n"
                   "Per-row throughput non-monotonic; B=128 = 791.8 ± 5.2 (4.20× MoE C1)",
                   fontsize=10, fontweight="bold")
    ax_c.set_xticks(moe_x)
    ax_c.legend(loc="upper left", fontsize=8)
    ax_c.set_ylim(0, 850)
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
    ax_d.axhline(204.5, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="C27 envelope KEEP = 204.5 (corrected)")
    ax_d.axhline(231.9, color="#d4a017", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="C28 hardware ceiling = 231.9 (corrected)")
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(E2E_LABELS, fontsize=10)
    ax_d.set_ylabel("decode_tok_s on warm-decode-bN", fontsize=11)
    ax_d.set_title("Panel D: corrected KEEP ladder — C10 → C13 → C27 (v10 phantom retracted)\n"
                   "C27 envelope: 204.5 ± 1.5 (4.85× C1, bf16-only). C28 ceiling: 231.9 ± 0.3 (5.50× C1).",
                   fontsize=11, fontweight="bold")
    ax_d.set_ylim(180, 245)
    ax_d.legend(loc="upper left", fontsize=9)
    ax_d.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.suptitle(
        "Silica-MLX P-6 Autoresearch — Summary (cycles 1-35, 2026-05-02 → 2026-05-04)\n"
        "Mission: push Qwen3.5 production checkpoints on M5 Pro 48 GB toward the hardware limit.\n"
        "Dense 27B (primary): 42.17 → 204.5 envelope KEEP (4.85×) / 231.9 hardware ceiling (5.50×) [C27/C28 corrected]; "
        "MoE 35B-A3B (secondary, cycles 34-35): 188.5 → 464.1 envelope / 791.8 hardware (4.20×; expert routing amortisation at B=128).",
        fontsize=12, fontweight="bold", y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUTDIR / "P6_AUTORESEARCH_SUMMARY.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
