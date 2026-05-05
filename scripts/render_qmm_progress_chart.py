"""Render the QMM kernel tuning progress chart (cycle 6/7).

Shows the simdgroup_matrix QMM kernel p50 latency at production shape
(B=4, K=5120, N=17408) across tuning iterations, with mlx's internal
QMM as a reference line and a theoretical bandwidth-bound floor.

Usage:
    uv run python scripts/render_qmm_progress_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path(__file__).resolve().parent.parent / "plans"

# Hand-curated from REPORT_KERNEL_CYCLE_{3,6,7}.md and ledger.
# (label, cycle, p50_ms, status, color)
KERNELS = [
    ("naive\n(1-thread/output)", 3, 0.66, "discard", "#e57373"),
    ("simdgroup v2\n(MMA)", 6, 0.97, "discard", "#ef9a9a"),
    ("v3\n(+hoist s/b)", 7, 0.89, "diagnostic", "#90caf9"),
    ("v4\n(+4 N/sg)", 7, 0.83, "diagnostic", "#64b5f6"),
    ("v5\n(group dequant)", 7, 1.21, "discard", "#e57373"),
    ("v6\n(uint4 view)", 7, 0.83, "diagnostic", "#64b5f6"),
    ("v7\n(2 sg/tg)", 8, 0.89, "discard", "#ef9a9a"),
    ("v8\n(8 N/sg)", 8, 1.24, "discard", "#e57373"),
    ("v9 ← BEST\n(prefetch+no #pragma)", 8, 0.59, "diagnostic", "#1976d2"),
    ("v10\n(2 K_TILE/barr)", 8, 0.61, "discard", "#90caf9"),
    ("v11\n(simdgroup_barrier)", 8, 0.65, "discard", "#90caf9"),
    ("v12\n(uint32 cache)", 9, 0.73, "discard", "#ef9a9a"),
    ("v13\n(uint4 vector)", 9, 0.78, "discard", "#ef9a9a"),
]

MLX_REF_MS = 0.45  # mx.quantized_matmul at production shape
# Theoretical bandwidth floor: read 50.2 MB of weights at 307 GB/s.
BW_FLOOR_MS = 50.2 / (307 * 1024) * 1000  # = 0.160 ms


def render() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.5))

    labels = [k[0] for k in KERNELS]
    p50s = [k[2] for k in KERNELS]
    colors = [k[4] for k in KERNELS]
    cycles = [k[1] for k in KERNELS]

    xs = list(range(len(KERNELS)))
    bars = ax.bar(xs, p50s, color=colors, edgecolor="black", linewidth=0.5,
                  zorder=3)

    # Annotate bar values
    for i, (bar, k) in enumerate(zip(bars, KERNELS)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.05,
                f"cycle {k[1]}", ha="center", va="bottom",
                fontsize=8, color="#444")

    # Reference lines
    ax.axhline(MLX_REF_MS, color="#2ca02c", linestyle="--", linewidth=2,
               label=f"mlx mx.quantized_matmul ({MLX_REF_MS:.3f} ms)", zorder=2)
    ax.axhline(BW_FLOOR_MS, color="#888888", linestyle=":", linewidth=1.5,
               label=f"weights-bandwidth floor ({BW_FLOOR_MS:.3f} ms @ 307 GB/s)",
               zorder=2)

    # Annotate gap to mlx for the best kernel (v9)
    best_idx = next(i for i, k in enumerate(KERNELS) if "BEST" in k[0])
    best_p50 = p50s[best_idx]
    ax.annotate("", xy=(best_idx, MLX_REF_MS), xytext=(best_idx, best_p50),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.5))
    ax.text(best_idx + 0.25, (best_p50 + MLX_REF_MS) / 2,
            f"gap: {best_p50 - MLX_REF_MS:.3f} ms\n"
            f"({best_p50 / MLX_REF_MS:.2f}× slower)",
            fontsize=9, color="#444", va="center", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("p50 latency (ms) — lower is better")
    ax.set_title(
        "Silica-MLX simdgroup_matrix QMM kernel tuning — "
        "B=4 K=5120 N=17408 4-bit affine, cycles 3-8\n"
        "Best kernel v9 = 0.59 ms (40% faster than v2 baseline; 1.27× from mlx ref); "
        "v5/v7/v8/v10/v11 retired",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim(0, max(p50s) * 1.18)

    fig.tight_layout()
    out = OUTDIR / "P6_AUTORESEARCH_PROGRESS_QMM_KERNEL.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def render_cycle_summary() -> None:
    """Render a per-cycle activity / kept-improvement panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # Per-cycle deliverables (from cycle reports). 35 cycles total; cycles
    # 24-26 contributed by Codex review (opus-codex branch); cycles 27-31
    # close dense 27B characterisation; cycles 32-35 add MoE secondary
    # track + post-23-cycle continuation work.
    cycles = list(range(1, 36))
    new_kernels = [0, 3, 1, 0, 0, 1, 4, 5, 2, 0, 8, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    new_probes = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
    discards = [0, 0, 5, 1, 1, 1, 1, 4, 2, 0, 1, 0, 1, 2, 4, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    diagnostics = [4, 1, 4, 4, 2, 1, 5, 1, 0, 4, 4, 3, 2, 1, 1, 3, 1,
                   1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1]

    width = 0.2
    xs = list(range(len(cycles)))
    ax1.bar([x - 1.5 * width for x in xs], new_kernels, width,
            label="new custom Metal kernels", color="#1f77b4")
    ax1.bar([x - 0.5 * width for x in xs], new_probes, width,
            label="new microbench probes", color="#9467bd")
    ax1.bar([x + 0.5 * width for x in xs], discards, width,
            label="discard ledger rows", color="#9e9e9e")
    ax1.bar([x + 1.5 * width for x in xs], diagnostics, width,
            label="diagnostic ledger rows", color="#42a5f5")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"C{c}" for c in cycles])
    ax1.set_ylabel("count")
    ax1.set_title("Per-cycle deliverables")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Running-best decode_tok_s over cycles
    # Cycle 10: BREAKTHROUGH — 42.17 -> 193.9 via batched-aggregate path
    # Cycle 11: FA-decode kernel-level beat over mlx (no E2E delta)
    # Cycle 12: bf16 DeltaNet state correct but no E2E delta (193 wall AT FIXED B=48)
    # Cycle 13: bf16 state's peak-memory headroom unlocks B-axis past 48 ->
    #           B=52 = 200.8 within strict envelope (NEW RUNNING-BEST);
    #           B=64 = 229.8 within hardware ceiling (DEMONSTRATED CEILING)
    # CYCLE 27 CORRECTION: cycle-14's claimed 206.2 was attribution error
    # (v10 wasn't firing due to dtype defect; small n=3 σ underestimated true
    # variance). Corrected running-best is 204.5 ± ~1.5 at B=52 bf16.
    running_best = [42.17] * 9 + [
        193.9,  # C10 BREAKTHROUGH
        193.3, 192.5,  # C11/C12: kernel + state probes 0% E2E at fixed B
        200.8,  # C13: bf16 peak save unlocks B=52
        204.5,  # C14 (corrected from 206.2)
        204.5, 204.5, 204.5, 204.5,  # C15-18: local-optimum confirmation
        204.5, 204.5, 204.5, 204.5, 204.5,  # C19-23: spec-decode arm closes
        204.5,  # C24: codex dep pin
        204.5, 204.5,  # C25/C26: codex reverify
        204.5,  # C27: corrected attribution (KEEP-revision)
        204.5,  # C28: B=64 ceiling re-measured (within-envelope unchanged)
        204.5, 204.5, 204.5,  # C29 cliff / C30 decomp / C31 deltanet parity
        204.5, 204.5,  # C32 chart re-render / C33 variance characterisation
        204.5, 204.5,  # C34 MoE portability / C35 MoE B=128 (dense unchanged)
    ]
    # MoE 35B-A3B secondary-track overlay (steps at C1, C34, C35)
    moe_running_best = (
        [188.5] * 33  # C1-C33: MoE B=4 cycle-1 baseline
        + [464.1]     # C34: first MoE secondary KEEP (B=64 within 36 GB)
        + [791.8]     # C35: B=128 hardware-ceiling (expert amortisation)
    )
    cycle_labels_full = [
        "C1 orient", "C2 simple", "C3 QMM naive", "C4 lazy chain",
        "C5 batcher", "C6 simdgroup", "C7 tune v3-v6", "C8 v7-v11",
        "C9 v12-v13", "C10 ⭐ B=48", "C11 FA-decode", "C12 bf16 probe",
        "C13 B=52 bf16", "C14 (retracted)", "C15 confirm", "C16 confirm",
        "C17 confirm", "C18 confirm", "C19 cov@64", "C20 verify cost",
        "C21 drafter", "C22 design", "C23 spec end", "C24 dep pin",
        "C25 reverify", "C26 bf16 FA fix", "C27 ⭐ correction",
        "C28 ceiling 232", "C29 cliff arch", "C30 deltanet 88%",
        "C31 deltanet parity", "C32 charts", "C33 variance",
        "C34 MoE port", "C35 MoE B=128 ⭐⭐",
    ]
    ax2.plot(xs, running_best, "o-", color="#2ca02c", linewidth=2.5,
             markersize=10, zorder=3, label="dense 27B running best (primary)")
    ax2.plot(xs, moe_running_best, "s-", color="#d4a017", linewidth=2.0,
             markersize=8, zorder=3, alpha=0.85,
             label="MoE 35B-A3B running best (secondary)")
    ax2.axhline(60, color="#ff7f0e", linestyle="--", linewidth=1.5,
                label="(1b) milestone (60 tok/s)")
    ax2.axhline(81.16, color="#444", linestyle=":", linewidth=1,
                label="B=4 weights ceiling (81.16 tok/s)")
    ax2.axhline(67, color="#1f77b4", linestyle=":", linewidth=1,
                label="demonstrated 82.7% util projection (67 tok/s)")

    for i, v in enumerate(running_best):
        ax2.annotate(f"{v:.2f}", (xs[i], v), xytext=(0, 12),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="#2ca02c")

    # C27 corrected envelope KEEP (replaces C14's retracted claim)
    ax2.annotate(
        "C27 ⭐ envelope KEEP (corrected):\nB=52 bf16 = 204.5 ± ~1.5 (4.85× C1)\n(C14 claim 206.2 was retracted —\nv10 didn't fire due to dtype bug;\nrunning-best stable since C13/C14)\nC30: DeltaNet 88% / C31: kernel parity\nLoop reaches genuine closed state",
        xy=(26, 204.5), xytext=(15, 78),
        fontsize=7, color="#2ca02c", ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c"),
    )
    # C28 corrected demonstrated ceiling (dense)
    ax2.axhline(231.9, color="#1976d2", linestyle=":", linewidth=1.5, alpha=0.6)
    ax2.text(34, 234.5, "C28 dense ceiling B=64 bf16 = 231.9 ± 0.3",
             fontsize=8, color="#1976d2", ha="right", fontweight="bold")
    # C35 MoE secondary track unlock annotation
    ax2.annotate(
        "C35 ⭐⭐ MoE B=128 = 791.8 ± 5.2\n(48 GB hardware ceiling; expert\nrouting amortisation crossed)",
        xy=(34, 791.8), xytext=(20, 720),
        fontsize=8, color="#d4a017", ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="#d4a017"),
    )
    ax2.annotate(
        "C34 MoE B=64 = 464.1 ± 0.7\n(36 GB envelope KEEP)",
        xy=(33, 464.1), xytext=(18, 400),
        fontsize=7, color="#d4a017", ha="center",
        arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.0),
    )

    ax2.set_xticks(xs)
    ax2.set_xticklabels(cycle_labels_full, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("decode_tok_s")
    ax2.set_title("Running best across 35 cycles — dense 27B primary at 204.5 envelope / 231.9 hardware (5.50× C1)\n"
                  "MoE 35B-A3B secondary unlocked C34-C35: 188.5 → 464.1 envelope → 791.8 hardware (4.20× MoE C1)")
    ax2.set_ylim(0, 850)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Silica-MLX P-6 Autoresearch — cycle-by-cycle progress (35 cycles, 2026-05-02 → 2026-05-04)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = OUTDIR / "P6_AUTORESEARCH_PROGRESS_CYCLES.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def render_fa_kernel_progress() -> None:
    """Render cycle 11 FA-decode kernel ablation: silica vs mlx across T_kv."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # Variant ablation at the most-illustrative shape (B=48, T_kv=128, plain)
    variants = ["v1\nbaseline", "v3\nGQA tile\nshare", "v6\n+ K-split\n+ stream\nsoftmax",
                "v7\n+ half4\nload", "v8\n+ half4\ninner ops",
                "v10\n+ single-pass\nfast path"]
    p50_T128 = [0.74, 0.45, 0.40, 0.32, 0.32, 0.28]
    colors = ["#e57373", "#ffb74d", "#fff176", "#aed581", "#4fc3f7", "#1976d2"]

    xs = list(range(len(variants)))
    bars = ax1.bar(xs, p50_T128, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    for x, v in zip(xs, p50_T128):
        ax1.annotate(f"{v:.2f}", (x, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax1.axhline(0.51, color="#2ca02c", linestyle="--", linewidth=2,
                label="mlx mx.fast.scaled_dot_product_attention (0.51 ms)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(variants, fontsize=8)
    ax1.set_ylabel("p50 latency at B=48 T_kv=128 (ms, lower better)")
    ax1.set_title("Cycle 11 FA-decode kernel ablation\n"
                  "v10 = 0.55× mlx (silica beats by 1.81×) at production T_kv=128",
                  fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax1.set_ylim(0, max(p50_T128) * 1.2)

    # silica vs mlx across T_kv sweep at B=48 (5-run median, plain variant)
    T_kv_values = [128, 256, 512, 1024]
    silica_p50 = [0.28, 0.44, 0.71, 1.07]
    mlx_p50 = [0.51, 0.70, 0.89, 1.35]
    ratio = [s / m for s, m in zip(silica_p50, mlx_p50)]

    ax2.plot(T_kv_values, silica_p50, "o-", color="#1976d2", linewidth=2.5,
             markersize=10, label="silica v8/v10")
    ax2.plot(T_kv_values, mlx_p50, "s-", color="#2ca02c", linewidth=2.5,
             markersize=10, label="mlx mx.fast.scaled_dot_product_attention")
    for tk, sp, mp, r in zip(T_kv_values, silica_p50, mlx_p50, ratio):
        ax2.annotate(f"{sp:.2f}", (tk, sp), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=8, color="#1976d2")
        ax2.annotate(f"{mp:.2f}", (tk, mp), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8, color="#2ca02c")
        ax2.annotate(f"silica/mlx={r:.2f}", (tk, (sp + mp) / 2),
                     xytext=(15, 0), textcoords="offset points",
                     fontsize=8, color="#888", fontweight="bold")
    ax2.set_xlabel("T_kv")
    ax2.set_ylabel("p50 latency at B=48 (ms)")
    ax2.set_title("Production B=48 sweep — silica beats mlx 1.25-1.81×\n"
                  "kernel-level victory; E2E flat at warm-decode-b48 (attn = 22% of step)",
                  fontsize=10, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(T_kv_values)
    ax2.set_xticklabels([str(t) for t in T_kv_values])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Silica-MLX FA-decode native MLX port (cycle 11) — kernel-level beat over mlx",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = OUTDIR / "P6_AUTORESEARCH_PROGRESS_FA_KERNEL.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
    render_cycle_summary()
    render_fa_kernel_progress()
