// Performance — animated SVG showcase of the P-6 autoresearch loop.
//
// The Karpathy-style ledger animates on scroll into view: dots fade
// in left-to-right at ~80 ms apart, and the running-best ladder
// strokes in synchronously through stroke-dashoffset interpolation.
// Each dot is clickable; clicking opens a detail panel below the
// chart with that experiment's cycle, measurement, and a short
// narrative. Cycle 14's retracted KEEPs render as dashed open
// circles and do *not* contribute to the running-best line — the
// ladder steps from cycle 13 (229.8) directly to cycle 28 (231.9),
// embedding the codex retraction into the visual.

const PerfDensePoints = [
  // [seq, tok_s, status, cycle?, label?, description?]
  [1, 42.17, "keep", "C1", "starting point",
    "Where we begin. Decoding 4 prompts in parallel (batch size 4) on Qwen3.5-27B reaches 42 tokens per second total throughput. The chip's memory bandwidth could in principle support more — we're using about 52% of it. Every later number is compared against this 42."],
  [2, 6.54, "discard", null, null,
    "First speculative-decoding probe — a small companion model guesses tokens, the big model verifies. Slower than plain decoding (about 40% the speed). Retired."],
  [3, 7.74, "discard", null, null,
    "Block-diffusion drafter probe. Drafter took longer than the speedup it provided. Retired."],
  [4, 17.10, "discard", null, null,
    "Tried letting the GPU evaluate several decode steps before syncing. Broke a stability check. Retired."],
  [5, 39.99, "diag", null, null,
    "Re-baselined at a different attention-head dimension. Just for measurement; no change."],
  [6, 41.00, "discard", null, null,
    "Skipped a small array allocation in the sampler hot path. Tiny regression. Retired."],
  [7, 42.39, "discard", null, null,
    "First custom GPU kernel attempt — fused output gate. No measurable benefit. Retired."],
  [8, 42.53, "discard", null, null,
    "Fused activation kernel (silu × multiply). No benefit. Retired."],
  [9, 42.68, "diag", null, null,
    "Re-baselined at production attention shape. Within noise of cycle 1."],
  [10, 43.0, "diag", null, null,
    "First step of the batch sweep: 8 prompts in parallel (batch size 8). Just above baseline because batch size 4 → batch size 8 doesn't yet pay off enough to dominate per-step overhead."],
  [11, 63.1, "keep", null, null,
    "First win: 12 prompts in parallel (batch size 12) = 63 tok/s. The first time we crossed the 60 tok/s stretch goal — without writing any new code. Just a different batch size: 12 prompts share the same per-step weight read instead of doing 12 sequential reads."],
  [12, 81.1, "diag", null, null,
    "batch size 16. Throughput keeps climbing cleanly — the per-step weight read amortizes across more in-flight tokens."],
  [13, 112.8, "diag", null, null,
    "batch size 24. About 2.7× the starting point already, just from packing more prompts into the same step."],
  [14, 150.6, "keep", null, null,
    "batch size 32: 150 tok/s, 3.6× the start. Memory peak ~26 GB, well within the 36 GB budget."],
  [15, 171.6, "keep", null, null,
    "batch size 40: 172 tok/s. About 4× the start."],
  [16, 183.3, "keep", null, null,
    "batch size 44: 183 tok/s. Approaching the strict 36 GB memory budget."],
  [17, 193.9, "keep", "C10", "the breakthrough · batch 48",
    "The breakthrough. Decoding 48 prompts in parallel (batch size 48) = 4.6× the starting throughput, with no kernel change at all. The trick was just re-reading our own goal — \"maximize total tokens per second across the batch\", not per individual prompt — and then increasing the batch size until we hit a memory boundary. Memory peak 34 GB, still inside the 36 GB budget. Nine prior cycles of GPU-kernel hacking had moved nothing; this single parameter choice did."],
  [18, 193.3, "diag", null, null,
    "Tested a custom GPU attention kernel at batch size 48. Wins on a small microbenchmark, but flat at the whole-system level — attention is only a small slice of total step time at high batch."],
  [19, 192.5, "diag", null, null,
    "Tried storing one piece of state — the model's recurrent memory — in 16-bit floats instead of 32-bit. Same speed at batch size 48, but frees about 3.5 GB of memory. That free headroom is the seed for the next breakthrough."],
  [20, 200.8, "keep", "C13", "the composition win",
    "Composition. The 16-bit memory save from cycle 12 doesn't speed anything up by itself, but it frees enough headroom to bump batch size from 48 to 52 (batch size 52) — and that bump pushes throughput past 200 tok/s, still inside the 36 GB budget. Two cheap parameter changes beat every kernel attempt."],
  [21, 212.2, "diag", null, null,
    "batch size 56. Past the strict 36 GB budget but still inside the 48 GB chip-memory cap."],
  [22, 219.1, "diag", null, null,
    "batch size 60."],
  [23, 229.8, "keep", null, null,
    "batch size 64: 230 tok/s. About 5.5× the starting point — pushing right against the 48 GB chip memory limit."],
  [24, 166.8, "discard", null, null,
    "Tried batch size 66. Throughput collapsed 26%. We hit a hardware cliff at the 40 GB memory peak — beyond it, the chip stops scaling. Three allocator settings tried; the cliff is in the chip itself, not in our code."],
  [25, 169.1, "discard", null, null,
    "batch size 68 — also past the cliff."],
  [26, 173.2, "discard", null, null,
    "batch size 72 — confirming the pattern. Dense 27B can't benefit from larger batches on this chip."],
  [27, 206.2, "retracted", "C14→C27", "retracted",
    "A retracted result. Cycle 14 claimed a 5-tok/s gain at batch size 52 from a custom GPU attention kernel. Two weeks later a code review caught a bug — the code checked for the wrong floating-point format and the kernel was silently being skipped on the production model. After the fix, the kernel's real contribution measured to within noise (about 0.5 tok/s, statistically zero). The honest credit goes to cycles 10 and 12; we kept this dot on the chart so the retraction stays visible."],
  [28, 232.2, "retracted", "C14→C27", "retracted",
    "Same retracted experiment at batch size 64. Cycle 14 reported 232 tok/s with the custom kernel; the code-review fix showed it was actually slightly slower than the simpler version (still within noise). The real ceiling result comes two indices later, at 231.9."],
  [29, 197.05, "diag", null, null,
    "First re-check at batch size 52 after the code-review fix. Within the cycle-13 range; no new claim."],
  [30, 200.14, "diag", null, null,
    "Re-check at batch size 52 under one Python environment — back near cycle-13."],
  [31, 185.30, "diag", null, null,
    "Re-check at batch size 52 under a different Python environment — about 10% lower. Identified as between-environment drift, not a regression."],
  [32, 231.9, "keep", "C28", "honest ceiling",
    "The honest hardware-ceiling result. batch size 64 → 232 tok/s, measured 3 times with tight agreement (±0.3 tok/s). About 5.5× the starting point, just inside the 48 GB chip memory limit. This number replaces the cycle-14 retracted claim."],
  [33, 204.0, "keep", "C33", "best within budget",
    "Final tighten. Re-measured at batch size 52 across 6 runs in two sessions: 204 ± 1 tok/s. This is the best result within the strict 36 GB memory budget. The variance protocol used here (multiple sessions, combined error check) is now the standard for any future claim."],
];

const PerfMoePoints = [
  [1, 188.5, "keep", "C1", "starting point",
    "Same starting line, this time for the mixture-of-experts version. 188 tok/s for 4 prompts in parallel (batch size 4) on Qwen3.5-35B-A3B — a 35-billion-parameter MoE model where only 8 of 256 experts are active per token, so each token's active-weight footprint is much smaller than dense 27B's. The original ≥100 tok/s goal was already satisfied here at the baseline."],
  [2, 181.4, "diag", null, null,
    "Tried the 16-bit recurrent state on MoE at batch size 4. Slightly below baseline — batch size 4 amortizes poorly on this architecture."],
  [3, 242.0, "diag", null, null,
    "batch size 8. Climbing."],
  [4, 306.3, "diag", null, null,
    "batch size 16."],
  [5, 384.8, "diag", null, null,
    "batch size 32."],
  [6, 434.3, "diag", null, null,
    "batch size 48."],
  [7, 464.4, "keep", "C34", "best within budget · batch 64",
    "MoE win within the strict memory budget. batch size 64 → 464 tok/s, peak memory 33.8 GB (inside the 36 GB budget). Same parameter changes from cycles 10 and 12 ported over to MoE via a shared memory hook. About 2.5× the MoE starting point."],
  [8, 447.9, "discard", null, null,
    "batch size 80 — past the strict budget, throughput dips."],
  [9, 444.5, "discard", null, null,
    "batch size 72."],
  [10, 467.1, "diag", null, null,
    "batch size 96 — climbing again as we sweep toward the chip-memory limit."],
  [11, 791.8, "keep", "C35", "biggest result · batch 128",
    "The biggest result of the entire 35-cycle effort. batch size 128 → 792 tok/s on the MoE model, sitting almost exactly at the 48 GB chip memory limit (peak 47.96 GB), measured 3 times. About 4.2× the MoE starting point. Unlike dense 27B, the MoE architecture doesn't hit a memory cliff at this size — because only 8 of 256 experts are active per token, the active-weight footprint per prompt is much smaller, leaving room for more parallel prompts before the chip caps out."],
];

const PerfTracks = {
  dense: {
    label: "Dense Qwen3.5-27B-4bit",
    sub: "38 measurements · 9 new bests · running best 232 tok/s",
    points: PerfDensePoints,
    yMax: 260,
    yTicks: [0, 50, 100, 150, 200, 250],
    baseline: 42.17,
    baselineLabel: "cycle-1 baseline 42.17",
  },
  moe: {
    label: "MoE Qwen3.5-35B-A3B-4bit",
    sub: "11 measurements · 2 new bests · running best 791.8 tok/s",
    points: PerfMoePoints,
    yMax: 850,
    yTicks: [0, 200, 400, 600, 800],
    baseline: 188.5,
    baselineLabel: "cycle-1 baseline 188.5",
  },
};

const PerfLedger = ({ trackKey, animateKey, onSelect, activeIdx }) => {
  const t = PerfTracks[trackKey];
  const data = t.points;

  // Animation progress — 0 (empty) to 1 (all dots visible).
  const [progress, setProgress] = React.useState(0);
  const lineRef = React.useRef(null);
  const [lineLength, setLineLength] = React.useState(0);

  // Reset + restart animation when animateKey changes (tab swap or replay).
  React.useEffect(() => {
    setProgress(0);
    let raf;
    const start = performance.now();
    const duration = trackKey === "moe" ? 1800 : 2600;
    const tick = (now) => {
      const t01 = Math.min(1, (now - start) / duration);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t01, 3);
      setProgress(eased);
      if (t01 < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animateKey, trackKey]);

  // Measure the line's geometric length once it's mounted so the
  // dashoffset trick produces a smooth left-to-right draw rather
  // than a sweep proportional to dot count.
  React.useLayoutEffect(() => {
    if (lineRef.current && typeof lineRef.current.getTotalLength === "function") {
      try {
        setLineLength(lineRef.current.getTotalLength());
      } catch (_) {
        setLineLength(0);
      }
    }
  }, [trackKey]);

  const W = 1100;
  const H = 360;
  const padL = 64;
  const padR = 36;
  const padT = 30;
  const padB = 44;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const xAt = i => padL + (i / Math.max(1, data.length - 1)) * innerW;
  const yAt = v => padT + (1 - v / t.yMax) * innerH;

  // Honest running-best (skip retracted).
  let best = 0;
  const bestSeries = data.map(p => {
    const status = p[2];
    const tok = p[1];
    if (status !== "retracted" && tok > best) best = tok;
    return best;
  });

  // Step polyline that hugs each measurement's x position.
  const stepPoints = [];
  bestSeries.forEach((b, i) => {
    if (i === 0) {
      stepPoints.push([xAt(i), yAt(b)]);
    } else {
      stepPoints.push([xAt(i), yAt(bestSeries[i - 1])]);
      stepPoints.push([xAt(i), yAt(b)]);
    }
  });

  // Closed area path under the running-best line, from each step point
  // down to the cycle-1 baseline. Used as a soft fill behind the line.
  const areaCmds = [];
  if (stepPoints.length > 0) {
    areaCmds.push(`M ${stepPoints[0][0].toFixed(2)},${yAt(t.baseline).toFixed(2)}`);
    stepPoints.forEach(p => {
      areaCmds.push(`L ${p[0].toFixed(2)},${p[1].toFixed(2)}`);
    });
    const lastX = stepPoints[stepPoints.length - 1][0];
    areaCmds.push(`L ${lastX.toFixed(2)},${yAt(t.baseline).toFixed(2)}`);
    areaCmds.push("Z");
  }
  const areaPath = areaCmds.join(" ");

  // Honest-vs-false retraction connector (dense only). When cycle 14's
  // claimed KEEPs at indices 26-27 are taken at face value, the running
  // best would have jumped to 232.2 at index 27 and stayed flat through
  // cycle 28's honest 231.9 reverify. The dashed connector visualises
  // that "missed shortcut" — the ladder bumps up over the retracted
  // points and drops back to the honest line.
  const retractionPath = (() => {
    if (trackKey !== "dense") return null;
    const lastHonestIdx = 22; // cycle-13 hardware ceiling at 229.8
    const falsePeakIdx = 27; // higher of the two retracted (232.2)
    const honestResumeIdx = 31; // cycle-28 honest KEEP at 231.9
    const lastHonest = bestSeries[lastHonestIdx]; // 229.8
    const falsePeakVal = data[falsePeakIdx][1]; // 232.2
    const honestResumeVal = data[honestResumeIdx][1]; // 231.9
    return [
      `M ${xAt(lastHonestIdx).toFixed(2)},${yAt(lastHonest).toFixed(2)}`,
      `L ${xAt(falsePeakIdx).toFixed(2)},${yAt(lastHonest).toFixed(2)}`,
      `L ${xAt(falsePeakIdx).toFixed(2)},${yAt(falsePeakVal).toFixed(2)}`,
      `L ${xAt(honestResumeIdx).toFixed(2)},${yAt(falsePeakVal).toFixed(2)}`,
      `L ${xAt(honestResumeIdx).toFixed(2)},${yAt(honestResumeVal).toFixed(2)}`,
    ].join(" ");
  })();

  // Draw progress: stroke-dashoffset interpolation reveals the line
  // from left to right.
  const dashOffset = lineLength * (1 - progress);
  const gradientId = `perf-area-${trackKey}`;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="perf-svg"
      role="img"
      aria-label={`${t.label} — Karpathy autoresearch ledger`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.18" />
          <stop offset="60%" stopColor="var(--accent)" stopOpacity="0.05" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* area fill under the running-best line — soft accent gradient */}
      <path
        d={areaPath}
        fill={`url(#${gradientId})`}
        className="perf-svg-area"
        style={lineLength ? {
          opacity: 0.95 * progress,
        } : { opacity: 0 }}
      />

      {/* gridlines */}
      {t.yTicks.map(g => (
        <g key={`grid-${g}`}>
          <line
            x1={padL}
            x2={W - padR}
            y1={yAt(g)}
            y2={yAt(g)}
            className="perf-svg-grid"
          />
          <text
            x={padL - 14}
            y={yAt(g) + 4}
            className="perf-svg-tick mono"
            textAnchor="end"
          >
            {g}
          </text>
        </g>
      ))}

      {/* axis labels */}
      <text
        x={padL - 14}
        y={padT - 12}
        className="perf-svg-axis-title mono"
        textAnchor="end"
      >
        tok/s
      </text>
      <text
        x={W / 2}
        y={H - 10}
        className="perf-svg-axis-title mono"
        textAnchor="middle"
      >
        experiment index · cycles 1 → 35
      </text>

      {/* baseline reference line */}
      <line
        x1={padL}
        x2={W - padR}
        y1={yAt(t.baseline)}
        y2={yAt(t.baseline)}
        className="perf-svg-baseline"
      />
      <text
        x={W - padR - 6}
        y={yAt(t.baseline) - 6}
        className="perf-svg-baseline-label mono"
        textAnchor="end"
      >
        {t.baselineLabel}
      </text>

      {/* running-best step polyline (animated draw) */}
      <polyline
        ref={lineRef}
        points={stepPoints.map(p => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ")}
        className="perf-svg-best"
        style={lineLength ? {
          strokeDasharray: lineLength,
          strokeDashoffset: dashOffset,
        } : { opacity: 0 }}
      />

      {/* retraction connector — the "missed shortcut" if cycle-14
          KEEPs had not been retracted; fades in after main line draws */}
      {retractionPath && (
        <path
          d={retractionPath}
          className="perf-svg-retract"
          style={{ opacity: progress > 0.85 ? Math.min(1, (progress - 0.85) / 0.15) : 0 }}
        />
      )}

      {/* dots — fade in based on progress */}
      {data.map((p, i) => {
        const [seq, tok, status, cycle, label] = p;
        const cx = xAt(i);
        const cy = yAt(tok);
        const threshold = i / Math.max(1, data.length - 1);
        const visible = progress >= threshold;
        const isKeep = status === "keep";
        const isRetracted = status === "retracted";
        const isActive = activeIdx === i;
        return (
          <g
            key={`d-${i}`}
            className={"perf-svg-dot-g" + (visible ? " perf-svg-dot-on" : "") + (isActive ? " perf-svg-dot-active" : "")}
            onClick={(e) => { e.stopPropagation(); onSelect(i); }}
          >
            {/* expanded hit area */}
            <circle cx={cx} cy={cy} r={14} fill="transparent" className="perf-svg-hit" />
            {isRetracted ? (
              <circle
                cx={cx}
                cy={cy}
                r={isActive ? 6 : 5}
                className="perf-svg-dot perf-svg-dot-retracted"
              />
            ) : (
              <circle
                cx={cx}
                cy={cy}
                r={isActive ? 6 : (isKeep ? 4.5 : 2.8)}
                className={`perf-svg-dot perf-svg-dot-${status}`}
              />
            )}
            {isActive && (
              <circle
                cx={cx}
                cy={cy}
                r={11}
                className="perf-svg-dot-ring"
                fill="none"
              />
            )}
          </g>
        );
      })}

      {/* milestone annotations */}
      {data.map((p, i) => {
        const [_, tok, status, cycle, label] = p;
        if (!cycle || !label) return null;
        const threshold = i / Math.max(1, data.length - 1);
        const visible = progress >= threshold;
        if (!visible) return null;
        const cx = xAt(i);
        const cy = yAt(tok);
        const above = cy > padT + innerH * 0.4;
        const yLine1 = above ? cy - 30 : cy + 22;
        const yLine2 = above ? cy - 16 : cy + 34;
        return (
          <g key={`anno-${i}`} className="perf-svg-anno-g">
            <line
              x1={cx}
              x2={cx}
              y1={cy}
              y2={above ? cy - 12 : cy + 12}
              className="perf-svg-anno-tick"
            />
            <text
              x={cx}
              y={yLine1}
              textAnchor="middle"
              className="perf-svg-anno-cycle mono"
            >
              {cycle}
            </text>
            <text
              x={cx}
              y={yLine2}
              textAnchor="middle"
              className="perf-svg-anno-label mono"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

const PerfDetailPanel = ({ point, trackKey, onClose }) => {
  if (!point) {
    return (
      <div className="perf-detail-panel perf-detail-panel-empty">
        <div className="perf-detail-panel-hint mono">Click any dot above to read that experiment's record.</div>
      </div>
    );
  }
  const [seq, tok, status, cycle, label, description] = point;
  const tokFmt = tok.toFixed(tok < 100 ? 2 : 1);
  return (
    <div className={"perf-detail-panel perf-detail-panel-active perf-detail-panel-" + status}>
      <button type="button" className="perf-detail-panel-close" onClick={onClose} aria-label="close">×</button>
      <div className="perf-detail-panel-meta">
        <span className="perf-detail-panel-seq mono">#{seq.toString().padStart(2, "0")}</span>
        {cycle && <span className="perf-detail-panel-cycle mono">{cycle}</span>}
        <span className={"perf-detail-panel-status mono perf-status-" + status}>{PerfStatusLabel[status] || status}</span>
      </div>
      <div className="perf-detail-panel-tok mono">
        {tokFmt}<span className="perf-detail-panel-unit"> tok/s</span>
      </div>
      {label && <div className="perf-detail-panel-label">{label}</div>}
      {description && <div className="perf-detail-panel-body">{description}</div>}
    </div>
  );
};

const PerfGates = [
  { gate: "(1a) Dense engineering", target: "≥40 tok/s", cleared: "204 ± 1 tok/s", mult: "4.85×", frame: "batch size 52 · 36 GB envelope" },
  { gate: "(1b) Dense stretch", target: "≥60 tok/s", cleared: "231.9 ± 0.3 tok/s", mult: "3.87×", frame: "batch size 64 · 48 GB hardware ceiling" },
  { gate: "(2a) MoE anchor", target: "≥100 tok/s", cleared: "120.93 tok/s", mult: "preserved", frame: "v1.7.13 baseline · MoE batch size 2" },
  { gate: "(2b) MoE stretch", target: "≥175 tok/s", cleared: "791.8 ± 5.2 tok/s", mult: "4.52×", frame: "MoE batch size 128 · 48 GB ceiling" },
];

const PerfDetails = [
  {
    id: "levers",
    num: "01",
    tag: "How it worked",
    title: "Two parameter changes; zero kernels.",
    points: [
      {
        head: "Lever 1 · re-read the goal.",
        body: "We were optimizing per-prompt speed; the metric we actually wanted was total throughput across the batch. Sweeping batch size 4 → 48 within the 36 GB memory budget gave 4.6× the starting throughput, with no code change.",
      },
      {
        head: "Lever 2 · 16-bit recurrent state.",
        body: "Storing one piece of model state in 16-bit floats instead of 32-bit frees ~3.5 GB of memory. By itself: no speedup. Combined with Lever 1: the batch can climb past 48 to 52 (best within the 36 GB budget) and 64 (best within the 48 GB chip cap).",
      },
      {
        head: "The lesson.",
        body: "17 custom GPU kernel attempts produced zero wins. The unlock was operating-point selection, not kernel hacking — because mlx's existing kernels are already at the chip's memory-bandwidth limit and the dominant cost is data movement, not compute.",
      },
    ],
  },
  {
    id: "honest",
    num: "02",
    tag: "What didn't work",
    title: "Three roads we walked before turning around.",
    points: [
      {
        head: "Speculative decoding — closed with a negative.",
        body: "Verify-cost grows roughly linearly with batch size. At batch size 52 the verifier alone takes 8 seconds per step, vs 0.25 s for plain decoding. Tree-spec produces ~10 tok/s, a 20× regression. No batch size in {1, 4, 16, 52} where any spec variant beats plain decode.",
      },
      {
        head: "Bigger batches past 64 — hardware cliff.",
        body: "batch size 64 → batch size 66 throughput drops 26% at a 40 GB memory boundary. Three allocator-tuning probes leave the cliff in place. The cliff is in the chip itself (likely SLC threshold or memory-bandwidth contention near the 48 GB cap), not in our code.",
      },
      {
        head: "Cycle 14's retracted kernel claim.",
        body: "Code review found a typo in a dtype check that silently skipped a custom GPU kernel for 14 cycles. Honest re-measure: the kernel adds about 0.5 tok/s, indistinguishable from noise. Both retracted measurements stay on the chart as dashed circles — public retraction.",
      },
    ],
  },
  {
    id: "method",
    num: "03",
    tag: "How we measured",
    title: "Karpathy-style ledger with variance discipline.",
    points: [
      {
        head: "One row per measurement.",
        body: "Main agent appends to the TSV ledger; sub-agents return findings rather than editing the ledger directly. 110 rows across the 35-cycle effort. Every dot on the chart is one row.",
      },
      {
        head: "Three reps × two sessions before declaring a win.",
        body: "Within-session error underestimates run-to-run variance — that's how cycle 14 published a result that needed retraction. The standard now: ≥3 reps per session, ≥2 sessions, combined error ≤ 1.5 tok/s before any new best is recorded.",
      },
      {
        head: "Toolchain pinned and gated.",
        body: "mlx 0.31.1 / mlx-lm 0.31.2 / mlx-metal 0.31.1. A determinism test (3/3 must pass) catches drift before any new measurement gets compared to the running best.",
      },
    ],
  },
];

const PerfStatusLabel = {
  keep: "new best",
  diag: "measurement",
  discard: "tried, didn't help",
  retracted: "retracted later",
};

const PerfLegend = () => (
  <div className="perf-legend mono">
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-keep"></span>
      new best
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-diag"></span>
      measurement
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-discard"></span>
      tried, didn't help
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-retracted"></span>
      retracted later
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-line"></span>
      best so far
    </span>
  </div>
);

const Performance = () => {
  const [activeTab, setActiveTab] = React.useState("dense");
  const [activeIdx, setActiveIdx] = React.useState(null);
  const [animateKey, setAnimateKey] = React.useState(0);
  const [hasAnimated, setHasAnimated] = React.useState(false);
  const sectionRef = React.useRef(null);
  const panelRef = React.useRef(null);

  const t = PerfTracks[activeTab];
  const point = activeIdx != null ? t.points[activeIdx] : null;

  // Smooth-scroll the detail panel into view when a dot is clicked,
  // so a click on a dot near the top of a tall SVG does not feel
  // unresponsive on a tall viewport.
  React.useEffect(() => {
    if (activeIdx == null) return;
    const el = panelRef.current;
    if (el && typeof el.scrollIntoView === "function") {
      try {
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } catch (_) {
        /* older browsers ignore the option object — no-op fallback */
      }
    }
  }, [activeIdx]);

  // Trigger animation on first scroll into view.
  React.useEffect(() => {
    if (hasAnimated) return;
    const node = sectionRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setAnimateKey(k => k + 1);
      setHasAnimated(true);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !hasAnimated) {
          setAnimateKey(k => k + 1);
          setHasAnimated(true);
        }
      });
    }, { threshold: 0.25 });
    io.observe(node);
    return () => io.disconnect();
  }, [hasAnimated]);

  const handleTab = (key) => {
    if (key === activeTab) return;
    setActiveTab(key);
    setActiveIdx(null);
    setAnimateKey(k => k + 1);
  };

  const replay = () => {
    setActiveIdx(null);
    setAnimateKey(k => k + 1);
  };

  return (
    <section className="block" id="performance" ref={sectionRef}>
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">Throughput Autoresearch</div>
          <h2>35 cycles. Server throughput 5.5×. Single-user, untouched.</h2>
          <p>
            We pushed Qwen3.5-27B-4bit <em>server-aggregate</em> decoding from 42 to 232 tokens per second on M5 Pro 48 GB across 35 experiments. The unlock was running more prompts in parallel, not faster decoding per prompt &mdash; per-row throughput moves the opposite way (10.5 tok/s/row at batch 4, 3.9 tok/s/row at batch 52, ~20 tok/s the bandwidth ceiling at batch 1). P-6 was a server-throughput phase; <strong>single-user interactive latency was not its goal</strong> &mdash; that's <span className="mono">D-022</span>, the next research line.
          </p>
          <p>
            The chart below is our lab notebook: each dot is one experiment, the rising line is the best result so far, the dashed circles are a claim we later retracted (visible on the chart so the correction stays public). Click any dot to read what that experiment tried.
          </p>
          <div className="perf-primer">
            <div className="perf-primer-card">
              <div className="perf-primer-tag mono">how to read</div>
              <div className="perf-primer-body">
                <p>
                  <strong>Throughput vs batch size.</strong> Every decode step loads the model's weights from memory once — and that one read can serve any number of in-flight prompts in parallel. So decoding 12 prompts at once produces about 12× the tokens per second of decoding one at a time, until either memory fills up or compute saturates. The model's speed depends critically on how many prompts run in parallel — that's the batch size.
                </p>
                <p>
                  <strong>Two memory budgets.</strong> The 36 GB working budget leaves headroom for the OS and other apps; 48 GB is the M5 Pro chip's hard ceiling. Bigger batches use more memory, so throughput climbs along the batch-size axis until one of these caps stops us.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Animated chart panel */}
        <div className="perf-chart-panel" onClick={() => setActiveIdx(null)}>
          <div className="perf-tabs" role="tablist">
            {Object.entries(PerfTracks).map(([key, track]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={activeTab === key}
                className={"perf-tab" + (activeTab === key ? " perf-tab-active" : "")}
                onClick={(e) => { e.stopPropagation(); handleTab(key); }}
              >
                <span className="perf-tab-label">{track.label}</span>
                <span className="perf-tab-sub mono">{track.sub}</span>
              </button>
            ))}
            <button
              type="button"
              className="perf-replay mono"
              onClick={(e) => { e.stopPropagation(); replay(); }}
              aria-label="Replay animation"
              title="Replay"
            >
              <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 2.5a5.5 5.5 0 1 0 5.46 6.18.75.75 0 1 0-1.49-.18A4 4 0 1 1 8 4v2L11.5 3 8 0v2.5Z" fill="currentColor"/>
              </svg>
              Replay
            </button>
          </div>
          <div className="perf-chart-frame" onClick={(e) => e.stopPropagation()}>
            <div className="perf-chart-title">
              <div className="perf-chart-title-left">
                <div className="perf-chart-title-eyebrow mono">{t.label}</div>
                <div className="perf-chart-title-line">
                  <span className="perf-chart-title-best mono">running best</span>
                  <span className="perf-chart-title-num mono">{activeTab === "dense" ? "232" : "791.8"}</span>
                  <span className="perf-chart-title-unit">tok/s</span>
                </div>
              </div>
              <div className="perf-chart-title-right">
                <div className="perf-chart-title-vs">
                  <span className="perf-chart-title-vs-label">vs starting point</span>
                  <span className="perf-chart-title-vs-num mono">{activeTab === "dense" ? "5.50×" : "4.20×"}</span>
                </div>
              </div>
            </div>
            <PerfLedger
              trackKey={activeTab}
              animateKey={animateKey}
              activeIdx={activeIdx}
              onSelect={(i) => setActiveIdx(i === activeIdx ? null : i)}
            />
            <PerfLegend />
            <div ref={panelRef}>
              <PerfDetailPanel point={point} trackKey={activeTab} onClose={() => setActiveIdx(null)} />
            </div>
          </div>
        </div>

        {/* Single-user reality — per-row math */}
        <div className="perf-section perf-perrow-section">
          <div className="perf-perrow-card">
            <div className="perf-perrow-eyebrow mono">Single-user reality</div>
            <h3 className="perf-perrow-title">All gains route through batch size. Per-row speed moves the opposite way.</h3>
            <p className="perf-perrow-body">
              Total throughput rises with batch because the per-step weight read amortises across more in-flight prompts. The flip side: each individual prompt receives a smaller share of the chip's bandwidth, so single-prompt speed <em>decreases</em> as batch grows. P-6 optimised aggregate, not single-user.
            </p>
            <div className="perf-perrow-table-wrap">
              <table className="perf-perrow-table mono">
                <thead>
                  <tr>
                    <th>Configuration</th>
                    <th className="num">Aggregate</th>
                    <th className="num">Per row</th>
                    <th>Frame</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>batch 1 (single user)</td>
                    <td className="num">~20 tok/s</td>
                    <td className="num">~20 tok/s</td>
                    <td>bandwidth ceiling, derived</td>
                  </tr>
                  <tr>
                    <td>batch 4 (cycle-1 baseline)</td>
                    <td className="num">42.17 tok/s</td>
                    <td className="num">10.54 tok/s</td>
                    <td>52% bandwidth utilisation</td>
                  </tr>
                  <tr className="perf-perrow-row-best">
                    <td>batch 52 (best within 36 GB)</td>
                    <td className="num">204 tok/s</td>
                    <td className="num">3.92 tok/s</td>
                    <td>strict envelope</td>
                  </tr>
                  <tr className="perf-perrow-row-best">
                    <td>batch 64 (48 GB ceiling)</td>
                    <td className="num">232 tok/s</td>
                    <td className="num">3.62 tok/s</td>
                    <td>hardware cap</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="perf-perrow-foot">
              P-6 was a <em>server-throughput</em> phase: more parallel users on the same chip. Single-user interactive latency &mdash; the silica-chat experience for one person sitting in front of an M5 Pro &mdash; was not its goal. <strong>D-022</strong> (in progress, v1.7.24) is the research line that attacks single-user latency directly: closing dispatch overhead and per-step time at batch &isin; <span className="mono">{"{1, 2, 4, 8, 12}"}</span>.
            </p>
          </div>
        </div>

        {/* Acceptance gates */}
        <div className="perf-section">
          <div className="perf-subhead">The four targets we set &middot; all cleared <span className="perf-subhead-qual">(server-aggregate)</span></div>
          <div className="perf-gates">
            {PerfGates.map((g, i) => (
              <div key={i} className="perf-gate">
                <div className="perf-gate-name mono">{g.gate}</div>
                <div className="perf-gate-row">
                  <div className="perf-gate-target">target {g.target}</div>
                  <div className="perf-gate-mult mono">{g.mult}</div>
                </div>
                <div className="perf-gate-cleared">{g.cleared}</div>
                <div className="perf-gate-frame">{g.frame}</div>
              </div>
            ))}
          </div>
          <div className="perf-gates-note">
            All four cleared by the same composition: a bigger batch size (lever 1) made possible by storing recurrent state in 16-bit floats (lever 2, which by itself doesn't speed anything up but frees ~3.5 GB so the bigger batch fits). The exact same combination ports cleanly to the MoE model. These targets are <strong>batch-aggregate throughput</strong>; per-row decreases with batch &mdash; see the Single-user reality table above. See card 01 below for the mechanics.
          </div>
        </div>

        {/* Behind the numbers — 3 cards, always visible */}
        <div className="perf-section">
          <div className="perf-subhead">Behind the numbers</div>
          <div className="perf-cards">
            {PerfDetails.map(d => (
              <div key={d.id} className="perf-card">
                <div className="perf-card-head">
                  <span className="perf-card-num mono">{d.num}</span>
                  <span className="perf-card-tag mono">{d.tag}</span>
                </div>
                <h3 className="perf-card-title">{d.title}</h3>
                <div className="perf-card-points">
                  {d.points.map((pt, idx) => (
                    <div key={idx} className="perf-card-point">
                      <div className="perf-card-point-head">{pt.head}</div>
                      <div className="perf-card-point-body">{pt.body}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="perf-foot">
          <div className="perf-foot-label mono">Read further</div>
          <div className="perf-foot-cta">
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_NOTES.md" target="_blank" rel="noreferrer">What we learned</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_FINAL_REPORT.md" target="_blank" rel="noreferrer">Full write-up</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_LOG.tsv" target="_blank" rel="noreferrer">Every measurement (raw data)</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/P6_AUTORESEARCH.md" target="_blank" rel="noreferrer">The original brief</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_SMALL_B_OPENING.md" target="_blank" rel="noreferrer">What we work on next</a>
          </div>
        </div>
      </div>

      <style>{`
        /* "How to read" primer */
        .perf-primer {
          margin-top: 28px;
          margin-bottom: 8px;
        }
        .perf-primer-card {
          background: var(--bg-sunken);
          border: 1px solid var(--rule);
          border-radius: var(--radius);
          padding: 20px 24px;
          display: grid;
          grid-template-columns: 110px 1fr;
          gap: 16px;
          align-items: start;
        }
        @media (max-width: 720px) {
          .perf-primer-card {
            grid-template-columns: 1fr;
            gap: 8px;
          }
        }
        .perf-primer-tag {
          font-size: 10px;
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 600;
          padding-top: 2px;
        }
        .perf-primer-body {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
        }
        .perf-primer-body p { margin: 0 0 10px; }
        .perf-primer-body p:last-child { margin-bottom: 0; }
        .perf-primer-body strong { color: var(--ink); font-weight: 600; }

        /* Tabs + chart frame */
        .perf-chart-panel { margin-bottom: 56px; }
        .perf-tabs {
          display: flex;
          gap: 1px;
          background: var(--rule);
          border: 1px solid var(--rule);
          border-bottom: none;
          border-radius: var(--radius) var(--radius) 0 0;
          overflow: hidden;
        }
        @media (max-width: 720px) { .perf-tabs { flex-wrap: wrap; } }
        .perf-tab {
          flex: 1;
          background: var(--bg-elev);
          border: none;
          padding: 16px 20px;
          text-align: left;
          cursor: pointer;
          display: flex;
          flex-direction: column;
          gap: 4px;
          color: var(--ink-3);
          transition: color 120ms, background 120ms;
          min-width: 200px;
        }
        .perf-tab:hover { background: var(--bg-sunken); color: var(--ink); }
        .perf-tab-active {
          background: var(--bg-elev);
          color: var(--ink);
          box-shadow: inset 0 -2px 0 var(--accent);
        }
        .perf-tab-label { font-size: 14px; font-weight: 600; }
        .perf-tab-sub { font-size: 11px; color: var(--ink-3); }
        .perf-tab-active .perf-tab-sub { color: var(--ink-2); }
        .perf-replay {
          background: var(--bg-elev);
          border: none;
          padding: 0 20px;
          cursor: pointer;
          color: var(--ink-3);
          font-size: 12px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          transition: color 120ms, background 120ms;
          min-width: 110px;
          justify-content: center;
        }
        .perf-replay:hover { color: var(--accent); background: var(--bg-sunken); }
        .perf-replay svg { display: block; }

        .perf-chart-frame {
          background:
            radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, var(--accent-soft) 65%, transparent) 0%, transparent 60%),
            linear-gradient(180deg, var(--bg-elev) 0%, color-mix(in srgb, var(--bg-sunken) 30%, var(--bg-elev)) 100%);
          border: 1px solid var(--rule);
          border-radius: 0 0 var(--radius) var(--radius);
          padding: 4px 28px 0;
          box-shadow: var(--shadow-sm);
        }

        /* Chart title strip */
        .perf-chart-title {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 24px;
          padding: 24px 4px 18px;
          border-bottom: 1px solid var(--rule-2);
          margin-bottom: 8px;
        }
        .perf-chart-title-eyebrow {
          font-size: 10px;
          color: var(--ink-3);
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .perf-chart-title-line {
          display: flex;
          align-items: baseline;
          gap: 10px;
          font-feature-settings: "tnum" 1, "ss01" 1;
        }
        .perf-chart-title-best {
          font-size: 11px;
          color: var(--ink-4);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 600;
        }
        .perf-chart-title-num {
          font-size: 38px;
          font-weight: 700;
          color: var(--ink);
          letter-spacing: -0.03em;
          line-height: 1;
          font-feature-settings: "tnum" 1, "ss01" 1;
        }
        .perf-chart-title-unit {
          font-size: 16px;
          color: var(--ink-3);
          font-weight: 500;
          letter-spacing: -0.01em;
        }
        .perf-chart-title-vs {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 4px;
        }
        .perf-chart-title-vs-label {
          font-size: 10px;
          color: var(--ink-4);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 600;
        }
        .perf-chart-title-vs-num {
          font-size: 22px;
          font-weight: 700;
          color: var(--ok);
          letter-spacing: -0.02em;
          font-feature-settings: "tnum" 1;
        }
        @media (max-width: 720px) {
          .perf-chart-title { flex-direction: column; align-items: flex-start; gap: 8px; }
          .perf-chart-title-vs { align-items: flex-start; }
          .perf-chart-title-num { font-size: 30px; }
          .perf-chart-title-vs-num { font-size: 18px; }
        }

        /* SVG */
        .perf-svg {
          width: 100%;
          height: auto;
          display: block;
          font-family: var(--font-sans);
          overflow: visible;
        }
        .perf-svg-grid { stroke: var(--rule-2); stroke-width: 0.75; opacity: 0.7; }
        .perf-svg-tick {
          fill: var(--ink-3);
          font-size: 11px;
          font-weight: 500;
          font-feature-settings: "tnum" 1;
        }
        .perf-svg-axis-title {
          fill: var(--ink-4);
          font-size: 10px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          font-weight: 500;
        }
        .perf-svg-baseline {
          stroke: var(--ink-4);
          stroke-width: 0.75;
          stroke-dasharray: 3 4;
          opacity: 0.45;
        }
        .perf-svg-baseline-label { fill: var(--ink-3); font-size: 10px; font-weight: 500; }
        .perf-svg-best {
          fill: none;
          stroke: var(--accent);
          stroke-width: 1.6;
          stroke-linejoin: round;
          stroke-linecap: round;
          transition: stroke-dashoffset 60ms linear;
          filter: drop-shadow(0 0.5px 1.5px color-mix(in srgb, var(--accent) 22%, transparent));
        }
        .perf-svg-area {
          transition: opacity 320ms ease-out;
        }
        .perf-svg-retract {
          fill: none;
          stroke: var(--ink-3);
          stroke-width: 1;
          stroke-dasharray: 3 3;
          stroke-linejoin: round;
          stroke-linecap: round;
          opacity: 0.45;
          transition: opacity 360ms ease-out;
        }
        .perf-svg-dot-g {
          opacity: 0;
          transform-origin: center;
          transition: opacity 240ms ease, transform 240ms ease;
          cursor: pointer;
        }
        .perf-svg-dot-g.perf-svg-dot-on {
          opacity: 1;
        }
        .perf-svg-dot {
          transition: r 200ms ease;
        }
        .perf-svg-dot-keep { fill: var(--accent); }
        .perf-svg-dot-diag { fill: var(--ink-3); opacity: 0.5; }
        .perf-svg-dot-discard { fill: var(--ink-4); opacity: 0.32; }
        .perf-svg-dot-retracted {
          fill: var(--bg-elev);
          stroke: var(--ink-3);
          stroke-width: 1.2;
          stroke-dasharray: 2 2;
          opacity: 0.85;
        }
        .perf-svg-dot-ring {
          stroke: var(--accent);
          stroke-width: 1;
          opacity: 0;
          animation: perfRing 1.8s infinite ease-out;
        }
        @keyframes perfRing {
          0% { opacity: 0.45; r: 7; }
          100% { opacity: 0; r: 14; }
        }
        .perf-svg-anno-g {
          opacity: 0;
          animation: perfFadeIn 360ms 200ms forwards ease-out;
        }
        @keyframes perfFadeIn {
          from { opacity: 0; transform: translateY(2px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .perf-svg-anno-tick {
          stroke: var(--accent);
          stroke-width: 0.75;
          opacity: 0.45;
        }
        .perf-svg-anno-cycle {
          fill: var(--accent);
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.04em;
        }
        .perf-svg-anno-label {
          fill: var(--ink-3);
          font-size: 10px;
          font-weight: 500;
          letter-spacing: 0.01em;
        }

        /* Legend */
        .perf-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 18px;
          padding: 14px 0 12px;
          font-size: 11px;
          color: var(--ink-3);
          border-top: 1px solid var(--rule-2);
          margin-top: 8px;
        }
        .perf-legend-item {
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }
        .perf-legend-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          display: inline-block;
        }
        .perf-legend-dot.perf-svg-dot-keep { background: var(--accent); }
        .perf-legend-dot.perf-svg-dot-diag { background: var(--ink-3); opacity: 0.55; }
        .perf-legend-dot.perf-svg-dot-discard { background: var(--ink-4); opacity: 0.6; }
        .perf-legend-dot.perf-svg-dot-retracted {
          background: var(--bg-elev);
          border: 1.6px dashed var(--ink-3);
        }
        .perf-legend-line {
          width: 18px;
          height: 2px;
          background: var(--accent);
          border-radius: 2px;
        }

        /* Detail panel */
        .perf-detail-panel {
          padding: 18px 22px 22px;
          border-top: 1px solid var(--rule-2);
          margin-top: 4px;
          min-height: 56px;
          position: relative;
          transition: background 200ms ease;
        }
        .perf-detail-panel-empty {
          background: transparent;
          color: var(--ink-3);
          padding: 22px;
        }
        .perf-detail-panel-hint { font-size: 12px; }
        .perf-detail-panel-active {
          background: var(--bg-sunken);
          border-radius: 0 0 var(--radius) var(--radius);
          margin: 4px -24px 0;
          padding: 22px 24px 24px;
          border-top: 1px solid var(--rule);
          animation: perfFadeIn 220ms ease-out;
        }
        .perf-detail-panel-close {
          position: absolute;
          top: 14px;
          right: 18px;
          background: transparent;
          border: none;
          color: var(--ink-3);
          font-size: 22px;
          cursor: pointer;
          line-height: 1;
          padding: 0 6px;
        }
        .perf-detail-panel-close:hover { color: var(--ink); }
        .perf-detail-panel-meta {
          display: flex;
          gap: 12px;
          align-items: center;
          margin-bottom: 6px;
          font-size: 11px;
          color: var(--ink-3);
        }
        .perf-detail-panel-seq { color: var(--ink-3); }
        .perf-detail-panel-cycle {
          color: var(--accent);
          font-weight: 600;
          letter-spacing: 0.02em;
        }
        .perf-detail-panel-status {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 10px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          font-weight: 600;
        }
        .perf-status-keep { background: var(--accent-soft); color: var(--accent-2); }
        .perf-status-diag { background: var(--bg-elev); color: var(--ink-3); border: 1px solid var(--rule); }
        .perf-status-discard { background: var(--bg-elev); color: var(--ink-4); border: 1px solid var(--rule); }
        .perf-status-retracted {
          background: var(--bg-elev);
          color: var(--warn);
          border: 1px dashed var(--warn);
        }
        .perf-detail-panel-tok {
          font-size: 32px;
          font-weight: 700;
          color: var(--ink);
          letter-spacing: -0.028em;
          line-height: 1.05;
          margin-bottom: 6px;
          font-feature-settings: "tnum" 1, "ss01" 1;
        }
        .perf-detail-panel-unit {
          font-size: 14px;
          font-weight: 500;
          color: var(--ink-3);
        }
        .perf-detail-panel-label {
          font-size: 14px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 8px;
        }
        .perf-detail-panel-body {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
          max-width: 800px;
        }

        /* Sections */
        .perf-section { margin-bottom: 48px; }
        .perf-section:last-child { margin-bottom: 0; }
        .perf-subhead {
          font-size: 11px;
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 600;
          margin-bottom: 18px;
        }
        .perf-subhead-qual {
          font-family: var(--font-mono);
          font-size: 10.5px;
          color: var(--ink-3);
          font-weight: 500;
          letter-spacing: 0.02em;
          text-transform: none;
          margin-left: 6px;
        }

        /* Single-user reality callout */
        .perf-perrow-section { margin-top: 16px; }
        .perf-perrow-card {
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: var(--radius);
          padding: 28px 30px 26px;
          position: relative;
          overflow: hidden;
        }
        .perf-perrow-card::before {
          content: "";
          position: absolute; left: 0; top: 0; bottom: 0;
          width: 3px;
          background: linear-gradient(180deg, var(--warn, #b58a00), color-mix(in srgb, var(--warn, #b58a00) 40%, transparent));
          opacity: 0.85;
        }
        .perf-perrow-eyebrow {
          font-size: 11px;
          color: var(--warn, #b58a00);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .perf-perrow-title {
          font-size: 19px;
          letter-spacing: -0.018em;
          font-weight: 600;
          line-height: 1.3;
          color: var(--ink);
          margin: 0 0 12px;
        }
        .perf-perrow-body {
          font-size: 14px;
          color: var(--ink-2);
          line-height: 1.6;
          margin: 0 0 18px;
        }
        .perf-perrow-table-wrap { overflow-x: auto; }
        .perf-perrow-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
          font-feature-settings: "tnum" 1, "ss01" 1;
        }
        .perf-perrow-table th,
        .perf-perrow-table td {
          padding: 11px 14px;
          border-bottom: 1px solid var(--rule);
          text-align: left;
          color: var(--ink-2);
          font-weight: 400;
        }
        .perf-perrow-table th {
          font-size: 11px;
          color: var(--ink-3);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
          border-bottom: 1px solid var(--rule);
        }
        .perf-perrow-table th.num,
        .perf-perrow-table td.num {
          text-align: right;
          color: var(--ink);
          font-weight: 500;
        }
        .perf-perrow-table tbody tr:last-child td { border-bottom: none; }
        .perf-perrow-row-best td { color: var(--ink); }
        .perf-perrow-row-best td.num { color: var(--accent); font-weight: 600; }
        .perf-perrow-foot {
          margin: 18px 0 0;
          padding: 14px 16px;
          background: var(--bg-sunken);
          border: 1px solid var(--rule);
          border-radius: var(--radius-sm);
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
        }
        @media (max-width: 720px) {
          .perf-perrow-card { padding: 22px 20px 20px; }
          .perf-perrow-title { font-size: 17px; }
          .perf-perrow-table th,
          .perf-perrow-table td { padding: 10px 8px; font-size: 12px; }
        }

        /* Gates */
        .perf-gates {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        @media (max-width: 920px) { .perf-gates { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 540px) { .perf-gates { grid-template-columns: 1fr; } }
        .perf-gate {
          padding: 22px 22px 20px;
          background: var(--bg-elev);
        }
        .perf-gate-name { font-size: 12px; color: var(--ink-3); margin-bottom: 12px; }
        .perf-gate-row {
          display: flex; align-items: baseline; justify-content: space-between;
          margin-bottom: 8px;
        }
        .perf-gate-target { font-size: 13px; color: var(--ink-3); }
        .perf-gate-mult {
          font-size: 22px; font-weight: 700; color: var(--ok);
          letter-spacing: -0.02em;
          font-feature-settings: "tnum" 1;
        }
        .perf-gate-cleared {
          font-size: 16px; font-weight: 600; color: var(--ink);
          margin-bottom: 6px;
        }
        .perf-gate-frame { font-size: 12px; color: var(--ink-3); line-height: 1.5; }
        .perf-gates-note {
          margin-top: 14px;
          padding: 14px 18px;
          background: var(--bg-sunken);
          border: 1px solid var(--rule);
          border-radius: var(--radius-sm);
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
        }

        /* Behind-the-numbers cards */
        .perf-cards {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
        }
        @media (max-width: 920px) { .perf-cards { grid-template-columns: 1fr; } }
        .perf-card {
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: var(--radius);
          padding: 26px 24px 28px;
          display: flex;
          flex-direction: column;
          transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
        }
        .perf-card:hover {
          border-color: var(--rule);
          transform: translateY(-1px);
          box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px -12px rgba(0,0,0,0.08), 0 0 0 1px var(--rule);
        }
        .perf-card-head {
          display: flex;
          align-items: baseline;
          gap: 12px;
          margin-bottom: 14px;
        }
        .perf-card-num {
          font-size: 22px;
          font-weight: 600;
          color: var(--accent);
          letter-spacing: -0.02em;
          line-height: 1;
          font-feature-settings: "tnum" 1;
        }
        .perf-card-tag {
          font-size: 10px;
          color: var(--ink-4);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 600;
        }
        .perf-card-title {
          font-size: 17px;
          font-weight: 600;
          color: var(--ink);
          letter-spacing: -0.012em;
          line-height: 1.3;
          margin: 0 0 18px;
        }
        .perf-card-points {
          display: flex;
          flex-direction: column;
          gap: 14px;
          flex: 1;
        }
        .perf-card-point {
          padding-top: 14px;
          border-top: 1px solid var(--rule-2);
        }
        .perf-card-point:first-child {
          padding-top: 0;
          border-top: none;
        }
        .perf-card-point-head {
          font-size: 12px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 6px;
          letter-spacing: -0.005em;
        }
        .perf-card-point-body {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
        }

        /* Foot */
        .perf-foot {
          margin-top: 36px;
          padding: 22px 24px;
          background: var(--bg-sunken);
          border-radius: var(--radius);
          border: 1px solid var(--rule);
        }
        .perf-foot-label {
          font-size: 10px;
          color: var(--ink-4);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-weight: 600;
          margin-bottom: 12px;
        }
        .perf-foot-cta { display: flex; flex-wrap: wrap; gap: 8px; }
        .perf-foot-cta .btn { padding: 7px 14px; font-size: 13px; }
      `}</style>
    </section>
  );
};

window.Performance = Performance;
