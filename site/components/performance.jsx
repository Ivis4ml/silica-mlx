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
    "Where we begin. Generating tokens for 4 simultaneous requests on Qwen3.5-27B (a 27-billion-parameter model) reaches 42 tokens per second per request — about 52% of what the M5 Pro chip's memory bandwidth ought to allow. Every later number is compared against this one."],
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
    "First step of the batch sweep: 8 simultaneous requests. Just above baseline."],
  [11, 63.1, "keep", null, null,
    "First win: 12 requests at once = 63 tok/s. The first time we crossed the 60 tok/s stretch goal — without writing any new code. Just a different batch size."],
  [12, 81.1, "diag", null, null,
    "16 requests at once. Throughput keeps climbing cleanly."],
  [13, 112.8, "diag", null, null,
    "24 requests at once — about 2.7× the starting point already."],
  [14, 150.6, "keep", null, null,
    "32 requests: 150 tok/s, 3.6× the start. Memory peak ~26 GB, well within budget."],
  [15, 171.6, "keep", null, null,
    "40 requests: 172 tok/s. About 4× the start."],
  [16, 183.3, "keep", null, null,
    "44 requests: 183 tok/s. Approaching the strict 36 GB memory budget."],
  [17, 193.9, "keep", "C10", "axis-shift @ B=48",
    "The breakthrough. Batching 48 requests together = 4.6× the starting throughput, with no kernel change at all. The trick was just re-reading our own goal: \"maximize total tokens per second across the batch\", not per individual request. Memory peak 34 GB, still inside the 36 GB budget. Nine prior cycles of GPU-kernel hacking had moved nothing — this single parameter choice did."],
  [18, 193.3, "diag", null, null,
    "Tested a custom GPU attention kernel at this batch size. Wins on a microbenchmark, but flat at the system level — attention is only a small slice of total step time at high batch."],
  [19, 192.5, "diag", null, null,
    "Tried storing the recurrent state in 16-bit floats instead of 32-bit. Same speed at this batch, but frees ~3.5 GB of memory — the seed for the next breakthrough."],
  [20, 200.8, "keep", "C13", "the composition win",
    "Composition. The 16-bit memory save from cycle 12 doesn't speed anything up by itself, but it frees enough headroom to push from 48 to 52 requests in batch — and that bump pushes throughput past 200 tok/s, still inside the 36 GB budget. Two cheap parameter changes beat every kernel attempt."],
  [21, 212.2, "diag", null, null,
    "56 requests at once. Past the strict 36 GB budget but inside the 48 GB chip-memory cap."],
  [22, 219.1, "diag", null, null,
    "60 requests."],
  [23, 229.8, "keep", null, null,
    "64 requests: 230 tok/s. About 5.5× the starting point — pushing right against the 48 GB chip memory limit."],
  [24, 166.8, "discard", null, null,
    "Tried 66 requests. Throughput collapsed 26%. We hit a hardware cliff at the 40 GB memory peak — beyond it, the chip stops scaling. Three allocator settings tried; the cliff is in the chip itself, not our code."],
  [25, 169.1, "discard", null, null,
    "68 requests — also past the cliff."],
  [26, 173.2, "discard", null, null,
    "72 requests — confirming the pattern. Dense 27B can't benefit from larger batches on this chip."],
  [27, 206.2, "retracted", "C14→C27", "retracted",
    "A retracted result. Cycle 14 claimed a 5-tok/s gain at batch=52 from a custom GPU attention kernel. Two weeks later a code review caught a bug — the code checked for the wrong floating-point format and the kernel was silently being skipped on the production model. After the fix, the kernel's real contribution measured to within noise (~0.5 tok/s, statistically zero). The honest credit goes to cycles 10 and 12; we kept this dot on the chart so the retraction stays visible."],
  [28, 232.2, "retracted", "C14→C27", "retracted",
    "Same retracted experiment at batch=64. Cycle 14 reported 232 tok/s with the custom kernel; the code-review fix showed it was actually slightly slower than the simpler version (still within noise). The real ceiling result comes two indices later, at 231.9."],
  [29, 197.05, "diag", null, null,
    "First re-check after the code-review fix. Within the cycle-13 range; no new claim."],
  [30, 200.14, "diag", null, null,
    "Re-check under one Python environment — back near cycle-13."],
  [31, 185.30, "diag", null, null,
    "Re-check under a different Python environment — about 10% lower. Identified as between-environment drift, not a regression."],
  [32, 231.9, "keep", "C28", "honest ceiling",
    "The honest hardware-ceiling result: 232 tok/s at batch=64, measured 3 times with tight agreement (±0.3 tok/s). About 5.5× the starting point, just inside the 48 GB chip memory limit. This number replaces the cycle-14 retracted claim."],
  [33, 204.0, "keep", "C33", "tightened envelope",
    "Final tighten. Re-measured at batch=52 across 6 runs in two sessions: 204 ± 1 tok/s. This is the running-best within the strict 36 GB memory budget. The variance protocol used here (multiple sessions, combined error check) is now the standard for any future claim."],
];

const PerfMoePoints = [
  [1, 188.5, "keep", "C1", "starting point",
    "Same starting line for the mixture-of-experts version. 188 tok/s for 4 simultaneous requests on Qwen3.5-35B-A3B (a 35-billion-parameter MoE model — larger total weights but only 8 of 256 experts active per token). The original ≥100 tok/s goal was already satisfied here at the baseline."],
  [2, 181.4, "diag", null, null,
    "Tried the 16-bit recurrent state on MoE at 4 requests. Slightly below baseline — 4 requests amortizes poorly on this architecture."],
  [3, 242.0, "diag", null, null,
    "8 requests. Climbing."],
  [4, 306.3, "diag", null, null,
    "16 requests."],
  [5, 384.8, "diag", null, null,
    "32 requests."],
  [6, 434.3, "diag", null, null,
    "48 requests."],
  [7, 464.4, "keep", "C34", "envelope KEEP @ B=64",
    "MoE win within the strict memory budget. 464 tok/s at 64 simultaneous requests, peak 33.8 GB. Same parameter changes from cycles 10 and 12 ported over to MoE via a shared memory hook. About 2.5× the MoE starting point."],
  [8, 447.9, "discard", null, null,
    "80 requests — past the strict budget, throughput dips."],
  [9, 444.5, "discard", null, null,
    "72 requests."],
  [10, 467.1, "diag", null, null,
    "96 requests — climbing again as we sweep toward the chip-memory limit."],
  [11, 791.8, "keep", "C35", "the biggest result",
    "The biggest result of the entire 35-cycle effort. 792 tok/s at 128 simultaneous requests on the MoE model, sitting almost exactly at the 48 GB chip memory limit (peak 47.96 GB), measured 3 times. About 4.2× the MoE starting point. Unlike dense 27B, the MoE architecture doesn't hit a memory cliff at this size — only 8 of 256 experts are active per token, so each token's active-weight footprint is much smaller."],
];

const PerfTracks = {
  dense: {
    label: "Dense Qwen3.5-27B-4bit",
    sub: "38 measurements · 9 KEEPs · running best 232 tok/s",
    points: PerfDensePoints,
    yMax: 260,
    yTicks: [0, 50, 100, 150, 200, 250],
    baseline: 42.17,
    baselineLabel: "cycle-1 baseline 42.17",
  },
  moe: {
    label: "MoE Qwen3.5-35B-A3B-4bit",
    sub: "11 measurements · 2 KEEPs · running best 791.8 tok/s",
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

  // Draw progress: stroke-dashoffset interpolation reveals the line
  // from left to right.
  const dashOffset = lineLength * (1 - progress);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="perf-svg"
      role="img"
      aria-label={`${t.label} — Karpathy autoresearch ledger`}
    >
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
        <span className={"perf-detail-panel-status mono perf-status-" + status}>{status}</span>
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
  { gate: "(1a) Dense engineering", target: "≥40 tok/s", cleared: "204 ± 1 tok/s", mult: "4.85×", frame: "B=52 · 36 GB envelope" },
  { gate: "(1b) Dense stretch", target: "≥60 tok/s", cleared: "231.9 ± 0.3 tok/s", mult: "3.87×", frame: "B=64 · 48 GB hardware ceiling" },
  { gate: "(2a) MoE anchor", target: "≥100 tok/s", cleared: "120.93 tok/s", mult: "preserved", frame: "v1.7.13 baseline · MoE B=2" },
  { gate: "(2b) MoE stretch", target: "≥175 tok/s", cleared: "791.8 ± 5.2 tok/s", mult: "4.52×", frame: "MoE B=128 · 48 GB ceiling" },
];

const PerfDetails = [
  {
    id: "levers",
    title: "Two load-bearing levers",
    tag: "How",
    body: (
      <>
        <p>The running-best is composition, not a custom kernel.</p>
        <ul>
          <li><strong>Cycle 10 — batched-aggregate axis-shift.</strong> Re-reading the AR.md metric definition moved the operating point B=4 → B=48 within the 36 GB envelope. Pure parameter selection. <em>4.60× on its own.</em></li>
          <li><strong>Cycle 12 — bf16 DeltaNet recurrent state.</strong> State shape <span className="mono">[B, Hv=48, Dv=128, Dk=128]</span> is 144 MB at fp32 per layer, 72 MB at bf16. ~3.5 GB peak save. Opens B≥48 within envelope, B=64 at ceiling.</li>
          <li><strong>Cycle 13 — composition.</strong> Cycle-12's peak save composed with cycle-10's B-axis lever produces the running-best line. The two levers are independent; together they dominate every later atomic probe.</li>
        </ul>
        <p className="perf-callout">
          <strong>17 custom Metal kernel attempts closed without a load-bearing E2E win.</strong> Cycle 30 explained why: at B=64 with the v10+bf16 stack, DeltaNet owns 88% of step time, full-attn 12.5%, dispatch 0.3% — and mlx's existing <span className="mono">gated_delta</span> kernel is already at HBM-bandwidth limit (cycle-31 silica <span className="mono">gated_delta_v2</span> = 1.001× vs mlx). Source-string Metal kernels in mlx 0.31.x do not pay back on dense 27B.
        </p>
      </>
    ),
  },
  {
    id: "honest",
    title: "Honest record — closures and a retraction",
    tag: "What didn't work",
    body: (
      <>
        <ul>
          <li><strong>Spec-decode at production B — closed with measurement-anchored negative.</strong> Cycle 23 measured the B × k verify-cost matrix: B=52 k=64 = 8105 ms vs same-B plain decode ~252 ms / step. Tree-spec recomputes to ~10 tok/s aggregate, a 20× regression vs plain. Track C settles: C.4 retired (η.1 = 0.482×), C.5 retired (cycle-23 closure), C.1/C.2/C.3/C.6 deprioritised.</li>
          <li><strong>Dense B-axis past 64 — closed at the architectural cliff.</strong> Cycles 28-29 measured a 26% drop at B=64 → B=66 (40 GB peak boundary). Three allocator-hint probes leave the cliff in place — architectural, not allocator policy.</li>
          <li><strong>Cycle 14's claimed v10 KEEP — retracted via codex review.</strong> A 14-cycle dtype-defect in <span className="mono">shadow_install</span> silently skipped the bf16 production path. After the fix, cycles 27/28 measured v10's E2E at +0.5 tok/s @ B=52 / −1.7 tok/s @ B=64 — both within noise. The retracted KEEPs are the dashed circles in the chart; click them to read the full story.</li>
        </ul>
      </>
    ),
  },
  {
    id: "method",
    title: "Methodology",
    tag: "How we measured",
    body: (
      <>
        <p>
          Karpathy-style autoresearch ledger (one TSV row per measurement; the main agent appends, sub-agents return findings). Variance discipline: ≥3 reps per session, ≥2 sessions, combined σ check before declaring a KEEP. Cycle 33's combined σ at B=52 across 2 sessions tightened to 0.83 tok/s on n=6 — the protocol standard, not the exception.
        </p>
        <p>
          Toolchain pin: <span className="mono">mlx==0.31.1</span>, <span className="mono">mlx-lm==0.31.2</span>, <span className="mono">mlx-metal==0.31.1</span>. Determinism gate: <span className="mono">tests/test_p2_preload_parity.py</span> (3/3 pass). The cycle-27 retraction reinforced a process rule: small-n within-session σ underestimates run-to-run variance, so an n=3 KEEP is provisional until a second session confirms it.
        </p>
      </>
    ),
  },
];

const PerfLegend = () => (
  <div className="perf-legend mono">
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-keep"></span>
      KEEP
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-diag"></span>
      diagnostic
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-discard"></span>
      discard
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-dot perf-svg-dot-retracted"></span>
      retracted (cycle 27)
    </span>
    <span className="perf-legend-item">
      <span className="perf-legend-line"></span>
      running best
    </span>
  </div>
);

const Performance = () => {
  const [activeTab, setActiveTab] = React.useState("dense");
  const [activeIdx, setActiveIdx] = React.useState(null);
  const [openDetail, setOpenDetail] = React.useState(null);
  const [animateKey, setAnimateKey] = React.useState(0);
  const [hasAnimated, setHasAnimated] = React.useState(false);
  const sectionRef = React.useRef(null);

  const t = PerfTracks[activeTab];
  const point = activeIdx != null ? t.points[activeIdx] : null;

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
          <div className="section-eyebrow">P-6 autoresearch · v1.7.23</div>
          <h2>35 cycles. Two levers. Every gate cleared 3.4-5.5×.</h2>
          <p>
            The opus autoresearch loop pushed Qwen3.5-27B-4bit warm decode 5.50× over the cycle-1 baseline on M5 Pro 48 GB; MoE Qwen3.5-35B-A3B-4bit hit 791.8 tok/s at the 48 GB hardware ceiling. The chart below replays the ledger — each dot a measurement, the ladder the running best, the dashed circles cycle 14's retracted KEEPs. Click any dot to read its experiment record.
          </p>
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
            <PerfLedger
              trackKey={activeTab}
              animateKey={animateKey}
              activeIdx={activeIdx}
              onSelect={(i) => setActiveIdx(i === activeIdx ? null : i)}
            />
            <PerfLegend />
            <PerfDetailPanel point={point} trackKey={activeTab} onClose={() => setActiveIdx(null)} />
          </div>
        </div>

        {/* Acceptance gates */}
        <div className="perf-section">
          <div className="perf-subhead">Acceptance gates · every P-6 gate cleared</div>
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
        </div>

        {/* Click-to-expand details */}
        <div className="perf-section">
          <div className="perf-subhead">Behind the numbers · click to expand</div>
          <div className="perf-details">
            {PerfDetails.map(d => {
              const open = openDetail === d.id;
              return (
                <div key={d.id} className={"perf-detail" + (open ? " perf-detail-open" : "")}>
                  <button
                    type="button"
                    className="perf-detail-head"
                    onClick={() => setOpenDetail(open ? null : d.id)}
                    aria-expanded={open}
                  >
                    <span className="perf-detail-tag mono">{d.tag}</span>
                    <span className="perf-detail-title">{d.title}</span>
                    <span className="perf-detail-chev mono">{open ? "−" : "+"}</span>
                  </button>
                  {open && <div className="perf-detail-body">{d.body}</div>}
                </div>
              );
            })}
          </div>
        </div>

        <div className="perf-foot">
          <div className="perf-foot-cta">
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_NOTES.md" target="_blank" rel="noreferrer">Take-home notes</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_FINAL_REPORT.md" target="_blank" rel="noreferrer">23-cycle final report</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_LOG.tsv" target="_blank" rel="noreferrer">Karpathy ledger (TSV)</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/AR.md" target="_blank" rel="noreferrer">AR.md directive</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_SMALL_B_OPENING.md" target="_blank" rel="noreferrer">D-022 next line</a>
          </div>
        </div>
      </div>

      <style>{`
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
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: 0 0 var(--radius) var(--radius);
          padding: 24px 24px 0;
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
        .perf-svg-tick { fill: var(--ink-3); font-size: 11px; font-weight: 500; }
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
          filter: drop-shadow(0 0.5px 1px var(--accent-soft));
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
          font-size: 30px;
          font-weight: 700;
          color: var(--ink);
          letter-spacing: -0.02em;
          line-height: 1.1;
          margin-bottom: 6px;
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
        }
        .perf-gate-cleared {
          font-size: 16px; font-weight: 600; color: var(--ink);
          margin-bottom: 6px;
        }
        .perf-gate-frame { font-size: 12px; color: var(--ink-3); line-height: 1.5; }

        /* Details accordion */
        .perf-details {
          display: flex;
          flex-direction: column;
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        .perf-detail { background: var(--bg-elev); }
        .perf-detail-head {
          all: unset;
          width: 100%;
          padding: 18px 22px;
          cursor: pointer;
          display: grid;
          grid-template-columns: 130px 1fr 24px;
          gap: 14px;
          align-items: center;
          transition: background 120ms;
        }
        .perf-detail-head:hover { background: var(--bg-sunken); }
        .perf-detail-tag {
          font-size: 11px;
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
        }
        .perf-detail-title {
          font-size: 15px;
          font-weight: 600;
          color: var(--ink);
        }
        .perf-detail-chev {
          font-size: 18px;
          color: var(--ink-3);
          font-weight: 600;
          text-align: right;
        }
        .perf-detail-body {
          padding: 0 22px 22px;
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.65;
        }
        .perf-detail-body p { margin: 0 0 12px; }
        .perf-detail-body ul { margin: 0; padding-left: 22px; }
        .perf-detail-body li { margin-bottom: 8px; }
        .perf-detail-body strong { color: var(--ink); font-weight: 600; }
        .perf-detail-body em { color: var(--accent); font-style: normal; font-weight: 600; }
        .perf-callout {
          margin-top: 14px !important;
          padding: 14px 18px;
          background: var(--bg-sunken);
          border-radius: var(--radius-sm);
          border: 1px solid var(--rule);
          font-size: 13px;
          color: var(--ink-2);
        }
        @media (max-width: 720px) {
          .perf-detail-head { grid-template-columns: 100px 1fr 20px; }
        }

        /* Foot */
        .perf-foot {
          margin-top: 36px;
          padding: 18px 22px;
          background: var(--bg-sunken);
          border-radius: var(--radius);
          border: 1px solid var(--rule);
        }
        .perf-foot-cta { display: flex; flex-wrap: wrap; gap: 8px; }
        .perf-foot-cta .btn { padding: 6px 12px; font-size: 12px; }
      `}</style>
    </section>
  );
};

window.Performance = Performance;
