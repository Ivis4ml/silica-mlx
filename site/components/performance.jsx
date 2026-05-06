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
  [1, 42.17, "keep", "C1", "baseline 42.17 tok/s",
    "Dense 27B B=4 warm decode at 52% bandwidth utilisation. Per-step decomposition: DeltaNet 74% / full-attn 22% / overhead 4%. The denominator for every later uplift number."],
  [2, 6.54, "discard", null, null,
    "C5 DDTree β.1 spec-on row. 0.408× speedup vs plain decode."],
  [3, 7.74, "discard", null, null,
    "C4 DFlash η.1 — drafter accept rate 0.088 with verify_cost ≪ draft_cost; 0.482× speedup. Track C.4 retired."],
  [4, 17.10, "discard", null, null,
    "Cycle 4 lazy-chain / chunked-decode probe. Fails the warm-decode oracle stability gate."],
  [5, 39.99, "diag", null, null,
    "Cycle 1 e2e baseline at D=128 head dim, no kernel changes."],
  [6, 41.00, "discard", null, null,
    "AR_GREEDY_SKIP_HISTORY — sampler-side history-array skip. Below baseline."],
  [7, 42.39, "discard", null, null,
    "AR_E2E_GATED_OUTPUT_D32 — first fused gated-output kernel attempt at D=32. Discard."],
  [8, 42.53, "discard", null, null,
    "AR_E2E_GATED_SILU_D32 — fused gated-silu kernel attempt. Discard."],
  [9, 42.68, "diag", null, null,
    "AR_E2E_BASELINE_D32 — production-shape e2e baseline at D=32 head dim."],
  [10, 43.0, "diag", null, null,
    "AR_BATCHED_AGG_B8 — first axis-shift probe. B=8 = ~43 tok/s; below the 60 bar."],
  [11, 63.1, "keep", null, null,
    "AR_BATCHED_AGG_B12 — first KEEP. B=12 = 63.1 tok/s, crosses the (1b) ≥60 stretch bar at the first non-baseline B."],
  [12, 81.1, "diag", null, null,
    "AR_BATCHED_AGG_B16. Aggregate keeps climbing cleanly."],
  [13, 112.8, "diag", null, null,
    "AR_BATCHED_AGG_B24. ~2.7× the cycle-1 baseline."],
  [14, 150.6, "keep", null, null,
    "AR_BATCHED_AGG_B32. KEEP at 3.6× cycle-1; peak ~26 GB, well within envelope."],
  [15, 171.6, "keep", null, null,
    "AR_BATCHED_AGG_B40. 4.07× cycle-1."],
  [16, 183.3, "keep", null, null,
    "AR_BATCHED_AGG_B44. 4.35× cycle-1; approaching the 36 GB envelope ceiling."],
  [17, 193.9, "keep", "C10", "axis-shift @ B=48",
    "AR_BATCHED_AGG_B48 — the cycle-10 breakthrough. 4.60× cycle-1 with no kernel change. Peak 33.95 GB. The single biggest leverage event in the loop, produced by re-reading the AR.md metric definition (\"B is chosen to maximise aggregate\")."],
  [18, 193.3, "diag", null, null,
    "AR_FA_V10_E2E_B48 — cycle-11 v10 FA-decode kernel at B=48. Microbench wins 1.25-1.81× over mlx SDPA but E2E 0% at this B. (Composition story: cycle-30 attribution shows DeltaNet at 88% step time when high-B; attention is small.)"],
  [19, 192.5, "diag", null, null,
    "AR_BF16_DELTANET_STATE_E2E_B48 — cycle-12 bf16 state at B=48. 0% E2E directly, but produces 3.5 GB peak save that opens B≥48."],
  [20, 200.8, "keep", "C13", "envelope KEEP @ B=52",
    "AR_BF16_AGG_B52 — cycle-13 composition KEEP. C12's peak save composed with C10's B-axis lever pushes B from 48 → 52. 18σ above C10. The 193 wall was a B=48 cap, not a hardware wall."],
  [21, 212.2, "diag", null, null,
    "AR_BF16_AGG_B56."],
  [22, 219.1, "diag", null, null,
    "AR_BF16_AGG_B60."],
  [23, 229.8, "keep", null, null,
    "AR_BF16_AGG_B64 — within the 48 GB hardware ceiling. 5.45× cycle-1."],
  [24, 166.8, "discard", null, null,
    "AR_BF16_AGG_B66 — across the architectural cliff at the 40 GB peak boundary. 26% drop vs B=64."],
  [25, 169.1, "discard", null, null,
    "AR_BF16_AGG_B68 — also past the cliff."],
  [26, 173.2, "discard", null, null,
    "AR_BF16_AGG_B72. Cycles 28-29 confirm the cliff is architectural — three allocator-hint probes (mx.metal.set_cache_limit / set_memory_limit / set_wired_limit) leave it in place."],
  [27, 206.2, "retracted", "C14→C27", "retracted",
    "Cycle 14 claimed v10+bf16 stack at B=52 = 206.2 tok/s as a +5.4 tok/s = 3.4σ KEEP over cycle-13's 200.8. Codex review on opus-codex caught the 14-cycle dtype defect: shadow_install checked queries.dtype == mx.float16 but the production path is bf16 — v10 was never firing. After the bf16-native fix, cycles 27/28 reverify measured v10's E2E contribution at +0.5 tok/s @ B=52, within noise. This KEEP is retracted; the dot is preserved as part of the research record."],
  [28, 232.2, "retracted", "C14→C27", "retracted",
    "Cycle 14 also claimed B=64 = 232.2 with the v10+bf16 stack. Same dtype defect; v10 was not firing. Cycle 28 reverify at corrected v10 path measured 230.2 ± 1.6 (with v10) vs 231.9 ± 0.3 (bf16-only) — v10 marginally hurts, within noise. The honest hardware-ceiling KEEP is the bf16-only number, two indices to the right of this dot."],
  [29, 197.05, "diag", null, null,
    "AR_C26_BF16_FA_E2E_B52 — first reverify after the codex bf16 fix. Within the cycle-25 envelope; not a KEEP."],
  [30, 200.14, "diag", null, null,
    "AR_C25_B52_REVERIFY_CONDA_MISS — codex re-run under conda; near cycle-13 envelope."],
  [31, 185.30, "diag", null, null,
    "AR_C25_B52_REVERIFY_UV_MISS — codex re-run under uv; ~10% below conda mean. Cycle 25 attributed this to between-environment drift, not regression."],
  [32, 231.9, "keep", "C28", "ceiling 231.9 ± 0.3",
    "AR_C28_HARDWARE_CEILING_REVERIFY — the honest hardware-ceiling KEEP at B=64 with bf16-only stack (n=3). Replaces cycle-14's retracted 232.2. 5.50× cycle-1 baseline; 3.87× the (1b) gate."],
  [33, 204.0, "keep", "C33", "envelope 204 ± 1 n=6",
    "AR_C33_B52_VARIANCE_TIGHTEN — n=6 reverify across 2 sessions tightened combined σ at B=52 to 0.83 tok/s. Final running-best within envelope: 204 ± 1 tok/s. This is the protocol standard for variance discipline going forward."],
];

const PerfMoePoints = [
  [1, 188.5, "keep", "C1", "baseline 188.5 tok/s",
    "MOE_27B_B4 baseline — Qwen3.5-35B-A3B-4bit at B=4. 92% bandwidth utilisation; the (2a) ≥100 anchor was already cleared at the v1.7.13 baseline."],
  [2, 181.4, "diag", null, null,
    "AR_C34_MOE_B4_BF16 — bf16 DeltaNet state at B=4 on MoE. Slightly below cycle-1 because B=4 amortises poorly on MoE."],
  [3, 242.0, "diag", null, null,
    "AR_C34_MOE_B8 — first cycle-12 lever on MoE B-axis sweep."],
  [4, 306.3, "diag", null, null,
    "AR_C34_MOE_B16."],
  [5, 384.8, "diag", null, null,
    "AR_C34_MOE_B32."],
  [6, 434.3, "diag", null, null,
    "AR_C34_MOE_B48."],
  [7, 464.4, "keep", "C34", "envelope KEEP @ B=64",
    "AR_C34_MOE_PORTABILITY_KEEP — peak 33.8 GB (within 36 GB envelope), n=3. Same C10×C12 lever stack transferred via the shared gated_delta shadow patch. 2.46× MoE cycle-1 baseline."],
  [8, 447.9, "discard", null, null,
    "AR_C34_MOE_B80 — past the envelope, throughput dips."],
  [9, 444.5, "discard", null, null,
    "AR_C35_MOE_B72 — also slightly below."],
  [10, 467.1, "diag", null, null,
    "AR_C35_MOE_B96 — climbing again as we approach the hardware-ceiling sweep."],
  [11, 791.8, "keep", "C35", "ceiling 791.8 @ B=128",
    "AR_C35_MOE_B128_HARDWARE_KEEP — peak 47.96 GB at the 48 GB cap, n=3. 4.20× MoE cycle-1; 1.71× cycle-34 B=64 KEEP. The largest absolute throughput observed across the full 35-cycle effort. Per-row throughput non-monotonic — expert routing amortisation crosses the threshold near B=128 (~4 activations per expert per step at B=128 vs 2 at B=64)."],
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
                r={isActive ? 8 : 6}
                className="perf-svg-dot perf-svg-dot-retracted"
              />
            ) : (
              <circle
                cx={cx}
                cy={cy}
                r={isActive ? 7.5 : (isKeep ? 5.5 : 3.4)}
                className={`perf-svg-dot perf-svg-dot-${status}`}
              />
            )}
            {isActive && (
              <circle
                cx={cx}
                cy={cy}
                r={14}
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
        .perf-svg-grid { stroke: var(--rule-2); stroke-width: 1; }
        .perf-svg-tick { fill: var(--ink-3); font-size: 11px; }
        .perf-svg-axis-title {
          fill: var(--ink-3);
          font-size: 11px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .perf-svg-baseline {
          stroke: var(--ink-4);
          stroke-width: 1;
          stroke-dasharray: 4 5;
          opacity: 0.5;
        }
        .perf-svg-baseline-label { fill: var(--ink-3); font-size: 10px; }
        .perf-svg-best {
          fill: none;
          stroke: var(--accent);
          stroke-width: 2.4;
          stroke-linejoin: round;
          stroke-linecap: round;
          transition: stroke-dashoffset 60ms linear;
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
          transition: r 180ms ease;
        }
        .perf-svg-dot-keep { fill: var(--accent); }
        .perf-svg-dot-diag { fill: var(--ink-3); opacity: 0.55; }
        .perf-svg-dot-discard { fill: var(--ink-4); opacity: 0.35; }
        .perf-svg-dot-retracted {
          fill: var(--bg-elev);
          stroke: var(--ink-3);
          stroke-width: 1.6;
          stroke-dasharray: 2.5 2;
        }
        .perf-svg-dot-ring {
          stroke: var(--accent);
          stroke-width: 1.4;
          opacity: 0;
          animation: perfRing 1.6s infinite ease-out;
        }
        @keyframes perfRing {
          0% { opacity: 0.6; r: 8; }
          100% { opacity: 0; r: 18; }
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
          stroke-width: 1;
          opacity: 0.55;
        }
        .perf-svg-anno-cycle {
          fill: var(--accent);
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.02em;
        }
        .perf-svg-anno-label {
          fill: var(--ink-2);
          font-size: 10px;
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
