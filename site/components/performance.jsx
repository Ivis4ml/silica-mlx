// Performance — visual showcase of the P-6 autoresearch results.
//
// The Karpathy-style ledger plots are redrawn as native SVG so they
// pick up the site's CSS variables (accent, ink, bg, fonts) and
// adapt to the active theme. Dots are coloured by status (keep /
// diagnostic / discard / retracted). The running-best ladder steps
// through every measurement that improved on the previous best,
// honest-attribution-style: the cycle-14 retracted KEEPs appear as
// dashed open circles and do *not* contribute to the running-best;
// the line resumes at cycle 28's reverify.

const PerfDensePoints = [
  // [seq, tok_s, status, cycle?, label?]
  // status: 'keep' | 'diag' | 'discard' | 'retracted'
  [1, 42.17, "keep", "C1", "baseline 42.17"],
  [2, 6.54, "discard"],
  [3, 7.74, "discard"],
  [4, 17.10, "discard"],
  [5, 39.99, "diag"],
  [6, 41.00, "discard"],
  [7, 42.39, "discard"],
  [8, 42.53, "discard"],
  [9, 42.68, "diag"],
  [10, 43.0, "diag"],
  [11, 63.1, "keep"],
  [12, 81.1, "diag"],
  [13, 112.8, "diag"],
  [14, 150.6, "keep"],
  [15, 171.6, "keep"],
  [16, 183.3, "keep"],
  [17, 193.9, "keep", "C10", "axis-shift @ B=48"],
  [18, 193.3, "diag"],
  [19, 192.5, "diag"],
  [20, 200.8, "keep", "C13", "envelope @ B=52"],
  [21, 212.2, "diag"],
  [22, 219.1, "diag"],
  [23, 229.8, "keep"],
  [24, 166.8, "discard"],
  [25, 169.1, "discard"],
  [26, 173.2, "discard"],
  [27, 206.2, "retracted"],
  [28, 232.2, "retracted"],
  [29, 197.05, "diag"],
  [30, 200.14, "diag"],
  [31, 185.30, "diag"],
  [32, 231.9, "keep", "C28", "ceiling @ B=64"],
  [33, 204.0, "keep"],
];

const PerfMoePoints = [
  [1, 188.5, "keep", "C1", "baseline 188.5"],
  [2, 181.4, "diag"],
  [3, 242.0, "diag"],
  [4, 306.3, "diag"],
  [5, 384.8, "diag"],
  [6, 434.3, "diag"],
  [7, 464.4, "keep", "C34", "envelope @ B=64"],
  [8, 447.9, "discard"],
  [9, 444.5, "discard"],
  [10, 467.1, "diag"],
  [11, 791.8, "keep", "C35", "ceiling @ B=128"],
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
    caption: "Each dot is a measurement; the ladder line is the running best honest-attributed (cycles 27/28 retraction respected). The two dashed circles at the top right are cycle 14's retracted KEEPs — published as part of the research record.",
  },
  moe: {
    label: "MoE Qwen3.5-35B-A3B-4bit",
    sub: "11 measurements · 2 KEEPs · running best 791.8 tok/s",
    points: PerfMoePoints,
    yMax: 850,
    yTicks: [0, 200, 400, 600, 800],
    baseline: 188.5,
    baselineLabel: "cycle-1 baseline 188.5",
    caption: "Same C10×C12 lever stack ported via the shared gated_delta shadow patch (cycles 34-35). Per-row throughput is non-monotonic; the cycle-35 hardware ceiling at B=128 is the largest absolute throughput observed across the full effort.",
  },
};

const PerfLedger = ({ trackKey }) => {
  const t = PerfTracks[trackKey];
  const data = t.points;
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

  // Honest running-best: skip retracted points, treat all measurements
  // (keep + diag + discard) as candidates for "best so far".
  let best = 0;
  const bestSeries = data.map(p => {
    const status = p[2];
    const tok = p[1];
    if (status !== "retracted" && tok > best) best = tok;
    return best;
  });

  // Build a step-style polyline that hugs the dot positions and
  // steps up at each new best.
  const stepPoints = [];
  bestSeries.forEach((b, i) => {
    if (i === 0) {
      stepPoints.push([xAt(i), yAt(b)]);
    } else {
      stepPoints.push([xAt(i), yAt(bestSeries[i - 1])]);
      stepPoints.push([xAt(i), yAt(b)]);
    }
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="perf-svg"
      role="img"
      aria-label={`${t.label} — Karpathy-style autoresearch ledger`}
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

      {/* running-best step polyline */}
      <polyline
        points={stepPoints.map(p => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ")}
        className="perf-svg-best"
      />

      {/* dots */}
      {data.map((p, i) => {
        const [seq, tok, status, cycle, label] = p;
        const cx = xAt(i);
        const cy = yAt(tok);
        const isKeep = status === "keep";
        const isRetracted = status === "retracted";
        return (
          <g key={`d-${i}`}>
            {isRetracted ? (
              <circle
                cx={cx}
                cy={cy}
                r={6}
                className="perf-svg-dot perf-svg-dot-retracted"
              />
            ) : (
              <circle
                cx={cx}
                cy={cy}
                r={isKeep ? 5.5 : 3.4}
                className={`perf-svg-dot perf-svg-dot-${status}`}
              />
            )}
            <title>{`#${seq}${cycle ? " " + cycle : ""}: ${tok} tok/s (${status})`}</title>
          </g>
        );
      })}

      {/* milestone annotations */}
      {data.map((p, i) => {
        const [_, tok, status, cycle, label] = p;
        if (!cycle || !label) return null;
        const cx = xAt(i);
        const cy = yAt(tok);
        // place label above for low values, below for high values
        const above = cy > padT + innerH * 0.4;
        const yLine1 = above ? cy - 30 : cy + 22;
        const yLine2 = above ? cy - 16 : cy + 34;
        return (
          <g key={`anno-${i}`}>
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

const PerfCycleHighlights = [
  {
    n: 1,
    cycle: "Cycle 1",
    headline: "Baseline",
    tok: "42.17 tok/s",
    blurb: "Dense 27B B=4 warm decode at 52% bandwidth utilisation. Per-step decomposition: DeltaNet 74% / full-attn 22% / overhead 4%.",
    detail: "Cycle 1 anchored P-6.0.5's measurement frame and proved 42.17 tok/s was not the chip ceiling. This baseline is the denominator for every later uplift number.",
  },
  {
    n: 2,
    cycle: "Cycle 10",
    headline: "Axis-shift breakthrough",
    tok: "+4.60× alone",
    blurb: "Re-reading the AR.md metric definition (\"B is chosen to maximise aggregate\") moved the operating point B=4 → B=48. Pure parameter selection; no kernel change.",
    detail: "Nine cycles of QMM kernel work at fixed B=4 produced zero KEEPs. Cycle 10 did not write any new code — it moved the operating point along the axis the metric definition pointed at. The single biggest leverage event in the loop.",
  },
  {
    n: 3,
    cycle: "Cycle 13",
    headline: "Composition KEEP",
    tok: "204 tok/s @ B=52",
    blurb: "Cycle-12's bf16 DeltaNet state save (3.5 GB peak) composed with cycle-10's B-axis lever. Within the 36 GB envelope; 18σ above C10.",
    detail: "C12 alone was 0% E2E at fixed B=48 — peak-memory headroom but no direct speedup. C10 alone was capped at B=48 by fp32 state's memory footprint. Composition pushed B from 48 → 52 within envelope and 64 at hardware ceiling.",
  },
  {
    n: 4,
    cycle: "Cycle 27",
    headline: "Codex retraction",
    tok: "−5.4 tok/s revised away",
    blurb: "Codex cross-review on opus-codex caught a 14-cycle dtype defect: shadow_install checked for fp16 but the production path is bf16. v10's claimed C14 KEEP was attribution error.",
    detail: "After the bf16-native v10 fix and 8-rep reverify, v10's E2E contribution measured +0.5 tok/s @ B=52 / -1.7 tok/s @ B=64 — both within noise. The honest running-best is C10+C12 alone. Publishing the retracted KEEP rather than quietly editing it out is part of the research record.",
  },
  {
    n: 5,
    cycle: "Cycle 28",
    headline: "Hardware ceiling",
    tok: "231.9 ± 0.3 tok/s",
    blurb: "Re-measured at B=64 with the corrected v10 path. v10 contribution within noise; bf16-only is the load-bearing piece.",
    detail: "At B=66 throughput drops 26% (40 GB peak boundary). Cycle 29 confirmed three allocator-hint probes leave the cliff in place — architectural, not allocator policy. Dense 27B B-axis extension is closed at this ceiling.",
  },
  {
    n: 6,
    cycle: "Cycle 35",
    headline: "MoE ceiling",
    tok: "791.8 ± 5.2 tok/s @ B=128",
    blurb: "MoE 35B-A3B at the 48 GB hardware ceiling. Same C10×C12 lever stack, transferred via the shared gated_delta shadow patch.",
    detail: "MoE does not have the dense 27B's 40 GB cliff because expert sparsity (8 of 256 experts active per token) bypasses dense activation pressure. Per-row throughput non-monotonic — amortisation crosses the threshold near B=128.",
  },
];

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
          <li><strong>Cycle 12 — bf16 DeltaNet recurrent state.</strong> State shape <span className="mono">[B, Hv=48, Dv=128, Dk=128]</span> is 144 MB at fp32 per layer, 72 MB at bf16. Across 48 DeltaNet layers, ~3.5 GB peak save. Opens B≥48 within envelope, B=64 at ceiling.</li>
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
          <li><strong>Cycle 14's claimed v10 KEEP — retracted via codex review.</strong> A 14-cycle dtype-defect in <span className="mono">shadow_install</span> silently skipped the bf16 production path. After the fix, cycles 27/28 measured v10's E2E at +0.5 tok/s @ B=52 / −1.7 tok/s @ B=64 — both within noise. The retraction is published as part of the research record (the two dashed circles in the chart).</li>
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
  const [expandedCycle, setExpandedCycle] = React.useState(null);
  const [openDetail, setOpenDetail] = React.useState(null);
  const t = PerfTracks[activeTab];

  return (
    <section className="block" id="performance">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">P-6 autoresearch · v1.7.23</div>
          <h2>35 cycles. Two levers. Every gate cleared 3.4-5.5×.</h2>
          <p>
            The opus autoresearch loop pushed Qwen3.5-27B-4bit warm decode 5.50× over the cycle-1 baseline on M5 Pro 48 GB, and Qwen3.5-35B-A3B-4bit MoE to 791.8 tok/s at the 48 GB hardware ceiling. Below is the Karpathy-style ledger that drove every decision — each dot a measurement, the ladder line the running best, the dashed circles cycle 14's retracted KEEPs.
          </p>
        </div>

        {/* Hero chart panel */}
        <div className="perf-chart-panel">
          <div className="perf-tabs" role="tablist">
            {Object.entries(PerfTracks).map(([key, track]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={activeTab === key}
                className={"perf-tab" + (activeTab === key ? " perf-tab-active" : "")}
                onClick={() => setActiveTab(key)}
              >
                <span className="perf-tab-label">{track.label}</span>
                <span className="perf-tab-sub mono">{track.sub}</span>
              </button>
            ))}
          </div>
          <div className="perf-chart-frame">
            <PerfLedger trackKey={activeTab} />
            <PerfLegend />
            <div className="perf-chart-cap">{t.caption}</div>
          </div>
        </div>

        {/* Cycle highlights — click to expand */}
        <div className="perf-section">
          <div className="perf-subhead">Cycle highlights · click to expand</div>
          <div className="perf-cycles">
            {PerfCycleHighlights.map(c => {
              const open = expandedCycle === c.n;
              return (
                <button
                  key={c.n}
                  type="button"
                  className={"perf-cycle" + (open ? " perf-cycle-open" : "")}
                  onClick={() => setExpandedCycle(open ? null : c.n)}
                  aria-expanded={open}
                >
                  <div className="perf-cycle-num mono">{c.n.toString().padStart(2, "0")}</div>
                  <div className="perf-cycle-meta">
                    <div className="perf-cycle-cycle mono">{c.cycle}</div>
                    <div className="perf-cycle-head">{c.headline}</div>
                    <div className="perf-cycle-tok mono">{c.tok}</div>
                    <div className="perf-cycle-blurb">{c.blurb}</div>
                    {open && <div className="perf-cycle-detail">{c.detail}</div>}
                  </div>
                  <div className="perf-cycle-chev mono">{open ? "−" : "+"}</div>
                </button>
              );
            })}
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
        /* Tabs */
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
        @media (max-width: 720px) { .perf-tabs { flex-direction: column; } }
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

        .perf-chart-frame {
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: 0 0 var(--radius) var(--radius);
          padding: 24px 24px 0;
        }

        /* SVG ledger */
        .perf-svg {
          width: 100%;
          height: auto;
          display: block;
          font-family: var(--font-sans);
          overflow: visible;
        }
        .perf-svg-grid {
          stroke: var(--rule-2);
          stroke-width: 1;
        }
        .perf-svg-tick {
          fill: var(--ink-3);
          font-size: 11px;
        }
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
        .perf-svg-baseline-label {
          fill: var(--ink-3);
          font-size: 10px;
        }
        .perf-svg-best {
          fill: none;
          stroke: var(--accent);
          stroke-width: 2.4;
          stroke-linejoin: round;
          stroke-linecap: round;
        }
        .perf-svg-dot {
          transition: r 120ms;
          cursor: pointer;
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

        .perf-chart-cap {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
          padding: 14px 0 22px;
          border-top: 1px solid var(--rule-2);
          margin-top: 0;
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

        /* Cycle highlights */
        .perf-cycles {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        @media (max-width: 920px) { .perf-cycles { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 540px) { .perf-cycles { grid-template-columns: 1fr; } }
        .perf-cycle {
          all: unset;
          background: var(--bg-elev);
          padding: 22px 22px 20px;
          cursor: pointer;
          display: grid;
          grid-template-columns: 36px 1fr 16px;
          gap: 14px;
          align-items: start;
          transition: background 120ms;
        }
        .perf-cycle:hover { background: var(--bg-sunken); }
        .perf-cycle-open { background: var(--bg-sunken); }
        .perf-cycle-num {
          font-size: 22px;
          font-weight: 700;
          color: var(--accent);
          line-height: 1;
          letter-spacing: -0.02em;
        }
        .perf-cycle-cycle {
          font-size: 11px;
          color: var(--ink-3);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 4px;
        }
        .perf-cycle-head {
          font-size: 16px;
          font-weight: 600;
          color: var(--ink);
          letter-spacing: -0.012em;
          margin-bottom: 4px;
        }
        .perf-cycle-tok {
          font-size: 13px;
          font-weight: 600;
          color: var(--ok);
          margin-bottom: 8px;
        }
        .perf-cycle-blurb {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.5;
        }
        .perf-cycle-detail {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid var(--rule-2);
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
        }
        .perf-cycle-chev {
          font-size: 18px;
          color: var(--ink-3);
          line-height: 1;
          font-weight: 600;
          padding-top: 2px;
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
