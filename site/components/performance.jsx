// Performance — visual showcase of the P-6 autoresearch results.
//
// Visual-first composition: a tabbed chart panel as the centrepiece
// (Karpathy-style ledger plots), a row of click-to-expand cycle-
// highlight cards beneath the chart, a compact acceptance-gates
// row, and a small gallery of secondary charts. Long-form text
// (load-bearing levers, honest closures, methodology) lives behind
// click-to-expand details rather than as a wall of paragraphs.

const PerfChartTabs = [
  {
    id: "dense",
    label: "Dense 27B",
    sub: "Karpathy ledger · 38 experiments · 9 kept",
    img: "uploads/p6-decode-dense.png",
    caption: "Running-best line: 42.17 tok/s @ B=4 (cycle 1) → 232 tok/s @ B=64 (cycle 28). Each dot is a measurement; the ladder is the running-best.",
    head: "Dense Qwen3.5-27B-4bit · running best 232.2 tok/s",
  },
  {
    id: "moe",
    label: "MoE 35B-A3B",
    sub: "Karpathy ledger · 11 experiments · 2 kept",
    img: "uploads/p6-decode-moe.png",
    caption: "Running-best line: 188.5 tok/s @ B=4 (cycle 1) → 791.8 tok/s @ B=128 (cycle 35). The 791.8 measurement is the largest absolute throughput observed across the full 35-cycle effort.",
    head: "MoE Qwen3.5-35B-A3B-4bit · running best 791.8 tok/s",
  },
  {
    id: "cycles",
    label: "Cycle deliverables",
    sub: "35 cycles · per-cycle outcomes",
    img: "uploads/p6-cycles.png",
    caption: "What each cycle produced: kept (running-best moved), discard (no improvement vs prior best), or correction (cycle 27 retraction).",
    head: "Cycle ledger · 35 cycles, 5 lasting load-bearing changes",
  },
  {
    id: "summary",
    label: "Summary",
    sub: "Multi-panel summary chart",
    img: "uploads/p6-summary.png",
    caption: "Multi-panel summary covering both tracks, kernel ablations, and the cycle-27 corrected attribution.",
    head: "Multi-panel summary",
  },
];

const PerfCycleHighlights = [
  {
    n: 1,
    cycle: "Cycle 1",
    headline: "Baseline",
    tok: "42.17 tok/s",
    blurb: "Dense 27B B=4 warm decode at 52% bandwidth utilisation. Per-step decomposition: DeltaNet 74% / full-attn 22% / overhead 4%.",
    detail: "Cycle 1 anchored P-6.0.5's measurement frame and proved 42.17 tok/s was not the chip ceiling. This baseline is the denominator for every later uplift number; preserving it as a 4.60-5.50× anchor takes work because run-to-run variance has to stay under control through every later cycle.",
  },
  {
    n: 2,
    cycle: "Cycle 10",
    headline: "Axis-shift breakthrough",
    tok: "+4.60× alone",
    blurb: "Re-reading the AR.md metric definition (\"B is chosen to maximise aggregate\") moved the operating point B=4 → B=48. Pure parameter selection; no kernel change.",
    detail: "Nine cycles of QMM kernel work at fixed B=4 produced zero KEEPs. Cycle 10 did not write any new code — it moved the operating point along the axis the metric definition pointed at. This is the single biggest leverage event in the loop, and it's why kernel work alone is the wrong frame for the M5 Pro envelope.",
  },
  {
    n: 3,
    cycle: "Cycle 13",
    headline: "Composition KEEP",
    tok: "204 tok/s @ B=52",
    blurb: "Cycle-12's bf16 DeltaNet state save (3.5 GB peak) composed with cycle-10's B-axis lever. Within the 36 GB envelope; 18σ above C10.",
    detail: "C12 alone was 0% E2E at fixed B=48 — it produced peak-memory headroom but no direct speedup. C10 alone was capped at B=48 by the fp32 state's memory footprint. Composing the two pushed B from 48 to 52 within the envelope and 64 at the hardware ceiling. The 193 tok/s wall observed earlier was a B=48 cap, not a hardware wall.",
  },
  {
    n: 4,
    cycle: "Cycle 27",
    headline: "Codex retraction",
    tok: "−5.4 tok/s revised away",
    blurb: "Codex cross-review on opus-codex caught a 14-cycle dtype defect: shadow_install checked for fp16 but the production path is bf16. v10's claimed C14 KEEP was attribution error.",
    detail: "After the bf16-native v10 fix and 8-rep reverify, v10's E2E contribution measured +0.5 tok/s @ B=52 / -1.7 tok/s @ B=64 — both within noise. The honest running-best is C10+C12 composition alone. Publishing the retracted KEEP (rather than quietly editing it out) is part of the research record; the take-home note for cross-reviews is to check small-n σ against between-session variance.",
  },
  {
    n: 5,
    cycle: "Cycle 28",
    headline: "Hardware ceiling",
    tok: "231.9 ± 0.3 tok/s",
    blurb: "Re-measured at B=64 with the corrected v10 path. v10 contribution within noise; bf16-only is the load-bearing piece.",
    detail: "At B=66 throughput drops 26% (40 GB peak boundary). Cycle 29 confirmed three allocator-hint probes (mx.metal.set_cache_limit / set_memory_limit / set_wired_limit) leave the cliff in place — architectural, not allocator policy. Dense 27B B-axis extension is closed at this ceiling.",
  },
  {
    n: 6,
    cycle: "Cycle 35",
    headline: "MoE ceiling",
    tok: "791.8 ± 5.2 tok/s @ B=128",
    blurb: "MoE 35B-A3B at the 48 GB hardware ceiling. Same C10×C12 lever stack, transferred via the shared gated_delta shadow patch.",
    detail: "MoE does not have the dense 27B's 40 GB cliff because expert sparsity (8 of 256 experts active per token) bypasses dense activation pressure. Per-row throughput is non-monotonic — amortisation crosses the threshold near B=128 where each expert sees ~4 activations per step vs 2 at B=64.",
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
          <strong>17 custom Metal kernel attempts closed without a load-bearing E2E win.</strong> Cycle-30 explained why: at B=64 with the v10+bf16 stack, DeltaNet owns 88% of step time, full-attn 12.5%, dispatch 0.3% — and mlx's existing <span className="mono">gated_delta</span> kernel is already at HBM-bandwidth limit (cycle-31 silica <span className="mono">gated_delta_v2</span> = 1.001× vs mlx). Source-string Metal kernels in mlx 0.31.x do not pay back on dense 27B.
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
          <li><strong>Cycle 14's claimed v10 KEEP — retracted via codex review.</strong> A 14-cycle dtype-defect in <span className="mono">shadow_install</span> silently skipped the bf16 production path. After the fix, cycles 27/28 measured v10's E2E at +0.5 tok/s @ B=52 / −1.7 tok/s @ B=64 — both within noise. The retraction is published as part of the research record.</li>
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

const PerfGallery = [
  { src: "uploads/p6-fa-kernel.png", title: "FA-decode kernel ablation", note: "Cycle 11 · v10 vs mlx SDPA across T_kv ∈ {128, 256, 512, 1024}." },
  { src: "uploads/p6-fa-bf16.png", title: "FA bf16 microbench", note: "Cycle 26 · post-codex bf16-native v10 microbench at T=512." },
  { src: "uploads/p6-qmm-kernel.png", title: "QMM kernel tuning", note: "Cycles 7-9 · 13 silica QMM versions vs mlx qmv_quad. mlx wins at the bandwidth limit." },
];

const Performance = () => {
  const [activeTab, setActiveTab] = React.useState("dense");
  const [expandedCycle, setExpandedCycle] = React.useState(null);
  const [openDetail, setOpenDetail] = React.useState(null);
  const [galleryOpen, setGalleryOpen] = React.useState(null);

  const tab = PerfChartTabs.find(t => t.id === activeTab) || PerfChartTabs[0];

  return (
    <section className="block" id="performance">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">P-6 autoresearch · v1.7.23</div>
          <h2>35 cycles. Two levers. Every gate cleared 3.4-5.5×.</h2>
          <p>
            The opus autoresearch loop pushed Qwen3.5-27B-4bit warm decode 5.50× over the cycle-1 baseline on M5 Pro 48 GB, and Qwen3.5-35B-A3B-4bit MoE to 791.8 tok/s at the 48 GB hardware ceiling. The chart below is the Karpathy-style ledger that drove every decision — each dot a measurement, the ladder line the running best.
          </p>
        </div>

        {/* Hero chart panel */}
        <div className="perf-chart-panel">
          <div className="perf-tabs" role="tablist">
            {PerfChartTabs.map(t => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={activeTab === t.id}
                className={"perf-tab" + (activeTab === t.id ? " perf-tab-active" : "")}
                onClick={() => setActiveTab(t.id)}
              >
                <span className="perf-tab-label">{t.label}</span>
                <span className="perf-tab-sub mono">{t.sub}</span>
              </button>
            ))}
          </div>
          <div className="perf-chart-frame">
            <div className="perf-chart-head mono">{tab.head}</div>
            <img src={tab.img} alt={tab.head} className="perf-chart-img" />
            <div className="perf-chart-cap">{tab.caption}</div>
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

        {/* Secondary chart gallery */}
        <div className="perf-section">
          <div className="perf-subhead">Secondary charts · click to enlarge</div>
          <div className="perf-gallery">
            {PerfGallery.map((g, i) => (
              <button
                key={i}
                type="button"
                className="perf-thumb"
                onClick={() => setGalleryOpen(galleryOpen === i ? null : i)}
                aria-expanded={galleryOpen === i}
              >
                <img src={g.src} alt={g.title} className="perf-thumb-img" />
                <div className="perf-thumb-meta">
                  <div className="perf-thumb-title">{g.title}</div>
                  <div className="perf-thumb-note">{g.note}</div>
                </div>
              </button>
            ))}
          </div>
          {galleryOpen !== null && (
            <div className="perf-gallery-modal" onClick={() => setGalleryOpen(null)}>
              <div className="perf-gallery-modal-inner" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="perf-gallery-close"
                  onClick={() => setGalleryOpen(null)}
                  aria-label="close"
                >
                  ×
                </button>
                <img src={PerfGallery[galleryOpen].src} alt={PerfGallery[galleryOpen].title} className="perf-gallery-modal-img" />
                <div className="perf-gallery-modal-meta">
                  <div className="perf-gallery-modal-title">{PerfGallery[galleryOpen].title}</div>
                  <div className="perf-gallery-modal-note">{PerfGallery[galleryOpen].note}</div>
                </div>
              </div>
            </div>
          )}
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
        .perf-chart-panel {
          margin-bottom: 56px;
        }
        .perf-tabs {
          display: flex;
          gap: 1px;
          background: var(--rule);
          border: 1px solid var(--rule);
          border-bottom: none;
          border-radius: var(--radius) var(--radius) 0 0;
          overflow: hidden;
        }
        @media (max-width: 720px) {
          .perf-tabs { flex-wrap: wrap; }
        }
        .perf-tab {
          flex: 1;
          background: var(--bg-elev);
          border: none;
          padding: 14px 18px;
          text-align: left;
          cursor: pointer;
          display: flex;
          flex-direction: column;
          gap: 3px;
          color: var(--ink-3);
          transition: color 120ms, background 120ms;
          min-width: 160px;
        }
        .perf-tab:hover {
          background: var(--bg-sunken);
          color: var(--ink);
        }
        .perf-tab-active {
          background: var(--bg-elev);
          color: var(--ink);
          box-shadow: inset 0 -2px 0 var(--accent);
        }
        .perf-tab-label {
          font-size: 14px;
          font-weight: 600;
        }
        .perf-tab-sub {
          font-size: 11px;
          color: var(--ink-3);
        }
        .perf-tab-active .perf-tab-sub { color: var(--ink-2); }

        .perf-chart-frame {
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: 0 0 var(--radius) var(--radius);
          padding: 0;
          overflow: hidden;
        }
        .perf-chart-head {
          font-size: 12px;
          color: var(--accent);
          padding: 14px 20px 0;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          font-weight: 600;
        }
        .perf-chart-img {
          display: block;
          width: 100%;
          height: auto;
          padding: 12px 20px 4px;
        }
        .perf-chart-cap {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.55;
          padding: 4px 24px 22px;
          border-top: 1px solid var(--rule-2);
          margin-top: 8px;
          padding-top: 16px;
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
        .perf-detail {
          background: var(--bg-elev);
        }
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

        /* Gallery */
        .perf-gallery {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }
        @media (max-width: 720px) { .perf-gallery { grid-template-columns: 1fr; } }
        .perf-thumb {
          all: unset;
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: var(--radius-sm);
          padding: 10px;
          cursor: pointer;
          transition: border-color 120ms, transform 120ms;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .perf-thumb:hover {
          border-color: var(--accent);
          transform: translateY(-1px);
        }
        .perf-thumb-img {
          display: block;
          width: 100%;
          height: 140px;
          object-fit: contain;
          background: var(--bg-sunken);
          border-radius: var(--radius-sm);
        }
        .perf-thumb-meta { padding: 4px 4px 8px; }
        .perf-thumb-title {
          font-size: 13px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 4px;
        }
        .perf-thumb-note {
          font-size: 12px;
          color: var(--ink-3);
          line-height: 1.45;
        }

        .perf-gallery-modal {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.72);
          z-index: 200;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
          cursor: zoom-out;
        }
        .perf-gallery-modal-inner {
          background: var(--bg-elev);
          border: 1px solid var(--rule);
          border-radius: var(--radius);
          max-width: 1100px;
          max-height: 90vh;
          width: 100%;
          overflow: auto;
          position: relative;
          cursor: default;
        }
        .perf-gallery-close {
          all: unset;
          position: absolute;
          top: 12px;
          right: 16px;
          font-size: 28px;
          color: var(--ink-3);
          cursor: pointer;
          line-height: 1;
          width: 32px;
          height: 32px;
          text-align: center;
        }
        .perf-gallery-close:hover { color: var(--ink); }
        .perf-gallery-modal-img {
          display: block;
          width: 100%;
          height: auto;
          padding: 24px 24px 8px;
        }
        .perf-gallery-modal-meta {
          padding: 16px 24px 24px;
          border-top: 1px solid var(--rule-2);
        }
        .perf-gallery-modal-title {
          font-size: 15px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 4px;
        }
        .perf-gallery-modal-note {
          font-size: 13px;
          color: var(--ink-3);
          line-height: 1.5;
        }

        /* Foot */
        .perf-foot {
          margin-top: 36px;
          padding: 18px 22px;
          background: var(--bg-sunken);
          border-radius: var(--radius);
          border: 1px solid var(--rule);
        }
        .perf-foot-cta {
          display: flex; flex-wrap: wrap; gap: 8px;
        }
        .perf-foot-cta .btn { padding: 6px 12px; font-size: 12px; }
      `}</style>
    </section>
  );
};

window.Performance = Performance;
