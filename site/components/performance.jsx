// Performance — P-6 autoresearch results showcase.
//
// The 35-cycle opus autoresearch loop cleared every P-6 acceptance
// gate 3.4-5.5x over the cycle-1 baseline on M5 Pro 48 GB. This
// section is in three movements: (1) headline numbers, (2) the four
// gate clearings, (3) the honest record — what closed with a
// negative, what got architecturally retired, and the codex-driven
// retraction that restored honest attribution. The closing card
// links the three load-bearing artefacts so readers can verify.

const Performance = () => {
  const headline = [
    {
      value: "232",
      unit: "tok/s",
      label: "Dense 27B decode",
      sub: "Qwen3.5-27B-4bit · B=64 · 48 GB hardware ceiling · n=3",
    },
    {
      value: "791.8",
      unit: "tok/s",
      label: "MoE 35B-A3B decode",
      sub: "Qwen3.5-35B-A3B-4bit · B=128 · peak 47.96 GB · n=3",
    },
    {
      value: "5.50",
      unit: "×",
      label: "Dense uplift from baseline",
      sub: "vs cycle-1 baseline 42.17 tok/s @ B=4",
    },
    {
      value: "35",
      unit: "cycles",
      label: "Karpathy-style autoresearch",
      sub: "opus branch · 110-row ledger · per-cycle reports",
    },
  ];

  const gates = [
    {
      gate: "(1a) Dense engineering",
      target: "≥40 tok/s",
      cleared: "204 ± 1 tok/s",
      mult: "4.85×",
      frame: "B=52 · 36 GB envelope · n=6 across 2 sessions",
    },
    {
      gate: "(1b) Dense stretch",
      target: "≥60 tok/s",
      cleared: "231.9 ± 0.3 tok/s",
      mult: "3.87×",
      frame: "B=64 · 48 GB hardware ceiling · n=3",
    },
    {
      gate: "(2a) MoE anchor",
      target: "≥100 tok/s",
      cleared: "120.93 tok/s",
      mult: "preserved",
      frame: "MoE B=2 · cleared at v1.7.13 baseline",
    },
    {
      gate: "(2b) MoE stretch",
      target: "≥175 tok/s",
      cleared: "791.8 ± 5.2 tok/s",
      mult: "4.52×",
      frame: "MoE B=128 · 48 GB hardware ceiling · n=3",
    },
  ];

  const levers = [
    {
      cycle: "Cycle 10",
      title: "Batched-aggregate axis-shift.",
      body: "Re-reading the AR.md metric definition (\"B is chosen to maximise aggregate\") moved the operating point from B=4 → B=48 within the 36 GB envelope. Pure parameter selection; no kernel change. 4.60× on its own.",
    },
    {
      cycle: "Cycle 12",
      title: "bf16 DeltaNet recurrent state.",
      body: "State shape [B, Hv=48, Dv=128, Dk=128] = 144 MB at fp32 per layer; 72 MB at bf16. Across 48 DeltaNet layers, peak-memory save is ~3.5 GB — opens B≥48 within the 36 GB envelope and unlocks B=64 within the 48 GB hardware ceiling. Greedy token-ID parity verified.",
    },
    {
      cycle: "Cycle 13",
      title: "Composition.",
      body: "Cycle-12's peak-memory save composed with cycle-10's B-axis lever produces the running-best line: 204 ± 1 tok/s within envelope, 231.9 ± 0.3 tok/s at hardware ceiling. The two levers are independent; together they dominate every later atomic probe in the loop.",
    },
  ];

  const honest = [
    {
      tag: "Closed with negative",
      title: "Speculative decoding at production B.",
      body: "Cycle 23 measured the B × k verify-cost matrix on Qwen3.5-27B-4bit / mlx 0.31.x: B=52 k=64 = 8105 ms versus same-B plain-decode ~252 ms. Tree-spec at b=64 recomputes to ~10 tok/s aggregate, a net regression by 20× vs plain. Track C settles: C.4 retired (η.1 = 0.482×), C.5 retired (cycle-23 closure), C.1/C.2/C.3/C.6 deprioritised — (1b) no longer needs them.",
    },
    {
      tag: "Closed at architectural cliff",
      title: "Dense B-axis past 64.",
      body: "Cycles 28-29 measured a 26% throughput drop at the B=64 → B=66 transition (40 GB peak boundary). Three allocator-hint probes (mx.metal.set_cache_limit / set_memory_limit / set_wired_limit) leave the cliff in place. The cliff is architectural — likely M5 Pro SLC threshold or unified-memory bandwidth contention near the 48 GB cap, not allocator policy.",
    },
    {
      tag: "Retracted via codex review",
      title: "Cycle 14's claimed v10 KEEP.",
      body: "A codex cross-review on opus-codex caught a 14-cycle dtype-defect in shadow_install (queries.dtype == mx.float16 silently skipped the bf16 production path). After the fix, an 8-rep reverify (cycles 27/28) measured v10's E2E contribution at +0.5 tok/s @ B=52 / -1.7 tok/s @ B=64 — both within noise. The honest running-best is C10 axis-shift × C12 bf16 DeltaNet state composition alone; the retracted KEEP is published as part of the research record.",
    },
  ];

  return (
    <section className="block" id="performance">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">P-6 autoresearch · v1.7.23</div>
          <h2>Cycle-1 baseline 42 tok/s → cycle-28 hardware ceiling 232 tok/s.</h2>
          <p>
            The 35-cycle opus autoresearch loop pushed Qwen3.5-27B-4bit warm decode 5.50× over the cycle-1 baseline on M5 Pro 48 GB. Two load-bearing levers — <span className="mono">cycle-10</span> batched-aggregate axis-shift (B=4 → B=52) and <span className="mono">cycle-12</span> bf16 DeltaNet recurrent state — composed to clear all four P-6 acceptance gates 3.4-5.5× over baseline. The MoE secondary track at B=128 = 791.8 tok/s is the largest absolute throughput observed across the full effort.
          </p>
        </div>

        <div className="perf-headline">
          {headline.map((h, i) => (
            <div key={i} className="perf-headline-card">
              <div className="perf-value mono">{h.value}<span className="perf-unit">{h.unit}</span></div>
              <div className="perf-label">{h.label}</div>
              <div className="perf-sub">{h.sub}</div>
            </div>
          ))}
        </div>

        <div className="perf-section">
          <div className="perf-subhead">Acceptance gates · every P-6 gate cleared</div>
          <div className="perf-gates">
            {gates.map((g, i) => (
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

        <div className="perf-section">
          <div className="perf-subhead">Two load-bearing levers · the running-best is composition, not a custom kernel</div>
          <div className="perf-levers">
            {levers.map((l, i) => (
              <div key={i} className="perf-lever-card">
                <div className="perf-lever-cycle mono">{l.cycle}</div>
                <div className="perf-lever-title">{l.title}</div>
                <p className="perf-lever-body">{l.body}</p>
              </div>
            ))}
          </div>
          <p className="perf-lever-foot">
            17 custom Metal kernel attempts (13 QMM versions + 7 FA-decode versions + DeltaNet vectorisation + 3 fused-op kernels) closed without a load-bearing E2E win. Cycle-30 explained why: at B=64 with the v10+bf16 stack, DeltaNet owns 88% of step time, full-attention 12.5%, dispatch 0.3% — and mlx's existing <span className="mono">gated_delta</span> is already at HBM-bandwidth limit (cycle-31 silica <span className="mono">gated_delta_v2</span> = 1.001× vs mlx). Source-string Metal kernels in mlx 0.31.x do not pay back on dense 27B; the unlock came from data layout (bf16 state) and operating-point selection (axis-shift).
          </p>
        </div>

        <div className="perf-section">
          <div className="perf-subhead">Honest record · what closed with a negative, what was retracted</div>
          <div className="perf-closed">
            {honest.map((c, i) => (
              <div key={i} className="perf-closed-card">
                <div className="perf-closed-tag mono">{c.tag}</div>
                <div className="perf-closed-title">{c.title}</div>
                <p className="perf-closed-body">{c.body}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="perf-foot">
          <div className="perf-foot-text">
            <strong>Methodology.</strong> Karpathy-style autoresearch ledger (one TSV row per measurement; the main agent appends, sub-agents return findings). Variance discipline: ≥3 reps per session, ≥2 sessions, combined σ check before declaring a KEEP. Cycle 33's combined σ at B=52 across 2 sessions tightened to 0.83 tok/s on n=6 — that protocol is the standard, not the exception. The cycle-27 retraction came from a codex cross-review that caught a defect 14 cycles after it was introduced; restoring honest attribution and publishing the retracted KEEP is treated as part of the research record. <strong>What's next.</strong> P-6 advances to <span className="mono">D-022</span> small-B interactive QoE — closing the dispatch and attention buckets at B ∈ {"{1, 2, 4, 8, 12}"} per the cycle-1 step-share decomposition.
          </div>
          <div className="perf-foot-cta">
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_NOTES.md" target="_blank" rel="noreferrer">Take-home notes</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_AUTORESEARCH_FINAL_REPORT.md" target="_blank" rel="noreferrer">23-cycle final report</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/AR.md" target="_blank" rel="noreferrer">AR.md directive</a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/P6_SMALL_B_OPENING.md" target="_blank" rel="noreferrer">D-022 small-B opening</a>
          </div>
        </div>
      </div>

      <style>{`
        .perf-headline {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
          margin-bottom: 56px;
        }
        @media (max-width: 920px) { .perf-headline { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 540px) { .perf-headline { grid-template-columns: 1fr; } }
        .perf-headline-card {
          padding: 28px 24px;
          background: var(--bg-elev);
          display: flex;
          flex-direction: column;
        }
        .perf-value {
          font-size: 44px;
          font-weight: 700;
          letter-spacing: -0.025em;
          line-height: 1;
          color: var(--ink);
          margin-bottom: 8px;
        }
        .perf-unit {
          font-size: 15px;
          font-weight: 500;
          color: var(--ink-3);
          margin-left: 4px;
        }
        .perf-label {
          font-size: 13px;
          font-weight: 600;
          color: var(--ink-2);
          margin-bottom: 4px;
        }
        .perf-sub {
          font-size: 12px;
          color: var(--ink-3);
          line-height: 1.45;
        }

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
        .perf-gate-name {
          font-size: 12px;
          color: var(--ink-3);
          margin-bottom: 12px;
        }
        .perf-gate-row {
          display: flex; align-items: baseline; justify-content: space-between;
          margin-bottom: 8px;
        }
        .perf-gate-target {
          font-size: 13px;
          color: var(--ink-3);
        }
        .perf-gate-mult {
          font-size: 22px;
          font-weight: 700;
          color: var(--ok);
          letter-spacing: -0.02em;
        }
        .perf-gate-cleared {
          font-size: 16px;
          font-weight: 600;
          color: var(--ink);
          margin-bottom: 6px;
        }
        .perf-gate-frame {
          font-size: 12px;
          color: var(--ink-3);
          line-height: 1.5;
        }

        .perf-levers {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
          margin-bottom: 18px;
        }
        @media (max-width: 920px) { .perf-levers { grid-template-columns: 1fr; } }
        .perf-lever-card {
          padding: 24px 24px 22px;
          background: var(--bg-elev);
        }
        .perf-lever-cycle {
          font-size: 11px;
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
          margin-bottom: 12px;
        }
        .perf-lever-title {
          font-size: 16px;
          font-weight: 600;
          letter-spacing: -0.012em;
          margin-bottom: 8px;
          color: var(--ink);
        }
        .perf-lever-body {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.55;
          margin: 0;
        }
        .perf-lever-foot {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
          margin: 0;
          padding: 18px 22px;
          background: var(--bg-sunken);
          border-radius: var(--radius);
          border: 1px solid var(--rule);
        }

        .perf-closed {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        @media (max-width: 920px) { .perf-closed { grid-template-columns: 1fr; } }
        .perf-closed-card {
          padding: 24px 24px 22px;
          background: var(--bg-elev);
        }
        .perf-closed-tag {
          font-size: 11px;
          color: var(--warn);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
          margin-bottom: 14px;
        }
        .perf-closed-title {
          font-size: 16px;
          font-weight: 600;
          letter-spacing: -0.012em;
          margin-bottom: 8px;
          color: var(--ink);
        }
        .perf-closed-body {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.55;
          margin: 0;
        }

        .perf-foot {
          margin-top: 36px;
          padding: 24px;
          background: var(--bg-sunken);
          border-radius: var(--radius);
          border: 1px solid var(--rule);
        }
        .perf-foot-text {
          font-size: 13px;
          color: var(--ink-2);
          line-height: 1.6;
          margin-bottom: 16px;
        }
        .perf-foot-text strong { color: var(--ink); font-weight: 600; }
        .perf-foot-cta {
          display: flex; flex-wrap: wrap; gap: 8px;
        }
        .perf-foot-cta .btn { padding: 6px 12px; font-size: 12px; }
      `}</style>
    </section>
  );
};

window.Performance = Performance;
