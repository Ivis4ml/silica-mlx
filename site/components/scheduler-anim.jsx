// Animated continuous-batching visualization.
// Shows requests arriving, getting admitted, prefilled, decoded, completing.

const SchedulerAnim = () => {
  const [tick, setTick] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTick(t => (t + 1) % 60), 220);
    return () => clearInterval(id);
  }, [playing]);

  // 7 rows; each request has: arrive_t, prefill_len, decode_len, prefix_hit
  const reqs = [
    { id: "r1", arrive: 0,  prefill: 4, decode: 18, hit: 0,  label: "Write a haiku about silicon." },
    { id: "r2", arrive: 1,  prefill: 6, decode: 14, hit: 0,  label: "Explain TTFT in one sentence." },
    { id: "r3", arrive: 3,  prefill: 1, decode: 22, hit: 5,  label: "The capital of France is" },
    { id: "r4", arrive: 5,  prefill: 1, decode: 16, hit: 5,  label: "The capital of Germany is" },
    { id: "r5", arrive: 8,  prefill: 7, decode: 24, hit: 0,  label: "Summarize the radix prefix cache." },
    { id: "r6", arrive: 14, prefill: 5, decode: 18, hit: 0,  label: "Why MLX unified memory?" },
    { id: "r7", arrive: 22, prefill: 1, decode: 20, hit: 5,  label: "The capital of Japan is" },
  ];

  const cols = 50;

  return (
    <section className="block" id="batching">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">Continuous batching</div>
          <h2>Prefill and decode, interleaved per step.</h2>
          <p>Requests enter at any time. The scheduler interleaves prefill chunks with active-row decode every step. Shared prefixes hit the radix cache and skip forward tokens entirely.</p>
        </div>

        <div className="sched-card">
          <div className="sched-head">
            <div className="sched-legend">
              <span className="lg lg-prefix"></span><span className="lg-label">prefix hit</span>
              <span className="lg lg-prefill"></span><span className="lg-label">prefill</span>
              <span className="lg lg-decode"></span><span className="lg-label">decode</span>
              <span className="lg lg-done"></span><span className="lg-label">complete</span>
            </div>
            <div className="sched-controls">
              <span className="sched-tick mono">step {String(tick).padStart(2, '0')} / {cols}</span>
              <button className="sched-btn" onClick={() => setPlaying(p => !p)}>
                <Icon name={playing ? "pause" : "play"} size={12} />
                {playing ? "Pause" : "Play"}
              </button>
            </div>
          </div>

          <div className="sched-grid">
            {reqs.map((r, ri) => {
              const cells = [];
              for (let c = 0; c < cols; c++) {
                let kind = "idle";
                let visible = c <= tick;
                const rel = c - r.arrive;
                if (rel < 0) kind = "queue";
                else if (rel < r.hit) kind = "prefix";
                else if (rel < r.hit + r.prefill) kind = "prefill";
                else if (rel < r.hit + r.prefill + r.decode) kind = "decode";
                else kind = "done";
                cells.push(
                  <div
                    key={c}
                    className={"sc sc-" + kind + (visible ? " sc-on" : "")}
                  ></div>
                );
              }
              return (
                <React.Fragment key={r.id}>
                  <div className="sched-row-label">
                    <div className="sched-row-id mono">{r.id}</div>
                    <div className="sched-row-prompt">{r.label}</div>
                  </div>
                  <div className="sched-row-cells">
                    {cells}
                  </div>
                </React.Fragment>
              );
            })}
          </div>

          <div className="sched-foot">
            <div className="sched-stat">
              <div className="sched-stat-v mono">{Math.min(tick, cols)}</div>
              <div className="sched-stat-l">scheduler steps</div>
            </div>
            <div className="sched-stat">
              <div className="sched-stat-v mono">{reqs.filter(r => tick >= r.arrive && tick < r.arrive + r.hit + r.prefill + r.decode).length}</div>
              <div className="sched-stat-l">active rows</div>
            </div>
            <div className="sched-stat">
              <div className="sched-stat-v mono">{reqs.filter(r => tick >= r.arrive + r.hit + r.prefill + r.decode).length}</div>
              <div className="sched-stat-l">complete</div>
            </div>
            <div className="sched-stat">
              <div className="sched-stat-v mono">{reqs.filter(r => r.hit > 0 && tick >= r.arrive).length}</div>
              <div className="sched-stat-l">prefix hits</div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .sched-card {
          background: var(--bg-elev);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          padding: 22px;
          overflow: hidden;
        }
        .sched-head {
          display: flex; justify-content: space-between; align-items: center;
          margin-bottom: 18px;
          flex-wrap: wrap; gap: 12px;
        }
        .sched-legend {
          display: flex; align-items: center; gap: 6px;
          font-size: 12px;
          color: var(--ink-3);
          flex-wrap: wrap;
        }
        .lg { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
        .lg-prefix { background: var(--ok); }
        .lg-prefill { background: var(--accent); }
        .lg-decode { background: color-mix(in srgb, var(--accent) 45%, var(--bg-sunken)); }
        .lg-done { background: var(--ink-4); opacity: 0.4; }
        .lg-label { margin-right: 14px; font-family: var(--font-mono); font-size: 11px; }
        .sched-controls {
          display: flex; align-items: center; gap: 12px;
          font-size: 12px;
        }
        .sched-tick { color: var(--ink-3); }
        .sched-btn {
          display: inline-flex; align-items: center; gap: 5px;
          background: var(--bg-sunken); border: 1px solid var(--rule);
          color: var(--ink-2);
          padding: 5px 10px; border-radius: 6px;
          font-size: 12px; cursor: pointer;
          font-family: inherit;
        }
        .sched-btn:hover { border-color: var(--ink-4); color: var(--ink); }

        .sched-grid {
          display: grid;
          grid-template-columns: minmax(200px, 1fr) 4fr;
          gap: 4px 16px;
          padding: 14px 0;
          border-top: 1px solid var(--rule-2);
          border-bottom: 1px solid var(--rule-2);
        }
        .sched-row-label {
          display: flex; align-items: center; gap: 10px;
          padding: 4px 0;
          min-height: 20px;
          overflow: hidden;
        }
        .sched-row-id {
          font-size: 11px; color: var(--ink-4);
          flex-shrink: 0;
        }
        .sched-row-prompt {
          font-size: 12.5px; color: var(--ink-2);
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .sched-row-cells {
          display: grid;
          grid-template-columns: repeat(50, 1fr);
          gap: 2px;
          align-items: center;
        }
        .sc {
          height: 14px; border-radius: 2px;
          background: var(--bg-sunken);
          opacity: 0.4;
          transition: background 0.2s, opacity 0.2s, transform 0.2s;
        }
        .sc-on { opacity: 1; }
        .sc-queue { background: var(--bg-sunken); opacity: 0.25; }
        .sc-idle { background: var(--bg-sunken); opacity: 0.25; }
        .sc-prefix { background: var(--ok); }
        .sc-prefill { background: var(--accent); }
        .sc-decode { background: color-mix(in srgb, var(--accent) 45%, var(--bg-sunken)); }
        .sc-done { background: var(--ink-4); opacity: 0.3; }

        .sched-foot {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 0;
          margin-top: 18px;
        }
        .sched-stat {
          padding: 4px 16px;
        }
        .sched-stat + .sched-stat { border-left: 1px solid var(--rule); }
        .sched-stat-v {
          font-size: 22px; font-weight: 500;
          color: var(--ink); letter-spacing: -0.02em;
          line-height: 1;
        }
        .sched-stat-l {
          font-size: 11.5px; color: var(--ink-4);
          text-transform: uppercase; letter-spacing: 0.06em;
          margin-top: 6px;
        }
        @media (max-width: 720px) {
          .sched-grid { grid-template-columns: minmax(80px, 1fr) 3fr; gap: 3px 8px; }
          .sched-row-prompt { display: none; }
          .sched-foot { grid-template-columns: repeat(2, 1fr); gap: 16px 0; }
          .sched-stat:nth-child(3) { border-left: none; padding-left: 16px; }
        }
      `}</style>
    </section>
  );
};

window.SchedulerAnim = SchedulerAnim;
