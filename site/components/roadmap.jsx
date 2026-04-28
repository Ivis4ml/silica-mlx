// Roadmap / phase board

const Roadmap = () => {
  const phases = [
    { id: "P-0", name: "Core skeleton + frozen interfaces + sampler", state: "done" },
    { id: "P-1", name: "Single-request Engine.generate", state: "done" },
    { id: "P-2", name: "Continuous batching · radix prefix cache · admission · preempt+replay", state: "done" },
    { id: "P-3", name: "Family adapters — Qwen3 · Qwen3.5 hybrid · Gemma4 · two MoE families", state: "done" },
    { id: "P-4", name: "Unified bench harness — runner · oracles · 15 scenarios · vqbench xcheck", state: "done" },
    { id: "P-4.5", name: "Chunked-prefill minimal + VectorCodec runtime spike", state: "done" },
    { id: "P-5", name: "VQ KV compression — BlockTQ · RaBitQ · ExtRaBitQ · per-head Haar", state: "done" },
    { id: "P-6", name: "Weight streaming — dense layer prefetch + per-expert MoE residency", state: "stub" },
    { id: "P-7", name: "Speculative decoding — DraftEngine · DraftTarget / EAGLE / Medusa", state: "stub" },
    { id: "P-8", name: "OpenAI-compatible HTTP server + session layer", state: "plan" },
  ];

  const pct = phases.filter(p => p.state === "done").length / phases.length * 100;

  return (
    <section className="block" id="roadmap" style={{ background: "var(--bg-sunken)" }}>
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">Roadmap</div>
          <h2>Seven phases shipped. Three on deck.</h2>
          <p>The engine main loop already carries stub implementations behind frozen interfaces. The planned phases progressively replace those stubs without changing call sites.</p>
        </div>

        <div className="rm-progress">
          <div className="rm-progress-bar"><div className="rm-progress-fill" style={{ width: pct + "%" }}></div></div>
          <div className="rm-progress-meta mono">
            <span>{phases.filter(p => p.state === "done").length} / {phases.length} phases shipped</span>
            <span>{Math.round(pct)}%</span>
          </div>
        </div>

        <div className="rm-list">
          {phases.map((p, i) => (
            <div key={p.id} className={"rm-row rm-" + p.state}>
              <div className="rm-marker">
                <div className="rm-dot"></div>
                {i < phases.length - 1 && <div className="rm-line"></div>}
              </div>
              <div className="rm-id mono">{p.id}</div>
              <div className="rm-name">{p.name}</div>
              <div className="rm-state">
                {p.state === "done" && <><Icon name="check" size={12} /><span>shipped</span></>}
                {p.state === "stub" && <><span className="rm-stub-dot"></span><span>stub · swappable</span></>}
                {p.state === "plan" && <><Icon name="dot" size={10} /><span>planned</span></>}
              </div>
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .rm-progress { margin-bottom: 32px; max-width: 720px; }
        .rm-progress-bar {
          height: 4px; background: var(--rule); border-radius: 2px; overflow: hidden;
        }
        .rm-progress-fill {
          height: 100%; background: var(--accent);
          transition: width 0.5s cubic-bezier(.2,.7,.2,1);
        }
        .rm-progress-meta {
          display: flex; justify-content: space-between;
          font-size: 11.5px; color: var(--ink-3); margin-top: 8px;
        }
        .rm-list {
          background: var(--bg-elev);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          padding: 8px 0;
        }
        .rm-row {
          display: grid;
          grid-template-columns: 28px 60px 1fr 140px;
          align-items: center;
          gap: 14px;
          padding: 14px 22px;
          position: relative;
        }
        .rm-marker {
          position: relative;
          height: 100%;
          display: flex; flex-direction: column; align-items: center;
        }
        .rm-dot {
          width: 10px; height: 10px;
          border-radius: 50%;
          margin-top: 6px;
          flex-shrink: 0;
        }
        .rm-done .rm-dot { background: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent); }
        .rm-stub .rm-dot { background: var(--bg-elev); border: 2px solid var(--ink-4); }
        .rm-plan .rm-dot { background: var(--bg-elev); border: 2px dashed var(--ink-4); }
        .rm-line {
          flex: 1;
          width: 2px;
          background: var(--rule);
          margin-top: 4px;
          margin-bottom: -22px;
        }
        .rm-done .rm-line { background: color-mix(in srgb, var(--accent) 35%, var(--rule)); }
        .rm-id {
          font-size: 12px;
          font-weight: 600;
          color: var(--ink-3);
          letter-spacing: 0.02em;
        }
        .rm-done .rm-id { color: var(--accent); }
        .rm-name {
          font-size: 14px;
          color: var(--ink);
          line-height: 1.4;
        }
        .rm-stub .rm-name, .rm-plan .rm-name { color: var(--ink-2); }
        .rm-state {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 12px;
          color: var(--ink-3);
          justify-self: end;
          font-family: var(--font-mono);
          font-feature-settings: normal;
        }
        .rm-done .rm-state { color: var(--ok); }
        .rm-stub-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: var(--warn);
        }
        @media (max-width: 720px) {
          .rm-row { grid-template-columns: 24px 48px 1fr; gap: 10px; padding: 12px 14px; }
          .rm-state { grid-column: 2 / -1; justify-self: start; padding-left: 48px; margin-top: 2px; font-size: 11px; }
          .rm-name { grid-column: 3; }
        }
      `}</style>
    </section>
  );
};

window.Roadmap = Roadmap;
