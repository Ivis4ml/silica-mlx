// Comparison table — silica-mlx vs mlx-lm / vLLM / SGLang

const Comparison = () => {
  const cols = [
    { id: "mlx", label: "mlx-lm", sub: "MLX" },
    { id: "vllm", label: "vLLM", sub: "CUDA" },
    { id: "sglang", label: "SGLang", sub: "CUDA" },
    { id: "silica", label: "silica-mlx", sub: "MLX", highlight: true },
  ];
  const rows = [
    ["Continuous batching", "no", "yes", "yes", "yes"],
    ["Radix prefix cache", "no", "block-level", "yes", "yes"],
    ["Memory-budget admission", "no", "yes", "yes", "yes"],
    ["Preempt + replay", "no", "yes", "yes", "yes"],
    ["KV codec compression", "no", "FP8 / INT8", "limited", "BlockTQ + RaBitQ"],
    ["Hybrid DeltaNet (batched)", "single-req", "no", "no", "yes"],
    ["MoE batched dispatch", "single-req", "yes", "yes", "yes"],
    ["OpenAI HTTP server", "no", "yes", "yes", "planned"],
    ["Speculative decoding", "no", "yes", "yes", "foundation"],
    ["Per-expert MoE residency", "no", "limited", "no", "planned"],
  ];

  const cellFor = (val, isSilica) => {
    if (val === "yes") return <Icon name="check" size={16} />;
    if (val === "no") return <Icon name="x" size={14} />;
    if (val === "planned") return <span className="cmp-planned">planned</span>;
    return <span className="cmp-text">{val}</span>;
  };

  return (
    <section className="block" id="compare">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">The niche</div>
          <h2>vLLM-core architecture, native to Apple Silicon.</h2>
          <p>mlx-lm is single-request and solves a different problem. vLLM and SGLang are CUDA-first and don't run on Apple Silicon. silica-mlx fills the gap with one integrated MLX-native runtime. Speculative decoding shipped at v1.7.19 as a foundation; production payoff settled with a measurement-anchored negative at v1.7.20-22 (see Performance below for cycle-23 verify-cost closure).</p>
        </div>

        <div className="cmp-card">
          <div className="cmp-grid">
            <div className="cmp-corner"></div>
            {cols.map(c => (
              <div key={c.id} className={"cmp-col " + (c.highlight ? "cmp-col-hi" : "")}>
                <div className="cmp-col-label">{c.label}</div>
                <div className="cmp-col-sub">{c.sub}</div>
              </div>
            ))}
            {rows.map((r, i) => (
              <React.Fragment key={i}>
                <div className="cmp-row-label">{r[0]}</div>
                {r.slice(1).map((v, j) => (
                  <div key={j} className={"cmp-cell " + (cols[j] && cols[j].highlight ? "cmp-cell-hi" : "") + " cmp-" + (v === "yes" ? "yes" : v === "no" ? "no" : v === "planned" ? "plan" : "txt")}>
                    {cellFor(v, cols[j] && cols[j].highlight)}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .cmp-card {
          background: var(--bg-elev);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          overflow: hidden;
        }
        .cmp-grid {
          display: grid;
          grid-template-columns: minmax(200px, 1.6fr) repeat(4, minmax(110px, 1fr));
        }
        .cmp-corner { background: var(--bg-elev); border-bottom: 1px solid var(--rule); }
        .cmp-col {
          padding: 18px 16px;
          text-align: center;
          border-bottom: 1px solid var(--rule);
          border-left: 1px solid var(--rule);
        }
        .cmp-col-hi {
          background: linear-gradient(180deg, var(--accent-soft), transparent);
          position: relative;
        }
        .cmp-col-hi::before {
          content: "";
          position: absolute; left: 0; right: 0; top: 0;
          height: 2px; background: var(--accent);
        }
        .cmp-col-label {
          font-size: 14.5px; font-weight: 600; letter-spacing: -0.01em;
        }
        .cmp-col-hi .cmp-col-label { color: var(--accent); }
        .cmp-col-sub {
          font-size: 11.5px; color: var(--ink-4);
          text-transform: uppercase; letter-spacing: 0.06em;
          margin-top: 4px; font-family: var(--font-mono);
        }
        .cmp-row-label {
          padding: 14px 20px;
          font-size: 14px;
          color: var(--ink);
          border-bottom: 1px solid var(--rule-2);
          font-weight: 500;
        }
        .cmp-cell {
          padding: 14px 16px;
          text-align: center;
          font-size: 13px;
          color: var(--ink-2);
          border-bottom: 1px solid var(--rule-2);
          border-left: 1px solid var(--rule-2);
          display: flex; align-items: center; justify-content: center;
          min-height: 48px;
        }
        .cmp-cell-hi {
          background: color-mix(in srgb, var(--accent-soft) 55%, transparent);
        }
        .cmp-yes { color: var(--ok); }
        .cmp-no { color: var(--ink-4); }
        .cmp-plan { color: var(--ink-3); }
        .cmp-planned {
          font-size: 11.5px; color: var(--ink-3);
          background: var(--bg-sunken);
          padding: 3px 9px; border-radius: 999px;
          border: 1px solid var(--rule);
          font-family: var(--font-mono);
          letter-spacing: 0.02em;
        }
        .cmp-cell-hi .cmp-yes { color: var(--accent); }
        .cmp-cell-hi .cmp-text { color: var(--accent); font-weight: 500; }
        .cmp-text { font-family: var(--font-mono); font-size: 12px; }
        .cmp-grid > div:last-child,
        .cmp-grid > div:nth-last-child(2),
        .cmp-grid > div:nth-last-child(3),
        .cmp-grid > div:nth-last-child(4),
        .cmp-grid > div:nth-last-child(5) { border-bottom: none; }
        @media (max-width: 720px) {
          .cmp-grid { font-size: 12px; grid-template-columns: minmax(140px, 1.4fr) repeat(4, 1fr); }
          .cmp-row-label { padding: 12px 10px; font-size: 12.5px; }
          .cmp-cell { padding: 10px 4px; }
          .cmp-col { padding: 12px 4px; }
          .cmp-text { font-size: 10.5px; }
          .cmp-planned { font-size: 10px; padding: 2px 6px; }
        }
      `}</style>
    </section>
  );
};

window.Comparison = Comparison;
