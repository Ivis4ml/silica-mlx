// Architecture / scheduler diagram — the centerpiece.
// Layered stack from Request → Engine → Scheduler → KV/Codec → MLX.

const Architecture = () => {
  const [hover, setHover] = React.useState(null);

  const layers = [
    {
      id: "client",
      title: "Client surface",
      sub: "CLI · ChatSession · Engine.generate · OpenAI HTTP server (silica serve, v1.7.33)",
      tone: "muted",
      pills: ["silica run", "silica chat", "silica serve", "Engine.generate_batch", "ChatSession", "silica.llm.LLM"],
    },
    {
      id: "engine",
      title: "Engine main loop",
      sub: "single-request and batched share one code path",
      tone: "ink",
      pills: ["Request FSM", "Sampling", "Profiling", "Metrics"],
    },
    {
      id: "scheduler",
      title: "Scheduler core",
      sub: "vLLM-style continuous batching with admission ladder",
      tone: "accent",
      isCenter: true,
      blocks: [
        { name: "ContinuousBatcher", note: "interleave prefill + decode" },
        { name: "MemoryBudgeter", note: "admit → evict → preempt → reject" },
        { name: "RadixPrefixCache", note: "block-granular trie reuse" },
        { name: "Preempt + Replay", note: "snapshot + re-enter queue" },
      ],
    },
    {
      id: "models",
      title: "Family adapters",
      sub: "frozen ModelAdapter protocol — batched parity vs mlx-lm",
      tone: "ink",
      grid: [
        { name: "Qwen3 dense", sizes: "0.6B – 32B" },
        { name: "Qwen3.5 hybrid", sizes: "DeltaNet · 0.8/4/27B" },
        { name: "Gemma4 dense", sizes: "31B" },
        { name: "Qwen3.5 MoE", sizes: "35B-A3B · 256×8" },
        { name: "Gemma4 MoE", sizes: "26B-A4B · 128×8" },
      ],
    },
    {
      id: "kv",
      title: "KV cache layer",
      sub: "VectorCodec[P] seam — codec lives at the prefix-block store",
      tone: "ink",
      pills: ["PagedKVCache", "RadixPrefixCache", "PrefixBlockStore", "KVCodec"],
    },
    {
      id: "vq",
      title: "Native KV codec",
      sub: "lossless-at-measurement-precision against vqbench baseline",
      tone: "warn",
      grid: [
        { name: "BlockTQ", sizes: "B=64 · 4-bit" },
        { name: "RaBitQ-1", sizes: "1-bit" },
        { name: "ExtRaBitQ", sizes: "2 / 3 / 4-bit" },
        { name: "Identity", sizes: "fp16 baseline" },
      ],
    },
    {
      id: "mlx",
      title: "MLX runtime",
      sub: "every hot-path tensor mx.array · unified-memory model · M5 Pro 48 GB",
      tone: "muted",
      pills: ["mx.array", "unified memory", "no torch dependency"],
    },
  ];

  return (
    <section className="block" id="architecture" style={{ background: "var(--bg-sunken)" }}>
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">Architecture</div>
          <h2>Seven layers, one integrated runtime.</h2>
          <p>Frozen <span className="mono">typing.Protocol</span> seams between every layer — ModelAdapter, KVManager, VectorCodec, WeightProvider, DraftEngine — so native capabilities slot in without changing call sites.</p>
        </div>

        <div className="arch-stack">
          {layers.map((l, i) => (
            <div
              key={l.id}
              className={"arch-layer arch-" + l.tone + (l.isCenter ? " arch-center" : "") + (hover === l.id ? " arch-hover" : "")}
              onMouseEnter={() => setHover(l.id)}
              onMouseLeave={() => setHover(null)}
            >
              <div className="arch-spine">
                <div className="arch-num mono">L{i}</div>
                <div className="arch-bar"></div>
              </div>
              <div className="arch-body">
                <div className="arch-head">
                  <div className="arch-title">{l.title}</div>
                  <div className="arch-sub">{l.sub}</div>
                </div>

                {l.pills && (
                  <div className="arch-pills">
                    {l.pills.map(p => <span key={p} className="arch-pill mono">{p}</span>)}
                  </div>
                )}

                {l.blocks && (
                  <div className="arch-blocks">
                    {l.blocks.map(b => (
                      <div key={b.name} className="arch-block">
                        <div className="arch-block-name mono">{b.name}</div>
                        <div className="arch-block-note">{b.note}</div>
                      </div>
                    ))}
                  </div>
                )}

                {l.grid && (
                  <div className="arch-grid">
                    {l.grid.map(g => (
                      <div key={g.name} className="arch-cell">
                        <div className="arch-cell-name">{g.name}</div>
                        <div className="arch-cell-sizes mono">{g.sizes}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .arch-stack {
          display: flex; flex-direction: column; gap: 4px;
          position: relative;
        }
        .arch-layer {
          display: grid;
          grid-template-columns: 64px 1fr;
          background: var(--bg-elev);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          transition: transform 0.18s, box-shadow 0.18s;
          overflow: hidden;
        }
        .arch-hover {
          transform: translateX(2px);
          box-shadow: var(--shadow-md);
        }
        .arch-spine {
          display: flex; flex-direction: column; align-items: center;
          padding: 22px 0;
          background: var(--bg-sunken);
          border-right: 1px solid var(--rule);
          position: relative;
        }
        .arch-num {
          font-size: 11px;
          color: var(--ink-4);
          font-weight: 500;
          letter-spacing: 0.04em;
          margin-bottom: 12px;
        }
        .arch-bar {
          flex: 1;
          width: 2px;
          background: var(--rule);
          border-radius: 1px;
          min-height: 32px;
        }
        .arch-center .arch-spine { background: var(--accent-soft); }
        .arch-center .arch-num { color: var(--accent); font-weight: 600; }
        .arch-center .arch-bar { background: var(--accent); opacity: 0.5; }
        .arch-warn .arch-spine { background: color-mix(in srgb, var(--warn) 12%, transparent); }
        .arch-warn .arch-num { color: var(--warn); font-weight: 600; }
        .arch-warn .arch-bar { background: var(--warn); opacity: 0.4; }

        .arch-body { padding: 22px 26px; }
        .arch-head { margin-bottom: 14px; }
        .arch-title {
          font-size: 16px; font-weight: 600;
          letter-spacing: -0.012em;
          color: var(--ink);
          margin-bottom: 4px;
        }
        .arch-center .arch-title { color: var(--accent); font-size: 18px; }
        .arch-warn .arch-title { color: var(--warn); }
        .arch-sub {
          font-size: 13.5px;
          color: var(--ink-3);
          line-height: 1.45;
        }
        .arch-pills {
          display: flex; flex-wrap: wrap; gap: 6px;
        }
        .arch-pill {
          font-size: 11.5px;
          padding: 4px 10px;
          border-radius: 6px;
          background: var(--bg-sunken);
          color: var(--ink-2);
          border: 1px solid var(--rule);
        }
        .arch-blocks {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 10px;
        }
        .arch-block {
          padding: 14px 14px;
          background: color-mix(in srgb, var(--accent-soft) 60%, transparent);
          border-radius: 10px;
          border: 1px solid color-mix(in srgb, var(--accent) 18%, transparent);
        }
        .arch-block-name {
          font-size: 12.5px;
          font-weight: 600;
          color: var(--accent);
          margin-bottom: 4px;
          letter-spacing: -0.01em;
        }
        .arch-block-note {
          font-size: 12px;
          color: var(--ink-2);
          line-height: 1.4;
        }
        .arch-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 0;
          border-radius: 10px;
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        .arch-cell {
          padding: 12px 14px;
          background: var(--bg-elev);
          border-right: 1px solid var(--rule);
          border-bottom: 1px solid var(--rule);
        }
        .arch-cell:last-child { border-right: none; }
        .arch-cell-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--ink);
          margin-bottom: 2px;
        }
        .arch-cell-sizes {
          font-size: 11.5px;
          color: var(--ink-3);
        }
        @media (max-width: 720px) {
          .arch-layer { grid-template-columns: 44px 1fr; }
          .arch-body { padding: 18px 16px; }
          .arch-spine { padding: 14px 0; }
        }
      `}</style>
    </section>
  );
};

window.Architecture = Architecture;
